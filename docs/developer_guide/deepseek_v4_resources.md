# DeepSeek V4 resource pools (C1)

Implements INFERENCE-93 on `primatrix/sglang-jax` `epic/dsv4`, following the
resource contract in the task's Outline revision 37. These are independently
testable resources, not a complete V4 model/backend or serving integration.

## Public objects and addresses

`DeepseekV4CacheSpec.from_config(hf_config)` takes only the first
`num_hidden_layers` compression ratios. Flash 0731 has 43 backbone layers:
2 SWA-only, 21 C4, 20 C128; draft entries in the ratio list are excluded.

`DeepseekV4TokenToKVPool(size, size_swa, page_size, spec, mesh, dp_size)` owns
BF16 buffer families `swa`, `c4`, `c128`, and `indexer`. `size` and `size_swa`
are **global usable original-token capacities**, divisible by `DP * P`.
`P` is 128 or 256 for both history and SWA. Each attention DP shard reserves
one extra history page and one extra SWA page, both numbered zero. History
uses one shared logical page ledger across its three buffer families; there
is no full uncompressed history tensor.

The KV pool directly implements the `KVCache` resource-management contract.
Backends use `get_swa_buffer(layer_id)`, `get_compressed_buffer(layer_id)`,
`get_compressed_page_size(layer_id)`, and `get_indexer_buffer(layer_id)`.
These accessors return existing global arrays (or the compressed page size),
routing by the layer's compression ratio. An out-of-range layer raises
`IndexError`; a layer without the requested resource raises `ValueError`.
The ordinary single-buffer `get_fused_kv_buffer`, `get_kv_buffer`, and
`set_kv_buffer` interfaces are deliberately unsupported: SWA alone does not
represent a compressed-attention layer. CPU snapshots are also unsupported.

The family-level `get_buffer/set_buffer/write` interfaces remain available
for reference paths. `get_kv_size_bytes()` returns global logical bytes,
including all four families and per-rank padding, excluding compressor state.
`mem_usage` is the same amount in GiB, not the sum of physical TP/EP replicas.

The per-rank shapes, including
padding, are:

| Family | Per-layer, per-DP-shard shape |
| --- | --- |
| `swa` | `[S + P, D]` |
| `c4` | `[G + 1, P/4, D]` |
| `c128` | `[G + 1, P/128, D]` |
| `indexer` | `[G + 1, P/4, Di]` |

All allocator locations are **rank-local original-token locations**, starting
at `P`. Compressed write addresses are `loc // ratio`, indexing the flattened
first two axes of the matching buffer. The backend must only write completed
groups and expose causally valid entries: reserving a page does not generate
compressed KV. SWA locations come from `full_to_swa_index_mapping`, a NumPy
array for DP=1 and a list of arrays for DP>1. Zero means unmapped/padding.

`write(family, layer_id, loc, values, valid_mask, dp_rank=0)` is a masked
reference update on **global** arrays; it adds the rank's array offset.
Invalid writes are dropped, including duplicate padded entries. In a future
`shard_map` consumer, use local addresses directly on local array slices;
do not add the global rank offset again. Backend kernels can instead produce
replacement arrays directly. These reference scatters are not optimized TPU
kernels.

## Pool layouts and dtypes on the current head

Anchors of the form `path:line` in the sections below are relative to
`python/sgl_jax/` at the current head; a bare file name refers to the full path
given at its first mention in this document.

The shape table above describes the logical layout. The arrays actually
allocated on the current head differ in one family: the ratio-128 record pool
`c128` is allocated in the HCA kernels' physical layout by default.

| Family | Owner | dtype | Allocated shape (all DP shards concatenated on axis 0) | Anchor |
| --- | --- | --- | --- | --- |
| `swa` | KV pool | BF16 | `(swa_slots_per_rank * dp, D)` with `swa_slots_per_rank = size_swa // dp + P` | `srt/mem_cache/deepseek_v4/pool.py:161`, `:164-167` |
| `c4` | KV pool | BF16 | `(pages, P/4, D)` with `pages = (pages_per_rank + 1) * dp` | `pool.py:162`, `:168` |
| `indexer` | KV pool | BF16 | `(pages, P/4, Di)` | `pool.py:179` |
| `c128` | KV pool | BF16 | `(pages, 1, P/128, D)` when `native_hca_layout()` is on (default); `(pages, P/128, D)` otherwise | `pool.py:169-178` |
| `c4` state | state pool | FP32 | `(slots, 8, 4*D)` with `slots = (size + 1) * dp` | `srt/mem_cache/deepseek_v4/state.py:59`, `:66` |
| `indexer` state | state pool | FP32 | `(slots, 8, 4*Di)` | `state.py:68` |
| `c128` state | state pool | FP32 | `(slots, 128, 2, D)` in the native layout; `(slots, 128, 2*D)` otherwise | `state.py:60-64` |

The KV pool constructor rejects any dtype other than BF16 (`pool.py:139-140`)
and any page size other than 128 or 256 (`pool.py:141-142`).
`native_hca_layout()` reads `DSV4_HCA_NATIVE_LAYOUT` (default on) and exists so
that neither the ratio-128 state nor the ratio-128 KV pool is relaid out on the
way into and out of every HCA layer (`pool.py:63-70`). Both layouts occupy the
same bytes; the reference `write()` still addresses `c128` as flattened rows and
treats the first `P/128` rows of each rank as the reserved page
(`pool.py:276-301`, reserved count at `:286`).

The state pool initialises the score half of every family to negative infinity
at construction (`state.py:48`); `score_slice` selects that half for both the
3-D and the 4-D layouts (`state.py:16-25`).

`DSV4_HCA_FLAT_COMPRESSED=1` is an optional view switch, not a different
allocation: the HCA mixin hands the 4-D `c128` pool to the kernels as is, or as
flat `[rows, D]` when the switch is set, and reshapes the returned array back to
the pool shape (`srt/layers/attention/dsv4/hca.py:56-60`, view selection
`:375-385`, reshape back `:402-405`). The HCA host page table pads the ratio-128
side per request: a request with `completed = length // 128` records contributes
`max(1, ceil(completed / (P/128)))` compressed pages, a request with no
completed record contributes the dummy page 0, and an inactive request
contributes one dummy page and one page's worth of cumulative length
(`srt/layers/attention/dsv4/hca.py:115-118`, inactive branch `:85-90`).

## Request state and updates

`DeepseekV4CompressStatePool(size, spec, mesh, dp_size)` binds directly to the
**global** `ReqToTokenPool` slot. Slot 0 is a legal request; local index `size`
is padding. `state_indices(request_slots, valid_mask)` maps invalid requests
to this dummy position. No second request/state free list is maintained.

The existing request free list is not DP partitioned. Accordingly, every DP
shard reserves `size + 1` positions rather than assuming `slot % (size/DP)`
identifies an owner. The model consumer uses the request's assigned DP rank
and its unchanged global request slot. This costs more memory than a future
DP-owned request pool, and the budget explicitly includes that cost.

State buffer families (all FP32) are `c4`: `[R+1, 8, 4D]`,
`c128`: `[R+1, 128, 2D]`, and `indexer`: `[R+1, 8, 4Di]` per layer/rank.
The first half of the last axis holds contents, the second half scores.
Empty contents are zero and empty scores are negative infinity.

C4 (the lifecycle task) supplies the validity and initialization event;
M2 initializes before a slot's first numerical consumption or reuse.
`reset(request_slots, valid_mask, dp_rank)` supplies this numerical empty
state for selected requests. SWA release does not reset state. Freeing a
request slot alone does not initialize its old state or erase old KV;
consumers must honor initialization masks and generated-history lengths.

Both pools are PyTrees with static metadata and `.buffers` dictionaries of
array tuples. PyTree reconstruction restores metadata without allocating arrays.
The KV pool and compressor state remain separate owners; state is not a
`KVCache` and continues to use request slots.

Each owner provides `build_buffer_updates(layer_updates)`, which merges partial
per-layer updates into a complete payload without mutating the owner:

```python
kv_updates = kv_pool.build_buffer_updates({
    layer_id: {"swa": new_swa, "compressed": new_compressed},
})
state_updates = state_pool.build_buffer_updates({
    layer_id: {"compressor": new_compressor_state},
})
updates = {
    "token_to_kv_pool": kv_updates,
    "compressor_state_pool": state_updates,
}
memory_pools.replace_all(updates)
```

KV resources are `swa`, `compressed` (C4 or C128 according to the layer), and
`indexer` (C4 only). State resources are `compressor` and `indexer` (C4 only).
Omitted resources retain their arrays. Unknown resources, invalid layers, or
shape/dtype changes fail before any replacement. Backend update dictionaries
use semantic `compressed` rather than physical `c4/c128` family names; the
final owner payload still contains the original physical family keys.

ModelRunner validates the complete two-owner result while tracing, before
input donation. `replace_buffer` validates and commits a complete family
payload after execution; these checks do not introduce host reads of array
contents. Both owners must be returned even when only one has changed.

## Compressor state slots and lifecycle

The slot is the request slot. `DeepseekV4CompressStatePool` is indexed by the
`ReqToTokenPool` slot `req_pool_idx`; there is no separate state allocator or
free list (`srt/mem_cache/deepseek_v4/state.py:29-34`;
`srt/layers/attention/dsv4/metadata.py:342-344`). The backend reads the slots
from `batch.req_pool_indices` for both the CSA path and the HCA path
(`srt/layers/attention/deepseek_v4_backend.py:306`;
`srt/layers/attention/dsv4/hca.py:140`). Each DP shard reserves `size + 1`
positions; position `size` is padding and `state_indices` maps every invalid
request there (`srt/mem_cache/deepseek_v4/state.py:57-58`, `:104-109`). A write
goes to `local + dp_rank * slots_per_rank` and an out-of-range or invalid entry
is dropped past the end of the buffer (`state.py:114-127`).

Initialisation happens on the first chunk only. `state_init_mask` marks
"requests whose state must be reset before use (first execution, or a recompute
from zero after retract)" (`metadata.py:351-352`). The backend derives it as
`(q_lens > 0) & (prefix_lens == 0)`
(`srt/layers/attention/deepseek_v4_backend.py:402`); the HCA path derives `init
= active & (prefixes == 0)` and accepts an explicit mask only if it is a subset
of that (`srt/layers/attention/dsv4/hca.py:176-181`).

A continuing chunk must not initialise. `derive_attention_metadata` raises when
`state_init_mask & (prefix_lens > 0)` with the message "a continuing chunk must
keep the state it accumulated" (`metadata.py:420-424`). Chunked prefill keeps
the request's `prefix_indices` through
`DeepseekV4ChunkCache.cache_unfinished_req`
(`srt/mem_cache/chunk_cache.py:126-127`), so the next chunk arrives with a
non-zero prefix and the accumulated state is consumed, not reset.

The reset itself is executed on device inside the step. CSA layers call
`_reset_state`, which writes one empty template (content zero, score negative
infinity) into the masked slots and excludes the last local slot because it is
padding (`srt/layers/attention/dsv4/execution.py:173-187`, used at `:249` and
`:257`). HCA layers build one template per forward and DMA it into the init
slots, skipping the padding sentinel
(`srt/layers/attention/dsv4/hca.py:326-347`). Both use `init_state_slots`
(`srt/kernels/dsv4/state_init.py:60`) when `DSV4_STATE_INIT_KERNEL` is on, which
is the default, and an XLA scatter otherwise (`state_init.py:53-57`).

Retract today is release plus full recompute. There is no DSv4-specific retract
code; the generic path is: `retract_decode`
(`srt/managers/schedule_batch.py:1466`) calls `release_req`
(`schedule_batch.py:1551-1560`), which goes through `release_kv_cache`
(`srt/mem_cache/common.py:120-132`) to `DeepseekV4ChunkCache.release_req`
(`chunk_cache.py:129-146`). That call frees the request's complete extent
(history pages together with their SWA pages), zeroes its `req_to_token` row and
frees the request slot. `reset_for_retract` then clears `prefix_indices` and
sets `extend_input_len = 0` (`srt/managers/schedule_batch.py:681-686`), and the
scheduler requeues the request (`srt/managers/scheduler.py:2617`,
`_extend_requests_to_queue` at `:1857`). On re-admission the request receives a
fresh slot and fresh pages and is prefilled again from position 0 with
`state_init_mask` set, so the compressor state is rebuilt from scratch; nothing
of the old state or history is salvaged. The allocator's `backup_state` /
`restore_state` transaction (`srt/mem_cache/deepseek_v4/allocator.py:237-257`)
is not used by this path.

Release on finish uses the same route (`release_kv_cache` ->
`DeepseekV4ChunkCache.release_req`). Only host ledgers are mutated; no device
buffer is touched at release time.

Slot reuse is by dropping the owner. A recycled slot is reset by the next
zero-prefix forward through `state_init_mask`, inside the donated model graph,
rather than by any host-side pool mutation
(`srt/mem_cache/chunk_cache.py:109-114`;
`srt/mem_cache/deepseek_v4/state.py:35-36`). Raw page addresses are not
generation handles: the lifecycle layer must clear the request owner and must
not submit stale addresses after the pages have been reused
(`allocator.py:182-184`).

## Allocator transactions and release

`DeepseekV4TokenToKVPoolAllocator(kvcache)` supplies `alloc`, `alloc_extend`,
`alloc_decode`, `free`, `free_swa`, and per-rank capacity queries.
`alloc` is page-aligned. Extend/decode retain the existing allocator argument
order, with `seq_lens` including this step's input and `last_loc=-1` or `0`
for an empty prefix. Request tails must identify distinct live pages.

`estimate_extend` / `estimate_decode` return history and SWA demand in pages
and original-token slots. They share the allocation planner, including
partially filled tails and remapping a reclaimed SWA tail. Use
`can_allocate(demand, dp_rank)` for exact admission; `available_size()` is a
conservative whole-page capacity and excludes already reserved tail space.

The planner validates the complete batch and checks both free lists before
committing. Capacity failure returns `None` and leaves the entire allocator
unchanged. It never rolls back by freeing the request's existing partial
page. `backup_state()` / `restore_state()` capture all ledgers and mappings;
restore retains the mapping object's identity. C2 must additionally restore
its own request slots, `req_to_token`, and host length reservations if a
wider batch transaction fails. Those objects are not mutated by this allocator.

A history page has one request owner. Releases must include all currently
written tokens in each page; partial live-page release raises before any
mutation. `free_swa` checks only still-mapped tokens and clears the whole SWA
page mapping while retaining history. The lifecycle caller decides when a
page is safe to release, considering the earliest query of a long chunk.
Finish/retract passes the complete original-token request mapping to `free`.
Repeated cleanup before reuse is harmless. Raw addresses are not generation
handles: the lifecycle caller must clear a released owner and must not submit
stale addresses after the pages have been reused by another request.

## Who writes pages and who builds read tables

Ratio-4 (CSA) layers are host-addressed and device-written. The host builds one
`ReadTables` per DP rank (`srt/layers/attention/dsv4/dispatch.py:98-135`) in
`padded_read_tables` (`execution.py:59`), called from `get_forward_metadata`
(`srt/layers/attention/deepseek_v4_backend.py:374`). `compressed_rows` are flat
compressed-entry addresses (`loc // ratio`); the decode-only fields
`decode_page_indices`, `decode_window_rows` and `decode_page_segments` are built
only for DECODE with page size 128
(`srt/layers/attention/deepseek_v4_backend.py:343`;
`srt/layers/attention/dsv4/execution.py:161-171`) and the builder asserts that
every compressed page starts at an allocated page boundary
(`execution.py:151-152`). On device, `run_layer` (`dispatch.py:222`) writes the
records with `_scatter_records` (`dispatch.py:778`), which uses the Pallas
`paged_row_write` when the per-page run is a multiple of 16, the buffer is 2-D
BF16 and at least `DSV4_PAGED_RECORD_WRITE_MIN_RECORDS` (default 256) records
are present, and a drop-mode scatter otherwise (`dispatch.py:511-513`,
`:786-806`); invalid boundaries are dropped, never aimed at entry 0. The window
is written by `update_window_kv`
(`srt/layers/attention/dsv4/attention.py:296-321`), paged when at least
`DSV4_PAGED_KV_WRITE_MIN_TOKENS` (default 256) tokens are written
(`srt/layers/attention/dsv4/attention.py:51-52`). Extend reads gather
`compressed_rows` with `jnp.take` (`dispatch.py:340`); decode branches into
`csa_decode_attention` with the page indices and window rows
(`dispatch.py:448-470`). `run_layer` returns `(out, updates)` with replacement
arrays only; nothing is written in place (`dispatch.py:255-261`).

Ratio-128 (HCA) layers are host-addressed and kernel-written. The host builds
the HCA kernel table in `_get_hca_metadata`
(`srt/layers/attention/dsv4/hca.py:123`), including the per-request window and
compressed page lists (`srt/layers/attention/dsv4/hca.py:80-121`), and
cross-checks page anchors, original-token offsets and SWA ownership against
`req_to_token` and `full_to_swa_index_mapping`
(`srt/layers/attention/dsv4/hca.py:93-110`). On device, `_forward_hca`
(`srt/layers/attention/dsv4/hca.py:294`) calls `run_hca`
(`srt/layers/attention/hca_execution.py:68`), which selects `decode`, `uniform`
or `ragged` from the forward mode and the uniform fast-path flag
(`hca_execution.py:147-151`) and runs `hca_step` (`srt/kernels/hca/hca.py:90`)
inside a `shard_map` whose pool specs follow the buffer rank
(`srt/layers/attention/hca_execution.py:195-241`, `_data_spec` at `:244-246`).
The `c128` page writes and the window writes happen inside the kernel layer:
`ragged_attention` (`srt/kernels/hca/attention.py:1494`, writes at `:1585-1604`)
and `uniform_prefill_attention` (`:1787`, writes at `:1856-1876`) use
`_write_cache_rows` (`:389`), `_scatter_compressed_pages` (`:359`) or
`_scatter_physical_rows` (`:171`); the Pallas paged writer is selected by
`DSV4_HCA_PAGED_ROW_WRITE` (default on, `:229-233`). `run_hca` returns `(output,
(state, window, compressed))` and the mixin reshapes the three arrays back to
the pool shapes (`srt/layers/attention/dsv4/hca.py:402-405`).

Commit is the same for both ratios. The model collects every layer's updates,
`pack_pool_updates` splits them into KV and state families and calls each
owner's `build_buffer_updates`
(`srt/layers/attention/deepseek_v4_backend.py:536-549`), and the runner commits
with `MemoryPools.replace_all` after the step
(`srt/model_executor/model_runner.py:1140`, `:1170`;
`srt/mem_cache/memory_pool.py:1729`). The jitted step donates `memory_pools`
(`srt/model_executor/model_runner.py:388-390`) and `_validate_v4_pool_updates`
checks the complete update set while tracing (`model_runner.py:106`, `:412`).

## Capacity and factory

`plan_deepseek_v4_pools` accepts post-weight, post-execution-reservation
**per-device bytes**, the request count, page size, DP size, existing
`swa_full_tokens_ratio`, and an optional per-DP `max_total_tokens` cap.
It accounts for BF16 KV, FP32 state, padding and full-page rounding before
choosing the largest fitting history capacity. When the request count is
unspecified, it derives a bounded count using approximately one quarter of
the budget for state (at least one request per DP, maximum 2048 globally).
An explicit request count that cannot fit fails rather than silently shrinking.

`build_deepseek_v4_pools` returns the real request pool, `MemoryPools`, and
allocator, and checks array bytes against the plan. The existing runner's
`init_memory_pool` branches for `deepseek_v4` / `DeepseekV4ForCausalLM` before
legacy MHA/MLA/SWA sizing. Its `_profile_available_bytes` already subtracts
`mem_fraction_static` execution headroom and the embedding pool; V4 then
reserves state and both KV families. The CI small-cache limit and user cap
cannot inflate capacity after budgeting.

Array capacity is sharded on existing mesh axis `data`; the single KV head
and state feature dimensions are replicated over TP/EP. The report is
per-device and does not divide these replicated dimensions by TP. The runner
requires page size 128/256, overlap/radix reuse disabled, ordinary (non-mixed)
forward and no speculative/draft execution. KV `auto` resolves to BF16.

## Per-step metadata and its sharding

`get_forward_metadata(batch, request_pool, allocator)`
(`srt/layers/attention/deepseek_v4_backend.py:289`) is the single host entry
point per step. It accepts ordinary EXTEND and DECODE only (`:316-317`),
validates lengths, query counts and slot ownership (`:318-328`, "active V4
requests must own distinct slots" at `:327-328`), and for EXTEND checks that
prefix plus query length equals the sequence length (`:329-333`).

Per DP rank it produces fixed-size arrays. Read-table sizes are power-of-two
buckets of the completed-group count per ratio, shared across ranks so that
every rank has identical local extents under `shard_map` (`:335-341`); the
decode page table is sized only for DECODE with page size 128 (`:342-344`); the
precompile context ladder can override both (`:345-349`). Each rank then gets
`padded_read_tables` (`:374`) and `derive_attention_metadata` (`:392`;
`srt/layers/attention/dsv4/metadata.py:320`), the latter yielding one
`DeepseekV4AttentionMetadata` whose per-query arrays have the padded token
length and whose per-request arrays have the padded batch length; padded entries
carry no plausible address (`metadata.py:170-177`, field list `:336-353`). Every
per-request quantity is carried by value and addressed by `request_slots`, never
by batch position (`metadata.py:355-357`).

Everything travels as one int32 vector. The attention tree, the three read
tables (ratios 0, 4, 128), the 14-leaf HCA kernel table and the state init slots
are concatenated across ranks on the host and packed by `pack_metadata`
(`srt/layers/attention/deepseek_v4_backend.py:42`, buffer at `:71`; call site
`:410-419`). The vector is uploaded with a single `device_put` under
`NamedSharding(mesh, P("data"))` (`:405`, `:427`), or kept as a host array when
`SGLANG_JAX_LAZY_HOST_ARGS` is on (default), in which case it enters the jit
replicated (`srt/utils/jax_utils.py:245-253`).

On device, `DeepseekV4RuntimeMetadata`
(`srt/layers/attention/deepseek_v4_backend.py:93`) unpacks the vector once per
trace and memoises the result (`:109-119`). `resolve()` returns `(attention,
read_tables)` for the CSA path (`:151`). `hca_metadata(mesh)` returns the HCA
view and reshards every leaf to `P("data")`, because leaves unpacked inside the
jitted step come out replicated while the HCA `shard_map` expects the sharding
the host upload used; for `dp == 1` this is a layout no-op (`:121-142`). The HCA
`shard_map` places `compressor_input`, `new_kv` and `positions` on `P("data")`,
`q` and `attention_sink` on `("data", "tensor")` / `("tensor")`, the three pools
on `P("data", None, ...)` matching the buffer rank, and weights replicated
(`srt/layers/attention/hca_execution.py:195-215`). The CSA path runs `run_layer`
inside its own `shard_map` over `("data", "tensor")`
(`srt/layers/attention/dsv4/execution.py:299`).

## Validation

Full-model acceptance requires GPU/TPU module and layer comparisons and a real
TPU serving check against a published static checkpoint. Check native-encoded
greedy token IDs and normal EOS separately from long-context, concurrency and
broad model-quality coverage.

The launch flags, client parameters, pass rules and GSM8K gate behind the
numbers quoted for the current head on one TPU v7x 2x2x1 host are described in
[deepseek_v4_tpu_v7x_baseline.md](deepseek_v4_tpu_v7x_baseline.md).
