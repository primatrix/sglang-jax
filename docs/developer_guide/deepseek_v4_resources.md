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

`get_buffer(family, layer_id)` returns a whole global array. Layer-to-buffer
maps are available as `layer_to_buffer[family]`. The per-rank shapes, including
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
array tuples. A model step returns **both** keys:

```python
updates = {
    "token_to_kv_pool": updated_kv_buffers,
    "compressor_state_pool": updated_state_buffers,
}
memory_pools.replace_all(updates)
```

Pool replacements validate all family counts, shapes and dtypes. Normal
model/backend integration and donation scheduling are owned by C3/M2.

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

## Validation and remaining integration

Run on a compatible CPU JAX/Flax installation:

```bash
PYTHONPATH=python JAX_PLATFORMS=cpu \
XLA_FLAGS=--xla_force_host_platform_device_count=4 \
python -m pytest -q python/sgl_jax/test/mem_cache/test_deepseek_v4_*.py \
  python/sgl_jax/test/test_model_runner_kv_cache_mixin.py \
  python/sgl_jax/test/mem_cache/test_swa_allocator.py \
  python/sgl_jax/test/mem_cache/test_paged_allocator_multi_dp.py \
  python/sgl_jax/test/mem_cache/test_req_to_token_pool.py
```

Coverage includes 43-layer mapping, 128/256 pages, BF16/FP32 bytes, complete
pool update round trips, same-shape JIT reuse, slot 0 and padding, state reuse,
DP/TP layouts, 127→128→129, nonaligned chunks, single-resource exhaustion,
atomic rollback, SWA-only release, grouped release, and randomized resource
conservation. No test loads complete Flash weights or runs TPU kernels.
C2–C4 and M2 still need to consume these interfaces; J0 owns real HBM,
full-model quality and serving/retract acceptance.

Local acceptance on 2026-09-07: **103 passed** (41 V4 tests and 62 existing
pool/allocator/runner regressions), JAX 0.11.1, Flax 0.12.9, four virtual CPU
devices. This includes an Explicit DP=2/TP=2 mesh JIT update, in addition to
Auto mesh coverage. Ruff 0.13.3, Black 24.10.0 and isort 5.13.2 checks passed.
The shared local environment had incompatible Flax 0.12.0; regression tests
used a temporary Flax 0.12.9 overlay without changing repository dependencies.
No remote CI or TPU/full-model validation is claimed by this result.
