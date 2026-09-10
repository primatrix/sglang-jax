# DeepSeek V4 model and runtime

`models/deepseek_v4.py` owns the full trunk parameter tree, checkpoint mapping
and paired MXFP4 expert conversion. The forward graph includes embeddings,
mHC pre/post, attention and FFN norms, SWA/CSA/HCA, grouped output projection,
hash/learned MoE routing, shared experts, mHC head collapse and LM logits.
MTP checkpoint tensors are explicitly excluded.

`DeepseekV4AttentionBackend` selects the native CSA/SWA backend or the tuned
Flash-geometry HCA backend. Native HCA also supports smaller validation models.
All three return `[tokens, heads, head_dim]` and complete cache updates.
CSA currently uses native JAX compression, indexing and masked attention;
this does not establish long-context throughput or production HBM efficiency.

`mem_cache/deepseek_v4/` groups `pool.py` (KV families), `state.py` (request
compressor state), `allocator.py` (history/SWA ownership) and `capacity.py`
(budget and construction). As in SGLang, request compression state remains
separate from KV storage and allocation. Existing global slot, original-token
addressing, rollback, reclamation and no-prefix-reuse contracts remain in force.

Real loading checks every trunk key, rejects unknown and missing tensors, and
validates target shapes and integer hash IDs. Non-expert FP8 uses K128/N128
E8M0 block scales (decoded explicitly from their bytes); routed MXFP4 is
converted strictly to per-channel power-of-two FP8.
Expert loading fills device slices one expert at a time, without assembling
all experts on the host. Conversion errors abort loading. Expert placement
must be identity; MTP, EPLB checkpoint remapping and static converted-checkpoint
export are outside this entry point.

## Initialization and ownership

`ModelRunner._get_attention_backend` routes V4 to `DeepseekV4AttentionBackend`
before loading the model, independently of the MLA/FA default. V4 does not
set the MLA absorption or V3 DSA flags. After C1 creates its request pool,
allocator and two device pool owners, `bind_attention_resources` binds the
actual request capacity. V4 freezes its model graph only after that binding.
Metadata production rejects calls made before binding.

The backend stores scalar capacities, not host ownership objects or device
pool snapshots. `ModelRunner.get_attention_metadata` passes the current host
request pool and allocator explicitly; both worker execution and precompilation
use this entry point. C2 must have reserved the entire step, filled the request
mapping and retained its required SWA pages before calling it.

The model call receives the current `MemoryPools`. It returns a complete
update dictionary for `token_to_kv_pool` and `compressor_state_pool`, including
untouched families/layers. Validation checks owner keys, family keys, layer
counts, shapes and dtypes while tracing, before dispatch donates the inputs.
The existing `ModelRunner._forward` replaces both owners immediately after
dispatch; the next call uses these new arrays. No host synchronization is added.
Do not retain array references from a prior donated step.

## Metadata transport

`DeepseekV4RuntimeMetadata` carries both the HCA kernel metadata and the
M2.1 `DeepseekV4AttentionMetadata`. The latter is exposed through
`ForwardBatch.deepseek_v4_metadata`; it is transported once as part of the
backend's dynamic PyTree child. M2.1 continues to own the derivation formulas.

Derive each DP rank separately and concatenate corresponding array leaves.
Thus `cu_q_lens` is `[DP * (local_B + 1)]`, and boundary arrays have
`DP * boundary_capacity(local_T, local_B, ratio)` entries. Query/request/event
indices and inert token-index sentinels are **rank-local**. Consumers must
partition the leading dimension on `data` before interpreting these indices.
Global request slots are preserved, including slot zero on any rank.

The HCA consumer supplies Q with `P("data", "tensor", None)` and shared K/V
with `P("data", None)`. Explicit mesh axis names must match even when a mesh
axis has size one; reshaping a packed input does not add the head sharding.

All lengths, masks, slot IDs, addresses and compression events are int32/bool
dynamic arrays. The runtime checks that live output addresses match the
request mapping and still own SWA storage. Zero-prefix requests initialize
state; continuing chunks retain it. Per-query attention masking still belongs
to the selected M2 kernel; this bridge does not change HCA's numerical rule.

## Bucket and dummy contract

For the runtime HCA path, static metadata choices come from the padded
request/token/cache capacities and configured context, rather than live
lengths. In particular, page-table capacities and maximum query capacity are
fixed, and prefill always uses the ragged path. Standalone HCA callers retain
the adaptive schedule and uniform-prefill optimization.

`CompilationManager` asks the runner to prepare V4 dummy batches before
creating `ForwardBatch`. Every request/query is inactive; request slots use
C1's padding slot, and no real request or page is allocated. The dummy still
has the full metadata structure and kernel schedule for its bucket. Executing
either dummy mode must preserve all pool arrays.
The compile key can change with mode or padded shape; changing only live
lengths, slots or events must not create a new model trace.

This favors compile reuse over tuning: page tables follow the configured cache
bucket, and the ragged path can cost more than uniform prefill. The compact
validation fixture's memory use is not a full-model HBM sizing result.

## Validation

Use the focused [real-weight precision workflows](../../test/manual/deepseek_v4_precision/README.md)
for GPU/TPU module and layer comparisons. Full-model request acceptance uses
`test/manual/deepseek_v4_precision/static_fp8_smoke.py` against a published static
checkpoint. The smoke checks native-encoded greedy token IDs and normal EOS;
it does not establish long-context, concurrency or broad model-quality coverage.
