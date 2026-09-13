# DeepSeek V4 C128 HCA

This branch rebases upstream PR #1549 (`df52b39cf8e074f7bc757927e7c65855eaaf68ed`)
onto `primatrix/sglang-jax` `epic/dsv4@1910da714`, then integrates
`epic/dsv4@7a0181618` (config and mHC v7x schedules). It retains the upstream
compressor, SWA-plus-compressed attention, standalone backend, benchmark, and
NumPy oracle. `DeepseekV4HCABackend` adapts those kernels to the C1 resources
already on the epic branch. It is a C128 backend consumer, not complete Flash
model construction or scheduler integration.

## Resource ownership

The V4 path uses `DeepseekV4TokenToKVPool`, `DeepseekV4CompressStatePool`,
`ReqToTokenPool`, and `DeepseekV4TokenToKVPoolAllocator`. It does not construct
the standalone `HCAKVPool`, `HCARecurrentStatePool`, or `HCAKVPoolAllocator`.
Those upstream classes remain available for the original standalone tests
and comparison benchmark.

An original-token page has 128 or 256 positions. Its C128 buffer page
therefore has **1 or 2 records**, while its SWA page still has 128 or 256 rows.
The two attention page tables use these different units. Both derive from
C1's original-token ledger: history page IDs are shared; SWA page IDs come
from `full_to_swa_index_mapping`. Only `seq_len // 128` completed records
become visible, with per-query causal masking inside the kernel.

The backend uses reshape views of the existing arrays. State changes from
`[slots,128,1024]` to the kernel's `[slots,128,2,512]` view. Slot zero is a
real request. Each DP rank indexes the unchanged global request slot;
`request_capacity` is padding. Zero-prefix EXTEND (including retract/recompute
and slot reuse) initializes content to zero and scores to negative infinity
before consuming the state. A nonzero-prefix call requires valid persisted
state; this backend does not reconstruct it from compressed KV.

## Calling contract

Construct once with the existing mesh and C1 capacities:

```python
backend = DeepseekV4HCABackend(
    mesh=mesh,
    page_size=kv_pool.page_size,
    max_context_len=req_pool.max_context_len,
    request_capacity=req_pool.size,
)
```

Before the model step, the scheduler must reserve the entire chunk, fill
`req_pool.req_to_token`, and keep every SWA page needed by the earliest query
of the chunk. Metadata construction is read-only:

```python
backend.forward_metadata = backend.get_forward_metadata(
    worker_batch, request_pool=req_pool, allocator=kv_allocator
)
```

Use ordinary `EXTEND` or `DECODE`. Request rows and tokens are padded into
equal contiguous DP sections, as in the existing attention batch contract.
Each rank gets its own cumulative lengths, request IDs, token validity and
page tables. Array capacities are padded equally across ranks. Changes in
compression-boundary counts do not change the decode metadata shapes.
Page-table and maximum-query capacities still grow at bucket boundaries;
this is not a promise of one compilation for arbitrary contexts.

The layer call accepts the standalone HCA query/weight arguments, with the
C1 state owner supplied as `compressor_state_pool`. It returns
`(output, (state, swa, c128))`, where all three updates have their native C1
shapes. Collect updates by **global layer ID** and replace both owners:

```python
updates = backend.pack_pool_updates(
    {layer_id: layer_update}, kv_pool, compressor_state_pool
)
memory_pools.replace_all(updates)
```

The complete family dictionaries preserve untouched C4, indexer, SWA-only
and other C128 layers. A model combining multiple attention families must
merge their updates into the same final dictionaries before replacement.
C2/C4 retain responsibility for allocation, rollback, request-end release,
retract and safe SWA reclamation. Host allocator/request objects are explicit
metadata inputs rather than fields captured in the Flax model graph.

## Platform and validation

The added v7x schedule starts from v6e's conservative 32 MiB **scoped** VMEM
budget. This does not transfer the upstream v6e benchmark speedups to v7x.
Small C1 record pages use the scatter update instead of the standalone
packed two-lane DMA writer. Attention reads each small page into a whole-page
VMEM scratch buffer, then inserts its values into the attention tile. This
avoids both sub-tile DMA slices and per-request compressed-history staging
in HBM. The transport is a correctness baseline, not a tuned performance result.

The standalone HCA kernel tests and benchmarks remain under
`test/srt/kernels/hca/`. For V4 model integration, use the
[real-weight attention and layer comparisons](../../test/manual/deepseek_v4_precision/README.md),
which exercise the production resource bridge against native SGLang GPU captures.
