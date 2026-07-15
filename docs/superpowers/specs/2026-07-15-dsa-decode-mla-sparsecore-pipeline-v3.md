# DSA decode MLA SparseCore pipeline v3

## Why v2 is not the final SparseCore shape

The functionally correct v2 gather uses a SparseCore `pallas_call` with one
128-row indirect DMA window per grid program. Falcon measurements show that
the gather dominates the composed operation and grows with a dispersed cache
footprint even though Top-K remains fixed at 2048:

| B | context | legacy SC gather | selected attention |
|---:|---:|---:|---:|
| 1 | 8K | 0.256 ms | 0.151 ms |
| 1 | 32K | 0.470 ms | 0.154 ms |
| 1 | 128K | 1.011 ms | 0.140 ms |

The gather window is already 128 rows, so increasing arithmetic tile size is
not the missing optimization. The v2 launch does not explicitly map windows
across the SparseCore vector-subcore mesh or pipeline index loads and output
stores.

JAX 0.8.1 exposes the required higher-level primitives:

- `pl.kernel(..., mesh=VectorSubcoreMesh(...))`;
- `pltpu.emit_pipeline(...)` for double-buffered window movement; and
- `pltpu.sync_copy(cache.at[index_ref], output_ref)` for indirect DMA.

This is also the structure used by the official JAX SparseCore gather example.

## v3 execution layout

Inputs remain the v2 contract:

- packed BF16 cache reshaped to `[capacity, 640]`;
- safe selected slots flattened from `[B, Kpad]` to `[1, B*Kpad]`;
- `Kpad` divisible by the 128-row gather window; and
- output `[B, Kpad, 640]` in HBM.

For v7x, one 128x640 BF16 output window is 160 KiB. Double buffering consumes
320 KiB, leaving room within each vector subcore's 512 KiB VMEM for the two
128-element int32 index buffers and pipeline state. Increasing the window
would exceed or leave too little VMEM headroom, so v3 keeps 128.

The mesh worker assignment is explicit and duplicate-free:

```text
worker_id = sparsecore_id * num_subcores + subcore_id
first_window = worker_id * windows_per_worker
window(step) = first_window + step
```

At GLM Top-K 2048 there are 16 windows per request:

- B=1: select 1 SparseCore, use 16 subcores, 1 window per worker;
- B=8: select 2 SparseCores, use 32 subcores, 4 windows per worker;
- B=32: select 2 SparseCores, use 32 subcores, 16 windows per worker.

For other shapes, use the largest available SparseCore count that evenly
divides the number of windows. Reject shapes with fewer than one window per
subcore instead of silently duplicating work or racing output stores. The
legacy and XLA paths remain available for those shapes.

Within each worker, `emit_pipeline` loads the next index block while the body
issues the current indirect gather and the previous output block is committed
to HBM. No global compiler option is attached to this `pl.kernel` path; the
official core-map API owns its device shape. The legacy `pallas_call` path
retains `xla_tpu_use_tc_device_shape_on_sc=false` for controlled comparison.

## Evidence gates

Do not switch the public default until all gates pass on Falcon v7x-8:

1. **Layout unit test:** B1 and B32 plans cover every window exactly once.
2. **Gather correctness:** pipeline output is bitwise equal to XLA gather at
   BF16 width 640 and K=2048, including dispersed physical slots.
3. **Composed correctness:** pipeline gather plus TensorCore attention matches
   the explicit packed-cache FP32 oracle for B1 and B32.
4. **Targeted performance:** compare pipeline, legacy SC, and XLA gather at
   B1/context160K and B32/context32K using 50 warmups and 200 timed iterations.
5. **Decision:** promote v3 only if it materially reduces legacy gather time
   without regressing correctness or p99 stability. Otherwise keep XLA gather
   as the current implementation recommendation and record the SparseCore API
   limitation.

The existing full context matrix remains the baseline dataset; v3 receives
only the two discriminating performance cases above, avoiding another broad
trial-and-error sweep.
