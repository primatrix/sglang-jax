# DSA Decode MLA Kernel v2 Design

## Status

This design supersedes the direct per-slot TensorCore DMA strategy in
`2026-07-14-dsa-decode-mla-kernel-design.md`. The earlier implementation was
useful for establishing the public contract and reference tests, but Falcon
compilation exposed two hardware constraints:

- a BF16 cache token is physically one lane of a packed `[2, width]` row, so a
  single-token HBM DMA is not a valid TensorCore DMA tile; and
- issuing one packed-row DMA and one tiny QK/PV update for each of 2,048 slots
  would leave the MXU underutilized even after fixing the tile shape.

The v2 design separates irregular address collection from dense matrix work.

## Scope

Implement decode-only sparse MLA attention over an already selected set of
physical KV slots. The kernel does not implement the DSA indexer, Top-K state,
IndexShare, prefill, MTP, cache lifecycle, or model-runner integration.

The public operation remains:

```python
def dsa_decode_mla_attention(
    ql_nope: jax.Array,       # bf16[B, H, L]
    q_pe: jax.Array,          # bf16[B, H, R]
    cache_kv: jax.Array,      # bf16[P, page_size / 2, 2, align(L) + align(R)]
    topk_slots: jax.Array,    # int32[B, K], physical token slots
    valid_counts: jax.Array,  # int32[B]
    *,
    sm_scale: float,
) -> jax.Array:               # bf16[B, H, L]
```

For the initial TP=8 Falcon GLM-5.2 decode benchmark, the important local shape
is `H=8`, `L=512`, `R=64`, and `K=2048`. The kernel remains generic in local
head count. The cache width is `512 + 128 = 640`. The model's attention scale
is derived from its unabsorbed
QK dimension and is `256**-0.5`; it is not `(512 + 64)**-0.5`.

The caller guarantees causal visibility. Slot order is not semantically
significant, while multiplicity is: permuting a row must not change the result,
but duplicate slots retain duplicate softmax mass. Entries at or beyond
`valid_counts[b]` are padding and never contribute.

## Hardware findings that determine the design

The existing MLA v2 kernel establishes the TPU-compatible physical layout:

- BF16 queries are packed as `[B, H / 2, 2, aligned_dim]`;
- BF16 cache pages are `[P, page_size / 2, 2, aligned_width]`;
- latent and RoPE dimensions are padded independently to 128; and
- large, aligned HBM-to-VMEM page copies feed TensorCore blocks.

Regular TensorCore Pallas DMA is contiguous. It cannot express a gather from
2,048 unrelated HBM rows, and a direct HBM load is not legal. JAX 0.8.1 does,
however, expose indirect DMA for SparseCore vector-subcore kernels. It accepts
a VMEM offset vector and gathers major-dimension HBM rows into a contiguous
VMEM target. TPU v7 supports this kernel type.

This creates a natural division of labor: SparseCore resolves irregular slot
addresses; TensorCore consumes a regular selected-KV matrix.

## Candidate architectures

### 1. Direct packed-row TensorCore DMA

Map slot `s` to packed row `s // 2`, DMA `[2, 640]`, and select lane `s % 2`.
This fixes the BF16 tile violation and reads at most about twice the selected
payload, but requires roughly 2,048 small DMA operations and 2-token compute
updates per request. Keep this only as a diagnostic fallback.

### 2. Page or fixed-microblock bucketing

Sort slots and load every touched page or block. This improves DMA size only
when selections are localized. For 2,048 uniformly dispersed selections in a
160K context, the expected traffic is approximately:

| Unit | Distinct units | Tokens loaded | Amplification |
| --- | ---: | ---: | ---: |
| page 128 | 1,007 | 129K | 63x |
| block 16 | 1,852 | 29.6K | 14.5x |
| packed row 2 | 2,022 | 4.0K | 2x |

This is not a robust primary kernel for DSA Top-K. It may become a locality-
aware specialization after real indexer traces exist.

### 3. SparseCore gather plus TensorCore attention

Gather only the selected rows in SparseCore, materialize them contiguously in
HBM, then run one regular TensorCore attention program per request. This is the
recommended v2 architecture.

For GLM BF16 shapes, the selected tensor is `2048 * 640 * 2 = 2.5 MiB` per
request. Cache read, intermediate write, and TensorCore read total about
7.5 MiB. A dense cache scan reads about 10 MiB at 8K, 40 MiB at 32K, 160 MiB
at 128K, and 195 MiB at 160K before accounting for compute. The data-volume
break-even is around 6K tokens; launch and synchronization overhead will move
the practical dispatch threshold higher and must be measured.

## Selected architecture

```text
packed paged HBM cache + physical slots
                 |
                 v
  SparseCore indirect DMA gather
  [B, Kpad] -> [B, Kpad, 640]
                 |
                 v
     contiguous selected-KV HBM
                 |
                 v
  TensorCore QK -> softmax -> PV
                 |
                 v
          latent output [B,H,L]
```

The two Pallas stages remain separately testable and benchmarkable. The public
wrapper composes them and owns padding, validation, and output slicing.

### Stage A: SparseCore selected-KV materialization

Normalize the packed cache to the logical token view `[capacity, width]`.
Because `reshape` alone preserves the BF16 packing order, logical token `s`
maps to row `s` in this view without a page-table lookup.

Before dispatch, replace padding slots with safe slot zero. Valid slots have
already been range-checked. The attention stage still uses `valid_counts`, so
the safe rows are never observed by softmax.

Use a static gather block `G`, initially 128:

- grid: `(B, ceil(K / G))`;
- slots input block: `[G]` in VMEM;
- cache source: `[capacity, width]` in HBM;
- indirect DMA target: `[G, width]` in VMEM;
- output block: `[G, width]` in HBM.

Each program performs one indirect HBM-to-VMEM gather using the `G` physical
slot offsets, then one regular VMEM-to-HBM copy. `K` is padded to `G`; `width`
is already a multiple of 128. The output is `[B, Kpad, width]`.

The stage uses
`CompilerParams(kernel_type=KernelType.SC_VECTOR_SUBCORE)`. JAX's SparseCore
tests require the compiler option
`xla_tpu_use_tc_device_shape_on_sc=false`; the prototype will compile this
stage through a small inner `jax.jit` that owns that option. Model integration
must later confirm whether to keep the separately compiled call or merge the
option into the model-runner JIT.

### Stage B: TensorCore selected MLA attention

Prepare queries with the same packing helpers as MLA v2:

- `ql_nope`: `[B, H / 2, 2, align(L)]`;
- `q_pe`: `[B, H / 2, 2, align(R)]`.

The initial kernel uses one program per request and consumes the contiguous
`[Kpad, width]` selected matrix. For the TP=8 Falcon case (`H=8`, `K=2048`,
and `width=640`), the selected matrix is about 2.5 MiB and fits within the
per-program VMEM budget
used by the existing MLA kernel. Compute:

```text
score[h,k] = scale * (
    dot(q_nope[h], selected[k, :align(L)])
  + dot(q_pe[h], selected[k, align(L):])
)
prob[h,:] = softmax(mask(score[h,:], k < valid_count))
out[h,:] = prob[h,:] @ selected[:, :L]
```

The first implementation evaluates the complete fixed-K score matrix and a
stable FP32 softmax. This gives the MXU a regular `H x K` by `K x D` workload
and simplifies correctness. If compilation or profiling shows VMEM pressure,
the predetermined fallback is the existing MLA v2 double-buffered K-block
pipeline with online softmax; it does not change Stage A or the public API.

Do not add split-K initially. Existing MLA assigns one program to a sequence
group, and additional Pallas grid programs do not imply useful intra-core
parallelism. Add split-K only after a trace identifies this stage as the
bottleneck and includes the reduction cost.

### Memory and dispatch

The intermediate allocation is significant at high batch sizes:

| Batch | Intermediate BF16 bytes |
| ---: | ---: |
| 1 | 2.5 MiB |
| 8 | 20 MiB |
| 32 | 80 MiB |
| 128 | 320 MiB |

The prototype accepts this allocation to validate the architecture. A later
serving integration can process batch chunks or overlap gather and attention
with double-buffered HBM workspaces. The wrapper should dispatch to dense MLA
below a measured context threshold rather than assuming sparse always wins.

## Correctness strategy

Correctness is checked at three independent boundaries.

### Packed-cache oracle

Build fixtures in the real production BF16 layout with packing factor two.
The oracle explicitly maps `slot -> page, packed_row, lane`; it must not use
the same flattening expression as the kernel. Cover page boundaries, odd/even
lanes, unsorted slots, duplicates, and padding. The previous packing-factor-one
fixtures are retained only for generic reference coverage and are not evidence
of TPU layout correctness.

### Stage tests

1. Compare SparseCore materialization with a dense JAX gather for production
   layout, including `K=2048`, `width=640`, cross-page slots, duplicates, and
   safe replacement of `-1` padding.
2. Compare TensorCore attention in Pallas interpret mode with an FP32 NumPy/JAX
   reference. Cover heterogeneous valid counts, masked padding, independently
   padded latent/RoPE dimensions, and the GLM shape and scale.
3. Compare the composed TPU operation with the explicit packed-cache FP32
   reference. Require finite outputs and use the existing BF16 MLA tolerance
   (`rtol=2e-2`, `atol=1e-2`) initially, while recording max absolute and
   relative errors.

Metamorphic checks provide additional independence:

- permuting the same selected multiset does not change the result beyond the
  reduction tolerance;
- adding masked padding does not change the result; and
- duplicating a valid slot has the same effect as duplicating its key/value in
  the dense selected reference.

After indexer integration, model-level validation must compare selected rows,
layer outputs, and final logits with GPU SGLang on the same GLM checkpoint and
prompts. The isolated kernel cannot establish indexer or cache-lifecycle
correctness.

## Falcon execution gates

Falcon runs are deliberately staged; a later gate is not started until the
earlier one passes.

1. **SparseCore feature smoke:** compile a tiny indirect gather on one v7x-8
   slice and confirm the required compiler option and layout.
2. **Production gather correctness:** run BF16 packing-two fixtures, then the
   GLM `K=2048`, `width=640` materialization test.
3. **TensorCore attention correctness:** run the selected-attention stage on
   GLM dimensions without SparseCore in the same job.
4. **End-to-end numerics:** compose both stages for batch 1, 8, and 32 and
   compare to the explicit packed-cache oracle.
5. **Performance:** only after all correctness gates pass, benchmark and
   profile SparseCore gather, TensorCore attention, and end-to-end latency
   separately.

Performance cases use `K=2048`, contexts 8K/32K/128K/160K, batch 1/8/32
(128 only if workspace memory is acceptable), and three slot distributions:
uniform dispersed, page-local clustered, and an actual indexer trace when one
is available. Compare:

- SparseCore indirect gather;
- plain JAX/XLA gather plus the same TensorCore stage;
- direct packed-row fallback; and
- existing dense MLA v2.

Record compile time separately, then median and p99 device latency after
warm-up. Capture XProf traces for batch 1 and 32, including HBM traffic and
VMEM spills when available. The current JAX/libtpu combination does not accept
the previously attempted LLO counter flags, so standard XProf is the supported
profiling path.

No sparse performance claim is made unless the composed operation is faster
than dense MLA v2 at long contexts and its cache work remains proportional to
`K`, not total context length.

## Failure containment and fallbacks

- If SparseCore indirect DMA is unavailable in the Falcon runtime, use plain
  JAX/XLA gather as the functional baseline and report the runtime limitation;
  do not silently relabel it as the final kernel.
- If full-K TensorCore attention exceeds VMEM or compiles poorly, switch only
  Stage B to aligned K blocks with double buffering and online softmax.
- If the two-stage launch dominates short contexts, dispatch dense MLA below a
  benchmark-derived threshold.
- If actual indexer traces show high page locality, benchmark a page-bucketed
  specialization before adding hybrid dispatch complexity.

These fallbacks are ordered hypotheses with explicit evidence gates, not a
sequence of unstructured kernel rewrites.
