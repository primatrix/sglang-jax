# DSA Decode MLA Adaptive SparseCore Window Design

## Decision

Keep the existing two-stage design:

```text
physical selected slots -> SparseCore indirect gather -> contiguous selected KV
                        -> TensorCore selected MLA -> latent output
```

Do not revisit direct TensorCore per-slot DMA, page-bucketed dense reads, or a
new attention tile in this iteration.  Falcon measurements show that selected
MLA attention is a minority of the operation time; the irregular gather is the
bottleneck.  The one new hypothesis is that the latency-critical single-request
decode case underutilizes SparseCore because the current 128-row window creates
only 16 windows for `K=2048`, enough for one 16-subcore SparseCore but not the
two active SparseCores exposed by the pinned Falcon runtime.

## Target layout and unchanged semantics

The production prototype shape remains BF16 MLA cache
`[pages, page_size / 2, 2, 640]`, local query heads `H=8`, latent dimension
`L=512`, RoPE dimension `R=64`, and fixed DSA capacity `K=2048`.  Scores and
output are unchanged:

```text
score[h, k] = 256^(-1/2) * (dot(q_nope[h], c_kv[k]) + dot(q_pe[h], k_rope[k]))
out[h] = softmax(score[h, :valid_count]) @ c_kv[:, :512]
```

Slots remain physical token slots, with duplicates retained.  The caller
continues to guarantee visibility/causality.  Entries after `valid_counts` are
replaced with the safe physical slot zero for the gather and masked in the
attention stage.  No Top-K, indexer, cache-lifecycle, IndexShare, prefill, or
model-runner integration is added here.

## Adaptive window policy

The policy is used only for the `sparsecore-pipeline` implementation and only
when the public caller requests the default `gather_block="auto"`:

| Static shape | Window | SparseCore workers | Windows per worker |
| --- | ---: | ---: | ---: |
| `B=1, Kpad=2048, W=640` | 64 rows | 2 cores x 16 subcores = 32 | 1 |
| `B>=2, Kpad=2048, W=640` | 128 rows | up to 2 x 16 | at least 1 |
| any unsupported shape or explicit integer | 128 rows / explicit value | existing planner | existing behavior |

At width 640, a 64-row BF16 output tile is 80 KiB; double buffering needs about
160 KiB per vector subcore.  This is below the existing 128-row design's
320 KiB budget, and 64 is still a multiple of SparseCore's required vector
width.  The change lets one request produce 32 independent gather windows,
which exactly fills the two active 16-subcore SparseCores already exercised by
the successful B=32 pipeline run.  It does not rely on or alter physical-slot
locality.

The explicit `gather_block=128` API remains available for reproduction.  The
automatic policy does not select a smaller block for unknown `K`, width, or
runtime topology; it conservatively retains the current 128-row choice.

## Why this is the next experiment

The full Falcon matrix at the pinned JAX 0.8.1/libtpu 0.0.30 runtime found
legacy SparseCore gather plus attention at `B=1,K=2048,context=160K` took
1.220 ms median, of which gather alone took 1.222 ms and attention alone 0.157
ms.  At `B=32,context=32K`, the manually pipelined implementation measured
1.139 ms median versus 1.887 ms for legacy SparseCore composition, so exposing
all SparseCore workers can matter.  The B=1 pipeline case has not yet been
measured.

The current `dense-jax-baseline` is a reshape/einsum workload, not the serving
MLA-v2 ragged-paged kernel.  It may be retained for a workload sanity check but
must not decide a sparse/dense serving threshold.  The focused performance gate
therefore records it separately and adds an equivalent production MLA-v2
decode invocation before any sparse-performance claim.

## Required validation

1. **Planner contract on CPU:** auto resolution chooses 64 only for the
   single-request GLM static shape; explicit 128 and non-pipeline paths remain
   128.
2. **Falcon numerical gate:** compare `B=1,H=8,L=512,R=64,K=2048` pipeline
   output using auto/64 against the independent explicit packed-cache FP32
   oracle.  Also compare `B=32` at the same GLM width and K, not the previous
   small `K=128,W=256` proxy.  Require BF16 `rtol=2e-2, atol=1e-2` and finite
   output.
3. **Falcon discriminator:** at `B=1,K=2048,context=160K`, time pipeline-64,
   pipeline-128, legacy SparseCore, XLA gather, and selected attention using
   the same slots and 50 warm-ups plus 200 device-timed iterations.  The new
   implementation is retained only if pipeline-64 improves the median and p99
   over pipeline-128 without a correctness regression.
4. **Production baseline:** run the existing MLA-v2 ragged-paged decode kernel
   with the same BF16 cache, query dimensions, request count, and context;
   record compilation and device latency separately.  No dispatch threshold or
   end-to-end DSA benefit is claimed until this comparison is available.

## Rejection criteria and next branch

Reject adaptive-64 if it fails to compile with the two-core SparseCore mesh,
is numerically different from the oracle, or does not beat pipeline-128 at
the B=1 gate.  In that case preserve the already-correct 128-row pipeline and
investigate the actual indexer slot distribution before considering a
locality-aware hybrid (contiguous TensorCore DMA for long physical runs plus
SparseCore gather for the remainder).  Page bucketing is intentionally not a
default: it can multiply read volume by orders of magnitude for dispersed
Top-K rows.
