# DSA Decode MLA Adaptive SparseCore Window Design

## Decision

Keep the existing two-stage design:

```text
physical selected slots -> SparseCore indirect gather -> contiguous selected KV
                        -> TensorCore selected MLA -> latent output
```

Do not revisit direct TensorCore per-slot DMA, page-bucketed dense reads, or a
new attention tile in this iteration. Falcon measurements show that selected
MLA attention is a minority of the operation time; the irregular gather is the
bottleneck. The 64-row automatic specialization is **rejected**: the focused
Falcon result did not establish a robust or meaningful performance win. The
automatic pipeline choice is therefore the proven 128-row window; 64 remains
an explicit diagnostic benchmark option only.

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

## Window policy after the Falcon experiment

The policy is used only for the `sparsecore-pipeline` implementation. When the
public caller requests the default `gather_block="auto"`, it resolves to 128
rows for every shape and topology:

| Static shape | Window | SparseCore workers | Windows per worker |
| --- | ---: | ---: | ---: |
| auto, including `B=1, Kpad=2048, W=640` | 128 rows | existing planner | existing behavior |
| explicit `gather_block=64` | 64 rows | explicit experiment only | existing planner |
| any other explicit valid integer | explicit value | existing planner | existing behavior |

At width 640, a 64-row BF16 output tile remains a valid explicit experiment:
it is 80 KiB and double buffering needs about 160 KiB per vector subcore. It
creates 32 independent windows for a single request, enough to fill two active
16-subcore SparseCores, but that theoretical occupancy advantage did not
translate into a reproducible latency improvement.

The explicit `gather_block=64` and `gather_block=128` APIs remain available for
reproduction. The automatic policy conservatively retains the 128-row choice.

## Why this is the next experiment

The full Falcon matrix at the pinned JAX 0.8.1/libtpu 0.0.30 runtime found
legacy SparseCore gather plus attention at `B=1,K=2048,context=160K` took
1.220 ms median, of which gather alone took 1.222 ms and attention alone 0.157
ms.  At `B=32,context=32K`, the manually pipelined implementation measured
1.139 ms median versus 1.887 ms for legacy SparseCore composition, so exposing
all SparseCore workers can matter. The B=1 pipeline case was measured in
Falcon experiment `exp-ezueatb6qw` at `B=1,K=2048,context=160K` with 50
warm-ups and 200 device-timed iterations:

| Variant | Median (ms) | p99 (ms) |
| --- | ---: | ---: |
| implicit `sparsecore-pipeline` (128 rows) | 1.1989 | 1.2370 |
| explicit `sparsecore-pipeline-64` | 1.2039 | 1.2301 |
| explicit `sparsecore-pipeline-128` | 1.2082 | 1.2438 |
| `xla-gather` (composed selected-MLA variant) | 0.2001 | 0.6374 |

The small, mixed differences between the three pipeline readings do not show a
robust 64-row win: explicit 64 is slower at the median than implicit 128 and
only marginally lower at p99, while explicit 128 has a nearby result. This
rejects the automatic adaptation without changing its mathematical result.

The current `dense-jax-baseline` is a reshape/einsum workload, not the serving
MLA-v2 ragged-paged kernel.  It may be retained for a workload sanity check but
must not decide a sparse/dense serving threshold.  The focused performance gate
therefore records it separately and adds an equivalent production MLA-v2
decode invocation before any sparse-performance claim.

## Required validation

1. **Planner contract on CPU:** auto resolution remains 128 for the
   single-request GLM static shape and all other shapes; explicit 64 and 128
   remain valid, unchanged requests.
2. **Falcon numerical gate:** compare `B=1,H=8,L=512,R=64,K=2048` pipeline
   output using auto against the independent explicit packed-cache FP32
   oracle.  Also compare `B=32` at the same GLM width and K, not the previous
   small `K=128,W=256` proxy.  Require BF16 `rtol=2e-2, atol=1e-2` and finite
   output.
3. **Falcon discriminator:** retain the `B=1,K=2048,context=160K` explicit
   64/128 variants, legacy SparseCore, XLA gather, and selected-attention
   measurements using the same slots and 50 warm-ups plus 200 device-timed
   iterations. They remain diagnostic gates; they do not alter automatic
   dispatch unless a future result shows a robust improvement.
4. **Production baseline:** run the existing MLA-v2 ragged-paged decode kernel
   with the same BF16 cache, query dimensions, request count, and context;
   record compilation and device latency separately.  No dispatch threshold or
   end-to-end DSA benefit is claimed until this comparison is available.

## Outcome and next branch

`exp-ezueatb6qw` met the rejection condition: adaptive 64 did not beat the
128-row policy robustly at the B=1 gate. Preserve the already-correct 128-row
pipeline, the explicit window variants, and all Falcon correctness gates.
Investigate actual indexer slot distributions and add the production MLA-v2
ragged-paged comparison before considering a locality-aware hybrid (contiguous
TensorCore DMA for long physical runs plus SparseCore gather for the remainder).
Page bucketing is intentionally not a default: it can multiply read volume by
orders of magnitude for dispersed Top-K rows.
