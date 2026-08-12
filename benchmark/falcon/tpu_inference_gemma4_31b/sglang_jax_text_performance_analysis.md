# Gemma 4 31B Text Performance Gap Analysis

This document records the investigation of the pure-text serving performance
gap between SGLang-JAX and the pinned `tpu-inference` Gemma 4 31B benchmark. It
separates experimentally confirmed causes from source-level differences that
still require isolated implementation A/B tests.

## Scope

The comparison uses `google/gemma-4-31B-it` on one TPU v7x-8 host with the
following common workload:

- 1,000 requests submitted at an infinite request rate;
- exactly 1,024 input tokens and 500 output tokens per request;
- temperature zero, ignore EOS;
- DP=4 and TP=2; and
- FP8 W8A8 text weights/activations and FP8 KV cache.

The upstream configuration and its reproduction manifests are documented in
[README.md](README.md). Unless otherwise noted, the results below are one
successful run per configuration and should not be used as a small-regression
threshold without repetitions.

## Executive summary

After correcting the DP scheduling-budget semantics, the best measured
SGLang-JAX output throughput is 8,013.90 token/s. It remains 43.33% below the
single-process `tpu-inference` SPMD control at 14,140.83 token/s, and its median
TPOT is 2.28 times higher.

The investigation found three layers of causes:

1. **Confirmed benchmark configuration mismatch.** The original SGLang-JAX
   configuration applied vLLM's per-DP-rank limits as global limits. Correcting
   the limits improved output throughput by 33.71% and reduced median TTFT from
   22.03 seconds to 6.61 seconds.
2. **Confirmed SWA cache pressure and late eviction.** The default eviction
   interval allows the SWA pool to fill during this workload, causing request
   retractions and recomputation. A shorter-interval runtime ablation improved
   throughput by only 5.19% and did not eliminate retractions, so this is real
   but not the dominant residual gap.
3. **Strongly supported residual kernel-path gap.** The pinned
   `tpu-inference` run enables its experimental batched RPA kernel and uses
   fused gate/up and QKV projections. SGLang-JAX uses an RPA v3 path that misses
   the tuned-block-size table for the relevant Gemma 4 shapes and uses separate
   projections. Source inspection, batch-scaling behavior, and XProf hotspots
   all point to this combined attention/projection path. The exact contribution
   of batched RPA versus projection fusion has not yet been isolated by porting
   each optimization independently.

The vision tower is not executed in the text workload. The four-process vLLM
DP architecture accounts for only about 8.42% of text throughput, and the
profile does not indicate communication as the dominant bottleneck.

## Results

| Implementation and configuration | Request throughput | Output token throughput | Total token throughput | Median TTFT | Median TPOT | Median E2EL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `tpu-inference`, default multiprocess DP | 30.88 req/s | 15,440.38 tok/s | 47,062.28 tok/s | 5,815.62 ms | 46.02 ms | 28,743.68 ms |
| `tpu-inference`, single-process SPMD | 28.28 req/s | 14,140.83 tok/s | 43,101.25 tok/s | 7,698.47 ms | 46.84 ms | 31,069.52 ms |
| SGLang-JAX, original limits | 11.395 req/s | 5,697.60 tok/s | 17,366.28 tok/s | 22,030 ms | 33.37 ms | 37,487 ms |
| SGLang-JAX, corrected DP limits | 15.237 req/s | 7,618.53 tok/s | 23,221.26 tok/s | 6,607.19 ms | 100.61 ms | 55,594.47 ms |
| SGLang-JAX, corrected limits plus shorter SWA eviction interval | 16.03 req/s | 8,013.90 tok/s | 24,426.36 tok/s | 6,511.46 ms | 106.88 ms | 59,845.82 ms |

The original SGLang-JAX median TPOT is not evidence of a faster decode path: it
ran at only 64 requests per DP rank, while the corrected and upstream runs
reach approximately 250--256 requests per DP rank. The large-batch behavior is
the comparable result.

Key relative changes are:

- corrected SGLang-JAX versus original SGLang-JAX: +33.71% output throughput;
- shorter SWA eviction interval versus corrected default: +5.19%;
- best SGLang-JAX versus multiprocess `tpu-inference`: -48.10%;
- best SGLang-JAX versus single-process SPMD `tpu-inference`: -43.33%; and
- single-process SPMD versus multiprocess `tpu-inference`: -8.42%.

## Finding 1: DP scheduling limits were not equivalent

The upstream server is launched with:

```text
--max-num-seqs 256
--max-num-batched-tokens 4096
--data-parallel-size 4
--tensor-parallel-size 2
```

In vLLM's DP engine, the sequence and batched-token limits apply to each DP
rank. The original SGLang-JAX comparison used `256` and `4096` as global
limits. SGLang-JAX computes the per-rank request limit by dividing the global
request limit by `dp_size` in
[`scheduler.py`](../../../python/sgl_jax/srt/managers/scheduler.py), and
`PrefillAdder` starts from a global remaining input-token budget in
[`schedule_policy.py`](../../../python/sgl_jax/srt/managers/schedule_policy.py).

The equivalent SGLang-JAX settings are:

```text
max_running_requests = 1024 global = 256 per DP rank
max_prefill_tokens = 16384 global = 4096 per DP rank
chunked_prefill_size = 4096 per DP rank
```

Runtime logs confirm the behavior. The original run admitted four 1,024-token
requests per prefill batch globally. The corrected run admitted 16 requests and
16,384 tokens globally, distributed as `[4, 4, 4, 4]` across the DP ranks.

This change explains most of the original TTFT regression, but it exposes the
large-batch decode cost: median TPOT rises from 33.37 ms at 64 requests per rank
to 100.61 ms near 250 requests per rank. The upstream single-process SPMD
control remains at 46.84 ms at the same scale.

## Finding 2: SWA eviction is too late for this workload

Gemma 4 has hybrid full and sliding-window attention. SGLang-JAX's scheduling
policy assumes the SWA pool will not constrain admission because slots outside
the active window will be evicted. In the current implementation, the default
decode eviction interval is:

```text
max(page_size, sliding_window * SGL_JAX_SWA_EVICTION_INTERVAL_MULTIPLIER)
```

With the default multiplier `1.0` and a sliding window of 1,024, eviction is
checked every 1,024 decode steps. The benchmark generates only 500 output
tokens. This conflicts with the nearby `ChunkCache` comment that says eviction
occurs on every decode step.

In the corrected-limit run, the global SWA pool reached 100% occupancy and 166
requests were retracted across three pressure events. Those requests had to be
scheduled and prefetched again, producing long ITL outliers.

A runtime-only ablation set:

```text
SGL_JAX_SWA_EVICTION_INTERVAL_MULTIPLIER=0.125
```

This changes the interval to 128 steps. Output throughput increased from
7,618.53 to 8,013.90 token/s, but the run still recorded 175 retractions. The
single-run result bounds the gain at roughly 5% for this implementation; it
does not demonstrate that interval reduction alone fixes SWA ownership and
admission behavior. Concurrent prefill/decode timing and allocator ownership
still need a targeted correctness test.

Relevant implementation locations are:

- admission assumption:
  [`schedule_policy.py`](../../../python/sgl_jax/srt/managers/schedule_policy.py);
- eviction interval and trigger:
  [`schedule_batch.py`](../../../python/sgl_jax/srt/managers/schedule_batch.py);
- slot reclamation: `ScheduleBatch._evict_swa` in `schedule_batch.py`.

## Finding 3: the residual gap is in the text kernel path

### Batched RPA

The pinned upstream benchmark explicitly sets:

```text
USE_BATCHED_RPA_KERNEL=1
```

Source inspection at `tpu-inference` commit
`a5596b27f02d1b1f1fb64c8bd4b0a73ae19b0336` shows that this experimental
kernel differs from the default RPA path in three important ways:

1. it batches multiple sequences instead of running per-request attention
   loops;
2. it uses triple buffering through the Pallas pipeline; and
3. it precomputes page locations and clipping metadata once in the scheduler
   and amortizes the work across model layers.

The current SGLang-JAX run logs an RPA v3 tuned-block-size lookup miss for both
Gemma 4 attention shapes and falls back to heuristic block sizes:

| Layer type | Local query heads | Local KV heads | Head dimension | Sliding window |
| --- | ---: | ---: | ---: | ---: |
| Sliding attention | 16 | 8 | 256 | 1,024 |
| Full attention | 16 | 2 | 512 | none |

The model has 60 decoder layers, including 50 sliding-attention layers, so
per-layer scheduling, metadata, and kernel inefficiencies are repeatedly paid
on the dominant layer type.

### Projection fusion

The same upstream Gemma 4 implementation uses
`JaxMergedColumnParallelLinear` for the gate/up projection and
`JaxQKVParallelLinear` for attention projections. SGLang-JAX currently invokes
separate `gate_proj` and `up_proj` linears in `Gemma4MLP`, and separate
`q_proj`, `k_proj`, and `v_proj` linears in
[`Gemma4Attention`](../../../python/sgl_jax/srt/models/gemma4.py).

This is a confirmed source-level difference. Its standalone throughput impact
has not yet been measured, so it should be evaluated independently after or
alongside the batched RPA port.

## Profile evidence and limitations

The XProf summary for `an-v1gou1et3t` reports the following accumulated TPU
operator time:

| Category | Time | Share |
| --- | ---: | ---: |
| Matmul | 11,396.16 ms | 39.8% |
| Elementwise | 6,632.01 ms | 23.1% |
| Custom kernel | 5,910.46 ms | 20.6% |
| Data formatting | 2,418.45 ms | 8.4% |
| Async | 2,264.10 ms | 7.9% |

All top-ten operator entries are sliding-window RPA custom calls. The trace
also reports only 14.85 ms of visible AllReduce stall, which makes
communication an unlikely explanation for the large residual gap.

The trace window primarily covers large-scale prefill; decode begins near the
end of the surrounding scheduler-log window. Therefore, the percentages above
must not be presented as a pure-decode breakdown. They confirm that RPA and
matmul paths are material during serving, while the decode attribution relies
on the controlled TPOT scaling result and the source comparison. A future
decode-only profile should start only after all 1,000 requests have completed
prefill.

## Alternatives ruled out or bounded

### Vision tower

The pure-text workload does not execute the vision tower. Differences in
vision-tower quantization cannot explain its throughput gap. Both text paths
use FP8 W8A8 and FP8 KV cache in the compared runs.

### Multiprocess DP

The pure-SPMD `tpu-inference` control changes only
`TPU_MULTIPROCESS_DP=0`. Output throughput decreases by 8.42%, from 15,440.38
to 14,140.83 token/s, while median TPOT changes by only 1.78%. SGLang-JAX still
trails that single-process control by 43.33%, so process fan-out is a secondary
effect rather than the primary explanation.

### KV sharing

The inspected Gemma 4 text configuration has no KV-shared layers. KV sharing
is therefore not an implementation advantage in the upstream result.

### Model source location

Candidate model paths checked under the mounted `/models` filesystem were not
present in the profiling experiment, so that run fell back to the Hugging Face
model identifier. This affects model acquisition and startup time, not the
steady-state serving measurements reported here.

## Recommended implementation order

1. Make DP configuration semantics explicit and keep the corrected benchmark
   values as the comparison contract.
2. Fix SWA eviction/admission invariants, add a long-output pressure test, and
   verify that no request is retracted solely because stale SWA slots remain
   allocated.
3. Port the upstream batched RPA ideas in SGLang-JAX style, including batched
   sequence handling, reusable scheduler metadata, and tuned parameters for
   both Gemma 4 shapes.
4. Add fused gate/up and QKV projections with compatible weight loading and
   quantization mappings.
5. Run isolated A/B benchmarks after each change, followed by at least three
   repetitions of the full 1,000-request workload.
6. Capture a decode-only XProf trace after prefill reaches 1,000 active
   requests, and compare it against an upstream trace collected with the same
   window.

The primary acceptance target should be the single-process SPMD result because
it controls for process architecture. Multiprocess parity can be evaluated
after the model/kernel path is competitive.

## Falcon records

| Purpose | Experiment | Artifact / analysis |
| --- | --- | --- |
| Upstream text, multiprocess DP | `exp-uxr7vn058s` | `art-1t44htxy4y`, `an-26bg18bwur` |
| Upstream text, pure SPMD | `exp-prrux3n44r` | `art-vpbdzhlrjs`, `an-w7jvuy9i1y` |
| SGLang-JAX original limits | `exp-y0kwk0m30c` | `art-5e10oqknvs` |
| SGLang-JAX corrected DP limits | `exp-nkhgv6c82x` | `art-i2u6p665ok`, `an-p683xivmi5` |
| SGLang-JAX shorter-interval ablation and profile | `exp-hjqdhqabqj` | `art-7tsoelfwp6` |
| XProf summary | `exp-hjqdhqabqj` | `an-v1gou1et3t` |
| Scheduler/profile-window forensics | `exp-hjqdhqabqj` | `an-es1zj03ceq` |

All listed experiments and required artifacts reached `SUCCEEDED`/ready state
when this analysis was written on 2026-08-10.
