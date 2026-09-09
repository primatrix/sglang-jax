# CSA attention layer A/B on TPU

`tpu_csa_layer_benchmark.py` invokes the production `DeepseekV4Attention` module
with real static FP8 attention weights from checkpoint layer 2 (CSA). It includes
Q/KV and indexer projections, normalization/RoPE, both compressors, gathered KV,
indexer/top-k, attention, output projections, and pool updates. It excludes the
FFN, mHC, other decoder layers, tokenizer, scheduler, and HTTP server.

The default mesh is TP8 / DP1. Hidden states, historical KV and continuation
state are deterministic synthetic inputs, shared exactly between revisions.
Long-history inputs are seeded directly into real runtime pools and the allocator
page ledger; they are not represented as having been produced by model prefill.
This tests layer performance and numerical differences, not model accuracy.

## Run

Run the same script in separate processes against two source-pinned worktrees,
using the same checkpoint and TPU configuration. Keep the outputs under a common
parent as `baseline/` and `candidate/`.

```bash
PYTHONPATH=python:test/manual/deepseek_v4_precision python \
  test/manual/deepseek_v4_precision/tpu_csa_layer_benchmark.py \
  --model /model --out /artifacts/baseline --tp 8 --repeats 7 --profile
```

Repeat on the candidate worktree with `--out /artifacts/candidate`, then compare:

```bash
PYTHONPATH=test/manual/deepseek_v4_precision python \
  test/manual/deepseek_v4_precision/tpu_csa_layer_compare.py /artifacts
```

`--cases` selects any subset:

| Case | Live / padded requests | Historical tokens per request | New tokens per request | Token bucket |
| --- | ---: | ---: | ---: | ---: |
| short18 | 1 / 1 | 0 | 18 | 128 |
| prefill2k | 1 / 64 | 0 | 2048 | 2048 |
| extend2k | 1 / 64 | 6144 | 2048 | 2048 |
| decode32 | 32 / 32 | 8192 | 1 | 32 |
| decode64 | 64 / 64 | 8192 | 1 | 64 |
| decode32_steady | 32 / 32 | 8224 | 1 | 32 |
| decode64_steady | 64 / 64 | 8224 | 1 | 64 |
| decode32_complete | 32 / 32 | 8227 | 1 | 32 |
| decode29_padded | 29 / 32 | 8227 | 1 | 32 |
| decode1_empty | 1 / 1 | 2 | 1 | 1 |
| decode1_first | 1 / 1 | 3 | 1 | 1 |

The 8192-token decode cases sit immediately below a compressed-capacity bucket
transition. The 8224-token cases cover the larger bucket after more decoding.
Report both; the initial bucket's speedup does not describe all later steps.
The 8227-token cases complete a new compression group during the measured call.
The final two cases cover zero and one completed group respectively; both have
a valid query and include the current-token SWA write.

## Measurements and checks

Each case lowers/compiles separately and warms up twice before collecting seven
synchronized host-wall timings. Compilation is reported separately. Pool buffers
are donated to the production update path. Every repetition receives a fresh
clone of the same pool snapshot; cloning and synchronization of that clone happen
outside the measured interval. Three optional profiled calls also use snapshots
prepared before tracing. No data-dependent model work is replaced by an oracle.

`result.json` records source SHA, checkpoint identity, exact input hashes, pool
output hashes, candidate extents, compile duration and all warm timing samples.
`weights.json` records every loaded weight payload hash. The comparison requires
identical checkpoint/weights/input hashes and finite outputs, then reports output
relative L2, worst-token errors, cosine, cache/state bitwise equality and speedup.
It deliberately does not turn an arbitrary numerical tolerance into a model
quality verdict. A GSM8K regression remains a separate acceptance failure.

`*-output.npy` stores complete live-token layer outputs as portable FP32.
`*-state-<family>-<index>.npy` stores the small FP32 continuation arrays, allowing
state hash differences to be quantified independently of BF16 cache writes.
`--profile` creates the standard `<case>/plugins/profile/<timestamp>/` hierarchy.
CPU execution is rejected by the benchmark; the small CPU harness sanity check
only validates resource construction and the donation/update call structure.

## Measured A/B (2026-09-09)

Baseline `ba6c9f9918ba96ea6f957db73ed6ea3add518088`; candidate `bea9ccd09cb22a33c8c1a59344b8d336dbb8f148`.
Falcon runs: `exp-inw0gsb5ok` (first five cases), `exp-ywtatvfcas` (steady decode).
TPU v7x, TP8, checkpoint `DeepSeek-V4-Flash-0731-SGLang-JAX-Expert-FP8-v1`.
Values below are medians of seven synchronized host-wall samples in milliseconds.

| Case | Baseline ms | Candidate ms | Speedup | Output relative L2 |
| --- | ---: | ---: | ---: | ---: |
| short18 | 1.441 | 1.078 | 1.34x | 0 |
| prefill2k | 501.125 | 4.720 | 106.18x | 0.000545 |
| extend2k | 500.963 | 7.077 | 70.79x | 0.000572 |
| decode32 | 16.619 | 4.429 | 3.75x | 0.000179 |
| decode64 | 31.194 | 7.917 | 3.94x | 9.11e-05 |
| decode32_steady | 16.487 | 12.509 | 1.32x | 0 |
| decode64_steady | 30.987 | 23.852 | 1.30x | 0 |

All seven cases matched input/weight hashes and updated cache/state hashes exactly.
Short18 and both steady-decode outputs were bitwise identical; the other four
outputs had small nonzero differences. This does not establish full-model parity.
The paired full-model arithmetic warmup separately returned 5 on baseline and 6
on candidate, so the layer results alone do not satisfy merge acceptance.

Chrome XProf Trace Viewer independently showed one candidate prefill2k
`jit_forward` execution at 4.327938 ms on TPU:0 (one device slice, not the median
host-wall measurement). Its timeline includes projection, CSA indexer/top-k and
the two attention contractions.

## Request-local decode after PR369 (2026-09-10)

PR369 was merged as `f6ba234209251c4e37373efeca0862f1bd4dd8f1`; its previously
reported full-model quality differences were accepted by the operator. This
follow-up reports its own numerical differences and does not claim to fix them.

Baseline is that merge; candidate production change is
`9300784a1dcaf787e08b3cd8054a70cc875db641`.
Falcon `exp-2dms60bkr3`, artifact `art-ehk709xlns`, v7x TP8/DP1,
JAX 0.11.1, libtpu 0.0.46.1, same real static Expert-FP8 attention weights.
Independent artifact analysis `an-23qi6efph4` re-read both output arrays and
verified matching checkpoint, weight and input hashes. All values are medians
of seven synchronized host-wall samples; CPU metadata construction is excluded.

| Case | Baseline ms | Candidate ms | Speedup | Output rel L2 | Worst token rel L2 | Cache/state bitwise equal |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| short18 | 1.059 | 1.097 | 0.97x | 0 | 0 | True |
| prefill2k | 4.712 | 4.687 | 1.01x | 0 | 0 | True |
| extend2k | 7.065 | 7.033 | 1.00x | 0 | 0 | True |
| decode32 | 4.505 | 1.273 | 3.54x | 0.000262 | 0.00108 | False |
| decode64 | 7.893 | 1.507 | 5.24x | 0.000386 | 0.00264 | True |
| decode32_steady | 12.215 | 1.247 | 9.79x | 0.00024 | 0.00105 | False |
| decode64_steady | 23.788 | 1.641 | 14.50x | 0.000222 | 0.000821 | True |

Both sides retain the same global padded extent in these comparisons. The decode
candidate scores per request at capacity 2048 (prefix 8192) or 4096 (prefix 8224),
then gathers selected compressed entries plus the SWA window. It retains native
exact top-k, completed-group visibility, invalid selection masking, and the sink
denominator. Prefill uses its existing implementation; its outputs and state
are bitwise unchanged in this run. The short18 timing is 3.6% slower in this
single paired run despite an unchanged path, and is not claimed as an improvement.

The BS32 hash differences are isolated to the FP32 CSA compressor continuation
state (pytree leaf 0). All BF16 KV/cache leaves and indexer continuation state
match exactly. A separate boundary run saves continuation arrays for attribution.
Decode outputs differ by at most 0.005859375 in this run; no nonfinite output
occurs. These are layer numerical measurements, not a full-model quality verdict.

### Trace Viewer evidence

The original prefix8224/BS64 profile exposed two full-history gather waits of
5.109 and 5.502 ms, gathering BF16 [262144,128] index keys and [262144,512]
compressed KV. Their asynchronous offload lifetimes overlap other work and must
not be added to scorer/sort durations as sequential time.

The new candidate profile was inspected directly in Chrome:
[BS64 steady Trace Viewer](https://falcon.infiscale-infra.org/v1/viewers/tpu-training-antgroup/data/plugin/profile/trace_viewer@;run=0c9b876178d25944%2F2026_09_09_16_15_50;tag=trace_viewer@;hosts=falcon-job-hc2u99cfri-workers-0-0?run=0c9b876178d25944%2F2026_09_09_16_15_50&tag=trace_viewer@&hosts=falcon-job-hc2u99cfri-workers-0-0).
The TPU:0 `jit_forward(5138153974612490034)` slice starts at 145060470 ns and
lasts **1.042411 ms**. This device slice is distinct from the 1.640942 ms host
median. The former multi-millisecond gather waits are absent from this timeline.
The visible `top_k.12` is a native sort over f32/s32 [64,4096], lasting 105974 ns;
`gather_fusion.16` lasts 200901 ns and maps to `decode.py`'s `take_along_axis`
page-address lookup, with s32 [32768] output. That small-address gather is a
remaining opportunity, rather than the old full-history BF16 gather. A neighboring
`region.230` lasts 209899 ns; its detailed context did not load, so its duration
alone is not evidence of DMA/compute overlap.

### Focused probes

```bash
PYTHONPATH=python python test/manual/deepseek_v4_precision/csa_decode_tables_probe.py
PYTHONPATH=python python test/manual/deepseek_v4_precision/csa_decode_probe.py --large --require-tpu --out probe.json
```

The table probe compares addresses with the original flat tables across two
DP-local geometries, permuted physical pages/slots and inactive requests. It is
host-only evidence, not a DP2 device run. The numerical probe compares the Pallas
scorer, selected sets and attention with the original native implementation,
including partial pages, score ties, BF16/FP32 index queries and extreme sinks.
The large TPU probe in the first experiment matched valid scores and selected
sets exactly; maximum raw attention output relative L2 was below 2.9e-8. CPU
interpretation also passed. This does not replace the complete real-weight layer
A/B, and no CPU timing is used for performance claims.

### Independent boundary and repeat run

Falcon `exp-apygdudr3n` / `art-6se3w37ixi` completed successfully against
candidate `2f248280c` (same production math, stricter required metadata fields,
plus benchmark state dumps). Independent analysis `an-ftp3aoi71s` checked all
output arrays and the 12 saved FP32 state arrays.

| Case | Baseline ms | Candidate ms | Speedup | Output rel L2 | Worst token rel L2 | Cache/state bitwise equal |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| decode32_steady | 12.059 | 1.322 | 9.12x | 0.00024 | 0.00105 | False |
| decode32_complete | 12.246 | 1.299 | 9.43x | 0.000206 | 0.00053 | False |
| decode29_padded | 4.099 | 1.415 | 2.90x | 0.000241 | 0.000624 | True |
| decode1_empty | 0.850 | 0.878 | 0.97x | 0 | 0 | True |
| decode1_first | 0.853 | 0.867 | 0.98x | 0 | 0 | True |
| decode64_steady | 23.637 | 1.663 | 14.21x | 0.000222 | 0.000821 | True |

Only two of the twelve continuation arrays differed: CSA c4 state for BS32
steady and BS32 completing a new group. Their maximum absolute differences
were 1.4305115e-6 and 1.9073486e-6, relative L2 6.7671e-8 and 6.9049e-8.
Nonfinite patterns matched. All BF16 cache leaves and indexer states were
bitwise identical, including the new-group completion cases. This quantifies
the change; it does not establish its compiler-level cause or long-run impact.
Zero/one-group valid decode outputs were bitwise identical. Small single-query
latencies varied by +3.2% / +1.6%; this change targets batched long-history decode.

Scope: ordinary decode with page_size=128. Page_size=256 and prefill retain the
existing path. DP-local address geometry is CPU-tested; device A/B covers DP1
only. CPU metadata building still creates the flat fallback tables and is
excluded from reported forward timings, so these speedups must not be quoted
as server throughput gains. Full-model quality and multi-step recurrent-state
accumulation remain unmeasured for this follow-up. The PR is left for review
and is not automatically merged.
