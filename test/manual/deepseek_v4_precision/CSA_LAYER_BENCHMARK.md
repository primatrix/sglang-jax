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

The 8192-token decode cases sit immediately below a compressed-capacity bucket
transition. The 8224-token cases cover the larger bucket after more decoding.
Report both; the initial bucket's speedup does not describe all later steps.

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
