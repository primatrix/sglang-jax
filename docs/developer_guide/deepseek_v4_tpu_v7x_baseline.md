# DeepSeek-V4-Flash on TPU v7x 2x2x1 with sglang-jax: how the baseline is measured

This page describes how the numbers quoted for `epic/dsv4` at 78c8d9ef0 were
measured on one TPU v7x 2x2x1 host, so that anyone with the same hardware can
reproduce them. It documents a measurement method; it is not an offer to run or
benchmark other people's code.

## Setup

- Hardware: one TPU v7x host, 2x2x1 topology (4 chips, 8 JAX devices), `tp8 /
  ep8`, single DP rank. The pod requests `google.com/tpu: 4` with
  `TPU_PROCESS_BOUNDS=1,1,1` and `TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1`,
  `JAX_PLATFORMS=tpu`.
- Code: `primatrix/sglang-jax` branch `epic/dsv4` at 78c8d9ef0, code defaults
  only (no `DSV4_*` feature environment variables set).
- Checkpoint: the static expert-FP8 export of DeepSeek-V4-Flash produced by
  `sgl_jax.srt.utils.quantization.deepseek_v4_static_fp8`
  (`python/sgl_jax/srt/utils/quantization/deepseek_v4_static_fp8.py`). Routed
  trunk experts are E4M3 with FP32 power-of-two scales; all other tensors keep
  their original bytes. KV cache is BF16.
- Compilation cache: `JAX_COMPILATION_CACHE_DIR` points at a persistent
  directory that survives pod restarts. With a warm cache the single-stream
  chain has no cold compile; without it the first 32K request alone compiles for
  roughly 200 s and must be discarded.
- The server process is pinned to the CPUs of one NUMA node with `taskset`
  (host-side path latency is otherwise higher and varies between hosts).
- The client is a separate pod in the same cluster; TTFT therefore includes
  about 1 ms of network round trip.

## Server launch flags

Single-stream gate ("regr"):

```
python -m sgl_jax.launch_server \
  --model-path <checkpoint> --trust-remote-code --device tpu \
  --tp-size 8 --ep-size 8 --moe-backend epmoe --dtype bfloat16 \
  --context-length 262144 --chunked-prefill-size 8192 --max-prefill-tokens 32768 \
  --max-total-tokens 1048576 --max-running-requests 16 --page-size 128 \
  --mem-fraction-static 0.8 --disable-radix-cache \
  --skip-server-warmup --watchdog-timeout 7200 \
  --precompile-token-paddings 256 8192 --precompile-bs-paddings 1 16 \
  --host 0.0.0.0 --port 30000
```

Throughput gate ("regrt"):

```
python -m sgl_jax.launch_server \
  --model-path <checkpoint> --trust-remote-code --device tpu \
  --tp-size 8 --ep-size 8 --moe-backend epmoe --dtype bfloat16 \
  --context-length 65536 --chunked-prefill-size 8192 --max-prefill-tokens 32768 \
  --max-total-tokens 4194304 --max-running-requests 112 --page-size 128 \
  --mem-fraction-static 0.8 --disable-radix-cache \
  --skip-server-warmup --watchdog-timeout 7200 \
  --precompile-token-paddings 256 8192 --precompile-bs-paddings 1 16 32 64 112 \
  --host 0.0.0.0 --port 30000
```

The `max-running-requests 160` working point uses the throughput flags with
`--max-running-requests 160 --precompile-bs-paddings 1 16 32 64 160`.
`--disable-radix-cache` is required by the DSv4 pool initialiser.

## Benchmark client parameters

All `bench_serving` runs use `python -m sgl_jax.bench_serving --backend sgl-jax
--dataset-name random --random-range-ratio 1` with the default seed and
`--warmup-requests 4` unless stated otherwise. Readiness is `GET
/get_server_info` returning 200.

Single-stream chain, in this order against the "regr" server:

| Step | in / out | num-prompts | max-concurrency | warmup | Reported |
| --- | --- | --- | --- | --- | --- |
| warm | 1024 / 32, then 8192 / 32 | 2 each | 1 | 1 | discarded |
| 8K TTFT | 8192 / 64 | 4 | 1 | 2 | Median TTFT, run twice, both passes reported |
| 32K TTFT | 32768 / 1 | 1 request, 3 repeats | 1 | none | server e2e latency of repeats 1 and 2 |
| GSM8K-40 | see below | 40 | 1 | none | correct / 40, truncations |
| cc=1 TPOT | 1024 / 128 | 8 | 1 | 2 | Median TPOT |

The 32K request is not a `bench_serving` run. It posts to `/generate` with
`input_ids` (real text tokenized to 32768 tokens including the native chat
template), `temperature 0`, `max_new_tokens 1`, and reads
`meta_info.e2e_latency`, which is TTFT plus one decode step. Each repeat uses a
different text offset so the request is fresh; repeat 0 is discarded because it
may carry compilation, repeats 1 and 2 are reported.

Throughput chain against the "regrt" server:

| Step | in / out | num-prompts | max-concurrency | warmup | Reported |
| --- | --- | --- | --- | --- | --- |
| warm | 1024 / 256 | 128 | 64 | 8 | discarded |
| cc=64 | 8192 / 1024 | 256 | 64 | 4 | Total token throughput, Median TPOT, Median TTFT, P99 TPOT |
| cc=256 | 8192 / 1024 | 512 | 256 | 4 | same |

The cc=64 and cc=256 runs are executed as two full passes (cc=64 then cc=256,
twice). The first pass carries the compilation of any batch shape not yet cached
and is discarded; the second pass is the reported number. An optional device
profile capture at cc=64 (1024 / 256) sits between the warm-up and the measured
passes and does not affect the reported numbers.

## Reporting rules

- Two passes, second pass reported. A first pass that includes compilation is
  never quoted.
- Configuration read-back: before quoting, the effective configuration is read
  from `GET /get_server_info` (context length, chunked prefill size, max running
  requests, max total tokens, page size, radix cache state, speculative
  settings) and must match the launch flags above. Numbers are attributed to the
  read-back values, not to the command line.
- Independent runs: a working point is only called reproducible after at least
  two independent server starts agree within 5 percent on throughput, TPOT and
  TTFT.

## GSM8K gate

The correctness gate is GSM8K, first N questions of the test set in file order,
N = 40 in the single-stream chain and N = 200 for the full gate.

- Prompting: each question is wrapped in an instruction asking for reasoning
  steps and a final line of the form `Answer: <number>`, encoded with the
  checkpoint's native chat encoding (`encoding_dsv4.encode_messages`, chat
  thinking mode) and sent as `input_ids` to `/generate`.
- Decoding: greedy (`temperature 0`), `max_new_tokens 8192`, concurrency 1.
- Scoring: the number after the last `Answer:` is compared with the reference
  after normalising commas and trailing zeros.
- Truncation: a request that ends with `finish_reason == length` counts as a
  failure of the gate regardless of the parsed answer; the gate requires zero
  truncations.
- Long-context variant: the same 200 questions with a deterministic filler
  prefix (later GSM8K questions, marked as unrelated background) prepended until
  the prompt reaches the target prefix length, so that every request decodes at
  long context. Short and long variants are reported as two counts.

## Numbers at 78c8d9ef0

| Metric | Value |
| --- | --- |
| cc=1 TPOT (1024 / 128) | 7.61 ms |
| 8K TTFT (8192 / 64, cc=1) | 263 ms (two passes 263.28 / 263.23) |
| 32K TTFT (single request, repeats 1 / 2) | 1.106 s / 1.102 s |
| GSM8K-40 (greedy, 8192) | 40 / 40, 0 truncations |
| GSM8K-200 short context (greedy, 8192) | 192 / 200, 0 truncations |
| GSM8K-200 long context (same questions after an 8K-token filler prefix) | 196 / 200, 0 truncations |
| cc=64, 8192 / 1024, 256 prompts | 13,088 tok/s total, TPOT 34.28 ms, TTFT 9.49 s, P99 TPOT 41.45 ms |
| cc=256, 8192 / 1024, 512 prompts, max-running-requests 112 | 12,838 tok/s total, TPOT 54.90 ms, TTFT 101.5 s, P99 TPOT 69.22 ms |

Working point `max-running-requests 160` at cc=256 (8192 / 1024, 512 prompts) on
78c8d9ef0: 13,929 tok/s total, TPOT 63.61 ms, TTFT 52.2 s (P99 TPOT 82.7 ms);
the cc=64 pass on the same server gave 12,819 tok/s, TPOT 34.27 ms, TTFT 10.1 s.
Repeated server starts with this shape reproduce the cc=256 throughput within
0.5 percent.
