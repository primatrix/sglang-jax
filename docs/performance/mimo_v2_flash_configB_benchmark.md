# MiMo-V2-Flash Config B Benchmark Results

## Test Configuration

- **Test Date**: April 16, 2026
- **Test Environment**: TPU v7x, 4 chips (2x2x1 topology), single-host
- **Pod**: brian-moe-cherry-pick
- **Model**: MiMo-V2-Flash
- **Branch**: `mimo-v2-test-configB` (commit: eb641405 — feat: swap FP8 configs to zipf (Config B) for serving benchmark)
- **Framework**: SGLang-JAX (sgl-jax)
- **Dtype**: bfloat16 (FP8 quantized MoE experts with zipf Config B)
- **Input Length**: 16384 tokens
- **Output Length**: 1024 tokens

### Server Configuration

```bash
JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache python -u -m sgl_jax.launch_server \
    --model-path /models/MiMo-V2-Flash \
    --trust-remote-code \
    --tp-size 8 --dp-size 2 --ep-size 8 \
    --moe-backend epmoe \
    --host 0.0.0.0 --port 30271 \
    --page-size 256 \
    --context-length 262144 \
    --disable-radix-cache \
    --chunked-prefill-size 2048 \
    --dtype bfloat16 \
    --mem-fraction-static 0.95 \
    --swa-full-tokens-ratio 0.2 \
    --skip-server-warmup \
    --log-level info \
    --max-running-requests 128 \
    --dp-schedule-policy round_robin
```

## Benchmark Results

### Max Concurrency = 64

```
============ Serving Benchmark Result ============
Backend:                                 sgl-jax
Traffic request rate:                    100.0
Max request concurrency:                 64
Successful requests:                     640
Benchmark duration (s):                  1415.23
Total input tokens:                      10485760
Total input text tokens:                 10485760
Total generated tokens:                  655360
Total generated tokens (retokenized):    655584
Request throughput (req/s):              0.45
Input token throughput (tok/s):          7409.22
Output token throughput (tok/s):         463.08
Peak output token throughput (tok/s):    1856.00
Peak concurrent requests:                128
Total token throughput (tok/s):          7872.30
Concurrency:                             63.97
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   141467.69
Median E2E Latency (ms):                 141405.74
P90 E2E Latency (ms):                    142186.08
P99 E2E Latency (ms):                    142401.71
---------------Time to First Token----------------
Mean TTFT (ms):                          50688.08
Median TTFT (ms):                        50722.28
P99 TTFT (ms):                           100539.44
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          88.74
Median TPOT (ms):                        88.87
P99 TPOT (ms):                           136.85
---------------Inter-Token Latency----------------
Mean ITL (ms):                           88.74
Median ITL (ms):                         37.34
P95 ITL (ms):                            41.04
P99 ITL (ms):                            48.02
Max ITL (ms):                            100647.77
==================================================
```

### Max Concurrency = 128

```
============ Serving Benchmark Result ============
Backend:                                 sgl-jax
Traffic request rate:                    100.0
Max request concurrency:                 128
Successful requests:                     1280
Benchmark duration (s):                  2760.69
Total input tokens:                      20971520
Total input text tokens:                 20971520
Total generated tokens:                  1310720
Total generated tokens (retokenized):    1311142
Request throughput (req/s):              0.46
Input token throughput (tok/s):          7596.49
Output token throughput (tok/s):         474.78
Peak output token throughput (tok/s):    2162.00
Peak concurrent requests:                256
Total token throughput (tok/s):          8071.27
Concurrency:                             127.95
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   275964.49
Median E2E Latency (ms):                 275843.00
P90 E2E Latency (ms):                    277000.88
P99 E2E Latency (ms):                    278603.46
---------------Time to First Token----------------
Mean TTFT (ms):                          101933.41
Median TTFT (ms):                        101849.90
P99 TTFT (ms):                           202392.32
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          170.12
Median TPOT (ms):                        170.36
P99 TPOT (ms):                           266.45
---------------Inter-Token Latency----------------
Mean ITL (ms):                           170.12
Median ITL (ms):                         67.40
P95 ITL (ms):                            83.69
P99 ITL (ms):                            93.96
Max ITL (ms):                            203191.12
==================================================
```

## Summary

| Metric | Concurrency 64 | Concurrency 128 |
|--------|----------------|-----------------|
| Successful requests | 640 | 1,280 |
| Benchmark duration (s) | 1,415.23 | 2,760.69 |
| Request throughput (req/s) | 0.45 | 0.46 |
| Input token throughput (tok/s) | 7,409.22 | 7,596.49 |
| Output token throughput (tok/s) | 463.08 | 474.78 |
| Peak output throughput (tok/s) | 1,856.00 | 2,162.00 |
| Total token throughput (tok/s) | 7,872.30 | 8,071.27 |
| Mean E2E Latency (ms) | 141,467.69 | 275,964.49 |
| Mean TTFT (ms) | 50,688.08 | 101,933.41 |
| Median TTFT (ms) | 50,722.28 | 101,849.90 |
| Mean TPOT (ms) | 88.74 | 170.12 |
| Median ITL (ms) | 37.34 | 67.40 |
| P99 ITL (ms) | 48.02 | 93.96 |
