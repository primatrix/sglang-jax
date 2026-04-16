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
    --moe-backend fused \
    --host 0.0.0.0 --port 30271 \
    --page-size 256 \
    --context-length 262144 \
    --disable-radix-cache \
    --chunked-prefill-size 4096 \
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
Benchmark duration (s):                  1685.61
Total input tokens:                      10485760
Total input text tokens:                 10485760
Total generated tokens:                  655360
Total generated tokens (retokenized):    654295
Request throughput (req/s):              0.38
Input token throughput (tok/s):          6220.74
Output token throughput (tok/s):         388.80
Peak output token throughput (tok/s):    2304.00
Peak concurrent requests:                128
Total token throughput (tok/s):          6609.54
Concurrency:                             63.98
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   168504.57
Median E2E Latency (ms):                 168286.95
P90 E2E Latency (ms):                    169042.09
P99 E2E Latency (ms):                    169643.17
---------------Time to First Token----------------
Mean TTFT (ms):                          68568.40
Median TTFT (ms):                        68655.19
P99 TTFT (ms):                           135053.26
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          97.69
Median TPOT (ms):                        97.81
P99 TPOT (ms):                           162.61
---------------Inter-Token Latency----------------
Mean ITL (ms):                           97.69
Median ITL (ms):                         29.50
P95 ITL (ms):                            31.98
P99 ITL (ms):                            38.66
Max ITL (ms):                            133771.47
==================================================
```

### Max Concurrency = 128

```
============ Serving Benchmark Result ============
Backend:                                 sgl-jax
Traffic request rate:                    100.0
Max request concurrency:                 128
Successful requests:                     1280
Benchmark duration (s):                  3460.01
Total input tokens:                      20971520
Total input text tokens:                 20971520
Total generated tokens:                  1310720
Total generated tokens (retokenized):    1308226
Request throughput (req/s):              0.37
Input token throughput (tok/s):          6061.12
Output token throughput (tok/s):         378.82
Peak output token throughput (tok/s):    2048.00
Peak concurrent requests:                256
Total token throughput (tok/s):          6439.93
Concurrency:                             127.96
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   345896.60
Median E2E Latency (ms):                 345938.90
P90 E2E Latency (ms):                    346578.04
P99 E2E Latency (ms):                    347236.33
---------------Time to First Token----------------
Mean TTFT (ms):                          136690.87
Median TTFT (ms):                        136889.15
P99 TTFT (ms):                           269546.66
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          204.50
Median TPOT (ms):                        204.42
P99 TPOT (ms):                           334.24
---------------Inter-Token Latency----------------
Mean ITL (ms):                           204.50
Median ITL (ms):                         69.60
P95 ITL (ms):                            76.96
P99 ITL (ms):                            86.37
Max ITL (ms):                            270417.62
==================================================
```

## Summary

| Metric | Concurrency 64 | Concurrency 128 |
|--------|----------------|-----------------|
| Successful requests | 640 | 1,280 |
| Benchmark duration (s) | 1,685.61 | 3,460.01 |
| Request throughput (req/s) | 0.38 | 0.37 |
| Input token throughput (tok/s) | 6,220.74 | 6,061.12 |
| Output token throughput (tok/s) | 388.80 | 378.82 |
| Peak output throughput (tok/s) | 2,304.00 | 2,048.00 |
| Total token throughput (tok/s) | 6,609.54 | 6,439.93 |
| Mean E2E Latency (ms) | 168,504.57 | 345,896.60 |
| Mean TTFT (ms) | 68,568.40 | 136,690.87 |
| Median TTFT (ms) | 68,655.19 | 136,889.15 |
| Mean TPOT (ms) | 97.69 | 204.50 |
| Median ITL (ms) | 29.50 | 69.60 |
| P99 ITL (ms) | 38.66 | 86.37 |
