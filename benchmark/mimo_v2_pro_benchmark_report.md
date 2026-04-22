# MiMo-V2-Pro Serving Benchmark Report

## Model & Hardware

| Item | Value |
|------|-------|
| Model | MiMo-V2-Pro (384 experts, top-k=8) |
| Architecture | 70 layers (60 SWA + 10 Full Attention) |
| Hidden size | 6144, Intermediate size 2048 |
| Weight dtype | FP8 (float8_e4m3fn, block_k=128) |
| Hardware | TPU v6e-64 (16 pods x 4 chips = 64 chips) |
| TP size | 64, EP size 64 |
| MoE backend | fused |
| JAX version | 0.8.1 |

## Server Configuration

| Parameter | 16K Config | 4K Config |
|-----------|-----------|-----------|
| chunked-prefill-size | 2048 | 2048 |
| mem-fraction-static | 0.90 | 0.90 |
| page-size | 256 | 256 |
| context-length | 262144 | 262144 |
| swa-full-tokens-ratio | 0.2 | 0.5 |
| max-running-requests | 128 | 35 |
| disable-radix-cache | yes | yes |
| disable-precompile | yes | yes |
| decode-distribution-opt | yes | yes |

**Note**: `swa-full-tokens-ratio` 需要根据 input length 调整。较短的 input 需要更大的 ratio 来避免 SWA pool OOM（详见"SWA Pool 瓶颈分析"章节）。

## Benchmark Results

### 16K input + 1K output (swa_ratio=0.2)

| Metric | R1 (warmup) | R2 (cached) |
|--------|-------------|-------------|
| Successful requests | 256 | 256 |
| Benchmark duration (s) | 768.31 | 768.12 |
| **Output token throughput (tok/s)** | **341.19** | **341.28** |
| Input token throughput (tok/s) | 5459.10 | 5460.49 |
| Total token throughput (tok/s) | 5800.29 | 5801.77 |
| Request throughput (req/s) | 0.33 | 0.33 |
| Mean E2E Latency (ms) | 299254.99 | 299156.10 |
| Median TTFT (ms) | 288608.64 | 288644.74 |
| **Median ITL (ms)** | **36.00** | **35.99** |
| P95 ITL (ms) | 36.76 | 36.74 |
| P99 ITL (ms) | 37.25 | 37.24 |

**Server-side metrics (16K):**

| Metric | Value |
|--------|-------|
| **Peak gen throughput (tok/s)** | **618.93** |
| Steady-state decode batch size | 22 |
| Full token pool size | 569,344 |
| SWA token pool size | 113,664 |
| Full pool usage (decode) | 67% |
| SWA pool usage (decode) | 5-10% |

### 4K input + 1K output (swa_ratio=0.5)

| Metric | R1 (warmup) | R2 (cached) |
|--------|-------------|-------------|
| Successful requests | 256 | 256 |
| Benchmark duration (s) | 352.18 | 353.20 |
| **Output token throughput (tok/s)** | **744.34** | **742.20** |
| Input token throughput (tok/s) | 2977.35 | 2968.81 |
| Total token throughput (tok/s) | 3721.69 | 3711.01 |
| Request throughput (req/s) | 0.73 | 0.72 |
| Mean E2E Latency (ms) | 135261.89 | 135693.25 |
| Median TTFT (ms) | 100413.30 | 100625.33 |
| **Median ITL (ms)** | **33.52** | **33.67** |
| P95 ITL (ms) | 35.39 | 35.61 |
| P99 ITL (ms) | 36.87 | 37.27 |

**Server-side metrics (4K):**

| Metric | Value |
|--------|-------|
| **Peak gen throughput (tok/s)** | **1066.33** |
| Steady-state decode batch size | 35 |
| Full token pool size | 313,088 |
| SWA token pool size | 156,416 |

## Summary

| Workload | Avg Output tok/s | Peak Gen tok/s | Decode BS | Median ITL |
|----------|------------------|----------------|-----------|------------|
| **16K in + 1K out** | 341.28 | **618.93** | 22 | 35.99 ms |
| **4K in + 1K out** | 742.20 | **1066.33** | 35 | 33.67 ms |

### vs MiMo-V2-Flash (reference)

| Model | Hardware | Peak Gen tok/s | Decode BS | Median ITL | DP |
|-------|----------|----------------|-----------|------------|----|
| Flash | 16x v6e (4x4 topo) | 3008 | 64 | 21.85 ms | dp=4 |
| **Pro** | **64x v6e (16 pods)** | **1066** | **35** | **33.67 ms** | **no DP** |

**Key difference**: Flash 使用 dp_size=4 做 data-parallel attention，每 4 卡处理独立的 batch，等效 4x 并发放大。Pro 目前不支持 DP attention 特性，decode batch size 受限于 SWA pool 容量。

## SWA Pool 瓶颈分析

### 问题

Scheduler 的 admission policy (`schedule_policy.py:rem_total_tokens`) 在 hybrid SWA 模式下**只检查 full attention pool 的可用空间**，不检查 SWA pool。当 full pool 仍有大量可用空间时，SWA pool 已耗尽，导致 decode OOM crash。

### 公式

```
swa_max_total_num_tokens = swa_full_tokens_ratio × full_max_total_num_tokens
```

| Ratio | Full Pool | SWA Pool | Max Concurrent (4K input) | Max Concurrent (16K input) |
|-------|-----------|----------|--------------------------|---------------------------|
| 0.2 | 569K | 113K | ~26 (SWA bottleneck) | ~22 (balanced) |
| 0.5 | 313K | 156K | ~35 (SWA bottleneck) | ~18 (full bottleneck) |
| 1.0 | 178K | 178K | ~44 (balanced) | ~10 (full bottleneck) |

### 建议

1. **短期**: 根据 input length 选择合适的 `swa_full_tokens_ratio`
   - 16K input: ratio=0.2 (当前最优)
   - 4K input: ratio=0.5
   - 1K input: ratio=0.8-1.0
2. **长期**: 修复 scheduler 的 `rem_total_tokens` 计算，同时考虑 SWA pool 可用空间

## 提升方向

1. **DP Attention**: 最关键的优化。Flash 通过 dp=4 将 decode batch 放大 4 倍。Pro 如果支持 dp=4，peak 可能达到 ~4200 tok/s
2. **Scheduler SWA-aware admission**: 修复 admission policy，避免 SWA OOM crash，允许更大 batch
3. **SWA token recycling**: 在 SWA pool 中实现 sliding window eviction，释放已滑出窗口的 token
4. **Per-request SWA ratio**: 根据每个 request 的 input length 动态调整 SWA/full 分配

## Benchmark Parameters

```bash
# bench_serving command (both 16K and 4K)
python3 -u -m sgl_jax.bench_serving \
  --backend sgl-jax --host 127.0.0.1 --port 30271 \
  --dataset-name random \
  --random-input-len {16384|4096} \
  --random-output-len 1024 \
  --random-range-ratio 1.0 \
  --flush-cache --seed 12345 \
  --request-rate 100 --num-prompts 256 --max-concurrency 128
```

## Git Branch

```
feat/mimo-pro-weight (rebased onto primatrix/perf/decode-distribution-opt)
Commits:
- 2c884399 feat: add MiMo-V2-Pro model, centralize FP8 dequant in WeightLoader
- 6362d798 fix: unittest failed
- 1d10e5bd perf: use pure decode distribution for decode-only batches
- 32c87848 perf: optimize get_forward_metadata for SWA models (#930)
- 55a85210 add tuned block configs for hidden_size=6144 intermediate_size=2048 n_experts=384 model (#932)
```
