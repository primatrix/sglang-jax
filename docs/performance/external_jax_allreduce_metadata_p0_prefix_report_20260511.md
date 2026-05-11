# External JAX Allreduce Metadata P0 Prefix 性能报告

更新日期：2026-05-11

## 结论

当前最佳实现是 `exp/external-jax-allreduce-metadata-p0-prefix`。这份报告只保留它和 `sgl-project/sglang-jax main` 的性能对比。

- 正确性：表内所有 case 都做过单条 chat 请求验证，结果 OK。
- Flash：EP8/32/64 在 cc512 下整体持平到小幅正向；EP32 cc256 的 3 组复测均值为 -0.25%，基本在方差范围内。
- Pro-Private：EP32/64 cc512 均有正向收益，EP32 +4.65%，EP64 +2.59%。
- 表格只保留 benchmark 原始输出里的关键吞吐和 median latency 指标。

## 实现摘要

`p0_prefix` 只优化 allreduce metadata 里当前 device 的 prefix starts 计算路径，避免对完整 `all_sizes` 做全量 `cumsum` 后再索引。它不引入额外跨设备通信，也不改变 fused MoE 主计算路径。

## 性能对比

| 模型 | EP | 并发 | 版本 | n | Input token throughput (tok/s) | Output token throughput (tok/s) | Output vs main | Peak output token throughput (tok/s) | Median E2E Latency (ms) | Median TTFT (ms) | Median TPOT (ms) | Median ITL (ms) | 正确性 |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| MiMo-V2-Flash | 32 | 256 | main | 3 | 13,681.65 | 855.10 | baseline | 2,560.00 | 305,121.41 | 96,253.20 | 204.22 | 112.03 | OK |
| MiMo-V2-Flash | 32 | 256 | p0_prefix | 3 | 13,647.32 | 852.96 | -0.25% | 2,389.33 | 305,924.24 | 93,975.41 | 207.23 | 117.02 | OK |
| MiMo-V2-Flash | 8 | 512 | main | 1 | 10,609.17 | 663.07 | baseline | 2,025.00 | 440,655.32 | 385,426.95 | 44.49 | 23.52 | OK |
| MiMo-V2-Flash | 8 | 512 | p0_prefix | 1 | 10,726.32 | 670.39 | +1.10% | 2,112.00 | 435,406.99 | 381,241.62 | 43.60 | 23.66 | OK |
| MiMo-V2-Flash | 32 | 512 | main | 1 | 14,530.56 | 908.16 | baseline | 3,200.00 | 357,522.29 | 191,395.99 | 185.83 | 113.57 | OK |
| MiMo-V2-Flash | 32 | 512 | p0_prefix | 1 | 14,751.02 | 921.94 | +1.52% | 3,400.00 | 356,354.85 | 186,823.12 | 184.57 | 117.63 | OK |
| MiMo-V2-Flash | 64 | 512 | main | 1 | 19,045.68 | 1,190.36 | baseline | 3,472.00 | 410,213.12 | 111,316.08 | 292.01 | 187.13 | OK |
| MiMo-V2-Flash | 64 | 512 | p0_prefix | 1 | 19,060.26 | 1,191.27 | +0.08% | 3,472.00 | 411,333.06 | 107,607.99 | 296.72 | 195.86 | OK |
| MiMo-V2-Pro-Private | 32 | 512 | main | 1 | 13,482.68 | 842.67 | baseline | 2,496.00 | 377,196.77 | 292,170.17 | 80.26 | 41.89 | OK |
| MiMo-V2-Pro-Private | 32 | 512 | p0_prefix | 1 | 14,110.22 | 881.89 | +4.65% | 2,808.00 | 360,018.44 | 279,896.99 | 75.45 | 39.48 | OK |
| MiMo-V2-Pro-Private | 64 | 512 | main | 1 | 13,559.54 | 847.47 | baseline | 3,104.00 | 386,604.09 | 212,668.08 | 195.57 | 119.36 | OK |
| MiMo-V2-Pro-Private | 64 | 512 | p0_prefix | 1 | 13,910.67 | 869.42 | +2.59% | 3,199.00 | 375,373.71 | 206,977.91 | 191.41 | 115.01 | OK |

## 原始结果

- Flash EP32 cc256：`/tmp/sglang_allreduce_prefix_validate_ep32_cc256_20260511/summary_ep32.jsonl`
- Flash EP8 cc512：`/tmp/sglang_allreduce_cc512_flash_ep8_20260511/summary_ep8.jsonl`
- Flash EP32 cc512：`/tmp/sglang_allreduce_cc512_flash_ep32_20260511/summary_ep32.jsonl`
- Flash EP64 cc512：`/tmp/sglang_allreduce_cc512_flash_ep64_20260511/summary_ep64.jsonl`
- Pro-Private EP32 cc512：`/tmp/sglang_allreduce_cc512_pro_ep32_20260511/summary_ep32.jsonl`
- Pro-Private EP64 cc512：`/tmp/sglang_allreduce_cc512_pro_ep64_20260511/summary_ep64.jsonl`

## 备注

- EP32 cc256 是 3 组复测，表内数值为均值。
- cc512 当前是 smoke 结果，每个 case 1 组。
- 表里的 `Input token throughput (tok/s)`、`Output token throughput (tok/s)`、`Peak output token throughput (tok/s)`、`Median E2E Latency (ms)`、`Median TTFT (ms)`、`Median TPOT (ms)`、`Median ITL (ms)` 均来自 bench serving 原始输出。
