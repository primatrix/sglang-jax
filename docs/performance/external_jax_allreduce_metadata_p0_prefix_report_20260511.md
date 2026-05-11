# External JAX Allreduce Metadata P0 Prefix 性能报告

更新日期：2026-05-11

## 结论

当前最佳实现是 `exp/external-jax-allreduce-metadata-p0-prefix`。这份报告只保留它和 `sgl-project/sglang-jax main` 的性能对比。

- 正确性：表内所有 case 都做过单条 chat 请求验证，结果 OK。
- Flash：EP8/32/64 在 cc512 下整体持平到小幅正向；EP32 cc256 的 3 组复测均值为 -0.25%，基本在方差范围内。
- Pro-Private：EP32/64 cc512 均有正向收益，EP32 +4.65%，EP64 +2.59%。
- `Median ITL` 是 benchmark 输出里的 median inter-token latency，也就是这里关心的 Medium/Median ITL 指标。

## 实现摘要

`p0_prefix` 只优化 allreduce metadata 里当前 device 的 prefix starts 计算路径，避免对完整 `all_sizes` 做全量 `cumsum` 后再索引。它不引入额外跨设备通信，也不改变 fused MoE 主计算路径。

## 性能对比

| 模型 | EP | 并发 | 版本 | n | Input tokens | Output tokens | Total tokens | Total tok/s | vs main | TTFT (ms) | TPOT (ms) | Median ITL (ms) | 正确性 |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| MiMo-V2-Flash | 32 | 256 | main | 3 | 4,194,304 | 262,144 | 4,456,448 | 14,536.75 | baseline | 97,147.56 | 203.29 | 112.03 | OK |
| MiMo-V2-Flash | 32 | 256 | p0_prefix | 3 | 4,194,304 | 262,144 | 4,456,448 | 14,500.28 | -0.25% | 94,854.36 | 206.32 | 117.02 | OK |
| MiMo-V2-Flash | 8 | 512 | main | 1 | 8,388,608 | 524,288 | 8,912,896 | 11,272.24 | baseline | 381,342.01 | 45.67 | 23.52 | OK |
| MiMo-V2-Flash | 8 | 512 | p0_prefix | 1 | 8,388,608 | 524,288 | 8,912,896 | 11,396.71 | +1.10% | 377,343.87 | 44.78 | 23.66 | OK |
| MiMo-V2-Flash | 32 | 512 | main | 1 | 8,388,608 | 524,288 | 8,912,896 | 15,438.72 | baseline | 239,788.37 | 196.59 | 113.57 | OK |
| MiMo-V2-Flash | 32 | 512 | p0_prefix | 1 | 8,388,608 | 524,288 | 8,912,896 | 15,672.96 | +1.52% | 236,863.89 | 195.45 | 117.63 | OK |
| MiMo-V2-Flash | 64 | 512 | main | 1 | 8,388,608 | 524,288 | 8,912,896 | 20,236.04 | baseline | 116,843.27 | 287.41 | 187.13 | OK |
| MiMo-V2-Flash | 64 | 512 | p0_prefix | 1 | 8,388,608 | 524,288 | 8,912,896 | 20,251.53 | +0.08% | 113,359.71 | 291.87 | 195.86 | OK |
| MiMo-V2-Pro-Private | 32 | 512 | main | 1 | 8,388,608 | 524,288 | 8,912,896 | 14,325.35 | baseline | 289,864.43 | 79.71 | 41.89 | OK |
| MiMo-V2-Pro-Private | 32 | 512 | p0_prefix | 1 | 8,388,608 | 524,288 | 8,912,896 | 14,992.11 | +4.65% | 278,018.86 | 74.93 | 39.48 | OK |
| MiMo-V2-Pro-Private | 64 | 512 | main | 1 | 8,388,608 | 524,288 | 8,912,896 | 14,407.02 | baseline | 262,112.78 | 207.85 | 119.36 | OK |
| MiMo-V2-Pro-Private | 64 | 512 | p0_prefix | 1 | 8,388,608 | 524,288 | 8,912,896 | 14,780.08 | +2.59% | 254,712.25 | 202.45 | 115.01 | OK |

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
- 表里的 token 数、TTFT、TPOT、Median ITL 均来自 bench serving 输出或对应 summary。
