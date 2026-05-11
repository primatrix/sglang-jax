# 外部 JAX Allreduce Metadata P0 Prefix 性能报告

更新日期：2026-05-11

## 结论

当前最佳实现是 `exp/external-jax-allreduce-metadata-p0-prefix`。这份报告只对比 `main` 和当前 `p0_prefix`，不再混入历史 `fixed` 数据。

- `p0_prefix` 代码对应测试 commit：`3eeed3cc`，后续报告提交不改变代码。
- 正确性：表内所有 `p0_prefix` case 都做过单条 chat 请求验证，结果 OK。
- `Δ` 统一表示相对 `main` 的改善：吞吐类指标越高越好，延迟类指标越低越好；正数表示更好，负数表示退化。
- Flash 小模型上的收益不显著，整体更接近持平或随 EP/并发波动：EP64 的 64/128 并发收益明显，但 EP8 的 64/128 单组略低，EP32 大多在噪声范围内持平。
- Pro-Private 上的正向趋势更稳定，尤其是较高并发下更明显：EP32/64 的 64/128/256/512 并发 output throughput 全部正向，Median E2E/TTFT/TPOT 多数同步改善。
- 当前证据支持继续推进 `p0_prefix` 作为候选优化；但除 Flash EP32/256 外，多数 `p0_prefix` case 仍是 1 组结果，合并前建议对 Pro 关键 case 做 3 到 5 组同时间复测。

## 统计口径

- 严格可做重复样本判断的是 Flash EP32/256：`main n=3`，`p0_prefix n=3`，output throughput 为 `855.10 -> 852.96 tok/s`，差异 `-0.25%`；Welch t 约 `-0.18`，不能证明收益，也不能证明退化。
- 其余 64/128/512 并发的 `p0_prefix` 多数是单组结果，因此只作为方向性判断，不写成严格统计显著结论。
- 从方向性看，主表 18 个 case 里 output throughput 有 15 个正向；其中 Pro-Private 8 个 case 全部正向，Flash 则混合了正向、持平和小幅负向。
- 因此当前可统计地表述为：Flash 小模型收益不显著；Pro-Private，特别是较高并发场景，有更一致的正向趋势。

## 实现摘要

`p0_prefix` 只优化 allreduce metadata 里当前 device 的 prefix starts 计算路径，避免对完整 `all_sizes` 做全量 `cumsum` 后再索引。它不引入额外跨设备通信，也不改变 fused MoE 主计算路径。

## MiMo-V2-Flash

| EP | 并发 | 版本 | n | Input tok/s | Δ | Output tok/s | Δ | Peak output tok/s | Δ | Median E2E ms | Δ | Median TTFT ms | Δ | Median TPOT ms | Δ | Median ITL ms | Δ | 正确性 |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 8 | 64 | main | 4 | 10,569.41 | baseline | 660.59 | baseline | 2,112.00 | baseline | 73,737.66 | baseline | 40,297.40 | baseline | 43.65 | baseline | 23.71 | baseline | OK |
| 8 | 64 | current_branch | 1 | 10,506.05 | -0.60% | 656.63 | -0.60% | 2,112.00 | +0.00% | 74,816.83 | -1.46% | 40,391.18 | -0.23% | 43.90 | -0.57% | 23.70 | +0.03% | OK |
| 8 | 128 | main | 5 | 10,588.30 | baseline | 661.77 | baseline | 2,112.00 | baseline | 146,626.80 | baseline | 119,060.19 | baseline | 43.58 | baseline | 23.61 | baseline | OK |
| 8 | 128 | current_branch | 1 | 10,470.19 | -1.12% | 654.39 | -1.11% | 2,111.00 | -0.05% | 149,122.14 | -1.70% | 120,527.84 | -1.23% | 43.81 | -0.52% | 23.57 | +0.18% | OK |
| 8 | 512 | main | 1 | 10,609.17 | baseline | 663.07 | baseline | 2,025.00 | baseline | 440,655.32 | baseline | 385,426.95 | baseline | 44.49 | baseline | 23.52 | baseline | OK |
| 8 | 512 | current_branch | 1 | 10,726.32 | +1.10% | 670.39 | +1.10% | 2,112.00 | +4.30% | 435,406.99 | +1.19% | 381,241.62 | +1.09% | 43.60 | +2.00% | 23.66 | -0.60% | OK |
| 32 | 64 | main | 5 | 12,754.03 | baseline | 797.13 | baseline | 2,099.20 | baseline | 82,209.10 | baseline | 26,817.83 | baseline | 53.81 | baseline | 32.83 | baseline | OK |
| 32 | 64 | current_branch | 1 | 12,889.38 | +1.06% | 805.59 | +1.06% | 1,984.00 | -5.49% | 81,551.78 | +0.80% | 26,160.11 | +2.45% | 53.53 | +0.53% | 33.09 | -0.80% | OK |
| 32 | 128 | main | 5 | 14,034.89 | baseline | 877.18 | baseline | 2,636.80 | baseline | 149,384.58 | baseline | 50,365.90 | baseline | 96.44 | baseline | 51.80 | baseline | OK |
| 32 | 128 | current_branch | 1 | 14,155.64 | +0.86% | 884.73 | +0.86% | 2,560.00 | -2.91% | 148,104.88 | +0.86% | 49,144.22 | +2.43% | 96.36 | +0.09% | 52.73 | -1.80% | OK |
| 32 | 256 | main | 3 | 13,681.65 | baseline | 855.10 | baseline | 2,560.00 | baseline | 305,121.41 | baseline | 96,253.20 | baseline | 204.22 | baseline | 112.03 | baseline | OK |
| 32 | 256 | current_branch | 3 | 13,647.32 | -0.25% | 852.96 | -0.25% | 2,389.33 | -6.67% | 305,924.24 | -0.26% | 93,975.41 | +2.37% | 207.23 | -1.48% | 117.02 | -4.46% | OK |
| 32 | 512 | main | 1 | 14,530.56 | baseline | 908.16 | baseline | 3,200.00 | baseline | 357,522.29 | baseline | 191,395.99 | baseline | 185.83 | baseline | 113.57 | baseline | OK |
| 32 | 512 | current_branch | 1 | 14,751.02 | +1.52% | 921.94 | +1.52% | 3,400.00 | +6.25% | 356,354.85 | +0.33% | 186,823.12 | +2.39% | 184.57 | +0.68% | 117.63 | -3.57% | OK |
| 64 | 64 | main | 5 | 17,072.80 | baseline | 1,067.05 | baseline | 2,126.80 | baseline | 61,453.48 | baseline | 15,631.50 | baseline | 44.49 | baseline | 30.60 | baseline | OK |
| 64 | 64 | current_branch | 1 | 18,091.37 | +5.97% | 1,130.71 | +5.97% | 2,240.00 | +5.32% | 58,199.51 | +5.30% | 15,202.20 | +2.75% | 41.54 | +6.63% | 29.56 | +3.41% | OK |
| 64 | 128 | main | 5 | 19,914.52 | baseline | 1,244.66 | baseline | 2,872.80 | baseline | 105,245.57 | baseline | 29,316.47 | baseline | 73.98 | baseline | 45.41 | baseline | OK |
| 64 | 128 | current_branch | 1 | 20,853.03 | +4.71% | 1,303.31 | +4.71% | 3,072.00 | +6.93% | 100,513.82 | +4.50% | 28,444.12 | +2.98% | 70.12 | +5.21% | 44.57 | +1.84% | OK |
| 64 | 512 | main | 1 | 19,045.68 | baseline | 1,190.36 | baseline | 3,472.00 | baseline | 410,213.12 | baseline | 111,316.08 | baseline | 292.01 | baseline | 187.13 | baseline | OK |
| 64 | 512 | current_branch | 1 | 19,060.26 | +0.08% | 1,191.27 | +0.08% | 3,472.00 | +0.00% | 411,333.06 | -0.27% | 107,607.99 | +3.33% | 296.72 | -1.61% | 195.86 | -4.67% | OK |

## MiMo-V2-Pro-Private

| EP | 并发 | 版本 | n | Input tok/s | Δ | Output tok/s | Δ | Peak output tok/s | Δ | Median E2E ms | Δ | Median TTFT ms | Δ | Median TPOT ms | Δ | Median ITL ms | Δ | 正确性 |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 32 | 64 | main | 5 | 12,634.70 | baseline | 789.67 | baseline | 2,233.40 | baseline | 82,957.74 | baseline | 28,725.40 | baseline | 52.83 | baseline | 30.40 | baseline | OK |
| 32 | 64 | current_branch | 1 | 12,708.40 | +0.58% | 794.27 | +0.58% | 2,161.00 | -3.24% | 82,581.45 | +0.45% | 28,327.38 | +1.39% | 52.66 | +0.31% | 30.49 | -0.30% | OK |
| 32 | 128 | main | 5 | 13,500.00 | baseline | 843.75 | baseline | 2,774.60 | baseline | 127,931.97 | baseline | 53,105.68 | baseline | 73.08 | baseline | 40.61 | baseline | OK |
| 32 | 128 | current_branch | 1 | 13,554.53 | +0.40% | 847.16 | +0.40% | 2,704.00 | -2.54% | 122,272.19 | +4.42% | 52,379.19 | +1.37% | 67.92 | +7.06% | 39.37 | +3.06% | OK |
| 32 | 256 | main | 5 | 13,447.90 | baseline | 840.50 | baseline | 2,807.20 | baseline | 260,269.10 | baseline | 146,716.53 | baseline | 76.69 | baseline | 41.85 | baseline | OK |
| 32 | 256 | current_branch | 1 | 13,594.10 | +1.09% | 849.63 | +1.09% | 2,704.00 | -3.68% | 240,262.36 | +7.69% | 142,377.01 | +2.96% | 67.50 | +11.99% | 39.03 | +6.73% | OK |
| 32 | 512 | main | 1 | 13,482.68 | baseline | 842.67 | baseline | 2,496.00 | baseline | 377,196.77 | baseline | 292,170.17 | baseline | 80.26 | baseline | 41.89 | baseline | OK |
| 32 | 512 | current_branch | 1 | 14,110.22 | +4.65% | 881.89 | +4.65% | 2,808.00 | +12.50% | 360,018.44 | +4.55% | 279,896.99 | +4.20% | 75.45 | +5.99% | 39.48 | +5.75% | OK |
| 64 | 64 | main | 5 | 10,094.94 | baseline | 630.93 | baseline | 1,344.00 | baseline | 104,084.30 | baseline | 29,804.31 | baseline | 71.89 | baseline | 48.90 | baseline | OK |
| 64 | 64 | current_branch | 1 | 10,316.84 | +2.20% | 644.80 | +2.20% | 1,408.00 | +4.76% | 101,717.07 | +2.27% | 29,099.60 | +2.36% | 70.53 | +1.90% | 47.93 | +1.98% | OK |
| 64 | 128 | main | 5 | 12,129.02 | baseline | 758.06 | baseline | 2,076.80 | baseline | 172,859.17 | baseline | 55,699.47 | baseline | 114.19 | baseline | 64.55 | baseline | OK |
| 64 | 128 | current_branch | 1 | 12,354.74 | +1.86% | 772.17 | +1.86% | 2,176.00 | +4.78% | 169,706.28 | +1.82% | 54,363.83 | +2.40% | 112.41 | +1.56% | 63.94 | +0.95% | OK |
| 64 | 256 | main | 5 | 13,628.39 | baseline | 851.77 | baseline | 2,966.60 | baseline | 306,273.17 | baseline | 106,600.82 | baseline | 195.23 | baseline | 93.08 | baseline | OK |
| 64 | 256 | current_branch | 1 | 13,750.95 | +0.90% | 859.43 | +0.90% | 2,816.00 | -5.08% | 303,540.66 | +0.89% | 104,058.24 | +2.39% | 195.05 | +0.09% | 95.27 | -2.35% | OK |
| 64 | 512 | main | 1 | 13,559.54 | baseline | 847.47 | baseline | 3,104.00 | baseline | 386,604.09 | baseline | 212,668.08 | baseline | 195.57 | baseline | 119.36 | baseline | OK |
| 64 | 512 | current_branch | 1 | 13,910.67 | +2.59% | 869.42 | +2.59% | 3,199.00 | +3.06% | 375,373.71 | +2.90% | 206,977.91 | +2.68% | 191.41 | +2.13% | 115.01 | +3.64% | OK |

## 数据说明

- `main` 的 64/128/256 并发主要复用 2026-05-10 的 5 组复测结果；`p0_prefix` 的 64/128/256 是 2026-05-11 用当前分支补跑的 1 组结果。
- Flash EP32/256 使用 2026-05-11 同时间 3 组复测；Flash EP8/64 的 256 并发没有用当前 `p0_prefix` 分支重测，所以不列入表格。
- 512 并发是 2026-05-11 的 smoke 结果，每个 case 1 组。
- 表格列名为了可读性做了缩写：`Input tok/s` 对应 `Input token throughput (tok/s)`，`Output tok/s` 对应 `Output token throughput (tok/s)`，`Peak output tok/s` 对应 `Peak output token throughput (tok/s)`，`Median E2E ms` 对应 `Median E2E Latency (ms)`，`Median TTFT/TPOT/ITL ms` 分别对应 `Median TTFT (ms)`、`Median TPOT (ms)`、`Median ITL (ms)`。

## 原始结果

- Flash EP8 64/128 p0_prefix：`/tmp/sglang_allreduce_p0prefix_cc64_128_1x_flash_ep8_20260511/summary_ep8.jsonl`
- Flash EP32 64/128 p0_prefix：`/tmp/sglang_allreduce_p0prefix_cc64_128_1x_flash_ep32_20260511/summary_ep32.jsonl`
- Flash EP64 64/128 p0_prefix：`/tmp/sglang_allreduce_p0prefix_cc64_128_1x_flash_ep64_20260511/summary_ep64.jsonl`
- Pro-Private EP32 64/128/256 p0_prefix：`/tmp/sglang_allreduce_p0prefix_cc64_128_256_1x_pro_ep32_20260511/summary_ep32.jsonl`
- Pro-Private EP64 64/128/256 p0_prefix：`/tmp/sglang_allreduce_p0prefix_cc64_128_256_1x_pro_ep64_20260511/summary_ep64.jsonl`
- 512 smoke：`/tmp/sglang_allreduce_cc512_flash_ep8_20260511`、`/tmp/sglang_allreduce_cc512_flash_ep32_20260511`、`/tmp/sglang_allreduce_cc512_flash_ep64_20260511`、`/tmp/sglang_allreduce_cc512_pro_ep32_20260511`、`/tmp/sglang_allreduce_cc512_pro_ep64_20260511`
- 历史 main baseline：`/tmp/sglang_allreduce_5x_20260510*`
