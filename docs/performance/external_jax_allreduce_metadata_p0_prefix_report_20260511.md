# External JAX Allreduce Metadata P0 Prefix 优化报告

更新日期：2026-05-11

## 结论

最终候选分支为 `exp/external-jax-allreduce-metadata-p0-prefix`，基于 `origin/exp/external-jax-allreduce-metadata` 的 `90055c0b`，只追加 `p0_prefix` 改动。

推荐合并该分支，不推荐合并完整 P0。完整 P0 中的 `p0_nozero` 改动在 serving 中表现为明确负向。

## 实现内容

`jax_allreduce_metadata_by_bt()` 原实现会为所有 device 计算 `starts_by_device`，再用 `dynamic_index_in_dim()` 取当前 device 的 starts：

```python
starts_by_device = jnp.cumsum(all_sizes, axis=1, dtype=jnp.int32) - all_sizes
starts = lax.dynamic_index_in_dim(starts_by_device, my_id, axis=1, keepdims=False)[:, None, :]
```

新实现只计算当前 device 需要的 prefix：

```python
device_ids = lax.broadcasted_iota(jnp.int32, (num_devices,), 0)
prefix_mask = device_ids < my_id
starts = jnp.sum(
    jnp.where(prefix_mask[None, :, None], all_sizes, jnp.zeros_like(all_sizes)),
    axis=1,
    keepdims=True,
).astype(jnp.int32)
```

该改动不改变输出 shape，不增加 HBM/SMEM metadata 体积，不改变 Pallas kernel 入参。

## IR 层面解释

旧实现会 materialize 全量 `starts_by_device`，StableHLO 中包含 `reduce_window`、`subtract` 和 `dynamic_slice`。prefix-only 实现只有 `iota/compare/select` 加一次沿 device 维度的 `reduce`。

因此该优化减少的是 JAX metadata 后处理的无用 work。它不是大算子优化，所以端到端收益会被 serving 调度、prefill/decode batch wave、Pallas custom call 边界、HBM handoff 和 run-to-run 波动稀释。

## P0 拆分结果

Flash EP32、cc256，单组定位：

| 版本 | 改动 | Correctness | Total tok/s | Mean TPOT ms | rank0 decode tok/s 均值 | 结论 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| fixed | `90055c0b` | OK | 14712.02 | 202.03 | 2267.09 | baseline |
| p0_nozero | 只去掉 Pallas 外部 metadata 路径里的 zero tensor | OK | 14212.76 | 212.46 | 2068.75 | 负向 |
| p0_prefix | 只优化 JAX starts prefix | OK | 14802.22 | 200.15 | 2300.16 | 正向候选 |
| 完整 P0 | `p0_nozero + p0_prefix` | OK | 14218.42 | 212.24 | 2071.39 | 被 p0_nozero 拖低 |

结论：只保留 `p0_prefix`，不要合并 `p0_nozero`。

## Flash EP32/cc256 复查

旧 5 组复测中 Flash EP32/cc256 出现 `fixed vs main = -5.62%`。后续同时间复查显示该结果主要来自 main 旧样本处在不可复现的高位。

最新同时间 3 组复测：

| 版本 | n | Total tok/s 样本 | Total tok/s 均值 | 标准差 | vs main | Mean TPOT ms | Mean TTFT ms | Correctness |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| main | 3 | 14440.07, 14780.99, 14389.18 | 14536.75 | 213.05 |  | 203.29 | 97147.56 | OK |
| fixed | 3 | 14680.46, 14665.63, 14100.93 | 14482.34 | 330.39 | -0.37% | 206.87 | 94709.38 | OK |
| p0_prefix | 3 | 14574.20, 14734.20, 14192.44 | 14500.28 | 278.34 | -0.25% | 206.32 | 94854.36 | OK |

结论：旧 `-5.62%` 没有复现。当前 p0_prefix 相对 main 的差异在 run-to-run 方差内，不作为合并阻塞点。

## 5 组复测主表

指标为 `Total token throughput (tok/s)`。

### MiMo-V2-Flash

| EP | 并发 | main 均值 | main 标准差 | fixed 均值 | fixed 标准差 | fixed vs main |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 64 | 11230.00 | 11.77 | 11216.38 | 16.34 | -0.12% |
| 8 | 128 | 11250.07 | 21.03 | 11220.27 | 22.26 | -0.26% |
| 8 | 256 | 11272.88 | 20.14 | 11220.83 | 6.29 | -0.46% |
| 32 | 64 | 13551.16 | 95.60 | 13642.24 | 63.87 | +0.67% |
| 32 | 128 | 14912.07 | 73.76 | 15115.04 | 45.86 | +1.36% |
| 32 | 256 | 15072.27 | 169.65 | 14225.94 | 125.80 | -5.62% |
| 64 | 64 | 18139.86 | 138.37 | 18315.58 | 180.63 | +0.97% |
| 64 | 128 | 21159.18 | 69.23 | 21679.37 | 108.45 | +2.46% |
| 64 | 256 | 21334.82 | 154.01 | 21377.12 | 63.32 | +0.20% |

Flash EP32/256 的 `-5.62%` 已由后续同时间复查解释为 baseline 漂移。

### MiMo-V2-Pro-Private

| EP | 并发 | main 均值 | main 标准差 | fixed 均值 | fixed 标准差 | fixed vs main |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 64 | 13424.37 | 21.46 | 13417.23 | 28.78 | -0.05% |
| 32 | 128 | 14343.75 | 32.24 | 14372.19 | 21.59 | +0.20% |
| 32 | 256 | 14288.39 | 41.33 | 14350.43 | 10.12 | +0.43% |
| 64 | 64 | 10725.87 | 1.09 | 10923.54 | 1.94 | +1.84% |
| 64 | 128 | 12887.09 | 2.30 | 13065.24 | 2.47 | +1.38% |
| 64 | 256 | 14480.17 | 23.01 | 14497.35 | 30.72 | +0.12% |

## 512 并发 Smoke

512 并发使用 `--num-prompts 512 --max-concurrency 512 --max-running-requests 512`。每个 case 跑 1 组，只作为 smoke。

| 模型 | EP | 版本 | Successful requests | Total tok/s | vs main | Mean TPOT ms | Correctness |
| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| Flash | 8 | main | 512 | 11272.24 |  | 45.67 | OK |
| Flash | 8 | p0_prefix | 512 | 11396.71 | +1.10% | 44.78 | OK |
| Flash | 32 | main | 512 | 15438.72 |  | 196.59 | OK |
| Flash | 32 | fixed | 512 | 15877.72 | +2.84% | 191.21 | OK |
| Flash | 32 | p0_prefix | 512 | 15672.96 | +1.52% | 195.45 | OK |
| Flash | 64 | main | 512 | 20236.04 |  | 287.41 | OK |
| Flash | 64 | p0_prefix | 512 | 20251.53 | +0.08% | 291.87 | OK |
| Pro-Private | 32 | main | 512 | 14325.35 |  | 79.71 | OK |
| Pro-Private | 32 | p0_prefix | 512 | 14992.11 | +4.65% | 74.93 | OK |
| Pro-Private | 64 | main | 512 | 14407.02 |  | 207.85 | OK |
| Pro-Private | 64 | p0_prefix | 512 | 14780.08 | +2.59% | 202.45 | OK |

512 smoke 没有发现 p0_prefix 的新回归。

## 合并建议

合并候选：

```text
90055c0b + p0_prefix
```

不要合并：

```text
origin/exp/external-jax-allreduce-metadata-p0
```

原因是完整 P0 包含 `p0_nozero`，该改动在 Flash EP32/cc256 上明显负向。

## 原始产物

| 内容 | 路径 |
| --- | --- |
| Flash EP32/cc256 P0 拆分实验 | `/tmp/sglang_allreduce_ablate_ep32_cc256_20260511/summary_ep32.jsonl` |
| Flash EP32/cc256 main/fixed/p0_prefix 3 组复测 | `/tmp/sglang_allreduce_prefix_validate_ep32_cc256_20260511/summary_ep32.jsonl` |
| 512 smoke: Flash EP8 | `/tmp/sglang_allreduce_cc512_flash_ep8_20260511/summary_ep8.jsonl` |
| 512 smoke: Flash EP32 | `/tmp/sglang_allreduce_cc512_flash_ep32_20260511/summary_ep32.jsonl` |
| 512 smoke: Flash EP64 | `/tmp/sglang_allreduce_cc512_flash_ep64_20260511/summary_ep64.jsonl` |
| 512 smoke: Pro-Private EP32 | `/tmp/sglang_allreduce_cc512_pro_ep32_20260511/summary_ep32.jsonl` |
| 512 smoke: Pro-Private EP64 | `/tmp/sglang_allreduce_cc512_pro_ep64_20260511/summary_ep64.jsonl` |
