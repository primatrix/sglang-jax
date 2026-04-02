# MiMo-V2-Flash MoE Backend 对比 — Block-128 量化

## 测试环境

- **硬件**: TPU v6e-16 (4x4 topology, 4 hosts x 4 chips)
- **模型**: MiMo-V2-Flash (XiaomiMiMo/MiMo-V2-Flash)
- **MoE 参数**: 256 experts, top_k=8, hidden=4096, intermediate=2048, activation=silu
- **量化**: FP8 (float8_e4m3fn), **weight_block_size=[128, 128]** (两个后端均使用 block-128)
- **Mesh**: ep_size=16, tp_size=1

与之前 COMPARE_RESULTS.md 的区别：
- 之前: EPMoE per-channel 量化, FusedEPMoE block-256 量化
- 本次: **两个后端均使用 block-128 量化**（控制变量）

## 1. EPMoE vs FusedEPMoE (Default Config) — Balanced

FusedEPMoE 默认 block config: `bt=32, bf=512, bd1=1024, bd2=1024`
Block-128 量化下 effective config: `bd1c=256, bfc=128`（由 effective_for 自动覆盖）

| tokens | EPMoE (ms) | Fused-Default (ms) | Speedup | Winner |
|--------|-----------|-------------------|---------|--------|
| | | | | |

_(待填充)_

## 2. EPMoE vs Tuned FusedEPMoE — Balanced

Tuned block configs (复用 block-256 下最优参数，bd1c/bfc 由 effective_for 覆盖为 256/128):
- 小 token (≤256): `bt=32 bf=2048 bd1=4096 bd2=4096 btc=16 bd2c=2048 bse=2048`
- 中大 token (≥512): `bt=32 bf=2048 bd1=4096 bd2=4096 btc=16 bd2c=4096 bse=512`

| tokens | EPMoE (ms) | Tuned Fused (ms) | Config | Speedup | Winner |
|--------|-----------|-----------------|--------|---------|--------|
| | | | | | |

_(待填充)_

## 3. Imbalanced 负载对比 — Hotspot (半数专家 2x 流量)

配置: `hotspot_count=128, hotspot_ratio=0.667` (128 个热专家获得 2/3 总流量)

### EPMoE vs Tuned FusedEPMoE under Hotspot

使用与 Balanced 相同的最优 tuned config，在 hotspot 条件下对比。

| tokens | EPMoE (ms) | Tuned Fused (ms) | Config | Speedup | Winner |
|--------|-----------|-----------------|--------|---------|--------|
| | | | | | |

_(待填充)_

### Balanced vs Hotspot — Tuned FusedEPMoE 退化幅度

| tokens | Balanced Fused (ms) | Hotspot Fused (ms) | Fused 退化 | EPMoE 退化 |
|--------|-------------------|-------------------|------------|----------|
| | | | | |

_(待填充)_

## 4. Block-128 vs Block-256 对比

对比两种量化 block size 下的性能差异（交叉引用 COMPARE_RESULTS.md）。

### Balanced — EPMoE

| tokens | EPMoE per-ch (ms) | EPMoE block-128 (ms) | 变化 |
|--------|------------------|---------------------|------|
| | | | |

_(待填充)_

### Balanced — Tuned FusedEPMoE

| tokens | Fused block-256 (ms) | Fused block-128 (ms) | 变化 |
|--------|---------------------|---------------------|------|
| | | | |

_(待填充)_

## 结论

_(待填充)_
