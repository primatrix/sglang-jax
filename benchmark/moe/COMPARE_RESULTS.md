# MiMo-V2-Flash MoE Backend 对比分析 — 最终结果

## 测试环境

- **硬件**: TPU v6e-16 (4x4 topology, 4 hosts × 4 chips)
- **模型**: MiMo-V2-Flash (XiaomiMiMo/MiMo-V2-Flash)
- **MoE 参数**: 256 experts, top_k=8, hidden=4096, intermediate=2048, activation=silu
- **量化**: FP8 (float8_e4m3fn), EPMoE per-channel, FusedEPMoE block-256
- **Mesh**: ep_size=16, tp_size=1 (两个后端相同)

## 1. EPMoE vs FusedEPMoE (Default Config) — Balanced

FusedEPMoE 默认 block config: `bt=32, bf=512, bd1=1024, bd2=1024`

| tokens | EPMoE (ms) | Fused-Default (ms) | Speedup | Winner |
|--------|-----------|-------------------|---------|--------|
| 32 | 0.786 | 1.315 | 0.60x | EPMoE |
| 64 | 0.786 | 1.327 | 0.59x | EPMoE |
| 128 | 0.863 | 1.316 | 0.66x | EPMoE |
| 256 | 1.071 | 1.415 | 0.76x | EPMoE |
| 512 | 1.241 | 1.531 | 0.81x | EPMoE |
| 1024 | 1.726 | 2.497 | 0.69x | EPMoE |
| 2048 | 2.588 | 4.399 | 0.59x | EPMoE |
| 4096 | 4.325 | 8.382 | 0.52x | EPMoE |
| 8192 | 8.096 | 14.867 | 0.54x | EPMoE |

**Geo-mean: 0.63x (EPMoE wins with untuned FusedEPMoE)**

## 2. EPMoE vs Tuned FusedEPMoE — Balanced 最终对比

Tuned block config (三轮 tuning 后的最优 config，详见 PROCESS_LOG.md):
- 小 token (≤256): `bt=32 bf=2048 bd1=4096 bd2=4096 btc=16 bd2c=2048 bse=2048` (FP8: bfc=256 bd1c=512)
- 中大 token (≥512): `bt=32 bf=2048 bd1=4096 bd2=4096 btc=16 bd2c=4096 bse=512` (FP8: bfc=256 bd1c=512)

| tokens | EPMoE (ms) | Tuned Fused (ms) | Config | Speedup | Winner |
|--------|-----------|-----------------|--------|---------|--------|
| 32 | 0.786 | 0.753 | R1 full_hidden | **1.04x** | Fused |
| 64 | 0.786 | 0.772 | R1 v7_fp8 | **1.02x** | Fused |
| 128 | 0.863 | 0.744 | R2 btc16+bse2048+bd2c2048 | **1.16x** | Fused |
| 256 | 1.071 | 0.777 | R2 btc16+bse2048+bd2c2048 | **1.38x** | Fused |
| 512 | 1.241 | 0.799 | R2 btc16+bse128 | **1.55x** | Fused |
| 1024 | 1.726 | 1.215 | R2 btc16+bse128 | **1.42x** | Fused |
| 2048 | 2.588 | 2.047 | R2 btc16 | **1.26x** | Fused |
| 4096 | 4.325 | 3.677 | R2 btc16+bse128 | **1.18x** | Fused |
| 8192 | 8.096 | 6.968 | R2 btc16 | **1.16x** | Fused |

**Geo-mean speedup: 1.23x (Tuned Fused wins at all token counts)**

## 3. Imbalanced 负载对比 — Hotspot (半数专家 2x 流量)

配置: `hotspot_count=128, hotspot_ratio=0.667` (128 个热专家获得 2/3 总流量)

### 3a. EPMoE vs Default FusedEPMoE under Hotspot

| tokens | EPMoE (ms) | Fused-Default (ms) | Speedup |
|--------|-----------|-------------------|---------|
| 32 | 0.739 | 1.428 | 0.52x |
| 64 | 0.797 | 1.306 | 0.61x |
| 128 | 0.904 | 1.306 | 0.69x |
| 256 | 0.978 | 1.418 | 0.69x |
| 512 | 1.271 | 1.511 | 0.84x |
| 1024 | 1.785 | 2.942 | 0.61x |
| 2048 | 2.839 | 5.513 | 0.51x |
| 4096 | 4.409 | 11.099 | 0.40x |
| 8192 | 8.378 | 21.124 | 0.40x |

**Geo-mean: 0.57x (EPMoE wins — default FusedEPMoE 在 imbalanced 下更差)**

### 3b. EPMoE vs Tuned FusedEPMoE under Hotspot

使用与 Balanced 相同的最优 tuned config，在 hotspot 条件下对比。

| tokens | EPMoE (ms) | Tuned Fused (ms) | Config | Speedup | Winner |
|--------|-----------|-----------------|--------|---------|--------|
| 128 | 1.082 | 1.302 | btc16+bse2048+bd2c2048 | 0.83x | EPMoE |
| 256 | 1.195 | 1.453 | btc16+bse2048+bd2c2048 | 0.82x | EPMoE |
| 512 | 1.439 | 1.402 | btc16+bse128 | **1.03x** | Fused |
| 1024 | 1.918 | 1.942 | btc16+bse128 | 0.99x | EPMoE |
| 2048 | 2.752 | 2.991 | btc16 | 0.92x | EPMoE |
| 4096 | 4.605 | 5.514 | btc16+bse128 | 0.84x | EPMoE |
| 8192 | 8.425 | 9.733 | btc16 | 0.87x | EPMoE |

**Geo-mean: 0.90x (EPMoE wins — tuned FusedEPMoE 在 hotspot 下仍不如 EPMoE)**

### Balanced vs Hotspot — Tuned FusedEPMoE 退化幅度

| tokens | Balanced Fused (ms) | Hotspot Fused (ms) | 退化 | EPMoE 退化 |
|--------|-------------------|-------------------|------|----------|
| 128 | 0.744 | 1.302 | +75% | +25% |
| 256 | 0.777 | 1.453 | +87% | +12% |
| 512 | 0.799 | 1.402 | +75% | +16% |
| 1024 | 1.215 | 1.942 | +60% | +11% |
| 2048 | 2.047 | 2.991 | +46% | +6% |
| 4096 | 3.677 | 5.514 | +50% | +6% |
| 8192 | 6.968 | 9.733 | +40% | +4% |

- **EPMoE**: 对 imbalance 不敏感 (4-25%)
- **FusedEPMoE (tuned)**: 退化 40-87%，小 token 更严重

## 结论

### 1. Block Config Tuning 是必须的

默认 FusedEPMoE config 在 v6e + MiMo-V2-Flash shape 下完全未优化，导致 FusedEPMoE 比 EPMoE 慢 37-60%。经过三轮 tuning 后反而快 2-55%。

### 2. btc=16 是关键优化（仅限 Balanced）

Deep tuning 发现将 token compute tile 从 btc=32 降到 btc=16，在 ≥512 tokens 下带来 5-32% 的提升。但在 hotspot imbalance 下，btc=16 反而不如 btc=32 — 更多 pipeline 迭代放大了 DMA 等待。

### 3. Tuned FusedEPMoE 优于 EPMoE（Balanced）

| 区间 | Speedup 范围 | 描述 |
|------|-----------|------|
| 32-64 tokens | 1.02-1.04x | 微弱优势 |
| 128-256 tokens | 1.16-1.38x | 明显优势 |
| 512-1024 tokens | 1.42-1.55x | 显著优势（甜蜜区间） |
| 2048-8192 tokens | 1.16-1.26x | 稳定优势 |

### 4. FusedEPMoE 的 Imbalance 敏感性（关键风险）

| 场景 | Geo-mean Speedup | 胜者 |
|------|----------------|------|
| Balanced | **1.23x** | Fused |
| Hotspot 2:1 (default config) | 0.57x | EPMoE |
| **Hotspot 2:1 (tuned config)** | **0.90x** | **EPMoE** |

Tuning 改善了 hotspot 下的表现 (0.57x→0.90x)，但仍然无法反超 EPMoE。
**生产环境中如果 routing 不均衡，应优先选择 EPMoE。**

### 5. v6e VMEM 限制

bt=64/128 + bd1=4096 的组合在 ≥1024 tokens 下触发 VMEM OOM (70.4M > 64M limit)。v6e 的 VMEM 比 v7 小，限制了 tile 大小组合空间。

### 6. 下一步

1. ~~将 v6e + MiMo-V2-Flash 的 tuned config 添加到 `tuned_block_configs.py`~~ ✅ 已完成
2. ~~测试 tuned FusedEPMoE 在 imbalanced 下的性能~~ ✅ 已完成 (Section 3b)
3. 验证 weight_block_size=[128,128] 量化对结果的影响
4. 探索 EPLB (expert load balancing) 对 FusedEPMoE imbalance 问题的缓解效果
