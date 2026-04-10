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

## 1. EPMoE vs Tuned FusedEPMoE — Balanced 全量结果

Tuned block configs (复用 block-256 下最优参数，bd1c/bfc 由 effective_for 覆盖为 256/128):
- BASE: `bt=32 bf=2048 bd1=4096 bd2=4096 btc=32 bfc=256 bd1c=512 bd2c=4096 bse=512`
- tuned_small: `btc=16 bse=2048 bd2c=2048`
- tuned_medium: `btc=16 bse=128`
- tuned_large: `btc=16`
- base_btc32: BASE 原始参数

| tokens | local | EPMoE (ms) | tuned_small | tuned_medium | tuned_large | base_btc32 | Best Fused vs EPMoE |
|--------|-------|-----------|-------------|--------------|-------------|------------|-------------------|
| 32 | 2 | 1.051 | 1.276 | 1.275 | 1.400 | 1.288 | 0.82x (EPMoE) |
| 64 | 4 | 1.065 | 1.342 | 1.304 | 1.298 | 1.379 | 0.82x (EPMoE) |
| 128 | 8 | 1.153 | 1.344 | 1.367 | 1.374 | 1.448 | 0.86x (EPMoE) |
| 256 | 16 | 1.586 | 1.429 | 1.510 | 1.405 | **1.379** | **1.15x (Fused)** |
| 512 | 32 | 2.073 | 1.484 | **1.464** | 1.523 | 1.621 | **1.42x (Fused)** |
| 1024 | 64 | 2.581 | 1.987 | **1.960** | 1.984 | 2.258 | **1.32x (Fused)** |
| 2048 | 128 | 4.092 | 3.015 | **2.950** | 3.196 | 3.873 | **1.39x (Fused)** |
| 4096 | 256 | 5.408 | 5.069 | **4.954** | 5.004 | 6.229 | **1.09x (Fused)** |
| 8192 | 512 | 9.091 | 9.214 | **9.006** | 9.010 | 11.569 | **1.01x (Fused)** |

**Geo-mean: 1.07x (Fused wins)**

Best config per token count:
- 32 tokens: tuned_medium (1.275 ms)
- 64 tokens: tuned_large (1.298 ms)
- 128 tokens: tuned_small (1.344 ms)
- 256 tokens: base_btc32 (1.379 ms)
- 512-8192 tokens: tuned_medium (1.464-9.006 ms)

## 2. Imbalanced 负载对比 — Hotspot (半数专家 2x 流量)

配置: `hotspot_count=128, hotspot_ratio=0.667` (128 个热专家获得 2/3 总流量)

### EPMoE vs Tuned FusedEPMoE under Hotspot

使用与 Balanced 相同的 tuned configs，在 hotspot 条件下对比。

| tokens | local | EPMoE (ms) | tuned_small | tuned_medium | tuned_large | base_btc32 | Best Fused vs EPMoE |
|--------|-------|-----------|-------------|--------------|-------------|------------|-------------------|
| 128 | 8 | 1.150 | 1.325 | 1.345 | 1.366 | 1.348 | 0.87x (EPMoE) |
| 256 | 16 | 1.454 | 1.402 | 1.432 | **1.397** | 1.422 | **1.04x (Fused)** |
| 512 | 32 | 2.161 | 1.696 | 1.696 | 1.745 | **1.606** | **1.35x (Fused)** |
| 1024 | 64 | 2.586 | 2.661 | **2.568** | 2.573 | 2.706 | **1.01x (Fused)** |
| 2048 | 128 | 3.856 | 4.415 | **4.265** | 4.350 | 4.512 | 0.90x (EPMoE) |
| 4096 | 256 | 5.693 | 9.105 | **8.453** | 8.457 | 8.568 | 0.67x (EPMoE) |
| 8192 | 512 | 9.839 | 16.271 | 15.731 | 15.697 | **15.343** | 0.64x (EPMoE) |

**Geo-mean: 0.90x (EPMoE wins)**

### Balanced vs Hotspot — Tuned FusedEPMoE 退化幅度

| tokens | Balanced Fused (ms) | Hotspot Fused (ms) | Fused 退化 | EPMoE 退化 |
|--------|-------------------|-------------------|------------|----------|
| 128 | 1.344 | 1.325 | -1% | 0% |
| 256 | 1.379 | 1.397 | +1% | -8% |
| 512 | 1.464 | 1.606 | +10% | +4% |
| 1024 | 1.960 | 2.568 | +31% | 0% |
| 2048 | 2.950 | 4.265 | +45% | -6% |
| 4096 | 4.954 | 8.453 | +71% | +5% |
| 8192 | 9.006 | 15.343 | +70% | +8% |

- **EPMoE**: 对 imbalance 几乎不敏感 (-8% ~ +8%)
- **FusedEPMoE (tuned)**: 小 token 下稳定 (-1% ~ +10%)，大 token 下严重退化 (+31% ~ +71%)

## 3. Block-128 vs Block-256 对比

对比两种量化 block size 下的性能差异（交叉引用 COMPARE_RESULTS.md）。

注：block-256 数据中 EPMoE 使用 per-channel 量化, FusedEPMoE 使用 block-256。
Block-128 数据中两个后端均使用 block-128。

### Balanced — EPMoE

| tokens | EPMoE per-ch (ms) | EPMoE block-128 (ms) | 变化 |
|--------|------------------|---------------------|------|
| 32 | 0.786 | 1.051 | +34% |
| 64 | 0.786 | 1.065 | +35% |
| 128 | 0.863 | 1.153 | +34% |
| 256 | 1.071 | 1.586 | +48% |
| 512 | 1.241 | 2.073 | +67% |
| 1024 | 1.726 | 2.581 | +50% |
| 2048 | 2.588 | 4.092 | +58% |
| 4096 | 4.325 | 5.408 | +25% |
| 8192 | 8.096 | 9.091 | +12% |

EPMoE block-128 比 per-channel 慢 12-67%。Block 量化引入的 scale 查表开销在 EPMoE 的 GMM kernel 中比较显著。

### Balanced — Tuned FusedEPMoE (best config)

| tokens | Fused block-256 (ms) | Fused block-128 (ms) | 变化 |
|--------|---------------------|---------------------|------|
| 32 | 0.753 | 1.275 | +69% |
| 64 | 0.772 | 1.298 | +68% |
| 128 | 0.744 | 1.344 | +81% |
| 256 | 0.777 | 1.379 | +77% |
| 512 | 0.799 | 1.464 | +83% |
| 1024 | 1.215 | 1.960 | +61% |
| 2048 | 2.047 | 2.950 | +44% |
| 4096 | 3.677 | 4.954 | +35% |
| 8192 | 6.968 | 9.006 | +29% |

FusedEPMoE block-128 比 block-256 慢 29-83%。小 token 下退化更严重 (68-83%)，大 token 下相对收敛 (29-44%)。

### 综合 Speedup 对比

| tokens | block-256 Speedup | block-128 Speedup | 变化 |
|--------|------------------|------------------|------|
| 32 | 1.04x | 0.82x | -0.22 |
| 64 | 1.02x | 0.82x | -0.20 |
| 128 | 1.16x | 0.86x | -0.30 |
| 256 | 1.38x | 1.15x | -0.23 |
| 512 | 1.55x | 1.42x | -0.13 |
| 1024 | 1.42x | 1.32x | -0.10 |
| 2048 | 1.26x | 1.39x | +0.13 |
| 4096 | 1.18x | 1.09x | -0.09 |
| 8192 | 1.16x | 1.01x | -0.15 |
| **Geo-mean** | **1.23x** | **1.07x** | **-0.16** |

Block-128 量化使 Fused vs EPMoE 的 speedup 整体下降约 0.16。Fused 的 block-128 退化（+29-83%）比 EPMoE 的退化（+12-67%）更大，导致 Fused 的相对优势缩小。

## 结论

### 1. Block-128 量化削弱了 FusedEPMoE 的优势

| 场景 | Block-256 Geo-mean | Block-128 Geo-mean | 变化 |
|------|-------------------|-------------------|------|
| Balanced | **1.23x (Fused)** | **1.07x (Fused)** | -0.16 |
| Hotspot | 0.90x (EPMoE) | 0.90x (EPMoE) | 0.00 |

Block-128 使 FusedEPMoE 在 balanced 下的优势从 1.23x 降至 1.07x。两个后端都变慢了，但 FusedEPMoE 退化更大 (+29-83%) vs EPMoE (+12-67%)。Hotspot 下 geo-mean 恰好一致 (0.90x)。

### 2. Balanced 下的 Crossover Point

Block-128 量化下，FusedEPMoE 仅在 ≥256 tokens 时才优于 EPMoE：
- ≤128 tokens: EPMoE 赢 (0.82-0.86x)
- 256-512 tokens: Fused 甜蜜区间 (1.15-1.42x)
- 1024-2048 tokens: Fused 中等优势 (1.32-1.39x)
- 4096-8192 tokens: Fused 微弱优势 (1.01-1.09x)

对比 block-256 下 FusedEPMoE 在所有 token count 均胜出的结果，block-128 量化显著缩小了优势区间。

### 3. Hotspot Imbalance 下 FusedEPMoE 仍然不可靠

- 256-1024 tokens: Fused 勉强持平或微弱优势 (1.01-1.35x)
- ≥2048 tokens: EPMoE 大幅领先 (0.64-0.90x)
- FusedEPMoE 在大 token + hotspot 下退化 +45-71%，EPMoE 几乎不受影响

**生产环境建议**: Block-128 量化下，FusedEPMoE 的优势非常有限 (geo-mean 仅 1.07x)。如果 routing 存在 imbalance，应使用 EPMoE。

### 4. Block Size 对 EPMoE 的影响

EPMoE 从 per-channel 切换到 block-128 后变慢 12-67%（平均约 40%）。这是 block 量化 scale 查表的固有开销。FusedEPMoE 的退化更大是因为其 Pallas kernel 需要额外处理 sub-channel scale 的 tiling。
