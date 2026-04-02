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

## 2. FusedEPMoE Block Config Deep Tuning

### Round 1: 初始 Tuning (4 configs)

| Config | bt | bf | bd1 | bd2 | btc | bd2c | bse |
|--------|----|----|-----|-----|-----|------|-----|
| default | 32 | 512 | 1024 | 1024 | 32 | 1024 | 512 |
| v7_fp8_pattern | 32 | 2048 | 4096 | 4096 | 32 | 4096 | 512 |
| full_hidden_tile | 32 | 2048 | 4096 | 4096 | 32 | 4096 | 512 |
| large_bf | 32 | 2048 | 2048 | 2048 | 32 | 2048 | 512 |

注: FP8 模式下 `effective_for()` 强制 bd1c=512, bfc=256

### Deep Tuning: 32 configs 系统性 sweep

对 bt, bf, bd1, bd2, bd2c, btc, bse 参数进行系统性扫描（6 个 token counts × 32 configs = 192 runs）。

**关键发现: btc=16 是最大优化点**

| tokens | local | base (btc=32) | btc=16 | 提升 |
|--------|-------|-------------|--------|------|
| 128 | 8 | 0.758 | 0.795 | -5% (不适用于小 token) |
| 512 | 32 | 0.866 | 0.823 | **5.0%** |
| 1024 | 64 | 1.831 | 1.240 | **32.3%** |
| 2048 | 128 | 2.220 | 2.051 | **7.6%** |
| 4096 | 256 | 3.990 | 3.696 | **7.4%** |
| 8192 | 512 | 8.194 | 6.968 | **15.0%** |

其他 sweep 结论:
- **bf=2048 >> bf=1024 >> bf=512** — intermediate dim tile 越大越好
- **bd1=4096 最优** — 缩小 bd1 始终变慢
- **bd2c=4096 最优**（小 token），**bd2c=2048**（大 token 时接近）
- **bse 影响小** — bse=128 到 bse=2048 差距 <3%
- **bt=64/128 + bd1=4096 在 ≥1024 tokens 下 VMEM OOM** (70.4M > 64M limit)

### Round 2: 最佳参数组合

基于 deep tuning 发现，测试 btc=16 与其他最优参数的组合 (11 configs × 7 tokens = 77 runs, iters=5)。

| tokens | 最优 config | Time (ms) |
|--------|-----------|----------|
| 128 | btc16 + bse2048 + bd2c2048 | **0.744** |
| 256 | btc16 + bse2048 + bd2c2048 | **0.777** |
| 512 | btc16 + bse128 | **0.799** |
| 1024 | btc16 + bse128 | **1.215** |
| 2048 | btc16 | **2.047** |
| 4096 | btc16 + bse128 | **3.677** |
| 8192 | btc16 | **6.968** |

### 推荐 Config

```python
# 小 token (≤256): btc16 + bse2048 + bd2c2048
FusedMoEBlockConfig(
    bt=32, bf=2048, bd1=4096, bd2=4096,
    btc=16, bfc=256, bd1c=512, bd2c=2048, bse=2048
)

# 中大 token (≥512): btc16
FusedMoEBlockConfig(
    bt=32, bf=2048, bd1=4096, bd2=4096,
    btc=16, bfc=256, bd1c=512, bd2c=4096, bse=512
)
```

## 3. EPMoE vs 深度优化 FusedEPMoE — Balanced 最终对比

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

## 4. Imbalanced 负载对比 — Hotspot (半数专家 2x 流量)

配置: `hotspot_count=128, hotspot_ratio=0.667` (128 个热专家获得 2/3 总流量)

### 4a. EPMoE vs Default FusedEPMoE under Hotspot

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

### 4b. EPMoE vs Tuned FusedEPMoE under Hotspot

测试 Round 2 最优 tuned configs 在同一 hotspot 条件下的性能。

| tokens | EPMoE (ms) | tuned_small | tuned_medium | tuned_large | base_btc32 | best Fused | Speedup |
|--------|-----------|------------|-------------|------------|-----------|-----------|---------|
| 128 | 1.082 | 1.302 | 1.292 | **1.280** | 1.282 | 1.280 | 0.84x |
| 256 | 1.195 | 1.453 | 1.305 | **1.282** | 1.457 | 1.282 | 0.93x |
| 512 | 1.439 | 1.792 | 1.402 | 1.411 | **1.379** | 1.379 | **1.04x** |
| 1024 | 1.918 | 2.024 | 1.942 | 1.932 | **1.925** | 1.925 | 1.00x |
| 2048 | 2.752 | 3.073 | 3.097 | 2.991 | **2.897** | 2.897 | 0.95x |
| 4096 | 4.605 | 5.650 | 5.514 | 5.454 | **5.284** | 5.284 | 0.87x |
| 8192 | 8.425 | 10.181 | 9.797 | 9.733 | **9.008** | 9.008 | 0.94x |

**Geo-mean: 0.94x (EPMoE wins — tuned FusedEPMoE 在 hotspot 下仍然不如 EPMoE)**

**关键发现:**
- btc=32 (`base_btc32`) 在 hotspot 下反而优于 btc=16 的 tuned configs（与 balanced 结论相反）
- Balanced 下 1.23x 的 Fused 优势在 hotspot 下逆转为 0.94x 的 EPMoE 优势
- 仅 512 tokens 时 Fused 微弱胜出 (1.04x)

### Balanced vs Hotspot 影响对比

| tokens | EPMoE 变化 | FusedEPMoE(default) 变化 | FusedEPMoE(tuned) 变化 |
|--------|----------|----------------------|---------------------|
| 128 | +5% → +25% | -1% | +72% |
| 256 | — | — | +65% |
| 512 | +2% | -1% | +73% |
| 1024 | +3% | +18% | +58% |
| 2048 | +10% | +25% | +42% |
| 4096 | +2% | **+32%** | +44% |
| 8192 | +3% | **+42%** | +29% |

- **EPMoE**: 通信为 replicate+psum（与 routing 分布无关），对 imbalance 不敏感 (±5-25%)
- **FusedEPMoE (tuned)**: 小 token 退化 65-73%，大 token 退化 29-44%
- **btc=16 在 hotspot 下更敏感**: pipeline 迭代次数增加放大了 DMA 等待时间

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
| **Hotspot 2:1 (tuned config)** | **0.94x** | **EPMoE** |

Tuning 改善了 hotspot 下的表现 (0.57x→0.94x)，但仍然无法反超 EPMoE。
**生产环境中如果 routing 不均衡，应优先选择 EPMoE。**

### 5. v6e VMEM 限制

bt=64/128 + bd1=4096 的组合在 ≥1024 tokens 下触发 VMEM OOM (70.4M > 64M limit)。v6e 的 VMEM 比 v7 小，限制了 tile 大小组合空间。

### 6. 下一步

1. 将 v6e + MiMo-V2-Flash 的 tuned config 添加到 `tuned_block_configs.py`
2. ~~测试 tuned FusedEPMoE 在 imbalanced 下的性能~~ ✅ 已完成 (Section 4b)
3. 验证 weight_block_size=[128,128] 量化对结果的影响
4. 探索 EPLB (expert load balancing) 对 FusedEPMoE imbalance 问题的缓解效果
