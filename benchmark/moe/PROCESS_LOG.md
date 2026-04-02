# MiMo-V2-Flash MoE Backend 对比 — 详细过程记录

## 1. 初始对比: EPMoE vs FusedEPMoE (Default Config)

**日期**: 2026-04-01
**目的**: 在相同配置下对比两个 MoE backend 的 kernel latency

### 配置

```
模型: MiMo-V2-Flash
  hidden_size: 4096, moe_intermediate_size: 2048
  n_routed_experts: 256, num_experts_per_tok: 8
  hidden_act: silu, norm_topk_prob: true
  quant_method: fp8 (e4m3), weight_block_size: [128, 128]

测试环境: TPU v6e-16 (16 chips, 4 hosts)
Mesh: ep_size=16, tp_size=1
负载模式: balanced
量化: QuantizationConfig(moe_weight_dtype=float8_e4m3fn, moe_activation_dtype=None)
  - EPMoE: weight_block_size=None → per-channel 量化
  - FusedEPMoE: subc_quant_wsz=256 → block-256 量化
```

### 原始数据 (Process 0)

```
num_tokens |  EPMoE (ms) |  Fused (ms) |  Speedup | Winner
----------------------------------------------------------
        32 |       0.786 |       1.315 |    0.60x |  EPMoE
        64 |       0.786 |       1.327 |    0.59x |  EPMoE
       128 |       0.863 |       1.316 |    0.66x |  EPMoE
       256 |       1.071 |       1.415 |    0.76x |  EPMoE
       512 |       1.241 |       1.531 |    0.81x |  EPMoE
      1024 |       1.726 |       2.497 |    0.69x |  EPMoE

Geo-mean speedup: 0.68x (EPMoE wins)
```

### 分析

EPMoE 在所有 token 数下都优于 FusedEPMoE (default)。但怀疑 FusedEPMoE 的
block config 未经 tune — 当前 `tuned_block_configs.py` 中没有 TPU v6e 的条目，
也没有 (hidden=4096, inter=2048, ep=16) 的配置，全部走 DEFAULT_FUSED_MOE_BLOCK_CONFIG。

---

## 2. FusedEPMoE Block Config 分析

**发现**: `tuned_block_configs.py` 中仅有 TPU v7 的 tuned 配置。
v6e 和我们的 model shape 完全没有 tuned config，使用的 default:
```python
DEFAULT_FUSED_MOE_BLOCK_CONFIG = FusedMoEBlockConfig(
    bt=32, bf=512, bd1=1024, bd2=1024,
    btc=32, bfc=512, bd1c=1024, bd2c=1024, bse=512,
)
```

**FP8 override**: `effective_for()` 中 `subc_quant_wsz=256` 会覆盖:
- `bd1c → 256 * t_packing(=2) = 512`
- `bfc → 256`
因此实际 compute tile: bd1c=512, bfc=256, 而 bd2c 不被覆盖。

**参考**: tpu-inference 的 heuristic (`get_default_block_sizes()`):
```
d = hidden_size / 256 = 16
f = intermediate_size / 256 = 8
bf = min(256 * f // 2, 1024) = 1024
bd1 = min(256 * d // 2, 1024) = 1024
bd2 = min(256 * d // 2, 2048) = 2048
```

---

## 3. Block Config Tuning — Round 1 (128 tokens)

**日期**: 2026-04-01
**目的**: 验证不同 block config 对 FusedEPMoE 性能的影响

### 测试的 7 个候选配置

| Config | bt | bf | bd1 | bd2 | btc | bfc | bd1c | bd2c |
|--------|----|----|-----|-----|-----|-----|------|------|
| default | 32 | 512 | 1024 | 1024 | 32 | 512 | 1024 | 1024 |
| heuristic_v1 | 128 | 1024 | 1024 | 2048 | 64 | 1024 | 1024 | 2048 |
| large_bf | 32 | 2048 | 2048 | 2048 | 32 | 2048 | 2048 | 2048 |
| v7_fp8_pattern | 32 | 2048 | 4096 | 4096 | 32 | 256 | 512 | 4096 |
| v7_mid_pattern | 32 | 1024 | 2048 | 2048 | 32 | 256 | 512 | 2048 |
| small_bf_large_bd | 32 | 512 | 2048 | 2048 | 32 | 512 | 2048 | 2048 |
| full_hidden_tile | 32 | 2048 | 4096 | 4096 | 32 | 2048 | 4096 | 4096 |

### 结果 (num_tokens=128, local=8)

```
Config               |  Time (ms) | vs default
---------------------+------------+-----------
default              |      1.373 |      1.00x
heuristic_v1         |      0.958 |      1.43x
large_bf             |      0.812 |      1.69x
v7_fp8_pattern       |      0.758 |      1.81x  ★
v7_mid_pattern       |      0.964 |      1.42x
small_bf_large_bd    |      1.105 |      1.24x
full_hidden_tile     |      0.763 |      1.80x
```

### 分析

- **大 bf (2048) 是关键因子** — 所有 bf=2048 的配置都显著优于 bf=512/1024
- **大 bd1/bd2 (4096) 进一步提升** — v7_fp8_pattern 和 full_hidden_tile 最优
- **small_bf_large_bd 证明 bf 比 bd 更重要** — bf=512 + bd=2048 (1.105ms) 远不如 bf=2048 + bd=2048 (0.812ms)
- 选出 top-3: v7_fp8_pattern, full_hidden_tile, large_bf

---

## 4. Block Config Tuning — Round 2 (全 token 数扫描)

**日期**: 2026-04-01
**目的**: 用 top-3 config 跑所有 token counts

### 结果

```
num_tokens=32 (local=2)
  default              |      1.265 |      1.00x
  v7_fp8_pattern       |      0.764 |      1.66x
  full_hidden_tile     |      0.753 |      1.68x  ★
  large_bf             |      0.782 |      1.62x

num_tokens=64 (local=4)
  default              |      1.299 |      1.00x
  v7_fp8_pattern       |      0.772 |      1.68x  ★
  full_hidden_tile     |      0.787 |      1.65x
  large_bf             |      0.809 |      1.61x

num_tokens=128 (local=8)
  default              |      1.398 |      1.00x
  v7_fp8_pattern       |      0.798 |      1.75x
  full_hidden_tile     |      0.769 |      1.82x  ★
  large_bf             |      0.828 |      1.69x

num_tokens=256 (local=16)
  default              |      1.339 |      1.00x
  v7_fp8_pattern       |      1.751 |      0.76x  ⚠️ 回退!
  full_hidden_tile     |      0.786 |      1.70x  ★
  large_bf             |      0.843 |      1.59x

num_tokens=512 (local=32)
  default              |      1.507 |      1.00x
  v7_fp8_pattern       |      0.850 |      1.77x  ★
  full_hidden_tile     |      0.903 |      1.67x
  large_bf             |      0.991 |      1.52x

num_tokens=1024 (local=64)
  default              |      2.609 |      1.00x
  v7_fp8_pattern       |      1.334 |      1.96x  ★
  full_hidden_tile     |      1.430 |      1.82x
  large_bf             |      1.623 |      1.61x
```

### 分析

- **v7_fp8_pattern 在 256 tokens (local=16) 出现严重回退 (1.751ms, 0.76x)**
  - 原因: bt=32 对 local_num_tokens=16 时，bd1=4096 + bfc=256 的组合可能导致
    tile 循环次数和 VMEM 溢出问题
  - full_hidden_tile 同样 bd1=4096 但 bfc=2048 不受影响，说明是 bfc=256 的问题
- **full_hidden_tile 最稳定**，所有 token 数都表现良好（0.753-1.430ms）
- **v7_fp8_pattern 在大 token 数更优**（512: 0.850, 1024: 1.334）
- **生产环境建议**: 小 token (≤256) 用 full_hidden_tile, 大 token (≥512) 用 v7_fp8_pattern

---

## 5. EPMoE 扩展测试 (2048/4096/8192)

**日期**: 2026-04-01
**目的**: 获取 EPMoE 在大 token 数下的 baseline 数据

### 测试方式

使用 `/tmp/launcher_epmoe_large.py` 运行 `bench_moe_compare.py`，
仅取 EPMoE 和 FusedEPMoE-default 的数据。

### 结果 (Process 0)

```
num_tokens |  EPMoE (ms) |  Fused-Default (ms) |  Speedup | Winner
------------------------------------------------------------------
      2048 |       2.588 |       4.399 |    0.59x |  EPMoE
      4096 |       4.325 |       8.382 |    0.52x |  EPMoE
      8192 |       8.096 |      14.867 |    0.54x |  EPMoE
```

### 分析

- EPMoE 在大 token 数下线性 scaling（2048→8192 约 3.1x latency for 4x tokens）
- FusedEPMoE (default config) 的 scaling 更差（2048→8192 约 3.4x latency）
- 差距在大 token 时更大（0.52x-0.59x vs 小 token 的 0.59x-0.81x）
- 再次确认默认 block config 在大 token 时效率极差

---

## 6. FusedEPMoE Tuning 扩展 (2048/4096/8192)

**日期**: 2026-04-01
**目的**: 对 top-3 block config 在大 token 数下进行 tuning

### 测试方式

使用 `/tmp/launcher_tune_fused.py` 独立 tuning 脚本，测试 4 个候选配置。

### 结果

```
num_tokens=2048 (local=128)
  Config               |  Time (ms) | vs default
  -------------------+-----------+-----------
  default              |      4.808 |      1.00x
  v7_fp8_pattern       |      2.224 |      2.16x
  full_hidden_tile     |      2.193 |      2.19x  ★
  large_bf             |      2.702 |      1.78x

num_tokens=4096 (local=256)
  default              |      9.218 |      1.00x
  v7_fp8_pattern       |      3.980 |      2.32x  ★
  full_hidden_tile     |      4.010 |      2.30x
  large_bf             |      5.022 |      1.84x

num_tokens=8192 (local=512)
  default              |     18.019 |      1.00x
  v7_fp8_pattern       |      7.562 |      2.38x
  full_hidden_tile     |      7.558 |      2.38x  ★
  large_bf             |      9.667 |      1.86x
```

### 分析

- **Tuning 提升随 token 数增大而增大**: 2048 时 2.19x → 8192 时 2.38x
  - 说明默认 tile 尺寸在高负载时浪费更严重
- **v7_fp8_pattern 和 full_hidden_tile 几乎持平**:
  - 2048: 2.224 vs 2.193 ms (差 1.4%)
  - 4096: 3.980 vs 4.010 ms (差 0.8%)
  - 8192: 7.562 vs 7.558 ms (差 0.05%)
  - 不像小 token (256) 那样出现 v7_fp8_pattern 的严重回退
- **large_bf 提升稳定但有限**: 始终约 1.8x，不如 top-2 的 2.2-2.4x
  - 说明 bd1/bd2=4096（完整 hidden_size tile）比 2048 更高效
- **v7_fp8_pattern 在大 token 未出现回退**: 256 tokens (local=16) 的回退问题
  在 local≥128 时不再出现，因为 token 数远大于 bt=32

### EPMoE vs Tuned FusedEPMoE 大 token 对比

```
num_tokens | EPMoE (ms) | Tuned Fused (ms) | Config          | Speedup
-----------------------------------------------------------------------
      2048 |      2.588 |            2.193 | full_hidden     |   1.18x
      4096 |      4.325 |            3.980 | v7_fp8_pattern  |   1.09x
      8192 |      8.096 |            7.558 | full_hidden     |   1.07x
```

- **FusedEPMoE 仍然赢**，但优势在大 token 时缩小（1.18x → 1.07x）
- **原因**: EPMoE 的 replicate + psum 通信开销是固定的（不随负载变化），
  而 FusedEPMoE 的 all-to-all 通信量和 kernel pipeline 效率在大 token 时
  计算占比更大，通信优势被稀释
- **中 token (256-1024) 是 FusedEPMoE 的甜蜜区间** (1.29x-1.46x)

---

## 7. Deep Tuning — 32 configs 系统性 sweep

**日期**: 2026-04-02
**目的**: 系统性探索 block config 参数空间，找到每个 token 数的最优 config

### 方法

对 6 个自由参数（FP8 固定 bd1c=512, bfc=256）进行 1D sweep + 组合测试:
- bt: 8, 16, 32, 64, 128
- bf: 256, 512, 1024, 2048
- bd1: 512, 1024, 2048, 4096
- bd2: 1024, 2048, 4096
- bd2c: 256, 512, 1024, 2048, 4096
- btc: 8, 16, 32
- bse: 128, 256, 512, 1024, 2048

共 32 个 config × 6 个 token counts (128, 512, 1024, 2048, 4096, 8192) = 192 runs

### 关键结果

**btc=16 是最大优化点** — 在 5/6 token counts 下排名第一:

```
tokens  | base (btc=32) | btc=16 | 提升
--------|-------------|--------|------
   128  |      0.758  |  0.795 |  -5% (小 token 不适用)
   512  |      0.866  |  0.823 |  5.0%
  1024  |      1.831  |  1.240 | 32.3%  ★
  2048  |      2.220  |  2.051 |  7.6%
  4096  |      3.990  |  3.696 |  7.4%
  8192  |      8.194  |  6.968 | 15.0%
```

**其他参数 sweep 结论**:

| 参数 | 最优值 | 结论 |
|------|-------|------|
| bt | 32 | bt=64/128 在 ≥1024 时 VMEM OOM (70.4M > 64M)；bt=8/16 太慢 |
| bf | 2048 | bf=2048 >> 1024 >> 512 >> 256，越大越好 |
| bd1 | 4096 | 缩小 bd1 始终变慢（bd1_2048: +8%, bd1_1024: +12%） |
| bd2 | 4096 | bd2=2048 接近但不如 4096 |
| bd2c | 4096 | 大 token 时 bd2c=4096 最优；小 token 时 bd2c=2048 可比 |
| bse | 128-2048 | 影响小 (<3%)，bse=2048 在 ≥1024 时微弱最优 |

**VMEM OOM 发现**: bt=64/128 + bd1=4096 在 ≥1024 tokens 下报错:
`RESOURCE_EXHAUSTED: scoped vmem 70.41M > 64M limit`
这是 v6e 的 VMEM 约束，v7 可能不受此限制。

---

## 8. Round 2 — 最佳参数组合

**日期**: 2026-04-02
**目的**: 基于 deep tuning 发现，测试 btc=16 与其他最优参数的组合

### 候选 Configs

| Config | btc | bse | bd2c | 其他 |
|--------|-----|-----|------|------|
| r1_btc16 | 16 | 512 | 4096 | base |
| r2_btc16_bse2048 | 16 | 2048 | 4096 | base |
| r2_btc16_bse128 | 16 | 128 | 4096 | base |
| r2_btc16_bd2c2048 | 16 | 512 | 2048 | base |
| r2_btc16_bse2048_bd2c2048 | 16 | 2048 | 2048 | base |
| r2_btc16_bd2_2048 | 16 | 512 | 2048 | bd2=2048 |
| r2_btc16_bd1_2048 | 16 | 512 | 4096 | bd1=2048 |
| r2_btc8_bse2048 | 8 | 2048 | 4096 | base |

11 configs × 7 tokens (128-8192) = 77 runs, iters=5, warmup=2

### 结果

```
tokens  | best config                     | time (ms) | vs R1 btc16
--------|--------------------------------|----------|------------
   128  | btc16+bse2048+bd2c2048          |    0.744 |  0.749→0.744 (-0.7%)
   256  | btc16+bse2048+bd2c2048          |    0.777 |  0.786→0.777 (-1.1%)
   512  | btc16+bse128                    |    0.799 |  0.823→0.799 (-2.9%)
  1024  | btc16+bse128                    |    1.215 |  1.240→1.215 (-2.0%)
  2048  | btc16 (plain)                   |    2.047 |  2.051→2.047 (-0.2%)
  4096  | btc16+bse128                    |    3.677 |  3.696→3.677 (-0.5%)
  8192  | btc16 (plain)                   |    6.999 |  6.968→6.999 (+0.4%)
```

### 分析

- **小 token (128-256)**: btc16+bse2048+bd2c2048 组合最优
  - bd2c=2048 减少 VMEM 占用，bse=2048 避免 shared expert scratch 碎片化
- **中 token (512-4096)**: btc16+bse128 最优
  - bse=128 最小化 shared expert scratch VMEM，为主计算留出更多空间
- **大 token (2048-8192)**: btc16 plain 与 btc16+bse128 差距 <1%
- **btc16_bd1_2048 始终最差** — 确认 bd1=4096 是硬要求

---

## 9. Imbalanced 负载测试

**日期**: 2026-04-02
**目的**: 测试半数专家 2x 流量时两个后端的性能变化

### 配置

```
imbalance_mode: hotspot
hotspot_count: 128 (256 个专家中 128 个为 "热" 专家)
hotspot_ratio: 0.6667 (2/3 流量到热专家 → 每个热专家 2x 负载)
non_hotspot_alpha: 10000.0 (冷专家均匀分布)
```

### EPMoE vs Default FusedEPMoE — Hotspot

```
num_tokens |  EPMoE (ms) |  Fused (ms) |  Speedup | Winner
----------------------------------------------------------
        32 |       0.739 |       1.428 |    0.52x |  EPMoE
        64 |       0.797 |       1.306 |    0.61x |  EPMoE
       128 |       0.904 |       1.306 |    0.69x |  EPMoE
       256 |       0.978 |       1.418 |    0.69x |  EPMoE
       512 |       1.271 |       1.511 |    0.84x |  EPMoE
      1024 |       1.785 |       2.942 |    0.61x |  EPMoE
      2048 |       2.839 |       5.513 |    0.51x |  EPMoE
      4096 |       4.409 |      11.099 |    0.40x |  EPMoE
      8192 |       8.378 |      21.124 |    0.40x |  EPMoE

Geo-mean: 0.57x (EPMoE wins)
```

### Balanced vs Hotspot 影响分析

```
tokens | EPMoE bal→hot | 变化  | Fused bal→hot      | 变化
-------|-------------|------|--------------------|-------
    32 | 0.786→0.739 |  -6% | 1.315→1.428        |   +9%
   128 | 0.863→0.904 |  +5% | 1.316→1.306        |   -1%
   512 | 1.241→1.271 |  +2% | 1.531→1.511        |   -1%
  1024 | 1.726→1.785 |  +3% | 2.497→2.942        |  +18%
  2048 | 2.588→2.839 | +10% | 4.399→5.513        |  +25%
  4096 | 4.325→4.409 |  +2% | 8.382→11.099       |  +32%
  8192 | 8.096→8.378 |  +3% | 14.867→21.124      |  +42%
```

### 分析

- **EPMoE 对 imbalance 免疫** (±2-10%): 通信为 replicate+psum，与 routing 分布无关
- **FusedEPMoE 对 imbalance 高度敏感**: 小 token 影响小 (<10%)，大 token 退化严重 (25-42%)
- **原因**: hotspot 导致某些设备接收更多 all-to-all DMA 流入 + 更长 FFN 计算，
  pipeline 中其他设备 stall 等待，破坏了 DMA-compute 重叠
- **影响**: 如果生产环境 routing 不够均衡，FusedEPMoE 的实际收益会打折扣
  - Balanced: FusedEPMoE 快 1.23x (geo-mean)
  - Hotspot 2:1: FusedEPMoE (default) 慢 0.57x — 需要验证 tuned config 下的 imbalance 敏感性

---

## 10. 全局总结

### 数据汇总 (Balanced, 最终优化)

```
tokens | EPMoE (ms) | Fused-Default | R1 Tuned | R2 Tuned | Best Config              | EPMoE/Best
-------------------------------------------------------------------------------------------------
    32 |      0.786 |         1.315 |    0.753 |    —     | R1 full_hidden           |    1.04x
    64 |      0.786 |         1.327 |    0.772 |    —     | R1 v7_fp8                |    1.02x
   128 |      0.863 |         1.398 |    0.769 |    0.744 | R2 btc16+bse2048+bd2c2048|    1.16x
   256 |      1.071 |         1.339 |    0.786 |    0.777 | R2 btc16+bse2048+bd2c2048|    1.38x
   512 |      1.241 |         1.507 |    0.850 |    0.799 | R2 btc16+bse128          |    1.55x
  1024 |      1.726 |         2.609 |    1.334 |    1.215 | R2 btc16+bse128          |    1.42x
  2048 |      2.588 |         4.808 |    2.193 |    2.047 | R2 btc16                 |    1.26x
  4096 |      4.325 |         9.218 |    3.980 |    3.677 | R2 btc16+bse128          |    1.18x
  8192 |      8.096 |        18.019 |    7.558 |    6.968 | R2 btc16                 |    1.16x
```

### 关键发现

1. **Block config tuning 带来 2-3 轮共 ~60-70% 的提升** (相对 default config)
2. **btc=16 是最关键的单一优化**: 比 btc=32 快 5-32%，尤其在 ≥1024 tokens
3. **Tuned FusedEPMoE 在 balanced 下全面优于 EPMoE**: geo-mean 1.23x
4. **FusedEPMoE 对 imbalance 敏感**: 大 token + hotspot 时退化 25-42%
5. **v6e VMEM 约束限制 tile 组合**: bt=64/128 + bd1=4096 导致 OOM
6. **推荐两个 config**:
   - 小 token (≤256): `bt=32 bf=2048 bd1=4096 bd2=4096 btc=16 bd2c=2048 bse=2048`
   - 大 token (≥512): `bt=32 bf=2048 bd1=4096 bd2=4096 btc=16 bd2c=4096 bse=512`
