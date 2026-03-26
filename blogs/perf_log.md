# Fused MoE Kernel 性能优化记录

## 测试配置
- Hardware: TPU v7, 单机 4 卡 (8 cores), EP=8
- Target shape: tokens=512, experts=64, top_k=8, hidden=5120, intermediate=2048

## 精度测试命令
```bash
python -m sgl_jax.test.kernels.fused_moe_v1_test
```

## 性能测试命令
```bash
python -m benchmark.moe.bench_fused_moe_kernel --iters 5
```

## 优化轮次记录

| Round | 改动内容 | Kernel Time (ms) | 变化 | 精度 | Commit |
|-------|---------|-----------------|------|------|--------|
| R0 (baseline) | 未修改 | 1.408 | - | 20/20 PASS | - |
| R1 | Pre-sort tokens + bulk scatter DMA | 1.319 | -6.3% | 20/20 PASS | 416634e6 |
| R2 | Block config tuning: bt=64, bf=2048, bd1/bd2=1280 | 0.312 | -76.3% | 20/20 PASS | 4c2b31de |
| R3 | Remove 4 redundant sync_barriers | 0.307 | -1.6% | 20/20 PASS | 9624b5d3 |
| R4 | Asymmetric block config: bd1=2560 (num_bd1: 4→2) | 0.299 | -2.6% | 20/20 PASS | ad780690 |
| R5 | Unroll 3 static fori_loops (prefix sum + gather acc) | 0.298 | -0.3% | 20/20 PASS | bfe353e3 |
| R6 | Pre-compute allgather metadata in JAX layer | 0.289 | -3.0% | 20/20 PASS | 62c43de0 |
| R7 | Bulk DMA wait in acc_and_store_output | 0.287 | -0.7% | 20/20 PASS | TBD |
| R8 | Skip last sync_barrier in run_bt | 0.286 | -0.3% | 20/20 PASS | TBD |

## 详细记录

### Round 0: Baseline
- **精度测试**: 20/20 PASS (336s), 1 skipped (需要真实模型权重)
- **性能测试**: mean=1.408ms, min=1.407ms, max=1.408ms
- **samples**: [1.4076, 1.4085, 1.4080, 1.4074]

### Round 1: Pre-sort tokens + bulk scatter DMA
- **改动**: 在 JAX 层 pre-sort tokens by (tile_id, expert_id)，kernel 内用 O(num_devices)=O(8) 次 bulk DMA 替代 O(bt*top_k)=O(2048) 次标量 fori_loop
- **精度测试**: 20/20 PASS (345s), 1 skipped
- **性能测试**: mean=1.319ms, min=1.317ms, max=1.321ms
- **samples**: [1.3206, 1.3175, 1.3186, 1.3204]
- **变化**: -0.089ms (-6.3%)

### Round 2: Block config tuning (bt=64, bf=2048, bd1/bd2=1280)
- **改动**: 为目标 shape (512, 64, 8, 5120, 2048, ep=8) 添加 tuned block config:
  - bt=64 (was 32): num_bt 从 2 降为 1，消除一次完整外层循环
  - bf=2048 (was 512): num_bf 从 4 降为 1，减少 weight loading 轮次
  - bd1=bd2=1280 (was 1024): num_bd1/bd2 从 5 降为 4
- **精度测试**: 20/20 PASS (329s), 1 skipped
- **性能测试**: mean=0.312ms, min=0.312ms, max=0.313ms
- **samples**: [0.3127, 0.3127, 0.3125, 0.3119]
- **变化**: -1.007ms (-76.3%) vs R1, -1.096ms (-77.8%) vs baseline
- **尝试的其他配置**:
  - bt=64, bf=512, bd1=1024: 0.664ms
  - bt=64, bf=2048, bd1=1024: 0.321ms
  - bt=64, bf=2048, bd1=2560: VMEM OOM (74.28M > 64M)
  - bt=64, bf=2048, bd1=5120: VMEM OOM (135.71M > 64M)

### Round 3: Remove 4 redundant sync_barriers
- **改动**: 移除 4 个冗余的 sync_barrier:
  1. all_reduce_metadata 内的 pre-allgather barrier（与 round 0 barrier 合并）
  2. all_reduce_metadata 内的 post-allgather barrier（各设备 recv_wait 已保证本地数据一致）
  3. all_reduce_metadata 后的 barrier（allreduce 内部已有足够同步）
  4. wait_a2a_gather_recv_all 后的 barrier（gather recv 已完成，后续操作均为本地）
- **精度测试**: 20/20 PASS (333s), 1 skipped
- **性能测试**: mean=0.307ms, min=0.307ms, max=0.308ms
- **samples**: [0.3065, 0.3076, 0.3070, 0.3067]
- **变化**: -0.005ms (-1.6%)
- **总变化**: -1.101ms (-78.2%) vs baseline

### Round 4: Asymmetric block config (bd1=2560, bd2=1280)
- **改动**: 使用非对称 block config，bd1=2560 (gate/up projection 从 4 tiles 降为 2 tiles)，bd2=1280 (down projection 保持 4 tiles)。bd1=2560 接近 VMEM 上限，需要 bd2 < bd1 才能编译成功。
- **精度测试**: 20/20 PASS (333s), 1 skipped
- **性能测试**: mean=0.299ms, min=0.297ms, max=0.300ms
- **samples**: [0.2987, 0.2974, 0.2990, 0.2996]
- **变化**: -0.008ms (-2.6%)
- **总变化**: -1.109ms (-78.8%) vs baseline

### Round 5: Unroll 3 static fori_loops
- **改动**: 将 3 个静态边界 fori_loop 改为 unroll=True，消除标量循环控制流开销:
  1. `_compute_sorted_starts`: 64 次 prefix sum 循环（每次写 SMEM + 读 SMEM + 加法）
  2. `start_load_acc_bt._load_one`: 16 次 gather DMA 启动循环（每次读 SMEM + 启动 8 个 DMA）
  3. `wait_load_acc_bt._count_valid`: 16 次有效 token 计数循环
- **精度测试**: 20/20 PASS (347s), 1 skipped
- **性能测试**: mean=0.298ms, min=0.297ms, max=0.299ms
- **samples**: [0.2976, 0.2983, 0.2978, 0.2970, 0.2981, 0.2980, 0.2981, 0.2975, 0.2990]
- **变化**: -0.001ms (-0.3%)
- **总变化**: -1.110ms (-78.8%) vs baseline
- **未生效的尝试**:
  - DMA priority=1 on weight fetches: 0.318ms (regression +6.3%，scatter/gather 被降优先级延迟)
  - Earlier weight prefetch (bd2 second-to-last): 0.300ms (无变化)
  - wait_a2a_gather_recv_all unroll=True: 无额外改善

### Round 6: Pre-compute allgather metadata in JAX layer
- **改动**: 将 O(log2(num_devices))=O(3) 轮 recursive doubling allgather 从 Pallas kernel 移至 JAX 图层：
  - 在 shard_map 内用 `jax.nn.one_hot` + `jax.lax.all_gather` 预计算 per-bt-tile 的 d2e_count, expert_starts, expert_sizes
  - 打包为 `ep_metadata_hbm` (num_bt, num_devices+2, 1, padded_num_experts) 传入 kernel
  - kernel 内 `all_reduce_metadata` 从 3 轮 remote copy + 3 sync_barrier 简化为 3 次 HBM→VMEM DMA + VMEM→SMEM copy
  - 修复 semaphore race: HBM→VMEM DMA 使用独立 local_sem 避免与 VMEM→SMEM 的 send_sem 计数器冲突
- **精度测试**: 20/20 PASS (341s), 1 skipped
- **性能测试**: mean=0.289ms, min=0.288ms, max=0.290ms
- **samples**: [0.2888, 0.2897, 0.2890, 0.2877]
- **变化**: -0.009ms (-3.0%)
- **总变化**: -1.119ms (-79.5%) vs baseline

### Round 7: Bulk DMA wait in acc_and_store_output
- **改动**: 将 `wait_load_acc_bt` 中 `num_valid * top_k` 次标量 fori_loop 等待替换为单次 bulk DMA wait:
  - 原代码: `fori_loop(0, num_valid * top_k, _wait_one, ...)` → 最多 128 次/tile × 4 tiles = 512 次标量等待
  - 新代码: `pltpu.make_async_copy(wait_ref, wait_ref, sem).wait()` → 1 次 bulk wait/tile
  - `wait_ref` 大小 = `(top_k, num_valid, t_packing, h_per_t_packing)` 匹配所有 DMA 传输的总字节数
- **精度测试**: 20/20 PASS (347s), 1 skipped
- **性能测试**: mean=0.287ms, min=0.286ms, max=0.288ms
- **samples**: [0.2869, 0.2878, 0.2862, 0.2862]
- **变化**: -0.002ms (-0.7%)
- **总变化**: -1.121ms (-79.6%) vs baseline

### Round 8: Skip last sync_barrier in run_bt
- **改动**: 将 `run_bt` 末尾的 `sync_barrier()` 改为 `@pl.when(bt_id + 1 < num_bt)` 条件执行：
  - 最后一轮 bt 不需要 barrier（无后续 bt 需要同步）
  - 所有跨设备通信已通过 wait_a2a_gather_recv_all + wait_a2a_gather_send 确保完成
  - 对 num_bt=1 场景，直接省去一次 barrier (~2μs)
- **精度测试**: 20/20 PASS (337s), 1 skipped
- **性能测试**: mean=0.286ms, min=0.285ms, max=0.286ms
- **samples**: [0.2857, 0.2855, 0.2862, 0.2851]
- **变化**: -0.001ms (-0.3%)
- **总变化**: -1.122ms (-79.7%) vs baseline

### Gap Analysis (after R8)
- **理论下限**: ~210μs (HBM 带宽限制: ~500MB 权重 / 2.4TB/s)
- **当前**: 286μs
- **差距**: ~76μs (27%)
- **差距组成** (估算):
  - DMA 启动开销 (~164 次 DMA × ~150ns): ~25μs
  - 标量 ALU 处理 (SMEM 读取, 条件判断, 循环控制): ~24μs
  - ICI 传输延迟 (scatter/gather remote copies): ~8μs
  - 流水线启动/排空成本: ~4μs
  - sync_barrier 同步: ~16μs
- **尝试但未生效的 R9 优化**:
  - 分离 barrier signal/wait (deferred barrier): 0% (TPU v7 单机 barrier 已足够快)
  - 批量 gather recv wait (bulk DMA wait 替代 64 次循环): 0% (DMA 等待已即时完成)
  - 预计算 sorted_scatter_starts 到 JAX 层: 0% (DMA 加载开销抵消循环节省)
  - 提前启动首个 expert 权重预取: 0% (已有足够流水线深度)
- **结论**: 此 shape (512, 64, 8, 5120, 2048) 已接近饱和，剩余差距分散在大量 <1μs 的微开销中

---

## Block Config Tuning: bf16 (N, 64, 8, 8192, 2048, EP=8)

### 目标
为 hidden=8192, intermediate=2048, 64 experts 的 bf16 shape 添加 tuned block configs。

### VMEM 约束分析
- hidden=8192 + intermediate=2048, bf16: 大型权重矩阵
- bd1=2048, bd2=2048 VMEM OOM (64.37M > 64M)
- bd1=4096 VMEM OOM (任何 bt 配置)
- 有效 bd 必须整除 8192: 1024, 2048, 4096, 8192

### 最优配置发现

| Tokens | bt | bf | bd1 | bd2 | Default (ms) | Tuned (ms) | 改善 |
|--------|-----|------|------|------|-------------|-----------|------|
| 64 | 8 | 2048 | 2048 | 1024 | 0.790 | 0.308 | -61.0% |
| 128 | 16 | 2048 | 2048 | 1024 | 0.825 | 0.323 | -60.8% |
| 256 | 32 | 2048 | 2048 | 1024 | 0.884 | 0.363 | -58.9% |
| 512 | 64 | 2048 | 2048 | 1024 | 1.969 | 0.459 | -76.7% |
| 1024 | 128 | 2048 | 1024 | 1024 | 3.647 | 0.782 | -78.6% |
| 2048 | 256 | 1024 | 1024 | 1024 | 6.977 | 1.697 | -75.7% |
| 4096 | 128 | 2048 | 1024 | 1024 | 14.095 | 3.516 | -75.1% |

### 配置选择洞察
- **bf=2048 > bf=1024** (num_bf=1 vs 2): 减少 weight loading 外层循环，在大多数 token counts 有利
- **VMEM 限制 bt**: bt≥128 时 bd1=2048 OOM，被迫降至 bd1=1024
- **2048 tokens 例外**: bt=256, bf=1024 (1.697ms) 优于 bt=128, bf=2048 (1.885ms)，因为减少 num_bt 从 2→1 的收益 > num_bf 从 1→2 的代价
- **精度测试**: 20/20 PASS

---

## Block Config Tuning: bf16 (N, 64, 8, 5120, 2048, EP=8)

### VMEM 约束
- hidden=5120: bd 必须整除 5120 → 1280, 2560, 5120
- bd1=2560 在 bt≤64 时可用 (R4 已验证)
- bt≥128 时 bd1=2560 OOM，降至 bd1=1280

### 最优配置

| Tokens | bt | bf | bd1 | bd2 | Default (ms) | Tuned (ms) | 改善 |
|--------|-----|------|------|------|-------------|-----------|------|
| 64 | 8 | 2048 | 2560 | 1280 | 0.522 | 0.193 | -63.0% |
| 128 | 16 | 2048 | 2560 | 1280 | 0.543 | 0.200 | -63.2% |
| 256 | 32 | 2048 | 2560 | 1280 | 0.583 | 0.229 | -60.7% |
| 512 | 64 | 2048 | 2560 | 1280 | (R2) | 0.286 | - |
| 1024 | 128 | 2048 | 1280 | 1280 | 2.354 | 0.498 | -78.8% |
| 2048 | 256 | 1024 | 1280 | 1280 | 4.543 | 1.087 | -76.1% |
| 4096 | 128 | 2048 | 1280 | 1280 | 9.167 | 2.225 | -75.7% |

- **精度测试**: 20/20 PASS

---

## Block Config Tuning: bf16 (N, 128, 8, 4096, 1536, EP=8) — Qwen3-235B

### VMEM 约束
- hidden=4096, inter=1536: bd 必须整除 4096, bf 必须整除 1536
- bf=1536 (num_bf=1) 最优
- bd1=4096 VMEM OOM (即使 bt=64)
- bd1=2048, bd2=2048 在 bt≤128 时可用

### 最优配置

| Tokens | bt | bf | bd1 | bd2 | Default (ms) | Tuned (ms) | 改善 |
|--------|-----|------|------|------|-------------|-----------|------|
| 64 | 8 | 1536 | 2048 | 2048 | 0.607 | 0.222 | -63.4% |
| 128 | 16 | 1536 | 2048 | 2048 | 0.626 | 0.229 | -63.4% |
| 256 | 32 | 1536 | 2048 | 2048 | 0.662 | 0.250 | -62.2% |
| 512 | 64 | 1536 | 2048 | 2048 | 1.014 | 0.286 | -71.8% |
| 1024 | 128 | 1536 | 2048 | 2048 | 2.179 | 0.390 | -82.1% |
| 2048 | 128 | 1536 | 2048 | 2048 | 3.674 | 0.962 | -73.8% |
| 4096 | 128 | 1536 | 2048 | 2048 | 7.242 | 2.003 | -72.3% |

- **精度测试**: 20/20 PASS

---

## Block Config Re-tune: bf16 (N, 128, 8, 2048, 768, EP=8) — Qwen3-30B

### 改动
将 bf 从 256 (num_bf=3) 提升到 768 (num_bf=1)，消除 2 轮 intermediate dimension 循环。

### 结果

| Tokens | bf (old→new) | Old (ms) | New (ms) | 改善 |
|--------|-------------|----------|---------|------|
| 64 | 256→768 | 0.193 | 0.089 | -53.9% |
| 128 | 256→768 | - | 0.093 | - |
| 256 | 256→768 | - | 0.108 | - |
| 512 | 256→768 | 0.258 | 0.143 | -44.6% |
| 1024 | 256→768 | 0.350 | 0.206 | -41.1% |
| 2048 | 256→768 | - | 0.432 | - |
| 4096 | 256→768 | 1.033 | 0.904 | -12.5% |
| 8192 | 256→768 | - | 1.914 | - |

- bt=512 with bf=768 OOM，4096/8192 tokens 降至 bt=256
- **精度测试**: 20/20 PASS

---

## Block Config Tuning: bf16 8192 tokens (64E, 8192h, 2048i)

### 问题
8192 tokens 无 tuned config，使用 default (bt=32, bf=512, bd1=1024, bd2=1024) → **28.265ms**。

### 调优结果

| Config | bt | bf | bd1 | bd2 | Time (ms) |
|--------|-----|------|------|------|-----------|
| Default | 32 | 512 | 1024 | 1024 | 28.265 |
| **A (最优)** | **128** | **2048** | **1024** | **1024** | **7.264** |
| B | 256 | 1024 | 1024 | 1024 | 7.853 |

- **改善**: 28.265ms → 7.264ms (**-74.3%**)
- Config A (bf=2048, num_bf=1) 优于 Config B (bf=1024, num_bf=2)，与之前 shape 的规律一致
- **精度测试**: 20/20 PASS

---

## Block Config Tuning: FP8 (64E, 8192h, 2048i) + Benchmark 支持

### 改动
1. 为 `bench_fused_moe_kernel.py` 添加 `--weight-dtype` 参数，支持 FP8 性能测试
2. 优化 FP8 block configs：主要改善 btc (contraction tile) 和 bd/bf 参数

### FP8 调优关键发现
- FP8 `subc_quant_wsz=256` 固定 `bd1c=512, bfc=256`，不可调
- FP8 权重半大小允许 **bd1=bd2=2048**（bf16 因 VMEM 限制只能 bd1=2048/bd2=1024）
- **btc=bt** 对小 token 更好，**btc=32** 对大 token (4096+) 反而更好（更细粒度流水线）
- FP8 8192t 原 config bf=512 (num_bf=4) 是严重瓶颈，改为 bf=2048 后提升 35%

### bf16 vs FP8 完整对比

| Tokens | bf16 (ms) | FP8 旧 (ms) | FP8 新 (ms) | FP8 优化 | FP8 vs bf16 |
|--------|-----------|-------------|-------------|---------|-------------|
| 64 | 0.308 | 0.164 | 0.164 | - | **-46.8%** |
| 128 | 0.323 | 0.217 | 0.217 | - | **-32.8%** |
| 256 | 0.363 | 0.273 | 0.273 | - | **-24.8%** |
| 512 | 0.459 | 0.449 | **0.438** | -2.4% | **-4.6%** |
| 1024 | 0.782 | 0.874 | **0.852** | -2.5% | +9.0% |
| 2048 | 1.697 | 1.936 | **1.860** | -3.9% | +9.6% |
| 4096 | 3.516 | 3.993 | **3.815** | -4.5% | +8.5% |
| 8192 | **7.264** | 12.316 | **7.952** | **-35.4%** | +9.5% |
| 16384 | - | 16.704 | 16.704 | - | - |

### FP8 vs bf16 分析
- **小 token (64-512)**: FP8 显著优于 bf16 (5-47%)，权重加载比例高时 FP8 的带宽优势明显
- **中大 token (1024-8192)**: FP8 反而慢 ~9%，subchannel quantization 的 scale 处理开销抵消了带宽节省
- **精度测试**: 20/20 PASS
