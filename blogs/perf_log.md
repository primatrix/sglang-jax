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
