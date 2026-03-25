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
| R3 | Remove 4 redundant sync_barriers | 0.307 | -1.6% | 20/20 PASS | TBD |

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
