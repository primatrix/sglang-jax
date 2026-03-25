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
