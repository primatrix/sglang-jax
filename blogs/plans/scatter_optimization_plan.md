# Fused MoE Kernel: 测试基础设施 + 通信掩盖优化 Plan

## Context

当前 Fused MoE Pallas Kernel 的 `start_a2a_scatter` 函数使用 `lax.fori_loop` 逐 token 散布（`bt * top_k = 2048` 次标量循环），导致 Scalar ALU 阻塞 MXU，通信与计算完全无法 Overlap。优化方案：在 JAX 图层面预排序 Token，在 Kernel 内用 O(1) Bulk DMA 替代逐 token 散布。

**本次计划分两步：先建立测试/基准测试基础设施并跑通 baseline，再实施优化。**

---

## Phase A: 构建精度测试和性能测试脚本

### A.1 精度测试脚本

**新建文件**: `benchmark/moe/test_fused_moe_accuracy.py`

**作用**: 对比 `fused_ep_moe()` vs `ref_moe()` 的数值精度，报告详细 diff 统计。

**复用已有代码**:
- `gen_moe_inputs()` from `python/sgl_jax/test/kernels/fused_moe_v1_test.py:37`
- `ref_moe()` from `python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py:276`
- `fused_ep_moe()` from `python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py:2556`
- `create_device_mesh()` from `python/sgl_jax/test/test_utils.py`
- `TopK` from `python/sgl_jax/srt/layers/moe.py`

**测试配置矩阵**:

| 名称 | tokens | experts | top_k | hidden | inter | weight_dtype |
|------|--------|---------|-------|--------|-------|-------------|
| `smoke` | 64 | 64 | 8 | 1024 | 512 | bf16 |
| `target` | 512 | 64 | 8 | 5120 | 2048 | bf16 |
| `qwen3_30b` | 256 | 128 | 8 | 2048 | 768 | bf16 |
| `target_fp8` | 512 | 64 | 8 | 5120 | 2048 | fp8_e4m3 |

**输出统计项**: max_abs_diff, mean_abs_diff, cosine_sim, pass/fail (atol=0.2, rtol=0.2)

**调用方式**:
```bash
python -m benchmark.moe.test_fused_moe_accuracy
python -m benchmark.moe.test_fused_moe_accuracy --configs target target_fp8
```

### A.2 性能基准测试脚本

**新建文件**: `benchmark/moe/bench_fused_moe_kernel.py`

**作用**: 针对目标 shape 的 kernel 级别精确计时 + profiling。

**复用已有代码**:
- `prepare_fused_moe_inputs()` from `benchmark/moe/utils.py:177`
- `build_mesh()` from `benchmark/moe/utils.py:360`
- `multiple_iteration_timeit_from_trace()` from `benchmark/utils.py:117`
- `MoEImbalanceSimulator` from `benchmark/moe/utils.py:372`
- `FusedEPMoE`, `TopK` layer 组装模式 from `benchmark/moe/bench_fused_moe.py:807-846`

**默认测试 shape** (匹配 profiling 场景):
- tokens=512, experts=64, top_k=8, hidden=5120, intermediate=2048, ep=8

**功能**:
- 计时模式 (默认): warmup + N 次迭代，报告 per-iter 时间和均值
- Profiling 模式 (`--profile`): 输出 xprof trace
- 支持 `--num-tokens`, `--num-experts` 等参数覆盖

**调用方式**:
```bash
# 计时
python -m benchmark.moe.bench_fused_moe_kernel --iters 5

# Profiling
LIBTPU_INIT_ARGS="--xla_enable_custom_call_region_trace=true --xla_xprof_register_llo_debug_info=true" \
python -m benchmark.moe.bench_fused_moe_kernel --profile --profile-dir ./profile_baseline
```

### A.3 跑通 Baseline

在当前 kernel（未修改）上执行：

```bash
# 1. 精度测试
python -m benchmark.moe.test_fused_moe_accuracy 2>&1 | tee baseline_accuracy.log

# 2. 性能基准
python -m benchmark.moe.bench_fused_moe_kernel --iters 5 2>&1 | tee baseline_perf.log

# 3. Profiling (可选)
LIBTPU_INIT_ARGS="--xla_enable_custom_call_region_trace=true --xla_xprof_register_llo_debug_info=true" \
python -m benchmark.moe.bench_fused_moe_kernel --profile --profile-dir ./profile_baseline_v1
bash blogs/tools/profile_export.sh ./profile_baseline_v1
```

---

## Phase B: Kernel 优化 (Baseline 通过后实施)

### B.1 JAX 层面 Token 预排序

**修改文件**: `kernel.py` (在 `fused_ep_moe()` 函数中，约 line 2556+)

在 `pallas_call` 之前添加预排序逻辑:
```python
# 复用 EPMoE._permute() 的模式 (moe.py:626-651)
flat_expert_ids = jnp.ravel(topk_ids)                    # (T*top_k,)
sorted_order = jnp.argsort(flat_expert_ids, stable=True)  # 按 expert_id 排序
sorted_token_ids = sorted_order // top_k                   # 映射回 token 索引
sorted_tokens = tokens[sorted_token_ids]                   # gather
expert_sizes = jnp.bincount(flat_expert_ids, length=num_experts)
expert_starts = jnp.cumsum(expert_sizes) - expert_sizes    # 前缀和
```

将 `sorted_tokens`, `expert_starts`, `expert_sizes` 作为新 HBM 输入传入 Pallas kernel。

### B.2 重写 `start_a2a_scatter` (核心优化)

**修改文件**: `kernel.py`, lines 791-849

**删除**: 内部 `lax.fori_loop(0, bt, _scatter_one, ...)` 和 `range(top_k)` 循环

**替换为**: 按 device 分段的 Bulk DMA（O(num_devices)=O(8) 次操作，而非 O(bt*top_k)=O(2048)）

```python
def start_a2a_scatter_bulk(*, e_sem_id, local_e_id):
    e_id = my_id * local_num_experts + local_e_id
    start = expert_starts_smem[e_id]
    sz = expert_sizes_smem[e_id]

    # 遍历 num_devices（8次），每次发一个 bulk DMA
    for dev_id in range(num_devices):
        dev_sz = per_device_expert_counts[dev_id, e_id]
        is_local = (dev_id == my_id)
        # ... 发射 local 或 remote bulk DMA
```

### B.3 适配 Gather 侧逻辑

**修改文件**: `kernel.py`, `acc_and_store_output` > `start_load_acc_bt` (lines 1924-1951)

`start_load_acc_bt` 有类似的 per-token 标量循环，需要用预计算的 gather 索引做批量 DMA 加载。

### B.4 验证与测量

用 Phase A 的脚本验证优化后的 kernel:
```bash
python -m benchmark.moe.test_fused_moe_accuracy         # 精度不退化
python -m benchmark.moe.bench_fused_moe_kernel --iters 5  # 性能提升
# + profiling 验证 Scalar ALU 气泡消失
```

---

## 关键文件清单

| 文件 | 用途 |
|------|------|
| `python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py` | 核心 kernel，Phase B 所有修改在此 |
| `python/sgl_jax/test/kernels/fused_moe_v1_test.py` | 已有测试，复用 `gen_moe_inputs()` |
| `benchmark/moe/bench_fused_moe.py` | 已有 benchmark，参考其 layer 组装和 profiling 模式 |
| `benchmark/moe/utils.py` | 复用 `prepare_fused_moe_inputs()`, `build_mesh()` |
| `benchmark/utils.py` | 复用 `multiple_iteration_timeit_from_trace()` |
| `python/sgl_jax/srt/layers/moe.py` | 参考 `_permute()` 的 argsort+bincount 模式 |
| `benchmark/moe/test_fused_moe_accuracy.py` | **新建** - 精度测试脚本 |
| `benchmark/moe/bench_fused_moe_kernel.py` | **新建** - 性能基准脚本 |

## 验证方法

1. **精度**: `test_fused_moe_accuracy.py` 全部 pass (atol=0.2, rtol=0.2, cosine_sim > 0.99)
2. **性能**: `bench_fused_moe_kernel.py` 报告的 per-iter 时间 vs baseline
3. **Profiling**: xprof trace 中 Scalar ALU 长条消失，MXU 计算块首尾相连
