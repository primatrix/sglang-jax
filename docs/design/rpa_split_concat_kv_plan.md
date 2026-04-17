# Split RPA Kernel 优化报告：消除 cross-sublane gather

日期：2026-04-10 | 硬件：TPU v6e-4 | 分支：`worktree-feat+rpa-split-opt`

## 摘要

Split RPA kernel 的 `unpack_heads` 函数在 VMEM 中执行 3D reshape + 中间维度 slice，触发 TPU 上的 cross-sublane gather（`vrot.slane + vsel` 级联），导致 split_128_128 比 fused_128 慢 2.54x。通过将数据路径改为 uint32 bitcast → 2D 行级 stride → bf16 unpack（与 fused kernel 的 `strided_load_bkv` 相同策略），消除了全部 ~28,000 条 gather 指令，**split_128_128 从 1.54ms 降至 0.58ms（2.64x 加速），现在与 fused_128 持平（0.97x）**。

---

## 1. 背景

### 1.1 问题

MiMo-V2-Flash 等模型使用不同的 K/V head 维度（如 k_dim=192, v_dim=128），需要 split KV cache 路径。Split kernel 在功能上正确，但性能远差于 fused kernel：

| Config | 时间 (ms) | vs fused_128 |
|---|---:|---:|
| fused_128 | 0.60 | 1.00x |
| split_128_128 | 1.54 | **2.54x** |
| split_192_128 | 2.31 | **3.82x** |

即使 k_dim == v_dim == 128（与 fused 完全相同的计算量），split 仍然慢 2.54x。

### 1.2 前序分析

- [04-08 Profiling Report](../2026-04-08/rpa_profiling_report.md)：识别出 split vs fused 的性能差距
- [04-09 Optimization Report](../2026-04-09/split_rpa_optimization_report.md)：Phase 1-2 优化（tuned_block_sizes、zero-copy 等），将 split_128 从 ~4.2ms 降至 ~1.54ms，但剩余 ~1ms 底线无法突破
- [04-10 LLO Analysis](llo_analysis_report.md)：LLO 指令级分析定位根因

---

## 2. 根因分析（LLO 指令级）

### 2.1 分析方法

对 split_128_128、fused_128、fused_256 三个 kernel 的 LLO dump（`final_bundles.txt`）做指令统计、per-bundle 功能单元利用率分析、编译 pipeline 阶段对比。

### 2.2 根因：`unpack_heads` 的 cross-sublane gather

Split kernel 的 `strided_load_kv_separate` → `unpack_heads` 函数：

```python
# 原始代码（慢）
def unpack_heads(ref, head_start_idx, num_heads_to_load, head_dim_val):
    ref_flat = ref.reshape(bkv_sz, -1, head_dim_val)        # 3D reshape
    heads = ref_flat[:, head_start_idx : head_start_idx + num_heads_to_load, :]  # 中间维 slice
    ...
```

在 TPU VMEM 中，数据按 `(sublanes=8, lanes=128)` tile 存储。对 3D array 做中间维 slice 时，目标 head 的数据散落在不同 sublane 位置。编译器生成 7 级 `vrot.slane + vsel` 级联链来跨 sublane 收集数据：

```
vrot.slane val, 6  →  vsel vm11, rotated, fallback
vrot.slane val, 5  →  vsel vm12, rotated, prev
...（共 7 级）
vrot.slane val, 1  →  vsel vm1,  rotated, prev   ← 最终结果
```

### 2.3 影响量化

| 指令 | split_128 | fused_128 | 差异 |
|---|---:|---:|---|
| vld.sshfl | 12,288 | 0 | **split 独有** |
| vrot.slane | 9,598 | 393 | **24.4x** |
| vsel | 6,872 | 164 | **41.9x** |
| **Gather 合计** | **28,758** | **557** | **+28,201** |
| 总 bundle 数 | 31,606 | 6,590 | **4.8x** |
| **MXU 利用率** | **1,120** | **1,120** | **完全相同** |

核心计算（MXU、XLU、EUP）一条指令不差。**100% 的性能差距来自数据搬运开销。**

### 2.4 瓶颈分解

| 因素 | 贡献 | 可优化？ |
|---|---|---|
| `unpack_heads` gather 模式 | ~70%（pre-RA 3.4x 差距） | **是 — 本次修复** |
| VLIW 打包降级 | ~30%（gather 占满 VALU/VLOAD 槽位） | 随 gather 消除自动恢复 |
| VREG 溢出 | 不直接增加 bundle（被 VLIW 打包吸收） | 随 gather 消除自动解决 |
| MXU / 核心计算 | 0% | 已最优 |

---

## 3. 修复方案

### 3.1 核心思路

参考 fused kernel 的 `strided_load_bkv` 实现：

1. **Bitcast to uint32**：`ref.bitcast(uint32)` 将 kv_packing 维度（bf16 时 size=2）折叠为 1，每个 uint32 包含 2 个 packed bf16 值
2. **Reshape to 2D**：`[bkv_sz * groups, head_dim]` uint32
3. **Row-stride selection**：`ref_2d[group_idx :: groups]` — 行级选择映射到 VMEM tile 行访问，无 cross-sublane 移动
4. **Unpack bf16**：`val.astype(uint16)` (低 16 位) 和 `(val >> 16).astype(uint16)` (高 16 位)，再 `pltpu.bitcast(uint16, bf16)`（同宽度，无 shape 变化）

### 3.2 关键技术障碍与解决

#### 障碍 1：Mosaic 不支持非 32 位数据的 strided load

直接在 bf16 ref 上做 `ref_2d[idx::step]` 会报错：

```
not implemented: Strided load with non 32-bit data
```

**解决**：先 bitcast 到 uint32 再做 stride。

#### 障碍 2：`pltpu.bitcast` 的维度语义

`pltpu.bitcast` 不是简单的内存重解释。从 JAX 源码（`pallas/mosaic/primitives.py:73`）确认：

```python
# 只改变 shape[-2]（sublane 维度），shape[-1]（lane 维度）不变
output_shape[-2] = input_shape[-2] * src_bits // dst_bits
```

| 转换 | 输入 shape | 输出 shape |
|---|---|---|
| `[N, M]` uint32 → bf16 | `[N, M]` | `[2*N, M]` — 第一维翻倍 |
| `[N, M]` uint16 → bf16 | `[N, M]` | `[N, M]` — 同宽度不变 |
| `[N, M]` uint32 → float32 | `[N, M]` | `[N, M]` — 同宽度不变 |

之前的尝试因为误解了这个语义而失败（以为是最后一维翻倍）。

**解决**：不对结果做 uint32→bf16 的 `pltpu.bitcast`（会导致维度翻倍）。改为先 truncate 到 uint16（低/高 16 位），再 `pltpu.bitcast(uint16, bf16)`（同宽度，不变 shape）。这正是 fused kernel 的 `_convert_to_target_bitwidth` 使用的方法。

#### 障碍 3：Fold trick 的 128 对齐约束

Fused kernel 的 `strided_load` 要求 `head_dim % 128 == 0`（TPU tile 宽度）。由于 `_prepare_single_kv` 会将 head_dim 对齐到 128（`align_to(192, 128) = 256`），kernel 内部的 head_dim 始终满足此约束。添加了 fallback 路径以防万一。

### 3.3 最终代码

```python
def _strided_load(ref, start, step):
    """Row-stride selection on a 2D uint32 ref. Uses fold trick for 128-aligned dims."""
    assert get_dtype_packing(ref.dtype) == 1
    assert len(ref.shape) == 2
    r, minor = ref.shape
    if minor % 128 == 0:
        folds = minor // 128
        ref = ref.reshape(r * folds, 128)
        start *= folds
        step *= folds
        return jnp.concat([ref[start + i :: step] for i in range(folds)], axis=1)
    else:
        return ref[start :: step]

def strided_load_kv_separate(bkv_sem_idx, start, step):
    head_idx_start = start // 2
    heads_per_load = max(1, kv_packing // 2)

    kv_k_dtype = bk_x2_ref.dtype
    kv_v_dtype = bv_x2_ref.dtype

    # Bitcast to uint32: collapses kv_packing dim (2→1 for bf16)
    k_ref = bk_x2_ref.bitcast(jnp.uint32).at[bkv_sem_idx]
    v_ref = bv_x2_ref.bitcast(jnp.uint32).at[bkv_sem_idx]

    # Flatten to 2D: [bkv_sz * groups, head_dim] uint32
    k_ref_2d = k_ref.reshape(bkv_sz * num_kv_heads_per_kv_packing, k_head_dim)
    v_ref_2d = v_ref.reshape(bkv_sz * num_kv_heads_per_kv_packing, v_head_dim)

    results = []
    for i in range(heads_per_load):
        h = head_idx_start + i
        g = h // kv_packing
        p = h % kv_packing

        k_packed = _strided_load(k_ref_2d, g, num_kv_heads_per_kv_packing)
        v_packed = _strided_load(v_ref_2d, g, num_kv_heads_per_kv_packing)

        if kv_packing == 1:  # float32
            k_val = pltpu.bitcast(k_packed, kv_k_dtype)
            v_val = pltpu.bitcast(v_packed, kv_v_dtype)
        else:  # bf16: unpack from uint32
            if p == 0:  # low 16 bits
                k_val = pltpu.bitcast(k_packed.astype(jnp.uint16), kv_k_dtype)
                v_val = pltpu.bitcast(v_packed.astype(jnp.uint16), kv_v_dtype)
            else:  # high 16 bits
                k_val = pltpu.bitcast((k_packed >> 16).astype(jnp.uint16), kv_k_dtype)
                v_val = pltpu.bitcast((v_packed >> 16).astype(jnp.uint16), kv_v_dtype)

        results.append((k_val, v_val))
    return results
```

---

## 4. 结果

### 4.1 精度验证

全部 7 个测试通过（decode/prefill × k_dim=128/192/256 × page_size=64/128）：

```
  [decode]  k_dim=128, v_dim=128  PASS  (max_diff=0.000639)
  [decode]  k_dim=256, v_dim=128  PASS  (max_diff=0.000709)
  [decode]  k_dim=192, v_dim=128  PASS  (max_diff=0.000714)
  [decode]  k_dim=192, v_dim=128, q_heads=8, kv_heads=4  PASS  (max_diff=0.000694)
  [prefill] k_dim=128, v_dim=128  PASS  (max_diff=0.007822)
  [prefill] k_dim=256, v_dim=128  PASS  (max_diff=0.007732)
  [prefill] k_dim=192, v_dim=128  PASS  (max_diff=0.007193)
```

### 4.2 Decode 性能（trace-based timing, batch=128, ctx=4096, ps=64）

| Config | 优化前 (ms) | 优化后 (ms) | 加速比 | vs fused_128 |
|---|---:|---:|---:|---:|
| fused_128 | 0.60 | 0.60 | — | 1.00x |
| **split_128_128** | **1.54** | **0.58** | **2.64x** | **0.97x** |
| **split_192_128** | **2.31** | **0.77** | **3.00x** | **1.28x** |
| **split_256_128** | **2.31** | **0.77** | **3.00x** | **1.28x** |
| fused_256 | 0.86 | 0.86 | — | 1.43x |

### 4.3 Prefill 性能

| Config | 优化前 (ms) | 优化后 (ms) | 加速比 | vs fused_128 |
|---|---:|---:|---:|---:|
| fused_128 | 0.55 | 0.55 | — | 1.00x |
| **split_128_128** | **1.62** | **0.57** | **2.84x** | **1.04x** |
| **split_192_128** | **2.38** | **0.70** | **3.40x** | **1.28x** |
| **split_256_128** | **2.38** | **0.70** | **3.40x** | **1.27x** |
| fused_256 | 1.00 | 1.00 | — | 1.83x |

### 4.4 关键结论

- **split_128_128 与 fused_128 持平**（decode 0.58ms vs 0.60ms, 0.97x）—— 证明 split 路径的固有开销已被完全消除
- **split_192_128 从 3.82x 降至 1.28x** —— 剩余差距来自 k_dim=256（192 对齐后）vs 128 的额外 DMA 和 MXU 计算，属于固有计算量差异
- split_192_128 == split_256_128（0.77ms vs 0.77ms）—— 确认 192 对齐到 256 后行为一致

---

## 5. 探索路径与失败尝试

在找到正确方案之前，经历了以下探索：

| 尝试 | 方法 | 结果 | 失败原因 |
|---|---|---|---|
| 1 | 2D reshape + stride（bf16 直接） | 编译失败 | Mosaic 不支持非 32 位 strided load |
| 2 | bitcast int32 + stride + `pltpu.bitcast(int32→bf16)` | Shape 不匹配 | `pltpu.bitcast` 改变 `shape[-2]` 而非 `shape[-1]`，导致维度翻倍 |
| 3 | bitcast int32 + stride + `pltpu.bitcast` + reshape | Shape 不整除 | reshape 无法正确恢复原始维度 |
| 4 | 直接 4D indexing `ref[:, g, p, :]` | 编译通过、精度正确 | **性能无变化** —— 编译器生成相同 gather |
| **5** | **bitcast uint32 + stride + uint16 truncate/shift + `pltpu.bitcast(uint16→bf16)`** | **成功** | **正确方案** |

关键 insight：必须在 uint16 级别做 `pltpu.bitcast`（同宽度 16→16 位，不改变 shape），而不是在 uint32 级别做（32→16 位会改变 shape[-2]）。

---

## 6. 文件变更

| 文件 | 变更 |
|---|---|
| `ragged_paged_attention_split.py` | 新增 `_strided_load` helper；重写 `strided_load_kv_separate`（~40 行替换） |

改动仅限 `strided_load_kv_separate` 函数，不涉及 kernel grid spec、DMA、attention 计算等其他路径。

## 7. 复现

```bash
# 1. Sync worktree
rsync -az --delete --exclude='.git' --exclude='__pycache__' \
  .claude/worktrees/feat+rpa-split-opt/ \
  sky-efe2-yuhao:~/sky_workdir/sglang-jax-feat-rpa-split-opt/

# 2. Accuracy test
ssh sky-efe2-yuhao "source ~/miniconda3/etc/profile.d/conda.sh && conda activate py312 && \
  cd ~/sky_workdir/sglang-jax-feat-rpa-split-opt && \
  PYTHONPATH=\$PWD/python:\$PWD/benchmark/kernels/flash_attention \
  python benchmark/kernels/flash_attention/test_split_ps128_accuracy.py"

# 3. Performance benchmark (trace-based)
ssh sky-efe2-yuhao "source ~/miniconda3/etc/profile.d/conda.sh && conda activate py312 && \
  cd ~/sky_workdir/sglang-jax-feat-rpa-split-opt && \
  PYTHONPATH=\$PWD/python:\$PWD/benchmark/kernels/flash_attention \
  python benchmark/kernels/flash_attention/benchmark_rpa_compare.py"
```
