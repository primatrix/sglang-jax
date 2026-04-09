# RPA Split Kernel 优化：拼接 K+V 布局方案

> 日期：2026-04-08 | 分支：`feat/mimo-rebase-v2`

## 问题

MiMo-V2-Flash 的 K/V head 维度不对称：K=192（对齐到 256），V=128。

- **Fused kernel**：K/V 交错存储，1次 DMA/page，但 V 必须 pad 到 256 → 浪费 25% 显存
- **Split kernel**：K/V 分开存储，保留原始维度 → 省 25% 显存，但 2次 DMA/page → **慢 2.5~4.3x**

实测（decode, batch=128, ctx=4096, page_size=64）：

| 配置 | 耗时 (ms) | vs fused_128 |
|------|---:|---:|
| fused_128 | 0.60 | 1.00x |
| split_192_128 (生产) | 2.31 | **3.82x** |

理论下限：如果只看 DMA 字节量，split_256_128 应为 fused_128 的 1.50x。纯结构开销 = 2.55x。

**根因**：`_fetch_bkv` 对 K 和 V 发两次 DMA，共享同一个 semaphore，完全串行。

### 串行 DMA 代码路径

位于 `ragged_paged_attention_split.py` L309-387。

```python
def _fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx, *, wait=False):
    sem = sems.at[0, bkv_sem_idx]           # K 和 V 共用 sems[0]
    k_vmem_ref = bk_x2_ref.at[bkv_sem_idx]  # K 的 VMEM 槽
    v_vmem_ref = bv_x2_ref.at[bkv_sem_idx]  # V 的 VMEM 槽
    ...
    # 每个 page 提交 2 次 DMA（L348-358）
    def loop_body(i, offset):
        _async_copy(k_cache → k_vmem, sem, wait)  # DMA 1: K
        _async_copy(v_cache → v_vmem, sem, wait)  # DMA 2: V（同一 sem，排队等 K 完成）
    lax.fori_loop(0, bkv_p_frm_cache, loop_body, 0)

    # extend token 也是 2 次（L367-378）
    _async_copy(k_hbm → k_vmem, sem, wait)
    _async_copy(v_hbm → v_vmem, sem, wait)
```

TPU DMA 引擎对同一 semaphore 的操作**按提交顺序串行**。`bkv_p` 个 page 共 `2×bkv_p` 次 DMA 命令。Writeback（`_update_kv_cache` L389-444）同理，K/V 共用 `sems[3]`。

对比 fused kernel（`ragged_paged_attention.py` L526-595）：KV 交错存储 `[pages, ps, 2, heads, dim]`，每个 page 只需 1 次 DMA。

## 方案：拼接 K+V 布局

将 K 和 V 沿 head_dim 轴拼接到同一个 buffer，恢复单次 DMA，同时保留非对称维度的显存节省：

```
当前 split：K cache [pages, ps, heads, 256]  +  V cache [pages, ps, heads, 128]  → 2次 DMA
拼接方案：  KV cache [pages, ps, heads, 384]                                      → 1次 DMA
                                        K = [..., :256], V = [..., 256:384]
```

384 = 3×128，满足 TPU 128 字节对齐。显存和 split 一致（768 B/head/token），DMA 和 fused 一致（1 次/page）。

## 分阶段实施

### Phase 1：分离 K/V semaphore（可独立上线）

在 `_fetch_bkv` 中给 K 和 V 用不同 semaphore（当前 6 对 → 7 对），使两次 DMA 可并行。预期改善 30-50%（受限于 HBM 带宽竞争）。

修改文件：`ragged_paged_attention_split.py`（仅 kernel）

### Phase 2：Block size 调优（可独立上线）

当前所有配置 fallback 到 `bkv_p=16, bq=32`。对 page_size={64,128} 做 sweep，将最优值写入 `TUNED_BLOCK_SIZES_SPLIT`。

修改文件：`tuned_block_sizes.py` + 新增调优脚本

**决策门**：Phase 1+2 后如果差距 < 2x fused_128，可推迟 Phase 3。

### Phase 3：拼接布局实现（核心优化）

| 子步骤 | 内容 | 文件 |
|--------|------|------|
| 3a | `ConcatMHATokenToKVPool`：单 buffer 存 K+V | `memory_pool.py` |
| 3b | KV cache update 适配：写入拼接 buffer | `update_kv_cache.py`（可能无需改动） |
| 3c | `FlashAttention._call_concat()`：新 dispatch 路径 | `flashattention_backend.py` |
| 3d | 新 kernel `ragged_paged_attention_concat.py`：基于 fused 结构，offset 切片取 K/V | 新文件 |
| 3e | 累加器优化：acc 从 256 缩到 128（V 的真实维度） | 同 3d |

关键注意：
- `k_head_dim` / `v_head_dim` 必须作为 `functools.partial` 编译期常量传入 kernel，否则 Pallas 会生成 `dynamic_slice` 导致 lowering 失败
- 保留 v2 分支的三处 SWA bugfix（`bkv_idx_start` 相关）
- LSE export + post-kernel attention sink 逻辑复用现有代码

### Phase 4：集成

`model_runner.py` 中根据 `v_head_dim != head_dim` 选择 `ConcatMHATokenToKVPool`，适配 `SWAKVPool`，E2E 测试。

## 风险

| 风险 | 应对 |
|------|------|
| 384 宽度 DMA/VMEM 对齐问题 | 384=3×128，满足对齐；MXU 分别操作 K(256) 和 V(128)，不涉及 384 |
| Pallas offset 切片生成 dynamic_slice | 将 k_head_dim/v_head_dim 作为编译期常量 |
| Zero-copy head expansion 不兼容 | 在 `_align_kv_heads` 中先拼接再 repeat |
| SWA 双池交互 | 早期用 `SWAKVPool` 包裹 `ConcatMHATokenToKVPool` 测试 |
