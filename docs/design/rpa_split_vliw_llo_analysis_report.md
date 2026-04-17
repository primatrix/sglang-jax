# LLO 指令级分析报告：Split vs Fused RPA Kernel

日期：2026-04-10 (v3) | 数据来源：TPU v6e-4 LLO dump (bq=32, bkv_p=16, ps=64)

## 目标

定位 split kernel 多出开销的来源，background 见同目录下的 `rpa_concat_kv_plan.md`

## 编译层级说明

Pallas kernel 的编译经过以下阶段，本报告分析的是最终阶段（VLIW bundle）：

```
Pallas Python
  │  Pallas tracing
  ▼
Mosaic MLIR
  │  Mosaic 编译器（不走 XLA 标准路径）
  ▼
LLO (Low-Level Optimizer)          ← post-finalize-llo.txt（每条指令独占一行）
  │  VLIW 打包：将可并行的指令打包到同一 bundle
  ▼
VLIW Bundles                       ← final_bundles.txt（本报告的主要分析对象）
  │                                   每行一个 bundle，用 ;; 分隔同 cycle 并行执行的多条指令
  ▼
Machine Code → TPU 执行
```

**`final_bundles.txt` vs `post-finalize-llo.txt`**：两者包含相同的 LLO 指令，区别在于 `final_bundles.txt` 经过了 VLIW 打包——多条独立指令被合并到同一个 bundle（同一 cycle 并行执行）。因此：
- `grep -o 'vrot.slane' final_bundles.txt` = 指令总数（与 `post-finalize-llo.txt` 中的 `grep -c` 一致）
- `grep -c 'vrot.slane' final_bundles.txt` = 包含该指令的 **bundle 行数**（≤ 指令总数，因为一个 bundle 可能包含多条同类指令）

bundle 是 TPU 的最小执行单位（1 bundle = 1 cycle），因此 **bundle 数直接决定执行时间**，是本报告的核心分析指标。

**术语约定**：本报告中，"指令"（如 `vrot.slane`、`vmatpush`）指 LLO 级别的单条硬件操作；"bundle" 指 VLIW 打包后的指令组（1 bundle 内的多条指令在同一 cycle 并行执行）。指令统计用 `grep -o`（指令总数），bundle 统计用 `grep -c`（包含该指令的 bundle 行数）。

## 分析方法

对三个 kernel 的 `final_bundles.txt`（LLO 经 VLIW 打包后的最终硬件指令）做：
1. 指令分类统计（`grep -o` 统计指令出现次数，`grep -c` 统计包含该指令的 bundle 行数）
2. 编译器生成的 per-bundle 功能单元利用率分析（`pre-delay_hlo-static-per-bundle-utilization.txt` 和 `final_hlo-static-per-bundle-utilization.txt`）
3. 编译 pipeline 各阶段 bundle 数对比（pre-RA → post-RA → final，每阶段对应独立的 `schedule-analysis_*.txt`）
4. 源码级 cross-check（Pallas 代码 → LLO 指令追溯）

## 文件引用

本报告所有数据来自 `profilings/` 目录下的以下文件。每个 kernel 的 LLO dump 以编译 ID 为前缀：

| 缩写 | Kernel | 编译 ID 前缀 |
|------|--------|------------|
| **split** | split_128_128 | `1775788512601440038-RPA-bq_32-bkvp_16-p_64-split.1` |
| **fused_128** | fused_128 | `1775788975109488757-RPA-bq_32-bkvp_16-p_64-.1` |
| **fused_256** | fused_256 | `1775789412421620008-RPA-bq_32-bkvp_16-p_64-.1` |

关键文件后缀及对应编译阶段：

| 后缀 | 内容 | 说明 |
|------|------|------|
| `-47-schedule-analysis_packed-bundles-pre-ra.txt` | pre-RA bundle 数 | 寄存器分配前 |
| `-64-schedule-analysis_packed-bundles-no-spills-fills.txt` | no-spills-fills bundle 数 | 去除 spill/fill 后 |
| `-70-schedule-analysis_packed-bundles-post-ra.txt` | post-RA bundle 数 | 寄存器分配后 |
| `-79-schedule-analysis_final_bundles.txt` | final bundle 数 | VLIW 最终优化后 |
| `-80-final_bundles.txt` | VLIW 最终指令文本 | 指令级分析的主要数据源 |
| `-76-pre-delay_hlo-static-per-bundle-utilization.txt` | 功能单元利用率（pre-delay） | 逻辑运算量的准确来源 |
| `-78-final_hlo-static-per-bundle-utilization.txt` | 功能单元利用率（final） | 最终 bundle 数对应的利用率 |

---

## 1. 根因定位：`unpack_heads` 的 cross-sublane gather 模式

### 1.1 编译 pipeline 各阶段 bundle 数

数据来源：各阶段对应的 `schedule-analysis_*.txt` 文件首行 `total scheduled bundles`。

| 编译阶段 | 文件后缀 | split | fused_128 | fused_256 | Split/F128 |
|---------|---------|-------|-----------|-----------|------------|
| **pre-RA**（寄存器分配前） | `-47-*pre-ra` | 34,468 | 10,245 | 12,010 | **3.4x** |
| no-ra-deps | `-58-*no-ra-deps` | 34,572 | 10,378 | 12,287 | 3.3x |
| no-spills-fills | `-64-*no-spills-fills` | 34,821 | 10,494 | 13,000 | 3.3x |
| post-RA | `-70-*post-ra` | 34,880 | 10,518 | 13,021 | 3.3x |
| **final**（最终优化） | `-79-*final_bundles` | **31,606** | **6,590** | **8,688** | **4.8x** |

**关键发现**：pre-RA（无任何寄存器溢出）阶段，split 已经是 fused 的 **3.4x**。post-RA 仅增加 412 bundles（34,468 → 34,880），说明 **VREG spilling 不增加 bundle 数**——spill/fill 指令被 VLIW 打包到已有 bundle 的空闲槽位。

4.8x 的 final 差距由两层因素叠加：
- **3.4x（源码级指令膨胀）**：`unpack_heads` 的 3D reshape + 中间维 slice 被 Mosaic 编译器展开为大量 gather 指令（vrot.slane + vsel + vld.sshfl），这在 pre-RA 阶段就已经存在，与编译器优化无关，是 Pallas 源码的数据访问模式直接决定的
- **1.4x（VLIW 打包降级）**：gather 指令占满了 VALU/VLOAD 槽位，导致 fused 能做到的双 vmatpush 打包在 split 中无法实现（post-RA → final 压缩 37% vs split 仅 9%，详见 §1.3）

### 1.2 源码级指令膨胀的来源：gather 指令

数据来源：`*-80-final_bundles.txt`。

> **统计方法说明**：VLIW bundle 中每行可打包多条并行指令（以 `;;` 分隔）。因此同一条助记符的 `grep -o`（出现总次数）和 `grep -c`（包含该指令的 bundle 行数）结果不同。下表使用 `grep -o` 统计实际指令数。

| 指令类别 | split 指令数 | fused_128 指令数 | 差异 | 说明 |
|---------|------------|----------------|------|------|
| **vld.sshfl** | **12,288** | **0** | **+12,288** | sublane-shuffled load（**split 独有**） |
| **vrot.slane** | **23,520** | **480** | **+23,040** | sublane 旋转（**49.0x**） |
| **vsel** | **10,964** | **212** | **+10,752** | 级联条件选择（**51.7x**） |
| **Gather 合计** | **46,772** | **692** | **+46,080** | |

> 补充：`grep -c` 统计表明 vrot.slane 占 9,598 个 bundle（每 bundle 平均 ~2.5 条）、vsel 占 6,872 个 bundle（每 bundle 平均 ~1.6 条）、vld.sshfl 占 12,288 个 bundle（每 bundle 1 条，因为 VLOAD 槽位每 bundle 仅 1 条 sshfl 可执行）。

**22,097 个 bundle（占 split 总 31,606 bundle 的 70%）包含至少一条 gather 模式指令**（`grep -cE 'vld\.sshfl|vrot\.slane|vsel'`）。

这些指令来自 `strided_load_kv_separate` → `unpack_heads` 函数（`ragged_paged_attention_split.py`）：

```python
# Split 路径：3D reshape + 中间维度 slice → 触发 cross-sublane gather
def unpack_heads(ref, head_start_idx, num_heads_to_load, head_dim_val):
    ref_flat = ref.reshape(bkv_sz, -1, head_dim_val)       # 3D reshape
    heads = ref_flat[:, head_start_idx : head_start_idx + num_heads_to_load, :]  # 中间维 slice
    return [heads[:, i, :] for i in range(num_heads_to_load)]
```

#### 为什么中间维 slice 触发 cross-sublane gather

TPU VMEM 中数据按 `(sublanes=8, lanes=128)` tile 布局。对 3D array `[bkv_sz, num_kv_heads, head_dim]` 做中间维度 slice `[:, idx, :]` 时，目标 head 的数据元素散布在 8 个不同 sublane 中（每个 sublane 存放了所有 head 的混合数据）。编译器必须从每个 sublane 提取正确的 head 数据并重组到目标寄存器的对应 sublane，因此生成 `vrot.slane`（旋转至目标 sublane 位置）+ `vsel`（用 mask 选择）级联链。

对比之下，2D array `[bkv_sz * num_kv_heads, head_dim]` 上的行级 stride `ref[idx::step]` 是整行操作——同一行的数据天然在同一个 sublane 内，无需跨 sublane 搬运。Mosaic 用 `vshrl`（shift right logical）高效实现。

#### VLIW 实证

以下为 split `final_bundles.txt` 中的实际 VLIW bundle 片段（bundle `0x14bd`–`0x14c5`，文件第 5319–5327 行），展示 vrot.slane + vsel 的流水线级联模式：

```asm
;; --- 阶段 1-2：批量 vrot.slane，旋转量从 7 递减到 1 ---
0x14bd: { vrot.slane %v..., 7  ;;  vrot.slane %v..., 6  ;;  vrot.slane %v..., 5  ;;  vrot.slane %v..., 4 }
0x14be: { vrot.slane %v..., 3  ;;  vrot.slane %v..., 2  ;;  vrot.slane %v..., 1  ;;  vrot.slane %v..., 7 }
;; --- 阶段 3+：vrot 与 vsel 交替流水线执行，逐级用递增 mask (vm11→vm15→vm0→vm1) 选择正确 sublane ---
0x14bf: { vsel vm11, %rotated_7, %original       ;;  vrot.slane %v..., 6  ;;  vrot.slane %v..., 5  ;;  vrot.slane %v..., 4 }
0x14c0: { vsel vm12, %rotated_6, %prev_selected  ;;  vsel vm11, %rotated_7b, %original_b   ;;  vrot.slane %v..., 3  ;;  vrot.slane %v..., 2 }
0x14c1: { vsel vm13, %rotated_5, %prev_selected  ;;  vsel vm12, %rotated_6b, %prev_sel_b   ;;  vrot.slane %v..., 1  ;;  ... }
0x14c2: { vsel vm14, %rotated_4, %prev_selected  ;;  vsel vm13, %rotated_5b, %prev_sel_b   ;;  ...  ;;  ... }
0x14c3: { vsel vm15, %rotated_3, %prev_selected  ;;  vsel vm14, %rotated_4b, %prev_sel_b   ;;  ...  ;;  ... }
0x14c4: { vsel vm0,  %rotated_2, %prev_selected  ;;  vsel vm15, %rotated_3b, %prev_sel_b   ;;  ...  ;;  ... }
0x14c5: { vsel vm1,  %rotated_1, %prev_selected  ;;  vsel vm0,  %rotated_2b, %prev_sel_b   ;;  ...  ;;  ... }
;; 每个 head 需要完整的 7 级 vrot + 7 级 vsel 才能完成 cross-sublane 数据重组
;; 对 N 个 head 重复 → 指令数 ∝ N × 7 × (vrot + vsel)
```

> 以上为简化表示。`final_bundles.txt` 中的实际 VLIW bundle 使用完整寄存器名（如 `%v26196_v16 = vrot.slane %v24986_v0, 7`），编译器将 2 个 head 的 vrot+vsel 链在 VLIW 中交叉流水线化以填满 VALU 槽位。

**对比 fused 路径**（`ragged_paged_attention.py:731`）：

```python
# Fused 路径：2D reshape + stride 行选取 → 高效 shift 路径
def strided_load(ref, start, step):
    ref = ref.reshape(r * folds, 128)            # 对齐到 128 lanes
    vec = jnp.concat([ref[start + i :: step]     # 行级 stride（VMEM 对齐）
                       for i in range(folds)], axis=1)
    return vec
```

`ref[start::step]` 是均匀间隔的行选取，Mosaic 编译器用 `vshrl`（shift right logical）高效实现（fused_128 有 791 条 `vshrl`（`grep -o`）vs split 仅 23 条），**无需跨 sublane 数据重排**。以下为 fused_128 `final_bundles.txt` 中的典型 vshrl 指令（bundle `0x46a`，文件第 1140 行）：

```asm
0x46a: { vst [vmem:spill] %v...  ;;  vld [vmem:+0x30 ss:2]  ;;  vld [vmem:+0x40 ss:2]  ;;  vshrl.u32 %v..., 7 }
;; vshrl 与 vld/vst 共享 bundle，充分利用 VLIW 槽位
```

### 1.3 VLIW 打包降级

数据来源：`*-80-final_bundles.txt`（`grep -o` 统计指令总数，`grep -c` 统计包含该指令的 bundle 行数）。

Gather 指令不仅自身占用 bundle，还阻碍其他指令的并行打包：

| 指标 | split | fused_128 | 统计方法 |
|------|-------|-----------|---------|
| vmatpush 指令总数 | 384 | 384 | `grep -o 'vmatpush'` |
| vmatpush 占用的 bundle 数 | 384 | **192**（2 条/bundle） | `grep -c 'vmatpush'` |
| post-RA → final 压缩比 | 9% | **37%** | `(post_ra - final) / post_ra` |

Split 中每个 vmatpush 独占一个 bundle（384 指令 / 384 bundle = 1.0 条/bundle），而 fused 将 mxu0 + mxu1 两条 vmatpush 打包到同一 bundle（384 指令 / 192 bundle = 2.0 条/bundle）。

以下为 fused_128 `final_bundles.txt` 中双 vmatpush 打包的实例（bundle `0x4f5`，文件第 1279 行）：

```asm
0x4f5: { vmatpush3.bf16.xpose.msra.mxu0 %v...  ;;  vmatpush3.bf16.xpose.msra.mxu1 %v...  ;;
         vpack.c.b16 ...  ;;  vpack.c.b16 ...  ;;  vpack.c.b16 ...  ;;
         vld [spill]  ;;  vld [spill]  ;;  vld [spill] }
;; 1 个 bundle = 2× MXU + 3× VALU(vpack) + 3× VLOAD(fill) — 充分利用所有 VLIW 槽位
```

Split 无法做到这一点，因为 gather 指令（vrot.slane、vsel、vld.sshfl）占满了同 bundle 的 VALU 和 VLOAD 槽位，导致 VLIW 打包从 post-RA 到 final 仅压缩 9%（34,880 → 31,606），而 fused 压缩了 37%（10,518 → 6,590）。

---

## 2. 核心计算完全相同

数据来源：`*-76-pre-delay_hlo-static-per-bundle-utilization.txt` 各列逐行求和（`awk 'NR>3 && NF==10 { for(i=1;i<=10;i++) sum[i]+=$i } END { ... }'`）。使用 pre-delay（而非 final）利用率是因为 pre-delay 反映逻辑运算量（不受 delay-slot 优化的 bundle 重排影响）。

列定义（文件前 3 行为 header）：`MXU, XLU, VALU, VPOP, EUP, VLOAD, VLOAD:FILL, VSTORE, VSTORE:SPILL, SALU`

| 功能单元 | 说明 | split | fused_128 | 差异 |
|---------|------|-------|-----------|------|
| **MXU** | 矩阵乘法 | **1,120** | **1,120** | **0** |
| **XLU** | 转置 | **1,040** | **1,040** | **0** |
| **EUP** | 超越函数 | **402** | **402** | **0** |
| **VPOP** | 向量 pop | **1,222** | **1,222** | **0** |
| VALU | 向量 ALU | 50,632 | 5,305 | +45,327 (**9.5x**) |
| VLOAD | 向量 load | 27,368 | 4,557 | +22,811 (**6.0x**) |
| VLOAD:FILL | spill 回填 | 12,646 | 2,000 | +10,646 (6.3x) |
| VSTORE | 向量 store | 18,952 | 3,866 | +15,086 (**4.9x**) |
| VSTORE:SPILL | spill 溢出 | 12,658 | 1,684 | +10,974 (7.5x) |
| SALU | 标量 ALU | 3,218 | 2,705 | +513 (1.2x) |

**两个 kernel 做的核心数学运算完全一致**。MXU 1,120、XLU 1,040、EUP 402、VPOP 1,222 ——一个不差。差异全在 VALU（9.5x）和 VLOAD（6.0x），即 gather 模式产生的数据搬运和重排上。

DMA 指令统计（`grep -o 'dma\.<type>' | wc -l`，数据来源：`*-80-final_bundles.txt`）：

| DMA 类型 | split | fused_128 | 倍数 |
|----------|-------|-----------|------|
| dma.hbm_to_vmem | 8 | 4 | **2.0x**（split 分别读 K/V） |
| dma.vmem_to_hbm | 9 | 6 | 1.5x |
| dma.vmem_to_smem | 9 | 9 | 1.0x |
| dma.done.wait | 36 | 24 | 1.5x |
| **DMA 总计** | **62** | **43** | **1.4x** |

> DMA hbm_to_vmem 翻倍反映了 split kernel 需要分别读取 K cache 和 V cache（各 4 条 DMA），而 fused kernel 读取 interleaved KV cache 只需 4 条。但 DMA 总数（62 vs 43）仅贡献 final bundle 差距的 <2%——gather 指令才是主导因素。

> 对比验证：final 阶段利用率（`*-78-final_hlo-*`）中 MXU 分别为 986 / 764。这是 delay-slot 优化将 bundle 重新打包后，部分 MXU 指令的槽位发生变化（例如一条 vmatpush 从双槽变为单槽），不代表计算量差异。pre-delay 的 1,120 / 1,120 是比较逻辑运算量的正确基准。

---

## 3. VREG 溢出是 gather 的连锁反应

SPILL/FILL 统计来源：`*-76-pre-delay_hlo-static-per-bundle-utilization.txt` 中 VSTORE:SPILL（第 9 列）和 VLOAD:FILL（第 7 列）的逐行求和。`_spill` 引用来源：`*-80-final_bundles.txt` 中包含 `_spill` 后缀的变量名（如 `#allocation143_spill`）。

| 指标 | split | fused_128 | Split/F128 | 统计方法 |
|------|-------|-----------|------------|---------|
| VSTORE:SPILL 总量 | 12,658 | 1,684 | 7.5x | pre-delay 利用率第 9 列求和 |
| VLOAD:FILL 总量 | 12,646 | 2,000 | 6.3x | pre-delay 利用率第 7 列求和 |
| `_spill` 引用（bundle 行数） | 20,143 | 1,917 | 10.5x | `grep -c '_spill'` final_bundles |
| `_spill` 引用（出现总次数） | 25,494 | 3,789 | 6.7x | `grep -o '_spill'` final_bundles |

> bundle 行数（20,143）> 出现总次数（25,494）的倒数关系说明很多 bundle 中有 2+ 条 spill/fill 相关操作——这些都被打包进了同一 bundle 的空闲槽位。

VREG 溢出是 gather 模式的**连锁反应**，不是独立瓶颈：
1. Gather 指令（46,772 条）在 VALU 和 VLOAD 单元上长期占用
2. 编译器无法释放用于 gather 中间结果的 VREG → 其他变量被迫溢出
3. 但 spill/fill 被 VLIW 打包到 gather 指令的 bundle 中（post-RA 仅增加 412 bundles：34,468 → 34,880），**不额外增加执行时间**

---

## 4. 瓶颈完整分解

| 因素 | 对 4.8x 差距的贡献 | 额外指令数 | 可优化？ |
|------|-------------------|-----------|----|
| **`unpack_heads` gather 模式** | **~70%（3.4x pre-RA 差距）** | 46,080 | **是 — 改用 `strided_load` 模式** |
| **VLIW 打包降级** | **~30%（3.4x → 4.8x）** | N/A | 是 — 消除 gather 后自然恢复 |
| VREG 溢出 | 不直接增加 bundle | SPILL+FILL 共 25,304 | 随 gather 消除自动解决 |
| Scalar ALU | <5% | 513 | 不值得优化 |
| DMA | <2% | 19 | 不值得优化 |
| MXU / 核心计算 | **0%** | 0 | 已最优 |

---

## 5. 结论：改写 `strided_load_kv_separate`（低风险高收益）

将 `unpack_heads` 的 3D reshape + 中间维度 slice 改为 fused 风格的 2D reshape + stride 行选取：

```python
# 当前（慢）：
ref_flat = ref.reshape(bkv_sz, -1, head_dim_val)
heads = ref_flat[:, head_start_idx : head_start_idx + num_heads_to_load, :]

# 目标（快）：类似 fused 的 strided_load
ref_2d = ref.reshape(bkv_sz * num_kv_heads, head_dim_val)
head = ref_2d[head_start_idx :: num_kv_heads]   # stride 行选取
```

这需要确保 VMEM 中 K/V cache buffer 的物理布局支持这种行级 stride 访问。如果 `bk_x2_ref` 的 layout 是 `[bkv_sz, num_kv_heads_packed, head_dim]`，那么 reshape 为 `[bkv_sz * num_kv_heads_packed, head_dim]` 后做 stride 访问，应该与 fused 的 `strided_load` 生成相同的 LLO 路径。

**效果**：消除 ~46,000 条 gather 指令 → bundle 数从 31,606 降到接近 ~6,500-8,000 → 等 flops 下性能接近 fused。

---
