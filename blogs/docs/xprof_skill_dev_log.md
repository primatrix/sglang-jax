# xprof-analysis Skill 开发日志

> 记录 xprof profiling 分析工具的开发和验证过程

## 动机

之前优化 Fused MoE Pallas kernel (R1-R8) 完全依赖 benchmark wall time。
需要 profiling-driven 分析来定量理解 kernel 内部瓶颈。

## 研究阶段

### 研究 1: accelerator-agents + xprof 仓库

Clone 了两个仓库到 `/root/`，深度研究了：
- **xprof** (OpenXLA): 核心 Python API `xspace_to_tools_data()`，17 个工具
- **accelerator-agents** (Google MaxKernel): Profiling Agent 设计模式

研究文档产出：`blogs/docs/01_xprof_core_api_reference.md` 等 4 个文档

### 关键发现

1. **xprof v2.22.0 已安装**，可直接用 Python API 编程化分析
2. 核心 API: `xprof.convert._pywrap_profiler_plugin.xspace_to_tools_data()`
3. accelerator-agents 使用两种模式：
   - 快速: trace_viewer → SyncWait ratio
   - 深度: xplane.pb → SQLite → SQL 查询

## 开发阶段

### Iteration 1: 测试 xprof API 可用性

直接测试 `xspace_to_tools_data()` 对 `profile_r8_current`：
- ✅ `tool_names`: 返回 12 个可用工具
- ✅ `utilization_viewer`: 返回 472 行数据，44 个指标/core
- ✅ `roofline_model`: 返回 per-op 时间分解

### Iteration 2: 发现 utilization_viewer 的局限

**问题**: 所有 utilization 值接近 0%（MXU=0.05%, Scalar=0.02%）

**原因**: 硬件计数器覆盖整个 profiling 窗口（~205ms），kernel 只运行 ~6ms。
390M cycles 是总窗口，不是 per-kernel。

**验证**: R6 和 R8 的 `achieved` 值几乎完全相同（Scalar=162360 vs 167022），
说明计数器不区分 kernel 内外——它们是 cumulative 的。

**结论**: `utilization_viewer` 对 per-kernel 分析**无效**。
`perf_counters` 也只有一个 kernel="counters_0"，没有 per-kernel 分解。

### Iteration 3: 确认 roofline_model 是核心工具

**roofline_model** 可以看到：
- Pallas kernel 作为 `custom-call` 出现，有确切时间（6606.6us for R8）
- 其他 XLA ops 的时间分解（allgather=1024us, gather_fusion=488us...）
- 每个 op 的 HBM/VMEM 带宽和 bound-by 分类
- DMA stall 百分比

**op_profile** 补充：
- 树形结构展示，含 rawFlops (52B BF16 FLOPS for fused-moe)
- 按 program 分组

**关键局限**: Pallas kernel 内部是黑盒——FLOP Rate=0.0 因为 xprof 不知道 custom-call 内部做了什么。

### Iteration 4: 构建 skill

基于实际可用数据构建 `skills/xprof-analysis/analyze.py`:
- `analyze_roofline()`: 解析 per-op 时间分解
- `analyze_op_profile()`: 解析 op 树 + FLOP 数据
- `print_comparison()`: 对比两个 profile 的 per-op delta
- CLI: `--compare`, `--ops-only`

### Iteration 5: 验证

**单 profile 分析** (R8): ✅
```
Kernel time: 6606.6us (76.7% of total)
Bound by: Compute
Top non-kernel ops: allgather(1024us), gather_fusion(488us)
```

**对比分析** (R6 vs R8): ✅
```
Kernel: 6901.4us → 6606.6us (-294.8us, -4.3%)
allgather: 834.1us → 1024.3us (+190.2us) ← R6→R8 precompute metadata 的副作用
```

**进化追踪** (R1→R8): ✅
```
R1: 31710us → R2: 15910us (-50%) → R4: 7127us (-78%) → R6: 6901us → R8: 6607us (-79%)
```

## 工具能力边界总结

| 能力 | 可行？ | 说明 |
|------|--------|------|
| Kernel 总时间 | ✅ | roofline_model custom-call 的 self time |
| Kernel FLOP 数 | ✅ | op_profile rawFlops/bf16Flops |
| Kernel 内部 MXU/Scalar 分解 | ❌ | 需要 LLO bundle 或专门的 kernel profiling |
| 非 kernel ops 时间分解 | ✅ | roofline 逐 op 列出 |
| 内存带宽 per op | ✅ | roofline HBM/VMEM BW 列 |
| Bottleneck 分类 per op | ✅ | roofline bound_by 列 |
| 对比分析 | ✅ | per-op delta + 总 kernel time delta |
| 硬件利用率 (全局) | ✅ | utilization_viewer（但不 per-kernel） |

## 文件清单

```
skills/xprof-analysis/
├── __init__.py
├── analyze.py      # 核心分析脚本
└── skill.md        # Skill 文档
```
