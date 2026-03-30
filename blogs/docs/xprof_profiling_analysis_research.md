# JAX/TPU Profiling 编程化分析研究报告（索引）

> **研究目标**: 研究 `accelerator-agents` 和 `xprof` 两个仓库，理解如何编程化分析 Pallas kernel profiling 数据，
> 超越纯 benchmark 的优化方式，实现 profiling-driven 的 kernel 性能分析。
>
> **日期**: 2026-03-26
> **仓库来源**:
> - https://github.com/AI-Hypercomputer/accelerator-agents (`/root/accelerator-agents/`)
> - https://github.com/openxla/xprof (`/root/xprof/`)

---

## 子文档索引

| # | 文档 | 内容 |
|---|------|------|
| 01 | [xprof Core API 参考手册](01_xprof_core_api_reference.md) | 完整的 xprof Python API 参考：17 个工具的用法、数据格式、TPU v7 硬件参数、Protobuf 结构 |
| 02 | [accelerator-agents 设计分析](02_accelerator_agents_design_analysis.md) | MaxKernel 的 Profiling Agent 架构、优化流程设计模式、对我们工作的借鉴 |
| 03 | [Profiling 分析方案](03_profiling_analysis_approach.md) | 两种分析场景（XLA Trace vs Kernel 内部）、完整分析脚本、瓶颈诊断决策树 |

---

## 目录（本文概述）

1. [概述](#1-概述)
2. [xprof 编程化 API](#2-xprof-编程化-api)
3. [accelerator-agents 的 Profiling 分析模式](#3-accelerator-agents-的-profiling-分析模式)
4. [可提取的关键指标](#4-可提取的关键指标)
5. [TPU v7 硬件常量与计数器](#5-tpu-v7-硬件常量与计数器)
6. [数据格式与转换流水线](#6-数据格式与转换流水线)
7. [Protobuf 数据结构](#7-protobuf-数据结构)
8. [实际应用：Pallas Kernel Profiling 分析方案](#8-实际应用pallas-kernel-profiling-分析方案)
9. [关键文件索引](#9-关键文件索引)

---

## 1. 概述

### 1.1 xprof 是什么

xprof（前身为 TensorBoard Profiler Plugin）是 OpenXLA 项目的性能分析工具套件。
核心能力：将 `.xplane.pb` profiling 数据转换为各类分析视图（roofline、op profile、utilization 等）。

**关键发现**：xprof v2.22.0 已安装在本机，提供了完整的 **Python 编程化 API**，
可以无需启动 Web 服务器，直接从 `.xplane.pb` 文件提取所有性能指标。

### 1.2 accelerator-agents 是什么

Google 的 AI 驱动加速器开发 Agent 集合，包含两个主要组件：
- **MaxCode**: PyTorch → JAX 代码迁移 Agent
- **MaxKernel**: TPU Pallas kernel 生成、优化、**profiling 分析** Agent

**关键发现**：MaxKernel 的 profiling 子系统展示了一种实用模式——
将 xplane.pb 加载到 SQLite 中进行灵活 SQL 查询分析，并结合 xprof Python API 提取高层指标。

---

## 2. xprof 编程化 API

### 2.1 核心函数

本机已安装路径: `/usr/local/lib/python3.12/site-packages/xprof/`

```python
from xprof.convert._pywrap_profiler_plugin import xspace_to_tools_data

# 主 API：从 .xplane.pb 提取任意工具数据
raw_data, success = xspace_to_tools_data(
    ['/path/to/file.xplane.pb'],  # xplane.pb 文件路径列表
    'tool_name',                    # 工具名称字符串
    {}                              # options 字典（工具特定）
)
# 返回: (bytes, bool) — raw_data 通常是 JSON，用 json.loads(raw_data) 解析
```

更高层的 Python wrapper：

```python
from xprof.convert.raw_to_tool_data import xspace_to_tool_data, xspace_to_tool_names

# 列出可用工具
tools = xspace_to_tool_names(['/path/to/file.xplane.pb'])

# 获取工具数据（自动处理 content_type）
data, content_type = xspace_to_tool_data(['/path/to/file.xplane.pb'], 'roofline_model', {})
```

### 2.2 可用工具列表

| 工具名 | 返回格式 | 用途 |
|--------|---------|------|
| `tool_names` | 逗号分隔字符串 | 列出可用工具 |
| `overview_page` | JSON (DataTable) | 高层摘要：MXU 利用率、设备空闲率、step 时间分解 |
| `op_profile` | JSON (树结构) | 按 program/category 分层的 op 时间分解 |
| `hlo_stats` | JSON (DataTable) | 每个 HLO op 的时间、FLOP rate、内存带宽、瓶颈分类 |
| `roofline_model` | JSON (DataTable) | Roofline 分析：每个 op 的实测/峰值 FLOP rate 与内存带宽 |
| `framework_op_stats` | JSON (DataTable) | 框架级 op 统计（pallas_call、reshard 等） |
| `utilization_viewer` | JSON | 硬件单元利用率：MXU、Scalar、Vector、XLU、HBM、ICI |
| `perf_counters` | JSON | 原始硬件性能计数器（数据量大，约 87MB） |
| `trace_viewer@` | Protobuf (Perfetto) | 流式时间线数据 |
| `memory_profile` | JSON | 内存分配/释放时间线 |
| `memory_viewer` | JSON/HTML | 内存布局 |
| `graph_viewer` | Various | HLO 图可视化 |
| `kernel_stats` | JSON | GPU kernel 统计（TPU 较少用） |
| `smart_suggestion` | JSON | AI 驱动的优化建议 |
| `inference_profile` | JSON | 推理延迟分析 |

### 2.3 工具特定 Options

```python
# op_profile: 按 program 分组
xspace_to_tools_data([path], 'op_profile', {'group_by': 'program'})

# trace_viewer@: 指定时间范围和分辨率
xspace_to_tools_data([path], 'trace_viewer@', {
    'resolution': 8000,
    'start_time_ms': 100,
    'end_time_ms': 200,
    'full_dma': True
})

# memory_viewer: 指定模块和内存空间
xspace_to_tools_data([path], 'memory_viewer', {
    'module_name': '<module>',
    'memory_space': '0',
    'view_memory_allocation_timeline': False
})
```

---

## 3. accelerator-agents 的 Profiling 分析模式

### 3.1 架构概览

MaxKernel 的 profiling 系统分两层：

```
Agent (ADK)  ──HTTP──>  Eval Server (:1245)  ──HTTP──>  TPU Server (:5463)
     │                                                       │
     │                                                  执行 profiling 脚本
     │                                                  查找 .xplane.pb
     │                                                  调用 analyze_trace()
     │                                                       │
     └─── 接收 JSON: {ratio, xplane_path} ──────────────────┘
```

### 3.2 xprof trace_viewer 分析模式

**文件**: `/root/accelerator-agents/MaxKernel/tpu_kernel_gen/agents/kernel_gen_agent/analyze_profile.py`

```python
from xprof.convert import raw_to_tool_data

def analyze_trace(path):
    tool_data_result, _ = raw_to_tool_data.xspace_to_tool_data(
        [path], "trace_viewer", {}
    )
    trace_data = json.loads(tool_data_result)
    events = trace_data.get("traceEvents", [])
    # 过滤 TPU:0 事件，找到最后一个 jit_computation
    # 累加 SyncWait 时长，计算 ratio = SyncWait / total_computation
```

核心指标：**DMA/内存传输占比**（`SyncWait_time / total_computation_time`）

### 3.3 xplane.pb → SQLite 分析模式（最灵活）

**文件**: `/root/accelerator-agents/MaxKernel/tpu_kernel_gen/agents/hitl_kernel_gen_agent/subagents/profiling/offline_tools.py`

这是最强大的分析模式——将 xplane.pb 直接解析为 SQLite 数据库，允许任意 SQL 查询：

```python
import sqlite3
from tensorflow.tsl.profiler.protobuf import xplane_pb2

def load_xplane_to_sqlite(xplane_path):
    """将 xplane.pb 加载到 SQLite 内存数据库"""
    xspace = xplane_pb2.XSpace()
    with open(xplane_path, 'rb') as f:
        xspace.ParseFromString(f.read())

    conn = sqlite3.connect(':memory:')
    conn.execute('''CREATE TABLE planes (id INTEGER, name TEXT)''')
    conn.execute('''CREATE TABLE lines (id INTEGER, plane_id INTEGER,
                     display_id INTEGER, name TEXT, timestamp_ns INTEGER)''')
    conn.execute('''CREATE TABLE events (plane_id INTEGER, line_id INTEGER,
                     name TEXT, offset_ps INTEGER, duration_ps INTEGER,
                     start_ps INTEGER, end_ps INTEGER)''')

    for plane in xspace.planes:
        conn.execute('INSERT INTO planes VALUES (?, ?)', (plane.id, plane.name))
        for line in plane.lines:
            conn.execute('INSERT INTO lines VALUES (?, ?, ?, ?, ?)',
                        (line.id, plane.id, line.display_id, line.name,
                         line.timestamp_ns))
            for event in line.events:
                start_ps = line.timestamp_ns * 1000 + event.offset_ps
                end_ps = start_ps + event.duration_ps
                # 解析 event metadata 获取 name
                conn.execute('INSERT INTO events VALUES (?, ?, ?, ?, ?, ?, ?)',
                            (plane.id, line.id, event_name, event.offset_ps,
                             event.duration_ps, start_ps, end_ps))
    return conn
```

**可执行的 SQL 查询示例**：

```sql
-- Top 10 最耗时操作
SELECT name, SUM(duration_ps) as total_ps, COUNT(*) as count
FROM events
GROUP BY name
ORDER BY total_ps DESC
LIMIT 10;

-- 计算 MXU 占比 vs Scalar 占比
SELECT
    SUM(CASE WHEN name LIKE '%matmul%' OR name LIKE '%dot%' THEN duration_ps ELSE 0 END) as mxu_ps,
    SUM(CASE WHEN name LIKE '%scalar%' THEN duration_ps ELSE 0 END) as scalar_ps,
    SUM(duration_ps) as total_ps
FROM events;

-- DMA 事件分析
SELECT name, SUM(duration_ps) as total_ps, COUNT(*) as count
FROM events
WHERE name LIKE '%DMA%' OR name LIKE '%copy%' OR name LIKE '%transfer%'
GROUP BY name
ORDER BY total_ps DESC;
```

### 3.4 Overview Metrics 提取

```python
def get_overview_page_metrics(xplane_path):
    """提取高层概览指标"""
    raw, ok = xspace_to_tools_data([xplane_path], 'overview_page')
    data = json.loads(raw)

    # data[0]['p'] 中的关键属性：
    metrics = data[0].get('p', {})
    return {
        'mxu_utilization_percent': metrics.get('mxu_utilization_percent'),
        'device_duty_cycle_percent': metrics.get('device_duty_cycle_percent'),
        'device_idle_time_percent': metrics.get('device_idle_time_percent'),
        'flop_rate_utilization': metrics.get('flop_rate_utilization_relative_to_roofline'),
        'memory_bw_utilization': metrics.get('memory_bw_utilization_relative_to_hw_limit'),
        'hbm_utilization_percent': metrics.get('hbm_utilization_percent'),
    }
```

### 3.5 Profiling 脚本生成模板

accelerator-agents 使用的 profiling 脚本模板：

```python
import jax
import jax.profiler

options = jax.profiler.ProfileOptions()
options.python_tracer_level = 0
options.host_tracer_level = 2
options.advanced_configuration = {"tpu_trace_mode": "TRACE_COMPUTE_AND_SYNC"}

jax.profiler.start_trace('jax_trace', profiler_options=options)
for i in range(3):
    C = jax.block_until_ready(computation(A, B))
jax.profiler.stop_trace()
```

### 3.6 分析 Agent 的 Prompt 策略

MaxKernel 的 profiling 分析 Agent 被指示：
1. 观察 `DMAs_and_memory_transfers_ratio` 和 `compute_ratio`
2. 使用 `load_xplane_and_query` + SQL 查找 top ops（按 `sum(duration_ps)`）
3. 使用 `get_overview_page_metrics` 获取高层指标
4. 使用 `create_chart_from_xplane` 生成可视化图表
5. 提供可操作的优化建议

---

## 4. 可提取的关键指标

### 4.1 MXU 利用率

**方法 1: overview_page（聚合值）**

```python
raw, ok = xspace_to_tools_data([xplane], 'overview_page')
data = json.loads(raw)
mxu_pct = data[0]['p']['mxu_utilization_percent']
```

**方法 2: utilization_viewer（硬件计数器级别）**

```python
raw, ok = xspace_to_tools_data([xplane], 'utilization_viewer')
data = json.loads(raw)
# DataTable 格式，每个 core 的行包括:
#   'Avg MXU Busy': achieved_cycles / peak_cycles
#   'MXU0', 'MXU1': 单个 MXU 的 busy cycles
#   'MXU BF16', 'MXU E4M3 + E5M2': 按数据类型分
#   '2 MXU Busy', '1 MXU Busy', 'No MXU Busy': 分布
```

**方法 3: roofline_model（每个 op 级别）**

```python
raw, ok = xspace_to_tools_data([xplane], 'roofline_model')
data = json.loads(raw)
# data[0]['p'] 包含峰值参数:
#   peak_flop_rate: 1028750 GFLOP/s (~1029 TFLOP/s per chip)
#   peak_hbm_bw: 3433 GiB/s
# data[0]['rows'] 每行是一个 op，列包括:
#   measured_flop_rate, model_flop_rate
#   HBM BW, VMEM Read BW, VMEM Write BW
#   Operational Intensity (FLOP/Byte)
#   Bound by (compute/HBM/VMEM)
#   Roofline efficiency
```

### 4.2 Scalar ALU vs Vector ALU vs MXU 时间

**通过 utilization_viewer 获取**：

| 指标名 | 含义 | 计算方式 |
|--------|------|---------|
| `Scalar Unit` | 标量指令利用率 | `(SCALAR_ALU_0 + _1) / (cycles * 2)` |
| `Vector ALUs` | 向量 ALU 利用率 | `sum(VECTOR_ALU_0..3) / (cycles * 4)` |
| `Avg MXU Busy` | MXU 平均忙碌率 | `(0.5 * MXU_BUSY_1 + MXU_BUSY_2) / cycles` |
| `Avg XLU Busy` | XLU 利用率 | 类似 MXU |
| `VMEM Loads` | VMEM 读指令率 | `(VLD_0 + _1) / (cycles * 2)` |
| `VMEM Stores` | VMEM 写指令率 | `VST / cycles` |

### 4.3 内存带宽

**HBM 带宽（utilization_viewer）**：
- `HBM Rd+Wr - core N`: 实测字节数 vs 峰值带宽字节数
- `HBM Read Ratio`, `HBM Write Ratio`: 读写比

**每 op 内存带宽（roofline_model）**：
- `HBM BW (GiB/s)`, `VMEM Read BW (GiB/s)`, `VMEM Write BW (GiB/s)`
- `Operational Intensity (FLOP/Byte)`: 运算强度

### 4.4 通信时间 (ICI)

**utilization_viewer**：
- `ICI (Read)`: 传输字节数 vs 峰值（~295 GB/s per direction on v7）
- `ICI (Write)`: 同上

### 4.5 Kernel 执行时间分解

**roofline_model（最详细）**：

```python
raw, ok = xspace_to_tools_data([xplane], 'roofline_model')
data = json.loads(raw)
cols = [c['label'] for c in data[0]['cols']]
for row in data[0]['rows']:
    vals = {cols[j]: row['c'][j].get('v') for j in range(len(cols))}
    # vals 包含:
    #   'Operation': op 名称
    #   'Total Time (us)': 总时间
    #   'Avg. time (us)': 平均时间
    #   'Total self time (us)': 自身时间
    #   'Measured FLOP Rate (GFLOP/s/core)': 实测 FLOP rate
    #   'HBM BW (GiB/s)': HBM 带宽
    #   'Bound by': 'Compute' / 'HBM' / 'VMEM Read' / 'VMEM Write'
    #   'Roofline Efficiency (%)': Roofline 效率
```

**op_profile（树结构，按 program 分层）**：

```python
raw, ok = xspace_to_tools_data([xplane], 'op_profile')
data = json.loads(raw)
# data['byProgram']['children'] = program 列表
# 每个 program 的 children 是 op 列表
# 每个 op 有 'metrics': {
#   'rawTime': <picoseconds>,
#   'rawFlops': <count>,
#   'bf16Flops': <count>,
#   'rawBytesAccessedArray': [HBM_bytes, CMEM_rd, CMEM_wr, VMEM_rd, VMEM_wr],
#   'normalizedTimePs': <picoseconds>,
#   'occurrences': <count>
# }
```

### 4.6 DMA Stall 分析

**roofline_model 提供 `DMA Stall Fraction`**：衡量每个 op 因 DMA 等待造成的空闲占比。

**hlo_stats 同样提供此指标**。

---

## 5. TPU v7 硬件常量与计数器

### 5.1 TPU v7 峰值参数（从真实 profile 验证）

| 参数 | 值 |
|------|-----|
| Peak FLOP Rate | 1,028,750 GFLOP/s (~1029 TFLOP/s per chip) |
| Peak HBM BW | 3,433 GiB/s |
| Peak VMEM Read BW | 27,180 GiB/s |
| Peak VMEM Write BW | 19,932 GiB/s |
| HBM Ridge Point | 279.085 FLOP/Byte |
| VMEM Read Ridge Point | 35.25 FLOP/Byte |
| VMEM Write Ridge Point | 48.07 FLOP/Byte |
| 频率 | 1.9 GHz |
| MXU 数量 | 2 per TC core |
| TC Core 数量 | 2 per chip (v7x) |
| Vector ALU 数量 | 4 per core |
| XLU 数量 | 2 per core |
| ICI 峰值读带宽 | ~295 GB/s |
| HBM Beat 大小 | 32 bytes/beat |
| ICI Flit 大小 | 128 bytes/flit |

### 5.2 MXU CPI（Cycles Per Instruction）常量

来自 `/root/xprof/xprof/convert/tpuv7generic_utilization_utils.cc`：

| 指令类型 | CPI |
|---------|-----|
| LMR BF16 | 4 |
| LMR I8 | 2 |
| LMR I4 | 2 |
| VREG BF16 | 8 |
| VREG F8 (E4M3/E5M2) | 8 |
| VREG I8 | 8 |
| VREG I4 | 8 |
| VREG F32 | 4 |

### 5.3 TPU v7 硬件计数器 ID

来自 `/root/xprof/xprof/utils/tpu_counter_ids_v7.h`，`TpuCounterIdsTpu7` 枚举，42 个计数器：

**MXU 相关**:
- `MATMUL_LMR_BF16_MXU_0/1` — LMR BF16 矩阵乘指令（每 MXU）
- `MATMUL_VREG_BF16/F8/I8/I4/F32_MXU_0/1` — VREG 矩阵乘指令
- `MXU_BUSY_0/1/2` — 0/1/2 个 MXU 忙碌的周期数
- `MATPUSH_CYCLES_MXU_0/1` — matpush 周期

**Scalar/Vector 相关**:
- `SCALAR_ALU_INSTRUCTION_0/1` — 标量 ALU 指令
- `VECTOR_ALU_INSTRUCTION_0/1/2/3` — 向量 ALU 指令（4 个 ALU）
- `VLD_INSTRUCTION_0/1` — VMEM load 指令
- `VST_INSTRUCTION` — VMEM store 指令

**XLU 相关**:
- `XLU_BUSY_0/1/2` — Cross-Lane Unit 忙碌周期
- `PACKED_XLU_0/1`, `ROTATE_PERMUTE_INSTRUCTION_XLU_0/1`, `TRANSPOSE_XLU_0/1`

**通用**:
- `CYCLES` — TC 周期计数器

### 5.4 Utilization 计算公式

来自 `/root/xprof/xprof/convert/tpuv7generic_utilization_utils.cc`：

```
Scalar Unit  = (SCALAR_ALU_0 + SCALAR_ALU_1) / (CYCLES * 2)
Vector ALUs  = (VECTOR_ALU_0 + _1 + _2 + _3) / (CYCLES * 4)
VMEM Loads   = (VLD_0 + VLD_1) / (CYCLES * 2)
VMEM Stores  = VST / CYCLES
Avg MXU Busy = (0.5 * MXU_BUSY_1 + MXU_BUSY_2) / CYCLES

Per-MXU utilization:
  mxu_busy_cycles = Σ (inst_count * CPI) for each instruction type
  utilization = mxu_busy_cycles / CYCLES

HBM BW:
  bytes = hbm_beats * 32
  utilization = bytes / (peak_hbm_bw * time)

ICI BW:
  bytes = flits * 128
  utilization = bytes / (peak_ici_bw * time)
```

---

## 6. 数据格式与转换流水线

### 6.1 核心数据格式

1. **XPlane protobuf (`.xplane.pb`)**：主格式，二进制 protobuf
   - `XSpace` → 多个 `XPlane`（每个设备/主机一个）
   - `XPlane` → 多个 `XLine`（线程/资源时间线）
   - `XLine` → 多个 `XEvent`（单个操作，皮秒精度）

2. **Google DataTable JSON**：大多数工具的输出格式
   ```json
   {
     "cols": [{"label": "col_name", "type": "string"}, ...],
     "rows": [{"c": [{"v": value}, ...]}, ...],
     "p": {"property_key": "value", ...}
   }
   ```

3. **Chrome Trace Event JSON**：trace_viewer 输出
   ```json
   {"traceEvents": [{"pid": 0, "tid": 1, "ts": 1000, "dur": 500, "name": "op", "ph": "X"}, ...]}
   ```

4. **Op Profile JSON**：嵌套树结构
   ```json
   {"byProgram": {"children": [...]}, "byCategory": {"children": [...]}}
   ```

### 6.2 转换流水线

```
                         ┌──> OpStats ──> OverviewPage (JSON)
                         │        │──> RooflineModel (JSON)
                         │        │──> OpProfile (JSON)
                         │        └──> HloStats (JSON)
.xplane.pb ──> XSpace ──┤
                         ├──> TraceEvents (Perfetto protobuf / Chrome JSON)
                         │
                         ├──> UtilizationViewer (直接从 HW counters，不经过 OpStats)
                         │
                         ├──> PerfCounters (原始 HW counter dump)
                         │
                         └──> MemoryProfile / MemoryViewer
```

### 6.3 中心调度器

**文件**: `/root/xprof/xprof/convert/xplane_to_tools_data.cc`

函数 `ConvertMultiXSpacesToToolData(session_snapshot, tool_name, options)` 根据 `tool_name` 分发到不同的转换器。

---

## 7. Protobuf 数据结构

### 7.1 本机安装路径

`/usr/local/lib/python3.12/site-packages/xprof/protobuf/` — 34 个 protobuf 模块

### 7.2 核心 Proto

**`OpMetrics`** (`op_metrics_pb2`):
- `name`, `long_name`, `category`
- `time_ps`, `self_time_ps` — 执行时间（皮秒）
- `flops_v2`, `model_flops_v2` — FLOP 计数
- `bytes_accessed` — 总内存访问字节
- `memory_accessed_breakdown[]` — 按内存空间分的读写字节
- `dma_stall_ps` — DMA 等待时间

**`OpStats`** (`op_stats_pb2`):
- `host_op_metrics_db`, `device_op_metrics_db` — Host/Device op 指标数据库
- `perf_env` → `PerfEnv`:
  - `peak_tera_flops_per_second`
  - `peak_hbm_bw_giga_bytes_per_second`
  - `peak_bws_giga_bytes_per_second[]` — [HBM, _, _, CMEM_rd, CMEM_wr, VMEM_rd, VMEM_wr]
- `run_environment` → `RunEnvironment`: device_type, device_core_count, hardware_type

**`RooflineModelDatabase`** (`roofline_model_pb2`):
- `peak_flop_rate`, `peak_hbm_bw`, `peak_vmem_read_bw`, `peak_vmem_write_bw`
- `roofline_model_record[]` → 每个 op:
  - `hlo_name`, `total_time_in_us`, `measured_flop_rate`, `model_flop_rate`
  - `hbm_bw`, `vmem_read_bw`, `vmem_write_bw`
  - `operational_intensity`, `bound_by`, `roofline_efficiency`
  - `dma_stall_fraction`, `flops_v2`, `bytes_accessed`

**`Profile`** (`op_profile_pb2`):
- `by_category`, `by_program`, `by_provenance` — Node 树
- 每个 `Node`:
  - `metrics` → `Metrics`: `raw_time`, `raw_flops`, `bf16_flops`, `raw_bytes_accessed_array[]`, `occurrences`
  - `xla` → `XLAInstruction`: op name, expression, category, provenance

**`HloStatsRecord`** (`hlo_stats_pb2`):
- `hlo_category`, `hlo_expression`, `occurrences`
- `total_time_in_us`, `measured_flop_rate`, `model_flop_rate`
- `hbm_bw`, `vmem_read_bw`, `vmem_write_bw`
- `bound_by`, `dma_stall_fraction`, `operational_intensity`

**`TpuStepBreakdown`** (`steps_db_pb2`):
- `infeed_duration_ps`, `crs_duration_ps`, `send_duration_ps`, `recv_duration_ps`
- `high_flops_compute_ps`, `tc_idle_ps`, `tc_busy_ps`
- `overlay_wait_duration_ps`

---

## 8. 实际应用：Pallas Kernel Profiling 分析方案

### 8.1 综合分析脚本设计

基于以上研究，设计一个编程化的 Pallas kernel profiling 分析流程：

```python
#!/usr/bin/env python3
"""Pallas Kernel Profiling Analyzer

用法:
    python analyze_kernel_profile.py /path/to/profile_dir
    python analyze_kernel_profile.py --compare baseline_dir optimized_dir
"""

import json
import glob
from xprof.convert._pywrap_profiler_plugin import xspace_to_tools_data


def find_xplane(profile_dir):
    """查找 profile 目录下的 .xplane.pb 文件"""
    pattern = f"{profile_dir}/**//*.xplane.pb"
    files = glob.glob(pattern, recursive=True)
    return files[0] if files else None


def extract_overview(xplane_path):
    """提取高层概览指标"""
    raw, ok = xspace_to_tools_data([xplane_path], 'overview_page')
    if not ok:
        return {}
    data = json.loads(raw)
    props = data[0].get('p', {}) if data else {}
    return {
        'mxu_utilization_pct': props.get('mxu_utilization_percent'),
        'device_idle_pct': props.get('device_idle_time_percent'),
        'flop_rate_util': props.get('flop_rate_utilization_relative_to_roofline'),
        'mem_bw_util': props.get('memory_bw_utilization_relative_to_hw_limit'),
    }


def extract_roofline(xplane_path):
    """提取 roofline 分析：每个 op 的性能特征"""
    raw, ok = xspace_to_tools_data([xplane_path], 'roofline_model')
    if not ok:
        return [], {}
    data = json.loads(raw)
    meta = data[0].get('p', {})
    cols = [c['label'] for c in data[0]['cols']]
    ops = []
    for row in data[0]['rows']:
        vals = {cols[j]: row['c'][j].get('v') for j in range(len(cols))}
        ops.append(vals)
    return ops, meta


def extract_utilization(xplane_path):
    """提取硬件利用率指标（MXU、Scalar、Vector、HBM、ICI）"""
    raw, ok = xspace_to_tools_data([xplane_path], 'utilization_viewer')
    if not ok:
        return []
    data = json.loads(raw)
    cols = [c['label'] for c in data['cols']]
    metrics = []
    for row in data['rows']:
        vals = {cols[j]: row['c'][j].get('v') for j in range(len(cols))}
        metrics.append(vals)
    return metrics


def extract_op_profile(xplane_path):
    """提取 op profile 树"""
    raw, ok = xspace_to_tools_data([xplane_path], 'op_profile')
    if not ok:
        return {}
    return json.loads(raw)


def analyze_kernel(profile_dir):
    """完整分析一个 profile"""
    xplane = find_xplane(profile_dir)
    if not xplane:
        print(f"No .xplane.pb found in {profile_dir}")
        return

    print(f"=== Analyzing: {profile_dir} ===")
    print(f"XPlane: {xplane}\n")

    # 1. Overview
    overview = extract_overview(xplane)
    print("--- Overview ---")
    for k, v in overview.items():
        print(f"  {k}: {v}")

    # 2. Top ops by time (roofline)
    ops, meta = extract_roofline(xplane)
    print(f"\n--- Roofline (Peak: {meta.get('peak_flop_rate')} GFLOP/s, "
          f"HBM: {meta.get('peak_hbm_bw')} GiB/s) ---")
    print(f"  {'Operation':<50} {'Time(us)':>10} {'FLOP Rate':>12} {'Bound By':>12}")
    for op in sorted(ops, key=lambda x: x.get('Total self time (us)', 0), reverse=True)[:15]:
        print(f"  {op.get('Operation', ''):<50} "
              f"{op.get('Total self time (us)', 0):>10.1f} "
              f"{op.get('Measured FLOP Rate (GFLOP/s/core)', 0):>12.1f} "
              f"{op.get('Bound by', ''):>12}")

    # 3. Hardware utilization
    utils = extract_utilization(xplane)
    print("\n--- Hardware Utilization ---")
    for m in utils:
        name = m.get('Name', '')
        if name in ('Avg MXU Busy', 'Scalar Unit', 'Vector ALUs',
                     'VMEM Loads', 'VMEM Stores', 'Avg XLU Busy'):
            achieved = m.get('Achieved', 0)
            peak = m.get('Peak', 1)
            pct = (achieved / peak * 100) if peak else 0
            print(f"  {name:<20}: {pct:.1f}%  ({achieved}/{peak})")

    return {
        'overview': overview,
        'roofline_ops': ops,
        'roofline_meta': meta,
        'utilization': utils,
    }


def compare_profiles(baseline_dir, optimized_dir):
    """对比两个 profile"""
    print("=" * 80)
    print("BASELINE")
    print("=" * 80)
    baseline = analyze_kernel(baseline_dir)

    print("\n" + "=" * 80)
    print("OPTIMIZED")
    print("=" * 80)
    optimized = analyze_kernel(optimized_dir)

    if baseline and optimized:
        print("\n" + "=" * 80)
        print("COMPARISON")
        print("=" * 80)
        for key in baseline.get('overview', {}):
            b = baseline['overview'].get(key)
            o = optimized['overview'].get(key)
            if b is not None and o is not None:
                delta = o - b
                print(f"  {key}: {b:.2f} → {o:.2f} (Δ{delta:+.2f})")
```

### 8.2 与当前 benchmark 工作流的结合

当前工作流（benchmark only）：
```
修改 kernel → bench_fused_moe_kernel → 看 wall time → 判断好坏
```

改进后的工作流（profiling-driven）：
```
修改 kernel → profiling → 分析脚本提取指标 → 定量诊断瓶颈 → 指导下一步优化
```

具体结合方式：

1. **每次 kernel 修改后**，自动运行 profiling + 分析脚本
2. **定量回答问题**：
   - Scalar ALU 利用率降了多少？（R1-R8 优化目标）
   - MXU 利用率提升了多少？
   - DMA stall fraction 变化？
   - 哪个 op 仍然是瓶颈？bound by compute 还是 memory？
3. **对比分析**：baseline vs optimized 的逐指标 diff

### 8.3 注意事项

1. **Step markers**：当前 profiling 使用 `--xla_enable_custom_call_region_trace=true`，
   可能缺少 step 标记，导致 overview_page 部分指标为空。需要确保 profiling 捕获完整计算步骤。

2. **perf_counters 数据量大**：单次 profile 约 87MB JSON，避免频繁使用。

3. **trace_viewer 可能崩溃**：大 profile 使用 `trace_viewer` 可能 segfault，推荐用 `trace_viewer@`（流式）。

4. **utilization_viewer 是最精确的**：直接来自硬件计数器，不经过 OpStats 中间层，
   是判断 Scalar ALU 阻塞 MXU 的最直接证据。

---

## 9. 关键文件索引

### 9.1 xprof 仓库 (`/root/xprof/`)

| 文件 | 用途 |
|------|------|
| `xprof/convert/xplane_to_tools_data.cc` | **中心调度器**：tool_name → 转换器 |
| `xprof/convert/tpu_counter_util.cc` | TPU 硬件计数器处理 |
| `xprof/convert/tpuv7generic_utilization_utils.cc` | **TPU v7 利用率计算**：MXU、Scalar、Vector、HBM、ICI |
| `xprof/utils/tpu_counter_ids_v7.h` | TPU v7 硬件计数器寄存器 ID |
| `xprof/utils/tpu_counter_ids_v7x.h` | TPU v7x 变体 |
| `xprof/convert/op_metrics_to_record.h` | Roofline 指标计算（per-op） |
| `xprof/convert/op_stats_to_roofline_model.cc` | OpStats → RooflineModel |
| `xprof/convert/op_stats_to_op_profile.h` | OpStats → OpProfile |
| `xprof/convert/xplane_to_utilization_viewer.cc` | XPlane → Utilization（直接路径） |
| `xprof/pywrap/pywrap_profiler_plugin.cc` | pybind11 Python 绑定 |
| `plugin/xprof/convert/raw_to_tool_data.py` | Python wrapper |
| `plugin/xprof/protobuf/*.proto` | 所有 protobuf 定义 |

### 9.2 accelerator-agents 仓库 (`/root/accelerator-agents/`)

| 文件 | 用途 |
|------|------|
| `MaxKernel/tpu_kernel_gen/agents/kernel_gen_agent/analyze_profile.py` | **xprof trace_viewer API 用法**：SyncWait ratio 计算 |
| `MaxKernel/tpu_kernel_gen/agents/hitl_kernel_gen_agent/subagents/profiling/offline_tools.py` | **SQLite 分析模式**：xplane → SQLite → SQL 查询 |
| `MaxKernel/tpu_kernel_gen/agents/hitl_kernel_gen_agent/subagents/profiling/agent.py` | Profiling Agent 编排（5 步） |
| `MaxKernel/tpu_kernel_gen/agents/hitl_kernel_gen_agent/subagents/profiling/prompts/analyze_profile_prompt.py` | 分析 Agent Prompt |
| `MaxKernel/tpu_kernel_gen/agents/hitl_kernel_gen_agent/subagents/profiling/prompts/gen_profiling_script.py` | Profiling 脚本生成模板 |
| `MaxKernel/tpu_kernel_gen/agents/kernel_gen_agent/kernel_eval/tpu_server.py` | TPU Server /profile 端点 |
| `MaxKernel/tpu_kernel_gen/agents/kernel_gen_agent/prompts/pallas_profiling_docs.py` | Pallas profiling 文档/指南 |
| `MaxKernel/tpu_kernel_gen/agents/hitl_kernel_gen_agent/tpu_specs.json` | TPU 规格数据库 (v4/v5e/v5p/v6e/v7x) |

### 9.3 本机已安装 xprof 包

| 文件 | 用途 |
|------|------|
| `/usr/local/lib/python3.12/site-packages/xprof/convert/_pywrap_profiler_plugin.so` | **核心 C++ 扩展** |
| `/usr/local/lib/python3.12/site-packages/xprof/convert/raw_to_tool_data.py` | Python API wrapper |
| `/usr/local/lib/python3.12/site-packages/xprof/protobuf/*.py` | 34 个 protobuf 模块 |
| `/usr/local/lib/python3.12/site-packages/xprof/profile_plugin.py` | Web 插件（1893 行） |
| `/usr/local/lib/python3.12/site-packages/xprof/api/continuous_profiling_snapshot.py` | 持续 profiling API |

---

## 附录 A: accelerator-agents MaxKernel Agent 架构

MaxKernel 包含三套 Agent 系统：

### A.1 kernel_gen_agent（全自动流水线）

```
OrganizeCode → ConvertToJax → JaxConversionLoop →
WriteBaseKernel → BaseKernelRefinementLoop →
AdjustInputShapes → AddKernelTiling → TiledKernelRefinementLoop →
GenerateTuningScript → KernelTilingOptimizer → Summary
```

- LLM: `gemini-3-pro-preview`，Temperature: 0.1
- 使用 Google ADK 框架（SequentialAgent, LoopAgent）
- 自动重试 + Refinement Loop

### A.2 hitl_kernel_gen_agent（人机协作）

```
KernelGenerationOrchestrationAgent (root)
  ├── ExplanationAgent
  ├── PlanKernelAgent
  ├── ImplementKernelAgent
  ├── ValidateKernelCompilationAgent (4 次自动修复)
  ├── ValidatedTestGenerationAgent (生成 + 验证 + 修复循环)
  ├── UnifiedTestAgent (在 TPU 上执行 pytest)
  ├── ProfileAgentOrchestrator
  │     ├── ReadFileForProfilingAgent
  │     ├── GenerateProfilingScriptAgent
  │     ├── ReadProfilingScriptAgent
  │     ├── ProfileEvalAgent (KernelProfiler)
  │     └── SummarizeProfileAgent (使用 offline xprof 工具)
  └── GpuToJaxAgent (12 步 CUDA/Triton/PyTorch→JAX)
```

- "One Agent, Then Wait" 模式
- 使用 BuiltInPlanner + ThinkingConfig(thinking_level="high")

### A.3 open_optimization_agent（进化式优化）

```
NeverExitLoopAgent (max 500 iterations)
  ├── IdeaGenerationAgent (LoopAgent: Idea + Judge, max 5)
  ├── KernelWriterAgent
  └── EvalAgent (compilation → correctness → performance → summary)
```

- 配合 OpenEvolve 使用
- 维护 "historical best" 防止回退

### A.4 关键设计模式

1. **State 传递**: 通过 `ctx.session.state` 共享字典，`input_key`/`output_key` 模式
2. **三层服务器**: Agent → Eval Server(:1245) → TPU Server(:5463) / CPU Server(:5464)
3. **MCP 文件系统**: 通过 `@modelcontextprotocol/server-filesystem` 访问文件
4. **RAG 检索**: Vertex AI RAG + BigQuery 向量搜索（UniXcoder embeddings）
5. **Kernel RAG**: 从相似 kernel 数据库中检索参考实现
