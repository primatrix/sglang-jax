# xprof Core API 参考手册

> **用途**: 编程化分析 JAX/TPU profiling 数据，替代手动查看 xprof Web UI
>
> **本机环境**: xprof v2.22.0, 路径 `/usr/local/lib/python3.12/site-packages/xprof/`

---

## 1. API 入口

### 1.1 底层 C++ 扩展

```python
from xprof.convert._pywrap_profiler_plugin import xspace_to_tools_data
```

**签名**:
```python
xspace_to_tools_data(
    xspace_paths: list[str],   # .xplane.pb 文件路径列表
    tool_name: str,             # 工具名（见下文工具列表）
    options: dict = {}          # 工具特定选项
) -> tuple[bytes, bool]        # (原始数据, 是否成功)
```

这是所有分析的基础。返回的 `bytes` 通常是 JSON 字符串，用 `json.loads()` 解析。

### 1.2 Python Wrapper

```python
from xprof.convert.raw_to_tool_data import xspace_to_tool_data, xspace_to_tool_names
```

**`xspace_to_tool_data`**:
```python
xspace_to_tool_data(
    xspace_paths: list[str],
    tool: str,
    params: dict,
    xspace_wrapper_func=_pywrap_profiler_plugin.xspace_to_tools_data
) -> tuple[data, content_type]
# data: 已处理的数据（JSON 字符串或 bytes）
# content_type: 'application/json' | 'text/html' | 'application/octet-stream'
```

**`xspace_to_tool_names`**:
```python
xspace_to_tool_names(xspace_paths: list[str]) -> list[str]
# 返回可用工具名列表
```

### 1.3 基本用法模式

```python
import json, glob
from xprof.convert._pywrap_profiler_plugin import xspace_to_tools_data

# 1. 查找 xplane.pb 文件
xplane = glob.glob('/path/to/profile/plugins/profile/*/*.xplane.pb')[0]

# 2. 调用工具
raw_data, success = xspace_to_tools_data([xplane], 'roofline_model')

# 3. 解析 JSON
if success:
    data = json.loads(raw_data)
```

---

## 2. 完整工具列表与详解

### 2.1 `tool_names` — 列出可用工具

```python
raw, ok = xspace_to_tools_data([xplane], 'tool_names')
tools = raw.decode().split(',')
# 典型输出:
# ['trace_viewer@', 'overview_page', 'input_pipeline_analyzer',
#  'framework_op_stats', 'memory_profile', 'op_profile', 'hlo_stats',
#  'roofline_model', 'memory_viewer', 'graph_viewer', 'perf_counters',
#  'utilization_viewer']
```

### 2.2 `overview_page` — 高层性能摘要

**返回格式**: JSON，包含多个 DataTable

```python
raw, ok = xspace_to_tools_data([xplane], 'overview_page')
data = json.loads(raw)

# data 是一个 list，通常 3 个 DataTable:
# data[0] — Top Operations 表 + 全局属性
# data[1] — Step Time Breakdown
# data[2] — Run Environment

# === 关键属性 (data[0]['p']) ===
props = data[0].get('p', {})
props['mxu_utilization_percent']                        # MXU 利用率
props['device_duty_cycle_percent']                      # 设备负载率
props['device_idle_time_percent']                       # 设备空闲率
props['flop_rate_utilization_relative_to_roofline']     # FLOP 利用率 vs roofline
props['memory_bw_utilization_relative_to_hw_limit']     # 内存带宽利用率
props['hbm_utilization_percent']                        # HBM 利用率
props['device_tf_op_percent']                           # Device op 时间占比
props['host_tf_op_percent']                             # Host op 时间占比

# === Top Operations 表 ===
# 列: Time(%), Cumulative time(%), Category, Operation, Bf16 Normalized Flop Rate
```

**实测示例**（profile_r6）:
```
Top Op: pallas_call (jit(run_fused)/jit(fused_ep_moe)/.../pallas_call)
  Time: 79.18%
  Flop Rate: 7563.78 GFLOP/s
```

**注意**: 如果 profiling 使用了 `--xla_enable_custom_call_region_trace=true` 且没有 step marker，
`mxu_utilization_percent` 可能显示 0.0%，因为缺少完整的 step 信息。

### 2.3 `roofline_model` — Roofline 分析（最详细的 per-op 分析）

**返回格式**: JSON DataTable，每行一个 op

```python
raw, ok = xspace_to_tools_data([xplane], 'roofline_model')
data = json.loads(raw)

# === 峰值参数 (data[0]['p']) ===
meta = data[0]['p']
meta['device_type']        # "TPU v7x"
meta['peak_flop_rate']     # "1.02875e+06" (GFLOP/s, ~1029 TFLOP/s)
meta['peak_hbm_bw']        # "3433" (GiB/s)
meta['peak_vmem_read_bw']  # "27180" (GiB/s)
meta['peak_vmem_write_bw'] # "19932" (GiB/s)
meta['hbm_ridge_point']    # "279.085" (FLOP/Byte)
meta['vmem_read_ridge_point']  # "35.2501"
meta['vmem_write_ridge_point'] # "48.0683"
meta['megacore']           # "0" or "1"

# === 37 列完整列表 ===
cols = [c['label'] for c in data[0]['cols']]
# ['Step', 'Rank', 'Category', 'Operation', '# Occurrences',
#  'Total Time (us)', 'Avg. time (us)', 'Total self time (us)',
#  'Avg. self time (us)', 'Total self time (%)',
#  'DMA Stall (%)',
#  'Measured FLOP Rate (GFLOP/s/core)',
#  'Normalized FLOP Rate (GFLOP/s/core)',     # BF16 normalized
#  'Model FLOP Rate (GFLOP/s/core)',
#  'Measured memory BW (GiB/s)',
#  'HBM BW (GiB/s)',
#  'CMEM Read BW (GiB/s)', 'CMEM Write BW (GiB/s)',
#  'VMEM Read BW (GiB/s)', 'VMEM Write BW (GiB/s)',
#  'Operational Intensity (FLOP/Byte)',
#  'HBM Operational Intensity (FLOP/Byte)',
#  'CMEM Read Operational Intensity', 'CMEM Write Operational Intensity',
#  'VMEM Read Operational Intensity', 'VMEM Write Operational Intensity',
#  'Bound by',                                 # 'Compute' | 'HBM' | 'VMEM Read' | ...
#  'Roofline Efficiency (%)',
#  'FLOP Rate / Peak (%)',
#  'MBW / Peak (%)',
#  'FLOPs (v2)', 'Model FLOPs', 'Bytes Accessed',
#  'HBM Bytes', 'CMEM Read Bytes', 'CMEM Write Bytes',
#  'VMEM Read Bytes', 'VMEM Write Bytes']

# === 解析 per-op 数据 ===
for row in data[0]['rows']:
    vals = {cols[j]: row['c'][j].get('v') for j in range(len(cols))}
    print(f"{vals['Operation']:<60} "
          f"Time={vals['Total self time (us)']:>10.1f}us "
          f"FLOP={vals['Measured FLOP Rate (GFLOP/s/core)']:>10.1f} "
          f"Bound={vals['Bound by']}")
```

**实测示例**（profile_r6, 84 ops）:
```
Total Program:  Time=27637.07us  FLOP Rate=1889.05 GFLOP/s  HBM BW=18.64 GiB/s  Bound=HBM
  → Roofline efficiency: 0.0054%  (说明被 HBM 严重瓶颈)
```

**关键指标解读**:
- `Bound by = "Compute"`: 计算密集型，MXU 是瓶颈
- `Bound by = "HBM"`: 内存带宽瓶颈
- `Bound by = "VMEM Read"/"VMEM Write"`: VMEM 带宽瓶颈
- `DMA Stall (%)`: DMA 等待占比，高值表示 DMA 调度不佳
- `Roofline Efficiency (%)`: 相对于 roofline 理论上限的效率

**Roofline 瓶颈判定算法** (来自 `op_metrics_to_record.h:SetRooflineMetrics`):
```
对每个资源 (Compute, HBM, CMEM_rd, CMEM_wr, VMEM_rd, VMEM_wr):
  utilization = measured / peak
瓶颈 = max(utilization) 对应的资源
roofline_efficiency = max(utilization)
```

### 2.4 `op_profile` — Op 时间分解树

**返回格式**: JSON 嵌套树结构

```python
raw, ok = xspace_to_tools_data([xplane], 'op_profile')
# 可选 options:
# raw, ok = xspace_to_tools_data([xplane], 'op_profile', {'group_by': 'program'})
data = json.loads(raw)

# === 顶层结构 ===
data['byProgram']               # 按 program 分组的 Node 树
data['byCategory']              # 按 category 分组
data['byProgramExcludeIdle']    # 排除 IDLE 的 program 树
data['deviceType']              # 设备类型字符串
data['aggDvfsTimeScaleMultiplier']  # DVFS 时间缩放因子

# === Node 结构 ===
# 每个 node 包含:
node = data['byProgram']['children'][0]  # 第一个 program
node['name']                     # program 名
node['metrics']                  # 性能指标
node['children']                 # 子 op 列表

# === Metrics 字段 ===
metrics = node['metrics']
metrics['rawTime']               # 原始时间 (picoseconds)
metrics['rawFlops']              # 原始 FLOP 计数
metrics['bf16Flops']             # BF16 FLOP 计数
metrics['rawBytesAccessedArray'] # [HBM, CMEM_rd, CMEM_wr, VMEM_rd, VMEM_wr]
metrics['normalizedTimePs']      # 归一化时间 (ps)
metrics['occurrences']           # 出现次数

# === XLA 元数据（子 op 级别）===
op = node['children'][0]
xla = op.get('xla', {})
xla['op']                        # HLO op 名
xla['expression']                # HLO 表达式
xla['category']                  # 类别 (e.g., 'custom-call')
xla['provenance']                # 源代码位置
```

**实测示例**（profile_r6）:
```
Program: main
  custom-call:    rawTime=6,935,405,742 ps  rawFlops=52,200,996,864
  async-done:     rawTime=  834,140,451 ps  rawFlops=0
  custom fusion:  rawTime=  488,415,367 ps  rawFlops=0
  data formatting: rawTime= 229,213,684 ps  rawFlops=0
  sort:           rawTime=  102,680,676 ps  rawFlops=1,290,240
```

### 2.5 `hlo_stats` — HLO Op 统计表

**返回格式**: JSON DataTable，与 roofline_model 类似但组织方式不同

```python
raw, ok = xspace_to_tools_data([xplane], 'hlo_stats')
data = json.loads(raw)

# 列类似 roofline_model，但按 HLO op 分组
# 关键列: hlo_category, hlo_expression, occurrences,
#         total_time_in_us, measured_flop_rate, bound_by,
#         dma_stall_fraction, operational_intensity
```

### 2.6 `utilization_viewer` — 硬件利用率（基于硬件计数器）

**返回格式**: JSON DataTable

```python
raw, ok = xspace_to_tools_data([xplane], 'utilization_viewer')
data = json.loads(raw)

# 列: Host, Device, Sample, Node, Name, Achieved, Peak, Unit
cols = [c['label'] for c in data['cols']]
for row in data['rows']:
    vals = {cols[j]: row['c'][j].get('v') for j in range(len(cols))}
    name = vals['Name']
    achieved = vals['Achieved']
    peak = vals['Peak']
    # ...
```

**可获取的指标列表**（每个 TC core 一套）:

| Name | 含义 | 峰值基准 | 单位 |
|------|------|---------|------|
| `Scalar Unit` | 标量 ALU 利用率 | cycles * 2 (2 slots) | instructions |
| `Vector ALUs` | 向量 ALU 利用率 | cycles * 4 (4 ALUs) | instructions |
| `VMEM Loads` | VMEM 读取利用率 | cycles * 2 (2 ports) | instructions |
| `VMEM Stores` | VMEM 写入利用率 | cycles | instructions |
| `Avg MXU Busy` | MXU 平均忙碌率 | cycles | cycles |
| `2 MXU Busy` | 两个 MXU 都忙碌的周期 | cycles | cycles |
| `1 MXU Busy` | 一个 MXU 忙碌的周期 | cycles | cycles |
| `No MXU Busy` | 无 MXU 忙碌的周期 | cycles | cycles |
| `MXU BF16` | BF16 MXU 忙碌周期 | cycles * 2 | cycles |
| `MXU E4M3 + E5M2` | FP8 MXU 忙碌周期 | cycles * 2 | cycles |
| `MXU matpush` | matpush 周期 | cycles * 2 | cycles |
| `MXU0` / `MXU1` | 各 MXU 忙碌周期 | cycles | cycles |
| `Avg XLU Busy` | XLU 利用率 | cycles | cycles |
| `HBM Rd+Wr - core N` | HBM 读写总量 | peak_bw * time | bytes |
| `HBM Read Ratio` | HBM 读占比 | 1 | ratio |
| `HBM Write Ratio` | HBM 写占比 | 1 | ratio |
| `ICI (Read)` | ICI 读带宽 | peak_ici_bw * time | bytes |
| `ICI (Write)` | ICI 写带宽 | peak_ici_bw * time | bytes |

**利用率计算**:
```python
utilization_pct = achieved / peak * 100  # 如果 peak > 0
```

**为什么这个工具最重要**: 它直接来自硬件计数器，不经过 OpStats 中间层。
对于 Pallas kernel profiling（无法看到内部 op 分解），这是**唯一能获取 MXU vs Scalar 占比的途径**。

### 2.7 `framework_op_stats` — 框架级 Op 统计

```python
raw, ok = xspace_to_tools_data([xplane], 'framework_op_stats')
data = json.loads(raw)

# 列: Rank, Host/device, Operation Type, Operation Name, #Occurrences,
#     Total time (us), Avg. time (us), Total self-time (us),
#     Total self-time on Device (%), Total self-time on Host (%),
#     Bf16 Normalized Flop Rate (GFLOP/s)
```

**实测**（profile_r6, 30 ops）:
```
Rank 1: IDLE, Total self-time: 18920.49 us (68.46% on Device)
```

### 2.8 `perf_counters` — 原始硬件计数器

```python
raw, ok = xspace_to_tools_data([xplane], 'perf_counters')
data = json.loads(raw)

# 列: Host, Chip, Kernel, Sample, Counter, Value, Description, Set
# 注意: 数据量很大（~87MB JSON），仅在需要原始计数器值时使用
```

### 2.9 `trace_viewer@` — 流式时间线

```python
raw, ok = xspace_to_tools_data([xplane], 'trace_viewer@', {
    'resolution': 8000,          # 分辨率
    'start_time_ms': 100,        # 起始时间
    'end_time_ms': 200,          # 结束时间
    'full_dma': True,            # 显示完整 DMA 信息
})
# 返回 Perfetto protobuf 格式（binary），非 JSON
```

**对比 `trace_viewer`（非流式）**:
```python
# 非流式版本返回 Chrome Trace Event 格式
from xprof.convert.raw_to_tool_data import xspace_to_tool_data
data, ct = xspace_to_tool_data([xplane], 'trace_viewer', {})
# data 是 JSON 字符串，包含 {"traceEvents": [...]}
# 注意: 大 profile 可能导致内存溢出或 segfault，推荐用 trace_viewer@
```

### 2.10 `memory_profile` — 内存使用时间线

```python
raw, ok = xspace_to_tools_data([xplane], 'memory_profile')
# 注意: 只支持单个 xplane 文件
data = json.loads(raw)
```

### 2.11 `memory_viewer` — 内存布局

```python
raw, ok = xspace_to_tools_data([xplane], 'memory_viewer', {
    'module_name': '<module_name>',
    'program_id': '',
    'memory_space': '',          # 空字符串 = 默认
    'view_memory_allocation_timeline': False,
})
```

### 2.12 `graph_viewer` — HLO 图

```python
raw, ok = xspace_to_tools_data([xplane], 'graph_viewer', {
    'graph_viewer_options': {
        'node_name': '<node>',
        'module_name': '<module>',
        'graph_width': 3,
        'type': 'graph',         # 'pb' | 'pbtxt' | 'json' | 'graph' | 'short_txt' | 'long_txt'
    }
})
```

### 2.13 `smart_suggestion` — AI 优化建议

```python
raw, ok = xspace_to_tools_data([xplane], 'smart_suggestion')
data = json.loads(raw)
# 返回 xprof 生成的优化建议文本
```

---

## 3. 数据格式详解

### 3.1 DataTable 格式（大多数工具）

```json
{
  "cols": [
    {"id": "col0", "label": "Column Name", "type": "string"},
    {"id": "col1", "label": "Value", "type": "number"}
  ],
  "rows": [
    {"c": [{"v": "row1_val"}, {"v": 42.0}]},
    {"c": [{"v": "row2_val"}, {"v": 99.0}]}
  ],
  "p": {
    "property_key": "property_value"
  }
}
```

**解析模板**:
```python
def parse_datatable(data):
    """通用 DataTable 解析器"""
    if isinstance(data, list):
        table = data[0]
    else:
        table = data
    cols = [c['label'] for c in table['cols']]
    props = table.get('p', {})
    rows = []
    for row in table.get('rows', []):
        vals = {}
        for j, col in enumerate(cols):
            cell = row['c'][j]
            vals[col] = cell.get('v') if cell else None
        rows.append(vals)
    return cols, rows, props
```

### 3.2 Op Profile 树结构

```json
{
  "byProgram": {
    "name": "",
    "metrics": {"rawTime": 12345, ...},
    "children": [
      {
        "name": "main",
        "metrics": {...},
        "children": [
          {
            "name": "custom-call.42",
            "metrics": {"rawTime": 6935405742, "rawFlops": 52200996864, ...},
            "xla": {
              "op": "custom-call",
              "expression": "f32[512,2048]{1,0} custom-call(...)",
              "category": "custom-call",
              "provenance": "jit(fused_ep_moe)/pallas_call"
            }
          }
        ]
      }
    ]
  },
  "byCategory": {...},
  "byProgramExcludeIdle": {...},
  "deviceType": "TPU v7x"
}
```

### 3.3 Chrome Trace Event 格式（trace_viewer）

```json
{
  "traceEvents": [
    {
      "pid": 0,          // 设备 ID
      "tid": 1,          // 线程/资源 ID
      "ts": 1000.5,      // 时间戳（微秒）
      "dur": 500.2,      // 持续时间（微秒）
      "ph": "X",         // 完成事件
      "name": "matmul",  // 操作名
      "args": {
        "name": "/device:TPU:0",
        "hlo_op": "dot.123"
      }
    }
  ]
}
```

---

## 4. Protobuf 结构参考

本机 protobuf 路径: `/usr/local/lib/python3.12/site-packages/xprof/protobuf/`

### 4.1 核心 Proto

**`OpMetrics`** (`op_metrics_pb2`):
```
name, long_name, category                # 标识
time_ps, self_time_ps                    # 时间（皮秒）
flops_v2, model_flops_v2                 # FLOP 计数
bytes_accessed                           # 总内存访问
memory_accessed_breakdown[]              # 按空间分: {memory_space, read/write, bytes}
dma_stall_ps                             # DMA 等待时间
```

**`PerfEnv`** (`op_stats_pb2`):
```
peak_tera_flops_per_second               # 峰值算力
peak_hbm_bw_giga_bytes_per_second        # HBM 峰值带宽
peak_bws_giga_bytes_per_second[]          # [HBM, _, _, CMEM_rd, CMEM_wr, VMEM_rd, VMEM_wr]
ridge_point                              # Roofline ridge point
has_cmem, has_merged_vmem, has_megacore  # 硬件特征标志
```

**`RooflineModelRecord`** (`roofline_model_pb2`):
```
hlo_name, total_time_in_us
measured_flop_rate, model_flop_rate      # GFLOP/s
measured_memory_bw                        # GiB/s (总)
hbm_bw, vmem_read_bw, vmem_write_bw     # GiB/s (分空间)
operational_intensity                     # FLOP/Byte
bound_by                                 # 瓶颈资源
roofline_efficiency                      # 效率
dma_stall_fraction                       # DMA 等待占比
flops_v2, bytes_accessed                 # 原始值
```

### 4.2 完整模块列表（34 个）

```
op_metrics_pb2          op_stats_pb2            op_profile_pb2
roofline_model_pb2      hlo_stats_pb2           kernel_stats_pb2
overview_page_pb2       memory_profile_pb2      memory_viewer_preprocess_pb2
trace_events_pb2        trace_events_old_pb2    trace_events_raw_pb2
steps_db_pb2            topology_pb2            hardware_types_pb2
diagnostics_pb2         dcn_collective_info_pb2 dcn_slack_analysis_pb2
input_pipeline_pb2      ...
```

---

## 5. TPU v7 硬件参数（从真实 profile 验证）

| 参数 | 值 | 来源 |
|------|-----|------|
| Peak FLOP Rate | 1,028,750 GFLOP/s (~1029 TFLOP/s) | roofline_model `p.peak_flop_rate` |
| Peak HBM BW | 3,433 GiB/s | roofline_model `p.peak_hbm_bw` |
| Peak VMEM Read BW | 27,180 GiB/s | roofline_model `p.peak_vmem_read_bw` |
| Peak VMEM Write BW | 19,932 GiB/s | roofline_model `p.peak_vmem_write_bw` |
| HBM Ridge Point | 279.085 FLOP/Byte | roofline_model `p.hbm_ridge_point` |
| VMEM Read Ridge Point | 35.25 FLOP/Byte | roofline_model `p.vmem_read_ridge_point` |
| VMEM Write Ridge Point | 48.07 FLOP/Byte | roofline_model `p.vmem_write_ridge_point` |
| 频率 | 1.9 GHz | xplane_to_utilization_viewer.cc 硬编码 |
| MXU 数量 | 2 per TC core | 同上 |
| TC Core 数量 | 2 per chip (v7x) | 同上 |
| Vector ALU 数量 | 4 per core | tpuv7generic_utilization_utils.cc |
| XLU 数量 | 2 per core | 同上 |
| HBM Beat 大小 | 32 bytes | 同上 |
| ICI Flit 大小 | 128 bytes | 同上 |
| ICI 峰值读带宽 | ~295 GB/s | 硬编码 kPeakIciRdBwBytesPerSecond |

### 5.1 MXU CPI 常量

| 指令类型 | CPI | 说明 |
|---------|-----|------|
| LMR BF16 | 4 | LMR 路径 BF16 矩阵乘 |
| LMR I8 | 2 | LMR 路径 INT8 |
| LMR I4 | 2 | LMR 路径 INT4 |
| VREG BF16 | 8 | VREG 路径 BF16 |
| VREG F8 | 8 | VREG 路径 FP8 (E4M3/E5M2) |
| VREG I8 | 8 | VREG 路径 INT8 |
| VREG I4 | 8 | VREG 路径 INT4 |
| VREG F32 | 4 | VREG 路径 FP32 |

### 5.2 Utilization 计算公式

```
Scalar Unit  = (SCALAR_ALU_INST_0 + _1) / (CYCLES * 2)
Vector ALUs  = (VEC_ALU_INST_0 + _1 + _2 + _3) / (CYCLES * 4)
VMEM Loads   = (VLD_INST_0 + VLD_INST_1) / (CYCLES * 2)
VMEM Stores  = VST_INST / CYCLES

Avg MXU Busy = (0.5 * MXU_BUSY_1 + MXU_BUSY_2) / CYCLES
  其中: MXU_BUSY_0 = 0 个 MXU 忙碌的周期
        MXU_BUSY_1 = 1 个 MXU 忙碌的周期
        MXU_BUSY_2 = 2 个 MXU 忙碌的周期

Per-MXU Utilization:
  busy_cycles = Σ (instruction_count_per_type * CPI_per_type)
  utilization = busy_cycles / CYCLES

HBM BW:
  total_bytes = Σ(hbm_beats_per_channel) * 32
  utilization = total_bytes / (peak_hbm_bw_bytes * time_seconds)

ICI BW:
  total_bytes = Σ(ici_flits_per_router) * 128
  utilization = total_bytes / (peak_ici_bw * time_seconds)
```

---

## 6. 持续 Profiling API

```python
from xprof.api import continuous_profiling_snapshot

# 前提: JAX 已启动 profiler server
import jax.profiler
jax.profiler.start_server(9999)

# 开始持续 profiling（常驻后台，~7us/packet 开销，~2GB 环形缓冲区）
continuous_profiling_snapshot.start_continuous_profiling('localhost:9999', {})

# 在检测到异常时快照（捕获最近 ~90 秒）
continuous_profiling_snapshot.get_snapshot('localhost:9999', '/tmp/profile-data/')

# 停止
continuous_profiling_snapshot.stop_continuous_profiling('localhost:9999')
```

---

## 7. 常见陷阱与最佳实践

1. **Step markers 缺失**: 使用 `--xla_enable_custom_call_region_trace=true` 时，
   overview_page 的 MXU 利用率可能为 0%。需要完整的 step 标记才能计算准确值。

2. **perf_counters 数据量大**: 单次 profile 约 87MB JSON。仅在需要原始计数器值时使用。
   优先使用 `utilization_viewer`（已经过聚合处理）。

3. **trace_viewer 内存溢出**: 大 profile 使用非流式 `trace_viewer` 可能 OOM 或 segfault。
   推荐使用流式 `trace_viewer@` 并指定时间范围。

4. **xspace_to_tools_data_from_byte_string 不稳定**: 测试中 `tool_names` 查询返回失败。
   优先使用文件路径版本。

5. **文件路径**: 传入的是 `.xplane.pb` 文件的完整路径，不是目录。
   Profile 目录结构: `<logdir>/plugins/profile/<timestamp>/<hostname>.xplane.pb`

6. **多 host**: 对于多 host profiling，传入所有 host 的 xplane.pb 路径列表。
   单 host 场景只需一个路径。
