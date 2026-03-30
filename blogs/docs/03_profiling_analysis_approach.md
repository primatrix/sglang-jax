# Pallas Kernel Profiling 分析方案

> **目标**: 设计一套编程化的 profiling 分析方案，覆盖两种不同的分析场景：
> 1. **XLA 编译级 trace 分析** — 可以看到算子操作（matmul, scatter, gather 等）
> 2. **Kernel 内部分析** — 只能看到硬件单元利用率（MXU, ALU, VMEM 等）
>
> **硬件**: TPU v7x (Ironwood), xprof v2.22.0

---

## 1. 两种分析场景的本质区别

### 1.1 场景对比

```
场景 A: XLA 编译级 Trace                      场景 B: Pallas Kernel 内部
┌──────────────────────────────┐             ┌──────────────────────────────┐
│  可以看到:                     │             │  可以看到:                     │
│  • 每个 HLO op 的时间           │             │  • MXU busy/idle 周期数        │
│  • matmul, scatter, reduce     │             │  • Scalar ALU 指令数          │
│  • 每个 op 的 FLOP/bytes/BW    │             │  • Vector ALU 指令数          │
│  • DMA stall per op            │             │  • VMEM load/store 指令数     │
│  • roofline per op             │             │  • HBM 读写字节数             │
│  • op 之间的 timeline           │             │  • ICI 通信字节数             │
│                                │             │  • XLU 利用率                 │
│  不能看到:                     │             │                               │
│  • Pallas kernel 内部           │             │  不能看到:                     │
│    (整个 kernel 是一个 op)      │             │  • 具体在执行什么操作           │
│                                │             │  • 哪一行 Pallas 代码在执行     │
│  适用: XLA 图级别优化           │             │  • op 之间的边界               │
│  工具: roofline, op_profile,   │             │                               │
│        hlo_stats               │             │  适用: Kernel 内部优化          │
└──────────────────────────────┘             │  工具: utilization_viewer,     │
                                             │        perf_counters           │
                                             └──────────────────────────────┘
```

### 1.2 为什么 Pallas kernel 是一个 "黑盒"

在 xprof 中，Pallas kernel 作为 `custom-call` 出现在 trace 中。
整个 kernel（包括内部的 matmul, scatter, 循环等）是**单个 HLO op**，
xprof 无法看到其内部细节。

```
XLA 层面看到的:
  jit_computation
    └── custom-call (pallas_call: fused-moe-k_8-bt_64-bf_2048-...)
          ├── Time: 6935 us (占总时间 79%)
          ├── FLOP Rate: 7563 GFLOP/s
          └── Bound by: VMEM Read

Kernel 内部实际在做的:
  for expert in local_experts:
    load_weights_from_hbm()          # HBM → VMEM DMA
    for token_block in token_blocks:
      scatter_tokens()               # ★ 逐 token 标量循环 (Scalar ALU)
      matmul_gate()                  # MXU: tokens × gate_weights
      activation()                   # Vector ALU: SiLU
      matmul_up()                    # MXU: tokens × up_weights
      elementwise_mul()              # Vector ALU: gate * up
      matmul_down()                  # MXU: tokens × down_weights
      gather_and_accumulate()        # ★ 逐 token 标量循环 (Scalar ALU)
```

所以对 Pallas kernel 的分析必须使用**硬件计数器级别的指标**（场景 B）。

---

## 2. 场景 A: XLA 编译级 Trace 分析

### 2.1 适用场景

- 分析 JAX 程序的整体性能（哪个 kernel 最慢）
- 比较不同 kernel 实现的总时间
- 识别 kernel 外部的瓶颈（数据搬运、allgather、host-device 传输）
- 分析 pipeline 重叠效果

### 2.2 使用 roofline_model

```python
import json
from xprof.convert._pywrap_profiler_plugin import xspace_to_tools_data

def analyze_xla_trace(xplane_path):
    """XLA 编译级分析: 每个 op 的 roofline 特征"""

    raw, ok = xspace_to_tools_data([xplane_path], 'roofline_model')
    data = json.loads(raw)
    meta = data[0].get('p', {})
    cols = [c['label'] for c in data[0]['cols']]

    print(f"=== Device: {meta.get('device_type')} ===")
    print(f"Peak FLOP: {float(meta.get('peak_flop_rate', 0)):.0f} GFLOP/s")
    print(f"Peak HBM BW: {meta.get('peak_hbm_bw')} GiB/s")
    print(f"HBM Ridge Point: {meta.get('hbm_ridge_point')} FLOP/Byte")
    print()

    # 解析每个 op
    ops = []
    for row in data[0]['rows']:
        vals = {cols[j]: row['c'][j].get('v') for j in range(len(cols))}
        ops.append(vals)

    # 按 self time 排序
    ops_sorted = sorted(ops, key=lambda x: x.get('Total self time (us)', 0), reverse=True)

    print(f"{'Rank':<5} {'Operation':<60} {'Self Time(us)':>12} "
          f"{'FLOP Rate':>12} {'HBM BW':>10} {'OI':>8} {'Bound':>12} {'Eff%':>8}")
    print("-" * 130)

    for i, op in enumerate(ops_sorted[:20]):
        name = op.get('Operation', '')
        # 截断过长的名字
        if len(name) > 57:
            name = name[:57] + '...'
        print(f"{i+1:<5} {name:<60} "
              f"{op.get('Total self time (us)', 0):>12.1f} "
              f"{op.get('Measured FLOP Rate (GFLOP/s/core)', 0):>12.1f} "
              f"{op.get('HBM BW (GiB/s)', 0):>10.1f} "
              f"{op.get('Operational Intensity (FLOP/Byte)', 0):>8.1f} "
              f"{op.get('Bound by', ''):>12} "
              f"{op.get('Roofline Efficiency (%)', 0):>8.4f}")

    # 分类统计
    bound_summary = {}
    for op in ops:
        bound = op.get('Bound by', 'Unknown')
        time = op.get('Total self time (us)', 0)
        bound_summary[bound] = bound_summary.get(bound, 0) + time

    print(f"\n=== 瓶颈分布 ===")
    total = sum(bound_summary.values())
    for bound, time in sorted(bound_summary.items(), key=lambda x: -x[1]):
        pct = time / total * 100 if total > 0 else 0
        print(f"  {bound:<20}: {time:>10.1f} us ({pct:>5.1f}%)")

    return ops
```

### 2.3 使用 op_profile（树形结构）

```python
def analyze_op_tree(xplane_path):
    """分析 op profile 树，找到最耗时的 program 和 op"""

    raw, ok = xspace_to_tools_data([xplane_path], 'op_profile')
    data = json.loads(raw)

    def walk_tree(node, depth=0):
        name = node.get('name', '')
        metrics = node.get('metrics', {})
        raw_time = metrics.get('rawTime', 0)
        raw_flops = metrics.get('rawFlops', 0)
        bytes_arr = metrics.get('rawBytesAccessedArray', [])
        xla = node.get('xla', {})
        category = xla.get('category', '')

        if raw_time > 0:
            time_us = raw_time / 1e6  # ps → us
            print(f"{'  ' * depth}{name:<50} "
                  f"Time={time_us:>10.1f}us "
                  f"FLOP={raw_flops:>15} "
                  f"Cat={category}")

        for child in node.get('children', []):
            walk_tree(child, depth + 1)

    # 按 program 分析
    by_program = data.get('byProgramExcludeIdle', data.get('byProgram', {}))
    walk_tree(by_program)
```

### 2.4 使用 framework_op_stats

```python
def analyze_framework_ops(xplane_path):
    """框架级 op 统计：识别 JAX 层面的热点"""

    raw, ok = xspace_to_tools_data([xplane_path], 'framework_op_stats')
    data = json.loads(raw)
    cols = [c['label'] for c in data[0]['cols']]

    print(f"{'Rank':<5} {'Type':<20} {'Name':<40} {'Self Time(us)':>12} {'Device%':>10}")
    print("-" * 90)

    for row in data[0].get('rows', [])[:15]:
        vals = {cols[j]: row['c'][j].get('v') for j in range(len(cols))}
        print(f"{vals.get('Rank', ''):<5} "
              f"{vals.get('Operation Type', ''):<20} "
              f"{str(vals.get('Operation Name', ''))[:40]:<40} "
              f"{vals.get('Total self-time (us)', 0):>12.1f} "
              f"{vals.get('Total self-time on Device (%)', ''):>10}")
```

### 2.5 使用 trace_viewer（时间线分析）

```python
from xprof.convert.raw_to_tool_data import xspace_to_tool_data

def analyze_timeline(xplane_path):
    """分析时间线事件，计算 DMA/Compute 占比"""

    data, _ = xspace_to_tool_data([xplane_path], 'trace_viewer', {})
    trace = json.loads(data)
    events = trace.get('traceEvents', [])

    # 按设备分组
    device_events = {}
    for event in events:
        pid = event.get('pid', -1)
        if pid not in device_events:
            device_events[pid] = []
        device_events[pid].append(event)

    # 分析每个设备
    for pid, evts in device_events.items():
        compute_us = 0
        sync_wait_us = 0
        dma_us = 0

        for evt in evts:
            dur = evt.get('dur', 0)
            name = evt.get('name', '')

            if 'SyncWait' in name:
                sync_wait_us += dur
            elif 'DMA' in name or 'transfer' in name.lower():
                dma_us += dur
            elif dur > 0:
                compute_us += dur

        total = compute_us + sync_wait_us + dma_us
        if total > 0:
            print(f"Device {pid}:")
            print(f"  Compute: {compute_us:.1f}us ({compute_us/total*100:.1f}%)")
            print(f"  SyncWait: {sync_wait_us:.1f}us ({sync_wait_us/total*100:.1f}%)")
            print(f"  DMA: {dma_us:.1f}us ({dma_us/total*100:.1f}%)")
```

---

## 3. 场景 B: Pallas Kernel 内部分析

### 3.1 适用场景

- 分析 Pallas kernel 的内部瓶颈（MXU vs Scalar vs VMEM）
- 验证优化效果（R1-R8: Scalar ALU 是否减少）
- 判断 kernel 是 compute-bound 还是 memory-bound
- 分析通信（ICI）overhead

### 3.2 核心工具: utilization_viewer

这是**场景 B 唯一真正有用的工具**。它直接读取硬件性能计数器，
报告每个 TC core 的各硬件单元利用率。

```python
def analyze_kernel_utilization(xplane_path):
    """分析 Pallas kernel 的硬件单元利用率"""

    raw, ok = xspace_to_tools_data([xplane_path], 'utilization_viewer')
    if not ok:
        print("utilization_viewer 不可用")
        return

    data = json.loads(raw)
    cols = [c['label'] for c in data['cols']]

    # 按 (Device, Sample, Node) 分组
    # Node 对应不同的 TC core
    metrics_by_core = {}
    for row in data['rows']:
        vals = {cols[j]: row['c'][j].get('v') for j in range(len(cols))}
        device = vals.get('Device', 0)
        node = vals.get('Node', 0)
        sample = vals.get('Sample', 0)
        key = (device, sample, node)
        if key not in metrics_by_core:
            metrics_by_core[key] = {}
        metrics_by_core[key][vals['Name']] = {
            'achieved': vals['Achieved'],
            'peak': vals['Peak'],
            'unit': vals.get('Unit', '')
        }

    # 报告每个 core 的关键指标
    for (device, sample, node), metrics in sorted(metrics_by_core.items()):
        print(f"\n=== Device {device}, Sample {sample}, Core {node} ===")

        key_metrics = [
            'Scalar Unit', 'Vector ALUs', 'VMEM Loads', 'VMEM Stores',
            'Avg MXU Busy', '2 MXU Busy', '1 MXU Busy', 'No MXU Busy',
            'MXU BF16', 'MXU E4M3 + E5M2', 'MXU matpush',
            'Avg XLU Busy',
        ]

        for name in key_metrics:
            if name in metrics:
                m = metrics[name]
                achieved = m['achieved']
                peak = m['peak']
                pct = (achieved / peak * 100) if peak > 0 else 0
                bar = '█' * int(pct / 2) + '░' * (50 - int(pct / 2))
                print(f"  {name:<20}: {pct:>6.1f}% |{bar}| "
                      f"({achieved:.0f}/{peak:.0f} {m['unit']})")

        # HBM 带宽
        for name in metrics:
            if 'HBM' in name:
                m = metrics[name]
                if m['peak'] > 0:
                    pct = m['achieved'] / m['peak'] * 100
                    print(f"  {name:<20}: {pct:>6.1f}% "
                          f"({m['achieved']:.2e}/{m['peak']:.2e} {m['unit']})")

        # ICI 带宽
        for name in metrics:
            if 'ICI' in name:
                m = metrics[name]
                if m['peak'] > 0:
                    pct = m['achieved'] / m['peak'] * 100
                    print(f"  {name:<20}: {pct:>6.1f}% "
                          f"({m['achieved']:.2e}/{m['peak']:.2e} {m['unit']})")

    return metrics_by_core
```

### 3.3 瓶颈自动分类

```python
def classify_kernel_bottleneck(metrics_by_core):
    """根据硬件利用率自动分类 kernel 瓶颈"""

    results = []

    for (device, sample, node), metrics in metrics_by_core.items():
        def get_pct(name):
            if name not in metrics:
                return 0.0
            m = metrics[name]
            return (m['achieved'] / m['peak'] * 100) if m['peak'] > 0 else 0.0

        mxu_pct = get_pct('Avg MXU Busy')
        scalar_pct = get_pct('Scalar Unit')
        vector_pct = get_pct('Vector ALUs')
        vmem_ld_pct = get_pct('VMEM Loads')
        vmem_st_pct = get_pct('VMEM Stores')

        # 瓶颈判定逻辑
        bottleneck = "Unknown"
        explanation = ""

        if mxu_pct > 70:
            bottleneck = "MXU-bound (Optimal)"
            explanation = "MXU 利用率高，kernel 已接近计算瓶颈"
        elif scalar_pct > mxu_pct and scalar_pct > 15:
            bottleneck = "Scalar ALU-bound"
            explanation = (f"Scalar ({scalar_pct:.1f}%) > MXU ({mxu_pct:.1f}%)。"
                          f"标量循环阻塞 MXU，需要 bulk DMA 或减少标量操作")
        elif vector_pct > mxu_pct and vector_pct > 20:
            bottleneck = "Vector ALU-bound"
            explanation = (f"Vector ({vector_pct:.1f}%) > MXU ({mxu_pct:.1f}%)。"
                          f"向量运算过多，考虑 fusion 或减少激活函数计算")
        elif vmem_ld_pct > 60 or vmem_st_pct > 60:
            bottleneck = "VMEM-bound"
            explanation = (f"VMEM Load ({vmem_ld_pct:.1f}%) / Store ({vmem_st_pct:.1f}%)。"
                          f"内存带宽瓶颈，增大 block 或优化数据布局")
        elif mxu_pct < 10 and scalar_pct < 10:
            bottleneck = "Likely DMA/Communication-bound"
            explanation = "所有计算单元利用率都低，可能在等待 DMA 或通信"
        else:
            bottleneck = "Mixed/Balanced"
            explanation = f"MXU={mxu_pct:.1f}% Scalar={scalar_pct:.1f}% Vector={vector_pct:.1f}%"

        results.append({
            'device': device, 'sample': sample, 'node': node,
            'bottleneck': bottleneck, 'explanation': explanation,
            'mxu_pct': mxu_pct, 'scalar_pct': scalar_pct,
            'vector_pct': vector_pct, 'vmem_ld_pct': vmem_ld_pct,
        })

        print(f"\nCore ({device},{sample},{node}): **{bottleneck}**")
        print(f"  {explanation}")

    return results
```

### 3.4 Baseline vs Optimized 对比

```python
def compare_kernel_profiles(baseline_xplane, optimized_xplane):
    """对比两个 kernel profile 的硬件利用率"""

    print("=" * 80)
    print("Analyzing BASELINE...")
    baseline_metrics = analyze_kernel_utilization(baseline_xplane)

    print("\n" + "=" * 80)
    print("Analyzing OPTIMIZED...")
    optimized_metrics = analyze_kernel_utilization(optimized_xplane)

    print("\n" + "=" * 80)
    print("COMPARISON (first core)")
    print("=" * 80)

    # 取第一个 core 的数据对比
    b_key = sorted(baseline_metrics.keys())[0]
    o_key = sorted(optimized_metrics.keys())[0]
    b = baseline_metrics[b_key]
    o = optimized_metrics[o_key]

    key_metrics = [
        'Scalar Unit', 'Vector ALUs', 'Avg MXU Busy',
        '2 MXU Busy', 'No MXU Busy',
        'VMEM Loads', 'VMEM Stores',
        'MXU BF16', 'MXU E4M3 + E5M2',
        'Avg XLU Busy',
    ]

    print(f"\n{'Metric':<20} {'Baseline%':>10} {'Optimized%':>10} {'Delta':>10} {'Direction':>10}")
    print("-" * 62)

    for name in key_metrics:
        if name in b and name in o:
            b_pct = (b[name]['achieved'] / b[name]['peak'] * 100) if b[name]['peak'] > 0 else 0
            o_pct = (o[name]['achieved'] / o[name]['peak'] * 100) if o[name]['peak'] > 0 else 0
            delta = o_pct - b_pct
            direction = "✓ Better" if (
                (name in ('Avg MXU Busy', '2 MXU Busy', 'MXU BF16') and delta > 0) or
                (name in ('Scalar Unit', 'No MXU Busy') and delta < 0)
            ) else ("✗ Worse" if delta != 0 else "—")

            print(f"  {name:<20} {b_pct:>9.1f}% {o_pct:>9.1f}% {delta:>+9.1f}% {direction:>10}")

    # 瓶颈分类
    print("\n--- 瓶颈变化 ---")
    b_results = classify_kernel_bottleneck({b_key: b})
    o_results = classify_kernel_bottleneck({o_key: o})

    if b_results and o_results:
        print(f"  Baseline: {b_results[0]['bottleneck']}")
        print(f"  Optimized: {o_results[0]['bottleneck']}")
```

---

## 4. 组合方案: 完整的 Profiling 分析流水线

### 4.1 整体架构

```
                    Profile 采集
                        │
                        ▼
               .xplane.pb 文件
                   /         \
                  /           \
    ┌────────────┐     ┌────────────┐
    │ 场景 A:     │     │ 场景 B:     │
    │ XLA Trace  │     │ Kernel 内部 │
    │ 分析       │     │ 分析       │
    ├────────────┤     ├────────────┤
    │roofline    │     │utilization │
    │op_profile  │     │_viewer     │
    │hlo_stats   │     │            │
    │trace_viewer│     │perf_       │
    │framework_  │     │counters    │
    │op_stats    │     │            │
    └─────┬──────┘     └─────┬──────┘
          │                  │
          ▼                  ▼
    ┌────────────────────────────┐
    │    综合分析 & 瓶颈诊断       │
    │                            │
    │ 1. 整体: 哪个 kernel 最慢?   │
    │ 2. Kernel 级: bound by?     │
    │ 3. 硬件级: MXU vs Scalar?   │
    │ 4. 对比: 优化前后差异?       │
    └────────────────────────────┘
```

### 4.2 完整分析脚本

```python
#!/usr/bin/env python3
"""
Pallas Kernel Profiling Analyzer

用法:
  # 单个 profile 分析
  python -m benchmark.moe.analyze_profile ./profile_r8_current

  # 对比分析
  python -m benchmark.moe.analyze_profile --compare ./profile_baseline ./profile_r8_current

  # 仅 kernel 内部分析
  python -m benchmark.moe.analyze_profile --kernel-only ./profile_r8_current
"""

import argparse
import json
import glob
import sys
from xprof.convert._pywrap_profiler_plugin import xspace_to_tools_data


def find_xplane(profile_dir):
    """在 profile 目录中查找 .xplane.pb 文件"""
    patterns = [
        f"{profile_dir}/plugins/profile/*/*.xplane.pb",
        f"{profile_dir}/**/*.xplane.pb",
        profile_dir,  # 直接传入文件路径
    ]
    for pattern in patterns:
        files = glob.glob(pattern, recursive=True)
        if files:
            return files[0]
    return None


def get_tool_data(xplane, tool, options=None):
    """安全地获取工具数据"""
    try:
        raw, ok = xspace_to_tools_data([xplane], tool, options or {})
        if ok:
            return json.loads(raw)
    except Exception as e:
        print(f"  Warning: {tool} failed: {e}", file=sys.stderr)
    return None


def analyze_full(xplane_path):
    """完整分析（场景 A + B）"""
    results = {}

    # === 场景 A: XLA Trace 分析 ===
    print("\n" + "=" * 80)
    print("PART 1: XLA TRACE ANALYSIS (算子级)")
    print("=" * 80)

    # 1. Roofline
    roofline = get_tool_data(xplane_path, 'roofline_model')
    if roofline:
        meta = roofline[0].get('p', {})
        cols = [c['label'] for c in roofline[0]['cols']]
        ops = [{cols[j]: row['c'][j].get('v') for j in range(len(cols))}
               for row in roofline[0].get('rows', [])]
        ops_sorted = sorted(ops, key=lambda x: x.get('Total self time (us)', 0), reverse=True)

        print(f"\nDevice: {meta.get('device_type')}")
        print(f"Peak: {float(meta.get('peak_flop_rate', 0)):.0f} GFLOP/s, "
              f"HBM: {meta.get('peak_hbm_bw')} GiB/s")
        print(f"\nTop 10 Ops by Self Time:")
        print(f"  {'#':<3} {'Operation':<55} {'Time(us)':>10} {'FLOP/s':>10} {'Bound':>12}")
        for i, op in enumerate(ops_sorted[:10]):
            name = op.get('Operation', '')[:55]
            print(f"  {i+1:<3} {name:<55} "
                  f"{op.get('Total self time (us)', 0):>10.1f} "
                  f"{op.get('Measured FLOP Rate (GFLOP/s/core)', 0):>10.0f} "
                  f"{op.get('Bound by', ''):>12}")

        results['roofline_ops'] = ops_sorted
        results['roofline_meta'] = meta

    # 2. Overview
    overview = get_tool_data(xplane_path, 'overview_page')
    if overview and isinstance(overview, list) and overview:
        props = overview[0].get('p', {})
        print(f"\nOverview Metrics:")
        for key in ['mxu_utilization_percent', 'device_duty_cycle_percent',
                     'device_idle_time_percent', 'flop_rate_utilization_relative_to_roofline']:
            print(f"  {key}: {props.get(key, 'N/A')}")
        results['overview'] = props

    # === 场景 B: Kernel 内部分析 ===
    print("\n" + "=" * 80)
    print("PART 2: KERNEL INTERNAL ANALYSIS (硬件单元级)")
    print("=" * 80)

    utilization = get_tool_data(xplane_path, 'utilization_viewer')
    if utilization:
        cols = [c['label'] for c in utilization['cols']]
        metrics_by_core = {}
        for row in utilization['rows']:
            vals = {cols[j]: row['c'][j].get('v') for j in range(len(cols))}
            key = (vals.get('Device', 0), vals.get('Sample', 0), vals.get('Node', 0))
            if key not in metrics_by_core:
                metrics_by_core[key] = {}
            metrics_by_core[key][vals['Name']] = {
                'achieved': vals['Achieved'],
                'peak': vals['Peak'],
            }

        # 报告第一个 core 的关键指标
        first_key = sorted(metrics_by_core.keys())[0]
        m = metrics_by_core[first_key]

        print(f"\nCore {first_key}:")
        key_names = ['Scalar Unit', 'Vector ALUs', 'Avg MXU Busy',
                     '2 MXU Busy', '1 MXU Busy', 'No MXU Busy',
                     'VMEM Loads', 'VMEM Stores', 'Avg XLU Busy',
                     'MXU BF16', 'MXU E4M3 + E5M2', 'MXU matpush']

        for name in key_names:
            if name in m:
                achieved = m[name]['achieved']
                peak = m[name]['peak']
                pct = (achieved / peak * 100) if peak > 0 else 0
                bar_len = min(int(pct / 2), 40)
                bar = '█' * bar_len + '░' * (40 - bar_len)
                print(f"  {name:<20}: {pct:>6.1f}% |{bar}|")

        # 瓶颈分类
        def pct(name):
            if name not in m or m[name]['peak'] == 0:
                return 0
            return m[name]['achieved'] / m[name]['peak'] * 100

        mxu = pct('Avg MXU Busy')
        scalar = pct('Scalar Unit')
        vector = pct('Vector ALUs')

        print(f"\n  === 瓶颈诊断 ===")
        if mxu > 70:
            print(f"  ✓ MXU-bound ({mxu:.1f}%): Kernel 已接近计算极限")
        elif scalar > mxu and scalar > 15:
            print(f"  ✗ Scalar ALU-bound: Scalar={scalar:.1f}% > MXU={mxu:.1f}%")
            print(f"    → 建议: 减少标量循环, 用 bulk DMA 替代逐元素操作")
        elif vector > mxu and vector > 20:
            print(f"  ! Vector ALU-bound: Vector={vector:.1f}% > MXU={mxu:.1f}%")
            print(f"    → 建议: 优化向量运算, 考虑 fusion")
        elif mxu < 10 and scalar < 10:
            print(f"  ? DMA/Comm-bound: 所有单元利用率低")
            print(f"    → 建议: 检查 DMA 等待, 优化通信 overlap")
        else:
            print(f"  ~ Mixed: MXU={mxu:.1f}% Scalar={scalar:.1f}% Vector={vector:.1f}%")

        results['utilization'] = metrics_by_core

    return results


def compare(baseline_dir, optimized_dir):
    """对比两个 profile"""
    b_xplane = find_xplane(baseline_dir)
    o_xplane = find_xplane(optimized_dir)

    if not b_xplane or not o_xplane:
        print("Error: Cannot find xplane.pb files")
        return

    # 分别分析
    print("█" * 80)
    print("  BASELINE")
    print("█" * 80)
    b_results = analyze_full(b_xplane)

    print("\n" + "█" * 80)
    print("  OPTIMIZED")
    print("█" * 80)
    o_results = analyze_full(o_xplane)

    # 对比
    if 'utilization' in b_results and 'utilization' in o_results:
        print("\n" + "█" * 80)
        print("  COMPARISON")
        print("█" * 80)

        b_key = sorted(b_results['utilization'].keys())[0]
        o_key = sorted(o_results['utilization'].keys())[0]
        b_m = b_results['utilization'][b_key]
        o_m = o_results['utilization'][o_key]

        print(f"\n{'Metric':<20} {'Baseline':>10} {'Optimized':>10} {'Delta':>10} {'Verdict':>10}")
        print("-" * 62)

        for name in ['Scalar Unit', 'Vector ALUs', 'Avg MXU Busy',
                     '2 MXU Busy', 'No MXU Busy', 'VMEM Loads', 'VMEM Stores']:
            if name in b_m and name in o_m:
                b_pct = (b_m[name]['achieved'] / b_m[name]['peak'] * 100
                         if b_m[name]['peak'] > 0 else 0)
                o_pct = (o_m[name]['achieved'] / o_m[name]['peak'] * 100
                         if o_m[name]['peak'] > 0 else 0)
                delta = o_pct - b_pct
                # 判断好坏
                good = (name in ('Avg MXU Busy', '2 MXU Busy') and delta > 1) or \
                       (name in ('Scalar Unit', 'No MXU Busy') and delta < -1)
                bad = (name in ('Avg MXU Busy', '2 MXU Busy') and delta < -1) or \
                      (name in ('Scalar Unit', 'No MXU Busy') and delta > 1)
                verdict = "Better" if good else ("Worse" if bad else "—")
                print(f"  {name:<20} {b_pct:>9.1f}% {o_pct:>9.1f}% {delta:>+9.1f}% {verdict:>10}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pallas Kernel Profiling Analyzer')
    parser.add_argument('profile_dir', nargs='?', help='Profile directory to analyze')
    parser.add_argument('--compare', nargs=2, metavar=('BASELINE', 'OPTIMIZED'),
                       help='Compare two profiles')
    parser.add_argument('--kernel-only', action='store_true',
                       help='Only analyze kernel-level utilization (Scene B)')
    args = parser.parse_args()

    if args.compare:
        compare(args.compare[0], args.compare[1])
    elif args.profile_dir:
        xplane = find_xplane(args.profile_dir)
        if not xplane:
            print(f"Error: No .xplane.pb found in {args.profile_dir}")
            sys.exit(1)
        if args.kernel_only:
            # 只做场景 B
            utilization = get_tool_data(xplane, 'utilization_viewer')
            # ... (简化)
        else:
            analyze_full(xplane)
    else:
        parser.print_help()
```

### 4.3 集成到现有工作流

**在 bench_fused_moe_kernel.py 中添加 `--analyze` 参数**:

```bash
# 采集 + 自动分析
python -m benchmark.moe.bench_fused_moe_kernel \
    --profile --profile-dir ./profile_test \
    --analyze

# 对比 baseline
python -m benchmark.moe.bench_fused_moe_kernel \
    --profile --profile-dir ./profile_optimized \
    --compare-baseline ./profile_baseline
```

---

## 5. 实际验证: 分析现有 Profile 数据

本机已有多个 profile 目录，可以直接验证:

```bash
# 分析 R6 优化后的 profile
python analyze_profile.py ./profile_r6

# 分析 R8 (当前最优) 的 profile
python analyze_profile.py ./profile_r8_current

# 对比 R6 vs R8
python analyze_profile.py --compare ./profile_r6 ./profile_r8_current
```

### 5.1 已知 Profile 数据

| 目录 | 说明 | xplane 路径 |
|------|------|------------|
| `profile_r6/` | R6 优化后 | `plugins/profile/2026_03_25_22_58_58/tpu7x-*.xplane.pb` |
| `profile_r8_current/` | R8 (当前最优) | `plugins/profile/2026_03_25_23_40_59/tpu7x-*.xplane.pb` |
| `profile_bf16_8192h/` | BF16 大模型 | ... |
| `profile_moe_256e_64t/` | 256 expert | ... |
| `profile_ring1t_fp8_64e/` | FP8 模式 | ... |

### 5.2 预期分析结果

对于我们的 Fused MoE kernel 优化 (R1-R8):

**优化前 (baseline)**:
- Scalar Unit 利用率 **高**（逐 token scatter 的标量循环）
- MXU 利用率 **中低**（被 Scalar 阻塞）
- 预期: Scalar > MXU → **Scalar ALU-bound**

**优化后 (R8)**:
- Scalar Unit 利用率 **降低**（bulk DMA 替代标量循环）
- MXU 利用率 **提升**（Scalar 不再阻塞）
- 预期: MXU > Scalar → 接近 **MXU-bound** 或 **Balanced**

---

## 6. 进阶: xplane → SQLite 灵活查询

当标准工具不够用时，可以将 xplane.pb 加载到 SQLite 进行任意分析:

```python
import sqlite3
# 注意: 需要 tensorflow 的 xplane_pb2
# 或者可以用 xprof 的 protobuf (需要找到对应的 XSpace proto)

def xplane_to_sqlite(xplane_path):
    """将 xplane.pb 加载到 SQLite 供灵活查询"""
    from tensorflow.tsl.profiler.protobuf import xplane_pb2

    xspace = xplane_pb2.XSpace()
    with open(xplane_path, 'rb') as f:
        xspace.ParseFromString(f.read())

    conn = sqlite3.connect(':memory:')
    conn.execute('CREATE TABLE planes (id INTEGER, name TEXT)')
    conn.execute('''CREATE TABLE lines (id INTEGER, plane_id INTEGER,
                     display_id INTEGER, name TEXT, timestamp_ns INTEGER)''')
    conn.execute('''CREATE TABLE events (plane_id INTEGER, line_id INTEGER,
                     name TEXT, offset_ps INTEGER, duration_ps INTEGER,
                     start_ps INTEGER, end_ps INTEGER)''')

    for plane in xspace.planes:
        conn.execute('INSERT INTO planes VALUES (?,?)', (plane.id, plane.name))
        for line in plane.lines:
            conn.execute('INSERT INTO lines VALUES (?,?,?,?,?)',
                        (line.id, plane.id, line.display_id, line.name, line.timestamp_ns))
            for event in line.events:
                name = plane.event_metadata[event.metadata_id].name \
                    if event.metadata_id in plane.event_metadata else str(event.metadata_id)
                start_ps = event.offset_ps
                conn.execute('INSERT INTO events VALUES (?,?,?,?,?,?,?)',
                            (plane.id, line.id, name, event.offset_ps,
                             event.duration_ps, start_ps, start_ps + event.duration_ps))
    conn.commit()
    return conn

# 使用示例
conn = xplane_to_sqlite(xplane_path)

# Top 10 耗时事件
df = pd.read_sql("""
    SELECT name, SUM(duration_ps) as total_ps,
           COUNT(*) as count, AVG(duration_ps) as avg_ps
    FROM events
    GROUP BY name
    ORDER BY total_ps DESC
    LIMIT 10
""", conn)

# 特定 plane (设备) 的事件分布
df = pd.read_sql("""
    SELECT e.name, SUM(e.duration_ps) as total_ps
    FROM events e
    JOIN planes p ON e.plane_id = p.id
    WHERE p.name LIKE '%TPU%'
    GROUP BY e.name
    ORDER BY total_ps DESC
    LIMIT 20
""", conn)
```

---

## 7. 总结: 分析决策树

```
开始分析 Pallas Kernel Profile
│
├─ Q1: 该 kernel 在整体程序中是否是瓶颈?
│   → 使用场景 A: roofline_model / framework_op_stats
│   → 找到 pallas_call 的 self_time 占比
│   → 如果 < 50%: 先优化其他部分
│   → 如果 > 50%: 继续分析 kernel 内部
│
├─ Q2: Kernel 是 compute-bound 还是 memory-bound?
│   → 使用场景 A: roofline_model 的 bound_by
│   │   → "Compute": 已在计算极限
│   │   → "HBM"/"VMEM": 内存瓶颈
│   → 使用场景 B: utilization_viewer
│       → MXU > 70%: compute-bound
│       → VMEM > 60%: memory-bound
│
├─ Q3: 计算密集型 → MXU 还是 ALU 主导?
│   → 使用场景 B: utilization_viewer
│   → Scalar > MXU: 标量循环阻塞 → 需要 bulk DMA
│   → Vector > MXU: 向量运算过多 → 需要 fusion
│   → MXU 最高: 理想状态
│
├─ Q4: 优化是否有效?
│   → 使用 compare 模式对比 baseline vs optimized
│   → 关注: Scalar ↓, MXU ↑, Wall Time ↓
│
└─ Q5: 还能优化什么?
    → DMA Stall (%) 高: 优化 DMA pipeline
    → ICI 利用率高: 通信是瓶颈
    → No MXU Busy 周期多: MXU 有闲置，可以更好地 overlap
```
