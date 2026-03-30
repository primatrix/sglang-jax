# accelerator-agents 设计分析：Kernel Profiling 优化的借鉴

> **目标**: 分析 Google 的 `accelerator-agents`（MaxKernel）如何设计 profiling 驱动的
> kernel 优化流程，提取可借鉴的设计模式，应用到我们的 Fused MoE Pallas kernel 优化中。
>
> **仓库**: `/root/accelerator-agents/`

---

## 1. MaxKernel 整体架构

MaxKernel 是 Google 为 TPU Pallas kernel 开发的 AI Agent 系统，包含三套 Agent：

```
                    MaxKernel
                   /    |     \
     kernel_gen_agent   hitl_agent   open_optimization_agent
     (全自动流水线)     (人机协作)     (进化式优化, 500轮)
```

### 1.1 kernel_gen_agent — 全自动

```
OrganizeCode → ConvertToJax → JaxConversionLoop →
WriteBaseKernel → BaseKernelRefinementLoop →
AdjustInputShapes → AddKernelTiling → TiledKernelRefinementLoop →
GenerateTuningScript → KernelTilingOptimizer → Summary
```

每个阶段使用 `SequentialAgent`，refinement loop 使用 `LoopAgent`（最多 5 轮）。
在 tiling 优化阶段，自动生成 tuning 脚本并在 TPU 上执行。

### 1.2 hitl_kernel_gen_agent — 人机协作（与我们的工作流最相关）

```
KernelGenerationOrchestrationAgent (root, 带 ThinkingConfig)
  ├── ExplanationAgent            # 解释代码
  ├── PlanKernelAgent             # 制定优化计划
  ├── ImplementKernelAgent        # 实现修改
  ├── ValidateKernelCompilationAgent  # 编译验证（4次重试+自动debug注入）
  ├── ValidatedTestGenerationAgent    # 生成+验证测试
  ├── UnifiedTestAgent            # TPU上运行pytest
  ├── ProfileAgentOrchestrator    # ★ Profiling 分析（本文重点）
  │     ├── ReadFileForProfilingAgent
  │     ├── GenerateProfilingScriptAgent
  │     ├── ReadProfilingScriptAgent
  │     ├── ProfileEvalAgent (KernelProfiler)
  │     └── SummarizeProfileAgent (使用 offline xprof 工具)
  └── GpuToJaxAgent              # GPU kernel → JAX 转换
```

**核心设计**: "One Agent, Then Wait" — 每个子 agent 完成后，编排器返回控制给用户。
用户决定下一步。

### 1.3 open_optimization_agent — 进化式优化

```
NeverExitLoopAgent (最多 500 轮)
  ├── IdeaGenerationAgent (LoopAgent: Idea + Judge, 最多 5 轮)
  │     ├── IdeaAgent (温度 0.5, 创造性)
  │     └── JudgeAgent (温度 0.1, 严格评估)
  ├── KernelWriterAgent (温度 0.5)
  └── EvalAgent (Sequential):
        ├── EvalCompilationAgent
        ├── GenCorrectnessTestAgent
        ├── EvalCorrectnessAgent
        ├── GeneratePerformanceTestAgent
        ├── EvalPerformanceAgent
        └── SummarizeEvaluationAgent
```

**关键设计**: "Skip on Failure" — 如果编译失败，跳过后续所有步骤，直接进入下一轮。
```python
def whether_to_skip(callback_context, result_key=None):
    if result_key and callback_context.state.get(result_key) != "Success":
        callback_context.state["skip_iter"] = True
```

---

## 2. Profiling 分析系统深度分析

### 2.1 两层 Profiling 架构

```
Layer 1: 快速指标 (kernel_gen_agent)
  → trace_viewer → SyncWait ratio
  → 一个数字：DMA等待/总时间 比率

Layer 2: 深度分析 (hitl_agent)
  → xplane.pb → SQLite → 任意 SQL 查询
  → xprof overview_page → 高层指标
  → 可视化图表
  → LLM 总结 + 优化建议
```

### 2.2 Layer 1: SyncWait Ratio（快速诊断）

**文件**: `analyze_profile.py` — 仅 50 行代码

```python
def analyze_trace(path):
    # 1. 用 xprof API 获取 trace_viewer JSON
    tool_data_result, _ = raw_to_tool_data.xspace_to_tool_data(
        [path], "trace_viewer", {}
    )
    trace_data = json.loads(tool_data_result)
    events = trace_data.get("traceEvents", [])

    # 2. 过滤 TPU:0 设备事件
    for event in events:
        if event["args"].get("name") == "/device:TPU:0":
            pid = event["pid"]

    # 3. 找到最后一个 jit_computation 的时间范围
    start_last = jit_computation_events[-2]["ts"] + jit_computation_events[-2]["dur"]
    end_last = jit_computation_events[-1]["ts"] + jit_computation_events[-1]["dur"]

    # 4. 在该范围内累加 SyncWait 时长
    for event in events_for_tpu_0:
        if event["ts"] >= start_last and (event["ts"] + event["dur"]) <= end_last:
            if "SyncWait" in event["name"]:
                sync_wait_total += event["dur"]

    # 5. 计算比率
    ratio = sync_wait_total / total_computation_time
    # 输出: "kernel spends X% waiting for synchronization and Y% computing"
    return ratio
```

**设计启示**:
- **简洁明了**: 一个比率就能快速判断 "计算密集" vs "DMA 密集"
- **取最后一次迭代**: 避免 warmup 噪声
- **SyncWait 是关键信号**: 在 TPU trace 中，`SyncWait` 事件表示设备在等待 DMA/通信完成

**对我们的借鉴**:
我们的 Fused MoE kernel 优化目标就是减少 Scalar ALU 中逐 token scatter 的时间，
可以类似地计算 `scalar_time / total_time` 来量化优化效果。

### 2.3 Layer 2: xplane → SQLite 深度分析

**文件**: `offline_tools.py` — 最核心的分析工具

```python
def load_xplane_and_query(xplane_path, sql_query):
    """将 xplane.pb 加载到 SQLite 内存数据库，执行 SQL 查询"""

    # 1. 解析 protobuf
    xspace = xplane_pb2.XSpace()
    with open(xplane_path, 'rb') as f:
        xspace.ParseFromString(f.read())

    # 2. 创建 SQLite schema
    conn = sqlite3.connect(":memory:")
    conn.execute("""CREATE TABLE planes (id INTEGER, name TEXT)""")
    conn.execute("""CREATE TABLE lines (id INTEGER, plane_id INTEGER,
                     display_id INTEGER, name TEXT, timestamp_ns INTEGER)""")
    conn.execute("""CREATE TABLE events (plane_id INTEGER, line_id INTEGER,
                     name TEXT, offset_ps INTEGER, duration_ps INTEGER,
                     start_ps INTEGER, end_ps INTEGER)""")

    # 3. 填充数据
    for plane in xspace.planes:
        for line in plane.lines:
            for event in line.events:
                name = plane.event_metadata[event.metadata_id].name
                start_ps = event.offset_ps
                end_ps = start_ps + event.duration_ps
                conn.execute('INSERT INTO events VALUES (?,?,?,?,?,?,?)', ...)

    # 4. 执行查询
    df = pd.read_sql_query(sql_query, conn)
    return df.to_markdown()
```

**典型 SQL 查询**:

```sql
-- Top 10 最耗时操作
SELECT name, SUM(duration_ps) as total_ps, COUNT(*) as count
FROM events GROUP BY name ORDER BY total_ps DESC LIMIT 10;

-- 特定 plane 的事件分布
SELECT e.name, SUM(e.duration_ps) as total_ps
FROM events e
JOIN planes p ON e.plane_id = p.id
WHERE p.name LIKE '%TPU%'
GROUP BY e.name ORDER BY total_ps DESC;

-- 时间线片段
SELECT * FROM events
WHERE start_ps BETWEEN ? AND ?
ORDER BY start_ps;
```

**设计启示**:
- **SQL 的灵活性**: 比固定的 API 更灵活，可以做任意聚合、过滤、关联
- **LLM 可以动态生成 SQL**: 分析 agent 可以根据问题生成不同的 SQL
- **轻量级**: SQLite 内存数据库，加载一次后查询极快

**限制**:
- 使用 `tensorflow.tsl.profiler.protobuf.xplane_pb2`（TensorFlow 依赖）
- 不包含 `xprof` 级别的高级指标（如 roofline、utilization）
- 只能看到事件级别的 name/duration，没有 FLOP/bytes 统计

### 2.4 Overview Metrics 提取

```python
def get_overview_page_metrics(xplane_path):
    """使用 xprof API 提取高层指标"""
    raw_data, success = raw_to_tool_data.xspace_to_tool_data(
        [xplane_path], "overview_page", {}
    )
    data = json.loads(raw_data)

    # 解析 DataTable 提取关键属性
    return {
        'device_count': ...,
        'host_count': ...,
        'total_duration_ms': ...,
        'device_duty_cycle_percent': ...,
        'average_step_time_ms': ...,
        'step_count': ...,
    }
```

### 2.5 可视化图表生成

```python
def create_chart_from_xplane(xplane_path, sql_query, chart_type, x_col, y_col, title):
    """从 xplane 数据生成 matplotlib 图表"""
    # 1. 执行 SQL 获取数据
    # 2. 用 matplotlib 画图
    # 3. 保存为 PNG 文件
    # 支持: bar, line, pie
```

---

## 3. 优化流程与 Prompt 设计

### 3.1 核心优化工作流（来自 `pallas_profiling_docs.py`）

这是 MaxKernel 内置的 Pallas kernel 优化指南，非常有参考价值：

```
Step 1: 判断 Compute-bound vs Memory-bound
  → 方法 A: Roofline 分析 (operational_intensity vs ridge_point)
  → 方法 B: xprof utilization_viewer (MXU busy vs VMEM/HBM 利用率)

Step 2: 判断 MXU-bound vs ALU-bound
  → 如果 MXU busy < 预期，可能是 Scalar/Vector ALU 阻塞 MXU
  → 查看 Scalar Unit 利用率: 高 Scalar + 低 MXU = ALU-bound

Step 3: 针对瓶颈优化
  → Compute-bound: 降低精度 (BF16→FP8), 优化 tiling, 算法改进
  → Memory-bound: 增大 block size, 减少 HBM 访问, fusion
  → ALU-bound: 减少标量循环, 用 DMA bulk 替代逐元素操作
  → Pipeline bubble: 调整 pipeline 深度和 block 大小
```

**这正是我们优化 Fused MoE kernel 的逻辑**:
- R1-R8 优化的核心就是 "ALU-bound → 减少标量循环 → Bulk DMA"
- 通过 profiling 验证 Scalar Unit 利用率下降、MXU 利用率上升

### 3.2 分析 Agent 的 Prompt 设计

**Profiling 分析 prompt** (`analyze_profile_prompt.py`) 指导 LLM:

```python
PROMPT = """You are a TPU kernel optimization expert...

Available tools:
1. load_xplane_and_query(xplane_path, sql_query) - SQL over xplane data
2. get_hlo_dump(xplane_path) - HLO extraction
3. create_chart_from_xplane(...) - Visualization
4. get_overview_page_metrics(xplane_path) - High-level metrics

Analysis workflow:
1. Check DMAs_and_memory_transfers_ratio and compute_ratio
2. Use SQL to find top ops by sum(duration_ps)
3. Get overview metrics (duty cycle, step time)
4. Generate visualization charts
5. Use RAG to find relevant optimization guides
6. Provide actionable recommendations

Key questions to answer:
- Is the kernel compute-bound or memory-bound?
- What is the DMA stall fraction?
- Which operations dominate execution time?
- What specific optimizations would help?
"""
```

### 3.3 Profiling 脚本生成模板

```python
PROMPT = """Generate a profiling script that:
1. Sets up JAX profiler with proper options
2. Runs the kernel for 3 iterations
3. Uses TRACE_COMPUTE_AND_SYNC mode

Template:
    options = jax.profiler.ProfileOptions()
    options.python_tracer_level = 0
    options.host_tracer_level = 2
    options.advanced_configuration = {"tpu_trace_mode": "TRACE_COMPUTE_AND_SYNC"}

    jax.profiler.start_trace('jax_trace', profiler_options=options)
    for i in range(3):
        C = jax.block_until_ready(computation(A, B))
    jax.profiler.stop_trace()
"""
```

---

## 4. 可借鉴的关键设计模式

### 4.1 三级指标体系

```
Level 0: Wall Time (benchmark)
  → 最终判断标准：kernel 是否变快了
  → 我们已有: bench_fused_moe_kernel.py

Level 1: 快速诊断指标 (SyncWait ratio)
  → 单个数字快速判断瓶颈类型
  → 可借鉴: 计算 Scalar/MXU/DMA 占比

Level 2: 深度分析 (xprof 全套工具)
  → Roofline, utilization, per-op breakdown
  → 用于定位具体哪个 op/阶段是瓶颈
```

### 4.2 Skip-on-Failure 快速迭代

```python
# 编译失败 → 跳过正确性测试 → 跳过性能测试 → 下一轮
if compilation != "Success":
    skip_iter = True
```

避免在失败分支上浪费时间。适用于我们的自动化优化循环。

### 4.3 Idea → Judge → Implement → Eval 循环

```
IdeaAgent (创造性, temp=0.5)
  ↓ 生成优化方案
JudgeAgent (严格, temp=0.1)
  ↓ 评估可行性，reject 不合理方案
KernelWriterAgent (创造性, temp=0.5)
  ↓ 实现代码
EvalAgent (Sequential)
  ↓ 编译 → 正确性 → 性能
SummarizeAgent
  ↓ 汇总结果，记录 "historical best"
  → 循环
```

**关键**: Judge 可以拒绝方案，避免浪费 TPU 时间在明显不合理的想法上。

### 4.4 State 传递模式

```python
# 所有 agent 通过 session.state 共享数据
callback_context.state["kernel_code"] = code
callback_context.state["compilation_results"] = "Success"
callback_context.state["performance_test_results"] = timing_data
```

输入/输出通过 `input_key`/`output_key` 解耦，每个 agent 只关心自己需要的数据。

### 4.5 TPU 规格数据库

```json
{
  "TPU 7x": {
    "generation": "Ironwood",
    "specs": {
      "performance": {"peak_bf16_tflops": 2157, "peak_int8_tops": 4314},
      "memory": {"type": "HBM", "capacity_gib": 192, "bandwidth_tb_per_s": 7.4},
      "interconnect": {"topology": "3D Torus"},
      "architecture": {"max_pod_size_chips": 9216}
    }
  }
}
```

每个 agent 在启动时通过 callback 注入 TPU 规格，确保优化建议符合硬件特性。

### 4.6 Performance 阈值

```python
PERF_THRESHOLD = 1.1  # 性能必须提升 10% 才算 "成功"
```

防止微小波动被误判为"优化"。

---

## 5. 对我们工作的具体借鉴

### 5.1 当前差距

| 能力 | 我们已有 | MaxKernel 有 | 差距 |
|------|---------|-------------|------|
| Wall time benchmark | ✅ bench_fused_moe_kernel | ✅ | - |
| 精度测试 | ✅ fused_moe_v1_test | ✅ | - |
| Profiling 采集 | ✅ --profile 参数 | ✅ | - |
| Profile 快速指标 | ❌ | ✅ SyncWait ratio | **需要** |
| Profile 深度分析 | ❌ 手动看 xprof UI | ✅ SQL + xprof API | **需要** |
| 瓶颈自动分类 | ❌ | ✅ roofline + utilization | **需要** |
| 对比分析 | ❌ | ✅ baseline vs optimized | **需要** |
| 可视化 | ❌ | ✅ matplotlib 图表 | 可选 |

### 5.2 建议实现的工具

**工具 1: 快速诊断脚本**
```bash
python -m benchmark.moe.analyze_profile ./profile_r8_current
# 输出:
#   MXU Busy: 45.2%
#   Scalar Unit: 23.1%
#   Vector ALU: 12.5%
#   DMA Stall: 8.3%
#   瓶颈类型: ALU-bound (高 Scalar + 中等 MXU)
```

**工具 2: 对比分析脚本**
```bash
python -m benchmark.moe.compare_profiles ./profile_baseline ./profile_optimized
# 输出:
#   MXU Busy: 35.2% → 45.2% (Δ+10.0%)
#   Scalar Unit: 38.1% → 23.1% (Δ-15.0%)
#   Wall Time: 0.310ms → 0.286ms (Δ-7.7%)
#   结论: Scalar ALU 减少 15%，MXU 利用率对应提升 10%
```

**工具 3: Roofline 分析**
```bash
python -m benchmark.moe.roofline_analysis ./profile_r8_current
# 输出:
#   Top Ops by Time:
#     custom-call (fused-moe): 6935us, 7563 GFLOP/s, Bound=VMEM Read
#     async-done: 834us
#   Peak Utilization:
#     FLOP Rate: 7563/1028750 = 0.74%
#     HBM BW: 18.6/3433 = 0.54%
#   Ridge Point: 279 FLOP/Byte
#   Operational Intensity: 94.4 FLOP/Byte → Below HBM ridge → Memory-bound
```

### 5.3 集成到优化工作流

```
修改 kernel 代码
    ↓
运行精度测试 (快速验证不回退)
    ↓
运行 benchmark (wall time)
    ↓
运行 profiling + 快速诊断
    ↓ 自动判断
├── Scalar-bound: 需要减少标量循环/用 bulk DMA
├── MXU-bound: 已达最优，考虑其他方向
├── HBM-bound: 需要减少 HBM 访问/增大 block
├── VMEM-bound: 需要优化内存访问模式
└── DMA-bound: 需要优化通信/overlap
    ↓
对比 baseline profile → 量化改进点
```

---

## 6. 技术细节补充

### 6.1 LLM 配置

| 参数 | 值 | 用途 |
|------|-----|------|
| Model | gemini-3-pro-preview | 所有 agent |
| Temperature (创造性) | 0.5 | Idea/Writer agent |
| Temperature (严格) | 0.1 | Judge/Test/Summary agent |
| Top-P | 0.9 | 全局 |
| Top-K | 5 | 全局 |
| Max Iterations | 5 | refinement loop |
| TPU Timeout | 120s | 单次 TPU 执行 |
| Request Timeout | 1800s | 总请求超时 |

### 6.2 三层服务器架构

```
Agent (ADK, Gemini)
  ↓ HTTP POST (code string)
Eval Server (:1245)
  ↓ HTTP POST (路由到对应 worker)
TPU Server (:5463)  /  CPU Server (:5464)
  ↓ subprocess 执行代码
  ↓ 返回结果 JSON
```

Eval Server 充当负载均衡/路由器，通过 `eval_config.yaml` 配置。
每次执行都在独立的 subprocess 沙箱中运行。

### 6.3 RAG 支持

- **Vertex AI RAG**: 用于检索 Pallas/JAX/TPU 文档
- **BigQuery 向量搜索**: 使用 UniXcoder embeddings，从类似 kernel 数据库中检索参考实现
- **API Search Tool**: 通过 `importlib` 动态解析 JAX API 文档

这些在优化建议生成阶段提供上下文支持，帮助 LLM 给出更准确的建议。
