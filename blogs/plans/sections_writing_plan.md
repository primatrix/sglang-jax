# Ring-1T 博客文章 — 4个章节写作计划

## 写作风格指南（基于 lmsys 博客分析）

参考文章：
- [SGLang-Jax: Native TPU Inference](https://lmsys.org/blog/2025-10-29-sglang-jax/)
- [GB300 NVL72 25x Performance](https://lmsys.org/blog/2026-02-20-gb300-inferencex/)
- [Unified FP8 for MoE RL](https://lmsys.org/blog/2025-11-25-unified-fp8/)

**核心风格要素：**
1. 每个优化项结构：问题/动机 → 方案 → 技术细节 → 收益/数据
2. 段落短小精悍，每段一个核心观点
3. 粗体标注关键数字和术语
4. 技术深度适中：解释 WHY 和 WHAT，不深入代码细节
5. 表格呈现 benchmark 数据
6. 图片引用（heatmap、pipeline diagram、profiling）用 `<p align="center"><img src="..." width="80%"></p>` 格式，图片路径留占位符
7. 语言：中文

---

## 章节 1: All Reduce Metadata 优化（Fused MoE 子节）

**所属位置**：Methods → 2. Fused MoE Kernel Optimization → 优化项之一
**预计篇幅**：300-400 字
**分配给**：Agent 1

### 大纲

#### 1.1 问题
- Fused MoE kernel 在 FFN 计算前需要通过 allgather 同步所有设备的 expert metadata（每个 expert 接收的 token 数量和起始偏移）
- tpu-inference 原始实现使用 **Ring allgather**，复杂度 O(N)
- EP=32 时需要 **31 轮**通信；EP=64 时需要 **63 轮**
- 每轮通信有固定的 barrier 同步开销，轮数多导致累积延迟显著

#### 1.2 方案：Recursive-Doubling Allgather
- 改用 **recursive-doubling** 算法，复杂度 O(log₂(N))
- EP=32 时仅需 **5 轮**（vs 31 轮），EP=64 时仅需 **6 轮**（vs 63 轮）
- 原理简述：每轮通过 XOR 选择 peer 设备交换数据，每轮数据量翻倍，log₂(N) 轮后所有设备拥有完整数据
- 非 power-of-2 设备数时回退到 ring 方案保证正确性

#### 1.3 收益
- 消融实验数据：metadata allgather 耗时约 **0.053ms**（128 tokens, EP=32）
- 绝对值不大，但对更大 EP 规模收益显著
- 减少了路由阶段（~98us 中的一部分）的 barrier 同步开销

**引用数据**：`blogs/docs/FusedMoE_benchmark.md` 最后一张消融表中的 `enable all_reduce metadata` 列

---

## 章节 2: Shared Expert 融合（Fused MoE 子节）

**所属位置**：Methods → 2. Fused MoE Kernel Optimization → 优化项之一
**预计篇幅**：600-800 字
**分配给**：Agent 1

### 大纲

#### 2.1 背景
- Ring-1T 模型有 **1 个 shared expert**（所有 token 共享的 dense FFN）+ 256 个 routed experts
- Shared expert 的算术强度 AI=1056（BS=512 时），远超临界值 312.6，是 **compute-bound** 的
- 如果 shared expert 在 routed expert 之后串行执行，会增加 MoE 层总延迟

#### 2.2 方案：穿插在通信间隙中
- 将 shared expert 的 FFN 计算切分为多个 block（按 `bse` 参数控制，如 bse=256 → 8 个 block）
- 将这些 block **穿插在 routed expert 流水线的通信等待间隙中**执行

**Pipeline 示意图**（文字版，正式文章配图）：
```
Expert 0:
  SE block 0        ← scatter 启动前
  SE block 1        ← scatter 启动后、等待前
  wait_scatter(E0)
  expert_ffn(E0)    ← MXU 密集计算
  start_gather(E0)
  SE block 2        ← gather 启动后

Expert 1:
  SE block 3
  ...
```

- 核心思想：routed expert 的间隙是 **bandwidth-bound**（A2A 通信 + barrier 等待），MXU 大部分空闲。而 SE 是 **compute-bound** 的，恰好填充闲置的 MXU 算力

#### 2.3 关键优化细节
- **独立的双缓冲权重**：SE 使用独立的 `b_se_w1_x2_vmem`、`b_se_w3_x2_vmem`、`b_se_w2_x2_vmem`，不与 routed expert 权重缓冲竞争
- **SE hidden_size 切分 + 重排 fetch 位置**：优化 SE 权重和 token 的预取位置，减少内存访问冲突
- **SE 预取优化**：FFN2 期间预取下一 block 的 W1/W3，进一步隐藏加载延迟
- **SE 计算插入 wait_a2a_scatter_recv 间隙**：利用 scatter 接收等待期间（~5-6us）的空闲 MXU

#### 2.4 消融实验数据

| 优化阶段 | all enable (hotspot-count=64, 128 tokens) |
|---------|----------------------------------------|
| 基线（无 SE 融合） | SE 串行，总耗时更高 |
| + SE 融合 | 0.261ms（独立 SE 开销） |
| + SE hidden_size 切分 + 重排 fetch | 0.282ms → 降低 |
| + SE 预取 + acc tile 预取 | 0.272ms |
| + FFN2→FFN1 权重预取 + SE 插入 wait 间隙 | 0.288ms |

（注：SE 开销数字代表 enable shared_expert 列的增量贡献，实际 SE 计算被有效隐藏在通信间隙中）

#### 2.5 理论分析
- 如果 SE 计算完全隐藏在通信间隙中 → MoE 层总耗时 = routed expert 耗时（**零额外 SE 开销**）
- 实际 bse=256 时有 8 个 SE block，可以在 8 experts × 2 + 1 = 17 个间隙中执行完，绑定关系宽松

---

## 章节 3: EPLB（Expert Parallelism Load Balancing）

**所属位置**：Methods → 3. EPLB
**预计篇幅**：600-800 字
**分配给**：Agent 2
**参考**：`blogs/docs/EPLB.md`、PR #818

### 大纲

#### 3.1 问题：Expert 负载不均衡
- Ring-1T 256 个 experts，EP=32 时每个 core 处理 8 个 experts
- 实测发现：
  - **平均每层只有个别 experts 是热点**，token 数差异可达 **10 倍**
  - **Device 视角的不均衡度是平均值的 2-2.5 倍**
- 不均衡直接影响 fused MoE kernel 性能：
  1. 随 decode 序列生成变长，kernel 性能劣化
  2. 同一次迭代前几层 kernel 性能逊于后面的层（2ms vs 1.1ms）
  3. `sync_barrier()` 时快设备等待慢设备，浪费算力

**配图**：EPLB 前的 expert/device 负载 heatmap（占位符）

#### 3.2 方案：DeepSeek EPLB 冗余专家放置
- 采用 DeepSeek 开源的 EPLB 算法
- **核心思想**：在原始 256 个 experts 基础上，增加 32 个**冗余 expert 副本**，总计 288 个 experts
- EPLB 接受 expert 分布统计作为输入，计算最优的 expert 放置方案，使各设备负载尽可能均衡
- 冗余 experts 被放置在负载较低的设备上，分担热点 expert 的 token

#### 3.3 实现
- **分布采集**：服务运行时自动记录 expert routing 分布，写入 `.npy` 文件
  - 参数：`--enable-expert-distribution-recorder --expert-distribution-recorder-output-file expert_dist.npy`
- **Expert 放置计算**：基于采集的分布数据，调用 EPLB 算法计算最优放置
  - 参数：`--init-expert-location expert_dist.npy`
- **动态 dispatch**：运行时根据放置方案动态路由 token 到正确的 expert 副本
  - 参数：`--ep-num-redundant-experts 32 --ep-dispatch-algorith dynamic`
- 精度验证：GSM8k 准确率 **0.9371**，确认冗余放置不影响模型精度

#### 3.4 效果

**配图**：EPLB 后的 expert/device 负载 heatmap（占位符）

- EPLB 后设备间负载更均衡，热点 expert 的 token 被分散
- 减少了 `sync_barrier()` 的等待时间，改善了 kernel 间隙中最不可预测的开销来源
- 对整体 decode 吞吐有显著提升（配合其他优化的端到端数据在 Performance 章节展示）

#### 3.5 Reference
- DeepSeek EPLB: https://github.com/deepseek-ai/EPLB
- SGLang GPU EPLB: sgl-project/sglang#5295

---

## 章节 4: DP Attention

**所属位置**：Methods → 4. DP Attention
**预计篇幅**：600-800 字
**分配给**：Agent 3
**参考**：用户提供的设计文档、PR #761、`blogs/docs/Ring1t-sync.md`

### 大纲

#### 4.1 动机
- Ring-1T 只有 **64 个 KV heads**（GQA），当 TP=32 时每个 shard 只分到 2 个 KV heads
- 纯 TP 模式下：
  - 每个 TP shard **复制相同的 KV heads** → KV cache 冗余存储
  - Attention 计算量被过度切分 → 每个设备只处理极少量 head，效率低
- 引入 DP 后：KV heads 按 DP 分片而非按 TP 复制，**节省内存，提高利用率**

#### 4.2 架构设计
- **2D Mesh**：`('data', 'tensor')`
  - Attention 层：DP=4 × TP=8，32 个 cores
  - MoE 层：EP=32，同样 32 个 cores
- **单控制器设计**（区别于 SGLang PyTorch 的多控制器架构）
  - 一个 Scheduler 管理所有 DP ranks 的请求调度
  - Prefix cache 和 KV allocator 按 `dp_id` 分区
  - 调度策略：shortest-queue / round-robin

**配图**：DP Attention + EP MoE 的并行策略示意图（占位符）

#### 4.3 关键实现

**KV Cache 分片：**
- KV cache shape: `(num_pages, page_size, num_kv_heads, head_dim)`
- Sharding: `P('data', None, 'tensor', None)` — pages 按 DP 分片，heads 按 TP 分片
- 每个 DP rank 独立管理 KV cache，互不干扰

**Attention Kernel 兼容：**
- Flash attention kernel 通过 `shard_map` 包装，每个 DP rank 看到从 index 0 开始的局部数据
- 无需内核内部通信，只需正确设置 `in_specs` 和 `out_specs`

**调度感知 DP：**
- 请求被分配 `dp_id`，元数据按 DP 重排后传入 model forward
- 输出结果按原始请求顺序重排返回

#### 4.4 收益
- **内存节省**：KV cache 不再跨 TP 复制，每设备可存储更多 token → batch size 增大
- **通信量降低**：DP 使每设备 local tokens 减少 → A2A scatter/gather 数据量下降
- **ITL 改善**：

| 配置 | batch_size=64 ITL | batch_size=256 ITL |
|------|------------------|-------------------|
| 纯 TP (TP=32) | 106.80ms | 123.71ms |
| DP=4 + TP=8 | **91.32ms** | **115.11ms** |

- **吞吐提升**：DP=4 配置下 output throughput 从 599 tok/s (bs=64) 提升至 700 tok/s

---

## 团队分配

| Agent | 负责章节 | 输出文件 |
|-------|---------|---------|
| Agent 1 | All Reduce Metadata 优化 + Shared Expert 融合 | `blogs/drafts/fused_moe_optimizations.md` |
| Agent 2 | EPLB | `blogs/drafts/eplb.md` |
| Agent 3 | DP Attention | `blogs/drafts/dp_attention.md` |

**Agent 3 需要**：checkout `epic/data-parallelism-rebase` 分支读取代码（PR #761）

---

## 待确认事项

1. **图片**：以上标注了配图占位符位置，实际图片是否需要 agent 生成描述文字，还是你后续提供？
2. **数据精确性**：消融实验数据直接引用 `FusedMoE_benchmark.md`，是否有更新的数据？
3. **EPLB 端到端数据**：EPLB 章节目前缺少 EPLB 前后端到端吞吐对比数据，是否在 Performance 章节统一呈现？
4. **DP Attention benchmark**：使用 `Ring1t-sync.md` 中的 bench_one_batch 数据，是否有更新的数据？
