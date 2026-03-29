---
title: "Serving Ring-1T on TPU v7: Fused MoE Kernel, FP8 Quantization, and Expert Load Balancing with SGLang-JAX"
author: "SGLang-JAX Team"
date: "2026-XX-XX"
previewImg: "/images/blog/ring_1t/cover.png"
---

## TL;DR

<!-- TODO: blockquote summary — teams, techniques, headline numbers -->

## Background

<!-- TODO: Ring-1T model intro, TPU inference challenges -->

## Methods

### 1. System Overview

<!-- TODO: SGLang-JAX architecture on TPU, parallelism config -->

### 2. Fused MoE Kernel Optimization

<!-- TODO: original kernel analysis -->

<!-- TODO: roofline analysis -->

#### All Reduce Metadata 优化

Fused MoE kernel 在执行 FFN 计算之前，需要通过 allgather 操作同步所有设备上的 expert metadata——包括每个 expert 接收的 token 数量和起始偏移。这些元数据是后续 scatter/gather 通信和 FFN 计算的前置依赖，必须在路由阶段完成。

tpu-inference 的原始实现使用 **ring allgather** 算法，时间复杂度为 **O(N)**。这意味着 EP=32 时需要 **31 轮**点对点通信，EP=64 时更是需要 **63 轮**。每轮通信都伴随固定的 barrier 同步开销，轮数越多，累积的同步延迟越显著。

我们将其替换为 **recursive-doubling allgather** 算法，将复杂度降至 **O(log2(N))**。该算法的核心思路是：每轮通过 XOR 运算选择 peer 设备交换数据，每轮交换的数据量翻倍，仅需 log2(N) 轮即可让所有设备拥有完整数据。

这一改动带来了显著的轮次缩减：

| EP 规模 | Ring Allgather | Recursive-Doubling | 轮次缩减 |
|---------|---------------|-------------------|----------|
| EP=32 | 31 轮 | **5 轮** | **6.2x** |
| EP=64 | 63 轮 | **6 轮** | **10.5x** |

对于非 2 的幂次的设备数，我们回退到 ring 方案以保证正确性。

消融实验数据显示，在 NumTokens=128、EP=32 的配置下，metadata allgather 的耗时约为 **0.053ms**，在各种负载分布下保持稳定：

| 负载分布 | all enable | all disabled | enable all_reduce metadata | enable topk | enable sync_barrier |
|---------|-----------|-------------|--------------------------|------------|-------------------|
| hotspot-count 8 | 2.756 | 0.223 | 0.053 | 0.006 | 0.039 |
| hotspot-count 64 | 0.895 | 0.224 | 0.053 | 0.006 | 0.039 |
| hotspot-count 256 | 0.615 | 0.223 | 0.053 | 0.006 | 0.039 |

虽然 **0.053ms** 的绝对值不算大，但这一优化减少了路由阶段的 barrier 同步次数，对更大 EP 规模（如 EP=64）的收益更为显著。更重要的是，它为后续扩展到更多设备的部署奠定了基础。

#### Shared Expert 融合

##### 问题：串行执行的额外延迟

Ring-1T 模型采用混合专家架构，包含 **1 个 shared expert**（所有 token 共享的 dense FFN）和 **256 个 routed experts**。Shared expert 的算术强度（Arithmetic Intensity）在 BS=512 时为 **AI=1056**，远超 TPU v7 的临界值 312.6，属于典型的 **compute-bound** 负载。

如果 shared expert 在所有 routed expert 之后串行执行，它将直接叠加在 MoE 层的总延迟上，造成不必要的性能损失。

##### 方案：穿插在通信间隙中

我们的核心观察是：routed expert 的流水线中存在大量 **MXU 空闲间隙**。profiling 数据显示，每个 expert 的 FFN 计算仅耗时约 **8us**，但 expert 之间的间隙长达 **35-70us**。这些间隙主要被标量控制流（scatter 路由循环，占 **50-60%**）、sync_barrier 同步（**15-25%**）和 ICI 通信（**10-15%**）占据，MXU 有效利用率仅约 **14%**。

Shared expert（SE）是 compute-bound 的，恰好可以填充这些间隙中闲置的 MXU 算力。具体做法是：将 SE 的 FFN 计算按 `bse` 参数切分为多个 block（如 bse=256 时切分为 **8 个 block**），然后将这些 block **穿插在 routed expert 流水线的通信等待间隙中**执行。

<p align="center"><img src="TODO_shared_expert_pipeline_diagram.png" width="80%"></p>

Pipeline 执行流程如下：

```
Expert 0:
  SE block 0         <-- scatter 启动前
  SE block 1         <-- scatter 启动后、等待前
  wait_scatter(E0)
  expert_ffn(E0)     <-- MXU 密集计算
  start_gather(E0)
  SE block 2         <-- gather 启动后

Expert 1:
  SE block 3
  ...
```

核心思想可以简洁地概括为：routed expert 的间隙是 **bandwidth-bound** 的（A2A 通信 + barrier 等待），MXU 大部分空闲；而 SE 是 **compute-bound** 的，两者在硬件资源需求上正好互补。

##### 关键优化细节

为了让 SE block 高效地穿插执行，我们进行了四项渐进式优化：

**独立的双缓冲权重。** SE 使用独立的权重缓冲区，不与 routed expert 的权重缓冲竞争 VMEM 带宽。这确保了 SE 的权重加载不会干扰 routed expert 的权重预取流水线。

**SE hidden_size 切分 + 重排 fetch 位置。** 将 SE 的 intermediate dimension 按 `bse` 粒度切分，并优化权重和 token 的预取位置，减少 SE 与 routed expert 之间的内存访问冲突。

**SE 预取优化。** 在 FFN2 计算期间预取下一个 SE block 的 W1/W3 权重，进一步隐藏加载延迟，使 SE block 的执行更加流畅。

**SE 计算插入 wait_a2a_scatter_recv 间隙。** 利用 scatter 接收等待期间（约 **5-6us**）的空闲 MXU，额外插入 SE 计算，充分榨取每一微秒的闲置算力。

##### 消融实验

<!-- TODO: 补充四个优化阶段的完整消融对比表（包含所有列），展示各阶段各组件的耗时变化趋势 -->

最终优化版本的完整消融数据（NumTokens=128）：

| 负载分布 | all enable | all disabled | enable a2a | enable dynamic_ffn1 | enable dynamic_ffn2 | enable weight_load | enable a2a_s_tile_read | enable a2a_s_acc_tile_write | enable shared_expert |
|---------|-----------|-------------|-----------|-------------------|-------------------|------------------|---------------------|--------------------------|-------------------|
| hotspot-count 8 | 2.756 | 0.223 | 0.295 | 1.689 | 1.004 | 0.467 | 0.479 | 0.353 | 0.288 |
| hotspot-count 64 | 0.895 | 0.224 | 0.257 | 0.516 | 0.378 | 0.466 | 0.273 | 0.258 | 0.288 |
| hotspot-count 256 | 0.615 | 0.223 | 0.259 | 0.320 | 0.273 | 0.466 | 0.249 | 0.243 | 0.288 |

> **注**：`enable shared_expert` 列的数值代表打开 SE 特性后的增量贡献。由于 SE 计算被有效隐藏在通信间隙中，这个增量对端到端延迟的影响远小于串行执行的情况。

##### 理论分析

如果 SE 计算完全隐藏在通信间隙中，MoE 层的总耗时将等于纯 routed expert 的耗时——即实现**零额外 SE 开销**。在 bse=256 的配置下，SE 被切分为 **8 个 block**，而 8 个 expert 的流水线提供了 **8 x 2 + 1 = 17 个**可用插入间隙，绑定关系相当宽松，SE 有充足的空间完成计算。

这一优化的本质是**资源互补调度**：利用通信密集阶段的计算空闲，执行计算密集的 shared expert，从而在不增加总延迟的前提下完成额外的计算工作。

<!-- TODO: ablation study with benchmark table -->

### 3. EPLB (Expert Parallelism Load Balancing)

#### 问题：Expert 负载不均衡

Ring-1T 模型包含 **256 个 routed experts**。在 EP=32 的部署配置下，每个 TPU core 承载 **8 个 experts**。理想情况下，所有 experts 接收的 token 数量应大致均匀，但实际分布远非如此。

我们在真实请求负载下进行了 expert 均衡度分析，发现了两个关键现象：

- **Expert 级别的热点效应**：平均每层只有个别 experts 成为热点，最热 expert 接收的 token 数可达均值的 **10 倍**以上。
- **Device 级别的不均衡放大**：由于每个 device 承载多个 experts，单个热点 expert 会拉高整个 device 的负载。Device 视角的不均衡度约为 expert 均值的 **2-2.5 倍**。

<p align="center"><img src="TODO_eplb_before_expert_heatmap.png" width="80%"></p>

上图展示了 EPLB 前 Layer 78 的 expert 负载 heatmap。大部分 expert 的 token 数接近均值（绿色区域），但存在明显的热点（黄色高亮），最热 expert 的 token 数约为均值的 **11.6 倍**。

这种不均衡对 fused MoE kernel 性能产生了直接影响：

1. **随 decode 序列变长，kernel 性能持续劣化**——越来越多的 token 被路由到热点 expert，加剧不均衡。
2. **同一次迭代内，前几层 kernel 耗时显著高于后续层**（**2ms** vs **1.1ms**），因为前几层的 expert 分布更不均匀。
3. **`sync_barrier()` 造成木桶效应**——每一轮 expert FFN 计算结束后，所有设备需要同步等待最慢的设备完成。负载最重的 device 决定了整体耗时，其他设备只能空等，浪费算力。

#### 方案：DeepSeek EPLB 冗余专家放置

为解决负载不均衡问题，我们采用了 DeepSeek 开源的 **EPLB（Expert Parallelism Load Balancing）** 算法。

核心思想非常直观：在原始 **256 个 experts** 的基础上，额外创建 **32 个冗余 expert 副本**，总计 **288 个 experts**（每个 device 从 8 个增加到 9 个）。EPLB 算法接受实际运行中采集的 expert 路由分布作为输入，计算出最优的 expert-to-device 放置方案。冗余副本被智能地放置在负载较低的设备上，分担热点 expert 的 token 流量。

这样，当某个 expert 成为热点时，其 token 会被分散到原始副本和冗余副本上，从而均衡各设备的计算负载。

#### 实现：三阶段流程

EPLB 的端到端使用分为三个阶段：

**阶段一：分布采集。** 服务运行时自动记录每个 expert 的 routing 分布统计，写入 `.npy` 文件。通过以下参数启用：

```
--enable-expert-distribution-recorder
--expert-distribution-recorder-output-file expert_dist.npy
```

**阶段二：放置计算。** 基于采集的分布数据，调用 EPLB 算法计算最优的 expert 放置方案，确定哪些热点 expert 需要冗余副本，以及副本放置在哪些设备上：

```
--init-expert-location expert_dist.npy
```

**阶段三：动态 dispatch。** 服务以冗余专家模式启动，运行时根据放置方案动态路由 token 到正确的 expert 副本：

```
--ep-num-redundant-experts 32
--ep-dispatch-algorith dynamic
```

精度验证方面，我们在 GSM8k 上测试了 EPLB 冗余放置后的模型准确率为 **0.9371**，确认冗余专家放置不影响模型精度。

#### 效果

<!-- TODO: 请补充以下数据 -->

<!-- 1. Heatmap 对齐：需要同一层（统一用 Layer 19 或 Layer 78）的 EPLB 前后 heatmap -->
<!-- 当前问题：EPLB 前用了 Layer 78，EPLB 后用了 Layer 19，不同层的对比说服力不足 -->
<p align="center"><img src="TODO_eplb_before_same_layer_heatmap.png" width="80%"></p>
<p align="center"><img src="TODO_eplb_after_same_layer_heatmap.png" width="80%"></p>

<!-- 2. 定量效果数据（至少补充其一）: -->
<!-- (a) Kernel 耗时对比：EPLB 前后同一层/同一 batch 的 fused MoE kernel 平均耗时，如 2ms → ?ms -->
<!-- (b) Device 不均衡比：EPLB 前后的 max/mean ratio，如从 2.5x → ?x -->
<!-- (c) 端到端 ITL 对比：EPLB 前后的 bench_one_batch ITL 数据 -->

EPLB 带来的核心收益体现在 `sync_barrier()` 等待时间的减少。在均匀分布的理想情况下，barrier 的基准开销仅为 **0.039ms**；但在实际不均衡分布下，barrier 等待时间远大于此。具体而言，如果某个 device 上的 expert 接收了 **40 个 token**（而其他 device 平均仅 **16 个**），该 device 的 FFN 计算耗时将达到平均值的 **2.5 倍**，其余所有 device 需要额外等待 **10-20us**。这种等待在 60 多层 MoE 中逐层累积，对端到端延迟的影响不可忽视。

EPLB 通过均衡设备间负载，直接缩小了最慢设备与最快设备之间的差异，从而减少了每层 barrier 的等待时间，改善了 fused MoE kernel 中最不可预测的性能开销来源。

#### 参考

- DeepSeek EPLB: [https://github.com/deepseek-ai/EPLB](https://github.com/deepseek-ai/EPLB)
- SGLang GPU EPLB: [sgl-project/sglang#5295](https://github.com/sgl-project/sglang/pull/5295)

### 4. DP Attention

#### 动机：纯 TP 模式的瓶颈

Ring-1T 采用 **GQA（Grouped Query Attention）** 架构，整个模型仅有 **8 个 KV heads**。这意味着 KV heads 最多只能被 **TP=8** 均匀切分——每个 TP shard 恰好分到 1 个 KV head。

当使用 TP=32 的纯 Tensor Parallelism 部署时，8 个 KV heads 无法在 32 个 shard 之间均分。多出的 24 个 shard 必须**复制** KV cache，导致相同的 KV cache 数据被冗余存储 **4 份**（32 / 8 = 4x 复制），极大地浪费了宝贵的 HBM 容量。

引入 **Data Parallelism（DP）** 后，这个问题迎刃而解。在 DP=4 × TP=8 的配置下，8 个 KV heads 恰好被 TP=8 完美切分，每个 shard 分到 1 个 KV head，**零复制**。4 个 DP rank 各自管理独立的 KV cache 分区，既消除了冗余存储，又提高了内存利用率。

#### 架构设计：2D Mesh + 单控制器

我们设计了 **2D Mesh** 并行策略，将 32 个 TPU cores 组织为 `('data', 'tensor')` 两个维度：

- **Attention 层**：DP=4 x TP=8。KV cache 沿 DP 维度分片，每个 DP rank 独立管理自己的 KV cache；attention 计算沿 TP 维度并行。
- **MoE 层**：EP=32。全部 32 个 cores 参与 expert 并行，与 attention 层共享相同的物理设备。

<p align="center"><img src="TODO_dp_attention_ep_moe_parallel_diagram.png" width="80%"></p>

在调度架构上，我们采用了**单控制器设计**，这与 SGLang PyTorch 版本的多控制器架构有本质区别。JAX 的 SPMD 编程模型天然支持单控制器——一个 Scheduler 拥有全局视野，统一管理所有 DP ranks 的请求调度。Prefix cache 和 KV allocator 按 `dp_id` 分区，每个 DP rank 的资源互不干扰。调度策略支持 **shortest-queue** 和 **round-robin** 两种模式。

单控制器的优势在于：全局调度视野使得 prefix cache 管理更加简洁高效，无需跨控制器的复杂协调，降低了系统复杂度。

#### 关键实现

**KV Cache 分片。** KV cache 的 shape 为 `(num_pages, page_size, num_kv_heads, head_dim)`，sharding 策略设置为 `P('data', None, 'tensor', None)`——pages 按 DP 维度分片，heads 按 TP 维度分片。这样每个 DP rank 独立管理各自的 KV cache 分区，互不干扰，也无需跨 DP rank 同步。

**Attention Kernel 兼容。** Flash attention kernel 通过 `shard_map` 包装，确保每个 DP rank 看到的是从 index 0 开始的局部数据视图。这意味着 attention kernel 内部无需任何通信操作，只需正确设置 `in_specs` 和 `out_specs` 即可——对已有 kernel 的侵入性极小。

**调度感知 DP。** 请求在进入系统时被分配 `dp_id`，元数据按 DP 维度重排后传入 model forward。计算完成后，输出结果按原始请求顺序重排返回，对上层调用者完全透明。

#### 收益

DP Attention 的收益体现在内存、通信和延迟三个维度。

**内存节省。** 纯 TP=32 模式下，8 个 KV heads 被 4x 复制，每个 core 存储的 KV cache 中有 **3/4 是冗余副本**。切换到 DP=4 × TP=8 后，KV heads 被 TP=8 完美切分，零复制，释放的 HBM 空间可用于存储更多 token。在 EP=32、DP=4、TP=8 的配置下，单 core 的模型权重占用 **68.26 GB**，剩余可分配给 KV cache 的空间为 **20.06 GB**，可存储约 **525,860 tokens**。这直接使得 batch size 可以从 256 扩展到 **512**。

**通信量降低。** DP 将请求分散到多个 DP rank，每个设备处理的 local tokens 数量减少，MoE 层的 A2A scatter/gather 数据量随之下降。

**延迟与吞吐改善。** 下表对比了纯 TP=32 与 DP=4 + TP=8 在 bench_one_batch 上的表现：

| 配置 | Batch Size | Output Throughput (tok/s) | ITL (ms) |
|------|-----------|--------------------------|----------|
| TP=32 | 64 | 599.27 | 106.80 |
| TP=32 | 128 | 710.07 | 180.26 |
| TP=32 | 256 | 2069.43 | 123.71 |
| DP=4, TP=8 | 64 | 700.83 | **91.32** |
| DP=4, TP=8 | 128 | 1269.19 | **100.85** |
| DP=4, TP=8 | 256 | 2223.90 | **115.11** |
| DP=4, TP=8 | 512 | 3092.44 | 165.57 |

关键对比点：

- **ITL 改善**：bs=64 时 ITL 从 106.80ms 降至 **91.32ms**（**-14.5%**）；bs=128 时从 180.26ms 降至 **100.85ms**（**-44%**）。
- **吞吐提升**：bs=64 时 output throughput 从 599 tok/s 提升至 **700 tok/s**（**+17%**）；bs=128 时从 710 tok/s 提升至 **1269 tok/s**（**+79%**）。
- **Batch size 扩展**：纯 TP 配置下 batch size 最大只能到 256，DP 配置可扩展到 **512**，output throughput 达到 **3092 tok/s**。

DP Attention 的本质是将冗余的 KV cache 复制转化为有效的数据分片，在不增加硬件资源的前提下，同时改善了内存效率、通信开销和计算利用率。

### 5. FP8 Quantization

<!-- TODO: activation/weight quantization -->

## Experiments

### TPU v7 vs H200 Performance

<!-- TODO: comparison table (bf16 32-chip + FP8 16-chip, ROI) -->

## How to Use

<!-- TODO: launch commands / config -->

## Future Work

<!-- TODO -->

## Acknowledgements

<!-- TODO -->

## References

<!-- TODO -->
