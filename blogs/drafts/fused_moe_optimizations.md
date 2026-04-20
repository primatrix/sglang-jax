## All Reduce Metadata 优化

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

---

## Shared Expert 融合

### 问题：串行执行的额外延迟

Ring-1T 模型采用混合专家架构，包含 **1 个 shared expert**（所有 token 共享的 dense FFN）和 **256 个 routed experts**。Shared expert 的算术强度（Arithmetic Intensity）在 BS=512 时为 **AI=1056**，远超 TPU v7 的临界值 312.6，属于典型的 **compute-bound** 负载。

如果 shared expert 在所有 routed expert 之后串行执行，它将直接叠加在 MoE 层的总延迟上，造成不必要的性能损失。

### 方案：穿插在通信间隙中

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

### 关键优化细节

为了让 SE block 高效地穿插执行，我们进行了四项渐进式优化：

**独立的双缓冲权重。** SE 使用独立的权重缓冲区，不与 routed expert 的权重缓冲竞争 VMEM 带宽。这确保了 SE 的权重加载不会干扰 routed expert 的权重预取流水线。

**SE hidden_size 切分 + 重排 fetch 位置。** 将 SE 的 intermediate dimension 按 `bse` 粒度切分，并优化权重和 token 的预取位置，减少 SE 与 routed expert 之间的内存访问冲突。

**SE 预取优化。** 在 FFN2 计算期间预取下一个 SE block 的 W1/W3 权重，进一步隐藏加载延迟，使 SE block 的执行更加流畅。

**SE 计算插入 wait_a2a_scatter_recv 间隙。** 利用 scatter 接收等待期间（约 **5-6us**）的空闲 MXU，额外插入 SE 计算，充分榨取每一微秒的闲置算力。

### 消融实验

<!-- TODO: 补充四个优化阶段的完整消融对比表（包含所有列），展示各阶段各组件的耗时变化趋势 -->

最终优化版本的完整消融数据（NumTokens=128）：

| 负载分布 | all enable | all disabled | enable a2a | enable dynamic_ffn1 | enable dynamic_ffn2 | enable weight_load | enable a2a_s_tile_read | enable a2a_s_acc_tile_write | enable shared_expert |
|---------|-----------|-------------|-----------|-------------------|-------------------|------------------|---------------------|--------------------------|-------------------|
| hotspot-count 8 | 2.756 | 0.223 | 0.295 | 1.689 | 1.004 | 0.467 | 0.479 | 0.353 | 0.288 |
| hotspot-count 64 | 0.895 | 0.224 | 0.257 | 0.516 | 0.378 | 0.466 | 0.273 | 0.258 | 0.288 |
| hotspot-count 256 | 0.615 | 0.223 | 0.259 | 0.320 | 0.273 | 0.466 | 0.249 | 0.243 | 0.288 |

> **注**：`enable shared_expert` 列的数值代表打开 SE 特性后的增量贡献。由于 SE 计算被有效隐藏在通信间隙中，这个增量对端到端延迟的影响远小于串行执行的情况。

### 理论分析

如果 SE 计算完全隐藏在通信间隙中，MoE 层的总耗时将等于纯 routed expert 的耗时——即实现**零额外 SE 开销**。在 bse=256 的配置下，SE 被切分为 **8 个 block**，而 8 个 expert 的流水线提供了 **8 x 2 + 1 = 17 个**可用插入间隙，绑定关系相当宽松，SE 有充足的空间完成计算。

这一优化的本质是**资源互补调度**：利用通信密集阶段的计算空闲，执行计算密集的 shared expert，从而在不增加总延迟的前提下完成额外的计算工作。
