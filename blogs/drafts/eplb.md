## EPLB: Expert Parallelism Load Balancing

### 问题：Expert 负载不均衡

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

### 方案：DeepSeek EPLB 冗余专家放置

为解决负载不均衡问题，我们采用了 DeepSeek 开源的 **EPLB（Expert Parallelism Load Balancing）** 算法。

核心思想非常直观：在原始 **256 个 experts** 的基础上，额外创建 **32 个冗余 expert 副本**，总计 **288 个 experts**（每个 device 从 8 个增加到 9 个）。EPLB 算法接受实际运行中采集的 expert 路由分布作为输入，计算出最优的 expert-to-device 放置方案。冗余副本被智能地放置在负载较低的设备上，分担热点 expert 的 token 流量。

这样，当某个 expert 成为热点时，其 token 会被分散到原始副本和冗余副本上，从而均衡各设备的计算负载。

### 实现：三阶段流程

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

### 效果

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

### 参考

- DeepSeek EPLB: [https://github.com/deepseek-ai/EPLB](https://github.com/deepseek-ai/EPLB)
- SGLang GPU EPLB: [sgl-project/sglang#5295](https://github.com/sgl-project/sglang/pull/5295)
