## DP Attention

### 动机：纯 TP 模式的瓶颈

Ring-1T 采用 **GQA（Grouped Query Attention）** 架构，整个模型仅有 **8 个 KV heads**。这意味着 KV heads 最多只能被 **TP=8** 均匀切分——每个 TP shard 恰好分到 1 个 KV head。

当使用 TP=32 的纯 Tensor Parallelism 部署时，8 个 KV heads 无法在 32 个 shard 之间均分。多出的 24 个 shard 必须**复制** KV cache，导致相同的 KV cache 数据被冗余存储 **4 份**（32 / 8 = 4x 复制），极大地浪费了宝贵的 HBM 容量。

引入 **Data Parallelism（DP）** 后，这个问题迎刃而解。在 DP=4 × TP=8 的配置下，8 个 KV heads 恰好被 TP=8 完美切分，每个 shard 分到 1 个 KV head，**零复制**。4 个 DP rank 各自管理独立的 KV cache 分区，既消除了冗余存储，又提高了内存利用率。

### 架构设计：2D Mesh + 单控制器

我们设计了 **2D Mesh** 并行策略，将 32 个 TPU cores 组织为 `('data', 'tensor')` 两个维度：

- **Attention 层**：DP=4 x TP=8。KV cache 沿 DP 维度分片，每个 DP rank 独立管理自己的 KV cache；attention 计算沿 TP 维度并行。
- **MoE 层**：EP=32。全部 32 个 cores 参与 expert 并行，与 attention 层共享相同的物理设备。

<p align="center"><img src="TODO_dp_attention_ep_moe_parallel_diagram.png" width="80%"></p>

在调度架构上，我们采用了**单控制器设计**，这与 SGLang PyTorch 版本的多控制器架构有本质区别。JAX 的 SPMD 编程模型天然支持单控制器——一个 Scheduler 拥有全局视野，统一管理所有 DP ranks 的请求调度。Prefix cache 和 KV allocator 按 `dp_id` 分区，每个 DP rank 的资源互不干扰。调度策略支持 **shortest-queue** 和 **round-robin** 两种模式。

单控制器的优势在于：全局调度视野使得 prefix cache 管理更加简洁高效，无需跨控制器的复杂协调，降低了系统复杂度。

### 关键实现

**KV Cache 分片。** KV cache 的 shape 为 `(num_pages, page_size, num_kv_heads, head_dim)`，sharding 策略设置为 `P('data', None, 'tensor', None)`——pages 按 DP 维度分片，heads 按 TP 维度分片。这样每个 DP rank 独立管理各自的 KV cache 分区，互不干扰，也无需跨 DP rank 同步。

**Attention Kernel 兼容。** Flash attention kernel 通过 `shard_map` 包装，确保每个 DP rank 看到的是从 index 0 开始的局部数据视图。这意味着 attention kernel 内部无需任何通信操作，只需正确设置 `in_specs` 和 `out_specs` 即可——对已有 kernel 的侵入性极小。

**调度感知 DP。** 请求在进入系统时被分配 `dp_id`，元数据按 DP 维度重排后传入 model forward。计算完成后，输出结果按原始请求顺序重排返回，对上层调用者完全透明。

### 收益

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
