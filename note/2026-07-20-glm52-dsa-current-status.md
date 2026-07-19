# GLM-5.2 DSA 当前状态快照（2026-07-20）

## 结论

截至 `develop/glm52-dsa-falcon` 的 `a974bdbb4`，GLM-5.2 DSA 已在 Falcon
v7x-32 上完成 correctness-first 的真实权重 E2E 闭环：TP32 / DP1 / EP32、fused
MoE、chunked prefill、ragged batch、真实 Top-K 截断和 decode 都可以运行并返回稳定结果。

当前可以声明：

- 独立 PyTorch CPU golden 覆盖 Indexer selection、logical-to-physical mapping 和
  sparse MLA；无 tie fixture 的整数结果要求精确一致。
- TPU Pallas kernel 对 Torch FP32 的最坏 max abs 为 `0.0019080639`，低于
  `atol=0.01`。
- 3072-token 请求中有 1025 个 query 超过 `index_topk=2048`，selection/mapping
  validator 零失败；相同请求重复运行的 output ID、logprob 和 top-20 完全一致。
- short、chunked 和 ragged 请求的 DSA/FA greedy output ID 相同，当前实现不存在已知的
  selection、paged mapping 或请求调度功能错误。

当前不能声明：

- short-context DSA 与 FA 的严格 logits/logprob 数值等价。两条路径关注相同 token 集，
  但 BF16 softmax/归约顺序不同，生成 logprob 最大绝对误差为
  `0.0703125 / 0.171875 / 0.203125`，没有通过 `0.05` 门。
- long-context DSA logits 应等于 dense attention logits。超过 2048 个可见 token 后，
  两者数学上选择不同 token 集；正确 gate 是 selection/mapping、sparse attention golden
  和任务质量，而不是 dense logits 相等。
- 当前 kernel 或 E2E 已获得性能收益。production-shape sparse prototype 仍是
  correctness-first 实现，尚未做 gather、DMA 或计算布局优化。
- GPQA 已复现官方准确率。4 题 smoke 中 3 题在 4096 completion token 被截断，样本和
  generation budget 都不足以形成质量结论。

完整过程、命令和逐层数据见
`note/2026-07-18-glm52-dsa-falcon-results.md`；本文只冻结进入性能 profiling 前的结论。

## Correctness 基线

### Kernel-level cross-framework matrix

CPU matrix 覆盖 candidate length：

```text
1 / 127 / 128 / 129 / 257 / 2047 / 2048 / 2049 / 3072 / 4096
```

验证口径：

| 部分 | Golden | Gate | 结果 |
| --- | --- | --- | --- |
| Indexer selection | PyTorch CPU | selected count、logical ID 精确一致 | 通过 |
| Paged mapping | PyTorch CPU | physical slot 精确一致 | 通过 |
| Sparse MLA FP32 | PyTorch CPU | JAX `rtol=1e-5 / atol=1e-5` | 通过 |
| Production Pallas | PyTorch CPU FP32 | `rtol=2e-2 / atol=1e-2` | 通过 |

Pallas 对 Torch 的最坏 max abs：

```text
length=128, max_abs=0.0019080639
```

### E2E 短上下文

真实 checkpoint、BF16、TP32 / DP1 / EP32、fused MoE 下：

| 请求 | DSA/FA output ID | 最大生成 logprob 绝对误差 | 最低 top-20 overlap |
| --- | --- | ---: | ---: |
| short | 相同 | 0.0703125 | 0.90 |
| 257-token chunked | 相同 | 0.171875 | 0.95 |
| 9/133-token ragged | 相同 | 0.203125 | 0.95 |

DSA 重复运行在以上请求上达到零误差、top-20 overlap `1.0`。PR #1062 风格逐层 dump
显示差异从 layer 0 的 BF16 量级开始，随 78 层平滑累积，没有单层突跳证据。

### E2E 真实稀疏上下文

3072-token prompt + one-token decode 覆盖 position `0..3072`：

- 3073 个 active query；其中 1025 个进入 `visible > 2048` 的真实截断区。
- `selected_count == min(position + 1, 2048)`。
- counted logical ID 无重复、非负、不指向 future；未截断时覆盖完整 causal set。
- physical slot 非负且无重复，logical/physical padding 分别为 `-1/0`。
- required boundary positions `2046/2047/2048/3071` 全部存在。
- 重复请求 output ID `[198]`、logprob `-4.1875`、top-20 均完全一致。

这证明真实模型 integration 和真实稀疏路径可运行；截断后的 Top-K 排名正确性由独立
CPU fixture matrix 保证，不使用 dense attention logits 作为 golden。

## 性能基线

### Kernel microbenchmark

Falcon v7x-8，JAX 0.9.0，编译和 20 次 warm-up 后同步计时：

| Workload | median | p99 |
| --- | ---: | ---: |
| sparse B1/H2/C160k/top-k2048 | 2.1947 ms | 2.3469 ms |
| sparse B1/H2/C160k/top-k2048，page-sorted | 2.1632 ms | 2.1844 ms |
| sparse B8/H2/C32k/top-k2048 | 16.5070 ms | 17.0481 ms |
| dense B1/H2/C160k workload baseline | 0.9916 ms | 1.0174 ms |

dense 行不是相同 attention domain，只能作为 workload baseline。当前 sparse B1 约慢
`2.21x`，不能据此声称 DSA kernel 有加速。

### Fresh-node 服务启动

metadata sidecar 已将 fresh-node server-ready 从 `65m04s` 降到 `43m09s`：

| 阶段 | 耗时 | 占 server-ready |
| --- | ---: | ---: |
| 初始化到权重加载 | 约 39s | 约 1.5% |
| 权重加载 | 26m48s | 约 62% |
| absorb | 37s | 约 1.4% |
| EXTEND precompile | 614s | 约 24% |
| DECODE precompile | 290s | 约 11% |

sidecar 消除了约 22 分钟 metadata header scan；剩余启动时间主要是实际 checkpoint
读取和 13 个显式 precompile variant，不是无界卡死。

### 当前 E2E wall time

完成显式 precompile 后，相同的 3072-token + one-token 请求两次分别为：

```text
99.9987s
99.9844s
```

两次 `cache_miss_count=1`，结果完全一致，没有只发生在首轮的额外编译迹象。但现有数据
只有请求 wall time，尚未拆分以下组成：

- prefill 各 chunk 的 device execution 和 host gap；
- decode 单步 device execution 和 host scheduler/sampling/result-processing gap；
- tokenizer、HTTP、IPC、batch construction、KV/cache bookkeeping 等 CPU overhead；
- DSA attention 之外的 MoE、dense layer、collective 和 sampling 占比。

因此 `~100s` 只能作为 profiling 前的 E2E 基线，不能直接归因于 DSA kernel。

## Profiling 阶段的范围

下一阶段先不优化 DSA Pallas kernel，而是：

1. 在完全预热、相同 shape 的请求上分别采集 prefill/decode JAX device trace。
2. 同时开启 host tracer 和受控 Python tracing，拆分 scheduler、batch preparation、
   dispatch/wait、sampling、result handling 与 HTTP/tokenizer 开销。
3. 将 request wall time、server batch step 和 device trace 对齐，先量化 device busy 与
   host gap，再选择最大的非 DSA-kernel 瓶颈。
4. 每次只修改一个已被 trace 证明的瓶颈，保持 correctness gates 不变，复测相同工作负载。

首轮优化不包含 DSA gather/DMA/计算布局；kernel 优化留到 non-kernel overhead 已可解释、
E2E 测量稳定之后。

## 证据索引

- 真实权重主 E2E：Falcon `exp-sff91uc6va`，artifact `art-i9c3i0yvvw`。
- 逐层与 3072 边界验证：Falcon `exp-8m0q7a4og9`，artifact `art-paz8z33izy`。
- CPU/Torch 与 Pallas matrix：Falcon `exp-6dqnqemqcd`。
- fresh-node sidecar + precompile：Falcon `exp-zeb34sbqwj`，artifact `art-9v9fkmhhvg`。
- sidecar 构建/完整性校验：Falcon `exp-9h2a7czdt2`，artifact `art-8bnaxyq6q2`。
