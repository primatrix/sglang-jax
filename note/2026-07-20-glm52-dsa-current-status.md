# GLM-5.2 DSA 当前状态快照（2026-07-20）

> 2026-07-20 scope update：服务侧 prefill/decode 性能差异已经能够由真实 static
> bucket、selected-cache 访存和当前 correctness-first Pallas 实现解释，本轮不再重复
> E2E profile。后续工作收敛为给算子同事交付独立 CPU PyTorch golden、冻结 ABI 和
> production-derived performance shapes；详见
> `note/2026-07-20-glm52-dsa-kernel-handoff.md`。

## 结论

截至 `develop/glm52-dsa-falcon` 的 profile manifest commit `8dd4aca49`（复测 source
`0324d5a0b`），GLM-5.2 DSA 已在 Falcon
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
- 首个 non-Pallas 优化已通过同 shape Falcon 复测：MLA cache 的 latent/RoPE 两次 scatter
  合成一次后，prefill/decode 的单层 cache-write device time 分别下降 `48.8% / 53.1%`；
  DSA custom-kernel 总量基本不变，correctness/schema gate 全部保持通过。

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

两次 `cache_miss_count=1`，结果完全一致，没有只发生在首轮的额外编译迹象。下面的
Falcon profile 已进一步确认，`~100s` 不是隐式 JIT，而是 3072-token 请求被拆成 24 个
128-token prefill chunk 后的真实执行时间；后半段 chunk 随可见上下文增长到约 6s。

### Falcon prefill/decode profile

基线 profile 使用 source `65b35076d53d4d0dc50432b17e3789e817361f74`，Falcon
`exp-7yo6j8fajf` / artifact `art-k06il4ul4q`，真实 checkpoint、TP32 / DP1 / EP32、
fused MoE、3072 input / 8 output。正确性和 response schema gate 全部通过。

server-ready 仍是 fresh-node 路径：权重加载约 `26m52s`，absorb 约 `38s`，EXTEND
precompile 约 `600s`，DECODE precompile 约 `291s`，总计 `2579s`。server-ready 后没有
新的 compile/cache-miss 日志。

rank-0 的阶段 trace 和 XProf 汇总如下。XProf category 是 local-device HLO self-time
aggregate，不能直接相加成 request wall；trace span/model envelope 才是 rank-0 wall
近似。

| 指标 | Prefill capture | Decode capture |
| --- | ---: | ---: |
| rank-0 trace span | 1127.888ms | 6059.490ms |
| `jit_jitted_run_model` | 604.401ms | 5886.527ms |
| XProf HLO self-time aggregate | 22432.095ms | 55143.598ms |
| custom kernel | 67.7% | 89.3% |
| elementwise | 17.6% | 6.0% |
| custom fusion | 10.9% | 2.1% |
| async | 2.8% | 2.0% |

decode 的 top operators 全是 `dsa-decode-mla.*`；因此 decode 已明确是 DSA Pallas
kernel 主导，而不是 Python/HTTP 主导。prefill 的非 Pallas device work 主要来自 DSA
selection/cache bookkeeping、fused MoE 和 collectives。

host trace 中，prefill/decode 的 `device_get` 分别为 `601.1ms / 5884.8ms`，与 device
model envelope 对齐，含义是 host 在等待 TPU 完成，不是等量的 CPU 计算。可见的
`copy_to_host_async` 仅 `0.05ms / 0.06ms`。decode 的 `run_batch=82.3ms` 和
`sample=78.7ms` 又包含约 `75.7ms` 的 profiler start；排除 profiler 控制后，没有发现
单步 `>=100ms` 的可避免 host/Python gap。

本次 profiled request wall 为 `167.518s`，比未 profile 的同 shape warmup
`102.616s` 慢 `64.902s`。server log 显示 prefill/decode 各有一次同步 trace flush，
分别暂停约 `31s / 33s`；profile finalize 在 request 返回后仅 `0.337s`。因此
`167.518s` 是诊断开销，不能用作 production E2E 基线。warmup 与此前两次约 100s 的
结果一致。

### 首个 non-Pallas 优化候选

按 rank-0 TPU:0 的 source 聚合，`write_mla_kv_cache` 在每层对同一 physical slot
连续执行两次 scatter：`reference.py:87` 写 latent，`reference.py:97` 写 RoPE。prefill
capture 中两组 scatter 分别为 `47.634ms / 45.456ms`，合计 `93.090ms`，占
`jit_jitted_run_model` 的 `15.4%`；decode 中合计 `127.852ms`，只占 `2.2%`。

两段目标区间不重叠，且都从同一 token/update 输入构造。把它们打包成一次连续 scatter
不改变 physical slot、有效位、latent/RoPE 数值或 Pallas attention kernel，是当前最大且
可单独测试的 non-Pallas prefill 候选。理论上只可能消除其中一次 scatter 的固定/循环
开销，不能把 `15.4%` 全部当成预期收益；最终只接受同 shape Falcon 复测中超出噪声的
实际改善。

### MLA cache single-scatter 复测结果

实现提交为 `d18e21c95`，补充 padding/invalid-slot contract 为 `6630ebb457`。第一次
candidate `exp-owybp7zc86` 在首次 EXTEND trace 暴露 JAX 0.9.0 explicit-sharding 问题：
新增 alignment padding 是 replicated sharding，而 latent/RoPE 是
`PartitionSpec("data", None)`，`concatenate` 抛出 `ShardingTypeError`。修复提交
`0324d5a0b` 让 padding 显式继承 `new_c_kv` 的 `out_sharding`，对应回归测试先 RED 后
GREEN；本地完整 DSA/profile regression 为 `117 passed`。

修复后的 source 经 staging `exp-9k9z3qx0ks` / `art-mtacuv18v3` 写入共享模型盘，SHA256
为 `4bc479b874183445b8cbbe5cfed2f7a04b11b4c6987d1a66cbda3de60d0cba75`。同 shape
candidate 为 `exp-6sroakc4lh` / `art-q8dine8622`，source
`0324d5a0bf98944417d0a563fced8bf02a704db7`。checkpoint、TP32/DP1/EP32、fused MoE、
3072 input / 8 output、chunk size、precompile variants 和 tracer levels 与 baseline 完全相同。

正确性 gate 全部通过：两次请求的 output ID、output logprob、shared top-20 logprob 都是
零误差，top-20 overlap `1.0`，schema 无错误，终态日志为
`GLM52_DSA_REAL_E2E_OK backend=dsa requests=2`。

XProf 和 rank-0 TPU:0 精确 source 聚合如下。不同 profile 的 trace 边界捕获到的 layer
event 数量不同，因此 cache write 使用每个 event 的平均时长；`jit_jitted_run_model` 使用
单次最大 envelope。XProf HLO self-time 是 32 个 local device 的 aggregate。

| 指标 | Baseline | Single-scatter | 变化 |
| --- | ---: | ---: | ---: |
| Prefill `jit_jitted_run_model` | 604.401ms | 561.371ms | -7.1% |
| Prefill HLO self-time aggregate | 22432.095ms | 20450.354ms | -8.8% |
| Prefill elementwise aggregate | 3947.990ms | 1993.050ms | -49.5% |
| Prefill custom-kernel aggregate | 15190.020ms | 15175.350ms | -0.1% |
| Prefill cache-write / event | 1.981ms | 1.014ms | -48.8% |
| Decode `jit_jitted_run_model` | 5886.527ms | 5804.940ms | -1.4% |
| Decode HLO self-time aggregate | 55143.598ms | 53448.362ms | -3.1% |
| Decode elementwise aggregate | 3297.700ms | 1671.080ms | -49.3% |
| Decode custom-kernel aggregate | 49251.400ms | 49245.740ms | -0.0% |
| Decode cache-write / event | 2.113ms | 0.991ms | -53.1% |

source event family 从 baseline 的 `reference.py:87 / :97` 两组变为 candidate 的
`reference.py:104` 一组；`reference.py:91` 的 padding fusion 每个 event 约 `0.01us`，
不是第二个 scatter。prefill DSA custom call 的归一化时长约
`2.367ms -> 2.366ms`，decode 约 `71.873ms -> 70.959ms`，没有把 Pallas kernel 的变化
误归因给本次优化。

未 profile 的同 shape warmup wall 从 `102.616s` 降到 `100.345s`（-2.2%）。profiled
request 从 `167.518s` 降到 `148.018s`，但两次都包含同步 trace flush，不能把 -11.6%
当作 production 收益。candidate 的 `device_get` 最大时长仍与 device model envelope
对齐（prefill `557.4ms`、decode `5800.1ms`），而 `copy_to_host_async` 仍为亚毫秒级；没有
出现新的 host/Python 瓶颈。

固定 acceptance gate 的结论是保留实现：correctness/schema 不变；两组 scatter 变一组；
prefill 目标区域和 model envelope 都有超出 trace noise 的下降；decode 无回退。当前
E2E 仍由 24 个 chunk 的模型执行和 DSA Pallas decode 主导，下一步再讨论 DSA kernel，
不继续挤压本轮 non-kernel 范围。

## Profiling 阶段的范围

下一阶段仍不优化 DSA Pallas kernel，而是：

1. 在完全预热、相同 shape 的请求上分别采集 prefill/decode JAX device trace。
2. 同时开启 host tracer 和受控 Python tracing，拆分 scheduler、batch preparation、
   dispatch/wait、sampling、result handling 与 HTTP/tokenizer 开销。
3. 将 request wall time、server batch step 和 device trace 对齐；当前已确认 host gap
   很小，decode 由 DSA Pallas 主导。
4. 首先合并 MLA cache 的两次连续 scatter，保持 correctness gates 不变，并复测相同
   3072/8 工作负载。

首轮优化不包含 DSA gather/DMA/计算布局；kernel 优化留到 non-kernel overhead 已可解释、
E2E 测量稳定之后。

## 证据索引

- 真实权重主 E2E：Falcon `exp-sff91uc6va`，artifact `art-i9c3i0yvvw`。
- 逐层与 3072 边界验证：Falcon `exp-8m0q7a4og9`，artifact `art-paz8z33izy`。
- CPU/Torch 与 Pallas matrix：Falcon `exp-6dqnqemqcd`。
- fresh-node sidecar + precompile：Falcon `exp-zeb34sbqwj`，artifact `art-9v9fkmhhvg`。
- sidecar 构建/完整性校验：Falcon `exp-9h2a7czdt2`，artifact `art-8bnaxyq6q2`。
- non-kernel profile：Falcon `exp-7yo6j8fajf`，artifact `art-k06il4ul4q`，source
  `65b35076d53d4d0dc50432b17e3789e817361f74`。
- prefill/decode XProf：`an-wbmml9rf8u` / `an-qxbsln3gxe`。
- host/Python trace：`an-wa47uzjlnm`；profile wall-gap：`an-tiifhhulg7`；non-DSA
  device source breakdown：`an-0emu7qinah`。
- single-scatter candidate：Falcon `exp-6sroakc4lh`，artifact `art-q8dine8622`，source
  `0324d5a0bf98944417d0a563fced8bf02a704db7`；source staging `exp-9k9z3qx0ks` /
  `art-mtacuv18v3`。
- candidate prefill/decode XProf：`an-u4d5j18uat` / `an-7ncficq29z`；candidate
  source/host/timing breakdown：`an-nd08uoij7w`；精确 baseline MLA cache 聚合：
  `an-o9l9bonmc2`。
