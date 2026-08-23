# SGLang JAX 适配 DSpark 设计文档

状态：Stage 1 已完成；Stage 2 第一版已实现（tuned config、动态预算与 compact ragged verify，TPU serving 验证待完成）

当前默认路径仍是 Stage 1：vanilla Markov Head、带 Markov embedding 的
Confidence Head、官方 checkpoint 直载、`gamma/draft_width=7` 与固定
`verify_width=8`。显式传入 `--enable-dspark-tuned-config` 且 deployment key
精确命中时，启用 Stage 2 第一版的 STS、SPS planner、逐请求 `verify_len`、
compact RPA 和静态 token bucket；key miss 自动回退固定 verify-all。

Stage 1 启动示例（checkpoint 的 `block_size=7` 对应 7 个 proposal，内部固定
验证宽度会归一化为 8）：

~~~bash
./.venv/bin/python -m sgl_jax.launch_server \
  --model-path /models/Qwen3-8B \
  --speculative-algorithm DSPARK \
  --speculative-draft-model-path deepseek-ai/dspark_qwen3_8b_block7 \
  --speculative-num-steps 1 \
  --speculative-num-draft-tokens 7 \
  --speculative-eagle-topk 1 \
  --tp-size 4 --dp-size 1 \
  --dtype bfloat16 --attention-backend fa \
  --mem-fraction-static 0.65 --page-size 64 \
  --chunked-prefill-size 2048 --max-running-requests 128 \
  --trust-remote-code --disable-radix-cache --grammar-backend none \
  --host 0.0.0.0 --port 30000
~~~

目标草稿模型：`deepseek-ai/dspark_qwen3_8b_block7`

目标主模型：`Qwen/Qwen3-8B`

参考实现：SGLang PyTorch PR [#30261](https://github.com/sgl-project/sglang/pull/30261)

参考论文：[DSpark](https://arxiv.org/abs/2607.05147)

## 1. 摘要

本文设计如何在 SGLang JAX/TPU 中适配 DSpark 推测解码。DSpark 可以拆成五部分：

1. DFlash 风格的并行草稿 Transformer。
2. 在并行 hidden states 之后执行的低秩串行 Markov Head。
3. 预测各草稿位置条件接受概率的 Confidence Head。
4. 根据历史 confidence 和硬件 SPS/step-time 曲线选择验证预算。
5. 按请求生成不同的 `verify_len`，紧凑打包后交给目标模型验证。

SGLang JAX 已有 DFlash、Qwen3 hidden capture、KV materialization、TP/DP、overlap relay、JIT 预编译和 `ragged_paged_attention_v3`。因此不需要重新实现 attention kernel。

主要新增工作是：

- DSpark Markov Head、Confidence Head 和 checkpoint 加载。
- 分离 `gamma`、`draft_width`、`verify_width` 三种长度。
- 建立不破坏因果性和 overlap 的 confidence relay。
- 使用 TPU 实测 SPS/step-time 数据计算验证预算。
- 在 device 上把固定总预算分配为逐请求 `verify_lens`。
- 把不同长度请求压紧到静态 token bucket。
- 将 compact 输出映射回请求坐标，并正确提交 KV、hidden state 和 sequence state。

第一版只支持 greedy decoding。Sampling、grammar、logprob、LoRA 和 disaggregation 暂不支持。

## 2. 核心判断

### 2.1 RPA 已原生支持 ragged query

当前 `ragged_paged_attention_v3` 接收：

~~~text
queries:    [max_num_tokens, num_q_heads, head_dim]
cu_q_lens: [max_num_seqs + 1]
~~~

每个请求的 query 范围为：

~~~text
q_start = cu_q_lens[r]
q_end   = cu_q_lens[r + 1]
~~~

所以不同请求本来就可以拥有不同 query length，尾部也允许未使用的 padding capacity。DSpark 不需要新 attention kernel，需要补齐的是：

- `verify_lens -> query_lens -> cu_q_lens`。
- 固定二维 verify window 到 compact token buffer 的 gather。
- compact logits/hidden 到请求坐标的 scatter。
- compact cache locations 和 KV commit mask。

### 2.2 Confidence Head 本身不难

Confidence Head 只是小投影：

~~~text
feature = concat(draft_hidden, markov_embedding)
raw_confidence = linear(feature)
confidence = sigmoid(raw_confidence / temperature)
~~~

真正困难的是它后面的控制闭环：

~~~text
current confidence
    +--> device 上分配已经确定的预算
    +--> 异步发布到 host，供未来轮次选择预算和 JIT bucket
~~~

这里同时涉及因果性、异步执行、JAX 静态 shape、DP 对齐、KV 状态和硬件成本模型。

### 2.3 风险排序

1. `verify_len`、draft token、bonus token 和 KV commit 的长度语义。
2. 使用历史 confidence 决定预算时的时序正确性。
3. TPU 静态 shape 下 compact token bucket 和预编译策略。
4. compact gather/scatter、attention metadata 和 cache location。
5. TP/DP 下 global argmax、共同 bucket 和状态一致性。
6. Markov/Confidence Head 与权重加载。

### 2.4 Target Verify 核心逻辑不需要修改

Stage 1 已经证明固定宽度复用路径成立：

~~~text
DSparkWorker(DFlashWorker)
    -> 继承 _init_jit_target_verify()
    -> target model forward
    -> dflash_greedy_verify()
~~~

DSpark 与 DFlash 的 target verify 都是同一条 top-1 greedy chain。区别只在于 DSpark 的 `verify_width = gamma + 1`，而不是另一套接受算法。

Stage 2 动态 `verify_len` 仍不修改 target model forward 的算法逻辑，并把 `dflash_greedy_verify()` 拆成 logits argmax 与共享 prediction verify 两层。新增内容位于它们的边界上：

- Verify 前：planner 选择 budget，构造 compact `ForwardBatch`、metadata 和 cache locations。
- Verify 后：把 compact prediction/hidden scatter 回固定 `[batch, verify_width]` 逻辑视图。
- 调用共享 greedy verify 核心前：在每个请求的 cutoff 位置把候选 token 替换为不可能匹配的 sentinel，使原逻辑自然停止。
- Verify 后的状态更新继续消费原函数产生的 `accept_lens_out` 和 bonus token。

当前实现遵守这一边界：新增的是 adapter 和 layout plumbing，不是另一套 target acceptance 算法。

## 3. 目标与非目标

目标：

- 无需转换即可加载官方 Qwen3-8B DSpark checkpoint。
- 实现 vanilla Markov Head 和带 Markov embedding 的 Confidence Head。
- 复用目标模型 embedding、LM Head 和 Qwen3 多层 hidden capture。
- greedy 输出与 target-only greedy 逐 token 完全一致。
- 支持官方 `gamma=7` 的长度约定。
- 使用 lagged confidence 做非前视预算决策。
- 使用 current confidence 在 device 上分配已确定预算。
- 使用现有 RPA 完成 variable-length compact verify。
- 用少量预编译 token buckets 控制 JIT 数量。
- 支持现有 DFlash 覆盖的 TP/DP 配置。

第一版非目标：

- 非 greedy sampling。
- grammar、logprob 和 logits penalty。
- target/draft LoRA。
- Qwen3 dense 以外的模型族。
- 在线 SPS 探索或在线训练 Confidence Head。
- 新 attention kernel。
- prefill/decode disaggregation。

不支持的组合必须在初始化或请求校验时明确报错。

## 4. 长度 Contract

| 名称 | 含义 | Qwen3-8B 值 |
|---|---|---:|
| `gamma` | Markov Head 生成的草稿 token 数 | 7 |
| `draft_width` | 草稿 Transformer 输入位置数 | 7 |
| `verify_width` | 目标模型最大 query 行数 | 8 |
| `verify_len[r]` | 请求实际验证行数，包含 anchor | 1～8 |
| `extra_budget` | 超出逐请求最低行数的总预算 | 动态 |
| `M` | 一个 DP rank 的实际 target query 总数 | `sum(verify_lens)` |
| `M_bucket` | executable 的静态 token capacity | 分桶 |

草稿输入：

~~~text
[anchor, mask, mask, mask, mask, mask, mask]  # draft_width = 7
~~~

七个 hidden states 通过共享 target LM Head 和 Markov Head 生成：

~~~text
[draft_1, draft_2, ..., draft_7]
~~~

最大 target verify chain：

~~~text
[anchor, draft_1, ..., draft_7]              # verify_width = 8
~~~

如果 `verify_len=4`，target 处理：

~~~text
[anchor, draft_1, draft_2, draft_3]
~~~

最多提交前三个 accepted draft token，最后一个有效 target prediction 作为 bonus/output token。

当前 JAX DFlash 把同一个 `block_size` 同时用于 draft input width 和 target verify width；DSpark 中二者分别是 `gamma` 和 `gamma+1`，因此必须拆开。

## 5. 可复用代码

### 5.1 Qwen3 目标模型

文件：`python/sgl_jax/srt/models/qwen3.py`

可复用 `get_embed_and_head()`、auxiliary hidden capture 和 target forward。建议把 `set_dflash_layers_to_capture()` 泛化或增加 DSpark alias。checkpoint 的 `target_layer_ids=[1,9,17,25,33]` 需要沿用当前 layer output 的 `+1` 对齐，并用 parity test 确认。

### 5.2 DFlash 草稿模型

文件：`python/sgl_jax/srt/models/dflash.py`

可复用五层 Qwen3 风格 draft backbone、target hidden projection、双向 block attention、draft KV materialization、attention/MLP weight mapping 和 TP sharding。

建议新增：

~~~text
DSparkDraftModel(DFlashDraftModel)
~~~

只增加 Markov Head、Confidence Head 和 checkpoint mapping。

### 5.3 DFlash Worker

文件：`python/sgl_jax/srt/speculative/dflash_worker.py`

可复用独立 draft worker、共享 allocator、embedding/LM Head 共享、DP/TP batch、overlap relay、fused JIT、KV materialization 和 precompile 基础设施。

尤其是 `_init_jit_target_verify()`、`_run_jit_target_verify()` 和 `dflash_greedy_verify()` 应保持算法语义不变。DSpark Stage 1 已经通过继承 `DFlashWorker` 复用了这三部分。

不建议复制整个 worker。先抽取 block-draft 公共 helper，DSpark 只维护算法特有状态。

### 5.4 FlashAttention metadata 与 RPA

相关文件：

- `python/sgl_jax/srt/layers/attention/flashattention_metadata.py`
- `python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention_v3.py`

底层 `_build_metadata_from_paged_layout()` 已支持 `query_lens/cu_q_lens`。当前 `build_target_verify_metadata()` 仍按统一 `draft_width` 构造 query length。只需让它接受逐请求 `query_lens` 并穿过 fused verify 路径。

## 6. 端到端数据流

### 6.1 Prefill

1. Target Qwen3 执行普通 prefill。
2. 捕获配置的五层 target hidden states。
3. DFlash projection 将多层 hidden 转为 draft context。
4. 为各 draft layer materialize KV。
5. Target 产生 bonus token，作为下一轮 anchor。

### 6.2 Draft

1. 构造 `[anchor, mask x 6]`。
2. 共享 target embedding。
3. 五层 draft backbone 并行处理七个位置。
4. 共享 target LM Head 得到 `[batch,gamma,vocab]` base logits。
5. Markov Head 静态串行七步，生成七个 draft tokens。
6. Confidence Head 产生七个条件接受概率。

### 6.3 动态验证

1. Lagged confidence 在 host 侧选择总 budget 和 `M_bucket`。
2. Current confidence 在 device 侧把固定 budget 分配给请求。
3. 生成 `verify_lens`。
4. compact token、position 和 cache location。
5. 既有 target model forward 使用 variable `cu_q_lens`，模型逻辑不变。
6. Target logits/hidden 映射回固定请求坐标。
7. Cutoff adapter 屏蔽 `verify_len` 以外的候选，再调用从 `dflash_greedy_verify()` 复用的 token-ID verify 核心。
8. Current confidence 异步发布给未来轮次。

~~~text
上一轮 target hidden + bonus
             |
             v
     draft KV materialization
             |
             v
   并行 DSpark draft backbone
             |
     base logits + hidden
             |
             v
       七步 Markov scan
             |
      tokens + confidence
        |              |
        |              +--> 异步 D2H，供未来预算
        v
current confidence 分配固定 budget
             |
             v
       verify_lens[batch]
             |
             v
 compact gather [dp,M_bucket]
             |
             v
 ragged_paged_attention_v3
             |
             v
 accept / scatter / KV commit
~~~

## 7. 模型设计

### 7.1 类结构

新增 `python/sgl_jax/srt/models/dspark.py`：

~~~text
VanillaMarkovHead
DSparkConfidenceHead
DSparkDraftModel(DFlashDraftModel)
Qwen3DSparkModel(DSparkDraftModel)
EntryClass = [DSparkDraftModel, Qwen3DSparkModel]
~~~

若 registry 只接受单个类，则注册 `Qwen3DSparkModel`，draft worker 显式 override model class。

### 7.2 Checkpoint 参数

初始化时校验：

~~~text
architectures               = [Qwen3DSparkModel]
num_hidden_layers           = 5
hidden_size                 = 4096
target_layer_ids            = [1, 9, 17, 25, 33]
block_size                  = 7
markov_rank                 = 256
markov_head_type            = vanilla
confidence_head_with_markov = true
mask_token_id               = 151669
~~~

影响长度或权重结构的字段不匹配时默认拒绝启动。

### 7.3 Markov Head

权重：

~~~text
markov_w1: [vocab_size, markov_rank]
markov_w2: [vocab_size, markov_rank]
~~~

逻辑：

~~~text
prev = anchor
for k in range(gamma):
    markov_embed = W1[prev]
    bias = markov_embed @ W2.T
    corrected_logits = base_logits[:, k, :] + bias
    token = global_argmax(corrected_logits)
    prev = token
~~~

串行部分只有七步，昂贵的五层 Transformer 仍然并行。实现时比较静态展开和 `jax.lax.scan`；`gamma` 必须是 static JIT 参数。

### 7.4 Confidence Head

~~~text
feature_k = concat(draft_hidden_k, markov_embed_k)
raw_k = feature_k @ confidence_weight + confidence_bias
c_k = sigmoid(raw_k / sts_temperature[k])
~~~

checkpoint 权重为 `[1, hidden_size+markov_rank]` 和 `[1]`。

### 7.5 权重和 sharding

| 权重 | 变换 | 第一版 sharding |
|---|---|---|
| `markov_w1 [V,R]` | 无 | replicated |
| `markov_w2 [V,R]` | transpose 为 `[R,V]` | `(None,tensor)` |
| `confidence proj [1,H+R]` | transpose | replicated |
| `confidence bias [1]` | 无 | replicated |

W1 在官方配置 BF16 下约 78 MB，第一版复制可降低复杂度。Markov W2 与共享 LM Head 应使用一致的 vocab sharding；argmax 必须有全局 TP 语义。

checkpoint 中重复的 `embed_tokens.weight` 和 `lm_head.weight` 应显式 skip，继续使用 target 的 live 参数，并做 weight coverage assertion。

## 8. Confidence 概率语义与时序

定义条件接受概率：

~~~text
c[r,k] = P(draft k 被接受 | draft 0..k-1 已接受)
~~~

前缀存活概率：

~~~text
a[r,k] = product(c[r,0:k+1])
~~~

因此每个请求内部 `a[r,0] >= a[r,1] >= ...`。额外验证一行的期望收益，就是到达该行所需的 prefix survival。

STS 使用 Sequential Temperature Scaling。Confidence Head 在运行时保留未经过
sigmoid 的 `confidence_logits`，统一计算
`confidence = sigmoid(confidence_logits / temperature)`；temperature 为 1 时就是
未校准概率。STS capture 保存 raw logits 和实际 `prefix_mask`，不再保存概率后通过
`logit()` 反解。采集必须关闭 tuned config 并走 fixed verify-all，否则 ragged cutoff
会让后缀标签受到当前调度策略截尾，不能再作为无偏 calibration 样本。

离线拟合从左到右进行。位置 `k` 固定前面已经选择的 temperature，针对候选
`T_k` 计算累计 survival
`a_k = a_{k-1} * sigmoid(confidence_logits[k] / T_k)`，并选择使 `a_k` 相对实际
prefix-survival label 的 ECE 最小的温度。这与论文及 SGLang GPU 实现一致；没有
STS 时使用 1.0，SPS 调度仍可退化到 verify-all。

不能等待当前完整 confidence 再选择当前 executable，否则会导致 device-to-host 同步。所有运行模式统一固定：

~~~text
capacity_lag = 2
M_bucket[t]  = planner(C[t-2], R[t], context[t], T(R,M))
verify_lens[t] = allocator(C[t], M_bucket[t])
~~~

其中 `t` 是每个请求自己的 decode round，不是 scheduler 的全局 batch/forward ID。请求被 filter、merge 或暂时未调度时，全局 forward ID 会跳变，但该请求的 decode round 只在它真正参与一次 decode 时递增。

- `C[t-2]`：只决定当前总 budget 和静态 executable bucket。
- `C[t]`：只在已经固定的 budget 内决定预算分给哪些请求，全程留在 device。
- `C[t-1]`：明确不用于容量选择，避免 overlap/non-overlap 产生两套时序。

### 8.1 Capacity Relay 的存储 Contract

现有 `DFlashRelayBuffers` 只携带下一轮 seed token 和 sequence length，不能承担 capacity relay。新增独立结构：

~~~text
DSparkConfidenceRelayDevice:
    confidence[ring_size,dp_size,req_pool_size,gamma]
    slot_generation[ring_size,dp_size,req_pool_size]
    source_decode_round[ring_size,dp_size,req_pool_size]

DSparkConfidenceRelayHost:
    confidence[ring_size,dp_size,req_pool_size,gamma]
    slot_generation[ring_size,dp_size,req_pool_size]
    source_decode_round[ring_size,dp_size,req_pool_size]
    ready[ring_size,dp_size,req_pool_size]
~~~

固定 `ring_size=3`。容量只读取两轮前的数据，但三槽可以避免当前 `C[t]` 发布时覆盖仍在被 host 读取的 `C[t-2]`，也给异步 D2H copy 留出额外一轮余量。ring slot 使用：

~~~text
write_slot = source_decode_round % 3
read_slot  = (current_decode_round - 2) % 3
~~~

`ReqToTokenPool` 增加 host-side `slot_generation[req_pool_size]`。每次把一个新请求分配到空闲 `req_pool_idx` 时 generation 加一，并把 `(req_pool_idx,slot_generation)` 带入 `ModelWorkerBatch`；chunked prefill 复用同一请求 slot 时不能增加。这样旧请求即使恰好在两轮前写过同一个 slot，也不能污染新请求。

### 8.2 Publish、读取与永不等待原则

Draft round `t` 生成校准后的 `C[t]` 后执行两条独立路径：

1. device hot path 立即用 `C[t]` 和已选 `extra_budget[t]` 生成 `verify_lens[t]`。
2. confidence、request pool index、slot generation 和 decode round scatter 到 device relay，并调用 `copy_to_host_async()`；后台 publisher 线程等待 copy 完成后更新 host relay。

Scheduler/worker 在准备容量时只读取已经 materialized 的 host relay，禁止对 confidence future 调用同步 `device_get()`：

~~~text
expected_round = current_decode_round - 2
slot = expected_round % 3
valid = (
    expected_round >= 0
    and host.ready[slot,dp,req_pool_idx]
    and host.slot_generation[slot,dp,req_pool_idx] == current_slot_generation
    and host.source_decode_round[slot,dp,req_pool_idx] == expected_round
)
survival = cumprod(host.confidence[slot,...]) if valid else ones(gamma)
~~~

后台 copy 未及时完成时不能等待；该请求本轮直接按 stale fallback 处理。第一次和第二次 decode、刚进入 decode 的新请求、slot generation 不匹配、ring tag 不匹配都属于正常 stale，不应告警刷屏，只累计 metric。

### 8.3 Batch、DP 与生命周期

容量选择按当前 batch 请求重新 gather，不保存 batch-row 状态：

1. 使用当前 `req_pool_indices/slot_generations/decode_rounds` 从 host relay gather `C[t-2]`。
2. 每个 DP rank 根据自己的 `R`、context bucket 和 survival 选择 `(local_M_bucket,local_extra_budget)`。
3. 所有 rank 对 `local_M_bucket` 取 max，得到共同静态 `M_bucket`；各 rank 保留自己的 `local_extra_budget`，多余容量为 padding。
4. current confidence 在 device 上按各 rank 的 local budget 生成 `verify_lens`。

Filter、merge 和请求顺序变化不需要搬运 relay 数据，因为 relay 始终以 `(dp_rank,req_pool_idx,slot_generation)` 寻址。请求结束时无需清零大 buffer；下一次 slot allocation 增加 generation 即完成逻辑失效。server reset/clear 时 generation 全量递增或显式清空 ready/tag，不能只重置 free list。

建议暴露以下观测指标：

~~~text
dspark_capacity_relay_hit
dspark_capacity_relay_stale_warmup
dspark_capacity_relay_stale_generation
dspark_capacity_relay_stale_not_ready
dspark_selected_token_bucket_per_dp
dspark_actual_verify_tokens_per_dp
dspark_ragged_padding_ratio
~~~

### 8.4 实现边界与落地顺序

`capacity_lag` 不提供 CLI 参数，固定常量为 2，避免线上出现未经 profile 的时序变体。建议按以下提交边界实现：

1. `ReqToTokenPool` 增加 slot generation，并把 generation/decode round 带到 `ModelWorkerBatch`；先用 slot-reuse 单测锁定生命周期。
2. 在 `relay_buffer.py` 新增 DSpark confidence device/host relay，不修改现有 DFlash state relay 的结构和语义。
3. DSpark draft JIT 发布 `C[t]`；后台 publisher 只负责异步 materialize host snapshot，scheduler thread 永不等待。
4. `_build_target_verify_plan()` 从 host relay gather `C[t-2]` 替换当前 `survival=1` bootstrap；current-confidence allocator 和 compact ragged verify 保持不变。
5. 增加 warmup、not-ready、request churn、filter/merge、slot reuse、DP unequal batch 和 ring wraparound 测试，再进行 TPU overlap timeline profile。

## 9. SPS 的定义和计算

### 9.1 SPS 是什么

SPS 是 `steps per second`：

~~~text
SPS = completed_decode_steps / elapsed_seconds
SPS = 1 / step_time
~~~

它不是直接的 token throughput。

### 9.2 PyTorch 参考实现

一维表保存：

~~~text
sample_batch_tokens  = [m0,m1,...]
sample_steps_per_sec = [s0,s1,...]
~~~

运行时：

~~~text
batch_tokens = num_requests + extra_budget
idx = bucketize(batch_tokens, probes, right=True) - 1
sps = sample_steps_per_sec[idx]
~~~

参考实现也支持 additive step-time：

~~~text
T(R,M) = bias + alpha(R) + theta(M)
~~~

`alpha` 和 `theta` 由离线 probes 线性插值。

### 9.3 SPS 依赖哪些参数

SPS 不是仅靠 FLOPs 在线推导，而来自目标部署环境实测，至少依赖：

- TPU 型号、topology。
- JAX/XLA/libtpu 版本。
- target/draft revision。
- dtype、quantization。
- TP/DP mesh。
- request batch bucket。
- verify token bucket。
- context length 和 page 分布。
- KV page size。
- attention backend 与 Pallas tuning。
- overlap 和 fused KV 模式。

这些字段必须写入 profiling manifest；环境发生实质变化时旧表默认失效。

### 9.4 TPU 为什么按 bucket 建表

XLA dense operator 按静态 shape 执行。即使 RPA 跳过无效 query，QKV、MLP 和 LM Head 仍主要受 `M_bucket` 影响。

因此初版用：

~~~text
T(R_bucket,M_bucket)
~~~

而不是只使用实际 `sum(verify_lens)` 的连续曲线。

### 9.5 如何 profile

1. 对每个 `(R_bucket,M_bucket)` 预编译。
2. 构造代表性 context/page 分布。
3. Warm up，排除编译和首次运行。
4. 连续运行足够多 decode steps。
5. 在已有同步边界等待关键 device 输出 ready。
6. 记录完整 device step time，不记录异步 Python dispatch latency。
7. 保存中位数、高分位数和完整环境 manifest。

## 10. 总验证预算公式

默认 `min_verify_len=1`，一个 DP rank 有 `R` 个请求。

~~~text
base rows = R
base expected output = R
survival = cumprod(history_confidence, axis=-1)
~~~

将所有 survival 展平降序为 `s1 >= s2 >= ...`。额外预算为 `m` 时：

~~~text
tau(m) = R + sum(s1..sm)
M(m) = R + m
~~~

legacy 一维 SPS bootstrap 可写成：

~~~text
theta(m) = tau(m) * SPS(M(m))
~~~

当前二维 step-time 表使用：

~~~text
theta(m) = tau(m) / T(R,M(m))
~~~

选择 `theta` 最大的 `m`。

TPU bucket 版本对每个 bucket `b`：

~~~text
m_b = min(b-R, R*gamma)
tau_b = R + sum(top_m_b(history_survival))
theta_b = tau_b / T(R_bucket,b)
~~~

选择最大 `theta_b` 对应的 `extra_budget` 和 `M_bucket`。

若配置 `L_min/L_max`：

~~~text
B0 = R * L_min
tau0 = R + sum(a[r,k] for k in [0,L_min-2])
selectable = a[r,k], k in [L_min-1,L_max-2]
m_b = min(b-B0, R*(L_max-L_min))
~~~

不能对全部位置排序后再 clamp，否则预算可能被分给最终被截掉的位置。

## 11. Current Confidence 如何生成 `verify_lens`

Host 已经选定 `extra_budget` 和 `M_bucket`。Device 只负责分配。

~~~text
survival = cumprod(current_confidence, axis=-1)
flat = reshape(survival, [dp,bs*gamma])
rank = descending_rank(flat)
selected = rank < extra_budget_per_rank
selected_extra = segment_sum(selected, request_index)
verify_lens = 1 + selected_extra
~~~

`jax.lax.top_k` 的 `k` 必须 static，不能把动态 budget 直接作为 `k`。可对静态最大候选集排序/rank，再动态比较 `rank < budget`；或者使用固定最大 k 的 `top_k` 后做 mask，二者需比较 HLO。

因为请求内 survival 单调不增，全局 top-k 自动保持每个请求的前缀结构。

Tie-break 固定为：

~~~text
survival descending
position ascending
request index ascending
~~~

## 12. Compact Ragged Verify

逻辑二维窗口：

~~~text
verify_ids_2d: [dp,bs_bucket,verify_width]
positions_2d:  [dp,bs_bucket,verify_width]
cache_loc_2d:  [dp,bs_bucket,verify_width]
verify_lens:   [dp,bs_bucket]
~~~

压紧为：

~~~text
compact_input_ids: [dp,M_bucket]
compact_positions: [dp,M_bucket]
compact_cache_loc: [dp,M_bucket]
query_lens:        [dp,bs_bucket]
cu_q_lens:        [dp,bs_bucket+1]
compact_to_row:   [dp,M_bucket]
compact_to_pos:   [dp,M_bucket]
valid_token_mask: [dp,M_bucket]
~~~

每个 DP rank 的 `cu_q_lens` 必须从 0 重新累计。JAX metadata 内部可继续 flatten。

`build_target_verify_metadata()` 增加可选 `query_lens`：DFlash 继续传统一 `draft_width`，DSpark 传 `verify_lens`。

仅让 RPA ragged 而不压缩 dense token dimension，只能节省 attention 内部工作。要节省 QKV、MLP 和 LM Head，必须同时使用更小的静态 `M_bucket`。

## 13. 复用 Target Verify、KV 与 Hidden Commit

Compact logits 先做 argmax，只把 token IDs scatter 到 `[dp,bs_bucket,verify_width]`。

Target model forward 和 greedy acceptance 不做算法修改。动态路径不应把 compact full-vocab logits scatter 成 `[batch,verify_width,vocab]`，否则会产生很大的临时张量。建议把当前函数按无语义变化的方式拆成：

~~~text
dflash_greedy_verify(draft_token, target_logits)
    -> argmax(target_logits)
    -> greedy_verify_predictions(candidates, target_predict)

DSpark compact path
    -> argmax(compact_logits)
    -> scatter compact token IDs
    -> greedy_verify_predictions(candidates, target_predict)
~~~

其中 `greedy_verify_predictions()` 直接搬用当前 `dflash_greedy_verify()` 中的 match、cumprod、bonus selection 和 `accept_lens_out` 计算。这是接口抽取，不是修改 target verify 逻辑。

Adapter 构造：

~~~text
target_predict_2d: [dp,bs_bucket,verify_width]
verify_candidates: [dp,bs_bucket,verify_width]
~~~

对 `verify_len=L < verify_width` 的请求，在 target forward 完成后设置：

~~~text
verify_candidates[r,L] = -1
~~~

这里的 `-1` 只用于 prediction/candidate 比较，不进入模型 embedding。它使原有比较：

~~~text
matches = candidates[:,1:] == target_predict[:,:-1]
~~~

在第 `L-1` 个 match 处必然停止。若之前的草稿全部匹配，共享逻辑会选择 `target_predict[r,L-1]` 作为 bonus，正好对应动态窗口最后一个有效 prediction。`L=verify_width` 时不需要 sentinel。

每个请求：

1. 只在 `verify_len` 内比较 target prediction 和 draft token。
2. 第一个 mismatch 停止。
3. 最后一个有效 target prediction 作为 bonus/output。
4. 生成 accepted length、commit length 和 new sequence length。

Target KV 可以写入预留物理 slot，但只有 committed 位置能进入 request-to-token mapping。Padding location 不得污染 live cache。

Compact hidden 可以先 scatter 回现有固定逻辑视图，再让当前 KV materialization 根据原有 `accept_lens_out` 选择 committed rows。Rejected、scheduler-trimmed 和 padding 位置都不写。优先复用当前 DFlash `_mask_draft_kv_writes()` 和 sequence-state 更新语义。

Host 可继续为每请求预留 `verify_width` 个候选 slot，使 allocator 不依赖动态 `verify_lens`。

Greedy lossless 要求最终 token sequence 与 target-only greedy 完全一致。Confidence 只能决定一次验证多少行，不能强制接受。

## 14. JIT、TP 与 DP

Draft executable 主要按 `padded_bs_per_dp` 分桶，`gamma` 固定。

Target verify cache key 建议包含：

~~~text
per_dp_bs_bucket
per_dp_verify_token_bucket
verify_width
page-index capacity
DP/TP mesh
~~~

默认 token buckets 可从 `R,2R,4R,...,R*verify_width` 开始，再按 TPU 友好倍数对齐，并由 profile 数据裁剪。

没有精确 bucket 时向上取最小可容纳 bucket；没有更大 bucket 时 verify-all。默认禁止无限运行时编译。

TP 要求共享 LM Head 和 Markov W2 的 vocab layout 一致，global argmax 语义一致，各 TP rank 的 `verify_lens` 一致。

DP 下全局 shape 是 `[dp_size,M_bucket,...]`。第一版各 rank 独立算期望 bucket，再取最大共同 bucket；每个 rank 在自己的 segment 内 compact。

## 15. 配置与 Fallback

新增 `SpeculativeAlgorithm.DSPARK` 和 `is_dflash_family()`。

内置调优表采用与 TPU kernel tune config 相同的精确 key 查找。用户只需启用：

~~~text
--speculative-algorithm DSPARK
--speculative-draft-model-path <path>
--enable-dspark-tuned-config
~~~

key 至少包含 target/draft checkpoint ID 与 revision、TPU 型号和 device 数、dtype/quantization、TP/DP、`gamma`、page size、attention backend 和 overlap 模式。SPS profile 内部再按 context bucket 选择。只有精确匹配才启用；路径 `/models/Qwen3-8B` 与 HF ID `Qwen/Qwen3-8B` 会规范化为相同 basename。

### 15.1 Tuned Config 命中后的 Ragged Verify 数据流

内置表不是直接返回某个固定 `verify_len`。它提供两类信息：

- STS temperature：把 Confidence Head 的输出校准成可比较的逐位置条件接受概率。
- SPS/step-time profile：估算不同静态 target token bucket 的执行成本。

运行时必须先选择整个 DP rank 共用的物理 bucket，再在这个 bucket 内为每个请求分配逻辑长度：

~~~text
tuned key exact hit
        |
        v
calibrate confidence with STS
        |
        +------------------------------+
        | lagged confidence            | current confidence
        v                              v
score SPS points and choose M_bucket   distribute extra_budget
                                       |
                                       v
                              verify_lens[request]
                                       |
                                       v
compact ids / positions / cache slots
                                       |
                                       v
query_lens + cu_q_lens -> target ragged verify
                                       |
                                       v
scatter predictions -> greedy acceptance -> KV/hidden commit
~~~

具体步骤如下：

1. 启动时根据实际部署参数构造 `DSparkTunedKey`。只有 exact hit 才保存 `DSparkTunedConfig`；miss 保持固定 verify-all。
2. draft forward 对每个位置应用 STS，得到 `c[r,k]`，再计算 `a[r,k]=cumprod(c[r,:])`。lagged survival 只用于选择未来 step 的 executable，避免 device-to-host 同步。
3. planner 取能覆盖当前最大 context 的最小 SPS profile。对每个候选 `M_bucket`，令 `extra_cap=min(M_bucket-R,R*(verify_width-1))`，用历史 survival 的前 `extra_cap` 项计算预期产出，再除以表中的 `median_step_time_ms`。得分最大的点决定 `M_bucket` 和 `extra_budget`。
4. 若启用 DP，每个 rank 先独立提出 bucket，再 collective max 得到所有 rank 共用的 `M_bucket`；每个 rank 仍按自己的 current confidence 分配预算，剩余位置只做静态 padding。
5. device 端对 current survival 做稳定排序，以 `rank < extra_budget` 生成 prefix-preserving mask，最终得到 `verify_lens[r]=1+selected_extra[r]`。这里 `sum(verify_lens) <= M_bucket`，而不是要求所有请求采用相同长度。
6. 按 `verify_lens` 压紧 draft token、position 和 cache location，并生成 `query_lens=verify_lens`、逐 DP rank 从零累计的 `cu_q_lens`、`compact_to_row/pos` 以及 padding mask。target forward 的静态 token shape 是 `M_bucket`，RPA 看到的有效 query 则是真实 ragged lengths。
7. target logits 只做 compact argmax；将 token ID scatter 回固定逻辑窗口后，用 sentinel 截断 `verify_len` 外的比较，复用现有 greedy acceptance。只有 accepted/bonus 对应的位置才能更新 request-to-token mapping、KV 和 hidden state。

上述流程中有两个不能混淆的长度：

~~~text
M_bucket                 = JIT executable 的每 DP rank 静态 token shape
sum(verify_lens[rank])   = 该 rank 本 step 的有效 ragged query 数
~~~

`M_bucket` 决定 QKV、MLP、LM Head 的实际计算桶；`verify_lens` 决定每个请求验证多少行以及 RPA 的 ragged 边界。只生成 `verify_lens` 而仍运行 verify-all dense shape，只能验证调度语义，不能获得完整性能收益。

当前代码已落地第一版完整执行链：exact-key tuned config、Qwen3-8B v7x8 STS/SPS 数据、SPS bucket planner、固定 `capacity_lag=2` confidence relay、current-confidence `verify_lens` 分配、compact gather/scatter、动态 `query_lens/cu_q_lens` 和 ragged target executable。它不会修改 checkpoint 的固定 `verify_width`，而是在 target JIT 内把固定逻辑窗口压入选中的静态 `M_bucket`。

Capacity relay 使用 `ReqToTokenPool.slot_generation` 和逐请求 `decode_batch_idx` 做身份校验。当前 `C[t]` 写入三槽 device ring 后调用异步 host copy；planner 只读取已经 materialize 且 generation/source-round 同时匹配的 `C[t-2]`，永不等待 future。前两轮、copy 未完成和 slot reuse 都逐请求回退到 `survival=1`。

当前 SPS 使用二维 `T(R,M)`：`R` 是每 DP 的 request bucket，`M` 是每 DP
的静态 compact token bucket。planner 选择能够覆盖实时请求数的最小 `R`
行，再只比较该行的 `M` 候选。实时请求数低于 `R` 时，多余行保持 padding，
不会因为 31/33 之类的 DP 瞬时不均衡回退到 verify-all。

Qwen3-8B 的内置 STS 温度已由 Falcon `exp-ksphva699n` 在 v7x-8 上用
GSM8K-500、raw confidence logits 和上述 Sequential STS 重新生成；完整拟合产物位于
`benchmark/dspark/tables/qwen3_8b_v7x8_sts.json`。v7x-8 的二维 SPS 原始表位于
`benchmark/dspark/tables/qwen3_8b_v7x8_sps_2d.json`。

部署时应预编译表中允许的 bucket variants，避免请求运行中出现意外编译。

高级预算参数在 planner 落地后再增加：

~~~text
--speculative-dspark-min-verify-len <int>
--speculative-dspark-max-verify-len <int>
--speculative-dspark-token-buckets <list[int]>
~~~

第一版强制 `topk=1`、`num_steps=1`、FA backend 和 greedy。

Fallback：

| 条件 | 行为 |
|---|---|
| 未启用 tuned config | verify-all |
| tuned config key miss | verify-all，并记录完整 miss key |
| 当前 context 超过所有 SPS bucket | verify-all |
| 无 STS | temperature=1.0，并告警 |
| confidence stale | survival=1.0 |
| 无精确 bucket | 向上取 bucket |
| 非 greedy/grammar/logprob | 拒绝请求 |
| 不支持的 Markov Head 或 gamma | 启动失败 |

所有 fallback 只能损失性能，不能损失 target correctness。

## 16. 测试计划

模型测试：config、weight coverage、Markov step、Confidence/STS、hidden/logits parity 和 exact greedy token。

调度测试：conditional-to-survival、预算 exhaustive case、stable tie-break、stale generation、DP common bucket 和边界长度。

Attention/layout 测试：`query_lens=[8,3,5,1]`、per-DP cumsum reset、compact RPA 对齐逐请求 reference、尾部 padding 和 KV 不污染。

端到端测试：target-only 等价、verify-all 与 compact 等价、batch filter/merge、slot reuse、overlap/no-overlap、TP/DP、page boundary、immediate/partial/full acceptance。

性能报告：draft latency、`(R_bucket,M_bucket)` verify latency、SPS、accepted tokens/step、output tok/s、ITL、padding ratio、预编译时间和 executable cache。

## 17. 实施阶段

### Phase 0：公共 Contract

注册算法，拆分三种长度，泛化 hidden capture 命名，抽取必要 DFlash helper。退出条件是 DFlash 不回退。

### Phase 1：模型与固定 Verify-all

实现 heads、加载 checkpoint、生成七个 draft tokens、固定验证八行，并与 PyTorch 对齐。

### Phase 2：Confidence 与逻辑 Cutoff

增加 STS、lagged relay、SPS planner、budget 和 `verify_lens`；物理 target shape 暂时 verify-all，通过 candidate sentinel adapter 验证动态长度的 acceptance 语义，继续调用原有 `dflash_greedy_verify()`。这一阶段不宣称获得 ragged verify 的性能收益。

### Phase 3：Compact RPA

将 Phase 2 生成的 `verify_lens` 变成实际 ragged target workload：选择 `M_bucket`，compact token/position/cache，构造 `query_lens/cu_q_lens`，运行现有 target forward/RPA，把 logits/hidden 映射回固定逻辑视图，然后复用原 target verify 和状态提交路径。退出条件是 `sum(verify_lens) < R*verify_width` 时 target dense shape 确实下降，且输出与 verify-all greedy 完全一致。

### Phase 4：TPU Bucket Scheduler

离线 profile、选择 bucket、预编译 variants、实现 DP common bucket。要求在 trimmed workload 上 target step cost 实际下降。

### Phase 5：生产化

RPA tuning、request churn、slot reuse、nightly accuracy/performance 和部署文档。

## 18. 工作量

| 模块 | 估算 |
|---|---:|
| Contract/config | 2～3 天 |
| Markov/Confidence Head 和权重 | 3～5 天 |
| 固定 verify-all parity | 3～5 天 |
| Relay、STS、cutoff | 4～7 天 |
| Compact layout 与现有 RPA | 4～7 天 |
| Token bucket、profiling、DP policy | 5～8 天 |
| TPU tuning、E2E、稳定性 | 5～10 天 |

可用 greedy 版本约 3～5 工程周；生产级 TPU 版本约 4～6 工程周。

RPA 已支持 ragged，删除的是新 kernel 的工作量；compact gather/scatter、dense bucket、JIT key、KV commit、hidden mapping 和 DP 对齐仍需约 1～2 周。

## 19. 主要待定问题

1. Relay lag 使用一轮还是两轮？
2. Markov W1 是否需要 vocab sharding？
3. Base logits 一次 materialize 还是与 scan 交错以降低峰值内存？
4. 各 TPU 代际的默认 token bucket 网格？
5. SPS 使用二维查表还是 additive fit？
6. DP common bucket 如何权衡 padding 和公平性？
7. 新 checkpoint/revision 的 STS 与 SPS 何时允许进入内置 tune table？
8. Sampling 后续如何保持 rejection sampling 分布严格正确？

## 20. 预期文件

~~~text
python/sgl_jax/srt/models/dspark.py
python/sgl_jax/srt/speculative/dspark_worker.py
python/sgl_jax/srt/speculative/dspark_planner.py
python/sgl_jax/srt/speculative/dspark_sps.py
python/sgl_jax/srt/speculative/dspark_sts.py
python/sgl_jax/srt/models/qwen3.py
python/sgl_jax/srt/speculative/spec_info.py
python/sgl_jax/srt/layers/attention/flashattention_metadata.py
python/sgl_jax/srt/server_args.py
~~~

## 21. 参考资料

- DSpark 论文：<https://arxiv.org/abs/2607.05147>
- SGLang PyTorch 实现：<https://github.com/sgl-project/sglang/pull/30261>
- 官方 checkpoint：<https://huggingface.co/deepseek-ai/dspark_qwen3_8b_block7>
- Speculative 架构：`docs/architecture/09-speculative-decoding.md`
- DFlash 模型：`python/sgl_jax/srt/models/dflash.py`
- DFlash Worker：`python/sgl_jax/srt/speculative/dflash_worker.py`
- Target verify metadata：`python/sgl_jax/srt/layers/attention/flashattention_metadata.py`
- TPU RPA：`python/sgl_jax/srt/kernels/ragged_paged_attention/ragged_paged_attention_v3.py`
