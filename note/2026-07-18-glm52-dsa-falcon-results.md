# GLM-5.2 DSA Falcon 调试与真实权重 E2E 记录

## 结论

当前 `develop/glm52-dsa-falcon` 已在 Falcon v7x-32 上完成 GLM-5.2 BF16 真实权重的 DSA 最小闭环：

- checkpoint 完整：282 个 safetensors shard，共 `1,506,667,387,408` bytes。
- TP32 / DP1 / EP32、fused MoE、DSA Pallas kernel 能完成真实权重加载、prefill、chunked prefill、ragged batch、decode 和 HTTP response。
- short、257-token chunked、9/133-token ragged 三类请求均通过响应 schema、finite logprob、top-20 宽度和请求数量校验。
- DSA 重复运行逐位可复现：output IDs、生成 token logprob、top-20 token/logprob 全部完全一致。
- 与 FA baseline 比较时，所有生成 token IDs 完全一致，top-20 最低重合率为 `0.90 / 0.95 / 0.95`。
- 严格 `max generated-token logprob abs error <= 0.05` 门未通过；三类请求分别为 `0.0703125 / 0.171875 / 0.203125`。因此可以声明功能 E2E 正常，不能声明严格 0.05 logprob 精度门通过。
- 独立 PyTorch CPU golden 已覆盖 select、logical-to-physical mapping 和 sparse MLA；TPU Pallas 对 Torch FP32 的最坏 max abs 为 `0.0019080639`。
- 3072-token 真实权重请求已越过 `index_topk=2048`：3073 个 prefill/decode query 中 1025 个发生真实截断，selection/mapping validator 零失败。

短序列 DSA/FA 对照均小于 checkpoint 的 `index_topk=2048`，DSA 选中了完整因果 token 集；其中的差异来自 Pallas 在线 softmax、Top-K 返回顺序与 FA block kernel 之间稳定的 BF16 数值路径。长序列从 position 2048 开始不再与 dense attention 数学等价，因此只对 selection/mapping 与独立 sparse-MLA golden 做 correctness gate，不要求长序列 DSA logits 等于 dense logits。

## 代码与运行环境

- Workspace: `/Users/jiongxuan/workspace/sglang-jax/.worktrees/glm52-dsa-falcon`
- Branch: `develop/glm52-dsa-falcon`
- 实验 source revision: `bd8ffe3f0+overlay-491dfb1e+runner-50015d51`
- 模型: `zai-org/GLM-5.2`
- 模型目录: `/models/GLM-5.2`
- Falcon v7x-32 experiment: `exp-sff91uc6va`
- topology: 4 replicas，`2x2x4`，每 host 8 个 local TPU devices，共 32 devices
- JAX / jaxlib: `0.9.0 / 0.9.0`
- libtpu: `0.0.34`
- flax: `0.12.4`
- transformers: `4.57.6`

checkpoint 关键配置：

- architecture: `GlmMoeDsaForCausalLM`
- hidden size / layers: `6144 / 78`
- `kv_lora_rank / q_lora_rank`: `512 / 2048`
- `qk_nope_head_dim / qk_rope_head_dim`: `192 / 64`
- attention heads: `64`
- `index_head_dim / index_n_heads / index_topk`: `128 / 32 / 2048`
- `index_topk_freq / index_skip_topk_offset`: `4 / 3`
- routed experts / selected experts: `256 / 8`

## Reference 与 production kernel

当前 reference 是 `python/sgl_jax/srt/kernels/dsa/reference.py` 中的 `dsa_sparse_mla_reference`：

- 按 `physical_slots + selected_counts` gather MLA cache。
- score、softmax 和 PV 使用 FP32。
- 同时提供 Index-K cache 和 MLA KV cache 的 reference write，以及 logical-to-physical slot 转换。

TPU production 路径是 `python/sgl_jax/srt/kernels/mla/dsa/kernel.py` 中的 `dsa_decode_mla_attention_unchecked`：

- Top-K metadata 保留在 HBM，按 128-wide chunk DMA 到 VMEM。
- 每个 selected slot DMA 完整 `[2, 640]` packing group。
- VMEM 内使用 FP32 online softmax 累积，输出 BF16。
- 显式 mesh 下由 `DsaAttentionBackend` 使用 `jax.shard_map` 进入每个 addressable shard 的 Pallas manual region。

v7x-32 reference/Pallas smoke 已在四个 process 全部通过：

```text
GLM52_DSA_V7X32_REFERENCE_AND_SHARDMAP_OK process_id=0 local_devices=8
GLM52_DSA_V7X32_REFERENCE_AND_SHARDMAP_OK process_id=1 local_devices=8
GLM52_DSA_V7X32_REFERENCE_AND_SHARDMAP_OK process_id=2 local_devices=8
GLM52_DSA_V7X32_REFERENCE_AND_SHARDMAP_OK process_id=3 local_devices=8
```

该 smoke 使用 32-device 显式 `data=1 / tensor=32` mesh、TopK=2048 和 `valid_count=129`，以 `rtol=2e-2 / atol=1e-2` 比较 Pallas 与 JAX reference。

## Checkpoint 下载与 GCSFuse 加速

`/models` 最初没有 GLM-5.2。CPU-only Falcon manifest 因 CPU pool 无可调度资源而未开始下载，最终使用 v7x-8 pod 的 host CPU/network 下载，TPU 不参与：

- downloader experiment: `exp-q7odgo8q9x`
- transport: `hf-xet` + writable GCSFuse streaming write
- completion marker: `/models/GLM-5.2/_DOWNLOAD_COMPLETE`
- marker: `2026-07-18T16:26:20Z`

下载完成后校验 index 中 282 个 shard 均存在且非空，总大小为 `1,506,667,387,408` bytes。

最初真实加载的主要问题不是模型下载，而是 GCSFuse 读取配置：

- 旧 mount 使用 `file-cache:cache-file-for-range-read:true`。
- 每 host 每个 MoE projection 实际需要 64/256 experts，约 1.6106 GB，但 experts 在文件中稀疏分布，跨度约 15.3 GB。
- 旧路径每组读取约 64--77 秒，225 组预计超过 4 小时。

最终 experiment 使用：

```text
file-cache:cache-file-for-range-read:false
SGLANG_JAX_METADATA_SCAN_THREADS=32
SGLANG_MOE_RANGE_LOAD_WORKERS=32
SGLANG_JAX_SKIP_GCSFUSE_WARMUP=1
```

代码侧增加：

- 只读取 safetensors 8-byte prefix + JSON header 的并发 metadata scan。
- 对稀疏 expert ranges 做受控并发精确读取，并检查 short read。
- 允许跳过会触发大文件下载的 GCSFuse warm-up。
- `scripts/kernels/inspect_glm52_moe_layout.py` 用于检查 expert byte-range 分布。

冷缓存真实 run008：

| 阶段 | rank 2 耗时 |
| --- | ---: |
| 282 shard metadata | 22m09s（8 threads） |
| 1194 regular weights | 22m33s |
| 225 MoE groups | 4m55s |
| server healthy | 3162s，总计约 52m42s |

MoE 每组读取 1.6106 GB，冷缓存常见 `0.63--1.82s`，约 `0.89--2.55 GB/s`；旧路径为 64--77 秒。

同一 pod 热缓存的 FA run：

| 阶段 | rank 2 耗时 |
| --- | ---: |
| 282 shard metadata | <1s（32 threads） |
| 1194 regular weights | 10s |
| 225 MoE groups | 32s |
| server healthy | 181s，总计 3m01s |

## Fresh-node 启动、metadata sidecar 与 precompile

### 为什么看起来像“precompile 卡死”

原 runner 把 server stdout/stderr 全部重定向到 rank 本地文件，只在进程退出时复制到
artifact。Falcon 实时 stdout 因而只有 health-check 的 `curl` 失败，看不到 metadata、
权重加载和预编译进度。

`1845199aa` 增加了 rank-0 只读日志 monitor。它从 server log 第一行开始跟随，只镜像
以下阶段边界到 Falcon stdout，并在退出时按独立 process group 清理：

- metadata scan 或 sidecar hit；
- parallel weight loading 完成；
- MLA/Fused MLP absorb 完成；
- 每个 EXTEND/DECODE precompile variant 的进度；
- server ready。

fresh-node baseline 为 `exp-mzbjj4o3f6`（job `job-w8hms841fl`，artifact
`art-5195yimbtv`，source `05ad390e9`）：

| 阶段 | 耗时 |
| --- | ---: |
| 282 shard metadata header scan | 22m12s |
| 权重加载 | 27m14s |
| absorb | 3s |
| EXTEND 8/8 | 604s |
| DECODE 5/5 | 295s |
| server ready | 3904s（65m04s） |

因此服务并没有卡在 precompile：前 49 分钟主要是在 fresh GCSFuse 节点读取 metadata
和真实权重，之后 15 分钟才是有界的 13 个预编译 variant。

### Versioned metadata sidecar

`16d3a7f09` 增加 gzip-JSON sidecar：

```text
/models/GLM-5.2/sglang_jax.safetensors_metadata.v1.json.gz
```

sidecar 保存 tensor key、shape、dtype、shard 相对路径、byte offset/size，以及 shard
basename/size fingerprint。process 0 先加载并校验 sidecar；miss、过期或损坏时安全回退
到原并发 header scanner，最终仍广播同一种 `weight_info` 结构。

独立 review 进一步发现“合法 JSON 但内部内容损坏”仍可能误命中，随后在
`97f05a640 / f20b2426f / ec1061e92` 中补齐：

- payload 和 `weight_info` 的结构/非空校验；
- canonical `weight_info` SHA256；
- shard 名称、dtype、shape、`shape × dtype == byte_size`；
- `byte_offset >= 8` 和 `offset + size <= shard size`。

测试覆盖旧无 digest 文件、stale shard、digest mismatch、空 metadata、未知 shard，
以及重算 digest 后的非法 shape/offset/size。最终独立复审结论为无 Critical/Important。

由于 Falcon CPU pool 当时不可调度，sidecar 仍使用 v7x-8 pod 的 host CPU 生成，TPU
不参与计算。首版任务 `exp-oyhebgisn4` 证明方案可用；完整性版最终任务为：

```text
experiment: exp-9h2a7czdt2
job:        job-jbka9fa213
artifact:   art-8bnaxyq6q2
source:     ec1061e92
shards:     282
tensors:    59,585
bytes:      621,430
GCS SHA256: 8d2b8157b7468b649f828c1f315fd330106dfa6c272da62593b1b44852897373
metadata SHA256: 6c7e0fcb14f7cf3a5ea1c3d3785a99549beab0fd5bf1ddb80e66047cedb10b24
build:      11s
validation: 5s
```

### Fresh-node v7x-32 验证

验证任务使用 4×v7x-8、TP32/DP1/EP32、fused MoE、DSA、3072-token
`precompile_repeat`，模型挂载保持 read-only：

```text
experiment: exp-zeb34sbqwj
job:        job-si7hvggx15
artifact:   art-9v9fkmhhvg
source:     16d3a7f09
run id:     glm52-dsa-metadata-cache-20260719-001
```

四个 pod 均 succeeded。可比的 runner 时间线为：

| 阶段 | 耗时 | 说明 |
| --- | ---: | --- |
| server launch 到 weight loading | 约 39s | 包含模型初始化；sidecar 独立冷加载为 5s |
| 权重加载 | 26m48s | 与 baseline 27m14s 一致 |
| absorb | 37s | fresh-node 波动 |
| EXTEND | 614s | 8/8，约 74--85s/variant |
| DECODE | 290s | 5/5 |
| server ready | 2589s（43m09s） | runner 实测 |

与 baseline 的 3904s 相比减少 1315s（21m55s，约 33.7%），基本等于被消除的
22m12s metadata header scan。现在 fresh-node 的主要启动成本是实际权重读取
（1608s，约 62%）和显式预编译（904s，约 35%），不是未解释的卡死。

两次完全相同的 3072-token、one-token generation 结果：

| 指标 | first | repeat |
| --- | ---: | ---: |
| latency | 99.9987s | 99.9844s |
| output ID | `[198]` | `[198]` |
| output logprob | `-4.1875` | `-4.1875` |
| cache_miss_count | 1 | 1 |

严格 first/repeat gate 为：

```text
output_ids_equal=true
max_output_logprob_abs_error=0
max_shared_topk_logprob_abs_error=0
min_topk_overlap=1.0
schema_valid=true
passed=true
```

两次 latency 基本相同，`cache_miss_count` 也都为 1，所以没有“只发生在首轮”的冷编译。
这里验证的是单个 active request、top-20、3072-token DSA bucket；不能外推为所有
active batch occupancy、混合 top-k 宽度都已预编译。

## 真实权重问题与修复

第一次完整 regular-weight load 暴露 fused shared expert 映射错误：

```text
model.layers.3.mlp.shared_experts.gate_proj.weight
```

旧逻辑映射到不存在的 `model.layers.3.shared_experts...`。fused MoE 下已改为：

```text
gate_proj -> model.layers.<n>.mlp.w1_shared
up_proj   -> model.layers.<n>.mlp.w3_shared
down_proj -> model.layers.<n>.mlp.w2_shared
```

三者均 transpose，并使用 replicated `(None, None)` sharding。修复有单测覆盖，真实 checkpoint 的 1194 regular weights 和 225 MoE groups 随后全部加载成功。

## 真实 DSA E2E

运行配置：

```text
load_format=safetensors
dtype=bfloat16
parallelism=TP32 / DP1 / EP32
moe_backend=fused
attention_backend=dsa
page_size=128
context_length=4096
chunked_prefill_size=128
max_prefill_tokens=256
max_total_tokens=4096
```

主验证 run：`glm52-real-dsa-20260719-008`。

请求集：

1. short：4 input tokens，生成 2 tokens。
2. chunked：257 input tokens，跨越 128-token chunk 边界，生成 2 tokens。
3. ragged：两个并发请求，input 长度 9 和 133，各生成 2 tokens。

结果：

| 请求 | output IDs | E2E latency |
| --- | --- | ---: |
| short | `[5, 6]` | 155.18s |
| chunked | `[198, 220]` | 270.58s |
| ragged-9 | `[209, 210]` | 150.37s |
| ragged-133 | `[69, 1589]` | 150.37s |

这些 latency 包含首次真实执行的编译、同步和日志回传，不是 steady-state benchmark。runner 最终输出：

```text
GLM52_DSA_REAL_E2E_OK backend=dsa requests=3
```

## FA baseline 与精度结果

FA run：`glm52-real-fa-20260719-001`。除 `attention_backend=fa` 外，checkpoint、并行配置、dtype 和请求完全一致。

| 请求 | IDs 相同 | 最大生成 logprob 绝对误差 | 最低 top-20 overlap | 0.05 严格门 |
| --- | --- | ---: | ---: | --- |
| short | 是 | 0.0703125 | 0.90 | 失败 |
| chunked | 是 | 0.171875 | 0.95 | 失败 |
| ragged | 是 | 0.203125 | 0.95 | 失败 |

其他事实：

- 所有 output logprob 都是 finite。
- 每一行都返回 20 个 top-logprobs。
- 所有 response count 和 schema 均一致。
- 若仅作为观察性检查使用 `max logprob abs error <= 0.25`，三类请求通过；这不是严格门结论。

DSA 重复 run：`glm52-real-dsa-20260719-009`。run008 与 run009 使用零容差比较：

| 请求 | output IDs | 最大生成 logprob 误差 | 最低 top-20 overlap |
| --- | --- | ---: | ---: |
| short | 完全一致 | 0 | 1.0 |
| chunked | 完全一致 | 0 | 1.0 |
| ragged | 完全一致 | 0 | 1.0 |

所以当前数值偏移稳定、可复现，不是并发竞态或随机采样导致。

## Artifact

Falcon artifact root：

```text
/gcs/experiments/exp-sff91uc6va/artifacts/art-i9c3i0yvvw
```

关键目录：

```text
rank-0/glm52-real-dsa-20260719-008/
rank-0/glm52-real-fa-20260719-001/
rank-0/glm52-real-dsa-20260719-009/
```

run008 的 `precision/` 中包含：

- `*.comparison.json`：DSA vs FA，严格 0.05 门，预期记录为失败。
- `*.comparison-observational.json`：DSA vs FA，观察性 0.25 上界。
- `*.dsa-repeat.json`：run008 vs run009，零容差可复现性比较。

每个 run 目录还包含 request、response、schema report、summary、server command、run context 和各 rank server log。

## 复现命令

runner：

```bash
GLM52_DSA_RUN_ID=<unique-run-id> \
GLM52_ATTENTION_BACKEND=dsa \
GLM52_DSA_ROOT=/tmp/glm52-dsa-v7x32/sglang-jax \
GLM52_DSA_PYBIN=/opt/venv/bin/python3 \
GLM52_DSA_PORT=<unique-port> \
SGLANG_JAX_METADATA_SCAN_THREADS=32 \
SGLANG_MOE_RANGE_LOAD_WORKERS=32 \
bash scripts/kernels/run_glm52_dsa_v7x32_real_e2e.sh
```

切换 FA baseline 时只改：

```bash
GLM52_ATTENTION_BACKEND=fa
```

比较：

```bash
python scripts/kernels/compare_glm52_e2e_results.py \
  --candidate <dsa-response.json> \
  --baseline <fa-response.json> \
  --max-logprob-abs-error 0.05 \
  --min-topk-overlap 0.90 \
  --expected-topk-width 20 \
  --output <comparison.json>
```

## 独立 PyTorch CPU golden

后续验证增加了 `python/sgl_jax/srt/kernels/dsa/torch_reference.py`。该模块只依赖
PyTorch，不导入 JAX，也不被 serving 路径引用；它独立实现：

- GLM Indexer score、causal mask、Top-K logical ID 和 selected count。
- logical ID 到 paged physical slot 的映射与 padding 语义。
- 按 selected count gather KV、FP32 softmax/PV 的 sparse MLA。

CPU cross-framework matrix 覆盖 candidate length
`1/127/128/129/257/2047/2048/2049/3072/4096`。无 tie 的确定性 fixture 中，
Torch 与 JAX 的 selected count 和 logical ID 精确一致；physical slot 精确一致；
sparse MLA FP32 输出通过 `rtol=1e-5 / atol=1e-5`。额外覆盖 count
`0/1/127/128/129/2047/2048`、乱序 page、ragged count、padding、重复和非法
counted slot。

Falcon TPU 上又直接比较了 production Pallas kernel 与同一批 Torch FP32 fixture，
四个 process 全部完成：

| candidate length | Pallas vs Torch max abs |
| ---: | ---: |
| 127 | 0.0013763905 |
| 128 | 0.0019080639 |
| 129 | 0.0013590455 |
| 2047 | 0.0018008351 |
| 2048 | 0.0008915663 |
| 2049 | 0.0004876256 |
| 3072 | 0.0006479323 |
| 4096 | 0.0007084906 |

所有 selected count 都精确等于 `min(candidate_length, 2048)`；最坏 max abs 为
`0.0019080639`，低于既定 `atol=0.01`。因此整数 selection/mapping 要求精确一致，
但 BF16 Pallas 输出不要求与 FP32 Torch 逐 bit 一致。

### CPU oracle 与 TPU kernel 稳态耗时

CPU 数字来自 Apple M4、PyTorch 2.13.0、4 个 intra-op threads。每组先 warm-up，
下表为单 query eager median；这些数字只描述独立 correctness oracle，不代表 serving
吞吐。

| candidate length | Indexer selection | logical-to-physical Python mapping | sparse MLA FP32（2 heads） |
| ---: | ---: | ---: | ---: |
| 127 | 0.0853 ms | 0.2587 ms | 0.1328 ms |
| 128 | 0.0832 ms | 0.2731 ms | 0.1243 ms |
| 129 | 0.0824 ms | 0.2715 ms | 0.1278 ms |
| 2047 | 0.2925 ms | 4.1620 ms | 1.6950 ms |
| 2048 | 0.2491 ms | 4.0577 ms | 1.2594 ms |

TPU 数字来自 Falcon v7x-8 experiment `exp-6dqnqemqcd`，JAX 0.9.0，均为编译和
20 次 warm-up 后的 device-synchronized steady-state。benchmark 的 JIT 函数显式接收
3/5 个 runtime array arguments，避免把闭包数组降成 StableHLO constants 后发生常量
折叠。`H=2 / latent=512 / rope=64` 对应 GLM-5.2 在 TP32 下的单 shard 形状。

| workload | median | p99 | 结论 |
| --- | ---: | ---: | --- |
| sparse B1/H2/C160k/top-k2048，unsorted | 2.1947 ms | 2.3469 ms | 当前 production-shape prototype |
| sparse B1/H2/C160k/top-k2048，page-sorted | 2.1632 ms | 2.1844 ms | 没有实质改善 |
| sparse B8/H2/C32k/top-k2048 | 16.5070 ms | 17.0481 ms | 约 485 query-kernel/s |
| dense B1/H2/C160k workload baseline | 0.9916 ms | 1.0174 ms | 不是 production MLA-v2 kernel |

dense 行扫描相同 packed cache 的完整 160k context，但与 Top-K sparse 的 attention
domain 不同，只能作为 workload baseline。即使按这个有利于 dense 的实现口径，当前
sparse prototype 在 B1 下仍约慢 `2.21x`。因此本轮可以声明 correctness-first kernel
已跑通，不能声明 DSA kernel 已获得性能收益；下一阶段需要优化 Pallas gather、DMA 和
计算布局，而不是用 E2E 首次编译 latency 评价 kernel 性能。

### GPQA-Diamond E2E smoke 口径

评测脚本 `scripts/kernels/eval_glm52_gpqa_smoke.py` 复用了公开
`zai-org/glm-simple-evals` 的 GPQA prompt、`Random(0)` 子集和选项排列、末尾 1024
字符上的 `ANSWER: [A-D]` 直接提取语义。请求显式启用 thinking，采用
`temperature=1.0 / top_p=0.95`，并保存逐题 prompt、原始响应、usage、finish reason
和 latency。

官方 GLM-5.2 model card 声明 GPQA-Diamond `91.2`，reasoning task 最大生成长度为
`163,840` token；公开 harness 的正式 GPQA 配置是全量 198 题、8 repeats。本轮受
Falcon 原型吞吐和 8k context 限制，只运行固定 seed 的小样本和最多 4096 completion
token。因此结果只用于发现明显功能/精度退化，不是官方准确率复现，百分点差异也不能
脱离 Wilson 置信区间和截断率解读。

先运行的单题 2048-token 探针在 `2311.2s` 后以 `finish_reason=length` 结束：
`prompt=366 / completion=2048`，输出仍在推理，没有产生 `ANSWER:`。这不是可判定的
模型答错，而是一次明确的 budget truncation；它也证明当前 DSA E2E 路径能连续完成
2048 次 decode，但单请求端到端速度还远未达到实用 serving 水平。

随后在 Falcon v7x-32 experiment `exp-zhtkxigcy9` 上运行固定 `Random(0)` 的 4 题
smoke，TP32 / DP1 / EP32、fused MoE、DSA、`max_tokens=4096`、concurrency 4。服务
token pool 为 8192，因此实际最多同时 decode 2 题；整组 wall time 为 `4203.8s`
（70.1 分钟），请求 median 为 `2919.6s`（48.7 分钟）。

| index | domain | gold | extracted | finish | completion tokens | latency | score |
| ---: | --- | --- | --- | --- | ---: | ---: | ---: |
| 0 | Chemistry | D | - | length | 4096 | 4203.8s | 0 |
| 1 | Physics | D | - | length | 4096 | 2665.0s | 0 |
| 2 | Physics | C | C | stop | 967 | 375.5s | 1 |
| 3 | Biology | C | - | length | 4096 | 3174.1s | 0 |

按公开 harness 的 direct-regex 计分，raw accuracy 为 `1/4 = 25.0%`，Wilson 95% CI
为 `[4.56%, 69.94%]`，相对官方 `91.2%` 低 `66.2pp`。但 3/4 样本全部用满
4096 token 且仍在连贯推理，截断率 `75%`；唯一自然 `stop` 的样本为 `1/1` 正确，
completed-only Wilson 95% CI 为 `[20.65%, 100%]`。因此 `-66.2pp` 主要反映本轮
generation budget，不是可以归因给 DSA kernel 的精度差异；该 smoke 证明真实模型能
在 DSA 路径长时间连续 decode、自然结束时产生可提取正确答案，但不足以判断是否复现
官方 GPQA 准确率。

结果与日志保存于 artifact `art-1qqd0xsl54`：

```text
rank-0/gpqa-smoke-20260719/gpqa4-max4096-final.json
rank-0/gpqa-smoke-20260719/eval_glm52_gpqa_smoke-final.py
rank-{0,1,2,3}/gpqa-smoke-20260719/server-v3-rank{0,1,2,3}.log
```

## PR #1062 风格逐层 dump 与短序列定位

在 `jax.debug.callback + np.save` 的基本模式上增加了 disabled-by-default 环境开关、
component/layer/name/process filter、原子 NPY、JSONL manifest、跨 DSA/FA 的 forward
fingerprint、完成 marker、padding valid-row mask 和多 controller rank-symmetric callback。
比较器会丢弃取消请求留下的不完整 forward，并只在有效 token row 上计算
max/mean/p99/cosine/top-k 指标。

当前 Falcon debug experiment：

```text
experiment: exp-8m0q7a4og9
job:        job-5lnyeg3uih
artifact:   art-paz8z33izy
root:       /gcs/experiments/exp-8m0q7a4og9/artifacts/art-paz8z33izy
```

短序列 DSA/FA 对照分别为：

```text
rank-0/glm52-dsa-smoke-layerwise-validrows-20260719-h/
rank-0/glm52-fa-smoke-layerwise-validrows-20260719-i/
```

二者都完成 9 个可对齐 forward、162 个有效 tensor；取消请求的 17/15 个 partial
tensor 被 completion marker 正确过滤。token valid mask 和 embedding 完全一致。
代表层 `hidden_states_post_mlp` 的跨 forward 最坏值如下：

| layer | max abs | max mean abs | max p99 abs | min cosine |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.0009766 | 0.00001675 | 0.0001221 | 0.9999933 |
| 8 | 0.0151367 | 0.0003710 | 0.0024414 | 0.9996736 |
| 16 | 0.0898438 | 0.0031990 | 0.0117188 | 0.9956896 |
| 24 | 0.6796875 | 0.0077235 | 0.0344238 | 0.9926439 |
| 32 | 1.046875 | 0.0203119 | 0.0859375 | 0.9797363 |
| 40 | 1.828125 | 0.0511159 | 0.2050781 | 0.9688853 |
| 48 | 5.6875 | 0.0832045 | 0.3398438 | 0.9685246 |
| 56 | 23.375 | 0.1303098 | 0.5742188 | 0.9647017 |
| 64 | 21.75 | 0.1780528 | 0.7421875 | 0.9697885 |
| 72 | 10.0 | 0.2492665 | 1.0410156 | 0.9719749 |
| 77 | 50.625 | 0.3631349 | 1.984375 | 0.9792287 |

layer 0 只有 BF16 量级的小差异，之后随 78 层平滑累积，没有发现某一层突然出现的
selection 或 layout 错误。这与短上下文 `visible <= index_topk` 时 DSA/FA 关注同一
causal token 集、但使用不同在线 softmax/归约顺序的预期一致。

在 one-token HTTP 对照中，三类请求 output ID 仍完全相同：

| 请求 | output ID | output logprob max abs | top-20 overlap | 0.25/0.90 gate |
| --- | --- | ---: | ---: | --- |
| short | `[5]` | 0.07421875 | 0.80 | 失败 |
| chunked | `[198]` | 0.0625 | 0.95 | 通过 |
| ragged | `[209] / [69]` | 0.09375 | 0.95 | 通过 |

所以“功能与 token 结果一致”和“严格 logits/top-k 精度门通过”必须分开陈述；short
的 top-20 overlap 仍是已知未通过项。

## 3072-token 真实稀疏 E2E

长上下文 run 使用 source revision `ca0f380c2`、TP32/DP1/EP32、DSA、diagnostic
EP-MoE、128-token chunked prefill 和 one-token generation：

```text
rank-0/glm52-dsa-boundary3072-selection-20260719-k/
```

单个 3072-token prompt 的所有 query positions 都会被逐块执行，因此一次请求同时覆盖
2047/2048 边界和 3072 长度，不需要把四个边界 prompt 串行重跑。结果：

- 四个 rank 均 exit 0，runner 输出 `GLM52_DSA_REAL_E2E_OK backend=dsa requests=1`。
- response schema 和 finite logprob 通过；output ID 为 `[198]`，logprob 为 `-4.15625`。
- 端到端 latency 为 `2032.53s`，包含逐层 debug callback 和每块同步，不是性能基准。
- 25 个完成 forward = 24 个 prefill chunk + 1 个 position 3072 decode。
- 3073 个 active query 中 1025 个满足 `position + 1 > 2048`，真实进入 Top-K 截断。

正式 validator report：

```text
rank-0/glm52-dsa-boundary3072-selection-20260719-k/precision/boundary-selection.json
```

报告结果为 `passed=true`、`failures=[]`、positions `0..3072`，并确认 required
positions `2046/2047/2048/3071` 全部存在。每个有效 query 都满足：

- `selected_count == min(position + 1, 2048)`。
- counted logical ID 无重复、非负且不指向 future token。
- 未截断时 logical ID set 精确覆盖完整 causal set。
- counted physical slot 非负且无重复。
- logical/physical padding suffix 分别为 `-1/0`。

真实长序列报告检查 integration invariants；截断后 2048 个 token 的 score 排名正确性由
独立 Torch/JAX selection fixture matrix 覆盖，sparse MLA 数值由 Torch/Pallas length
matrix 覆盖。长序列 DSA 与 dense attention 本来选择不同 token 集，二者最终 logits
不作为数学相等 gate。

## 已知限制与下一步

1. short 的严格 logits/top-20 gate 未全部通过；逐层证据显示是从 layer 0 的 BF16 小误差平滑累积，不是已确认的 selection/layout 单点错误。
2. 3072-token run 证明 sparse integration 和 one-token response 正常，不等于任务级模型质量通过；仍需更多长 prompt、更多 decode token 和下游 eval。
3. 长序列 diagnostic run 使用 EP-MoE。带全量 debug dump 的 fused shared-expert BSE 512 路径需要约 96 MiB VMEM、超过单核 64 MiB；短序列无全量 debug 的 fused MoE E2E 已通过。
4. 当前 kernel 是 correctness-first 实现。逐块长跑约 80 秒且含 debug callback/同步，不能作为 production 性能数据；尚未完成 kernel 性能优化。
5. 可评估 `visible <= index_topk` 时保持 causal slot 顺序以收紧 DSA/FA 数值差，但必须先定义排序语义和模型质量 gate，不能仅为通过阈值而改变行为。

## 清理

验证结束后已完成资源清理：

- v7x32 验证实验 `exp-sff91uc6va` 已 abort；artifact payload 已在 abort 前确认并上传到上述路径。
- v7x32 layerwise/long-context 实验 `exp-8m0q7a4og9` 已 abort；`art-paz8z33izy` 中的 response 和 boundary report 已在 abort 前确认。
- v7x8 debug 实验 `exp-gq082n4nzb` 已 abort。
- 权重下载实验 `exp-q7odgo8q9x` 已成功结束。
- 旧实验 `exp-x9...` 和 range benchmark `exp-r78...` 已 abort，clone 实验 `exp-1vt...` 已失败退出，没有继续占用计算资源。

Falcon 对已 abort 实验的 artifact 链接可能显示为 `failed`，但本轮列出的结果文件已实际落入 GCS artifact root。runner 结束时出现的 `Killed` 是生成结果后清理 server process group；本轮成功 run 均已写入 `SUCCESS` 和全部 follower ACK，不代表 E2E 失败。
