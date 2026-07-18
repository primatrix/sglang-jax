# GLM-5.2 DSA Falcon 调试与真实权重 E2E 记录

## 结论

当前 `develop/glm52-dsa-falcon` 已在 Falcon v7x-32 上完成 GLM-5.2 BF16 真实权重的 DSA 最小闭环：

- checkpoint 完整：282 个 safetensors shard，共 `1,506,667,387,408` bytes。
- TP32 / DP1 / EP32、fused MoE、DSA Pallas kernel 能完成真实权重加载、prefill、chunked prefill、ragged batch、decode 和 HTTP response。
- short、257-token chunked、9/133-token ragged 三类请求均通过响应 schema、finite logprob、top-20 宽度和请求数量校验。
- DSA 重复运行逐位可复现：output IDs、生成 token logprob、top-20 token/logprob 全部完全一致。
- 与 FA baseline 比较时，所有生成 token IDs 完全一致，top-20 最低重合率为 `0.90 / 0.95 / 0.95`。
- 严格 `max generated-token logprob abs error <= 0.05` 门未通过；三类请求分别为 `0.0703125 / 0.171875 / 0.203125`。因此可以声明功能 E2E 正常，不能声明严格 0.05 logprob 精度门通过。

本轮序列长度均小于 checkpoint 的 `index_topk=2048`，DSA 选中了完整因果 token 集。当前 DSA/FA 差异不是稀疏截断，而是 Pallas 在线 softmax、Top-K 返回顺序与 FA block kernel 之间稳定的 BF16 数值路径差异。是否接受 `0.25` 的观察性上界，需要模型质量或产品侧另行确认，本文不把放宽阈值包装成严格精度通过。

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

## 已知限制与下一步

1. 当前 E2E 只覆盖到 257 input tokens，尚未覆盖 `context > index_topk` 的真实稀疏截断质量。
2. 0.05 logprob 严格门未通过。若要求收紧，优先增加逐层 attention output trace，比较 DSA Pallas、DSA FP32 reference 和 FA，定位误差累积起点。
3. 可评估“候选数小于等于 `index_topk` 时保持 causal slot 顺序”的数值路径，但必须先补 reference/排序语义测试，不能仅为让 E2E 阈值通过而改行为。
4. 当前 kernel 是 correctness-first 实现。DSA 首次执行 latency 明显高于 FA，尚未做 production 性能优化。
5. 真实模型质量仍需更长 prompt、更多 decode tokens 和任务级 eval；本轮只证明最小功能闭环及短序列 logits 一致性范围。

## 清理

验证结束后已完成资源清理：

- v7x32 验证实验 `exp-sff91uc6va` 已 abort；artifact payload 已在 abort 前确认并上传到上述路径。
- v7x8 debug 实验 `exp-gq082n4nzb` 已 abort。
- 权重下载实验 `exp-q7odgo8q9x` 已成功结束。
- 旧实验 `exp-x9...` 和 range benchmark `exp-r78...` 已 abort，clone 实验 `exp-1vt...` 已失败退出，没有继续占用计算资源。

Falcon 对已 abort 实验的 artifact 链接可能显示为 `failed`，但本轮列出的结果文件已实际落入 GCS artifact root。runner 结束时出现的 `Killed` 是生成结果后清理 server process group；本轮成功 run 均已写入 `SUCCESS` 和全部 follower ACK，不代表 E2E 失败。
