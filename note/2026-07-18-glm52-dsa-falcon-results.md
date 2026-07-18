# GLM-5.2 DSA Falcon 调试记录

## 目标与当前结论

目标是在 Falcon v7x 上跑通 GLM-5.2 的 DSA 最小闭环。当前状态：

- baseline sparse MLA Pallas kernel 已通过 v7x-8 真 TPU correctness。
- 32 芯显式 mesh 下的 TopK=2048 reference + Pallas smoke 已在 4 个 process 全部通过。
- GLM-5.2 dummy-weight server 已在 v7x-32 完成 prefill → decode → HTTP response 的分布式 E2E。
- 真实 checkpoint 尚未下载完成，因此真实权重、最终 logits 和模型质量闭环仍未完成，不能声明真实模型 E2E 已通过。

资源与分支：

- Branch: `develop/glm52-dsa-falcon`
- Active v7x-8 debug experiment: `exp-gq082n4nzb`
- Active v7x-32 debug experiment: `exp-x9ghpgedxk`
- Active v7x-8 host-CPU checkpoint downloader: `exp-q7odgo8q9x`
- Model source: [zai-org/GLM-5.2](https://huggingface.co/zai-org/GLM-5.2)

## 已确认的 checkpoint 配置

官方 `config.json` 与当前 fixture 对齐：

- architecture: `GlmMoeDsaForCausalLM`
- dtype: BF16
- hidden size / layers: `6144 / 78`
- `kv_lora_rank`: `512`
- `q_lora_rank`: `2048`
- `qk_nope_head_dim / qk_rope_head_dim`: `192 / 64`
- attention heads: `64`
- `index_head_dim / index_n_heads / index_topk`: `128 / 32 / 2048`
- `index_topk_freq / index_skip_topk_offset`: `4 / 3`
- experts / top-k experts: `256 / 8`

模型仓库约 1.51 TB、包含 282 个 safetensors shard。v7x-8 主要用于 kernel 调试；完整模型使用 v7x-32（4 replicas、32 JAX devices、`2x2x4`）。

## `/models` 探查与下载

通过 Falcon pod 内的 GCSFuse mount 检查 `inference-model-storage-poc-tpu-hns`，已有 GLM 目录只有：

- `/models/GLM-4.5`
- `/models/GLM-4.5-Air`

没有发现 GLM-5 或 GLM-5.2 checkpoint/config。未使用 provider CLI 直接读取 bucket。

最初提交的 CPU 下载 manifest：

```text
scripts/kernels/falcon_glm52_model_download_cpu.yaml
source      = zai-org/GLM-5.2
destination = /models/GLM-5.2
completion  = /models/GLM-5.2/_DOWNLOAD_COMPLETE
experiment  = exp-9oj6o2sohe
```

该任务 `exp-9oj6o2sohe` 实际没有开始下载。`device_type: cpu` 将它限制到 6 个 CPU node，这些节点全部报 `Insufficient cpu`；其余节点因 taint 或 affinity 不匹配无法承载。降低 250m CPU request 没有意义，因此已 abort。两个不声明 device 的旧式 generic manifest 和一个在 v7x cluster 显式请求 CPU 的试验也没有进入资源调度，均已 abort。

当前改用 v7x-8 pod 的 host CPU 和网络，TPU 本身不参与下载：

```text
manifest    = scripts/kernels/falcon_glm52_model_download_v7x8.yaml
source      = zai-org/GLM-5.2
destination = /models/GLM-5.2
completion  = /models/GLM-5.2/_DOWNLOAD_COMPLETE
experiment  = exp-q7odgo8q9x
transport   = hf-xet -> writable GCSFuse streaming write
```

任务约 7 秒完成调度并进入 `RUNNING`。host 有 224 CPU、有效内存约 945 GiB，启用了 Hugging Face Xet high-performance 和 sequential reconstruction；GCSFuse mount 启用了 streaming writes。metadata 13 秒下载完成，index 校验得到 282 个预期 shard。最近两次 60 秒进度采样：

```text
15/282 shards,  80,391,064,128 bytes
24/282 shards, 128,645,946,912 bytes
```

最近一分钟约写入 48.3 GB，启动以来平均约 0.71 GB/s；按当前样本粗估剩余约 30--40 分钟。分片完成是突发式的，因此这里只作为观察值，不作为完成承诺。下载支持断点续传；只有 config、index 和全部非空 shard 校验通过后才写 `_DOWNLOAD_COMPLETE`。

## Falcon 环境

v7x-8 experiment `exp-gq082n4nzb`：

```text
device_topo=2x2x1
replica=1
local_device_count=8
jax/jaxlib=0.9.0
libtpu=0.0.34
```

v7x-32 experiment `exp-x9ghpgedxk`：

```text
device_topo=2x2x4
replica=4
global/local devices=32/8
jax/jaxlib=0.9.0
libtpu=0.0.34
flax=0.12.4
```

当前分支尚未 push，实验最初从 upstream main clone，再使用 `falcon exp cp` 注入本地 patch。manifest 默认从 `https://github.com/cjx0709/sglang-jax.git` 获取 `develop/glm52-dsa-falcon`；远端分支存在后可直接复现。

## 真机故障与修复

### Mosaic DMA 与内存布局

1. q/RoPE HBM slice 的最内维 64 不满足 Mosaic 128-wide tile。在 Pallas 外分别把 latent 和 RoPE query pad 到 128 对齐，DMA 完整行。
2. flatten 后 DMA 单个 `[1, 640]` cache row 不满足 HBM first-dimension tile 2。把 physical slot 解码为 `page / packed_row / packing_index`，DMA 完整 `[2, 640]` packing group。
3. VMEM 上用动态 `packing_index` 直接取 row，Mosaic 无法证明 index 对齐。改为 one-hot VPU 运算选择 packing row。
4. GLM prefill batch 的 `[128, 2048]` Top-K 表单独就占满 v7x 的 1 MiB SMEM，再加 valid count 会编译失败。现在只把 valid count scalar-prefetch 到 SMEM；Top-K 表留在 HBM，reshape 为 128-wide chunks，每个 program 按需 DMA 一个 chunk 到 VMEM，再用 128-way VPU mask 选择 slot。

validated API 明确要求 production cache `packing == 2`，避免接受真机无法 lower 的 layout。

### JAX 0.9 显式 mesh

1. 对 Index-K cache、candidate rows/keys、logical → physical slots 和 MLA cache write 的 gather/scatter 补充显式 `out_sharding` 或 operand sharding。MLA reference gather 使用 physical-slot 的 data sharding，保持 Top-K 轴在 tensor mesh 上 replicated；如果沿用 query 的 head sharding，会把 tensor 轴错误放到 Top-K 并触发 `DuplicateSpecError`。
2. 外层 selection compaction 不再使用会生成 replicated iota 的 stable `jnp.argsort`；改用显式 sharding 的 `lax.broadcasted_iota` 和 `lax.sort_key_val`。
3. 显式 mesh 内不能直接调用 Pallas。DSA backend 现在用 `jax.shard_map` 建立 manual region：query/output 按 `P(data, tensor, None)`，cache/slots/counts 按 data axis 分片，再执行本地 Pallas kernel。

## Correctness 证据

本地完整 DSA 回归：

```bash
../../.venv/bin/python -m pytest -q \
  python/sgl_jax/test/kernels/test_dsa_decode_mla.py \
  python/sgl_jax/test/test_dsa_backend.py \
  python/sgl_jax/test/test_dsa_reference.py \
  python/sgl_jax/test/test_dsa_glm52.py \
  python/sgl_jax/test/test_model_runner_kv_cache_mixin.py
```

结果：`95 passed, 2 skipped, 1 warning, 39 subtests passed in 12.22s`。两个 skip 是本机没有 TPU 的 non-interpret case；warning 是仓库已有的 `jax.experimental.shard_map` deprecation。

Falcon v7x-8 完整 kernel 文件：

```bash
scripts/kernels/run_glm52_dsa_v7x8_debug.sh full
```

最新结果：`34 passed, 1 skipped in 10.70s`。测试覆盖 dynamic physical slots，`valid_count=0/1/127/128/129/255/256`，非对齐 `max_selected=129/2050`，以及 `latent=512 / rope=64 / topk=2048 / page_size=128` 的 GLM shape。

Falcon v7x-32 manual-shard smoke：

```text
GLM52_DSA_V7X32_REFERENCE_AND_SHARDMAP_OK process_id=0 local_devices=8
GLM52_DSA_V7X32_REFERENCE_AND_SHARDMAP_OK process_id=1 local_devices=8
GLM52_DSA_V7X32_REFERENCE_AND_SHARDMAP_OK process_id=2 local_devices=8
GLM52_DSA_V7X32_REFERENCE_AND_SHARDMAP_OK process_id=3 local_devices=8
```

该 smoke 使用 32-device 显式 `data=1 / tensor=32` mesh 和 TopK=2048。输入按 slot/head 构造为可区分的非零值，`valid_count=129`，未计数 tail 指向高值哨兵；四个 process 都逐 addressable shard 比较 reference 和 Pallas 数值，Falcon command exit 0。

## v7x-32 dummy-weight E2E

运行入口：

```text
manifest = scripts/kernels/falcon_glm52_dsa_v7x32_dummy_debug.yaml
runner   = scripts/kernels/run_glm52_dsa_v7x32_dummy_e2e.sh
model    = zai-org/GLM-5.2
load     = dummy
parallel = TP32 / DP1 / EP32
backend  = DSA + fused MoE
context  = 4096
prefill  = chunked, max 128
```

runner 要求每次调用提供跨 rank 共享且不可复用的 nonce，例如：

```bash
GLM52_DSA_RUN_ID=glm52-dsa-e2e-<unique> \
  scripts/kernels/run_glm52_dsa_v7x32_dummy_e2e.sh
```

关键结果：

- 543 个缺失参数由 dummy initializer 补齐；MLA absorb 和 fused MLP 初始化完成。
- KV 可用约 38.6 GB，模型配置 4096 tokens，cache 总量约 0.39 GB。
- EXTEND precompile 约 79 秒，通过。
- DECODE precompile 约 67 秒，通过。
- `/health` 返回 200。
- 请求：`input_ids=[1,2,3,4]`，`max_new_tokens=2`。
- 稳定 runner 复跑响应：`output_ids=[0,0]`，`prompt_tokens=4`，`completion_tokens=2`，`e2e_latency=1.4566s`。
- runner 输出 `GLM52_DSA_DUMMY_PREFILL_DECODE_OK`，Falcon command exit 0。

dummy 权重下输出 0 是预期现象。这个结果验证的是 32 芯 topology、模型构建、DSA selection/cache、Pallas dispatch、prefill/decode 控制流和 HTTP server 链路，不验证真实模型 logits。

runner 使用本次 nonce 隔离 control directory，只有 rank 0 在 HTTP 响应校验通过后写 `SUCCESS + STOP`；失败 rank 写独立 `FAIL-rank-N`，follower 完成限时 teardown 后写 `ACK-rank-N`。本次稳定复跑最终得到 `SUCCESS`、`STOP` 和 rank 1/2/3 的全部 ACK，无 FAIL。health/generate、follower wait、ACK wait 和 TERM → KILL teardown 都有 deadline；末尾的 rank 0 `Killed` 是 server 在 60 秒 TERM deadline 后被 runner 强制清理，不是 E2E 失败。结束后确认四个 host 均无残留 `sgl_jax.launch_server` 进程。

## Kernel 微基准

运行：

```bash
scripts/kernels/run_glm52_dsa_v7x8_bench.sh
```

Shape：B1、context 160000、TopK 2048、8 local heads、latent 512、RoPE 64、page 128、BF16、`sm_scale=0.0625`，2 次 warmup、5 次计时。

最新 HBM chunk metadata 实现：

```text
median_ms = 2.168382
mean_ms   = 2.170685
p99_ms    = 2.178637
min_ms    = 2.166540
```

改造前 SMEM metadata 实现的 median 为 1.956691 ms；当前约慢 10.8%，但解除 `[128, 2048]` Top-K 表在 GLM prefill 编译中的 SMEM OOM。后续可以考虑流水化 metadata DMA，或为 decode B1 和 prefill batch 拆分 specialization。

这是 baseline correctness kernel 的短微基准，不代表完整模型 token latency，也未包含 Indexer、Top-K、cache write 或跨 host 通信。

## 下一步

1. 观察 `exp-q7odgo8q9x`，确认 `/models/GLM-5.2/_DOWNLOAD_COMPLETE`；完成后 abort 下载 pod，释放 v7x-8 reservation。
2. 在 v7x-32 加载真实 checkpoint，先跑短 prefill + reference sparse decode，确认内存和参数映射。
3. 保持同一 selection/cache state 切换到 Pallas，比较 latent output 和最终 logits。
4. 覆盖 `full → shared → full` 的 IndexShare，以及两个 ragged request 的 chunked prefill → decode。
5. 补齐真实权重下的 compile time、峰值内存和 token latency；这些证据完成前，不声明真实模型 E2E 通过。

## 资源清理

`exp-gq082n4nzb` 和 `exp-x9ghpgedxk` 暂时保留用于继续调试。`exp-q7odgo8q9x` 保持运行以完成 checkpoint 下载；原 CPU 下载及三个不可调度的替代试验均已 abort。真实权重验证结束后再显式清理调试资源。
