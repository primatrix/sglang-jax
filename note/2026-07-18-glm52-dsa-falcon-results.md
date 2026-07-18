# GLM-5.2 DSA Falcon 调试记录

## 目标与状态

目标是在 Falcon v7x 上跑通 GLM-5.2 的 DSA 最小真实闭环。当前已经完成 baseline sparse MLA Pallas kernel 的真 TPU correctness 和微基准；真实 checkpoint 的 prefill → decode E2E 尚未完成，不能标记为 E2E 通过。

- Branch: `develop/glm52-dsa-falcon`
- Kernel/Falcon checkpoint: `e85252a34` (`fix(kernels): align GLM DSA Falcon DMAs`)
- Active v7x-8 debug experiment: `exp-gq082n4nzb`
- Pending CPU checkpoint downloader: `exp-9oj6o2sohe`
- Model source: [zai-org/GLM-5.2](https://huggingface.co/zai-org/GLM-5.2)

## 已确认的 checkpoint 配置

官方 `config.json` 与当前 fixture 对齐：

- architecture: `GlmMoeDsaForCausalLM`
- dtype: BF16
- hidden size / layers: `6144 / 78`
- `kv_lora_rank`: `512`
- `qk_nope_head_dim / qk_rope_head_dim`: `192 / 64`
- attention heads: `64`
- `index_head_dim / index_n_heads / index_topk`: `128 / 32 / 2048`
- `index_topk_freq / index_skip_topk_offset`: `4 / 3`
- experts / top-k experts: `256 / 8`

模型仓库约 1.51 TB、包含 282 个 safetensors shard。v7x-8 只用于 kernel 调试；真实模型计划使用 v7x-32（4 replicas、32 JAX devices、`2x2x4`）。

## `/models` 探查与下载

通过 Falcon pod 内的 GCSFuse mount 检查 `inference-model-storage-poc-tpu-hns`，已有 GLM 目录只有：

- `/models/GLM-4.5`
- `/models/GLM-4.5-Air`

没有发现 GLM-5 或 GLM-5.2 checkpoint/config。未使用 provider CLI 直接读取 bucket。

CPU 下载 manifest：

```text
scripts/kernels/falcon_glm52_model_download_cpu.yaml
source      = zai-org/GLM-5.2
destination = /models/GLM-5.2
completion  = /models/GLM-5.2/_DOWNLOAD_COMPLETE
experiment  = exp-9oj6o2sohe
```

截至本记录，任务为 `PENDING/Unschedulable`。Falcon 报告匹配的 6 个 CPU node 都是 `Insufficient cpu`；任务保留排队并支持 `hf download` 断点续传。`tpu-service` 没有可用 CPU quota，因此没有可迁移的第二个 active cluster。

## Falcon v7x-8 环境

Experiment `exp-gq082n4nzb`：

```text
cluster=tpu-training-antgroup
device_type=v7x
device_topo=2x2x1
replica=1
local_device_count=8
jax=0.9.0
jaxlib=0.9.0
libtpu=0.0.34
flax=0.12.3
```

当前分支尚未 push，首次实验从 upstream main clone 后使用 `falcon exp cp` 注入本地 patch。已提交的 manifest 默认从 `https://github.com/cjx0709/sglang-jax.git` 获取 `develop/glm52-dsa-falcon`；在远端分支存在后可直接复现。

## 真机故障与修复

1. q/RoPE HBM slice 的最内维 64 不满足 Mosaic 128-wide tile。修复：在 Pallas 外分别把 latent 和 RoPE query pad 到 128 对齐，DMA 完整行。
2. flatten 后 DMA 单个 `[1, 640]` cache row 不满足 HBM first-dimension tile 2。修复：把 physical slot 解码为 `page / packed_row / packing_index`，DMA 完整 `[2, 640]` packing group。
3. VMEM 上用动态 `packing_index` 直接取 row，Mosaic 无法证明 index 满足 tile 对齐。修复：以 one-hot mask 在 VPU 中选择 packing row。

validated API 现在明确要求 production cache `packing == 2`，避免接受真机无法 lower 的 layout。

## Correctness 证据

本地完整 DSA 回归：

```bash
PYTHONPATH=python /Users/jiongxuan/workspace/sglang-jax/.venv/bin/python -m pytest -q \
  python/sgl_jax/test/kernels/test_dsa_decode_mla.py \
  python/sgl_jax/test/test_dsa_backend.py \
  python/sgl_jax/test/test_dsa_reference.py \
  python/sgl_jax/test/test_dsa_glm52.py \
  python/sgl_jax/test/test_model_runner_kv_cache_mixin.py
```

结果：`87 passed, 2 skipped, 37 subtests passed`。两个 skip 是本机没有 TPU 的 non-interpret case。

Falcon 真 TPU 完整 kernel 文件：

```bash
scripts/kernels/run_glm52_dsa_v7x8_debug.sh full
```

结果：`33 passed, 1 skipped in 8.76s`。skip 是只应在 CPU 执行的 non-TPU guard。测试包含 dynamic physical slots，以及 `latent=512 / rope=64 / topk=2048 / page_size=128` 的 GLM shape。

代码质量门禁：Ruff、Black、YAML parse 和 `git diff --check` 均通过。独立代码审查未发现 Critical 或 kernel correctness 阻塞问题。

## Kernel 微基准

运行：

```bash
scripts/kernels/run_glm52_dsa_v7x8_bench.sh
```

Shape：B1、context 160000、TopK 2048、8 local heads、latent 512、RoPE 64、page 128、BF16、`sm_scale=0.0625`，2 次 warmup、5 次计时。

```text
median_ms = 1.956691
mean_ms   = 1.955294
p99_ms    = 1.963744
min_ms    = 1.948020
```

这是 baseline correctness kernel 的短微基准，不代表完整模型 token latency，也未包含 Indexer、Top-K、cache write 或跨 host 通信。

## 下一步

1. 等待 `exp-9oj6o2sohe` 获得 CPU 并完成 `/models/GLM-5.2/_DOWNLOAD_COMPLETE`。
2. 用 v7x-32 启动真实 checkpoint，先跑短 prefill + reference sparse decode。
3. 保持同一 selection/cache state 切换到 Pallas，比较 latent output 和最终 logits。
4. 覆盖 `full -> shared -> full` 的 IndexShare，以及两个 ragged request 的 chunked prefill → decode。
5. 补齐 compile time、峰值内存和 trace；在这些证据和真实 logits 闭环完成前，不声明 E2E 通过。

## 资源清理

`exp-gq082n4nzb` 暂时保留用于继续调试，`exp-9oj6o2sohe` 保留用于模型下载。工作结束后再显式 abort，避免误停仍在使用的下载任务。
