# GLM-5.2 DSA 算子输入交接

## 目标与范围

本交接提供两类输入：

1. 独立的 CPU PyTorch FP32 golden，用来验证 Indexer selection、logical-to-physical
   mapping 和 final sparse MLA read 的数值语义。
2. 单 TPU device 的 final sparse MLA microbenchmark，用来调试和测量算子本身。

v0.1 不纳入 CP；也不要求 TP32 或创建多卡 mesh。TP 不改变一次 chunked prefill 的
`Q`，而是决定每张卡上的 local Q-head 数。这里的 `H=8/16` 是已经投影到单设备的
local Q-head 代表点，不是启动 8/16/32 张设备的要求；`H=1` 只保留给快速 debug。

prefill 的 `Q=2K/4K` 是代表性的 chunk 范围，目的是覆盖真实 chunk prefill 的
kernel 工作量，而不是从某个服务参数机械推导出的固定值，也不是服务调度器的上限。
实际集成若采用别的 chunk bucket，保持同一 ABI、在该区间附近增补参数化 case 即可。

## 分层边界与数据流

端到端 DSA 数据流如下：

```text
Index-K cache write
  -> per-query causal Top-K
  -> logical-to-physical mapping
  -> main MLA cache write
  -> final sparse MLA read
```

本交接的 Pallas 算子范围仅是最后一步。它消费已经选好的 **physical slots**，不负责
Indexer、Top-K、mapping、Index-K/main MLA cache write，也不消费 Index-K 的 FP8+
scale 存储。GPU 代码存在非-absorb MLA fallback，但本交接只覆盖 GLM-5.2 的 absorb
MLA 路径。

### final sparse MLA read ABI

```text
q_latent        [Q, H, 512]       BF16
q_rope          [Q, H, 64]        BF16
cache           [P, 64, 2, 640]   BF16
physical_slots  [Q, K]            INT32, K <= 2048
selected_counts [Q]               INT32
sm_scale        scalar            FP32, 256^-0.5
output          [Q, H, 512]       BF16
```

`640 = align128(512) + align128(64)`；其中真正参与语义的 KV feature 是
`512 + 64`，尾部 64 是 physical packing padding。cache 的物理 layout 为
`[page, packed_row, lane, feature]`，slot 解码如下：

```text
page = slot // 128
offset = slot % 128
row = offset // 2
lane = offset % 2
```

只有 `physical_slots[q, :selected_counts[q]]` 有语义；`slot=0` 是合法地址，
`selected_counts=0` 时输出全 0。selected rank 以外的 slot 值可以是 0；不能把它
解释成 padding sentinel。logical Top-K ID 用 `-1` 表示 padding，mapping 后的
physical slots 用 counted prefix 区分有效范围。

### selection 与 mapping contract（framework-owned）

selection 不是 final sparse MLA read ABI 的一部分，但 CPU golden 必须覆盖它。一个
生产实现至少需要等价地提供：

```text
q_index                 [Q, 32, 128]
head_weights            [Q, 32]
index_k_cache_after_write
req_to_token_slots
query_request_indices   [Q]
query_positions         [Q]
```

对 prefill 中请求 `r` 的第 `i` 个 query row，若该 row 的绝对位置为 `p`，候选的
logical token 集合是该请求的 `[0, p]`。当前 chunk 的 Index-K 必须先写入，之后每个
row 独立打分并做 causal Top-K；未来 token 仍被 mask。Top-K 输出 logical IDs 和
`selected_counts=min(visible_count, 2048)`，随后经 `req_to_token_slots` 映射成 final
read 所需的 physical slots。多个请求/ragged prefill 时，`query_request_indices` 和
`query_positions` 使每个 row 仍使用自己的 request 边界与候选范围。

golden 为了可审计可以 materialize `[Q, C]` candidate matrix；生产路径不需要持久化该
矩阵。Index-K 在 GPU 路径可用 FP8+scale 存储，selection contract 的整数输出固定为
INT32；final MLA cache/read 在本 handoff 中为 BF16。

## CPU golden

Reference：

- `python/sgl_jax/srt/kernels/dsa/torch_reference.py`
- `benchmark/kernels/mla/export_glm52_dsa_golden.py`

生成：

```bash
PYTHONPATH=python:. python \
  benchmark/kernels/mla/export_glm52_dsa_golden.py \
  --output-dir /tmp/glm52-dsa-golden
```

输出为 `.npz + manifest.json`；manifest 记录 ABI、shape、dtype 和每个 fixture 的
SHA-256。PyTorch CPU FP32 是独立 source of truth；JAX reference 用同一 bundle 做
集成比较，不能替代该 golden。

当前 bundle 覆盖：

- Indexer candidate length：`1, 127, 128, 129, 257, 2047, 2048, 2049, 3072, 4096`。
- realistic Indexer：全部 32 个 index heads、128 维、signed weights 和 ReLU 两侧。
- mapping：duplicate、future、invalid、missing slot。
- sparse MLA：`H=1`，selected count 为 `0, 1, 128, 2048`，slot order 为 unsorted。

在编写 selection kernel 前，还需把以下 fixture 加入 exporter：多个 request 的 ragged
prefill、当前 chunk 写入后的 per-query causal candidates，以及跨 `K=2048` 边界的
logical Top-K 再映射 physical slots。这个 fixture 是 selection/mapping kernel 的
gate，不是把 synthetic sparse-read slot table 误当作真实 Indexer Top-K。

精度 gate：

- Indexer logical IDs、selected counts 和 mapping counted prefix：精确一致。
- Indexer scores：`rtol=1e-6, atol=1e-6`。
- sparse MLA FP32 reference：`rtol=2e-5, atol=2e-5`。
- BF16 TPU kernel 输出 cast 到 FP32 后：`rtol=2e-2, atol=1e-2`。

当 visible count 大于 2048 时，full dense attention 不是 golden；正确比较链路是 CPU
Indexer Top-K + CPU mapping + CPU sparse MLA。

## 性能场景

场景均只使用一个 local TPU device，不创建 mesh。`H` 是传给 kernel 的 local Q-head
维度；这些 cases 保持在 `SparseMlaPerfCase` 和 CLI 中参数化，因而不把 2K/4K、8/16
编码为 TP 或调度器限制。

| Case | Q | H | cache tokens | K | 用途 |
| --- | ---: | ---: | ---: | ---: | --- |
| `debug-q1-h1-c128-k128` | 1 | 1 | 128 | 128 | 快速编译和地址/精度调试 |
| `decode-q1-h8-c8192-k2048` | 1 | 8 | 8192 | 2048 | 代表性 decode 单 query 延迟 |
| `prefill-q2048-h8-start8192-k2048` | 2048 | 8 | 10240 | 2048 | 2K chunk prefill；每个 row 已过 Top-K 饱和点 |
| `prefill-q4096-h16-start16384-k2048` | 4096 | 16 | 20480 | 2048 | 4K chunk prefill；更大的 local-head/workload 点 |

两个 prefill case 的 `start_position` 分别为 8192/16384，故每个 row 的
`selected_counts=2048`；这隔离 final sparse MLA read 的稳定态访问模式。它们不测
selection/mapping，也不代表 chunk 的全局服务配置。

运行：

```bash
BENCH=benchmark/kernels/mla/bench_dsa_decode_mla.py
COMMON="--latent-dim 512 --rope-dim 64 --page-size 128 \
--slot-order unsorted --variant sparse --warmup-iters 50 --iters 200"

PYTHONPATH=python:. python "$BENCH" \
  --batch-size 1 --num-heads 1 --context-length 128 --top-k 128 $COMMON

PYTHONPATH=python:. python "$BENCH" \
  --batch-size 1 --num-heads 8 --context-length 8192 --top-k 2048 $COMMON

PYTHONPATH=python:. python "$BENCH" \
  --batch-size 2048 --num-heads 8 --context-length 10240 --top-k 2048 \
  --valid-count-pattern causal --start-position 8192 $COMMON

PYTHONPATH=python:. python "$BENCH" \
  --batch-size 4096 --num-heads 16 --context-length 20480 --top-k 2048 \
  --valid-count-pattern causal --start-position 16384 $COMMON
```

脚本将首次 compile + execute 单独记为 `compile_ms`，计时前同步 warmup，并输出
median、p95、p99、mean 和 min。CPU reference 不进入性能计时。final sparse MLA read
的 cache traffic 和 QK/PV 计算应与 selection/mapping 分开归因；前者的语义 cache
feature 宽度是 576（而非 packed 640）。

## 集成边界与验证

GPU 的 `--enable-attn-tp-input-scattered` 明确不支持 DSA；DSA prefill CP 是另一条独立
feature 路径。因此 v0.1 不应把 CP/TP communication 结果混入这个 kernel handoff。

```bash
PYTHONPATH=python:. python -m pytest \
  python/sgl_jax/test/test_dsa_cross_framework.py \
  python/sgl_jax/test/kernels/test_bench_dsa_decode_mla.py \
  python/sgl_jax/test/kernels/test_glm52_dsa_handoff.py -q
```

final sparse MLA read 的交接完成标准：CPU bundle 可复现、已有三阶段 correctness gate
通过、四个单设备场景能完成 precompile 和稳定计时。selection/mapping kernel 的完成标准
额外包括上文 ragged prefill causal fixture；TP/CP E2E 只在这些 kernel-local gate
稳定后再做。
