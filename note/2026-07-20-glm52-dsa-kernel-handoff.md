# GLM-5.2 DSA 算子输入交接

## 目标

交付两样东西：

1. CPU PyTorch golden，用来验证 Indexer selection、slot mapping 和 sparse MLA 精度。
2. 单 TPU device 的 sparse MLA microbenchmark，用来调试和做 kernel 性能优化。

这里不要求 TP32，也不复现服务的 Q64 decode bucket。多卡 mesh、padding bucket、并发
request pool 和 160k 长上下文属于后续 E2E 集成，不是算子 microbenchmark 的前置条件。

## Kernel-local ABI

```text
q_latent        [Q, H, 512]       BF16
q_rope          [Q, H, 64]        BF16
cache           [P, 64, 2, 640]   BF16
physical_slots  [Q, K]            INT32, K <= 2048
selected_counts [Q]               INT32
sm_scale        scalar            FP32, 256^-0.5
output          [Q, H, 512]       BF16
```

microbenchmark 固定 `H=1`，独立测一个 head。cache 本身没有 KV-head 轴，因此无需把模型
TP 拓扑搬进 benchmark。

物理 slot 的解码方式：

```text
page = slot // 128
offset = slot % 128
row = offset // 2
lane = offset % 2
```

只有 `physical_slots[q, :selected_counts[q]]` 有语义；`slot=0` 是合法地址；
`selected_counts=0` 时输出全 0。算子只做 final sparse MLA，不包含 Indexer、Top-K、
logical-to-physical mapping 或 cache write。

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

输出为 `.npz + manifest.json`，manifest 记录 shape、dtype 和每个 fixture 的 SHA-256。
默认覆盖：

- Indexer candidate length：`1, 127, 128, 129, 257, 2047, 2048, 2049, 3072, 4096`。
- realistic Indexer：全部 32 个 index heads、128 维、signed weights 和 ReLU 两侧。
- mapping：duplicate、future、invalid、missing slot。
- sparse MLA：`H=1`，selected count 为 `0, 1, 128, 2048`，slot order 为 unsorted。

精度 gate：

- Indexer logical IDs、selected counts 和 mapping counted prefix：精确一致。
- Indexer scores：`rtol=1e-6, atol=1e-6`。
- sparse MLA FP32 reference：`rtol=2e-5, atol=2e-5`。
- BF16 TPU kernel 输出 cast 到 FP32 后：`rtol=2e-2, atol=1e-2`。

当 visible count 大于 2048 时，full dense attention 不是 golden；正确结果是 CPU Indexer
Top-K + CPU mapping + CPU sparse MLA。

## 性能场景

三个场景都只用第一个 local TPU device，不创建 mesh：

| Case | Q | H | cache tokens | K | 用途 |
| --- | ---: | ---: | ---: | ---: | --- |
| `debug-q1-h1-c128-k128` | 1 | 1 | 128 | 128 | 快速编译和调试 |
| `decode-q1-h1-c8192-k2048` | 1 | 1 | 8192 | 2048 | decode 单 query 延迟 |
| `prefill-q128-h1-start2048-k2048` | 128 | 1 | 2176 | 2048 | steady-state prefill chunk |

运行：

```bash
BENCH=benchmark/kernels/mla/bench_dsa_decode_mla.py
COMMON="--num-heads 1 --latent-dim 512 --rope-dim 64 --page-size 128 \
--slot-order unsorted --variant sparse --warmup-iters 50 --iters 200"

PYTHONPATH=python:. python "$BENCH" \
  --batch-size 1 --context-length 128 --top-k 128 $COMMON

PYTHONPATH=python:. python "$BENCH" \
  --batch-size 1 --context-length 8192 --top-k 2048 $COMMON

PYTHONPATH=python:. python "$BENCH" \
  --batch-size 128 --context-length 2176 --top-k 2048 \
  --valid-count-pattern causal --start-position 2048 $COMMON
```

脚本把首次 compile + execute 单独记为 `compile_ms`，计时前同步 warmup，并输出 median、
p95、p99、mean 和 min。CPU reference 不进入性能计时。

## 验证

```bash
PYTHONPATH=python:. python -m pytest \
  python/sgl_jax/test/test_dsa_cross_framework.py \
  python/sgl_jax/test/kernels/test_bench_dsa_decode_mla.py \
  python/sgl_jax/test/kernels/test_glm52_dsa_handoff.py -q
```

交接完成标准：CPU bundle 可复现、三阶段 correctness gate 通过、三个单设备场景都能完成
precompile 和稳定计时。TP32 E2E 只在 kernel-local 结果稳定后再做。
