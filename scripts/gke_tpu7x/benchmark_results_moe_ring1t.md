# MoE Fused Kernel Benchmark: sglang-jax vs tpu-inference

**Branch:** `feat/gke-tpu7x-bench`
**Hardware:** TPU v7x-8 (8 chips, GKE)
**Date:** 2026-03-25
**Reference model:** [Ring-1T-FP8](https://huggingface.co/inclusionAI/Ring-1T-FP8/blob/main/config.json) (DeepseekV3-like MoE)

---

## Model Config (Ring-1T-FP8)

Full model config (ep=32):
- `num_experts = 256`, `top_k = 8`, `topk_group = 4`, `num_shared_experts = 1`
- `hidden_size = 8192`, `moe_intermediate_size = 2048`
- `activation = silu`, `norm_topk_prob = True`

Scaled config for ep=8 (experts / 4):
- `num_experts = 64`, `top_k = 8`
- `hidden_size = 8192`, `intermediate_size = 2048`

---

## Benchmark Setup

### sglang-jax

```bash
python3 -u /tmp/launcher.py -m benchmark.moe.bench_fused_moe \
    --num-experts 64 --topk 8 --hidden-size 8192 --intermediate-size 2048 \
    --num-tokens 64 128 256 512 1024 --iters 5 --warmup-iters 2 \
    --imbalance-mode balanced
```

- Kernel: `sgl_jax.srt.kernels.fused_moe.v1.kernel` (Pallas GMM-based)
- Mesh: 1D mesh with `ep_size=8`
- Timing: `multiple_iteration_timeit_from_trace` (XLA trace-based, drops first sample)

### tpu-inference

```bash
python3 -u /tmp/launcher.py scripts/gke_tpu7x/bench_tpu_inference_moe.py \
    --num-experts 64 --top-k 8 --hidden-size 8192 --intermediate-size 2048 \
    --num-tokens 64 128 256 512 1024 --iters 5 --warmup-iters 2
```

- Kernel: `tpu_inference.kernels.fused_moe.v1.kernel.fused_ep_moe` (Pallas ring-based all-to-all DMA)
- Mesh: 2D mesh `("data", "model")` with `ep_size=8`
- Timing: same `multiple_iteration_timeit_from_trace`

---

## Benchmark Results (Ring-1T, ep=8)

| num_tokens | sglang-jax (ms) | tpu-inference (ms) | speedup |
|:----------:|:---------------:|:------------------:|:-------:|
| 64         | 0.814           | 1.734              | 2.1x    |
| 128        | 0.863           | 1.604              | 1.9x    |
| 256        | 0.943           | 2.763              | 2.9x    |
| 512        | 2.099           | compilation failed | -       |
| 1024       | 3.839           | compilation failed | -       |

**sglang-jax is ~2-3x faster on the Ring-1T MoE config.**

### Notes

- tpu-inference fails to compile at 512/1024 tokens due to VMEM overflow. The kernel's scratch buffer management cannot handle large batch sizes combined with large hidden/intermediate dimensions (8192/2048).
- sglang-jax handles all token counts without issues.
- Both kernels were run with balanced routing (no imbalance simulation).

---

## Profiling (xprof with LLO utilization)

Profile traces were captured for sglang-jax using xprof custom call profiling to inspect LLO-level utilization of the Pallas kernels.

### Setup

Required LIBTPU flags (set before importing JAX):
```bash
LIBTPU_INIT_ARGS="--xla_enable_custom_call_region_trace=true --xla_xprof_register_llo_debug_info=true"
```

Required package: `libtpu-nightly==0.0.38.dev*` (standard libtpu may not support LLO debug info registration).

### Profile command

```bash
python3 -u /tmp/launcher.py -m benchmark.moe.bench_fused_moe \
    --num-experts 64 --topk 8 --hidden-size 8192 --intermediate-size 2048 \
    --num-tokens 128 --iters 3 --warmup-iters 1 \
    --imbalance-mode balanced --profile --profile-dir profile_ring1t_moe
```

### Trace files

Traces saved to `profile_ring1t_moe/case_128t_64e_ep8/`:
- `*.xplane.pb` (80 MB) — full XPlane data, viewable with TensorBoard
- `*.trace.json.gz` (10 MB) — trace events, viewable with Perfetto UI (https://ui.perfetto.dev)

### Viewing traces

**Option 1: TensorBoard**
```bash
pip install tensorboard-plugin-profile
tensorboard --logdir=profile_ring1t_moe/
```

**Option 2: Perfetto UI**
1. Open https://ui.perfetto.dev
2. Upload `*.trace.json.gz`
3. Look for the "LLO utilization" line under each TPU core to inspect custom call performance

---

## Conclusions

1. **sglang-jax fused_moe is 2-3x faster** than tpu-inference's ring-based implementation on the Ring-1T MoE config (64 experts, top_k=8, hidden=8192, intermediate=2048, ep=8).

2. **tpu-inference has scalability limitations** — it fails to compile with 512+ tokens on Ring-1T dimensions due to VMEM overflow in scratch buffer allocation.

3. **sglang-jax scales smoothly** from 64 to 1024 tokens, with latency growing sub-linearly (0.8ms → 3.8ms for 16x more tokens).

4. **Profile traces confirm** that LLO utilization is visible for Pallas custom calls when the correct LIBTPU flags and libtpu-nightly are used.

---

## How to Run with Other Configs

### sglang-jax benchmark
```bash
python3 -u /tmp/launcher.py -m benchmark.moe.bench_fused_moe \
    --num-experts <E> --topk <K> --hidden-size <D> --intermediate-size <F> \
    --num-tokens <T1> <T2> ... --iters 5 --warmup-iters 2 \
    --imbalance-mode balanced
```

### tpu-inference benchmark
```bash
python3 -u /tmp/launcher.py scripts/gke_tpu7x/bench_tpu_inference_moe.py \
    --num-experts <E> --top-k <K> --hidden-size <D> --intermediate-size <F> \
    --num-tokens <T1> <T2> ... --iters 5 --warmup-iters 2
```

### sglang-jax profiling
```bash
LIBTPU_INIT_ARGS="--xla_enable_custom_call_region_trace=true --xla_xprof_register_llo_debug_info=true" \
python3 -u /tmp/launcher.py -m benchmark.moe.bench_fused_moe \
    --num-experts <E> --topk <K> --hidden-size <D> --intermediate-size <F> \
    --num-tokens <T> --iters 3 --warmup-iters 1 \
    --imbalance-mode balanced --profile --profile-dir <output_dir>
```

### Pull traces to local
```bash
# From the GKE pod
kubectl cp <pod>:/tmp/<profile_dir> ./<local_dir>
```
