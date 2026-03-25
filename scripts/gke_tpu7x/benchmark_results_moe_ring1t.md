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
    --num-experts 64 --top-k 8 --hidden-size 8192 --intermediate-size 2048 \
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

### With Default Block Config

| num_tokens | sglang-jax (ms) | tpu-inference (ms) | speedup |
|:----------:|:---------------:|:------------------:|:-------:|
| 64         | 0.814           | 1.734              | 2.1x    |
| 128        | 0.863           | 1.604              | 1.9x    |
| 256        | 0.943           | 2.763              | 2.9x    |
| 512        | 2.099           | compilation failed | -       |
| 1024       | 3.839           | compilation failed | -       |

### With Tuned Block Config (from Ring-1T ep=32, scaled by local_num_tokens)

Tuned configs are adapted from Ring-1T (256 experts, ep=32) to our scaled config (64 experts, ep=8)
by mapping `local_num_tokens = num_tokens / ep_size`:
- ep=8/64t (local=8) uses ep=32/256t (local=8) config
- ep=8/128t (local=16) uses ep=32/512t (local=16) config
- ep=8/256t (local=32) uses ep=32/1024t (local=32) config
- ep=8/512t (local=64) uses ep=32/2048t (local=64) config
- ep=8/1024t (local=128) uses ep=32/4096t (local=128) config

| num_tokens | Default (ms) | Tuned (ms) | vs Default | tpu-inference (ms) | vs tpu-inference |
|:----------:|:------------:|:----------:|:----------:|:------------------:|:----------------:|
| 64         | 0.814        | 0.300      | **2.7x**   | 1.734              | **5.8x**         |
| 128        | 0.863        | 0.329      | **2.6x**   | 1.604              | **4.9x**         |
| 256        | 0.943        | 0.522      | **1.8x**   | 2.763              | **5.3x**         |
| 512        | 2.099        | 0.667      | **3.1x**   | compilation failed | -                |
| 1024       | 3.839        | 1.144      | **3.4x**   | compilation failed | -                |

**With tuned configs, sglang-jax is 1.8-3.4x faster than default, and 4.9-5.8x faster than tpu-inference.**

### Notes

- tpu-inference fails to compile at 512/1024 tokens due to VMEM overflow. The kernel's scratch buffer management cannot handle large batch sizes combined with large hidden/intermediate dimensions (8192/2048).
- sglang-jax handles all token counts without issues.
- Both kernels were run with balanced routing (no imbalance simulation).
- Tuned block configs are stored in `python/sgl_jax/srt/kernels/fused_moe/v1/tuned_block_configs.py`.

---

## Profiling (xprof with LLO utilization)

Profile traces were captured for sglang-jax using xprof custom call profiling to inspect LLO-level
utilization of the Pallas kernels. This section documents the complete end-to-end flow.

### Background: What is LLO Utilization?

XLA Custom Calls (i.e. Pallas kernels) are opaque to the standard XLA profiler. By enabling two
LIBTPU flags, xprof can show per-hardware-unit utilization **inside** each custom call:

| Row in Trace Viewer | What it shows |
|---|---|
| MXU | Matrix Unit utilization (matmuls) |
| Scalar ALU | Scalar arithmetic |
| Vector ALU | Vector arithmetic |
| Vector Load / Store | HBM ↔ VMEM data movement |
| Vector Fills / Spills | VMEM spill traffic |
| XLU | Cross-Lane Unit (permutes, reductions) |

Reference: [xprof custom_call_profiling.md](https://github.com/openxla/xprof/blob/master/docs/custom_call_profiling.md)

---

### Step 1: Prepare a Profile Launcher

**The key requirement**: `LIBTPU_INIT_ARGS` must be set **before** JAX/libtpu is imported.
A regular `launcher.py` imports JAX immediately, so we need a separate `profile_launcher.py`.

Create `/tmp/profile_launcher.py` on the pod (or see `scripts/gke_tpu7x/profile_moe.py` for reference):

```python
#!/usr/bin/env python3
import os, sys, runpy

# ---- MUST be set BEFORE importing JAX ----
_xla_flags = (
    "--xla_enable_custom_call_region_trace=true "
    "--xla_xprof_register_llo_debug_info=true"
)
existing = os.environ.get("LIBTPU_INIT_ARGS", "")
os.environ["LIBTPU_INIT_ARGS"] = (existing + " " + _xla_flags).strip()

# Standard launcher setup
REPO_ROOT = "/tmp/sglang-jax"
sys.path.insert(0, os.path.join(REPO_ROOT, "python"))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

import jax
jax.distributed.initialize()

script_path = os.path.join(REPO_ROOT, sys.argv[1])
sys.argv = [sys.argv[1]] + sys.argv[2:]
runpy.run_path(script_path, run_name="__main__")
```

Copy to both containers:
```bash
for C in <WORKLOAD>-1 <WORKLOAD>-2; do
  kubectl cp /tmp/profile_launcher.py <POD>:/tmp/profile_launcher.py -c $C
done
```

### Step 2: Run Profiling on TPU Pod

Both containers must run the same command simultaneously (JAX multi-process requirement):

```bash
POD=<pod-name>
WL=<workload-name>

PROFILE_CMD="python3 -u /tmp/profile_launcher.py benchmark/moe/bench_fused_moe.py \
  --num-experts 64 --top-k 8 --hidden-size 8192 --intermediate-size 2048 \
  --num-tokens 128 --iters 3 --warmup-iters 1 \
  --imbalance-mode balanced --profile --profile-dir /tmp/profile_output"

# Worker in background
kubectl exec $POD -c ${WL}-2 -- bash -c "$PROFILE_CMD" 2>&1 &
BGPID=$!

# Main in foreground
kubectl exec $POD -c ${WL}-1 -- bash -c "$PROFILE_CMD" 2>&1

kill $BGPID 2>/dev/null; wait $BGPID 2>/dev/null
```

Successful output looks like:
```
LIBTPU_INIT_ARGS=--xla_enable_custom_call_region_trace=true --xla_xprof_register_llo_debug_info=true
[Process 0] JAX 0.8.1, 8 devices, local 4
  Profiling to: /tmp/profile_output/case_128t_64e_ep8
  Profile saved to: /tmp/profile_output/case_128t_64e_ep8
```

### Step 3: Pull Trace Files to Local

The trace files are large (~90 MB total). Use GCS as intermediate for reliable transfer:

```bash
# On pod: upload to GCS
kubectl exec $POD -c ${WL}-1 -- bash -c '
TRACE_DIR=$(find /tmp/profile_output -name "*.xplane.pb" -exec dirname {} \;)
gsutil cp ${TRACE_DIR}/*.xplane.pb gs://<bucket>/profile_tmp/
gsutil cp ${TRACE_DIR}/*.trace.json.gz gs://<bucket>/profile_tmp/
'

# On local: download from GCS
mkdir -p profile_output/
gsutil cp gs://<bucket>/profile_tmp/xplane.pb profile_output/
gsutil cp gs://<bucket>/profile_tmp/trace.json.gz profile_output/
```

Alternative (direct kubectl cp, may truncate large files):
```bash
kubectl cp $POD:/tmp/profile_output ./profile_output -c ${WL}-1
```

Generated files:
- `*.xplane.pb` (~83 MB) — full XPlane protobuf, contains LLO utilization data
- `*.trace.json.gz` (~10 MB) — pre-converted trace events

### Step 4: View Results with TensorBoard

**Important**: TensorBoard must run on **Linux** (not macOS). The xprof native module
(`_pywrap_profiler_plugin`) does not have macOS ARM64 builds. Run TensorBoard on the
TPU pod itself and port-forward to local.

#### 4a. Install TensorBoard on Pod

Version requirements (must be compatible):
```bash
kubectl exec $POD -c ${WL}-1 -- pip install \
  'tensorflow>=2.21' \
  'tensorboard>=2.20' \
  'tensorboard-plugin-profile>=2.22' \
  'xprof>=2.22' \
  'protobuf>=5,<7' \
  'setuptools<81'
```

#### 4b. Start TensorBoard on Pod

```bash
kubectl exec $POD -c ${WL}-1 -- bash -c "
nohup python3 -c '
from tensorboard import main as tb
import sys
sys.argv = [\"tensorboard\", \"--logdir=/tmp/profile_output/\", \"--port=6006\", \"--bind_all\", \"--load_fast=false\"]
tb.run_main()
tb.main()
' > /tmp/tb.log 2>&1 &"
```

Verify no profile plugin errors:
```bash
kubectl exec $POD -c ${WL}-1 -- grep -i error /tmp/tb.log
# Should only show CUDA-related warnings (expected on TPU), no profile plugin errors
```

#### 4c. Port-Forward to Local

```bash
kubectl port-forward $POD 6006:6006
```

#### 4d. Open in Browser

1. Open **http://localhost:6006/**
2. Top-right dropdown: select **Profile**
3. Left panel "Tools": select **trace_viewer**
4. Left panel "Runs": select the run (e.g. `case_128t_64e_ep8/2026_03_25_03_31_38`)
5. Left panel "Hosts": select the TPU host

#### 4e. Navigate the Trace Viewer

The trace viewer uses keyboard shortcuts (scroll wheel does NOT zoom):

| Key | Action |
|-----|--------|
| **W** | Zoom in |
| **S** | Zoom out |
| **A** | Pan left |
| **D** | Pan right |
| **1** | Select mode (click to inspect events) |
| **2** | Pan mode (drag to move) |
| **3** | Zoom mode (drag to zoom into region) |
| **4** | Timing mode (measure duration between points) |

**What to look for:**
- **XLA Modules / XLA Ops** rows: shows the fused_moe kernel execution span
- **MXU** row: matrix unit utilization (higher = better for matmul-bound kernels)
- **Vector Load/Store** rows: HBM bandwidth utilization
- **Framework Name Scope** row: shows the JAX call hierarchy (jit → fused_ep_moe → pallas_call)
- **Source code** row: links back to the kernel source file and line number

### Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| "No dashboards are active" | TensorBoard can't find profile data | Ensure `--logdir` points to parent of the run dir that contains `plugins/profile/` |
| Profile tab says "plugin has moved" | `tensorboard-plugin-profile` not installed | `pip install tensorboard-plugin-profile>=2.22` |
| `_pywrap_profiler_plugin` import error | Running on macOS (no native module) | Run TensorBoard on the Linux pod, port-forward to local |
| `proto.id() > INT_MAX` | tensorboard-plugin-profile too old for JAX 0.8.1 | Upgrade to `tensorboard-plugin-profile>=2.22` + `tensorflow>=2.21` |
| `'MessageFactory' has no attribute 'GetPrototype'` | protobuf version mismatch | `pip install 'protobuf>=5,<7'` |
| `ModuleNotFoundError: pkg_resources` | setuptools too new (>=82) | `pip install 'setuptools<81'` |
| No LLO utilization rows (MXU etc.) | LIBTPU flags not set before JAX import | Use `profile_launcher.py` that sets flags before `import jax` |
| Profiling hangs after warmup | libtpu version issue | Ensure `libtpu-nightly>=0.0.38.dev` is installed |

---

## Conclusions

1. **Tuned block configs provide 1.8-3.4x speedup** over default configs on the Ring-1T MoE config, bringing sglang-jax to 0.3-1.1ms range.

2. **With tuned configs, sglang-jax fused_moe is 4.9-5.8x faster** than tpu-inference's ring-based implementation (64 experts, top_k=8, hidden=8192, intermediate=2048, ep=8).

3. **tpu-inference has scalability limitations** — it fails to compile with 512+ tokens on Ring-1T dimensions due to VMEM overflow in scratch buffer allocation.

4. **sglang-jax scales smoothly** from 64 to 1024 tokens, with tuned latency growing sub-linearly (0.3ms → 1.1ms for 16x more tokens).

5. **Profile traces confirm** that LLO utilization is visible for Pallas custom calls when the correct LIBTPU flags and libtpu-nightly are used.

---

## How to Run with Other Configs

### sglang-jax benchmark
```bash
python3 -u /tmp/profile_launcher.py benchmark/moe/bench_fused_moe.py \
    --num-experts <E> --top-k <K> --hidden-size <D> --intermediate-size <F> \
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
# Use profile_launcher.py (NOT launcher.py) — it sets LIBTPU flags before JAX import
python3 -u /tmp/profile_launcher.py benchmark/moe/bench_fused_moe.py \
    --num-experts <E> --top-k <K> --hidden-size <D> --intermediate-size <F> \
    --num-tokens <T> --iters 3 --warmup-iters 1 \
    --imbalance-mode balanced --profile --profile-dir <output_dir>
```

### Pull traces to local
```bash
# Reliable method: via GCS
kubectl exec <POD> -c <WORKLOAD>-1 -- gsutil cp /tmp/<profile_dir>/**/*.xplane.pb gs://<bucket>/profile_tmp/
gsutil cp gs://<bucket>/profile_tmp/*.xplane.pb ./<local_dir>/

# Quick method (may truncate files > ~50MB):
kubectl cp <POD>:/tmp/<profile_dir> ./<local_dir> -c <WORKLOAD>-1
```
