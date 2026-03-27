# xprof-analysis Skill

Analyze JAX/TPU profiling data using the xprof Python API. Extracts per-op time breakdown, roofline metrics, and bottleneck classification from `.xplane.pb` files.

## Usage

```bash
# Analyze a single profile
python -m skills.xprof_analysis.analyze ./profile_r8_current

# Compare two profiles (baseline vs optimized)
python -m skills.xprof_analysis.analyze --compare ./profile_r6 ./profile_r8_current

# Quick ops-only view
python -m skills.xprof_analysis.analyze --ops-only ./profile_r8_current
```

## What This Tool Shows

### Per-Op Roofline Analysis
- Each XLA op's **self time**, **HBM/VMEM bandwidth**, **operational intensity**
- **Bottleneck classification**: Compute / HBM / VMEM Read / VMEM Write
- **DMA stall percentage** per op
- **Bottleneck distribution** across all ops

### Comparison Mode
- **Kernel time delta** (e.g., R6 → R8: -294.8us, -4.3%)
- **Per-op time delta** for every shared op
- Visual markers for improved (✓) or regressed (✗) ops

## Key Insight: What xprof CAN and CANNOT See

### CAN see (XLA trace level):
- Time spent in each XLA op (matmul, allgather, sort, copy, custom-call...)
- The Pallas kernel as ONE custom-call op with total time
- Memory bandwidth per op (HBM, VMEM)
- Roofline bound classification per op

### CANNOT see (Pallas kernel internals):
- Internal breakdown within a Pallas kernel (scatter vs matmul vs activation)
- Per-instruction MXU/Scalar/Vector utilization within the kernel
- `utilization_viewer` gives global window-averaged counters, NOT per-kernel

### Why `utilization_viewer` doesn't help for per-kernel analysis:
The hardware counters in `utilization_viewer` accumulate over the ENTIRE profiling window
(~200ms), but the kernel runs for only ~6ms. The resulting utilization percentages (~0.05%)
are meaningless for kernel-level analysis. The same absolute counter values appear in both
R6 and R8 profiles because the kernel internals produce identical instruction counts —
the speedup comes from reduced pipeline stalls, not fewer instructions.

## Files

| File | Purpose |
|------|---------|
| `analyze.py` | Main analysis script |
| `skill.md` | This file |

## Dependencies

- `xprof` (v2.22.0, already installed on TPU pod)
- Profile data in standard xprof directory layout: `<dir>/plugins/profile/<timestamp>/<host>.xplane.pb`

## Profiling Capture

```bash
# Capture a profile with custom call region trace
LIBTPU_INIT_ARGS="--xla_enable_custom_call_region_trace=true --xla_xprof_register_llo_debug_info=true" \
python -m benchmark.moe.bench_fused_moe_kernel --profile --profile-dir ./profile_test --iters 3
```
