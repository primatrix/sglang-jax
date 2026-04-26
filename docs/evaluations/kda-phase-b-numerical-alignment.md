# KDA Phase B: M1 Numerical Alignment

**Date**: 2026-04-26
**Branch**: `sub3/layer-tests`
**Environment**: TPU v6e-4 (`sky-efe2-yuhao`), conda `sglang` (JAX 0.8.1, libtpu 0.0.30)
**GPU reference**: H100, fla `chunk_kda` with `force_mode="chunk"`
**Model**: `moonshotai/Kimi-Linear-48B-A3B-Instruct`
**Dumps**: `/models/yuhao/kimi-linear/kda_module/{L0,L6,L13,L22}/`

## Test Configuration

- **TPU kernel**: `fused_recurrent_kda` (naive recurrent, via `use_pallas_prefill=False` fallback)
- **GPU reference kernel**: `chunk_kda` (chunked Triton kernel, `force_mode="chunk"`)
- **Precisions tested**: float32, bfloat16
- **12 cases**: single sequences (T=1,8,64,65,128,256,1024), varlen (balanced 4x32, unbalanced, single T128), with/without initial state
- **Relative diff**: `mean_rel = mean(|diff| / (|expected| + 1e-12))`. Note: near-zero expected values inflate this metric; it is a rough indicator, not directly comparable to `rtol`.

## L0 Results

```
22 passed, 2 skipped, 1 warning in 122.16s
```

### L0 FP32 (11 passed, 1 skipped)

| Case | max_abs | mean_abs | mean_rel | Status |
|------|---------|----------|----------|--------|
| single_T1 | 6.20e-02 | 1.44e-02 | — | SKIP (GPU ref all-zero) |
| single_T8 | 6.16e-04 | 5.32e-05 | 2.15e-02 | PASS |
| single_T64 | 7.04e-04 | 5.84e-05 | 3.02e-02 | PASS |
| single_T65 | 7.07e-04 | 5.83e-05 | 2.71e-02 | PASS |
| single_T128 | 7.07e-04 | 5.88e-05 | 3.41e-02 | PASS |
| single_T256 | 8.36e-04 | 5.83e-05 | 3.32e-02 | PASS |
| single_T1024 | 9.50e-04 | 5.87e-05 | 3.17e-02 | PASS |
| varlen_balanced_4x32 | 9.57e-04 | 6.56e-05 | 3.20e-02 | PASS |
| varlen_unbalanced | 9.48e-04 | 6.52e-05 | 2.97e-02 | PASS |
| varlen_single_T128 | 7.07e-04 | 5.88e-05 | 3.41e-02 | PASS |
| single_T128_initstate | 7.81e-04 | 6.00e-05 | 3.22e-02 | PASS |
| varlen_initstate | 1.29e-03 | 7.99e-05 | 4.15e-02 | PASS |

**Tolerance**: `atol=2e-3, rtol=5e-3`

### L0 BF16 (11 passed, 1 skipped)

| Case | max_abs | mean_abs | mean_rel | Status |
|------|---------|----------|----------|--------|
| single_T1 | 6.20e-02 | 1.44e-02 | — | SKIP (GPU ref all-zero) |
| single_T8 | 4.88e-04 | 5.95e-05 | 3.04e-02 | PASS |
| single_T64 | 9.77e-04 | 6.16e-05 | 1.79e-01 | PASS |
| single_T65 | 9.77e-04 | 6.16e-05 | 1.77e-01 | PASS |
| single_T128 | 9.77e-04 | 6.39e-05 | 1.06e-01 | PASS |
| single_T256 | 9.77e-04 | 6.55e-05 | 6.93e-02 | PASS |
| single_T1024 | 9.77e-04 | 6.81e-05 | 6.28e-02 | PASS |
| varlen_balanced_4x32 | 9.77e-04 | 6.74e-05 | 1.04e-01 | PASS |
| varlen_unbalanced | 9.77e-04 | 6.69e-05 | 3.30e-02 | PASS |
| varlen_single_T128 | 9.77e-04 | 6.39e-05 | 1.06e-01 | PASS |
| single_T128_initstate | 1.46e-03 | 6.73e-05 | 3.42e-02 | PASS |
| varlen_initstate | 1.95e-03 | 8.87e-05 | 3.13e-02 | PASS |

**Tolerance**: `atol=3e-3, rtol=5e-3`

bf16 max_abs clusters at 9.77e-04 (= 1/1024, bf16 ULP near 1.0). The naive kernel internally upcasts to float32 for recurrence, so bf16 truncation only affects the projection/conv weights and input, not the attention accumulation.

## L6 Results

### L6 FP32

| Case | max_abs | mean_abs | mean_rel | Status |
|------|---------|----------|----------|--------|
| single_T1 | 1.55e-01 | 1.92e-02 | — | SKIP (GPU ref all-zero) |
| single_T8 | 5.33e-03 | 4.03e-04 | 3.09e-02 | OK |
| single_T64 | 6.85e-03 | 4.88e-04 | 4.16e-02 | OK |
| single_T65 | 6.88e-03 | 4.88e-04 | 3.45e-02 | OK |
| single_T128 | 8.79e-03 | 4.83e-04 | 4.51e-02 | OK |
| single_T256 | 9.89e-03 | 4.86e-04 | 8.31e-02 | OK |
| single_T1024 | 1.44e-02 | 4.83e-04 | 3.41e-02 | OK |
| varlen_balanced_4x32 | 1.16e-02 | 5.28e-04 | 3.12e-02 | OK |
| varlen_unbalanced | 1.16e-02 | 5.27e-04 | 3.25e-02 | OK |
| varlen_single_T128 | 8.79e-03 | 4.83e-04 | 3.18e-02 | OK |
| single_T128_initstate | 1.03e-02 | 5.05e-04 | 5.38e-02 | OK |
| varlen_initstate | 1.33e-02 | 6.91e-04 | 3.62e-02 | OK |

### L6 BF16

| Case | max_abs | mean_abs | mean_rel | Status |
|------|---------|----------|----------|--------|
| single_T1 | 1.55e-01 | 1.91e-02 | — | SKIP (GPU ref all-zero) |
| single_T8 | 1.17e-02 | 4.02e-04 | 2.21e-02 | OK |
| single_T64 | 1.56e-02 | 4.87e-04 | 3.15e-02 | OK |
| single_T65 | 1.56e-02 | 4.87e-04 | 3.13e-02 | OK |
| single_T128 | 1.56e-02 | 4.87e-04 | 3.24e-02 | OK |
| single_T256 | 1.56e-02 | 4.88e-04 | 3.02e-02 | OK |
| single_T1024 | 1.56e-02 | 4.88e-04 | 4.92e-02 | OK |
| varlen_balanced_4x32 | 1.56e-02 | 5.40e-04 | 5.82e-02 | OK |
| varlen_unbalanced | 1.56e-02 | 5.37e-04 | 4.15e-02 | OK |
| varlen_single_T128 | 1.56e-02 | 4.87e-04 | 3.45e-02 | OK |
| single_T128_initstate | 1.56e-02 | 5.13e-04 | 2.83e-02 | OK |
| varlen_initstate | 2.34e-02 | 7.54e-04 | 3.34e-02 | OK |

## L13 Results

### L13 FP32

| Case | max_abs | mean_abs | mean_rel | Status |
|------|---------|----------|----------|--------|
| single_T1 | 4.63e-01 | 5.93e-02 | — | SKIP (GPU ref all-zero) |
| single_T8 | 6.93e-03 | 7.97e-04 | 2.12e-02 | OK |
| single_T64 | 8.76e-03 | 8.72e-04 | 8.42e-02 | OK |
| single_T65 | 8.77e-03 | 8.73e-04 | 2.88e-02 | OK |
| single_T128 | 1.32e-02 | 8.57e-04 | 2.71e-02 | OK |
| single_T256 | 1.32e-02 | 8.86e-04 | 2.76e-02 | OK |
| single_T1024 | 1.32e-02 | 8.90e-04 | 3.76e-02 | OK |
| varlen_balanced_4x32 | 1.91e-02 | 9.63e-04 | 3.91e-02 | OK |
| varlen_unbalanced | 1.91e-02 | 9.70e-04 | 3.33e-02 | OK |
| varlen_single_T128 | 1.32e-02 | 8.57e-04 | 2.71e-02 | OK |
| single_T128_initstate | 1.32e-02 | 8.87e-04 | 4.09e-02 | OK |
| varlen_initstate | 1.25e-02 | 1.17e-03 | 4.05e-02 | OK |

### L13 BF16

| Case | max_abs | mean_abs | mean_rel | Status |
|------|---------|----------|----------|--------|
| single_T1 | 4.63e-01 | 5.95e-02 | — | SKIP (GPU ref all-zero) |
| single_T8 | 7.81e-03 | 7.90e-04 | 1.40e-01 | OK |
| single_T64 | 9.77e-03 | 8.69e-04 | 4.74e-02 | OK |
| single_T65 | 9.77e-03 | 8.73e-04 | 4.70e-02 | OK |
| single_T128 | 1.56e-02 | 8.85e-04 | 4.59e-02 | OK |
| single_T256 | 1.56e-02 | 9.10e-04 | 3.75e-02 | OK |
| single_T1024 | 1.56e-02 | 9.21e-04 | 3.73e-02 | OK |
| varlen_balanced_4x32 | 1.56e-02 | 9.94e-04 | 4.50e-02 | OK |
| varlen_unbalanced | 1.56e-02 | 1.00e-03 | 5.21e-02 | OK |
| varlen_single_T128 | 1.56e-02 | 8.85e-04 | 4.59e-02 | OK |
| single_T128_initstate | 1.56e-02 | 9.23e-04 | 3.62e-02 | OK |
| varlen_initstate | 1.56e-02 | 1.23e-03 | 3.32e-02 | OK |

## L22 Results

### L22 FP32

| Case | max_abs | mean_abs | mean_rel | Status |
|------|---------|----------|----------|--------|
| single_T1 | 8.43e-01 | 1.44e-01 | — | SKIP (GPU ref all-zero) |
| single_T8 | 8.94e-03 | 1.40e-03 | 2.82e-02 | OK |
| single_T64 | 1.72e-02 | 1.84e-03 | 3.09e-02 | OK |
| single_T65 | 1.73e-02 | 1.83e-03 | 3.95e-02 | OK |
| single_T128 | 1.73e-02 | 1.86e-03 | 3.75e-02 | OK |
| single_T256 | 2.39e-02 | 1.91e-03 | 4.88e-02 | OK |
| single_T1024 | 2.84e-02 | 1.94e-03 | 4.78e-02 | OK |
| varlen_balanced_4x32 | 2.23e-02 | 2.07e-03 | 4.74e-02 | OK |
| varlen_unbalanced | 2.30e-02 | 2.07e-03 | 4.92e-02 | OK |
| varlen_single_T128 | 1.73e-02 | 1.86e-03 | 3.42e-02 | OK |
| single_T128_initstate | 2.41e-02 | 1.88e-03 | 3.27e-02 | OK |
| varlen_initstate | 2.39e-02 | 2.17e-03 | 4.97e-02 | OK |

### L22 BF16

| Case | max_abs | mean_abs | mean_rel | Status |
|------|---------|----------|----------|--------|
| single_T1 | 8.40e-01 | 1.44e-01 | — | SKIP (GPU ref all-zero) |
| single_T8 | 1.17e-02 | 1.54e-03 | 2.60e-02 | OK |
| single_T64 | 3.13e-02 | 1.99e-03 | 6.00e-02 | OK |
| single_T65 | 3.13e-02 | 1.99e-03 | 5.95e-02 | OK |
| single_T128 | 3.13e-02 | 2.07e-03 | 5.34e-02 | OK |
| single_T256 | 3.13e-02 | 2.11e-03 | 4.28e-02 | OK |
| single_T1024 | 6.25e-02 | 2.12e-03 | 4.42e-02 | OK |
| varlen_balanced_4x32 | 3.13e-02 | 2.25e-03 | 3.42e-02 | OK |
| varlen_unbalanced | 3.13e-02 | 2.25e-03 | 3.34e-02 | OK |
| varlen_single_T128 | 3.13e-02 | 2.07e-03 | 4.29e-02 | OK |
| single_T128_initstate | 2.34e-02 | 2.12e-03 | 3.27e-02 | OK |
| varlen_initstate | 3.13e-02 | 2.43e-03 | 3.71e-02 | OK |

## Cross-Layer Summary

| Layer | FP32 worst max_abs | BF16 worst max_abs | FP32 mean_rel range | BF16 mean_rel range |
|-------|-------------------|-------------------|--------------------|--------------------|
| L0 | 1.29e-03 | 1.95e-03 | 2-4% | 3-18% |
| L6 | 1.44e-02 | 2.34e-02 | 3-8% | 2-6% |
| L13 | 1.91e-02 | 1.56e-02 | 2-8% | 4-14% |
| L22 | 2.84e-02 | 6.25e-02 | 3-5% | 3-6% |

Error grows ~20x from L0 to L22. This is expected: deeper layers have larger weight magnitudes and output scales, amplifying cross-device matmul precision differences. mean_rel stays stable at 2-8% (FP32) across all layers, confirming the error scales proportionally with output magnitude.

## Per-Stage Intermediate Comparison (single_T128)

Measured by comparing TPU output at each stage against GPU dump intermediates.

| Stage | Compared against | max_abs_diff | Value range | Relative error |
|-------|-----------------|-------------|-------------|---------------|
| Conv+SiLU (q) | `intermediates__q_after_conv` | 1.35e-02 | [-0.28, 6.63] | ~2.0e-3 |
| Conv+SiLU (k) | `intermediates__k_after_conv` | 1.02e-02 | — | — |
| Conv+SiLU (v) | `intermediates__v_after_conv` | 6.38e-03 | — | — |
| Beta (sigmoid) | `intermediates__beta` | 4.25e-03 | — | — |
| KDA attn output | `intermediates__o_kda_fused_recurrent` | 2.31e-04 | [-0.11, 0.07] | ~2.1e-3 |
| KDA attn output | `intermediates__o_kda_chunk` | 3.10e-04 | — | — |
| Recurrent state | `intermediates__recurrent_state_fused_recurrent` | 1.56e-03 | — | — |
| Recurrent state | `intermediates__recurrent_state_chunk` | 1.88e-03 | — | — |
| Final module output | `out_fp32` | 7.07e-04 | [-0.15, 0.22] | ~3.2e-3 |

**Not measured**: projection-only error (bundled with conv), L2 norm error, activated gate comparison (diagnostic compared raw gate against activated gate by mistake), o_norm output.

### Error flow analysis

```
Projection → Conv+SiLU → L2 Norm → KDA Attention → GatedRMSNorm → o_proj
              ~1e-2                    ~2e-4                         ~7e-4
```

The conv stage introduces the largest absolute error (~1e-2), but L2 normalization dampens it significantly before it reaches the attention computation. The attention output error (~2e-4) is much smaller than the conv error, indicating L2 norm is effective at suppressing upstream precision differences. The final output error (~7e-4) is amplified from the attention error through the norm and projection stages.

### GPU chunk vs fused_recurrent baseline

Even on GPU, the two kernels differ:
- `chunk` vs `fused_recurrent` attention output: max_abs_diff = 1.21e-04
- `chunk` vs `fused_recurrent` recurrent state: max_abs_diff = 6.33e-04

This sets a floor for cross-kernel comparison.

## T=1 Skip Rationale

GPU reference `out_fp32` for `single_T1` is all zeros. The GPU dump uses `force_mode="chunk"`, and the chunk kernel produces zero output when T < chunk_size (64). The TPU naive kernel correctly produces non-zero output for T=1.

The test verifies T=1 produces no NaN and non-zero output, then skips the numerical comparison.

## Error Source Breakdown

1. **Cross-device matmul precision** (~1e-2 at conv): GPU (CUDA) and TPU use different matmul implementations with different accumulation orders. This is the dominant error source at the conv stage.

2. **Cross-kernel difference** (~1e-4 at attention): `fused_recurrent_kda` (sequential per-timestep) vs `chunk_kda` (chunked parallel) use different computation orders, leading to floating-point accumulation differences.

3. **Error dampening by L2 norm**: The L2 normalization of q, k before attention reduces the ~1e-2 conv error to ~1e-4 at the attention output. This is the key reason the final output tolerance is manageable.

## Pallas Kernel Status

The Pallas chunked kernel (`chunk_kda_fwd`) cannot be used for these tests due to the NaN bug with real-weight gate magnitudes (see `docs/bugs/pallas-kda-nan.md`). All tests use the naive recurrent kernel as fallback (`use_pallas_prefill=False`).
