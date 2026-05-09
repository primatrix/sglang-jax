"""Layer 0 GEMM/MXU envelope rows for Phase 1 calibration."""

from __future__ import annotations

import os
from collections.abc import Iterable
from typing import Any

from benchmark.moe.calibration.common import build_observation_row

SCENARIO_LAYER0_GEMM_ENVELOPE = "layer0_gemm_envelope"
STATUS_MEASURED = "measured"
STATUS_NOT_IMPLEMENTED = "not_implemented"

DEFAULT_WARMUP_RUNS = 3
DEFAULT_SAMPLE_RUNS = 9
DEFAULT_TRACE_DISCARD_RUNS = 1
BF16_BYTES = 2


def build_rows(
    *,
    suite: str,
    shapes: Iterable[Any],
    execution_mode: str,
    runtime: dict[str, Any],
    dtype: str,
    weight_dtype: str,
    t_packing: int,
    source: dict[str, Any],
    metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    if execution_mode != "jax_trace":
        return [
            _make_row(
                suite=suite,
                shape=shape,
                execution_mode=execution_mode,
                runtime=runtime,
                dtype=dtype,
                weight_dtype=weight_dtype,
                t_packing=t_packing,
                source=source,
                metadata=metadata,
                status=STATUS_NOT_IMPLEMENTED,
                latency_ms_samples=[],
                implementation_note=_not_implemented_note(execution_mode, runtime),
            )
            for shape in shapes
        ]

    unavailable_note = _jax_trace_unavailable_note(runtime)
    if unavailable_note is not None:
        return [
            _make_row(
                suite=suite,
                shape=shape,
                execution_mode=execution_mode,
                runtime=runtime,
                dtype=dtype,
                weight_dtype=weight_dtype,
                t_packing=t_packing,
                source=source,
                metadata=metadata,
                status=STATUS_NOT_IMPLEMENTED,
                latency_ms_samples=[],
                implementation_note=unavailable_note,
            )
            for shape in shapes
        ]

    rows: list[dict[str, Any]] = []
    warmup_runs = _positive_int_env("CALIBRATION_LAYER0_GEMM_WARMUP_RUNS", DEFAULT_WARMUP_RUNS)
    sample_runs = _positive_int_env("CALIBRATION_LAYER0_GEMM_SAMPLE_RUNS", DEFAULT_SAMPLE_RUNS)
    discard_runs = _nonnegative_int_env(
        "CALIBRATION_LAYER0_GEMM_TRACE_DISCARD_RUNS", DEFAULT_TRACE_DISCARD_RUNS
    )
    trace_root = os.getenv(
        "CALIBRATION_LAYER0_GEMM_TRACE_ROOT", "/tmp/sglang_jax_layer0_gemm_trace"
    )

    for shape in shapes:
        measured_metadata = _with_measurement_metadata(
            metadata,
            shape=shape,
            warmup_runs=warmup_runs,
            sample_runs=sample_runs,
            discard_runs=discard_runs,
            trace_root=trace_root,
        )
        try:
            samples = _measure_gemm_ms(
                shape,
                warmup_runs=warmup_runs,
                sample_runs=sample_runs,
                discard_runs=discard_runs,
                trace_root=trace_root,
            )
        except Exception as exc:
            rows.append(
                _make_row(
                    suite=suite,
                    shape=shape,
                    execution_mode=execution_mode,
                    runtime=runtime,
                    dtype=dtype,
                    weight_dtype=weight_dtype,
                    t_packing=t_packing,
                    source=source,
                    metadata=measured_metadata,
                    status=STATUS_NOT_IMPLEMENTED,
                    latency_ms_samples=[],
                    implementation_note=_measurement_failed_note(exc),
                )
            )
            continue

        rows.append(
            _make_row(
                suite=suite,
                shape=shape,
                execution_mode=execution_mode,
                runtime=runtime,
                dtype=dtype,
                weight_dtype=weight_dtype,
                t_packing=t_packing,
                source=source,
                metadata=measured_metadata,
                status=STATUS_MEASURED,
                latency_ms_samples=samples,
                implementation_note=(
                    "Measured with a JAX bf16 matmul envelope using profiler "
                    "trace device duration. This is a generic MXU envelope row, "
                    "not fused-MoE dot scheduling."
                ),
            )
        )
    return rows


def _make_row(
    *,
    suite: str,
    shape: Any,
    execution_mode: str,
    runtime: dict[str, Any],
    dtype: str,
    weight_dtype: str,
    t_packing: int,
    source: dict[str, Any],
    metadata: dict[str, Any],
    status: str,
    latency_ms_samples: list[float],
    implementation_note: str,
) -> dict[str, Any]:
    flops = _gemm_flops(shape)
    bytes_hbm = _gemm_bytes_hbm(shape)
    row = build_observation_row(
        scenario=SCENARIO_LAYER0_GEMM_ENVELOPE,
        suite=suite,
        layer=0,
        path=shape.path,
        path_class=shape.path,
        dtype=dtype,
        weight_dtype=weight_dtype,
        t_packing=t_packing,
        bf=shape.m,
        bd=shape.k,
        tile_shape=(shape.m, shape.k, shape.n),
        bytes_hbm=bytes_hbm,
        bytes_per_fetch=bytes_hbm,
        dma_count=0,
        status=status,
        execution_mode=execution_mode,
        latency_ms_samples=latency_ms_samples,
        runtime=runtime,
        source=source,
        metadata=_metadata_for_shape(metadata, shape),
        implementation_note=implementation_note,
    )
    row.update(
        {
            "m": shape.m,
            "k": shape.k,
            "n": shape.n,
            "flops": flops,
            "bytes_lhs": shape.m * shape.k * BF16_BYTES,
            "bytes_rhs": shape.k * shape.n * BF16_BYTES,
            "bytes_out": shape.m * shape.n * BF16_BYTES,
            "tflops_per_device": _tflops_per_device(flops, row.get("latency_ms_p50")),
        }
    )
    return row


def _metadata_for_shape(metadata: dict[str, Any], shape: Any) -> dict[str, Any]:
    enriched = dict(metadata)
    enriched["gemm"] = {
        "path": shape.path,
        "m": shape.m,
        "k": shape.k,
        "n": shape.n,
        "flops": _gemm_flops(shape),
        "bytes_lhs": shape.m * shape.k * BF16_BYTES,
        "bytes_rhs": shape.k * shape.n * BF16_BYTES,
        "bytes_out": shape.m * shape.n * BF16_BYTES,
        "operation": "jax_bfloat16_matmul",
    }
    return enriched


def _with_measurement_metadata(
    metadata: dict[str, Any],
    *,
    shape: Any,
    warmup_runs: int,
    sample_runs: int,
    discard_runs: int,
    trace_root: str,
) -> dict[str, Any]:
    enriched = _metadata_for_shape(metadata, shape)
    enriched["benchmark"] = {
        "name": "layer0_jax_gemm_envelope",
        "warmup_runs": warmup_runs,
        "sample_runs": sample_runs,
        "trace_discard_runs": discard_runs,
        "timing": "jax_profiler_trace_device_duration_ms",
        "trace_root": trace_root,
    }
    return enriched


def _not_implemented_note(execution_mode: str, runtime: dict[str, Any]) -> str:
    if execution_mode == "local_smoke":
        return (
            "layer0_gemm_envelope emitted schema-only rows on local_smoke; "
            "JAX trace-derived GEMM measurements require a TPU backend."
        )
    backend = runtime.get("default_backend")
    return (
        "layer0_gemm_envelope did not emit synthetic latency samples. "
        f"execution_mode={execution_mode!r} is not a measured JAX trace mode "
        f"for this runtime; observed JAX default_backend={backend!r}."
    )


def _jax_trace_unavailable_note(runtime: dict[str, Any]) -> str | None:
    backend = runtime.get("default_backend")
    if backend != "tpu":
        return (
            "layer0_gemm_envelope did not emit synthetic latency samples. "
            "trace-derived GEMM measurements require JAX default_backend='tpu'; "
            f"observed default_backend={backend!r}."
        )

    try:
        import jax  # noqa: F401
        import jax.numpy as jnp  # noqa: F401

        from benchmark.utils import multiple_iteration_timeit_from_trace  # noqa: F401
    except Exception as exc:
        return (
            "layer0_gemm_envelope could not import the JAX trace APIs needed "
            f"for measured GEMM; {type(exc).__name__}: {exc}. "
            "No synthetic latency samples were emitted."
        )

    return None


def _measurement_failed_note(exc: Exception) -> str:
    return (
        "layer0_gemm_envelope JAX GEMM measurement failed before producing "
        f"trustworthy samples: {type(exc).__name__}: {exc}. "
        "No synthetic latency samples were emitted for this shape."
    )


def _positive_int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if not value:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def _nonnegative_int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if not value:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return parsed if parsed >= 0 else default


def _measure_gemm_ms(
    shape: Any,
    *,
    warmup_runs: int,
    sample_runs: int,
    discard_runs: int,
    trace_root: str,
) -> list[float]:
    import jax
    import jax.numpy as jnp

    from benchmark.utils import multiple_iteration_timeit_from_trace

    lhs = jnp.ones((shape.m, shape.k), dtype=jnp.bfloat16)
    rhs = jnp.ones((shape.k, shape.n), dtype=jnp.bfloat16)
    jax.block_until_ready((lhs, rhs))

    @jax.jit
    def gemm(lhs_arg, rhs_arg):
        return lhs_arg @ rhs_arg

    jax.block_until_ready(gemm(lhs, rhs))
    task = f"layer0_gemm_{shape.path}_{shape.m}x{shape.k}x{shape.n}"
    return multiple_iteration_timeit_from_trace(
        compute_func=gemm,
        data_generator=lambda: (lhs, rhs),
        task=task,
        tries=sample_runs,
        warmup=warmup_runs,
        discard_initial_samples=discard_runs,
        trace_root=trace_root,
    )


def _gemm_flops(shape: Any) -> int:
    return 2 * shape.m * shape.k * shape.n


def _gemm_bytes_hbm(shape: Any) -> int:
    return (shape.m * shape.k + shape.k * shape.n + shape.m * shape.n) * BF16_BYTES


def _tflops_per_device(flops: int, latency_ms_p50: Any) -> float | None:
    if not isinstance(latency_ms_p50, (int, float)) or latency_ms_p50 <= 0:
        return None
    return flops / (float(latency_ms_p50) * 1e9)
