"""Layer 0 HBM copy envelope rows for Phase 1 calibration."""

from __future__ import annotations

import os
from collections.abc import Iterable
from math import prod
from typing import Any

from benchmark.moe.calibration.common import build_observation_row

SCENARIO_LAYER0_HBM_ENVELOPE = "layer0_hbm_envelope"
MATRIX_KIND = "hbm_equivalent_weight_tile"
STATUS_MEASURED = "measured"
STATUS_NOT_IMPLEMENTED = "not_implemented"

DEFAULT_WARMUP_RUNS = 3
DEFAULT_SAMPLE_RUNS = 9
DEFAULT_TRACE_DISCARD_RUNS = 1


def build_rows(
    *,
    suite: str,
    shapes: Iterable[Any],
    execution_mode: str,
    runtime: dict[str, Any],
    dtype: str,
    weight_dtype: str,
    t_packing: int,
    dma_count: int,
    source: dict[str, Any],
    metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build Layer 0 rows using a JAX HBM copy envelope on TPU.

    This follows the Ironwood accelerator microbenchmark style: benchmark a
    jitted `a.copy()`, collect profiler trace device durations, and account for
    read plus write HBM traffic. The fused-MoE tile dimensions are still used as
    the sweep matrix so Layer 1 can join on equivalent shapes.
    """

    if execution_mode == "pallas":
        execution_mode = "jax_trace"

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
                dma_count=dma_count,
                bytes_hbm=shape.bytes_per_fetch * 2,
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
                dma_count=dma_count,
                bytes_hbm=shape.bytes_per_fetch * 2,
                source=source,
                metadata=metadata,
                status=STATUS_NOT_IMPLEMENTED,
                latency_ms_samples=[],
                implementation_note=unavailable_note,
            )
            for shape in shapes
        ]

    rows: list[dict[str, Any]] = []
    warmup_runs = _positive_int_env("CALIBRATION_LAYER0_WARMUP_RUNS", DEFAULT_WARMUP_RUNS)
    sample_runs = _positive_int_env("CALIBRATION_LAYER0_SAMPLE_RUNS", DEFAULT_SAMPLE_RUNS)
    discard_runs = _nonnegative_int_env(
        "CALIBRATION_LAYER0_TRACE_DISCARD_RUNS", DEFAULT_TRACE_DISCARD_RUNS
    )
    trace_root = os.getenv("CALIBRATION_LAYER0_TRACE_ROOT", "/tmp/sglang_jax_layer0_hbm_trace")
    measured_metadata = _with_measurement_metadata(
        metadata,
        warmup_runs=warmup_runs,
        sample_runs=sample_runs,
        discard_runs=discard_runs,
        trace_root=trace_root,
    )

    for shape in shapes:
        try:
            samples = _measure_hbm_copy_ms(
                tuple(shape.tile_shape),
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
                    dma_count=dma_count,
                    bytes_hbm=shape.bytes_per_fetch * 2,
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
                dma_count=dma_count,
                bytes_hbm=shape.bytes_per_fetch * 2,
                source=source,
                metadata=measured_metadata,
                status=STATUS_MEASURED,
                latency_ms_samples=samples,
                implementation_note=(
                    "Measured with a JAX HBM copy envelope following the "
                    "Ironwood accelerator microbenchmark pattern. Each sample "
                    "uses profiler trace device duration for a jitted a.copy(); "
                    "bytes_hbm is read plus write traffic, while bytes_per_fetch "
                    "remains the fused-MoE equivalent tile size for joins."
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
    dma_count: int,
    bytes_hbm: int,
    source: dict[str, Any],
    metadata: dict[str, Any],
    status: str,
    latency_ms_samples: list[float],
    implementation_note: str,
) -> dict[str, Any]:
    return build_observation_row(
        scenario=SCENARIO_LAYER0_HBM_ENVELOPE,
        suite=suite,
        layer=0,
        path=shape.path_class,
        path_class=shape.path_class,
        dtype=dtype,
        weight_dtype=weight_dtype,
        t_packing=t_packing,
        bf=shape.bf,
        bd=shape.bd,
        tile_shape=shape.tile_shape,
        bytes_hbm=bytes_hbm,
        bytes_per_fetch=shape.bytes_per_fetch,
        dma_count=dma_count,
        status=status,
        execution_mode=execution_mode,
        latency_ms_samples=latency_ms_samples,
        runtime=runtime,
        source=source,
        metadata=metadata,
        implementation_note=implementation_note,
    )


def _not_implemented_note(execution_mode: str, runtime: dict[str, Any]) -> str:
    if execution_mode == "local_smoke":
        return (
            "layer0_hbm_envelope emitted schema-only rows on local_smoke; "
            "JAX trace-derived HBM copy measurements require a TPU backend."
        )
    backend = runtime.get("default_backend")
    return (
        "layer0_hbm_envelope did not emit synthetic latency samples. "
        f"execution_mode={execution_mode!r} is not a measured JAX trace mode "
        f"for this runtime; observed JAX default_backend={backend!r}."
    )


def _jax_trace_unavailable_note(runtime: dict[str, Any]) -> str | None:
    backend = runtime.get("default_backend")
    if backend != "tpu":
        return (
            "layer0_hbm_envelope did not emit synthetic latency samples. "
            "trace-derived HBM copy measurements require JAX default_backend='tpu'; "
            f"observed default_backend={backend!r}."
        )

    try:
        import jax  # noqa: F401
        import jax.numpy as jnp  # noqa: F401

        from benchmark.utils import multiple_iteration_timeit_from_trace  # noqa: F401
    except Exception as exc:
        return (
            "layer0_hbm_envelope could not import the JAX trace APIs needed for "
            f"measured HBM copy; {type(exc).__name__}: {exc}. "
            "No synthetic latency samples were emitted."
        )

    return None


def _measurement_failed_note(exc: Exception) -> str:
    return (
        "layer0_hbm_envelope JAX HBM copy measurement failed before "
        f"producing trustworthy samples: {type(exc).__name__}: {exc}. "
        "No synthetic latency samples were emitted for this shape. Remaining "
        "work is to run/debug the trace-derived copy benchmark on v7x-32 and "
        "replace this fallback with measured samples once profiler trace "
        "durations are available."
    )


def _with_measurement_metadata(
    metadata: dict[str, Any],
    *,
    warmup_runs: int,
    sample_runs: int,
    discard_runs: int,
    trace_root: str,
) -> dict[str, Any]:
    enriched = dict(metadata)
    enriched["benchmark"] = {
        "name": "layer0_jax_hbm_copy_envelope",
        "reference": "AI-Hypercomputer/accelerator-microbenchmarks Ironwood single_device_hbm_copy",
        "warmup_runs": warmup_runs,
        "sample_runs": sample_runs,
        "trace_discard_runs": discard_runs,
        "copy_direction": "hbm_to_hbm_copy",
        "traffic_class": "copy_read_write",
        "measured_hbm_traffic": "bytes_per_fetch read plus bytes_per_fetch write",
        "traffic_multiplier": 2,
        "timing": "jax_profiler_trace_device_duration_ms",
        "trace_root": trace_root,
    }
    return enriched


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


def _measure_hbm_copy_ms(
    tile_shape: tuple[int, int, int],
    *,
    warmup_runs: int,
    sample_runs: int,
    discard_runs: int,
    trace_root: str,
) -> list[float]:
    import jax
    import jax.numpy as jnp

    from benchmark.utils import multiple_iteration_timeit_from_trace

    src = jnp.ones((prod(tile_shape),), dtype=jnp.bfloat16)
    jax.block_until_ready(src)

    @jax.jit
    def copy_hbm(array):
        with jax.named_scope("SGLANG_JAX_LAYER0_HBM"):
            return array.copy()

    jax.block_until_ready(copy_hbm(src))
    task = f"layer0_hbm_copy_{tile_shape[0]}x{tile_shape[1]}x{tile_shape[2]}"
    return multiple_iteration_timeit_from_trace(
        compute_func=copy_hbm,
        data_generator=lambda: (src,),
        task=task,
        tries=sample_runs,
        warmup=warmup_runs,
        discard_initial_samples=discard_runs,
        trace_root=trace_root,
    )
