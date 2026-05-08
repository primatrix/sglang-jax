"""Layer 0 HBM local DMA envelope rows for Phase 1 calibration."""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Iterable
from typing import Any

from benchmark.moe.calibration.common import build_observation_row

SCENARIO_LAYER0_HBM_ENVELOPE = "layer0_hbm_envelope"
MATRIX_KIND = "hbm_equivalent_weight_tile"
STATUS_MEASURED = "measured"
STATUS_NOT_IMPLEMENTED = "not_implemented"

DEFAULT_WARMUP_RUNS = 3
DEFAULT_SAMPLE_RUNS = 9


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
    """Build Layer 0 rows, measuring HBM->VMEM DMA on TPU/Pallas when available."""

    if execution_mode != "pallas":
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
                source=source,
                metadata=metadata,
                status=STATUS_NOT_IMPLEMENTED,
                latency_ms_samples=[],
                implementation_note=_not_implemented_note(execution_mode, runtime),
            )
            for shape in shapes
        ]

    unavailable_note = _pallas_unavailable_note(runtime)
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
    measured_metadata = _with_measurement_metadata(
        metadata,
        warmup_runs=warmup_runs,
        sample_runs=sample_runs,
    )

    for shape in shapes:
        try:
            samples = _measure_hbm_local_dma_ms(
                tuple(shape.tile_shape),
                warmup_runs=warmup_runs,
                sample_runs=sample_runs,
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
                source=source,
                metadata=measured_metadata,
                status=STATUS_MEASURED,
                latency_ms_samples=samples,
                implementation_note=(
                    "Measured with a Pallas TPU HBM-to-VMEM local DMA kernel. "
                    "Each sample starts two bf16 DMA copies, one per t_packing slice, "
                    "and excludes compilation plus warmup runs."
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
        bytes_hbm=shape.bytes_per_fetch,
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
            "TPU/Pallas local HBM DMA measurements require execution_mode=pallas "
            "on a TPU backend."
        )
    backend = runtime.get("default_backend")
    return (
        "layer0_hbm_envelope did not emit synthetic latency samples. "
        f"execution_mode={execution_mode!r} is not a measured TPU/Pallas mode "
        f"for this runtime; observed JAX default_backend={backend!r}."
    )


def _pallas_unavailable_note(runtime: dict[str, Any]) -> str | None:
    backend = runtime.get("default_backend")
    if backend != "tpu":
        return (
            "layer0_hbm_envelope did not emit synthetic latency samples. "
            "execution_mode='pallas' requires JAX default_backend='tpu' for the "
            f"HBM local DMA kernel; observed default_backend={backend!r}."
        )

    try:
        import jax  # noqa: F401
        import jax.numpy as jnp  # noqa: F401
        from jax.experimental import pallas as pl  # noqa: F401
        from jax.experimental.pallas import tpu as pltpu  # noqa: F401
    except Exception as exc:
        return (
            "layer0_hbm_envelope could not import the JAX/Pallas TPU APIs needed "
            f"for measured HBM local DMA; {type(exc).__name__}: {exc}. "
            "No synthetic latency samples were emitted."
        )

    return None


def _measurement_failed_note(exc: Exception) -> str:
    return (
        "layer0_hbm_envelope Pallas HBM local DMA measurement failed before "
        f"producing trustworthy samples: {type(exc).__name__}: {exc}. "
        "No synthetic latency samples were emitted for this shape. Remaining "
        "work is to run/debug this kernel on v7x-32 and replace this fallback "
        "with measured samples once the Pallas DMA call compiles and executes."
    )


def _with_measurement_metadata(
    metadata: dict[str, Any],
    *,
    warmup_runs: int,
    sample_runs: int,
) -> dict[str, Any]:
    enriched = dict(metadata)
    enriched["benchmark"] = {
        "name": "layer0_hbm_local_dma",
        "warmup_runs": warmup_runs,
        "sample_runs": sample_runs,
        "copy_direction": "hbm_to_vmem",
        "timing": "host_perf_counter_ms_with_block_until_ready",
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


def _measure_hbm_local_dma_ms(
    tile_shape: tuple[int, int, int],
    *,
    warmup_runs: int,
    sample_runs: int,
) -> list[float]:
    import jax
    import jax.numpy as jnp

    run_dma = _build_hbm_local_dma_call(tile_shape)
    src = jnp.ones(tile_shape, dtype=jnp.bfloat16)
    jax.block_until_ready(src)

    jax.block_until_ready(run_dma(src))
    for _ in range(warmup_runs):
        jax.block_until_ready(run_dma(src))

    samples: list[float] = []
    for _ in range(sample_runs):
        start_ns = time.perf_counter_ns()
        result = run_dma(src)
        jax.block_until_ready(result)
        end_ns = time.perf_counter_ns()
        samples.append((end_ns - start_ns) / 1_000_000.0)
    return samples


def _build_hbm_local_dma_call(tile_shape: tuple[int, int, int]) -> Callable[[Any], Any]:
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    def _hbm_local_dma_kernel(src_ref, out_ref, scratch_ref, sem):
        del out_ref
        for packing_idx in range(src_ref.shape[0]):
            pltpu.make_async_copy(
                src_ref=src_ref.at[packing_idx],
                dst_ref=scratch_ref.at[packing_idx],
                sem=sem,
            ).start()
        pltpu.make_async_copy(src_ref=scratch_ref, dst_ref=scratch_ref, sem=sem).wait()

    @jax.jit
    def _run(src):
        kernel = pl.pallas_call(
            _hbm_local_dma_kernel,
            out_shape=jax.ShapeDtypeStruct((1,), jnp.bfloat16),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                in_specs=[pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)],
                out_specs=pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
                grid=(1,),
                scratch_shapes=[
                    pltpu.VMEM(tile_shape, jnp.bfloat16),
                    pltpu.SemaphoreType.DMA,
                ],
            ),
            compiler_params=pltpu.CompilerParams(
                has_side_effects=True,
                vmem_limit_bytes=32 * 1024 * 1024,
            ),
            name=f"layer0-hbm-local-dma-{tile_shape[0]}x{tile_shape[1]}x{tile_shape[2]}",
        )
        return kernel(src)

    return _run
