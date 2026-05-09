"""Layer 0 all-to-all collective envelope rows for Phase 1 calibration."""

from __future__ import annotations

import os
from collections.abc import Iterable
from math import prod
from typing import Any

from benchmark.moe.calibration.common import build_observation_row

SCENARIO_LAYER0_A2A_ENVELOPE = "layer0_a2a_envelope"
STATUS_MEASURED = "measured"
STATUS_NOT_IMPLEMENTED = "not_implemented"

DEFAULT_WARMUP_RUNS = 3
DEFAULT_SAMPLE_RUNS = 5
DEFAULT_TRACE_DISCARD_RUNS = 1
BF16_BYTES = 2
BASE_N = 8
BASE_K = 128


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
    warmup_runs = _positive_int_env("CALIBRATION_LAYER0_A2A_WARMUP_RUNS", DEFAULT_WARMUP_RUNS)
    sample_runs = _positive_int_env("CALIBRATION_LAYER0_A2A_SAMPLE_RUNS", DEFAULT_SAMPLE_RUNS)
    discard_runs = _nonnegative_int_env(
        "CALIBRATION_LAYER0_A2A_TRACE_DISCARD_RUNS", DEFAULT_TRACE_DISCARD_RUNS
    )
    trace_root = os.getenv("CALIBRATION_LAYER0_A2A_TRACE_ROOT", "/tmp/sglang_jax_layer0_a2a_trace")

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
            samples = _measure_all_to_all_ms(
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
                    "Measured with a JAX lax.all_to_all collective envelope, "
                    "following the Ironwood ICI collectives microbenchmark "
                    "shape and shard_map pattern. This is a Layer 0 collective "
                    "hardware envelope, not the fused-MoE Pallas remote DMA "
                    "scatter/gather implementation."
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
    return build_observation_row(
        scenario=SCENARIO_LAYER0_A2A_ENVELOPE,
        suite=suite,
        layer=0,
        path=shape.path_class,
        path_class=shape.path_class,
        dtype=dtype,
        weight_dtype=weight_dtype,
        t_packing=t_packing,
        bf=shape.matrix_dim,
        bd=BASE_N * BASE_K,
        tile_shape=_matrix_shape(shape),
        bytes_hbm=_nominal_transferred_bytes_per_device(shape),
        bytes_per_fetch=_input_bytes(shape),
        dma_count=0,
        status=status,
        execution_mode=execution_mode,
        latency_ms_samples=latency_ms_samples,
        runtime=runtime,
        source=source,
        metadata=_metadata_for_shape(metadata, shape),
        implementation_note=implementation_note,
    )


def _metadata_for_shape(metadata: dict[str, Any], shape: Any) -> dict[str, Any]:
    enriched = dict(metadata)
    enriched["collective"] = {
        "operation": "jax_lax_all_to_all",
        "path_class": shape.path_class,
        "matrix_shape": list(_matrix_shape(shape)),
        "matrix_dim": shape.matrix_dim,
        "dtype_bytes": BF16_BYTES,
        "mesh_shape": shape.mesh_shape,
        "slice_topology": shape.slice_topology,
        "ici_size": shape.ici_size,
        "sharding_strategy": shape.sharding_strategy,
        "sharding_axes": list(_sharding_axes(shape)),
        "collective_rank": _collective_rank(shape),
        "bytes_input_per_device": _input_bytes(shape),
        "nominal_transferred_bytes_per_device": _nominal_transferred_bytes_per_device(shape),
        "traffic_formula": "input_bytes * (rank - 1) / rank",
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
        "name": "layer0_jax_a2a_collective_envelope",
        "reference": "AI-Hypercomputer/accelerator-microbenchmarks Ironwood all_to_all",
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
            "layer0_a2a_envelope emitted schema-only rows on local_smoke; "
            "JAX trace-derived all_to_all measurements require a TPU backend."
        )
    backend = runtime.get("default_backend")
    return (
        "layer0_a2a_envelope did not emit synthetic latency samples. "
        f"execution_mode={execution_mode!r} is not a measured JAX trace mode "
        f"for this runtime; observed JAX default_backend={backend!r}."
    )


def _jax_trace_unavailable_note(runtime: dict[str, Any]) -> str | None:
    backend = runtime.get("default_backend")
    if backend != "tpu":
        return (
            "layer0_a2a_envelope did not emit synthetic latency samples. "
            "trace-derived all_to_all measurements require JAX default_backend='tpu'; "
            f"observed default_backend={backend!r}."
        )

    try:
        import jax  # noqa: F401
        import jax.numpy as jnp  # noqa: F401
        from jax.experimental import mesh_utils  # noqa: F401
        from jax.experimental.shard_map import shard_map  # noqa: F401
        from jax.sharding import Mesh  # noqa: F401
        from jax.sharding import PartitionSpec as P  # noqa: F401

        from benchmark.utils import multiple_iteration_timeit_from_trace  # noqa: F401
    except Exception as exc:
        return (
            "layer0_a2a_envelope could not import the JAX trace APIs needed "
            f"for measured all_to_all; {type(exc).__name__}: {exc}. "
            "No synthetic latency samples were emitted."
        )

    return None


def _measurement_failed_note(exc: Exception) -> str:
    return (
        "layer0_a2a_envelope JAX all_to_all measurement failed before "
        f"producing trustworthy samples: {type(exc).__name__}: {exc}. "
        "No synthetic latency samples were emitted for this shape."
    )


def _measure_all_to_all_ms(
    shape: Any,
    *,
    warmup_runs: int,
    sample_runs: int,
    discard_runs: int,
    trace_root: str,
) -> list[float]:
    import jax
    import jax.numpy as jnp
    from jax.experimental import mesh_utils
    from jax.experimental.shard_map import shard_map
    from jax.sharding import Mesh
    from jax.sharding import PartitionSpec as P

    from benchmark.utils import multiple_iteration_timeit_from_trace

    mesh_dims = _mesh_dims(shape.mesh_shape)
    axis_names = tuple(f"d_{idx}" for idx in range(len(mesh_dims)))
    devices = jax.devices()[: shape.ici_size]
    if len(devices) < shape.ici_size:
        raise ValueError(f"Need {shape.ici_size} devices, found {len(devices)}.")
    mesh = Mesh(mesh_utils.create_device_mesh(mesh_dims, devices=devices), axis_names)
    sharding_axis = _sharding_axes(shape)
    matrix_shape = _matrix_shape(shape)

    def all_to_all(x):
        with jax.named_scope("SGLANG_JAX_LAYER0_A2A"):
            return jax.lax.all_to_all(
                x,
                sharding_axis,
                split_axis=0,
                concat_axis=0,
                tiled=True,
            )

    collective = jax.jit(
        shard_map(
            all_to_all,
            mesh,
            in_specs=P(None, None, None),
            out_specs=P(None, None, None),
            check_rep=False,
        )
    )
    src = jnp.ones(matrix_shape, dtype=jnp.bfloat16)
    jax.block_until_ready(collective(src))
    task = (
        "layer0_a2a_"
        f"{shape.path_class}_{shape.matrix_dim}_{shape.mesh_shape}_{shape.sharding_strategy}"
    )
    return multiple_iteration_timeit_from_trace(
        compute_func=collective,
        data_generator=lambda: (src,),
        task=task,
        tries=sample_runs,
        warmup=warmup_runs,
        discard_initial_samples=discard_runs,
        trace_root=trace_root,
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


def _matrix_shape(shape: Any) -> tuple[int, int, int]:
    return (shape.matrix_dim, BASE_N, BASE_K)


def _input_bytes(shape: Any) -> int:
    return prod(_matrix_shape(shape)) * BF16_BYTES


def _nominal_transferred_bytes_per_device(shape: Any) -> int:
    rank = _collective_rank(shape)
    return int(_input_bytes(shape) * (rank - 1) / rank)


def _mesh_dims(mesh_shape: str) -> tuple[int, ...]:
    return tuple(int(part) for part in mesh_shape.split("x") if part)


def _strategy_dims(sharding_strategy: str) -> tuple[int, ...]:
    return tuple(int(part) for part in sharding_strategy.split("x") if part)


def _sharding_axes(shape: Any) -> tuple[str, ...]:
    return tuple(
        f"d_{idx}" for idx, dim in enumerate(_strategy_dims(shape.sharding_strategy)) if dim > 1
    )


def _collective_rank(shape: Any) -> int:
    return prod(dim for dim in _strategy_dims(shape.sharding_strategy) if dim > 1)
