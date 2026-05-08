"""Layer 1 fused-MoE dense weight tile DMA calibration.

This module records the Phase 1 mapping from #2 JSONL rows to the real
`start_fetch_bw1`, `start_fetch_bw2`, and `start_fetch_bw3` HBM->VMEM copies in
`python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py`.

On TPU with `execution_mode=pallas`, it runs a small Pallas kernel that mirrors
the primary bf16 weight tile async copies and self-copy wait anchors used by
fused-MoE. Non-TPU and local smoke paths continue to emit schema-only rows.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from typing import Any, Literal, Protocol, cast

from benchmark.moe.calibration.common import (
    build_observation_row,
    collect_runtime_identity,
)

SCENARIO_LAYER1_WEIGHT_TILE_DMA = "layer1_weight_tile_dma"
SUITE_V7X32_BF16_WEIGHT_TILES = "v7x32_bf16_weight_tiles"
DTYPE = "bfloat16"
WEIGHT_DTYPE = "bfloat16"
BF16_BYTES = 2
T_PACKING = 2
DMA_COUNT = T_PACKING

KERNEL_PATH = "python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py"
STATUS_MEASURED = "measured"
STATUS_NOT_IMPLEMENTED = "not_implemented"
DEFAULT_WARMUP_RUNS = 3
DEFAULT_SAMPLE_RUNS = 9
DEFAULT_TRACE_DISCARD_RUNS = 1
HIDDEN_SIZE = 8192
H_PER_T_PACKING = HIDDEN_SIZE // T_PACKING
VMEM_LIMIT_BYTES = 96 * 1024 * 1024
LOCAL_SEMAPHORE_SHAPE = (2, 14)
IMPLEMENTATION_NOTE = (
    "Measured with a Pallas TPU microkernel that issues the fused-MoE "
    "start_fetch_bw1/bw2/bw3 primary bf16 weight HBM->VMEM async copies for "
    "t_packing=2, then waits with the same VMEM self-copy semaphore pattern. "
    "The kernel is side-effect-only and does not add an HBM anchor write, so "
    "bytes_hbm reports only the primary fused-MoE weight read traffic."
)

TARGET_RUNTIME_V7X32 = {
    "device_type": "v7x",
    "falcon_device_count": 32,
    "falcon_device_topo": "2x2x4",
    "replica": 4,
    "jax_device_count": 32,
    "jax_local_device_count": 8,
    "jax_process_count": 4,
    "chip_count": 16,
    "tensorcore_or_jax_device_count": 32,
}

WeightPath = Literal["w1", "w2", "w3"]
PathClass = Literal["w1w3", "w2"]


class WeightTileShapeLike(Protocol):
    path_class: str
    bf: int
    bd: int
    bytes_per_fetch: int
    tile_shape: tuple[int, int, int]


@dataclass(frozen=True)
class WeightDMAPathSpec:
    path: WeightPath
    path_class: PathClass
    start_fetch: str
    wait_fetch: str
    kernel_line: int
    weight_ref: str
    vmem_ref: str
    semaphore_index: int
    source_slice: str
    destination_slice: str
    scratch_shape: str


@dataclass(frozen=True)
class WeightTileDMAPlan:
    path: WeightPath
    path_class: PathClass
    bf: int
    bd: int
    tile_shape: tuple[int, int, int]
    bytes_per_fetch: int
    dma_count: int
    spec: WeightDMAPathSpec


WEIGHT_DMA_PATH_SPECS: dict[WeightPath, WeightDMAPathSpec] = {
    "w1": WeightDMAPathSpec(
        path="w1",
        path_class="w1w3",
        start_fetch="start_fetch_bw1(local_e_id, bw1_sem_id, bf_id, bd1_id)",
        wait_fetch="wait_fetch_bw1(local_e_id, bw1_sem_id, bf_id, bd1_id)",
        kernel_line=1182,
        weight_ref="w1_hbm",
        vmem_ref="b_w1_x2_vmem",
        semaphore_index=1,
        source_slice=(
            "w1_hbm[local_e_id, "
            "p * h_per_t_packing + bd1_id * bd1_per_t_packing : "
            "+ bd1_per_t_packing, bf_id * bf : + bf]"
        ),
        destination_slice="b_w1_x2_vmem[bw1_sem_id, p]",
        scratch_shape="(2, t_packing, bd1 // t_packing, bf)",
    ),
    "w2": WeightDMAPathSpec(
        path="w2",
        path_class="w2",
        start_fetch="start_fetch_bw2(local_e_id, bw2_sem_id, bf_id, bd2_id)",
        wait_fetch="wait_fetch_bw2(local_e_id, bw2_sem_id, bf_id, bd2_id)",
        kernel_line=1223,
        weight_ref="w2_hbm",
        vmem_ref="b_w2_x2_vmem",
        semaphore_index=2,
        source_slice=(
            "w2_hbm[local_e_id, bf_id * bf : + bf, "
            "p * h_per_t_packing + bd2_id * bd2_per_t_packing : "
            "+ bd2_per_t_packing]"
        ),
        destination_slice="b_w2_x2_vmem[bw2_sem_id, p]",
        scratch_shape="(2, t_packing, bf, bd2 // t_packing)",
    ),
    "w3": WeightDMAPathSpec(
        path="w3",
        path_class="w1w3",
        start_fetch="start_fetch_bw3(local_e_id, bw3_sem_id, bf_id, bd3_id)",
        wait_fetch="wait_fetch_bw3(local_e_id, bw3_sem_id, bf_id, bd3_id)",
        kernel_line=1258,
        weight_ref="w3_hbm",
        vmem_ref="b_w3_x2_vmem",
        semaphore_index=3,
        source_slice=(
            "w3_hbm[local_e_id, "
            "p * h_per_t_packing + bd3_id * bd1_per_t_packing : "
            "+ bd1_per_t_packing, bf_id * bf : + bf]"
        ),
        destination_slice="b_w3_x2_vmem[bw3_sem_id, p]",
        scratch_shape="(2, t_packing, bd1 // t_packing, bf)",
    ),
}


def dense_bf16_tile_shape(path: WeightPath, *, bf: int, bd: int) -> tuple[int, int, int]:
    """Return the primary weight tile shape copied by the start_fetch p-loop."""

    _validate_bf16_shape_inputs(bf=bf, bd=bd)
    if path in ("w1", "w3"):
        return (T_PACKING, bd // T_PACKING, bf)
    if path == "w2":
        return (T_PACKING, bf, bd // T_PACKING)
    raise ValueError(f"Unsupported Layer 1 weight path: {path}")


def dense_bf16_bytes_per_fetch(*, bf: int, bd: int) -> int:
    _validate_bf16_shape_inputs(bf=bf, bd=bd)
    return bf * bd * BF16_BYTES


def plans_for_shape(shape: WeightTileShapeLike) -> tuple[WeightTileDMAPlan, ...]:
    path_class = _coerce_path_class(shape.path_class)
    paths: tuple[WeightPath, ...] = ("w1", "w3") if path_class == "w1w3" else ("w2",)
    return tuple(_plan_for_path(path, path_class=path_class, shape=shape) for path in paths)


def build_not_implemented_rows(
    *,
    shapes: Iterable[WeightTileShapeLike],
    suite: str = SUITE_V7X32_BF16_WEIGHT_TILES,
    execution_mode: str,
    runtime: dict[str, Any] | None = None,
    source: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Build Layer 1 rows, measuring TPU/Pallas mode and preserving smoke rows."""

    if suite != SUITE_V7X32_BF16_WEIGHT_TILES:
        raise ValueError(f"Unsupported suite: {suite}")
    runtime = collect_runtime_identity() if runtime is None else runtime

    if execution_mode != "pallas":
        return _build_schema_only_rows(
            shapes=shapes,
            suite=suite,
            execution_mode=execution_mode,
            runtime=runtime,
            source=source,
            metadata=metadata,
            implementation_note=_not_implemented_note(execution_mode, runtime),
        )

    unavailable_note = _pallas_unavailable_note(runtime)
    if unavailable_note is not None:
        return _build_schema_only_rows(
            shapes=shapes,
            suite=suite,
            execution_mode=execution_mode,
            runtime=runtime,
            source=source,
            metadata=metadata,
            implementation_note=unavailable_note,
        )

    rows: list[dict[str, Any]] = []
    warmup_runs = _positive_int_env("CALIBRATION_LAYER1_WARMUP_RUNS", DEFAULT_WARMUP_RUNS)
    sample_runs = _positive_int_env("CALIBRATION_LAYER1_SAMPLE_RUNS", DEFAULT_SAMPLE_RUNS)
    discard_runs = _nonnegative_int_env(
        "CALIBRATION_LAYER1_TRACE_DISCARD_RUNS", DEFAULT_TRACE_DISCARD_RUNS
    )
    trace_root = os.getenv("CALIBRATION_LAYER1_TRACE_ROOT", "/tmp/sglang_jax_layer1_dma_trace")

    for shape in shapes:
        for plan in plans_for_shape(shape):
            row_metadata = _metadata_for_plan(
                plan,
                _with_measurement_metadata(
                    metadata,
                    plan=plan,
                    warmup_runs=warmup_runs,
                    sample_runs=sample_runs,
                    discard_runs=discard_runs,
                    trace_root=trace_root,
                ),
            )
            try:
                samples = _measure_weight_tile_dma_ms(
                    plan,
                    warmup_runs=warmup_runs,
                    sample_runs=sample_runs,
                    discard_runs=discard_runs,
                    trace_root=trace_root,
                )
            except Exception as exc:
                rows.append(
                    _make_row(
                        suite=suite,
                        plan=plan,
                        execution_mode=execution_mode,
                        runtime=runtime,
                        source=dict(source or _source()),
                        metadata=row_metadata,
                        status=STATUS_NOT_IMPLEMENTED,
                        latency_ms_samples=[],
                        implementation_note=_measurement_failed_note(exc),
                    )
                )
                continue

            rows.append(
                _make_row(
                    suite=suite,
                    plan=plan,
                    execution_mode=execution_mode,
                    runtime=runtime,
                    source=dict(source or _source()),
                    metadata=row_metadata,
                    status=STATUS_MEASURED,
                    latency_ms_samples=samples,
                    implementation_note=IMPLEMENTATION_NOTE,
                )
            )
    return rows


def _build_schema_only_rows(
    *,
    shapes: Iterable[WeightTileShapeLike],
    suite: str,
    execution_mode: str,
    runtime: dict[str, Any],
    source: Mapping[str, Any] | None,
    metadata: Mapping[str, Any] | None,
    implementation_note: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for shape in shapes:
        for plan in plans_for_shape(shape):
            rows.append(
                _make_row(
                    suite=suite,
                    plan=plan,
                    execution_mode=execution_mode,
                    runtime=runtime,
                    source=dict(source or _source()),
                    metadata=_metadata_for_plan(plan, metadata),
                    status=STATUS_NOT_IMPLEMENTED,
                    latency_ms_samples=[],
                    implementation_note=implementation_note,
                )
            )
    return rows


def _make_row(
    *,
    suite: str,
    plan: WeightTileDMAPlan,
    execution_mode: str,
    runtime: dict[str, Any],
    source: dict[str, Any],
    metadata: dict[str, Any],
    status: str,
    latency_ms_samples: list[float],
    implementation_note: str,
) -> dict[str, Any]:
    return build_observation_row(
        scenario=SCENARIO_LAYER1_WEIGHT_TILE_DMA,
        suite=suite,
        layer=1,
        path=plan.path,
        path_class=plan.path_class,
        dtype=DTYPE,
        weight_dtype=WEIGHT_DTYPE,
        t_packing=T_PACKING,
        bf=plan.bf,
        bd=plan.bd,
        tile_shape=plan.tile_shape,
        bytes_hbm=plan.bytes_per_fetch,
        bytes_per_fetch=plan.bytes_per_fetch,
        dma_count=plan.dma_count,
        status=status,
        execution_mode=execution_mode,
        latency_ms_samples=latency_ms_samples,
        runtime=runtime,
        source=source,
        metadata=metadata,
        implementation_note=implementation_note,
    )


def _plan_for_path(
    path: WeightPath, *, path_class: PathClass, shape: WeightTileShapeLike
) -> WeightTileDMAPlan:
    spec = WEIGHT_DMA_PATH_SPECS[path]
    if spec.path_class != path_class:
        raise ValueError(f"Path {path} is not valid for {path_class=}.")

    tile_shape = dense_bf16_tile_shape(path, bf=shape.bf, bd=shape.bd)
    expected_bytes = dense_bf16_bytes_per_fetch(bf=shape.bf, bd=shape.bd)
    if tuple(shape.tile_shape) != tile_shape:
        raise ValueError(
            f"Shape row tile_shape={shape.tile_shape} does not match {path=} tile_shape={tile_shape}."
        )
    if shape.bytes_per_fetch != expected_bytes:
        raise ValueError(
            f"Shape row bytes_per_fetch={shape.bytes_per_fetch} does not match "
            f"{path=} expected_bytes={expected_bytes}."
        )

    return WeightTileDMAPlan(
        path=path,
        path_class=path_class,
        bf=shape.bf,
        bd=shape.bd,
        tile_shape=tile_shape,
        bytes_per_fetch=expected_bytes,
        dma_count=DMA_COUNT,
        spec=spec,
    )


def _metadata_for_plan(
    plan: WeightTileDMAPlan, metadata: Mapping[str, Any] | None
) -> dict[str, Any]:
    base: dict[str, Any] = {
        "matrix_kind": "fused_moe_weight_tile_dma",
        "target_runtime": TARGET_RUNTIME_V7X32,
        "target_family": {
            "dtype": DTYPE,
            "weight_dtype": WEIGHT_DTYPE,
            "num_experts": 256,
            "top_k": 8,
            "hidden_size": 8192,
            "intermediate_size": 2048,
        },
        "kernel_mapping": {
            **asdict(plan.spec),
            "kernel_path": KERNEL_PATH,
            "p_loop": "for p in range(t_packing), with t_packing=2 for bf16",
            "primary_copy_only": True,
            "excluded_from_phase1_row": (
                "dot",
                "A2A",
                "expert traversal",
                "full fused-MoE control flow",
                "quant scale side copies",
                "bias side copies",
            ),
        },
        "traffic": {
            "traffic_class": "primary_hbm_to_vmem_read",
            "bytes_hbm_primary_read": plan.bytes_per_fetch,
            "bytes_hbm_anchor_write": 0,
            "bytes_hbm_total_accounted": plan.bytes_per_fetch,
            "anchor_write_required_for_observability": False,
        },
    }
    if metadata:
        base.update(dict(metadata))
    return base


def _with_measurement_metadata(
    metadata: Mapping[str, Any] | None,
    *,
    plan: WeightTileDMAPlan,
    warmup_runs: int,
    sample_runs: int,
    discard_runs: int,
    trace_root: str,
) -> dict[str, Any]:
    enriched = dict(metadata or {})
    enriched["benchmark"] = {
        "name": "layer1_pallas_weight_tile_dma",
        "warmup_runs": warmup_runs,
        "sample_runs": sample_runs,
        "trace_discard_runs": discard_runs,
        "timing": "jax_profiler_trace_device_duration_ms",
        "trace_root": trace_root,
        "pallas_grid": (1,),
        "vmem_limit_bytes": VMEM_LIMIT_BYTES,
        "local_semaphore_shape": LOCAL_SEMAPHORE_SHAPE,
        "has_side_effects": True,
    }
    enriched["traffic"] = {
        "traffic_class": "primary_hbm_to_vmem_read_side_effect_only",
        "bytes_hbm_primary_read": plan.bytes_per_fetch,
        "bytes_hbm_anchor_write": 0,
        "bytes_hbm_total_accounted": plan.bytes_per_fetch,
        "anchor_write_required_for_observability": False,
        "observability_note": (
            "This first measured Layer 1 variant relies on Pallas side effects "
            "and the fused-MoE VMEM self-copy wait pattern. It intentionally "
            "avoids scalar HBM stores, which failed in earlier diagnostics."
        ),
    }
    return enriched


def _source() -> dict[str, Any]:
    return {
        "coordination_repo": "jimoosciuc/fused-moe-calibration-lab",
        "implementation_issue": "jimoosciuc/fused-moe-calibration-lab#6",
        "schema_issue": "jimoosciuc/fused-moe-calibration-lab#2",
        "suite_source": "docs/phase-1-input-matrix.md",
        "kernel_path": KERNEL_PATH,
    }


def _validate_bf16_shape_inputs(*, bf: int, bd: int) -> None:
    if bf <= 0 or bd <= 0:
        raise ValueError(f"Expected positive bf/bd, got {bf=} {bd=}.")
    if bd % T_PACKING != 0:
        raise ValueError(f"Expected bd={bd} to be divisible by t_packing={T_PACKING}.")
    if bd > HIDDEN_SIZE:
        raise ValueError(f"Expected bd={bd} to be <= target hidden_size={HIDDEN_SIZE}.")


def _coerce_path_class(path_class: str) -> PathClass:
    if path_class in ("w1w3", "w2"):
        return cast(PathClass, path_class)
    raise ValueError(f"Unsupported Layer 1 path_class: {path_class}")


def _not_implemented_note(execution_mode: str, runtime: dict[str, Any]) -> str:
    if execution_mode == "local_smoke":
        return (
            "layer1_weight_tile_dma emitted schema-only rows on local_smoke; "
            "Pallas DMA measurements require a TPU backend."
        )
    backend = runtime.get("default_backend")
    return (
        "layer1_weight_tile_dma did not emit synthetic latency samples. "
        f"execution_mode={execution_mode!r} is not a measured Pallas mode for "
        f"this runtime; observed JAX default_backend={backend!r}."
    )


def _pallas_unavailable_note(runtime: dict[str, Any]) -> str | None:
    backend = runtime.get("default_backend")
    if backend != "tpu":
        return (
            "layer1_weight_tile_dma did not emit synthetic latency samples. "
            "Pallas DMA measurements require JAX default_backend='tpu'; "
            f"observed default_backend={backend!r}."
        )

    try:
        import jax  # noqa: F401
        import jax.numpy as jnp  # noqa: F401
        from jax.experimental import pallas as pl  # noqa: F401
        from jax.experimental.pallas import tpu as pltpu  # noqa: F401

        from benchmark.utils import multiple_iteration_timeit_from_trace  # noqa: F401
    except Exception as exc:
        return (
            "layer1_weight_tile_dma could not import the JAX/Pallas trace APIs "
            f"needed for measured DMA; {type(exc).__name__}: {exc}. "
            "No synthetic latency samples were emitted."
        )

    return None


def _measurement_failed_note(exc: Exception) -> str:
    return (
        "layer1_weight_tile_dma Pallas DMA measurement failed before producing "
        f"trustworthy samples: {type(exc).__name__}: {exc}. No synthetic latency "
        "samples were emitted for this shape."
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


def _measure_weight_tile_dma_ms(
    plan: WeightTileDMAPlan,
    *,
    warmup_runs: int,
    sample_runs: int,
    discard_runs: int,
    trace_root: str,
) -> list[float]:
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    from benchmark.utils import multiple_iteration_timeit_from_trace

    source_shape = _hbm_source_shape(plan)
    source = jnp.ones(source_shape, dtype=jnp.bfloat16)
    jax.block_until_ready(source)

    @jax.jit
    def run_dma(weight_hbm):
        return _pallas_weight_tile_dma_call(
            weight_hbm, plan=plan, pl=pl, pltpu=pltpu, jax=jax, jnp=jnp
        )

    jax.block_until_ready(run_dma(source))
    task = f"layer1_dma_{plan.path}_{plan.bf}x{plan.bd}"
    return multiple_iteration_timeit_from_trace(
        compute_func=run_dma,
        data_generator=lambda: (source,),
        task=task,
        tries=sample_runs,
        warmup=warmup_runs,
        discard_initial_samples=discard_runs,
        trace_root=trace_root,
    )


def _hbm_source_shape(plan: WeightTileDMAPlan) -> tuple[int, int, int]:
    if plan.path == "w2":
        return (1, plan.bf, HIDDEN_SIZE)
    return (1, HIDDEN_SIZE, plan.bf)


def _pallas_weight_tile_dma_call(weight_hbm, *, plan, pl, pltpu, jax, jnp):
    bd_per_t_packing = plan.bd // T_PACKING
    sem_index = plan.spec.semaphore_index
    is_w2 = plan.path == "w2"
    scratch_shape = (
        (2, T_PACKING, plan.bf, bd_per_t_packing)
        if is_w2
        else (2, T_PACKING, bd_per_t_packing, plan.bf)
    )

    def kernel(weight_ref, out_ref, weight_scratch, local_sems):
        del out_ref
        for p in range(T_PACKING):
            offset = p * H_PER_T_PACKING
            if is_w2:
                src_ref = weight_ref.at[
                    0,
                    pl.ds(0, plan.bf),
                    pl.ds(offset, bd_per_t_packing),
                ]
            else:
                src_ref = weight_ref.at[
                    0,
                    pl.ds(offset, bd_per_t_packing),
                    pl.ds(0, plan.bf),
                ]
            pltpu.make_async_copy(
                src_ref=src_ref,
                dst_ref=weight_scratch.at[0, p],
                sem=local_sems.at[0, sem_index],
            ).start()

        pltpu.make_async_copy(
            src_ref=weight_scratch.at[0],
            dst_ref=weight_scratch.at[0],
            sem=local_sems.at[0, sem_index],
        ).wait()

    compiler_params_kwargs = {
        "has_side_effects": True,
        "vmem_limit_bytes": VMEM_LIMIT_BYTES,
    }
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((1,), jnp.bfloat16),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)],
            out_specs=pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
            grid=(1,),
            scratch_shapes=[
                pltpu.VMEM(scratch_shape, jnp.bfloat16),
                pltpu.SemaphoreType.DMA(LOCAL_SEMAPHORE_SHAPE),
            ],
        ),
        compiler_params=pltpu.CompilerParams(**compiler_params_kwargs),
        name=f"layer1_weight_tile_dma_{plan.path}_{plan.bf}x{plan.bd}",
    )(weight_hbm)
