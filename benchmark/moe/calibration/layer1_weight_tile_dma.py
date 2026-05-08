"""Layer 1 fused-MoE dense weight tile DMA calibration skeleton.

This module records the Phase 1 mapping from #2 JSONL rows to the real
`start_fetch_bw1`, `start_fetch_bw2`, and `start_fetch_bw3` HBM->VMEM copies in
`python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py`.

It does not run a measured benchmark yet. Until a TPU/Pallas timing kernel is
added and wired into `bench_calibration.py`, rows built here stay
`status=not_implemented` with empty latency samples.
"""

from __future__ import annotations

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
IMPLEMENTATION_BLOCKER = (
    "Layer 1 dense bf16 weight tile DMA mapping is specified, but the measured "
    "Pallas microkernel is not implemented in this pass. A valid implementation "
    "still needs a TPU-side timing kernel that starts and waits on only the "
    "start_fetch_bw1/start_fetch_bw2/start_fetch_bw3 primary weight async copies, "
    "plus leader-owned dispatch from bench_calibration.py."
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
    """Build schema-compatible Layer 1 rows without reporting fake latency."""

    if suite != SUITE_V7X32_BF16_WEIGHT_TILES:
        raise ValueError(f"Unsupported suite: {suite}")
    runtime = collect_runtime_identity() if runtime is None else runtime
    rows: list[dict[str, Any]] = []
    for shape in shapes:
        for plan in plans_for_shape(shape):
            row_metadata = _metadata_for_plan(plan, metadata)
            rows.append(
                build_observation_row(
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
                    status="not_implemented",
                    execution_mode=execution_mode,
                    latency_ms_samples=[],
                    runtime=runtime,
                    source=dict(source or _source()),
                    metadata=row_metadata,
                    implementation_note=IMPLEMENTATION_BLOCKER,
                )
            )
    return rows


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
    }
    if metadata:
        base.update(dict(metadata))
    return base


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


def _coerce_path_class(path_class: str) -> PathClass:
    if path_class in ("w1w3", "w2"):
        return cast(PathClass, path_class)
    raise ValueError(f"Unsupported Layer 1 path_class: {path_class}")
