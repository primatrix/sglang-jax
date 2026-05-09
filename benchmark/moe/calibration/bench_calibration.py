"""Fused-MoE performance investigation benchmark CLI.

This CLI dispatches Layer 0 hardware envelope, Layer 1 kernel-pattern module,
and Layer 2 composed fused-MoE diagnostics into one JSONL observation schema.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Any

from benchmark.moe.calibration import (
    layer0_a2a_envelope,
    layer0_gemm_envelope,
    layer0_hbm_envelope,
    layer1_a2a_gather,
    layer1_a2a_scatter,
    layer1_local_dma,
    layer1_weight_tile_dma,
)
from benchmark.moe.calibration.common import (
    build_observation_row,
    collect_runtime_identity,
    write_jsonl,
)

SCENARIO_LAYER0_HBM_ENVELOPE = "layer0_hbm_envelope"
SCENARIO_LAYER0_GEMM_ENVELOPE = "layer0_gemm_envelope"
SCENARIO_LAYER0_A2A_ENVELOPE = "layer0_a2a_envelope"
SCENARIO_LAYER1_A2A_METADATA = "layer1_a2a_metadata"
SCENARIO_LAYER1_A2A_SCATTER = "layer1_a2a_scatter"
SCENARIO_LAYER1_A2A_GATHER = "layer1_a2a_gather"
SCENARIO_LAYER1_WEIGHT_TILE_DMA = "layer1_weight_tile_dma"
SCENARIO_LAYER1_LOCAL_DMA = "layer1_local_dma"
SCENARIO_LAYER2_FUSED_MOE_E2E = "layer2_fused_moe_e2e"
SCENARIOS = (
    SCENARIO_LAYER0_HBM_ENVELOPE,
    SCENARIO_LAYER0_GEMM_ENVELOPE,
    SCENARIO_LAYER0_A2A_ENVELOPE,
    SCENARIO_LAYER1_A2A_METADATA,
    SCENARIO_LAYER1_A2A_SCATTER,
    SCENARIO_LAYER1_A2A_GATHER,
    SCENARIO_LAYER1_WEIGHT_TILE_DMA,
    SCENARIO_LAYER1_LOCAL_DMA,
    SCENARIO_LAYER2_FUSED_MOE_E2E,
)

SUITE_V7X32_BF16_WEIGHT_TILES = "v7x32_bf16_weight_tiles"
SUITE_V7X32_BF16_HBM_COPY_ENVELOPE = "v7x32_bf16_hbm_copy_envelope"
SUITE_V7X32_BF16_HBM_CURVE_V2 = "v7x32_bf16_hbm_curve_v2"
SUITE_V7X32_BF16_HBM_DENSE_CURVE_V3 = "v7x32_bf16_hbm_dense_curve_v3"
SUITE_V7X8_BF16_HBM_DENSE_CURVE_V3 = "v7x8_bf16_hbm_dense_curve_v3"
SUITE_V7X32_BF16_GEMM_ENVELOPE = "v7x32_bf16_gemm_envelope"
SUITE_V7X32_BF16_GEMM_CURVE_V2 = "v7x32_bf16_gemm_curve_v2"
SUITE_V7X8_BF16_GEMM_SATURATION_CURVE_V3 = "v7x8_bf16_gemm_saturation_curve_v3"
SUITE_V7X32_BF16_A2A_CURVE_V1 = "v7x32_bf16_a2a_curve_v1"
SUITE_V7X32_BF16_A2A_CURVE_V2 = "v7x32_bf16_a2a_curve_v2"
SUITE_V7X32_BF16_A2A_TOPK8_V1 = "v7x32_bf16_a2a_topk8_v1"
SUITE_V7X32_BF16_A2A_TOPK8_PREFLIGHT_V1 = "v7x32_bf16_a2a_topk8_preflight_v1"
SUITE_V7X32_BF16_LOCAL_DMA_TOPK8_V1 = "v7x32_bf16_local_dma_topk8_v1"
SUITE_V7X32_BF16_FUSED_MOE_E2E_DIAG_V1 = "v7x32_bf16_fused_moe_e2e_diag_v1"
SUITES = (
    SUITE_V7X32_BF16_WEIGHT_TILES,
    SUITE_V7X32_BF16_HBM_COPY_ENVELOPE,
    SUITE_V7X32_BF16_HBM_CURVE_V2,
    SUITE_V7X32_BF16_HBM_DENSE_CURVE_V3,
    SUITE_V7X8_BF16_HBM_DENSE_CURVE_V3,
    SUITE_V7X32_BF16_GEMM_ENVELOPE,
    SUITE_V7X32_BF16_GEMM_CURVE_V2,
    SUITE_V7X8_BF16_GEMM_SATURATION_CURVE_V3,
    SUITE_V7X32_BF16_A2A_CURVE_V1,
    SUITE_V7X32_BF16_A2A_CURVE_V2,
    SUITE_V7X32_BF16_A2A_TOPK8_V1,
    SUITE_V7X32_BF16_A2A_TOPK8_PREFLIGHT_V1,
    SUITE_V7X32_BF16_LOCAL_DMA_TOPK8_V1,
    SUITE_V7X32_BF16_FUSED_MOE_E2E_DIAG_V1,
)

DTYPE = "bfloat16"
WEIGHT_DTYPE = "bfloat16"
T_PACKING = 2
DMA_COUNT = T_PACKING

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

TARGET_RUNTIME_V7X8 = {
    "device_type": "v7x",
    "falcon_device_count": 8,
    "falcon_device_topo": "2x2x1",
    "replica": 1,
    "jax_device_count": 8,
    "jax_local_device_count": 8,
    "jax_process_count": 1,
    "chip_count": 4,
    "tensorcore_or_jax_device_count": 8,
}


@dataclass(frozen=True)
class WeightTileShape:
    path_class: str
    bf: int
    bd: int
    bytes_per_fetch: int
    tile_shape: tuple[int, int, int]


@dataclass(frozen=True)
class GemmShape:
    path: str
    m: int
    k: int
    n: int


@dataclass(frozen=True)
class CollectiveShape:
    path_class: str
    matrix_dim: int
    mesh_shape: str
    sharding_strategy: str
    slice_topology: str
    ici_size: int


PHASE1_HBM_EQUIVALENT_SHAPES: tuple[WeightTileShape, ...] = (
    WeightTileShape("w1w3", 512, 1024, 1048576, (2, 512, 512)),
    WeightTileShape("w1w3", 512, 2048, 2097152, (2, 1024, 512)),
    WeightTileShape("w1w3", 512, 4096, 4194304, (2, 2048, 512)),
    WeightTileShape("w1w3", 512, 8192, 8388608, (2, 4096, 512)),
    WeightTileShape("w1w3", 1024, 1024, 2097152, (2, 512, 1024)),
    WeightTileShape("w1w3", 1024, 2048, 4194304, (2, 1024, 1024)),
    WeightTileShape("w1w3", 1024, 4096, 8388608, (2, 2048, 1024)),
    WeightTileShape("w1w3", 2048, 1024, 4194304, (2, 512, 2048)),
    WeightTileShape("w1w3", 2048, 2048, 8388608, (2, 1024, 2048)),
    WeightTileShape("w2", 512, 1024, 1048576, (2, 512, 512)),
    WeightTileShape("w2", 512, 2048, 2097152, (2, 512, 1024)),
    WeightTileShape("w2", 512, 4096, 4194304, (2, 512, 2048)),
    WeightTileShape("w2", 512, 8192, 8388608, (2, 512, 4096)),
    WeightTileShape("w2", 1024, 1024, 2097152, (2, 1024, 512)),
    WeightTileShape("w2", 1024, 2048, 4194304, (2, 1024, 1024)),
    WeightTileShape("w2", 1024, 4096, 8388608, (2, 1024, 2048)),
    WeightTileShape("w2", 2048, 1024, 4194304, (2, 2048, 512)),
    WeightTileShape("w2", 2048, 2048, 8388608, (2, 2048, 1024)),
)

PHASE1_HBM_COPY_LADDER_BYTES = (
    64 * 1024,
    256 * 1024,
    1024 * 1024,
    4 * 1024 * 1024,
    16 * 1024 * 1024,
    64 * 1024 * 1024,
    256 * 1024 * 1024,
    512 * 1024 * 1024,
    1024 * 1024 * 1024,
)

PHASE1_HBM_CURVE_V2_BYTES_HBM = (
    64 * 1024,
    128 * 1024,
    256 * 1024,
    512 * 1024,
    1024 * 1024,
    2 * 1024 * 1024,
    4 * 1024 * 1024,
    8 * 1024 * 1024,
    16 * 1024 * 1024,
    32 * 1024 * 1024,
    64 * 1024 * 1024,
    128 * 1024 * 1024,
    256 * 1024 * 1024,
    512 * 1024 * 1024,
    1024 * 1024 * 1024,
    2 * 1024 * 1024 * 1024,
    4 * 1024 * 1024 * 1024,
)

PHASE1_HBM_DENSE_CURVE_V3_BYTES_HBM = (
    4 * 1024,
    8 * 1024,
    16 * 1024,
    32 * 1024,
    64 * 1024,
    128 * 1024,
    256 * 1024,
    512 * 1024,
    1024 * 1024,
    2 * 1024 * 1024,
    4 * 1024 * 1024,
    8 * 1024 * 1024,
    16 * 1024 * 1024,
    32 * 1024 * 1024,
    64 * 1024 * 1024,
    128 * 1024 * 1024,
    256 * 1024 * 1024,
    512 * 1024 * 1024,
    1024 * 1024 * 1024,
    2 * 1024 * 1024 * 1024,
    4 * 1024 * 1024 * 1024,
)

PHASE1_HBM_COPY_LADDER_SHAPES: tuple[WeightTileShape, ...] = tuple(
    WeightTileShape(
        path_class="hbm_ladder",
        bf=bytes_per_fetch // 2,
        bd=1,
        bytes_per_fetch=bytes_per_fetch,
        tile_shape=(1, 1, bytes_per_fetch // 2),
    )
    for bytes_per_fetch in PHASE1_HBM_COPY_LADDER_BYTES
)

PHASE1_HBM_CURVE_V2_SHAPES: tuple[WeightTileShape, ...] = tuple(
    WeightTileShape(
        path_class="hbm_curve_v2",
        bf=(bytes_hbm // 2) // 2,
        bd=1,
        bytes_per_fetch=bytes_hbm // 2,
        tile_shape=(1, 1, (bytes_hbm // 2) // 2),
    )
    for bytes_hbm in PHASE1_HBM_CURVE_V2_BYTES_HBM
)

PHASE1_HBM_DENSE_CURVE_V3_SHAPES: tuple[WeightTileShape, ...] = tuple(
    WeightTileShape(
        path_class="hbm_dense_curve_v3",
        bf=(bytes_hbm // 2) // 2,
        bd=1,
        bytes_per_fetch=bytes_hbm // 2,
        tile_shape=(1, 1, (bytes_hbm // 2) // 2),
    )
    for bytes_hbm in PHASE1_HBM_DENSE_CURVE_V3_BYTES_HBM
)

PHASE1_GEMM_EQUIVALENT_SHAPES: tuple[GemmShape, ...] = (
    GemmShape("ffn1", 2, 1024, 2048),
    GemmShape("ffn1", 4, 1024, 2048),
    GemmShape("ffn1", 4, 2048, 1024),
    GemmShape("ffn1", 8, 512, 2048),
    GemmShape("ffn1", 8, 1024, 2048),
    GemmShape("ffn1", 16, 512, 2048),
    GemmShape("ffn1", 16, 1024, 1024),
    GemmShape("ffn1", 16, 1024, 2048),
    GemmShape("ffn1", 32, 512, 2048),
    GemmShape("ffn1", 32, 1024, 1024),
    GemmShape("ffn1", 32, 2048, 512),
    GemmShape("ffn1", 32, 4096, 512),
    GemmShape("ffn1", 64, 512, 1024),
    GemmShape("ffn1", 64, 1024, 512),
    GemmShape("ffn1", 64, 1024, 1024),
    GemmShape("ffn1", 64, 1024, 2048),
    GemmShape("ffn1", 64, 2048, 512),
    GemmShape("ffn1", 128, 512, 512),
    GemmShape("ffn1", 128, 512, 1024),
    GemmShape("ffn1", 128, 1024, 2048),
    GemmShape("ffn1", 256, 512, 2048),
    GemmShape("ffn1", 512, 512, 2048),
    GemmShape("ffn1", 1024, 512, 512),
    GemmShape("ffn2", 2, 2048, 1024),
    GemmShape("ffn2", 4, 1024, 2048),
    GemmShape("ffn2", 4, 2048, 1024),
    GemmShape("ffn2", 8, 2048, 512),
    GemmShape("ffn2", 8, 2048, 1024),
    GemmShape("ffn2", 16, 1024, 1024),
    GemmShape("ffn2", 16, 2048, 512),
    GemmShape("ffn2", 16, 2048, 1024),
    GemmShape("ffn2", 32, 512, 2048),
    GemmShape("ffn2", 32, 512, 4096),
    GemmShape("ffn2", 32, 1024, 1024),
    GemmShape("ffn2", 32, 2048, 512),
    GemmShape("ffn2", 64, 512, 1024),
    GemmShape("ffn2", 64, 512, 2048),
    GemmShape("ffn2", 64, 1024, 512),
    GemmShape("ffn2", 64, 1024, 1024),
    GemmShape("ffn2", 64, 2048, 1024),
    GemmShape("ffn2", 128, 512, 512),
    GemmShape("ffn2", 128, 1024, 512),
    GemmShape("ffn2", 128, 2048, 1024),
    GemmShape("ffn2", 256, 2048, 512),
    GemmShape("ffn2", 512, 2048, 512),
    GemmShape("ffn2", 1024, 512, 512),
)

PHASE1_GEMM_M_SWEEP_SHAPES: tuple[GemmShape, ...] = tuple(
    GemmShape("mxu_m_sweep", m, k, n)
    for m in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048)
    for k, n in ((1024, 1024), (2048, 2048), (4096, 4096))
)

PHASE1_GEMM_V3_M_SATURATION_SHAPES: tuple[GemmShape, ...] = tuple(
    GemmShape("mxu_m_saturation", m, k, n)
    for m in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)
    for k, n in ((512, 512), (1024, 1024), (2048, 2048), (4096, 4096))
)

PHASE1_GEMM_ASPECT_SWEEP_SHAPES: tuple[GemmShape, ...] = tuple(
    GemmShape("mxu_aspect_sweep", m, k, n)
    for m in (64, 256, 1024)
    for k, n in ((512, 4096), (1024, 2048), (2048, 1024), (4096, 512))
)

PHASE1_GEMM_V3_ASPECT_SHAPES: tuple[GemmShape, ...] = tuple(
    GemmShape("mxu_aspect", m, k, n)
    for m in (16, 64, 256, 1024, 4096)
    for k, n in ((512, 4096), (1024, 2048), (2048, 1024), (4096, 512))
)

PHASE1_GEMM_CURVE_V2_SHAPES: tuple[GemmShape, ...] = (
    PHASE1_GEMM_EQUIVALENT_SHAPES + PHASE1_GEMM_M_SWEEP_SHAPES + PHASE1_GEMM_ASPECT_SWEEP_SHAPES
)

PHASE1_GEMM_SATURATION_CURVE_V3_SHAPES: tuple[GemmShape, ...] = (
    PHASE1_GEMM_V3_M_SATURATION_SHAPES
    + PHASE1_GEMM_V3_ASPECT_SHAPES
    + PHASE1_GEMM_EQUIVALENT_SHAPES
)

PHASE1_A2A_CURVE_V1_MATRIX_DIMS = (
    32,
    64,
    128,
    256,
    512,
    1024,
    2048,
    4096,
    8192,
    16384,
)

PHASE1_A2A_CURVE_V2_MATRIX_DIMS = (
    16,
    32,
    64,
    128,
    256,
    512,
    1024,
    2048,
    4096,
    8192,
    16384,
    32768,
)

PHASE1_A2A_CURVE_V1_SHAPES: tuple[CollectiveShape, ...] = tuple(
    CollectiveShape(
        path_class=path_class,
        matrix_dim=matrix_dim,
        mesh_shape="4x4x2",
        sharding_strategy=sharding_strategy,
        slice_topology="2x2x4",
        ici_size=32,
    )
    for path_class, sharding_strategy in (
        ("a2a_4x4x1", "4x4x1"),
        ("a2a_4x4x2", "4x4x2"),
    )
    for matrix_dim in PHASE1_A2A_CURVE_V1_MATRIX_DIMS
)

PHASE1_A2A_CURVE_V2_SHAPES: tuple[CollectiveShape, ...] = tuple(
    CollectiveShape(
        path_class=path_class,
        matrix_dim=matrix_dim,
        mesh_shape="4x4x2",
        sharding_strategy=sharding_strategy,
        slice_topology="2x2x4",
        ici_size=32,
    )
    for path_class, sharding_strategy in (
        ("a2a_4x4x1", "4x4x1"),
        ("a2a_4x4x2", "4x4x2"),
    )
    for matrix_dim in PHASE1_A2A_CURVE_V2_MATRIX_DIMS
    if not (path_class == "a2a_4x4x2" and matrix_dim < 32)
)

PHASE1_A2A_TOPK8_BT_VALUES = (1, 2, 4, 8, 16, 32)
PHASE1_A2A_TOPK8_PREFLIGHT_BT_VALUES = (1,)

PHASE1_A2A_SCATTER_TOPK8_SHAPES: tuple[layer1_a2a_scatter.A2AScatterShape, ...] = tuple(
    layer1_a2a_scatter.A2AScatterShape(path_class="scatter_topk8_ring", bt=bt)
    for bt in PHASE1_A2A_TOPK8_BT_VALUES
)

PHASE1_A2A_SCATTER_TOPK8_PREFLIGHT_SHAPES: tuple[layer1_a2a_scatter.A2AScatterShape, ...] = tuple(
    layer1_a2a_scatter.A2AScatterShape(path_class="scatter_topk8_preflight", bt=bt)
    for bt in PHASE1_A2A_TOPK8_PREFLIGHT_BT_VALUES
)

PHASE1_A2A_METADATA_TOPK8_SHAPES: tuple[layer1_a2a_scatter.A2AMetadataShape, ...] = tuple(
    layer1_a2a_scatter.A2AMetadataShape(path_class="metadata_topk8_ring", bt=bt)
    for bt in PHASE1_A2A_TOPK8_BT_VALUES
)

PHASE1_A2A_METADATA_TOPK8_PREFLIGHT_SHAPES: tuple[layer1_a2a_scatter.A2AMetadataShape, ...] = tuple(
    layer1_a2a_scatter.A2AMetadataShape(path_class="metadata_topk8_preflight", bt=bt)
    for bt in PHASE1_A2A_TOPK8_PREFLIGHT_BT_VALUES
)

PHASE1_A2A_GATHER_TOPK8_SHAPES: tuple[layer1_a2a_gather.A2AGatherShape, ...] = tuple(
    layer1_a2a_gather.A2AGatherShape(path_class="gather_topk8_ring", bt=bt)
    for bt in PHASE1_A2A_TOPK8_BT_VALUES
)

PHASE1_A2A_GATHER_TOPK8_PREFLIGHT_SHAPES: tuple[layer1_a2a_gather.A2AGatherShape, ...] = tuple(
    layer1_a2a_gather.A2AGatherShape(path_class="gather_topk8_preflight", bt=bt)
    for bt in PHASE1_A2A_TOPK8_PREFLIGHT_BT_VALUES
)

PHASE1_LOCAL_DMA_TOPK8_PATH_CLASSES: dict[layer1_local_dma.LocalDMAPath, str] = {
    "topk_fetch": "local_topk_fetch",
    "a2a_s_tile_read": "local_a2a_s_tile_read",
    "accumulator_store_or_rmw": "local_accumulator_rmw",
    "output_gather_load": "local_output_gather_load",
    "output_store": "local_output_store",
}

PHASE1_LOCAL_DMA_TOPK8_SHAPES: tuple[layer1_local_dma.LocalDMAShape, ...] = tuple(
    layer1_local_dma.LocalDMAShape(path=path, path_class=path_class, bt=bt)
    for bt in PHASE1_A2A_TOPK8_BT_VALUES
    for path, path_class in PHASE1_LOCAL_DMA_TOPK8_PATH_CLASSES.items()
)


def load_suite_shapes(suite: str) -> tuple[WeightTileShape, ...]:
    if suite != SUITE_V7X32_BF16_WEIGHT_TILES:
        raise ValueError(f"Unsupported suite: {suite}")
    return PHASE1_HBM_EQUIVALENT_SHAPES


def load_layer0_suite_shapes(suite: str) -> tuple[WeightTileShape, ...]:
    if suite == SUITE_V7X32_BF16_WEIGHT_TILES:
        return PHASE1_HBM_EQUIVALENT_SHAPES
    if suite == SUITE_V7X32_BF16_HBM_COPY_ENVELOPE:
        return PHASE1_HBM_COPY_LADDER_SHAPES + PHASE1_HBM_EQUIVALENT_SHAPES
    if suite == SUITE_V7X32_BF16_HBM_CURVE_V2:
        return PHASE1_HBM_CURVE_V2_SHAPES + PHASE1_HBM_EQUIVALENT_SHAPES
    if suite in (SUITE_V7X32_BF16_HBM_DENSE_CURVE_V3, SUITE_V7X8_BF16_HBM_DENSE_CURVE_V3):
        return PHASE1_HBM_DENSE_CURVE_V3_SHAPES + PHASE1_HBM_EQUIVALENT_SHAPES
    raise ValueError(f"Unsupported Layer 0 suite: {suite}")


def load_layer0_gemm_suite_shapes(suite: str) -> tuple[GemmShape, ...]:
    if suite == SUITE_V7X32_BF16_GEMM_ENVELOPE:
        return PHASE1_GEMM_EQUIVALENT_SHAPES
    if suite == SUITE_V7X32_BF16_GEMM_CURVE_V2:
        return PHASE1_GEMM_CURVE_V2_SHAPES
    if suite == SUITE_V7X8_BF16_GEMM_SATURATION_CURVE_V3:
        return PHASE1_GEMM_SATURATION_CURVE_V3_SHAPES
    raise ValueError(f"Unsupported Layer 0 GEMM suite: {suite}")


def load_layer0_a2a_suite_shapes(suite: str) -> tuple[CollectiveShape, ...]:
    if suite == SUITE_V7X32_BF16_A2A_CURVE_V1:
        return PHASE1_A2A_CURVE_V1_SHAPES
    if suite == SUITE_V7X32_BF16_A2A_CURVE_V2:
        return PHASE1_A2A_CURVE_V2_SHAPES
    raise ValueError(f"Unsupported Layer 0 A2A suite: {suite}")


def load_layer1_a2a_scatter_suite_shapes(
    suite: str,
) -> tuple[layer1_a2a_scatter.A2AScatterShape, ...]:
    if suite == SUITE_V7X32_BF16_A2A_TOPK8_V1:
        return PHASE1_A2A_SCATTER_TOPK8_SHAPES
    if suite == SUITE_V7X32_BF16_A2A_TOPK8_PREFLIGHT_V1:
        return PHASE1_A2A_SCATTER_TOPK8_PREFLIGHT_SHAPES
    raise ValueError(f"Unsupported Layer 1 A2A scatter suite: {suite}")


def load_layer1_a2a_metadata_suite_shapes(
    suite: str,
) -> tuple[layer1_a2a_scatter.A2AMetadataShape, ...]:
    if suite == SUITE_V7X32_BF16_A2A_TOPK8_V1:
        return PHASE1_A2A_METADATA_TOPK8_SHAPES
    if suite == SUITE_V7X32_BF16_A2A_TOPK8_PREFLIGHT_V1:
        return PHASE1_A2A_METADATA_TOPK8_PREFLIGHT_SHAPES
    raise ValueError(f"Unsupported Layer 1 A2A metadata suite: {suite}")


def load_layer1_a2a_gather_suite_shapes(
    suite: str,
) -> tuple[layer1_a2a_gather.A2AGatherShape, ...]:
    if suite == SUITE_V7X32_BF16_A2A_TOPK8_V1:
        return PHASE1_A2A_GATHER_TOPK8_SHAPES
    if suite == SUITE_V7X32_BF16_A2A_TOPK8_PREFLIGHT_V1:
        return PHASE1_A2A_GATHER_TOPK8_PREFLIGHT_SHAPES
    raise ValueError(f"Unsupported Layer 1 A2A gather suite: {suite}")


def load_layer1_local_dma_suite_shapes(
    suite: str,
) -> tuple[layer1_local_dma.LocalDMAShape, ...]:
    if suite == SUITE_V7X32_BF16_LOCAL_DMA_TOPK8_V1:
        return PHASE1_LOCAL_DMA_TOPK8_SHAPES
    raise ValueError(f"Unsupported Layer 1 local DMA suite: {suite}")


def load_layer2_fused_moe_e2e_suite_shapes(
    suite: str,
) -> tuple[Any, ...]:
    if suite == SUITE_V7X32_BF16_FUSED_MOE_E2E_DIAG_V1:
        return _layer2_fused_moe_e2e().default_shapes()
    raise ValueError(f"Unsupported Layer 2 fused MoE E2E suite: {suite}")


def _layer2_fused_moe_e2e():
    from benchmark.moe.calibration import layer2_fused_moe_e2e

    return layer2_fused_moe_e2e


def _source() -> dict[str, Any]:
    return {
        "coordination_repo": "jimoosciuc/fused-moe-calibration-lab",
        "coordination_docs": [
            "docs/implementation-plan.md",
        ],
        "suite_source": "docs/implementation-plan.md",
    }


def _suite_metadata(*, matrix_kind: str) -> dict[str, Any]:
    return {
        "matrix_kind": matrix_kind,
        "target_runtime": TARGET_RUNTIME_V7X32,
        "target_family": {
            "dtype": DTYPE,
            "weight_dtype": WEIGHT_DTYPE,
            "num_experts": 256,
            "top_k": 8,
            "hidden_size": 8192,
            "intermediate_size": 2048,
        },
    }


def _suite_metadata_for_runtime(
    *, matrix_kind: str, target_runtime: dict[str, Any]
) -> dict[str, Any]:
    metadata = _suite_metadata(matrix_kind=matrix_kind)
    metadata["target_runtime"] = target_runtime
    return metadata


def _layer0_hbm_metadata(suite: str) -> dict[str, Any]:
    target_runtime = (
        TARGET_RUNTIME_V7X8 if suite == SUITE_V7X8_BF16_HBM_DENSE_CURVE_V3 else TARGET_RUNTIME_V7X32
    )
    metadata = _suite_metadata_for_runtime(
        matrix_kind="hbm_equivalent_weight_tile",
        target_runtime=target_runtime,
    )
    if suite in (SUITE_V7X32_BF16_HBM_DENSE_CURVE_V3, SUITE_V7X8_BF16_HBM_DENSE_CURVE_V3):
        metadata["matrix_kind"] = "hbm_dense_curve_v3"
        metadata["includes"] = [
            "tiny_hbm_copy_rows",
            "knee_hbm_copy_rows",
            "plateau_hbm_copy_rows",
            "fused_moe_weight_tile_marker_rows",
        ]
        metadata["excludes"] = [
            "fused_moe_pallas_weight_dma",
            "remote_dma",
            "gemm_compute",
        ]
        metadata["hbm_dense_curve_v3"] = {
            "measured_bytes_hbm": list(PHASE1_HBM_DENSE_CURVE_V3_BYTES_HBM),
            "traffic_class": "copy_read_write",
            "bytes_hbm_formula": "2 * bytes_per_fetch",
            "byte_groups": {
                "tiny": [4 * 1024, 8 * 1024, 16 * 1024, 32 * 1024],
                "small_knee": [
                    64 * 1024,
                    128 * 1024,
                    256 * 1024,
                    512 * 1024,
                    1024 * 1024,
                    2 * 1024 * 1024,
                    4 * 1024 * 1024,
                    8 * 1024 * 1024,
                    16 * 1024 * 1024,
                    32 * 1024 * 1024,
                    64 * 1024 * 1024,
                ],
                "large_plateau": [
                    128 * 1024 * 1024,
                    256 * 1024 * 1024,
                    512 * 1024 * 1024,
                    1024 * 1024 * 1024,
                    2 * 1024 * 1024 * 1024,
                    4 * 1024 * 1024 * 1024,
                ],
            },
        }
    return metadata


def _layer0_gemm_metadata(suite: str) -> dict[str, Any]:
    metadata = _suite_metadata_for_runtime(
        matrix_kind="gemm_equivalent_shape",
        target_runtime=TARGET_RUNTIME_V7X32,
    )
    if suite == SUITE_V7X8_BF16_GEMM_SATURATION_CURVE_V3:
        metadata = _suite_metadata_for_runtime(
            matrix_kind="gemm_saturation_curve_v3",
            target_runtime=TARGET_RUNTIME_V7X8,
        )
        metadata["includes"] = [
            "m_saturation_sweep",
            "aspect_ratio_sweep",
            "fused_moe_marker_rows",
        ]
        metadata["excludes"] = [
            "fused_moe_pallas_dot_scheduling",
            "hbm_weight_dma",
            "remote_dma",
            "expert_routing",
        ]
        metadata["gemm_saturation_curve_v3"] = {
            "operation": "jax_bfloat16_matmul",
            "primary_metric": "tflops_per_device",
            "target_slice": "v7x-8",
            "m_saturation": {
                "m": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096],
                "kn": [[512, 512], [1024, 1024], [2048, 2048], [4096, 4096]],
            },
            "aspect": {
                "m": [16, 64, 256, 1024, 4096],
                "kn": [[512, 4096], [1024, 2048], [2048, 1024], [4096, 512]],
            },
            "marker_paths": ["ffn1", "ffn2"],
            "shape_counts": {
                "m_saturation": len(PHASE1_GEMM_V3_M_SATURATION_SHAPES),
                "aspect": len(PHASE1_GEMM_V3_ASPECT_SHAPES),
                "marker": len(PHASE1_GEMM_EQUIVALENT_SHAPES),
                "total": len(PHASE1_GEMM_SATURATION_CURVE_V3_SHAPES),
            },
        }
    return metadata


def _not_implemented_note(scenario: str, execution_mode: str) -> str:
    if execution_mode == "local_smoke":
        return (
            f"{scenario} emitted schema-only rows on local_smoke; TPU/Pallas measurement "
            "kernels are pending Phase 1 implementation."
        )
    return f"{scenario} TPU/Pallas measurement kernel is pending Phase 1 implementation."


def _make_row(
    *,
    scenario: str,
    layer: int,
    path: str,
    path_class: str,
    shape: WeightTileShape,
    execution_mode: str,
    runtime: dict[str, Any],
    matrix_kind: str,
) -> dict[str, Any]:
    return build_observation_row(
        scenario=scenario,
        suite=SUITE_V7X32_BF16_WEIGHT_TILES,
        layer=layer,
        path=path,
        path_class=path_class,
        dtype=DTYPE,
        weight_dtype=WEIGHT_DTYPE,
        t_packing=T_PACKING,
        bf=shape.bf,
        bd=shape.bd,
        tile_shape=shape.tile_shape,
        bytes_hbm=shape.bytes_per_fetch,
        bytes_per_fetch=shape.bytes_per_fetch,
        dma_count=DMA_COUNT,
        status="not_implemented",
        execution_mode=execution_mode,
        latency_ms_samples=[],
        runtime=runtime,
        source=_source(),
        metadata=_suite_metadata(matrix_kind=matrix_kind),
        implementation_note=_not_implemented_note(scenario, execution_mode),
    )


def _layer0_hbm_envelope_rows(
    suite: str, execution_mode: str, runtime: dict[str, Any]
) -> list[dict[str, Any]]:
    return layer0_hbm_envelope.build_rows(
        suite=suite,
        shapes=load_layer0_suite_shapes(suite),
        execution_mode=execution_mode,
        runtime=runtime,
        dtype=DTYPE,
        weight_dtype=WEIGHT_DTYPE,
        t_packing=T_PACKING,
        dma_count=DMA_COUNT,
        source=_source(),
        metadata=_layer0_hbm_metadata(suite),
    )


def _layer0_gemm_envelope_rows(
    suite: str, execution_mode: str, runtime: dict[str, Any]
) -> list[dict[str, Any]]:
    return layer0_gemm_envelope.build_rows(
        suite=suite,
        shapes=load_layer0_gemm_suite_shapes(suite),
        execution_mode=execution_mode,
        runtime=runtime,
        dtype=DTYPE,
        weight_dtype=WEIGHT_DTYPE,
        t_packing=T_PACKING,
        source=_source(),
        metadata=_layer0_gemm_metadata(suite),
    )


def _layer0_a2a_envelope_rows(
    suite: str, execution_mode: str, runtime: dict[str, Any]
) -> list[dict[str, Any]]:
    return layer0_a2a_envelope.build_rows(
        suite=suite,
        shapes=load_layer0_a2a_suite_shapes(suite),
        execution_mode=execution_mode,
        runtime=runtime,
        dtype=DTYPE,
        weight_dtype=WEIGHT_DTYPE,
        t_packing=T_PACKING,
        source=_source(),
        metadata=_suite_metadata(matrix_kind="a2a_collective_curve"),
    )


def _layer1_weight_tile_dma_rows(
    suite: str, execution_mode: str, runtime: dict[str, Any]
) -> list[dict[str, Any]]:
    if suite != SUITE_V7X32_BF16_WEIGHT_TILES:
        raise ValueError(
            f"Layer 1 weight tile DMA supports only {SUITE_V7X32_BF16_WEIGHT_TILES}, "
            f"got {suite}."
        )
    return layer1_weight_tile_dma.build_not_implemented_rows(
        suite=suite,
        shapes=load_suite_shapes(suite),
        execution_mode=execution_mode,
        runtime=runtime,
    )


def _layer1_a2a_metadata_rows(
    suite: str, execution_mode: str, runtime: dict[str, Any]
) -> list[dict[str, Any]]:
    return layer1_a2a_scatter.build_metadata_rows(
        suite=suite,
        shapes=load_layer1_a2a_metadata_suite_shapes(suite),
        execution_mode=execution_mode,
        runtime=runtime,
        dtype=DTYPE,
        weight_dtype=WEIGHT_DTYPE,
        t_packing=T_PACKING,
        source=_source(),
        metadata=_suite_metadata(matrix_kind="a2a_metadata_topk8"),
    )


def _layer1_a2a_scatter_rows(
    suite: str, execution_mode: str, runtime: dict[str, Any]
) -> list[dict[str, Any]]:
    return layer1_a2a_scatter.build_rows(
        suite=suite,
        shapes=load_layer1_a2a_scatter_suite_shapes(suite),
        execution_mode=execution_mode,
        runtime=runtime,
        dtype=DTYPE,
        weight_dtype=WEIGHT_DTYPE,
        t_packing=T_PACKING,
        source=_source(),
        metadata=_suite_metadata(matrix_kind="a2a_scatter_topk8"),
    )


def _filter_shapes_by_bf(shapes, bf_values: tuple[int, ...] | None):
    if bf_values is None:
        return shapes
    allowed = set(bf_values)
    return tuple(shape for shape in shapes if getattr(shape, "bt", None) in allowed)


def _layer1_a2a_gather_rows(
    suite: str,
    execution_mode: str,
    runtime: dict[str, Any],
    bf_values: tuple[int, ...] | None = None,
) -> list[dict[str, Any]]:
    return layer1_a2a_gather.build_rows(
        suite=suite,
        shapes=_filter_shapes_by_bf(load_layer1_a2a_gather_suite_shapes(suite), bf_values),
        execution_mode=execution_mode,
        runtime=runtime,
        dtype=DTYPE,
        weight_dtype=WEIGHT_DTYPE,
        t_packing=T_PACKING,
        source=_source(),
        metadata=_suite_metadata(matrix_kind="a2a_gather_topk8"),
    )


def _layer1_local_dma_rows(
    suite: str, execution_mode: str, runtime: dict[str, Any]
) -> list[dict[str, Any]]:
    return layer1_local_dma.build_rows(
        suite=suite,
        shapes=load_layer1_local_dma_suite_shapes(suite),
        execution_mode=execution_mode,
        runtime=runtime,
        dtype=DTYPE,
        weight_dtype=WEIGHT_DTYPE,
        t_packing=T_PACKING,
        source=_source(),
        metadata=_suite_metadata(matrix_kind="local_dma_topk8"),
    )


def _jax_backend(runtime: dict[str, Any]) -> str | None:
    backend = runtime.get("default_backend")
    return str(backend) if backend is not None else None


def resolve_execution_mode(scenario: str, requested: str, runtime: dict[str, Any]) -> str:
    if requested != "auto":
        return requested
    if _jax_backend(runtime) != "tpu":
        raise RuntimeError(
            "Performance investigation runs require a TPU backend. Use --execution-mode local_smoke "
            "only for explicit local schema validation; local_smoke rows are not "
            "diagnostic results."
        )
    if scenario in (
        SCENARIO_LAYER0_HBM_ENVELOPE,
        SCENARIO_LAYER0_GEMM_ENVELOPE,
        SCENARIO_LAYER0_A2A_ENVELOPE,
        SCENARIO_LAYER2_FUSED_MOE_E2E,
    ):
        return "jax_trace"
    return "pallas"


def build_rows(
    scenario: str,
    suite: str,
    execution_mode: str,
    bf_values: tuple[int, ...] | None = None,
) -> list[dict[str, Any]]:
    runtime = collect_runtime_identity()
    resolved_mode = resolve_execution_mode(scenario, execution_mode, runtime)
    if scenario == SCENARIO_LAYER0_HBM_ENVELOPE:
        return _layer0_hbm_envelope_rows(suite, resolved_mode, runtime)
    if scenario == SCENARIO_LAYER0_GEMM_ENVELOPE:
        return _layer0_gemm_envelope_rows(suite, resolved_mode, runtime)
    if scenario == SCENARIO_LAYER0_A2A_ENVELOPE:
        return _layer0_a2a_envelope_rows(suite, resolved_mode, runtime)
    if scenario == SCENARIO_LAYER1_A2A_METADATA:
        return _layer1_a2a_metadata_rows(suite, resolved_mode, runtime)
    if scenario == SCENARIO_LAYER1_A2A_SCATTER:
        return _layer1_a2a_scatter_rows(suite, resolved_mode, runtime)
    if scenario == SCENARIO_LAYER1_A2A_GATHER:
        return _layer1_a2a_gather_rows(suite, resolved_mode, runtime, bf_values)
    if scenario == SCENARIO_LAYER1_WEIGHT_TILE_DMA:
        return _layer1_weight_tile_dma_rows(suite, resolved_mode, runtime)
    if scenario == SCENARIO_LAYER1_LOCAL_DMA:
        return _layer1_local_dma_rows(suite, resolved_mode, runtime)
    if scenario == SCENARIO_LAYER2_FUSED_MOE_E2E:
        layer2_fused_moe_e2e = _layer2_fused_moe_e2e()
        return layer2_fused_moe_e2e.build_rows(
            suite=suite,
            shapes=load_layer2_fused_moe_e2e_suite_shapes(suite),
            execution_mode=resolved_mode,
            runtime=runtime,
            source=_source(),
            metadata=_suite_metadata(matrix_kind="fused_moe_e2e_overlap_diagnostic"),
        )
    raise ValueError(f"Unsupported scenario: {scenario}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run fused-MoE performance investigation benchmarks."
    )
    parser.add_argument(
        "--scenario",
        choices=SCENARIOS,
        required=True,
        help="Benchmark scenario to dispatch.",
    )
    parser.add_argument(
        "--suite",
        choices=SUITES,
        required=True,
        help="Static Phase 1 shape suite to run.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to write JSONL observations.",
    )
    parser.add_argument(
        "--execution-mode",
        choices=("auto", "local_smoke", "jax_trace", "pallas"),
        default="auto",
        help=(
            "Execution backend. auto selects jax_trace for Layer 0/2 on TPU, "
            "pallas for Layer 1 on TPU, and fails outside TPU. Use local_smoke "
            "only for explicit schema validation."
        ),
    )
    parser.add_argument(
        "--bf-values",
        type=str,
        default=None,
        help=(
            "Optional comma-separated bf/bt values to run. Currently used for "
            "Layer 1 gather process isolation, e.g. --bf-values 1 or 1,2."
        ),
    )
    parser.add_argument(
        "--print",
        action="store_true",
        help="Also print emitted JSON rows to stdout.",
    )
    return parser.parse_args()


def _parse_bf_values(raw: str | None) -> tuple[int, ...] | None:
    if raw is None or not raw.strip():
        return None
    values: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(int(part))
    return tuple(values)


def main() -> None:
    args = parse_args()
    rows = build_rows(
        args.scenario,
        args.suite,
        args.execution_mode,
        _parse_bf_values(args.bf_values),
    )
    write_jsonl(args.output, rows)
    if args.print:
        for row in rows:
            print(json.dumps(row, sort_keys=True))


if __name__ == "__main__":
    main()
