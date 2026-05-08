"""Phase 1 fused-MoE calibration benchmark CLI.

This first pass establishes the command shape, suite matrix, scenario dispatch,
and JSONL schema. TPU/Pallas measurement kernels are intentionally represented
as not_implemented rows until the Layer 0 and Layer 1 benchmark issues land.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Any

from benchmark.moe.calibration import (
    layer0_gemm_envelope,
    layer0_hbm_envelope,
    layer1_weight_tile_dma,
)
from benchmark.moe.calibration.common import (
    build_observation_row,
    collect_runtime_identity,
    write_jsonl,
)

SCENARIO_LAYER0_HBM_ENVELOPE = "layer0_hbm_envelope"
SCENARIO_LAYER0_GEMM_ENVELOPE = "layer0_gemm_envelope"
SCENARIO_LAYER1_WEIGHT_TILE_DMA = "layer1_weight_tile_dma"
SCENARIOS = (
    SCENARIO_LAYER0_HBM_ENVELOPE,
    SCENARIO_LAYER0_GEMM_ENVELOPE,
    SCENARIO_LAYER1_WEIGHT_TILE_DMA,
)

SUITE_V7X32_BF16_WEIGHT_TILES = "v7x32_bf16_weight_tiles"
SUITE_V7X32_BF16_HBM_COPY_ENVELOPE = "v7x32_bf16_hbm_copy_envelope"
SUITE_V7X32_BF16_GEMM_ENVELOPE = "v7x32_bf16_gemm_envelope"
SUITES = (
    SUITE_V7X32_BF16_WEIGHT_TILES,
    SUITE_V7X32_BF16_HBM_COPY_ENVELOPE,
    SUITE_V7X32_BF16_GEMM_ENVELOPE,
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


def load_suite_shapes(suite: str) -> tuple[WeightTileShape, ...]:
    if suite != SUITE_V7X32_BF16_WEIGHT_TILES:
        raise ValueError(f"Unsupported suite: {suite}")
    return PHASE1_HBM_EQUIVALENT_SHAPES


def load_layer0_suite_shapes(suite: str) -> tuple[WeightTileShape, ...]:
    if suite == SUITE_V7X32_BF16_WEIGHT_TILES:
        return PHASE1_HBM_EQUIVALENT_SHAPES
    if suite == SUITE_V7X32_BF16_HBM_COPY_ENVELOPE:
        return PHASE1_HBM_COPY_LADDER_SHAPES + PHASE1_HBM_EQUIVALENT_SHAPES
    raise ValueError(f"Unsupported Layer 0 suite: {suite}")


def load_layer0_gemm_suite_shapes(suite: str) -> tuple[GemmShape, ...]:
    if suite == SUITE_V7X32_BF16_GEMM_ENVELOPE:
        return PHASE1_GEMM_EQUIVALENT_SHAPES
    raise ValueError(f"Unsupported Layer 0 GEMM suite: {suite}")


def _source() -> dict[str, Any]:
    return {
        "coordination_repo": "jimoosciuc/fused-moe-calibration-lab",
        "coordination_docs": [
            "docs/phase-1-execution-plan.md",
            "docs/phase-1-input-matrix.md",
        ],
        "implementation_issue": "jimoosciuc/fused-moe-calibration-lab#2",
        "suite_source": "docs/phase-1-input-matrix.md",
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
        metadata=_suite_metadata(matrix_kind="hbm_equivalent_weight_tile"),
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
        metadata=_suite_metadata(matrix_kind="gemm_equivalent_shape"),
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


def _jax_backend(runtime: dict[str, Any]) -> str | None:
    backend = runtime.get("default_backend")
    return str(backend) if backend is not None else None


def resolve_execution_mode(scenario: str, requested: str, runtime: dict[str, Any]) -> str:
    if requested != "auto":
        return requested
    if _jax_backend(runtime) != "tpu":
        return "local_smoke"
    if scenario in (SCENARIO_LAYER0_HBM_ENVELOPE, SCENARIO_LAYER0_GEMM_ENVELOPE):
        return "jax_trace"
    return "pallas"


def build_rows(scenario: str, suite: str, execution_mode: str) -> list[dict[str, Any]]:
    runtime = collect_runtime_identity()
    resolved_mode = resolve_execution_mode(scenario, execution_mode, runtime)
    if scenario == SCENARIO_LAYER0_HBM_ENVELOPE:
        return _layer0_hbm_envelope_rows(suite, resolved_mode, runtime)
    if scenario == SCENARIO_LAYER0_GEMM_ENVELOPE:
        return _layer0_gemm_envelope_rows(suite, resolved_mode, runtime)
    if scenario == SCENARIO_LAYER1_WEIGHT_TILE_DMA:
        return _layer1_weight_tile_dma_rows(suite, resolved_mode, runtime)
    raise ValueError(f"Unsupported scenario: {scenario}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 1 fused-MoE calibration benchmarks.")
    parser.add_argument(
        "--scenario",
        choices=SCENARIOS,
        required=True,
        help="Calibration scenario to dispatch.",
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
        help="Path to write JSONL calibration observations.",
    )
    parser.add_argument(
        "--execution-mode",
        choices=("auto", "local_smoke", "jax_trace", "pallas"),
        default="auto",
        help=(
            "Execution backend. auto selects jax_trace for Layer 0 on TPU, "
            "pallas for Layer 1 on TPU, and local_smoke otherwise."
        ),
    )
    parser.add_argument(
        "--print",
        action="store_true",
        help="Also print emitted JSON rows to stdout.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_rows(args.scenario, args.suite, args.execution_mode)
    write_jsonl(args.output, rows)
    if args.print:
        for row in rows:
            print(json.dumps(row, sort_keys=True))


if __name__ == "__main__":
    main()
