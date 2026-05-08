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

from benchmark.moe.calibration import layer0_hbm_envelope, layer1_weight_tile_dma
from benchmark.moe.calibration.common import (
    build_observation_row,
    collect_runtime_identity,
    write_jsonl,
)

SCENARIO_LAYER0_HBM_ENVELOPE = "layer0_hbm_envelope"
SCENARIO_LAYER1_WEIGHT_TILE_DMA = "layer1_weight_tile_dma"
SCENARIOS = (SCENARIO_LAYER0_HBM_ENVELOPE, SCENARIO_LAYER1_WEIGHT_TILE_DMA)

SUITE_V7X32_BF16_WEIGHT_TILES = "v7x32_bf16_weight_tiles"
SUITES = (SUITE_V7X32_BF16_WEIGHT_TILES,)

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


def load_suite_shapes(suite: str) -> tuple[WeightTileShape, ...]:
    if suite != SUITE_V7X32_BF16_WEIGHT_TILES:
        raise ValueError(f"Unsupported suite: {suite}")
    return PHASE1_HBM_EQUIVALENT_SHAPES


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
        shapes=load_suite_shapes(suite),
        execution_mode=execution_mode,
        runtime=runtime,
        dtype=DTYPE,
        weight_dtype=WEIGHT_DTYPE,
        t_packing=T_PACKING,
        dma_count=DMA_COUNT,
        source=_source(),
        metadata=_suite_metadata(matrix_kind="hbm_equivalent_weight_tile"),
    )


def _layer1_weight_tile_dma_rows(
    suite: str, execution_mode: str, runtime: dict[str, Any]
) -> list[dict[str, Any]]:
    return layer1_weight_tile_dma.build_not_implemented_rows(
        suite=suite,
        shapes=load_suite_shapes(suite),
        execution_mode=execution_mode,
        runtime=runtime,
    )


def _jax_backend(runtime: dict[str, Any]) -> str | None:
    backend = runtime.get("default_backend")
    return str(backend) if backend is not None else None


def resolve_execution_mode(requested: str, runtime: dict[str, Any]) -> str:
    if requested != "auto":
        return requested
    return "pallas" if _jax_backend(runtime) == "tpu" else "local_smoke"


def build_rows(scenario: str, suite: str, execution_mode: str) -> list[dict[str, Any]]:
    runtime = collect_runtime_identity()
    resolved_mode = resolve_execution_mode(execution_mode, runtime)
    if scenario == SCENARIO_LAYER0_HBM_ENVELOPE:
        return _layer0_hbm_envelope_rows(suite, resolved_mode, runtime)
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
        choices=("auto", "local_smoke", "pallas"),
        default="auto",
        help="Execution backend. auto selects pallas on TPU and local_smoke otherwise.",
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
