"""Layer 1 fused-MoE top-k communication calibration.

This module intentionally splits communication into sub-scenarios:

* metadata d2e_count allgather: mirrors only the recursive-doubling remote-DMA
  exchange over per-device expert counts.
* metadata full: mirrors the full `all_reduce_metadata` stage closely enough
  for pipeline scheduling: t2e routing staging, offsets/starts/sizes SMEM
  copies, d2e_count allgather, and local prefix reduction.
* scatter: mirrors `start_a2a_scatter_batch` payload local/remote DMA for
  top_k=8.

The combined metadata+scatter scenario should be added after both sub-scenarios
compile and measure cleanly on v7x-32.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass
from math import comb
from typing import Any

from benchmark.moe.calibration.common import build_observation_row

SCENARIO_LAYER1_A2A_SCATTER = "layer1_a2a_scatter"
SCENARIO_LAYER1_A2A_METADATA = "layer1_a2a_metadata"
SCENARIO_LAYER1_A2A_METADATA_FULL = "layer1_a2a_metadata_full"
STATUS_MEASURED = "measured"
STATUS_NOT_IMPLEMENTED = "not_implemented"


def _dtype_bytes(dtype: str) -> int:
    if dtype in ("bfloat16", "float16"):
        return 2
    if dtype in ("float32", "int32"):
        return 4
    raise ValueError(f"Unsupported dtype for byte accounting: {dtype}")


DTYPE = "bfloat16"
WEIGHT_DTYPE = "bfloat16"
DTYPE_BYTES = _dtype_bytes(DTYPE)
T_PACKING = 32 // (DTYPE_BYTES * 8)
HIDDEN_SIZE = 8192
H_PER_T_PACKING = HIDDEN_SIZE // T_PACKING
TOP_K = 8
PADDED_TOP_K = 128
HBM_TOKEN_ALIGNMENT = 128
EP_SIZE = 32
LOCAL_NUM_EXPERTS = 8
NUM_EXPERTS = EP_SIZE * LOCAL_NUM_EXPERTS
PADDED_NUM_EXPERTS = 256
VMEM_LIMIT_BYTES = 96 * 1024 * 1024

DEFAULT_WARMUP_RUNS = 3
DEFAULT_SAMPLE_RUNS = 7
DEFAULT_TRACE_DISCARD_RUNS = 1


@dataclass(frozen=True)
class A2AScatterShape:
    path_class: str
    bt: int
    local_routes_per_token: int = 0
    routing_mode: str = "fixed_local"
    routing_seed: int = 17
    scatter_mode: str = "batch"
    top_k: int = TOP_K
    hidden_size: int = HIDDEN_SIZE
    ep_size: int = EP_SIZE
    local_num_experts: int = LOCAL_NUM_EXPERTS


@dataclass(frozen=True)
class A2AMetadataShape:
    path_class: str
    bt: int
    top_k: int = TOP_K
    ep_size: int = EP_SIZE
    local_num_experts: int = LOCAL_NUM_EXPERTS
    padded_num_experts: int = PADDED_NUM_EXPERTS


def build_metadata_full_rows(
    *,
    suite: str,
    shapes: Iterable[A2AMetadataShape],
    execution_mode: str,
    runtime: dict[str, Any],
    dtype: str,
    weight_dtype: str,
    t_packing: int,
    source: dict[str, Any],
    metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    if execution_mode != "pallas":
        return [
            _make_metadata_full_row(
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

    unavailable_note = _pallas_unavailable_note(runtime)
    if unavailable_note is not None:
        return [
            _make_metadata_full_row(
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
    warmup_runs = _positive_int_env("CALIBRATION_LAYER1_A2A_WARMUP_RUNS", DEFAULT_WARMUP_RUNS)
    sample_runs = _positive_int_env("CALIBRATION_LAYER1_A2A_SAMPLE_RUNS", DEFAULT_SAMPLE_RUNS)
    discard_runs = _nonnegative_int_env(
        "CALIBRATION_LAYER1_A2A_TRACE_DISCARD_RUNS", DEFAULT_TRACE_DISCARD_RUNS
    )
    trace_root = os.getenv("CALIBRATION_LAYER1_A2A_TRACE_ROOT", "/tmp/sglang_jax_layer1_a2a_trace")

    for shape in shapes:
        measured_metadata = _with_metadata_full_measurement_metadata(
            metadata,
            shape=shape,
            warmup_runs=warmup_runs,
            sample_runs=sample_runs,
            discard_runs=discard_runs,
            trace_root=trace_root,
        )
        try:
            samples = _measure_a2a_metadata_full_ms(
                shape,
                warmup_runs=warmup_runs,
                sample_runs=sample_runs,
                discard_runs=discard_runs,
                trace_root=trace_root,
            )
        except Exception as exc:
            rows.append(
                _make_metadata_full_row(
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
            _make_metadata_full_row(
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
                    "Measured with a Pallas TPU microkernel aligned to the full "
                    "fused-MoE all_reduce_metadata stage: synthetic t2e routing "
                    "HBM-to-VMEM staging, offsets/starts/sizes/d2e_count SMEM copies, "
                    "recursive-doubling d2e_count remote DMA, and local "
                    "reduced_sizes/reduced_starts computation. It excludes "
                    "scatter, expert compute, and gather."
                ),
            )
        )

    return rows


def build_metadata_rows(
    *,
    suite: str,
    shapes: Iterable[A2AMetadataShape],
    execution_mode: str,
    runtime: dict[str, Any],
    dtype: str,
    weight_dtype: str,
    t_packing: int,
    source: dict[str, Any],
    metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    if execution_mode != "pallas":
        return [
            _make_metadata_row(
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

    unavailable_note = _pallas_unavailable_note(runtime)
    if unavailable_note is not None:
        return [
            _make_metadata_row(
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
    warmup_runs = _positive_int_env("CALIBRATION_LAYER1_A2A_WARMUP_RUNS", DEFAULT_WARMUP_RUNS)
    sample_runs = _positive_int_env("CALIBRATION_LAYER1_A2A_SAMPLE_RUNS", DEFAULT_SAMPLE_RUNS)
    discard_runs = _nonnegative_int_env(
        "CALIBRATION_LAYER1_A2A_TRACE_DISCARD_RUNS", DEFAULT_TRACE_DISCARD_RUNS
    )
    trace_root = os.getenv("CALIBRATION_LAYER1_A2A_TRACE_ROOT", "/tmp/sglang_jax_layer1_a2a_trace")

    for shape in shapes:
        measured_metadata = _with_metadata_measurement_metadata(
            metadata,
            shape=shape,
            warmup_runs=warmup_runs,
            sample_runs=sample_runs,
            discard_runs=discard_runs,
            trace_root=trace_root,
        )
        try:
            samples = _measure_a2a_metadata_ms(
                shape,
                warmup_runs=warmup_runs,
                sample_runs=sample_runs,
                discard_runs=discard_runs,
                trace_root=trace_root,
            )
        except Exception as exc:
            rows.append(
                _make_metadata_row(
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
            _make_metadata_row(
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
                    "Measured with a Pallas TPU microkernel that mirrors the "
                    "fused-MoE recursive-doubling remote DMA allgather over "
                    "d2e_count only. It includes the mesh barriers and "
                    "send/recv wait anchors, but excludes t2e routing staging, "
                    "offset/start/size SMEM copies, scatter, expert compute, "
                    "and gather."
                ),
            )
        )

    return rows


def build_rows(
    *,
    suite: str,
    shapes: Iterable[A2AScatterShape],
    execution_mode: str,
    runtime: dict[str, Any],
    dtype: str,
    weight_dtype: str,
    t_packing: int,
    source: dict[str, Any],
    metadata: dict[str, Any],
) -> list[dict[str, Any]]:
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
                source=source,
                metadata=metadata,
                status=STATUS_NOT_IMPLEMENTED,
                latency_ms_samples=[],
                implementation_note=unavailable_note,
            )
            for shape in shapes
        ]

    rows: list[dict[str, Any]] = []
    warmup_runs = _positive_int_env("CALIBRATION_LAYER1_A2A_WARMUP_RUNS", DEFAULT_WARMUP_RUNS)
    sample_runs = _positive_int_env("CALIBRATION_LAYER1_A2A_SAMPLE_RUNS", DEFAULT_SAMPLE_RUNS)
    discard_runs = _nonnegative_int_env(
        "CALIBRATION_LAYER1_A2A_TRACE_DISCARD_RUNS", DEFAULT_TRACE_DISCARD_RUNS
    )
    trace_root = os.getenv("CALIBRATION_LAYER1_A2A_TRACE_ROOT", "/tmp/sglang_jax_layer1_a2a_trace")

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
            samples = _measure_a2a_scatter_ms(
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
                    "Measured with a Pallas TPU microkernel that mirrors the "
                    "fused-MoE start_a2a_scatter_batch remote-DMA structure: "
                    "top_k=8, one token payload per async remote copy, "
                    "rank-dependent routing, per-slot send/recv semaphores, "
                    "and self-copy wait anchors. It measures scatter only, not "
                    "metadata allgather, expert compute, or gather."
                ),
            )
        )

    return rows


def _make_metadata_row(
    *,
    suite: str,
    shape: A2AMetadataShape,
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
        scenario=SCENARIO_LAYER1_A2A_METADATA,
        suite=suite,
        layer=1,
        path="metadata_allgather",
        path_class=shape.path_class,
        dtype=dtype,
        weight_dtype=weight_dtype,
        t_packing=t_packing,
        bf=shape.bt,
        bd=shape.padded_num_experts,
        tile_shape=(shape.ep_size, 1, shape.padded_num_experts),
        bytes_hbm=_metadata_remote_bytes_per_device(shape),
        bytes_per_fetch=_metadata_remote_bytes_per_device(shape),
        dma_count=_metadata_rounds(shape),
        status=status,
        execution_mode=execution_mode,
        latency_ms_samples=latency_ms_samples,
        runtime=runtime,
        source=source,
        metadata=_metadata_for_metadata_shape(metadata, shape),
        implementation_note=implementation_note,
    )


def _make_metadata_full_row(
    *,
    suite: str,
    shape: A2AMetadataShape,
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
        scenario=SCENARIO_LAYER1_A2A_METADATA_FULL,
        suite=suite,
        layer=1,
        path="metadata_full",
        path_class=shape.path_class,
        dtype=dtype,
        weight_dtype=weight_dtype,
        t_packing=t_packing,
        bf=shape.bt,
        bd=shape.padded_num_experts,
        tile_shape=(shape.ep_size, 1, shape.padded_num_experts),
        bytes_hbm=_metadata_full_bytes_per_device(shape),
        bytes_per_fetch=_metadata_remote_bytes_per_device(shape),
        dma_count=_metadata_rounds(shape) + 5,
        status=status,
        execution_mode=execution_mode,
        latency_ms_samples=latency_ms_samples,
        runtime=runtime,
        source=source,
        metadata=_metadata_for_metadata_full_shape(metadata, shape),
        implementation_note=implementation_note,
    )


def _make_row(
    *,
    suite: str,
    shape: A2AScatterShape,
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
        scenario=SCENARIO_LAYER1_A2A_SCATTER,
        suite=suite,
        layer=1,
        path="remote_scatter",
        path_class=shape.path_class,
        dtype=dtype,
        weight_dtype=weight_dtype,
        t_packing=t_packing,
        bf=shape.bt,
        bd=shape.hidden_size,
        tile_shape=(shape.bt, shape.top_k, T_PACKING, shape.hidden_size // T_PACKING),
        bytes_hbm=_payload_bytes_per_device(shape),
        bytes_per_fetch=_payload_bytes_per_device(shape),
        dma_count=shape.bt * shape.top_k,
        status=status,
        execution_mode=execution_mode,
        latency_ms_samples=latency_ms_samples,
        runtime=runtime,
        source=source,
        metadata=_metadata_for_shape(metadata, shape),
        implementation_note=implementation_note,
    )


def _metadata_for_shape(metadata: dict[str, Any], shape: A2AScatterShape) -> dict[str, Any]:
    routing_stats = _routing_stats(shape)
    enriched = dict(metadata)
    enriched["a2a_scatter"] = {
        "kernel_reference": "python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py:start_a2a_scatter_batch",
        "operation": "pallas_make_async_remote_copy",
        "path_class": shape.path_class,
        "bt": shape.bt,
        "top_k": shape.top_k,
        "padded_top_k": PADDED_TOP_K,
        "hidden_size": shape.hidden_size,
        "t_packing": T_PACKING,
        "h_per_t_packing": shape.hidden_size // T_PACKING,
        "ep_size": shape.ep_size,
        "local_num_experts": shape.local_num_experts,
        "num_experts": shape.ep_size * shape.local_num_experts,
        "routing_mode": shape.routing_mode,
        "scatter_mode": shape.scatter_mode,
        "routing_seed": (shape.routing_seed if shape.routing_mode == "uniform_topk" else None),
        "routing_pattern": (
            "seeded uniform top-k without replacement over all experts"
            if shape.routing_mode == "uniform_topk"
            else "e_id=(rank+k+1)%ep_size*local_num_experts+k"
        ),
        "local_routes_per_token": shape.local_routes_per_token,
        "topk_id_layout": "HBM rows are padded to 128 columns; only first top_k entries route payloads.",
        "scratch_token_capacity": _scratch_token_capacity(shape),
        "local_copies_per_device": routing_stats["local_copies_per_device"],
        "remote_copies_per_device": routing_stats["remote_copies_per_device"],
        "remote_payload_bytes_per_copy": shape.hidden_size * DTYPE_BYTES,
        "local_payload_bytes_per_device": routing_stats["local_payload_bytes_per_device"],
        "remote_payload_bytes_per_device": routing_stats["remote_payload_bytes_per_device"],
        "payload_bytes_per_device": _payload_bytes_per_device(shape),
        "nonzero_send_peers_per_device": routing_stats["nonzero_send_peers_per_device"],
        "nonzero_recv_peers_per_device": routing_stats["nonzero_recv_peers_per_device"],
        "local_routes_histogram": routing_stats["local_routes_histogram"],
        "expert_token_count_max": routing_stats["expert_token_count_max"],
        "expert_token_count_nonzero": routing_stats["expert_token_count_nonzero"],
        "destination_layout": "scratch[local_expert_slot, expert_start + offset, t_packing, hidden/t_packing]",
        "traffic_class": (
            "realistic_uniform_topk_scatter_payload"
            if shape.routing_mode == "uniform_topk"
            else (
                "remote_dma_scatter_payload_only"
                if shape.local_routes_per_token == 0
                else "mixed_local_remote_scatter_payload"
            )
        ),
        "includes": [
            "precomputed_t2e_routing_hbm_to_vmem",
            "precomputed_starts_sizes_hbm_to_vmem",
            *(
                [
                    "local_scatter_start",
                    "remote_scatter_start",
                    "send_wait",
                    "recv_wait",
                ]
                if shape.scatter_mode in ("batch", "descriptor_issue_only")
                else ["routing_scan", "offset_update", "send_count_update"]
            ),
            "mesh_barrier",
        ],
        "excludes": [
            "metadata_allgather",
            "expert_compute",
            "a2a_gather",
            "output_accumulation",
        ],
    }
    return enriched


def _metadata_for_metadata_shape(
    metadata: dict[str, Any], shape: A2AMetadataShape
) -> dict[str, Any]:
    enriched = dict(metadata)
    enriched["a2a_metadata"] = {
        "kernel_reference": "python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py:all_reduce_metadata",
        "operation": "recursive_doubling_remote_dma_allgather",
        "path_class": shape.path_class,
        "bt": shape.bt,
        "top_k": shape.top_k,
        "ep_size": shape.ep_size,
        "local_num_experts": shape.local_num_experts,
        "num_experts": shape.ep_size * shape.local_num_experts,
        "padded_num_experts": shape.padded_num_experts,
        "rounds": _metadata_rounds(shape),
        "remote_payload_bytes_per_device": _metadata_remote_bytes_per_device(shape),
        "traffic_class": "remote_dma_metadata_allgather",
        "includes": [
            "d2e_count_remote_allgather",
            "send_wait",
            "recv_wait",
            "mesh_barrier",
        ],
        "excludes": [
            "t2e_routing_smem_copy",
            "offsets_starts_sizes_smem_copy",
            "scatter",
            "expert_compute",
            "gather",
        ],
    }
    return enriched


def _metadata_for_metadata_full_shape(
    metadata: dict[str, Any], shape: A2AMetadataShape
) -> dict[str, Any]:
    enriched = dict(metadata)
    enriched["a2a_metadata_full"] = {
        "kernel_reference": "python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py:all_reduce_metadata",
        "operation": "full_metadata_stage_microbench",
        "path_class": shape.path_class,
        "bt": shape.bt,
        "top_k": shape.top_k,
        "ep_size": shape.ep_size,
        "local_num_experts": shape.local_num_experts,
        "num_experts": shape.ep_size * shape.local_num_experts,
        "padded_top_k": PADDED_TOP_K,
        "padded_num_experts": shape.padded_num_experts,
        "rounds": _metadata_rounds(shape),
        "remote_payload_bytes_per_device": _metadata_remote_bytes_per_device(shape),
        "local_metadata_bytes_per_device": _metadata_local_bytes_per_device(shape),
        "traffic_class": "metadata_full_stage",
        "includes": [
            "synthetic_t2e_routing_hbm_to_vmem_staging",
            "t2e_routing_smem_copy",
            "offsets_zero_and_smem_copy",
            "d2e_count_init",
            "d2e_count_remote_allgather",
            "reduced_sizes_reduced_starts_loop",
            "starts_sizes_d2e_count_smem_copy",
            "send_wait",
            "recv_wait",
            "mesh_barrier",
        ],
        "excludes": [
            "scatter",
            "expert_compute",
            "gather",
            "output_accumulation",
        ],
    }
    return enriched


def _with_measurement_metadata(
    metadata: dict[str, Any],
    *,
    shape: A2AScatterShape,
    warmup_runs: int,
    sample_runs: int,
    discard_runs: int,
    trace_root: str,
) -> dict[str, Any]:
    enriched = _metadata_for_shape(metadata, shape)
    enriched["benchmark"] = {
        "name": "layer1_pallas_a2a_scatter_topk8",
        "warmup_runs": warmup_runs,
        "sample_runs": sample_runs,
        "trace_discard_runs": discard_runs,
        "timing": "jax_profiler_trace_device_duration_ms",
        "trace_root": trace_root,
    }
    return enriched


def _with_metadata_measurement_metadata(
    metadata: dict[str, Any],
    *,
    shape: A2AMetadataShape,
    warmup_runs: int,
    sample_runs: int,
    discard_runs: int,
    trace_root: str,
) -> dict[str, Any]:
    enriched = _metadata_for_metadata_shape(metadata, shape)
    enriched["benchmark"] = {
        "name": "layer1_pallas_a2a_metadata_allgather",
        "warmup_runs": warmup_runs,
        "sample_runs": sample_runs,
        "trace_discard_runs": discard_runs,
        "timing": "jax_profiler_trace_device_duration_ms",
        "trace_root": trace_root,
    }
    return enriched


def _with_metadata_full_measurement_metadata(
    metadata: dict[str, Any],
    *,
    shape: A2AMetadataShape,
    warmup_runs: int,
    sample_runs: int,
    discard_runs: int,
    trace_root: str,
) -> dict[str, Any]:
    enriched = _metadata_for_metadata_full_shape(metadata, shape)
    enriched["benchmark"] = {
        "name": "layer1_pallas_a2a_metadata_full",
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
            "layer1_a2a communication emitted schema-only rows on local_smoke; "
            "Pallas remote-DMA measurements require a TPU backend."
        )
    backend = runtime.get("default_backend")
    return (
        "layer1_a2a communication did not emit synthetic latency samples. "
        f"execution_mode={execution_mode!r} is not a measured Pallas mode "
        f"for this runtime; observed JAX default_backend={backend!r}."
    )


def _pallas_unavailable_note(runtime: dict[str, Any]) -> str | None:
    backend = runtime.get("default_backend")
    if backend != "tpu":
        return (
            "layer1_a2a communication did not emit synthetic latency samples. "
            "Pallas remote-DMA measurements require JAX default_backend='tpu'; "
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
            "layer1_a2a communication could not import the JAX/Pallas APIs needed "
            f"for measured remote DMA; {type(exc).__name__}: {exc}. "
            "No synthetic latency samples were emitted."
        )
    return None


def _measurement_failed_note(exc: Exception) -> str:
    return (
        "layer1_a2a Pallas remote-DMA measurement failed before "
        f"producing trustworthy samples: {type(exc).__name__}: {exc}. "
        "No synthetic latency samples were emitted for this shape."
    )


def _measure_a2a_scatter_ms(
    shape: A2AScatterShape,
    *,
    warmup_runs: int,
    sample_runs: int,
    discard_runs: int,
    trace_root: str,
) -> list[float]:
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    from benchmark.utils import multiple_iteration_timeit_from_trace

    mesh = _build_tensor_mesh(jax=jax, np=np, ep_size=shape.ep_size, tp_size=1)
    token_sharding = NamedSharding(mesh, P("tensor", None, None))
    topk_sharding = NamedSharding(mesh, P("tensor", None))
    scratch_sharding = NamedSharding(mesh, P())
    starts_sharding = NamedSharding(mesh, P("tensor", None))
    sizes_sharding = NamedSharding(mesh, P())

    tokens = _make_tokens(jax=jax, np=np, sharding=token_sharding, shape=shape)
    topk_ids = _make_topk_ids(jax=jax, np=np, sharding=topk_sharding, shape=shape)
    starts = _make_starts_by_rank(jax=jax, np=np, sharding=starts_sharding, shape=shape)
    sizes = _make_sizes(jax=jax, np=np, sharding=sizes_sharding, shape=shape)
    scratch = jax.device_put(
        jnp.zeros(
            (
                shape.local_num_experts,
                _scratch_token_capacity(shape),
                T_PACKING,
                shape.hidden_size // T_PACKING,
            ),
            dtype=jnp.bfloat16,
        ),
        scratch_sharding,
    )
    jax.block_until_ready((tokens, topk_ids, starts, sizes, scratch))

    with jax.set_mesh(mesh):

        @jax.jit
        def run_scatter(tokens_hbm, topk_ids_hbm, starts_hbm, sizes_hbm, scratch_hbm):
            return _sharded_scatter_call(
                tokens_hbm,
                topk_ids_hbm,
                starts_hbm,
                sizes_hbm,
                scratch_hbm,
                shape=shape,
                mesh=mesh,
                jax=jax,
                jnp=jnp,
                pl=pl,
                pltpu=pltpu,
                P=P,
            )

        jax.block_until_ready(run_scatter(tokens, topk_ids, starts, sizes, scratch))
        task = f"layer1_a2a_scatter_topk8_bt{shape.bt}_{shape.path_class}"
        return multiple_iteration_timeit_from_trace(
            compute_func=run_scatter,
            data_generator=lambda: (tokens, topk_ids, starts, sizes, scratch),
            task=task,
            tries=sample_runs,
            warmup=warmup_runs,
            discard_initial_samples=discard_runs,
            trace_root=trace_root,
        )


def _sharded_scatter_call(
    tokens_hbm,
    topk_ids_hbm,
    starts_hbm,
    sizes_hbm,
    scratch_hbm,
    *,
    shape: A2AScatterShape,
    mesh,
    jax,
    jnp,
    pl,
    pltpu,
    P,
):
    @jax.shard_map(
        mesh=mesh,
        in_specs=(
            P("tensor", None, None),
            P("tensor", None),
            P("tensor", None),
            P(),
            P(),
        ),
        out_specs=P("tensor"),
        check_vma=False,
    )
    def per_rank(tokens_local, topk_ids_local, starts_local, sizes_replicated, scratch_replicated):
        return _pallas_a2a_scatter_call(
            tokens_local,
            topk_ids_local,
            starts_local,
            sizes_replicated,
            scratch_replicated,
            shape=shape,
            jax=jax,
            jnp=jnp,
            pl=pl,
            pltpu=pltpu,
        )

    return per_rank(tokens_hbm, topk_ids_hbm, starts_hbm, sizes_hbm, scratch_hbm)


def _measure_a2a_metadata_ms(
    shape: A2AMetadataShape,
    *,
    warmup_runs: int,
    sample_runs: int,
    discard_runs: int,
    trace_root: str,
) -> list[float]:
    import jax
    import numpy as np
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    from benchmark.utils import multiple_iteration_timeit_from_trace

    mesh = _build_tensor_mesh(jax=jax, np=np, ep_size=shape.ep_size, tp_size=1)
    count_sharding = NamedSharding(mesh, P("tensor", None))
    local_counts = _make_local_counts(jax=jax, np=np, sharding=count_sharding, shape=shape)
    jax.block_until_ready(local_counts)

    with jax.set_mesh(mesh):

        @jax.jit
        def run_metadata(local_counts_hbm):
            return _sharded_metadata_call(
                local_counts_hbm,
                shape=shape,
                mesh=mesh,
                jax=jax,
                pl=pl,
                pltpu=pltpu,
                P=P,
            )

        jax.block_until_ready(run_metadata(local_counts))
        task = f"layer1_a2a_metadata_bt{shape.bt}_{shape.path_class}"
        return multiple_iteration_timeit_from_trace(
            compute_func=run_metadata,
            data_generator=lambda: (local_counts,),
            task=task,
            tries=sample_runs,
            warmup=warmup_runs,
            discard_initial_samples=discard_runs,
            trace_root=trace_root,
        )


def _measure_a2a_metadata_full_ms(
    shape: A2AMetadataShape,
    *,
    warmup_runs: int,
    sample_runs: int,
    discard_runs: int,
    trace_root: str,
) -> list[float]:
    import jax
    import numpy as np
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    from benchmark.utils import multiple_iteration_timeit_from_trace

    mesh = _build_tensor_mesh(jax=jax, np=np, ep_size=shape.ep_size, tp_size=1)
    count_sharding = NamedSharding(mesh, P("tensor", None))
    topk_sharding = NamedSharding(mesh, P("tensor", None))
    local_counts = _make_local_counts(jax=jax, np=np, sharding=count_sharding, shape=shape)
    topk_ids = _make_metadata_topk_ids(jax=jax, np=np, sharding=topk_sharding, shape=shape)
    jax.block_until_ready((local_counts, topk_ids))

    with jax.set_mesh(mesh):

        @jax.jit
        def run_metadata_full(local_counts_hbm, topk_ids_hbm):
            return _sharded_metadata_full_call(
                local_counts_hbm,
                topk_ids_hbm,
                shape=shape,
                mesh=mesh,
                jax=jax,
                pl=pl,
                pltpu=pltpu,
                P=P,
            )

        jax.block_until_ready(run_metadata_full(local_counts, topk_ids))
        task = f"layer1_a2a_metadata_full_bt{shape.bt}_{shape.path_class}"
        return multiple_iteration_timeit_from_trace(
            compute_func=run_metadata_full,
            data_generator=lambda: (local_counts, topk_ids),
            task=task,
            tries=sample_runs,
            warmup=warmup_runs,
            discard_initial_samples=discard_runs,
            trace_root=trace_root,
        )


def _sharded_metadata_call(local_counts_hbm, *, shape: A2AMetadataShape, mesh, jax, pl, pltpu, P):
    @jax.shard_map(
        mesh=mesh,
        in_specs=P("tensor", None),
        out_specs=P("tensor"),
        check_vma=False,
    )
    def per_rank(local_counts):
        return _pallas_a2a_metadata_call(
            local_counts,
            shape=shape,
            jax=jax,
            pl=pl,
            pltpu=pltpu,
        )

    return per_rank(local_counts_hbm)


def _sharded_metadata_full_call(
    local_counts_hbm, topk_ids_hbm, *, shape: A2AMetadataShape, mesh, jax, pl, pltpu, P
):
    @jax.shard_map(
        mesh=mesh,
        in_specs=(P("tensor", None), P("tensor", None)),
        out_specs=P("tensor"),
        check_vma=False,
    )
    def per_rank(local_counts, topk_ids):
        return _pallas_a2a_metadata_full_call(
            local_counts,
            topk_ids,
            shape=shape,
            jax=jax,
            pl=pl,
            pltpu=pltpu,
        )

    return per_rank(local_counts_hbm, topk_ids_hbm)


def _pallas_a2a_metadata_call(local_counts_hbm, *, shape, jax, pl, pltpu):
    def kernel(local_counts_ref, out_ref, d2e_count_vmem, send_sem, recv_sem, barrier_sem):
        import jax.numpy as jnp
        from jax import lax

        del out_ref
        rank = lax.axis_index("tensor")

        def get_mesh_device_id(ep_rank):
            return (0, ep_rank)

        def sync_barrier():
            for peer in range(shape.ep_size):
                pltpu.semaphore_signal(
                    barrier_sem,
                    device_id=get_mesh_device_id(peer),
                    device_id_type=pltpu.DeviceIdType.MESH,
                )
            pltpu.semaphore_wait(barrier_sem, shape.ep_size)

        d2e_count_vmem[...] = jnp.zeros_like(d2e_count_vmem)
        local_count_copy = pltpu.make_async_copy(
            src_ref=local_counts_ref.at[pl.ds(0, 1), pl.ds(0, shape.padded_num_experts)],
            dst_ref=d2e_count_vmem.at[rank, pl.ds(0, 1), pl.ds(0, shape.padded_num_experts)],
            sem=send_sem,
        )
        local_count_copy.start()
        local_count_copy.wait()

        sync_barrier()
        for round_id in range(_metadata_rounds(shape)):
            sync_barrier()
            chunk = 1 << round_id
            peer_id = rank ^ jnp.int32(chunk)
            send_start = (rank >> round_id) << round_id
            recv_start = (peer_id >> round_id) << round_id

            pltpu.make_async_remote_copy(
                src_ref=d2e_count_vmem.at[
                    pl.ds(send_start, chunk),
                    pl.ds(0, 1),
                    pl.ds(0, shape.padded_num_experts),
                ],
                dst_ref=d2e_count_vmem.at[
                    pl.ds(send_start, chunk),
                    pl.ds(0, 1),
                    pl.ds(0, shape.padded_num_experts),
                ],
                send_sem=send_sem,
                recv_sem=recv_sem,
                device_id=get_mesh_device_id(peer_id),
                device_id_type=pltpu.DeviceIdType.MESH,
            ).start()

            recv_ref = d2e_count_vmem.at[
                pl.ds(recv_start, chunk),
                pl.ds(0, 1),
                pl.ds(0, shape.padded_num_experts),
            ]
            pltpu.make_async_copy(src_ref=recv_ref, dst_ref=recv_ref, sem=recv_sem).wait()

            send_ref = d2e_count_vmem.at[
                pl.ds(send_start, chunk),
                pl.ds(0, 1),
                pl.ds(0, shape.padded_num_experts),
            ]
            pltpu.make_async_copy(src_ref=send_ref, dst_ref=send_ref, sem=send_sem).wait()

        sync_barrier()

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((1,), local_counts_hbm.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)],
            out_specs=pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
            grid=(1,),
            scratch_shapes=[
                pltpu.VMEM((shape.ep_size, 1, shape.padded_num_experts), local_counts_hbm.dtype),
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.BARRIER,
            ],
        ),
        compiler_params=pltpu.CompilerParams(
            collective_id=2,
            allow_collective_id_without_custom_barrier=True,
            has_side_effects=True,
            vmem_limit_bytes=VMEM_LIMIT_BYTES,
        ),
        name=f"layer1_a2a_metadata_bt{shape.bt}",
    )(local_counts_hbm)


def _pallas_a2a_metadata_full_call(local_counts_hbm, topk_ids_hbm, *, shape, jax, pl, pltpu):
    def kernel(
        local_counts_ref,
        topk_ids_ref,
        out_ref,
        t2e_routing_vmem,
        d2e_count_vmem,
        offsets_vmem,
        starts_vmem,
        sizes_vmem,
        t2e_routing_smem,
        d2e_count_smem,
        offsets_smem,
        starts_smem,
        sizes_smem,
        send_sem,
        recv_sem,
        barrier_sem,
    ):
        import jax.numpy as jnp
        from jax import lax

        del out_ref
        rank = lax.axis_index("tensor")

        def get_mesh_device_id(ep_rank):
            return (0, ep_rank)

        def sync_barrier():
            for peer in range(shape.ep_size):
                pltpu.semaphore_signal(
                    barrier_sem,
                    device_id=get_mesh_device_id(peer),
                    device_id_type=pltpu.DeviceIdType.MESH,
                )
            pltpu.semaphore_wait(barrier_sem, shape.ep_size)

        offsets_vmem[...] = jnp.zeros_like(offsets_vmem)
        offsets_copy = pltpu.async_copy(
            src_ref=offsets_vmem,
            dst_ref=offsets_smem,
            sem=send_sem,
        )

        topk_copy = pltpu.make_async_copy(
            src_ref=topk_ids_ref.at[pl.ds(0, shape.bt)],
            dst_ref=t2e_routing_vmem,
            sem=send_sem,
        )
        topk_copy.start()
        topk_copy.wait()

        t2e_routing_copy = pltpu.async_copy(
            src_ref=t2e_routing_vmem,
            dst_ref=t2e_routing_smem,
            sem=send_sem,
        )

        d2e_count_vmem[...] = jnp.zeros_like(d2e_count_vmem)
        local_count_copy = pltpu.make_async_copy(
            src_ref=local_counts_ref.at[pl.ds(0, 1), pl.ds(0, shape.padded_num_experts)],
            dst_ref=d2e_count_vmem.at[rank, pl.ds(0, 1), pl.ds(0, shape.padded_num_experts)],
            sem=send_sem,
        )
        local_count_copy.start()
        local_count_copy.wait()

        sync_barrier()
        for round_id in range(_metadata_rounds(shape)):
            sync_barrier()
            chunk = 1 << round_id
            peer_id = rank ^ jnp.int32(chunk)
            send_start = (rank >> round_id) << round_id
            recv_start = (peer_id >> round_id) << round_id

            pltpu.make_async_remote_copy(
                src_ref=d2e_count_vmem.at[
                    pl.ds(send_start, chunk),
                    pl.ds(0, 1),
                    pl.ds(0, shape.padded_num_experts),
                ],
                dst_ref=d2e_count_vmem.at[
                    pl.ds(send_start, chunk),
                    pl.ds(0, 1),
                    pl.ds(0, shape.padded_num_experts),
                ],
                send_sem=send_sem,
                recv_sem=recv_sem,
                device_id=get_mesh_device_id(peer_id),
                device_id_type=pltpu.DeviceIdType.MESH,
            ).start()

            recv_ref = d2e_count_vmem.at[
                pl.ds(recv_start, chunk),
                pl.ds(0, 1),
                pl.ds(0, shape.padded_num_experts),
            ]
            pltpu.make_async_copy(src_ref=recv_ref, dst_ref=recv_ref, sem=recv_sem).wait()

            send_ref = d2e_count_vmem.at[
                pl.ds(send_start, chunk),
                pl.ds(0, 1),
                pl.ds(0, shape.padded_num_experts),
            ]
            pltpu.make_async_copy(src_ref=send_ref, dst_ref=send_ref, sem=send_sem).wait()

        sync_barrier()

        reduced_sizes = jnp.zeros_like(sizes_vmem)
        reduced_starts = jnp.zeros_like(starts_vmem)
        for dev_id in range(shape.ep_size):
            dev_sizes = d2e_count_vmem[dev_id]
            reduced_sizes += dev_sizes
            reduced_starts += lax.select(
                jnp.int32(dev_id) < rank, dev_sizes, jnp.zeros_like(dev_sizes)
            )

        starts_vmem[...] = reduced_starts
        sizes_vmem[...] = reduced_sizes

        starts_copy = pltpu.async_copy(src_ref=starts_vmem, dst_ref=starts_smem, sem=send_sem)
        sizes_copy = pltpu.async_copy(src_ref=sizes_vmem, dst_ref=sizes_smem, sem=send_sem)
        d2e_count_copy = pltpu.async_copy(
            src_ref=d2e_count_vmem,
            dst_ref=d2e_count_smem,
            sem=send_sem,
        )

        t2e_routing_copy.wait()
        d2e_count_copy.wait()
        offsets_copy.wait()
        starts_copy.wait()
        sizes_copy.wait()

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((1,), local_counts_hbm.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
                pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
            ],
            out_specs=pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
            grid=(1,),
            scratch_shapes=[
                pltpu.VMEM((shape.bt, PADDED_TOP_K), local_counts_hbm.dtype),
                pltpu.VMEM((shape.ep_size, 1, shape.padded_num_experts), local_counts_hbm.dtype),
                pltpu.VMEM((2, shape.padded_num_experts), local_counts_hbm.dtype),
                pltpu.VMEM((1, shape.padded_num_experts), local_counts_hbm.dtype),
                pltpu.VMEM((1, shape.padded_num_experts), local_counts_hbm.dtype),
                pltpu.SMEM((shape.bt, PADDED_TOP_K), local_counts_hbm.dtype),
                pltpu.SMEM((shape.ep_size, 1, shape.padded_num_experts), local_counts_hbm.dtype),
                pltpu.SMEM((2, shape.padded_num_experts), local_counts_hbm.dtype),
                pltpu.SMEM((1, shape.padded_num_experts), local_counts_hbm.dtype),
                pltpu.SMEM((1, shape.padded_num_experts), local_counts_hbm.dtype),
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.BARRIER,
            ],
        ),
        compiler_params=pltpu.CompilerParams(
            collective_id=3,
            allow_collective_id_without_custom_barrier=True,
            has_side_effects=True,
            vmem_limit_bytes=VMEM_LIMIT_BYTES,
        ),
        name=f"layer1_a2a_metadata_full_bt{shape.bt}",
    )(local_counts_hbm, topk_ids_hbm)


def _pallas_a2a_scatter_call(
    tokens_hbm,
    topk_ids_hbm,
    starts_hbm,
    sizes_hbm,
    scratch_hbm,
    *,
    shape,
    jax,
    jnp,
    pl,
    pltpu,
):
    def kernel(
        tokens_ref,
        topk_ids_ref,
        starts_ref,
        sizes_ref,
        scratch_ref,
        out_ref,
        topk_ids_vmem,
        starts_vmem,
        sizes_vmem,
        starts_smem,
        sizes_smem,
        offsets_smem,
        send_counts_smem,
        out_vmem,
        topk_sem,
        metadata_sem,
        send_sems,
        recv_sems,
        barrier_sem,
    ):
        my_id = pl.program_id(0) * 0 + 0
        del my_id

        def axis_rank():
            # The surrounding shard_map has data=1 and tensor=ep_size.
            from jax import lax

            return lax.axis_index("tensor")

        rank = axis_rank()

        def get_mesh_device_id(ep_rank):
            return (0, ep_rank)

        for peer in range(shape.ep_size):
            pltpu.semaphore_signal(
                barrier_sem,
                device_id=get_mesh_device_id(peer),
                device_id_type=pltpu.DeviceIdType.MESH,
            )
        pltpu.semaphore_wait(barrier_sem, shape.ep_size)

        topk_copy = pltpu.make_async_copy(
            src_ref=topk_ids_ref.at[pl.ds(0, shape.bt)],
            dst_ref=topk_ids_vmem,
            sem=topk_sem,
        )
        topk_copy.start()
        topk_copy.wait()
        starts_copy = pltpu.make_async_copy(
            src_ref=starts_ref.at[0, pl.ds(0, PADDED_NUM_EXPERTS)],
            dst_ref=starts_vmem.at[0, pl.ds(0, PADDED_NUM_EXPERTS)],
            sem=metadata_sem,
        )
        sizes_copy = pltpu.make_async_copy(
            src_ref=sizes_ref.at[0, pl.ds(0, PADDED_NUM_EXPERTS)],
            dst_ref=sizes_vmem.at[0, pl.ds(0, PADDED_NUM_EXPERTS)],
            sem=metadata_sem,
        )
        starts_copy.start()
        starts_copy.wait()
        sizes_copy.start()
        sizes_copy.wait()
        starts_to_smem = pltpu.async_copy(
            src_ref=starts_vmem,
            dst_ref=starts_smem,
            sem=metadata_sem,
        )
        sizes_to_smem = pltpu.async_copy(
            src_ref=sizes_vmem,
            dst_ref=sizes_smem,
            sem=metadata_sem,
        )
        starts_to_smem.wait()
        sizes_to_smem.wait()
        for e_id in range(PADDED_NUM_EXPERTS):
            offsets_smem[e_id] = jnp.int32(0)
        for slot in range(shape.local_num_experts):
            send_counts_smem[slot] = jnp.int32(0)

        def scatter_one_batch(t_id, _):
            for k_id in range(shape.top_k):
                e_id = topk_ids_vmem[t_id, k_id]
                is_valid = e_id >= 0
                e_id_safe = jnp.where(is_valid, e_id, jnp.int32(0))
                recv_id = e_id_safe // shape.local_num_experts
                e_sem_id = e_id_safe % shape.local_num_experts
                offset = offsets_smem[e_id_safe]
                offsets_smem[e_id_safe] = offset + jnp.where(is_valid, jnp.int32(1), jnp.int32(0))
                dst_start = starts_smem[0, e_id_safe] + offset
                is_local = recv_id == rank
                local_sz = jnp.where(is_valid & is_local, jnp.int32(1), jnp.int32(0))
                remote_sz = jnp.where(is_valid & ~is_local, jnp.int32(1), jnp.int32(0))
                send_counts_smem[e_sem_id] = send_counts_smem[e_sem_id] + remote_sz

                @pl.when(local_sz != 0)
                def _local_copy(e_sem_id=e_sem_id, dst_start=dst_start, local_sz=local_sz):
                    pltpu.make_async_copy(
                        src_ref=tokens_ref.at[pl.ds(t_id, local_sz)],
                        dst_ref=scratch_ref.at[e_sem_id, pl.ds(dst_start, local_sz)],
                        sem=recv_sems.at[e_sem_id],
                    ).start()

                @pl.when(remote_sz != 0)
                def _remote_copy(
                    recv_id=recv_id,
                    e_sem_id=e_sem_id,
                    dst_start=dst_start,
                    remote_sz=remote_sz,
                ):
                    pltpu.make_async_remote_copy(
                        src_ref=tokens_ref.at[pl.ds(t_id, remote_sz)],
                        dst_ref=scratch_ref.at[e_sem_id, pl.ds(dst_start, remote_sz)],
                        send_sem=send_sems.at[e_sem_id],
                        recv_sem=recv_sems.at[e_sem_id],
                        device_id=get_mesh_device_id(recv_id),
                        device_id_type=pltpu.DeviceIdType.MESH,
                    ).start()

            return None

        def scan_one_batch(t_id, _):
            for k_id in range(shape.top_k):
                e_id = topk_ids_vmem[t_id, k_id]
                is_valid = e_id >= 0
                e_id_safe = jnp.where(is_valid, e_id, jnp.int32(0))
                e_sem_id = e_id_safe % shape.local_num_experts
                offset = offsets_smem[e_id_safe]
                offsets_smem[e_id_safe] = offset + jnp.where(is_valid, jnp.int32(1), jnp.int32(0))
                recv_id = e_id_safe // shape.local_num_experts
                remote_sz = jnp.where(is_valid & (recv_id != rank), jnp.int32(1), jnp.int32(0))
                send_counts_smem[e_sem_id] = send_counts_smem[e_sem_id] + remote_sz
            return None

        def scan_expert_one(local_e_id, _):
            def scan_token_one(t_id, __):
                for k_id in range(shape.top_k):
                    e_id = topk_ids_vmem[t_id, k_id]
                    is_valid = e_id >= 0
                    e_id_safe = jnp.where(is_valid, e_id, jnp.int32(0))
                    recv_id = e_id_safe // shape.local_num_experts
                    is_active_expert = is_valid & (
                        e_id_safe % shape.local_num_experts == local_e_id
                    )
                    offset = offsets_smem[e_id_safe]
                    offsets_smem[e_id_safe] = offset + jnp.where(
                        is_active_expert, jnp.int32(1), jnp.int32(0)
                    )
                    remote_sz = jnp.where(
                        is_active_expert & (recv_id != rank), jnp.int32(1), jnp.int32(0)
                    )
                    send_counts_smem[local_e_id] = send_counts_smem[local_e_id] + remote_sz
                return None

            lax.fori_loop(0, shape.bt, scan_token_one, None, unroll=False)
            return None

        def issue_descriptor_one(item_id, _):
            # Descriptor-only path: predecoded work items, no t2e routing scan.
            # This isolates remote-copy issue/wait cost from predicate-heavy routing logic.
            t_id = item_id // shape.top_k
            k_id = item_id - t_id * shape.top_k
            recv_id = (rank + k_id + 1) % shape.ep_size
            e_sem_id = k_id % shape.local_num_experts
            dst_start = t_id
            send_counts_smem[e_sem_id] = send_counts_smem[e_sem_id] + jnp.int32(1)

            pltpu.make_async_remote_copy(
                src_ref=tokens_ref.at[pl.ds(t_id, 1)],
                dst_ref=scratch_ref.at[e_sem_id, pl.ds(dst_start, 1)],
                send_sem=send_sems.at[e_sem_id],
                recv_sem=recv_sems.at[e_sem_id],
                device_id=get_mesh_device_id(recv_id),
                device_id_type=pltpu.DeviceIdType.MESH,
            ).start()
            return None

        def commit_send_count_summary():
            total = jnp.int32(0)
            for slot in range(shape.local_num_experts):
                total += send_counts_smem[slot]
            out_vmem[0] = total
            pltpu.make_async_copy(
                src_ref=out_vmem,
                dst_ref=out_ref,
                sem=metadata_sem,
            ).wait()

        from jax import lax

        if shape.scatter_mode == "scan_only_batch":
            lax.fori_loop(0, shape.bt, scan_one_batch, None, unroll=False)
            commit_send_count_summary()
            return
        if shape.scatter_mode == "scan_only_scatter_one_x8":
            lax.fori_loop(0, shape.local_num_experts, scan_expert_one, None, unroll=False)
            commit_send_count_summary()
            return
        if shape.scatter_mode == "descriptor_issue_only":
            lax.fori_loop(0, shape.bt * shape.top_k, issue_descriptor_one, None, unroll=False)
        else:
            lax.fori_loop(0, shape.bt, scatter_one_batch, None, unroll=False)

        def wait_send_one(slot, _):
            scatter_send_sz = send_counts_smem[slot]

            @pl.when(scatter_send_sz != 0)
            def _():
                ref = scratch_ref.at[slot, pl.ds(0, scatter_send_sz)]
                pltpu.make_async_copy(src_ref=ref, dst_ref=ref, sem=send_sems.at[slot]).wait()

            return None

        lax.fori_loop(0, shape.local_num_experts, wait_send_one, None, unroll=False)

        def wait_recv_one(local_e_id, _):
            e_id = rank * shape.local_num_experts + local_e_id
            recv_sz = sizes_smem[0, e_id]

            @pl.when(recv_sz != 0)
            def _():
                ref = scratch_ref.at[local_e_id, pl.ds(0, recv_sz)]
                pltpu.make_async_copy(
                    src_ref=ref,
                    dst_ref=ref,
                    sem=recv_sems.at[local_e_id],
                ).wait()

            return None

        lax.fori_loop(0, shape.local_num_experts, wait_recv_one, None, unroll=False)
        commit_send_count_summary()

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((1,), jnp.int32),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
                pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
                pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
                pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
                pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
            ],
            out_specs=pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
            grid=(1,),
            scratch_shapes=[
                pltpu.VMEM((shape.bt, PADDED_TOP_K), topk_ids_hbm.dtype),
                pltpu.VMEM((1, PADDED_NUM_EXPERTS), topk_ids_hbm.dtype),
                pltpu.VMEM((1, PADDED_NUM_EXPERTS), topk_ids_hbm.dtype),
                pltpu.SMEM((1, PADDED_NUM_EXPERTS), topk_ids_hbm.dtype),
                pltpu.SMEM((1, PADDED_NUM_EXPERTS), topk_ids_hbm.dtype),
                pltpu.SMEM((PADDED_NUM_EXPERTS,), topk_ids_hbm.dtype),
                pltpu.SMEM((shape.local_num_experts,), topk_ids_hbm.dtype),
                pltpu.VMEM((1,), jnp.int32),
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA((shape.local_num_experts,)),
                pltpu.SemaphoreType.DMA((shape.local_num_experts,)),
                pltpu.SemaphoreType.BARRIER,
            ],
        ),
        compiler_params=pltpu.CompilerParams(
            collective_id=1,
            allow_collective_id_without_custom_barrier=True,
            has_side_effects=True,
            vmem_limit_bytes=VMEM_LIMIT_BYTES,
        ),
        name=f"layer1_a2a_scatter_topk8_bt{shape.bt}",
    )(tokens_hbm, topk_ids_hbm, starts_hbm, sizes_hbm, scratch_hbm)


def _make_tokens(*, jax: Any, np: Any, sharding: Any, shape: A2AScatterShape):
    try:
        from ml_dtypes import bfloat16 as numpy_bfloat16
    except Exception:
        numpy_bfloat16 = np.float32

    global_shape = (shape.bt * shape.ep_size, T_PACKING, shape.hidden_size // T_PACKING)

    def data_callback(index):
        leading = index[0]
        start = int(leading.start or 0)
        stop = int(leading.stop or global_shape[0])
        rank_id = start // shape.bt
        local_shape = (stop - start, T_PACKING, shape.hidden_size // T_PACKING)
        return np.full(local_shape, rank_id % 128, dtype=numpy_bfloat16)

    return jax.make_array_from_callback(global_shape, sharding, data_callback)


def _make_topk_ids(*, jax: Any, np: Any, sharding: Any, shape: A2AScatterShape):
    global_shape = (shape.bt * shape.ep_size, PADDED_TOP_K)

    def data_callback(index):
        leading = index[0]
        start = int(leading.start or 0)
        stop = int(leading.stop or global_shape[0])
        rank_id = start // shape.bt
        local_rows = stop - start
        out = np.full((local_rows, PADDED_TOP_K), -1, dtype=np.int32)
        for local_row, global_token_id in enumerate(range(start, stop)):
            out[local_row, : shape.top_k] = _topk_ids_for_token(
                np=np,
                global_token_id=global_token_id,
                rank_id=rank_id,
                shape=shape,
            )
        return out

    return jax.make_array_from_callback(global_shape, sharding, data_callback)


def _make_starts_by_rank(*, jax: Any, np: Any, sharding: Any, shape: A2AScatterShape):
    counts = _counts_by_rank(np=np, shape=shape)
    starts = np.zeros((shape.ep_size, PADDED_NUM_EXPERTS), dtype=np.int32)
    running = np.zeros((PADDED_NUM_EXPERTS,), dtype=np.int32)
    for rank_id in range(shape.ep_size):
        starts[rank_id] = running
        running = running + counts[rank_id]
    return jax.make_array_from_callback(
        starts.shape,
        sharding,
        lambda index: starts[index],
    )


def _make_sizes(*, jax: Any, np: Any, sharding: Any, shape: A2AScatterShape):
    sizes = _counts_by_rank(np=np, shape=shape).sum(axis=0).astype(np.int32)[None, :]
    return jax.make_array_from_callback(
        sizes.shape,
        sharding,
        lambda index: sizes[index],
    )


def _topk_ids_for_token(*, np: Any, global_token_id: int, rank_id: int, shape: A2AScatterShape):
    if shape.routing_mode == "uniform_topk":
        rng = np.random.default_rng(shape.routing_seed + int(global_token_id))
        return rng.choice(
            shape.ep_size * shape.local_num_experts,
            size=shape.top_k,
            replace=False,
        ).astype(np.int32)

    out = np.empty((shape.top_k,), dtype=np.int32)
    for k_id in range(shape.top_k):
        if k_id < shape.local_routes_per_token:
            recv_id = rank_id
        else:
            remote_offset = k_id - shape.local_routes_per_token
            recv_id = (rank_id + remote_offset + 1) % shape.ep_size
        out[k_id] = recv_id * shape.local_num_experts + k_id
    return out


def _counts_by_rank(*, np: Any, shape: A2AScatterShape):
    counts = np.zeros((shape.ep_size, PADDED_NUM_EXPERTS), dtype=np.int32)
    for rank_id in range(shape.ep_size):
        for token_id in range(shape.bt):
            global_token_id = rank_id * shape.bt + token_id
            topk_ids = _topk_ids_for_token(
                np=np,
                global_token_id=global_token_id,
                rank_id=rank_id,
                shape=shape,
            )
            for e_id in topk_ids:
                counts[rank_id, int(e_id)] += 1
    return counts


def _make_local_counts(*, jax: Any, np: Any, sharding: Any, shape: A2AMetadataShape):
    global_shape = (shape.ep_size, shape.padded_num_experts)

    def data_callback(index):
        leading = index[0]
        start = int(leading.start or 0)
        stop = int(leading.stop or global_shape[0])
        out = np.zeros((stop - start, shape.padded_num_experts), dtype=np.int32)
        for local_row, rank_id in enumerate(range(start, stop)):
            for k_id in range(shape.top_k):
                recv_id = (rank_id + k_id + 1) % shape.ep_size
                e_id = recv_id * shape.local_num_experts + k_id
                out[local_row, e_id] = shape.bt
        return out

    return jax.make_array_from_callback(global_shape, sharding, data_callback)


def _make_metadata_topk_ids(*, jax: Any, np: Any, sharding: Any, shape: A2AMetadataShape):
    global_shape = (shape.bt * shape.ep_size, PADDED_TOP_K)

    def data_callback(index):
        leading = index[0]
        start = int(leading.start or 0)
        stop = int(leading.stop or global_shape[0])
        rank_id = start // shape.bt
        local_rows = stop - start
        out = np.full((local_rows, PADDED_TOP_K), -1, dtype=np.int32)
        for k_id in range(shape.top_k):
            recv_id = (rank_id + k_id + 1) % shape.ep_size
            out[:, k_id] = recv_id * shape.local_num_experts + k_id
        return out

    return jax.make_array_from_callback(global_shape, sharding, data_callback)


def _build_tensor_mesh(*, jax: Any, np: Any, ep_size: int, tp_size: int):
    from jax.experimental import mesh_utils
    from jax.sharding import Mesh

    if ep_size <= 0 or tp_size <= 0:
        raise ValueError(f"Expected {ep_size=} and {tp_size=} to be > 0.")
    devices = jax.devices()[: ep_size * tp_size]
    device_mesh = mesh_utils.create_device_mesh((tp_size, ep_size), devices=devices)
    return Mesh(np.asarray(device_mesh), ("data", "tensor"))


def _payload_bytes_per_device(shape: A2AScatterShape) -> int:
    return shape.bt * shape.top_k * shape.hidden_size * DTYPE_BYTES


def _remote_scatter_copies_per_device(shape: A2AScatterShape) -> int:
    if shape.routing_mode == "uniform_topk":
        return int(round(_routing_stats(shape)["remote_copies_per_device"]))
    return shape.bt * max(shape.top_k - shape.local_routes_per_token, 0)


def _local_scatter_copies_per_device(shape: A2AScatterShape) -> int:
    if shape.routing_mode == "uniform_topk":
        return int(round(_routing_stats(shape)["local_copies_per_device"]))
    return shape.bt * min(shape.local_routes_per_token, shape.top_k)


def _remote_scatter_bytes_per_device(shape: A2AScatterShape) -> int:
    return _remote_scatter_copies_per_device(shape) * shape.hidden_size * DTYPE_BYTES


def _local_scatter_bytes_per_device(shape: A2AScatterShape) -> int:
    return _local_scatter_copies_per_device(shape) * shape.hidden_size * DTYPE_BYTES


def _local_routes_probability(shape: A2AScatterShape) -> float:
    total_experts = shape.ep_size * shape.local_num_experts
    local_experts = shape.local_num_experts
    local_routes = shape.local_routes_per_token
    if local_routes < 0 or local_routes > min(shape.top_k, local_experts):
        return 0.0
    remote_experts = total_experts - local_experts
    remote_routes = shape.top_k - local_routes
    if remote_routes < 0 or remote_routes > remote_experts:
        return 0.0
    return (
        comb(local_experts, local_routes)
        * comb(remote_experts, remote_routes)
        / comb(total_experts, shape.top_k)
    )


def _routing_stats(shape: A2AScatterShape) -> dict[str, Any]:
    if shape.routing_mode != "uniform_topk":
        local_copies = shape.bt * min(shape.local_routes_per_token, shape.top_k)
        remote_copies = shape.bt * max(shape.top_k - shape.local_routes_per_token, 0)
        return {
            "local_copies_per_device": local_copies,
            "remote_copies_per_device": remote_copies,
            "local_payload_bytes_per_device": local_copies * shape.hidden_size * DTYPE_BYTES,
            "remote_payload_bytes_per_device": remote_copies * shape.hidden_size * DTYPE_BYTES,
            "nonzero_send_peers_per_device": (
                0 if remote_copies == 0 else shape.top_k - shape.local_routes_per_token
            ),
            "nonzero_recv_peers_per_device": (
                0 if remote_copies == 0 else shape.top_k - shape.local_routes_per_token
            ),
            "local_routes_histogram": {str(shape.local_routes_per_token): shape.bt},
            "expert_token_count_max": shape.bt,
            "expert_token_count_nonzero": shape.top_k,
        }

    import numpy as np

    counts = _counts_by_rank(np=np, shape=shape)
    local_counts = []
    remote_counts = []
    send_peer_counts = []
    recv_peer_counts = []
    local_routes_histogram: dict[int, int] = {}

    for rank_id in range(shape.ep_size):
        local = 0
        remote = 0
        send_peers = set()
        recv_peers = set()
        for token_id in range(shape.bt):
            global_token_id = rank_id * shape.bt + token_id
            topk_ids = _topk_ids_for_token(
                np=np,
                global_token_id=global_token_id,
                rank_id=rank_id,
                shape=shape,
            )
            token_local = 0
            for e_id in topk_ids:
                recv_id = int(e_id) // shape.local_num_experts
                if recv_id == rank_id:
                    local += 1
                    token_local += 1
                else:
                    remote += 1
                    send_peers.add(recv_id)
            local_routes_histogram[token_local] = local_routes_histogram.get(token_local, 0) + 1
        for peer_id in range(shape.ep_size):
            local_expert_start = rank_id * shape.local_num_experts
            local_expert_stop = local_expert_start + shape.local_num_experts
            if counts[peer_id, local_expert_start:local_expert_stop].sum() > 0:
                recv_peers.add(peer_id)
        local_counts.append(local)
        remote_counts.append(remote)
        send_peer_counts.append(len(send_peers))
        recv_peer_counts.append(len(recv_peers))

    def avg(values: list[int]) -> float:
        return float(sum(values) / len(values)) if values else 0.0

    total_expert_counts = counts.sum(axis=0)
    return {
        "local_copies_per_device": avg(local_counts),
        "remote_copies_per_device": avg(remote_counts),
        "local_payload_bytes_per_device": avg(local_counts) * shape.hidden_size * DTYPE_BYTES,
        "remote_payload_bytes_per_device": avg(remote_counts) * shape.hidden_size * DTYPE_BYTES,
        "nonzero_send_peers_per_device": avg(send_peer_counts),
        "nonzero_recv_peers_per_device": avg(recv_peer_counts),
        "local_routes_histogram": {str(k): v for k, v in sorted(local_routes_histogram.items())},
        "expert_token_count_max": int(total_expert_counts.max(initial=0)),
        "expert_token_count_nonzero": int((total_expert_counts > 0).sum()),
    }


def _scratch_token_capacity(shape: A2AScatterShape) -> int:
    tokens = shape.bt * shape.ep_size
    return ((tokens + HBM_TOKEN_ALIGNMENT - 1) // HBM_TOKEN_ALIGNMENT) * HBM_TOKEN_ALIGNMENT


def _metadata_rounds(shape: A2AMetadataShape) -> int:
    return (shape.ep_size.bit_length() - 1) if shape.ep_size > 1 else 0


def _metadata_remote_bytes_per_device(shape: A2AMetadataShape) -> int:
    copied_rows = shape.ep_size - 1
    return copied_rows * shape.padded_num_experts * 4


def _metadata_local_bytes_per_device(shape: A2AMetadataShape) -> int:
    routing_bytes = shape.bt * PADDED_TOP_K * 4
    offsets_bytes = 2 * shape.padded_num_experts * 4
    starts_bytes = shape.padded_num_experts * 4
    sizes_bytes = shape.padded_num_experts * 4
    d2e_count_bytes = shape.ep_size * shape.padded_num_experts * 4
    return routing_bytes + offsets_bytes + starts_bytes + sizes_bytes + d2e_count_bytes


def _metadata_full_bytes_per_device(shape: A2AMetadataShape) -> int:
    return _metadata_remote_bytes_per_device(shape) + _metadata_local_bytes_per_device(shape)


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
