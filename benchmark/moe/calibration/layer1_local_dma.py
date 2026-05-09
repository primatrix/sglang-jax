"""Layer 1 fused-MoE local HBM/VMEM DMA calibration rows.

This first local-DMA pass covers non-weight fused-MoE primitives that are local
to a device. It emits complete, parseable smoke rows for the path taxonomy and
byte model; TPU/Pallas measurement kernels are intentionally left unavailable
until the primitive microkernels are implemented.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal

from benchmark.moe.calibration.common import build_observation_row

SCENARIO_LAYER1_LOCAL_DMA = "layer1_local_dma"
SUITE_V7X32_BF16_LOCAL_DMA_TOPK8_V1 = "v7x32_bf16_local_dma_topk8_v1"

STATUS_NOT_IMPLEMENTED = "not_implemented"
DTYPE = "bfloat16"
WEIGHT_DTYPE = "bfloat16"
BF16_BYTES = 2
F32_BYTES = 4
I32_BYTES = 4
T_PACKING = 2
TOP_K = 8
PADDED_TOP_K = 128
HIDDEN_SIZE = 8192
H_PER_T_PACKING = HIDDEN_SIZE // T_PACKING
KERNEL_PATH = "python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py"

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

LocalDMAPath = Literal[
    "topk_fetch",
    "a2a_s_tile_read",
    "accumulator_store_or_rmw",
    "output_gather_load",
    "output_store",
]


@dataclass(frozen=True)
class LocalDMAShape:
    path: LocalDMAPath
    path_class: str
    bt: int
    top_k: int = TOP_K
    hidden_size: int = HIDDEN_SIZE
    t_packing: int = T_PACKING


@dataclass(frozen=True)
class LocalDMAPlan:
    shape: LocalDMAShape
    tile_shape: tuple[int, ...]
    bytes_hbm: int
    bytes_per_fetch: int
    dma_count: int
    includes: tuple[str, ...]
    excludes: tuple[str, ...]
    layout: dict[str, Any]
    traffic_class: str
    kernel_reference: str


def build_rows(
    *,
    suite: str,
    shapes: Iterable[LocalDMAShape],
    execution_mode: str,
    runtime: dict[str, Any],
    dtype: str,
    weight_dtype: str,
    t_packing: int,
    source: dict[str, Any],
    metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    if suite != SUITE_V7X32_BF16_LOCAL_DMA_TOPK8_V1:
        raise ValueError(f"Unsupported Layer 1 local DMA suite: {suite}")
    if dtype != DTYPE or weight_dtype != WEIGHT_DTYPE or t_packing != T_PACKING:
        raise ValueError("layer1_local_dma v1 supports only bf16 tokens/weights with t_packing=2.")

    return [
        _make_row(
            suite=suite,
            plan=plan_for_shape(shape),
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


def plan_for_shape(shape: LocalDMAShape) -> LocalDMAPlan:
    _validate_shape(shape)
    if shape.path == "topk_fetch":
        return _topk_fetch_plan(shape)
    if shape.path == "a2a_s_tile_read":
        return _a2a_s_tile_read_plan(shape)
    if shape.path == "accumulator_store_or_rmw":
        return _accumulator_store_or_rmw_plan(shape)
    if shape.path == "output_gather_load":
        return _output_gather_load_plan(shape)
    if shape.path == "output_store":
        return _output_store_plan(shape)
    raise ValueError(f"Unsupported Layer 1 local DMA path: {shape.path}")


def _make_row(
    *,
    suite: str,
    plan: LocalDMAPlan,
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
    shape = plan.shape
    return build_observation_row(
        scenario=SCENARIO_LAYER1_LOCAL_DMA,
        suite=suite,
        layer=1,
        path=shape.path,
        path_class=shape.path_class,
        dtype=dtype,
        weight_dtype=weight_dtype,
        t_packing=t_packing,
        bf=shape.bt,
        bd=shape.hidden_size,
        tile_shape=plan.tile_shape,
        bytes_hbm=plan.bytes_hbm,
        bytes_per_fetch=plan.bytes_per_fetch,
        dma_count=plan.dma_count,
        status=status,
        execution_mode=execution_mode,
        latency_ms_samples=latency_ms_samples,
        runtime=runtime,
        source=source,
        metadata=_metadata_for_plan(metadata, plan),
        implementation_note=implementation_note,
    )


def _metadata_for_plan(metadata: dict[str, Any], plan: LocalDMAPlan) -> dict[str, Any]:
    shape = plan.shape
    enriched = dict(metadata)
    enriched["includes"] = list(plan.includes)
    enriched["excludes"] = list(plan.excludes)
    enriched["local_dma"] = {
        "kernel_path": KERNEL_PATH,
        "kernel_reference": plan.kernel_reference,
        "path": shape.path,
        "path_class": shape.path_class,
        "bt": shape.bt,
        "top_k": shape.top_k,
        "padded_top_k": PADDED_TOP_K,
        "hidden_size": shape.hidden_size,
        "t_packing": shape.t_packing,
        "h_per_t_packing": shape.hidden_size // shape.t_packing,
        "tile_shape": plan.tile_shape,
        "layout": plan.layout,
        "traffic_class": plan.traffic_class,
        "bytes_hbm": plan.bytes_hbm,
        "bytes_per_fetch": plan.bytes_per_fetch,
        "dma_count": plan.dma_count,
        "includes": list(plan.includes),
        "excludes": list(plan.excludes),
        "measurement_status": "schema_only",
    }
    return enriched


def _topk_fetch_plan(shape: LocalDMAShape) -> LocalDMAPlan:
    bytes_per_fetch = shape.bt * PADDED_TOP_K * F32_BYTES
    ids_bytes_per_fetch = shape.bt * PADDED_TOP_K * I32_BYTES
    return LocalDMAPlan(
        shape=shape,
        tile_shape=(shape.bt, PADDED_TOP_K),
        bytes_hbm=bytes_per_fetch + ids_bytes_per_fetch,
        bytes_per_fetch=bytes_per_fetch,
        dma_count=2,
        includes=("topk_weights_hbm_to_vmem", "topk_ids_hbm_to_vmem"),
        excludes=("routing_compute", "metadata_allgather", "a2a_scatter", "expert_compute"),
        layout={
            "source": "topk_{weights,ids}_hbm[bt_start:bt_start+bt, 0:padded_top_k]",
            "destination": "b_topk_{weights,ids}_x2_vmem[bt_sem_id, 0:bt, 0:padded_top_k]",
            "topk_weight_dtype": "float32",
            "topk_id_dtype": "int32",
            "padding_note": "kernel pads top-k rows to padded_top_k=128 before local fetch",
        },
        traffic_class="local_hbm_to_vmem_topk_fetch",
        kernel_reference=f"{KERNEL_PATH}:start_fetch_topk",
    )


def _a2a_s_tile_read_plan(shape: LocalDMAShape) -> LocalDMAPlan:
    bytes_per_fetch = _bf16_token_tile_bytes(shape.bt, shape.hidden_size)
    return LocalDMAPlan(
        shape=shape,
        tile_shape=(shape.bt, shape.t_packing, shape.hidden_size // shape.t_packing),
        bytes_hbm=bytes_per_fetch,
        bytes_per_fetch=bytes_per_fetch,
        dma_count=1,
        includes=("a2a_s_x2_hbm_to_b_stage_x2_vmem",),
        excludes=("remote_a2a_scatter", "w1_w3_weight_fetch", "ffn1_dot", "ffn2_dot"),
        layout={
            "source": "a2a_s_x2_hbm[e_sem_id, tile_start:tile_start+bt, :, bd1_slice]",
            "destination": "b_stage_x2_vmem[buf_id, 0:bt, :, 0:hidden/t_packing]",
            "bd1": shape.hidden_size,
            "bd1_per_t_packing": shape.hidden_size // shape.t_packing,
        },
        traffic_class="local_hbm_to_vmem_a2a_s_tile_read",
        kernel_reference=f"{KERNEL_PATH}:start_stage_a2a_s_tile_from_hbm",
    )


def _accumulator_store_or_rmw_plan(shape: LocalDMAShape) -> LocalDMAPlan:
    bytes_per_fetch = _bf16_token_tile_bytes(shape.bt, shape.hidden_size)
    return LocalDMAPlan(
        shape=shape,
        tile_shape=(shape.bt, shape.t_packing, shape.hidden_size // shape.t_packing),
        bytes_hbm=2 * bytes_per_fetch,
        bytes_per_fetch=bytes_per_fetch,
        dma_count=2,
        includes=("a2a_s_acc_tile_load_for_rmw", "a2a_s_acc_tile_store_to_hbm"),
        excludes=("a2a_gather_remote_dma", "output_weighted_sum", "output_hbm_store"),
        layout={
            "load_source": "a2a_s_acc_x2_hbm[e_sem_id, tile_start:tile_start+bt, :, bd2_slice]",
            "load_destination": "a2a_s_acc_stage_x3_vmem[buf_id, 0:bt, :, 0:hidden/t_packing]",
            "store_source": "a2a_s_acc_stage_x3_vmem[buf_id, 0:bt, :, 0:hidden/t_packing]",
            "store_destination": "a2a_s_acc_x2_hbm[e_sem_id, tile_start:tile_start+bt, :, bd2_slice]",
            "bd2": shape.hidden_size,
            "bd2_per_t_packing": shape.hidden_size // shape.t_packing,
            "rmw_note": "v1 accounts one local read plus one local write for the RMW-capable path",
        },
        traffic_class="local_hbm_vmem_accumulator_rmw",
        kernel_reference=(
            f"{KERNEL_PATH}:start_load_stage_a2a_s_acc_tile_from_hbm/"
            "start_store_stage_a2a_s_acc_tile_to_hbm"
        ),
    )


def _output_gather_load_plan(shape: LocalDMAShape) -> LocalDMAPlan:
    bytes_per_fetch = _bf16_token_tile_bytes(1, shape.hidden_size)
    return LocalDMAPlan(
        shape=shape,
        tile_shape=(shape.bt, shape.top_k, shape.t_packing, shape.hidden_size // shape.t_packing),
        bytes_hbm=shape.bt * shape.top_k * bytes_per_fetch,
        bytes_per_fetch=bytes_per_fetch,
        dma_count=shape.bt * shape.top_k,
        includes=("a2a_g_hbm_to_a2a_g_acc_vmem",),
        excludes=("topk_weight_fetch", "weighted_sum", "shared_expert_add", "output_hbm_store"),
        layout={
            "source": "a2a_g_hbm[e_id, offset:offset+1, :, 0:hidden/t_packing]",
            "destination": "a2a_g_acc_vmem[buf_id, k_id, t_i:t_i+1, :, 0:hidden/t_packing]",
            "copies": "one local HBM->VMEM DMA per valid token per top_k expert",
            "valid_token_assumption": "smoke rows model all bt tokens as valid",
        },
        traffic_class="local_hbm_to_vmem_output_gather_load",
        kernel_reference=f"{KERNEL_PATH}:start_load_acc_bt",
    )


def _output_store_plan(shape: LocalDMAShape) -> LocalDMAPlan:
    bytes_per_fetch = _bf16_token_tile_bytes(shape.bt, shape.hidden_size)
    return LocalDMAPlan(
        shape=shape,
        tile_shape=(shape.bt, shape.hidden_size),
        bytes_hbm=bytes_per_fetch,
        bytes_per_fetch=bytes_per_fetch,
        dma_count=1,
        includes=("b_output_x2_vmem_to_output_hbm", "output_store_wait_anchor"),
        excludes=("output_gather_load", "weighted_sum", "shared_expert_add", "remote_dma"),
        layout={
            "source": "b_output_x2_vmem[bt_sem_id, 0:bt, 0:hidden]",
            "destination": "output_hbm[bt_start:bt_start+bt, 0:hidden]",
            "wait_anchor": "output_hbm[bt_start:bt_start+bt]",
        },
        traffic_class="local_vmem_to_hbm_output_store",
        kernel_reference=f"{KERNEL_PATH}:start_send_bo/wait_store_output",
    )


def _bf16_token_tile_bytes(bt: int, hidden_size: int) -> int:
    return bt * hidden_size * BF16_BYTES


def _validate_shape(shape: LocalDMAShape) -> None:
    if shape.bt <= 0:
        raise ValueError(f"Expected positive bt, got {shape.bt}.")
    if shape.top_k != TOP_K:
        raise ValueError(f"Expected top_k={TOP_K}, got {shape.top_k}.")
    if shape.hidden_size != HIDDEN_SIZE:
        raise ValueError(f"Expected hidden_size={HIDDEN_SIZE}, got {shape.hidden_size}.")
    if shape.t_packing != T_PACKING:
        raise ValueError(f"Expected t_packing={T_PACKING}, got {shape.t_packing}.")
    if shape.hidden_size % shape.t_packing != 0:
        raise ValueError(
            f"Expected hidden_size={shape.hidden_size} to be divisible by "
            f"t_packing={shape.t_packing}."
        )


def _not_implemented_note(execution_mode: str, runtime: dict[str, Any]) -> str:
    if execution_mode == "local_smoke":
        return (
            "layer1_local_dma emitted schema-only rows on local_smoke; local "
            "HBM/VMEM Pallas primitive measurements are pending."
        )
    backend = runtime.get("default_backend")
    return (
        "layer1_local_dma does not yet emit measured latency samples. "
        f"execution_mode={execution_mode!r}, observed JAX default_backend={backend!r}."
    )
