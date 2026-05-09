"""Layer 1 fused-MoE top-k communication calibration.

This module intentionally splits communication into sub-scenarios:

* metadata allgather: mirrors the recursive-doubling `all_reduce_metadata`
  remote-DMA exchange over per-device expert counts.
* scatter: mirrors `start_a2a_scatter_batch` payload remote DMA for top_k=8.

The combined metadata+scatter scenario should be added after both sub-scenarios
compile and measure cleanly on v7x-32.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from benchmark.moe.calibration.common import build_observation_row

SCENARIO_LAYER1_A2A_SCATTER = "layer1_a2a_scatter"
SCENARIO_LAYER1_A2A_METADATA = "layer1_a2a_metadata"
STATUS_MEASURED = "measured"
STATUS_NOT_IMPLEMENTED = "not_implemented"

DTYPE = "bfloat16"
WEIGHT_DTYPE = "bfloat16"
BF16_BYTES = 2
T_PACKING = 2
HIDDEN_SIZE = 8192
H_PER_T_PACKING = HIDDEN_SIZE // T_PACKING
TOP_K = 8
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
                    "fused-MoE all_reduce_metadata recursive-doubling remote "
                    "DMA allgather over d2e_count. It includes the mesh "
                    "barriers and send/recv wait anchors, but not scatter, "
                    "expert compute, or gather."
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
    enriched = dict(metadata)
    enriched["a2a_scatter"] = {
        "kernel_reference": "python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py:start_a2a_scatter_batch",
        "operation": "pallas_make_async_remote_copy",
        "path_class": shape.path_class,
        "bt": shape.bt,
        "top_k": shape.top_k,
        "hidden_size": shape.hidden_size,
        "t_packing": T_PACKING,
        "h_per_t_packing": shape.hidden_size // T_PACKING,
        "ep_size": shape.ep_size,
        "local_num_experts": shape.local_num_experts,
        "num_experts": shape.ep_size * shape.local_num_experts,
        "routing_pattern": "e_id=(rank+k+1)%ep_size*local_num_experts+k",
        "remote_copies_per_device": shape.bt * shape.top_k,
        "remote_payload_bytes_per_copy": shape.hidden_size * BF16_BYTES,
        "remote_payload_bytes_per_device": _payload_bytes_per_device(shape),
        "destination_layout": "scratch[topk_slot, source_rank*bt + token_id, t_packing, hidden/t_packing]",
        "traffic_class": "remote_dma_scatter_payload_only",
        "includes": ["remote_scatter_start", "send_wait", "recv_wait", "mesh_barrier"],
        "excludes": ["metadata_allgather", "expert_compute", "a2a_gather", "output_accumulation"],
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
        "includes": ["d2e_count_remote_allgather", "send_wait", "recv_wait", "mesh_barrier"],
        "excludes": [
            "t2e_routing_smem_copy",
            "offsets_starts_sizes_smem_copy",
            "scatter",
            "expert_compute",
            "gather",
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

        from benchmark.moe.utils import build_mesh  # noqa: F401
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
        "layer1_a2a_scatter Pallas remote-DMA measurement failed before "
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

    from benchmark.moe.utils import build_mesh
    from benchmark.utils import multiple_iteration_timeit_from_trace

    mesh = build_mesh(ep_size=shape.ep_size, tp_size=1)
    token_sharding = NamedSharding(mesh, P("tensor", None, None))
    topk_sharding = NamedSharding(mesh, P("tensor", None))
    scratch_sharding = NamedSharding(mesh, P())

    tokens = _make_tokens(jax=jax, np=np, sharding=token_sharding, shape=shape)
    topk_ids = _make_topk_ids(jax=jax, np=np, sharding=topk_sharding, shape=shape)
    scratch = jax.device_put(
        jnp.zeros(
            (
                shape.top_k,
                shape.ep_size * shape.bt,
                T_PACKING,
                shape.hidden_size // T_PACKING,
            ),
            dtype=jnp.bfloat16,
        ),
        scratch_sharding,
    )
    jax.block_until_ready((tokens, topk_ids, scratch))

    with jax.set_mesh(mesh):

        @jax.jit
        def run_scatter(tokens_hbm, topk_ids_hbm, scratch_hbm):
            return _sharded_scatter_call(
                tokens_hbm,
                topk_ids_hbm,
                scratch_hbm,
                shape=shape,
                mesh=mesh,
                jax=jax,
                jnp=jnp,
                pl=pl,
                pltpu=pltpu,
                P=P,
            )

        jax.block_until_ready(run_scatter(tokens, topk_ids, scratch))
        task = f"layer1_a2a_scatter_topk8_bt{shape.bt}_{shape.path_class}"
        return multiple_iteration_timeit_from_trace(
            compute_func=run_scatter,
            data_generator=lambda: (tokens, topk_ids, scratch),
            task=task,
            tries=sample_runs,
            warmup=warmup_runs,
            discard_initial_samples=discard_runs,
            trace_root=trace_root,
        )


def _sharded_scatter_call(
    tokens_hbm,
    topk_ids_hbm,
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
        in_specs=(P("tensor", None, None), P("tensor", None), P()),
        out_specs=P("tensor"),
        check_vma=False,
    )
    def per_rank(tokens_local, topk_ids_local, scratch_replicated):
        return _pallas_a2a_scatter_call(
            tokens_local,
            topk_ids_local,
            scratch_replicated,
            shape=shape,
            jax=jax,
            jnp=jnp,
            pl=pl,
            pltpu=pltpu,
        )

    return per_rank(tokens_hbm, topk_ids_hbm, scratch_hbm)


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

    from benchmark.moe.utils import build_mesh
    from benchmark.utils import multiple_iteration_timeit_from_trace

    mesh = build_mesh(ep_size=shape.ep_size, tp_size=1)
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


def _pallas_a2a_metadata_call(local_counts_hbm, *, shape, jax, pl, pltpu):
    def kernel(local_counts_ref, out_ref, d2e_count_vmem, send_sem, recv_sem, barrier_sem):
        import jax.numpy as jnp
        from jax import lax

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
        d2e_count_vmem.at[rank, 0, pl.ds(0, shape.padded_num_experts)][...] = local_counts_ref[
            0, pl.ds(0, shape.padded_num_experts)
        ]

        sync_barrier()
        for round_id in range(_metadata_rounds(shape)):
            sync_barrier()
            chunk = 1 << round_id
            peer_id = rank ^ jnp.int32(chunk)
            send_start = (rank >> round_id) << round_id
            recv_start = (peer_id >> round_id) << round_id

            pltpu.make_async_remote_copy(
                src_ref=d2e_count_vmem.at[
                    pl.ds(send_start, chunk), pl.ds(0, 1), pl.ds(0, shape.padded_num_experts)
                ],
                dst_ref=d2e_count_vmem.at[
                    pl.ds(send_start, chunk), pl.ds(0, 1), pl.ds(0, shape.padded_num_experts)
                ],
                send_sem=send_sem,
                recv_sem=recv_sem,
                device_id=get_mesh_device_id(peer_id),
                device_id_type=pltpu.DeviceIdType.MESH,
            ).start()

            recv_ref = d2e_count_vmem.at[
                pl.ds(recv_start, chunk), pl.ds(0, 1), pl.ds(0, shape.padded_num_experts)
            ]
            pltpu.make_async_copy(src_ref=recv_ref, dst_ref=recv_ref, sem=recv_sem).wait()

            send_ref = d2e_count_vmem.at[
                pl.ds(send_start, chunk), pl.ds(0, 1), pl.ds(0, shape.padded_num_experts)
            ]
            pltpu.make_async_copy(src_ref=send_ref, dst_ref=send_ref, sem=send_sem).wait()

        sync_barrier()
        out_ref[0] = d2e_count_vmem[rank, 0, 0]

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


def _pallas_a2a_scatter_call(tokens_hbm, topk_ids_hbm, scratch_hbm, *, shape, jax, jnp, pl, pltpu):
    del jnp

    def kernel(tokens_ref, topk_ids_ref, scratch_ref, out_ref, send_sems, recv_sems, barrier_sem):
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

        def scatter_one(t_id, _):
            for k_id in range(shape.top_k):
                e_id = topk_ids_ref[t_id, k_id]
                recv_id = e_id // shape.local_num_experts
                e_sem_id = e_id % shape.local_num_experts
                dst_start = rank * shape.bt + t_id
                pltpu.make_async_remote_copy(
                    src_ref=tokens_ref.at[pl.ds(t_id, 1)],
                    dst_ref=scratch_ref.at[e_sem_id, pl.ds(dst_start, 1)],
                    send_sem=send_sems.at[e_sem_id],
                    recv_sem=recv_sems.at[e_sem_id],
                    device_id=get_mesh_device_id(recv_id),
                    device_id_type=pltpu.DeviceIdType.MESH,
                ).start()
            return None

        from jax import lax

        lax.fori_loop(0, shape.bt, scatter_one, None, unroll=False)

        for k_id in range(shape.top_k):
            send_ref = scratch_ref.at[k_id, pl.ds(0, shape.bt)]
            pltpu.make_async_copy(
                src_ref=send_ref,
                dst_ref=send_ref,
                sem=send_sems.at[k_id],
            ).wait()

            source_rank = (rank + shape.ep_size - k_id - 1) % shape.ep_size
            recv_ref = scratch_ref.at[k_id, pl.ds(source_rank * shape.bt, shape.bt)]
            pltpu.make_async_copy(
                src_ref=recv_ref,
                dst_ref=recv_ref,
                sem=recv_sems.at[k_id],
            ).wait()

        out_ref[0] = tokens_ref[0, 0, 0]

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((1,), tokens_hbm.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
                pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
                pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
            ],
            out_specs=pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
            grid=(1,),
            scratch_shapes=[
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
    )(tokens_hbm, topk_ids_hbm, scratch_hbm)


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
    global_shape = (shape.bt * shape.ep_size, shape.top_k)

    def data_callback(index):
        leading = index[0]
        start = int(leading.start or 0)
        stop = int(leading.stop or global_shape[0])
        rank_id = start // shape.bt
        local_rows = stop - start
        out = np.zeros((local_rows, shape.top_k), dtype=np.int32)
        for k_id in range(shape.top_k):
            recv_id = (rank_id + k_id + 1) % shape.ep_size
            out[:, k_id] = recv_id * shape.local_num_experts + k_id
        return out

    return jax.make_array_from_callback(global_shape, sharding, data_callback)


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


def _payload_bytes_per_device(shape: A2AScatterShape) -> int:
    return shape.bt * shape.top_k * shape.hidden_size * BF16_BYTES


def _metadata_rounds(shape: A2AMetadataShape) -> int:
    return (shape.ep_size.bit_length() - 1) if shape.ep_size > 1 else 0


def _metadata_remote_bytes_per_device(shape: A2AMetadataShape) -> int:
    copied_rows = shape.ep_size - 1
    return copied_rows * shape.padded_num_experts * 4


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
