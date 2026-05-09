"""Layer 1 fused-MoE gather communication calibration.

This scenario mirrors the fused-MoE gather start/wait path: computed expert
outputs are copied from the target rank back to the source rank after expert
compute. It intentionally excludes metadata allgather, scatter, expert compute,
and output accumulation.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from benchmark.moe.calibration.common import build_observation_row

SCENARIO_LAYER1_A2A_GATHER = "layer1_a2a_gather"
STATUS_MEASURED = "measured"
STATUS_NOT_IMPLEMENTED = "not_implemented"

DTYPE = "bfloat16"
WEIGHT_DTYPE = "bfloat16"
BF16_BYTES = 2
T_PACKING = 2
HIDDEN_SIZE = 8192
TOP_K = 8
EP_SIZE = 32
LOCAL_NUM_EXPERTS = 8
NUM_EXPERTS = EP_SIZE * LOCAL_NUM_EXPERTS
VMEM_LIMIT_BYTES = 96 * 1024 * 1024

DEFAULT_WARMUP_RUNS = 3
DEFAULT_SAMPLE_RUNS = 7
DEFAULT_TRACE_DISCARD_RUNS = 1


@dataclass(frozen=True)
class A2AGatherShape:
    path_class: str
    bt: int
    top_k: int = TOP_K
    hidden_size: int = HIDDEN_SIZE
    ep_size: int = EP_SIZE
    local_num_experts: int = LOCAL_NUM_EXPERTS


def build_rows(
    *,
    suite: str,
    shapes: Iterable[A2AGatherShape],
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
            samples = _measure_a2a_gather_ms(
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
                    "fused-MoE start_a2a_gather and wait_a2a_gather_* remote-DMA "
                    "structure: computed expert outputs are returned from target "
                    "rank to source rank for top_k=8, with per-expert send waits "
                    "and per-source gather receive waits. It measures gather only, "
                    "not metadata allgather, scatter, expert compute, or output "
                    "accumulation."
                ),
            )
        )

    return rows


def _make_row(
    *,
    suite: str,
    shape: A2AGatherShape,
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
        scenario=SCENARIO_LAYER1_A2A_GATHER,
        suite=suite,
        layer=1,
        path="remote_gather",
        path_class=shape.path_class,
        dtype=dtype,
        weight_dtype=weight_dtype,
        t_packing=t_packing,
        bf=shape.bt,
        bd=shape.hidden_size,
        tile_shape=(shape.bt, shape.top_k, t_packing, shape.hidden_size // t_packing),
        bytes_hbm=_payload_bytes_per_device(shape),
        bytes_per_fetch=_payload_bytes_per_device(shape),
        dma_count=_remote_copy_count_per_device(shape),
        status=status,
        execution_mode=execution_mode,
        latency_ms_samples=latency_ms_samples,
        runtime=runtime,
        source=source,
        metadata=_metadata_for_shape(metadata, shape),
        implementation_note=implementation_note,
    )


def _metadata_for_shape(metadata: dict[str, Any], shape: A2AGatherShape) -> dict[str, Any]:
    enriched = dict(metadata)
    enriched["a2a_gather"] = {
        "kernel_reference": "python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py:start_a2a_gather",
        "wait_references": [
            "python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py:wait_a2a_gather_send",
            "python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py:wait_a2a_gather_recv_all",
        ],
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
        "routing_pattern": "target rank sends local expert k back to source_rank=(target_rank-k-1)%ep_size; source rank waits for recv_rank=(source_rank+k+1)%ep_size",
        "local_copy_count_per_device": 0,
        "remote_copies_per_device": _remote_copy_count_per_device(shape),
        "remote_tokens_per_device": shape.bt * shape.top_k,
        "tokens_per_remote_copy": shape.bt,
        "remote_payload_bytes_per_copy": shape.hidden_size * BF16_BYTES,
        "remote_payload_bytes_per_device": _payload_bytes_per_device(shape),
        "copy_granularity": "bulk_per_local_expert",
        "source_layout": "a2a_s_acc[local_expert, token, t_packing, hidden/t_packing]",
        "destination_layout": "a2a_g[global_expert, token, t_packing, hidden/t_packing] on source rank",
        "traffic_class": "remote_dma_gather_payload_only",
        "includes": [
            "remote_gather_start",
            "gather_send_wait",
            "gather_recv_wait_all",
            "mesh_barrier",
        ],
        "excludes": [
            "metadata_allgather",
            "scatter",
            "expert_compute",
            "output_accumulation",
        ],
    }
    return enriched


def _with_measurement_metadata(
    metadata: dict[str, Any],
    *,
    shape: A2AGatherShape,
    warmup_runs: int,
    sample_runs: int,
    discard_runs: int,
    trace_root: str,
) -> dict[str, Any]:
    enriched = _metadata_for_shape(metadata, shape)
    enriched["benchmark"] = {
        "name": "layer1_pallas_a2a_gather_topk8",
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
            "layer1_a2a_gather emitted schema-only rows on local_smoke; "
            "Pallas remote-DMA measurements require a TPU backend."
        )
    backend = runtime.get("default_backend")
    return (
        "layer1_a2a_gather did not emit synthetic latency samples. "
        f"execution_mode={execution_mode!r} is not a measured Pallas mode "
        f"for this runtime; observed JAX default_backend={backend!r}."
    )


def _pallas_unavailable_note(runtime: dict[str, Any]) -> str | None:
    backend = runtime.get("default_backend")
    if backend != "tpu":
        return (
            "layer1_a2a_gather did not emit synthetic latency samples. "
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
            "layer1_a2a_gather could not import the JAX/Pallas APIs needed "
            f"for measured remote DMA; {type(exc).__name__}: {exc}. "
            "No synthetic latency samples were emitted."
        )
    return None


def _measurement_failed_note(exc: Exception) -> str:
    return (
        "layer1_a2a_gather Pallas remote-DMA measurement failed before "
        f"producing trustworthy samples: {type(exc).__name__}: {exc}. "
        "No synthetic latency samples were emitted for this shape."
    )


def _measure_a2a_gather_ms(
    shape: A2AGatherShape,
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
    acc_sharding = NamedSharding(mesh, P("tensor", None, None, None))
    gather_sharding = NamedSharding(mesh, P())

    expert_outputs = _make_expert_outputs(jax=jax, np=np, sharding=acc_sharding, shape=shape)
    gathered = jax.device_put(
        jnp.zeros(
            (
                shape.ep_size * shape.local_num_experts,
                shape.bt,
                T_PACKING,
                shape.hidden_size // T_PACKING,
            ),
            dtype=jnp.bfloat16,
        ),
        gather_sharding,
    )
    jax.block_until_ready((expert_outputs, gathered))

    with jax.set_mesh(mesh):

        @jax.jit
        def run_gather(expert_outputs_hbm, gathered_hbm):
            return _sharded_gather_call(
                expert_outputs_hbm,
                gathered_hbm,
                shape=shape,
                mesh=mesh,
                jax=jax,
                pl=pl,
                pltpu=pltpu,
                P=P,
            )

        jax.block_until_ready(run_gather(expert_outputs, gathered))
        task = f"layer1_a2a_gather_topk8_bt{shape.bt}_{shape.path_class}"
        return multiple_iteration_timeit_from_trace(
            compute_func=run_gather,
            data_generator=lambda: (expert_outputs, gathered),
            task=task,
            tries=sample_runs,
            warmup=warmup_runs,
            discard_initial_samples=discard_runs,
            trace_root=trace_root,
        )


def _sharded_gather_call(
    expert_outputs_hbm,
    gathered_hbm,
    *,
    shape: A2AGatherShape,
    mesh,
    jax,
    pl,
    pltpu,
    P,
):
    @jax.shard_map(
        mesh=mesh,
        in_specs=(P("tensor", None, None, None), P()),
        out_specs=P("tensor"),
        check_vma=False,
    )
    def per_rank(expert_outputs_local, gathered_replicated):
        return _pallas_a2a_gather_call(
            expert_outputs_local,
            gathered_replicated,
            shape=shape,
            jax=jax,
            pl=pl,
            pltpu=pltpu,
        )

    return per_rank(expert_outputs_hbm, gathered_hbm)


def _pallas_a2a_gather_call(expert_outputs_hbm, gathered_hbm, *, shape, jax, pl, pltpu):
    def kernel(expert_outputs_ref, gathered_ref, out_ref, send_sems, gather_sem, barrier_sem):
        del out_ref

        from jax import lax

        rank = lax.axis_index("tensor")

        def get_mesh_device_id(ep_rank):
            return (0, ep_rank)

        for peer in range(shape.ep_size):
            pltpu.semaphore_signal(
                barrier_sem,
                device_id=get_mesh_device_id(peer),
                device_id_type=pltpu.DeviceIdType.MESH,
            )
        pltpu.semaphore_wait(barrier_sem, shape.ep_size)

        def start_one(local_e_id, _):
            source_rank = (rank + shape.ep_size - local_e_id - 1) % shape.ep_size
            global_e_id = rank * shape.local_num_experts + local_e_id
            pltpu.make_async_remote_copy(
                src_ref=expert_outputs_ref.at[pl.ds(local_e_id, 1), pl.ds(0, shape.bt)],
                dst_ref=gathered_ref.at[pl.ds(global_e_id, 1), pl.ds(0, shape.bt)],
                send_sem=send_sems.at[local_e_id],
                recv_sem=gather_sem,
                device_id=get_mesh_device_id(source_rank),
                device_id_type=pltpu.DeviceIdType.MESH,
            ).start()
            return None

        lax.fori_loop(0, shape.top_k, start_one, None, unroll=False)

        def wait_send_one(local_e_id, _):
            ref = expert_outputs_ref.at[pl.ds(local_e_id, 1), pl.ds(0, shape.bt)]
            pltpu.make_async_copy(
                src_ref=ref,
                dst_ref=ref,
                sem=send_sems.at[local_e_id],
            ).wait()
            return None

        lax.fori_loop(0, shape.top_k, wait_send_one, None, unroll=False)

        def wait_recv_one(k_id, _):
            recv_rank = (rank + k_id + 1) % shape.ep_size
            global_e_id = recv_rank * shape.local_num_experts + k_id
            ref = gathered_ref.at[pl.ds(global_e_id, 1), pl.ds(0, shape.bt)]
            pltpu.make_async_copy(src_ref=ref, dst_ref=ref, sem=gather_sem).wait()
            return None

        lax.fori_loop(0, shape.top_k, wait_recv_one, None, unroll=False)

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((1,), expert_outputs_hbm.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
                pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
            ],
            out_specs=pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
            grid=(1,),
            scratch_shapes=[
                pltpu.SemaphoreType.DMA((shape.local_num_experts,)),
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
        name=f"layer1_a2a_gather_topk8_bt{shape.bt}",
    )(expert_outputs_hbm, gathered_hbm)


def _make_expert_outputs(*, jax: Any, np: Any, sharding: Any, shape: A2AGatherShape):
    try:
        from ml_dtypes import bfloat16 as numpy_bfloat16
    except Exception:
        numpy_bfloat16 = np.float32

    global_shape = (
        shape.ep_size * shape.local_num_experts,
        shape.bt,
        T_PACKING,
        shape.hidden_size // T_PACKING,
    )

    def data_callback(index):
        leading = index[0]
        start = int(leading.start or 0)
        stop = int(leading.stop or global_shape[0])
        local_shape = (stop - start, shape.bt, T_PACKING, shape.hidden_size // T_PACKING)
        return np.full(local_shape, (start // shape.local_num_experts) % 128, dtype=numpy_bfloat16)

    return jax.make_array_from_callback(global_shape, sharding, data_callback)


def _build_tensor_mesh(*, jax: Any, np: Any, ep_size: int, tp_size: int):
    from jax.experimental import mesh_utils
    from jax.sharding import Mesh

    if ep_size <= 0 or tp_size <= 0:
        raise ValueError(f"Expected {ep_size=} and {tp_size=} to be > 0.")
    devices = jax.devices()[: ep_size * tp_size]
    device_mesh = mesh_utils.create_device_mesh((tp_size, ep_size), devices=devices)
    return Mesh(np.asarray(device_mesh), ("data", "tensor"))


def _payload_bytes_per_device(shape: A2AGatherShape) -> int:
    return shape.bt * shape.top_k * shape.hidden_size * BF16_BYTES


def _remote_copy_count_per_device(shape: A2AGatherShape) -> int:
    # The fused-MoE gather path issues one bulk remote copy for each local
    # expert/source-rank pair with nonzero tokens. This benchmark's ring pattern
    # gives one source rank per top-k local expert.
    return shape.top_k


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
