"""Layer 1 fused-MoE wait/barrier calibration."""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal

from benchmark.moe.calibration.common import build_observation_row

SCENARIO_LAYER1_WAIT = "layer1_wait"
SUITE_V7X32_BF16_WAIT_PRIMITIVES = "v7x32_bf16_wait_primitives"
SUPPORTED_SUITES = (SUITE_V7X32_BF16_WAIT_PRIMITIVES,)

STATUS_MEASURED = "measured"
STATUS_NOT_IMPLEMENTED = "not_implemented"
DEFAULT_WARMUP_RUNS = 3
DEFAULT_SAMPLE_RUNS = 7
DEFAULT_TRACE_DISCARD_RUNS = 1

DTYPE = "bfloat16"
WEIGHT_DTYPE = "bfloat16"
BF16_BYTES = 2
T_PACKING = 2
HIDDEN_SIZE = 8192
EP_SIZE = 32
KERNEL_PATH = "python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py"
VMEM_LIMIT_BYTES = 96 * 1024 * 1024

WaitPath = Literal["mesh_barrier", "remote_dma_wait_1token"]


@dataclass(frozen=True)
class WaitShape:
    path: WaitPath
    path_class: str
    repetitions: int
    hidden_size: int = HIDDEN_SIZE
    ep_size: int = EP_SIZE


def build_rows(
    *,
    suite: str,
    shapes: Iterable[WaitShape],
    execution_mode: str,
    runtime: dict[str, Any],
    dtype: str,
    weight_dtype: str,
    t_packing: int,
    source: dict[str, Any],
    metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    if suite not in SUPPORTED_SUITES:
        raise ValueError(f"Unsupported Layer 1 wait suite: {suite}")
    if dtype != DTYPE or weight_dtype != WEIGHT_DTYPE or t_packing != T_PACKING:
        raise ValueError("layer1_wait supports only bf16 with t_packing=2.")

    if execution_mode != "pallas" or runtime.get("default_backend") != "tpu":
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

    rows: list[dict[str, Any]] = []
    warmup_runs = _positive_int_env("CALIBRATION_LAYER1_WAIT_WARMUP_RUNS", DEFAULT_WARMUP_RUNS)
    sample_runs = _positive_int_env("CALIBRATION_LAYER1_WAIT_SAMPLE_RUNS", DEFAULT_SAMPLE_RUNS)
    discard_runs = _nonnegative_int_env(
        "CALIBRATION_LAYER1_WAIT_TRACE_DISCARD_RUNS", DEFAULT_TRACE_DISCARD_RUNS
    )
    trace_root = os.getenv("CALIBRATION_LAYER1_WAIT_TRACE_ROOT", "/tmp/sglang_jax_layer1_wait")

    for shape in shapes:
        try:
            samples = _measure_wait_ms(
                shape,
                warmup_runs=warmup_runs,
                sample_runs=sample_runs,
                discard_runs=discard_runs,
                trace_root=trace_root,
            )
            status = STATUS_MEASURED
            note = (
                "Measured fused-MoE synchronization primitive with Pallas TPU trace timing. "
                "Rows isolate barrier or remote-DMA wait anchors from payload-heavy scatter/gather."
            )
        except Exception as exc:
            samples = []
            status = STATUS_NOT_IMPLEMENTED
            note = f"Layer1 wait measurement failed: {type(exc).__name__}: {exc}"

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
                metadata=metadata,
                status=status,
                latency_ms_samples=samples,
                implementation_note=note,
            )
        )
    return rows


def _make_row(
    *,
    suite: str,
    shape: WaitShape,
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
        scenario=SCENARIO_LAYER1_WAIT,
        suite=suite,
        layer=1,
        path=shape.path,
        path_class=shape.path_class,
        dtype=dtype,
        weight_dtype=weight_dtype,
        t_packing=t_packing,
        bf=shape.repetitions,
        bd=shape.hidden_size,
        tile_shape=_tile_shape(shape),
        bytes_hbm=_bytes_hbm(shape),
        bytes_per_fetch=_bytes_hbm(shape),
        dma_count=_dma_count(shape),
        status=status,
        execution_mode=execution_mode,
        latency_ms_samples=latency_ms_samples,
        runtime=runtime,
        source=source,
        metadata=_metadata_for_shape(metadata, shape, status=status),
        implementation_note=implementation_note,
    )


def _metadata_for_shape(
    metadata: dict[str, Any], shape: WaitShape, *, status: str
) -> dict[str, Any]:
    enriched = dict(metadata)
    enriched["wait_primitive"] = {
        "kernel_path": KERNEL_PATH,
        "kernel_references": [
            f"{KERNEL_PATH}:sync_barrier",
            f"{KERNEL_PATH}:wait_a2a_scatter_send_batch",
            f"{KERNEL_PATH}:wait_a2a_gather_recv_all",
        ],
        "path": shape.path,
        "path_class": shape.path_class,
        "repetitions": shape.repetitions,
        "hidden_size": shape.hidden_size,
        "ep_size": shape.ep_size,
        "tile_shape": _tile_shape(shape),
        "bytes_hbm": _bytes_hbm(shape),
        "dma_count": _dma_count(shape),
        "measurement_status": "measured" if status == STATUS_MEASURED else "schema_only",
        "includes": _includes(shape),
        "excludes": ["expert_compute", "weight_prefetch", "topk", "output_accumulate"],
    }
    return enriched


def _measure_wait_ms(
    shape: WaitShape,
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
    src_sharding = NamedSharding(mesh, P("tensor", None))
    dst_sharding = NamedSharding(mesh, P())
    payload_elems = _payload_elems(shape)
    src = jax.device_put(jnp.ones((shape.ep_size, payload_elems), dtype=jnp.bfloat16), src_sharding)
    dst = jax.device_put(
        jnp.zeros((shape.ep_size, payload_elems), dtype=jnp.bfloat16), dst_sharding
    )
    jax.block_until_ready((src, dst))

    with jax.set_mesh(mesh):

        @jax.jit
        def run_wait(src_hbm, dst_hbm):
            return _sharded_wait_call(
                src_hbm,
                dst_hbm,
                shape=shape,
                mesh=mesh,
                jax=jax,
                pl=pl,
                pltpu=pltpu,
                P=P,
            )

        jax.block_until_ready(run_wait(src, dst))
        task = f"layer1_wait_{shape.path}_rep{shape.repetitions}"
        return multiple_iteration_timeit_from_trace(
            compute_func=run_wait,
            data_generator=lambda: (src, dst),
            task=task,
            tries=sample_runs,
            warmup=warmup_runs,
            discard_initial_samples=discard_runs,
            trace_root=trace_root,
        )


def _sharded_wait_call(src_hbm, dst_hbm, *, shape: WaitShape, mesh, jax, pl, pltpu, P):
    @jax.shard_map(
        mesh=mesh,
        in_specs=(P("tensor", None), P()),
        out_specs=P("tensor"),
        check_vma=False,
    )
    def per_rank(src_local, dst_replicated):
        return _pallas_wait_call(
            src_local,
            dst_replicated,
            shape=shape,
            jax=jax,
            pl=pl,
            pltpu=pltpu,
        )

    return per_rank(src_hbm, dst_hbm)


def _pallas_wait_call(src_hbm, dst_hbm, *, shape: WaitShape, jax, pl, pltpu):
    def kernel(src_ref, dst_ref, out_ref, send_sem, recv_sem, barrier_sem):
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

        def remote_roundtrip():
            peer = (rank + 1) % shape.ep_size
            recv_peer = (rank + shape.ep_size - 1) % shape.ep_size
            pltpu.make_async_remote_copy(
                src_ref=src_ref.at[pl.ds(0, 1), pl.ds(0, _payload_elems(shape))],
                dst_ref=dst_ref.at[pl.ds(rank, 1), pl.ds(0, _payload_elems(shape))],
                send_sem=send_sem,
                recv_sem=recv_sem,
                device_id=get_mesh_device_id(peer),
                device_id_type=pltpu.DeviceIdType.MESH,
            ).start()
            recv_ref = dst_ref.at[pl.ds(recv_peer, 1), pl.ds(0, _payload_elems(shape))]
            pltpu.make_async_copy(src_ref=recv_ref, dst_ref=recv_ref, sem=recv_sem).wait()
            send_ref = src_ref.at[pl.ds(0, 1), pl.ds(0, _payload_elems(shape))]
            pltpu.make_async_copy(src_ref=send_ref, dst_ref=send_ref, sem=send_sem).wait()

        for _ in range(shape.repetitions):
            if shape.path == "mesh_barrier":
                sync_barrier()
            else:
                sync_barrier()
                remote_roundtrip()
                sync_barrier()
        out_ref[0] = src_ref[0, 0]

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((1,), src_hbm.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
                pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
            ],
            out_specs=pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
            grid=(1,),
            scratch_shapes=[
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.BARRIER,
            ],
        ),
        compiler_params=pltpu.CompilerParams(
            collective_id=5,
            allow_collective_id_without_custom_barrier=True,
            has_side_effects=True,
            vmem_limit_bytes=VMEM_LIMIT_BYTES,
        ),
        name=f"layer1_wait_{shape.path}_rep{shape.repetitions}",
    )(src_hbm, dst_hbm)


def _build_tensor_mesh(*, jax: Any, np: Any, ep_size: int, tp_size: int):
    from jax.experimental import mesh_utils
    from jax.sharding import Mesh

    devices = jax.devices()[: ep_size * tp_size]
    device_mesh = mesh_utils.create_device_mesh((tp_size, ep_size), devices=devices)
    return Mesh(np.asarray(device_mesh), ("data", "tensor"))


def _tile_shape(shape: WaitShape) -> tuple[int, ...]:
    return (shape.repetitions, _payload_elems(shape))


def _payload_elems(shape: WaitShape) -> int:
    return 1 if shape.path == "mesh_barrier" else shape.hidden_size


def _bytes_hbm(shape: WaitShape) -> int:
    if shape.path == "mesh_barrier":
        return 0
    return shape.repetitions * shape.hidden_size * BF16_BYTES


def _dma_count(shape: WaitShape) -> int:
    if shape.path == "mesh_barrier":
        return 0
    return shape.repetitions


def _includes(shape: WaitShape) -> list[str]:
    if shape.path == "mesh_barrier":
        return ["mesh_barrier_signal_all_peers", "mesh_barrier_wait_all_peers"]
    return ["mesh_barrier", "one_token_remote_dma_start", "recv_wait", "send_wait"]


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


def _not_implemented_note(execution_mode: str, runtime: dict[str, Any]) -> str:
    if execution_mode == "local_smoke":
        return "layer1_wait emitted schema-only rows on local_smoke."
    return (
        "layer1_wait requires TPU Pallas execution; "
        f"execution_mode={execution_mode!r}, default_backend={runtime.get('default_backend')!r}."
    )
