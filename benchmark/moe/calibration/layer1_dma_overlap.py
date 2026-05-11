"""Layer 1 remote-DMA and weight-DMA overlap calibration.

This scenario is intentionally narrower than the kernel-aligned scatter/gather
benchmarks.  It answers one scheduling question: when a fused-MoE remote A2A DMA
and an HBM->VMEM weight DMA are concurrently issued, does the measured device
time behave like max(remote, weight), sum(remote, weight), or something between?
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal

from benchmark.moe.calibration.common import build_observation_row

SCENARIO_LAYER1_DMA_OVERLAP = "layer1_dma_overlap"
STATUS_MEASURED = "measured"
STATUS_NOT_IMPLEMENTED = "not_implemented"

DTYPE = "bfloat16"
WEIGHT_DTYPE = "bfloat16"
BF16_BYTES = 2
T_PACKING = 2
EP_SIZE = 32
HIDDEN_SIZE = 8192
LOCAL_NUM_EXPERTS = 8
VMEM_LIMIT_BYTES = 96 * 1024 * 1024

DEFAULT_WARMUP_RUNS = 3
DEFAULT_SAMPLE_RUNS = 7
DEFAULT_TRACE_DISCARD_RUNS = 1

PathClass = Literal[
    "barrier_only",
    "remote_only",
    "weight_only",
    "remote_plus_weight",
]


@dataclass(frozen=True)
class DMAOverlapShape:
    path: str
    path_class: PathClass
    bt: int
    remote_bytes: int
    remote_copy_count: int
    weight_bytes: int
    weight_copy_count: int
    hidden_size: int = HIDDEN_SIZE
    ep_size: int = EP_SIZE

    @property
    def has_remote(self) -> bool:
        return self.path_class in ("remote_only", "remote_plus_weight")

    @property
    def has_weight(self) -> bool:
        return self.path_class in ("weight_only", "remote_plus_weight")


def build_rows(
    *,
    suite: str,
    shapes: Iterable[DMAOverlapShape],
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
    warmup_runs = _positive_int_env("CALIBRATION_LAYER1_OVERLAP_WARMUP_RUNS", DEFAULT_WARMUP_RUNS)
    sample_runs = _positive_int_env("CALIBRATION_LAYER1_OVERLAP_SAMPLE_RUNS", DEFAULT_SAMPLE_RUNS)
    discard_runs = _nonnegative_int_env(
        "CALIBRATION_LAYER1_OVERLAP_TRACE_DISCARD_RUNS", DEFAULT_TRACE_DISCARD_RUNS
    )
    trace_root = os.getenv(
        "CALIBRATION_LAYER1_OVERLAP_TRACE_ROOT", "/tmp/sglang_jax_layer1_dma_overlap_trace"
    )

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
            samples = _measure_dma_overlap_ms(
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
                    "Measured with a controlled Pallas TPU microkernel that can issue "
                    "remote A2A make_async_remote_copy, HBM->VMEM make_async_copy, "
                    "or both in the same marker window. Compare barrier-adjusted "
                    "remote_only, weight_only, and remote_plus_weight rows to estimate "
                    "DMA overlap/contention for fused-MoE scheduling."
                ),
            )
        )

    return rows


def _make_row(
    *,
    suite: str,
    shape: DMAOverlapShape,
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
        scenario=SCENARIO_LAYER1_DMA_OVERLAP,
        suite=suite,
        layer=1,
        path=shape.path,
        path_class=shape.path_class,
        dtype=dtype,
        weight_dtype=weight_dtype,
        t_packing=t_packing,
        bf=shape.bt,
        bd=shape.hidden_size,
        tile_shape=_tile_shape(shape),
        bytes_hbm=shape.remote_bytes + shape.weight_bytes,
        bytes_per_fetch=max(shape.remote_bytes, shape.weight_bytes),
        dma_count=shape.remote_copy_count + shape.weight_copy_count,
        status=status,
        execution_mode=execution_mode,
        latency_ms_samples=latency_ms_samples,
        runtime=runtime,
        source=source,
        metadata=_metadata_for_shape(metadata, shape),
        implementation_note=implementation_note,
    )


def _metadata_for_shape(metadata: dict[str, Any], shape: DMAOverlapShape) -> dict[str, Any]:
    enriched = dict(metadata)
    enriched["dma_overlap"] = {
        "purpose": (
            "Determine whether fused-MoE remote A2A DMA and HBM->VMEM weight DMA "
            "can be treated as overlapping resources in the decode64 pipeline."
        ),
        "path": shape.path,
        "path_class": shape.path_class,
        "bt": shape.bt,
        "hidden_size": shape.hidden_size,
        "ep_size": shape.ep_size,
        "remote": {
            "enabled": shape.has_remote,
            "operation": "pltpu.make_async_remote_copy",
            "payload_bytes_per_device": shape.remote_bytes,
            "copy_count_per_device": shape.remote_copy_count,
            "payload_bytes_per_copy": _bytes_per_remote_copy(shape),
            "pattern": "each rank sends to (rank + 1) % ep_size and receives from previous rank",
        },
        "weight": {
            "enabled": shape.has_weight,
            "operation": "pltpu.make_async_copy",
            "payload_bytes_per_device": shape.weight_bytes,
            "copy_count_per_device": shape.weight_copy_count,
            "payload_bytes_per_copy": _bytes_per_weight_copy(shape),
            "pattern": "HBM->VMEM copy with the same self-copy wait anchor family as weight prefetch",
        },
        "interpretation": {
            "barrier_adjustment": (
                "Use barrier_only to adjust remote_only and remote_plus_weight before "
                "comparing with weight_only."
            ),
            "full_overlap_test": "T_both_adj ~= max(T_remote_adj, T_weight)",
            "no_overlap_test": "T_both_adj ~= T_remote_adj + T_weight",
        },
    }
    return enriched


def _with_measurement_metadata(
    metadata: dict[str, Any],
    *,
    shape: DMAOverlapShape,
    warmup_runs: int,
    sample_runs: int,
    discard_runs: int,
    trace_root: str,
) -> dict[str, Any]:
    enriched = _metadata_for_shape(metadata, shape)
    enriched["benchmark"] = {
        "name": "layer1_pallas_dma_overlap",
        "warmup_runs": warmup_runs,
        "sample_runs": sample_runs,
        "trace_discard_runs": discard_runs,
        "timing": "jax_profiler_trace_device_duration_ms",
        "trace_root": trace_root,
        "pallas_grid": (1,),
        "vmem_limit_bytes": VMEM_LIMIT_BYTES,
    }
    return enriched


def _not_implemented_note(execution_mode: str, runtime: dict[str, Any]) -> str:
    if execution_mode == "local_smoke":
        return (
            "layer1_dma_overlap emitted schema-only rows on local_smoke; "
            "Pallas DMA overlap measurements require a TPU backend."
        )
    backend = runtime.get("default_backend")
    return (
        "layer1_dma_overlap did not emit measured samples. "
        f"execution_mode={execution_mode!r} is not a measured Pallas mode "
        f"for this runtime; observed JAX default_backend={backend!r}."
    )


def _pallas_unavailable_note(runtime: dict[str, Any]) -> str | None:
    backend = runtime.get("default_backend")
    if backend != "tpu":
        return (
            "layer1_dma_overlap did not emit measured samples. "
            "Pallas DMA overlap measurements require JAX default_backend='tpu'; "
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
            "layer1_dma_overlap could not import the JAX/Pallas APIs needed "
            f"for measured DMA overlap; {type(exc).__name__}: {exc}. "
            "No measured latency samples were emitted."
        )
    return None


def _measurement_failed_note(exc: Exception) -> str:
    return (
        "layer1_dma_overlap Pallas measurement failed before producing trustworthy "
        f"samples: {type(exc).__name__}: {exc}. No measured latency samples were emitted."
    )


def _measure_dma_overlap_ms(
    shape: DMAOverlapShape,
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
    remote_src_sharding = NamedSharding(mesh, P("tensor", None, None, None))
    replicated_sharding = NamedSharding(mesh, P())

    remote_src = _make_remote_src(jax=jax, np=np, sharding=remote_src_sharding, shape=shape)
    remote_dst = jax.device_put(
        jnp.zeros(_remote_dst_shape(shape), dtype=jnp.bfloat16),
        replicated_sharding,
    )
    weight_src = jax.device_put(
        jnp.ones(_weight_src_shape(shape), dtype=jnp.bfloat16),
        replicated_sharding,
    )
    jax.block_until_ready((remote_src, remote_dst, weight_src))

    with jax.set_mesh(mesh):

        @jax.jit
        def run_overlap(remote_src_hbm, remote_dst_hbm, weight_src_hbm):
            return _sharded_dma_overlap_call(
                remote_src_hbm,
                remote_dst_hbm,
                weight_src_hbm,
                shape=shape,
                mesh=mesh,
                jax=jax,
                jnp=jnp,
                pl=pl,
                pltpu=pltpu,
                P=P,
            )

        jax.block_until_ready(run_overlap(remote_src, remote_dst, weight_src))
        task = f"layer1_dma_overlap_{shape.path}"
        return multiple_iteration_timeit_from_trace(
            compute_func=run_overlap,
            data_generator=lambda: (remote_src, remote_dst, weight_src),
            task=task,
            tries=sample_runs,
            warmup=warmup_runs,
            discard_initial_samples=discard_runs,
            trace_root=trace_root,
        )


def _sharded_dma_overlap_call(
    remote_src_hbm,
    remote_dst_hbm,
    weight_src_hbm,
    *,
    shape: DMAOverlapShape,
    mesh,
    jax,
    jnp,
    pl,
    pltpu,
    P,
):
    @jax.shard_map(
        mesh=mesh,
        in_specs=(P("tensor", None, None, None), P(), P()),
        out_specs=P("tensor"),
        check_vma=False,
    )
    def per_rank(remote_src_local, remote_dst_replicated, weight_src_replicated):
        return _pallas_dma_overlap_call(
            remote_src_local,
            remote_dst_replicated,
            weight_src_replicated,
            shape=shape,
            jax=jax,
            jnp=jnp,
            pl=pl,
            pltpu=pltpu,
        )

    return per_rank(remote_src_hbm, remote_dst_hbm, weight_src_hbm)


def _pallas_dma_overlap_call(
    remote_src_hbm, remote_dst_hbm, weight_src_hbm, *, shape, jax, jnp, pl, pltpu
):
    weight_elems_per_copy = _elements_per_weight_copy(shape)
    remote_elems_per_copy = _elements_per_remote_copy(shape)

    def kernel(
        remote_src_ref,
        remote_dst_ref,
        weight_src_ref,
        out_ref,
        weight_scratch,
        remote_send_sem,
        remote_recv_sem,
        weight_sem,
        barrier_sem,
    ):
        from jax import lax

        del out_ref
        rank = lax.axis_index("tensor")
        peer = (rank + 1) % shape.ep_size
        recv_peer = (rank + shape.ep_size - 1) % shape.ep_size

        def get_mesh_device_id(ep_rank):
            return (0, ep_rank)

        def sync_barrier():
            for target in range(shape.ep_size):
                pltpu.semaphore_signal(
                    barrier_sem,
                    device_id=get_mesh_device_id(target),
                    device_id_type=pltpu.DeviceIdType.MESH,
                )
            pltpu.semaphore_wait(barrier_sem, shape.ep_size)

        sync_barrier()

        if shape.has_remote:
            for copy_id in range(shape.remote_copy_count):
                pltpu.make_async_remote_copy(
                    src_ref=remote_src_ref.at[
                        copy_id,
                        pl.ds(0, T_PACKING),
                        pl.ds(0, remote_elems_per_copy),
                    ],
                    dst_ref=remote_dst_ref.at[
                        rank,
                        copy_id,
                        pl.ds(0, T_PACKING),
                        pl.ds(0, remote_elems_per_copy),
                    ],
                    send_sem=remote_send_sem,
                    recv_sem=remote_recv_sem,
                    device_id=get_mesh_device_id(peer),
                    device_id_type=pltpu.DeviceIdType.MESH,
                ).start()

        if shape.has_weight:
            for copy_id in range(shape.weight_copy_count):
                pltpu.make_async_copy(
                    src_ref=weight_src_ref.at[
                        copy_id,
                        pl.ds(0, T_PACKING),
                        pl.ds(0, weight_elems_per_copy),
                    ],
                    dst_ref=weight_scratch.at[
                        copy_id,
                        pl.ds(0, T_PACKING),
                        pl.ds(0, weight_elems_per_copy),
                    ],
                    sem=weight_sem,
                ).start()

        if shape.has_remote:
            for copy_id in range(shape.remote_copy_count):
                recv_ref = remote_dst_ref.at[
                    recv_peer,
                    copy_id,
                    pl.ds(0, T_PACKING),
                    pl.ds(0, remote_elems_per_copy),
                ]
                pltpu.make_async_copy(
                    src_ref=recv_ref, dst_ref=recv_ref, sem=remote_recv_sem
                ).wait()

            for copy_id in range(shape.remote_copy_count):
                send_ref = remote_src_ref.at[
                    copy_id,
                    pl.ds(0, T_PACKING),
                    pl.ds(0, remote_elems_per_copy),
                ]
                pltpu.make_async_copy(
                    src_ref=send_ref, dst_ref=send_ref, sem=remote_send_sem
                ).wait()

        if shape.has_weight:
            for copy_id in range(shape.weight_copy_count):
                ref = weight_scratch.at[
                    copy_id,
                    pl.ds(0, T_PACKING),
                    pl.ds(0, weight_elems_per_copy),
                ]
                pltpu.make_async_copy(src_ref=ref, dst_ref=ref, sem=weight_sem).wait()

        sync_barrier()

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((1,), jnp.bfloat16),
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
                pltpu.VMEM(_weight_scratch_shape(shape), jnp.bfloat16),
                pltpu.SemaphoreType.DMA,
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
        name=f"layer1_dma_overlap_{shape.path}",
    )(remote_src_hbm, remote_dst_hbm, weight_src_hbm)


def _make_remote_src(*, jax: Any, np: Any, sharding: Any, shape: DMAOverlapShape):
    try:
        from ml_dtypes import bfloat16 as numpy_bfloat16
    except Exception:
        numpy_bfloat16 = np.float32

    global_shape = (shape.ep_size, *(_remote_src_local_shape(shape)))

    def data_callback(index):
        leading = index[0]
        start = int(leading.start or 0)
        stop = int(leading.stop or global_shape[0])
        local_shape = (stop - start, *(_remote_src_local_shape(shape)))
        return np.full(local_shape, start % 128, dtype=numpy_bfloat16)

    return jax.make_array_from_callback(global_shape, sharding, data_callback)


def _build_tensor_mesh(*, jax: Any, np: Any, ep_size: int, tp_size: int):
    from jax.experimental import mesh_utils
    from jax.sharding import Mesh

    devices = jax.devices()[: ep_size * tp_size]
    device_mesh = mesh_utils.create_device_mesh((tp_size, ep_size), devices=devices)
    return Mesh(np.asarray(device_mesh), ("data", "tensor"))


def _remote_src_local_shape(shape: DMAOverlapShape) -> tuple[int, int, int]:
    return (max(1, shape.remote_copy_count), T_PACKING, max(1, _elements_per_remote_copy(shape)))


def _remote_dst_shape(shape: DMAOverlapShape) -> tuple[int, int, int, int]:
    return (shape.ep_size, *_remote_src_local_shape(shape))


def _weight_src_shape(shape: DMAOverlapShape) -> tuple[int, int, int]:
    return _weight_scratch_shape(shape)


def _weight_scratch_shape(shape: DMAOverlapShape) -> tuple[int, int, int]:
    return (max(1, shape.weight_copy_count), T_PACKING, max(1, _elements_per_weight_copy(shape)))


def _tile_shape(shape: DMAOverlapShape) -> tuple[int, int, int, int]:
    return (
        max(1, shape.remote_copy_count),
        max(1, shape.weight_copy_count),
        T_PACKING,
        max(_elements_per_remote_copy(shape), _elements_per_weight_copy(shape), 1),
    )


def _bytes_per_remote_copy(shape: DMAOverlapShape) -> int:
    if shape.remote_copy_count <= 0:
        return 0
    return shape.remote_bytes // shape.remote_copy_count


def _bytes_per_weight_copy(shape: DMAOverlapShape) -> int:
    if shape.weight_copy_count <= 0:
        return 0
    return shape.weight_bytes // shape.weight_copy_count


def _elements_per_remote_copy(shape: DMAOverlapShape) -> int:
    bytes_per_copy = _bytes_per_remote_copy(shape)
    if bytes_per_copy <= 0:
        return 1
    _validate_copy_bytes(bytes_per_copy, label=f"{shape.path}.remote")
    return bytes_per_copy // (T_PACKING * BF16_BYTES)


def _elements_per_weight_copy(shape: DMAOverlapShape) -> int:
    bytes_per_copy = _bytes_per_weight_copy(shape)
    if bytes_per_copy <= 0:
        return 1
    _validate_copy_bytes(bytes_per_copy, label=f"{shape.path}.weight")
    return bytes_per_copy // (T_PACKING * BF16_BYTES)


def _validate_copy_bytes(bytes_per_copy: int, *, label: str) -> None:
    divisor = T_PACKING * BF16_BYTES
    if bytes_per_copy % divisor != 0:
        raise ValueError(f"{label} bytes_per_copy={bytes_per_copy} is not divisible by {divisor}.")


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
