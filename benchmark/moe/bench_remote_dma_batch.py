"""Pure remote DMA launch benchmark for fused EP-MoE Stage2 analysis.

This benchmark measures the upper bound of DMA launch reduction without paying
any pack/permute cost:

  small_dma
      For each destination peer, issue one remote DMA per token-expert entry.

  batch_dma
      For each destination peer, issue one remote DMA containing all entries for
      that peer. The input is already laid out contiguously by peer, so this is
      the idealized upper bound for pack-based Stage2 scatter.

`entries_per_peer` is already after top-k expansion. For example, if
local_tokens=128 and top_k=8, local token-expert entries are 1024. With
fanout=8, entries_per_peer is roughly 128.
"""

from __future__ import annotations

import argparse
import functools
import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from benchmark.moe.utils import MARKER, build_mesh
from benchmark.utils import multiple_iteration_timeit_from_trace

MODES = ("small_dma", "batch_dma")


def _dtype_packing(dtype: jnp.dtype) -> int:
    bits = jnp.dtype(dtype).itemsize * 8
    if 32 % bits != 0:
        raise ValueError(f"Unsupported dtype packing for {dtype=} ({bits=} bits).")
    return 32 // bits


def _wait_dma(ref, sem, count):
    @pl.when(count != 0)
    def _():
        pltpu.make_async_copy(
            src_ref=ref.at[pl.ds(0, count)],
            dst_ref=ref.at[pl.ds(0, count)],
            sem=sem,
        ).wait()


def _sync_barrier(barrier_sem, *, num_devices: int, tp_size: int):
    def _mesh_device_id(ep_rank):
        return (ep_rank // tp_size, ep_rank % tp_size)

    for dev in range(num_devices):
        pltpu.semaphore_signal(
            barrier_sem,
            device_id=_mesh_device_id(dev),
            device_id_type=pltpu.DeviceIdType.MESH,
        )
    pltpu.semaphore_wait(barrier_sem, num_devices)


def _store_zero(out_hbm, out_vmem, out_sem):
    out_vmem[...] = jnp.zeros((128,), dtype=jnp.float32)
    copy = pltpu.make_async_copy(src_ref=out_vmem, dst_ref=out_hbm, sem=out_sem)
    copy.start()
    copy.wait()


def _small_dma_kernel(
    src_hbm,
    recv_hbm,
    out_hbm,
    send_sems,
    recv_sems,
    barrier_sem,
    out_vmem,
    out_sem,
    *,
    fanout: int,
    entries_per_peer: int,
    dp_axis_name: str,
    tp_axis_name: str,
):
    dp_rank = lax.axis_index(dp_axis_name)
    tp_rank = lax.axis_index(tp_axis_name)
    tp_size = lax.axis_size(tp_axis_name)
    dp_size = lax.axis_size(dp_axis_name)
    my_id = dp_rank * tp_size + tp_rank
    num_devices = dp_size * tp_size

    def _mesh_device_id(ep_rank):
        return (ep_rank // tp_size, ep_rank % tp_size)

    for lane in range(fanout):
        dst = (my_id + jnp.int32(1 + lane)) % jnp.int32(num_devices)

        def _entry_body(entry_id, _):
            src_offset = lane * entries_per_peer + entry_id
            pltpu.make_async_remote_copy(
                src_ref=src_hbm.at[pl.ds(src_offset, 1)],
                dst_ref=recv_hbm.at[my_id, pl.ds(entry_id, 1)],
                send_sem=send_sems.at[lane],
                recv_sem=recv_sems.at[my_id],
                device_id=_mesh_device_id(dst),
                device_id_type=pltpu.DeviceIdType.MESH,
            ).start()
            return None

        lax.fori_loop(0, entries_per_peer, _entry_body, None, unroll=False)

    for lane in range(fanout):
        src = (my_id - jnp.int32(1 + lane) + jnp.int32(num_devices)) % jnp.int32(num_devices)
        _wait_dma(recv_hbm.at[src], recv_sems.at[src], jnp.int32(entries_per_peer))
        _wait_dma(
            src_hbm.at[pl.ds(lane * entries_per_peer, entries_per_peer)],
            send_sems.at[lane],
            jnp.int32(entries_per_peer),
        )

    _sync_barrier(barrier_sem, num_devices=num_devices, tp_size=tp_size)
    _store_zero(out_hbm, out_vmem, out_sem)


def _batch_dma_kernel(
    src_hbm,
    recv_hbm,
    out_hbm,
    send_sems,
    recv_sems,
    barrier_sem,
    out_vmem,
    out_sem,
    *,
    fanout: int,
    entries_per_peer: int,
    dp_axis_name: str,
    tp_axis_name: str,
):
    dp_rank = lax.axis_index(dp_axis_name)
    tp_rank = lax.axis_index(tp_axis_name)
    tp_size = lax.axis_size(tp_axis_name)
    dp_size = lax.axis_size(dp_axis_name)
    my_id = dp_rank * tp_size + tp_rank
    num_devices = dp_size * tp_size

    def _mesh_device_id(ep_rank):
        return (ep_rank // tp_size, ep_rank % tp_size)

    for lane in range(fanout):
        dst = (my_id + jnp.int32(1 + lane)) % jnp.int32(num_devices)
        pltpu.make_async_remote_copy(
            src_ref=src_hbm.at[pl.ds(lane * entries_per_peer, entries_per_peer)],
            dst_ref=recv_hbm.at[my_id, pl.ds(0, entries_per_peer)],
            send_sem=send_sems.at[lane],
            recv_sem=recv_sems.at[my_id],
            device_id=_mesh_device_id(dst),
            device_id_type=pltpu.DeviceIdType.MESH,
        ).start()

    for lane in range(fanout):
        src = (my_id - jnp.int32(1 + lane) + jnp.int32(num_devices)) % jnp.int32(num_devices)
        _wait_dma(recv_hbm.at[src], recv_sems.at[src], jnp.int32(entries_per_peer))
        _wait_dma(
            src_hbm.at[pl.ds(lane * entries_per_peer, entries_per_peer)],
            send_sems.at[lane],
            jnp.int32(entries_per_peer),
        )

    _sync_barrier(barrier_sem, num_devices=num_devices, tp_size=tp_size)
    _store_zero(out_hbm, out_vmem, out_sem)


def _make_kernel(
    *,
    mode: str,
    mesh: jax.sharding.Mesh,
    fanout: int,
    entries_per_peer: int,
    hidden_size: int,
    dtype: jnp.dtype,
    vmem_limit_bytes: int,
):
    ep_size = mesh.shape["tensor"] * mesh.shape["data"]
    if not (1 <= fanout <= ep_size - 1):
        raise ValueError(f"Expected 1 <= {fanout=} <= {ep_size - 1}.")
    t_packing = _dtype_packing(dtype)
    if hidden_size % t_packing:
        raise ValueError(f"Expected {hidden_size=} aligned to {t_packing=}.")
    hidden_per_pack = hidden_size // t_packing
    recv_shape = (ep_size, entries_per_peer, t_packing, hidden_per_pack)
    hbm_block_spec = pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)
    out_shape = jax.ShapeDtypeStruct((128,), jnp.float32)
    kernel = _small_dma_kernel if mode == "small_dma" else _batch_dma_kernel

    pallas_fn = pl.pallas_call(
        functools.partial(
            kernel,
            fanout=fanout,
            entries_per_peer=entries_per_peer,
            dp_axis_name="data",
            tp_axis_name="tensor",
        ),
        out_shape=out_shape,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[hbm_block_spec, hbm_block_spec],
            out_specs=hbm_block_spec,
            scratch_shapes=[
                pltpu.SemaphoreType.DMA((fanout,)),
                pltpu.SemaphoreType.DMA((ep_size,)),
                pltpu.SemaphoreType.BARRIER,
                pltpu.VMEM((128,), jnp.float32),
                pltpu.SemaphoreType.DMA,
            ],
        ),
        compiler_params=pltpu.CompilerParams(
            collective_id=0,
            allow_collective_id_without_custom_barrier=True,
            has_side_effects=True,
            vmem_limit_bytes=vmem_limit_bytes,
        ),
        name=f"{MARKER}_remote_dma_{mode}_fanout{fanout}_entries{entries_per_peer}_hidden{hidden_size}",
    )

    @jax.jit
    @jax.shard_map(
        mesh=mesh,
        in_specs=(P("tensor", None, None),),
        out_specs=P(None),
        check_vma=False,
    )
    def run(src):
        recv = pl.empty(recv_shape, dtype)
        return pallas_fn(
            pltpu.with_memory_space_constraint(src, pltpu.HBM),
            pltpu.with_memory_space_constraint(recv, pltpu.HBM),
        )

    return run


def _make_src(
    *,
    mesh: jax.sharding.Mesh,
    fanout: int,
    entries_per_peer: int,
    hidden_size: int,
    dtype: jnp.dtype,
):
    t_packing = _dtype_packing(dtype)
    hidden_per_pack = hidden_size // t_packing
    ep_size = mesh.shape["tensor"] * mesh.shape["data"]
    total_entries = fanout * entries_per_peer
    sharding = NamedSharding(mesh, P("tensor", None, None))
    return jax.jit(
        lambda: jnp.ones((ep_size * total_entries, t_packing, hidden_per_pack), dtype),
        out_shardings=sharding,
    )()


def _init_distributed(args: argparse.Namespace) -> None:
    if args.dist_init_addr is None and args.num_processes is None and args.process_id is None:
        return
    missing = [
        name
        for name in ("dist_init_addr", "num_processes", "process_id")
        if getattr(args, name) is None
    ]
    if missing:
        raise ValueError(f"Missing distributed args: {missing}")
    if jax.distributed.is_initialized():
        return
    jax.distributed.initialize(
        coordinator_address=args.dist_init_addr,
        num_processes=args.num_processes,
        process_id=args.process_id,
        local_device_ids=args.local_device_ids,
        initialization_timeout=args.distributed_init_timeout,
    )


def _summary(values: list[float]) -> dict[str, float]:
    arr = jnp.asarray(values, dtype=jnp.float32)
    return {
        "mean_ms": float(jnp.mean(arr)),
        "median_ms": float(jnp.median(arr)),
        "min_ms": float(jnp.min(arr)),
        "max_ms": float(jnp.max(arr)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark pure remote DMA batch upper bound.")
    parser.add_argument("--ep-size", type=int, required=True)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--hidden-size", nargs="+", type=int, default=[8192])
    parser.add_argument("--entries-per-peer", nargs="+", type=int, default=[1, 4, 16, 64, 256])
    parser.add_argument("--fanout", nargs="+", type=int, default=[1, 4, 8])
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--warmup-iters", type=int, default=1)
    parser.add_argument("--trace-root", default="/tmp/sglang_remote_dma_batch_trace")
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--vmem-limit-mb", type=int, default=96)
    parser.add_argument("--dist-init-addr", type=str, default=None)
    parser.add_argument("--num-processes", "--nnodes", dest="num_processes", type=int, default=None)
    parser.add_argument("--process-id", type=int, default=None)
    parser.add_argument("--local-device-ids", type=int, nargs="+", default=None)
    parser.add_argument("--distributed-init-timeout", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _init_distributed(args)
    mesh = build_mesh(ep_size=args.ep_size, tp_size=args.tp_size)
    dtype = jnp.bfloat16
    dtype_bytes = jnp.dtype(dtype).itemsize
    vmem_limit_bytes = args.vmem_limit_mb * 1024 * 1024
    rows = []

    print(
        "remote DMA batch benchmark: "
        f"process={jax.process_index()}/{jax.process_count()}, "
        f"local_devices={len(jax.local_devices())}, global_devices={len(jax.devices())}, "
        f"ep_size={args.ep_size}, dtype={dtype}, vmem_limit_mb={args.vmem_limit_mb}"
    )

    for hidden_size in args.hidden_size:
        entry_bytes = hidden_size * dtype_bytes
        for fanout_arg in args.fanout:
            fanout = max(1, min(int(fanout_arg), args.ep_size - 1))
            for entries_per_peer in args.entries_per_peer:
                src_data = _make_src(
                    mesh=mesh,
                    fanout=fanout,
                    entries_per_peer=entries_per_peer,
                    hidden_size=hidden_size,
                    dtype=dtype,
                )
                for mode in args.modes:
                    try:
                        run = _make_kernel(
                            mode=mode,
                            mesh=mesh,
                            fanout=fanout,
                            entries_per_peer=entries_per_peer,
                            hidden_size=hidden_size,
                            dtype=dtype,
                            vmem_limit_bytes=vmem_limit_bytes,
                        )

                        def _data():
                            return (src_data,)

                        samples = multiple_iteration_timeit_from_trace(
                            run,
                            _data,
                            task=(
                                f"remote_dma_{mode}_ep{args.ep_size}_h{hidden_size}"
                                f"_fanout{fanout}_entries{entries_per_peer}"
                            ),
                            tries=args.iters,
                            warmup=args.warmup_iters,
                            trace_root=args.trace_root,
                        )
                        stats = _summary(samples)
                        status = "ok"
                        error = None
                    except Exception as exc:  # noqa: BLE001 - keep the sweep running.
                        samples = []
                        stats = {
                            "mean_ms": math.nan,
                            "median_ms": math.nan,
                            "min_ms": math.nan,
                            "max_ms": math.nan,
                        }
                        status = "error"
                        error = f"{type(exc).__name__}: {exc}"
                        print(
                            "ERROR "
                            f"mode={mode} hidden_size={hidden_size} fanout={fanout} "
                            f"entries_per_peer={entries_per_peer}: {error}"
                        )

                    remote_bytes = fanout * entries_per_peer * entry_bytes
                    row = {
                        "ep_size": args.ep_size,
                        "hidden_size": hidden_size,
                        "dtype": str(dtype),
                        "entry_bytes": int(entry_bytes),
                        "fanout": fanout,
                        "entries_per_peer": entries_per_peer,
                        "batch_bytes_per_peer": int(entries_per_peer * entry_bytes),
                        "remote_bytes_per_device": int(remote_bytes),
                        "mode": mode,
                        "status": status,
                        "error": error,
                        "samples_ms": samples,
                        "effective_remote_gbs_mean": (
                            float(remote_bytes) / (stats["mean_ms"] / 1000.0) / 1e9
                            if stats["mean_ms"] and not math.isnan(stats["mean_ms"])
                            else math.nan
                        ),
                        **stats,
                    }
                    rows.append(row)
                    print("RESULT " + json.dumps(row, sort_keys=True))

    if args.out_json and jax.process_index() == 0:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            for row in rows:
                f.write(json.dumps(row, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
