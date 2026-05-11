"""Stage2 scatter staging microbenchmark for fused EP-MoE.

This benchmark isolates the RFC-0026 Stage2 scatter question:

  direct: token/top-k remote copies into the receiver buffer
  staged: local pack -> per-destination contiguous remote copy -> local demux

The staged path is an upper-bound test for a per-device DMA merge design.  It
keeps the routing pattern deterministic so the same script can run on
EP=8/32/64 without model weights.
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


def _dtype_packing(dtype: jnp.dtype) -> int:
    bits = jnp.dtype(dtype).itemsize * 8
    if 32 % bits != 0:
        raise ValueError(f"Unsupported dtype packing for {dtype=} ({bits=} bits).")
    return 32 // bits


def _dest_for(src_rank, t_id, k_id, *, top_k: int, fanout: int, num_devices: int):
    del top_k
    # Keep the benchmark focused on remote scatter.  fanout is capped by the
    # caller to [1, num_devices - 1].
    lane = (t_id + k_id) % jnp.int32(fanout)
    return (src_rank + jnp.int32(1) + lane) % jnp.int32(num_devices)


def _compute_send_counts(
    send_counts, *, my_id, local_num_tokens: int, top_k: int, fanout: int, num_devices: int
):
    for dev in range(num_devices):
        send_counts[dev] = jnp.int32(0)

    def _token_body(t_id, _):
        for k_id in range(top_k):
            dst = _dest_for(
                my_id,
                t_id,
                jnp.int32(k_id),
                top_k=top_k,
                fanout=fanout,
                num_devices=num_devices,
            )
            send_counts[dst] = send_counts[dst] + jnp.int32(1)
        return None

    lax.fori_loop(0, local_num_tokens, _token_body, None, unroll=False)


def _recv_count_from_src(
    src_rank, *, my_id, local_num_tokens: int, top_k: int, fanout: int, num_devices: int
):
    def _token_body(t_id, count):
        for k_id in range(top_k):
            dst = _dest_for(
                src_rank,
                t_id,
                jnp.int32(k_id),
                top_k=top_k,
                fanout=fanout,
                num_devices=num_devices,
            )
            count += lax.select(dst == my_id, jnp.int32(1), jnp.int32(0))
        return count

    return lax.fori_loop(0, local_num_tokens, _token_body, jnp.int32(0), unroll=False)


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
    copy = pltpu.make_async_copy(
        src_ref=out_vmem,
        dst_ref=out_hbm,
        sem=out_sem,
    )
    copy.start()
    copy.wait()


def _direct_scatter_kernel(
    tokens_hbm,
    direct_hbm,
    out_hbm,
    send_counts_smem,
    send_sems,
    recv_sems,
    barrier_sem,
    out_vmem,
    out_sem,
    *,
    top_k: int,
    fanout: int,
    dp_axis_name: str,
    tp_axis_name: str,
):
    dp_rank = lax.axis_index(dp_axis_name)
    tp_rank = lax.axis_index(tp_axis_name)
    tp_size = lax.axis_size(tp_axis_name)
    dp_size = lax.axis_size(dp_axis_name)
    my_id = dp_rank * tp_size + tp_rank
    num_devices = dp_size * tp_size
    local_num_tokens = tokens_hbm.shape[0]

    def _mesh_device_id(ep_rank):
        return (ep_rank // tp_size, ep_rank % tp_size)

    _compute_send_counts(
        send_counts_smem,
        my_id=my_id,
        local_num_tokens=local_num_tokens,
        top_k=top_k,
        fanout=fanout,
        num_devices=num_devices,
    )
    for dev in range(num_devices):
        send_counts_smem[dev] = jnp.int32(0)

    def _token_body(t_id, _):
        for k_id in range(top_k):
            dst = _dest_for(
                my_id,
                t_id,
                jnp.int32(k_id),
                top_k=top_k,
                fanout=fanout,
                num_devices=num_devices,
            )
            offset = send_counts_smem[dst]
            send_counts_smem[dst] = offset + jnp.int32(1)
            pltpu.make_async_remote_copy(
                src_ref=tokens_hbm.at[pl.ds(t_id, 1)],
                dst_ref=direct_hbm.at[my_id, pl.ds(offset, 1)],
                send_sem=send_sems.at[dst],
                recv_sem=recv_sems.at[my_id],
                device_id=_mesh_device_id(dst),
                device_id_type=pltpu.DeviceIdType.MESH,
            ).start()
        return None

    lax.fori_loop(0, local_num_tokens, _token_body, None, unroll=False)

    for src in range(num_devices):
        recv_count = _recv_count_from_src(
            jnp.int32(src),
            my_id=my_id,
            local_num_tokens=local_num_tokens,
            top_k=top_k,
            fanout=fanout,
            num_devices=num_devices,
        )
        _wait_dma(direct_hbm.at[src], recv_sems.at[src], recv_count)

    for dst in range(num_devices):
        _wait_dma(direct_hbm.at[0], send_sems.at[dst], send_counts_smem[dst])

    _sync_barrier(barrier_sem, num_devices=num_devices, tp_size=tp_size)
    _store_zero(out_hbm, out_vmem, out_sem)


def _staged_scatter_kernel(
    tokens_hbm,
    pack_hbm,
    recv_hbm,
    demux_hbm,
    out_hbm,
    send_counts_smem,
    pack_sems,
    demux_sems,
    send_sems,
    recv_sems,
    barrier_sem,
    out_vmem,
    out_sem,
    *,
    top_k: int,
    fanout: int,
    dp_axis_name: str,
    tp_axis_name: str,
    remote_only: bool,
):
    dp_rank = lax.axis_index(dp_axis_name)
    tp_rank = lax.axis_index(tp_axis_name)
    tp_size = lax.axis_size(tp_axis_name)
    dp_size = lax.axis_size(dp_axis_name)
    my_id = dp_rank * tp_size + tp_rank
    num_devices = dp_size * tp_size
    local_num_tokens = tokens_hbm.shape[0]

    def _mesh_device_id(ep_rank):
        return (ep_rank // tp_size, ep_rank % tp_size)

    _compute_send_counts(
        send_counts_smem,
        my_id=my_id,
        local_num_tokens=local_num_tokens,
        top_k=top_k,
        fanout=fanout,
        num_devices=num_devices,
    )

    if not remote_only:
        for dev in range(num_devices):
            send_counts_smem[dev] = jnp.int32(0)

        def _pack_token_body(t_id, _):
            for k_id in range(top_k):
                dst = _dest_for(
                    my_id,
                    t_id,
                    jnp.int32(k_id),
                    top_k=top_k,
                    fanout=fanout,
                    num_devices=num_devices,
                )
                offset = send_counts_smem[dst]
                send_counts_smem[dst] = offset + jnp.int32(1)
                pltpu.make_async_copy(
                    src_ref=tokens_hbm.at[pl.ds(t_id, 1)],
                    dst_ref=pack_hbm.at[dst, pl.ds(offset, 1)],
                    sem=pack_sems.at[dst],
                ).start()
            return None

        lax.fori_loop(0, local_num_tokens, _pack_token_body, None, unroll=False)

        for dst in range(num_devices):
            _wait_dma(pack_hbm.at[dst], pack_sems.at[dst], send_counts_smem[dst])

    for dst in range(num_devices):
        count = send_counts_smem[dst]

        @pl.when(count != 0)
        def _copy_to_dst(dst=dst, count=count):
            pltpu.make_async_remote_copy(
                src_ref=pack_hbm.at[dst, pl.ds(0, count)],
                dst_ref=recv_hbm.at[my_id, pl.ds(0, count)],
                send_sem=send_sems.at[dst],
                recv_sem=recv_sems.at[my_id],
                device_id=_mesh_device_id(jnp.int32(dst)),
                device_id_type=pltpu.DeviceIdType.MESH,
            ).start()

    for src in range(num_devices):
        recv_count = _recv_count_from_src(
            jnp.int32(src),
            my_id=my_id,
            local_num_tokens=local_num_tokens,
            top_k=top_k,
            fanout=fanout,
            num_devices=num_devices,
        )
        _wait_dma(recv_hbm.at[src], recv_sems.at[src], recv_count)

    for dst in range(num_devices):
        _wait_dma(pack_hbm.at[dst], send_sems.at[dst], send_counts_smem[dst])

    _sync_barrier(barrier_sem, num_devices=num_devices, tp_size=tp_size)

    if not remote_only:
        for src in range(num_devices):
            recv_count = _recv_count_from_src(
                jnp.int32(src),
                my_id=my_id,
                local_num_tokens=local_num_tokens,
                top_k=top_k,
                fanout=fanout,
                num_devices=num_devices,
            )

            def _demux_body(i, _):
                pltpu.make_async_copy(
                    src_ref=recv_hbm.at[src, pl.ds(i, 1)],
                    dst_ref=demux_hbm.at[src, pl.ds(i, 1)],
                    sem=demux_sems.at[src],
                ).start()
                return None

            lax.fori_loop(0, recv_count, _demux_body, None, unroll=False)
            _wait_dma(demux_hbm.at[src], demux_sems.at[src], recv_count)

    _store_zero(out_hbm, out_vmem, out_sem)


def _make_kernel(
    *,
    mode: str,
    mesh: jax.sharding.Mesh,
    global_tokens: int,
    hidden_size: int,
    top_k: int,
    fanout: int,
    dtype: jnp.dtype,
):
    ep_size = mesh.shape["tensor"] * mesh.shape["data"]
    if global_tokens % ep_size:
        raise ValueError(f"Expected {global_tokens=} divisible by {ep_size=}.")
    local_tokens = global_tokens // ep_size
    t_packing = _dtype_packing(dtype)
    if hidden_size % t_packing:
        raise ValueError(f"Expected {hidden_size=} aligned to {t_packing=}.")
    hidden_per_pack = hidden_size // t_packing
    max_copies_per_peer = local_tokens * top_k
    hbm_shape = (ep_size, max_copies_per_peer, t_packing, hidden_per_pack)
    hbm_block_spec = pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)
    out_shape = jax.ShapeDtypeStruct((128,), jnp.float32)

    if mode == "direct":
        pallas_fn = pl.pallas_call(
            functools.partial(
                _direct_scatter_kernel,
                top_k=top_k,
                fanout=fanout,
                dp_axis_name="data",
                tp_axis_name="tensor",
            ),
            out_shape=out_shape,
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                in_specs=[
                    hbm_block_spec,
                    hbm_block_spec,
                ],
                out_specs=hbm_block_spec,
                scratch_shapes=[
                    pltpu.SMEM((ep_size,), jnp.int32),
                    pltpu.SemaphoreType.DMA((ep_size,)),
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
                vmem_limit_bytes=96 * 1024 * 1024,
            ),
            name=f"{MARKER}_stage2_scatter_direct_fanout{fanout}",
        )
    elif mode in ("staged", "staged_remote_only"):
        pallas_fn = pl.pallas_call(
            functools.partial(
                _staged_scatter_kernel,
                top_k=top_k,
                fanout=fanout,
                dp_axis_name="data",
                tp_axis_name="tensor",
                remote_only=(mode == "staged_remote_only"),
            ),
            out_shape=out_shape,
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                in_specs=[
                    hbm_block_spec,
                    hbm_block_spec,
                    hbm_block_spec,
                    hbm_block_spec,
                ],
                out_specs=hbm_block_spec,
                scratch_shapes=[
                    pltpu.SMEM((ep_size,), jnp.int32),
                    pltpu.SemaphoreType.DMA((ep_size,)),
                    pltpu.SemaphoreType.DMA((ep_size,)),
                    pltpu.SemaphoreType.DMA((ep_size,)),
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
                vmem_limit_bytes=96 * 1024 * 1024,
            ),
            name=f"{MARKER}_stage2_scatter_{mode}_fanout{fanout}",
        )
    else:
        raise ValueError(f"Unsupported {mode=}.")

    @jax.jit
    @jax.shard_map(
        mesh=mesh,
        in_specs=(P("tensor", None, None),),
        out_specs=P(None),
        check_vma=False,
    )
    def run(tokens):
        if mode == "direct":
            direct = pl.empty(hbm_shape, dtype)
            return pallas_fn(
                pltpu.with_memory_space_constraint(tokens, pltpu.HBM),
                pltpu.with_memory_space_constraint(direct, pltpu.HBM),
            )

        pack = pl.empty(hbm_shape, dtype)
        recv = pl.empty(hbm_shape, dtype)
        demux = pl.empty(hbm_shape, dtype)
        return pallas_fn(
            pltpu.with_memory_space_constraint(tokens, pltpu.HBM),
            pltpu.with_memory_space_constraint(pack, pltpu.HBM),
            pltpu.with_memory_space_constraint(recv, pltpu.HBM),
            pltpu.with_memory_space_constraint(demux, pltpu.HBM),
        )

    return run


def _make_tokens(
    *, mesh: jax.sharding.Mesh, global_tokens: int, hidden_size: int, dtype: jnp.dtype
):
    t_packing = _dtype_packing(dtype)
    hidden_per_pack = hidden_size // t_packing
    sharding = NamedSharding(mesh, P("tensor", None, None))
    return jax.jit(
        lambda: jnp.ones((global_tokens, t_packing, hidden_per_pack), dtype=dtype),
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
    parser = argparse.ArgumentParser(description="Benchmark Stage2 scatter staging candidates.")
    parser.add_argument("--ep-size", type=int, required=True)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--num-tokens", nargs="+", type=int, default=[512, 8192])
    parser.add_argument("--hidden-size", type=int, default=8192)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument(
        "--fanout",
        nargs="+",
        type=int,
        default=None,
        help="Remote device fanouts to benchmark. Default: min(top_k, ep_size - 1).",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["direct", "staged_remote_only", "staged"],
        default=["direct", "staged_remote_only", "staged"],
    )
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--warmup-iters", type=int, default=1)
    parser.add_argument("--trace-root", default="/tmp/sglang_stage2_scatter_staging_trace")
    parser.add_argument("--out-json", default=None)
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
    fanouts = args.fanout or [min(args.top_k, args.ep_size - 1)]
    rows = []

    print(
        "stage2 scatter staging benchmark: "
        f"process={jax.process_index()}/{jax.process_count()}, "
        f"local_devices={len(jax.local_devices())}, global_devices={len(jax.devices())}, "
        f"ep_size={args.ep_size}, hidden_size={args.hidden_size}, top_k={args.top_k}"
    )

    for tokens in args.num_tokens:
        token_data = _make_tokens(
            mesh=mesh,
            global_tokens=tokens,
            hidden_size=args.hidden_size,
            dtype=dtype,
        )
        for fanout in fanouts:
            fanout = max(1, min(int(fanout), args.ep_size - 1))
            for mode in args.modes:
                run = _make_kernel(
                    mode=mode,
                    mesh=mesh,
                    global_tokens=tokens,
                    hidden_size=args.hidden_size,
                    top_k=args.top_k,
                    fanout=fanout,
                    dtype=dtype,
                )

                def _data():
                    return (token_data,)

                samples = multiple_iteration_timeit_from_trace(
                    run,
                    _data,
                    task=f"stage2_scatter_{mode}_ep{args.ep_size}_tok{tokens}_fanout{fanout}",
                    tries=args.iters,
                    warmup=args.warmup_iters,
                    trace_root=args.trace_root,
                )
                stats = _summary(samples)
                bytes_remote = tokens * args.top_k * args.hidden_size * jnp.dtype(dtype).itemsize
                row = {
                    "ep_size": args.ep_size,
                    "tokens": tokens,
                    "hidden_size": args.hidden_size,
                    "top_k": args.top_k,
                    "fanout": fanout,
                    "mode": mode,
                    "samples_ms": samples,
                    "remote_bytes": int(bytes_remote),
                    "effective_remote_gbs_mean": (
                        float(bytes_remote) / (stats["mean_ms"] / 1000.0) / 1e9
                        if stats["mean_ms"] > 0
                        else math.nan
                    ),
                    **stats,
                }
                rows.append(row)
                print(
                    "RESULT "
                    + json.dumps(
                        row,
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                )

    if args.out_json and jax.process_index() == 0:
        path = Path(args.out_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
