"""Stage2 scatter pack/permute microbenchmark for fused EP-MoE.

This benchmark isolates the Stage2 scatter design space:

  direct
      token-wise HBM -> remote HBM scatter, matching the current launch pattern.

  hbm_pack_serial
      token-wise HBM -> HBM pack by destination, then one large remote DMA per
      destination. Pack and remote DMA are serialized.

  hbm_pack_overlap
      token-wise HBM -> HBM pack by destination, then start the large remote DMA
      for that destination immediately. The remote DMA for dst=N can overlap
      with HBM packing for dst=N+1.

  hbm_pack_demux
      hbm_pack_serial plus a receiver-side token-wise local demux copy. This is
      a pessimistic proxy for designs that pack by device and still need to
      restore per-expert layout after receive.

  vmem_pack
      token-wise HBM -> VMEM pack by destination, then one large remote DMA from
      VMEM to remote HBM. A single VMEM pack buffer is reused, so pack and remote
      DMA are serialized.

  vmem_pack_overlap
      Same as vmem_pack, but with two VMEM pack buffers so remote DMA for
      dst=N can overlap with VMEM packing for dst=N+1.

  direct_expert
      token-wise HBM -> remote HBM scatter, but preserving the real fused MoE
      receiver layout: remote buffers are split by local expert.

  hbm_expert_pack
      HBM pack by (destination, local expert), then one remote DMA per
      destination/expert pair. This avoids receiver demux.

  hbm_expert_pack_early
      HBM pack by (destination, local expert), but start the remote DMA for a
      pair as soon as that pair's final packed token is written. This keeps a
      single token scan and tries to overlap later pack work with earlier remote
      sends.

  vmem_expert_pack
      Streaming per-expert VMEM pack. One small VMEM buffer is reused for each
      destination/expert pair, then sent with VMEM -> remote HBM DMA.

The routing pattern is deterministic and intentionally controlled by fanout, so
the same script can run on EP=8/32/64 without model weights.
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

MODES = (
    "direct",
    "direct_run_merge",
    "hbm_pack_serial",
    "hbm_pack_overlap",
    "hbm_pack_demux",
    "vmem_pack",
    "vmem_pack_overlap",
    "direct_expert",
    "hbm_expert_pack",
    "hbm_expert_pack_early",
    "vmem_expert_pack",
)


def _dtype_packing(dtype: jnp.dtype) -> int:
    bits = jnp.dtype(dtype).itemsize * 8
    if 32 % bits != 0:
        raise ValueError(f"Unsupported dtype packing for {dtype=} ({bits=} bits).")
    return 32 // bits


def _dest_for(src_rank, t_id, k_id, *, fanout: int, num_devices: int):
    # Keep the benchmark focused on remote scatter. fanout is capped by the
    # caller to [1, num_devices - 1].
    lane = (t_id + k_id) % jnp.int32(fanout)
    return (src_rank + jnp.int32(1) + lane) % jnp.int32(num_devices)


def _expert_for(t_id, k_id, *, local_num_experts: int):
    return (t_id + k_id) % jnp.int32(local_num_experts)


def _compute_send_counts(
    send_counts,
    *,
    my_id,
    local_num_tokens: int,
    top_k: int,
    fanout: int,
    num_devices: int,
):
    for dev in range(num_devices):
        send_counts[dev] = jnp.int32(0)

    def _token_body(t_id, _):
        for k_id in range(top_k):
            dst = _dest_for(
                my_id,
                t_id,
                jnp.int32(k_id),
                fanout=fanout,
                num_devices=num_devices,
            )
            send_counts[dst] = send_counts[dst] + jnp.int32(1)
        return None

    lax.fori_loop(0, local_num_tokens, _token_body, None, unroll=False)


def _recv_count_from_src(
    src_rank,
    *,
    my_id,
    local_num_tokens: int,
    top_k: int,
    fanout: int,
    num_devices: int,
):
    def _token_body(t_id, count):
        for k_id in range(top_k):
            dst = _dest_for(
                src_rank,
                t_id,
                jnp.int32(k_id),
                fanout=fanout,
                num_devices=num_devices,
            )
            count += lax.select(dst == my_id, jnp.int32(1), jnp.int32(0))
        return count

    return lax.fori_loop(0, local_num_tokens, _token_body, jnp.int32(0), unroll=False)


def _recv_count_expert_from_src(
    src_rank,
    local_e_id,
    *,
    my_id,
    local_num_tokens: int,
    local_num_experts: int,
    top_k: int,
    fanout: int,
    num_devices: int,
):
    def _token_body(t_id, count):
        for k_id in range(top_k):
            dst = _dest_for(
                src_rank,
                t_id,
                jnp.int32(k_id),
                fanout=fanout,
                num_devices=num_devices,
            )
            expert = _expert_for(
                t_id,
                jnp.int32(k_id),
                local_num_experts=local_num_experts,
            )
            count += lax.select(
                (dst == my_id) & (expert == local_e_id),
                jnp.int32(1),
                jnp.int32(0),
            )
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
    recv_hbm,
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

    for dev in range(num_devices):
        send_counts_smem[dev] = jnp.int32(0)

    def _token_body(t_id, _):
        for k_id in range(top_k):
            dst = _dest_for(
                my_id,
                t_id,
                jnp.int32(k_id),
                fanout=fanout,
                num_devices=num_devices,
            )
            offset = send_counts_smem[dst]
            send_counts_smem[dst] = offset + jnp.int32(1)
            pltpu.make_async_remote_copy(
                src_ref=tokens_hbm.at[pl.ds(t_id, 1)],
                dst_ref=recv_hbm.at[my_id, pl.ds(offset, 1)],
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
        _wait_dma(recv_hbm.at[src], recv_sems.at[src], recv_count)

    for dst in range(num_devices):
        _wait_dma(recv_hbm.at[0], send_sems.at[dst], send_counts_smem[dst])

    _sync_barrier(barrier_sem, num_devices=num_devices, tp_size=tp_size)
    _store_zero(out_hbm, out_vmem, out_sem)


def _direct_run_merge_scatter_kernel(
    tokens_hbm,
    recv_hbm,
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
    """Merge naturally contiguous source-token runs without a pack buffer.

    This is a timing-only upper-bound experiment: it scans by dst/k and writes a
    k-major receiver order, so it does not preserve the exact direct scatter
    ordering required by the fused MoE gather path.
    """
    dp_rank = lax.axis_index(dp_axis_name)
    tp_rank = lax.axis_index(tp_axis_name)
    tp_size = lax.axis_size(tp_axis_name)
    dp_size = lax.axis_size(dp_axis_name)
    my_id = dp_rank * tp_size + tp_rank
    num_devices = dp_size * tp_size
    local_num_tokens = tokens_hbm.shape[0]

    def _mesh_device_id(ep_rank):
        return (ep_rank // tp_size, ep_rank % tp_size)

    for dev in range(num_devices):
        send_counts_smem[dev] = jnp.int32(0)

    for dst in range(num_devices):
        for k_id in range(top_k):

            def _token_while_body(t_id):
                dst_i = jnp.int32(dst)
                k_i = jnp.int32(k_id)
                is_match = (
                    _dest_for(
                        my_id,
                        t_id,
                        k_i,
                        fanout=fanout,
                        num_devices=num_devices,
                    )
                    == dst_i
                )

                def _run_cond(run_len):
                    next_t = t_id + run_len
                    return (
                        is_match
                        & (next_t < local_num_tokens)
                        & (
                            _dest_for(
                                my_id,
                                next_t,
                                k_i,
                                fanout=fanout,
                                num_devices=num_devices,
                            )
                            == dst_i
                        )
                    )

                run_len = lax.while_loop(
                    _run_cond,
                    lambda run_len: run_len + jnp.int32(1),
                    jnp.int32(1),
                )

                @pl.when(is_match)
                def _copy_run():
                    offset = send_counts_smem[dst]
                    send_counts_smem[dst] = offset + run_len
                    pltpu.make_async_remote_copy(
                        src_ref=tokens_hbm.at[pl.ds(t_id, run_len)],
                        dst_ref=recv_hbm.at[my_id, pl.ds(offset, run_len)],
                        send_sem=send_sems.at[dst],
                        recv_sem=recv_sems.at[my_id],
                        device_id=_mesh_device_id(jnp.int32(dst)),
                        device_id_type=pltpu.DeviceIdType.MESH,
                    ).start()

                return lax.select(is_match, t_id + run_len, t_id + jnp.int32(1))

            lax.while_loop(
                lambda t_id: t_id < local_num_tokens,
                _token_while_body,
                jnp.int32(0),
            )

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
        _wait_dma(recv_hbm.at[0], send_sems.at[dst], send_counts_smem[dst])

    _sync_barrier(barrier_sem, num_devices=num_devices, tp_size=tp_size)
    _store_zero(out_hbm, out_vmem, out_sem)


def _pack_tokens_to_hbm_by_dst(
    tokens_hbm,
    pack_hbm,
    send_counts_smem,
    pack_offsets_smem,
    pack_sems,
    *,
    my_id,
    local_num_tokens: int,
    top_k: int,
    fanout: int,
    num_devices: int,
):
    for dev in range(num_devices):
        pack_offsets_smem[dev] = jnp.int32(0)

    def _token_body(t_id, _):
        for k_id in range(top_k):
            dst = _dest_for(
                my_id,
                t_id,
                jnp.int32(k_id),
                fanout=fanout,
                num_devices=num_devices,
            )
            offset = pack_offsets_smem[dst]
            pack_offsets_smem[dst] = offset + jnp.int32(1)
            pltpu.make_async_copy(
                src_ref=tokens_hbm.at[pl.ds(t_id, 1)],
                dst_ref=pack_hbm.at[dst, pl.ds(offset, 1)],
                sem=pack_sems.at[dst],
            ).start()
        return None

    lax.fori_loop(0, local_num_tokens, _token_body, None, unroll=False)

    for dst in range(num_devices):
        _wait_dma(pack_hbm.at[dst], pack_sems.at[dst], send_counts_smem[dst])


def _start_hbm_remote_by_dst(
    pack_hbm,
    recv_hbm,
    send_counts_smem,
    send_sems,
    recv_sems,
    *,
    my_id,
    num_devices: int,
    tp_size: int,
):
    def _mesh_device_id(ep_rank):
        return (ep_rank // tp_size, ep_rank % tp_size)

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


def _wait_remote_by_dst(
    pack_hbm,
    recv_hbm,
    send_counts_smem,
    send_sems,
    recv_sems,
    *,
    my_id,
    local_num_tokens: int,
    top_k: int,
    fanout: int,
    num_devices: int,
):
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


def _hbm_pack_scatter_kernel(
    tokens_hbm,
    pack_hbm,
    recv_hbm,
    demux_hbm,
    out_hbm,
    send_counts_smem,
    pack_offsets_smem,
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
    overlap_remote: bool,
    demux: bool,
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

    if overlap_remote:
        for dst in range(num_devices):
            pack_offsets_smem[dst] = jnp.int32(0)

        for dst in range(num_devices):

            def _pack_dst_token_body(t_id, _, dst=dst):
                for k_id in range(top_k):
                    route_dst = _dest_for(
                        my_id,
                        t_id,
                        jnp.int32(k_id),
                        fanout=fanout,
                        num_devices=num_devices,
                    )
                    should_pack = route_dst == jnp.int32(dst)

                    @pl.when(should_pack)
                    def _copy_one(dst=dst, t_id=t_id):
                        offset = pack_offsets_smem[dst]
                        pack_offsets_smem[dst] = offset + jnp.int32(1)
                        pltpu.make_async_copy(
                            src_ref=tokens_hbm.at[pl.ds(t_id, 1)],
                            dst_ref=pack_hbm.at[dst, pl.ds(offset, 1)],
                            sem=pack_sems.at[dst],
                        ).start()

                return None

            lax.fori_loop(0, local_num_tokens, _pack_dst_token_body, None, unroll=False)
            count = send_counts_smem[dst]
            _wait_dma(pack_hbm.at[dst], pack_sems.at[dst], count)

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

    else:
        _pack_tokens_to_hbm_by_dst(
            tokens_hbm,
            pack_hbm,
            send_counts_smem,
            pack_offsets_smem,
            pack_sems,
            my_id=my_id,
            local_num_tokens=local_num_tokens,
            top_k=top_k,
            fanout=fanout,
            num_devices=num_devices,
        )
        _start_hbm_remote_by_dst(
            pack_hbm,
            recv_hbm,
            send_counts_smem,
            send_sems,
            recv_sems,
            my_id=my_id,
            num_devices=num_devices,
            tp_size=tp_size,
        )

    _wait_remote_by_dst(
        pack_hbm,
        recv_hbm,
        send_counts_smem,
        send_sems,
        recv_sems,
        my_id=my_id,
        local_num_tokens=local_num_tokens,
        top_k=top_k,
        fanout=fanout,
        num_devices=num_devices,
    )

    _sync_barrier(barrier_sem, num_devices=num_devices, tp_size=tp_size)

    if demux:
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


def _pack_one_dst_to_vmem(
    tokens_hbm,
    pack_vmem,
    pack_count_smem,
    pack_sem,
    *,
    dst: int,
    my_id,
    local_num_tokens: int,
    top_k: int,
    fanout: int,
    num_devices: int,
):
    pack_count_smem[0] = jnp.int32(0)

    def _token_body(t_id, _):
        for k_id in range(top_k):
            route_dst = _dest_for(
                my_id,
                t_id,
                jnp.int32(k_id),
                fanout=fanout,
                num_devices=num_devices,
            )
            should_pack = route_dst == jnp.int32(dst)

            @pl.when(should_pack)
            def _copy_one(t_id=t_id):
                offset = pack_count_smem[0]
                pack_count_smem[0] = offset + jnp.int32(1)
                pltpu.make_async_copy(
                    src_ref=tokens_hbm.at[pl.ds(t_id, 1)],
                    dst_ref=pack_vmem.at[pl.ds(offset, 1)],
                    sem=pack_sem,
                ).start()

        return None

    lax.fori_loop(0, local_num_tokens, _token_body, None, unroll=False)


def _vmem_pack_scatter_kernel(
    tokens_hbm,
    recv_hbm,
    out_hbm,
    send_counts_smem,
    pack_count_smem,
    pack_vmem_x2,
    pack_sems,
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
    overlap_remote: bool,
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

    if overlap_remote:
        for dst in range(num_devices):
            buf = dst % 2

            if dst >= 2:
                prev_dst = dst - 2
                prev_count = send_counts_smem[prev_dst]
                _wait_dma(pack_vmem_x2.at[buf], send_sems.at[prev_dst], prev_count)

            _pack_one_dst_to_vmem(
                tokens_hbm,
                pack_vmem_x2.at[buf],
                pack_count_smem,
                pack_sems.at[buf],
                dst=dst,
                my_id=my_id,
                local_num_tokens=local_num_tokens,
                top_k=top_k,
                fanout=fanout,
                num_devices=num_devices,
            )
            count = send_counts_smem[dst]
            _wait_dma(pack_vmem_x2.at[buf], pack_sems.at[buf], count)

            @pl.when(count != 0)
            def _copy_to_dst(dst=dst, count=count, buf=buf):
                pltpu.make_async_remote_copy(
                    src_ref=pack_vmem_x2.at[buf, pl.ds(0, count)],
                    dst_ref=recv_hbm.at[my_id, pl.ds(0, count)],
                    send_sem=send_sems.at[dst],
                    recv_sem=recv_sems.at[my_id],
                    device_id=_mesh_device_id(jnp.int32(dst)),
                    device_id_type=pltpu.DeviceIdType.MESH,
                ).start()

        if num_devices >= 2:
            last_2 = num_devices - 2
            _wait_dma(pack_vmem_x2.at[last_2 % 2], send_sems.at[last_2], send_counts_smem[last_2])
        if num_devices >= 1:
            last_1 = num_devices - 1
            _wait_dma(pack_vmem_x2.at[last_1 % 2], send_sems.at[last_1], send_counts_smem[last_1])
    else:
        for dst in range(num_devices):
            _pack_one_dst_to_vmem(
                tokens_hbm,
                pack_vmem_x2.at[0],
                pack_count_smem,
                pack_sems.at[0],
                dst=dst,
                my_id=my_id,
                local_num_tokens=local_num_tokens,
                top_k=top_k,
                fanout=fanout,
                num_devices=num_devices,
            )
            count = send_counts_smem[dst]
            _wait_dma(pack_vmem_x2.at[0], pack_sems.at[0], count)

            @pl.when(count != 0)
            def _copy_to_dst(dst=dst, count=count):
                pltpu.make_async_remote_copy(
                    src_ref=pack_vmem_x2.at[0, pl.ds(0, count)],
                    dst_ref=recv_hbm.at[my_id, pl.ds(0, count)],
                    send_sem=send_sems.at[dst],
                    recv_sem=recv_sems.at[my_id],
                    device_id=_mesh_device_id(jnp.int32(dst)),
                    device_id_type=pltpu.DeviceIdType.MESH,
                ).start()
                _wait_dma(pack_vmem_x2.at[0], send_sems.at[dst], count)

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

    _sync_barrier(barrier_sem, num_devices=num_devices, tp_size=tp_size)
    _store_zero(out_hbm, out_vmem, out_sem)


def _wait_expert_remote_by_src(
    recv_hbm,
    send_counts_smem,
    send_sems,
    recv_sems,
    *,
    my_id,
    local_num_tokens: int,
    local_num_experts: int,
    top_k: int,
    fanout: int,
    num_devices: int,
):
    total_pairs = num_devices * local_num_experts

    def _recv_pair_body(pair_id, _):
        local_e_id = pair_id // jnp.int32(num_devices)
        src = pair_id - local_e_id * jnp.int32(num_devices)
        recv_count = _recv_count_expert_from_src(
            src,
            local_e_id,
            my_id=my_id,
            local_num_tokens=local_num_tokens,
            local_num_experts=local_num_experts,
            top_k=top_k,
            fanout=fanout,
            num_devices=num_devices,
        )
        _wait_dma(recv_hbm.at[local_e_id, src], recv_sems.at[local_e_id, src], recv_count)
        return None

    lax.fori_loop(0, jnp.int32(total_pairs), _recv_pair_body, None, unroll=False)

    def _send_pair_body(pair_id, _):
        dst = pair_id // jnp.int32(local_num_experts)
        local_e_id = pair_id - dst * jnp.int32(local_num_experts)
        send_count = send_counts_smem[dst, local_e_id]
        _wait_dma(recv_hbm.at[0, 0], send_sems.at[dst, local_e_id], send_count)
        return None

    lax.fori_loop(0, jnp.int32(total_pairs), _send_pair_body, None, unroll=False)


def _direct_expert_scatter_kernel(
    tokens_hbm,
    recv_hbm,
    out_hbm,
    send_counts_smem,
    send_sems,
    recv_sems,
    barrier_sem,
    out_vmem,
    out_sem,
    *,
    local_num_experts: int,
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

    for dst in range(num_devices):
        for local_e_id in range(local_num_experts):
            send_counts_smem[dst, local_e_id] = jnp.int32(0)

    def _token_body(t_id, _):
        for k_id in range(top_k):
            dst = _dest_for(
                my_id,
                t_id,
                jnp.int32(k_id),
                fanout=fanout,
                num_devices=num_devices,
            )
            local_e_id = _expert_for(
                t_id,
                jnp.int32(k_id),
                local_num_experts=local_num_experts,
            )
            offset = send_counts_smem[dst, local_e_id]
            send_counts_smem[dst, local_e_id] = offset + jnp.int32(1)
            pltpu.make_async_remote_copy(
                src_ref=tokens_hbm.at[pl.ds(t_id, 1)],
                dst_ref=recv_hbm.at[local_e_id, my_id, pl.ds(offset, 1)],
                send_sem=send_sems.at[dst, local_e_id],
                recv_sem=recv_sems.at[local_e_id, my_id],
                device_id=_mesh_device_id(dst),
                device_id_type=pltpu.DeviceIdType.MESH,
            ).start()
        return None

    lax.fori_loop(0, local_num_tokens, _token_body, None, unroll=False)

    _wait_expert_remote_by_src(
        recv_hbm,
        send_counts_smem,
        send_sems,
        recv_sems,
        my_id=my_id,
        local_num_tokens=local_num_tokens,
        local_num_experts=local_num_experts,
        top_k=top_k,
        fanout=fanout,
        num_devices=num_devices,
    )
    _sync_barrier(barrier_sem, num_devices=num_devices, tp_size=tp_size)
    _store_zero(out_hbm, out_vmem, out_sem)


def _hbm_expert_pack_scatter_kernel(
    tokens_hbm,
    pack_hbm,
    recv_hbm,
    out_hbm,
    send_counts_smem,
    pack_offsets_smem,
    pack_sems,
    send_sems,
    recv_sems,
    barrier_sem,
    out_vmem,
    out_sem,
    *,
    local_num_experts: int,
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

    for dst in range(num_devices):
        for local_e_id in range(local_num_experts):
            send_counts_smem[dst, local_e_id] = jnp.int32(0)
            pack_offsets_smem[dst, local_e_id] = jnp.int32(0)

    def _pack_body(t_id, _):
        for k_id in range(top_k):
            dst = _dest_for(
                my_id,
                t_id,
                jnp.int32(k_id),
                fanout=fanout,
                num_devices=num_devices,
            )
            local_e_id = _expert_for(
                t_id,
                jnp.int32(k_id),
                local_num_experts=local_num_experts,
            )
            offset = pack_offsets_smem[dst, local_e_id]
            pack_offsets_smem[dst, local_e_id] = offset + jnp.int32(1)
            send_counts_smem[dst, local_e_id] = offset + jnp.int32(1)
            pltpu.make_async_copy(
                src_ref=tokens_hbm.at[pl.ds(t_id, 1)],
                dst_ref=pack_hbm.at[dst, local_e_id, pl.ds(offset, 1)],
                sem=pack_sems.at[dst, local_e_id],
            ).start()
        return None

    lax.fori_loop(0, local_num_tokens, _pack_body, None, unroll=False)

    total_pairs = num_devices * local_num_experts

    def _remote_pair_body(pair_id, _):
        dst = pair_id // jnp.int32(local_num_experts)
        local_e_id = pair_id - dst * jnp.int32(local_num_experts)
        count = send_counts_smem[dst, local_e_id]
        _wait_dma(pack_hbm.at[dst, local_e_id], pack_sems.at[dst, local_e_id], count)

        @pl.when(count != 0)
        def _copy_to_dst(dst=dst, local_e_id=local_e_id, count=count):
            pltpu.make_async_remote_copy(
                src_ref=pack_hbm.at[dst, local_e_id, pl.ds(0, count)],
                dst_ref=recv_hbm.at[local_e_id, my_id, pl.ds(0, count)],
                send_sem=send_sems.at[dst, local_e_id],
                recv_sem=recv_sems.at[local_e_id, my_id],
                device_id=_mesh_device_id(dst),
                device_id_type=pltpu.DeviceIdType.MESH,
            ).start()

        return None

    lax.fori_loop(0, jnp.int32(total_pairs), _remote_pair_body, None, unroll=False)

    _wait_expert_remote_by_src(
        recv_hbm,
        send_counts_smem,
        send_sems,
        recv_sems,
        my_id=my_id,
        local_num_tokens=local_num_tokens,
        local_num_experts=local_num_experts,
        top_k=top_k,
        fanout=fanout,
        num_devices=num_devices,
    )
    _sync_barrier(barrier_sem, num_devices=num_devices, tp_size=tp_size)
    _store_zero(out_hbm, out_vmem, out_sem)


def _hbm_expert_pack_early_scatter_kernel(
    tokens_hbm,
    pack_hbm,
    recv_hbm,
    out_hbm,
    send_counts_smem,
    pack_offsets_smem,
    pack_sems,
    send_sems,
    recv_sems,
    barrier_sem,
    out_vmem,
    out_sem,
    *,
    local_num_experts: int,
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

    for dst in range(num_devices):
        for local_e_id in range(local_num_experts):
            send_counts_smem[dst, local_e_id] = jnp.int32(0)
            pack_offsets_smem[dst, local_e_id] = jnp.int32(0)

    def _count_body(t_id, _):
        for k_id in range(top_k):
            dst = _dest_for(
                my_id,
                t_id,
                jnp.int32(k_id),
                fanout=fanout,
                num_devices=num_devices,
            )
            local_e_id = _expert_for(
                t_id,
                jnp.int32(k_id),
                local_num_experts=local_num_experts,
            )
            send_counts_smem[dst, local_e_id] = send_counts_smem[dst, local_e_id] + jnp.int32(1)
        return None

    lax.fori_loop(0, local_num_tokens, _count_body, None, unroll=False)

    def _pack_body(t_id, _):
        for k_id in range(top_k):
            dst = _dest_for(
                my_id,
                t_id,
                jnp.int32(k_id),
                fanout=fanout,
                num_devices=num_devices,
            )
            local_e_id = _expert_for(
                t_id,
                jnp.int32(k_id),
                local_num_experts=local_num_experts,
            )
            offset = pack_offsets_smem[dst, local_e_id]
            next_offset = offset + jnp.int32(1)
            pack_offsets_smem[dst, local_e_id] = next_offset
            count = send_counts_smem[dst, local_e_id]
            pltpu.make_async_copy(
                src_ref=tokens_hbm.at[pl.ds(t_id, 1)],
                dst_ref=pack_hbm.at[dst, local_e_id, pl.ds(offset, 1)],
                sem=pack_sems.at[dst, local_e_id],
            ).start()

            @pl.when(next_offset == count)
            def _copy_completed_pair(dst=dst, local_e_id=local_e_id, count=count):
                _wait_dma(pack_hbm.at[dst, local_e_id], pack_sems.at[dst, local_e_id], count)
                pltpu.make_async_remote_copy(
                    src_ref=pack_hbm.at[dst, local_e_id, pl.ds(0, count)],
                    dst_ref=recv_hbm.at[local_e_id, my_id, pl.ds(0, count)],
                    send_sem=send_sems.at[dst, local_e_id],
                    recv_sem=recv_sems.at[local_e_id, my_id],
                    device_id=_mesh_device_id(dst),
                    device_id_type=pltpu.DeviceIdType.MESH,
                ).start()

        return None

    lax.fori_loop(0, local_num_tokens, _pack_body, None, unroll=False)

    _wait_expert_remote_by_src(
        recv_hbm,
        send_counts_smem,
        send_sems,
        recv_sems,
        my_id=my_id,
        local_num_tokens=local_num_tokens,
        local_num_experts=local_num_experts,
        top_k=top_k,
        fanout=fanout,
        num_devices=num_devices,
    )
    _sync_barrier(barrier_sem, num_devices=num_devices, tp_size=tp_size)
    _store_zero(out_hbm, out_vmem, out_sem)


def _vmem_expert_pack_scatter_kernel(
    tokens_hbm,
    recv_hbm,
    out_hbm,
    send_counts_smem,
    pack_count_smem,
    pack_vmem,
    pack_sem,
    send_sem,
    recv_sems,
    barrier_sem,
    out_vmem,
    out_sem,
    *,
    local_num_experts: int,
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

    for dst in range(num_devices):
        for local_e_id in range(local_num_experts):
            send_counts_smem[dst, local_e_id] = jnp.int32(0)

    total_pairs = num_devices * local_num_experts

    def _pair_body(pair_id, _):
        dst = pair_id // jnp.int32(local_num_experts)
        local_e_id = pair_id - dst * jnp.int32(local_num_experts)
        pack_count_smem[0] = jnp.int32(0)

        def _pack_token_body(t_id, _):
            for k_id in range(top_k):
                route_dst = _dest_for(
                    my_id,
                    t_id,
                    jnp.int32(k_id),
                    fanout=fanout,
                    num_devices=num_devices,
                )
                route_expert = _expert_for(
                    t_id,
                    jnp.int32(k_id),
                    local_num_experts=local_num_experts,
                )
                should_pack = (route_dst == dst) & (route_expert == local_e_id)

                @pl.when(should_pack)
                def _copy_one(t_id=t_id):
                    offset = pack_count_smem[0]
                    pack_count_smem[0] = offset + jnp.int32(1)
                    pltpu.make_async_copy(
                        src_ref=tokens_hbm.at[pl.ds(t_id, 1)],
                        dst_ref=pack_vmem.at[pl.ds(offset, 1)],
                        sem=pack_sem,
                    ).start()

            return None

        lax.fori_loop(0, local_num_tokens, _pack_token_body, None, unroll=False)

        count = pack_count_smem[0]
        send_counts_smem[dst, local_e_id] = count
        _wait_dma(pack_vmem, pack_sem, count)

        @pl.when(count != 0)
        def _copy_to_dst(dst=dst, local_e_id=local_e_id, count=count):
            pltpu.make_async_remote_copy(
                src_ref=pack_vmem.at[pl.ds(0, count)],
                dst_ref=recv_hbm.at[local_e_id, my_id, pl.ds(0, count)],
                send_sem=send_sem,
                recv_sem=recv_sems.at[local_e_id, my_id],
                device_id=_mesh_device_id(dst),
                device_id_type=pltpu.DeviceIdType.MESH,
            ).start()
            _wait_dma(pack_vmem, send_sem, count)

        return None

    lax.fori_loop(0, jnp.int32(total_pairs), _pair_body, None, unroll=False)

    def _recv_pair_body(pair_id, _):
        local_e_id = pair_id // jnp.int32(num_devices)
        src = pair_id - local_e_id * jnp.int32(num_devices)
        recv_count = _recv_count_expert_from_src(
            src,
            local_e_id,
            my_id=my_id,
            local_num_tokens=local_num_tokens,
            local_num_experts=local_num_experts,
            top_k=top_k,
            fanout=fanout,
            num_devices=num_devices,
        )
        _wait_dma(recv_hbm.at[local_e_id, src], recv_sems.at[local_e_id, src], recv_count)
        return None

    lax.fori_loop(0, jnp.int32(total_pairs), _recv_pair_body, None, unroll=False)
    _sync_barrier(barrier_sem, num_devices=num_devices, tp_size=tp_size)
    _store_zero(out_hbm, out_vmem, out_sem)


def _make_kernel(
    *,
    mode: str,
    mesh: jax.sharding.Mesh,
    global_tokens: int,
    hidden_size: int,
    num_experts: int,
    top_k: int,
    fanout: int,
    dtype: jnp.dtype,
    vmem_limit_bytes: int,
):
    ep_size = mesh.shape["tensor"] * mesh.shape["data"]
    if global_tokens % ep_size:
        raise ValueError(f"Expected {global_tokens=} divisible by {ep_size=}.")
    if num_experts % ep_size:
        raise ValueError(f"Expected {num_experts=} divisible by {ep_size=}.")
    local_tokens = global_tokens // ep_size
    local_num_experts = num_experts // ep_size
    t_packing = _dtype_packing(dtype)
    if hidden_size % t_packing:
        raise ValueError(f"Expected {hidden_size=} aligned to {t_packing=}.")
    hidden_per_pack = hidden_size // t_packing
    max_copies_per_peer = local_tokens * top_k
    expert_capacity = math.ceil(max_copies_per_peer / local_num_experts) + top_k
    hbm_shape = (ep_size, max_copies_per_peer, t_packing, hidden_per_pack)
    expert_recv_shape = (
        local_num_experts,
        ep_size,
        expert_capacity,
        t_packing,
        hidden_per_pack,
    )
    expert_pack_shape = (
        ep_size,
        local_num_experts,
        expert_capacity,
        t_packing,
        hidden_per_pack,
    )
    expert_vmem_pack_shape = (expert_capacity, t_packing, hidden_per_pack)
    vmem_pack_shape = (2, max_copies_per_peer, t_packing, hidden_per_pack)
    hbm_block_spec = pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)
    out_shape = jax.ShapeDtypeStruct((128,), jnp.float32)

    if mode in ("direct", "direct_run_merge"):
        kernel = _direct_scatter_kernel if mode == "direct" else _direct_run_merge_scatter_kernel
        pallas_fn = pl.pallas_call(
            functools.partial(
                kernel,
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
                vmem_limit_bytes=vmem_limit_bytes,
            ),
            name=f"{MARKER}_stage2_scatter_{mode}_fanout{fanout}",
        )
    elif mode in ("hbm_pack_serial", "hbm_pack_overlap", "hbm_pack_demux"):
        pallas_fn = pl.pallas_call(
            functools.partial(
                _hbm_pack_scatter_kernel,
                top_k=top_k,
                fanout=fanout,
                dp_axis_name="data",
                tp_axis_name="tensor",
                overlap_remote=(mode == "hbm_pack_overlap"),
                demux=(mode == "hbm_pack_demux"),
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
                vmem_limit_bytes=vmem_limit_bytes,
            ),
            name=f"{MARKER}_stage2_scatter_{mode}_fanout{fanout}",
        )
    elif mode in ("vmem_pack", "vmem_pack_overlap"):
        pallas_fn = pl.pallas_call(
            functools.partial(
                _vmem_pack_scatter_kernel,
                top_k=top_k,
                fanout=fanout,
                dp_axis_name="data",
                tp_axis_name="tensor",
                overlap_remote=(mode == "vmem_pack_overlap"),
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
                    pltpu.SMEM((1,), jnp.int32),
                    pltpu.VMEM(vmem_pack_shape, dtype),
                    pltpu.SemaphoreType.DMA((2,)),
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
                vmem_limit_bytes=vmem_limit_bytes,
            ),
            name=f"{MARKER}_stage2_scatter_{mode}_fanout{fanout}",
        )
    elif mode == "direct_expert":
        pallas_fn = pl.pallas_call(
            functools.partial(
                _direct_expert_scatter_kernel,
                local_num_experts=local_num_experts,
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
                    pltpu.SMEM((ep_size, local_num_experts), jnp.int32),
                    pltpu.SemaphoreType.DMA((ep_size, local_num_experts)),
                    pltpu.SemaphoreType.DMA((local_num_experts, ep_size)),
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
            name=f"{MARKER}_stage2_scatter_{mode}_fanout{fanout}",
        )
    elif mode == "hbm_expert_pack":
        pallas_fn = pl.pallas_call(
            functools.partial(
                _hbm_expert_pack_scatter_kernel,
                local_num_experts=local_num_experts,
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
                    hbm_block_spec,
                ],
                out_specs=hbm_block_spec,
                scratch_shapes=[
                    pltpu.SMEM((ep_size, local_num_experts), jnp.int32),
                    pltpu.SMEM((ep_size, local_num_experts), jnp.int32),
                    pltpu.SemaphoreType.DMA((ep_size, local_num_experts)),
                    pltpu.SemaphoreType.DMA((ep_size, local_num_experts)),
                    pltpu.SemaphoreType.DMA((local_num_experts, ep_size)),
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
            name=f"{MARKER}_stage2_scatter_{mode}_fanout{fanout}",
        )
    elif mode == "hbm_expert_pack_early":
        pallas_fn = pl.pallas_call(
            functools.partial(
                _hbm_expert_pack_early_scatter_kernel,
                local_num_experts=local_num_experts,
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
                    hbm_block_spec,
                ],
                out_specs=hbm_block_spec,
                scratch_shapes=[
                    pltpu.SMEM((ep_size, local_num_experts), jnp.int32),
                    pltpu.SMEM((ep_size, local_num_experts), jnp.int32),
                    pltpu.SemaphoreType.DMA((ep_size, local_num_experts)),
                    pltpu.SemaphoreType.DMA((ep_size, local_num_experts)),
                    pltpu.SemaphoreType.DMA((local_num_experts, ep_size)),
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
            name=f"{MARKER}_stage2_scatter_{mode}_fanout{fanout}",
        )
    elif mode == "vmem_expert_pack":
        pallas_fn = pl.pallas_call(
            functools.partial(
                _vmem_expert_pack_scatter_kernel,
                local_num_experts=local_num_experts,
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
                    pltpu.SMEM((ep_size, local_num_experts), jnp.int32),
                    pltpu.SMEM((1,), jnp.int32),
                    pltpu.VMEM(expert_vmem_pack_shape, dtype),
                    pltpu.SemaphoreType.DMA,
                    pltpu.SemaphoreType.DMA,
                    pltpu.SemaphoreType.DMA((local_num_experts, ep_size)),
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
        if mode in ("direct", "direct_run_merge"):
            recv = pl.empty(hbm_shape, dtype)
            return pallas_fn(
                pltpu.with_memory_space_constraint(tokens, pltpu.HBM),
                pltpu.with_memory_space_constraint(recv, pltpu.HBM),
            )
        if mode.startswith("hbm_pack"):
            pack = pl.empty(hbm_shape, dtype)
            recv = pl.empty(hbm_shape, dtype)
            demux = pl.empty(hbm_shape, dtype)
            return pallas_fn(
                pltpu.with_memory_space_constraint(tokens, pltpu.HBM),
                pltpu.with_memory_space_constraint(pack, pltpu.HBM),
                pltpu.with_memory_space_constraint(recv, pltpu.HBM),
                pltpu.with_memory_space_constraint(demux, pltpu.HBM),
            )
        if mode == "direct_expert":
            recv = pl.empty(expert_recv_shape, dtype)
            return pallas_fn(
                pltpu.with_memory_space_constraint(tokens, pltpu.HBM),
                pltpu.with_memory_space_constraint(recv, pltpu.HBM),
            )
        if mode in ("hbm_expert_pack", "hbm_expert_pack_early"):
            pack = pl.empty(expert_pack_shape, dtype)
            recv = pl.empty(expert_recv_shape, dtype)
            return pallas_fn(
                pltpu.with_memory_space_constraint(tokens, pltpu.HBM),
                pltpu.with_memory_space_constraint(pack, pltpu.HBM),
                pltpu.with_memory_space_constraint(recv, pltpu.HBM),
            )
        if mode == "vmem_expert_pack":
            recv = pl.empty(expert_recv_shape, dtype)
            return pallas_fn(
                pltpu.with_memory_space_constraint(tokens, pltpu.HBM),
                pltpu.with_memory_space_constraint(recv, pltpu.HBM),
            )
        recv = pl.empty(hbm_shape, dtype)
        return pallas_fn(
            pltpu.with_memory_space_constraint(tokens, pltpu.HBM),
            pltpu.with_memory_space_constraint(recv, pltpu.HBM),
        )

    return run


def _make_tokens(
    *,
    mesh: jax.sharding.Mesh,
    global_tokens: int,
    hidden_size: int,
    dtype: jnp.dtype,
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
    parser = argparse.ArgumentParser(description="Benchmark Stage2 scatter pack candidates.")
    parser.add_argument("--ep-size", type=int, required=True)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--num-tokens", nargs="+", type=int, default=[512, 8192])
    parser.add_argument("--hidden-size", type=int, default=8192)
    parser.add_argument("--num-experts", type=int, default=256)
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
        choices=MODES,
        default=list(MODES),
    )
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--warmup-iters", type=int, default=1)
    parser.add_argument("--trace-root", default="/tmp/sglang_stage2_scatter_pack_trace")
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
    fanouts = args.fanout or [min(args.top_k, args.ep_size - 1)]
    rows = []
    vmem_limit_bytes = args.vmem_limit_mb * 1024 * 1024

    print(
        "stage2 scatter pack benchmark: "
        f"process={jax.process_index()}/{jax.process_count()}, "
        f"local_devices={len(jax.local_devices())}, global_devices={len(jax.devices())}, "
        f"ep_size={args.ep_size}, hidden_size={args.hidden_size}, "
        f"num_experts={args.num_experts}, top_k={args.top_k}, "
        f"vmem_limit_mb={args.vmem_limit_mb}"
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
                try:
                    run = _make_kernel(
                        mode=mode,
                        mesh=mesh,
                        global_tokens=tokens,
                        hidden_size=args.hidden_size,
                        num_experts=args.num_experts,
                        top_k=args.top_k,
                        fanout=fanout,
                        dtype=dtype,
                        vmem_limit_bytes=vmem_limit_bytes,
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
                    status = "ok"
                    error = None
                except (
                    Exception
                ) as exc:  # noqa: BLE001 - benchmark should keep remaining modes running.
                    samples = []
                    stats = {
                        "mean_ms": math.nan,
                        "median_ms": math.nan,
                        "min_ms": math.nan,
                        "max_ms": math.nan,
                    }
                    status = "error"
                    error = f"{type(exc).__name__}: {exc}"
                    print(f"ERROR mode={mode} tokens={tokens} fanout={fanout}: {error}")

                remote_bytes = tokens * args.top_k * args.hidden_size * jnp.dtype(dtype).itemsize
                row = {
                    "ep_size": args.ep_size,
                    "tokens": tokens,
                    "hidden_size": args.hidden_size,
                    "top_k": args.top_k,
                    "fanout": fanout,
                    "mode": mode,
                    "status": status,
                    "error": error,
                    "samples_ms": samples,
                    "remote_bytes": int(remote_bytes),
                    "effective_remote_gbs_mean": (
                        float(remote_bytes) / (stats["mean_ms"] / 1000.0) / 1e9
                        if stats["mean_ms"] > 0
                        else math.nan
                    ),
                    **stats,
                }
                rows.append(row)
                print("RESULT " + json.dumps(row, ensure_ascii=False, sort_keys=True))

    if args.out_json and jax.process_index() == 0:
        path = Path(args.out_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
