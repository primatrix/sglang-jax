from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np


def apply_rebalance_mapping_global(
    *,
    weights: jax.Array,
    src_for_dst_physical: jax.Array,  # (num_physical_experts,) int32
    expert_axis: int = 0,
) -> jax.Array:
    """Apply a physical-slot remapping to a weight tensor in a single-process setting.

    This is a pure gather-based implementation useful for:
      - correctness testing
      - single-device debugging

    In distributed EP, each rank does not hold all physical experts, so this must be
    replaced with a collective exchange (e.g., ragged all-to-all).
    """
    if src_for_dst_physical.ndim != 1:
        raise ValueError(
            f"Expected src_for_dst_physical to be 1D, got {src_for_dst_physical.ndim=}"
        )
    src_for_dst_physical = src_for_dst_physical.astype(jnp.int32)
    return jnp.take(weights, src_for_dst_physical, axis=expert_axis)


def compute_rebalance_sources(
    *,
    old_physical_to_logical_map: np.ndarray,
    new_physical_to_logical_map: np.ndarray,
    ep_size: int,
) -> np.ndarray:
    """Compute a source physical id for each destination physical id.

    The returned array has shape (num_layers, num_physical_experts). For each destination
    physical id `dst`, we choose a source `src` such that:
      old_map[src] == new_map[dst]

    Preference:
      - pick a source on the same EP rank as the destination when possible
      - otherwise pick the lowest-index source
    """
    if ep_size <= 0:
        raise ValueError(f"Expected {ep_size=} to be > 0.")

    old_map = np.asarray(old_physical_to_logical_map)
    new_map = np.asarray(new_physical_to_logical_map)
    if old_map.shape != new_map.shape:
        raise ValueError(
            "Expected old_map and new_map to have the same shape, got "
            f"{old_map.shape} vs {new_map.shape}"
        )
    if old_map.ndim == 1:
        old_map = old_map[None, :]
        new_map = new_map[None, :]
    if old_map.ndim != 2:
        raise ValueError(f"Expected maps to be 1D or 2D, got {old_map.ndim=}")

    num_layers, num_physical = old_map.shape
    if num_physical % ep_size != 0:
        raise ValueError(f"Expected {num_physical=} to be divisible by {ep_size=}.")
    local_e = num_physical // ep_size

    src_for_dst = np.empty((num_layers, num_physical), dtype=np.int32)
    for layer in range(num_layers):
        p2l_old = old_map[layer].astype(np.int32, copy=False)
        p2l_new = new_map[layer].astype(np.int32, copy=False)

        num_logical = int(max(np.max(p2l_old), np.max(p2l_new))) + 1
        logical_to_phys_old: list[np.ndarray] = [
            np.where(p2l_old == logical_id)[0].astype(np.int32) for logical_id in range(num_logical)
        ]
        logical_to_phys_new: list[np.ndarray] = [
            np.where(p2l_new == logical_id)[0].astype(np.int32) for logical_id in range(num_logical)
        ]

        layer_src_for_dst = np.full((num_physical,), -1, dtype=np.int32)
        used_src = np.zeros((num_physical,), dtype=np.bool_)

        # 1) Keep fixed points (src == dst) when the logical id doesn't change.
        fixed = np.where(p2l_old == p2l_new)[0].astype(np.int32)
        layer_src_for_dst[fixed] = fixed
        used_src[fixed] = True

        # 2) For each logical id, greedily match unused sources to destinations, preferring same-rank.
        for logical_id in range(num_logical):
            srcs_all = logical_to_phys_old[logical_id]
            dsts_all = logical_to_phys_new[logical_id]
            if srcs_all.size == 0 and dsts_all.size > 0:
                raise ValueError(
                    f"Layer {layer}: logical expert {logical_id} requested in new placement but absent in old."
                )
            if dsts_all.size == 0:
                continue

            # Filter out fixed destinations already assigned.
            dsts = dsts_all[layer_src_for_dst[dsts_all] < 0]
            if dsts.size == 0:
                continue

            # Build rank buckets for unused sources.
            srcs_unused = srcs_all[~used_src[srcs_all]]
            src_unused_by_rank: list[list[int]] = [[] for _ in range(ep_size)]
            for s in srcs_unused.tolist():
                src_unused_by_rank[int(s // local_e)].append(int(s))

            # Destinations grouped by rank.
            dst_by_rank: list[list[int]] = [[] for _ in range(ep_size)]
            for d in dsts.tolist():
                dst_by_rank[int(d // local_e)].append(int(d))

            # 2a) Same-rank matches using unused sources (unique assignment).
            for rank in range(ep_size):
                src_bucket = src_unused_by_rank[rank]
                dst_bucket = dst_by_rank[rank]
                if not src_bucket or not dst_bucket:
                    continue
                m = min(len(src_bucket), len(dst_bucket))
                for i in range(m):
                    s = src_bucket.pop()
                    d = dst_bucket.pop()
                    layer_src_for_dst[d] = s
                    used_src[s] = True

            # 2b) Cross-rank matches using any remaining unused sources.
            remaining_srcs: list[int] = []
            for rank in range(ep_size):
                remaining_srcs.extend(src_unused_by_rank[rank])
            if remaining_srcs:
                # Deterministic order.
                remaining_srcs.sort()
                for rank in range(ep_size):
                    while dst_by_rank[rank] and remaining_srcs:
                        d = dst_by_rank[rank].pop()
                        s = remaining_srcs.pop()  # take largest remaining to keep deterministic
                        layer_src_for_dst[d] = s
                        used_src[s] = True

            # 2c) If new placement needs more replicas than exist in old, reuse sources (duplicates).
            # Prefer reusing a source on the same destination rank when possible to minimize comm.
            still_missing: list[int] = []
            for rank in range(ep_size):
                still_missing.extend(dst_by_rank[rank])
            if still_missing:
                # Rank-local source lists (including already-used).
                src_all_by_rank: list[list[int]] = [[] for _ in range(ep_size)]
                for s in srcs_all.tolist():
                    src_all_by_rank[int(s // local_e)].append(int(s))
                for rank in range(ep_size):
                    src_all_by_rank[rank].sort()

                # Deterministic round-robin reuse per rank.
                rr_pos = [0 for _ in range(ep_size)]
                for d in sorted(still_missing):
                    r = int(d // local_e)
                    if src_all_by_rank[r]:
                        pos = rr_pos[r] % len(src_all_by_rank[r])
                        rr_pos[r] += 1
                        s = src_all_by_rank[r][pos]
                    else:
                        # Fallback: should be rare (means this logical has zero sources on dst rank).
                        s = int(srcs_all[int((d + layer) % int(srcs_all.size))])
                    layer_src_for_dst[d] = int(s)

        src_for_dst[layer] = layer_src_for_dst

    if np.any(src_for_dst < 0):
        bad = np.where(src_for_dst < 0)
        raise ValueError(f"Unable to find rebalance sources for some destinations: {bad}")
    return src_for_dst


@dataclass(frozen=True)
class RebalanceAllToAllPlan:
    """Host-computed plan for distributed weight rebalance via ragged all-to-all.

    Arrays are numpy and intended to be broadcast to devices. Each EP rank `r`
    extracts its own slice `plan.for_rank(r)`.
    """

    ep_size: int
    num_physical_experts: int
    local_num_physical_experts: int
    send_src_local_indices: list[np.ndarray]
    send_sizes: list[np.ndarray]
    input_offsets: list[np.ndarray]
    recv_dst_local_indices: list[np.ndarray]
    recv_sizes: list[np.ndarray]
    output_offsets: list[np.ndarray]

    def for_rank(self, ep_rank: int) -> dict[str, np.ndarray]:
        if not (0 <= ep_rank < self.ep_size):
            raise ValueError(f"Expected {ep_rank=} to be in [0, {self.ep_size}).")
        return {
            "send_src_local_indices": self.send_src_local_indices[ep_rank],
            "send_sizes": self.send_sizes[ep_rank],
            "input_offsets": self.input_offsets[ep_rank],
            "recv_dst_local_indices": self.recv_dst_local_indices[ep_rank],
            "recv_sizes": self.recv_sizes[ep_rank],
            "output_offsets": self.output_offsets[ep_rank],
        }


def build_rebalance_all_to_all_plan(
    *,
    src_for_dst_physical: np.ndarray,  # (num_physical_experts,)
    ep_size: int,
) -> RebalanceAllToAllPlan:
    """Build a ragged all-to-all plan for distributing expert rows for rebalance.

    `src_for_dst_physical[dst] = src` indicates destination physical slot `dst`
    should receive the weight row currently stored at source physical slot `src`.

    The plan is expressed in terms of *local* indices:
      - local index range per rank: [0, local_num_physical_experts)
    """
    if ep_size <= 0:
        raise ValueError(f"Expected {ep_size=} to be > 0.")
    src = np.asarray(src_for_dst_physical).astype(np.int64)
    if src.ndim != 1:
        raise ValueError(f"Expected src_for_dst_physical to be 1D, got {src.ndim=}.")

    num_physical = int(src.shape[0])
    if num_physical % ep_size != 0:
        raise ValueError(f"Expected {num_physical=} to be divisible by {ep_size=}.")
    local_e = num_physical // ep_size

    if np.any(src < 0) or np.any(src >= num_physical):
        raise ValueError("src_for_dst_physical contains out-of-range physical ids")

    edges: list[list[list[tuple[int, int]]]] = [
        [[[] for _ in range(ep_size)] for _ in range(ep_size)]
    ][0]

    for dst_physical in range(num_physical):
        src_physical = int(src[dst_physical])
        src_rank, src_local = divmod(src_physical, local_e)
        dst_rank, dst_local = divmod(dst_physical, local_e)
        edges[src_rank][dst_rank].append((src_local, dst_local))

    send_src_local_indices: list[np.ndarray] = []
    send_sizes: list[np.ndarray] = []
    input_offsets: list[np.ndarray] = []

    recv_dst_local_indices: list[np.ndarray] = []
    recv_sizes: list[np.ndarray] = []
    # Important: ragged_all_to_all's `output_offsets` are specified on the *sender*
    # but consumed (after an all_to_all) on the receiver. We'll first compute the
    # receiver-side offsets (prefix sums over senders), then transpose them so each
    # sender rank has its per-destination `output_offsets`.
    output_offsets_receiver: list[np.ndarray] = []

    for src_rank in range(ep_size):
        sizes = np.array(
            [len(edges[src_rank][dst_rank]) for dst_rank in range(ep_size)],
            dtype=np.int32,
        )
        offs = np.zeros((ep_size,), dtype=np.int32)
        if ep_size > 1:
            offs[1:] = np.cumsum(sizes[:-1], dtype=np.int32)
        flat = []
        for dst_rank in range(ep_size):
            flat.extend([src_local for (src_local, _dst_local) in edges[src_rank][dst_rank]])
        send_src_local_indices.append(np.array(flat, dtype=np.int32))
        send_sizes.append(sizes)
        input_offsets.append(offs)

    for dst_rank in range(ep_size):
        sizes = np.array(
            [len(edges[src_rank][dst_rank]) for src_rank in range(ep_size)],
            dtype=np.int32,
        )
        offs = np.zeros((ep_size,), dtype=np.int32)
        if ep_size > 1:
            offs[1:] = np.cumsum(sizes[:-1], dtype=np.int32)
        flat = []
        for src_rank in range(ep_size):
            flat.extend([dst_local for (_src_local, dst_local) in edges[src_rank][dst_rank]])
        recv_dst_local_indices.append(np.array(flat, dtype=np.int32))
        recv_sizes.append(sizes)

        # Receiver-side prefix sums: offset for data coming from each src_rank.
        output_offsets_receiver.append(offs)

    output_offsets_sender: list[np.ndarray] = []
    for src_rank in range(ep_size):
        # Sender-side: for each destination rank, store the offset that this sender's
        # slice should be written at on that destination.
        output_offsets_sender.append(
            np.array(
                [output_offsets_receiver[dst_rank][src_rank] for dst_rank in range(ep_size)],
                dtype=np.int32,
            )
        )

    return RebalanceAllToAllPlan(
        ep_size=ep_size,
        num_physical_experts=num_physical,
        local_num_physical_experts=local_e,
        send_src_local_indices=send_src_local_indices,
        send_sizes=send_sizes,
        input_offsets=input_offsets,
        recv_dst_local_indices=recv_dst_local_indices,
        recv_sizes=recv_sizes,
        output_offsets=output_offsets_sender,
    )


@dataclass(frozen=True)
class RebalanceAllToAllDevicePlan:
    """Stacked/padded plan representation suitable for SPMD compilation."""

    ep_size: int
    local_num_physical_experts: int
    max_total_send: int
    send_src_local_indices: np.ndarray  # (ep_size, max_total_send)
    send_sizes: np.ndarray  # (ep_size, ep_size)
    input_offsets: np.ndarray  # (ep_size, ep_size)
    output_offsets: np.ndarray  # (ep_size, ep_size) sender-side
    recv_sizes: np.ndarray  # (ep_size, ep_size) receiver-side
    recv_dst_local_indices: np.ndarray  # (ep_size, local_e)

    def for_rank(self, ep_rank: int) -> dict[str, np.ndarray]:
        if not (0 <= ep_rank < self.ep_size):
            raise ValueError(f"Expected {ep_rank=} to be in [0, {self.ep_size}).")
        return {
            "send_src_local_indices": self.send_src_local_indices[ep_rank],
            "send_sizes": self.send_sizes[ep_rank],
            "input_offsets": self.input_offsets[ep_rank],
            "output_offsets": self.output_offsets[ep_rank],
            "recv_sizes": self.recv_sizes[ep_rank],
            "recv_dst_local_indices": self.recv_dst_local_indices[ep_rank],
        }


def build_rebalance_all_to_all_device_plan(
    *,
    src_for_dst_physical: np.ndarray,  # (num_physical_experts,)
    ep_size: int,
) -> RebalanceAllToAllDevicePlan:
    """Build a stacked/padded plan suitable for `jax.lax.ragged_all_to_all` in pjit."""
    host_plan = build_rebalance_all_to_all_plan(
        src_for_dst_physical=src_for_dst_physical, ep_size=ep_size
    )

    local_e = host_plan.local_num_physical_experts
    max_total_send = int(max(int(np.sum(sz)) for sz in host_plan.send_sizes))
    send_src = np.zeros((ep_size, max_total_send), dtype=np.int32)
    for r in range(ep_size):
        flat = host_plan.send_src_local_indices[r].astype(np.int32)
        if flat.shape[0] > max_total_send:
            raise ValueError("Unexpected: max_total_send smaller than a rank's send list.")
        send_src[r, : flat.shape[0]] = flat

    recv_dst = np.zeros((ep_size, local_e), dtype=np.int32)
    for r in range(ep_size):
        flat = host_plan.recv_dst_local_indices[r].astype(np.int32)
        if flat.shape[0] != local_e:
            raise ValueError(
                f"Expected each rank to receive exactly {local_e} expert rows, got {flat.shape[0]}."
            )
        recv_dst[r] = flat

    send_sizes = np.stack([x.astype(np.int32) for x in host_plan.send_sizes], axis=0)
    input_offsets = np.stack([x.astype(np.int32) for x in host_plan.input_offsets], axis=0)
    output_offsets = np.stack([x.astype(np.int32) for x in host_plan.output_offsets], axis=0)
    recv_sizes = np.stack([x.astype(np.int32) for x in host_plan.recv_sizes], axis=0)

    return RebalanceAllToAllDevicePlan(
        ep_size=ep_size,
        local_num_physical_experts=local_e,
        max_total_send=max_total_send,
        send_src_local_indices=send_src,
        send_sizes=send_sizes,
        input_offsets=input_offsets,
        output_offsets=output_offsets,
        recv_sizes=recv_sizes,
        recv_dst_local_indices=recv_dst,
    )


def pack_expert_rows(weights_local: jax.Array, send_src_local_indices: jax.Array) -> jax.Array:
    """Pack local expert rows into a contiguous buffer in the requested order."""
    if send_src_local_indices.ndim != 1:
        raise ValueError(
            f"Expected send_src_local_indices to be 1D, got {send_src_local_indices.ndim=}"
        )
    send_src_local_indices = send_src_local_indices.astype(jnp.int32)
    return jnp.take(weights_local, send_src_local_indices, axis=0)


def scatter_expert_rows(
    *,
    recv_buffer: jax.Array,
    recv_dst_local_indices: jax.Array,
    out_shape: tuple[int, ...],
) -> jax.Array:
    """Scatter received rows into a local expert buffer."""
    if recv_dst_local_indices.ndim != 1:
        raise ValueError(
            f"Expected recv_dst_local_indices to be 1D, got {recv_dst_local_indices.ndim=}"
        )
    if recv_buffer.shape[0] != recv_dst_local_indices.shape[0]:
        raise ValueError(
            "Expected recv_buffer and recv_dst_local_indices to have the same leading size."
        )

    recv_dst_local_indices = recv_dst_local_indices.astype(jnp.int32)
    out = jnp.empty(out_shape, dtype=recv_buffer.dtype)
    out = out.at[recv_dst_local_indices].set(recv_buffer)
    return out


def rebalance_weights_all_to_all(
    *,
    weights_local: jax.Array,  # (local_E, ...)
    send_src_local_indices: jax.Array,  # (total_send,)
    input_offsets: jax.Array,  # (ep_size,)
    send_sizes: jax.Array,  # (ep_size,)
    output_offsets: jax.Array,  # (ep_size,)
    recv_sizes: jax.Array,  # (ep_size,)
    recv_dst_local_indices: jax.Array,  # (total_recv,)
    axis_name,
) -> jax.Array:
    """Distributed online weight rebalance using `jax.lax.ragged_all_to_all`."""
    send_sizes = send_sizes.astype(jnp.int32)
    recv_sizes = recv_sizes.astype(jnp.int32)
    input_offsets = input_offsets.astype(jnp.int32)
    output_offsets = output_offsets.astype(jnp.int32)

    send_buf = pack_expert_rows(weights_local, send_src_local_indices)
    total_recv = recv_dst_local_indices.shape[0]
    recv_buf = jnp.empty((total_recv,) + weights_local.shape[1:], dtype=weights_local.dtype)

    recv_buf = jax.lax.ragged_all_to_all(
        send_buf,
        recv_buf,
        input_offsets,
        send_sizes,
        output_offsets,
        recv_sizes,
        axis_name=axis_name,
    )

    return scatter_expert_rows(
        recv_buffer=recv_buf,
        recv_dst_local_indices=recv_dst_local_indices,
        out_shape=weights_local.shape,
    )
