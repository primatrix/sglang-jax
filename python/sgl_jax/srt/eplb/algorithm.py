from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .metadata import ExpertLocationMetadata, choose_num_physical_experts


def _stable_hash_u32(*values: int) -> int:
    """Deterministic 32-bit mixing for dispatch offsets."""
    # Use Python ints to avoid numpy overflow warnings; wrap to 32-bit manually.
    h = 2166136261  # FNV-ish
    for v in values:
        h ^= int(v) & 0xFFFFFFFF
        h = (h * 16777619) & 0xFFFFFFFF
    return h


@dataclass(frozen=True)
class GreedyEplbConfig:
    ep_size: int
    num_redundant_experts: int = 0
    max_num_redundant_experts: int = 128
    seed: int = 0


def rebalance_experts_greedy(
    *,
    tokens_per_logical_expert: np.ndarray,
    ep_size: int,
    num_redundant_experts: int,
    max_num_redundant_experts: int = 128,
    seed: int = 0,
) -> ExpertLocationMetadata:
    """Greedy baseline EPLB: replicate hot experts, then place to balance rank load.

    Input:
      tokens_per_logical_expert:
        - shape (num_layers, num_logical_experts) or (num_logical_experts,)
        - non-negative weights (ints/floats ok)

    Output:
      ExpertLocationMetadata with:
        - physical_to_logical_map[layer, num_physical_experts]
        - logical_to_rank_dispatch_physical_map[layer, num_logical_experts, ep_size]

    Notes:
      - This is a baseline algorithm intended for correctness + a reasonable first-order balance.
      - It does not model topology; it spreads replica selection evenly across ranks.
    """
    if ep_size <= 0:
        raise ValueError(f"Expected {ep_size=} to be > 0.")
    if max_num_redundant_experts < 0:
        raise ValueError(f"Expected {max_num_redundant_experts=} to be >= 0.")
    if num_redundant_experts < 0:
        raise ValueError(f"Expected {num_redundant_experts=} to be >= 0.")

    w = np.asarray(tokens_per_logical_expert)
    if w.ndim == 1:
        w = w[None, :]
    if w.ndim != 2:
        raise ValueError(f"Expected tokens_per_logical_expert to be 1D or 2D, got {w.ndim=}")
    if np.any(w < 0):
        raise ValueError("Expected tokens_per_logical_expert to be non-negative")

    num_layers, num_logical_experts = w.shape
    num_physical_experts, _ = choose_num_physical_experts(
        num_logical_experts=num_logical_experts,
        ep_size=ep_size,
        requested_num_redundant_experts=num_redundant_experts,
        max_num_redundant_experts=max_num_redundant_experts,
    )
    local_e = num_physical_experts // ep_size
    r = num_physical_experts - num_logical_experts

    physical_to_logical = np.empty((num_layers, num_physical_experts), dtype=np.int32)
    logical_to_rank_dispatch_physical = np.empty(
        (num_layers, num_logical_experts, ep_size), dtype=np.int32
    )

    base_logicals = np.arange(num_logical_experts, dtype=np.int32)

    for layer in range(num_layers):
        layer_w = w[layer].astype(np.float64)

        # Choose which logical experts to replicate (may contain duplicates if r > num_logical_experts).
        if r > 0:
            order = np.argsort(-layer_w, kind="stable")  # hot first
            replicate = order[np.arange(r) % num_logical_experts].astype(np.int32)
            items = np.concatenate([base_logicals, replicate], axis=0)
        else:
            items = base_logicals

        # Place items into ranks to balance total load (LPT-style greedy).
        # Each item has weight = tokens for its logical id.
        item_weights = layer_w[items.astype(np.int64)]
        sort_idx = np.argsort(-item_weights, kind="stable")
        items_sorted = items[sort_idx]
        weights_sorted = item_weights[sort_idx]

        rank_bins: list[list[int]] = [[] for _ in range(ep_size)]
        rank_loads = np.zeros((ep_size,), dtype=np.float64)
        rank_remaining = np.full((ep_size,), local_e, dtype=np.int32)

        for logical_id, wt in zip(items_sorted.tolist(), weights_sorted.tolist(), strict=True):
            # Pick the least-loaded rank with remaining capacity.
            candidates = np.where(rank_remaining > 0)[0]
            if candidates.size == 0:
                raise RuntimeError("Internal error: no rank has remaining slots")
            best = candidates[np.argmin(rank_loads[candidates])]
            rank_bins[int(best)].append(int(logical_id))
            rank_loads[int(best)] += float(wt)
            rank_remaining[int(best)] -= 1

        if not np.all(rank_remaining == 0):
            raise RuntimeError("Internal error: not all ranks filled")

        # Flatten rank-local layouts into the global physical id space.
        layout = np.array([x for rbin in rank_bins for x in rbin], dtype=np.int32)
        if layout.shape != (num_physical_experts,):
            raise RuntimeError("Internal error: unexpected layout shape")
        physical_to_logical[layer] = layout

        # Build dispatch map: for each logical expert, choose (possibly different) physical replicas
        # across ep ranks to spread traffic.
        p2l = physical_to_logical[layer]
        logical_to_phys: list[np.ndarray] = []
        for logical_id in range(num_logical_experts):
            phys = np.where(p2l == logical_id)[0].astype(np.int32)
            if phys.size == 0:
                raise RuntimeError("Internal error: missing logical expert in placement")
            logical_to_phys.append(phys)

        for logical_id in range(num_logical_experts):
            phys = logical_to_phys[logical_id]
            phys_sorted = np.sort(phys)
            base_offset = _stable_hash_u32(seed, layer, logical_id) % int(phys_sorted.size)
            for rank in range(ep_size):
                chosen = phys_sorted[(base_offset + rank) % int(phys_sorted.size)]
                logical_to_rank_dispatch_physical[layer, logical_id, rank] = chosen

    meta = ExpertLocationMetadata(
        ep_size=ep_size,
        physical_to_logical_map=physical_to_logical,
        logical_to_rank_dispatch_physical_map=logical_to_rank_dispatch_physical,
    )
    meta.validate()
    return meta
