from __future__ import annotations

import jax
import jax.numpy as jnp


def dense_logits_from_topk(
    *,
    topk_ids: jax.Array,  # (num_tokens, top_k) int32
    topk_weights: jax.Array,  # (num_tokens, top_k) float/bf16/f32
    num_experts: int,
    fill_value: float = float("-inf"),
) -> jax.Array:
    """Build a dense `(num_tokens, num_experts)` logits matrix from top-k ids/weights.

    This is primarily intended as a compatibility bridge for kernels that only accept
    dense gating inputs. For EPLB with redundant experts, top-k should be computed in
    *logical* expert space, then mapped to *physical* ids before building dense logits.

    Notes:
      - If a token contains duplicate expert ids, later writes win (undefined behavior).
      - `fill_value=-inf` ensures non-selected experts are never chosen by argmax top-k.
    """
    if num_experts <= 0:
        raise ValueError(f"Expected {num_experts=} to be > 0.")
    if topk_ids.ndim != 2 or topk_weights.ndim != 2:
        raise ValueError(
            f"Expected topk_ids and topk_weights to be 2D, got {topk_ids.ndim=} {topk_weights.ndim=}"
        )
    if topk_ids.shape != topk_weights.shape:
        raise ValueError(f"Shape mismatch: {topk_ids.shape=} vs {topk_weights.shape=}")

    num_tokens, top_k = topk_ids.shape
    if top_k <= 0:
        raise ValueError(f"Expected {top_k=} to be > 0.")

    topk_ids = topk_ids.astype(jnp.int32)
    if topk_weights.dtype != jnp.float32:
        topk_weights = topk_weights.astype(jnp.float32)

    dense = jnp.full((num_tokens, num_experts), jnp.float32(fill_value), dtype=jnp.float32)
    rows = jnp.arange(num_tokens, dtype=jnp.int32)[:, None]
    dense = dense.at[rows, topk_ids].set(topk_weights)
    return dense


def map_logical_to_physical_topk_ids(
    *,
    topk_ids_logical: jax.Array,  # (num_tokens, top_k) int32 in [0, E_logical)
    logical_to_rank_dispatch_physical_map_layer: jax.Array,  # (E_logical, ep_size) int32
    ep_rank: int | jax.Array,
) -> jax.Array:
    """Map logical expert ids to physical ids using a per-layer static dispatch map."""
    if topk_ids_logical.ndim != 2:
        raise ValueError(f"Expected topk_ids_logical to be 2D, got {topk_ids_logical.ndim=}")
    if logical_to_rank_dispatch_physical_map_layer.ndim != 2:
        raise ValueError(
            "Expected logical_to_rank_dispatch_physical_map_layer to be 2D, got "
            f"{logical_to_rank_dispatch_physical_map_layer.ndim=}"
        )
    ep_size = int(logical_to_rank_dispatch_physical_map_layer.shape[1])
    if isinstance(ep_rank, int):
        if not (0 <= ep_rank < ep_size):
            raise ValueError(f"Expected {ep_rank=} to be in [0, {ep_size}).")
        dispatch_col = logical_to_rank_dispatch_physical_map_layer[:, ep_rank]
    else:
        ep_rank = ep_rank.astype(jnp.int32)
        dispatch_col = jax.lax.dynamic_index_in_dim(
            logical_to_rank_dispatch_physical_map_layer, ep_rank, axis=1, keepdims=False
        )
    topk_ids_logical = topk_ids_logical.astype(jnp.int32)
    return dispatch_col[topk_ids_logical]
