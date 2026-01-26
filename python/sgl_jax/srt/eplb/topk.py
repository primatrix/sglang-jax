from __future__ import annotations

import jax
import jax.numpy as jnp


def compute_topk(
    *,
    router_logits: jax.Array,  # (num_tokens, num_experts)
    top_k: int,
    correction_bias: jax.Array | None = None,  # (num_experts,)
    use_grouped_topk: bool = False,
    num_groups: int = 1,
    top_k_groups: int = 1,
    renormalize_topk_logits: bool = False,
    routed_scaling_factor: float | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Compute top-k routing ids/weights for MoE (matches fused kernel semantics).

    Semantics:
      - Expert selection is done on `router_logits + correction_bias` (if bias provided).
      - Returned weights are always taken from *unbiased* `router_logits`.
      - Grouped-topk uses the same group scoring as the fused kernel:
          * no bias: group score = max score in group
          * with bias: group score = sum of top2 scores in group
      - `renormalize_topk_logits` renormalizes weights to sum to 1 for each token.
      - `routed_scaling_factor` (if set) scales weights after optional renormalization.
    """
    if router_logits.ndim != 2:
        raise ValueError(f"Expected {router_logits.ndim=} to be 2.")
    if top_k <= 0:
        raise ValueError(f"Expected {top_k=} to be > 0.")

    router_logits_f32 = router_logits.astype(jnp.float32)
    routing_scores = router_logits_f32

    if correction_bias is not None:
        if correction_bias.ndim != 1:
            raise ValueError(f"Expected {correction_bias.ndim=} to be 1.")
        if correction_bias.shape[0] != router_logits.shape[1]:
            raise ValueError(
                "Expected correction_bias.shape[0] to equal num_experts, got "
                f"{correction_bias.shape[0]=} vs {router_logits.shape[1]=}."
            )
        routing_scores = routing_scores + jnp.expand_dims(correction_bias.astype(jnp.float32), 0)

    if use_grouped_topk:
        num_tokens, num_experts = router_logits.shape
        if num_groups <= 0 or top_k_groups <= 0:
            raise ValueError(f"Expected {num_groups=} and {top_k_groups=} to be > 0.")
        if num_experts % num_groups != 0:
            raise ValueError(f"Expected {num_experts=} to be divisible by {num_groups=}.")
        experts_per_group = num_experts // num_groups
        reshaped = routing_scores.reshape(num_tokens, num_groups, experts_per_group)

        if correction_bias is not None:
            top2_vals, _ = jax.lax.top_k(reshaped, 2)
            group_scores = jnp.sum(top2_vals, axis=-1)
        else:
            group_scores = jnp.max(reshaped, axis=-1)

        group_mask_accum = jnp.zeros((num_tokens, num_groups), dtype=jnp.bool_)
        temp_group_scores = group_scores
        group_iota = jax.lax.broadcasted_iota(jnp.int32, (num_tokens, num_groups), 1)

        for _ in range(top_k_groups):
            curr_max_group_idx = jnp.argmax(temp_group_scores, axis=1, keepdims=True)
            curr_mask = group_iota == curr_max_group_idx
            group_mask_accum = jnp.logical_or(group_mask_accum, curr_mask)
            temp_group_scores = jnp.where(curr_mask, -jnp.float32(jnp.inf), temp_group_scores)

        expert_mask = jnp.repeat(
            jnp.expand_dims(group_mask_accum, axis=2), experts_per_group, axis=2
        ).reshape(num_tokens, num_experts)
        routing_scores = jnp.where(expert_mask, routing_scores, -jnp.float32(jnp.inf))

    _, topk_ids = jax.lax.top_k(routing_scores, top_k)
    topk_ids = topk_ids.astype(jnp.int32)
    topk_weights = jnp.take_along_axis(router_logits_f32, topk_ids, axis=-1)

    if renormalize_topk_logits:
        topk_weights = topk_weights / (jnp.sum(topk_weights, axis=-1, keepdims=True) + 1e-6)

    if routed_scaling_factor is not None:
        topk_weights = topk_weights * routed_scaling_factor

    return topk_weights.astype(jnp.float32), topk_ids


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
