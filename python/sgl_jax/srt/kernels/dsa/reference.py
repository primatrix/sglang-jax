"""CPU-safe reference helpers for DSA cache writes and slot transforms."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from sgl_jax.srt.layers.attention.dsa_types import DsaSelection


def _validate_packed_index_cache(
    cache: jax.Array,
    *,
    page_size: int,
    index_head_dim: int,
) -> None:
    if cache.ndim != 4:
        raise ValueError(f"Index-K cache must have rank 4; got {cache.ndim}")
    if page_size <= 0 or index_head_dim <= 0:
        raise ValueError("page_size and index_head_dim must be positive")
    if cache.shape[1] * cache.shape[2] < page_size:
        raise ValueError(
            "Index-K cache row/lane axes are too small for page_size; got "
            f"shape={cache.shape}, page_size={page_size}"
        )
    if cache.shape[3] < index_head_dim:
        raise ValueError(f"Index-K cache width {cache.shape[3]} is smaller than {index_head_dim=}")


def write_indexer_k_cache(
    cache: jax.Array,
    *,
    index_k: jax.Array,
    write_slots: jax.Array,
    page_size: int,
    index_head_dim: int,
) -> jax.Array:
    """Functionally write current Indexer-K values to token-granular slots."""
    _validate_packed_index_cache(
        cache,
        page_size=page_size,
        index_head_dim=index_head_dim,
    )
    if index_k.ndim != 2 or index_k.shape[1] != index_head_dim:
        raise ValueError(f"index_k must have shape [tokens, {index_head_dim}]; got {index_k.shape}")
    if write_slots.shape != (index_k.shape[0],) or write_slots.dtype != jnp.int32:
        raise ValueError(
            "write_slots must be int32 with shape [tokens]; got "
            f"shape={write_slots.shape}, dtype={write_slots.dtype}"
        )

    packing = cache.shape[2]
    capacity = cache.shape[0] * page_size
    valid = (write_slots >= 0) & (write_slots < capacity)
    safe_slots = jnp.where(valid, write_slots, 0)
    page = safe_slots // page_size
    offset = safe_slots % page_size
    row = offset // packing
    lane = offset % packing
    page = jnp.where(valid, page, cache.shape[0])
    return cache.at[page, row, lane, :index_head_dim].set(
        index_k.astype(cache.dtype),
        mode="drop",
    )


def gather_indexer_k_cache(
    cache: jax.Array,
    *,
    physical_slots: jax.Array,
    page_size: int,
    index_head_dim: int,
) -> jax.Array:
    """Gather Indexer-K rows from token-granular physical slots."""
    _validate_packed_index_cache(
        cache,
        page_size=page_size,
        index_head_dim=index_head_dim,
    )
    if physical_slots.dtype != jnp.int32:
        raise TypeError(f"physical_slots must have dtype int32; got {physical_slots.dtype}")

    packing = cache.shape[2]
    capacity = cache.shape[0] * page_size
    valid = (physical_slots >= 0) & (physical_slots < capacity)
    safe_slots = jnp.where(valid, physical_slots, 0)
    page = safe_slots // page_size
    offset = safe_slots % page_size
    row = offset // packing
    lane = offset % packing
    gathered = cache[page, row, lane, :index_head_dim]
    return jnp.where(valid[..., None], gathered, jnp.zeros_like(gathered))


def _duplicate_after_first(logical_ids: jax.Array) -> jax.Array:
    """Mark duplicate IDs after their first occurrence while preserving score order."""

    def per_row(row):
        order = jnp.argsort(row, stable=True)
        sorted_row = row[order]
        duplicate_sorted = jnp.concatenate(
            [
                jnp.zeros((1,), dtype=jnp.bool_),
                sorted_row[1:] == sorted_row[:-1],
            ]
        )
        return jnp.zeros_like(duplicate_sorted).at[order].set(duplicate_sorted)

    return jax.vmap(per_row)(logical_ids)


def logical_topk_to_physical_slots(
    *,
    logical_topk_ids: jax.Array,
    selected_counts: jax.Array,
    req_to_token_slots: jax.Array,
    query_request_indices: jax.Array,
    query_positions: jax.Array,
    producer_layer: int,
) -> DsaSelection:
    """Map logical Top-K IDs to physical slots and compact invalid entries."""
    if logical_topk_ids.ndim != 2 or logical_topk_ids.dtype != jnp.int32:
        raise ValueError("logical_topk_ids must be a rank-2 int32 array")
    token_count, topk_width = logical_topk_ids.shape
    for name, value in (
        ("selected_counts", selected_counts),
        ("query_request_indices", query_request_indices),
        ("query_positions", query_positions),
    ):
        if value.shape != (token_count,) or value.dtype != jnp.int32:
            raise ValueError(f"{name} must be int32 with shape {(token_count,)}")
    if req_to_token_slots.ndim != 2 or req_to_token_slots.dtype != jnp.int32:
        raise ValueError("req_to_token_slots must be a rank-2 int32 array")
    if type(producer_layer) is not int or producer_layer < 0:
        raise ValueError("producer_layer must be a nonnegative Python int")

    request_count, max_request_tokens = req_to_token_slots.shape
    rank_valid = jnp.arange(topk_width)[None, :] < jnp.clip(selected_counts[:, None], 0, topk_width)
    request_valid = (query_request_indices >= 0) & (query_request_indices < request_count)
    logical_valid = (logical_topk_ids >= 0) & (logical_topk_ids < max_request_tokens)
    causal_valid = logical_topk_ids <= query_positions[:, None]
    duplicate = _duplicate_after_first(logical_topk_ids)

    safe_requests = jnp.clip(query_request_indices, 0, max(request_count - 1, 0))
    safe_logical = jnp.clip(logical_topk_ids, 0, max(max_request_tokens - 1, 0))
    physical_slots = req_to_token_slots[safe_requests[:, None], safe_logical]
    valid = (
        rank_valid
        & request_valid[:, None]
        & logical_valid
        & causal_valid
        & ~duplicate
        & (physical_slots >= 0)
    )

    compact_order = jnp.argsort(~valid, axis=1, stable=True)
    compact_logical = jnp.take_along_axis(logical_topk_ids, compact_order, axis=1)
    compact_physical = jnp.take_along_axis(physical_slots, compact_order, axis=1)
    compact_counts = jnp.sum(valid, axis=1, dtype=jnp.int32)
    compact_valid = jnp.arange(topk_width)[None, :] < compact_counts[:, None]
    compact_logical = jnp.where(compact_valid, compact_logical, -1).astype(jnp.int32)
    compact_physical = jnp.where(compact_valid, compact_physical, 0).astype(jnp.int32)

    return DsaSelection(
        logical_topk_ids=compact_logical,
        physical_slots=compact_physical,
        selected_counts=compact_counts,
        producer_layer=producer_layer,
    )
