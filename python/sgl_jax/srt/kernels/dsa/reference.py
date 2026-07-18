"""CPU-safe JAX references for DSA cache writes, selection, and sparse MLA."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.layers.attention.dsa_types import DsaSelection


def _align_to_128(dim: int) -> int:
    return ((dim + 127) // 128) * 128


def _validate_packed_mla_cache(
    cache: jax.Array,
    *,
    page_size: int,
    latent_dim: int,
    rope_dim: int,
) -> tuple[int, int]:
    if cache.ndim != 4:
        raise ValueError(f"MLA cache must have rank 4; got {cache.ndim}")
    if page_size <= 0 or latent_dim <= 0 or rope_dim <= 0:
        raise ValueError("page_size, latent_dim, and rope_dim must be positive")
    if cache.shape[1] * cache.shape[2] < page_size:
        raise ValueError(
            "MLA cache row/lane axes are too small for page_size; got "
            f"shape={cache.shape}, page_size={page_size}"
        )
    latent_aligned = _align_to_128(latent_dim)
    rope_aligned = _align_to_128(rope_dim)
    required_width = latent_aligned + rope_aligned
    if cache.shape[3] < required_width:
        raise ValueError(
            f"MLA cache width {cache.shape[3]} is smaller than required {required_width}"
        )
    if not jnp.issubdtype(cache.dtype, jnp.floating):
        raise TypeError(f"MLA cache must have floating dtype; got {cache.dtype}")
    return latent_aligned, rope_aligned


def write_mla_kv_cache(
    cache: jax.Array,
    *,
    new_c_kv: jax.Array,
    new_k_pe: jax.Array,
    write_slots: jax.Array,
    page_size: int,
    latent_dim: int,
    rope_dim: int,
) -> jax.Array:
    """Functionally write latent MLA KV values to token-granular physical slots."""
    latent_aligned, _ = _validate_packed_mla_cache(
        cache,
        page_size=page_size,
        latent_dim=latent_dim,
        rope_dim=rope_dim,
    )
    if new_c_kv.ndim != 2 or new_c_kv.shape[1] != latent_dim:
        raise ValueError(f"new_c_kv must have shape [tokens, {latent_dim}]; got {new_c_kv.shape}")
    if new_k_pe.shape != (new_c_kv.shape[0], rope_dim):
        raise ValueError(
            f"new_k_pe must have shape {(new_c_kv.shape[0], rope_dim)}; " f"got {new_k_pe.shape}"
        )
    if not jnp.issubdtype(new_c_kv.dtype, jnp.floating) or not jnp.issubdtype(
        new_k_pe.dtype, jnp.floating
    ):
        raise TypeError("new_c_kv and new_k_pe must have floating dtypes")
    if write_slots.shape != (new_c_kv.shape[0],) or write_slots.dtype != jnp.int32:
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

    updated = cache.at[page, row, lane, :latent_dim].set(
        new_c_kv.astype(cache.dtype),
        mode="drop",
    )
    return updated.at[
        page,
        row,
        lane,
        latent_aligned : latent_aligned + rope_dim,
    ].set(
        new_k_pe.astype(cache.dtype),
        mode="drop",
    )


def dsa_sparse_mla_reference(
    q_latent: jax.Array,
    q_rope: jax.Array,
    cache: jax.Array,
    physical_slots: jax.Array,
    selected_counts: jax.Array,
    *,
    sm_scale: float,
    page_size: int,
    latent_dim: int,
    rope_dim: int,
) -> jax.Array:
    """Gather selected MLA slots and compute FP32 score, softmax, and PV."""
    latent_aligned, _ = _validate_packed_mla_cache(
        cache,
        page_size=page_size,
        latent_dim=latent_dim,
        rope_dim=rope_dim,
    )
    if q_latent.ndim != 3 or q_latent.shape[-1] != latent_dim:
        raise ValueError(
            f"q_latent must have shape [tokens, heads, {latent_dim}]; got {q_latent.shape}"
        )
    if q_rope.ndim != 3 or q_rope.shape[-1] != rope_dim:
        raise ValueError(f"q_rope must have shape [tokens, heads, {rope_dim}]; got {q_rope.shape}")
    if q_latent.shape[:2] != q_rope.shape[:2]:
        raise ValueError("q_latent and q_rope must have matching token and head dimensions")
    if not jnp.issubdtype(q_latent.dtype, jnp.floating) or not jnp.issubdtype(
        q_rope.dtype, jnp.floating
    ):
        raise TypeError("q_latent and q_rope must have floating dtypes")

    token_count = q_latent.shape[0]
    if physical_slots.ndim != 2 or physical_slots.shape[0] != token_count:
        raise ValueError("physical_slots must have shape [tokens, max_selected]")
    if physical_slots.dtype != jnp.int32:
        raise TypeError(f"physical_slots must have dtype int32; got {physical_slots.dtype}")
    if selected_counts.shape != (token_count,):
        raise ValueError(f"selected_counts must have shape {(token_count,)}")
    if selected_counts.dtype != jnp.int32:
        raise TypeError(f"selected_counts must have dtype int32; got {selected_counts.dtype}")
    if physical_slots.shape[1] == 0:
        raise ValueError("physical_slots must reserve at least one slot per token")

    scale = jnp.asarray(sm_scale, dtype=jnp.float32)
    if scale.ndim != 0:
        raise ValueError("sm_scale must be a scalar")

    max_selected = physical_slots.shape[1]
    capacity = cache.shape[0] * page_size
    if not isinstance(physical_slots, jax.core.Tracer) and not isinstance(
        selected_counts, jax.core.Tracer
    ):
        concrete_slots = np.asarray(jax.device_get(physical_slots))
        concrete_counts = np.asarray(jax.device_get(selected_counts))
        if np.any(concrete_counts < 0) or np.any(concrete_counts > max_selected):
            raise ValueError("selected_counts entries must be in [0, max_selected]")
        for token, count in enumerate(concrete_counts):
            counted_slots = concrete_slots[token, : int(count)]
            if np.any(counted_slots < 0) or np.any(counted_slots >= capacity):
                raise ValueError(
                    "counted physical_slots must be valid cache addresses; "
                    f"token={token}, count={int(count)}"
                )

    # selected_counts is the only validity mask. Slot values outside its prefix
    # may be arbitrary padding; addresses inside the prefix are an upstream ABI
    # obligation and are checked above whenever inputs are concrete.
    valid = jnp.arange(max_selected)[None, :] < selected_counts[:, None]
    safe_slots = jnp.where(valid, physical_slots, 0)

    packing = cache.shape[2]
    page = safe_slots // page_size
    offset = safe_slots % page_size
    row = offset // packing
    lane = offset % packing
    gathered = cache[page, row, lane].astype(jnp.float32)
    selected_latent = gathered[..., :latent_dim]
    selected_rope = gathered[..., latent_aligned : latent_aligned + rope_dim]

    q_latent_fp32 = q_latent.astype(jnp.float32)
    q_rope_fp32 = q_rope.astype(jnp.float32)
    scores = jnp.einsum("thc,tkc->thk", q_latent_fp32, selected_latent)
    scores += jnp.einsum("thr,tkr->thk", q_rope_fp32, selected_rope)
    scores *= scale

    valid_per_head = valid[:, None, :]
    masked_scores = jnp.where(valid_per_head, scores, -jnp.inf)
    has_selection = jnp.any(valid, axis=-1)[:, None, None]
    row_max = jnp.max(masked_scores, axis=-1, keepdims=True)
    row_max = jnp.where(has_selection, row_max, 0.0)
    exponentials = jnp.where(valid_per_head, jnp.exp(scores - row_max), 0.0)
    denominator = jnp.sum(exponentials, axis=-1, keepdims=True)
    weights = exponentials / jnp.where(denominator > 0, denominator, 1.0)
    return jnp.einsum("thk,tkc->thc", weights, selected_latent)


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
