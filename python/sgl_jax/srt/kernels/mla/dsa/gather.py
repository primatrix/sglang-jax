"""Selected-slot materialization helpers for sparse DSA MLA decode."""

from __future__ import annotations

import jax
import jax.numpy as jnp

_DEFAULT_GATHER_BLOCK = 128
_MIN_SC_VECTOR_WIDTH = 8


def _validate_gather_block(gather_block: int) -> None:
    if not isinstance(gather_block, int):
        raise ValueError("gather_block must be a Python integer")
    if gather_block <= 0 or gather_block % _MIN_SC_VECTOR_WIDTH:
        raise ValueError("gather_block must be a positive multiple of 8")


def prepare_safe_topk_slots(
    topk_slots: jax.Array,
    valid_counts: jax.Array,
    *,
    gather_block: int = _DEFAULT_GATHER_BLOCK,
) -> jax.Array:
    """Pad selected slots to a static gather block and make padding safe.

    Entries outside ``valid_counts`` and negative entries are replaced with
    physical slot zero. The validated public wrapper rejects negative valid
    entries; handling them here as padding additionally guarantees that an
    unchecked gather never uses a negative address.
    """
    _validate_gather_block(gather_block)
    topk_slots = jnp.asarray(topk_slots, dtype=jnp.int32)
    valid_counts = jnp.asarray(valid_counts, dtype=jnp.int32)

    max_selected = topk_slots.shape[1]
    padded_selected = (
        (max_selected + gather_block - 1) // gather_block
    ) * gather_block
    if padded_selected != max_selected:
        topk_slots = jnp.pad(
            topk_slots,
            ((0, 0), (0, padded_selected - max_selected)),
            constant_values=0,
        )

    positions = jnp.arange(padded_selected, dtype=jnp.int32)[None, :]
    is_valid = (positions < valid_counts[:, None]) & (topk_slots >= 0)
    return jnp.where(is_valid, topk_slots, jnp.int32(0))


def materialize_selected_kv_xla(
    cache_kv: jax.Array,
    topk_slots: jax.Array,
    valid_counts: jax.Array,
    *,
    gather_block: int = _DEFAULT_GATHER_BLOCK,
) -> jax.Array:
    """Materialize selected logical cache rows with a regular XLA gather."""
    cache_kv = jnp.asarray(cache_kv)
    safe_slots = prepare_safe_topk_slots(
        topk_slots, valid_counts, gather_block=gather_block
    )
    logical_cache = cache_kv.reshape((-1, cache_kv.shape[-1]))
    return logical_cache[safe_slots]


__all__ = ["materialize_selected_kv_xla", "prepare_safe_topk_slots"]
