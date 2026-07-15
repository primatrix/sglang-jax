"""Selected-slot materialization helpers for sparse DSA MLA decode."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

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


def _sparsecore_gather_kernel(cache_hbm_ref, slot_indices_ref, output_vmem_ref):
    """Gather major-dimension cache rows with one SparseCore indirect DMA."""
    pltpu.sync_copy(cache_hbm_ref.at[slot_indices_ref], output_vmem_ref)


def materialize_selected_kv_sparsecore_unchecked(
    cache_kv: jax.Array,
    safe_topk_slots: jax.Array,
    *,
    gather_block: int = _DEFAULT_GATHER_BLOCK,
) -> jax.Array:
    """Run SparseCore gather on already-safe, gather-block-padded slots."""
    _validate_gather_block(gather_block)
    if jax.default_backend() != "tpu":
        raise RuntimeError("SparseCore selected KV materialization requires a TPU")

    cache_kv = jnp.asarray(cache_kv)
    safe_topk_slots = jnp.asarray(safe_topk_slots, dtype=jnp.int32)
    batch_size, padded_selected = safe_topk_slots.shape
    if padded_selected % gather_block:
        raise ValueError("safe_topk_slots width must be divisible by gather_block")

    cache_width = cache_kv.shape[-1]
    logical_cache = cache_kv.reshape((-1, cache_width))
    out_shape = jax.ShapeDtypeStruct(
        (batch_size, padded_selected, cache_width), cache_kv.dtype
    )

    gather_call = pl.pallas_call(
        _sparsecore_gather_kernel,
        out_shape=out_shape,
        grid=(batch_size, padded_selected // gather_block),
        in_specs=(
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(
                (None, gather_block),
                lambda batch_index, block_index: (batch_index, block_index),
                memory_space=pltpu.VMEM,
            ),
        ),
        out_specs=pl.BlockSpec(
            (None, gather_block, cache_width),
            lambda batch_index, block_index: (batch_index, block_index, 0),
            memory_space=pltpu.VMEM,
        ),
        compiler_params=pltpu.CompilerParams(
            kernel_type=pltpu.KernelType.SC_VECTOR_SUBCORE
        ),
        name="dsa-selected-kv-sparsecore-gather",
    )
    compiled_gather = jax.jit(
        gather_call,
        compiler_options={"xla_tpu_use_tc_device_shape_on_sc": "false"},
    )
    return compiled_gather(logical_cache, safe_topk_slots)


def materialize_selected_kv_sparsecore(
    cache_kv: jax.Array,
    topk_slots: jax.Array,
    valid_counts: jax.Array,
    *,
    gather_block: int = _DEFAULT_GATHER_BLOCK,
) -> jax.Array:
    """Make selected slots safe, then materialize them with SparseCore."""
    safe_slots = prepare_safe_topk_slots(
        topk_slots, valid_counts, gather_block=gather_block
    )
    return materialize_selected_kv_sparsecore_unchecked(
        cache_kv, safe_slots, gather_block=gather_block
    )


__all__ = [
    "materialize_selected_kv_sparsecore",
    "materialize_selected_kv_sparsecore_unchecked",
    "materialize_selected_kv_xla",
    "prepare_safe_topk_slots",
]
