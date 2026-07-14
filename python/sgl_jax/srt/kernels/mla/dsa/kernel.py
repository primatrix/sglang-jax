"""Pallas implementation of sparse DSA MLA decode attention.

The selected cache slots are consumed one at a time inside each batch program.
That deliberately avoids materialising a ``[batch, top_k, cache_width]`` gather
while retaining the caller's selected-slot order (including duplicates).
"""

from __future__ import annotations

import functools
import math

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

_ALIGNMENT = 128


def _align_to_128(dim: int) -> int:
    return ((dim + _ALIGNMENT - 1) // _ALIGNMENT) * _ALIGNMENT


def _is_floating_dtype(dtype: np.dtype) -> bool:
    return bool(np.issubdtype(dtype, np.floating) or dtype == jnp.bfloat16)


def _validate_inputs(
    ql_nope,
    q_pe,
    cache_kv,
    topk_slots,
    valid_counts,
    sm_scale: float,
) -> None:
    """Check the public contract on the host before launching a Pallas call."""
    ql_nope = np.asarray(ql_nope)
    q_pe = np.asarray(q_pe)
    cache_kv = np.asarray(cache_kv)
    topk_slots = np.asarray(topk_slots)
    valid_counts = np.asarray(valid_counts)
    sm_scale_array = np.asarray(sm_scale)

    if ql_nope.ndim != 3 or q_pe.ndim != 3:
        raise ValueError("ql_nope and q_pe must be rank-3 [batch, heads, width] arrays")
    if cache_kv.ndim != 4:
        raise ValueError("cache_kv must be rank-4 [pages, packed_rows, packing, width]")
    if topk_slots.ndim != 2:
        raise ValueError("topk_slots must be rank-2 [batch, max_selected] array")
    if valid_counts.ndim != 1:
        raise ValueError("valid_counts must be rank-1 [batch] array")
    if sm_scale_array.ndim != 0 or not np.issubdtype(sm_scale_array.dtype, np.number):
        raise ValueError("sm_scale must be a numeric scalar")
    if not math.isfinite(float(sm_scale_array)):
        raise ValueError("sm_scale must be finite")

    if ql_nope.shape[:2] != q_pe.shape[:2]:
        raise ValueError("ql_nope and q_pe must have matching batch and head dimensions")
    batch_size, num_heads, latent_dim = ql_nope.shape
    rope_dim = q_pe.shape[-1]
    if batch_size == 0 or num_heads == 0 or latent_dim == 0 or rope_dim == 0:
        raise ValueError("query dimensions must be nonzero")
    if topk_slots.shape[0] != batch_size or valid_counts.shape[0] != batch_size:
        raise ValueError("topk_slots and valid_counts must have one entry per batch item")
    if topk_slots.shape[1] == 0:
        raise ValueError("topk_slots must reserve at least one slot per batch item")
    if any(dim == 0 for dim in cache_kv.shape[:3]):
        raise ValueError("cache_kv pages, packed_rows, and packing must be nonzero")

    for name, array in (("ql_nope", ql_nope), ("q_pe", q_pe), ("cache_kv", cache_kv)):
        if not _is_floating_dtype(array.dtype):
            raise ValueError(f"{name} must have a floating-point dtype")
    if topk_slots.dtype != np.int32:
        raise ValueError("topk_slots must have dtype int32")
    if valid_counts.dtype != np.int32:
        raise ValueError("valid_counts must have dtype int32")

    expected_cache_width = _align_to_128(latent_dim) + _align_to_128(rope_dim)
    if cache_kv.shape[-1] != expected_cache_width:
        raise ValueError(
            "cache_kv width must equal the independently 128-aligned latent and rope widths"
        )

    max_selected = topk_slots.shape[1]
    if np.any(valid_counts <= 0) or np.any(valid_counts > max_selected):
        raise ValueError("valid_counts entries must be in [1, max_selected]")

    capacity = int(np.prod(cache_kv.shape[:3]))
    if np.any(topk_slots < -1):
        raise ValueError("topk_slots may not contain values below -1")
    for batch_index, valid_count in enumerate(valid_counts):
        valid_slots = topk_slots[batch_index, : int(valid_count)]
        if np.any(valid_slots < 0):
            raise ValueError("-1 is permitted only after valid_counts[batch]")
        if np.any(valid_slots >= capacity):
            raise ValueError("valid topk_slots must be within cache capacity")


def _dsa_decode_mla_kernel(
    ql_nope_ref,
    q_pe_ref,
    cache_kv_ref,
    topk_slots_ref,
    valid_counts_ref,
    output_ref,
    *,
    latent_dim: int,
    rope_dim: int,
    padded_latent_dim: int,
    padded_rope_dim: int,
    page_size: int,
    max_selected: int,
    sm_scale: float,
):
    """Process every selected cache vector for one batch element online."""
    batch_index = pl.program_id(0)
    num_heads = ql_nope_ref.shape[1]
    ql_nope = ql_nope_ref[batch_index].astype(jnp.float32)
    q_pe = q_pe_ref[batch_index].astype(jnp.float32)

    # The cache reserves independent 128-aligned segments for the latent and
    # RoPE dimensions.  Padding query segments separately preserves that layout.
    if latent_dim != padded_latent_dim:
        ql_nope = jnp.concatenate(
            (ql_nope, jnp.zeros((num_heads, padded_latent_dim - latent_dim), jnp.float32)),
            axis=-1,
        )
    if rope_dim != padded_rope_dim:
        q_pe = jnp.concatenate(
            (q_pe, jnp.zeros((num_heads, padded_rope_dim - rope_dim), jnp.float32)),
            axis=-1,
        )

    running_max = jnp.full((num_heads,), -jnp.inf, dtype=jnp.float32)
    running_sum = jnp.zeros((num_heads,), dtype=jnp.float32)
    running_value = jnp.zeros((num_heads, latent_dim), dtype=jnp.float32)
    valid_count = valid_counts_ref[batch_index]

    for selected_index in range(max_selected):

        def update(state, selected_index=selected_index):
            max_logits, exp_sums, weighted_values = state
            physical_slot = topk_slots_ref[batch_index, selected_index]
            page_index = physical_slot // page_size
            offset_in_page = physical_slot % page_size
            packed_row = offset_in_page // cache_kv_ref.shape[2]
            packed_offset = offset_in_page % cache_kv_ref.shape[2]

            # This is the only cache read for this selected slot.  The physical
            # slot mapping is page * (packed_rows * packing) + offset.
            cache_vector = cache_kv_ref[page_index, packed_row, packed_offset, :].astype(
                jnp.float32
            )
            scores = jnp.sum(ql_nope * cache_vector[:padded_latent_dim], axis=-1)
            scores += jnp.sum(q_pe * cache_vector[padded_latent_dim:], axis=-1)
            scores *= jnp.float32(sm_scale)

            new_max = jnp.maximum(max_logits, scores)
            old_scale = jnp.exp(max_logits - new_max)
            new_scale = jnp.exp(scores - new_max)
            new_sums = exp_sums * old_scale + new_scale
            values = cache_vector[:latent_dim]
            new_values = weighted_values * old_scale[:, None] + new_scale[:, None] * values
            return new_max, new_sums, new_values

        running_max, running_sum, running_value = lax.cond(
            selected_index < valid_count,
            update,
            lambda state: state,
            (running_max, running_sum, running_value),
        )

    output_ref[batch_index] = (running_value / running_sum[:, None]).astype(output_ref.dtype)


def dsa_decode_mla_attention(
    ql_nope: jax.Array,
    q_pe: jax.Array,
    cache_kv: jax.Array,
    topk_slots: jax.Array,
    valid_counts: jax.Array,
    *,
    sm_scale: float,
    interpret: bool = False,
    validate: bool = True,
) -> jax.Array:
    """Apply sparse DSA decode MLA attention to selected packed-cache slots.

    ``interpret=True`` executes Pallas locally and is intended for correctness
    tests.  A real Pallas launch requires a TPU; host validation is enabled by
    default and can be disabled by trusted benchmark callers.
    """
    if validate:
        _validate_inputs(ql_nope, q_pe, cache_kv, topk_slots, valid_counts, sm_scale)

    if not interpret and jax.default_backend() != "tpu":
        raise RuntimeError(
            "dsa_decode_mla_attention requires a TPU when interpret=False; "
            "use interpret=True for local execution"
        )

    ql_nope = jnp.asarray(ql_nope)
    q_pe = jnp.asarray(q_pe)
    cache_kv = jnp.asarray(cache_kv)
    topk_slots = jnp.asarray(topk_slots)
    valid_counts = jnp.asarray(valid_counts)

    batch_size, _num_heads, latent_dim = ql_nope.shape
    rope_dim = q_pe.shape[-1]
    padded_latent_dim = _align_to_128(latent_dim)
    padded_rope_dim = _align_to_128(rope_dim)
    page_size = cache_kv.shape[1] * cache_kv.shape[2]
    max_selected = topk_slots.shape[1]

    kernel = pl.pallas_call(
        functools.partial(
            _dsa_decode_mla_kernel,
            latent_dim=latent_dim,
            rope_dim=rope_dim,
            padded_latent_dim=padded_latent_dim,
            padded_rope_dim=padded_rope_dim,
            page_size=page_size,
            max_selected=max_selected,
            sm_scale=float(sm_scale),
        ),
        out_shape=jax.ShapeDtypeStruct(ql_nope.shape, ql_nope.dtype),
        grid=(batch_size,),
        in_specs=(
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.HBM),
        ),
        out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("arbitrary",)),
        interpret=interpret,
        name="dsa-decode-mla",
    )
    return kernel(
        ql_nope,
        q_pe,
        cache_kv,
        topk_slots,
        valid_counts,
    )


__all__ = ["dsa_decode_mla_attention"]
