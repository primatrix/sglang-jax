"""Composed SparseCore-gather plus TensorCore DSA MLA decode attention."""

from __future__ import annotations

import functools
import math
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.kernels.mla.dsa.attention import (
    selected_mla_attention_unchecked,
)
from sgl_jax.srt.kernels.mla.dsa.gather import (
    SPARSECORE_COMPILER_OPTIONS,
    materialize_selected_kv_sparsecore_pipeline_unchecked,
    materialize_selected_kv_sparsecore_unchecked,
    materialize_selected_kv_xla,
    prepare_safe_topk_slots,
)

_ALIGNMENT = 128
_DEFAULT_GATHER_BLOCK = 128
GatherImplementation = Literal[
    "auto", "sparsecore", "sparsecore-pipeline", "xla"
]


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
    """Check the public contract on the host before launching either stage."""
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

    for name, array in (
        ("ql_nope", ql_nope),
        ("q_pe", q_pe),
        ("cache_kv", cache_kv),
    ):
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


def _resolve_gather_impl(
    gather_impl: GatherImplementation, *, interpret: bool
) -> Literal["sparsecore", "sparsecore-pipeline", "xla"]:
    if gather_impl not in ("auto", "sparsecore", "sparsecore-pipeline", "xla"):
        raise ValueError(
            "gather_impl must be 'auto', 'sparsecore', "
            "'sparsecore-pipeline', or 'xla'"
        )
    if gather_impl == "auto":
        return "xla" if interpret else "sparsecore"
    if gather_impl in ("sparsecore", "sparsecore-pipeline") and interpret:
        raise ValueError("SparseCore gather does not support Pallas interpret mode")
    return gather_impl


def dsa_decode_mla_attention_unchecked(
    ql_nope: jax.Array,
    q_pe: jax.Array,
    cache_kv: jax.Array,
    topk_slots: jax.Array,
    valid_counts: jax.Array,
    *,
    sm_scale: float,
    interpret: bool = False,
    gather_impl: GatherImplementation = "auto",
    gather_block: int = _DEFAULT_GATHER_BLOCK,
) -> jax.Array:
    """Compose selected-KV materialization and attention without host checks."""
    resolved_gather = _resolve_gather_impl(gather_impl, interpret=interpret)
    if resolved_gather in ("sparsecore", "sparsecore-pipeline"):
        safe_slots = prepare_safe_topk_slots(
            topk_slots,
            valid_counts,
            gather_block=gather_block,
        )
        if resolved_gather == "sparsecore-pipeline":
            selected_kv = materialize_selected_kv_sparsecore_pipeline_unchecked(
                cache_kv,
                safe_slots,
                gather_block=gather_block,
            )
        else:
            selected_kv = materialize_selected_kv_sparsecore_unchecked(
                cache_kv,
                safe_slots,
                gather_block=gather_block,
            )
    else:
        selected_kv = materialize_selected_kv_xla(
            cache_kv,
            topk_slots,
            valid_counts,
            gather_block=gather_block,
        )

    return selected_mla_attention_unchecked(
        ql_nope,
        q_pe,
        selected_kv,
        valid_counts,
        sm_scale=sm_scale,
        interpret=interpret,
    )


@functools.cache
def _sparsecore_composed_launcher(sm_scale: float, gather_block: int):
    launch = functools.partial(
        dsa_decode_mla_attention_unchecked,
        sm_scale=sm_scale,
        interpret=False,
        gather_impl="sparsecore",
        gather_block=gather_block,
    )
    return jax.jit(launch, compiler_options=SPARSECORE_COMPILER_OPTIONS)


@functools.cache
def _sparsecore_pipeline_composed_launcher(sm_scale: float, gather_block: int):
    launch = functools.partial(
        dsa_decode_mla_attention_unchecked,
        sm_scale=sm_scale,
        interpret=False,
        gather_impl="sparsecore-pipeline",
        gather_block=gather_block,
    )
    return jax.jit(launch)


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
    gather_impl: GatherImplementation = "auto",
    gather_block: int = _DEFAULT_GATHER_BLOCK,
) -> jax.Array:
    """Apply sparse DSA decode MLA attention to selected physical slots."""
    _resolve_gather_impl(gather_impl, interpret=interpret)
    if validate:
        _validate_inputs(
            ql_nope,
            q_pe,
            cache_kv,
            topk_slots,
            valid_counts,
            sm_scale,
        )
    resolved_gather = _resolve_gather_impl(gather_impl, interpret=interpret)
    if resolved_gather == "sparsecore":
        return _sparsecore_composed_launcher(float(sm_scale), gather_block)(
            ql_nope,
            q_pe,
            cache_kv,
            topk_slots,
            valid_counts,
        )
    if resolved_gather == "sparsecore-pipeline":
        return _sparsecore_pipeline_composed_launcher(
            float(sm_scale), gather_block
        )(
            ql_nope,
            q_pe,
            cache_kv,
            topk_slots,
            valid_counts,
        )
    return dsa_decode_mla_attention_unchecked(
        ql_nope,
        q_pe,
        cache_kv,
        topk_slots,
        valid_counts,
        sm_scale=sm_scale,
        interpret=interpret,
        gather_impl=gather_impl,
        gather_block=gather_block,
    )


__all__ = ["dsa_decode_mla_attention", "dsa_decode_mla_attention_unchecked"]
