"""Host-side, NumPy references for selected-slot DSA decode MLA.

These routines intentionally do not use Pallas or JAX transformations.  They
are correctness oracles for a decode batch with one query per sequence.  A
physical slot is laid out as ``page * page_size + offset`` where
``page_size = packed_rows * packing``.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

_ALIGNMENT = 128


def _align_to_128(dim: int) -> int:
    return ((dim + _ALIGNMENT - 1) // _ALIGNMENT) * _ALIGNMENT


def _is_floating_dtype(dtype: np.dtype) -> bool:
    """Recognize NumPy floating types plus JAX's host-materialized BF16."""
    return bool(np.issubdtype(dtype, np.floating) or dtype == jnp.bfloat16)


def _validate_inputs(
    ql_nope: np.ndarray,
    q_pe: np.ndarray,
    cache_kv: np.ndarray,
    selected_slots: np.ndarray,
    valid_counts: np.ndarray,
    sm_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int, np.float32]:
    """Materialize and validate every prototype input on the host."""
    ql_nope = np.asarray(ql_nope)
    q_pe = np.asarray(q_pe)
    cache_kv = np.asarray(cache_kv)
    selected_slots = np.asarray(selected_slots)
    valid_counts = np.asarray(valid_counts)
    sm_scale = np.asarray(sm_scale)

    if ql_nope.ndim != 3 or q_pe.ndim != 3:
        raise ValueError("ql_nope and q_pe must be rank-3 [batch, heads, width] arrays")
    if cache_kv.ndim != 4:
        raise ValueError("cache_kv must be rank-4 [pages, packed_rows, packing, width]")
    if selected_slots.ndim != 2:
        raise ValueError("selected_slots must be rank-2 [batch, max_selected] array")
    if valid_counts.ndim != 1:
        raise ValueError("valid_counts must be rank-1 [batch] array")
    if sm_scale.ndim != 0 or not np.issubdtype(sm_scale.dtype, np.number):
        raise ValueError("sm_scale must be a numeric scalar")
    sm_scale = np.float32(sm_scale)
    if not np.isfinite(sm_scale):
        raise ValueError("sm_scale must be finite")

    if ql_nope.shape[:2] != q_pe.shape[:2]:
        raise ValueError("ql_nope and q_pe must have matching batch and head dimensions")
    batch_size, num_heads, lkv_dim = ql_nope.shape
    rope_dim = q_pe.shape[-1]
    if batch_size == 0 or num_heads == 0 or lkv_dim == 0 or rope_dim == 0:
        raise ValueError("query dimensions must be nonzero")
    if selected_slots.shape[0] != batch_size or valid_counts.shape[0] != batch_size:
        raise ValueError("selected_slots and valid_counts must have one entry per batch item")
    if selected_slots.shape[1] == 0:
        raise ValueError("selected_slots must reserve at least one slot per batch item")
    if any(dim == 0 for dim in cache_kv.shape[:3]):
        raise ValueError("cache_kv pages, packed_rows, and packing must be nonzero")

    for name, array in (("ql_nope", ql_nope), ("q_pe", q_pe), ("cache_kv", cache_kv)):
        if not _is_floating_dtype(array.dtype):
            raise ValueError(f"{name} must have a floating-point dtype")
    if selected_slots.dtype != np.int32:
        raise ValueError("selected_slots must have dtype int32")
    if valid_counts.dtype != np.int32:
        raise ValueError("valid_counts must have dtype int32")

    lkv_padded = _align_to_128(lkv_dim)
    rope_padded = _align_to_128(rope_dim)
    if cache_kv.shape[-1] != lkv_padded + rope_padded:
        raise ValueError(
            "cache_kv width must equal the independently 128-aligned latent and rope widths"
        )

    max_selected = selected_slots.shape[1]
    if np.any(valid_counts < 0) or np.any(valid_counts > max_selected):
        raise ValueError("valid_counts entries must be in [0, max_selected]")

    capacity = int(np.prod(cache_kv.shape[:3]))
    if np.any(selected_slots < -1):
        raise ValueError("selected_slots may not contain values below -1")
    for batch_index, valid_count in enumerate(valid_counts):
        valid_slots = selected_slots[batch_index, : int(valid_count)]
        if np.any(valid_slots < 0):
            raise ValueError("-1 is permitted only after valid_counts[batch]")
        if np.any(valid_slots >= capacity):
            raise ValueError("valid selected_slots must be within cache capacity")

    return (
        ql_nope.astype(np.float32, copy=False),
        q_pe.astype(np.float32, copy=False),
        cache_kv.astype(np.float32, copy=False),
        selected_slots,
        valid_counts,
        lkv_padded,
        rope_padded,
        sm_scale,
    )


def reference_dsa_decode_mla_attention(
    ql_nope: np.ndarray,
    q_pe: np.ndarray,
    cache_kv: np.ndarray,
    selected_slots: np.ndarray,
    valid_counts: np.ndarray,
    *,
    sm_scale: float,
) -> np.ndarray:
    """Compute selected-slot decode MLA by explicitly decoding physical slots.

    Returns an FP32 ``[batch, heads, lkv_dim]`` latent output.  The selected
    slots are gathered in caller order, so repeated and nonmonotonic slots are
    both valid.
    """
    (
        ql_nope,
        q_pe,
        cache_kv,
        selected_slots,
        valid_counts,
        lkv_padded,
        rope_padded,
        sm_scale,
    ) = _validate_inputs(ql_nope, q_pe, cache_kv, selected_slots, valid_counts, sm_scale)

    lkv_dim = ql_nope.shape[-1]
    if lkv_dim != lkv_padded:
        ql_nope = np.pad(ql_nope, ((0, 0), (0, 0), (0, lkv_padded - lkv_dim)))
    if q_pe.shape[-1] != rope_padded:
        q_pe = np.pad(q_pe, ((0, 0), (0, 0), (0, rope_padded - q_pe.shape[-1])))

    batch_size, num_heads, _ = ql_nope.shape
    packed_rows = cache_kv.shape[1]
    packing = cache_kv.shape[2]
    page_size = packed_rows * packing
    output = np.zeros((batch_size, num_heads, lkv_dim), dtype=np.float32)

    for batch_index in range(batch_size):
        valid_count = int(valid_counts[batch_index])
        if valid_count == 0:
            continue
        gathered = np.empty((valid_count, cache_kv.shape[-1]), dtype=np.float32)
        for gathered_index, physical_slot in enumerate(selected_slots[batch_index, :valid_count]):
            page, offset = divmod(int(physical_slot), page_size)
            packed_row, packed_offset = divmod(offset, packing)
            gathered[gathered_index] = cache_kv[page, packed_row, packed_offset]

        query = np.concatenate((ql_nope[batch_index], q_pe[batch_index]), axis=-1)
        scores = query @ gathered.T
        scores *= sm_scale
        scores -= np.max(scores, axis=-1, keepdims=True)
        weights = np.exp(scores)
        weights /= np.sum(weights, axis=-1, keepdims=True)
        output[batch_index] = weights @ gathered[:, :lkv_dim]

    return output


def dense_selected_mla_attention(
    ql_nope: np.ndarray,
    q_pe: np.ndarray,
    cache_kv: np.ndarray,
    selected_slots: np.ndarray,
    valid_counts: np.ndarray,
    *,
    sm_scale: float,
) -> np.ndarray:
    """Compute selected-slot decode MLA through an independently dense gather.

    Flattening the packed cache makes physical slot IDs direct dense indices;
    this provides a separate oracle for the explicit page/row/packing mapping
    in :func:`reference_dsa_decode_mla_attention`.
    """
    (
        ql_nope,
        q_pe,
        cache_kv,
        selected_slots,
        valid_counts,
        lkv_padded,
        rope_padded,
        sm_scale,
    ) = _validate_inputs(ql_nope, q_pe, cache_kv, selected_slots, valid_counts, sm_scale)

    latent_width = ql_nope.shape[-1]
    padded_nope = np.zeros((*ql_nope.shape[:2], lkv_padded), dtype=np.float32)
    padded_nope[..., :latent_width] = ql_nope
    padded_rope = np.zeros((*q_pe.shape[:2], rope_padded), dtype=np.float32)
    padded_rope[..., : q_pe.shape[-1]] = q_pe

    batch_size, num_heads, _ = padded_nope.shape
    dense_cache = cache_kv.reshape(-1, cache_kv.shape[-1])
    output = np.zeros((batch_size, num_heads, latent_width), dtype=np.float32)

    for batch_index in range(batch_size):
        valid_count = int(valid_counts[batch_index])
        if valid_count == 0:
            continue
        gathered = dense_cache[selected_slots[batch_index, :valid_count]]
        query = np.concatenate((padded_nope[batch_index], padded_rope[batch_index]), axis=-1)
        logits = query @ gathered.T
        logits *= sm_scale
        shifted_logits = logits - np.max(logits, axis=-1, keepdims=True)
        probabilities = np.exp(shifted_logits)
        probabilities /= np.sum(probabilities, axis=-1, keepdims=True)
        output[batch_index] = probabilities @ gathered[:, :latent_width]

    return output
