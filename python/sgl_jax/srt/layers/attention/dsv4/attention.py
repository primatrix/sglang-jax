"""M2.4 -- SWA / CSA / HCA numerical attention.

The three V4 layer types are one formula. They differ only in which compressed
records a query is allowed to see:

===========  ==========================================================
layer        admitted compressed records
===========  ==========================================================
SWA-only     none (ratio 0 layers have no compressed history)
HCA (128)    every completed group
CSA (4)      the indexer's top-k among the completed groups (M2.3)
===========  ==========================================================

The keys are the **union** of the sliding window and the admitted records, not a
partition of history: at ratio == window == 128 a query at position 255 attends to
window tokens 128..255 *and* to the record of group 1, which covers those same
tokens. Confirmed against `kernels/hca/attention.py` and the independent fp32
oracle in `test/srt/kernels/hca/oracle.py`; see #352 for what happens when this is
guessed instead of read.

MLA shape: one tensor serves as both K and V, so the output is
``probs @ keys`` over the same rows the scores were computed from.

The attention sink is a phantom key that contributes to the **denominator only**::

    shift = max(scores.max(over keys), sink)      # per head
    probs = exp(scores - shift)
    out   = (probs @ keys) / (probs.sum(over keys) + exp(sink - shift))

Writing it as an extra key in the numerator would be wrong -- it has no value
vector. The `shift` taking the sink into account is what keeps `exp(sink - shift)`
from overflowing when the sink dominates.

This module never touches compressor state. The SWA cache write is a separate
function so that separation is structural rather than a comment: M2.2 owns state,
M2.4 owns the window.
"""

from __future__ import annotations

import jax.numpy as jnp

__all__ = [
    "admissible_mask",
    "dsv4_attention",
    "update_window_kv",
]

_NEG_INF = jnp.finfo(jnp.float32).min


def admissible_mask(
    *,
    query_positions,
    query_request_ids,
    valid_token_mask,
    window_positions,
    window_request_ids,
    compressed_entry_ids,
    compressed_request_ids,
    window_size: int,
    ratio: int,
    selected_entries=None,
):
    """Which keys each query may attend to.

    Returns ``(window_mask [T, W], compressed_mask [T, E])``.

    A window row is admissible when it belongs to the query's request and sits
    inside ``(position - window_size, position]`` -- causal on the right, window on
    the left.

    A compressed record is admissible when it belongs to the query's request and its
    group is **complete** at the query: ``entry_id < (position + 1) // ratio``. That
    is the same rule as `dsv4.metadata.visible_groups_for_positions` and
    `dsv4.indexer.visible_entries_for_query`. `ratio == 0` admits nothing.

    `selected_entries`, when given, is ``[T, k]`` of **row indices into the
    compressed key array** (exactly what `dsv4.indexer.csa_indexer_topk` returns for
    the same row ordering), with -1 for unused slots. Selection intersects the
    completeness rule rather than replacing it, so a stale or over-eager selection
    still cannot reach an unwritten group.
    """
    qpos = jnp.asarray(query_positions)[:, None]
    qreq = jnp.asarray(query_request_ids)[:, None]
    valid = jnp.asarray(valid_token_mask, bool)[:, None]

    wpos = jnp.asarray(window_positions)[None, :]
    wreq = jnp.asarray(window_request_ids)[None, :]
    window_mask = valid & (wreq == qreq) & (wpos <= qpos) & (wpos > qpos - window_size)

    entry_ids = jnp.asarray(compressed_entry_ids)[None, :]
    creq = jnp.asarray(compressed_request_ids)[None, :]
    if ratio <= 0:
        compressed_mask = jnp.zeros(window_mask.shape[:1] + entry_ids.shape[1:], bool)
    else:
        complete = entry_ids < ((qpos + 1) // ratio)
        compressed_mask = valid & (creq == qreq) & complete

    if selected_entries is not None:
        selected = jnp.asarray(selected_entries)
        num_entries = entry_ids.shape[1]
        rows = jnp.arange(num_entries, dtype=selected.dtype)[None, None, :]
        chosen = jnp.any((selected[:, :, None] == rows) & (selected[:, :, None] >= 0), axis=1)
        compressed_mask = compressed_mask & chosen

    return window_mask, compressed_mask


def dsv4_attention(
    q,
    window_kv,
    compressed_kv,
    *,
    query_positions,
    query_request_ids,
    valid_token_mask,
    window_positions,
    window_request_ids,
    compressed_entry_ids,
    compressed_request_ids,
    attention_sink,
    softmax_scale: float,
    window_size: int,
    ratio: int,
    selected_entries=None,
):
    """Attention output for one V4 layer.

    Args:
      q: ``[T, H, D]``.
      window_kv: ``[W, D]`` sliding-window KV rows (MLA: K and V are the same).
      compressed_kv: ``[E, D]`` compressed records gathered for this step.
      attention_sink: ``[H]`` per-head sink logit.
      ratio: 0 for SWA-only layers, 4 for CSA, 128 for HCA.
      selected_entries: CSA top-k row indices, or None to admit every completed
        record (HCA) / nothing (ratio 0).

    Returns:
      ``[T, H, D]`` float32. Padded query rows are zero.
    """
    q = jnp.asarray(q, jnp.float32)
    window_kv = jnp.asarray(window_kv, jnp.float32)
    compressed_kv = jnp.asarray(compressed_kv, jnp.float32)
    if q.ndim != 3:
        raise ValueError(f"q must be [T, H, D], got {q.shape}")
    if window_kv.ndim != 2 or window_kv.shape[-1] != q.shape[-1]:
        raise ValueError(f"window_kv must be [W, {q.shape[-1]}], got {window_kv.shape}")
    if compressed_kv.ndim != 2 or compressed_kv.shape[-1] != q.shape[-1]:
        raise ValueError(f"compressed_kv must be [E, {q.shape[-1]}], got {compressed_kv.shape}")
    sink = jnp.asarray(attention_sink, jnp.float32)
    if sink.shape != (q.shape[1],):
        raise ValueError(f"attention_sink must be [H] = [{q.shape[1]}], got {sink.shape}")

    window_mask, compressed_mask = admissible_mask(
        query_positions=query_positions,
        query_request_ids=query_request_ids,
        valid_token_mask=valid_token_mask,
        window_positions=window_positions,
        window_request_ids=window_request_ids,
        compressed_entry_ids=compressed_entry_ids,
        compressed_request_ids=compressed_request_ids,
        window_size=window_size,
        ratio=ratio,
        selected_entries=selected_entries,
    )

    keys = jnp.concatenate((window_kv, compressed_kv), axis=0)  # [W+E, D]
    mask = jnp.concatenate((window_mask, compressed_mask), axis=1)  # [T, W+E]

    scores = jnp.einsum("thd,kd->thk", q, keys, preferred_element_type=jnp.float32)
    scores = scores * softmax_scale
    scores = jnp.where(mask[:, None, :], scores, _NEG_INF)

    # The sink participates in the shift so `exp(sink - shift)` cannot overflow when
    # the sink is the largest logit, which is exactly when it matters.
    shift = jnp.maximum(jnp.max(scores, axis=-1), sink[None, :])[..., None]
    probs = jnp.where(mask[:, None, :], jnp.exp(scores - shift), 0.0)
    denominator = jnp.sum(probs, axis=-1, keepdims=True) + jnp.exp(sink[None, :, None] - shift)
    out = jnp.einsum("thk,kd->thd", probs, keys, preferred_element_type=jnp.float32)
    out = out / denominator
    return jnp.where(jnp.asarray(valid_token_mask, bool)[:, None, None], out, 0.0)


def update_window_kv(window_kv, new_kv, write_loc, valid_mask):
    """Scatter this step's KV into the sliding-window cache.

    Args:
      window_kv: ``[W, D]`` cache.
      new_kv: ``[T, D]`` rows for this step's tokens.
      write_loc: ``[T]`` destination row per token; padded tokens must carry a
        negative or out-of-range value.
      valid_mask: ``[T]``.

    Invalid rows are sent past the end of the cache and dropped rather than to row
    zero -- a duplicate write to row zero would race with whatever really lives
    there. Compressor state is untouched; that belongs to M2.2.
    """
    window_kv = jnp.asarray(window_kv)
    new_kv = jnp.asarray(new_kv, window_kv.dtype)
    loc = jnp.asarray(write_loc)
    keep = jnp.asarray(valid_mask, bool) & (loc >= 0) & (loc < window_kv.shape[0])
    loc = jnp.where(keep, loc, window_kv.shape[0])
    return window_kv.at[loc].set(new_kv, mode="drop")
