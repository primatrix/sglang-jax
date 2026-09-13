"""M2.2 -- offline compressor and continuation state.

One compressed record is produced per completed group. The record pools a window
of per-token *state* rows, normalises, and applies RoPE:

    kv     = x @ wkv^T                                  [T, coff*D]
    score  = x @ wgate^T + ape[position % ratio]        [T, coff*D]
    weight = softmax(score_window, axis=window)         per feature, over the window
    pooled = sum(weight * kv_window, axis=window)
    normed = pooled * rsqrt(mean(pooled^2) + eps) * norm_weight
    record = interleaved_rope(normed, cos_sin[compressed_pos], rope_head_dim)

Semantics are taken from tpu-inference's reference
(`kernels/experimental/deepseek_v4/compress_and_store/compress_store_ref.py`:
`gather_state_windows` / `compress_norm_rope`) and agree with the independent fp32
oracle that landed with the HCA kernels in `test/srt/kernels/hca/oracle.py`. They
are not re-derived here -- an earlier attempt to reason out a related rule from
first principles produced a wrong visibility mask that reached `epic/dsv4`.

Overlap: why a ratio-4 ring is eight deep
-----------------------------------------
CSA (ratio 4) compresses with *overlapping* windows and HCA (ratio 128) does not::

    coff   = 1 + int(overlap)          # CSA 2, HCA 1
    window = coff * ratio              # CSA 8, HCA 128

Each token stores ``coff`` content fields and ``coff`` score fields, each `head_dim`
wide. A record ending at position ``p`` pools the ``window`` token-states
``p-window+1 .. p``, reading **field 0 for the older half of the window and field 1
for the newer half**. That is the whole of "overlap": not two records, one record
over twice as many tokens with a different projection per half.

So the ring is exactly `window` deep, and `position % window` addresses it. That is
also why a long chunk must not be written into the ring before pooling: with
`window` = 8 and a 512-token chunk, the earliest groups' states would be overwritten
before they were ever read. This module computes from *old state plus this chunk's
projections* and only then saves the tail, which is the option the M2.2 brief
allows and the one that needs no ordering discipline from the caller.

State layout matches C1's pool (`DeepseekV4CompressStatePool`):
``[slot, window, 2*coff*head_dim]`` float32, contents in the first half of the last
axis and scores in the second, empty contents zero and empty scores ``-inf``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

__all__ = [
    "compress_chunk",
    "interleaved_rope",
    "overlap_factor",
    "pool_normalize_rope",
    "project_tokens",
    "state_window",
]


def overlap_factor(ratio: int) -> int:
    """``coff``: 2 for the overlapping CSA ratio, 1 otherwise.

    Mirrors tpu-inference's ``1 + int(overlap)`` with ``Mode.CSA`` selected by
    ratio 4, and ``coff = 1 + (compress_ratio == 4)`` in its V4 compressor.
    """
    if ratio <= 0:
        raise ValueError(f"ratio must be positive, got {ratio}")
    return 2 if ratio == 4 else 1


def state_window(ratio: int) -> int:
    """Token-states pooled into one record, and therefore the ring depth."""
    return overlap_factor(ratio) * ratio


def interleaved_rope(x, cos, sin, rope_head_dim: int):
    """Interleaved (GPT-J) RoPE on the trailing ``rope_head_dim`` features.

    Pairs are ``(even, odd)`` *within* the trailing block, which is why this cannot
    be written as a half-rotate over the whole head.
    """
    head_dim = x.shape[-1]
    if rope_head_dim % 2 or rope_head_dim > head_dim:
        raise ValueError(
            f"rope_head_dim must be even and <= head_dim; got {rope_head_dim} vs {head_dim}"
        )
    if rope_head_dim == 0:
        return x
    start = head_dim - rope_head_dim
    head, tail = x[..., :start], x[..., start:]
    even, odd = tail[..., 0::2], tail[..., 1::2]
    rotated_even = even * cos - odd * sin
    rotated_odd = even * sin + odd * cos
    tail = jnp.stack((rotated_even, rotated_odd), axis=-1).reshape(tail.shape)
    return jnp.concatenate((head, tail), axis=-1)


def project_tokens(x, wkv, wgate, ape, positions, *, ratio: int):
    """Per-token content and score projections.

    Args:
      x: ``[T, hidden]`` activations.
      wkv: ``[coff*D, hidden]`` content projection (checkpoint layout, ``[out, in]``).
      wgate: ``[coff*D, hidden]`` score projection.
      ape: ``[ratio, coff*D]`` absolute-position embedding added to the score,
        indexed by the token's offset *within its group*.
      positions: ``[T]`` absolute positions.
      ratio: compression ratio.

    Returns:
      ``(kv, score)``, each ``[T, coff*D]`` float32.

    The APE index is ``position % ratio``, not the position -- it encodes where in
    the group a token sits, so it repeats every group.
    """
    x = jnp.asarray(x, jnp.float32)
    coff_width = jnp.asarray(wkv).shape[0]
    if jnp.asarray(wgate).shape[0] != coff_width:
        raise ValueError("wkv and wgate must have the same output width")
    if jnp.asarray(ape).shape != (ratio, coff_width):
        raise ValueError(
            f"ape must be [ratio, coff*D] = [{ratio}, {coff_width}], got {jnp.asarray(ape).shape}"
        )
    kv = jnp.einsum(
        "th,oh->to", x, jnp.asarray(wkv, jnp.float32), precision=jax.lax.Precision.HIGHEST
    )
    score = jnp.einsum(
        "th,oh->to", x, jnp.asarray(wgate, jnp.float32), precision=jax.lax.Precision.HIGHEST
    )
    score = score + jnp.asarray(ape, jnp.float32)[jnp.asarray(positions) % ratio]
    return kv, score


def pool_normalize_rope(
    kv_window,
    score_window,
    valid_mask,
    norm_weight,
    cos,
    sin,
    *,
    rope_head_dim: int,
    norm_eps: float,
):
    """Window softmax-pool, RMSNorm, interleaved RoPE.

    Args:
      kv_window: ``[N, W, D]`` content rows of each record's window.
      score_window: ``[N, W, D]`` score rows.
      valid_mask: ``[N, W]`` False where the window runs off the start of the
        sequence (a record near position 0 pools fewer than W rows).
      cos, sin: ``[N, rope_head_dim//2]``.

    The softmax is per feature over the window axis, and masked entries go to
    ``-inf`` so they contribute nothing -- not zero, which would still take a share
    of the normalisation.
    """
    score_window = jnp.where(valid_mask[..., None], score_window, -jnp.inf)
    weights = jax.nn.softmax(score_window, axis=1)
    pooled = jnp.sum(weights * kv_window, axis=1)
    variance = jnp.mean(jnp.square(pooled), axis=-1, keepdims=True)
    normed = pooled * jax.lax.rsqrt(variance + norm_eps) * jnp.asarray(norm_weight, jnp.float32)
    return interleaved_rope(normed, cos, sin, rope_head_dim)


def _window_rows(state, chunk_rows, positions_in_window, chunk_index, from_chunk):
    """One window's rows, taken from the old ring or from this chunk.

    `state` is read as it was *before* this chunk, so there is no read-after-write
    hazard and the caller needs no ordering discipline.
    """
    window = positions_in_window.shape[-1]
    ring_slot = jnp.clip(jnp.mod(positions_in_window, window), 0, window - 1)
    from_state = jnp.take_along_axis(state, ring_slot[..., None], axis=1)
    from_new = chunk_rows[jnp.clip(chunk_index, 0, chunk_rows.shape[0] - 1)]
    return jnp.where(from_chunk[..., None], from_new, from_state)


def compress_chunk(
    x,
    *,
    wkv,
    wgate,
    ape,
    norm_weight,
    cos_sin_cache,
    state,
    positions,
    query_request_ids,
    prefix_lens,
    cu_q_lens,
    state_slots,
    boundary_token_indices,
    boundary_valid_mask,
    boundary_compressed_pos,
    ratio: int,
    head_dim: int,
    rope_head_dim: int = 64,
    norm_eps: float = 1e-6,
):
    """Compress one chunk: emit a record per completed group and update the state.

    Args:
      x: ``[T, hidden]`` activations for this chunk.
      state: ``[S, W, 2*coff*D]`` float32 state ring, **as of before this chunk**.
      positions: ``[T]`` absolute positions.
      query_request_ids: ``[T]``.
      prefix_lens, cu_q_lens: ``[B]`` / ``[B+1]``, to map an absolute position back
        to a chunk row.
      state_slots: ``[B]`` state-pool slot per request.
      boundary_token_indices, boundary_valid_mask: from M2.1's ratio metadata.
      boundary_compressed_pos: ``[N]`` original-token group starts used to index
        `cos_sin_cache`, not compressed group ids (for ratio 4: 0, 4, 8, ...).

    Returns:
      ``(records, record_valid, new_state)`` with ``records`` ``[N, D]``.

    Records for padded boundary slots are produced but masked by `record_valid`;
    the caller drops them rather than this function compacting, so the output shape
    stays a function of the padded metadata only.
    """
    coff = overlap_factor(ratio)
    window = state_window(ratio)
    width = coff * head_dim
    state = jnp.asarray(state, jnp.float32)
    if state.shape[1] != window or state.shape[2] != 2 * width:
        raise ValueError(
            f"state must be [S, {window}, {2 * width}] for ratio {ratio}, got {state.shape}"
        )

    kv, score = project_tokens(x, wkv, wgate, ape, positions, ratio=ratio)
    rows = jnp.concatenate((kv, score), axis=-1)  # [T, 2*width]

    positions = jnp.asarray(positions)
    query_request_ids = jnp.asarray(query_request_ids)
    prefix_lens = jnp.asarray(prefix_lens)
    cu_q_lens = jnp.asarray(cu_q_lens)
    state_slots = jnp.asarray(state_slots)
    bidx = jnp.asarray(boundary_token_indices)
    bvalid = jnp.asarray(boundary_valid_mask, bool)

    num_tokens = rows.shape[0]
    safe_bidx = jnp.clip(bidx, 0, num_tokens - 1)
    end_pos = positions[safe_bidx]  # [N] last position of each record's window
    req = query_request_ids[safe_bidx]  # [N]
    slot = state_slots[req]

    offsets = jnp.arange(window)
    win_pos = (end_pos - window + 1)[:, None] + offsets[None, :]  # [N, W]
    in_sequence = win_pos >= 0

    # A window position is in this chunk when it is at or after the chunk's first
    # position for that request; anything older can only come from the old ring.
    chunk_start = prefix_lens[req][:, None]
    from_chunk = in_sequence & (win_pos >= chunk_start)
    chunk_index = cu_q_lens[req][:, None] + (win_pos - chunk_start)

    slot_state = state[jnp.clip(slot, 0, state.shape[0] - 1)]  # [N, W, 2*width]
    combined = _window_rows(slot_state, rows, win_pos, chunk_index, from_chunk)

    # The overlap: the older half of the window reads content/score field 0, the
    # newer half reads field 1. With coff == 1 both halves read the same field.
    field = (
        (offsets >= ratio).astype(jnp.int32) * head_dim
        if coff == 2
        else jnp.zeros((window,), jnp.int32)
    )
    cols = field[None, :, None] + jnp.arange(head_dim)[None, None, :]
    kv_window = jnp.take_along_axis(combined[..., :width], cols, axis=2)
    score_window = jnp.take_along_axis(combined[..., width:], cols, axis=2)

    cos_sin = jnp.asarray(cos_sin_cache, jnp.float32)[jnp.asarray(boundary_compressed_pos)]
    half = rope_head_dim // 2
    cos, sin = cos_sin[:, :half], cos_sin[:, half : 2 * half]

    records = pool_normalize_rope(
        kv_window,
        score_window,
        in_sequence,
        norm_weight,
        cos,
        sin,
        rope_head_dim=rope_head_dim,
        norm_eps=norm_eps,
    )
    records = jnp.where(bvalid[:, None], records, 0.0)

    # Metadata pads query_request_ids with zero. Those rows must never write
    # request zero's ring, even when their zero position is inside its tail.
    valid_tokens = jnp.arange(num_tokens) < cu_q_lens[-1]
    new_state = _save_tail(
        state, rows, positions, query_request_ids, state_slots, window, valid_tokens
    )
    return records, bvalid, new_state


def _save_tail(state, rows, positions, query_request_ids, state_slots, window, valid_tokens):
    """Write only each request's last `window` chunk tokens into its ring.

    Restricting the write to the tail is what makes a chunk longer than the ring
    safe: without it, several chunk tokens map to the same ring slot and the
    surviving value is whichever scatter happened to land last.
    """
    token_slot = state_slots[query_request_ids]
    # Distance from the end of this request's contribution, computed from
    # positions so it needs no per-request loop: the last token of a request has
    # the largest position among that request's tokens.
    last_position = jax.ops.segment_max(
        jnp.where(valid_tokens, positions, -1),
        query_request_ids,
        num_segments=state_slots.shape[0],
        indices_are_sorted=False,
    )
    from_end = last_position[query_request_ids] - positions
    keep = valid_tokens & (from_end < window)
    flat_index = token_slot * window + jnp.mod(positions, window)
    flat = state.reshape(-1, state.shape[-1])
    flat_index = jnp.where(keep, flat_index, flat.shape[0])
    return flat.at[flat_index].set(rows, mode="drop").reshape(state.shape)
