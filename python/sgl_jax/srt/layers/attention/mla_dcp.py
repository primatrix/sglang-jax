"""Correctness-first decode context parallel attention for absorbed MLA.

The production MLA v2 Pallas kernel fuses cache update and attention and only
returns the normalized output. DCP needs the softmax state from every context
shard. This module provides a JAX implementation that keeps the same paged-cache
contract, computes a local online-softmax state, and combines it across the
``dcp`` mesh axis with ``pmax``/``psum``.

It is intentionally an initial implementation rather than a performance kernel:
the local attention materializes scores for the static batch bucket. DCP=1 keeps
using the existing Pallas path.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax

from sgl_jax.srt.kernels.mla.v2.kernel import align_to


def _sequence_ids(offsets: jax.Array, indices: jax.Array) -> jax.Array:
    """Map flattened indices to ragged sequence ids without dynamic shapes."""
    # Equivalent to searchsorted(offsets[1:], indices, side="right"), but the
    # compare/reduce form has predictable TPU lowering for small request buckets.
    return jnp.sum(indices[..., None] >= offsets[None, 1:], axis=-1).astype(jnp.int32)


def mla_dcp_attention_local(
    ql_nope: jax.Array,
    q_pe: jax.Array,
    new_kv_c: jax.Array,
    new_k_pe: jax.Array,
    cache_kv: jax.Array,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    cu_kv_lens: jax.Array,
    distribution: jax.Array,
    out_cache_loc: jax.Array,
    *,
    page_size: int,
    dcp_size: int,
    sm_scale: float,
    sliding_window: int | None,
    soft_cap: float | None,
    dcp_axis_name: str = "dcp",
) -> tuple[jax.Array, jax.Array]:
    """Run one local MLA context shard and exactly combine DCP softmax state.

    This function executes inside ``jax.shard_map``. ``cache_kv`` is already a
    local shard of global ``P("data", "dcp", None, None)`` storage. The packed
    words in that shard are treated as a dense sequence of owner-local token
    slots, where global position ``p`` maps to rank ``p % dcp_size`` and local
    slot ``p // dcp_size`` within its logical page.
    """
    if page_size % dcp_size != 0:
        raise ValueError(f"page_size={page_size} must be divisible by dcp_size={dcp_size}")
    if ql_nope.shape[:2] != q_pe.shape[:2]:
        raise ValueError("ql_nope and q_pe must have matching token/head dimensions")
    if ql_nope.shape[0] != new_kv_c.shape[0] or new_kv_c.shape[0] != new_k_pe.shape[0]:
        raise ValueError("Q and new latent-KV token dimensions must match")

    dcp_rank = lax.axis_index(dcp_axis_name)
    q_dtype = ql_nope.dtype
    cache_dtype = cache_kv.dtype
    actual_lkv_dim = ql_nope.shape[-1]
    actual_rope_dim = q_pe.shape[-1]
    lkv_dim = align_to(actual_lkv_dim, 128)
    rope_dim = align_to(actual_rope_dim, 128)

    if cache_kv.shape[-1] != lkv_dim + rope_dim:
        raise ValueError(
            "MLA cache feature dimension does not match padded latent+rope dimensions: "
            f"cache={cache_kv.shape[-1]}, expected={lkv_dim + rope_dim}."
        )

    local_page_size = page_size // dcp_size
    num_local_pages, local_page_words, kv_packing, _ = cache_kv.shape
    if local_page_words * kv_packing != local_page_size:
        raise ValueError(
            "Local MLA cache page does not match the DCP shard: "
            f"words={local_page_words}, packing={kv_packing}, "
            f"local_page_size={local_page_size}."
        )

    new_kv_c_padded = jnp.pad(
        new_kv_c,
        ((0, 0), (0, lkv_dim - actual_lkv_dim)),
        constant_values=0,
    )
    new_k_pe_padded = jnp.pad(
        new_k_pe,
        ((0, 0), (0, rope_dim - actual_rope_dim)),
        constant_values=0,
    )
    new_kv = jnp.concatenate((new_kv_c_padded, new_k_pe_padded), axis=-1).astype(cache_dtype)

    cache_tokens = cache_kv.reshape(num_local_pages, local_page_size, lkv_dim + rope_dim)

    def _write_token(i, cache):
        loc = out_cache_loc[i].astype(jnp.int32)
        page = loc // page_size
        offset = loc % page_size
        owner = offset % dcp_size
        local_offset = offset // dcp_size
        valid = jnp.logical_and(loc > 0, page < num_local_pages)
        should_write = jnp.logical_and(valid, owner == dcp_rank)

        def _write(c):
            return c.at[page, local_offset].set(new_kv[i])

        return lax.cond(should_write, _write, lambda c: c, cache)

    cache_tokens = lax.fori_loop(0, new_kv.shape[0], _write_token, cache_tokens)

    # Build a dense description of every rank-local KV slot. page_indices is
    # ragged by sequence; cu_kv_lens contains the page-aligned cumulative size.
    num_page_entries = page_indices.shape[0]
    page_entry = jnp.arange(num_page_entries, dtype=jnp.int32)
    local_offset = jnp.arange(local_page_size, dtype=jnp.int32)
    page_cu = cu_kv_lens // page_size
    kv_seq_for_page = _sequence_ids(page_cu, page_entry)
    max_num_seqs = kv_lens.shape[0]
    kv_seq_for_page = jnp.minimum(kv_seq_for_page, max_num_seqs - 1)
    num_seqs = distribution[-1].astype(jnp.int32)
    valid_page = jnp.logical_and(page_entry < page_cu[num_seqs], kv_seq_for_page < num_seqs)
    safe_page_ids = jnp.where(valid_page, page_indices, 0)
    safe_page_ids = jnp.clip(safe_page_ids, 0, num_local_pages - 1)

    gathered = cache_tokens[safe_page_ids]
    kv_seq = jnp.broadcast_to(kv_seq_for_page[:, None], gathered.shape[:2]).reshape(-1)
    kv_position = (
        (page_entry - page_cu[kv_seq_for_page])[:, None] * page_size
        + local_offset[None, :] * dcp_size
        + dcp_rank
    ).reshape(-1)
    valid_kv = jnp.broadcast_to(valid_page[:, None], gathered.shape[:2]).reshape(-1)
    valid_kv = jnp.logical_and(valid_kv, kv_position < kv_lens[kv_seq])
    flat_kv = gathered.reshape(-1, lkv_dim + rope_dim)
    local_kv_c = flat_kv[:, :lkv_dim]
    local_k_pe = flat_kv[:, lkv_dim:]

    ql_nope_padded = jnp.pad(
        ql_nope,
        ((0, 0), (0, 0), (0, lkv_dim - actual_lkv_dim)),
        constant_values=0,
    ).astype(jnp.float32)
    q_pe_padded = jnp.pad(
        q_pe,
        ((0, 0), (0, 0), (0, rope_dim - actual_rope_dim)),
        constant_values=0,
    ).astype(jnp.float32)

    num_queries = ql_nope.shape[0]
    q_index = jnp.arange(num_queries, dtype=jnp.int32)
    q_seq = _sequence_ids(cu_q_lens, q_index)
    q_seq = jnp.minimum(q_seq, max_num_seqs - 1)
    q_len = cu_q_lens[q_seq + 1] - cu_q_lens[q_seq]
    q_position = kv_lens[q_seq] - q_len + (q_index - cu_q_lens[q_seq])
    valid_q = jnp.logical_and(q_index < cu_q_lens[num_seqs], q_seq < num_seqs)

    scores = jnp.einsum(
        "qhl,kl->qhk",
        ql_nope_padded,
        local_kv_c.astype(jnp.float32),
        precision=lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )
    scores += jnp.einsum(
        "qhr,kr->qhk",
        q_pe_padded,
        local_k_pe.astype(jnp.float32),
        precision=lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )
    scores *= sm_scale
    if soft_cap is not None:
        scores = soft_cap * jnp.tanh(scores / soft_cap)

    visible = jnp.logical_and(
        q_seq[:, None] == kv_seq[None, :],
        kv_position[None, :] <= q_position[:, None],
    )
    visible = jnp.logical_and(visible, valid_q[:, None])
    visible = jnp.logical_and(visible, valid_kv[None, :])
    if sliding_window is not None:
        visible = jnp.logical_and(
            visible,
            kv_position[None, :] > q_position[:, None] - sliding_window,
        )

    scores = jnp.where(visible[:, None, :], scores, -jnp.inf)
    local_max = jnp.max(scores, axis=-1)
    safe_local_max = jnp.where(jnp.isfinite(local_max), local_max, 0.0)
    probs = jnp.where(
        visible[:, None, :],
        jnp.exp(scores - safe_local_max[..., None]),
        0.0,
    )
    local_sum = jnp.sum(probs, axis=-1)
    local_acc = jnp.einsum(
        "qhk,kl->qhl",
        probs,
        local_kv_c.astype(jnp.float32),
        precision=lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )

    global_max = lax.pmax(local_max, dcp_axis_name)
    local_weight = jnp.where(jnp.isfinite(local_max), jnp.exp(local_max - global_max), 0.0)
    global_sum = lax.psum(local_sum * local_weight, dcp_axis_name)
    global_acc = lax.psum(local_acc * local_weight[..., None], dcp_axis_name)
    output = jnp.where(
        global_sum[..., None] > 0,
        global_acc / jnp.maximum(global_sum[..., None], jnp.finfo(jnp.float32).tiny),
        0.0,
    )
    output = output[..., :actual_lkv_dim].astype(q_dtype)

    return output, cache_tokens.reshape(cache_kv.shape)
