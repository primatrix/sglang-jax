"""CSA decode over request-local pages and exact selected cache slots."""

import jax
import jax.numpy as jnp

from sgl_jax.srt.kernels.csa_decode import paged_csa_decode_scores

_NEG_INF = jnp.finfo(jnp.float32).min


def csa_decode_attention(
    q,
    index_q,
    index_weights,
    index_cache,
    compressed_cache,
    window_cache,
    pages,
    window_rows,
    *,
    query_positions,
    valid_token_mask,
    attention_sink,
    softmax_scale,
    compressed_page_size,
    index_topk,
    ratio,
):
    """Read selected compressed slots plus the SWA union, including the sink.

    Page and SWA addresses have one row per packed decode query; padded queries
    own no entries. Both caches already contain this step's compressor/KV writes.
    This keeps prefill on its shared-history path and avoids cross-request decode
    score matrices, global top-k, and gathering every compressed KV candidate.
    """
    lengths = jnp.where(valid_token_mask, (query_positions + 1) // ratio, 0)
    scores = paged_csa_decode_scores(
        index_q,
        index_weights,
        index_cache.reshape(-1, compressed_page_size, index_cache.shape[-1]),
        lengths,
        pages,
        interpret=jax.default_backend() != "tpu",
    )
    take = min(index_topk, scores.shape[-1])
    values, selected = jax.lax.top_k(scores, take)
    selected_valid = (selected < lengths[:, None]) & (values > _NEG_INF)
    # Preserve original entry order during attention and gather nearby slots
    # together. Exact top-k is unchanged; invalid selections sort to the end.
    selected = jnp.sort(jnp.where(selected_valid, selected, scores.shape[-1]), axis=-1)
    selected_valid = selected < lengths[:, None]
    safe_selected = jnp.where(selected_valid, selected, 0)
    physical_pages = jnp.take_along_axis(pages, safe_selected // compressed_page_size, axis=1)
    slots = physical_pages * compressed_page_size + safe_selected % compressed_page_size

    compressed = jnp.take(compressed_cache, slots, axis=0)
    window = jnp.take(window_cache, window_rows, axis=0)
    window_positions = (
        query_positions[:, None] - window_rows.shape[1] + 1 + jnp.arange(window_rows.shape[1])
    )
    window_valid = valid_token_mask[:, None] & (window_positions >= 0)
    mask = jnp.concatenate((window_valid, selected_valid), axis=1)
    keys = jnp.concatenate((window, compressed), axis=1).astype(jnp.float32)
    keys = jnp.where(mask[:, :, None], keys, 0.0)
    scores = jnp.einsum(
        "thd,tkd->thk", q.astype(jnp.float32), keys, preferred_element_type=jnp.float32
    )
    scores = jnp.where(mask[:, None, :], scores * softmax_scale, _NEG_INF)
    sink = attention_sink.astype(jnp.float32)[None, :, None]
    shift = jnp.maximum(jnp.max(scores, axis=-1, keepdims=True), sink)
    probs = jnp.where(mask[:, None, :], jnp.exp(scores - shift), 0.0)
    denominator = jnp.sum(probs, axis=-1, keepdims=True) + jnp.exp(sink - shift)
    out = jnp.einsum("thk,tkd->thd", probs, keys, preferred_element_type=jnp.float32) / denominator
    return jnp.where(valid_token_mask[:, None, None], out, 0.0)
