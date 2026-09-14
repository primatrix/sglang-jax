"""CSA decode over request-local pages and exact selected cache slots."""

import os

import jax
import jax.numpy as jnp

from sgl_jax.srt.kernels.csa_decode import paged_csa_decode_scores
from sgl_jax.srt.kernels.dsa.streamindex_topk import select_topk_indices

_NEG_INF = jnp.finfo(jnp.float32).min

# Which indexer scores + selects the decode query's compressed entries:
#   "kernel": kernels/dsa/streamindex_topk over the request-local page table (same kernel
#             the prefill indexer uses; Pallas scoring + exact SparseCore selection);
#   "p370":   paged_csa_decode_scores (request-local Pallas scorer) + exact selector;
#   "auto":   "kernel" on TPU, "p370" elsewhere (the p370 scorer can run interpreted).
DECODE_INDEXER_BACKEND_ENV = "DSV4_DECODE_INDEXER_BACKEND"
_KERNEL_KV_PAGES_PER_BLOCK = int(os.environ.get("DSV4_INDEXER_KV_PAGES_PER_BLOCK", "64"))
_KERNEL_QUERIES_PER_BLOCK = (1, 64, 64)


def resolve_decode_indexer_backend(backend: str = "auto") -> str:
    if backend not in ("auto", "kernel", "p370"):
        raise ValueError(f"unknown CSA decode indexer backend {backend!r}")
    if backend == "auto":
        backend = os.environ.get(DECODE_INDEXER_BACKEND_ENV, "auto")
        if backend not in ("auto", "kernel", "p370"):
            raise ValueError(
                f"{DECODE_INDEXER_BACKEND_ENV}={backend!r} must be auto, kernel or p370"
            )
    if backend == "auto":
        return "kernel" if jax.default_backend() == "tpu" else "p370"
    return backend


def csa_decode_select_p370(
    index_q,
    index_weights,
    index_cache,
    pages,
    lengths,
    *,
    take,
    ratio,
    compressed_page_size,
    topk_backend="auto",
):
    """Request-local Pallas scorer + exact selector: ``(selected [B, take], valid [B, take])``."""
    del ratio
    scores = paged_csa_decode_scores(
        index_q,
        index_weights,
        index_cache.reshape(-1, compressed_page_size, index_cache.shape[-1]),
        lengths,
        pages,
        interpret=jax.default_backend() != "tpu",
    )
    return select_decode_entries(scores, lengths, take=take, topk_backend=topk_backend)


def csa_decode_select_kernel(
    index_q,
    index_weights,
    index_cache,
    pages,
    lengths,
    *,
    take,
    ratio,
    compressed_page_size,
    topk_backend="auto",
):
    """Same contract as `csa_decode_select_p370` through ``kernels/dsa/streamindex_topk``.

    ``pages`` is the request-local page table (``decode_page_indices``, one row per packed
    decode query, zeros for padded rows), which is exactly the paged layout the kernel
    reads; ``lengths`` are completed compressed groups, so ``seq_lens = lengths * ratio``
    makes the kernel's completed-groups rule admit exactly those entries. Selections come
    back in request-local entry coordinates, the same as the p370 path.
    """
    from sgl_jax.srt.kernels.dsa.streamindex_topk import streamindex_topk

    if compressed_page_size % 2:
        raise ValueError("compressed_page_size must be even for the paged indexer layout")
    num_queries = index_q.shape[0]
    lengths = jnp.asarray(lengths, jnp.int32)
    cache_kv = jnp.asarray(index_cache).reshape(-1, compressed_page_size // 2, 2, index_q.shape[-1])
    selected = streamindex_topk(
        q=jnp.asarray(index_q).astype(jnp.bfloat16),
        indexer_weights=jnp.asarray(index_weights, jnp.float32),
        cache_kv=cache_kv,
        seq_lens=lengths * ratio,
        page_indices=jnp.asarray(pages, jnp.int32).reshape(-1),
        cu_q_lens=jnp.arange(num_queries + 1, dtype=jnp.int32),
        distribution=jnp.asarray((num_queries, num_queries, num_queries), jnp.int32),
        k=take,
        compression_ratio=ratio,
        num_kv_pages_per_block=_KERNEL_KV_PAGES_PER_BLOCK,
        num_queries_per_block=_KERNEL_QUERIES_PER_BLOCK,
        topk_backend=topk_backend,
    )
    valid = (selected >= 0) & (selected < lengths[:, None])
    return selected, valid


def select_decode_entries(scores, lengths, *, take: int, topk_backend: str = "auto"):
    """Exact top-``take`` entries per decode query, ``(selected [B, take], valid [B, take])``.

    Uses the shared DSA exit-stage selector (SparseCore radix select on chips that have
    one, XLA otherwise) instead of a full-row sort. ``scores`` marks unusable entries
    with ``_NEG_INF``; entries at or beyond ``lengths`` are also invalid.
    """
    masked = jnp.where(scores > _NEG_INF, scores, -jnp.inf)
    selected = select_topk_indices(masked, take, backend=topk_backend)
    valid = (selected >= 0) & (selected < lengths[:, None])
    return selected, valid


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
    take = min(index_topk, pages.shape[1] * compressed_page_size)
    select = (
        csa_decode_select_kernel
        if resolve_decode_indexer_backend() == "kernel"
        else csa_decode_select_p370
    )
    selected, selected_valid = select(
        index_q,
        index_weights,
        index_cache,
        pages,
        lengths,
        take=take,
        ratio=ratio,
        compressed_page_size=compressed_page_size,
    )
    scores_width = pages.shape[1] * compressed_page_size
    # Preserve original entry order during attention and gather nearby slots
    # together. Exact top-k is unchanged; invalid selections sort to the end.
    selected = jnp.sort(jnp.where(selected_valid, selected, scores_width), axis=-1)
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
