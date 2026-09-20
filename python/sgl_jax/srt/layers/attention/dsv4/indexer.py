"""M2.3 -- CSA indexer scoring and compressed-space top-k.

The ratio-4 (CSA) layers do not attend over their whole compressed history.
A lightning indexer scores every visible compressed record against the query and
only the top ``index_topk`` records go on to the sparse attention in M2.4.

Conventions, stated because the surrounding kernels disagree about them:

* **Index unit is a compressed entry**, not a token. Entry ``e`` covers original
  positions ``[e*ratio, (e+1)*ratio - 1]``.
* **Invalid selections are ``-1``**, packed at the end of each row. This matches
  ``kernels/dsa`` so a downstream gather can use one convention.
* **Scores are ``sum_h relu(q_h . k_{h_kv}) * w_h``** with the ReLU inside the head
  sum, and GQA head folding ``h_kv = h // (num_q_heads // num_kv_heads)``. Same
  as the DSA lightning indexer.

Why this is native JAX rather than ``kernels/dsa/streamindex_topk``
------------------------------------------------------------------
That kernel is the obvious candidate and the M2.3 brief points at it, but its
causal mask is ``entry * ratio <= query_position`` (see
``streamindex_topk_ref`` and the NumPy oracle in
``test/srt/kernels/dsa/test_streamindex_topk.py``). That admits the entry the
query *sits inside*, which for V4 is an entry whose record has not been written
yet -- the compressor only emits at a group boundary. So a query could select a
slot holding the previous occupant's data, and worse, the selection a query makes
would change once later tokens in the same chunk complete that group.

V4 needs completed groups only: ``(entry + 1) * ratio - 1 <= query_position``.
The difference is exactly the queries that are not the last token of their group,
which under chunked prefill is most of them, so it is not an edge case. The two
rules cannot be reconciled by rescaling ``seq_lens`` either, because that kernel
derives the query's absolute position from ``seq_lens`` as well.

The scoring half is identical, so if that kernel gains a completed-groups mask
this module becomes a thin wrapper over it.
"""

from __future__ import annotations

import functools
import os

import jax
import jax.numpy as jnp

__all__ = [
    "INDEXER_BACKEND_ENV",
    "INVALID_ENTRY",
    "csa_indexer_scores",
    "csa_indexer_topk",
    "csa_indexer_topk_kernel",
    "kernel_read_layout",
    "resolve_indexer_backend",
    "visible_entries_for_query",
]

# Which implementation scores and selects compressed entries during prefill:
#   "reference": the native-JAX path below (materializes [T, E] scores, lax.top_k);
#   "kernel":    kernels/dsa/streamindex_topk (Pallas scoring straight from the paged
#                indexer cache, completed-groups mask, exact SparseCore selection);
#   "auto":      "kernel" on TPU, "reference" elsewhere.
INDEXER_BACKEND_ENV = "DSV4_INDEXER_BACKEND"
# Kernel block sizes. The compressed page holds page_size // 4 = 32 entries, and the
# kernel needs whole 128-entry KV blocks, so kv pages per block must be a multiple of 4.
_KERNEL_KV_PAGES_PER_BLOCK = int(os.environ.get("DSV4_INDEXER_KV_PAGES_PER_BLOCK", "64"))
# ``DSV4_INDEXER_QUERIES_PER_BLOCK``: query rows per grid step of the prefill /
# mixed indexer kernel (decode stays at 1). 64 is the decode-era default; the
# 8K prefill scores block is [64, N] per step.
_KERNEL_QUERIES_PER_BLOCK = (
    1,
    int(os.environ.get("DSV4_INDEXER_QUERIES_PER_BLOCK", "64")),
    int(os.environ.get("DSV4_INDEXER_QUERIES_PER_BLOCK", "64")),
)

# Packed at the end of each top-k row. -1 rather than an out-of-range positive
# value because that is what kernels/dsa already emits and what the downstream
# sparse gather in M2.4 will check.
INVALID_ENTRY = -1

_NEG_INF = jnp.finfo(jnp.float32).min


def visible_entries_for_query(query_positions, ratio: int):
    """Compressed entries a query at each position may select: ``(position+1)//ratio``.

    An entry is selectable only once its group is **complete**, i.e. the token at
    ``(entry + 1) * ratio - 1`` has been consumed. Returns the count, so entry
    ids ``0 .. count-1`` are legal.

    This is what makes a selection stable: it depends only on tokens at or before
    the query, so later tokens in the same chunk cannot change it.

    Same rule as `dsv4.metadata.visible_groups_for_positions` and as the HCA path
    in `kernels/hca` -- kept as a jnp implementation here because this runs inside
    a jitted selection, while the metadata one is host-side NumPy. The equality is
    pinned by a test.
    """
    if ratio <= 0:
        raise ValueError(f"ratio must be positive, got {ratio}")
    return (jnp.asarray(query_positions) + 1) // ratio


def csa_indexer_scores(q, weights, keys, *, num_kv_heads: int | None = None):
    """Lightning-indexer scores for every query against every compressed entry.

    Args:
      q: ``[T, H, D]`` indexer query heads.
      weights: ``[T, H]`` per-head mixing weights.
      keys: ``[E, H_kv, D]`` or ``[E, D]`` compressed indexer keys. A 2-D array is
        treated as a single shared KV head.
      num_kv_heads: number of KV heads; inferred from `keys` when 3-D.

    Returns:
      ``[T, E]`` float32 scores.

    The ReLU sits *inside* the head sum, so the heads cannot be collapsed into one
    matmul; that is the DSA indexer's definition, not an accident.
    """
    q = jnp.asarray(q)
    weights = jnp.asarray(weights, jnp.float32)
    keys = jnp.asarray(keys)
    if q.ndim != 3:
        raise ValueError(f"q must be [T, H, D], got {q.shape}")
    if keys.ndim == 2:
        keys = keys[:, None, :]
    if keys.ndim != 3:
        raise ValueError(f"keys must be [E, H_kv, D] or [E, D], got {keys.shape}")
    num_heads, head_dim = q.shape[1], q.shape[2]
    if keys.shape[2] != head_dim:
        raise ValueError(f"key head_dim {keys.shape[2]} != query head_dim {head_dim}")
    kv_heads = keys.shape[1] if num_kv_heads is None else num_kv_heads
    if num_heads % kv_heads:
        raise ValueError(f"{num_heads} query heads do not fold onto {kv_heads} kv heads")
    if weights.shape != (q.shape[0], num_heads):
        raise ValueError(f"weights must be [T, H], got {weights.shape}")

    if kv_heads != keys.shape[1]:
        raise ValueError("num_kv_heads must match the key head dimension")
    group = num_heads // kv_heads
    if q.shape[0] == 0 or keys.shape[0] == 0:
        return jnp.zeros((q.shape[0], keys.shape[0]), jnp.float32)

    # As in the private DSA query-tile scorer, expose all heads to the MXU
    # together. Target an 8 MiB [Bq, H, E] FP32 temporary, with at least
    # one query per tile, instead of materializing all T queries at once.
    tile = min(32, max(1, (2 << 20) // max(1, num_heads * keys.shape[0])))
    padding = (-q.shape[0]) % tile
    tiled_q = jnp.pad(q, ((0, padding), (0, 0), (0, 0))).reshape(
        -1, tile, kv_heads, group, head_dim
    )
    tiled_w = jnp.pad(weights, ((0, padding), (0, 0))).reshape(-1, tile, kv_heads, group)

    def score_tile(inputs):
        queries, head_weights = inputs
        similarities = jnp.einsum(
            "tngd,end->tnge", queries, keys, preferred_element_type=jnp.float32
        )
        # Keep the FP32 weights out of a second MXU dot: its default
        # precision can round them to BF16 on TPU.
        return jnp.sum(jax.nn.relu(similarities) * head_weights[..., None], axis=(1, 2))

    scores = jax.lax.map(score_tile, (tiled_q, tiled_w))
    return scores.reshape(-1, keys.shape[0])[: q.shape[0]]


@functools.partial(jax.jit, static_argnames=("k", "ratio", "num_kv_heads"))
def csa_indexer_topk(
    q,
    weights,
    keys,
    query_positions,
    query_request_ids,
    entry_request_ids,
    valid_token_mask,
    *,
    entry_group_ids,
    k: int,
    ratio: int,
    num_kv_heads: int | None = None,
):
    """Select up to `k` compressed entries per query.

    Args:
      q: ``[T, H, D]`` indexer queries.
      weights: ``[T, H]`` per-head weights.
      keys: ``[E, H_kv, D]`` compressed indexer keys, gathered for this step.
      query_positions: ``[T]`` absolute position of each query in its request.
      query_request_ids: ``[T]`` request each query belongs to.
      entry_request_ids: ``[E]`` request each key row belongs to. Rows that belong
        to no request must carry a value no query has.
      valid_token_mask: ``[T]`` padded query slots are False.
      entry_group_ids: `[E]` group number within each row's own request,
        distinct from the gathered row index returned by top-k.
      k: selection budget (`index_topk`).
      ratio: compression ratio of these layers (4 for CSA).

    Returns:
      ``[T, k]`` int32 entry ids, `INVALID_ENTRY` padded at the end of each row.

    Three masks decide legality, and all three are necessary:
      * the entry belongs to the same request as the query -- otherwise one
        request reads another's history;
      * the entry's group is complete at or before the query -- so a selection
        never depends on a future token;
      * the query slot is real -- padded queries select nothing.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    q = jnp.asarray(q)
    query_positions = jnp.asarray(query_positions)
    query_request_ids = jnp.asarray(query_request_ids)
    entry_request_ids = jnp.asarray(entry_request_ids)
    valid_token_mask = jnp.asarray(valid_token_mask, bool)
    num_entries = jnp.asarray(keys).shape[0]

    scores = csa_indexer_scores(q, weights, keys, num_kv_heads=num_kv_heads)

    entry_ids = jnp.asarray(entry_group_ids, jnp.int32)[None, :]
    if entry_ids.shape[1] != num_entries:
        raise ValueError("entry_group_ids must describe every gathered key row")
    same_request = query_request_ids[:, None] == entry_request_ids[None, :]
    complete = (entry_ids >= 0) & (
        entry_ids < visible_entries_for_query(query_positions, ratio)[:, None]
    )
    legal = same_request & complete & valid_token_mask[:, None]

    scores = jnp.where(legal, scores, _NEG_INF)

    # Ask for min(k, E) so a history shorter than the budget does not force the
    # selector to invent entries; the row is then padded out to k.
    take = min(k, int(num_entries))
    values, indices = jax.lax.top_k(scores, take)
    legal_count = legal.sum(-1, keepdims=True)
    rank = jnp.arange(take, dtype=jnp.int32)[None, :]
    # Guard by rank *and* by score: an all-illegal row still yields `take`
    # indices from top_k, and they must not be mistaken for selections.
    keep = (rank < legal_count) & (values > _NEG_INF)
    selected = jnp.where(keep, indices.astype(jnp.int32), INVALID_ENTRY)

    if take < k:
        selected = jnp.pad(selected, ((0, 0), (0, k - take)), constant_values=INVALID_ENTRY)
    return selected


def resolve_indexer_backend(backend: str = "auto") -> str:
    """Resolve ``auto`` (env ``DSV4_INDEXER_BACKEND`` first, then the JAX backend)."""
    if backend not in ("auto", "kernel", "reference"):
        raise ValueError(f"unknown CSA indexer backend {backend!r}")
    if backend == "auto":
        backend = os.environ.get(INDEXER_BACKEND_ENV, "auto")
        if backend not in ("auto", "kernel", "reference"):
            raise ValueError(f"{INDEXER_BACKEND_ENV}={backend!r} must be auto, kernel or reference")
    if backend == "auto":
        return "kernel" if jax.default_backend() == "tpu" else "reference"
    return backend


def kernel_read_layout(compressed_rows, seq_lens, q_lens, *, ratio: int, compressed_page_size: int):
    """Per-request page table and gathered-row offsets from the flat read tables.

    ``read_tables`` lists every request's completed entries in order, request after
    request, at flat row ``page * compressed_page_size + entry % compressed_page_size``.
    So request ``r`` owns rows ``[offset_r, offset_r + count_r)`` of the gathered array,
    and the page of its ``p``-th page-aligned entry is that row's page.

    Returns ``(page_indices [B, pages_per_seq], offsets [B])``; entries past a request's
    count are clamped to real rows so every page index is a valid page, and the kernel
    never reads them because ``seq_lens`` bounds each request.
    """
    compressed_rows = jnp.asarray(compressed_rows, jnp.int32)
    seq_lens = jnp.asarray(seq_lens, jnp.int32)
    counts = jnp.where(jnp.asarray(q_lens) > 0, seq_lens // ratio, 0).astype(jnp.int32)
    offsets = jnp.cumsum(counts) - counts
    capacity = compressed_rows.shape[0]
    pages_per_seq = max(1, -(-capacity // compressed_page_size))
    first = (
        offsets[:, None]
        + jnp.arange(pages_per_seq, dtype=jnp.int32)[None, :] * compressed_page_size
    )
    first = jnp.clip(first, 0, capacity - 1)
    return compressed_rows[first] // compressed_page_size, offsets


def csa_indexer_topk_kernel(
    q,
    weights,
    indexer_buffer,
    *,
    compressed_rows,
    seq_lens,
    q_lens,
    cu_q_lens,
    query_request_ids,
    valid_token_mask,
    k: int,
    ratio: int,
    compressed_page_size: int,
    topk_backend: str = "auto",
    return_scores: bool = False,
):
    """`csa_indexer_topk` semantics through ``kernels/dsa/streamindex_topk``.

    Args:
      q: ``[T, H, D]`` indexer queries, request-major in ``cu_q_lens`` order.
      weights: ``[T, H]`` per-head weights.
      indexer_buffer: ``[pages * compressed_page_size, D]`` flat paged indexer cache
        (the ``kv_buffers["indexer"]`` view), already holding this step's records.
      compressed_rows: flat read table (see `kernel_read_layout`).
      seq_lens, q_lens, cu_q_lens: ``[B]``, ``[B]``, ``[B + 1]`` from the metadata.
      query_request_ids, valid_token_mask: ``[T]``.
      k, ratio, compressed_page_size: ``index_topk``, 4, ``page_size // ratio``.

    Returns:
      ``[T, k]`` int32 row indices into the gathered compressed key array (the same
      coordinates `csa_indexer_topk` returns), ``INVALID_ENTRY`` packed at the tail.
      With ``return_scores``: ``(scores [T, E_padded] f32, offsets [B] int32)`` -- the
      raw indexer scores (-inf where a query may not see the entry; column = entry
      index within the query's own request) and each request's first gathered row,
      for `membership_from_scores`.
    """
    from sgl_jax.srt.kernels.dsa.streamindex_topk import streamindex_topk

    if compressed_page_size % 2:
        raise ValueError("compressed_page_size must be even for the paged indexer layout")
    if _KERNEL_KV_PAGES_PER_BLOCK * compressed_page_size % 128:
        raise ValueError("kv pages per block times compressed page size must be a multiple of 128")
    q = jnp.asarray(q)
    num_requests = jnp.asarray(seq_lens).shape[0]
    pages, offsets = kernel_read_layout(
        compressed_rows, seq_lens, q_lens, ratio=ratio, compressed_page_size=compressed_page_size
    )
    cache_kv = jnp.asarray(indexer_buffer).reshape(-1, compressed_page_size // 2, 2, q.shape[-1])
    active = jnp.asarray(q_lens) > 0
    if return_scores:
        scores = streamindex_topk(
            q=q.astype(jnp.bfloat16),
            indexer_weights=jnp.asarray(weights, jnp.float32),
            cache_kv=cache_kv,
            seq_lens=jnp.where(active, jnp.asarray(seq_lens), 0).astype(jnp.int32),
            page_indices=pages.reshape(-1).astype(jnp.int32),
            cu_q_lens=jnp.asarray(cu_q_lens, jnp.int32),
            distribution=jnp.asarray((0, 0, num_requests), jnp.int32),
            k=k,
            compression_ratio=ratio,
            num_kv_pages_per_block=_KERNEL_KV_PAGES_PER_BLOCK,
            num_queries_per_block=_KERNEL_QUERIES_PER_BLOCK,
            topk_backend=topk_backend,
            return_scores=True,
        )
        return scores, offsets
    selected = streamindex_topk(
        q=q.astype(jnp.bfloat16),
        indexer_weights=jnp.asarray(weights, jnp.float32),
        cache_kv=cache_kv,
        seq_lens=jnp.where(active, jnp.asarray(seq_lens), 0).astype(jnp.int32),
        page_indices=pages.reshape(-1).astype(jnp.int32),
        cu_q_lens=jnp.asarray(cu_q_lens, jnp.int32),
        distribution=jnp.asarray((0, 0, num_requests), jnp.int32),
        k=k,
        compression_ratio=ratio,
        num_kv_pages_per_block=_KERNEL_KV_PAGES_PER_BLOCK,
        num_queries_per_block=_KERNEL_QUERIES_PER_BLOCK,
        topk_backend=topk_backend,
    )
    request = jnp.clip(jnp.asarray(query_request_ids), 0, num_requests - 1)
    keep = jnp.asarray(valid_token_mask, bool)[:, None] & (selected >= 0)
    return jnp.where(keep, selected + offsets[request][:, None], INVALID_ENTRY).astype(jnp.int32)


def membership_from_scores(
    scores,
    offsets,
    *,
    q_lens,
    query_request_ids,
    valid_token_mask,
    k: int,
    num_entries: int,
    topk_backend: str = "auto",
):
    """``[T, num_entries]`` bool top-k membership from the indexer scores.

    Two ways to the same mask, chosen at run time: when exactly one request is
    active its gathered rows start at 0, so a row's score column *is* its gathered
    row and the mask is ``score >= k-th largest`` (`kernels/dsv4/topk_threshold`,
    linear in E, no sort; ties admit every tied entry). Otherwise the index path is
    kept: `select_topk_indices` + per-request offsets + `packed_membership`.
    """
    from sgl_jax.srt.kernels.dsa.streamindex_topk import select_topk_indices
    from sgl_jax.srt.kernels.dsv4.topk_threshold import topk_membership_mask
    from sgl_jax.srt.layers.attention.dsv4.attention import packed_membership

    scores = jnp.asarray(scores, jnp.float32)
    offsets = jnp.asarray(offsets, jnp.int32)
    num_requests = offsets.shape[0]
    rows_valid = jnp.asarray(valid_token_mask, bool)[:, None]
    request = jnp.clip(jnp.asarray(query_request_ids), 0, num_requests - 1)

    def _fit(mask):
        width = mask.shape[1]
        if width >= num_entries:
            return mask[:, :num_entries]
        return jnp.pad(mask, ((0, 0), (0, num_entries - width)))

    def by_threshold(s):
        return _fit(topk_membership_mask(s, k)) & rows_valid

    def by_indices(s):
        selected = select_topk_indices(s, k, backend=topk_backend)[:, :k]
        keep = rows_valid & (selected >= 0)
        selected = jnp.where(keep, selected + offsets[request][:, None], INVALID_ENTRY)
        return packed_membership(selected.astype(jnp.int32), num_entries)

    single = jnp.sum((jnp.asarray(q_lens) > 0).astype(jnp.int32)) == 1
    return jax.lax.cond(single, by_threshold, by_indices, scores)
