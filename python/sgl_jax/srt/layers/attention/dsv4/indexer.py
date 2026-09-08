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

import jax
import jax.numpy as jnp

__all__ = [
    "INVALID_ENTRY",
    "csa_indexer_scores",
    "csa_indexer_topk",
    "visible_entries_for_query",
]

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

    group = num_heads // kv_heads

    # [T, H, E] would be the natural intermediate but it is the memory blow-up the
    # DSA reference also avoids; accumulate over heads instead.
    def head_step(h, acc):
        q_h = jax.lax.dynamic_index_in_dim(q, h, axis=1, keepdims=False)  # [T, D]
        k_h = jax.lax.dynamic_index_in_dim(keys, h // group, axis=1, keepdims=False)  # [E, D]
        w_h = jax.lax.dynamic_index_in_dim(weights, h, axis=1, keepdims=True)  # [T, 1]
        inner = jnp.einsum("td,ed->te", q_h, k_h, preferred_element_type=jnp.float32)
        return acc + jax.nn.relu(inner) * w_h

    zeros = jnp.zeros((q.shape[0], keys.shape[0]), jnp.float32)
    return jax.lax.fori_loop(0, num_heads, head_step, zeros)


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
