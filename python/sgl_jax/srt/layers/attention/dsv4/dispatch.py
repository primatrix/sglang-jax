"""M2.5 -- layer dispatch: compose M2.1-M2.4 over C1's resources.

C3 (`dsv4/runtime.py`) already bridges `ForwardBatch` to metadata and reconciled the
three metadata paths into one pytree, and its consumer raises for anything that is
not a C128 HCA layer. This module supplies the missing dispatch: the three V4 layer
types over C1's pools, composed from the pieces that landed separately.

    ratio 0    SWA-only      window attention                          (M2.4)
    ratio 4    CSA           compress -> indexer top-k -> attention    (M2.2, M2.3, M2.4)
    ratio 128  HCA           compress -> attention over all complete   (M2.2, M2.4)

Two halves, deliberately separated so each is testable on its own:

**Host** (`read_tables`) turns C1's ownership map into flat read addresses. C1 owns
`req_to_token` and the allocator's original-token -> SWA mapping; nothing here
invents an address. `deepseek_v4_hca_backend._page_tables` does the same job at page
granularity for its Pallas kernel; this produces per-row indices because the M2.4
path is dense native JAX.

**Device** (`run_layer`) is pure JAX over gathered arrays and the metadata.

What this module does not do: the Q/KV/output projections. `wq_a`/`wq_b`/`wkv` and
the grouped output LoRA (`wo_a`/`wo_b`, `o_groups`) are per-layer weights the model
owns, and the M2.5 brief scopes this to *composing* the M2.1-M2.4 calls. `run_layer`
therefore takes projected Q and KV, exactly as M2.4 does. The grouped output LoRA in
particular has no implementation anywhere in the repo yet.

Production HCA stays on the Pallas kernel from #349 when one is available; the native
path here covers all three ratios so that CPU tests can exercise the dispatch and so
that the Pallas path has a same-repo reference to be compared against.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.layers.attention.dsv4.attention import dsv4_attention, update_window_kv
from sgl_jax.srt.layers.attention.dsv4.compressor import compress_chunk
from sgl_jax.srt.layers.attention.dsv4.indexer import csa_indexer_topk

__all__ = [
    "ReadTables",
    "read_tables",
    "run_layer",
]


@jax.tree_util.register_pytree_node_class
class ReadTables:
    """Flat read addresses for one step, one DP rank.

    Attributes:
      window_rows: SWA physical rows every query in this step may need.
      window_positions / window_request_ids: what each row is.
      compressed_rows: flat compressed-entry addresses (C1's ``loc // ratio``).
      compressed_entry_ids / compressed_request_ids: what each entry is.

    Rows are a superset of any single query's window -- narrowing per query is the
    mask's job in M2.4, not this table's.
    """

    __slots__ = (
        "window_rows",
        "window_positions",
        "window_request_ids",
        "compressed_rows",
        "compressed_entry_ids",
        "compressed_request_ids",
        "decode_page_indices",
        "decode_window_rows",
    )

    def __init__(self, **kw):
        for name in self.__slots__:
            setattr(self, name, kw.get(name) if name.startswith("decode_") else kw[name])

    def tree_flatten(self):
        return tuple(getattr(self, name) for name in self.__slots__), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(**dict(zip(cls.__slots__, children, strict=True)))

    def __repr__(self):  # pragma: no cover - debugging aid
        return f"ReadTables(window={len(self.window_rows)}, compressed={len(self.compressed_rows)})"


def read_tables(
    *,
    request_pool,
    allocator,
    slots,
    lengths,
    q_lens,
    ratio: int,
    window_size: int,
    page_size: int,
    rank: int = 0,
) -> ReadTables:
    """Build the step's read addresses from C1's ownership map.

    Args:
      request_pool: C1's `ReqToTokenPool`; `req_to_token[slot, p]` is the
        original-token location of position `p`.
      allocator: C1's V4 allocator, for `full_to_swa_index_mapping`.
      slots: [B] request slots. lengths: [B] seq_lens after this step.
      q_lens: [B] query tokens this step (0 for an inactive request).
      ratio: 0, 4 or 128.

    Raises rather than guessing when a required SWA page has been released -- a
    zero in the mapping means "not allocated", and reading row 0 would silently
    return another request's data.
    """
    mapping = allocator.full_to_swa_index_mapping
    mapping = mapping[rank] if isinstance(mapping, list) else mapping
    slots = np.asarray(slots, np.int64)
    lengths = np.asarray(lengths, np.int64)
    q_lens = np.asarray(q_lens, np.int64)

    w_rows, w_pos, w_req = [], [], []
    c_rows, c_ids, c_req = [], [], []
    for r, (slot, length, n) in enumerate(zip(slots, lengths, q_lens, strict=True)):
        if not n:
            continue
        last = int(length) - 1
        # Every position any query of this request may reach: its own window plus the
        # window of the oldest query in the chunk.
        first = max(0, int(length) - int(n) - window_size + 1)
        positions = np.arange(first, last + 1, dtype=np.int64)
        locations = np.asarray(request_pool.req_to_token[int(slot), positions], np.int64)
        if np.any(locations < page_size) or np.any(locations >= mapping.size):
            raise ValueError("V4 window positions must name allocated original-token slots")
        swa = mapping[locations]
        if np.any(swa == 0):
            raise ValueError(
                "V4 SWA rows required by this step were released; row 0 is the padding "
                "row and would return another request's data"
            )
        w_rows.append(swa)
        w_pos.append(positions)
        w_req.append(np.full(positions.shape, r, np.int64))

        if ratio > 0:
            complete = (last + 1) // ratio
            if complete:
                entries = np.arange(complete, dtype=np.int64)
                anchors = np.asarray(
                    request_pool.req_to_token[int(slot), entries * ratio], np.int64
                )
                if np.any(anchors < page_size):
                    raise ValueError("V4 compressed anchors must name allocated slots")
                c_rows.append(anchors // ratio)
                c_ids.append(entries)
                c_req.append(np.full(entries.shape, r, np.int64))

    def pack(parts, empty_fill):
        if not parts:
            return np.full((1,), empty_fill, np.int32)
        return np.concatenate(parts).astype(np.int32)

    return ReadTables(
        window_rows=pack(w_rows, 0),
        window_positions=pack(w_pos, -1),
        window_request_ids=pack(w_req, -1),
        compressed_rows=pack(c_rows, 0),
        compressed_entry_ids=pack(c_ids, -1),
        compressed_request_ids=pack(c_req, -1),
    )


def run_layer(
    *,
    q,
    new_kv,
    layer_id: int,
    ratio: int,
    metadata,
    tables: ReadTables,
    kv_buffers,
    state,
    compressor_weights=None,
    compressor_input=None,
    indexer=None,
    attention_sink,
    softmax_scale: float,
    window_size: int,
    head_dim: int,
    index_topk: int | None = None,
    rope_head_dim: int = 64,
    norm_eps: float = 1e-6,
):
    """One V4 attention layer: compress, select, attend, and update both tiers.

    Args:
      q: [T, H, D] projected queries. new_kv: [T, D] this step's KV rows.
      metadata: `DeepseekV4AttentionMetadata` from M2.1.
      kv_buffers: dict with `"swa"` [W, D] and, for ratio > 0, `"compressed"` [E_all, D]
        and (ratio 4) `"indexer"` [E_all, Di]; C1 buffers flattened over their first
        two axes so a compressed address indexes them directly.
      state: [S, window, 2*coff*D] compressor state, or None for ratio 0.
      compressor_input: original [T, hidden] sublayer input, separate from projected KV.
      compressor_weights: `wkv`/`wgate`/`ape`/`norm_weight`/`cos_sin_cache` for the
        main compressor; required when ratio > 0.
      indexer: for ratio 4, a dict with the indexer's own `compressor_weights`,
        `q` [T, Hi, Di], `weights` [T, Hi] and `state`.

    Returns:
      `(out, updates)` where `updates` carries the new `swa`, `compressed`,
      `indexer` and `state` arrays that were actually touched. Nothing is written in
      place; the caller hands the results to `MemoryPools.replace_all`.
    """
    ratio_md = metadata.ratio(ratio) if ratio > 0 else None
    updates = {}

    compressed_kv = jnp.zeros((1, head_dim), jnp.float32)
    selected = None

    if ratio > 0:
        if compressor_weights is None:
            raise ValueError(f"ratio {ratio} needs compressor weights")
        if compressor_input is None:
            raise ValueError("compressed layers require original hidden compressor_input")
        # Cache addresses and visibility use group ids; RoPE uses the group's
        # start in original-token coordinates (SGLang: seq_len - ratio).
        rope_positions = ratio_md.boundary_group_ids * ratio
        records, record_valid, new_state = compress_chunk(
            compressor_input,
            state=state,
            positions=metadata.query_positions,
            query_request_ids=metadata.query_request_ids,
            prefix_lens=metadata.prefix_lens,
            cu_q_lens=metadata.cu_q_lens,
            state_slots=metadata.request_slots,
            boundary_token_indices=ratio_md.boundary_token_indices,
            boundary_valid_mask=ratio_md.boundary_valid_mask,
            boundary_compressed_pos=rope_positions,
            ratio=ratio,
            head_dim=head_dim,
            rope_head_dim=rope_head_dim,
            norm_eps=norm_eps,
            **compressor_weights,
        )
        updates["state"] = new_state
        # Records land at the addresses M2.1 derived; invalid boundaries are dropped
        # rather than aimed at entry 0.
        compressed_buffer = _scatter_records(
            kv_buffers["compressed"], records, ratio_md.boundary_write_entries, record_valid
        )
        updates["compressed"] = compressed_buffer
        if tables.decode_page_indices is None:
            compressed_kv = jnp.take(compressed_buffer, jnp.asarray(tables.compressed_rows), axis=0)

        if ratio == 4:
            if indexer is None or index_topk is None:
                raise ValueError("CSA layers need the indexer and index_topk")
            idx_records, idx_valid, idx_state = compress_chunk(
                indexer["compressor_input"],
                state=indexer["state"],
                positions=metadata.query_positions,
                query_request_ids=metadata.query_request_ids,
                prefix_lens=metadata.prefix_lens,
                cu_q_lens=metadata.cu_q_lens,
                state_slots=metadata.request_slots,
                boundary_token_indices=ratio_md.boundary_token_indices,
                boundary_valid_mask=ratio_md.boundary_valid_mask,
                boundary_compressed_pos=rope_positions,
                ratio=ratio,
                head_dim=indexer["head_dim"],
                rope_head_dim=indexer.get("rope_head_dim", rope_head_dim),
                norm_eps=norm_eps,
                **indexer["compressor_weights"],
            )
            updates["indexer_state"] = idx_state
            indexer_buffer = _scatter_records(
                kv_buffers["indexer"], idx_records, ratio_md.boundary_write_entries, idx_valid
            )
            updates["indexer"] = indexer_buffer
            if tables.decode_page_indices is None:
                indexer_keys = jnp.take(indexer_buffer, jnp.asarray(tables.compressed_rows), axis=0)
                selected = csa_indexer_topk(
                    indexer["q"],
                    indexer["weights"],
                    indexer_keys,
                    metadata.query_positions,
                    metadata.query_request_ids,
                    jnp.asarray(tables.compressed_request_ids),
                    metadata.valid_token_mask,
                    entry_group_ids=jnp.asarray(tables.compressed_entry_ids),
                    k=index_topk,
                    ratio=ratio,
                )

    # C1 retains all SWA pages read by this chunk until it completes. Publish
    # current-token KV before the read; causal positions, not write ordering,
    # prevent a query from attending to later tokens in the chunk.
    updates["swa"] = update_window_kv(
        kv_buffers["swa"], new_kv, metadata.swa_write_loc, metadata.valid_token_mask
    )
    if tables.decode_page_indices is not None:
        from sgl_jax.srt.layers.attention.dsv4.decode import csa_decode_attention

        out = csa_decode_attention(
            q,
            indexer["q"],
            indexer["weights"],
            updates["indexer"],
            updates["compressed"],
            updates["swa"],
            tables.decode_page_indices,
            tables.decode_window_rows,
            query_positions=metadata.query_positions,
            valid_token_mask=metadata.valid_token_mask,
            attention_sink=attention_sink,
            softmax_scale=softmax_scale,
            compressed_page_size=metadata.page_size // ratio,
            index_topk=index_topk,
            ratio=ratio,
        )
        return out, updates
    window_kv = jnp.take(updates["swa"], jnp.asarray(tables.window_rows), axis=0)
    out = dsv4_attention(
        q,
        window_kv,
        compressed_kv,
        query_positions=metadata.query_positions,
        query_request_ids=metadata.query_request_ids,
        valid_token_mask=metadata.valid_token_mask,
        window_positions=jnp.asarray(tables.window_positions),
        window_request_ids=jnp.asarray(tables.window_request_ids),
        compressed_entry_ids=jnp.asarray(tables.compressed_entry_ids),
        compressed_request_ids=jnp.asarray(tables.compressed_request_ids),
        attention_sink=attention_sink,
        softmax_scale=softmax_scale,
        window_size=window_size,
        ratio=ratio,
        selected_entries=selected,
    )

    return out, updates


def _scatter_records(buffer, records, write_entries, valid):
    """Place records at their compressed addresses, dropping invalid boundaries."""
    buffer = jnp.asarray(buffer)
    entries = jnp.asarray(write_entries)
    keep = jnp.asarray(valid, bool) & (entries >= 0) & (entries < buffer.shape[0])
    entries = jnp.where(keep, entries, buffer.shape[0])
    return buffer.at[entries].set(jnp.asarray(records, buffer.dtype), mode="drop")
