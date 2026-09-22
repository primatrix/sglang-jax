"""Mixed chunked prefill: decoding requests get request-local decode tables and leave
the shared-history tables (host builder), and the CSA layer overlays their rows."""

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.layers.attention.dsv4.execution import padded_read_tables

PAGE, WIN, RATIO = 128, 128, 4


def _pool(lengths, slots):
    total = 1 + 64 * len(slots)
    r2t = np.zeros((max(slots) + 1, 65536), np.int32)
    nxt = 1
    for s, L in zip(slots, lengths):
        n = -(-L // PAGE)
        pages = np.arange(nxt, nxt + n)
        nxt += n
        r2t[s, : n * PAGE] = (pages[:, None] * PAGE + np.arange(PAGE)[None, :]).reshape(-1)
    mapping = np.arange(total * PAGE + 4096, dtype=np.int32) + 1
    return SimpleNamespace(req_to_token=r2t), SimpleNamespace(full_to_swa_index_mapping=mapping)


def test_mixed_tables_split_decode_rows_from_prefill():
    # requests: prefill chunk (8192 new, no prefix), two decoding (q=1, history), one idle slot
    lengths = np.array([8192, 9000, 4100, 0], np.int64)
    q_lens = np.array([8192, 1, 1, 0], np.int64)
    slots = np.array([5, 2, 9, 0], np.int64)
    pool, alloc = _pool(lengths, slots)
    decode_rows = (q_lens == 1) & (lengths - q_lens > 0)
    tables = padded_read_tables(
        request_pool=pool,
        allocator=alloc,
        slots=slots,
        lengths=lengths,
        q_lens=q_lens,
        ratio=RATIO,
        window_size=WIN,
        page_size=PAGE,
        max_context_len=65536,
        token_capacity=8320,
        rank=0,
        compressed_capacity=4096,
        decode_capacity=4096,
        decode_rows=decode_rows,
    )
    # shared-history tables hold only the prefill request (request id 0)
    ids = np.asarray(tables.compressed_request_ids)
    assert set(ids[ids >= 0].tolist()) == {0}
    assert np.asarray(tables.compressed_entry_ids).max() == 8192 // RATIO - 1
    # decode tables: one row per request slot, token index = cumulative query offset
    assert tables.decode_page_indices.shape == (4, 4096 // (PAGE // RATIO))
    np.testing.assert_array_equal(tables.decode_token_index, [8192, 8193, -1, -1])
    assert (tables.decode_page_indices[0] > 0).sum() == -(-(9000 // RATIO) // (PAGE // RATIO))
    assert (tables.decode_page_indices[1] > 0).sum() == -(-(4100 // RATIO) // (PAGE // RATIO))
    assert np.all(tables.decode_page_indices[2:] == 0)
    assert np.all(tables.decode_window_rows[0] > 0)


def test_pure_decode_tables_unchanged_shape_and_token_index():
    lengths = np.array([9000, 4100, 0], np.int64)
    q_lens = np.array([1, 1, 0], np.int64)
    slots = np.array([2, 9, 0], np.int64)
    pool, alloc = _pool(lengths, slots)
    tables = padded_read_tables(
        request_pool=pool,
        allocator=alloc,
        slots=slots,
        lengths=lengths,
        q_lens=q_lens,
        ratio=RATIO,
        window_size=WIN,
        page_size=PAGE,
        max_context_len=65536,
        token_capacity=3,
        rank=0,
        compressed_capacity=4096,
        decode_capacity=4096,
    )
    assert tables.decode_page_indices.shape[0] == 3
    np.testing.assert_array_equal(tables.decode_token_index, [0, 1, -1])
    assert tables.compressed_rows.shape == (1,)
