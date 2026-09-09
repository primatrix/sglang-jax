import json
from types import SimpleNamespace

import numpy as np
from sgl_jax.srt.layers.attention.deepseek_v4_csa_backend import padded_read_tables

rng = np.random.default_rng(44)
req = np.zeros((8, 16384), np.int32)
all_pages = rng.permutation(np.arange(1, 1 + 8 * 128)).reshape(8, 128)
for i in range(8):
    req[i] = (all_pages[i, :, None] * 128 + np.arange(128)).reshape(-1)
# Two rank-local SWA maps, with deliberately different physical placement.
mapping = [
    np.arange(1025 * 128, dtype=np.int32),
    np.arange(1025 * 128, dtype=np.int32) + 128,
]
results = []
for rank, lengths in enumerate(([8193, 0, 8225, 17], [0, 517, 8193, 0])):
    lengths = np.array(lengths, np.int32)
    queries = (lengths > 0).astype(np.int32)
    tables = padded_read_tables(
        request_pool=SimpleNamespace(req_to_token=req),
        allocator=SimpleNamespace(full_to_swa_index_mapping=mapping),
        slots=np.array([5, 2, 7, 1]),
        lengths=lengths,
        q_lens=queries,
        ratio=4,
        window_size=128,
        page_size=128,
        max_context_len=16384,
        token_capacity=4,
        rank=rank,
        compressed_capacity=8192,
        decode_capacity=4096,
    )
    token = 0
    for r, n in enumerate(queries):
        if not n:
            continue
        mask = tables.compressed_request_ids == r
        ids = tables.compressed_entry_ids[mask]
        actual = tables.decode_page_indices[token, ids // 32] * 32 + ids % 32
        np.testing.assert_array_equal(actual, tables.compressed_rows[mask])
        wmask = tables.window_request_ids == r
        np.testing.assert_array_equal(
            tables.decode_window_rows[token, -wmask.sum() :], tables.window_rows[wmask]
        )
        token += 1
    assert np.all(tables.decode_page_indices[token:] == 0)
    assert np.all(tables.decode_window_rows[token:] == 0)
    results.append(
        {
            "rank": rank,
            "live": token,
            "page_shape": list(tables.decode_page_indices.shape),
            "window_shape": list(tables.decode_window_rows.shape),
        }
    )
print(json.dumps(results, indent=2))
print("CSA_DECODE_TABLE_PROBE_COMPLETE")
