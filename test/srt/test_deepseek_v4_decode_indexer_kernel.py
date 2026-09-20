"""CSA decode indexer through ``kernels/dsa/streamindex_topk`` versus the request-local
``paged_csa_decode_scores`` + exact selector path (primatrix #370 scorer).

Both consume the same request-local page table (``decode_page_indices``) and must select the
same compressed entries per decode query, including padded rows (no entries) and requests
with fewer completed groups than ``index_topk``.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sgl_jax.srt.layers.attention.dsv4.decode import (
    csa_decode_select_kernel,
    csa_decode_select_p370,
    resolve_decode_indexer_backend,
)

RATIO, CPS, H, D, K = 4, 32, 64, 128, 512


def _batch(lengths_tokens, *, valid, pages_per_seq, seed=0):
    """lengths_tokens: per-row query position + 1 (tokens); valid: per-row real query flag.
    Each row owns pages_per_seq contiguous pages starting at 1 + row * pages_per_seq."""
    rng = np.random.default_rng(seed)
    B = len(lengths_tokens)
    pages = np.zeros((B, pages_per_seq), np.int32)
    for r in range(B):
        pages[r] = 1 + r * pages_per_seq + np.arange(pages_per_seq)
    total_pages = 1 + B * pages_per_seq
    return dict(
        index_q=jnp.asarray(rng.standard_normal((B, H, D), np.float32), jnp.bfloat16),
        index_weights=jnp.asarray(rng.standard_normal((B, H), np.float32)),
        index_cache=jnp.asarray(
            rng.standard_normal((total_pages * CPS, D), np.float32), jnp.bfloat16
        ),
        pages=jnp.asarray(pages),
        query_positions=jnp.asarray(np.array(lengths_tokens, np.int32) - 1),
        valid_token_mask=jnp.asarray(np.array(valid, bool)),
    )


def test_resolve_decode_indexer_backend(monkeypatch):
    monkeypatch.delenv("DSV4_DECODE_INDEXER_BACKEND", raising=False)
    assert resolve_decode_indexer_backend("auto") == "p370"  # default since pfbase14 (09-19)
    monkeypatch.setenv("DSV4_DECODE_INDEXER_BACKEND", "auto")
    expected = "kernel" if jax.default_backend() == "tpu" else "p370"
    assert resolve_decode_indexer_backend("auto") == expected
    assert resolve_decode_indexer_backend("p370") == "p370"
    monkeypatch.setenv("DSV4_DECODE_INDEXER_BACKEND", "p370")
    assert resolve_decode_indexer_backend("auto") == "p370"
    with pytest.raises(ValueError):
        resolve_decode_indexer_backend("reference")


@pytest.mark.skipif(jax.default_backend() != "tpu", reason="Pallas kernels need TPU")
@pytest.mark.parametrize(
    "lengths_tokens,valid,pages_per_seq",
    [
        # 8K-token histories, one padded row, one short request (< k completed groups)
        ([8192, 8000, 1000, 1, 8192, 4097, 8192, 8192], [1, 1, 1, 0, 1, 1, 1, 0], 64),
        # long histories (128K) at B=16
        ([131072] * 12 + [65536, 100, 131072, 131072], [1] * 14 + [0, 1], 1024),
    ],
)
def test_decode_kernel_matches_p370_selection(lengths_tokens, valid, pages_per_seq):
    b = _batch(lengths_tokens, valid=valid, pages_per_seq=pages_per_seq)
    lengths = jnp.where(b["valid_token_mask"], (b["query_positions"] + 1) // RATIO, 0)
    take = min(K, pages_per_seq * CPS)
    common = dict(
        index_q=b["index_q"],
        index_weights=b["index_weights"],
        index_cache=b["index_cache"],
        pages=b["pages"],
        lengths=lengths,
        take=take,
        ratio=RATIO,
        compressed_page_size=CPS,
    )
    ref_sel, ref_valid = (np.asarray(x) for x in csa_decode_select_p370(**common))
    got_sel, got_valid = (np.asarray(x) for x in csa_decode_select_kernel(**common))
    assert got_sel.shape == ref_sel.shape == (len(lengths_tokens), take)
    np.testing.assert_array_equal(got_valid.sum(1), ref_valid.sum(1))
    np.testing.assert_array_equal(got_valid.sum(1), np.minimum(np.asarray(lengths), take))
    for r in range(len(lengths_tokens)):
        assert set(got_sel[r][got_valid[r]].tolist()) == set(ref_sel[r][ref_valid[r]].tolist()), r
        assert np.all(got_sel[r][got_valid[r]] < int(lengths[r]))
