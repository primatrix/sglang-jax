"""The CSA decode page lookup (one-hot matmul by default) must equal the batched
``take_along_axis`` it replaces, for every fallback mode and for page ids that need
more than one bf16-exact byte."""

import jax.numpy as jnp
import numpy as np
import pytest

from sgl_jax.srt.layers.attention.dsv4 import decode as m


def _run(pages, idx, monkeypatch, mode):
    if mode is None:
        monkeypatch.delenv("DSV4_DECODE_PAGE_TAKE", raising=False)
    else:
        monkeypatch.setenv("DSV4_DECODE_PAGE_TAKE", mode)
    return np.asarray(m._take_pages(jnp.asarray(pages), jnp.asarray(idx)))


@pytest.mark.parametrize("mode", [None, "onehot", "gather", "2d"])
def test_take_pages_matches_take_along_axis(monkeypatch, mode):
    rng = np.random.default_rng(0)
    pages = rng.integers(0, 5000, size=(64, 2048), dtype=np.int32)
    idx = rng.integers(0, 2048, size=(64, 512), dtype=np.int32)
    got = _run(pages, idx, monkeypatch, mode)
    assert got.dtype == np.int32
    np.testing.assert_array_equal(got, np.take_along_axis(pages, idx, axis=1))


@pytest.mark.parametrize(
    "rows,table,k,hi",
    [(3, 72, 1, 300), (8, 512, 640, 70000), (2, 130, 5, (1 << 24) - 1)],
)
def test_take_pages_onehot_exact_for_awkward_shapes_and_large_ids(monkeypatch, rows, table, k, hi):
    rng = np.random.default_rng(1)
    pages = rng.integers(0, hi + 1, size=(rows, table), dtype=np.int32)
    pages[0, 0] = hi  # the largest id must survive the byte split
    idx = rng.integers(0, table, size=(rows, k), dtype=np.int32)
    idx[0, 0] = 0
    got = _run(pages, idx, monkeypatch, "onehot")
    np.testing.assert_array_equal(got, np.take_along_axis(pages, idx, axis=1))
