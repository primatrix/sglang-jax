"""Decode HCA on one-record pages through the request-major dense view.

With C1 ``page_size`` 128 a compressed page holds one record, and the decode
streaming kernel gathered it row by row from the 4D pool; XLA then relaid out the
whole pool for every HCA layer of the decode step.  ``_dense_compressed_view``
gathers each request's records with XLA into ``[B, tile, D]`` and hands the
kernel that buffer instead.  This checks the view itself against NumPy and the
decode output against the dense HCA reference and the row-by-row path.
"""

import importlib.util
import os
import pathlib

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.kernels.hca import attention as hca_attention
from sgl_jax.srt.kernels.hca.attention import _dense_compressed_view, ragged_attention
from sgl_jax.srt.kernels.hca.tuned_block_sizes import get_hca_kernel_schedule

_spec = importlib.util.spec_from_file_location(
    "hca_small_page_gather_test",
    pathlib.Path(__file__).with_name("test_deepseek_v4_hca_small_page_gather.py"),
)
_small = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_small)
H, D, P = _small.H, _small.D, _small.P


def test_dense_view_gathers_each_request_in_table_order():
    rng = np.random.default_rng(3)
    rows, tile = 40, 16
    flat = jnp.asarray(rng.standard_normal((rows, D)), jnp.bfloat16)
    # request 0: records at pages 5,9,2 (3 live); request 1: none; request 2: pages 30,31.
    table = np.array([5, 9, 2, 0, 0, 30, 31, 0], np.int32)
    starts = np.array([0, 3, 5], np.int32)
    lens = np.array([3, 0, 2], np.int32)
    cache, pages, page_starts, page_size = _dense_compressed_view(
        flat,
        jnp.asarray(table),
        jnp.asarray(starts),
        jnp.asarray(lens),
        head_dim=D,
        page_size=1,
        tile=tile,
        sublanes=8,
    )
    assert page_size == tile and cache.shape == (3 * tile, D)
    np.testing.assert_array_equal(np.asarray(pages), [0, 1, 2])
    np.testing.assert_array_equal(np.asarray(page_starts), [0, 1, 2])
    got = np.asarray(cache.astype(jnp.float32)).reshape(3, tile, D)
    want = np.zeros((3, tile, D), np.float32)
    src = np.asarray(flat.astype(jnp.float32))
    for r in range(3):
        for j in range(lens[r]):
            want[r, j] = src[table[starts[r] + j]]
    np.testing.assert_array_equal(got, want)


def test_dense_view_passes_sublane_pages_through():
    flat = jnp.zeros((64, D), jnp.bfloat16)
    table, starts, lens = (jnp.zeros((4,), jnp.int32),) * 3
    out = _dense_compressed_view(
        flat, table, starts, lens, head_dim=D, page_size=8, tile=128, sublanes=8
    )
    assert out[0] is flat and out[3] == 8


def _decode(reqs, key):
    args, aux = _small._build(reqs, 1, key)
    sched = get_hca_kernel_schedule(
        "TPU7x", page_size=1, max_compressed_entries=64, local_heads=H, head_dim=D
    )
    assert sched.compressed_tile == 128
    ref = _small._reference(reqs, args, aux)
    out = ragged_attention(*args, schedule=sched, softmax_scale=D**-0.5, page_size=P)[0]
    return np.asarray(out.astype(jnp.float32)), ref


def test_decode_matches_reference_and_row_gather(monkeypatch):
    # Pure decode batch: tokens == batch, one query each; 0..12 records per request,
    # one request crossing a boundary (its new record is written this step).
    reqs = [(1536, 1), (100, 1), (127, 1), (1000, 1), (383, 1)]
    dense, ref = _decode(reqs, jax.random.PRNGKey(11))
    assert np.isfinite(dense).all()
    np.testing.assert_allclose(dense, ref, rtol=2e-2, atol=2e-2)
    monkeypatch.setenv("DSV4_HCA_DECODE_DENSE", "0")
    assert not hca_attention._decode_dense_view_enabled()
    rows, _ = _decode(reqs, jax.random.PRNGKey(11))
    np.testing.assert_allclose(dense, rows, rtol=1e-3, atol=1e-3)


def test_decode_rows_per_step_is_output_invariant(monkeypatch):
    # 11 decode rows: with 4 rows per grid step the last step holds 3 real rows plus
    # one padded row; every row must equal the one-row-per-step result bit for bit.
    reqs = [
        (1536, 1),
        (100, 1),
        (127, 1),
        (1000, 1),
        (383, 1),
        (0, 1),
        (2048, 1),
        (5, 1),
        (700, 1),
        (129, 1),
        (256, 1),
    ]
    monkeypatch.setenv(hca_attention.STREAM_ROWS_ENV, "1")
    single, ref = _decode(reqs, jax.random.PRNGKey(5))
    monkeypatch.setenv(hca_attention.STREAM_ROWS_ENV, "4")
    multi, _ = _decode(reqs, jax.random.PRNGKey(5))
    np.testing.assert_array_equal(multi, single)
    np.testing.assert_allclose(multi, ref, rtol=2e-2, atol=2e-2)
