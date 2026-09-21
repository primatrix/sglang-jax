"""Page-run DMA writer (interpret) == scatter with mode=drop, for contiguous and ragged layouts."""

import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.kernels.dsv4.paged_row_write import paged_row_write


def _check(cache, values, loc, valid, run=32):
    keep = valid & (loc >= 0) & (loc < cache.shape[0])
    safe = np.where(keep, loc, cache.shape[0])
    want = jnp.asarray(cache).at[jnp.asarray(safe)].set(jnp.asarray(values), mode="drop")
    got = paged_row_write(
        jnp.asarray(cache),
        jnp.asarray(values),
        jnp.asarray(loc),
        jnp.asarray(valid),
        run=run,
        interpret=True,
    )
    np.testing.assert_array_equal(np.asarray(got, np.float32), np.asarray(want, np.float32))


def test_contiguous_pages_and_ragged_segments():
    rng = np.random.default_rng(0)
    # Small sizes: the interpret-mode DMA loop runs one step per row.
    R, D, T = 1024, 128, 160
    cache = rng.standard_normal((R, D)).astype(jnp.bfloat16)
    values = rng.standard_normal((T, D)).astype(jnp.bfloat16)
    loc = np.zeros(T, np.int32)
    valid = np.ones(T, bool)
    loc[0:32] = 256 + np.arange(32)  # aligned page run
    loc[32:64] = 700 + np.arange(32)  # contiguous but unaligned start -> row path
    loc[64:96] = rng.permutation(R)[:32]  # scattered
    loc[96:128] = 512 + np.arange(32)
    valid[100:110] = False  # a hole in an otherwise contiguous segment
    loc[128:160] = 128 + np.arange(32)
    loc[150] = -1  # out of range -> dropped
    _check(cache, values, loc, valid)


def test_padding_and_all_invalid():
    rng = np.random.default_rng(1)
    R, D, T = 512, 128, 50  # T not a multiple of run
    cache = rng.standard_normal((R, D)).astype(jnp.bfloat16)
    values = rng.standard_normal((T, D)).astype(jnp.bfloat16)
    loc = np.arange(T, dtype=np.int32) + 32
    valid = np.ones(T, bool)
    _check(cache, values, loc, valid)
    _check(cache, values, loc, np.zeros(T, bool))


def test_long_runs_env_default(monkeypatch):
    """DSV4_PAGED_ROW_RUN picks the segment length; 256-row segments take the DMA
    path when a request's pages are physically consecutive and the row path otherwise."""
    from sgl_jax.srt.kernels.dsv4 import paged_row_write as prw

    monkeypatch.setenv("DSV4_PAGED_ROW_RUN", "256")
    assert prw.default_run() == 256
    rng = np.random.default_rng(1)
    R, D, T = 2048, 128, 640
    cache = rng.standard_normal((R, D)).astype(jnp.bfloat16)
    values = rng.standard_normal((T, D)).astype(jnp.bfloat16)
    loc = np.zeros(T, np.int32)
    valid = np.ones(T, bool)
    loc[0:256] = 512 + np.arange(256)  # aligned contiguous run
    loc[256:512] = 1024 + np.arange(256)
    loc[380:400] = rng.permutation(R)[:20]  # breaks the second run -> row path
    loc[512:640] = 16 + np.arange(128)  # tail segment (padded) -> row path
    valid[530:540] = False
    _check(cache, values, loc, valid, run=None)
