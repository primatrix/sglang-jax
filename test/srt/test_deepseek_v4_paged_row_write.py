"""Page-run DMA writer (interpret) == scatter with mode=drop, for contiguous and ragged layouts."""

import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.kernels.dsv4.paged_row_write import paged_row_write


def _check(cache, values, loc, valid, run=128):
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
    R, D, T = 4096, 256, 640
    cache = rng.standard_normal((R, D)).astype(jnp.bfloat16)
    values = rng.standard_normal((T, D)).astype(jnp.bfloat16)
    loc = np.zeros(T, np.int32)
    valid = np.ones(T, bool)
    loc[0:128] = 1024 + np.arange(128)  # aligned page run
    loc[128:256] = 3000 + np.arange(128)  # contiguous but unaligned start -> row path
    loc[256:384] = rng.permutation(R)[:128]  # scattered
    loc[384:512] = 2048 + np.arange(128)
    valid[400:410] = False  # a hole in an otherwise contiguous segment
    loc[512:640] = 512 + np.arange(128)
    loc[600] = -1  # out of range -> dropped
    _check(cache, values, loc, valid)


def test_padding_and_all_invalid():
    rng = np.random.default_rng(1)
    R, D, T = 1024, 128, 200  # T not a multiple of run
    cache = rng.standard_normal((R, D)).astype(jnp.bfloat16)
    values = rng.standard_normal((T, D)).astype(jnp.bfloat16)
    loc = np.arange(T, dtype=np.int32) + 128
    valid = np.ones(T, bool)
    _check(cache, values, loc, valid)
    _check(cache, values, loc, np.zeros(T, bool))


def test_long_runs_env_default(monkeypatch):
    """DSV4_PAGED_ROW_RUN picks the segment length; 1024-row segments take the DMA
    path when a request's pages are physically consecutive and the row path otherwise."""
    from sgl_jax.srt.kernels.dsv4 import paged_row_write as prw

    monkeypatch.setenv("DSV4_PAGED_ROW_RUN", "1024")
    assert prw.default_run() == 1024
    rng = np.random.default_rng(1)
    R, D, T = 8192, 256, 2560
    cache = rng.standard_normal((R, D)).astype(jnp.bfloat16)
    values = rng.standard_normal((T, D)).astype(jnp.bfloat16)
    loc = np.zeros(T, np.int32)
    valid = np.ones(T, bool)
    loc[0:1024] = 2048 + np.arange(1024)  # aligned contiguous run
    loc[1024:2048] = 4096 + np.arange(1024)
    loc[1500:1600] = rng.permutation(R)[:100]  # breaks the second run -> row path
    loc[2048:2560] = 16 + np.arange(512)  # tail segment (padded) -> row path
    valid[2100:2110] = False
    _check(cache, values, loc, valid, run=None)
