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


def test_hca_pad_queries_matches_scatter(monkeypatch):
    from sgl_jax.srt.kernels.hca import attention as hca

    monkeypatch.setenv("DSV4_PAGED_KV_WRITE", "1")
    monkeypatch.setenv("DSV4_PAGED_KV_WRITE_MIN_TOKENS", "1")
    monkeypatch.setenv("PALLAS_INTERPRET", "1")
    rng = np.random.default_rng(2)
    B, S, D = 3, 512, 128
    lens = [300, 512, 100]  # ragged chunk: request rows contiguous in new_kv
    T = sum(lens)
    new_kv = rng.standard_normal((T, D)).astype(jnp.bfloat16)
    seq_ids = np.repeat(np.arange(B), lens).astype(np.int32)
    local = np.concatenate([np.arange(n) for n in lens]).astype(np.int32)
    valid = np.ones(T, bool)
    valid[350:360] = False
    got = hca._pad_queries(
        jnp.asarray(new_kv),
        jnp.asarray(seq_ids),
        jnp.asarray(local),
        jnp.asarray(valid),
        batch=B,
        max_queries=S,
    )
    monkeypatch.setenv("DSV4_PAGED_KV_WRITE", "0")
    want = hca._pad_queries(
        jnp.asarray(new_kv),
        jnp.asarray(seq_ids),
        jnp.asarray(local),
        jnp.asarray(valid),
        batch=B,
        max_queries=S,
    )
    # the reference scatter ignores valid (padded tokens carry seq 0 / local 0 in real
    # metadata); compare only valid rows plus untouched zeros
    g, w = np.asarray(got, np.float32), np.asarray(want, np.float32)
    for b, n in enumerate(lens):
        rows = np.arange(n)
        ok = valid[np.cumsum([0] + lens)[b] + rows]
        np.testing.assert_array_equal(g[b, rows[ok]], w[b, rows[ok]])
        np.testing.assert_array_equal(g[b, n:], 0)


def test_scatter_records_paged_matches_xla(monkeypatch):
    """DSV4_PAGED_RECORD_WRITE routes prefill-sized record sets through the page-run writer."""
    from sgl_jax.srt.layers.attention.dsv4 import dispatch

    monkeypatch.setenv("PALLAS_INTERPRET", "1")
    rng = np.random.default_rng(3)
    entries_total, D, run = 4096, 128, 32
    buffer = rng.standard_normal((entries_total, D)).astype(jnp.bfloat16)
    cap = 640
    records = rng.standard_normal((cap, D)).astype(np.float32)
    ent = np.full(cap, -1, np.int32)
    valid = np.zeros(cap, bool)
    ent[0:256] = 1024 + np.arange(256)  # eight aligned page runs
    ent[256:320] = 3001 + np.arange(64)  # contiguous but off a tile boundary
    ent[320:400] = rng.permutation(entries_total)[:80]  # scattered
    valid[0:400] = True
    valid[100:104] = False  # hole inside a run
    ent[410] = entries_total + 5  # out of range, marked valid -> dropped
    valid[410] = True
    args = (jnp.asarray(buffer), jnp.asarray(records), jnp.asarray(ent), jnp.asarray(valid))
    monkeypatch.setattr(dispatch, "_PAGED_RECORD_WRITE", False)
    want = dispatch._scatter_records(*args, run=run)
    monkeypatch.setattr(dispatch, "_PAGED_RECORD_WRITE", True)
    monkeypatch.setattr(dispatch, "_PAGED_RECORD_MIN", 256)
    got = dispatch._scatter_records(*args, run=run)
    np.testing.assert_array_equal(np.asarray(got, np.float32), np.asarray(want, np.float32))
    # below the size gate the XLA path is used unchanged
    small = tuple(a[:128] if a.ndim else a for a in args[1:])
    got_small = dispatch._scatter_records(args[0], *small, run=run)
    want_small = dispatch._scatter_records(args[0], *small, run=None)
    np.testing.assert_array_equal(
        np.asarray(got_small, np.float32), np.asarray(want_small, np.float32)
    )


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
