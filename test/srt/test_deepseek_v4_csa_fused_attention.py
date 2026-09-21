"""The fused CSA attention kernel (interpret mode) matches the dense reference."""

import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.layers.attention.dsv4 import attention as att


def _case(T, W, E, H, D, K, ratio, seed):
    rng = np.random.default_rng(seed)
    reqs = rng.integers(0, 2, size=T).astype(np.int32)
    qpos = (rng.integers(0, 4 * ratio, size=T)).astype(np.int32)
    valid = rng.random(T) > 0.1
    wpos = rng.integers(0, 4 * ratio, size=W).astype(np.int32)
    wreq = rng.integers(0, 2, size=W).astype(np.int32)
    entry = rng.integers(0, 4, size=E).astype(np.int32)
    creq = rng.integers(0, 2, size=E).astype(np.int32)
    sel = rng.integers(-1, E, size=(T, K)).astype(np.int32)
    sel[0] = -1  # a query that selects nothing
    return dict(
        q=jnp.asarray(rng.standard_normal((T, H, D)), jnp.bfloat16),
        window_kv=jnp.asarray(rng.standard_normal((W, D)), jnp.bfloat16),
        compressed_kv=jnp.asarray(rng.standard_normal((E, D)), jnp.bfloat16),
        query_positions=jnp.asarray(qpos),
        query_request_ids=jnp.asarray(reqs),
        valid_token_mask=jnp.asarray(valid),
        window_positions=jnp.asarray(wpos),
        window_request_ids=jnp.asarray(wreq),
        compressed_entry_ids=jnp.asarray(entry),
        compressed_request_ids=jnp.asarray(creq),
        attention_sink=jnp.asarray(rng.standard_normal(H) * 2, jnp.float32),
        softmax_scale=D**-0.5,
        window_size=ratio,
        ratio=ratio,
        selected_entries=jnp.asarray(sel),
    )


def test_fused_matches_dense():
    for T, W, E, H, D, K, seed in ((16, 40, 100, 8, 128, 8, 0), (13, 300, 700, 4, 128, 32, 1)):
        kw = _case(T, W, E, H, D, K, 4, seed)
        want = np.asarray(att.dsv4_dense_attention(**kw))
        got = np.asarray(att.csa_fused_attention(**kw, interpret=True))
        assert got.shape == want.shape
        np.testing.assert_allclose(got, want, rtol=1e-3, atol=1e-3)
        # invalid queries and queries admitting nothing are exactly zero
        np.testing.assert_array_equal(got[~np.asarray(kw["valid_token_mask"])], 0.0)


def test_fused_zero_when_nothing_admitted():
    kw = _case(8, 16, 32, 2, 128, 4, 4, 3)
    kw["selected_entries"] = jnp.full((8, 4), -1, jnp.int32)
    kw["window_request_ids"] = jnp.full((16,), 7, jnp.int32)  # no window row matches
    got = np.asarray(att.csa_fused_attention(**kw, interpret=True))
    assert np.all(np.isfinite(got)) and np.all(got == 0.0)


def test_selected_mask_equals_selected_entries():
    """A precomputed membership mask (threshold path) gives the same output as the index list."""
    from sgl_jax.srt.layers.attention.dsv4.attention import packed_membership

    for T, W, E, H, D, K, seed in ((16, 40, 100, 8, 128, 8, 0), (13, 300, 700, 4, 128, 32, 1)):
        kw = _case(T, W, E, H, D, K, 4, seed)
        want = np.asarray(att.csa_fused_attention(**kw, interpret=True))
        mask = packed_membership(kw["selected_entries"], E)
        kw2 = dict(kw, selected_entries=None, selected_mask=mask)
        got = np.asarray(att.csa_fused_attention(**kw2, interpret=True))
        np.testing.assert_array_equal(got, want)
        want_d = np.asarray(att.dsv4_dense_attention(**kw))
        got_d = np.asarray(att.dsv4_dense_attention(**kw2))
        np.testing.assert_array_equal(got_d, want_d)


def test_fused_other_tilings_match_dense(monkeypatch):
    """Block sizes are a pure tiling choice: any (block_q, block_k) gives the dense result."""
    for bq, bk in ((8, 128), (16, 256), (64, 1024)):
        monkeypatch.setattr(att, "_CSA_FUSED_BLOCK_Q", bq)
        monkeypatch.setattr(att, "_CSA_FUSED_BLOCK_K", bk)
        kw = _case(13, 300, 700, 4, 128, 32, 4, 1)
        want = np.asarray(att.dsv4_dense_attention(**kw))
        got = np.asarray(att.csa_fused_attention(**kw, interpret=True))
        np.testing.assert_allclose(got, want, rtol=1e-3, atol=1e-3)
