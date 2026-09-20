"""csa_sparse_attention (gather-then-dense) matches the dense dsv4_attention (CPU-only)."""

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.layers.attention.dsv4.attention import (
    csa_sparse_attention,
    dsv4_attention,
)


def _case(seed, T=6, H=4, D=128, W=8, E=40, K=5, ratio=4, window_size=6):
    rng = np.random.default_rng(seed)
    # two requests: rows 0..3 -> req 0 (positions 20..23), rows 4..5 -> req 1 (positions 9..10); last row padded
    qpos = np.array([20, 21, 22, 23, 9, 10], np.int32)
    qreq = np.array([0, 0, 0, 0, 1, 1], np.int32)
    valid = np.array([1, 1, 1, 1, 1, 0], bool)
    wpos = np.array([18, 19, 20, 21, 22, 8, 9, 10], np.int32)
    wreq = np.array([0, 0, 0, 0, 0, 1, 1, 1], np.int32)
    entry_ids = np.concatenate([np.arange(24), np.arange(16)]).astype(
        np.int32
    )  # req0: 24 entries, req1: 16
    creq = np.concatenate([np.zeros(24), np.ones(16)]).astype(np.int32)
    selected = np.full((T, K), -1, np.int32)
    for t in range(T):
        pool = np.flatnonzero(creq == qreq[t])
        pick = rng.choice(pool, size=min(K, len(pool)), replace=False)
        selected[t, : len(pick)] = pick
    selected[1, 0] = -1  # padding slot
    selected[2, 1] = int(np.flatnonzero(creq == 1)[0])  # wrong-request entry: must be ignored
    q = (rng.standard_normal((T, H, D)) * 0.5).astype(np.float32)
    window_kv = (rng.standard_normal((W, D)) * 0.5).astype(np.float32)
    compressed_kv = (rng.standard_normal((E, D)) * 0.5).astype(np.float32)
    sink = rng.uniform(-2, 2, size=H).astype(np.float32)
    # round to bf16 so the dense f32 reference sees the same operands the kernel does
    rb = lambda a: np.asarray(jnp.asarray(a).astype(jnp.bfloat16).astype(jnp.float32))
    kw = dict(
        query_positions=qpos,
        query_request_ids=qreq,
        valid_token_mask=valid,
        window_positions=wpos,
        window_request_ids=wreq,
        compressed_entry_ids=entry_ids,
        compressed_request_ids=creq,
        attention_sink=sink,
        softmax_scale=1.0 / np.sqrt(D),
        window_size=window_size,
        ratio=ratio,
        selected_entries=selected,
    )
    return rb(q), rb(window_kv), rb(compressed_kv), kw


def test_sparse_matches_dense():
    for seed in (0, 1, 2):
        q, wkv, ckv, kw = _case(seed)
        dense = np.asarray(dsv4_attention(q, wkv, ckv, **kw))
        sparse = np.asarray(csa_sparse_attention(q, wkv, ckv, interpret=True, **kw))
        assert dense.shape == sparse.shape
        np.testing.assert_allclose(sparse, dense, rtol=2e-2, atol=2e-2)
        assert np.all(sparse[-1] == 0.0)  # padded row


def test_sparse_no_selection_row_gives_window_only():
    q, wkv, ckv, kw = _case(3)
    sel = kw["selected_entries"].copy()
    sel[0, :] = -1
    kw["selected_entries"] = sel
    dense = np.asarray(dsv4_attention(q, wkv, ckv, **kw))
    sparse = np.asarray(csa_sparse_attention(q, wkv, ckv, interpret=True, **kw))
    np.testing.assert_allclose(sparse, dense, rtol=2e-2, atol=2e-2)
