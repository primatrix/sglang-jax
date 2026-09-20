"""Bisection top-k threshold (interpret) == exact top-k membership from lax.top_k."""

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.kernels.dsv4.topk_threshold import (
    score_key,
    topk_mask,
    topk_threshold,
    topk_threshold_xla,
)


def _reference_mask(scores, k):
    """Membership of the exact top-k (ties: every entry equal to the k-th value)."""
    s = np.asarray(scores, np.float32)
    out = np.zeros(s.shape, bool)
    for t in range(s.shape[0]):
        finite = np.isfinite(s[t])
        if finite.sum() == 0:
            continue
        vals = np.sort(s[t][finite])[::-1]
        kth = vals[min(k, vals.size) - 1]
        out[t] = finite & (s[t] >= kth)
    return out


def test_score_key_is_monotone():
    x = np.array([-np.inf, -3.5, -1e-30, -0.0, 0.0, 1e-30, 2.0, np.inf], np.float32)
    keys = np.asarray(score_key(jnp.asarray(x)))
    assert np.all(np.diff(keys.astype(np.int64)) >= 0)
    assert keys[3] <= keys[4]  # -0.0 orders no later than +0.0


def test_threshold_matches_reference():
    rng = np.random.default_rng(0)
    for T, E, k in ((16, 256, 8), (40, 1024, 512), (9, 384, 64)):
        s = rng.standard_normal((T, E)).astype(np.float32) * 3
        s[rng.random((T, E)) < 0.3] = -np.inf  # illegal entries
        s[0] = -np.inf  # a row with nothing legal
        s[1, : k // 2] = 1.0  # fewer legal than k
        s[1, k // 2 :] = -np.inf
        s[2, :] = 0.25  # all ties
        s[3, 5] = s[3, 6]  # a tie at some rank
        want = _reference_mask(s, k)
        got = np.asarray(topk_mask(jnp.asarray(s), k, interpret=True))
        np.testing.assert_array_equal(got, want)
        thr_k = np.asarray(topk_threshold(jnp.asarray(s), k, interpret=True))
        thr_x = np.asarray(topk_threshold_xla(jnp.asarray(s), k))
        np.testing.assert_array_equal(thr_k, thr_x)


def test_padding_rows_and_lanes():
    rng = np.random.default_rng(1)
    T, E, k = 13, 200, 16  # E not a lane multiple, T not a sublane multiple
    s = rng.standard_normal((T, E)).astype(np.float32)
    want = _reference_mask(s, k)
    got = np.asarray(topk_mask(jnp.asarray(s), k, interpret=True))
    np.testing.assert_array_equal(got, want)
    assert got.sum(1).min() == k


def _membership_case(rng, T, Ep, k, q_lens, counts):
    """Scores in request-local columns for a request-major token layout."""
    from sgl_jax.srt.layers.attention.dsv4.indexer import INVALID_ENTRY  # noqa: F401

    B = len(q_lens)
    req_ids = np.concatenate([np.full(n, r, np.int32) for r, n in enumerate(q_lens)])
    assert req_ids.size == T
    scores = np.full((T, Ep), -np.inf, np.float32)
    for t in range(T):
        r = req_ids[t]
        scores[t, : counts[r]] = rng.standard_normal(counts[r])
    offsets = np.cumsum(counts) - np.asarray(counts)
    valid = np.ones(T, bool)
    return (
        scores,
        offsets.astype(np.int32),
        np.asarray(q_lens, np.int32),
        req_ids,
        valid,
        sum(counts),
    )
