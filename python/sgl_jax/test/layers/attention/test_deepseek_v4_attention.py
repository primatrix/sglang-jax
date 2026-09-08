"""M2.4 -- SWA / CSA / HCA numerical attention.

Fixed Q, fixed KV, fixed top-k; no compressor and no device kernel. The oracle is a
per-query NumPy loop that builds each key set explicitly, so it shares no masking or
reduction structure with the vectorised implementation.

Cases are the ones the M2.4 brief names: the SWA window, CSA sparse selection, the
HCA completed-group boundary, and multi-request isolation -- plus the requirement
that an early query in a chunk cannot read a group its later tokens complete.
"""

import numpy as np
import pytest

from sgl_jax.srt.layers.attention.dsv4.attention import (
    admissible_mask,
    dsv4_attention,
    update_window_kv,
)

D = 8
H = 2
SCALE = 1.0 / np.sqrt(D)


def _oracle(
    q,
    window_kv,
    compressed_kv,
    *,
    query_positions,
    query_request_ids,
    valid_token_mask,
    window_positions,
    window_request_ids,
    compressed_entry_ids,
    compressed_request_ids,
    attention_sink,
    softmax_scale,
    window_size,
    ratio,
    selected_entries=None,
):
    """Per-query reference: gather the admitted keys, then softmax with the sink."""
    q = np.asarray(q, np.float64)
    keys_all = np.concatenate(
        (np.asarray(window_kv, np.float64), np.asarray(compressed_kv, np.float64)), axis=0
    )
    W = np.asarray(window_kv).shape[0]
    T = q.shape[0]
    out = np.zeros((T, H, D), np.float64)

    for t in range(T):
        if not valid_token_mask[t]:
            continue
        p, r = int(query_positions[t]), int(query_request_ids[t])
        rows = []
        for j in range(W):
            if (
                window_request_ids[j] == r
                and window_positions[j] <= p
                and window_positions[j] > p - window_size
            ):
                rows.append(j)
        if ratio > 0:
            complete = (p + 1) // ratio
            allowed = None
            if selected_entries is not None:
                allowed = {int(e) for e in np.asarray(selected_entries)[t] if int(e) >= 0}
            for e in range(keys_all.shape[0] - W):
                if compressed_request_ids[e] != r or compressed_entry_ids[e] >= complete:
                    continue
                if allowed is not None and e not in allowed:
                    continue
                rows.append(W + e)
        if not rows:
            continue
        keys = keys_all[rows]
        for h in range(H):
            scores = (q[t, h] @ keys.T) * softmax_scale
            sink = float(attention_sink[h])
            shift = max(scores.max(), sink)
            probs = np.exp(scores - shift)
            out[t, h] = (probs @ keys) / (probs.sum() + np.exp(sink - shift))
    return out


def _fixture(
    q_lens, prefix_lens, entries_per_request, *, seed=0, pad_tokens=0, window_size=8, ratio=4
):
    rng = np.random.default_rng(seed)
    positions, request_ids = [], []
    for r, (pre, n) in enumerate(zip(prefix_lens, q_lens)):
        positions.extend(range(pre, pre + n))
        request_ids.extend([r] * n)
    live = len(positions)
    T = live + pad_tokens
    query_positions = np.array(positions + [0] * pad_tokens, np.int32)
    query_request_ids = np.array(request_ids + [0] * pad_tokens, np.int32)
    valid = np.arange(T) < live

    # The window holds every token of every request, which is a superset of what any
    # query may see; the mask is what must narrow it.
    wpos, wreq = [], []
    for r, (pre, n) in enumerate(zip(prefix_lens, q_lens)):
        for p in range(0, pre + n):
            wpos.append(p)
            wreq.append(r)
    entry_ids, entry_req = [], []
    for r, n in enumerate(entries_per_request):
        entry_ids.extend(range(n))
        entry_req.extend([r] * n)

    return dict(
        q=rng.normal(size=(T, H, D)).astype(np.float32),
        window_kv=rng.normal(size=(max(len(wpos), 1), D)).astype(np.float32),
        compressed_kv=rng.normal(size=(max(len(entry_ids), 1), D)).astype(np.float32),
        query_positions=query_positions,
        query_request_ids=query_request_ids,
        valid_token_mask=valid,
        window_positions=np.array(wpos or [-1], np.int32),
        window_request_ids=np.array(wreq or [-1], np.int32),
        compressed_entry_ids=np.array(entry_ids or [-1], np.int32),
        compressed_request_ids=np.array(entry_req or [-1], np.int32),
        attention_sink=rng.normal(size=(H,)).astype(np.float32),
        softmax_scale=SCALE,
        window_size=window_size,
        ratio=ratio,
    )


_MASK_ARGS = (
    "query_positions",
    "query_request_ids",
    "valid_token_mask",
    "window_positions",
    "window_request_ids",
    "compressed_entry_ids",
    "compressed_request_ids",
    "window_size",
    "ratio",
)


def _masks(fx, **kw):
    """`admissible_mask` over a fixture, as numpy."""
    wm, cm = admissible_mask(**{k: fx[k] for k in _MASK_ARGS}, **kw)
    return np.asarray(wm), np.asarray(cm)


def _both(fx, **kw):
    got = np.asarray(
        dsv4_attention(
            fx["q"],
            fx["window_kv"],
            fx["compressed_kv"],
            **{k: v for k, v in fx.items() if k not in ("q", "window_kv", "compressed_kv")},
            **kw,
        )
    )
    want = _oracle(
        fx["q"],
        fx["window_kv"],
        fx["compressed_kv"],
        **{k: v for k, v in fx.items() if k not in ("q", "window_kv", "compressed_kv")},
        **kw,
    )
    return got, want


def _assert_match(fx, **kw):
    got, want = _both(fx, **kw)
    np.testing.assert_allclose(got, want, rtol=2e-5, atol=2e-5)
    return got


# --------------------------------------------------------------------------
# SWA-only layers
# --------------------------------------------------------------------------


def test_swa_only_layer_ignores_compressed_history():
    """ratio 0: even with records present and attractive, none may be read."""
    fx = _fixture([8], [0], [4], seed=1, ratio=0)
    fx["compressed_kv"] = fx["compressed_kv"] * 100.0
    got = _assert_match(fx)
    _, cmask = _masks(fx)
    assert not np.any(cmask)
    assert np.all(np.isfinite(got))


def test_window_is_causal_on_the_right_and_bounded_on_the_left():
    fx = _fixture([16], [0], [0], seed=2, ratio=0, window_size=4)
    wmask, _ = _masks(fx)
    for t, p in enumerate(fx["query_positions"]):
        allowed = np.flatnonzero(wmask[t])
        assert allowed.tolist() == [j for j in range(max(0, p - 3), p + 1)]
    _assert_match(fx)


# --------------------------------------------------------------------------
# HCA: every completed group, union with the window
# --------------------------------------------------------------------------


def test_hca_admits_every_completed_group_including_the_window_overlap():
    """ratio == window: the newest complete group covers the same tokens the window
    already carries, and it is still admitted. That overlap is the semantics."""
    fx = _fixture([8], [0], [2], seed=3, ratio=4, window_size=4)
    _, cmask = _masks(fx)
    for t, p in enumerate(fx["query_positions"]):
        assert cmask[t].sum() == min(2, (p + 1) // 4), (t, p)
    # Position 7 sees group 1 (tokens 4..7) *and* window tokens 4..7.
    assert cmask[7].tolist() == [True, True]
    _assert_match(fx)


def test_an_early_query_cannot_read_a_group_its_later_tokens_complete():
    """The whole chunk is presented at once; query at position 4 must still see only
    group 0, even though position 7 in the same chunk completes group 1."""
    fx = _fixture([8], [0], [2], seed=4, ratio=4, window_size=4)
    _, cmask = _masks(fx)
    assert cmask[4].tolist() == [True, False]
    assert cmask[6].tolist() == [True, False]
    assert cmask[7].tolist() == [True, True]


def test_a_group_completed_later_cannot_change_an_earlier_output():
    """Extend the chunk so another group completes; earlier queries must be bit-stable."""
    short = _fixture([4], [0], [2], seed=5, ratio=4, window_size=4)
    long = _fixture([8], [0], [2], seed=5, ratio=4, window_size=4)
    long["q"][:4] = short["q"]
    long["compressed_kv"] = short["compressed_kv"]
    long["window_kv"][: short["window_kv"].shape[0]] = short["window_kv"]
    long["attention_sink"] = short["attention_sink"]
    got_short = _assert_match(short)
    got_long = _assert_match(long)
    np.testing.assert_allclose(got_short, got_long[:4], rtol=2e-5, atol=2e-5)


# --------------------------------------------------------------------------
# CSA: top-k selection
# --------------------------------------------------------------------------


def test_csa_admits_only_the_selected_entries():
    fx = _fixture([8], [0], [2], seed=6, ratio=4, window_size=4)
    selected = np.full((fx["q"].shape[0], 1), -1, np.int32)
    selected[7, 0] = 0  # position 7 could see both groups; allow only entry 0
    _, cmask = _masks(fx, selected_entries=selected)
    assert cmask[7].tolist() == [True, False]
    _assert_match(fx, selected_entries=selected)


def test_selection_cannot_override_completeness():
    """A stale or over-eager selection must not reach an unwritten group: selection
    intersects the completeness rule, it does not replace it."""
    fx = _fixture([8], [0], [2], seed=7, ratio=4, window_size=4)
    selected = np.zeros((fx["q"].shape[0], 2), np.int32)
    selected[:, 0] = 0
    selected[:, 1] = 1  # ask for group 1 from *every* query, including position 0
    _, cmask = _masks(fx, selected_entries=selected)
    assert cmask[0].tolist() == [False, False]  # nothing complete yet
    assert cmask[4].tolist() == [True, False]  # group 1 requested but incomplete
    assert cmask[7].tolist() == [True, True]
    _assert_match(fx, selected_entries=selected)


def test_all_minus_one_selection_reads_window_only():
    fx = _fixture([8], [0], [2], seed=8, ratio=4, window_size=4)
    none = np.full((fx["q"].shape[0], 3), -1, np.int32)
    swa_only = dict(fx)
    swa_only["ratio"] = 0
    np.testing.assert_allclose(
        _assert_match(fx, selected_entries=none),
        _assert_match(swa_only),
        rtol=2e-5,
        atol=2e-5,
    )


# --------------------------------------------------------------------------
# the sink
# --------------------------------------------------------------------------


def test_sink_enters_the_denominator_only():
    """A large sink shrinks the output toward zero without contributing a value."""
    fx = _fixture([4], [0], [0], seed=9, ratio=0, window_size=4)
    small = dict(fx, attention_sink=np.full((H,), -30.0, np.float32))
    large = dict(fx, attention_sink=np.full((H,), 30.0, np.float32))
    out_small = _assert_match(small)
    out_large = _assert_match(large)
    assert np.all(np.abs(out_large) < np.abs(out_small) + 1e-6)
    assert np.abs(out_large).max() < 1e-3


def test_a_dominant_sink_does_not_overflow():
    """`shift` has to include the sink; otherwise exp(sink - shift) overflows exactly
    when the sink is the largest logit."""
    fx = _fixture([4], [0], [0], seed=10, ratio=0, window_size=4)
    fx["attention_sink"] = np.full((H,), 200.0, np.float32)
    got = _assert_match(fx)
    assert np.all(np.isfinite(got))


def test_per_head_sink_is_not_shared():
    fx = _fixture([4], [0], [0], seed=11, ratio=0, window_size=4)
    fx["attention_sink"] = np.array([-30.0, 30.0], np.float32)
    got = _assert_match(fx)
    assert np.abs(got[:, 1]).max() < np.abs(got[:, 0]).max()


def test_rejects_wrong_sink_shape():
    fx = _fixture([4], [0], [0], seed=12, ratio=0)
    fx["attention_sink"] = np.zeros((H + 1,), np.float32)
    with pytest.raises(ValueError, match="attention_sink must be"):
        dsv4_attention(
            fx["q"],
            fx["window_kv"],
            fx["compressed_kv"],
            **{k: v for k, v in fx.items() if k not in ("q", "window_kv", "compressed_kv")},
        )


# --------------------------------------------------------------------------
# batching
# --------------------------------------------------------------------------


def test_requests_are_isolated():
    fx = _fixture([6, 6], [0, 0], [1, 1], seed=13, ratio=4, window_size=4)
    fx["window_kv"][6:] *= 50.0
    fx["compressed_kv"][1:] *= 50.0
    _assert_match(fx)
    wmask, cmask = _masks(fx)
    for t in range(fx["q"].shape[0]):
        r = fx["query_request_ids"][t]
        assert set(fx["window_request_ids"][wmask[t]].tolist()) <= {r}
        assert set(fx["compressed_request_ids"][cmask[t]].tolist()) <= {r}


def test_unequal_lengths_and_padding():
    fx = _fixture([6, 2], [0, 10], [1, 3], seed=14, ratio=4, window_size=4, pad_tokens=3)
    got = _assert_match(fx)
    assert np.all(got[-3:] == 0.0)


def test_a_query_with_no_admissible_key_is_zero_not_nan():
    """Padded queries mask everything; the denominator is then exp(0) = 1, so the
    result must be a clean zero rather than 0/0."""
    fx = _fixture([2], [0], [0], seed=15, ratio=0, window_size=4, pad_tokens=5)
    got = _assert_match(fx)
    assert np.all(np.isfinite(got))
    assert np.all(got[2:] == 0.0)


# --------------------------------------------------------------------------
# the SWA cache write
# --------------------------------------------------------------------------


def test_window_write_places_rows_and_drops_padding():
    cache = np.zeros((6, D), np.float32)
    new = np.arange(3 * D, dtype=np.float32).reshape(3, D) + 1.0
    loc = np.array([2, 4, -1], np.int32)
    valid = np.array([True, True, False])
    out = np.asarray(update_window_kv(cache, new, loc, valid))
    np.testing.assert_array_equal(out[2], new[0])
    np.testing.assert_array_equal(out[4], new[1])
    # Row 0 must not have been used as a dumping ground for the padded token.
    assert np.all(out[0] == 0.0)
    assert np.all(out[1] == 0.0) and np.all(out[3] == 0.0) and np.all(out[5] == 0.0)


def test_window_write_drops_out_of_range_locations():
    cache = np.zeros((4, D), np.float32)
    new = np.ones((2, D), np.float32)
    out = np.asarray(
        update_window_kv(cache, new, np.array([9, 1], np.int32), np.array([True, True]))
    )
    assert np.all(out[1] == 1.0)
    assert np.all(out[0] == 0.0) and np.all(out[2] == 0.0) and np.all(out[3] == 0.0)
