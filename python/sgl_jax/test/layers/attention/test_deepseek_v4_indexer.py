"""M2.3 -- CSA indexer scoring and compressed-space top-k.

Fixed keys, fixed queries, no compressor and no device kernel. The M2.3 brief asks
for scoring/selection checked against a reference over: history shorter than K,
padding, unequal request lengths, per-query group boundaries, and the property
that changing a future token cannot change an earlier legal selection.

The oracle here is deliberately a plain triple loop -- if it shared structure with
the implementation it would not be independent evidence.
"""

import numpy as np
import pytest

from sgl_jax.srt.layers.attention.dsv4.indexer import (
    INVALID_ENTRY,
    csa_indexer_scores,
    csa_indexer_topk,
    visible_entries_for_query,
)

RATIO = 4
HEAD_DIM = 8


def _oracle(
    q,
    weights,
    keys,
    query_positions,
    query_request_ids,
    entry_request_ids,
    valid_token_mask,
    *,
    entry_group_ids,
    k,
    ratio,
    num_kv_heads=None,
):
    """Independent NumPy reference: score every legal entry, sort, pad with -1."""
    q = np.asarray(q, np.float64)
    weights = np.asarray(weights, np.float64)
    keys = np.asarray(keys, np.float64)
    if keys.ndim == 2:
        keys = keys[:, None, :]
    T, H, _ = q.shape
    E = keys.shape[0]
    kv_heads = keys.shape[1] if num_kv_heads is None else num_kv_heads
    group = H // kv_heads

    out = np.full((T, k), INVALID_ENTRY, np.int32)
    for t in range(T):
        if not valid_token_mask[t]:
            continue
        complete = (query_positions[t] + 1) // ratio
        cand = []
        for e in range(E):
            if (
                entry_request_ids[e] != query_request_ids[t]
                or not 0 <= entry_group_ids[e] < complete
            ):
                continue
            s = 0.0
            for h in range(H):
                inner = float(np.dot(q[t, h], keys[e, h // group]))
                s += max(0.0, inner) * weights[t, h]
            cand.append((s, e))
        cand.sort(key=lambda sv: (-sv[0], sv[1]))
        for i, (_, e) in enumerate(cand[:k]):
            out[t, i] = e
    return out


def _fixture(
    q_lens,
    prefix_lens,
    entries_per_request,
    *,
    heads=4,
    kv_heads=1,
    seed=0,
    pad_tokens=0,
    pad_entries=0,
):
    rng = np.random.default_rng(seed)
    q_lens = list(q_lens)
    prefix_lens = list(prefix_lens)
    positions, request_ids = [], []
    for r, (pre, n) in enumerate(zip(prefix_lens, q_lens)):
        positions.extend(range(pre, pre + n))
        request_ids.extend([r] * n)
    live = len(positions)
    T = live + pad_tokens
    positions = np.array(positions + [0] * pad_tokens, np.int32)
    request_ids = np.array(request_ids + [0] * pad_tokens, np.int32)
    valid = np.arange(T) < live

    entry_request_ids = []
    for r, n in enumerate(entries_per_request):
        entry_request_ids.extend([r] * n)
    # Padded key rows belong to a request id no query carries.
    entry_request_ids.extend([-1] * pad_entries)
    entry_request_ids = np.array(entry_request_ids, np.int32)
    E = entry_request_ids.size

    return dict(
        q=rng.normal(size=(T, heads, HEAD_DIM)).astype(np.float32),
        weights=rng.normal(size=(T, heads)).astype(np.float32),
        keys=rng.normal(size=(E, kv_heads, HEAD_DIM)).astype(np.float32),
        query_positions=positions,
        query_request_ids=request_ids,
        entry_request_ids=entry_request_ids,
        entry_group_ids=np.concatenate(
            [np.arange(n) for n in entries_per_request] + [np.full(pad_entries, -1)]
        ).astype(np.int32),
        valid_token_mask=valid,
    )


def _both(fx, *, k, ratio=RATIO, num_kv_heads=None):
    got = np.asarray(csa_indexer_topk(**fx, k=k, ratio=ratio, num_kv_heads=num_kv_heads))
    want = _oracle(**fx, k=k, ratio=ratio, num_kv_heads=num_kv_heads)
    return got, want


# --------------------------------------------------------------------------
# the causality rule -- the reason this is not a wrapper over streamindex_topk
# --------------------------------------------------------------------------


def test_only_completed_groups_are_selectable():
    """Entry e needs its whole group consumed: position (e+1)*ratio - 1."""
    pos = np.array([0, 2, 3, 4, 6, 7, 8])
    vis = np.asarray(visible_entries_for_query(pos, 4))
    # positions 0..2 complete nothing; 3 completes entry 0; 7 completes entry 1.
    assert vis.tolist() == [0, 0, 1, 1, 1, 2, 2]


def test_the_group_a_query_sits_inside_is_not_selectable():
    """This is exactly where kernels/dsa's mask (`e*ratio <= position`) differs:
    it would admit the unwritten group the query is inside."""
    for position in (4, 5, 6):  # all inside group 1, which completes at 7
        vis = int(np.asarray(visible_entries_for_query(np.array([position]), 4))[0])
        assert vis == 1, position
        # dsa's rule would have allowed entry 1 here:
        assert position // 4 == 1


def test_a_later_token_cannot_change_an_earlier_selection():
    """The stability property: extend the chunk so more groups complete, and every
    query that existed before must keep the identical selection."""
    short = _fixture([8], [0], [4], seed=7)
    long = _fixture([16], [0], [4], seed=7)
    # Same queries/keys for the overlapping prefix.
    for key in ("q", "weights"):
        long[key][:8] = short[key]
    long["keys"] = short["keys"]

    got_short, _ = _both(short, k=3)
    got_long, _ = _both(long, k=3)
    np.testing.assert_array_equal(got_short, got_long[:8])


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------


def test_scores_match_the_relu_inside_head_sum_definition():
    rng = np.random.default_rng(3)
    q = rng.normal(size=(5, 4, HEAD_DIM)).astype(np.float32)
    w = rng.normal(size=(5, 4)).astype(np.float32)
    keys = rng.normal(size=(6, 1, HEAD_DIM)).astype(np.float32)
    got = np.asarray(csa_indexer_scores(q, w, keys))
    want = np.zeros((5, 6), np.float64)
    for t in range(5):
        for e in range(6):
            want[t, e] = sum(
                max(0.0, float(np.dot(q[t, h], keys[e, 0]))) * float(w[t, h]) for h in range(4)
            )
    np.testing.assert_allclose(got, want, rtol=2e-5, atol=2e-5)


def test_relu_placement_actually_matters():
    """Guards against collapsing the heads into one matmul: with the ReLU inside
    the sum, a negative-inner head contributes nothing rather than cancelling."""
    q = np.zeros((1, 2, HEAD_DIM), np.float32)
    q[0, 0, 0] = 1.0
    q[0, 1, 0] = -1.0
    w = np.ones((1, 2), np.float32)
    keys = np.zeros((1, 1, HEAD_DIM), np.float32)
    keys[0, 0, 0] = 1.0
    # relu(1)*1 + relu(-1)*1 = 1, whereas relu(1 + -1) would be 0.
    assert float(np.asarray(csa_indexer_scores(q, w, keys))[0, 0]) == pytest.approx(1.0)


def test_gqa_head_folding():
    fx = _fixture([4], [0], [1], heads=4, kv_heads=2, seed=11)
    got, want = _both(fx, k=1)
    np.testing.assert_array_equal(got, want)


# --------------------------------------------------------------------------
# selection vs the oracle
# --------------------------------------------------------------------------


@pytest.mark.parametrize("k", [1, 2, 4])
def test_matches_oracle_single_request(k):
    fx = _fixture([16], [0], [4], seed=k)
    got, want = _both(fx, k=k)
    np.testing.assert_array_equal(got, want)


def test_matches_oracle_unequal_request_lengths():
    fx = _fixture([12, 4, 8], [0, 20, 8], [3, 6, 4], seed=5)
    got, want = _both(fx, k=3)
    np.testing.assert_array_equal(got, want)


def test_requests_are_isolated():
    """A query must never select another request's entry, even a high-scoring one."""
    fx = _fixture([4, 4], [40, 0], [2, 2], seed=9)
    # Make request 1's keys overwhelmingly attractive.
    fx["keys"][2:] *= 100.0
    got, _ = _both(fx, k=2)
    for t in range(got.shape[0]):
        req = fx["query_request_ids"][t]
        for e in got[t]:
            if e != INVALID_ENTRY:
                assert fx["entry_request_ids"][e] == req


# --------------------------------------------------------------------------
# history shorter than k, padding
# --------------------------------------------------------------------------


def test_history_shorter_than_k_pads_with_invalid_at_the_end():
    # 8 tokens at ratio 4 completes 2 entries; k=5 cannot be filled.
    fx = _fixture([8], [0], [2], seed=2)
    got, want = _both(fx, k=5)
    np.testing.assert_array_equal(got, want)
    last_row = got[-1]
    assert (last_row != INVALID_ENTRY).sum() == 2
    # Padding sits at the end, never interleaved.
    valid = last_row != INVALID_ENTRY
    assert valid.tolist() == sorted(valid.tolist(), reverse=True)


def test_no_completed_history_selects_nothing():
    fx = _fixture([3], [0], [0], seed=1)
    got, want = _both(fx, k=4)
    np.testing.assert_array_equal(got, want)
    assert (got == INVALID_ENTRY).all()


def test_k_larger_than_the_key_array():
    fx = _fixture([8], [0], [2], seed=4)
    got, want = _both(fx, k=16)
    assert got.shape == (8, 16)
    np.testing.assert_array_equal(got, want)


def test_padded_query_slots_select_nothing():
    fx = _fixture([8], [0], [2], seed=6, pad_tokens=4)
    got, want = _both(fx, k=2)
    np.testing.assert_array_equal(got, want)
    assert (got[8:] == INVALID_ENTRY).all()


def test_padded_key_rows_are_never_selected():
    fx = _fixture([8], [0], [2], seed=8, pad_entries=6)
    fx["keys"][2:] *= 1000.0  # make the padding look attractive
    got, want = _both(fx, k=3)
    np.testing.assert_array_equal(got, want)
    assert not np.any(got >= 2)


# --------------------------------------------------------------------------
# per-query group boundaries
# --------------------------------------------------------------------------


@pytest.mark.parametrize("prefix", [0, 1, 2, 3, 4])
def test_group_boundary_within_a_chunk(prefix):
    """Queries in one chunk straddling a boundary must see different histories."""
    fx = _fixture([5], [prefix], [3], seed=prefix)
    got, want = _both(fx, k=3)
    np.testing.assert_array_equal(got, want)
    counts = [(row != INVALID_ENTRY).sum() for row in got]
    expected = [
        min(3, (p + 1) // RATIO) for p in fx["query_positions"][: fx["valid_token_mask"].sum()]
    ]
    assert counts == expected
    assert counts == sorted(counts)  # never shrinks as the chunk advances


def test_decode_step_at_a_group_boundary():
    """A single-token decode right after a group closed must see it."""
    closed = _fixture([1], [4], [1], seed=12)  # position 4: entry 0 complete
    got, want = _both(closed, k=2)
    np.testing.assert_array_equal(got, want)
    assert (got[0] != INVALID_ENTRY).sum() == 1

    open_group = _fixture([1], [3], [1], seed=12)  # position 3 completes entry 0 too
    got2, want2 = _both(open_group, k=2)
    np.testing.assert_array_equal(got2, want2)
    assert (got2[0] != INVALID_ENTRY).sum() == 1

    before = _fixture([1], [2], [1], seed=12)  # position 2: nothing complete
    got3, _ = _both(before, k=2)
    assert (got3[0] == INVALID_ENTRY).all()


# --------------------------------------------------------------------------
# input validation
# --------------------------------------------------------------------------


def test_rejects_bad_shapes_and_budgets():
    fx = _fixture([4], [0], [1], seed=0)
    with pytest.raises(ValueError, match="k must be positive"):
        csa_indexer_topk(**fx, k=0, ratio=RATIO)
    with pytest.raises(ValueError, match="ratio must be positive"):
        visible_entries_for_query(np.array([0]), 0)
    with pytest.raises(ValueError, match="do not fold onto"):
        csa_indexer_scores(fx["q"], fx["weights"], fx["keys"], num_kv_heads=3)
    with pytest.raises(ValueError, match="weights must be"):
        csa_indexer_scores(fx["q"], fx["weights"][:, :1], fx["keys"])
    with pytest.raises(ValueError, match="q must be"):
        csa_indexer_scores(fx["q"][0], fx["weights"], fx["keys"])


def test_each_requests_first_completed_group_is_selectable():
    fx = _fixture([4, 4], [0, 0], [1, 1])
    got, _ = _both(fx, k=1)
    # Both requests have completed group zero, at different gathered row indices.
    np.testing.assert_array_equal(got[[3, 7]], [[0], [1]])
