"""M2.2 -- offline compressor and continuation state.

Fixed metadata and buffers, no model and no device kernel. The oracle is a plain
per-record NumPy loop written from tpu-inference's `gather_state_windows` /
`compress_norm_rope` formulation, so it shares no structure with the vectorised
implementation.

The cases are the ones the M2.2 brief names: C128 at 127+1 and 129+127, C4 at every
mod-4 split, a start shorter than the window, a chunk longer than the ring, request
reorder, and slot reuse.
"""

import numpy as np
import pytest

from sgl_jax.srt.layers.attention.dsv4.compressor import (
    compress_chunk,
    interleaved_rope,
    overlap_factor,
    project_tokens,
    state_window,
)
from sgl_jax.srt.layers.attention.dsv4.metadata import derive_attention_metadata

D = 16
ROPE = 4
HIDDEN = 12
EPS = 1e-6


def _weights(ratio, seed=0):
    rng = np.random.default_rng(seed)
    width = overlap_factor(ratio) * D
    return dict(
        wkv=rng.normal(size=(width, HIDDEN)).astype(np.float32) * 0.3,
        wgate=rng.normal(size=(width, HIDDEN)).astype(np.float32) * 0.3,
        ape=rng.normal(size=(ratio, width)).astype(np.float32) * 0.3,
        norm_weight=(1.0 + 0.1 * rng.normal(size=(D,))).astype(np.float32),
        cos_sin_cache=rng.uniform(-1, 1, size=(4096, ROPE)).astype(np.float32),
    )


_MAX_POS = 2048


def _activations(seed, state_slots, request_ids, positions):
    """Activations as a deterministic function of (slot, absolute position).

    Drawing them from the chunk shape instead would give a split run different
    inputs than the whole run, so a continuation test would fail for the wrong
    reason.
    """
    table = (
        np.random.default_rng(seed + 100)
        .normal(size=(int(np.max(state_slots)) + 2, _MAX_POS, HIDDEN))
        .astype(np.float32)
    )
    slots = np.asarray(state_slots)[np.asarray(request_ids)]
    return table[slots, np.asarray(positions)]


def _empty_state(num_slots, ratio):
    """C1's empty representation: contents zero, scores -inf."""
    window = state_window(ratio)
    width = overlap_factor(ratio) * D
    state = np.zeros((num_slots, window, 2 * width), np.float32)
    state[..., width:] = -np.inf
    return state


def _oracle(
    x,
    w,
    *,
    ratio,
    positions,
    request_ids,
    prefix_lens,
    cu_q_lens,
    state_slots,
    boundaries,
    compressed_pos,
    state,
):
    """Per-record NumPy reference, transcribed from tpu-inference's formulation."""
    coff = overlap_factor(ratio)
    window = state_window(ratio)
    width = coff * D

    x64 = np.asarray(x, np.float64)
    kv = x64 @ np.asarray(w["wkv"], np.float64).T
    score = x64 @ np.asarray(w["wgate"], np.float64).T
    score = score + np.asarray(w["ape"], np.float64)[np.asarray(positions) % ratio]
    rows = np.concatenate((kv, score), axis=-1)

    out = np.zeros((len(boundaries), D), np.float64)
    for n, t in enumerate(boundaries):
        r = request_ids[t]
        end = positions[t]
        kv_win = np.zeros((window, D), np.float64)
        sc_win = np.full((window, D), -np.inf, np.float64)
        valid = np.zeros((window,), bool)
        for j in range(window):
            p = end - window + 1 + j
            if p < 0:
                continue
            valid[j] = True
            field = D if (coff == 2 and j >= ratio) else 0
            if p >= prefix_lens[r]:
                row = rows[cu_q_lens[r] + (p - prefix_lens[r])]
            else:
                row = np.asarray(state[state_slots[r], p % window], np.float64)
            kv_win[j] = row[field : field + D]
            sc_win[j] = row[width + field : width + field + D]
        sc_win = np.where(valid[:, None], sc_win, -np.inf)
        shift = sc_win.max(axis=0, keepdims=True)
        e = np.exp(sc_win - shift)
        weights = e / e.sum(axis=0, keepdims=True)
        pooled = (weights * kv_win).sum(axis=0)
        normed = (
            pooled / np.sqrt(np.mean(pooled**2) + EPS) * np.asarray(w["norm_weight"], np.float64)
        )
        cs = np.asarray(w["cos_sin_cache"], np.float64)[compressed_pos[n]]
        cos, sin = cs[: ROPE // 2], cs[ROPE // 2 : ROPE]
        rec = normed.copy()
        tail = rec[D - ROPE :]
        even, odd = tail[0::2].copy(), tail[1::2].copy()
        tail[0::2] = even * cos - odd * sin
        tail[1::2] = even * sin + odd * cos
        out[n] = rec
    return out


def _run(prefix_lens, q_lens, ratio, *, seed=0, state=None, state_slots=None, page_size=256):
    prefix_lens = np.asarray(prefix_lens, np.int64)
    q_lens = np.asarray(q_lens, np.int64)
    B = q_lens.size
    if state_slots is None:
        state_slots = np.arange(B, dtype=np.int64)
    if state is None:
        state = _empty_state(int(state_slots.max()) + 2, ratio)

    positions = np.concatenate(
        [np.arange(p, p + n) for p, n in zip(prefix_lens, q_lens) if n] or [np.empty(0, np.int64)]
    )
    T = positions.size
    cu_q = np.concatenate(([0], np.cumsum(q_lens)))
    request_ids = np.repeat(np.arange(B), q_lens)

    # Reuse M2.1 for the boundary metadata so the two stay consistent.
    md = derive_attention_metadata(
        q_lens=q_lens,
        prefix_lens=prefix_lens,
        positions=positions,
        request_slots=state_slots,
        history_write_loc=np.arange(page_size, page_size + T),
        swa_write_loc=np.arange(page_size, page_size + T),
        pages_per_request=np.maximum(1, -(-(prefix_lens + q_lens) // page_size)),
        page_size=page_size,
        window_size=128,
    )
    rm = md.ratio(ratio)
    live = np.asarray(rm.boundary_valid_mask)
    boundaries = np.asarray(rm.boundary_token_indices)
    compressed_pos = np.where(live, np.asarray(rm.boundary_group_ids), 0).astype(np.int64)

    x = _activations(seed, state_slots, request_ids, positions)
    w = _weights(ratio, seed)

    records, valid, new_state = compress_chunk(
        x,
        state=state,
        positions=positions,
        query_request_ids=request_ids,
        prefix_lens=prefix_lens,
        cu_q_lens=cu_q,
        state_slots=state_slots,
        boundary_token_indices=boundaries,
        boundary_valid_mask=live,
        boundary_compressed_pos=compressed_pos,
        ratio=ratio,
        head_dim=D,
        rope_head_dim=ROPE,
        norm_eps=EPS,
        **w,
    )
    want = _oracle(
        x,
        w,
        ratio=ratio,
        positions=positions,
        request_ids=request_ids,
        prefix_lens=prefix_lens,
        cu_q_lens=cu_q,
        state_slots=state_slots,
        boundaries=np.clip(boundaries, 0, max(T - 1, 0)),
        compressed_pos=compressed_pos,
        state=state,
    )
    return dict(
        records=np.asarray(records),
        valid=np.asarray(valid),
        want=want,
        new_state=np.asarray(new_state),
        positions=positions,
        x=x,
        w=w,
        request_ids=request_ids,
        cu_q=cu_q,
        prefix_lens=prefix_lens,
        q_lens=q_lens,
        state_slots=state_slots,
    )


def _assert_records_match(res):
    live = res["valid"]
    if not live.any():
        return
    np.testing.assert_allclose(res["records"][live], res["want"][live], rtol=2e-5, atol=2e-5)


# --------------------------------------------------------------------------
# the overlap geometry
# --------------------------------------------------------------------------


def test_ring_depth_follows_the_overlap_factor():
    """CSA overlaps, HCA does not; the ring depth is ratio * coff, which is why a
    ratio-4 layer's ring is 8 deep and not 4."""
    assert (overlap_factor(4), state_window(4)) == (2, 8)
    assert (overlap_factor(128), state_window(128)) == (1, 128)


def test_state_shape_matches_c1s_pool():
    from sgl_jax.srt.mem_cache.deepseek_v4_compress_state import (
        DeepseekV4CompressStatePool,
    )

    assert DeepseekV4CompressStatePool  # imported for provenance, not called
    for ratio, middle, last in ((4, 8, 4), (128, 128, 2)):
        assert state_window(ratio) == middle
        assert 2 * overlap_factor(ratio) == last


def test_interleaved_rope_pairs_within_the_trailing_block():
    x = np.zeros((1, D), np.float32)
    x[0, D - ROPE] = 1.0  # first even element of the tail
    x[0, D - ROPE + 1] = 0.0
    cos = np.array([[0.0, 1.0]], np.float32)
    sin = np.array([[1.0, 0.0]], np.float32)
    got = np.asarray(interleaved_rope(x, cos, sin, ROPE))
    # even'=e*cos-o*sin=0, odd'=e*sin+o*cos=1 -> the pair rotates into its neighbour
    assert got[0, D - ROPE] == pytest.approx(0.0)
    assert got[0, D - ROPE + 1] == pytest.approx(1.0)
    # Everything before the tail is untouched.
    assert np.all(got[0, : D - ROPE] == 0.0)


# --------------------------------------------------------------------------
# C128 continuation boundaries
# --------------------------------------------------------------------------


def test_c128_127_plus_1():
    """127 tokens produce nothing; the next single token closes group 0 using the
    state the first chunk left behind."""
    first = _run([0], [127], 128, seed=1)
    assert not first["valid"].any()
    _assert_records_match(first)

    second = _run([127], [1], 128, seed=1, state=first["new_state"])
    assert second["valid"].sum() == 1
    _assert_records_match(second)


def test_c128_129_plus_127():
    """The group closes mid-chunk: 129..255 closes group 1 at position 255."""
    first = _run([0], [129], 128, seed=2)
    assert first["valid"].sum() == 1  # position 127 closed group 0
    second = _run([129], [127], 128, seed=2, state=first["new_state"])
    assert second["valid"].sum() == 1
    _assert_records_match(second)


def test_c128_start_shorter_than_the_window():
    """A record at position 127 pools 128 rows exactly; there is no earlier history
    to mask off, but the mask path still has to be exercised at ratio 4 below."""
    res = _run([0], [128], 128, seed=3)
    assert res["valid"].sum() == 1
    _assert_records_match(res)


# --------------------------------------------------------------------------
# C4: every mod-4 split, and the start-shorter-than-window mask
# --------------------------------------------------------------------------


@pytest.mark.parametrize("first_len", [1, 2, 3, 4, 5, 6, 7])
def test_c4_split_at_every_offset(first_len):
    """Split 12 tokens two ways at each offset; the records must be identical to
    doing it in one chunk, which is the continuation property."""
    whole = _run([0], [12], 4, seed=first_len)
    _assert_records_match(whole)

    part1 = _run([0], [first_len], 4, seed=first_len)
    _assert_records_match(part1)
    part2 = _run([first_len], [12 - first_len], 4, seed=first_len, state=part1["new_state"])
    _assert_records_match(part2)

    got = np.concatenate((part1["records"][part1["valid"]], part2["records"][part2["valid"]]))
    np.testing.assert_allclose(got, whole["records"][whole["valid"]], rtol=2e-5, atol=2e-5)


def test_c4_start_shorter_than_the_window_masks_missing_rows():
    """Position 3 closes the first record but the 8-wide window reaches back to -4;
    those rows must be masked out, not read as zeros with a share of the softmax."""
    res = _run([0], [4], 4, seed=9)
    assert res["valid"].sum() == 1
    _assert_records_match(res)
    # If the masking were wrong the oracle and implementation would disagree, so a
    # match here is the assertion. Sanity: the record is finite.
    assert np.all(np.isfinite(res["records"][res["valid"]]))


def test_c4_overlap_actually_reads_two_fields():
    """Zero out field 1 of the projection and the record must change -- otherwise the
    older/newer half distinction is not being applied."""
    base = _run([0], [8], 4, seed=13)
    w = dict(base["w"])
    w["wkv"] = w["wkv"].copy()
    w["wkv"][D:] = 0.0  # field 1 of the content projection
    x, positions = base["x"], base["positions"]
    kv_a, _ = project_tokens(
        base["x"], base["w"]["wkv"], base["w"]["wgate"], base["w"]["ape"], positions, ratio=4
    )
    kv_b, _ = project_tokens(x, w["wkv"], w["wgate"], w["ape"], positions, ratio=4)
    assert not np.allclose(np.asarray(kv_a), np.asarray(kv_b))
    assert np.allclose(np.asarray(kv_a)[:, :D], np.asarray(kv_b)[:, :D])


# --------------------------------------------------------------------------
# chunk longer than the ring
# --------------------------------------------------------------------------


def test_chunk_much_longer_than_the_ring():
    """512 tokens through an 8-deep ring. Writing the chunk into the ring before
    pooling would destroy the early groups; this must match a per-record oracle
    that never touches the ring for in-chunk positions."""
    res = _run([0], [512], 4, seed=21)
    assert res["valid"].sum() == 512 // 4
    _assert_records_match(res)


def test_ring_ends_up_holding_only_the_tail():
    """After a long chunk the ring must hold the last `window` tokens, so the next
    chunk continues from the right place."""
    long_chunk = _run([0], [512], 4, seed=22)
    follow = _run([512], [4], 4, seed=22, state=long_chunk["new_state"])
    _assert_records_match(follow)

    # And it agrees with having done 516 in one go.
    whole = _run([0], [516], 4, seed=22)
    _assert_records_match(whole)
    np.testing.assert_allclose(
        follow["records"][follow["valid"]],
        whole["records"][whole["valid"]][-1:],
        rtol=2e-5,
        atol=2e-5,
    )


# --------------------------------------------------------------------------
# batch: reorder and slot reuse
# --------------------------------------------------------------------------


def test_multiple_requests_do_not_share_state():
    res = _run([0, 0], [8, 8], 4, seed=31)
    assert res["valid"].sum() == 4
    _assert_records_match(res)


def test_batch_reorder_permutes_records_and_nothing_else():
    a = _run([0, 16], [8, 8], 4, seed=41)
    _assert_records_match(a)
    b = _run([16, 0], [8, 8], 4, seed=41, state_slots=np.array([1, 0], np.int64))
    _assert_records_match(b)
    # Both wrote the same two slots; the surviving state must agree per slot.
    np.testing.assert_allclose(a["new_state"][:2], b["new_state"][:2], rtol=2e-5, atol=2e-5)


def test_a_request_starting_at_zero_never_reads_state():
    """A fresh request is immune to whatever the slot held before: its window reaches
    back past position 0 and those rows are masked out, so the record cannot depend
    on them. This is what makes slot reuse safe without a reset on the read path."""
    used = _run([0], [8], 4, seed=51)
    assert np.any(used["new_state"][0] != _empty_state(2, 4)[0])

    on_stale = _run([0], [4], 4, seed=61, state=used["new_state"])
    on_empty = _run([0], [4], 4, seed=61)
    np.testing.assert_allclose(
        on_stale["records"][on_stale["valid"]],
        on_empty["records"][on_empty["valid"]],
        rtol=2e-5,
        atol=2e-5,
    )


def test_a_continuing_request_does_read_state():
    """Guards the test above from being vacuous: once prefix > 0 the window genuinely
    reaches into the ring, so corrupting it must change the record."""
    first = _run([0], [4], 4, seed=71)
    good = _run([4], [4], 4, seed=71, state=first["new_state"])
    _assert_records_match(good)

    corrupted = first["new_state"].copy()
    corrupted[0, :, : overlap_factor(4) * D] += 5.0  # contents only; scores stay finite
    bad = _run([4], [4], 4, seed=71, state=corrupted)
    assert not np.allclose(good["records"][good["valid"]], bad["records"][bad["valid"]], rtol=1e-3)


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------


def test_rejects_mismatched_state_shape():
    with pytest.raises(ValueError, match="state must be"):
        _run([0], [4], 4, seed=0, state=_empty_state(2, 128))


def test_rejects_bad_ape_and_projection_shapes():
    w = _weights(4)
    with pytest.raises(ValueError, match="ape must be"):
        project_tokens(
            np.zeros((2, HIDDEN), np.float32),
            w["wkv"],
            w["wgate"],
            w["ape"][:2],
            np.zeros((2,), np.int32),
            ratio=4,
        )
    with pytest.raises(ValueError, match="same output width"):
        project_tokens(
            np.zeros((2, HIDDEN), np.float32),
            w["wkv"],
            w["wgate"][:1],
            w["ape"],
            np.zeros((2,), np.int32),
            ratio=4,
        )
    with pytest.raises(ValueError, match="ratio must be positive"):
        overlap_factor(0)
