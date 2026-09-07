"""M2.1 -- DeepSeek-V4 attention metadata derivation.

Fixed requests, positions, masks and page counts; no device and no model. The
cases are the ones the M2 acceptance list names: P=128/256, unaligned chunks,
batch reorder, padding, and group-completion boundaries.
"""

import numpy as np
import pytest

from sgl_jax.srt.layers.attention.dsv4.metadata import (
    INERT_BOUNDARY,
    boundary_capacity,
    complete_groups,
    derive_attention_metadata,
    visible_groups_for_positions,
)

HEAD_PAGE = 128


def _addresses(prefix_lens, q_lens, page_size, first_page=1):
    """Lay out one page run per request, the way C1's allocator does.

    Rank-local original token slots start at page 1 (page 0 is padding), and a
    request's position p sits at page offset ``p % page_size``.
    """
    history, swa, pages_per_request = [], [], []
    page = first_page
    for pre, n in zip(prefix_lens, q_lens):
        span = pre + n
        pages = max(1, -(-span // page_size)) if span else 0
        base = page * page_size
        for p in range(pre, pre + n):
            history.append(base + p)
            swa.append(1_000_000 + base + p)
        pages_per_request.append(pages)
        page += max(pages, 1)
    return (
        np.array(history, np.int64),
        np.array(swa, np.int64),
        np.array(pages_per_request, np.int64),
    )


def _derive(prefix_lens, q_lens, *, page_size=256, window_size=128, pad_tokens=0, **kw):
    prefix_lens = np.asarray(prefix_lens, np.int64)
    q_lens = np.asarray(q_lens, np.int64)
    history, swa, pages = _addresses(prefix_lens, q_lens, page_size)
    positions = np.concatenate(
        [np.arange(p, p + n) for p, n in zip(prefix_lens, q_lens) if n] or [np.empty(0, np.int64)]
    )
    live = positions.size
    total = live + pad_tokens
    pad = lambda a, fill: np.concatenate((a, np.full((pad_tokens,), fill, np.int64)))
    return derive_attention_metadata(
        q_lens=q_lens,
        prefix_lens=prefix_lens,
        positions=pad(positions, 0),
        request_slots=kw.pop("request_slots", np.arange(q_lens.size, dtype=np.int64)),
        history_write_loc=pad(history, 0),
        swa_write_loc=pad(swa, 0),
        pages_per_request=kw.pop("pages_per_request", pages),
        page_size=page_size,
        window_size=window_size,
        num_tokens=total,
        **kw,
    )


# --------------------------------------------------------------------------
# the two rules the rest of M2 has to agree with
# --------------------------------------------------------------------------


@pytest.mark.parametrize("ratio", [4, 128])
def test_complete_groups_counts_only_filled_groups(ratio):
    # ratio-1 tokens is not yet a group; ratio tokens is exactly one.
    assert complete_groups(np.array([0]), ratio)[0] == 0
    assert complete_groups(np.array([ratio - 1]), ratio)[0] == 0
    assert complete_groups(np.array([ratio]), ratio)[0] == 1
    assert complete_groups(np.array([2 * ratio - 1]), ratio)[0] == 1
    assert complete_groups(np.array([2 * ratio]), ratio)[0] == 2


def test_visible_groups_excludes_the_sliding_window():
    """A query only reaches compressed groups that are entirely older than its
    window; the window itself carries those tokens uncompressed."""
    ratio, window = 128, 128
    # Inside the first window: nothing compressed is reachable yet.
    assert visible_groups_for_positions(np.array([0, 127]), ratio, window).tolist() == [0, 0]
    # One full group has aged out of the window.
    assert visible_groups_for_positions(np.array([255]), ratio, window).tolist() == [1]
    assert visible_groups_for_positions(np.array([383]), ratio, window).tolist() == [2]
    # Boundary: position 254 is one short of retiring group 1.
    assert visible_groups_for_positions(np.array([254]), ratio, window).tolist() == [0]


def test_visible_groups_is_monotone_and_never_reaches_the_present():
    ratio, window = 4, 128
    pos = np.arange(0, 2048)
    vis = visible_groups_for_positions(pos, ratio, window)
    assert np.all(np.diff(vis) >= 0)
    # The newest group a query can see must end before its window opens.
    assert np.all(vis * ratio <= np.maximum(0, pos - window + 1))


def test_boundary_capacity_depends_only_on_padded_shape():
    # Same padded shape must give the same capacity regardless of content, or
    # every step with a different boundary count is a fresh compilation.
    assert boundary_capacity(512, 4, 128) == boundary_capacity(512, 4, 128)
    assert boundary_capacity(512, 4, 128) == 512 // 128 + 4
    assert boundary_capacity(0, 0, 4) == 1  # never zero-length


# --------------------------------------------------------------------------
# capacity vs visible -- the distinction the acceptance list calls out
# --------------------------------------------------------------------------


def test_reserved_capacity_is_not_reported_as_visible_data():
    """A fresh 256-token page reserves 64 ratio-4 entries and 2 ratio-128 entries
    while only a few groups have actually been produced."""
    md = _derive([0], [130], page_size=256)
    c4, c128 = md.c4, md.c128
    assert c4.visible_entries_after.tolist() == [130 // 4]  # 32
    assert c4.capacity_entries.tolist() == [256 // 4]  # 64
    assert c128.visible_entries_after.tolist() == [1]
    assert c128.capacity_entries.tolist() == [2]
    assert np.all(c4.capacity_entries >= c4.visible_entries_after)
    assert np.all(c128.capacity_entries >= c128.visible_entries_after)


def test_visible_before_and_after_bracket_this_step():
    md = _derive([100], [60], page_size=256)  # 100 -> 160
    assert md.c128.visible_entries_before.tolist() == [0]
    assert md.c128.visible_entries_after.tolist() == [1]
    assert md.c4.visible_entries_before.tolist() == [25]
    assert md.c4.visible_entries_after.tolist() == [40]


def test_capacity_below_completed_groups_is_rejected():
    """Under-reserved history must fail loudly here, not scatter out of bounds."""
    with pytest.raises(ValueError, match="compressed capacity is below"):
        _derive([0], [512], page_size=256, pages_per_request=np.array([1], np.int64))


# --------------------------------------------------------------------------
# group-completion boundaries
# --------------------------------------------------------------------------


def test_c128_boundary_at_127_plus_1():
    """The 127+1 case: a chunk that lands exactly on a group boundary."""
    first = _derive([0], [127], page_size=256)
    assert (first.c128.boundary_valid_mask).sum() == 0
    assert first.c128.visible_entries_after.tolist() == [0]

    second = _derive([127], [1], page_size=256)
    live = second.c128.boundary_valid_mask
    assert live.sum() == 1
    assert second.c128.boundary_token_indices[live].tolist() == [0]
    assert second.c128.boundary_group_ids[live].tolist() == [0]
    assert second.c128.visible_entries_after.tolist() == [1]


def test_c128_boundary_at_129_plus_127():
    """The 129+127 case: the group completes mid-chunk, not at either end."""
    md = _derive([129], [127], page_size=256)  # positions 129..255
    live = md.c128.boundary_valid_mask
    assert live.sum() == 1
    # position 255 completes group 1, and it is the last token of the chunk here.
    assert md.query_positions[md.c128.boundary_token_indices[live][0]] == 255
    assert md.c128.boundary_group_ids[live].tolist() == [1]
    # Group 0 was completed by an earlier chunk, so it is visible from the start.
    assert md.c128.visible_entries_before.tolist() == [1]
    assert md.c128.visible_entries_after.tolist() == [2]


def test_c4_boundaries_hit_every_fourth_position():
    md = _derive([0], [16], page_size=128)
    live = md.c4.boundary_valid_mask
    assert md.query_positions[md.c4.boundary_token_indices[live]].tolist() == [3, 7, 11, 15]
    assert md.c4.boundary_group_ids[live].tolist() == [0, 1, 2, 3]


@pytest.mark.parametrize("offset", [0, 1, 2, 3])
def test_c4_unaligned_chunk_start(offset):
    """Each residue mod 4: a chunk starting mid-group completes it and no more."""
    md = _derive([offset], [4], page_size=128)
    live = md.c4.boundary_valid_mask
    positions = md.query_positions[md.c4.boundary_token_indices[live]].tolist()
    expected = [p for p in range(offset, offset + 4) if (p + 1) % 4 == 0]
    assert positions == expected
    assert md.c4.visible_entries_before.tolist() == [offset // 4]
    assert md.c4.visible_entries_after.tolist() == [(offset + 4) // 4]


def test_chunk_shorter_than_a_group_produces_no_boundary():
    md = _derive([0], [3], page_size=128)
    assert md.c4.boundary_valid_mask.sum() == 0
    assert (md.c128.boundary_valid_mask).sum() == 0
    assert md.c4.visible_entries_after.tolist() == [0]


# --------------------------------------------------------------------------
# addresses
# --------------------------------------------------------------------------


@pytest.mark.parametrize("page_size", [128, 256])
def test_compressed_write_address_is_loc_over_ratio(page_size):
    """C1's contract: `loc // ratio` is the flat compressed-entry address."""
    md = _derive([0], [page_size], page_size=page_size)
    for meta in (md.c4, md.c128):
        live = meta.boundary_valid_mask
        tokens = meta.boundary_token_indices[live]
        assert np.array_equal(
            meta.boundary_write_entries[live],
            md.history_write_loc[tokens] // meta.ratio,
        )


@pytest.mark.parametrize("page_size", [128, 256])
def test_every_token_of_a_group_shares_one_compressed_address(page_size):
    """Only true because page_size % ratio == 0; if a group straddled a page,
    `loc // ratio` would name two different entries for one group."""
    md = _derive([0], [page_size], page_size=page_size)
    for ratio in (4, 128):
        loc = md.history_write_loc[: md.q_lens[0]].astype(np.int64)
        groups = md.query_positions[: md.q_lens[0]].astype(np.int64) // ratio
        entries = loc // ratio
        for g in np.unique(groups):
            assert len(np.unique(entries[groups == g])) == 1


def test_page_size_not_divisible_by_ratio_is_rejected():
    with pytest.raises(ValueError, match="must be 128 or 256"):
        _derive([0], [8], page_size=100)


# --------------------------------------------------------------------------
# padding, batch reorder, multiple requests
# --------------------------------------------------------------------------


def test_padded_token_slots_get_no_usable_address():
    md = _derive([0], [5], page_size=128, pad_tokens=11)
    assert md.num_tokens == 16
    assert md.valid_token_mask.tolist() == [True] * 5 + [False] * 11
    # Padding must not land on a real slot -- least of all slot 0.
    assert np.all(md.history_write_loc[5:] == -1)
    assert np.all(md.swa_write_loc[5:] == -1)


def test_padded_requests_are_inert():
    md = _derive([0, 0], [8, 0], page_size=128)
    assert md.request_valid_mask.tolist() == [True, False]
    assert md.c4.visible_entries_after.tolist() == [2, 0]
    assert md.c4.capacity_entries.tolist() == [32, 0]


def test_boundaries_carry_their_own_request_and_state_slot():
    """Multi-request: each boundary must name its own request, so a reorder or a
    recycled slot cannot send one request's group into another's state."""
    slots = np.array([7, 3], np.int64)
    md = _derive([0, 0], [8, 4], page_size=128, request_slots=slots)
    live = md.c4.boundary_valid_mask
    req = md.c4.boundary_request_ids[live]
    assert req.tolist() == [0, 0, 1]
    assert md.c4.boundary_state_slots[live].tolist() == [7, 7, 3]


def test_batch_order_does_not_change_any_per_request_result():
    """Reordering the batch permutes the rows and nothing else."""
    prefix, q, slots = [0, 100, 8], [16, 60, 4], [5, 2, 9]
    md = _derive(prefix, q, page_size=256, request_slots=np.array(slots, np.int64))
    order = [2, 0, 1]
    md_r = _derive(
        [prefix[i] for i in order],
        [q[i] for i in order],
        page_size=256,
        request_slots=np.array([slots[i] for i in order], np.int64),
    )
    for field in ("visible_entries_before", "visible_entries_after"):
        for name in ("c4", "c128"):
            a = getattr(getattr(md, name), field)
            b = getattr(getattr(md_r, name), field)
            assert a[order].tolist() == b.tolist(), (name, field)
    assert md.seq_lens[order].tolist() == md_r.seq_lens.tolist()
    assert md.request_slots[order].tolist() == md_r.request_slots.tolist()


def test_boundary_arrays_have_the_same_length_for_the_same_padded_shape():
    """Different boundary counts, identical array shapes -- the property that
    keeps a changing batch from recompiling."""
    # Both padded to a 256-token axis and 2 requests; only the content differs.
    # a: positions 0..127 per request, so position 127 completes group 0 twice.
    # b: positions 1..126 per request, which never reaches a multiple of 128.
    a = _derive([0, 0], [128, 128], page_size=256)
    b = _derive([1, 1], [126, 126], page_size=256, pad_tokens=4)
    assert a.num_tokens == b.num_tokens == 256
    assert a.num_requests == b.num_requests == 2
    for name in ("c4", "c128"):
        assert getattr(a, name).num_boundaries == getattr(b, name).num_boundaries
    # ...while the live counts genuinely differ.
    live_a = int((a.c128.boundary_valid_mask).sum())
    live_b = int((b.c128.boundary_valid_mask).sum())
    assert (live_a, live_b) == (2, 0)


# --------------------------------------------------------------------------
# input validation
# --------------------------------------------------------------------------


def test_positions_must_match_the_declared_lengths():
    with pytest.raises(ValueError, match="positions must be"):
        derive_attention_metadata(
            q_lens=np.array([4]),
            prefix_lens=np.array([0]),
            positions=np.array([0, 1, 2, 9]),  # 9 is not position 3
            request_slots=np.array([0]),
            history_write_loc=np.arange(128, 132),
            swa_write_loc=np.arange(128, 132),
            pages_per_request=np.array([1]),
            page_size=128,
            window_size=128,
        )


def test_state_init_is_refused_for_a_continuing_chunk():
    """Re-initialising mid-request would throw away the accumulated state; only a
    request restarting from zero may reset."""
    with pytest.raises(ValueError, match="only be initialised for a request starting from zero"):
        _derive([64], [4], page_size=128, state_init_mask=np.array([True]))
    md = _derive([0], [4], page_size=128, state_init_mask=np.array([True]))
    assert md.state_init_mask.tolist() == [True]


def test_token_axis_must_hold_every_query_token():
    """A token axis shorter than the live tokens would silently truncate the
    batch, so it is rejected."""
    with pytest.raises(ValueError, match="cannot hold"):
        derive_attention_metadata(
            q_lens=np.array([16]),
            prefix_lens=np.array([0]),
            positions=np.arange(16),
            request_slots=np.array([0]),
            history_write_loc=np.arange(128, 144),
            swa_write_loc=np.arange(128, 144),
            pages_per_request=np.array([1]),
            page_size=128,
            window_size=128,
            num_tokens=8,
        )


def test_ratio_lookup_rejects_unknown_ratios():
    md = _derive([0], [4], page_size=128)
    assert md.ratio(4) is md.c4
    assert md.ratio(128) is md.c128
    with pytest.raises(ValueError, match="no V4 metadata for compression ratio 8"):
        md.ratio(8)


def test_metadata_is_a_pytree_that_survives_a_round_trip():
    import jax

    md = _derive([0, 4], [8, 8], page_size=128)
    leaves, treedef = jax.tree_util.tree_flatten(md)
    back = jax.tree_util.tree_unflatten(treedef, leaves)
    assert back.page_size == md.page_size and back.window_size == md.window_size
    assert back.c4.ratio == 4 and back.c128.ratio == 128
    assert np.array_equal(back.query_positions, md.query_positions)
    assert np.array_equal(back.c4.boundary_write_entries, md.c4.boundary_write_entries)


def test_padded_boundary_slots_cannot_wrap_into_live_data():
    """The reason the token sentinel is `num_tokens` and not -1: JAX wraps
    negative indices, so -1 would gather the last live token instead of nothing."""
    md = _derive([0], [5], page_size=128, pad_tokens=3)
    pad = ~md.c128.boundary_valid_mask
    assert pad.all()  # 5 tokens completes no ratio-128 group
    # Out of range on the query axis, so a gather is droppable, and never negative.
    assert np.all(md.c128.boundary_token_indices[pad] >= md.num_tokens)
    assert not np.any(md.c128.boundary_token_indices < 0)
    # The non-index payload stays at the inert fill and is gated by the mask.
    assert np.all(md.c128.boundary_group_ids[pad] == INERT_BOUNDARY)
    assert np.all(md.c128.boundary_state_slots[pad] == INERT_BOUNDARY)
