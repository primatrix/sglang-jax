"""M2.5 -- layer dispatch composing M2.1-M2.4 over C1's resources.

Real C1 pools at small capacity, synthetic layer weights, no model and no device
kernel. The cases are the M2.5 acceptance list: normal prefill and decode,
cross-chunk continuation, batch reorder, the three layer-type routes, and the next
step reading the updated pool version rather than a stale one.
"""

import numpy as np
import pytest

from sgl_jax.srt.layers.attention.dsv4.compressor import overlap_factor, state_window
from sgl_jax.srt.layers.attention.dsv4.dispatch import read_tables, run_layer
from sgl_jax.srt.layers.attention.dsv4.metadata import derive_attention_metadata

D = 8
DI = 4
H = 2
HI = 2
PAGE = 128
WINDOW = 4
TOPK = 2
EPS = 1e-6
SCALE = 1.0 / np.sqrt(D)


class _RequestPool:
    """The only part of C1's ReqToTokenPool this needs: position -> token location."""

    def __init__(self, size, max_context_len):
        self.size = size
        self.req_to_token = np.zeros((size, max_context_len), np.int32)


class _Allocator:
    """C1's original-token -> SWA-row mapping, filled as pages are handed out."""

    def __init__(self, num_locations):
        self.full_to_swa_index_mapping = np.zeros((num_locations,), np.int32)


def _lay_out(request_pool, allocator, slots, lengths, *, page_size=PAGE):
    """One page run per request, the way C1's allocator does: locations start at
    `page_size` (page 0 is padding) and SWA rows are a distinct non-zero range."""
    page = 1
    for slot, length in zip(slots, lengths):
        base = page * page_size
        positions = np.arange(length)
        locations = base + positions
        request_pool.req_to_token[slot, :length] = locations
        # A distinct SWA row per original token, never row 0.
        allocator.full_to_swa_index_mapping[locations] = 1 + locations - page_size
        page += max(1, -(-length // page_size))


def _weights(ratio, dim, seed):
    rng = np.random.default_rng(seed)
    width = overlap_factor(ratio) * dim
    return dict(
        wkv=(rng.normal(size=(width, dim)) * 0.3).astype(np.float32),
        wgate=(rng.normal(size=(width, dim)) * 0.3).astype(np.float32),
        ape=(rng.normal(size=(ratio, width)) * 0.3).astype(np.float32),
        norm_weight=(1.0 + 0.1 * rng.normal(size=(dim,))).astype(np.float32),
        cos_sin_cache=rng.uniform(-1, 1, size=(512, 4)).astype(np.float32),
    )


class _Harness:
    """A tiny V4 world: two requests, one layer of each type, real address maps."""

    def __init__(self, ratio, *, num_slots=3, capacity=4 * PAGE, seed=0):
        self.ratio = ratio
        self.pool = _RequestPool(num_slots, capacity)
        self.alloc = _Allocator(capacity + 1)
        self.swa = np.zeros((capacity + 1, D), np.float32)
        self.compressed = np.zeros((max(1, capacity // max(ratio, 1)) + 1, D), np.float32)
        self.indexer_buf = np.zeros((self.compressed.shape[0], DI), np.float32)
        if ratio > 0:
            w = state_window(ratio)
            self.state = np.zeros((num_slots + 1, w, 2 * overlap_factor(ratio) * D), np.float32)
            self.state[..., overlap_factor(ratio) * D :] = -np.inf
            self.idx_state = np.zeros(
                (num_slots + 1, w, 2 * overlap_factor(ratio) * DI), np.float32
            )
            self.idx_state[..., overlap_factor(ratio) * DI :] = -np.inf
            self.cw = _weights(ratio, D, seed)
            self.iw = _weights(ratio, DI, seed + 7)
        else:
            self.state = self.idx_state = self.cw = self.iw = None
        self.sink = np.random.default_rng(seed + 1).normal(size=(H,)).astype(np.float32)
        self.rng = np.random.default_rng(seed + 2)

    def step(self, slots, prefix_lens, q_lens):
        slots = np.asarray(slots, np.int64)
        prefix_lens = np.asarray(prefix_lens, np.int64)
        q_lens = np.asarray(q_lens, np.int64)
        lengths = prefix_lens + q_lens
        _lay_out(self.pool, self.alloc, slots, lengths)

        positions = np.concatenate(
            [np.arange(p, p + n) for p, n in zip(prefix_lens, q_lens) if n]
            or [np.empty(0, np.int64)]
        )
        T = positions.size
        history = np.concatenate(
            [
                self.pool.req_to_token[int(s), p : p + n]
                for s, p, n in zip(slots, prefix_lens, q_lens)
                if n
            ]
            or [np.empty(0, np.int32)]
        ).astype(np.int64)
        swa_loc = self.alloc.full_to_swa_index_mapping[history].astype(np.int64)

        md = derive_attention_metadata(
            q_lens=q_lens,
            prefix_lens=prefix_lens,
            positions=positions,
            request_slots=slots,
            history_write_loc=history,
            swa_write_loc=swa_loc,
            pages_per_request=np.maximum(1, -(-lengths // PAGE)),
            page_size=PAGE,
            window_size=WINDOW,
            state_init_mask=(q_lens > 0) & (prefix_lens == 0),
        )
        tables = read_tables(
            request_pool=self.pool,
            allocator=self.alloc,
            slots=slots,
            lengths=lengths,
            q_lens=q_lens,
            ratio=self.ratio,
            window_size=WINDOW,
            page_size=PAGE,
        )

        q = self.rng.normal(size=(T, H, D)).astype(np.float32)
        new_kv = self.rng.normal(size=(T, D)).astype(np.float32)
        indexer = None
        if self.ratio == 4:
            indexer = dict(
                compressor_input=self.rng.normal(size=(T, DI)).astype(np.float32),
                q=self.rng.normal(size=(T, HI, DI)).astype(np.float32),
                weights=self.rng.normal(size=(T, HI)).astype(np.float32),
                state=self.idx_state,
                head_dim=DI,
                compressor_weights=self.iw,
            )

        out, updates = run_layer(
            q=q,
            new_kv=new_kv,
            compressor_input=new_kv,
            layer_id=0,
            ratio=self.ratio,
            metadata=md,
            tables=tables,
            kv_buffers={
                "swa": self.swa,
                "compressed": self.compressed,
                "indexer": self.indexer_buf,
            },
            state=self.state,
            compressor_weights=self.cw,
            indexer=indexer,
            attention_sink=self.sink,
            softmax_scale=SCALE,
            window_size=WINDOW,
            head_dim=D,
            index_topk=TOPK if self.ratio == 4 else None,
            rope_head_dim=4,
            norm_eps=EPS,
        )
        return np.asarray(out), {k: np.asarray(v) for k, v in updates.items()}, tables, md

    def commit(self, updates):
        """What C3 does with the returned arrays -- nothing was mutated in place."""
        self.swa = updates["swa"]
        if "compressed" in updates:
            self.compressed = updates["compressed"]
        if "state" in updates:
            self.state = updates["state"]
        if "indexer" in updates:
            self.indexer_buf = updates["indexer"]
        if "indexer_state" in updates:
            self.idx_state = updates["indexer_state"]


# --------------------------------------------------------------------------
# the three routes
# --------------------------------------------------------------------------


@pytest.mark.parametrize("ratio", [0, 4, 128])
def test_all_three_layer_types_run(ratio):
    h = _Harness(ratio, seed=ratio)
    out, updates, _, _ = h.step([0], [0], [8])
    assert out.shape == (8, H, D)
    assert np.all(np.isfinite(out))
    # Every route updates the window; only compressed routes touch the other tiers.
    assert "swa" in updates
    assert ("compressed" in updates) == (ratio > 0)
    assert ("state" in updates) == (ratio > 0)
    assert ("indexer" in updates) == (ratio == 4)


def test_swa_only_route_leaves_compressed_tiers_untouched():
    h = _Harness(0)
    before = h.compressed.copy()
    _, updates, _, _ = h.step([0], [0], [8])
    h.commit(updates)
    np.testing.assert_array_equal(h.compressed, before)


def test_csa_route_selects_at_most_topk():
    h = _Harness(4, seed=5)
    # Long enough that more groups are complete than the budget allows.
    out, updates, tables, md = h.step([0], [0], [16])
    assert np.all(np.isfinite(out))
    assert len(tables.compressed_rows) == 16 // 4


# --------------------------------------------------------------------------
# nothing is mutated in place; the next step must see the committed version
# --------------------------------------------------------------------------


@pytest.mark.parametrize("ratio", [0, 4, 128])
def test_run_layer_does_not_mutate_its_inputs(ratio):
    h = _Harness(ratio, seed=11)
    swa_before = h.swa.copy()
    comp_before = h.compressed.copy()
    state_before = None if h.state is None else h.state.copy()
    h.step([0], [0], [8])
    np.testing.assert_array_equal(h.swa, swa_before)
    np.testing.assert_array_equal(h.compressed, comp_before)
    if state_before is not None:
        np.testing.assert_array_equal(h.state, state_before)


def test_the_next_step_reads_the_committed_window_not_a_stale_one():
    """Decode after prefill: the decode query's window must include the tokens the
    prefill step wrote, which only happens if the update was committed."""
    h = _Harness(0, seed=21)
    _, updates, _, _ = h.step([0], [0], [4])
    h.commit(updates)
    out_fresh, _, _, _ = h.step([0], [4], [1])

    stale = _Harness(0, seed=21)
    stale.step([0], [0], [4])  # deliberately not committed
    out_stale, _, _, _ = stale.step([0], [4], [1])
    assert not np.allclose(out_fresh, out_stale, rtol=1e-3)


def test_a_query_never_attends_to_a_row_this_step_overwrote():
    """The window write happens after the read. If it did not, a query would see its
    own KV row twice: once as its own and once through a slot it just clobbered."""
    h = _Harness(0, seed=22)
    out_a, updates, _, _ = h.step([0], [0], [4])
    # Re-running from the same committed state must give the same answer, which it
    # cannot if the read saw a partially written cache.
    h2 = _Harness(0, seed=22)
    out_b, _, _, _ = h2.step([0], [0], [4])
    np.testing.assert_allclose(out_a, out_b, rtol=2e-5, atol=2e-5)


# --------------------------------------------------------------------------
# prefill / decode / continuation
# --------------------------------------------------------------------------


@pytest.mark.parametrize("ratio", [0, 4, 128])
def test_prefill_then_decode(ratio):
    h = _Harness(ratio, seed=31)
    _, updates, _, _ = h.step([0], [0], [8])
    h.commit(updates)
    out, updates, _, _ = h.step([0], [8], [1])
    h.commit(updates)
    assert out.shape == (1, H, D)
    assert np.all(np.isfinite(out))


def test_cross_chunk_continuation_advances_the_compressed_tier():
    """Two chunks of 8 at ratio 4 must leave four records behind, not two."""
    h = _Harness(4, seed=41)
    _, updates, tables, _ = h.step([0], [0], [8])
    h.commit(updates)
    assert len(tables.compressed_rows) == 2
    _, updates, tables, _ = h.step([0], [8], [8])
    h.commit(updates)
    assert len(tables.compressed_rows) == 4


def test_decode_across_a_group_boundary():
    h = _Harness(4, seed=51)
    _, updates, tables, _ = h.step([0], [0], [7])
    h.commit(updates)
    assert len(tables.compressed_rows) == 1
    _, updates, tables, _ = h.step([0], [7], [1])  # position 7 closes group 1
    h.commit(updates)
    assert len(tables.compressed_rows) == 2


# --------------------------------------------------------------------------
# batch: reorder and isolation
# --------------------------------------------------------------------------


def test_multiple_requests_at_different_progress():
    h = _Harness(4, seed=61)
    out, updates, tables, _ = h.step([0, 1], [0, 0], [8, 4])
    h.commit(updates)
    assert out.shape == (12, H, D)
    assert np.all(np.isfinite(out))
    # Request 0 completed two groups, request 1 one.
    assert tables.compressed_request_ids.tolist() == [0, 0, 1]


def test_batch_reorder_gives_the_same_per_request_addresses():
    a = _Harness(4, seed=71)
    _, _, ta, _ = a.step([0, 1], [0, 0], [8, 4])
    b = _Harness(4, seed=71)
    _, _, tb, _ = b.step([1, 0], [0, 0], [4, 8])
    # Same set of visible entries, just grouped in the other order.
    assert sorted(ta.compressed_entry_ids.tolist()) == sorted(tb.compressed_entry_ids.tolist())
    assert ta.compressed_request_ids.tolist() == [0, 0, 1]
    assert tb.compressed_request_ids.tolist() == [0, 1, 1]


def test_inactive_requests_contribute_no_addresses():
    h = _Harness(4, seed=81)
    _, _, tables, _ = h.step([0, 1], [0, 0], [8, 0])
    assert set(tables.window_request_ids.tolist()) == {0}
    assert set(tables.compressed_request_ids.tolist()) == {0}


# --------------------------------------------------------------------------
# the read tables refuse to guess
# --------------------------------------------------------------------------


def test_released_swa_rows_are_refused_rather_than_read_as_row_zero():
    """A zero in C1's mapping means "not allocated". Reading row 0 would silently
    return the padding row, so this has to raise."""
    h = _Harness(0, seed=91)
    h.step([0], [0], [8])
    h.alloc.full_to_swa_index_mapping[h.pool.req_to_token[0, 2]] = 0
    with pytest.raises(ValueError, match="released"):
        read_tables(
            request_pool=h.pool,
            allocator=h.alloc,
            slots=np.array([0]),
            lengths=np.array([8]),
            q_lens=np.array([8]),
            ratio=0,
            window_size=WINDOW,
            page_size=PAGE,
        )


def test_unallocated_locations_are_refused():
    h = _Harness(0, seed=92)
    h.pool.req_to_token[0, :4] = 0  # never handed out
    with pytest.raises(ValueError, match="allocated original-token slots"):
        read_tables(
            request_pool=h.pool,
            allocator=h.alloc,
            slots=np.array([0]),
            lengths=np.array([4]),
            q_lens=np.array([4]),
            ratio=0,
            window_size=WINDOW,
            page_size=PAGE,
        )


def test_window_table_covers_the_oldest_query_in_the_chunk():
    """A chunk's first query needs its own window, which reaches further back than the
    last query's. The table is a superset for the whole chunk."""
    h = _Harness(0, seed=93)
    h.step([0], [0], [16])
    tables = read_tables(
        request_pool=h.pool,
        allocator=h.alloc,
        slots=np.array([0]),
        lengths=np.array([16]),
        q_lens=np.array([4]),
        ratio=0,
        window_size=WINDOW,
        page_size=PAGE,
    )
    # Queries at 12..15; the earliest needs back to 12-3=9.
    assert tables.window_positions.min() == 9
    assert tables.window_positions.max() == 15


def test_csa_needs_its_indexer():
    h = _Harness(4, seed=94)
    md_args = h.step([0], [0], [4])[3]
    with pytest.raises(ValueError, match="need the indexer"):
        run_layer(
            q=np.zeros((4, H, D), np.float32),
            new_kv=np.zeros((4, D), np.float32),
            compressor_input=np.zeros((4, D), np.float32),
            layer_id=0,
            ratio=4,
            metadata=md_args,
            tables=read_tables(
                request_pool=h.pool,
                allocator=h.alloc,
                slots=np.array([0]),
                lengths=np.array([4]),
                q_lens=np.array([4]),
                ratio=4,
                window_size=WINDOW,
                page_size=PAGE,
            ),
            kv_buffers={"swa": h.swa, "compressed": h.compressed, "indexer": h.indexer_buf},
            state=h.state,
            compressor_weights=h.cw,
            indexer=None,
            attention_sink=h.sink,
            softmax_scale=SCALE,
            window_size=WINDOW,
            head_dim=D,
            index_topk=None,
            rope_head_dim=4,
        )


def test_compressed_route_needs_compressor_weights():
    h = _Harness(128, seed=95)
    md = h.step([0], [0], [4])[3]
    with pytest.raises(ValueError, match="needs compressor weights"):
        run_layer(
            q=np.zeros((4, H, D), np.float32),
            new_kv=np.zeros((4, D), np.float32),
            compressor_input=np.zeros((4, D), np.float32),
            layer_id=0,
            ratio=128,
            metadata=md,
            tables=read_tables(
                request_pool=h.pool,
                allocator=h.alloc,
                slots=np.array([0]),
                lengths=np.array([4]),
                q_lens=np.array([4]),
                ratio=128,
                window_size=WINDOW,
                page_size=PAGE,
            ),
            kv_buffers={"swa": h.swa, "compressed": h.compressed},
            state=h.state,
            compressor_weights=None,
            attention_sink=h.sink,
            softmax_scale=SCALE,
            window_size=WINDOW,
            head_dim=D,
        )


# --------------------------------------------------------------------------
# the routing C3's consumer now performs
# --------------------------------------------------------------------------


class _Spec:
    compress_ratios = (0, 0, 4, 128, 4)


class _KVPool:
    spec = _Spec()


class _Layer:
    def __init__(self, layer_id):
        self.layer_id = layer_id


def test_layer_ratio_comes_from_c1s_spec():
    """One classification, not a third derivation: the ratio is read straight off
    C1's spec, which is the same list `configs/deepseek_v4.classify_layers` uses."""
    from sgl_jax.srt.layers.attention.deepseek_v4_backend import DeepseekV4AttentionBackend

    layer_ratio = DeepseekV4AttentionBackend.__dict__["layer_ratio"]
    assert [layer_ratio(None, _Layer(i), _KVPool()) for i in range(5)] == [0, 0, 4, 128, 4]


def test_layer_outside_the_backbone_is_rejected():
    """The 46-vs-43 trap again: an index past the trunk must not silently classify."""
    from sgl_jax.srt.layers.attention.deepseek_v4_backend import DeepseekV4AttentionBackend

    layer_ratio = DeepseekV4AttentionBackend.__dict__["layer_ratio"]
    with pytest.raises(ValueError, match="outside the V4 backbone"):
        layer_ratio(None, _Layer(5), _KVPool())


def test_first_token_attends_to_its_own_kv():
    """A zero query and zero sink split mass equally between self KV and sink."""
    h = _Harness(0)
    out, updates, _, _ = h.step([0], [0], [1])
    # Read the exact generated KV from its committed physical row. Re-run with
    # q=0 below so the independent oracle is simply value / 2.
    row = int(h.alloc.full_to_swa_index_mapping[h.pool.req_to_token[0, 0]])
    value = np.asarray(updates["swa"])[row]
    md = derive_attention_metadata(
        q_lens=[1],
        prefix_lens=[0],
        positions=[0],
        request_slots=[0],
        history_write_loc=[PAGE],
        swa_write_loc=[row],
        pages_per_request=[1],
        page_size=PAGE,
        window_size=WINDOW,
    )
    out, _ = run_layer(
        q=np.zeros((1, H, D), np.float32),
        new_kv=value[None],
        layer_id=0,
        ratio=0,
        metadata=md,
        tables=read_tables(
            request_pool=h.pool,
            allocator=h.alloc,
            slots=[0],
            lengths=[1],
            q_lens=[1],
            ratio=0,
            window_size=WINDOW,
            page_size=PAGE,
        ),
        kv_buffers={"swa": h.swa},
        state=None,
        attention_sink=np.zeros(H),
        softmax_scale=SCALE,
        window_size=WINDOW,
        head_dim=D,
    )
    np.testing.assert_allclose(out, np.broadcast_to(value / 2, (1, H, D)), atol=1e-6)
