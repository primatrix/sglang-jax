"""C4 real host allocation/release and scheduler lifecycle contracts."""

from types import SimpleNamespace
from unittest.mock import Mock

import jax
import numpy as np
import pytest

from sgl_jax.srt.managers.schedule_batch import Req, ScheduleBatch
from sgl_jax.srt.mem_cache.cache_init_params import CacheInitParams
from sgl_jax.srt.mem_cache.chunk_cache import DeepseekV4ChunkCache
from sgl_jax.srt.mem_cache.common import reclaim_completed_v4_swa, release_kv_cache
from sgl_jax.srt.mem_cache.deepseek_v4.allocator import DeepseekV4TokenToKVPoolAllocator
from sgl_jax.srt.mem_cache.deepseek_v4.pool import (
    DeepseekV4CacheSpec,
    DeepseekV4TokenToKVPool,
)
from sgl_jax.srt.mem_cache.memory_pool import ReqToTokenPool
from sgl_jax.srt.mem_cache.registry import TreeCacheBuildContext, create_tree_cache
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode
from sgl_jax.srt.sampling.sampling_params import SamplingParams


@pytest.fixture(autouse=True)
def isolated_mesh():
    with jax.set_mesh(None):
        yield


def cache(p=128, dp=1, swa_pages=12):
    if jax.device_count() < dp:
        pytest.skip("requires multiple devices")
    mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:dp]).reshape(dp, 1), ("data", "tensor"))
    pool = DeepseekV4TokenToKVPool(
        p * 24 * dp, p * swa_pages * dp, p, DeepseekV4CacheSpec((0, 4, 128), 8, 4), mesh, dp
    )
    allocator = DeepseekV4TokenToKVPoolAllocator(pool)
    return DeepseekV4ChunkCache(ReqToTokenPool(4, 4096), allocator, p, 128)


def request(rid="a", rank=0, prompt=700):
    return Req(
        rid=rid,
        origin_input_text="",
        origin_input_ids=[3] * prompt,
        sampling_params=SamplingParams(max_new_tokens=16, ignore_eos=True),
        dp_rank=rank,
        vocab_size=100,
    )


def extend(c, r, end):
    if r.req_pool_idx is None:
        assert c.req_to_token_pool.alloc([r]) is not None
    start = r.kv_committed_len
    old = c.req_to_token_pool.read(r.req_pool_idx, start)
    loc = c.token_to_kv_pool_allocator.alloc_extend(
        [start], [end], [old[-1] if start else -1], end - start, dp_rank=r.dp_rank
    )
    assert loc is not None
    c.req_to_token_pool.write((r.req_pool_idx, slice(start, end)), loc)
    r.kv_committed_len = r.kv_allocated_len = end
    r.fill_ids = (r.origin_input_ids + r.output_ids)[:end]
    return c.req_to_token_pool.read(r.req_pool_idx, end)


@pytest.mark.parametrize("p", [128, 256])
@pytest.mark.parametrize("rank", [0, 1])
def test_reclaim_only_completed_chunk_retains_history_and_reuses_swa(p, rank):
    c = cache(p, 2)
    a = c.token_to_kv_pool_allocator
    r = request(rank=rank)
    first = extend(c, r, 127)
    reclaim_completed_v4_swa(r, c)
    assert a.count_swa_mapped(first, rank) == 127
    loc = extend(c, r, 641)
    batch = object.__new__(ScheduleBatch)
    batch.tree_cache = c
    batch.forward_mode = ForwardMode.EXTEND
    # Prepared end=641 must not remove the early query's [0,127] window.
    batch.maybe_evict_swa()
    assert a.count_swa_mapped(loc, rank) == 641
    history_before = a.full_available_size(rank)
    reclaim_completed_v4_swa(r, c)
    assert r.swa_evicted_seqlen == 512
    assert a.count_swa_mapped(loc, rank) == 129
    assert a.full_available_size(rank) == history_before
    np.testing.assert_array_equal(c.req_to_token_pool.read(r.req_pool_idx, 641), loc)
    before = a.swa_available_size(rank)
    reclaim_completed_v4_swa(r, c)
    assert a.swa_available_size(rank) == before
    # A second request can consume released SWA while the first keeps history.
    other = request("b", rank)
    extend(c, other, 256)
    assert a.swa_available_size(rank) == before - 256
    release_kv_cache(r, c)
    release_kv_cache(other, c)
    assert a.full_available_size(rank) == 24 * p
    assert a.swa_available_size(rank) == 12 * p
    assert c.req_to_token_pool.available_size() == 4


@pytest.mark.parametrize("p", [128, 256])
def test_same_request_chunk_match_and_no_cross_request_reuse(p):
    c = cache(p)
    r = request()
    loc = extend(c, r, 129)
    c.cache_unfinished_req(r)
    r.init_next_round_input(c)
    np.testing.assert_array_equal(r.prefix_indices, loc)
    assert r.extend_input_len == 700 - 129
    other = request("b")
    other.init_next_round_input(c)
    assert len(other.prefix_indices) == 0
    assert other.extend_input_len == 700


@pytest.mark.parametrize("p", [128, 256])
@pytest.mark.parametrize("group", [False, True])
def test_release_full_extent_partial_tail_and_reentry_after_slot_reuse(p, group):
    c = cache(p)
    a = c.token_to_kv_pool_allocator
    r = request()
    extend(c, r, 257)
    slot = r.req_pool_idx
    r.kv_committed_len = 129  # allocated tail shares the committed page for P=256
    if group:
        a.free_group_begin()
    release_kv_cache(r, c)
    assert r.req_pool_idx is None
    assert not c.req_to_token_pool.req_to_token[slot].any()
    if group:
        a.free_group_end()
    # Force B onto A's exact global request slot and physical pages.
    c.req_to_token_pool.free_slots.remove(slot)
    c.req_to_token_pool.free_slots.insert(0, slot)
    other = request("b")
    loc = extend(c, other, 257)
    assert other.req_pool_idx == slot
    before = (a.full_available_size(), a.swa_available_size())
    release_kv_cache(r, c)
    assert before == (a.full_available_size(), a.swa_available_size())
    np.testing.assert_array_equal(c.req_to_token_pool.read(slot, 257), loc)
    release_kv_cache(other, c)
    assert a.full_available_size() == 24 * p and a.swa_available_size() == 12 * p


@pytest.mark.parametrize("end", [127, 128, 129, 255, 256, 257])
def test_retract_keeps_generated_history_and_stream_cursors(end):
    c = cache()
    r = request(prompt=end - 7)
    r.output_ids = list(range(10, 18))
    # Logical history fixture may extend through a later decode position.
    extend(c, r, end)
    r.send_token_offset = 5
    r.send_decode_id_offset = 7
    r.decoded_text = "already sent"
    r.surr_offset = 3
    r.read_offset = 9
    batch = object.__new__(ScheduleBatch)
    batch.tree_cache = c
    batch.reqs_info = [SimpleNamespace(reqs=[r])]
    batch._evict_tree_cache_if_needed = Mock()
    tokens = r.output_ids.copy()
    prompt = r.origin_input_ids.copy()
    batch.release_req(0, 0, 0, SimpleNamespace())
    assert r.is_retracted and r.req_pool_idx is None
    assert r.output_ids == tokens and r.origin_input_ids == prompt
    assert (
        r.send_token_offset,
        r.send_decode_id_offset,
        r.decoded_text,
        r.surr_offset,
        r.read_offset,
    ) == (5, 7, "already sent", 3, 9)
    r.init_next_round_input(c)
    assert r.fill_ids == prompt + tokens and len(r.prefix_indices) == 0
    assert r.extend_input_len == len(prompt) + len(tokens)
    assert c.req_to_token_pool.available_size() == 4


def test_factory_routes_v4_without_chunking_and_rejects_overlap():
    c = cache()
    ctx = TreeCacheBuildContext(
        server_args=SimpleNamespace(disable_overlap_schedule=True),
        params=CacheInitParams(
            c.req_to_token_pool, c.token_to_kv_pool_allocator, 128, sliding_window_size=128
        ),
        is_hybrid_swa=False,
        disable_radix_cache=True,
        effective_chunked_prefill_size=None,
        model_config=SimpleNamespace(),
        tp_size=1,
    )
    assert isinstance(create_tree_cache(ctx), DeepseekV4ChunkCache)
    ctx.server_args.disable_overlap_schedule = False
    with pytest.raises(ValueError, match="overlap"):
        create_tree_cache(ctx)


def scheduler(c, r):
    from sgl_jax.test.test_scheduler_chunked_ownership import (
        TestSchedulerChunkedOwnership,
    )

    helper = TestSchedulerChunkedOwnership()
    s, _ = helper._make_scheduler(r, active_reqs=r)
    s.tree_cache = c
    s.req_to_token_pool = c.req_to_token_pool
    s.token_to_kv_pool_allocator = c.token_to_kv_pool_allocator
    return s, helper


@pytest.mark.parametrize("kind", ["finish", "abort_running", "abort_chunk", "abort_parked"])
def test_scheduler_finish_and_abort_release_every_owner(kind):
    from sgl_jax.srt.managers.io_struct import AbortReq

    c = cache()
    r = request(prompt=641)
    extend(c, r, 641)
    s, helper = scheduler(c, r)
    if kind == "finish":
        r.sampling_params.max_new_tokens = 1
    if kind == "abort_chunk":
        r.is_chunked = 1
    if kind.startswith("abort"):
        s.abort_request(AbortReq(rid=r.rid))
    if kind == "abort_parked":
        s._process_pending_chunked_aborts()
    else:
        s.process_batch_result_prefill(s.last_batch, helper._make_prefill_result([9]))
    assert r.req_pool_idx is None
    assert r.finished()
    assert c.req_to_token_pool.available_size() == 4
    assert c.token_to_kv_pool_allocator.full_available_size() == 24 * 128
    assert c.token_to_kv_pool_allocator.swa_available_size() == 12 * 128
    release_kv_cache(r, c)


def test_retract_requeue_chunk_recompute_and_stream_sends_only_new_token():
    from sgl_jax.srt.managers.scheduler_output_processor_mixin import (
        SchedulerOutputProcessorMixin,
    )

    c = cache()
    r = request(prompt=127)
    r.output_ids = [10, 11, 12]
    r.stream = True
    extend(c, r, 129)
    s, helper = scheduler(c, r)
    s.skip_tokenizer_init = True
    SchedulerOutputProcessorMixin.stream_output(s, [r], False, False)
    s.send_to_detokenizer.send_pyobj.reset_mock()
    sent = (r.send_token_offset, r.send_decode_id_offset)
    s._retract_parked_chunked_reqs([])
    assert s.waiting_queue == [r] and r.is_retracted
    r.init_next_round_input(c)
    assert r.fill_ids == [3] * 127 + [10, 11, 12]
    # Rebuild two chunks; middle-chunk sampled output must never enter history.
    for end in (128, 130):
        extend(c, r, end)
        r.is_retracted = False
        r.is_chunked = int(end < 130)
        b = helper._make_batch([[r]], [r if r.is_chunked else None])
        s._pending_chunked_abort_reqs = [None]
        s.process_batch_result_prefill(b, helper._make_prefill_result([99 if end < 130 else 13]))
        if end < 130:
            assert r.output_ids == [10, 11, 12]
            assert sent == (r.send_token_offset, r.send_decode_id_offset)
            c.cache_unfinished_req(r)
            r.init_next_round_input(c)
            assert r.extend_input_len == 2
    assert r.output_ids == [10, 11, 12, 13]
    SchedulerOutputProcessorMixin.stream_output(s, [r], False, False)
    out = s.send_to_detokenizer.send_pyobj.call_args.args[0]
    assert out.output_ids == [[13]] and out.decode_ids == [[13]]
    release_kv_cache(r, c)


def test_idle_check_counts_both_pools_independently():
    c = cache(swa_pages=5)
    r = request()
    s, _ = scheduler(c, r)
    s.is_hybrid = True
    s.check_memory()
    extend(c, r, 129)
    with pytest.raises(ValueError, match="V4 history/SWA"):
        s.check_memory()
    release_kv_cache(r, c)
    s.check_memory()


@pytest.mark.parametrize("p", [128, 256])
def test_schedule_prepare_continuation_and_retract_from_zero(p):
    c = cache(p)
    r = request(prompt=385)
    r.init_next_round_input(c)
    r.fill_ids = r.fill_ids[:129]
    r.extend_input_len = 129

    def batch():
        return ScheduleBatch.init_new(
            reqs=[[r]],
            req_to_token_pool=c.req_to_token_pool,
            token_to_kv_pool_allocator=c.token_to_kv_pool_allocator,
            tree_cache=c,
            model_config=SimpleNamespace(vocab_size=100),
            enable_overlap=False,
            dp_size=1,
        )

    b = batch()
    b.prepare_for_extend()
    assert r.kv_committed_len == 129 and r.req_pool_idx == 0
    c.cache_unfinished_req(r)
    reclaim_completed_v4_swa(r, c)
    r.init_next_round_input(c)
    b = batch()
    b.prepare_for_extend()
    assert b.reqs_info[0].input_ids.tolist() == [3] * 256
    assert b.reqs_info[0].prefix_lens == [129]
    reclaim_completed_v4_swa(r, c)
    b.release_req(0, 0, 0, SimpleNamespace())
    r.init_next_round_input(c)
    b = batch()
    b.prepare_for_extend()
    assert b.reqs_info[0].prefix_lens == [0]
    assert b.reqs_info[0].input_ids.tolist() == [3] * 385
    release_kv_cache(r, c)
