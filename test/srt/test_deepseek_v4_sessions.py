"""Host allocator/ownership tests. No TPU or model weights are needed."""

from types import SimpleNamespace

import numpy as np
import pytest

from sgl_jax.srt.mem_cache.base_prefix_cache import MatchPrefixParams
from sgl_jax.srt.mem_cache.deepseek_v4.allocator import DeepseekV4TokenToKVPoolAllocator
from sgl_jax.srt.mem_cache.deepseek_v4.session_cache import DeepseekV4SessionCache
from sgl_jax.srt.mem_cache.memory_pool import ReqToTokenPool


def request(tokens, output=()):
    req = SimpleNamespace(
        origin_input_ids=list(tokens),
        output_ids=list(output),
        req_pool_idx=None,
        dp_rank=0,
        kv_committed_len=0,
        kv_allocated_len=0,
        kv_committed_freed=False,
        kv_overallocated_freed=False,
        swa_evicted_seqlen=0,
        session_id=None,
        session_restored=False,
        is_chunked=0,
        prefix_indices=np.empty(0, np.int32),
        extra_key=None,
        lora_id=None,
        return_logprob=False,
        return_hidden_states=False,
        mm_inputs=None,
        finished_reason=None,
    )
    req.finished = lambda: req.finished_reason is not None
    return req


@pytest.fixture
def cache():
    pool = ReqToTokenPool(4, 128)
    allocator = DeepseekV4TokenToKVPoolAllocator(
        SimpleNamespace(size=128, size_swa=128, page_size=4, dp_size=1)
    )
    return DeepseekV4SessionCache(pool, allocator, 4, 8)


def allocate(cache, req, length):
    pool, allocator = cache.req_to_token_pool, cache.token_to_kv_pool_allocator
    pool.alloc([req])
    prefix = req.kv_committed_len
    last = pool.read(req.req_pool_idx, prefix)[-1] if prefix else -1
    slots = allocator.alloc_extend(
        np.array([prefix]), np.array([length]), np.array([last]), length - prefix
    )
    assert slots is not None
    pool.req_to_token[req.req_pool_idx, prefix:length] = slots
    req.kv_committed_len = req.kv_allocated_len = length


def finish(cache, req, reason="length"):
    req.finished_reason = SimpleNamespace(to_json=lambda: {"type": reason})
    cache.release_req(req)


def match(cache, req):
    return cache.match_prefix(MatchPrefixParams(key=None, req=req)).device_indices


def assert_empty(cache):
    assert cache.req_to_token_pool.available_size() == 4
    assert cache.token_to_kv_pool_allocator.full_available_size() == 128
    assert cache.token_to_kv_pool_allocator.swa_available_size() == 128


@pytest.mark.parametrize("overlap", [False, True])
def test_append_preserves_slot_and_allocation_extent(cache, overlap):
    first = request([1, 2, 3, 4], [5])
    cache.attach(first, {"id": "a"})
    allocate(cache, first, 5 if overlap else 4)
    slot = first.req_pool_idx
    original = cache.req_to_token_pool.read(slot, first.kv_committed_len)
    finish(cache, first)
    second = request([1, 2, 3, 4, 5, 6, 7])
    cache.attach(second, {"id": "a"})
    np.testing.assert_array_equal(match(cache, second), original)
    assert second.req_pool_idx == slot and first.req_pool_idx is None
    np.testing.assert_array_equal(match(cache, second), original)  # admission retry
    allocate(cache, second, 7)
    finish(cache, second)
    cache.sessions.close("a")
    assert_empty(cache)


@pytest.mark.parametrize("tokens", [[1, 9, 3, 4, 5], [1, 2], [1, 2, 3, 4]])
def test_mismatch_shorter_and_no_query_are_cold(cache, tokens):
    first = request([1, 2, 3, 4], [5])
    cache.attach(first, {"id": "a"})
    allocate(cache, first, 4)
    finish(cache, first)
    second = request(tokens)
    cache.attach(second, {"id": "a"})
    assert len(match(cache, second)) == 0
    assert_empty(cache)
    cache.cancel(second)


def test_uncached_final_output_is_not_required_for_reuse(cache):
    first = request([1, 2, 3, 4], [5])
    cache.attach(first, {"id": "a"})
    allocate(cache, first, 4)
    finish(cache, first)
    second = request([1, 2, 3, 4, 99])
    cache.attach(second, {"id": "a"})
    assert len(match(cache, second)) == 4
    cache.cancel(second)
    assert_empty(cache)


def test_abort_and_deferred_close_release_resources(cache):
    for reason in ("abort", "length"):
        req = request([1, 2, 3, 4])
        cache.attach(req, {"id": "a"})
        allocate(cache, req, 4)
        if reason == "length":
            cache.sessions.close("a")
        finish(cache, req, reason)
        assert_empty(cache)


def test_retraction_keeps_lease_but_not_kv(cache):
    req = request([1, 2, 3, 4])
    cache.attach(req, {"id": "a"})
    allocate(cache, req, 4)
    cache.release_req(req)  # not finished: retraction
    assert_empty(cache)
    with pytest.raises(ValueError, match="in-flight"):
        cache.attach(request([1, 2, 3, 4, 5]), {"id": "a"})
    cache.cancel(req)


def test_multiple_restored_sessions_share_a_batch(cache):
    restored = []
    for sid in ("a", "b"):
        first = request([1, 2, 3, 4])
        cache.attach(first, {"id": sid})
        allocate(cache, first, 4)
        finish(cache, first)
        second = request([1, 2, 3, 4, 5])
        cache.attach(second, {"id": sid})
        assert len(match(cache, second)) == 4
        restored.append(second)
    slots = cache.req_to_token_pool.alloc(restored)
    assert len(set(slots)) == 2
    for req in restored:
        cache.cancel(req)
    assert_empty(cache)


def test_swa_accounting_and_eviction(cache):
    req = request(list(range(12)))
    cache.attach(req, {"id": "a"})
    allocate(cache, req, 12)
    indices = cache.req_to_token_pool.read(req.req_pool_idx, 12)
    cache.token_to_kv_pool_allocator.free_swa(indices[:4])
    req.swa_evicted_seqlen = 4
    finish(cache, req)
    assert cache.held_sizes() == (12, 8, 1)
    cache.reserve_headroom(128)
    assert_empty(cache)


def test_unknown_allocated_tail_is_not_retained(cache):
    req = request([1, 2, 3, 4], [5])
    cache.attach(req, {"id": "a"})
    allocate(cache, req, 6)
    finish(cache, req)
    assert_empty(cache)


def test_completion_is_idempotent(cache):
    req = request([1, 2, 3, 4])
    cache.attach(req, {"id": "a"})
    allocate(cache, req, 4)
    finish(cache, req)
    cache.release_req(req)
    assert cache.held_sizes() == (4, 4, 1)
    cache.sessions.close("a")
    cache.release_req(req)
    assert_empty(cache)


def test_plain_requests_still_release_immediately(cache):
    req = request([1, 2, 3, 4])
    allocate(cache, req, 4)
    finish(cache, req)
    assert not cache.sessions.sessions
    assert_empty(cache)


def test_cache_namespace_change_is_a_miss(cache):
    first = request([1, 2, 3, 4])
    first.extra_key = "tenant-a"
    cache.attach(first, {"id": "a"})
    allocate(cache, first, 4)
    finish(cache, first)
    second = request([1, 2, 3, 4, 5])
    second.extra_key = "tenant-b"
    cache.attach(second, {"id": "a"})
    assert len(match(cache, second)) == 0
    cache.cancel(second)
    assert_empty(cache)


def test_continuation_state_survives_other_request_and_transfer(cache):
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh

    from sgl_jax.srt.mem_cache.deepseek_v4.pool import DeepseekV4CacheSpec
    from sgl_jax.srt.mem_cache.deepseek_v4.state import DeepseekV4CompressStatePool

    mesh = Mesh(np.array(jax.devices()[:1]), ("data",))
    states = DeepseekV4CompressStatePool(4, DeepseekV4CacheSpec((4, 128), 2, 2), mesh)
    first = request([1, 2, 3, 4])
    cache.attach(first, {"id": "a"})
    allocate(cache, first, 4)
    slot = first.req_pool_idx
    for family, (layers, shape) in states.layout.items():
        for layer in layers:
            states.write(
                family,
                layer,
                jnp.array([slot]),
                jnp.full((1, *shape[1:]), 7.0),
                jnp.array([True]),
            )
    finish(cache, first)
    other = request([9, 9, 9, 9])
    allocate(cache, other, 4)
    states.reset(jnp.array([other.req_pool_idx, slot]), jnp.array([True, False]))
    finish(cache, other)
    second = request([1, 2, 3, 4, 5])
    cache.attach(second, {"id": "a"})
    assert len(match(cache, second)) == 4
    for family, (layers, _) in states.layout.items():
        for layer in layers:
            np.testing.assert_array_equal(states.get_buffer(family, layer)[second.req_pool_idx], 7)
    cache.cancel(second)
    assert_empty(cache)


@pytest.mark.parametrize(
    "override",
    [
        {"dp_size": 2},
        {"pd_disaggregation": "prefill"},
        {"disaggregation_mode": "decode"},
        {"has_speculative": True},
        {"disable_radix_cache": False},
    ],
)
def test_factory_rejects_unsupported_combinations(cache, override):
    from sgl_jax.srt.mem_cache.cache_init_params import CacheInitParams
    from sgl_jax.srt.mem_cache.registry import (
        TreeCacheBuildContext,
        default_radix_cache_factory,
    )

    args = SimpleNamespace(
        enable_streaming_session=True,
        dp_size=1,
        pd_disaggregation="",
        disaggregation_mode="null",
        enable_unified_radix_tree=False,
    )
    ctx = TreeCacheBuildContext(
        args,
        CacheInitParams(
            cache.req_to_token_pool,
            cache.token_to_kv_pool_allocator,
            4,
            sliding_window_size=8,
        ),
        True,
        True,
        16,
        SimpleNamespace(),
        1,
    )
    for key, value in override.items():
        setattr(
            ctx if key in ("has_speculative", "disable_radix_cache") else args,
            key,
            value,
        )
    with pytest.raises(ValueError, match="Streaming sessions require"):
        default_radix_cache_factory(ctx)
