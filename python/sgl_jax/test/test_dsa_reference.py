from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

PAGE_SIZE = 4
LATENT_DIM = 3
ROPE_DIM = 2
LATENT_ALIGNED = 128
ROPE_ALIGNED = 128
CACHE_WIDTH = LATENT_ALIGNED + ROPE_ALIGNED


def _empty_cache(pages=3, dtype=jnp.bfloat16):
    packing = 2 if dtype == jnp.bfloat16 else 1
    return jnp.zeros(
        (pages, PAGE_SIZE // packing, packing, CACHE_WIDTH),
        dtype=dtype,
    )


def _write(cache, latent, rope, slots):
    from sgl_jax.srt.kernels.dsa.reference import write_mla_kv_cache

    return write_mla_kv_cache(
        cache,
        new_c_kv=jnp.asarray(latent, dtype=cache.dtype),
        new_k_pe=jnp.asarray(rope, dtype=cache.dtype),
        write_slots=jnp.asarray(slots, dtype=jnp.int32),
        page_size=PAGE_SIZE,
        latent_dim=LATENT_DIM,
        rope_dim=ROPE_DIM,
    )


def _slot_rows(cache):
    cache = np.asarray(cache, dtype=np.float32).reshape(-1, CACHE_WIDTH)
    return cache[:, :LATENT_DIM], cache[:, LATENT_ALIGNED : LATENT_ALIGNED + ROPE_DIM]


def _naive_selected_attention(q_latent, q_rope, cache, slots, counts, sm_scale):
    cached_latent, cached_rope = _slot_rows(cache)
    q_latent = np.asarray(q_latent, dtype=np.float32)
    q_rope = np.asarray(q_rope, dtype=np.float32)
    slots = np.asarray(slots, dtype=np.int32)
    counts = np.asarray(counts, dtype=np.int32)
    output = np.zeros_like(q_latent, dtype=np.float32)
    for token, count in enumerate(counts):
        if count == 0:
            continue
        selected = slots[token, :count]
        scores = (
            np.einsum("hc,kc->hk", q_latent[token], cached_latent[selected])
            + np.einsum("hr,kr->hk", q_rope[token], cached_rope[selected])
        ) * np.float32(sm_scale)
        scores -= scores.max(axis=-1, keepdims=True)
        weights = np.exp(scores)
        weights /= weights.sum(axis=-1, keepdims=True)
        output[token] = weights @ cached_latent[selected]
    return output


def test_write_mla_kv_cache_uses_token_slots_across_page_boundary_and_drops_padding():
    latent = np.arange(12, dtype=np.float32).reshape(4, LATENT_DIM) + 1
    rope = np.arange(8, dtype=np.float32).reshape(4, ROPE_DIM) + 21

    cache = jax.jit(_write)(_empty_cache(), latent, rope, [0, 3, 4, -1])
    cached_latent, cached_rope = _slot_rows(cache)

    np.testing.assert_array_equal(cached_latent[[0, 3, 4]], latent[:3])
    np.testing.assert_array_equal(cached_rope[[0, 3, 4]], rope[:3])
    np.testing.assert_array_equal(cached_latent[1], np.zeros(LATENT_DIM))
    np.testing.assert_array_equal(cached_rope[1], np.zeros(ROPE_DIM))
    assert cache.dtype == jnp.bfloat16


def test_write_mla_kv_cache_zeroes_latent_gap_preserves_tail_and_drops_capacity_slot():
    sentinel = np.float32(7)
    cache = jnp.full_like(_empty_cache(), sentinel)
    capacity = cache.shape[0] * PAGE_SIZE
    valid_slot = 1
    latent = np.array([[1, 2, 3], [31, 32, 33]], dtype=np.float32)
    rope = np.array([[4, 5], [34, 35]], dtype=np.float32)

    updated = jax.jit(_write)(cache, latent, rope, [valid_slot, capacity])
    slot_rows = np.asarray(updated, dtype=np.float32).reshape(capacity, CACHE_WIDTH)

    np.testing.assert_array_equal(slot_rows[valid_slot, :LATENT_DIM], latent[0])
    np.testing.assert_array_equal(
        slot_rows[valid_slot, LATENT_DIM:LATENT_ALIGNED],
        np.zeros(LATENT_ALIGNED - LATENT_DIM, dtype=np.float32),
    )
    np.testing.assert_array_equal(
        slot_rows[valid_slot, LATENT_ALIGNED : LATENT_ALIGNED + ROPE_DIM],
        rope[0],
    )
    np.testing.assert_array_equal(
        slot_rows[valid_slot, LATENT_ALIGNED + ROPE_DIM :],
        np.full(CACHE_WIDTH - LATENT_ALIGNED - ROPE_DIM, sentinel, dtype=np.float32),
    )
    np.testing.assert_array_equal(
        np.delete(slot_rows, valid_slot, axis=0),
        np.full((capacity - 1, CACHE_WIDTH), sentinel, dtype=np.float32),
    )


def test_write_mla_kv_cache_packs_latent_and_rope_into_one_sharded_scatter(monkeypatch):
    from sgl_jax.srt.kernels.dsa.reference import write_mla_kv_cache

    expected_sharding = object()
    captured = []

    class FakeAt:
        def __init__(self, cache):
            self.cache = cache

        def __getitem__(self, index):
            captured.append({"index": index})
            return self

        def set(self, values, **kwargs):
            captured[-1]["values"] = values
            captured[-1]["kwargs"] = kwargs
            return self.cache

    class FakeCache:
        ndim = 4
        shape = (2, 2, 2, CACHE_WIDTH)
        dtype = jnp.bfloat16

        def __init__(self):
            self.at = FakeAt(self)

    monkeypatch.setattr(jax, "typeof", lambda _array: SimpleNamespace(sharding=expected_sharding))
    result = write_mla_kv_cache(
        FakeCache(),
        new_c_kv=jnp.ones((1, LATENT_DIM), dtype=jnp.bfloat16),
        new_k_pe=jnp.ones((1, ROPE_DIM), dtype=jnp.bfloat16),
        write_slots=jnp.array([0], dtype=jnp.int32),
        page_size=PAGE_SIZE,
        latent_dim=LATENT_DIM,
        rope_dim=ROPE_DIM,
    )

    assert isinstance(result, FakeCache)
    assert len(captured) == 1
    scatter = captured[0]
    assert scatter["kwargs"]["out_sharding"] is expected_sharding
    assert scatter["values"].shape == (1, LATENT_ALIGNED + ROPE_DIM)
    assert scatter["index"][-1] == slice(None, LATENT_ALIGNED + ROPE_DIM)


def test_sparse_mla_reference_uses_slot_sharding_for_cache_gather(monkeypatch):
    from sgl_jax.srt.kernels.dsa.reference import dsa_sparse_mla_reference

    expected_sharding = object()
    captured = {}

    class FakeAt:
        def __getitem__(self, index):
            captured["index"] = index
            return self

        def get(self, **kwargs):
            captured["kwargs"] = kwargs
            return jnp.ones((1, 2, CACHE_WIDTH), dtype=jnp.bfloat16)

    class FakeCache:
        ndim = 4
        shape = (1, 2, 2, CACHE_WIDTH)
        dtype = jnp.bfloat16
        at = FakeAt()

    physical_slots = jnp.array([[0, 1]], dtype=jnp.int32)
    original_typeof = jax.typeof

    def fake_typeof(array):
        if array is physical_slots:
            return SimpleNamespace(sharding=expected_sharding)
        return original_typeof(array)

    monkeypatch.setattr(jax, "typeof", fake_typeof)
    output = dsa_sparse_mla_reference(
        q_latent=jnp.ones((1, 1, LATENT_DIM), dtype=jnp.bfloat16),
        q_rope=jnp.ones((1, 1, ROPE_DIM), dtype=jnp.bfloat16),
        cache=FakeCache(),
        physical_slots=physical_slots,
        selected_counts=jnp.array([2], dtype=jnp.int32),
        sm_scale=1.0,
        page_size=PAGE_SIZE,
        latent_dim=LATENT_DIM,
        rope_dim=ROPE_DIM,
    )

    assert output.shape == (1, 1, LATENT_DIM)
    assert captured["kwargs"]["out_sharding"] is expected_sharding


def test_sparse_mla_reference_matches_dense_all_visible_and_is_slot_order_invariant():
    from sgl_jax.srt.kernels.dsa.reference import dsa_sparse_mla_reference
    from sgl_jax.srt.kernels.mla.v1.ref import ref_mla_ragged_paged_attention

    latent = np.array(
        [[1.0, 0.5, -0.5], [0.25, 2.0, 0.5], [-1.0, 0.5, 1.5], [0.5, -1.0, 2.0]],
        dtype=np.float32,
    )
    rope = np.array([[0.5, 1.0], [1.5, -0.5], [-1.0, 2.0], [0.25, 0.75]], dtype=np.float32)
    cache = _write(_empty_cache(dtype=jnp.float32), latent, rope, [0, 1, 2, 3])
    q_latent = jnp.asarray([[[0.5, 1.0, -0.25], [1.0, -0.5, 0.75]]], dtype=jnp.float32)
    q_rope = jnp.asarray([[[0.25, 1.0], [-0.75, 0.5]]], dtype=jnp.float32)
    ordered = jnp.array([[0, 1, 2, 3]], dtype=jnp.int32)
    permuted = jnp.array([[2, 0, 3, 1]], dtype=jnp.int32)
    counts = jnp.array([4], dtype=jnp.int32)

    output = dsa_sparse_mla_reference(
        q_latent,
        q_rope,
        cache,
        ordered,
        counts,
        sm_scale=0.75,
        page_size=PAGE_SIZE,
        latent_dim=LATENT_DIM,
        rope_dim=ROPE_DIM,
    )
    expected = _naive_selected_attention(q_latent, q_rope, cache, ordered, counts, 0.75)
    dense_output, _ = ref_mla_ragged_paged_attention(
        q_latent,
        q_rope,
        jnp.asarray(latent[-1:], dtype=jnp.float32),
        jnp.asarray(rope[-1:], dtype=jnp.float32),
        jnp.copy(cache),
        jnp.array([4], dtype=jnp.int32),
        jnp.array([0], dtype=jnp.int32),
        jnp.array([0, 1], dtype=jnp.int32),
        jnp.array([0, 0, 1], dtype=jnp.int32),
        sm_scale=0.75,
    )
    permuted_output = dsa_sparse_mla_reference(
        q_latent,
        q_rope,
        cache,
        permuted,
        counts,
        sm_scale=0.75,
        page_size=PAGE_SIZE,
        latent_dim=LATENT_DIM,
        rope_dim=ROPE_DIM,
    )

    np.testing.assert_allclose(output, expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        output,
        dense_output[..., :LATENT_DIM],
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(permuted_output, output, rtol=1e-6, atol=1e-6)
    assert output.dtype == jnp.float32


def test_sparse_mla_reference_handles_two_ragged_requests_and_zero_count_padding():
    from sgl_jax.srt.kernels.dsa.reference import dsa_sparse_mla_reference

    latent = np.arange(15, dtype=np.float32).reshape(5, LATENT_DIM) / 4
    rope = np.arange(10, dtype=np.float32).reshape(5, ROPE_DIM) / 3
    cache = _write(_empty_cache(), latent, rope, [0, 1, 4, 5, 6])
    q_latent = jnp.asarray(np.arange(18).reshape(3, 2, 3) / 7, dtype=jnp.bfloat16)
    q_rope = jnp.asarray(np.arange(12).reshape(3, 2, 2) / 5, dtype=jnp.bfloat16)
    slots = jnp.array(
        [[0, 1, -99, -99], [6, 4, 5, -99], [-123, -123, -123, -123]],
        dtype=jnp.int32,
    )
    counts = jnp.array([2, 3, 0], dtype=jnp.int32)

    output = jax.jit(
        lambda q, r, s, n: dsa_sparse_mla_reference(
            q,
            r,
            cache,
            s,
            n,
            sm_scale=0.5,
            page_size=PAGE_SIZE,
            latent_dim=LATENT_DIM,
            rope_dim=ROPE_DIM,
        )
    )(q_latent, q_rope, slots, counts)
    expected = _naive_selected_attention(q_latent, q_rope, cache, slots, counts, 0.5)

    np.testing.assert_allclose(output, expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(output[2], np.zeros((2, LATENT_DIM), dtype=np.float32))


def test_decode_after_prefill_reads_cache_written_in_both_steps():
    from sgl_jax.srt.kernels.dsa.reference import dsa_sparse_mla_reference

    prefill_latent = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]], dtype=np.float32)
    prefill_rope = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.float32)
    cache_after_prefill = _write(_empty_cache(), prefill_latent, prefill_rope, [0, 1, 4])
    cache_after_decode = _write(
        cache_after_prefill,
        [[4, 1, -1]],
        [[-1, 2]],
        [5],
    )
    q_latent = jnp.asarray([[[1.0, 0.5, 0.25]]], dtype=jnp.bfloat16)
    q_rope = jnp.asarray([[[0.5, 1.0]]], dtype=jnp.bfloat16)
    slots = jnp.array([[0, 1, 4, 5]], dtype=jnp.int32)
    counts = jnp.array([4], dtype=jnp.int32)

    output = dsa_sparse_mla_reference(
        q_latent,
        q_rope,
        cache_after_decode,
        slots,
        counts,
        sm_scale=1.0,
        page_size=PAGE_SIZE,
        latent_dim=LATENT_DIM,
        rope_dim=ROPE_DIM,
    )
    expected = _naive_selected_attention(q_latent, q_rope, cache_after_decode, slots, counts, 1.0)

    np.testing.assert_allclose(output, expected, rtol=1e-6, atol=1e-6)
    assert not np.allclose(output, np.zeros_like(output))


def test_sparse_mla_reference_rejects_shape_and_dtype_abi_violations():
    from sgl_jax.srt.kernels.dsa.reference import dsa_sparse_mla_reference

    kwargs = dict(
        q_latent=jnp.zeros((1, 1, LATENT_DIM), dtype=jnp.bfloat16),
        q_rope=jnp.zeros((1, 1, ROPE_DIM), dtype=jnp.bfloat16),
        cache=_empty_cache(),
        physical_slots=jnp.zeros((1, 2), dtype=jnp.int32),
        selected_counts=jnp.ones((1,), dtype=jnp.int32),
        sm_scale=1.0,
        page_size=PAGE_SIZE,
        latent_dim=LATENT_DIM,
        rope_dim=ROPE_DIM,
    )

    with pytest.raises(ValueError, match="matching token and head dimensions"):
        dsa_sparse_mla_reference(**{**kwargs, "q_rope": jnp.zeros((2, 1, ROPE_DIM))})
    with pytest.raises(TypeError, match="physical_slots must have dtype int32"):
        dsa_sparse_mla_reference(**{**kwargs, "physical_slots": jnp.zeros((1, 2))})
    with pytest.raises(TypeError, match="selected_counts must have dtype int32"):
        dsa_sparse_mla_reference(**{**kwargs, "selected_counts": jnp.ones((1,))})
    with pytest.raises(ValueError, match="counted physical_slots"):
        dsa_sparse_mla_reference(
            **{
                **kwargs,
                "physical_slots": jnp.array([[-1, 0]], dtype=jnp.int32),
            }
        )
    with pytest.raises(ValueError, match="selected_counts entries"):
        dsa_sparse_mla_reference(
            **{
                **kwargs,
                "selected_counts": jnp.array([3], dtype=jnp.int32),
            }
        )
