"""L1 padding contract tests.

Verifies the host-side padding helpers and contracts (pad_to_bucket,
_pick_context_len, spec EXTEND batch padding). These do not load a model
and run in <1s on CPU/pod single chip.
"""

from __future__ import annotations

import numpy as np
import pytest

from sgl_jax.srt.utils.common_utils import pad_to_bucket

# ---------- L1.1: pad_to_bucket contract ----------


def test_pad_to_bucket_exact_match():
    padded, idx = pad_to_bucket(8, [1, 8, 16])
    assert (padded, idx) == (8, 1)


def test_pad_to_bucket_ceil_to_nearest():
    padded, idx = pad_to_bucket(5, [1, 8, 16])
    assert (padded, idx) == (8, 1)


def test_pad_to_bucket_smallest_bucket():
    padded, idx = pad_to_bucket(1, [1, 8, 16])
    assert (padded, idx) == (1, 0)


def test_pad_to_bucket_overflow_raises():
    with pytest.raises(ValueError, match="No bucket >="):
        pad_to_bucket(20, [1, 8, 16])


def test_pad_to_bucket_all_real_bs_in_1_to_16_hit_bucket():
    """Padding 契约: any real_bs in [1, 16] must land on a bucket from
    [1,4,8,16]."""
    buckets = [1, 4, 8, 16]
    for real_bs in range(1, 17):
        padded, idx = pad_to_bucket(real_bs, buckets)
        assert padded in buckets
        assert padded >= real_bs
        assert buckets[idx] == padded


def test_pad_to_bucket_all_tokens_in_1_to_4096_hit_bucket():
    buckets = [256, 1024, 2048, 4096]
    for n in [1, 100, 256, 257, 1023, 1024, 2000, 4096]:
        padded, idx = pad_to_bucket(n, buckets)
        assert padded in buckets
        assert padded >= n


# ---------- L1.2: _pick_context_len strict bucket ----------


class _StubDraftWorker:
    """Mimics EagleDraftWorker._pick_context_len."""

    def __init__(self, token_paddings):
        self.precompile_token_paddings = token_paddings

    # Import the real method under test.
    from sgl_jax.srt.speculative.eagle_draft_worker import EagleDraftWorker

    _pick_context_len = EagleDraftWorker._pick_context_len


def test_pick_context_len_in_bucket():
    w = _StubDraftWorker([256, 1024, 2048, 4096])
    assert w._pick_context_len(300) == 1024
    assert w._pick_context_len(4096) == 4096
    assert w._pick_context_len(1) == 256


def test_pick_context_len_overflow_raises():
    w = _StubDraftWorker([256, 1024, 2048, 4096])
    with pytest.raises(ValueError, match="No bucket >="):
        w._pick_context_len(9999)


def test_pick_context_len_empty_buckets_raises():
    w = _StubDraftWorker([])
    with pytest.raises(RuntimeError, match="precompile_token_paddings is empty"):
        w._pick_context_len(100)


# ---------- L1.3: spec EXTEND array padding helper ----------


def _call_pad():
    from sgl_jax.srt.managers.schedule_batch import ScheduleBatch

    return ScheduleBatch._pad_spec_extend_arrays


def test_spec_extend_pad_shapes_hit_buckets():
    pad_fn = _call_pad()
    real_bs = 7
    real_tokens = 182
    padded_bs = 8
    padded_tokens = 256
    padded_cache_loc = 1024

    result = pad_fn(
        padded_bs=padded_bs,
        padded_tokens=padded_tokens,
        padded_cache_loc=padded_cache_loc,
        input_ids_cpu=np.arange(real_tokens, dtype=np.int32),
        positions_cpu=np.arange(real_tokens, dtype=np.int32),
        mrope_positions_cpu=None,
        seq_lens_cpu=np.full(real_bs, 26, dtype=np.int32),
        req_pool_indices_cpu=np.arange(real_bs, dtype=np.int32),
        extend_prefix_lens=np.zeros(real_bs, dtype=np.int32),
        extend_seq_lens=np.full(real_bs, 26, dtype=np.int32),
        logits_indices=np.cumsum(np.full(real_bs, 26, dtype=np.int32)) - 1,
        cache_loc_flat=np.zeros(900, dtype=np.int32),
        out_cache_loc_cpu=np.zeros(real_tokens, dtype=np.int32),
    )
    (
        input_ids,
        positions,
        _mrope,
        seq_lens,
        req_pool,
        ext_prefix,
        ext_seq,
        logits_indices,
        cache_loc,
        out_cache_loc,
    ) = result

    assert input_ids.shape == (padded_tokens,)
    assert positions.shape == (padded_tokens,)
    assert seq_lens.shape == (padded_bs,)
    assert req_pool.shape == (padded_bs,)
    assert ext_prefix.shape == (padded_bs,)
    assert ext_seq.shape == (padded_bs,)
    assert logits_indices.shape == (padded_bs,)
    assert cache_loc.shape == (padded_cache_loc,)
    assert out_cache_loc.shape == (padded_tokens,)

    # Padding rows must produce zero attention contribution
    assert (seq_lens[real_bs:] == 0).all()
    assert (ext_seq[real_bs:] == 0).all()
    assert (ext_prefix[real_bs:] == 0).all()

    # Logits padding rows reuse last real index so gather stays legal
    last_real = logits_indices[real_bs - 1]
    assert (logits_indices[real_bs:] == last_real).all()


def test_spec_extend_pad_real_bs_equals_padded_is_noop():
    """When real_bs already matches the bucket, arrays pass through unchanged."""
    pad_fn = _call_pad()
    real_bs = 8
    real_tokens = 256
    input_ids = np.arange(real_tokens, dtype=np.int32)
    seq_lens = np.full(real_bs, 32, dtype=np.int32)

    result = pad_fn(
        padded_bs=real_bs,
        padded_tokens=real_tokens,
        padded_cache_loc=512,
        input_ids_cpu=input_ids,
        positions_cpu=np.arange(real_tokens, dtype=np.int32),
        mrope_positions_cpu=None,
        seq_lens_cpu=seq_lens,
        req_pool_indices_cpu=np.arange(real_bs, dtype=np.int32),
        extend_prefix_lens=np.zeros(real_bs, dtype=np.int32),
        extend_seq_lens=np.full(real_bs, 32, dtype=np.int32),
        logits_indices=np.cumsum(np.full(real_bs, 32, dtype=np.int32)) - 1,
        cache_loc_flat=np.zeros(512, dtype=np.int32),
        out_cache_loc_cpu=np.zeros(real_tokens, dtype=np.int32),
    )
    (out_input_ids, _, _, out_seq_lens, *_rest) = result
    np.testing.assert_array_equal(out_input_ids, input_ids)
    np.testing.assert_array_equal(out_seq_lens, seq_lens)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
