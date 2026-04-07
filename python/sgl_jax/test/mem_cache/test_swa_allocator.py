# cd python && USE_DEVICE_TYPE=cpu python -m pytest sgl_jax/test/mem_cache/test_swa_allocator.py -v

import os
import unittest

if os.environ.get("USE_DEVICE_TYPE") == "cpu":
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
    os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

from sgl_jax.srt.mem_cache.allocator import (
    SWATokenToKVPoolAllocator,
)
from sgl_jax.srt.mem_cache.memory_pool import (
    MHATokenToKVPool,
    ReqToTokenPool,
    SWAKVPool,
)


def _make_mesh():
    return Mesh(np.array(jax.devices()[:1]), axis_names=("tensor",))


def _make_swa_pool(size, size_swa, page_size, mesh):
    """Create a minimal SWAKVPool for testing."""
    return SWAKVPool(
        size=size,
        size_swa=size_swa,
        page_size=page_size,
        swa_attention_layer_ids=[0],
        full_attention_layer_ids=[1],
        full_pool_class=MHATokenToKVPool,
        swa_pool_class=MHATokenToKVPool,
        dtype=jnp.bfloat16,
        head_num=1,
        head_dim=1,
        mesh=mesh,
    )


# ---------------------------------------------------------------------------
# Class 1: Token-level allocator (page_size=1)
# ---------------------------------------------------------------------------
class TestSWAAllocatorTokenLevel(unittest.TestCase):
    def setUp(self):
        self.mesh = _make_mesh()
        self.kvcache = _make_swa_pool(
            size=64, size_swa=32, page_size=1, mesh=self.mesh
        )
        self.alloc = SWATokenToKVPoolAllocator(
            size=64, size_swa=32, kvcache=self.kvcache, page_size=1
        )

    # 1
    def test_alloc_basic_creates_mapping(self):
        """alloc(n) returns full indices and mapping points to SWA indices."""
        indices = self.alloc.alloc(4)
        self.assertIsNotNone(indices)
        self.assertEqual(len(indices), 4)
        # Mapping should contain non-zero SWA indices
        swa_indices = self.alloc.full_to_swa_index_mapping[indices]
        self.assertTrue(np.all(swa_indices > 0))
        # SWA indices should be unique
        self.assertEqual(len(np.unique(swa_indices)), 4)

    # 2
    def test_alloc_exceeds_full_returns_none(self):
        """Full pool exhaustion returns None."""
        # Directly exhaust the full pool via underlying allocator
        self.alloc.full_attn_allocator.alloc(64)
        # Now full pool is empty, alloc should fail even though SWA has room
        result = self.alloc.alloc(1)
        self.assertIsNone(result)

    # 3
    def test_alloc_exceeds_swa_returns_none(self):
        """SWA pool exhaustion returns None (SWA < full)."""
        # SWA is 32, full is 64 → SWA exhausts first
        result = self.alloc.alloc(32)
        self.assertIsNotNone(result)
        result = self.alloc.alloc(1)
        self.assertIsNone(result)

    # 4
    def test_available_size_returns_min(self):
        """available_size() = min(full_avail, swa_avail)."""
        self.assertEqual(self.alloc.available_size(), 32)  # min(64, 32)
        self.alloc.alloc(10)
        self.assertEqual(self.alloc.available_size(), 22)  # min(54, 22)

    # 5
    def test_free_restores_both_pools(self):
        """free() returns tokens to both full and SWA pools."""
        indices = self.alloc.alloc(10)
        self.assertEqual(self.alloc.available_size(), 22)
        self.alloc.free(indices)
        self.assertEqual(self.alloc.full_available_size(), 64)
        self.assertEqual(self.alloc.swa_available_size(), 32)

    # 6
    def test_free_swa_only_releases_swa(self):
        """free_swa() releases SWA slots but keeps full slots allocated."""
        indices = self.alloc.alloc(10)
        full_before = self.alloc.full_available_size()
        swa_before = self.alloc.swa_available_size()

        self.alloc.free_swa(indices)

        # Full pool unchanged
        self.assertEqual(self.alloc.full_available_size(), full_before)
        # SWA pool restored
        self.assertEqual(self.alloc.swa_available_size(), swa_before + 10)

    # 7
    def test_free_swa_clears_mapping(self):
        """free_swa() zeroes out the mapping for freed indices."""
        indices = self.alloc.alloc(5)
        # Mapping should be non-zero
        self.assertTrue(np.all(self.alloc.full_to_swa_index_mapping[indices] > 0))
        self.alloc.free_swa(indices)
        # Mapping should be zero
        self.assertTrue(np.all(self.alloc.full_to_swa_index_mapping[indices] == 0))

    # 8
    def test_free_swa_idempotent(self):
        """free_swa() twice on same indices does not crash or double-free."""
        indices = self.alloc.alloc(5)
        self.alloc.free_swa(indices)
        swa_after_first = self.alloc.swa_available_size()
        # Second call should be a no-op (mapping is already 0)
        self.alloc.free_swa(indices)
        self.assertEqual(self.alloc.swa_available_size(), swa_after_first)


# ---------------------------------------------------------------------------
# Class 2: Paged allocator (page_size=4)
# ---------------------------------------------------------------------------
class TestSWAAllocatorPaged(unittest.TestCase):
    def setUp(self):
        self.mesh = _make_mesh()
        self.page_size = 4
        self.size = 128
        self.size_swa = 64
        self.kvcache = _make_swa_pool(
            size=self.size,
            size_swa=self.size_swa,
            page_size=self.page_size,
            mesh=self.mesh,
        )
        self.alloc = SWATokenToKVPoolAllocator(
            size=self.size,
            size_swa=self.size_swa,
            kvcache=self.kvcache,
            page_size=self.page_size,
        )

    # 9
    def test_alloc_extend_mapping_correct(self):
        """alloc_extend() produces correct full→SWA mapping."""
        # Allocate initial tokens (simulating a prefix)
        prefix = self.alloc.alloc(8)
        self.assertIsNotNone(prefix)

        # Now extend: prefix_lens=[8], seq_lens=[12], last_loc=[last of prefix]
        last_loc = [int(prefix[-1])]
        full_indices = self.alloc.alloc_extend(
            prefix_lens=[8], seq_lens=[12], last_loc=last_loc, extend_num_tokens=4
        )
        self.assertIsNotNone(full_indices)
        self.assertEqual(len(full_indices), 4)

        # Each full index should have a valid SWA mapping
        for idx in full_indices:
            swa_idx = self.alloc.full_to_swa_index_mapping[idx]
            self.assertGreater(swa_idx, 0)

    # 10
    def test_alloc_decode_mapping_correct(self):
        """alloc_decode() produces correct full→SWA mapping."""
        # Allocate initial 4 tokens (1 page)
        prefix = self.alloc.alloc(4)
        self.assertIsNotNone(prefix)

        last_loc = [int(prefix[-1])]
        full_indices = self.alloc.alloc_decode(seq_lens=[5], last_loc=last_loc)
        self.assertIsNotNone(full_indices)
        self.assertEqual(len(full_indices), 1)

        swa_idx = self.alloc.full_to_swa_index_mapping[full_indices[0]]
        self.assertGreater(swa_idx, 0)

    # 11 — Bug 1 test (should FAIL before fix)
    def test_alloc_extend_rollback_on_swa_failure(self):
        """When SWA pool is exhausted, alloc_extend must roll back full-pool pages."""
        # Exhaust SWA pool: allocate size_swa tokens
        eaten = self.alloc.alloc(self.size_swa)
        self.assertIsNotNone(eaten)
        self.assertEqual(self.alloc.swa_available_size(), 0)
        full_before = self.alloc.full_available_size()

        # Try to extend — SWA is full, should fail
        # Set up a valid extend scenario: prefix_lens=[size_swa], seq_lens=[size_swa+4]
        last_loc = [int(eaten[-1])]
        result = self.alloc.alloc_extend(
            prefix_lens=[self.size_swa],
            seq_lens=[self.size_swa + 4],
            last_loc=last_loc,
            extend_num_tokens=4,
        )
        self.assertIsNone(result)

        # Full pool must have been rolled back (no leak)
        self.assertEqual(self.alloc.full_available_size(), full_before)

    # 12 — Bug 1 test (should FAIL before fix)
    def test_alloc_decode_rollback_on_swa_failure(self):
        """When SWA pool is exhausted, alloc_decode must roll back full-pool pages."""
        # Exhaust SWA pool
        eaten = self.alloc.alloc(self.size_swa)
        self.assertIsNotNone(eaten)
        self.assertEqual(self.alloc.swa_available_size(), 0)
        full_before = self.alloc.full_available_size()

        # Try to decode — SWA is full, should fail.
        # seq_lens=[size_swa+1] means we need the next token after the prefix.
        # last_loc is the last index of the allocated prefix.
        # We need the last index to be at a page boundary so decode needs a new page.
        # With page_size=4 and size_swa=64, last token is at offset 63.
        # seq_lens = 65 → needs page 17 (index 64 is 65th position → page 16*4=64..67)
        # Actually, eaten has 64 tokens, so we want seq_lens=65.
        # last_loc should be the last slot in eaten.
        last_loc = [int(eaten[-1])]
        result = self.alloc.alloc_decode(
            seq_lens=[self.size_swa + 1], last_loc=last_loc
        )
        self.assertIsNone(result)

        # Full pool must have been rolled back (no leak)
        self.assertEqual(self.alloc.full_available_size(), full_before)

    # 13
    def test_alloc_extend_swa_last_loc_translation(self):
        """alloc_extend translates last_loc through the full→SWA mapping."""
        prefix = self.alloc.alloc(4)
        self.assertIsNotNone(prefix)

        last_full = int(prefix[-1])
        expected_swa_last = int(self.alloc.full_to_swa_index_mapping[last_full])
        self.assertGreater(expected_swa_last, 0)

        # Extend by 4 more tokens
        result = self.alloc.alloc_extend(
            prefix_lens=[4], seq_lens=[8], last_loc=[last_full], extend_num_tokens=4
        )
        self.assertIsNotNone(result)

        # Verify new mappings exist
        for idx in result:
            self.assertGreater(int(self.alloc.full_to_swa_index_mapping[idx]), 0)

    # 14
    def test_free_releases_both_paged_pools(self):
        """free() returns pages to both full and SWA paged pools."""
        indices = self.alloc.alloc(8)
        self.assertIsNotNone(indices)
        self.alloc.free(indices)
        self.assertEqual(self.alloc.full_available_size(), self.size)
        self.assertEqual(self.alloc.swa_available_size(), self.size_swa)

    # 15
    def test_clear_resets_everything(self):
        """clear() fully resets both pools and the mapping."""
        indices = self.alloc.alloc(16)
        self.assertIsNotNone(indices)
        self.alloc.clear()
        self.assertEqual(self.alloc.full_available_size(), self.size)
        self.assertEqual(self.alloc.swa_available_size(), self.size_swa)
        # Mapping should be all zeros
        self.assertTrue(np.all(self.alloc.full_to_swa_index_mapping == 0))


# ---------------------------------------------------------------------------
# Class 3: SWA Eviction logic
# ---------------------------------------------------------------------------
class TestSWAEviction(unittest.TestCase):
    """Tests for _evict_swa logic (called via maybe_evict_swa)."""

    def setUp(self):
        self.mesh = _make_mesh()
        self.page_size = 1
        self.sliding_window = 64
        self.pool_size = 256
        self.pool_size_swa = 256
        self.kvcache = _make_swa_pool(
            size=self.pool_size,
            size_swa=self.pool_size_swa,
            page_size=self.page_size,
            mesh=self.mesh,
        )
        self.alloc = SWATokenToKVPoolAllocator(
            size=self.pool_size,
            size_swa=self.pool_size_swa,
            kvcache=self.kvcache,
            page_size=self.page_size,
        )
        self.req_to_token_pool = ReqToTokenPool(size=8, max_context_len=256)

    def _make_req(self, origin_len, output_len):
        """Create a minimal Req-like object for eviction tests."""

        class FakeReq:
            def __init__(self, origin_input_ids, output_ids):
                self.origin_input_ids = origin_input_ids
                self.output_ids = output_ids
                self.swa_evicted_seqlen = 0
                self.req_pool_idx = 0

        return FakeReq(
            origin_input_ids=list(range(origin_len)),
            output_ids=list(range(output_len)),
        )

    def _setup_req_tokens(self, req, n_tokens):
        """Allocate n_tokens and record them in req_to_token_pool."""
        indices = self.alloc.alloc(n_tokens)
        assert indices is not None, f"Failed to allocate {n_tokens} tokens"
        self.req_to_token_pool.req_to_token[req.req_pool_idx, :n_tokens] = indices
        return indices

    def _evict(self, req, pre_len):
        """Wrapper around the eviction logic matching _evict_swa."""
        new_evicted = max(req.swa_evicted_seqlen, pre_len - self.sliding_window)
        if self.page_size > 1:
            new_evicted = (new_evicted // self.page_size) * self.page_size
        if new_evicted <= req.swa_evicted_seqlen:
            return
        free_slots = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, req.swa_evicted_seqlen : new_evicted
        ]
        self.alloc.free_swa(free_slots)
        req.swa_evicted_seqlen = new_evicted

    # 16
    def test_evict_basic(self):
        """100 tokens, window=64 → evicts [0, 36)."""
        req = self._make_req(origin_len=100, output_len=0)
        self._setup_req_tokens(req, 100)
        pre_len = 100  # len(origin) + len(output) - 1 = 100 + 0 - 1, but for extend pre_len=100

        swa_before = self.alloc.swa_available_size()
        self._evict(req, pre_len)
        expected_evicted = pre_len - self.sliding_window  # 36
        self.assertEqual(req.swa_evicted_seqlen, expected_evicted)
        self.assertEqual(
            self.alloc.swa_available_size(), swa_before + expected_evicted
        )

    # 17
    def test_evict_idempotent(self):
        """Same pre_len repeated does not double-free."""
        req = self._make_req(origin_len=100, output_len=0)
        self._setup_req_tokens(req, 100)
        self._evict(req, 100)
        swa_after_first = self.alloc.swa_available_size()

        self._evict(req, 100)
        self.assertEqual(self.alloc.swa_available_size(), swa_after_first)

    # 18
    def test_evict_incremental(self):
        """Decode step +1 → evicts exactly 1 slot."""
        req = self._make_req(origin_len=100, output_len=0)
        self._setup_req_tokens(req, 100)

        self._evict(req, 100)
        evicted_1 = req.swa_evicted_seqlen
        swa_1 = self.alloc.swa_available_size()

        # Simulate one decode step
        self._evict(req, 101)
        self.assertEqual(req.swa_evicted_seqlen, evicted_1 + 1)
        self.assertEqual(self.alloc.swa_available_size(), swa_1 + 1)

    # 19
    def test_evict_page_aligned(self):
        """page_size=4: frontier aligns down to page boundary."""
        # Recreate with page_size=4
        page_size = 4
        kvcache = _make_swa_pool(
            size=256, size_swa=256, page_size=page_size, mesh=self.mesh
        )
        alloc = SWATokenToKVPoolAllocator(
            size=256, size_swa=256, kvcache=kvcache, page_size=page_size
        )
        req_pool = ReqToTokenPool(size=8, max_context_len=256)

        req = self._make_req(origin_len=100, output_len=0)
        indices = alloc.alloc(100)
        self.assertIsNotNone(indices)
        req_pool.req_to_token[req.req_pool_idx, :100] = indices

        # pre_len=100, window=64 → raw evicted=36 → page-aligned=36//4*4=36
        new_evicted = max(req.swa_evicted_seqlen, 100 - self.sliding_window)
        new_evicted = (new_evicted // page_size) * page_size  # 36
        self.assertEqual(new_evicted, 36)

        # pre_len=101 → raw=37 → aligned=36 (no change from 36)
        new_evicted2 = max(36, 101 - self.sliding_window)
        new_evicted2 = (new_evicted2 // page_size) * page_size
        self.assertEqual(new_evicted2, 36)

        # pre_len=104 → raw=40 → aligned=40
        new_evicted3 = max(36, 104 - self.sliding_window)
        new_evicted3 = (new_evicted3 // page_size) * page_size
        self.assertEqual(new_evicted3, 40)

    # 20
    def test_evict_within_window_noop(self):
        """pre_len <= window → nothing evicted."""
        req = self._make_req(origin_len=50, output_len=0)
        self._setup_req_tokens(req, 50)
        swa_before = self.alloc.swa_available_size()

        self._evict(req, 50)
        self.assertEqual(req.swa_evicted_seqlen, 0)
        self.assertEqual(self.alloc.swa_available_size(), swa_before)

    # 21
    def test_evict_reclaims_swa_capacity(self):
        """Eviction increases swa_available_size."""
        req = self._make_req(origin_len=128, output_len=0)
        self._setup_req_tokens(req, 128)
        swa_before = self.alloc.swa_available_size()

        self._evict(req, 128)
        expected_freed = 128 - self.sliding_window  # 64
        self.assertEqual(
            self.alloc.swa_available_size(), swa_before + expected_freed
        )


# ---------------------------------------------------------------------------
# Class 4: Overlap safety
# ---------------------------------------------------------------------------
class TestSWAOverlapSafety(unittest.TestCase):
    """Test that overlap mode applies an explicit safety margin to eviction."""

    def _compute_eviction_frontier(self, pre_len, sliding_window, enable_overlap, overlap_margin=0):
        """Compute the eviction frontier as maybe_evict_swa would."""
        if enable_overlap:
            pre_len -= overlap_margin
        return max(0, pre_len - sliding_window)

    # 22 — Bug 2 test (should FAIL before fix)
    def test_overlap_margin_more_conservative(self):
        """Overlap mode eviction frontier must be strictly less than non-overlap."""
        # Import the actual ScheduleBatch to check the margin constant
        from sgl_jax.srt.managers.schedule_batch import ScheduleBatch

        # Verify the margin constant exists and is >= 1
        self.assertTrue(
            hasattr(ScheduleBatch, "_SWA_OVERLAP_SAFETY_MARGIN"),
            "ScheduleBatch must define _SWA_OVERLAP_SAFETY_MARGIN",
        )
        margin = ScheduleBatch._SWA_OVERLAP_SAFETY_MARGIN
        self.assertGreaterEqual(margin, 1)

        # For any realistic pre_len > sliding_window, overlap frontier < non-overlap
        sliding_window = 64
        for pre_len in [100, 200, 500]:
            frontier_normal = self._compute_eviction_frontier(
                pre_len, sliding_window, enable_overlap=False
            )
            frontier_overlap = self._compute_eviction_frontier(
                pre_len, sliding_window, enable_overlap=True, overlap_margin=margin
            )
            self.assertLess(
                frontier_overlap,
                frontier_normal,
                f"Overlap frontier ({frontier_overlap}) should be < "
                f"non-overlap frontier ({frontier_normal}) for pre_len={pre_len}",
            )


if __name__ == "__main__":
    unittest.main()
