# cd python && USE_DEVICE_TYPE=cpu python -m pytest sgl_jax/test/mem_cache/test_swa_paged.py -v

import os

if os.environ.get("USE_DEVICE_TYPE") == "cpu":
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
    os.environ["JAX_PLATFORMS"] = "cpu"

import unittest

import jax
import numpy as np
from jax.sharding import Mesh

from sgl_jax.srt.mem_cache.allocator import SWATokenToKVPoolAllocator
from sgl_jax.srt.mem_cache.memory_pool import (
    MHATokenToKVPool,
    ReqToTokenPool,
    SWAKVPool,
)
from sgl_jax.srt.mem_cache.swa_radix_cache import SWARadixCache


class TestSWAPaged(unittest.TestCase):
    """Tests for SWA + page_size > 1 correctness.

    These tests verify page alignment invariants required when using
    PagedTokenToKVPoolAllocator with the SWA dual-pool architecture.
    """

    PAGE_SIZE = 4
    FULL_SIZE = 512  # 128 pages
    SWA_SIZE = 256  # 64 pages
    SLIDING_WINDOW = 16

    def setUp(self):
        self.devices = jax.devices()
        self.mesh = Mesh([self.devices[0]], axis_names=("tensor",))

        self.kv_pool = SWAKVPool(
            size=self.FULL_SIZE,
            size_swa=self.SWA_SIZE,
            page_size=self.PAGE_SIZE,
            swa_attention_layer_ids=[0],
            full_attention_layer_ids=[1],
            full_pool_class=MHATokenToKVPool,
            swa_pool_class=MHATokenToKVPool,
            dtype=jax.numpy.bfloat16,
            head_num=1,
            head_dim=1,
            mesh=self.mesh,
        )

        self.allocator = SWATokenToKVPoolAllocator(
            size=self.FULL_SIZE,
            size_swa=self.SWA_SIZE,
            kvcache=self.kv_pool,
            page_size=self.PAGE_SIZE,
        )

        self.req_pool = ReqToTokenPool(size=64, max_context_len=512)

        self.cache = SWARadixCache(
            req_to_token_pool=self.req_pool,
            token_to_kv_pool_allocator=self.allocator,
            sliding_window_size=self.SLIDING_WINDOW,
            page_size=self.PAGE_SIZE,
            disable=False,
        )

    def _alloc_paged(self, n: int) -> np.ndarray:
        """Allocate n tokens (must be page-aligned) from the SWA allocator."""
        assert n % self.PAGE_SIZE == 0, f"n={n} must be page-aligned"
        idx = self.allocator.alloc(n)
        self.assertIsNotNone(idx, f"Failed to allocate {n} tokens")
        self.assertEqual(len(idx), n)
        return idx

    # ------------------------------------------------------------------
    # Bug 1: alloc_extend does not rollback full allocation on SWA failure
    # ------------------------------------------------------------------
    def test_alloc_extend_rollback_on_swa_failure(self):
        """When SWA pool is exhausted but full pool has space, alloc_extend
        should return None AND rollback the full allocation so no pages leak."""

        # Exhaust the SWA pool by allocating almost everything
        # SWA pool has 256 tokens = 64 pages. Allocate 252 tokens = 63 pages.
        # Leave only 1 page (4 tokens) free in SWA.
        exhaust_size = (self.SWA_SIZE // self.PAGE_SIZE - 1) * self.PAGE_SIZE
        _exhaust = self.allocator.alloc(exhaust_size)
        self.assertIsNotNone(_exhaust)

        full_before = self.allocator.full_available_size()

        # Try to allocate more tokens than SWA has available.
        # We need 2 pages (8 tokens) but SWA only has 1 page (4 tokens).
        result = self.allocator.alloc_extend(
            prefix_lens=[0],
            seq_lens=[8],
            last_loc=[-1],
            extend_num_tokens=8,
        )

        # alloc_extend should fail
        self.assertIsNone(result)

        # Full pool should NOT have leaked pages
        full_after = self.allocator.full_available_size()
        self.assertEqual(
            full_before, full_after, f"Full pool leaked: before={full_before}, after={full_after}"
        )

    # ------------------------------------------------------------------
    # Bug 1b: alloc_decode does not rollback full allocation on SWA failure
    # ------------------------------------------------------------------
    def test_alloc_decode_rollback_on_swa_failure(self):
        """When SWA pool is exhausted, alloc_decode should rollback full allocation."""

        # First, allocate a sequence so we have valid last_loc
        initial = self.allocator.alloc_extend(
            prefix_lens=[0],
            seq_lens=[4],
            last_loc=[-1],
            extend_num_tokens=4,
        )
        self.assertIsNotNone(initial)

        # Exhaust the SWA pool
        remaining_swa = self.allocator.swa_available_size()
        remaining_pages = remaining_swa // self.PAGE_SIZE
        exhaust_needed = remaining_pages * self.PAGE_SIZE
        if exhaust_needed > 0:
            _exhaust = self.allocator.alloc(exhaust_needed)
            self.assertIsNotNone(_exhaust)

        full_before = self.allocator.full_available_size()
        swa_before = self.allocator.swa_available_size()
        self.assertEqual(swa_before, 0, "SWA pool should be exhausted")

        # Try alloc_decode -- the sequence needs a decode token at position 5
        # which crosses a page boundary, requiring a new SWA page
        result = self.allocator.alloc_decode(
            seq_lens=[5],
            last_loc=[int(initial[-1])],
        )

        # Should fail since SWA has no pages
        self.assertIsNone(result)

        # Full pool should not leak
        full_after = self.allocator.full_available_size()
        self.assertEqual(
            full_before, full_after, f"Full pool leaked: before={full_before}, after={full_after}"
        )

    # ------------------------------------------------------------------
    # Bug 2: free_swa with non-page-aligned indices releases entire pages
    # ------------------------------------------------------------------
    def test_free_swa_non_page_aligned_releases_whole_page(self):
        """Calling free_swa with fewer than page_size indices should be
        rejected because PagedTokenToKVPoolAllocator.free() releases
        entire pages, which would corrupt still-in-use tokens."""

        # Allocate 2 pages (8 tokens)
        indices = self._alloc_paged(8)

        # Free only the first 2 indices (not a complete page) -- should raise
        partial_free = indices[:2]
        with self.assertRaises(
            AssertionError, msg="free_swa should reject non-page-aligned indices"
        ):
            self.allocator.free_swa(partial_free)

    def test_free_swa_page_aligned_works(self):
        """free_swa with a complete page of indices should succeed."""

        # Allocate 2 pages (8 tokens)
        indices = self._alloc_paged(8)
        swa_before = self.allocator.swa_available_size()

        # Free the first complete page (4 indices)
        full_page_free = indices[: self.PAGE_SIZE]
        self.allocator.free_swa(full_page_free)

        swa_after = self.allocator.swa_available_size()
        freed = swa_after - swa_before
        self.assertEqual(freed, self.PAGE_SIZE)

    # ------------------------------------------------------------------
    # Bug 3: _insert_helper tombstone restore with non-page-aligned boundary
    # ------------------------------------------------------------------
    def test_insert_tombstone_restore_page_aligned(self):
        """When tombstone nodes are restored during re-insertion, the free
        boundary (first_diff_idx) must be page-aligned to avoid freeing
        partial pages in PagedTokenToKVPoolAllocator."""

        # Insert two sequences sharing a prefix to create internal nodes.
        # key_a: [100..115] (4 pages), key_b: [100..107, 200..207] (shared 2 pages + 2 different)
        # This creates: root -> [100..107] (internal) -> [108..115] (leaf A), [200..207] (leaf B)
        key_a = list(range(100, 116))
        key_b = list(range(100, 108)) + list(range(200, 208))

        val_a = self._alloc_paged(16)
        self.cache.insert(key_a, value=val_a, prev_prefix_len=0)

        val_b = self._alloc_paged(16)
        self.cache.insert(key_b, value=val_b, prev_prefix_len=0)

        total_before, swa_before = self.cache.total_size()
        # 8 (shared) + 8 (A suffix) + 8 (B suffix) = 24
        self.assertEqual(total_before, 24)

        # SWA-evict: tombstone the shared internal node [100..107] (8 tokens = 2 pages)
        # SWA eviction picks LRU non-leaf nodes first for tombstoning
        self.cache.evict(full_num_tokens=0, swa_num_tokens=8)

        total_after_evict, swa_after_evict = self.cache.total_size()
        # Full tokens should still be 24 (tombstone only frees SWA)
        self.assertEqual(total_after_evict, 24)
        self.assertLess(swa_after_evict, swa_before)

        # Now re-insert key_a with prev_prefix_len=6 (not page-aligned).
        # This creates first_diff_idx = max(0, 6 - 0) = 6 in _insert_helper,
        # which is NOT page-aligned. The tombstone restore path will call
        # free(node.value[6:]) passing non-page-aligned indices to PagedAllocator.
        val_c = self._alloc_paged(16)
        self.cache.insert(key_a, value=val_c, prev_prefix_len=6)

        # The cache should be in a consistent state
        total_final, swa_final = self.cache.total_size()
        self.assertGreater(total_final, 0)

        # Verify the key can still be matched
        match = self.cache.match_prefix(key_a)
        self.assertEqual(len(match.device_indices), 16)

    # ------------------------------------------------------------------
    # Sanity: page-aligned operations should work correctly end-to-end
    # ------------------------------------------------------------------
    def test_paged_swa_insert_match_evict_cycle(self):
        """End-to-end test: insert, match, SWA evict, re-match with page alignment."""

        # Insert 2 sequences sharing a prefix
        key_a = list(range(0, 16))  # 4 pages
        key_b = list(range(0, 8)) + list(range(200, 208))  # shares first 2 pages

        val_a = self._alloc_paged(16)
        self.cache.insert(key_a, value=val_a, prev_prefix_len=0)

        val_b = self._alloc_paged(16)
        self.cache.insert(key_b, value=val_b, prev_prefix_len=0)

        # Both should match fully
        match_a = self.cache.match_prefix(key_a)
        self.assertEqual(len(match_a.device_indices), 16)
        match_b = self.cache.match_prefix(key_b)
        self.assertEqual(len(match_b.device_indices), 16)

        # Shared prefix (first 8 tokens) should use the same indices
        np.testing.assert_array_equal(
            np.asarray(match_a.device_indices[:8]),
            np.asarray(match_b.device_indices[:8]),
        )

        # Evict SWA tokens -- this tombstones internal nodes
        self.cache.evict(full_num_tokens=0, swa_num_tokens=8)

        # Sanity check should pass
        self.cache.sanity_check()


if __name__ == "__main__":
    unittest.main()
