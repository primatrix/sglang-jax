"""Tests for SWATokenToKVPoolAllocator."""

import numpy as np
import pytest

from sgl_jax.srt.mem_cache.allocator import (
    SWATokenToKVPoolAllocator,
    TokenToKVPoolAllocator,
)


class FakeKVPool:
    """Minimal fake KV pool for testing."""

    def __init__(self, size):
        self.size = size


class FakeSWAKVPool:
    """Minimal fake SWA KV pool for testing."""

    def __init__(self, size_full, size_swa):
        self.full_kv_pool = FakeKVPool(size_full)
        self.swa_kv_pool = FakeKVPool(size_swa)
        self.full_to_swa_index_mapping = None


class TestSWAAllocatorTokenLevel:
    """Test SWATokenToKVPoolAllocator with page_size=1."""

    def setup_method(self):
        self.size = 64
        self.size_swa = 32
        self.kvcache = FakeSWAKVPool(self.size, self.size_swa)
        self.allocator = SWATokenToKVPoolAllocator(
            self.size, self.size_swa, self.kvcache, page_size=1
        )

    def test_alloc_basic_creates_mapping(self):
        indices = self.allocator.alloc(4)
        assert indices is not None
        assert len(indices) == 4
        # Mapping should be set for allocated indices
        swa_mapped = self.allocator.full_to_swa_index_mapping[indices]
        assert np.all(swa_mapped > 0)

    def test_alloc_exceeds_full_returns_none(self):
        result = self.allocator.alloc(self.size + 1)
        assert result is None

    def test_alloc_exceeds_swa_returns_none(self):
        result = self.allocator.alloc(self.size_swa + 1)
        assert result is None

    def test_available_size_returns_min(self):
        avail = self.allocator.available_size()
        assert avail == min(
            self.allocator.full_available_size(),
            self.allocator.swa_available_size(),
        )

    def test_free_restores_both_pools(self):
        indices = self.allocator.alloc(4)
        full_before = self.allocator.full_available_size()
        swa_before = self.allocator.swa_available_size()

        self.allocator.free(indices)

        assert self.allocator.full_available_size() == full_before + 4
        assert self.allocator.swa_available_size() == swa_before + 4

    def test_free_swa_only_releases_swa(self):
        indices = self.allocator.alloc(4)
        full_before = self.allocator.full_available_size()
        swa_before = self.allocator.swa_available_size()

        self.allocator.free_swa(indices)

        # Full pool unchanged, SWA pool freed
        assert self.allocator.full_available_size() == full_before
        assert self.allocator.swa_available_size() == swa_before + 4

    def test_free_swa_clears_mapping(self):
        indices = self.allocator.alloc(4)
        self.allocator.free_swa(indices)
        swa_mapped = self.allocator.full_to_swa_index_mapping[indices]
        assert np.all(swa_mapped == 0)

    def test_free_swa_idempotent(self):
        indices = self.allocator.alloc(4)
        self.allocator.free_swa(indices)
        swa_before = self.allocator.swa_available_size()
        # Second free_swa should be no-op
        self.allocator.free_swa(indices)
        assert self.allocator.swa_available_size() == swa_before

    def test_clear_resets_everything(self):
        self.allocator.alloc(10)
        self.allocator.clear()
        assert self.allocator.full_available_size() == self.size
        assert self.allocator.swa_available_size() == self.size_swa
        assert np.all(self.allocator.full_to_swa_index_mapping == 0)
