"""CPU-capable V4 resource tests; use two JAX devices to exercise DP isolation."""

import os
import unittest
from unittest.mock import patch

if os.environ.get("USE_DEVICE_TYPE") == "cpu":
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

from sgl_jax.srt.mem_cache.deepseek_v4.capacity import (
    build_deepseek_v4_pools,
    plan_deepseek_v4_pools,
)
from sgl_jax.srt.mem_cache.deepseek_v4.pool import (
    DeepseekV4CacheSpec,
    DeepseekV4TokenToKVPool,
)
from sgl_jax.srt.mem_cache.deepseek_v4.state import DeepseekV4CompressStatePool
from sgl_jax.srt.mem_cache.memory_pool import KVCache, MemoryPools


class TestDeepseekV4Pool(unittest.TestCase):
    def make_pools(self, ratios=(0, 4, 128, 4), page_size=128, dp=1):
        mesh = Mesh(np.array(jax.devices()[:dp]), ("data",))
        spec = DeepseekV4CacheSpec(ratios, head_dim=8, index_head_dim=4)
        kv = DeepseekV4TokenToKVPool(2 * page_size * dp, page_size * dp, page_size, spec, mesh, dp)
        state = DeepseekV4CompressStatePool(2, spec, mesh, dp)
        return kv, state

    def test_layout_capacity_and_pytree(self):
        for ratios in ((0,), (4,), (128,), (0, 4, 128, 4)):
            for page in (128, 256):
                for dp in range(1, min(2, len(jax.devices())) + 1):
                    with self.subTest(ratios=ratios, page=page, dp=dp):
                        kv, state = self.make_pools(ratios, page, dp)
                        self.assertIsInstance(kv, KVCache)
                        self.assertEqual((kv.start_layer, kv.end_layer), (0, len(ratios) - 1))
                        expected = (
                            2
                            * dp
                            * (
                                len(ratios) * 2 * page * 8
                                + ratios.count(4) * 3 * (page // 4) * (8 + 4)
                                + ratios.count(128) * 3 * (page // 128) * 8
                            )
                        )
                        self.assertEqual(kv.get_kv_size_bytes(), expected)
                        self.assertEqual(kv.mem_usage, expected / 2**30)
                        for arrays in kv.buffers.values():
                            for array in arrays:
                                self.assertEqual(array.dtype, jnp.bfloat16)
                                self.assertEqual(array.sharding.spec[0], "data")
                                self.assertTrue(
                                    all(axis is None for axis in array.sharding.spec[1:])
                                )
                        for owner in (kv, state):
                            leaves, treedef = jax.tree_util.tree_flatten(owner)
                            with patch(
                                "sgl_jax.srt.mem_cache.deepseek_v4.pool.allocate_buffer",
                                side_effect=AssertionError("unflatten must not allocate"),
                            ):
                                restored = jax.tree_util.tree_unflatten(treedef, leaves)
                            self.assertEqual(restored.layout, owner.layout)
                            self.assertEqual(restored.mem_usage, owner.mem_usage)
                            for a, b in zip(leaves, jax.tree_util.tree_leaves(restored)):
                                self.assertIs(a, b)
                        leaves, treedef = jax.tree_util.tree_flatten(kv)
                        restored_kv = jax.tree_util.tree_unflatten(treedef, leaves)
                        self.assertEqual(restored_kv.start_layer, 0)
                        self.assertEqual(restored_kv.end_layer, len(ratios) - 1)

    def test_accessors_and_unsupported_interfaces(self):
        kv, _ = self.make_pools()
        self.assertIs(kv.get_swa_buffer(0), kv.buffers["swa"][0])
        self.assertIs(kv.get_compressed_buffer(1), kv.buffers["c4"][0])
        self.assertIs(kv.get_compressed_buffer(3), kv.buffers["c4"][1])
        self.assertIs(kv.get_compressed_buffer(2), kv.buffers["c128"][0])
        self.assertIs(kv.get_indexer_buffer(3), kv.buffers["indexer"][1])
        self.assertEqual(kv.get_compressed_page_size(1), 32)
        self.assertEqual(kv.get_compressed_page_size(2), 1)
        for method in (kv.get_swa_buffer, kv.get_compressed_buffer, kv.get_indexer_buffer):
            for layer in (-1, 4):
                with self.assertRaises(IndexError):
                    method(layer)
        with self.assertRaises(ValueError):
            kv.get_compressed_buffer(0)
        with self.assertRaises(ValueError):
            kv.get_indexer_buffer(2)
        for method, args in (
            (kv.get_fused_kv_buffer, (0,)),
            (kv.get_kv_buffer, (0,)),
            (kv.set_kv_buffer, (0, None, None, None, False)),
            (kv.get_cpu_copy, (None,)),
            (kv.load_cpu_copy, (None, None)),
        ):
            with self.assertRaises(NotImplementedError):
                method(*args)

    def test_update_routing_validation_and_atomicity(self):
        kv, state = self.make_pools()
        original = kv.buffers
        c4 = jnp.ones_like(kv.get_compressed_buffer(3))
        updates = kv.build_buffer_updates({3: {"compressed": c4}})
        self.assertIs(kv.buffers, original)
        self.assertIs(updates["c4"][1], c4)
        self.assertIs(updates["c4"][0], original["c4"][0])
        for bad in (
            {3: {"compressed": c4}, 0: {"compressed": c4}},
            {0: {"compressed": c4}},
            {2: {"indexer": c4}},
            {1: {"unknown": c4}},
            {1: {"compressed": c4.astype(jnp.float32)}},
            {1: {"compressed": c4[:1]}},
        ):
            with self.assertRaises(ValueError):
                kv.build_buffer_updates(bad)
            self.assertIs(kv.buffers, original)
        for bad in ({}, {**updates, "extra": ()}, {**updates, "c4": ()}):
            with self.assertRaises(ValueError):
                kv.replace_buffer(bad)
            self.assertIs(kv.buffers, original)
        kv.replace_buffer(updates)
        self.assertIs(kv.get_compressed_buffer(3), c4)
        state_original = state.buffers
        new_state = jnp.ones_like(state.get_buffer("c128", 2))
        state_updates = state.build_buffer_updates({2: {"compressor": new_state}})
        self.assertIs(state.buffers, state_original)
        self.assertIs(state_updates["c128"][0], new_state)
        with self.assertRaises(ValueError):
            state.build_buffer_updates({0: {"compressor": new_state}})
        state.replace_buffer(state_updates)
        state.reset(jnp.array([0]), jnp.array([True]))
        reset = np.asarray(state.get_buffer("c128", 2))
        np.testing.assert_array_equal(reset[0, :, :8], 0)
        self.assertTrue(np.isneginf(reset[0, :, 8:]).all())
        np.testing.assert_array_equal(reset[1:], 1)

    def test_reference_write_address_units_and_dp(self):
        dp = min(2, len(jax.devices()))
        kv, state = self.make_pools(dp=dp)
        for family, layer, first in (
            ("swa", 1, 128),
            ("c4", 1, 32),
            ("c128", 2, 1),
            ("indexer", 1, 32),
        ):
            buf = kv.get_buffer(family, layer)
            rows_per_rank = buf.size // buf.shape[-1] // dp
            loc = jnp.array([0, first, first + 1, rows_per_rank, -1])
            values = jnp.full((5, buf.shape[-1]), 7, dtype=buf.dtype)
            kv.write(family, layer, loc, values, jnp.array([True, True, False, True, True]), dp - 1)
            actual = np.asarray(kv.get_buffer(family, layer)).reshape(-1, buf.shape[-1])
            expected = np.zeros_like(actual)
            expected[(dp - 1) * rows_per_rank + first] = 7
            np.testing.assert_array_equal(actual, expected)
        for arrays in state.buffers.values():
            for array in arrays:
                half = array.shape[-1] // 2
                np.testing.assert_array_equal(np.asarray(array)[..., :half], 0)
                self.assertTrue(np.isneginf(np.asarray(array)[..., half:]).all())

    def test_jit_donation_and_complete_commit(self):
        kv, state = self.make_pools()
        pools = MemoryPools(token_to_kv_pool=kv, compressor_state_pool=state)
        traces = []

        @jax.jit(donate_argnums=(0,))
        def step(owners):
            traces.append(1)
            kv_owner = owners.token_to_kv_pool
            state_owner = owners.compressor_state_pool
            return {
                "token_to_kv_pool": kv_owner.build_buffer_updates(
                    {
                        1: {
                            "swa": kv_owner.get_swa_buffer(1) + 1,
                            "compressed": kv_owner.get_compressed_buffer(1) + 2,
                            "indexer": kv_owner.get_indexer_buffer(1) + 3,
                        }
                    }
                ),
                "compressor_state_pool": state_owner.build_buffer_updates(
                    {1: {"compressor": state_owner.get_buffer("c4", 1) + 4}}
                ),
            }

        for i in range(1, 3):
            pools.replace_all(step(pools))
            np.testing.assert_array_equal(np.asarray(kv.get_swa_buffer(1)), i)
            np.testing.assert_array_equal(np.asarray(kv.get_compressed_buffer(1)), 2 * i)
            np.testing.assert_array_equal(np.asarray(kv.get_indexer_buffer(1)), 3 * i)
            np.testing.assert_array_equal(np.asarray(state.get_buffer("c4", 1))[..., :16], 4 * i)
        self.assertEqual(len(traces), 1)

    def test_capacity_builder(self):
        kv, _ = self.make_pools()
        budget = plan_deepseek_v4_pools(kv.spec, 1 << 20, 2, 128)
        req, pools, allocator = build_deepseek_v4_pools(kv.spec, budget, 128, kv.mesh, 512)
        self.assertIsInstance(pools.token_to_kv_pool, KVCache)
        self.assertIs(allocator.get_kvcache(), pools.token_to_kv_pool)
        self.assertEqual(req.size, 2)
        self.assertEqual(
            pools.token_to_kv_pool.nbytes + pools.compressor_state_pool.nbytes,
            budget.allocated_bytes_per_device,
        )


if __name__ == "__main__":
    unittest.main()
