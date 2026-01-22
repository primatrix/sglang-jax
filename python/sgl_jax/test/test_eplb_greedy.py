import unittest

import numpy as np

from sgl_jax.srt.eplb import (
    choose_num_physical_experts,
    compute_rebalance_sources,
    rebalance_experts_greedy,
)


class TestEplbGreedy(unittest.TestCase):
    def test_choose_num_physical_experts_adjusts_for_divisibility(self):
        # 256 logical experts, ep_size=96 => need (256 + r) % 96 == 0.
        e_physical, r = choose_num_physical_experts(
            num_logical_experts=256,
            ep_size=96,
            requested_num_redundant_experts=128,
            max_num_redundant_experts=128,
        )
        self.assertEqual(e_physical, 256 + r)
        self.assertTrue(0 <= r <= 128)
        self.assertEqual(e_physical % 96, 0)

    def test_rebalance_greedy_validates_invariants(self):
        rng = np.random.default_rng(0)
        num_layers = 4
        e_logical = 256
        ep_size = 32
        r = 128

        # Make a very skewed distribution.
        weights = rng.exponential(scale=1.0, size=(num_layers, e_logical)).astype(np.float32)
        weights[:, 0] *= 1e4

        meta = rebalance_experts_greedy(
            tokens_per_logical_expert=weights,
            ep_size=ep_size,
            num_redundant_experts=r,
            max_num_redundant_experts=128,
            seed=123,
        )

        self.assertEqual(meta.ep_size, ep_size)
        self.assertEqual(meta.num_layers, num_layers)
        self.assertTrue(meta.num_physical_experts >= e_logical)
        self.assertTrue(meta.num_physical_experts - e_logical <= 128)
        self.assertEqual(meta.num_physical_experts % ep_size, 0)

        # validate() already checks deep invariants; re-check a couple here.
        p2l = meta.physical_to_logical_map
        self.assertEqual(p2l.shape[0], num_layers)
        self.assertTrue(np.all(p2l >= 0))
        self.assertTrue(np.all(p2l < e_logical))

        dispatch = meta.logical_to_rank_dispatch_physical_map
        self.assertEqual(dispatch.shape, (num_layers, e_logical, ep_size))
        self.assertTrue(np.all(dispatch >= 0))
        self.assertTrue(np.all(dispatch < meta.num_physical_experts))

    def test_rebalance_sources_matches_logical_id(self):
        rng = np.random.default_rng(1)
        num_layers = 3
        e_logical = 256
        ep_size = 64
        r = 128

        weights_a = rng.uniform(low=0.0, high=1.0, size=(num_layers, e_logical)).astype(np.float32)
        weights_b = rng.uniform(low=0.0, high=1.0, size=(num_layers, e_logical)).astype(np.float32)
        weights_a[:, 3] *= 1000.0
        weights_b[:, 7] *= 1000.0

        old_meta = rebalance_experts_greedy(
            tokens_per_logical_expert=weights_a,
            ep_size=ep_size,
            num_redundant_experts=r,
            max_num_redundant_experts=128,
            seed=0,
        )
        new_meta = rebalance_experts_greedy(
            tokens_per_logical_expert=weights_b,
            ep_size=ep_size,
            num_redundant_experts=r,
            max_num_redundant_experts=128,
            seed=1,
        )

        src_for_dst = compute_rebalance_sources(
            old_physical_to_logical_map=old_meta.physical_to_logical_map,
            new_physical_to_logical_map=new_meta.physical_to_logical_map,
            ep_size=ep_size,
        )
        self.assertEqual(src_for_dst.shape, new_meta.physical_to_logical_map.shape)

        for layer in range(num_layers):
            old_map = old_meta.physical_to_logical_map[layer]
            new_map = new_meta.physical_to_logical_map[layer]
            src = src_for_dst[layer]
            self.assertTrue(np.array_equal(old_map[src], new_map))


if __name__ == "__main__":
    unittest.main()
