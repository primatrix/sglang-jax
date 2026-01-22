import unittest

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.eplb import dense_logits_from_topk, map_logical_to_physical_topk_ids


class TestEplbTopk(unittest.TestCase):
    def test_dense_logits_from_topk(self):
        topk_ids = jnp.array([[1, 3], [0, 2]], dtype=jnp.int32)
        topk_w = jnp.array([[0.2, 0.8], [1.0, 0.5]], dtype=jnp.float32)
        dense = dense_logits_from_topk(topk_ids=topk_ids, topk_weights=topk_w, num_experts=5)

        dense_np = np.asarray(dense)
        self.assertEqual(dense_np.shape, (2, 5))
        self.assertTrue(np.isneginf(dense_np[0, 0]))
        self.assertAlmostEqual(dense_np[0, 1], 0.2, places=6)
        self.assertAlmostEqual(dense_np[0, 3], 0.8, places=6)
        self.assertTrue(np.isneginf(dense_np[0, 2]))
        self.assertTrue(np.isneginf(dense_np[0, 4]))

    def test_map_logical_to_physical_topk_ids(self):
        # E_logical=4, ep_size=2. For rank0 choose ids [0,1,2,3] -> [5,6,7,8]
        dispatch = jnp.array(
            [
                [5, 10],
                [6, 11],
                [7, 12],
                [8, 13],
            ],
            dtype=jnp.int32,
        )
        topk_logical = jnp.array([[0, 3], [2, 1]], dtype=jnp.int32)
        topk_phys = map_logical_to_physical_topk_ids(
            topk_ids_logical=topk_logical,
            logical_to_rank_dispatch_physical_map_layer=dispatch,
            ep_rank=0,
        )
        self.assertTrue(
            np.array_equal(np.asarray(topk_phys), np.array([[5, 8], [7, 6]], dtype=np.int32))
        )

    def test_map_logical_to_physical_topk_ids_dynamic_rank(self):
        dispatch = jnp.array(
            [
                [5, 10],
                [6, 11],
                [7, 12],
                [8, 13],
            ],
            dtype=jnp.int32,
        )
        topk_logical = jnp.array([[0, 3], [2, 1]], dtype=jnp.int32)

        @jax.jit
        def f(ep_rank):
            return map_logical_to_physical_topk_ids(
                topk_ids_logical=topk_logical,
                logical_to_rank_dispatch_physical_map_layer=dispatch,
                ep_rank=ep_rank,
            )

        topk_phys = f(jnp.int32(1))
        self.assertTrue(
            np.array_equal(np.asarray(topk_phys), np.array([[10, 13], [12, 11]], dtype=np.int32))
        )


if __name__ == "__main__":
    unittest.main()
