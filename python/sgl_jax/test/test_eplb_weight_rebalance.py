import unittest

import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.eplb import apply_rebalance_mapping_global


class TestEplbWeightRebalance(unittest.TestCase):
    def test_apply_rebalance_mapping_global_gathers_rows(self):
        # weights[expert, ...]
        w = jnp.arange(5 * 3, dtype=jnp.float32).reshape(5, 3)
        # dst0 <- src3, dst1 <- src1, ...
        src_for_dst = jnp.array([3, 1, 4, 0, 2], dtype=jnp.int32)
        out = apply_rebalance_mapping_global(weights=w, src_for_dst_physical=src_for_dst)

        out_np = np.asarray(out)
        w_np = np.asarray(w)
        self.assertTrue(np.array_equal(out_np[0], w_np[3]))
        self.assertTrue(np.array_equal(out_np[1], w_np[1]))
        self.assertTrue(np.array_equal(out_np[2], w_np[4]))
        self.assertTrue(np.array_equal(out_np[3], w_np[0]))
        self.assertTrue(np.array_equal(out_np[4], w_np[2]))


if __name__ == "__main__":
    unittest.main()
