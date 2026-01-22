import unittest

import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.eplb import pack_expert_rows, scatter_expert_rows


class TestEplbWeightRebalanceHelpers(unittest.TestCase):
    def test_pack_and_scatter(self):
        weights = jnp.arange(6 * 2, dtype=jnp.float32).reshape(6, 2)
        send_idx = jnp.array([3, 1, 5], dtype=jnp.int32)
        packed = pack_expert_rows(weights, send_idx)
        self.assertTrue(np.array_equal(np.asarray(packed), np.asarray(weights)[[3, 1, 5]]))

        recv_dst = jnp.array([2, 0, 1], dtype=jnp.int32)
        scattered = scatter_expert_rows(
            recv_buffer=packed, recv_dst_local_indices=recv_dst, out_shape=weights.shape
        )
        scattered_np = np.asarray(scattered)
        self.assertTrue(np.array_equal(scattered_np[2], np.asarray(weights)[3]))
        self.assertTrue(np.array_equal(scattered_np[0], np.asarray(weights)[1]))
        self.assertTrue(np.array_equal(scattered_np[1], np.asarray(weights)[5]))


if __name__ == "__main__":
    unittest.main()
