import unittest

import numpy as np

from sgl_jax.srt.eplb import EplbController


class TestEplbRuntime(unittest.TestCase):
    def test_controller_rebalances_after_window_filled(self):
        controller = EplbController(
            num_layers=2,
            num_logical_experts=8,
            num_experts_per_tok=2,
            ep_size=4,
            window_size=2,
            update_interval=1,
            num_redundant_experts=0,
            max_num_redundant_experts=128,
            seed=0,
        )

        # step 1: only layer 0 has routing info
        topk0 = np.array([[0, 1], [0, 1], [0, 2]], dtype=np.int32)
        controller.record_step(topk_ids_by_layer=[topk0, None])
        self.assertIsNone(controller.maybe_rebalance())

        # step 2: both layers have routing info
        topk1 = np.array([[3, 4], [3, 4], [3, 5]], dtype=np.int32)
        controller.record_step(topk_ids_by_layer=[topk0, topk1])
        update = controller.maybe_rebalance()
        self.assertIsNotNone(update)

        meta = update.metadata
        self.assertEqual(meta.num_layers, 2)
        self.assertEqual(meta.ep_size, 4)
        self.assertEqual(meta.num_logical_experts, 8)
        self.assertEqual(meta.num_physical_experts % 4, 0)
        self.assertEqual(meta.physical_to_logical_map.shape[0], 2)
        self.assertEqual(meta.logical_to_rank_dispatch_physical_map.shape, (2, 8, 4))


if __name__ == "__main__":
    unittest.main()
