import unittest

import numpy as np

from sgl_jax.srt.eplb import build_rebalance_all_to_all_plan


class TestEplbAllToAllPlan(unittest.TestCase):
    def test_plan_sizes_are_consistent(self):
        ep_size = 4
        num_physical = 16
        local_e = num_physical // ep_size

        # Build a mapping that sends everything to the same physical id (not valid for real rebalance,
        # but still tests sizing). We'll instead make a valid permutation-like mapping.
        src_for_dst = np.arange(num_physical, dtype=np.int32)
        src_for_dst = np.roll(src_for_dst, 3)

        plan = build_rebalance_all_to_all_plan(src_for_dst_physical=src_for_dst, ep_size=ep_size)
        self.assertEqual(plan.ep_size, ep_size)
        self.assertEqual(plan.num_physical_experts, num_physical)
        self.assertEqual(plan.local_num_physical_experts, local_e)

        # send_sizes[src][dst] must equal recv_sizes[dst][src]
        for src_rank in range(ep_size):
            for dst_rank in range(ep_size):
                self.assertEqual(
                    int(plan.send_sizes[src_rank][dst_rank]),
                    int(plan.recv_sizes[dst_rank][src_rank]),
                )

        # Each dst rank must receive exactly local_e rows.
        for dst_rank in range(ep_size):
            self.assertEqual(int(np.sum(plan.recv_sizes[dst_rank])), local_e)

        # Total sends across all ranks equals total destinations (one row per dst slot).
        total_sent = int(sum(np.sum(plan.send_sizes[r]) for r in range(ep_size)))
        self.assertEqual(total_sent, num_physical)

        # Offsets must be prefix sums.
        for r in range(ep_size):
            sizes = plan.send_sizes[r]
            offs = plan.input_offsets[r]
            self.assertTrue(np.array_equal(offs[1:], np.cumsum(sizes[:-1])))

            rsizes = plan.recv_sizes[r]
            roffs = plan.output_offsets[r]
            self.assertTrue(np.array_equal(roffs[1:], np.cumsum(rsizes[:-1])))

        # Per-rank flattened index vectors must match total sizes.
        for r in range(ep_size):
            self.assertEqual(
                plan.send_src_local_indices[r].shape[0], int(np.sum(plan.send_sizes[r]))
            )
            self.assertEqual(
                plan.recv_dst_local_indices[r].shape[0], int(np.sum(plan.recv_sizes[r]))
            )


if __name__ == "__main__":
    unittest.main()
