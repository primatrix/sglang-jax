import unittest

import numpy as np

from sgl_jax.srt.eplb import EplbStatsRecorder, counts_from_topk_ids


class TestEplbRecorder(unittest.TestCase):
    def test_counts_from_topk_ids(self):
        ids = np.array([[0, 1], [1, 1], [2, 0]], dtype=np.int32)
        counts = counts_from_topk_ids(ids, num_logical_experts=4, top_k=2)
        # Flatten: [0,1,1,1,2,0] => 0:2, 1:3, 2:1, 3:0
        self.assertTrue(np.array_equal(counts, np.array([2, 3, 1, 0], dtype=np.int64)))

    def test_sliding_window_sum(self):
        rec = EplbStatsRecorder(
            num_layers=2,
            num_logical_experts=4,
            num_experts_per_tok=2,
            window_size=3,
        )

        step0 = [
            np.array([[0, 1], [1, 1]], dtype=np.int32),  # layer0
            np.array([[2, 3]], dtype=np.int32),  # layer1
        ]
        rec.record_step(topk_ids_by_layer=step0)

        step1 = [
            np.array([[0, 0]], dtype=np.int32),
            None,  # treat as non-moe layer
        ]
        rec.record_step(topk_ids_by_layer=step1)

        dump = rec.dump().logical_count
        # layer0: step0 => [1,3,0,0], step1 => [2,0,0,0] => [3,3,0,0]
        self.assertTrue(np.array_equal(dump[0], np.array([3, 3, 0, 0], dtype=np.int64)))
        # layer1: step0 => [0,0,1,1], step1 => [0,0,0,0] => [0,0,1,1]
        self.assertTrue(np.array_equal(dump[1], np.array([0, 0, 1, 1], dtype=np.int64)))

        # Overwrite oldest when window slides.
        step2 = [
            np.array([[3, 3]], dtype=np.int32),
            np.array([[0, 0], [0, 0]], dtype=np.int32),
        ]
        rec.record_step(topk_ids_by_layer=step2)

        step3 = [
            np.array([[2, 2]], dtype=np.int32),
            np.array([[1, 1]], dtype=np.int32),
        ]
        rec.record_step(topk_ids_by_layer=step3)

        # Now window holds steps 1,2,3 (step0 evicted).
        dump = rec.dump().logical_count
        # layer0: step1 [2,0,0,0] + step2 [0,0,0,2] + step3 [0,0,2,0] => [2,0,2,2]
        self.assertTrue(np.array_equal(dump[0], np.array([2, 0, 2, 2], dtype=np.int64)))
        # layer1: step1 [0,0,0,0] + step2 [4,0,0,0] + step3 [0,2,0,0] => [4,2,0,0]
        self.assertTrue(np.array_equal(dump[1], np.array([4, 2, 0, 0], dtype=np.int64)))


if __name__ == "__main__":
    unittest.main()
