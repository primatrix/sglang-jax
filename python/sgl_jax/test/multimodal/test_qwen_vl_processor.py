import unittest

import numpy as np

from sgl_jax.srt.multimodal.common.modality_enum import Modality
from sgl_jax.srt.multimodal.processors.qwen_vl import QwenVLProcessor


class TestQwenVLProcessor(unittest.TestCase):
    def test_compute_image_placeholder_ranges_uses_expanded_image_token_spans(self):
        image_token_id = 151655
        input_ids = [
            1,
            2,
            image_token_id,
            image_token_id,
            3,
            image_token_id,
            image_token_id,
            image_token_id,
            image_token_id,
            4,
        ]
        grids = [(1, 2, 4), (1, 4, 4)]

        placeholder_ranges = QwenVLProcessor._compute_image_placeholder_ranges(
            input_ids=input_ids,
            grids=grids,
            image_token_id=image_token_id,
            spatial_merge_size=2,
        )

        self.assertEqual(placeholder_ranges, [(2, 3), (5, 8)])

    def test_build_items_attaches_per_image_placeholder_ranges(self):
        features = np.arange(24).reshape(24, 1)
        grids = [(1, 2, 4), (1, 4, 4)]
        placeholder_ranges = [(2, 3), (5, 8)]

        items = QwenVLProcessor._build_items(
            features,
            grids,
            placeholder_ranges,
            Modality.IMAGE,
            "image_grid_thw",
        )

        self.assertEqual(len(items), 2)
        self.assertEqual(items[0].placeholder_ranges, [(2, 3)])
        self.assertEqual(items[1].placeholder_ranges, [(5, 8)])
        self.assertEqual(items[0].feature.shape, (8, 1))
        self.assertEqual(items[1].feature.shape, (16, 1))
        np.testing.assert_array_equal(
            items[0].model_specific_data["image_grid_thw"],
            np.array([[1, 2, 4]], dtype=np.int32),
        )
        np.testing.assert_array_equal(
            items[1].model_specific_data["image_grid_thw"],
            np.array([[1, 4, 4]], dtype=np.int32),
        )


if __name__ == "__main__":
    unittest.main()
