from types import SimpleNamespace

import numpy as np

from sgl_jax.srt.multimodal.processors.qwen_vl import QwenVLProcessor


def test_collect_encoder_images_builds_features_without_text_metadata():
    processor = object.__new__(QwenVLProcessor)
    processor.hf_config = SimpleNamespace(vision_config=SimpleNamespace(spatial_merge_size=2))
    features = np.arange(24).reshape(12, 2)

    result = processor._collect_encoder_images(
        {
            "pixel_values": features,
            "image_grid_thw": np.asarray([[1, 2, 2], [1, 4, 2]]),
        }
    )

    assert result.input_ids is None
    assert [item.placeholder_ranges for item in result.mm_items] == [[(0, 1)], [(1, 3)]]
    np.testing.assert_array_equal(result.mm_items[0].feature, features[:4])
    np.testing.assert_array_equal(result.mm_items[1].feature, features[4:])
