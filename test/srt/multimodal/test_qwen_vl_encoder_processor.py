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


def test_process_encoder_images_records_processor_timing():
    class ImageProcessor:
        def __call__(self, *, images, return_tensors):
            assert len(images) == 1
            assert return_tensors == "pt"
            return {
                "pixel_values": np.arange(8).reshape(4, 2),
                "image_grid_thw": np.asarray([[1, 2, 2]]),
            }

    processor = object.__new__(QwenVLProcessor)
    processor.hf_config = SimpleNamespace(vision_config=SimpleNamespace(spatial_merge_size=2))
    timing = {}

    processor._process_encoder_images(
        [object()],
        processor=SimpleNamespace(image_processor=ImageProcessor()),
        encoder_timing=timing,
    )

    assert timing["processor_start_ns"] <= timing["processor_done_ns"]
