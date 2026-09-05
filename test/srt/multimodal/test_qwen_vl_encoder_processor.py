import asyncio
import threading
from types import SimpleNamespace

import numpy as np
import pytest

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


@pytest.mark.parametrize("workers", [1, 2])
def test_encoder_loads_and_processes_requests_on_processor_workers(monkeypatch, workers):
    stages = {}
    hf_processors = {}
    barrier = threading.Barrier(workers)

    def load_image(source):
        stages[source] = [threading.get_ident()]
        assert threading.current_thread().name.startswith("sgl-jax-mm-processor")
        barrier.wait(timeout=2)
        return source

    class ImageProcessor:
        def __call__(self, *, images, return_tensors):
            assert len(images) == 1
            assert return_tensors == "pt"
            source = images[0]
            stages[source].append(threading.get_ident())
            hf_processors[source] = id(self)
            return {
                "pixel_values": np.full((4, 2), source),
                "image_grid_thw": np.asarray([[1, 2, 2]]),
            }

    processor = QwenVLProcessor(
        SimpleNamespace(
            architectures=["Qwen2_5_VLForConditionalGeneration"],
            vision_config=SimpleNamespace(spatial_merge_size=2, window_size=4, patch_size=1),
        ),
        SimpleNamespace(mm_processor_worker_num=workers),
        SimpleNamespace(image_processor=ImageProcessor()),
    )
    monkeypatch.setattr(processor, "load_image", load_image)
    prepare_layouts = processor._prepare_vision_layouts

    def prepare_on_worker(items):
        source = int(items[0].feature[0, 0])
        stages[source].append(threading.get_ident())
        prepare_layouts(items)

    monkeypatch.setattr(processor, "_prepare_vision_layouts", prepare_on_worker)
    timings = [{} for _ in range(workers)]
    event_loop_thread_id = threading.get_ident()

    async def run():
        return await asyncio.wait_for(
            asyncio.gather(
                *(
                    processor.process_encoder_mm_data_async(
                        [source], "prompt", SimpleNamespace(), encoder_timing=timing
                    )
                    for source, timing in enumerate(timings)
                )
            ),
            timeout=3,
        )

    try:
        results = asyncio.run(run())
    finally:
        processor.shutdown()

    assert len(set(hf_processors.values())) == workers
    assert len({thread_ids[0] for thread_ids in stages.values()}) == workers
    for source, (result, timing) in enumerate(zip(results, timings)):
        assert stages[source][0] == stages[source][1] == stages[source][2] != event_loop_thread_id
        np.testing.assert_array_equal(result.mm_items[0].feature, np.full((4, 2), source))
        layout = result.mm_items[0].get("vision_layout")
        np.testing.assert_array_equal(layout.indices, [[0, 0]])
        np.testing.assert_array_equal(layout.position_ids, [[0, 0], [0, 1], [1, 0], [1, 1]])
        np.testing.assert_array_equal(layout.window_ends, [4])
        np.testing.assert_array_equal(layout.frame_ends, [4])
        assert (
            timing["processor_submit_ns"]
            <= timing["processor_start_ns"]
            <= timing["image_load_start_ns"]
            <= timing["image_load_done_ns"]
            <= timing["processor_done_ns"]
        )


def test_encoder_worker_accepts_next_request_after_load_failure(monkeypatch):
    class ImageProcessor:
        def __call__(self, *, images, **kwargs):
            return {
                "pixel_values": np.zeros((4, 2)),
                "image_grid_thw": np.asarray([[1, 2, 2]]),
            }

    processor = QwenVLProcessor(
        SimpleNamespace(vision_config=SimpleNamespace(spatial_merge_size=2)),
        SimpleNamespace(mm_processor_worker_num=1),
        SimpleNamespace(image_processor=ImageProcessor()),
    )

    def load_image(source):
        if source == "invalid":
            raise ValueError("invalid image")
        return source

    monkeypatch.setattr(processor, "load_image", load_image)

    async def run():
        with pytest.raises(ValueError, match="invalid image"):
            await processor.process_encoder_mm_data_async("invalid", "prompt", SimpleNamespace())
        return await processor.process_encoder_mm_data_async("valid", "prompt", SimpleNamespace())

    try:
        result = asyncio.run(asyncio.wait_for(run(), timeout=3))
    finally:
        processor.shutdown()
    assert len(result.mm_items) == 1
