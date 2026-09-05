"""Local Qwen2.5-VL preprocessing benchmark, without model weights or a TPU.

Run each worker/thread configuration in a fresh process with PYTHONPATH=python.
Inputs match the image benchmark: random 512x512 JPEGs, quality 85, data URIs.
"""

import argparse
import asyncio
import functools
import io
import json
import os
import statistics
import time
from collections import defaultdict
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pybase64
import torch
from PIL import Image
from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
from transformers.models.qwen2_vl.image_processing_qwen2_vl_fast import (
    Qwen2VLImageProcessorFast,
)

from sgl_jax.srt.multimodal.common.modality_enum import MultimodalDataItem
from sgl_jax.srt.multimodal.processors.qwen_vl import QwenVLProcessor

# Qwen/Qwen2.5-VL-3B-Instruct/preprocessor_config.json. Loading the image
# processors directly avoids downloading a tokenizer or model weights.
IMAGE_CONFIG = dict(
    min_pixels=3136,
    max_pixels=12845056,
    patch_size=14,
    temporal_patch_size=2,
    merge_size=2,
    image_mean=[0.48145466, 0.4578275, 0.40821073],
    image_std=[0.26862954, 0.26130258, 0.27577711],
)
SAMPLES = defaultdict(list)


def measured(name):
    def decorate(function):
        @functools.wraps(function)
        def wrapped(*args, **kwargs):
            wall_start, cpu_start = time.perf_counter_ns(), time.thread_time_ns()
            result = function(*args, **kwargs)
            SAMPLES[name].append(
                (
                    (time.perf_counter_ns() - wall_start) / 1e6,
                    (time.thread_time_ns() - cpu_start) / 1e6,
                )
            )
            return result

        return wrapped

    return decorate


class TimedImageProcessor:
    def __init__(self, processor):
        self.processor = processor

    @measured("image_processor")
    def __call__(self, *args, **kwargs):
        return self.processor(*args, **kwargs)


class MeasuredQwenProcessor(QwenVLProcessor):
    @measured("decode")
    def load_image(self, source):
        return super().load_image(source)

    @measured("collect_including_layout_hash")
    def _collect_encoder_images(self, output):
        return super()._collect_encoder_images(output)

    @measured("layout")
    def _prepare_vision_layouts(self, items):
        return super()._prepare_vision_layouts(items)


def make_processor(kind, workers):
    cls = Qwen2VLImageProcessor if kind == "slow" else Qwen2VLImageProcessorFast
    return MeasuredQwenProcessor(
        SimpleNamespace(
            architectures=["Qwen2_5_VLForConditionalGeneration"],
            vision_config=SimpleNamespace(spatial_merge_size=2, window_size=112, patch_size=14),
        ),
        SimpleNamespace(mm_processor_worker_num=workers),
        SimpleNamespace(image_processor=TimedImageProcessor(cls(**IMAGE_CONFIG))),
    )


def make_images(count, seed):
    rng = np.random.RandomState(seed)
    images = []
    for _ in range(count):
        pixels = (rng.rand(512, 512, 3) * 255).astype(np.uint8)
        buffer = io.BytesIO()
        Image.fromarray(pixels).save(buffer, format="jpeg", quality=85)
        encoded = pybase64.b64encode(buffer.getvalue()).decode("ascii")
        images.append(f"data:image/jpeg;base64,{encoded}")
    return images


async def run_round(processor, images, count, workers):
    requests = iter(range(count))
    timings = []

    async def consume():
        for index in requests:
            timing = {}
            result = await processor.process_encoder_mm_data_async(
                [images[index % len(images)]], "", SimpleNamespace(), encoder_timing=timing
            )
            assert len(result.mm_items) == 1
            assert result.mm_items[0].feature.shape == (1296, 1176)
            assert result.mm_items[0].feature.dtype == np.float32
            timings.append(timing)

    cpu_start, wall_start = time.process_time(), time.perf_counter()
    await asyncio.wait_for(asyncio.gather(*(consume() for _ in range(workers))), timeout=60)
    elapsed = time.perf_counter() - wall_start
    return {
        "images_per_s": count / elapsed,
        "elapsed_s": elapsed,
        "average_cpu_cores": (time.process_time() - cpu_start) / elapsed,
        "worker_request_median_ms": statistics.median(
            (t["processor_done_ns"] - t["processor_start_ns"]) / 1e6 for t in timings
        ),
        "executor_queue_median_ms": statistics.median(
            (t["processor_start_ns"] - t["processor_submit_ns"]) / 1e6 for t in timings
        ),
        "phases": {
            name: {
                "count": len(values),
                "wall_median_ms": statistics.median(wall for wall, _ in values),
                "caller_thread_cpu_median_ms": statistics.median(cpu for _, cpu in values),
            }
            for name, values in SAMPLES.items()
        },
    }


def compare_processors(images):
    import ml_dtypes

    processors = {kind: make_processor(kind, 1) for kind in ("slow", "fast")}
    comparisons = []
    try:
        for image in images[:8]:
            results = {
                kind: processor._process_encoder_images(
                    [image], processor=processor.processor
                ).mm_items[0]
                for kind, processor in processors.items()
            }
            slow, fast = results["slow"], results["fast"]
            a, b = slow.feature, fast.feature
            assert a.shape == b.shape
            comparisons.append(
                {
                    "shape": a.shape,
                    "grid_equal": np.array_equal(slow.image_grid_thw, fast.image_grid_thw),
                    "hash_equal": slow.hash == fast.hash,
                    "max_abs_difference": float(np.max(np.abs(a - b))),
                    "mean_abs_difference": float(np.mean(np.abs(a - b))),
                    "fp32_equal_fraction": float(np.mean(a == b)),
                    "bf16_equal_fraction": float(
                        np.mean(a.astype(ml_dtypes.bfloat16) == b.astype(ml_dtypes.bfloat16))
                    ),
                }
            )
    finally:
        for processor in processors.values():
            processor.shutdown()
    return comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processor", choices=("slow", "fast"), default="slow")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--torch-threads", type=int, default=0, help="0 keeps the PyTorch default")
    parser.add_argument("--requests", type=int, default=256)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.workers, args.requests, args.rounds) < 1 or args.torch_threads < 0:
        parser.error(
            "worker/request/round counts must be positive; thread count must be nonnegative"
        )
    if args.torch_threads:
        torch.set_num_threads(args.torch_threads)
    metadata = {
        "versions": {
            name: version(name)
            for name in ("torch", "torchvision", "torchcodec", "transformers", "numpy", "pillow")
        },
        "cpu_affinity_count": len(os.sched_getaffinity(0)),
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "processor": args.processor,
        "workers": args.workers,
        "requests_per_round": args.requests,
        "image_config": IMAGE_CONFIG,
    }
    print(json.dumps(metadata), flush=True)
    images = make_images(min(args.requests, 128), args.seed)
    if args.compare:
        result = {**metadata, "comparisons": compare_processors(images)}
    else:
        original_set_pad_value = MultimodalDataItem.set_pad_value
        MultimodalDataItem.set_pad_value = measured("hash")(original_set_pad_value)
        processor = make_processor(args.processor, args.workers)

        async def run():
            await run_round(processor, images, max(32, 2 * args.workers), args.workers)
            rounds = []
            for index in range(args.rounds):
                SAMPLES.clear()
                round_result = await run_round(processor, images, args.requests, args.workers)
                rounds.append(round_result)
                print(json.dumps({"round": index, **round_result}), flush=True)
            return rounds

        try:
            rounds = asyncio.run(run())
        finally:
            processor.shutdown()
            MultimodalDataItem.set_pad_value = original_set_pad_value
        result = {
            **metadata,
            "rounds": rounds,
            "median_images_per_s": statistics.median(r["images_per_s"] for r in rounds),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
