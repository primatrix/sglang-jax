"""Measure encoder packing while preserving the processor feature dtype.

Run with PYTHONPATH=python. Includes lane planning, allocation and copying.
"""

import argparse
import json
import statistics
import time
from types import SimpleNamespace

import numpy as np

from sgl_jax.srt.multimodal.in_model import lane_packing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=int, nargs="+", default=[1, 4, 5, 8])
    parser.add_argument("--patches", type=int, nargs="+", default=[1296])
    parser.add_argument("--width", type=int, default=1176)
    parser.add_argument("--lanes", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=30)
    args = parser.parse_args()
    if min(*args.images, *args.patches, args.width, args.lanes, args.iterations) < 1:
        parser.error("all sizes and iteration counts must be positive")
    if any(count % 4 for count in args.patches):
        parser.error("patch counts must be divisible by the merge unit (4)")
    rng = np.random.default_rng(0)
    kwargs = dict(
        num_lanes=args.lanes,
        buckets=(2048, 4096, 8192),
        merge_unit=4,
    )
    for count in args.images:
        items = [
            SimpleNamespace(
                feature=rng.normal(size=(args.patches[i % len(args.patches)], args.width)).astype(
                    np.float32
                )
            )
            for i in range(count)
        ]
        samples = []
        for iteration in range(args.iterations + 5):
            start = time.perf_counter_ns()
            result = lane_packing.pack_lanes(items, **kwargs)
            elapsed_ms = (time.perf_counter_ns() - start) / 1e6
            shape = result.features.shape
            packed_dtype = str(result.features.dtype)
            del result
            if iteration >= 5:
                samples.append(elapsed_ms)
        print(
            json.dumps(
                {
                    "images": count,
                    "patches": [item.feature.shape[0] for item in items],
                    "packed_shape": shape,
                    "packed_dtype": packed_dtype,
                    "iterations": args.iterations,
                    "median_ms": {"packing": statistics.median(samples)},
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
