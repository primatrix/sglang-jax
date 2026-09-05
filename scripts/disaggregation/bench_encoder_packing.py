"""Compare encoder packing with NumPy and compiled FP32-to-BF16 copies.

Run with PYTHONPATH=python. Compilation happens at import and is excluded.
Both paths use the same lane planning, allocation, padding and annotations.
"""

import argparse
import json
import statistics
import time
from types import SimpleNamespace
from unittest.mock import patch

import ml_dtypes
import numpy as np

from sgl_jax.srt.multimodal.in_model import lane_packing


def _numpy_copy(destination, source):
    destination[...] = source


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
    implementations = {"numpy": _numpy_copy, "compiled": lane_packing.copy_features}
    kwargs = dict(
        num_lanes=args.lanes,
        buckets=(2048, 4096, 8192),
        merge_unit=4,
        dtype=ml_dtypes.bfloat16,
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
        expected = None
        for copy in implementations.values():
            with patch.object(lane_packing, "copy_features", copy):
                actual = lane_packing.pack_lanes(items, **kwargs)
            if expected is not None:
                assert actual.cap == expected.cap and actual.lanes == expected.lanes
                np.testing.assert_array_equal(
                    actual.features.view(np.uint16), expected.features.view(np.uint16)
                )
                np.testing.assert_array_equal(actual.output_indices, expected.output_indices)
            expected = actual
        shape = expected.features.shape
        del actual, expected

        samples = {name: [] for name in implementations}
        for iteration in range(args.iterations + 5):
            # Alternate execution order to limit allocator/cache ordering bias.
            order = list(implementations.items())
            if iteration % 2:
                order.reverse()
            for name, copy in order:
                with patch.object(lane_packing, "copy_features", copy):
                    start = time.perf_counter_ns()
                    result = lane_packing.pack_lanes(items, **kwargs)
                    elapsed_ms = (time.perf_counter_ns() - start) / 1e6
                del result
                if iteration >= 5:
                    samples[name].append(elapsed_ms)
        print(
            json.dumps(
                {
                    "images": count,
                    "patches": [item.feature.shape[0] for item in items],
                    "packed_shape": shape,
                    "iterations": args.iterations,
                    "median_ms": {
                        name: statistics.median(values) for name, values in samples.items()
                    },
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
