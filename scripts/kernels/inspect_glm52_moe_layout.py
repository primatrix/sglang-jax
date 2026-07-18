#!/usr/bin/env python3
import argparse
import json
import os
import struct
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/models/GLM-5.2")
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--projection", default="gate_proj")
    parser.add_argument("--expert-start", type=int, default=0)
    parser.add_argument("--expert-stop", type=int, default=64)
    parser.add_argument("--benchmark-workers", type=int, default=0)
    parser.add_argument("--benchmark-mode", choices=("range", "file"), default="range")
    args = parser.parse_args()

    index_path = os.path.join(args.model, "model.safetensors.index.json")
    with open(index_path) as fp:
        weight_map = json.load(fp)["weight_map"]

    keys = [
        f"model.layers.{args.layer}.mlp.experts.{expert}.{args.projection}.weight"
        for expert in range(args.expert_start, args.expert_stop)
    ]
    missing = [key for key in keys if key not in weight_map]
    if missing:
        raise KeyError(f"missing keys, first={missing[0]} count={len(missing)}")

    keys_by_file = defaultdict(list)
    for key in keys:
        keys_by_file[weight_map[key]].append(key)

    print(
        f"LAYOUT layer={args.layer} projection={args.projection} "
        f"experts={len(keys)} files={len(keys_by_file)}"
    )
    total_actual = 0
    total_span = 0
    all_ranges = []
    for filename, file_keys in sorted(keys_by_file.items()):
        path = os.path.join(args.model, filename)
        with open(path, "rb") as fp:
            header_size = struct.unpack("<Q", fp.read(8))[0]
            header = json.loads(fp.read(header_size))
        data_offset = 8 + header_size
        ranges = []
        for key in file_keys:
            start, stop = header[key]["data_offsets"]
            ranges.append((data_offset + start, data_offset + stop, key))
        ranges.sort()
        all_ranges.extend((path, start, stop, key) for start, stop, key in ranges)
        actual = sum(stop - start for start, stop, _ in ranges)
        span = ranges[-1][1] - ranges[0][0]
        gaps = [ranges[i + 1][0] - ranges[i][1] for i in range(len(ranges) - 1)]
        total_actual += actual
        total_span += span
        print(
            f"FILE name={filename} entries={len(ranges)} "
            f"actual_mb={actual / 1e6:.1f} span_mb={span / 1e6:.1f} "
            f"span_ratio={span / actual:.2f} max_gap_mb={max(gaps, default=0) / 1e6:.1f}"
        )
    print(
        f"TOTAL actual_mb={total_actual / 1e6:.1f} span_mb={total_span / 1e6:.1f} "
        f"span_ratio={total_span / total_actual:.2f}"
    )

    if args.benchmark_workers:
        def read_range(entry):
            path, start, stop, key = entry
            with open(path, "rb") as fp:
                fp.seek(start)
                raw = fp.read(stop - start)
            if len(raw) != stop - start:
                raise OSError(f"short read for {key}: {len(raw)} != {stop - start}")
            return len(raw), raw[0], raw[-1]

        started = time.monotonic()
        if args.benchmark_mode == "range":
            with ThreadPoolExecutor(max_workers=args.benchmark_workers) as executor:
                results = list(executor.map(read_range, all_ranges))
        else:
            ranges_by_path = defaultdict(list)
            for entry in all_ranges:
                ranges_by_path[entry[0]].append(entry)

            def read_file_ranges(entries):
                return [read_range(entry) for entry in entries]

            with ThreadPoolExecutor(
                max_workers=min(args.benchmark_workers, len(ranges_by_path))
            ) as executor:
                nested_results = list(executor.map(read_file_ranges, ranges_by_path.values()))
            results = [result for file_results in nested_results for result in file_results]
        elapsed = time.monotonic() - started
        bytes_read = sum(result[0] for result in results)
        checksum = sum(result[1] + result[2] for result in results)
        print(
            f"BENCH mode={args.benchmark_mode} workers={args.benchmark_workers} "
            f"bytes={bytes_read} "
            f"seconds={elapsed:.3f} mbps={bytes_read / 1e6 / elapsed:.1f} "
            f"checksum={checksum}"
        )


if __name__ == "__main__":
    main()
