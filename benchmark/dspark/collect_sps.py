#!/usr/bin/env python3
"""Collect DSpark verify-all SPS points from a live sglang-jax server."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import time
import urllib.request
from pathlib import Path


def _server_info(base_url: str) -> dict:
    with urllib.request.urlopen(f"{base_url}/get_server_info", timeout=30) as response:
        return json.load(response)


def _step_samples_by_state(
    server_info: dict, bucket: int, lower_exclusive: int
) -> list[dict[int, list[float]]]:
    samples = []
    for state in server_info.get("internal_states", []):
        table = state.get("decode_step_time_by_batch_size", {})
        samples.append(
            {
                int(batch_size): [float(value) for value in values]
                for batch_size, values in table.items()
                if lower_exclusive < int(batch_size) <= bucket
            }
        )
    return samples


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot compute a percentile from no samples.")
    index = (len(ordered) - 1) * q
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _run_point(
    *,
    python: str,
    base_url: str,
    model: str,
    concurrency: int,
    lower_exclusive: int,
    input_len: int,
    output_len: int,
    prompts_multiplier: int,
    output_file: Path,
) -> dict:
    command = [
        python,
        "-m",
        "sgl_jax.bench_serving",
        "--backend",
        "sgl-jax",
        "--base-url",
        base_url,
        "--model",
        model,
        "--dataset-name",
        "random",
        "--num-prompts",
        str(concurrency * prompts_multiplier),
        "--max-concurrency",
        str(concurrency),
        "--random-input-len",
        str(input_len),
        "--random-output-len",
        str(output_len),
        "--random-range-ratio",
        "1",
        "--output-file",
        str(output_file),
        "--disable-tqdm",
        "--seed",
        "980406",
        "--extra-request-body",
        '{"temperature":0.0}',
    ]
    warmup_command = list(command)
    warmup_command[warmup_command.index("--num-prompts") + 1] = str(concurrency)
    warmup_command[warmup_command.index("--random-output-len") + 1] = "64"
    warmup_command[warmup_command.index("--output-file") + 1] = str(
        output_file.with_name("warmup.jsonl")
    )
    subprocess.run(warmup_command, check=True)
    before = _server_info(base_url)
    before_samples = _step_samples_by_state(before, concurrency, lower_exclusive)
    started = time.perf_counter()
    subprocess.run(command, check=True)
    elapsed = time.perf_counter() - started
    after = _server_info(base_url)
    after_samples = _step_samples_by_state(after, concurrency, lower_exclusive)
    samples = []
    observed_batch_sizes = []
    for state_index, state_table in enumerate(after_samples):
        old_table = before_samples[state_index] if state_index < len(before_samples) else {}
        for batch_size, state_samples in state_table.items():
            new_samples = state_samples[len(old_table.get(batch_size, [])) :]
            samples.extend(new_samples)
            observed_batch_sizes.extend([batch_size] * len(new_samples))
    if not samples:
        raise RuntimeError(
            f"No decode-step samples in bucket ({lower_exclusive}, {concurrency}] were recorded. "
            "Launch with SGLANG_RECORD_STEP_TIME=1 and --decode-log-interval 1."
        )
    result = json.loads(output_file.read_text(encoding="utf-8").splitlines()[-1])
    median = statistics.median(samples)
    return {
        "global_concurrency": concurrency,
        "global_request_bucket": concurrency,
        "requests_per_dp": concurrency // 2,
        "verify_tokens_per_dp": concurrency // 2 * 8,
        "step_samples": len(samples),
        "observed_running_requests_min": min(observed_batch_sizes),
        "observed_running_requests_max": max(observed_batch_sizes),
        "median_step_time_ms": median * 1000.0,
        "p90_step_time_ms": _percentile(samples, 0.90) * 1000.0,
        "p99_step_time_ms": _percentile(samples, 0.99) * 1000.0,
        "steps_per_second": 1.0 / median,
        "client_elapsed_s": elapsed,
        "output_throughput": result["output_throughput"],
        "mean_itl_ms": result["mean_itl_ms"],
        "accept_length": result.get("accept_length"),
        "cache_hit_rate": result.get("cache_hit_rate"),
        "result_jsonl": str(output_file),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default="python/.venv/bin/python")
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument(
        "--concurrency",
        nargs="+",
        type=int,
        default=[2, 4, 8, 16, 32, 64, 128, 256, 512],
    )
    parser.add_argument("--input-len", type=int, default=256)
    parser.add_argument("--output-len", type=int, default=512)
    parser.add_argument("--prompts-multiplier", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    lower_exclusive = 0
    for concurrency in args.concurrency:
        if concurrency % 2:
            raise ValueError("This TP=8/DP=2 experiment requires even global concurrency.")
        point_dir = args.output_dir / f"bs_{concurrency}"
        point_dir.mkdir(parents=True, exist_ok=True)
        rows.append(
            _run_point(
                python=args.python,
                base_url=args.base_url,
                model=args.model,
                concurrency=concurrency,
                lower_exclusive=lower_exclusive,
                input_len=args.input_len,
                output_len=args.output_len,
                prompts_multiplier=args.prompts_multiplier,
                output_file=point_dir / "result.jsonl",
            )
        )
        lower_exclusive = concurrency
    table_path = args.output_dir / "sps_table.json"
    table = {
        "schema_version": 1,
        "kind": "dspark_verify_all_sps_frontier",
        "gamma": 7,
        "verify_width": 8,
        "tp_size": 8,
        "dp_size": 2,
        "input_len": args.input_len,
        "output_len": args.output_len,
        "points": rows,
    }
    table_path.write_text(json.dumps(table, indent=2) + "\n", encoding="utf-8")
    with (args.output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(table, indent=2))


if __name__ == "__main__":
    main()
