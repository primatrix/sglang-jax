#!/usr/bin/env python3
"""Measure one DSpark ragged verify T(R, M) point from a live server."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
import urllib.request
from pathlib import Path


def _server_info(base_url: str) -> dict:
    with urllib.request.urlopen(f"{base_url}/get_server_info", timeout=30) as response:
        return json.load(response)


def _samples_for_batch(server_info: dict, batch_size: int) -> list[list[float]]:
    samples = []
    for state in server_info.get("internal_states", []):
        table = state.get("decode_step_time_by_batch_size", {})
        samples.append([float(value) for value in table.get(str(batch_size), [])])
    return samples


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _bench_command(
    args: argparse.Namespace, *, output_len: int, output_file: Path
) -> list[str]:
    return [
        args.python,
        "-m",
        "sgl_jax.bench_serving",
        "--backend",
        "sgl-jax",
        "--base-url",
        args.base_url,
        "--model",
        args.model,
        "--dataset-name",
        "random",
        "--num-prompts",
        str(args.global_concurrency * args.prompts_multiplier),
        "--max-concurrency",
        str(args.global_concurrency),
        "--random-input-len",
        str(args.input_len),
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default="python/.venv/bin/python")
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--global-concurrency", type=int, required=True)
    parser.add_argument("--request-bucket-per-dp", type=int, required=True)
    parser.add_argument("--verify-token-bucket-per-dp", type=int, required=True)
    parser.add_argument("--dp-size", type=int, default=2)
    parser.add_argument("--input-len", type=int, default=256)
    parser.add_argument("--output-len", type=int, default=256)
    parser.add_argument("--prompts-multiplier", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.global_concurrency != args.request_bucket_per_dp * args.dp_size:
        raise ValueError("global concurrency must equal request bucket * dp size")
    if not (
        args.request_bucket_per_dp
        <= args.verify_token_bucket_per_dp
        <= args.request_bucket_per_dp * 8
    ):
        raise ValueError("verify token bucket must be within [R, R * verify_width]")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    warmup_file = args.output.with_name(f"{args.output.stem}_warmup.jsonl")
    result_file = args.output.with_name(f"{args.output.stem}_serving.jsonl")
    subprocess.run(
        _bench_command(args, output_len=64, output_file=warmup_file),
        check=True,
    )
    before = _samples_for_batch(_server_info(args.base_url), args.global_concurrency)
    started = time.perf_counter()
    subprocess.run(
        _bench_command(args, output_len=args.output_len, output_file=result_file),
        check=True,
    )
    elapsed = time.perf_counter() - started
    after = _samples_for_batch(_server_info(args.base_url), args.global_concurrency)

    samples = []
    for state_index, state_samples in enumerate(after):
        old_count = len(before[state_index]) if state_index < len(before) else 0
        samples.extend(state_samples[old_count:])
    if not samples:
        raise RuntimeError(
            f"No decode-step samples were recorded at exact global batch {args.global_concurrency}."
        )

    serving_result = json.loads(
        result_file.read_text(encoding="utf-8").splitlines()[-1]
    )
    median = statistics.median(samples)
    point = {
        "request_bucket_per_dp": args.request_bucket_per_dp,
        "verify_tokens_per_dp": args.verify_token_bucket_per_dp,
        "global_concurrency": args.global_concurrency,
        "step_samples": len(samples),
        "median_step_time_ms": median * 1000.0,
        "p90_step_time_ms": _percentile(samples, 0.90) * 1000.0,
        "p99_step_time_ms": _percentile(samples, 0.99) * 1000.0,
        "steps_per_second": 1.0 / median,
        "client_elapsed_s": elapsed,
        "output_throughput": serving_result["output_throughput"],
        "mean_itl_ms": serving_result["mean_itl_ms"],
        "accept_length": serving_result.get("accept_length"),
        "result_jsonl": str(result_file),
    }
    args.output.write_text(json.dumps(point, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(point, indent=2))


if __name__ == "__main__":
    main()
