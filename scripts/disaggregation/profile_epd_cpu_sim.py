#!/usr/bin/env python3
"""Arm profiling across the EPD CPU-sim tiers, drive image requests, and stop.

Pairs with scripts/disaggregation/run_epd_cpu_sim.sh. Captures a jax.profiler
trace on every encoder process and on the language server (prefill+decode),
all under one profiler dir, so you can align them into a single EPD flame graph.

Example:
    python scripts/disaggregation/profile_epd_cpu_sim.py \
        --lang-url http://127.0.0.1:30000 \
        --encoder-url http://127.0.0.1:31001 \
        --image https://.../demo.jpeg --n-requests 4 --max-tokens 32

Only the standard library is required.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import mimetypes
import os
import re
import statistics
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

from sgl_jax.srt.disaggregation.encoder.metrics import (
    summarize_raiden_transfer_inflight,
)

DEFAULT_PROFILER_DIR = "/tmp/epd-sim-profile"
_QUEUE_RE = re.compile(
    r"enqueue_ns=(\d+).*?queue_ms=([0-9.]+).*?batch_size=(\d+).*?queue_depth=(\d+)"
)
_PIPELINE_RE = re.compile(r"ENCODER-PIPELINE-TIME (?P<body>[^\n]+)")
_KEY_VALUE_RE = re.compile(r"([a-z_]+)=([^\s]+)")
_BENCHMARK_KEYS = (
    "completed",
    "duration",
    "request_throughput",
    "input_throughput",
    "output_throughput",
    "total_throughput",
    "mean_ttft_ms",
    "median_ttft_ms",
    "p99_ttft_ms",
    "mean_tpot_ms",
    "median_tpot_ms",
    "p99_tpot_ms",
    "median_e2e_latency_ms",
    "p99_e2e_latency_ms",
    "total_input_text_tokens",
    "total_input_vision_tokens",
    "total_output_tokens",
)


def _post(url: str, payload: dict | None, timeout: float = 1200.0) -> dict:
    data = json.dumps(payload or {}).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return {"raw": body}


def _image_content(image: str) -> dict:
    """Build an OpenAI image_url content block; inline local files as data URIs."""
    if image.startswith(("http://", "https://", "data:")):
        url = image
    else:
        mime = mimetypes.guess_type(image)[0] or "image/jpeg"
        with open(image, "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode("ascii")
        url = f"data:{mime};base64,{b64}"
    return {"type": "image_url", "image_url": {"url": url}}


def _chat_request(args, image_blocks: list) -> dict:
    return {
        "model": args.model,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": args.prompt}, *image_blocks],
            }
        ],
        "max_tokens": args.max_tokens,
        "temperature": 0,
        # Force a fixed number of decode steps regardless of the (meaningless
        # under --simulate-compute) sampled tokens, so decode is actually
        # exercised and profiled.
        "ignore_eos": True,
    }


def _arm(url: str, output_dir: str, args) -> None:
    body = {"output_dir": output_dir}
    if args.host_tracer_level is not None:
        body["host_tracer_level"] = args.host_tracer_level
    if args.python_tracer_level is not None:
        body["python_tracer_level"] = args.python_tracer_level
    print(f"  start_profile -> {url}  ({output_dir})")
    _post(f"{url}/start_profile", body)


def _stop(url: str) -> None:
    _post(f"{url}/stop_profile", None)
    print(f"  stop_profile -> {url}")


def _run_bench_serving(
    args,
    *,
    num_prompts: int,
    concurrency: int,
    warmup_requests: int,
    output_file: Path,
    output_details: bool = False,
) -> dict:
    endpoint = urlparse(args.lang_url)
    if endpoint.scheme not in ("http", "https") or not endpoint.hostname:
        raise ValueError(f"invalid language URL: {args.lang_url!r}")
    port = endpoint.port or (443 if endpoint.scheme == "https" else 80)
    output_file.unlink(missing_ok=True)
    command = [
        sys.executable,
        "-m",
        "sgl_jax.bench_serving",
        "--backend",
        "sglang-oai-chat",
        "--host",
        endpoint.hostname,
        "--port",
        str(port),
        "--model",
        args.model_path,
        "--tokenizer",
        args.model_path,
        "--dataset-name",
        "image",
        "--num-prompts",
        str(num_prompts),
        "--max-concurrency",
        str(concurrency),
        "--random-input-len",
        str(args.random_input_len),
        "--random-output-len",
        str(args.max_tokens),
        "--random-range-ratio",
        "1.0",
        "--image-count",
        str(args.images_per_request),
        "--image-resolution",
        args.image_resolution,
        "--image-format",
        args.image_format,
        "--image-content",
        args.image_content,
        "--request-rate",
        "inf",
        "--seed",
        str(args.seed),
        "--warmup-requests",
        str(warmup_requests),
        "--flush-cache",
        "--disable-tqdm",
        "--output-file",
        str(output_file),
    ]
    if output_details:
        command.append("--output-details")
    subprocess.run(command, check=True)
    return json.loads(output_file.read_text().splitlines()[-1])


def _summarize_encoder_queue(
    log_paths: list[Path],
    *,
    start_ns: int,
    end_ns: int,
) -> dict:
    rows = []
    by_encoder = {}
    for path in log_paths:
        encoder_rows = []
        for match in _QUEUE_RE.finditer(path.read_text()):
            enqueue_ns = int(match.group(1))
            if start_ns <= enqueue_ns <= end_ns:
                encoder_rows.append(
                    {
                        "enqueue_ns": enqueue_ns,
                        "queue_ms": float(match.group(2)),
                        "batch_size": int(match.group(3)),
                        "queue_depth": int(match.group(4)),
                    }
                )
        by_encoder[path.name] = len(encoder_rows)
        rows.extend(encoder_rows)

    queue_ms = [row["queue_ms"] for row in rows]
    batch_sizes = [row["batch_size"] for row in rows]
    return _summarize_latencies(queue_ms) | {
        "n": len(rows),
        "by_encoder": by_encoder,
        "mean_batch_size": statistics.fmean(batch_sizes) if batch_sizes else None,
        "max_batch_size": max(batch_sizes) if batch_sizes else None,
    }


def _summarize_latencies(values: list[float]) -> dict:
    def percentile(q: float) -> float | None:
        if not values:
            return None
        ordered = sorted(values)
        return ordered[max(0, math.ceil(q * len(ordered)) - 1)]

    return {
        "mean_ms": statistics.fmean(values) if values else None,
        "p50_ms": statistics.median(values) if values else None,
        "p95_ms": percentile(0.95),
        "p99_ms": percentile(0.99),
        "max_ms": max(values) if values else None,
    }


def _summarize_encoder_pipeline(log_path: Path, *, start_ns: int, end_ns: int) -> dict:
    rows = []
    for match in _PIPELINE_RE.finditer(log_path.read_text()):
        row = dict(_KEY_VALUE_RE.findall(match.group("body")))
        enqueue_ns = int(row["enqueue_ns"])
        if start_ns <= enqueue_ns <= end_ns:
            rows.append(row)

    phases = (
        "queue_ms",
        "encode_stage_wait_ms",
        "preprocess_ms",
        "encode_wait_ms",
        "transfer_reserve_ms",
        "encode_dispatch_ms",
        "encode_compute_ms",
        "encode_ms",
        "post_vit_to_copy_ms",
        "server_postprocess_ms",
        "server_token_count_ms",
        "server_embedding_slice_ms",
        "server_split_compile_wait_ms",
        "server_split_dispatch_ms",
        "server_metadata_ms",
        "server_result_pack_ms",
        "server_postprocess_residual_ms",
        "runtime_return_gap_ms",
        "runtime_postprocess_ms",
        "runtime_metadata_prepare_ms",
        "runtime_embedding_data_ms",
        "runtime_result_pack_ms",
        "runtime_postprocess_residual_ms",
        "runtime_timing_attach_ms",
        "runtime_to_copy_gap_ms",
        "publish_ms",
        "transfer_handoff_ms",
        "transfer_queue_ms",
        "transfer_pool_setup_ms",
        "transfer_copy_submit_ms",
        "transfer_copy_wait_ms",
        "transfer_worker_wait_ms",
        "transfer_post_copy_queue_ms",
        "transfer_register_ms",
        "transfer_publish_finalize_ms",
        "transfer_total_ms",
        "receive_ms",
        "mm_prepare_ms",
        "receive_metadata_wait_ms",
        "receive_setup_ms",
        "receive_transfer_wait_ms",
        "receive_completion_to_materialize_ms",
        "receive_materialize_wait_ms",
        "receive_poll_delay_ms",
        "receive_finalize_ms",
        "receive_concat_ms",
        "receive_extra_meta_ms",
        "receive_result_pack_ms",
        "language_pickup_wait_ms",
        "language_get_mm_data_ms",
        "language_radix_finalize_ms",
        "receive_mm_ms",
        "language_admission_wait_ms",
        "language_queue_after_pickup_ms",
        "language_queue_ms",
        "prefill_ms",
        "total_to_prefill_ms",
        "total_to_prefill_done_ms",
    )
    return {
        "n": len(rows),
        "phases": {
            phase: _summarize_latencies([float(row[phase]) for row in rows]) for phase in phases
        },
    }


def _run_aligned_benchmark(args, encoder_urls: list[str]) -> int:
    output_dir = Path(args.profiler_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prewarm_file = output_dir / "prewarm.jsonl"
    benchmark_file = output_dir / "profile-load.jsonl"

    print(f"prewarming {args.prewarm_requests} requests, concurrency={args.prewarm_concurrency}")
    _run_bench_serving(
        args,
        num_prompts=args.prewarm_requests,
        concurrency=args.prewarm_concurrency,
        warmup_requests=1,
        output_file=prewarm_file,
    )

    print("arming profilers:")
    for idx, url in enumerate(encoder_urls):
        _arm(url, os.path.join(args.profiler_dir, f"encoder_{idx}"), args)
    _arm(args.lang_url, os.path.join(args.profiler_dir, "language"), args)

    start_ns = time.time_ns()
    try:
        print(f"driving {args.n_requests} aligned requests, concurrency={args.concurrency}")
        result = _run_bench_serving(
            args,
            num_prompts=args.n_requests,
            concurrency=args.concurrency,
            warmup_requests=0,
            output_file=benchmark_file,
            output_details=True,
        )
    finally:
        end_ns = time.time_ns()
        print("stopping profilers:")
        for url in encoder_urls:
            _stop(url)
        _stop(args.lang_url)

    queue = _summarize_encoder_queue(
        [output_dir / f"encoder_{idx}.log" for idx in range(len(encoder_urls))],
        start_ns=start_ns,
        end_ns=end_ns,
    )
    pipeline = _summarize_encoder_pipeline(
        output_dir / "language.log",
        start_ns=start_ns,
        end_ns=end_ns,
    )
    transfer_inflight = summarize_raiden_transfer_inflight(
        [output_dir / f"encoder_{idx}.log" for idx in range(len(encoder_urls))],
        start_ns=start_ns,
        end_ns=end_ns,
    )
    if queue["n"] != args.n_requests or pipeline["n"] != args.n_requests:
        raise RuntimeError(
            "formal-window coverage: "
            f"queue={queue['n']}/{args.n_requests}, "
            f"pipeline={pipeline['n']}/{args.n_requests}"
        )
    summary = {
        "schema_version": 1,
        "measurement_start_ns": start_ns,
        "measurement_end_ns": end_ns,
        "workload": {
            "num_prompts": args.n_requests,
            "max_concurrency": args.concurrency,
            "random_input_len": args.random_input_len,
            "random_output_len": args.max_tokens,
            "image_count": args.images_per_request,
            "image_resolution": args.image_resolution,
            "image_format": args.image_format,
            "image_content": args.image_content,
            "seed": args.seed,
        },
        "benchmark": {key: result[key] for key in _BENCHMARK_KEYS if key in result},
        "encoder_queue": queue,
        "encoder_pipeline": pipeline,
        "encoder_transfer_inflight": transfer_inflight,
    }
    summary_path = output_dir / "aligned-summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print("EPD_ALIGNED_RESULT " + json.dumps(summary, sort_keys=True))
    print(f"aligned summary -> {summary_path}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lang-url", default="http://127.0.0.1:30000")
    p.add_argument(
        "--encoder-url",
        action="append",
        default=[],
        help="Encoder base URL (repeat for multiple encoders).",
    )
    p.add_argument("--image", help="Image URL or local file path for the legacy fixed-image mode.")
    p.add_argument(
        "--bench-serving",
        action="store_true",
        help="Drive the same generated image workload as sgl_jax.bench_serving.",
    )
    p.add_argument("--model-path", help="Model/tokenizer path for --bench-serving.")
    p.add_argument(
        "--images-per-request",
        type=int,
        default=1,
        help="How many image items to attach per request (drives encoder + prefill load).",
    )
    p.add_argument("--prompt", default="Describe this image in detail.")
    p.add_argument("--model", default="model")
    p.add_argument("--n-requests", type=int, default=4)
    p.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Requests in flight at once. >1 exercises the scheduler's prefill/"
        "decode batching (the interesting EPD orchestration under load). 1 = "
        "sequential (clean per-request timeline).",
    )
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--max-tokens", type=int, default=32)
    p.add_argument("--prewarm-requests", type=int, default=32)
    p.add_argument("--prewarm-concurrency", type=int, default=16)
    p.add_argument("--random-input-len", type=int, default=1024)
    p.add_argument("--image-resolution", default="512x512")
    p.add_argument("--image-format", choices=("jpeg", "png"), default="jpeg")
    p.add_argument("--image-content", choices=("random", "blank"), default="random")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--profiler-dir",
        default=DEFAULT_PROFILER_DIR,
        help="Must match PROFILER_DIR used by run_epd_cpu_sim.sh.",
    )
    p.add_argument(
        "--host-tracer-level",
        type=int,
        default=None,
        help="XProf host tracer level (0-3). Default keeps the server default.",
    )
    p.add_argument(
        "--python-tracer-level",
        type=int,
        default=1,
        help="XProf python tracer level. 1 (default) = full per-call Python "
        "frames in Perfetto (zoom in to resolve the many tiny slices; keep the "
        "workload small, e.g. --n-requests 1 --max-tokens 8, to avoid the 1M-event "
        "truncation). 0 = stage annotations only (clean flame-graph view).",
    )
    args = p.parse_args()

    encoder_urls = args.encoder_url or ["http://127.0.0.1:31001"]
    if args.bench_serving:
        if not args.model_path:
            p.error("--model-path is required with --bench-serving")
        return _run_aligned_benchmark(args, encoder_urls)
    if not args.image:
        p.error("--image is required unless --bench-serving is used")

    image_blocks = [_image_content(args.image)] * max(1, args.images_per_request)

    # Warmup outside the trace window to keep the flame graph focused.
    for i in range(args.warmup):
        print(f"warmup {i + 1}/{args.warmup}")
        _post(f"{args.lang_url}/v1/chat/completions", _chat_request(args, image_blocks))

    print("arming profilers:")
    for idx, url in enumerate(encoder_urls):
        _arm(url, os.path.join(args.profiler_dir, f"encoder_{idx}"), args)
    _arm(args.lang_url, os.path.join(args.profiler_dir, "language"), args)

    def one(_):
        t = time.monotonic()
        _post(f"{args.lang_url}/v1/chat/completions", _chat_request(args, image_blocks))
        return time.monotonic() - t

    conc = max(1, args.concurrency)
    print(f"driving {args.n_requests} requests, concurrency={conc}")
    t0 = time.monotonic()
    if conc == 1:
        lats = [one(i) for i in range(args.n_requests)]
    else:
        with ThreadPoolExecutor(max_workers=conc) as ex:
            lats = list(ex.map(one, range(args.n_requests)))
    elapsed = time.monotonic() - t0

    lats_ms = sorted(x * 1000 for x in lats)

    def pctl(p):
        return lats_ms[min(len(lats_ms) - 1, int(p / 100 * len(lats_ms)))]

    print(
        f"\n{args.n_requests} requests, concurrency {conc}, in {elapsed:.2f}s "
        f"-> {args.n_requests / elapsed:.1f} req/s"
    )
    print(
        f"per-request latency ms: p50 {pctl(50):.0f}  p99 {pctl(99):.0f}  "
        f"mean {statistics.mean(lats_ms):.0f}  max {lats_ms[-1]:.0f}"
    )

    print("stopping profilers:")
    for url in encoder_urls:
        _stop(url)
    _stop(args.lang_url)

    print(f"\nTraces under {args.profiler_dir}:")
    print("  encoder_*/plugins/profile/.../*.trace.json.gz")
    print("  language/plugins/profile/.../*.trace.json.gz")
    print("\nView the full EPD chain:")
    print(
        "  - Drag all trace.json.gz files into https://ui.perfetto.dev/ (multi-trace, same clock)"
    )
    print(f"  - or: xprof --logdir={args.profiler_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
