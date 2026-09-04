#!/usr/bin/env python3
"""Run a fast EPD configuration matrix on one Falcon TPU allocation."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import SimpleNamespace

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from scripts.disaggregation.profile_epd_cpu_sim import (
    _BENCHMARK_KEYS,
    _run_bench_serving,
    _summarize_encoder_pipeline,
    _summarize_encoder_queue,
)
from sgl_jax.srt.disaggregation.encoder.metrics import (
    summarize_raiden_transfer_inflight,
)


@dataclass(frozen=True, slots=True)
class Variant:
    name: str
    pool_size: int = 256
    encoder_batch_size: int = 16
    inflight_batches: int = 1
    io_workers: int = 4
    processor_workers: int = 4
    channels: int = 4
    max_prefill_tokens: int = 8192
    encoder_cpu_threads: int | None = None
    language_cpu_threads: int | None = None
    dispatch_orjson: bool = True


def _variants() -> list[Variant]:
    base = Variant("baseline")
    return [
        base,
        replace(base, name="pool-64", pool_size=64),
        replace(base, name="pool-128", pool_size=128),
        replace(base, name="batch-8", encoder_batch_size=8),
        replace(base, name="batch-32", encoder_batch_size=32),
        replace(base, name="inflight-2", inflight_batches=2),
        replace(base, name="inflight-4", inflight_batches=4),
        replace(base, name="io-workers-2", io_workers=2),
        replace(base, name="io-workers-8", io_workers=8),
        replace(base, name="processor-workers-2", processor_workers=2),
        replace(base, name="processor-workers-8", processor_workers=8),
        replace(base, name="channels-2", channels=2),
        replace(base, name="channels-8", channels=8),
        replace(base, name="prefill-4096", max_prefill_tokens=4096),
        replace(base, name="prefill-16384", max_prefill_tokens=16384),
        replace(base, name="cpu-threads-1", encoder_cpu_threads=1, language_cpu_threads=1),
        replace(base, name="cpu-threads-2", encoder_cpu_threads=2, language_cpu_threads=2),
        replace(base, name="cpu-threads-4", encoder_cpu_threads=4, language_cpu_threads=4),
        replace(base, name="cpu-threads-8", encoder_cpu_threads=8, language_cpu_threads=8),
        replace(base, name="cpu-threads-16", encoder_cpu_threads=16, language_cpu_threads=16),
        replace(base, name="encoder-cpu-threads-2", encoder_cpu_threads=2),
        replace(base, name="encoder-cpu-threads-4", encoder_cpu_threads=4),
        replace(base, name="encoder-cpu-threads-8", encoder_cpu_threads=8),
        replace(base, name="language-cpu-threads-4", language_cpu_threads=4),
        replace(base, name="encoder-cpu4-json", encoder_cpu_threads=4, dispatch_orjson=False),
        replace(base, name="encoder-cpu4-orjson", encoder_cpu_threads=4, dispatch_orjson=True),
        Variant(
            "wide-pipeline",
            pool_size=128,
            encoder_batch_size=32,
            inflight_batches=2,
            io_workers=8,
            processor_workers=8,
            channels=8,
            max_prefill_tokens=16384,
        ),
    ]


def _wait_for_url(url: str, process: subprocess.Popen, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited with status {process.returncode}: {url}")
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status < 400:
                    return
        except Exception:
            time.sleep(2)
    raise TimeoutError(f"server did not become ready: {url}")


def _signal_process_group(process: subprocess.Popen, sig: int) -> None:
    if process.poll() is None:
        os.killpg(process.pid, sig)


def _stop_many(*processes: subprocess.Popen | None) -> None:
    """Stop all server trees concurrently so teardown is paid only once."""
    running = [process for process in processes if process is not None and process.poll() is None]
    for process in running:
        _signal_process_group(process, signal.SIGTERM)
    deadline = time.monotonic() + 10
    while running and time.monotonic() < deadline:
        running = [process for process in running if process.poll() is None]
        if running:
            time.sleep(0.1)
    for process in running:
        _signal_process_group(process, signal.SIGKILL)
    for process in running:
        process.wait(timeout=5)


def _server_env(
    code_root: Path,
    cache_dir: Path,
    chips: str,
    cpu_threads: int | None,
    dispatch_orjson: bool | None = None,
) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "TPU_CHIPS_PER_PROCESS_BOUNDS": "1,2,1",
            "TPU_PROCESS_BOUNDS": "1,1,1",
            "TPU_VISIBLE_CHIPS": chips,
            "ALLOW_MULTIPLE_LIBTPU_LOAD": "true",
            "JAX_COMPILATION_CACHE_DIR": str(cache_dir),
            "PYTHONPATH": f"{code_root / 'python'}:{code_root}",
        }
    )
    if cpu_threads is not None:
        env["OMP_NUM_THREADS"] = str(cpu_threads)
        env["MKL_NUM_THREADS"] = str(cpu_threads)
    if dispatch_orjson is not None:
        env["SGLANG_ENCODER_DISPATCH_ORJSON"] = "1" if dispatch_orjson else "0"
    return env


def _common_server_args(args: argparse.Namespace, variant: Variant) -> list[str]:
    return [
        sys.executable,
        "-u",
        "-m",
        "sgl_jax.launch_server",
        "--model-path",
        args.model_path,
        "--trust-remote-code",
        "--skip-server-warmup",
        "--device",
        "tpu",
        "--tp-size",
        "4",
        "--dp-size",
        "2",
        "--dtype",
        "bfloat16",
        "--vision-encoder-parallel",
        "dp",
        "--mm-io-worker-num",
        str(variant.io_workers),
        "--mm-processor-worker-num",
        str(variant.processor_workers),
        "--encoder-transfer-backend",
        "raiden",
        "--encoder-transfer-pool-size",
        str(variant.pool_size),
        "--encoder-control-timeout-seconds",
        "300",
        "--encoder-request-timeout-seconds",
        "900",
        "--disaggregation-channel-number",
        str(variant.channels),
        "--disaggregation-host-ip",
        args.host_ip,
        "--enable-request-time-stats-logging",
        "--random-seed",
        "0",
        "--download-dir",
        args.download_dir,
    ]


def _start_servers(
    args: argparse.Namespace,
    variant: Variant,
    variant_dir: Path,
) -> tuple[subprocess.Popen, subprocess.Popen, object, object]:
    common = _common_server_args(args, variant)
    encoder_log = (variant_dir / "encoder.log").open("w")
    language_log = (variant_dir / "language.log").open("w")
    encoder = subprocess.Popen(
        [
            *common,
            "--encoder-only",
            "--encoder-max-batch-size",
            str(variant.encoder_batch_size),
            "--encoder-max-inflight-batches",
            str(variant.inflight_batches),
            "--encoder-batch-coalesce-ms",
            "0",
            "--host",
            "0.0.0.0",
            "--port",
            "30001",
        ],
        cwd=args.code_root,
        env=_server_env(
            args.code_root,
            args.cache_root / "encoder",
            "0,1",
            variant.encoder_cpu_threads,
        ),
        stdout=encoder_log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    language = subprocess.Popen(
        [
            *common,
            "--language-only",
            "--encoder-urls",
            "http://127.0.0.1:30001",
            "--kv-cache-dtype",
            "bf16",
            "--context-length",
            "2048",
            "--max-seq-len",
            "2048",
            "--max-running-requests",
            "512",
            "--max-prefill-tokens",
            str(variant.max_prefill_tokens),
            "--chunked-prefill-size",
            "4096",
            "--mem-fraction-static",
            "0.9",
            "--page-size",
            "128",
            "--disable-radix-cache",
            "--dp-schedule-policy",
            "min_running_queue",
            "--encoder-receiver-background-progress",
            "--host",
            "0.0.0.0",
            "--port",
            "30000",
        ],
        cwd=args.code_root,
        env=_server_env(
            args.code_root,
            args.cache_root / "language",
            "2,3",
            variant.language_cpu_threads,
            variant.dispatch_orjson,
        ),
        stdout=language_log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        _wait_for_url("http://127.0.0.1:30001/health", encoder, args.startup_timeout)
        _wait_for_url(
            "http://127.0.0.1:30000/get_server_info",
            language,
            args.startup_timeout,
        )
        return encoder, language, encoder_log, language_log
    except BaseException:
        _stop_many(language, encoder)
        encoder_log.close()
        language_log.close()
        raise


def _benchmark_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        lang_url="http://127.0.0.1:30000",
        model_path=args.model_path,
        random_input_len=128,
        max_tokens=1,
        images_per_request=1,
        image_resolution="512x512",
        image_format="jpeg",
        image_content="random",
        seed=0,
    )


def _run_variant(
    args: argparse.Namespace,
    variant: Variant,
    index: int,
) -> dict:
    variant_dir = args.output_dir / f"{index:02d}-{variant.name}"
    variant_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    encoder = language = None
    encoder_log = language_log = None
    try:
        encoder, language, encoder_log, language_log = _start_servers(
            args,
            variant,
            variant_dir,
        )
        startup_s = time.monotonic() - started
        bench_args = _benchmark_args(args)
        prewarm_prompts = args.first_prewarm if index == 0 else args.prewarm
        _run_bench_serving(
            bench_args,
            num_prompts=prewarm_prompts,
            concurrency=args.concurrency,
            warmup_requests=1,
            output_file=variant_dir / "prewarm.jsonl",
        )
        formal_start_ns = time.time_ns()
        result = _run_bench_serving(
            bench_args,
            num_prompts=args.prompts,
            concurrency=args.concurrency,
            warmup_requests=0,
            output_file=variant_dir / "benchmark.jsonl",
        )
        formal_end_ns = time.time_ns()
        if result.get("completed") != args.prompts:
            raise RuntimeError(f"incomplete benchmark: {result.get('completed')}/{args.prompts}")
        encoder_log.flush()
        language_log.flush()
        queue = _summarize_encoder_queue(
            [variant_dir / "encoder.log"],
            start_ns=formal_start_ns,
            end_ns=formal_end_ns,
        )
        pipeline = _summarize_encoder_pipeline(
            variant_dir / "language.log",
            start_ns=formal_start_ns,
            end_ns=formal_end_ns,
        )
        if queue["n"] != args.prompts or pipeline["n"] != args.prompts:
            raise RuntimeError(f"formal coverage queue={queue['n']} pipeline={pipeline['n']}")
        summary = {
            "schema_version": 1,
            "status": "SUCCEEDED",
            "index": index,
            "variant": asdict(variant),
            "source_commit": args.source_commit,
            "startup_s": startup_s,
            "benchmark": {key: result.get(key) for key in _BENCHMARK_KEYS},
            "encoder_queue": queue,
            "pipeline": pipeline,
            "transfer_inflight": summarize_raiden_transfer_inflight(
                [variant_dir / "encoder.log"],
                start_ns=formal_start_ns,
                end_ns=formal_end_ns,
            ),
        }
    except BaseException as exc:
        summary = {
            "schema_version": 1,
            "status": "FAILED",
            "index": index,
            "variant": asdict(variant),
            "source_commit": args.source_commit,
            "error": f"{type(exc).__name__}: {exc}",
        }
    finally:
        _stop_many(language, encoder)
        if encoder_log is not None:
            encoder_log.close()
        if language_log is not None:
            language_log.close()
    summary["wall_s"] = time.monotonic() - started
    (variant_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, default=Path.cwd())
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--download-dir", required=True)
    parser.add_argument("--host-ip", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--prompts", type=int, default=384)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--prewarm", type=int, default=32)
    parser.add_argument("--first-prewarm", type=int, default=128)
    parser.add_argument("--startup-timeout", type=float, default=900)
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_root.mkdir(parents=True, exist_ok=True)

    matrix_path = args.output_dir / "matrix.jsonl"
    variants = _variants()
    if args.only:
        selected = set(args.only)
        variants = [variant for variant in variants if variant.name in selected]
        missing = selected - {variant.name for variant in variants}
        if missing:
            parser.error(f"unknown variants: {sorted(missing)}")
    succeeded = 0
    with matrix_path.open("w") as matrix:
        for index, variant in enumerate(variants):
            print(f"EPD-MATRIX-START index={index} variant={variant.name}", flush=True)
            summary = _run_variant(args, variant, index)
            matrix.write(json.dumps(summary) + "\n")
            matrix.flush()
            print(
                f"EPD-MATRIX-DONE index={index} variant={variant.name} "
                f"status={summary['status']} wall_s={summary['wall_s']:.1f}",
                flush=True,
            )
            succeeded += summary["status"] == "SUCCEEDED"

    aggregate = {
        "schema_version": 1,
        "source_commit": args.source_commit,
        "attempted": len(variants),
        "succeeded": succeeded,
        "failed": len(variants) - succeeded,
    }
    (args.output_dir / "matrix-summary.json").write_text(json.dumps(aggregate, indent=2))
    print(json.dumps(aggregate), flush=True)
    required = min(10, len(variants))
    return 0 if succeeded >= required else 1


if __name__ == "__main__":
    raise SystemExit(main())
