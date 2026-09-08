#!/usr/bin/env python3
"""Run the fixed Qwen3-VL-8B N1 baseline, without selecting SLO thresholds.

Run inside the prepared TPU environment. Optionally launch the N1 server;
otherwise benchmark an existing N1 endpoint. Outputs go to a new directory.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


def percentile(values, q):
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    index = (len(values) - 1) * q
    lo = int(index)
    hi = min(lo + 1, len(values) - 1)
    return (values[lo] + (values[hi] - values[lo]) * (index - lo)) * 1000


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--deployment", choices=["N1", "E1", "E2"], default="N1")
    p.add_argument("--request-dir", type=Path, help="Directory of saved image requests to replay")
    p.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=30000)
    p.add_argument("--launch-server", action="store_true")
    p.add_argument("--num-prompts", type=int, default=100)
    p.add_argument(
        "--input-len",
        type=int,
        default=1024,
        help="Text input tokens; excludes vision tokens",
    )
    p.add_argument("--image-resolution", default="640x640")
    p.add_argument("--output-len", type=int, default=128)
    p.add_argument("--concurrency", type=int, nargs="+", default=[2, 4, 8, 16, 32])
    p.add_argument("--groups", choices=["A", "C"], nargs="+", default=["A", "C"])
    p.add_argument(
        "--skip-points",
        nargs="*",
        default=[],
        help="Completed group:concurrency points, e.g. A:2 A:4",
    )
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    if (
        min(
            args.num_prompts,
            args.repeats,
            args.input_len,
            args.output_len,
            *args.concurrency,
        )
        < 1
    ):
        p.error("request counts, repeats, and concurrency must be positive")
    if args.launch_server and args.host != "127.0.0.1":
        p.error("--launch-server requires --host 127.0.0.1")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    server_cmd = [
        sys.executable,
        "-u",
        "-m",
        "sgl_jax.launch_server",
        "--model-path",
        args.model,
        "--trust-remote-code",
        "--device",
        "tpu",
        "--tp-size",
        "8",
        "--dp-size",
        "4",
        "--vision-encoder-parallel",
        "dp",
        "--dtype",
        "bfloat16",
        "--kv-cache-dtype",
        "bf16",
        "--context-length",
        "16384",
        "--max-seq-len",
        "16384",
        "--max-running-requests",
        "96",
        "--max-prefill-tokens",
        "16384",
        "--chunked-prefill-size",
        "4096",
        "--mem-fraction-static",
        "0.8",
        "--page-size",
        "128",
        "--disable-radix-cache",
        "--mm-processor-worker-num",
        "2",
        "--random-seed",
        "42",
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]
    encoder_cmd = None
    if args.deployment != "N1":
        encoder_devices = 4 if args.deployment == "E1" else 2
        pd_devices = 8 - encoder_devices
        server_cmd[server_cmd.index("--tp-size") + 1] = str(pd_devices)
        server_cmd[server_cmd.index("--dp-size") + 1] = str(pd_devices // 2)
        server_cmd += [
            "--language-only",
            "--encoder-urls",
            "http://127.0.0.1:31001",
            "--device-indexes",
            *map(str, range(encoder_devices, 8)),
        ]
        encoder_cmd = [
            sys.executable,
            "-u",
            "-m",
            "sgl_jax.launch_server",
            "--model-path",
            args.model,
            "--trust-remote-code",
            "--device",
            "tpu",
            "--encoder-only",
            "--tp-size",
            str(encoder_devices),
            "--dp-size",
            str(encoder_devices),
            "--vision-encoder-parallel",
            "dp",
            "--dtype",
            "bfloat16",
            "--disable-radix-cache",
            "--mm-processor-worker-num",
            "2",
            "--host",
            "127.0.0.1",
            "--port",
            "31001",
            "--device-indexes",
            *map(str, range(encoder_devices)),
        ]
    metadata = vars(args) | {
        "output_dir": str(args.output_dir),
        "server_command": server_cmd,
        "encoder_command": encoder_cmd,
        "request_dir": str(args.request_dir) if args.request_dir else None,
    }
    metadata["revision"] = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    (args.output_dir / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")

    def run_bench(group, concurrency, count, stem, seed):
        output = args.output_dir / f"{stem}.jsonl"
        cmd = [
            sys.executable,
            "-u",
            "-m",
            "sgl_jax.bench_serving",
            "--backend",
            "sglang-oai-chat",
            "--host",
            args.host,
            "--port",
            str(args.port),
            "--model",
            args.model,
            "--tokenizer",
            args.model,
            "--dataset-name",
            "image",
            "--image-count",
            "1" if group == "A" else "4",
            "--image-resolution",
            args.image_resolution,
            "--image-format",
            "jpeg",
            "--image-content",
            "random",
            "--random-input-len",
            str(args.input_len),
            "--random-output-len",
            str(args.output_len),
            "--random-range-ratio",
            "1.0",
            "--request-rate",
            "inf",
            "--max-concurrency",
            str(concurrency),
            "--num-prompts",
            str(count),
            "--warmup-requests",
            "1",
            "--extra-request-body",
            '{"stream_options":{"include_usage":true}}',
            "--seed",
            str(seed),
            "--output-details",
            "--disable-tqdm",
            "--output-file",
            str(output),
        ]
        if not stem.startswith("prewarm_"):
            cmd += [
                "--image-request-file",
                str((args.request_dir or args.output_dir) / f"requests_{group}_seed{seed}.json"),
            ]
        print(json.dumps({"run": stem, "command": cmd}), flush=True)
        if args.dry_run:
            return None
        # Dataset generation is CPU work; this client must not initialize TPU.
        env = os.environ | {"JAX_PLATFORMS": "cpu"}
        with (args.output_dir / f"{stem}.log").open("w") as log:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, check=True)
        result = json.loads(output.read_text().splitlines()[-1])
        if result["completed"] != count:
            raise RuntimeError(f"{stem}: only {result['completed']}/{count} requests completed")
        if any(length != args.output_len for length in result["output_lens"]):
            raise RuntimeError(
                f"{stem}: output lengths differ from the {args.output_len}-token budget"
            )
        if (
            not all(
                u and u.get("completion_tokens") == args.output_len
                for u in result.get("usages", [])
            )
            or len(result.get("usages", [])) != count
        ):
            raise RuntimeError(f"{stem}: missing or mismatched server token usage")
        if not all(key in result for key in ("latencies", "tpots", "successes")):
            raise RuntimeError(
                "benchmark lacks request-level timing additions; deploy the matching code"
            )
        return result

    server = None
    log = None
    encoder = None
    encoder_log = None
    try:
        if args.launch_server:
            print(json.dumps({"server_command": server_cmd}), flush=True)
            if not args.dry_run:
                log = (args.output_dir / "server.log").open("w")
                server_env = os.environ.copy()
                if encoder_cmd:
                    server_env["ALLOW_MULTIPLE_LIBTPU_LOAD"] = "1"
                    encoder_log = (args.output_dir / "encoder.log").open("w")
                    print(json.dumps({"encoder_command": encoder_cmd}), flush=True)
                    encoder = subprocess.Popen(
                        encoder_cmd, stdout=encoder_log, stderr=subprocess.STDOUT, env=server_env
                    )
                    deadline = time.monotonic() + 1800
                    while True:
                        if encoder.poll() is not None:
                            raise RuntimeError("encoder exited before readiness")
                        try:
                            with urllib.request.urlopen("http://127.0.0.1:31001/health", timeout=5):
                                break
                        except OSError:
                            if time.monotonic() >= deadline:
                                raise RuntimeError("encoder readiness timed out")
                            time.sleep(5)
                server = subprocess.Popen(
                    server_cmd, stdout=log, stderr=subprocess.STDOUT, env=server_env
                )
        if not args.dry_run:
            deadline = time.monotonic() + (1800 if server else 30)
            while True:
                if server and server.poll() is not None:
                    raise RuntimeError("server exited before readiness; inspect server.log")
                try:
                    with urllib.request.urlopen(
                        f"http://{args.host}:{args.port}/get_server_info", timeout=5
                    ) as response:
                        info = response.read()
                    (args.output_dir / "server-info.json").write_bytes(info)
                    break
                except (OSError, ValueError):
                    if time.monotonic() >= deadline:
                        raise RuntimeError("server readiness timed out")
                    time.sleep(5)
        for group in args.groups:
            for concurrency in args.concurrency:
                if f"{group}:{concurrency}" in args.skip_points:
                    continue
                # Warm each shape/concurrency separately; preserve logs so that
                # remaining compilation in measured runs can be detected.
                run_bench(
                    group,
                    concurrency,
                    max(16, 2 * concurrency),
                    f"prewarm_{group}_c{concurrency}",
                    args.seed,
                )
                for repeat in range(args.repeats):
                    stem = f"{args.deployment}_{group}_c{concurrency}_r{repeat + 1}_seed{args.seed + repeat}"
                    result = run_bench(
                        group, concurrency, args.num_prompts, stem, args.seed + repeat
                    )
                    if result is None:
                        continue
                    summary = {
                        "run": stem,
                        "deployment": args.deployment,
                        "group": group,
                        "concurrency": concurrency,
                        "image_resolution": args.image_resolution,
                        "image_count": 1 if group == "A" else 4,
                        "text_input_len": args.input_len,
                        "output_len": args.output_len,
                        "submitted": args.num_prompts,
                        "completed": result["completed"],
                        "failed": args.num_prompts - result["completed"],
                        "failure_rate": 1 - result["completed"] / args.num_prompts,
                        "duration_s": result["duration"],
                        "request_throughput": result["request_throughput"],
                        "output_throughput": result["output_throughput"],
                        "input_vision_tokens": result["total_input_vision_tokens"],
                        "input_text_tokens": result["total_input_text_tokens"],
                        "server_prompt_tokens": sum(u["prompt_tokens"] for u in result["usages"]),
                    }
                    for field, prefix in [
                        ("ttfts", "ttft"),
                        ("tpots", "tpot"),
                        ("latencies", "e2e"),
                    ]:
                        values = [
                            v
                            for v, ok in zip(result[field], result["successes"])
                            if ok and v is not None
                        ]
                        summary[f"samples_{prefix}"] = len(values)
                        for name, value in (
                            ("mean", sum(values) / len(values) if values else None),
                            ("min", min(values) if values else None),
                            ("max", max(values) if values else None),
                        ):
                            summary[f"{name}_{prefix}_ms"] = (
                                value * 1000 if value is not None else None
                            )
                        for q in (50, 90, 95, 99):
                            summary[f"p{q}_{prefix}_ms"] = percentile(values, q / 100)
                    with (args.output_dir / "summary.jsonl").open("a") as output:
                        output.write(json.dumps(summary) + "\n")
                    print("BASELINE_RESULT " + json.dumps(summary), flush=True)
    finally:
        if sys.exc_info()[0] is not None:
            # Keep diagnostics visible in Falcon's durable workload log even
            # when a failed artifact cannot be enrolled for analysis.
            if log:
                log.flush()
            for path in sorted(args.output_dir.glob("*.log")):
                print(f"Failure log: {path.name}", flush=True)
                print(path.read_text(errors="replace")[-16000:], flush=True)
        if server and server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
        if encoder and encoder.poll() is None:
            encoder.terminate()
            try:
                encoder.wait(timeout=30)
            except subprocess.TimeoutExpired:
                encoder.kill()
                encoder.wait()
        if encoder_log:
            encoder_log.close()
        if log:
            log.close()


if __name__ == "__main__":
    main()
