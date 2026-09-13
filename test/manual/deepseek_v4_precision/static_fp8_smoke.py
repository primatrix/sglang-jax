"""Serve a static V4 checkpoint and compare short replies to the original-load baseline."""

import argparse
import json
import os
import pathlib
import signal
import subprocess
import sys
import time

import requests

BASELINE_EXP = "exp-lsez7lg3cf"
CASES = [
    ("What is 2 + 3? Answer with only the number.", [23]),
    ("What is the capital of France? Answer briefly.", [671, 6102, 294, 8760, 344, 11111, 16]),
    (
        "请用中文说一句简短的问候。",
        [30594, 1175, 70037, 17611, 804, 303, 77075, 5237, 8488, 14789, 1175],
    ),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.model / "encoding"))
    from encoding_dsv4 import encode_messages
    from sgl_jax.srt.hf_transformers_utils import get_tokenizer

    tokenizer = get_tokenizer(str(args.model), trust_remote_code=True)
    command = [
        sys.executable,
        "-m",
        "sgl_jax.launch_server",
        "--model-path",
        str(args.model),
        "--trust-remote-code",
        "--device",
        "tpu",
        "--tp-size",
        "8",
        "--ep-size",
        "1",
        "--moe-backend",
        "epmoe",
        "--dtype",
        "bfloat16",
        "--context-length",
        "2048",
        "--chunked-prefill-size",
        "256",
        "--max-prefill-tokens",
        "2048",
        "--max-total-tokens",
        "4096",
        "--max-running-requests",
        "2",
        "--page-size",
        "128",
        "--mem-fraction-static",
        "0.8",
        "--disable-overlap-schedule",
        "--disable-radix-cache",
        "--skip-server-warmup",
        "--watchdog-timeout",
        "1800",
        "--disable-precompile",
        "--host",
        "0.0.0.0",
        "--port",
        "30000",
    ]
    (args.out / "launch-command.json").write_text(json.dumps(command, indent=2))
    started = time.monotonic()
    process = subprocess.Popen(command, start_new_session=True)
    try:
        deadline = started + 7200
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"static server exited: {process.returncode}")
            try:
                response = requests.get("http://127.0.0.1:30000/get_server_info", timeout=3)
                if response.ok:
                    (args.out / "server-info.json").write_text(
                        json.dumps(response.json(), indent=2)
                    )
                    break
            except requests.RequestException:
                pass
            time.sleep(5)
        else:
            raise TimeoutError("static server readiness timeout")
        ready_seconds = time.monotonic() - started
        print("STATIC_SERVER_READY", ready_seconds, flush=True)
        results = []
        for question, expected_ids in CASES:
            prompt = encode_messages([{"role": "user", "content": question}], thinking_mode="chat")
            ids = tokenizer.encode(prompt, add_special_tokens=False)
            request_start = time.monotonic()
            response = requests.post(
                "http://127.0.0.1:30000/generate",
                json={
                    "input_ids": ids,
                    "sampling_params": {"temperature": 0, "max_new_tokens": 128},
                },
                timeout=1800,
            )
            response.raise_for_status()
            data = response.json()
            result = {
                "question": question,
                "input_ids": ids,
                "response": data,
                "expected_output_ids": expected_ids,
                "baseline_exp": BASELINE_EXP,
                "output_ids_match": data.get("output_ids") == expected_ids,
                "seconds": time.monotonic() - request_start,
            }
            results.append(result)
            (args.out / "generation-results.json").write_text(
                json.dumps(results, ensure_ascii=False, indent=2)
            )
            print("STATIC_GENERATION", json.dumps(result, ensure_ascii=False), flush=True)
        assert all(
            x["output_ids_match"] and x["response"]["meta_info"]["finish_reason"]["type"] == "stop"
            for x in results
        ), "Static replies differ from original-load baseline"
        (args.out / "smoke-summary.json").write_text(
            json.dumps(
                {
                    "baseline_exp": BASELINE_EXP,
                    "requests": 3,
                    "all_token_ids_match": True,
                    "server_ready_seconds": ready_seconds,
                },
                indent=2,
            )
        )
        print("STATIC_FP8_SMOKE_PASS", flush=True)
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()


if __name__ == "__main__":
    main()
