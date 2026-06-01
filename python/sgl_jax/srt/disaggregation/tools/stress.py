"""Stage 4 H-F: PD stress harness.

Drives a configurable QPS against a deployed router for a fixed
duration, logs P50/P95/P99 latency + throughput + error rate. Saves
a JSON summary so the runbook can paste it into the perf baseline
doc.

Usage:
    python -m sgl_jax.srt.disaggregation.tools.stress \\
        --router http://router:8001 \\
        --qps 100 --duration-seconds 600 \\
        --prompt-tokens 1024 --max-new-tokens 32 \\
        --out stress.json

This script intentionally uses ``httpx`` directly rather than the
sgl_jax client SDK so it can run from a thin operator pod without
JAX installed.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

import httpx


def _fire_one(client: httpx.Client, router: str, prompt: str,
              max_new: int) -> Dict[str, object]:
    t0 = time.perf_counter()
    try:
        r = client.post(
            f"{router}/generate",
            json={
                "text": prompt,
                "sampling_params": {
                    "max_new_tokens": max_new,
                    "temperature": 0.0,
                },
            },
            timeout=60.0,
        )
        ok = 200 <= r.status_code < 300
    except Exception:  # noqa: BLE001
        ok = False
    return {"ok": ok, "elapsed": time.perf_counter() - t0}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--router", required=True)
    p.add_argument("--qps", type=float, default=10.0)
    p.add_argument("--duration-seconds", type=float, default=60.0)
    p.add_argument("--prompt-tokens", type=int, default=512)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--concurrency", type=int, default=64)
    p.add_argument("--out", default="stress.json")
    args = p.parse_args()

    prompt = "Hello " * args.prompt_tokens
    interval = 1.0 / max(args.qps, 1e-6)
    deadline = time.perf_counter() + args.duration_seconds

    # Stage 4 review I4: stream completions through a bounded queue so
    # 10k qps × 1h doesn't accumulate 36M futures in RAM. We keep
    # rolling aggregates (sum / count / sorted-sample) instead.
    import collections
    import heapq

    n_total = 0
    n_ok = 0
    sum_ok = 0.0
    # Reservoir of up-to-N elapsed times for percentile estimation;
    # bounded so memory stays O(N) regardless of duration. 200k is
    # accurate enough for P50/P95/P99 reporting.
    SAMPLE_CAP = 200_000
    samples: collections.deque = collections.deque(maxlen=SAMPLE_CAP)

    results_lock = threading.Lock()
    pending: collections.deque = collections.deque()

    def _drain(force_all: bool = False) -> None:
        # Drain completed futures from the head of the deque.
        nonlocal n_total, n_ok, sum_ok
        while pending and (force_all or pending[0].done()):
            f = pending.popleft()
            r = f.result()
            with results_lock:
                n_total += 1
                if r["ok"]:
                    n_ok += 1
                    sum_ok += r["elapsed"]
                    samples.append(r["elapsed"])

    client = httpx.Client()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        next_t = time.perf_counter()
        while time.perf_counter() < deadline:
            sleep_for = next_t - time.perf_counter()
            if sleep_for > 0:
                time.sleep(sleep_for)
            pending.append(
                pool.submit(
                    _fire_one, client, args.router, prompt,
                    args.max_new_tokens,
                )
            )
            next_t += interval
            _drain(force_all=False)

        _drain(force_all=True)

    def _pct(p):
        if not samples:
            return None
        s = sorted(samples)
        return s[int(p / 100.0 * (len(s) - 1))]

    summary = {
        "router": args.router,
        "qps_target": args.qps,
        "duration_seconds": args.duration_seconds,
        "concurrency": args.concurrency,
        "total_requests": n_total,
        "success_requests": n_ok,
        "error_rate": (n_total - n_ok) / max(n_total, 1),
        "p50_seconds": _pct(50),
        "p95_seconds": _pct(95),
        "p99_seconds": _pct(99),
        "mean_seconds": (sum_ok / n_ok) if n_ok else None,
        "qps_achieved": n_ok / args.duration_seconds,
        "sampled_for_percentiles": len(samples),
    }
    print(json.dumps(summary, indent=2))
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
