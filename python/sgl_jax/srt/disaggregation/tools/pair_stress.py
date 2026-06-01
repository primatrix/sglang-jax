"""Normal-path PD pair stress benchmark.

Unlike `tools.stress`, this driver matches the current PD topology:
every logical request is fanned out to both P and D, while only D's
completion is treated as the service result. P is validated against the
prefill-only contract (`length=0`, empty completion).
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

from sgl_jax.srt.disaggregation.tools.e2e import _common as C


PROMPTS = [
    "Hello, my name is",
    "The capital of France is",
    "Once upon a time, in a small village near the sea, lived a wise",
    "2 + 2 =",
]


def _fire_one(
    topo: C.Topology,
    room: int,
    rid: str,
    prompt: str,
    max_new: int,
    *,
    use_explicit_endpoints: bool,
):
    t0 = time.perf_counter()
    try:
        rsp = C.fire_pd_pair(
            topo,
            rid=rid,
            prompt=prompt,
            bootstrap_room=room,
            max_new_tokens=max_new,
            p_url=topo.first_p if use_explicit_endpoints else None,
            d_url=topo.first_d if use_explicit_endpoints else None,
            timeout=120.0,
        )
        p_meta = rsp["P"].get("meta_info", {})
        d_meta = rsp["D"].get("meta_info", {})
        p_finish = p_meta.get("finish_reason") or {}
        d_finish = d_meta.get("finish_reason") or {}
        p_ok = (
            rsp["P"].get("text") == ""
            and (rsp["P"].get("output_ids") or []) == []
            and p_meta.get("completion_tokens") == 0
            and p_finish.get("type") == "length"
            and p_finish.get("length") == 0
        )
        d_ok = (
            bool(rsp["D"].get("text"))
            and isinstance(rsp["D"].get("output_ids"), list)
            and len(rsp["D"]["output_ids"]) > 0
            and int(d_meta.get("completion_tokens") or 0) > 0
            and d_finish.get("type") in {"length", "stop", "abort"}
        )
        ok = p_ok and d_ok
        return {
            "ok": ok,
            "elapsed": time.perf_counter() - t0,
            "error": None if ok else "contract",
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "elapsed": time.perf_counter() - t0,
            "error": repr(exc),
        }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    C.add_topology_args(p)
    p.add_argument("--qps", type=float, default=1.0)
    p.add_argument("--duration-seconds", type=float, default=60.0)
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument("--concurrency", type=int, default=16)
    args = p.parse_args()

    topo = C.parse_topology(args)
    use_explicit_endpoints = len(topo.p_urls) == 1 and len(topo.d_urls) == 1
    interval = 1.0 / max(args.qps, 1e-6)
    deadline = time.perf_counter() + args.duration_seconds

    n_total = 0
    n_ok = 0
    sum_ok = 0.0
    samples: collections.deque[float] = collections.deque(maxlen=200_000)
    error_samples: List[str] = []
    lock = threading.Lock()
    pending: collections.deque = collections.deque()

    def _drain(force_all: bool = False):
        nonlocal n_total, n_ok, sum_ok
        while pending and (force_all or pending[0].done()):
            fut = pending.popleft()
            row = fut.result()
            with lock:
                n_total += 1
                if row["ok"]:
                    n_ok += 1
                    sum_ok += row["elapsed"]
                    samples.append(row["elapsed"])
                elif row["error"] and len(error_samples) < 20:
                    error_samples.append(str(row["error"]))

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        next_t = time.perf_counter()
        i = 0
        while time.perf_counter() < deadline:
            sleep_for = next_t - time.perf_counter()
            if sleep_for > 0:
                time.sleep(sleep_for)
            rid = f"pair-stress-{i}"
            room = random.randint(0, 2**31 - 1)
            prompt = random.choice(PROMPTS)
            pending.append(
                pool.submit(
                    _fire_one,
                    topo,
                    room,
                    rid,
                    prompt,
                    args.max_new_tokens,
                    use_explicit_endpoints=use_explicit_endpoints,
                )
            )
            i += 1
            next_t += interval
            _drain(force_all=False)
        _drain(force_all=True)

    def _pct(pct: int):
        if not samples:
            return None
        s = sorted(samples)
        return s[int(pct / 100.0 * (len(s) - 1))]

    summary = {
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
        "error_samples": error_samples,
    }
    print(json.dumps(summary, indent=2))
    out_path = args.out or "pair_stress.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
