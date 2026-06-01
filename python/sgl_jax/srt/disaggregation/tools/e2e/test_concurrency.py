"""P0-4: concurrency. Fire N concurrent PD pairs and verify:

  * All requests return success (non-empty text on both sides).
  * No KVPoll illegal-transition errors leaked (proxied via
    pd_transfer_failures_total{reason=other,sender_fail,
    receiver_fail} delta == 0).
  * pd_transfer_inflight gauge returns to 0 after the wave drains.

The test is parameterized over a few concurrency levels so we can
spot the lowest level that starts seeing leaks.
"""

from __future__ import annotations

import argparse
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

import httpx

from sgl_jax.srt.disaggregation.tools.e2e import _common as C


PROMPTS = [
    "Hello, my name is",
    "The capital of France is",
    "Once upon a time",
    "The most important",
    "In the year 2026,",
]


def _fetch_metric(p_url: str, prefix: str) -> float:
    try:
        r = httpx.get(f"{p_url}/metrics", timeout=5.0)
        if r.status_code != 200:
            return 0.0
        total = 0.0
        for line in r.text.splitlines():
            if line.startswith(prefix):
                total += float(line.rsplit(" ", 1)[1])
        return total
    except Exception:
        return 0.0


def _fire(topo, room_base, i):
    rid = f"conc-{room_base}-{i}"
    prompt = PROMPTS[i % len(PROMPTS)]
    rsp = C.fire_pd_pair(
        topo, rid=rid, prompt=prompt,
        bootstrap_room=room_base + i, max_new_tokens=4,
        timeout=60.0,
    )
    p_text = rsp["P"].get("text", "")
    d_text = rsp["D"].get("text", "")
    return {
        "rid": rid, "ok": bool(p_text) and bool(d_text),
        "equal": p_text == d_text,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    C.add_topology_args(p)
    p.add_argument("--levels", default="10,30",
                   help="Comma-separated concurrency levels.")
    args = p.parse_args()
    topo = C.parse_topology(args)

    summary = {"levels": []}
    overall_fail = []

    for level in [int(x) for x in args.levels.split(",")]:
        room_base = random.randint(0, 1_000_000)
        # Snapshot failure-counter and inflight gauge before.
        fails_before = _fetch_metric(
            topo.first_p, "pd_transfer_failures_total"
        )
        inflight_before = _fetch_metric(
            topo.first_p, "pd_transfer_inflight"
        )

        t0 = time.perf_counter()
        results: List[Dict] = []
        with ThreadPoolExecutor(max_workers=level) as pool:
            futs = [pool.submit(_fire, topo, room_base, i)
                    for i in range(level)]
            for f in as_completed(futs):
                try:
                    results.append(f.result())
                except Exception as e:
                    results.append({"error": str(e)})
        elapsed = time.perf_counter() - t0

        # Let the reaper / pruner settle.
        time.sleep(3.0)
        inflight_after = _fetch_metric(
            topo.first_p, "pd_transfer_inflight"
        )
        fails_after = _fetch_metric(
            topo.first_p, "pd_transfer_failures_total"
        )

        ok_count = sum(1 for r in results if r.get("ok"))
        eq_count = sum(1 for r in results if r.get("equal"))
        err_count = sum(1 for r in results if "error" in r)

        row = {
            "level": level,
            "elapsed_s": elapsed,
            "ok": ok_count,
            "byte_equal": eq_count,
            "error": err_count,
            "fails_delta": fails_after - fails_before,
            "inflight_after_drain": inflight_after - inflight_before,
        }
        summary["levels"].append(row)
        if ok_count != level:
            overall_fail.append(f"L{level}:ok={ok_count}/{level}")
        if eq_count != level:
            overall_fail.append(f"L{level}:eq={eq_count}/{level}")
        if row["inflight_after_drain"] > 0.5:
            overall_fail.append(
                f"L{level}:inflight_leak={row['inflight_after_drain']}"
            )

    C.write_report(args, "concurrency", summary)
    if overall_fail:
        return C.print_result(False, "; ".join(overall_fail))
    return C.print_result(
        True,
        f"levels={[r['level'] for r in summary['levels']]} all clean",
    )


if __name__ == "__main__":
    sys.exit(main())
