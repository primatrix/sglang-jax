"""Normal-path soak for current PD phase-1 semantics.

This repeatedly fires the same logical requests to one P and one D and
checks the current production-path contract:

  * P returns an empty completion (`length=0`).
  * D returns the real completion.
  * No request errors occur.
  * `pd_transfer_failures_total` does not increase.
  * `pd_transfer_inflight` returns to its starting level after the wave.
"""

from __future__ import annotations

import argparse
import sys
import time

import httpx

from sgl_jax.srt.disaggregation.tools.e2e import _common as C


CASES = [
    (1251, 12, "soak-1251"),
    (1637, 17, "soak-1637"),
    (2501, 25, "soak-2501"),
    (3900, 39, "soak-3900"),
]


def _fetch_metric_sum(url: str, prefix: str) -> float | None:
    try:
        r = httpx.get(f"{url}/metrics", timeout=5.0)
        if r.status_code != 200:
            return None
        total = 0.0
        found = False
        for line in r.text.splitlines():
            if line.startswith(prefix):
                total += float(line.rsplit(" ", 1)[1])
                found = True
        return total if found else None
    except Exception:
        return None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    C.add_topology_args(p)
    p.add_argument("--rounds", type=int, default=10)
    p.add_argument("--max-new-tokens", type=int, default=8)
    args = p.parse_args()
    topo = C.parse_topology(args)
    use_explicit_endpoints = len(topo.p_urls) == 1 and len(topo.d_urls) == 1

    fail_before = _fetch_metric_sum(
        topo.first_p, "pd_transfer_failures_total"
    )
    inflight_before = _fetch_metric_sum(
        topo.first_p, "pd_transfer_inflight"
    )

    rows = []
    failed = []
    for round_idx in range(args.rounds):
        for n_tokens, room, rid_base in CASES:
            rid = f"{rid_base}-r{round_idx}"
            prompt = " hello" * n_tokens
            try:
                rsp = C.fire_pd_pair(
                    topo,
                    rid=rid,
                    prompt=prompt,
                    bootstrap_room=room,
                    max_new_tokens=args.max_new_tokens,
                    p_url=topo.first_p if use_explicit_endpoints else None,
                    d_url=topo.first_d if use_explicit_endpoints else None,
                    timeout=240.0,
                )
            except Exception as exc:  # noqa: BLE001
                rows.append(
                    {
                        "rid": rid,
                        "round": round_idx,
                        "error": repr(exc),
                    }
                )
                failed.append(f"{rid}:exc={type(exc).__name__}")
                continue

            p_meta = rsp["P"].get("meta_info", {})
            d_meta = rsp["D"].get("meta_info", {})
            p_finish = p_meta.get("finish_reason") or {}
            d_finish = d_meta.get("finish_reason") or {}
            p_ids = rsp["P"].get("output_ids") or []
            d_ids = rsp["D"].get("output_ids") or []
            p_text = rsp["P"].get("text")
            d_text = rsp["D"].get("text")

            p_ok = (
                p_text == ""
                and p_ids == []
                and p_meta.get("completion_tokens") == 0
                and p_finish.get("type") == "length"
                and p_finish.get("length") == 0
            )
            d_ok = (
                bool(d_text)
                and isinstance(d_ids, list)
                and len(d_ids) > 0
                and int(d_meta.get("completion_tokens") or 0) > 0
                and d_finish.get("type") in {"length", "stop", "abort"}
            )
            rows.append(
                {
                    "rid": rid,
                    "round": round_idx,
                    "p_ok": p_ok,
                    "d_ok": d_ok,
                    "p_completion_tokens": p_meta.get("completion_tokens"),
                    "d_completion_tokens": d_meta.get("completion_tokens"),
                }
            )
            if not p_ok:
                failed.append(f"{rid}:p-contract")
            if not d_ok:
                failed.append(f"{rid}:d-contract")

    time.sleep(3.0)
    fail_after = _fetch_metric_sum(
        topo.first_p, "pd_transfer_failures_total"
    )
    inflight_after = _fetch_metric_sum(
        topo.first_p, "pd_transfer_inflight"
    )

    if (
        fail_before is not None
        and fail_after is not None
        and fail_after > fail_before
    ):
        failed.append(f"failures_delta={fail_after - fail_before}")

    if (
        inflight_before is not None
        and inflight_after is not None
        and inflight_after - inflight_before > 0.5
    ):
        failed.append(f"inflight_leak={inflight_after - inflight_before}")

    C.write_report(
        args,
        "normal_path_soak",
        {
            "rounds": args.rounds,
            "rows": rows,
            "failures_before": fail_before,
            "failures_after": fail_after,
            "inflight_before": inflight_before,
            "inflight_after": inflight_after,
            "failed": failed,
        },
    )
    if failed:
        return C.print_result(
            False, f"{len(failed)} soak failures: {failed[:3]}"
        )
    return C.print_result(
        True, f"{args.rounds} rounds over {len(CASES)} cases stayed clean"
    )


if __name__ == "__main__":
    sys.exit(main())
