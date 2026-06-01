"""Orthogonal: PD × dp_size=2.

The Stage 3 RFC only validated PD with dp_size=1. Sharding paths
through ``_extract_req_kv`` / ``_write_kv_to_pool`` haven't been
exercised with multiple data-parallel replicas. This test re-runs
the byte-equal correctness subset against a P+D pair deployed with
--dp-size 2, surfacing any sharding-spec mismatch.

This script does NOT manage the deployment — it assumes the
operator has restarted P and D with --dp-size 2 (and --tp-size
adjusted) before running it. The matrix driver documents the
manual steps in pd_e2e_matrix.md.
"""

from __future__ import annotations

import argparse
import sys

from sgl_jax.srt.disaggregation.tools.e2e import _common as C


PROMPTS = [
    ("hello", "Hello, my name is"),
    ("math",  "2 + 2 ="),
    ("story", "Once upon a time, in a small village near the sea, lived a wise"),
]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    C.add_topology_args(p)
    args = p.parse_args()
    topo = C.parse_topology(args)

    mismatches = []
    cases = []
    for ptag, prompt in PROMPTS:
        for mn in (1, 8):
            rid = f"dp2-{ptag}-{mn}"
            room = abs(hash(rid)) % 100000
            try:
                rsp = C.fire_pd_pair(
                    topo, rid=rid, prompt=prompt,
                    bootstrap_room=room, max_new_tokens=mn,
                )
            except Exception as e:
                mismatches.append(f"{rid}:exc={type(e).__name__}")
                cases.append({"rid": rid, "error": repr(e)})
                continue
            p_text = rsp["P"].get("text")
            d_text = rsp["D"].get("text")
            equal = p_text == d_text
            cases.append({
                "rid": rid, "prompt": prompt, "max_new": mn,
                "p_text": p_text, "d_text": d_text, "equal": equal,
            })
            if not equal:
                mismatches.append(rid)

    C.write_report(args, "orthogonal_dp", {
        "total": len(cases),
        "mismatch_count": len(mismatches),
        "cases": cases,
    })
    if mismatches:
        return C.print_result(
            False,
            f"{len(mismatches)}/{len(cases)} mismatched: {mismatches[:3]}",
        )
    return C.print_result(True, f"all {len(cases)} byte-equal under dp_size=2")


if __name__ == "__main__":
    sys.exit(main())
