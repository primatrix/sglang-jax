"""P0-2: prefill-only contract correctness across multiple prompts.

Drives 12 PD pairs (short / medium / long prompts × max_new_tokens
1/8/32) and checks the current phase-1 production-path contract:

  * P returns an empty completion with ``finish_reason=length(0)``.
  * D returns the actual generated completion.

This is the correct acceptance test once prefill workers stop serving
their own continuation. It intentionally does NOT compare P and D
output byte-for-byte anymore — that old oracle only made sense while P
was still locally decoding in debug mode.

Each case gets a unique salt prefix so the P-side prefix cache doesn't
hit between cases. Stage 4 e2e finding C: P-side ``_extract_req_kv``
crashes with a sharding error when the prefill batch's #new-token == 1
because of a prefix-cache hit (the gather indices become empty). Until
that is fixed, the salt avoids exercising the bug from the matrix.
"""

from __future__ import annotations

import argparse
import sys
import time

from sgl_jax.srt.disaggregation.tools.e2e import _common as C


PROMPTS = [
    ("hello", "Hello, my name is"),
    ("france", "The capital of France is"),
    ("math",   "2 + 2 ="),
    ("story",  "Once upon a time, in a small village near the sea, lived a wise"),
]
MAX_NEW = [1, 8, 32]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    C.add_topology_args(p)
    args = p.parse_args()
    topo = C.parse_topology(args)

    salt = f"#{int(time.time())}-"
    violations = []
    cases = []
    for ptag, prompt in PROMPTS:
        for mn in MAX_NEW:
            rid = f"correct-{ptag}-{mn}"
            # Salt the prompt so each case has a unique prefix on P.
            salted_prompt = salt + rid + " " + prompt
            room = abs(hash(rid)) % 100000
            rsp = C.fire_pd_pair(
                topo, rid=rid, prompt=salted_prompt,
                bootstrap_room=room, max_new_tokens=mn,
            )
            p_ids = rsp["P"].get("output_ids") or rsp["P"].get(
                "meta_info", {}
            ).get("output_token_ids")
            d_ids = rsp["D"].get("output_ids") or rsp["D"].get(
                "meta_info", {}
            ).get("output_token_ids")
            p_text = rsp["P"].get("text")
            d_text = rsp["D"].get("text")
            p_meta = rsp["P"].get("meta_info", {})
            d_meta = rsp["D"].get("meta_info", {})
            p_finish = p_meta.get("finish_reason") or {}
            d_finish = d_meta.get("finish_reason") or {}
            p_ok = (
                p_text == ""
                and (p_ids == [] or p_ids is None)
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
            cases.append({
                "rid": rid, "prompt": prompt, "max_new": mn,
                "p_text": p_text, "d_text": d_text,
                "p_output_ids": p_ids,
                "d_output_ids": d_ids,
                "p_ok": p_ok,
                "d_ok": d_ok,
            })
            if not (p_ok and d_ok):
                violations.append(rid)

    C.write_report(args, "correctness_byte_equal", {
        "total": len(cases),
        "violation_count": len(violations),
        "violating_rids": violations,
        "cases": cases,
    })
    if violations:
        return C.print_result(
            False,
            f"{len(violations)}/{len(cases)} contract violations: {violations[:3]}",
        )
    return C.print_result(True, f"all {len(cases)} prefill-only contracts hold")


if __name__ == "__main__":
    sys.exit(main())
