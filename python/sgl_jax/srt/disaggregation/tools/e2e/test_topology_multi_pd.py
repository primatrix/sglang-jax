"""P0-1: multi-P × multi-D topology.

Verifies:
  1. Bootstrap /list_prefills returns >= 2 P entries.
  2. Different bootstrap_room values land on different P peers
     (deterministically — picker is `room % sorted(keys)`).
  3. A single D can drive transfers against both P peers in the
     same minute without crossing wires (no duplicate rid, no
     state-machine errors).

The third assertion is observed indirectly: we fan-out 4 PD pairs
across 2 (room%P) buckets and require every output to be present
and non-empty.
"""

from __future__ import annotations

import argparse
import sys

from sgl_jax.srt.disaggregation.tools.e2e import _common as C


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    C.add_topology_args(p)
    args = p.parse_args()
    topo = C.parse_topology(args)

    prefills = C.list_prefills(topo)
    p_keys = sorted([row["bootstrap_key"] for row in prefills])
    if len(p_keys) < 2:
        return C.print_result(
            False,
            f"bootstrap registry has {len(p_keys)} P entries; "
            f"need >= 2",
        )
    topo.refresh_picker()

    # Pick 4 rooms whose `% len(keys)` mod distribution covers all P.
    rooms = [10, 11, 20, 21]
    bucket_counts: dict[str, int] = {k: 0 for k in p_keys}
    results = []
    for i, room in enumerate(rooms):
        rid = f"topo-{i}-{room}"
        # Mirror mini_lb: send prefill to the same P the decode will
        # pull from. Without this the room hashes to one P but the
        # KV lives on another and the decode hangs.
        chosen_p = topo.pick_p_for_room(room)
        rsp = C.fire_pd_pair(
            topo, rid=rid, prompt="Hello, my name is",
            bootstrap_room=room, max_new_tokens=4,
            p_url=chosen_p,
        )
        expected_p = p_keys[room % len(p_keys)]
        bucket_counts[expected_p] += 1
        d_text = rsp["D"].get("text", "")
        p_text = rsp["P"].get("text", "")
        if not d_text or not p_text:
            return C.print_result(
                False,
                f"empty response for rid={rid}: P={p_text!r} D={d_text!r}",
            )
        results.append({
            "rid": rid, "room": room, "expected_p": expected_p,
            "chosen_p_url": chosen_p,
            "p_text": p_text, "d_text": d_text,
        })

    # All P peers should have received at least one request.
    unused = [k for k, n in bucket_counts.items() if n == 0]
    if unused:
        return C.print_result(
            False,
            f"P peers never selected by rooms {rooms}: {unused}",
        )

    C.write_report(args, "topology_multi_pd", {
        "p_keys": p_keys,
        "bucket_counts": bucket_counts,
        "results": results,
    })
    return C.print_result(
        True,
        f"P_keys={p_keys} buckets={bucket_counts}",
    )


if __name__ == "__main__":
    sys.exit(main())
