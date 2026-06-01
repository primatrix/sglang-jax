"""I3: chaos via tools/chaos.sh wrapper.

This test calls the existing chaos.sh script for each scenario and
parses its exit code. It's a thin Python wrapper so the matrix
driver can include chaos with the same RESULT: PASS|FAIL contract.

Note: chaos.sh requires kubectl + the deployment to be labelled
(pd-role=prefill|decode), which our debug topology isn't. Until we
add those labels, this test is gated behind --enable; otherwise it
prints SKIP and exits 0.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

from sgl_jax.srt.disaggregation.tools.e2e import _common as C


SCENARIOS = ["kill_p", "drop_dcn", "bootstrap"]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    C.add_topology_args(p)
    p.add_argument(
        "--enable", action="store_true",
        help="Actually run chaos.sh. Default: skip (most deployments "
        "lack the pd-role= labels chaos.sh needs).",
    )
    p.add_argument("--router-url", default=None)
    p.add_argument(
        "--scenarios", default=",".join(SCENARIOS),
        help="Comma-separated subset to run.",
    )
    args = p.parse_args()

    if not args.enable:
        C.write_report(args, "chaos", {"skipped": True})
        return C.print_result(
            True, "SKIPPED (rerun with --enable on a label-ed cluster)"
        )

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    chaos_sh = os.path.join(here, "chaos.sh")
    env = dict(os.environ)
    if args.router_url:
        env["ROUTER_URL"] = args.router_url

    results = {}
    failed = []
    for scen in args.scenarios.split(","):
        proc = subprocess.run(
            ["bash", chaos_sh, scen],
            env=env, capture_output=True, text=True,
            timeout=300,
        )
        results[scen] = {
            "rc": proc.returncode,
            "stdout_tail": proc.stdout[-2000:],
            "stderr_tail": proc.stderr[-2000:],
        }
        if proc.returncode != 0:
            failed.append(scen)

    C.write_report(args, "chaos", {"results": results})
    if failed:
        return C.print_result(False, f"failed scenarios: {failed}")
    return C.print_result(True, f"all {len(results)} chaos scenarios passed")


if __name__ == "__main__":
    sys.exit(main())
