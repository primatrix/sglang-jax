#!/usr/bin/env python3
"""Check concurrent E/PD chip visibility on a fresh TPU process, without weights."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

CHILD = r"""
import json,os,time
from pathlib import Path
import jax
import jax.numpy as jnp
root=Path(os.environ['PROBE_OUTPUT'])
devices=jax.devices()
checks=[]
for device in devices:
 x=jax.device_put([1,2,3],device)
 checks.append(int(jnp.sum(x).block_until_ready())==6)
result={'devices':[str(d) for d in devices], 'count':len(devices),'compute_ok':all(checks),
        'visible_chips':os.environ['TPU_VISIBLE_CHIPS']}
(root/(os.environ['PROBE_ROLE']+'.json')).write_text(json.dumps(result,indent=2))
print(json.dumps(result),flush=True)
while not (root/'release').exists():time.sleep(1)
"""


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--layout", choices=["E1", "E2"], required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    n = 2 if args.layout == "E1" else 1
    procs = []
    logs = []
    report = {"layout": args.layout, "full_device_preflight": False}
    try:
        for role, chips in [("encoder", list(range(n))), ("pd", list(range(n, 4)))]:
            env = os.environ.copy()
            env.pop("ALLOW_MULTIPLE_LIBTPU_LOAD", None)
            env.update(
                TPU_VISIBLE_CHIPS=",".join(map(str, chips)),
                TPU_CHIPS_PER_PROCESS_BOUNDS=f"{len(chips)},1,1",
                TPU_PROCESS_BOUNDS="1,1,1",
                PROBE_OUTPUT=str(args.output_dir),
                PROBE_ROLE=role,
            )
            log = (args.output_dir / f"{role}.log").open("w")
            logs.append(log)
            proc = subprocess.Popen(
                [sys.executable, "-u", "-c", CHILD],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            procs.append(proc)
            deadline = time.monotonic() + 180
            while not (args.output_dir / f"{role}.json").exists():
                if proc.poll() is not None:
                    raise RuntimeError(f"{role} initialization exited {proc.returncode}")
                if time.monotonic() > deadline:
                    raise RuntimeError(f"{role} initialization timed out")
                time.sleep(2)
            report[role] = json.loads((args.output_dir / f"{role}.json").read_text())
            if report[role]["count"] != 2 * len(chips):
                raise RuntimeError(f"{role} device count mismatch")
        report["passed"] = all(report[r]["compute_ok"] for r in ("encoder", "pd"))
    except Exception as exc:
        report["passed"] = False
        report["error"] = str(exc)
    finally:
        (args.output_dir / "release").touch()
        for proc in procs:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                import signal

                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
        for log in logs:
            log.close()
        (args.output_dir / "report.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report), flush=True)
        for path in args.output_dir.glob("*.log"):
            print(path.name, flush=True)
            print(path.read_text(errors="replace")[-12000:], flush=True)


if __name__ == "__main__":
    main()
