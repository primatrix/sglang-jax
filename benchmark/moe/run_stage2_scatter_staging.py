from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import time
from pathlib import Path

from benchmark.moe.run_fused_moe_stage2_scatter_ablation import (
    GROUPS,
    get_pod_ip,
    now,
    remote,
    remote_popen,
    setup_repo,
)


def bench_args(args: argparse.Namespace, group: dict[str, object], rank: int, dist_addr: str):
    items: list[object] = [
        "--ep-size",
        group["ep_size"],
        "--num-tokens",
        *args.num_tokens,
        "--hidden-size",
        args.hidden_size,
        "--top-k",
        args.top_k,
        "--fanout",
        *args.fanout,
        "--modes",
        *args.modes,
        "--iters",
        args.iters,
        "--warmup-iters",
        args.warmup_iters,
        "--trace-root",
        args.trace_root,
    ]
    if len(group["pods"]) > 1:
        items.extend(
            [
                "--dist-init-addr",
                dist_addr,
                "--num-processes",
                len(group["pods"]),
                "--process-id",
                rank,
                "--distributed-init-timeout",
                args.distributed_init_timeout,
            ]
        )
    return " ".join(shlex.quote(str(x)) for x in items)


def parse_results(text: str) -> list[dict[str, object]]:
    rows = []
    for line in text.splitlines():
        if line.startswith("RESULT "):
            rows.append(json.loads(line.removeprefix("RESULT ")))
    return rows


def run_group(args: argparse.Namespace, group_name: str, out_dir: Path, run_id: str) -> None:
    group = GROUPS[group_name]
    pods = group["pods"]
    dist_addr = f"{get_pod_ip(pods[0])}:{group['dist_port'] + 100}"
    procs: dict[str, subprocess.Popen] = {}
    remote_logs: dict[str, str] = {}

    for rank in list(range(1, len(pods))) + [0]:
        pod = pods[rank]
        remote_log = f"/tmp/stage2_scatter_staging_{run_id}_{group_name}_rank{rank}.log"
        remote_logs[pod] = remote_log
        cmd_args = bench_args(args, group, rank, dist_addr)
        script = f"""
set -euo pipefail
cd {args.repo_dir}
export PYTHONPATH=$PWD/python
export JAX_COMPILATION_CACHE_DIR={args.compilation_cache_dir}
/tmp/tpu_logs/venv/bin/python -u -m benchmark.moe.bench_stage2_scatter_staging {cmd_args} > {remote_log} 2>&1
tail -n 80 {remote_log}
"""
        print(f"start {group_name} rank={rank} pod={pod}")
        procs[pod] = remote_popen(pod, script)
        time.sleep(0.5)

    errors: dict[str, str] = {}
    for pod, proc in procs.items():
        stdout, stderr = proc.communicate(timeout=args.timeout)
        (out_dir / f"{group_name}_{pod}.tail.log").write_text(stdout + stderr)
        if proc.returncode != 0:
            errors[pod] = f"rc={proc.returncode}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"

    rank0_rows = []
    for pod, remote_log in remote_logs.items():
        proc = remote(pod, f"cat {remote_log}", timeout=120, check=False)
        text = proc.stdout + proc.stderr
        (out_dir / f"{group_name}_{pod}.full.log").write_text(text)
        if pod == pods[0]:
            rank0_rows = parse_results(text)

    record = {
        "group": group_name,
        "remote_logs": remote_logs,
        "rows": rank0_rows,
        "errors": errors,
    }
    with (out_dir / "summary.jsonl").open("a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    if errors:
        raise RuntimeError(f"{group_name} failed: {errors}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage2 scatter staging benchmark on pods.")
    parser.add_argument("--groups", nargs="+", choices=sorted(GROUPS), default=["ep8"])
    parser.add_argument("--num-tokens", nargs="+", type=int, default=[512, 8192])
    parser.add_argument("--hidden-size", type=int, default=8192)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--fanout", nargs="+", type=int, default=[1, 4, 8])
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["direct", "staged_remote_only", "staged"],
        default=["direct", "staged_remote_only", "staged"],
    )
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--warmup-iters", type=int, default=1)
    parser.add_argument("--distributed-init-timeout", type=int, default=300)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--repo-dir", default="/tmp/tpu_logs/sglang-jax")
    parser.add_argument("--remote", default="primatrix")
    parser.add_argument("--ref", default="exp/fused-ep-moe-stage2-scatter")
    parser.add_argument("--skip-setup", action="store_true")
    parser.add_argument("--compilation-cache-dir", default="/tmp/jit_cache")
    parser.add_argument("--trace-root", default="/tmp/stage2_scatter_staging_trace")
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = now()
    for group_name in args.groups:
        group = GROUPS[group_name]
        if not args.skip_setup:
            setup_repo(group["pods"], repo_dir=args.repo_dir, remote_name=args.remote, ref=args.ref)
        run_group(args, group_name, out_dir, run_id)


if __name__ == "__main__":
    main()
