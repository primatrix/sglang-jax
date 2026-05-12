from __future__ import annotations

import argparse
import datetime as dt
import json
import shlex
import subprocess
import time
from pathlib import Path

KUBECTL = [
    "kubectl",
    "--context",
    "gke_poc-tpu-partner_us-central1_tpuv7x-64-node",
]

KUBECTL_STREAM_RESET_MARKERS = (
    "error reading from error stream",
    "connection reset by peer",
)

GROUPS = {
    "ep8": {
        "pods": ["s1c-ep8-0-lrpqn"],
        "ep_size": 8,
        "dist_port": 30508,
    },
    "ep32": {
        "pods": [
            "s1c-ep32-0-g9tbq",
            "s1c-ep32-1-xbp2v",
            "s1c-ep32-2-dk77m",
            "s1c-ep32-3-vdvhs",
        ],
        "ep_size": 32,
        "dist_port": 30532,
    },
    "ep64": {
        "pods": [
            "s1c-ep64-0-6dlsw",
            "s1c-ep64-1-7z26r",
            "s1c-ep64-2-wngxd",
            "s1c-ep64-3-798th",
            "s1c-ep64-4-9kl9r",
            "s1c-ep64-5-scpb7",
            "s1c-ep64-6-8vlq2",
            "s1c-ep64-7-wqhmp",
        ],
        "ep_size": 64,
        "dist_port": 30564,
    },
}


def now() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def run(cmd: list[str], *, timeout: int | None = None, check: bool = True):
    proc = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout)
    if check and proc.returncode != 0:
        raise RuntimeError(
            f"command failed rc={proc.returncode}: {' '.join(cmd)}\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    return proc


def remote(pod: str, script: str, *, timeout: int | None = None, check: bool = True):
    return run(
        KUBECTL + ["exec", "-c", "bench", pod, "--", "bash", "-lc", script],
        timeout=timeout,
        check=check,
    )


def remote_popen(pod: str, script: str) -> subprocess.Popen:
    return subprocess.Popen(
        KUBECTL + ["exec", "-c", "bench", pod, "--", "bash", "-lc", script],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def get_pod_ip(pod: str) -> str:
    proc = run(KUBECTL + ["get", "pod", pod, "-o", "jsonpath={.status.podIP}"], timeout=30)
    return proc.stdout.strip()


def setup_repo(pods: list[str], *, repo_dir: str, remote_name: str, ref: str) -> None:
    for pod in pods:
        script = f"""
set -euo pipefail
cd {repo_dir}
if [ -n "$(git status --porcelain)" ]; then
  git stash push -u -m codex-before-stage2-scatter-pack-bench >/dev/null
fi
git fetch {remote_name} {shlex.quote(ref)}
git switch -C stage2-scatter-pack-bench FETCH_HEAD
PYTHONPATH=$PWD/python /tmp/tpu_logs/venv/bin/python -m py_compile \
  benchmark/moe/bench_stage2_scatter_pack.py benchmark/moe/run_stage2_scatter_pack.py
git rev-parse --short HEAD
"""
        proc = remote(pod, script, timeout=240)
        print(f"{pod}: {proc.stdout.strip().splitlines()[-1]}")


def bench_args(args: argparse.Namespace, group: dict[str, object], rank: int, dist_addr: str):
    items: list[object] = [
        "--ep-size",
        group["ep_size"],
        "--tp-size",
        1,
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
        "--vmem-limit-mb",
        args.vmem_limit_mb,
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


def is_kubectl_stream_reset(error: str) -> bool:
    return any(marker in error for marker in KUBECTL_STREAM_RESET_MARKERS)


def run_group(args: argparse.Namespace, group_name: str, out_dir: Path, run_id: str) -> None:
    group = GROUPS[group_name]
    pods = group["pods"]
    dist_addr = f"{get_pod_ip(pods[0])}:{group['dist_port']}"
    procs: dict[str, subprocess.Popen] = {}
    remote_logs: dict[str, str] = {}

    for rank in list(range(1, len(pods))) + [0]:
        pod = pods[rank]
        remote_log = f"/tmp/stage2_scatter_pack_{run_id}_{group_name}_rank{rank}.log"
        remote_logs[pod] = remote_log
        cmd_args = bench_args(args, group, rank, dist_addr)
        script = f"""
set -euo pipefail
cd {args.repo_dir}
export PYTHONPATH=$PWD/python
export JAX_COMPILATION_CACHE_DIR={args.compilation_cache_dir}
/tmp/tpu_logs/venv/bin/python -u -m benchmark.moe.bench_stage2_scatter_pack {cmd_args} > {remote_log} 2>&1
tail -n 120 {remote_log}
"""
        print(f"start {group_name} rank={rank} pod={pod}")
        procs[pod] = remote_popen(pod, script)
        time.sleep(0.5)

    errors: dict[str, str] = {}
    for pod, proc in procs.items():
        try:
            stdout, stderr = proc.communicate(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate()
            errors[pod] = f"timeout after {args.timeout}s\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
            continue
        (out_dir / f"{group_name}_{pod}.tail.log").write_text(stdout + stderr)
        if proc.returncode != 0:
            err = f"rc={proc.returncode}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
            if not is_kubectl_stream_reset(err):
                errors[pod] = err

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
    print(json.dumps(record, ensure_ascii=False, indent=2))
    if errors:
        raise RuntimeError(f"{group_name} failed: {errors}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage2 scatter pack benchmark on pods.")
    parser.add_argument("--groups", nargs="+", choices=sorted(GROUPS), default=["ep8"])
    parser.add_argument("--num-tokens", nargs="+", type=int, default=[512, 8192])
    parser.add_argument("--hidden-size", type=int, default=8192)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--fanout", nargs="+", type=int, default=[1, 4, 8])
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=[
            "direct",
            "hbm_pack_serial",
            "hbm_pack_overlap",
            "hbm_pack_demux",
            "vmem_pack",
            "vmem_pack_overlap",
        ],
        default=[
            "direct",
            "hbm_pack_serial",
            "hbm_pack_overlap",
            "hbm_pack_demux",
            "vmem_pack",
            "vmem_pack_overlap",
        ],
    )
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--warmup-iters", type=int, default=1)
    parser.add_argument("--distributed-init-timeout", type=int, default=300)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--repo-dir", default="/tmp/tpu_logs/sglang-jax")
    parser.add_argument("--remote", default="primatrix")
    parser.add_argument("--ref", default="exp/fused-ep-moe-stage2-scatter-pack-study")
    parser.add_argument("--skip-setup", action="store_true")
    parser.add_argument("--compilation-cache-dir", default="/tmp/jit_cache")
    parser.add_argument("--trace-root", default="/tmp/stage2_scatter_pack_trace")
    parser.add_argument("--vmem-limit-mb", type=int, default=96)
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
