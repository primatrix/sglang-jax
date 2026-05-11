#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import subprocess
import sys
import time
from pathlib import Path

KUBECTL = [
    "kubectl",
    "--context",
    "gke_poc-tpu-partner_us-central1_tpuv7x-64-node",
]

GROUPS = {
    "ep8": {
        "pods": ["s1c-ep8-0-lrpqn"],
        "ep_size": 8,
        "dist_port": 30208,
    },
    "ep32": {
        "pods": [
            "s1c-ep32-0-g9tbq",
            "s1c-ep32-1-xbp2v",
            "s1c-ep32-2-dk77m",
            "s1c-ep32-3-vdvhs",
        ],
        "ep_size": 32,
        "dist_port": 30232,
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
        "dist_port": 30264,
    },
}

CASE_ENVS = {
    "all_enable": {
        "FUSED_MOE_BENCHMARK_ALL_DISABLE": "False",
        "FUSED_MOE_BENCHMARK_DISABLE_A2A": "False",
        "FUSED_MOE_BENCHMARK_DISABLE_DYNAMIC_FFN1": "False",
        "FUSED_MOE_BENCHMARK_DISABLE_DYNAMIC_FFN2": "False",
        "FUSED_MOE_BENCHMARK_DISABLE_WEIGHT_LOAD": "False",
        "FUSED_MOE_BENCHMARK_DISABLE_A2A_S_TILE_READ": "False",
        "FUSED_MOE_BENCHMARK_DISABLE_A2A_S_ACC_TILE_WRITE": "False",
        "FUSED_MOE_BENCHMARK_DISABLE_SHARED_EXPERT": "False",
        "FUSED_MOE_BENCHMARK_DISABLE_ALL_REDUCE_METADATA": "False",
        "FUSED_MOE_BENCHMARK_DISABLE_SYNC_BARRIER": "False",
    },
    "no_a2a": {
        "FUSED_MOE_BENCHMARK_ALL_DISABLE": "False",
        "FUSED_MOE_BENCHMARK_DISABLE_A2A": "True",
        "FUSED_MOE_BENCHMARK_DISABLE_DYNAMIC_FFN1": "False",
        "FUSED_MOE_BENCHMARK_DISABLE_DYNAMIC_FFN2": "False",
        "FUSED_MOE_BENCHMARK_DISABLE_WEIGHT_LOAD": "False",
        "FUSED_MOE_BENCHMARK_DISABLE_A2A_S_TILE_READ": "False",
        "FUSED_MOE_BENCHMARK_DISABLE_A2A_S_ACC_TILE_WRITE": "False",
        "FUSED_MOE_BENCHMARK_DISABLE_SHARED_EXPERT": "False",
        "FUSED_MOE_BENCHMARK_DISABLE_ALL_REDUCE_METADATA": "True",
        "FUSED_MOE_BENCHMARK_DISABLE_SYNC_BARRIER": "True",
    },
    "control_only": {
        "FUSED_MOE_BENCHMARK_ALL_DISABLE": "True",
        "FUSED_MOE_BENCHMARK_DISABLE_A2A": "True",
        "FUSED_MOE_BENCHMARK_DISABLE_DYNAMIC_FFN1": "True",
        "FUSED_MOE_BENCHMARK_DISABLE_DYNAMIC_FFN2": "True",
        "FUSED_MOE_BENCHMARK_DISABLE_WEIGHT_LOAD": "True",
        "FUSED_MOE_BENCHMARK_DISABLE_A2A_S_TILE_READ": "True",
        "FUSED_MOE_BENCHMARK_DISABLE_A2A_S_ACC_TILE_WRITE": "True",
        "FUSED_MOE_BENCHMARK_DISABLE_SHARED_EXPERT": "True",
        "FUSED_MOE_BENCHMARK_DISABLE_ALL_REDUCE_METADATA": "False",
        "FUSED_MOE_BENCHMARK_DISABLE_SYNC_BARRIER": "False",
    },
    "a2a_only": {
        "FUSED_MOE_BENCHMARK_ALL_DISABLE": "True",
        "FUSED_MOE_BENCHMARK_DISABLE_A2A": "False",
        "FUSED_MOE_BENCHMARK_DISABLE_DYNAMIC_FFN1": "True",
        "FUSED_MOE_BENCHMARK_DISABLE_DYNAMIC_FFN2": "True",
        "FUSED_MOE_BENCHMARK_DISABLE_WEIGHT_LOAD": "True",
        "FUSED_MOE_BENCHMARK_DISABLE_A2A_S_TILE_READ": "True",
        "FUSED_MOE_BENCHMARK_DISABLE_A2A_S_ACC_TILE_WRITE": "True",
        "FUSED_MOE_BENCHMARK_DISABLE_SHARED_EXPERT": "True",
        "FUSED_MOE_BENCHMARK_DISABLE_ALL_REDUCE_METADATA": "False",
        "FUSED_MOE_BENCHMARK_DISABLE_SYNC_BARRIER": "False",
    },
}

CASE_RE = re.compile(
    r"\[case=(?P<case>[^\]]+)\] tokens=(?P<num_tokens>\d+), .*ep_size=(?P<ep_size>\d+)"
)
TIME_RE = re.compile(
    r"fused_moe\[(?P<tag>[^\]]+)\]: (?P<mean_ms>[0-9.]+) ms .*samples=(?P<samples>\[[^\]]*\])"
)


def now() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def run(
    cmd: list[str], *, timeout: int | None = None, check: bool = True
) -> subprocess.CompletedProcess:
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
  git stash push -u -m codex-before-stage2-a2a-bench >/dev/null
fi
git fetch {remote_name} {ref}
git switch -C stage2-a2a-bench FETCH_HEAD
PYTHONPATH=$PWD/python /tmp/tpu_logs/venv/bin/python -m py_compile benchmark/moe/bench_fused_moe.py benchmark/moe/utils.py
git rev-parse --short HEAD
"""
        proc = remote(pod, script, timeout=240)
        print(f"{pod}: {proc.stdout.strip().splitlines()[-1]}")


def parse_rank0_log(text: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    current_case: dict[str, object] | None = None
    for line in text.splitlines():
        case_match = CASE_RE.search(line)
        if case_match:
            current_case = {
                "case": case_match.group("case"),
                "num_tokens": int(case_match.group("num_tokens")),
                "ep_size": int(case_match.group("ep_size")),
            }
            continue
        time_match = TIME_RE.search(line)
        if time_match and current_case is not None:
            row = dict(current_case)
            row.update(
                {
                    "tag": time_match.group("tag"),
                    "mean_ms": float(time_match.group("mean_ms")),
                    "samples": time_match.group("samples"),
                }
            )
            rows.append(row)
    return rows


def run_case(
    *,
    group_name: str,
    case_name: str,
    args: argparse.Namespace,
    out_dir: Path,
    run_id: str,
) -> list[dict[str, object]]:
    group = GROUPS[group_name]
    pods = group["pods"]
    ep_size = group["ep_size"]
    rank0_ip = get_pod_ip(pods[0])
    dist_addr = f"{rank0_ip}:{group['dist_port']}"
    remote_logs: dict[str, str] = {}
    env = " ".join(f"{k}={v}" for k, v in CASE_ENVS[case_name].items())
    token_args = " ".join(str(x) for x in args.num_tokens)
    extra_args = " ".join(args.extra_args or [])

    base_cmd = f"""
cd {args.repo_dir}
export PYTHONPATH=$PWD/python
export JAX_COMPILATION_CACHE_DIR={args.compilation_cache_dir}
{env} /tmp/tpu_logs/venv/bin/python -u -m benchmark.moe.bench_fused_moe \\
  --shape-preset {args.shape_preset} \\
  --ep-size {ep_size} --tp-size 1 \\
  --num-tokens {token_args} \\
  --iters {args.iters} --warmup-iters {args.warmup_iters} \\
  --imbalance-mode {args.imbalance_mode} \\
  --hotspot-ratio {args.hotspot_ratio} \\
  --hotspot-count {args.hotspot_count} \\
  {extra_args}
"""

    procs: dict[str, subprocess.Popen] = {}
    order = list(range(1, len(pods))) + [0]
    for rank in order:
        pod = pods[rank]
        remote_log = f"/tmp/stage2_a2a_{run_id}_{group_name}_{case_name}_rank{rank}.log"
        remote_logs[pod] = remote_log
        dist_args = ""
        if len(pods) > 1:
            dist_args = (
                f" --dist-init-addr {dist_addr}"
                f" --num-processes {len(pods)}"
                f" --process-id {rank}"
                f" --distributed-init-timeout {args.distributed_init_timeout}"
            )
        script = f"""
set -euo pipefail
({base_cmd} {dist_args}) > {remote_log} 2>&1
tail -n 80 {remote_log}
"""
        print(f"start {group_name}/{case_name} rank={rank} pod={pod}")
        procs[pod] = remote_popen(pod, script)
        time.sleep(0.5)

    failures = []
    for pod, proc in procs.items():
        try:
            stdout, stderr = proc.communicate(timeout=args.timeout_s)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate()
            failures.append((pod, "timeout", stdout, stderr))
            continue
        if proc.returncode != 0:
            failures.append((pod, proc.returncode, stdout, stderr))

    for pod, rc, stdout, stderr in failures:
        print(f"FAILED {group_name}/{case_name} pod={pod} rc={rc}", file=sys.stderr)
        print(stdout[-4000:], file=sys.stderr)
        print(stderr[-4000:], file=sys.stderr)
    if failures:
        raise RuntimeError(f"{group_name}/{case_name} failed on {len(failures)} pod(s)")

    rank0_log = remote(pods[0], f"cat {remote_logs[pods[0]]}", timeout=120, check=False).stdout
    local_rank0_log = out_dir / f"{group_name}_{case_name}_rank0.log"
    local_rank0_log.write_text(rank0_log)
    rows = parse_rank0_log(rank0_log)
    for row in rows:
        row.update(
            {
                "run_id": run_id,
                "group": group_name,
                "ablation": case_name,
                "remote_logs": remote_logs,
                "local_rank0_log": str(local_rank0_log),
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RFC-0026 Stage 2 fused MoE ablations.")
    parser.add_argument("--groups", default="ep8,ep32,ep64")
    parser.add_argument("--cases", default="all_enable,no_a2a,control_only,a2a_only")
    parser.add_argument("--num-tokens", type=int, nargs="+", default=[512, 4096, 8192])
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--warmup-iters", type=int, default=1)
    parser.add_argument("--shape-preset", default="ling2_6_1t")
    parser.add_argument("--imbalance-mode", default="balanced")
    parser.add_argument("--hotspot-ratio", type=float, default=1.0)
    parser.add_argument("--hotspot-count", type=int, default=48)
    parser.add_argument("--repo-dir", default="/tmp/tpu_logs/sglang-jax")
    parser.add_argument("--remote", default="origin")
    parser.add_argument("--ref", default="exp/fused-ep-moe-stage2-a2a")
    parser.add_argument("--skip-setup", action="store_true")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--timeout-s", type=int, default=7200)
    parser.add_argument("--distributed-init-timeout", type=int, default=600)
    parser.add_argument("--compilation-cache-dir", default="/tmp/jit_cache")
    parser.add_argument("--extra-args", nargs=argparse.REMAINDER, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    groups = [x for x in args.groups.split(",") if x]
    cases = [x for x in args.cases.split(",") if x]
    for group in groups:
        if group not in GROUPS:
            raise ValueError(f"Unknown group {group!r}; choices={sorted(GROUPS)}")
    for case in cases:
        if case not in CASE_ENVS:
            raise ValueError(f"Unknown case {case!r}; choices={sorted(CASE_ENVS)}")

    run_id = now()
    out_dir = Path(args.out_dir or f"/tmp/sglang_stage2_a2a_{run_id}")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.jsonl"

    if not args.skip_setup:
        for group in groups:
            setup_repo(
                GROUPS[group]["pods"], repo_dir=args.repo_dir, remote_name=args.remote, ref=args.ref
            )

    with summary_path.open("a") as f:
        for group in groups:
            for case in cases:
                print(f"\n=== {group} / {case} ===")
                rows = run_case(
                    group_name=group, case_name=case, args=args, out_dir=out_dir, run_id=run_id
                )
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    f.flush()
                    print(json.dumps(row, ensure_ascii=False))

    print(f"\nsummary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
