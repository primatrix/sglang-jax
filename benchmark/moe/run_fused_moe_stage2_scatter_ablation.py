from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import shlex
import subprocess
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
        "dist_port": 30308,
    },
    "ep32": {
        "pods": [
            "s1c-ep32-0-g9tbq",
            "s1c-ep32-1-xbp2v",
            "s1c-ep32-2-dk77m",
            "s1c-ep32-3-vdvhs",
        ],
        "ep_size": 32,
        "dist_port": 30332,
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
        "dist_port": 30364,
    },
}


def flags(**overrides: str) -> dict[str, str]:
    base = {
        "FUSED_MOE_BENCHMARK_ALL_DISABLE": "False",
        "FUSED_MOE_BENCHMARK_DISABLE_A2A": "False",
        "FUSED_MOE_BENCHMARK_DISABLE_A2A_SCATTER": "False",
        "FUSED_MOE_BENCHMARK_DISABLE_A2A_GATHER": "False",
        "FUSED_MOE_BENCHMARK_DISABLE_DYNAMIC_FFN1": "False",
        "FUSED_MOE_BENCHMARK_DISABLE_DYNAMIC_FFN2": "False",
        "FUSED_MOE_BENCHMARK_DISABLE_WEIGHT_LOAD": "False",
        "FUSED_MOE_BENCHMARK_DISABLE_A2A_S_TILE_READ": "False",
        "FUSED_MOE_BENCHMARK_DISABLE_A2A_S_ACC_TILE_WRITE": "False",
        "FUSED_MOE_BENCHMARK_DISABLE_OUTPUT_ACCUMULATE": "False",
        "FUSED_MOE_BENCHMARK_DISABLE_SHARED_EXPERT": "False",
        "FUSED_MOE_BENCHMARK_DISABLE_ALL_REDUCE_METADATA": "False",
        "FUSED_MOE_BENCHMARK_DISABLE_SYNC_BARRIER": "False",
    }
    base.update(overrides)
    return base


STAGE2_SKELETON = {
    "FUSED_MOE_BENCHMARK_DISABLE_A2A_GATHER": "True",
    "FUSED_MOE_BENCHMARK_DISABLE_DYNAMIC_FFN1": "True",
    "FUSED_MOE_BENCHMARK_DISABLE_DYNAMIC_FFN2": "True",
    "FUSED_MOE_BENCHMARK_DISABLE_WEIGHT_LOAD": "True",
    "FUSED_MOE_BENCHMARK_DISABLE_A2A_S_TILE_READ": "True",
    "FUSED_MOE_BENCHMARK_DISABLE_A2A_S_ACC_TILE_WRITE": "True",
    "FUSED_MOE_BENCHMARK_DISABLE_OUTPUT_ACCUMULATE": "True",
    "FUSED_MOE_BENCHMARK_DISABLE_SHARED_EXPERT": "True",
}

CASE_ENVS = {
    "full": flags(),
    "stage2_control": flags(
        **STAGE2_SKELETON,
        FUSED_MOE_BENCHMARK_DISABLE_A2A_SCATTER="True",
    ),
    "stage2_scatter_only": flags(**STAGE2_SKELETON),
    "a2a_total_only": flags(
        FUSED_MOE_BENCHMARK_DISABLE_DYNAMIC_FFN1="True",
        FUSED_MOE_BENCHMARK_DISABLE_DYNAMIC_FFN2="True",
        FUSED_MOE_BENCHMARK_DISABLE_WEIGHT_LOAD="True",
        FUSED_MOE_BENCHMARK_DISABLE_A2A_S_TILE_READ="True",
        FUSED_MOE_BENCHMARK_DISABLE_A2A_S_ACC_TILE_WRITE="True",
        FUSED_MOE_BENCHMARK_DISABLE_OUTPUT_ACCUMULATE="True",
        FUSED_MOE_BENCHMARK_DISABLE_SHARED_EXPERT="True",
    ),
}

CASE_RE = re.compile(
    r"\[case=(?P<case>[^\]]+)\] tokens=(?P<num_tokens>\d+), .*ep_size=(?P<ep_size>\d+)"
)
TIME_RE = re.compile(
    r"fused_moe\[(?P<tag>[^\]]+)\]: (?P<mean_ms>[0-9.]+) ms .*samples=(?P<samples>\[[^\]]*\])"
)


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
  git stash push -u -m codex-before-stage2-scatter-bench >/dev/null
fi
git fetch {remote_name} {shlex.quote(ref)}
git switch -C stage2-scatter-bench FETCH_HEAD
PYTHONPATH=$PWD/python /tmp/tpu_logs/venv/bin/python -m py_compile \
  benchmark/moe/bench_fused_moe.py benchmark/moe/utils.py
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


def bench_args(args: argparse.Namespace, group: dict[str, object], rank: int, dist_addr: str):
    items: list[object] = [
        "--shape-preset",
        args.shape_preset,
        "--ep-size",
        group["ep_size"],
        "--tp-size",
        1,
        "--num-tokens",
        *args.num_tokens,
        "--iters",
        args.iters,
        "--warmup-iters",
        args.warmup_iters,
        "--imbalance-mode",
        args.imbalance_mode,
        "--hotspot-ratio",
        args.hotspot_ratio,
        "--hotspot-count",
        args.hotspot_count,
        "--compilation-cache-dir",
        args.compilation_cache_dir,
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
    items.extend(args.extra_args or [])
    return " ".join(shlex.quote(str(x)) for x in items)


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
    dist_addr = f"{get_pod_ip(pods[0])}:{group['dist_port']}"
    env = " ".join(f"{k}={v}" for k, v in CASE_ENVS[case_name].items())
    remote_logs: dict[str, str] = {}
    procs: dict[str, subprocess.Popen] = {}

    for rank in list(range(1, len(pods))) + [0]:
        pod = pods[rank]
        remote_log = f"/tmp/stage2_scatter_{run_id}_{group_name}_{case_name}_rank{rank}.log"
        remote_logs[pod] = remote_log
        cmd_args = bench_args(args, group, rank, dist_addr)
        script = f"""
set -euo pipefail
cd {args.repo_dir}
export PYTHONPATH=$PWD/python
export JAX_COMPILATION_CACHE_DIR={args.compilation_cache_dir}
{env} /tmp/tpu_logs/venv/bin/python -u -m benchmark.moe.bench_fused_moe {cmd_args} > {remote_log} 2>&1
tail -n 80 {remote_log}
"""
        print(f"start {group_name}/{case_name} rank={rank} pod={pod}")
        procs[pod] = remote_popen(pod, script)
        time.sleep(0.5)

    rows: list[dict[str, object]] = []
    errors: dict[str, str] = {}
    for pod, proc in procs.items():
        stdout, stderr = proc.communicate(timeout=args.case_timeout)
        (out_dir / f"{group_name}_{case_name}_{pod}.tail.log").write_text(stdout + stderr)
        if proc.returncode != 0:
            errors[pod] = f"rc={proc.returncode}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"

    for pod, remote_log in remote_logs.items():
        local_name = out_dir / f"{group_name}_{case_name}_{pod}.full.log"
        proc = remote(pod, f"cat {remote_log}", timeout=120, check=False)
        local_name.write_text(proc.stdout + proc.stderr)
        if pod == pods[0]:
            rows = parse_rank0_log(proc.stdout)

    record = {
        "group": group_name,
        "case_name": case_name,
        "shape_preset": args.shape_preset,
        "env": CASE_ENVS[case_name],
        "remote_logs": remote_logs,
        "rows": rows,
        "errors": errors,
    }
    with (out_dir / f"summary_{group_name}.jsonl").open("a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    if errors:
        raise RuntimeError(f"{group_name}/{case_name} failed: {errors}")
    return rows


def print_delta_summary(out_dir: Path, group_name: str) -> None:
    records = []
    path = out_dir / f"summary_{group_name}.jsonl"
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        if line.strip():
            records.append(json.loads(line))
    by_case: dict[str, dict[int, float]] = {}
    for record in records:
        values: dict[int, float] = {}
        for row in record["rows"]:
            values[int(row["num_tokens"])] = float(row["mean_ms"])
        by_case[record["case_name"]] = values
    if "stage2_control" not in by_case or "stage2_scatter_only" not in by_case:
        return
    print("\nStage2 scatter delta:")
    print("tokens | control_ms | scatter_only_ms | scatter_delta_ms | scatter_delta_pct")
    for tokens in sorted(set(by_case["stage2_control"]) & set(by_case["stage2_scatter_only"])):
        base = by_case["stage2_control"][tokens]
        scatter = by_case["stage2_scatter_only"][tokens]
        delta = scatter - base
        pct = delta / base * 100 if base else float("nan")
        print(f"{tokens:6d} | {base:10.3f} | {scatter:15.3f} | {delta:16.3f} | {pct:17.2f}%")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fused MoE Stage2 scatter ablation.")
    parser.add_argument("--groups", nargs="+", choices=sorted(GROUPS), default=["ep8"])
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=sorted(CASE_ENVS),
        default=["stage2_control", "stage2_scatter_only", "a2a_total_only", "full"],
    )
    parser.add_argument(
        "--shape-preset",
        choices=["ling2_6_1t", "mimo_v2_flash", "mimo_v2_pro"],
        default="ling2_6_1t",
    )
    parser.add_argument("--num-tokens", nargs="+", type=int, default=[512, 8192])
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--warmup-iters", type=int, default=1)
    parser.add_argument("--imbalance-mode", default="balanced")
    parser.add_argument("--hotspot-ratio", type=float, default=1.0)
    parser.add_argument("--hotspot-count", type=int, default=48)
    parser.add_argument("--distributed-init-timeout", type=int, default=300)
    parser.add_argument("--case-timeout", type=int, default=1800)
    parser.add_argument("--repo-dir", default="/tmp/tpu_logs/sglang-jax")
    parser.add_argument("--remote", default="origin")
    parser.add_argument("--ref", default="exp/fused-ep-moe-stage2-scatter")
    parser.add_argument("--skip-setup", action="store_true")
    parser.add_argument("--compilation-cache-dir", default="/tmp/jit_cache")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("extra_args", nargs=argparse.REMAINDER)
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
        for case_name in args.cases:
            run_case(
                group_name=group_name,
                case_name=case_name,
                args=args,
                out_dir=out_dir,
                run_id=run_id,
            )
        print_delta_summary(out_dir, group_name)


if __name__ == "__main__":
    main()
