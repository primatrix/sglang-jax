"""Benchmark/profile grouped_topk v1 vs archived v2 Pallas kernels on TPU.

This entrypoint is intentionally operator-benchmark scoped:

* It compares the current production stable-tiebreak v1 kernel with the
  archived v2 padded-output kernel on the same seeded logits and bias.
* It extracts device timing from profiler trace events instead of host wall
  time.
* For one selected token size, it preserves per-variant XProf trace trees and
  copies newly emitted Mosaic LLO dumps into variant-specific directories.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import pathlib
import shutil
import statistics
from collections.abc import Callable, Iterable
from typing import Any

import jax
import jax.numpy as jnp

from sgl_jax.srt.kernels.grouped_topk.archive.v2_lane_padded_kernel import (
    _tuned_v2_config,
    grouped_topk_pallas_v2,
)
from sgl_jax.srt.kernels.grouped_topk.v1.kernel import (
    SAFE_AUTO_BT,
    _largest_safe_divisor,
    grouped_topk_pallas,
)

SCOPE_V1 = "bench_grouped_topk_v1"
SCOPE_V2 = "bench_grouped_topk_v2"


def _parse_csv_ints(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _logits(tokens: int, experts: int, seed: int) -> jax.Array:
    return jax.nn.sigmoid(
        jax.random.normal(jax.random.PRNGKey(seed), (tokens, experts), dtype=jnp.float32)
    )


def _bias(experts: int, seed: int) -> jax.Array:
    return jax.random.normal(jax.random.PRNGKey(seed), (experts,), dtype=jnp.float32) * 0.1


def _duration_ms(event: dict[str, Any]) -> float | None:
    args = event.get("args", {})
    device_duration_ps = args.get("device_duration_ps")
    if device_duration_ps:
        return float(device_duration_ps) / 1e9
    if "dur" in event:
        return float(event["dur"]) / 1e3
    return None


def _latest_trace_dir(trace_root: pathlib.Path) -> pathlib.Path | None:
    profile_root = trace_root / "plugins" / "profile"
    if not profile_root.exists():
        return None
    profile_dirs = [path for path in profile_root.iterdir() if path.is_dir()]
    if not profile_dirs:
        return None
    return max(profile_dirs, key=os.path.getmtime)


def _latest_trace_events(trace_root: pathlib.Path) -> list[dict[str, Any]]:
    latest = _latest_trace_dir(trace_root)
    if latest is None:
        return []
    events: list[dict[str, Any]] = []
    for trace_file in sorted(latest.glob("*.trace.json.gz")):
        with gzip.open(trace_file, "rb") as f:
            trace = json.load(f)
        shard_events = trace.get("traceEvents", [])
        if isinstance(shard_events, list):
            events.extend(shard_events)
    return events


def _metadata_maps(
    events: Iterable[dict[str, Any]],
) -> tuple[dict[int, str], dict[tuple[int, int], str]]:
    process_names: dict[int, str] = {}
    thread_names: dict[tuple[int, int], str] = {}
    for event in events:
        if event.get("ph") != "M":
            continue
        args = event.get("args", {})
        if event.get("name") == "process_name" and isinstance(event.get("pid"), int):
            process_names[event["pid"]] = args.get("name", "")
        elif (
            event.get("name") == "thread_name"
            and isinstance(event.get("pid"), int)
            and isinstance(event.get("tid"), int)
        ):
            thread_names[(event["pid"], event["tid"])] = args.get("name", "")
    return process_names, thread_names


def _xla_module_durations_ms(events: list[dict[str, Any]]) -> list[float]:
    process_names, thread_names = _metadata_maps(events)
    durations: list[float] = []
    for event in events:
        if event.get("ph") != "X":
            continue
        pid = event.get("pid")
        tid = event.get("tid")
        if process_names.get(pid) != "/device:TPU:0":
            continue
        if thread_names.get((pid, tid)) != "XLA Modules":
            continue
        duration = _duration_ms(event)
        if duration is not None:
            durations.append(duration)
    return durations


def _scope_op_summary_ms(
    events: list[dict[str, Any]], scope: str, num_modules: int
) -> tuple[float, int]:
    process_names, thread_names = _metadata_maps(events)
    total = 0.0
    names: set[str] = set()
    for event in events:
        if event.get("ph") != "X":
            continue
        pid = event.get("pid")
        tid = event.get("tid")
        if process_names.get(pid) != "/device:TPU:0":
            continue
        if thread_names.get((pid, tid)) != "XLA Ops":
            continue
        args = event.get("args", {})
        blob = " ".join(
            str(value)
            for value in (
                event.get("name", ""),
                args.get("tf_op", ""),
                args.get("hlo_op", ""),
                args.get("long_name", ""),
            )
        )
        if scope not in blob:
            continue
        duration = _duration_ms(event)
        if duration is None:
            continue
        total += duration
        names.add(str(event.get("name", "")))
    return total / max(num_modules, 1), len(names)


def _p90(samples: list[float]) -> float:
    ordered = sorted(samples)
    return ordered[min(len(ordered) - 1, int(0.9 * (len(ordered) - 1)))]


def _snapshot_files(root: pathlib.Path) -> set[pathlib.Path]:
    if not root.exists():
        return set()
    return {path for path in root.rglob("*") if path.is_file()}


def _copy_new_files(before: set[pathlib.Path], root: pathlib.Path, dst: pathlib.Path) -> list[str]:
    dst.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for src in sorted(_snapshot_files(root) - before):
        rel = src.relative_to(root)
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
        copied.append(str(rel))
    return copied


def _copy_trace_tree(src: pathlib.Path, dst: pathlib.Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _resolved_bt(
    variant: str,
    tokens: int,
    experts: int,
    groups: int,
    topk_groups: int,
    topk: int,
    block_tokens: int | str,
) -> int:
    if block_tokens != "auto":
        return min(int(block_tokens), tokens)
    if variant == "v2":
        tuned_bt, _ = _tuned_v2_config(tokens, experts, groups, topk_groups, topk)
        if tuned_bt is not None and tokens % tuned_bt == 0:
            return tuned_bt
    return _largest_safe_divisor(tokens, cap=SAFE_AUTO_BT, align=128) or tokens


def _make_variant(
    variant: str,
    *,
    groups: int,
    topk_groups: int,
    topk: int,
    block_tokens: int | str,
    interpret: bool,
) -> tuple[str, Callable[[jax.Array, jax.Array], tuple[jax.Array, jax.Array]]]:
    if variant == "v1":

        def fn(logits, bias):
            with jax.named_scope(SCOPE_V1):
                return grouped_topk_pallas(
                    logits,
                    bias,
                    num_expert_group=groups,
                    topk_group=topk_groups,
                    topk=topk,
                    block_tokens=block_tokens,
                    interpret=interpret,
                )

        return SCOPE_V1, jax.jit(fn)

    if variant == "v2":

        def fn(logits, bias):
            with jax.named_scope(SCOPE_V2):
                return grouped_topk_pallas_v2(
                    logits,
                    bias,
                    num_expert_group=groups,
                    topk_group=topk_groups,
                    topk=topk,
                    block_tokens=block_tokens,
                    interpret=interpret,
                )

        return SCOPE_V2, jax.jit(fn)

    raise ValueError(f"unknown variant {variant!r}")


def _profile_variant(
    run_fn: Callable[[], tuple[jax.Array, jax.Array]],
    *,
    scope: str,
    trace_root: pathlib.Path,
    warmup: int,
    iters: int,
) -> tuple[list[float], dict[str, Any]]:
    for _ in range(warmup):
        jax.block_until_ready(run_fn())

    trace_root.mkdir(parents=True, exist_ok=True)
    with jax.profiler.trace(str(trace_root)):
        for step in range(iters):
            with (
                jax.profiler.StepTraceAnnotation(scope, step_num=step),
                jax.named_scope(f"{scope}_call"),
            ):
                jax.block_until_ready(run_fn())

    events = _latest_trace_events(trace_root)
    module_samples = _xla_module_durations_ms(events)
    scope_ms, scope_op_count = _scope_op_summary_ms(events, scope, len(module_samples))
    profile_dir = _latest_trace_dir(trace_root)
    meta = {
        "trace_root": str(trace_root),
        "profile_dir": str(profile_dir) if profile_dir is not None else None,
        "num_events": len(events),
        "scope_ops_ms_per_iter": scope_ms,
        "scope_op_name_count": scope_op_count,
    }
    if not module_samples:
        raise RuntimeError(f"No TPU XLA Modules durations found under {trace_root}")
    return module_samples, meta


def _summary_row(
    *,
    tokens: int,
    experts: int,
    groups: int,
    topk_groups: int,
    topk: int,
    variant: str,
    scope: str,
    block_tokens_arg: int | str,
    block_tokens_resolved: int,
    samples_ms: list[float],
    trace_meta: dict[str, Any],
    llo_files: list[str] | None,
) -> dict[str, Any]:
    return {
        "T": tokens,
        "E": experts,
        "G": groups,
        "Gtop": topk_groups,
        "topk": topk,
        "variant": variant,
        "scope": scope,
        "block_tokens_arg": block_tokens_arg,
        "block_tokens_resolved": block_tokens_resolved,
        "median_ms": statistics.median(samples_ms),
        "mean_ms": statistics.fmean(samples_ms),
        "p90_ms": _p90(samples_ms),
        "num_samples": len(samples_ms),
        "samples_ms": samples_ms,
        "timing_source": "trace_xla_modules",
        "trace_path": trace_meta["trace_root"],
        "profile_dir": trace_meta["profile_dir"],
        "scope_ops_ms_per_iter": trace_meta["scope_ops_ms_per_iter"],
        "scope_op_name_count": trace_meta["scope_op_name_count"],
        "llo_file_count": len(llo_files or []),
        "llo_files": llo_files or [],
    }


def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    output = pathlib.Path(args.output)
    trace_root = pathlib.Path(args.trace_root)
    artifact_profile_root = pathlib.Path(args.profile_output_root)
    llo_root = pathlib.Path(args.llo_root)
    llo_variant_root = pathlib.Path(args.llo_variant_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    trace_root.mkdir(parents=True, exist_ok=True)
    artifact_profile_root.mkdir(parents=True, exist_ok=True)
    llo_root.mkdir(parents=True, exist_ok=True)
    llo_variant_root.mkdir(parents=True, exist_ok=True)

    token_sizes = _parse_csv_ints(args.tokens)
    block_tokens: int | str = "auto" if args.block_tokens == "auto" else int(args.block_tokens)

    print(
        "ENV "
        + json.dumps(
            {
                "jax": jax.__version__,
                "devices": [str(device) for device in jax.devices()],
                "device_kind": jax.devices()[0].device_kind,
                "LIBTPU_INIT_ARGS": os.environ.get("LIBTPU_INIT_ARGS", ""),
                "XLA_FLAGS": os.environ.get("XLA_FLAGS", ""),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    rows: list[dict[str, Any]] = []
    with output.open("w") as out:
        for tokens in token_sizes:
            logits = jax.device_put(_logits(tokens, args.experts, seed=tokens + args.experts))
            bias = jax.device_put(_bias(args.experts, seed=args.experts + args.topk))
            compiled = {
                variant: _make_variant(
                    variant,
                    groups=args.groups,
                    topk_groups=args.topk_groups,
                    topk=args.topk,
                    block_tokens=block_tokens,
                    interpret=args.interpret,
                )
                for variant in ("v1", "v2")
            }

            token_rows: list[dict[str, Any]] = []
            outputs: dict[str, tuple[jax.Array, jax.Array]] = {}
            for variant in ("v1", "v2"):
                scope, fn = compiled[variant]

                def run_fn(fn=fn):
                    return fn(logits, bias)

                tag = f"{variant}_T{tokens}"
                current_trace_root = trace_root / tag
                before_llo = _snapshot_files(llo_root)
                samples_ms, trace_meta = _profile_variant(
                    run_fn,
                    scope=scope,
                    trace_root=current_trace_root,
                    warmup=args.warmup,
                    iters=args.iters,
                )
                outputs[variant] = jax.block_until_ready(fn(logits, bias))
                llo_files: list[str] = []
                if tokens == args.profile_tokens:
                    _copy_trace_tree(current_trace_root, artifact_profile_root / tag)
                    llo_files = _copy_new_files(before_llo, llo_root, llo_variant_root / tag)

                row = _summary_row(
                    tokens=tokens,
                    experts=args.experts,
                    groups=args.groups,
                    topk_groups=args.topk_groups,
                    topk=args.topk,
                    variant=variant,
                    scope=scope,
                    block_tokens_arg=block_tokens,
                    block_tokens_resolved=_resolved_bt(
                        variant,
                        tokens,
                        args.experts,
                        args.groups,
                        args.topk_groups,
                        args.topk,
                        block_tokens,
                    ),
                    samples_ms=samples_ms,
                    trace_meta=trace_meta,
                    llo_files=llo_files,
                )
                token_rows.append(row)

            v1_weights, v1_ids = outputs["v1"]
            v2_weights, v2_ids = outputs["v2"]
            ids_equal = bool(jnp.array_equal(v1_ids, v2_ids))
            max_weight_abs_diff = float(jnp.max(jnp.abs(v1_weights - v2_weights)))
            compare = {
                "T": tokens,
                "ids_equal": ids_equal,
                "max_weight_abs_diff": max_weight_abs_diff,
            }
            print("COMPARE " + json.dumps(compare, sort_keys=True), flush=True)

            for row in token_rows:
                row.update(compare)
                out.write(json.dumps(row, sort_keys=True) + "\n")
                out.flush()
                rows.append(row)
                print("METRIC " + json.dumps(row, sort_keys=True), flush=True)

    print("\n=== grouped_topk v1 vs v2 ===", flush=True)
    print(
        f"{'T':>7} {'v1_ms':>10} {'v2_ms':>10} {'v2/v1':>8} {'v1_BT':>7} {'v2_BT':>7}",
        flush=True,
    )
    by_t = {(row["T"], row["variant"]): row for row in rows}
    for tokens in token_sizes:
        v1 = by_t[(tokens, "v1")]
        v2 = by_t[(tokens, "v2")]
        ratio = v2["median_ms"] / v1["median_ms"]
        print(
            f"{tokens:>7} {v1['median_ms']:>10.4f} {v2['median_ms']:>10.4f} {ratio:>8.3f} "
            f"{v1['block_tokens_resolved']:>7} {v2['block_tokens_resolved']:>7}",
            flush=True,
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", default="128,512,4096,16384")
    parser.add_argument("--experts", type=int, default=256)
    parser.add_argument("--groups", type=int, default=8)
    parser.add_argument("--topk-groups", type=int, default=4)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--block-tokens", default="auto")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--profile-tokens", type=int, default=512)
    parser.add_argument("--interpret", action="store_true")
    parser.add_argument("--output", required=True)
    parser.add_argument("--trace-root", required=True)
    parser.add_argument("--profile-output-root", required=True)
    parser.add_argument("--llo-root", required=True)
    parser.add_argument("--llo-variant-root", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
