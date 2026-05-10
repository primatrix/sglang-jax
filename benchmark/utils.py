from __future__ import annotations

import gzip
import json
import os
import pathlib
import random
import re
import string
import time
from typing import Any

import jax

MARKER = "SGLANG_JAX_BENCH"
TRACE_TIMING_SUMMARY = "trace_timing_summary.json"

_COMPILATION_CACHE_ENV_VARS = ("SGLANG_JAX_COMPILATION_CACHE_DIR", "JAX_COMPILATION_CACHE_DIR")


def _maybe_enable_compilation_cache_from_env() -> None:
    cache_dir = None
    for key in _COMPILATION_CACHE_ENV_VARS:
        value = os.environ.get(key)
        if value:
            cache_dir = value
            break
    if not cache_dir:
        return

    try:
        from jax.experimental.compilation_cache import (
            compilation_cache as _compilation_cache,
        )
    except Exception:
        return
    _compilation_cache.set_cache_dir(cache_dir)


_maybe_enable_compilation_cache_from_env()


def _event_text(event: dict[str, Any]) -> str:
    args = event.get("args", {})
    return " ".join(
        str(part)
        for part in (
            event.get("name", ""),
            event.get("cat", ""),
            args.get("long_name", ""),
            args.get("tf_op", ""),
            args.get("op_name", ""),
            args.get("hlo_module", ""),
        )
        if part
    )


def _extract_marker_durations_ms(trace: dict[str, Any], task: str | None = None) -> list[float]:
    def _durations_by_pid(events: list[dict[str, Any]]) -> dict[int, list[float]]:
        by_pid: dict[int, list[dict[str, Any]]] = {}
        for e in events:
            pid = e.get("pid")
            if isinstance(pid, int):
                by_pid.setdefault(pid, []).append(e)

        durations: dict[int, list[float]] = {}
        for pid, pid_events in by_pid.items():
            pid_events.sort(key=lambda ev: float(ev.get("ts", 0.0)))
            pid_durations: list[float] = []
            for e in pid_events:
                args = e.get("args", {})
                if args.get("device_duration_ps"):
                    pid_durations.append(float(args["device_duration_ps"]) / 1e9)
                elif "dur" in e:
                    pid_durations.append(float(e["dur"]) / 1e3)
            if pid_durations:
                durations[pid] = pid_durations
        return durations

    def _dominant_pid_durations(events: list[dict[str, Any]]) -> list[float]:
        durations_by_pid = _durations_by_pid(events)
        if not durations_by_pid:
            return []
        return max(sorted(durations_by_pid.items()), key=lambda kv: len(kv[1]))[1]

    trace_events = trace.get("traceEvents", [])

    if task:
        preferred_patterns = []
        if task.startswith("layer0_a2a_"):
            preferred_patterns.append("SGLANG_JAX_LAYER0_A2A")
        scatter_match = re.search(r"(layer1_a2a_scatter_topk8_bt\d+)", task)
        if scatter_match:
            preferred_patterns.append(scatter_match.group(1))
        gather_match = re.search(r"(layer1_a2a_gather_topk8_bt\d+)", task)
        if gather_match:
            preferred_patterns.append(gather_match.group(1))
        if task.startswith("layer1_wait_"):
            preferred_patterns.append(task)

        preferred_events = [
            e
            for e in trace_events
            if preferred_patterns
            and any(pattern in _event_text(e) for pattern in preferred_patterns)
            and (e.get("args", {}) or {}).get("device_duration_ps")
            and not str(e.get("name", "")).startswith("jit_")
        ]
        preferred_durations = _dominant_pid_durations(preferred_events)
        if preferred_durations:
            return preferred_durations

    # Prefer compiled/device marker events, matching the Ironwood
    # microbenchmarks. StepTraceAnnotation events and outer jit call-done
    # events are host/envelope windows and are kept only as fallbacks.
    marker_events = [
        e
        for e in trace_events
        if MARKER in _event_text(e)
        and not str(e.get("name", "")).startswith(f"{MARKER}:")
        and not str((e.get("args", {}) or {}).get("long_name", "")).startswith(f"{MARKER}:")
        and not str(e.get("name", "")).endswith("call-done")
    ]
    marker_durations = _dominant_pid_durations(marker_events)
    if marker_durations:
        return marker_durations

    if task:
        event_matcher = re.compile(task)
        events = []
        for e in trace_events:
            if "name" in e and event_matcher.match(e["name"]):
                events.append(e)
        task_durations = _dominant_pid_durations(events)
        if task_durations:
            return task_durations

    marker_step_events = [
        e
        for e in trace_events
        if (
            str(e.get("name", "")).startswith(f"{MARKER}:")
            or str((e.get("args", {}) or {}).get("long_name", "")).startswith(f"{MARKER}:")
        )
        and "step_num" in (e.get("args", {}) or {})
    ]
    marker_step_durations = _dominant_pid_durations(marker_step_events)
    if marker_step_durations:
        return marker_step_durations

    return []


def _event_duration_ms(event: dict[str, Any]) -> float | None:
    args = event.get("args", {})
    if args.get("device_duration_ps"):
        return float(args["device_duration_ps"]) / 1e9
    if "dur" in event:
        return float(event["dur"]) / 1e3
    return None


def _percentile(samples: list[float], percent: float) -> float | None:
    if not samples:
        return None
    ordered = sorted(samples)
    if len(ordered) == 1:
        return ordered[0]
    index = (len(ordered) - 1) * percent
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return ordered[lower]
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _event_group_name(event: dict[str, Any]) -> str | None:
    name = str(event.get("name", ""))
    text = _event_text(event).lower()
    if name == "barrier-cores":
        return "barrier_cores"
    if "all_to_all" in text or "all-to-all" in text:
        return "all_to_all_op"
    if name.startswith("jit_all_to_all("):
        return "jit_all_to_all"
    if "remote" in text and ("dma" in text or "copy" in text):
        return "remote_dma_or_copy"
    if "dma" in text:
        return "dma"
    if "wait" in text or "sync" in text:
        return "wait_or_sync"
    return None


def _samples_summary(samples: list[float]) -> dict[str, Any]:
    return {
        "count": len(samples),
        "min_ms": min(samples) if samples else None,
        "p50_ms": _percentile(samples, 0.5),
        "p90_ms": _percentile(samples, 0.9),
        "max_ms": max(samples) if samples else None,
        "samples_ms": samples,
    }


def _trace_timing_summary(
    trace: dict[str, Any],
    *,
    task: str,
    marker_ms: list[float],
    discard_initial_samples: int,
) -> dict[str, Any]:
    event_groups: dict[str, list[float]] = {}
    for event in trace.get("traceEvents", []):
        group = _event_group_name(event)
        if group is None:
            continue
        duration_ms = _event_duration_ms(event)
        if duration_ms is None:
            continue
        event_groups.setdefault(group, []).append(duration_ms)

    discarded = max(0, int(discard_initial_samples))
    measured_marker_ms = marker_ms[discarded:]
    barrier_ms = sorted(event_groups.get("barrier_cores", []))
    barrier_representative_ms = _percentile(barrier_ms, 0.5)
    marker_minus_barrier_ms = None
    if barrier_representative_ms is not None:
        marker_minus_barrier_ms = [
            max(0.0, sample - barrier_representative_ms) for sample in measured_marker_ms
        ]

    return {
        "schema_version": 1,
        "task": task,
        "marker": MARKER,
        "timing_semantics": {
            "marker_ms": "Preferred device marker duration from named_scope/tf_op device_duration_ps; falls back to task-matched device event, then host StepTraceAnnotation only if no device marker exists.",
            "barrier_cores_ms": "Device trace barrier-cores events; diagnostic only.",
            "marker_minus_barrier_cores_ms": (
                "Deprecated diagnostic residual. Do not use as communication time."
            ),
            "event_groups": "Best-effort trace decomposition by event name/tf_op.",
        },
        "discard_initial_samples": discarded,
        "marker_ms_all": marker_ms,
        "marker_ms": measured_marker_ms,
        "marker_summary": _samples_summary(measured_marker_ms),
        "barrier_cores_summary": _samples_summary(barrier_ms),
        "marker_minus_barrier_cores_ms": marker_minus_barrier_ms,
        "marker_minus_barrier_cores_summary": (
            _samples_summary(marker_minus_barrier_ms)
            if marker_minus_barrier_ms is not None
            else _samples_summary([])
        ),
        "event_group_summaries": {
            group: _samples_summary(sorted(samples))
            for group, samples in sorted(event_groups.items())
        },
    }


def _write_trace_timing_summary(trace_dir: str, summary: dict[str, Any]) -> None:
    path = pathlib.Path(trace_dir) / TRACE_TIMING_SUMMARY
    try:
        path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    except Exception:
        return


def _load_trace(trace_root: str) -> dict[str, Any]:
    trace_dir = pathlib.Path(trace_root) / "plugins" / "profile"
    if not trace_dir.exists():
        raise FileNotFoundError(f"No trace output under {trace_dir}")
    latest_dir = max(trace_dir.iterdir(), key=os.path.getmtime)
    trace_files = list(latest_dir.glob("*.trace.json.gz"))
    if not trace_files:
        raise FileNotFoundError(f"No trace json.gz under {latest_dir}")

    combined: dict[str, Any] = {"traceEvents": []}
    for trace_file in sorted(trace_files):
        with gzip.open(trace_file, "rb") as f:
            shard = json.load(f)
        shard_events = shard.get("traceEvents", [])
        if isinstance(shard_events, list):
            combined["traceEvents"].extend(shard_events)
        if "displayTimeUnit" in shard and "displayTimeUnit" not in combined:
            combined["displayTimeUnit"] = shard["displayTimeUnit"]
        if "otherData" in shard and "otherData" not in combined:
            combined["otherData"] = shard["otherData"]
    return combined


def multiple_iteration_timeit_from_trace(
    compute_func,
    data_generator,
    task: str,
    tries: int = 5,
    warmup: int = 0,
    discard_initial_samples: int = 0,
    trace_root: str = "/tmp/sglang_jax_moe_trace",
) -> list[float]:
    """
    Profile multiple iterations and pull per-iteration kernel time from trace.
    """
    trace_name = f"{task}_" + "".join(random.choices(string.ascii_uppercase + string.digits, k=6))
    trace_dir = os.path.join(trace_root, trace_name)
    os.makedirs(trace_dir, exist_ok=True)

    start = time.perf_counter()
    for _ in range(max(0, int(warmup))):
        data_args = data_generator()
        out = compute_func(*data_args)
        jax.block_until_ready(out)
    print(f"warmed up in {(time.perf_counter() - start) * 1000} ms")

    profiler_options = None
    try:
        profiler_options = jax.profiler.ProfileOptions()
        profiler_options.advanced_configuration = {
            "tpu_trace_mode": "TRACE_ONLY_XLA",
            "tpu_num_sparse_cores_to_trace": 0,
            "tpu_num_sparse_core_tiles_to_trace": 0,
        }
    except Exception:
        profiler_options = None

    trace_kwargs = {"profiler_options": profiler_options} if profiler_options is not None else {}
    traced_tries = int(tries) + max(0, int(discard_initial_samples))
    marker_task = f"{MARKER}:{task}"
    with jax.profiler.trace(trace_dir, **trace_kwargs):
        for i in range(traced_tries):
            data_args = data_generator()
            with jax.profiler.StepTraceAnnotation(marker_task, step_num=i):
                with jax.named_scope(f"{MARKER}_{i}"):
                    out = compute_func(*data_args)
                    jax.block_until_ready(out)

    trace = _load_trace(trace_dir)
    durations = _extract_marker_durations_ms(trace, task=task)
    summary = _trace_timing_summary(
        trace,
        task=task,
        marker_ms=durations,
        discard_initial_samples=discard_initial_samples,
    )
    _write_trace_timing_summary(trace_dir, summary)
    return summary["marker_ms"]
