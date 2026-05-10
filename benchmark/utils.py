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

    # Prefer explicit benchmark step markers. These are host trace annotation
    # boundaries around one timed iteration and exist even when compute_func is
    # already jitted, where an outer jax.named_scope may not enter XLA.
    marker_step_events = [
        e
        for e in trace_events
        if (
            str(e.get("name", "")).startswith(f"{MARKER}:")
            or str((e.get("args", {}) or {}).get("long_name", "")).startswith(f"{MARKER}:")
        )
        and "step_num" in (e.get("args", {}) or {})
    ]
    marker_durations = _dominant_pid_durations(marker_step_events)
    if marker_durations:
        return marker_durations

    # If a benchmark marker is emitted inside compiled work, it may show up in
    # tf_op/op_name/hlo_module instead of the event name.
    marker_events = [e for e in trace_events if MARKER in _event_text(e)]
    marker_call_done_events = [e for e in marker_events if e.get("name", "").endswith("call-done")]
    if marker_call_done_events:
        marker_events = marker_call_done_events
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

    return []


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
    return durations[max(0, int(discard_initial_samples)) :]
