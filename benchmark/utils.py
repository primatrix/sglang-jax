from __future__ import annotations

import gzip
import json
import os
import pathlib
import random
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


def _extract_marker_durations_ms(trace: dict[str, Any], task: str | None = None) -> list[float]:
    """Extract per-iteration device durations (ms) from profiler trace.

    Finds top-level JIT XLA Module events that carry ``device_duration_ps``
    (the actual TPU kernel time). Groups by device (pid), then **averages**
    across devices per iteration to smooth measurement noise. Each device's
    duration already includes collective communication wait time.

    Never falls back to host-side ``dur`` which includes Python / host-device
    synchronisation overhead.
    """
    all_events = trace.get("traceEvents", [])

    # ── 1. Collect top-level JIT module events with device_duration_ps ──
    jit_events: list[dict[str, Any]] = []
    for e in all_events:
        dev_dur = e.get("args", {}).get("device_duration_ps")
        if dev_dur and str(e.get("name", "")).startswith("jit_"):
            jit_events.append(e)

    if not jit_events:
        return []

    # ── 2. Group by function name and pick the benchmark target ──
    # The benchmark target is the JIT function with the most total device
    # time (tiny helpers like jit__multi_slice are excluded automatically).
    by_func: dict[str, list[dict[str, Any]]] = {}
    for e in jit_events:
        by_func.setdefault(e["name"], []).append(e)

    target_func = max(
        by_func,
        key=lambda n: sum(int(e["args"]["device_duration_ps"]) for e in by_func[n]),
    )
    target_events = by_func[target_func]

    # ── 3. Group by pid (TPU device), sort by timestamp ──
    by_pid: dict[int, list[tuple[float, float]]] = {}
    for e in target_events:
        pid = e.get("pid")
        if not isinstance(pid, int):
            continue
        ts = float(e.get("ts", 0.0))
        dur_ms = int(e["args"]["device_duration_ps"]) / 1e9  # ps → ms
        by_pid.setdefault(pid, []).append((ts, dur_ms))

    if not by_pid:
        return []

    for pid in by_pid:
        by_pid[pid].sort(key=lambda x: x[0])

    # ── 4. Per iteration: average across devices ──
    # Each device's device_duration_ps already includes collective wait time
    # (all-to-all is synchronous). Averaging smooths measurement noise.
    n_iters = max(len(v) for v in by_pid.values())
    result: list[float] = []
    for i in range(n_iters):
        durs = [pid_events[i][1] for pid_events in by_pid.values() if i < len(pid_events)]
        result.append(sum(durs) / len(durs))

    return result


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

    with jax.profiler.trace(trace_dir):
        for i in range(tries):
            data_args = data_generator()
            with jax.profiler.StepTraceAnnotation(task, step_num=i):
                with jax.named_scope(f"{MARKER}_{i}"):
                    out = compute_func(*data_args)
                    jax.block_until_ready(out)

    trace = _load_trace(trace_dir)
    return _extract_marker_durations_ms(trace, task=task)
