#!/usr/bin/env python3
"""Slim a level-1 jax.profiler trace down to a readable "middle ground" for
Perfetto: keep sglang's own functions + the sim stage annotations (+ XLA ops),
drop the stdlib / framework firehose (builtins, sys, zmq, asyncio, threading,
importlib, <frozen>, ...) that python_tracer_level=1 floods in.

Capture a SMALL level-1 slice first (so it isn't truncated at ~1M events), then:

    python scripts/disaggregation/traceslim_trace.py --profiler-dir /tmp/epd-sim-profile

Writes <profiler-dir>/<tier>.slim.trace.json.gz for each tier; open those in
Perfetto. Tune with --drop (extra file tokens to remove) / --keep (force-keep).
Only the standard library is required.
"""
from __future__ import annotations

import argparse
import glob
import os

if __package__:
    from .trace_io import read_trace, write_trace
else:
    from trace_io import read_trace, write_trace

# File tokens (module/basename before the line number) whose Python frames are
# framework/stdlib noise, not EPD orchestration. Matched by prefix.
_NOISE = {
    "builtins",
    "sys",
    "<frozen",
    "<unknown>",
    "<string>",
    "<genexpr>",
    "<listcomp>",
    "<dictcomp>",
    "<lambda>",
    "_zmq",
    "zmq",
    "error.py",
    "threading",
    "socket",
    "selectors",
    "asyncio",
    "base_events",
    "_base",
    "queue",
    "concurrent",
    "time",
    "enum",
    "abc",
    "typing",
    "functools",
    "contextlib",
    "os.py",
    "posixpath",
    "ntpath",
    "genericpath",
    "re.py",
    "sre_",
    "json",
    "warnings",
    "logging",
    "inspect",
    "copy.py",
    "copyreg",
    "weakref",
    "_weakrefset",
    "importlib",
    "_collections_abc",
    "codecs",
    "encodings",
    "ssl",
    "hashlib",
    "urllib",
    "http",
    "email",
    # jax/numpy plumbing that isn't sglang orchestration
    "tree_util",
    "dtypes.py",
    "literals.py",
    "numpy",
    "array.py",
    "traceback",
}


def frame_source(name: str) -> str:
    body = name[1:]
    head = body.split(" ", 1)[0]  # "scheduler.py:2770" or "importlib._bootstrap>:1390"
    return head.rsplit(":", 1)[0]


def is_noise_frame(tok: str, drop: set[str], keep: set[str]) -> bool:
    if any(tok.startswith(k) for k in keep):
        return False
    return any(tok.startswith(n) for n in (_NOISE | drop))


def slim_trace(path: str, out: str, drop: set[str], keep: set[str]) -> tuple[int, int]:
    trace = read_trace(path)
    events = trace.get("traceEvents", [])
    trace["traceEvents"] = [
        event
        for event in events
        if event.get("ph") != "X"
        or not event.get("name", "").startswith("$")
        or not is_noise_frame(frame_source(event["name"]), drop, keep)
    ]
    write_trace(out, trace)
    return len(events), len(trace["traceEvents"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profiler-dir", default="/tmp/epd-sim-profile")
    ap.add_argument("--drop", default="", help="comma-separated extra file tokens to drop")
    ap.add_argument("--keep", default="", help="comma-separated file tokens to force-keep")
    args = ap.parse_args()
    drop = {x for x in args.drop.split(",") if x}
    keep = {x for x in args.keep.split(",") if x}

    traces = glob.glob(
        os.path.join(args.profiler_dir, "*", "plugins", "profile", "*", "*.trace.json.gz")
    )
    if not traces:
        print(f"no traces under {args.profiler_dir}")
        return 1
    for t in sorted(traces):
        tier = t.split(args.profiler_dir.rstrip("/") + "/", 1)[1].split("/plugins")[0]
        out = os.path.join(args.profiler_dir, f"{tier}.slim.trace.json.gz")
        before, after = slim_trace(t, out, drop, keep)
        print(
            f"{tier:10s} {before:>8d} -> {after:>7d} events  ({after / max(before, 1) * 100:.0f}% kept)"
        )
        print(f"   {out}")
    print("\nOpen the .slim.trace.json.gz files in Perfetto / chrome://tracing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
