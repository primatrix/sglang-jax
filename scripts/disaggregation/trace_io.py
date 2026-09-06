"""Shared readers for plain or compressed Chrome trace files."""

from __future__ import annotations

import glob
import gzip
import json
from pathlib import Path


def read_trace(path: str | Path) -> dict:
    path = Path(path)
    open_trace = gzip.open if path.suffix == ".gz" else open
    with open_trace(path, "rt") as source:
        return json.load(source)


def write_trace(path: str | Path, trace: dict) -> None:
    path = Path(path)
    open_trace = gzip.open if path.suffix == ".gz" else open
    with open_trace(path, "wt") as destination:
        json.dump(trace, destination)


def complete_events(trace: dict) -> list[dict]:
    return [
        event for event in trace.get("traceEvents", []) if event.get("ph") == "X" and "dur" in event
    ]


def load_spans(pattern: str, prefixes: set[str] | None = None) -> list[tuple[float, float, str]]:
    paths = sorted(glob.glob(pattern))
    if not paths:
        return []
    return sorted(
        (event["ts"], event["dur"], event["name"])
        for event in complete_events(read_trace(paths[0]))
        if prefixes is None or event["name"].startswith(tuple(prefixes))
    )
