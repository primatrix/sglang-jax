"""Derive request phase durations offline from serving logs, including older traces."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

_PHASES = {
    "queue_ms": ("enqueue_ns", "preprocess_start_ns"),
    "preprocess_ms": ("preprocess_start_ns", "preprocess_done_ns"),
    "encode_wait_ms": ("preprocess_done_ns", "transfer_reserve_start_ns"),
    "transfer_reserve_ms": ("transfer_reserve_start_ns", "encoder_dispatch_start_ns"),
    "encode_host_dispatch_ms": ("encoder_dispatch_start_ns", "encoder_dispatch_done_ns"),
    "transfer_handoff_ms": ("encoder_dispatch_done_ns", "transfer_enqueue_ns"),
    "transfer_queue_ms": ("transfer_enqueue_ns", "transfer_start_ns"),
    "publish_observed_ms": ("transfer_start_ns", "publish_done_ns"),
    "receive_ms": ("publish_done_ns", "receive_done_ns"),
    "mm_prepare_ms": ("receive_done_ns", "language_ready_ns"),
    "language_queue_ms": ("language_ready_ns", "language_prefill_start_ns"),
    "prefill_observed_ms": ("language_prefill_start_ns", "language_prefill_done_ns"),
    "server_ttft_ms": ("server_asgi_enter_ns", "server_first_content_send_done_ns"),
}


def pipeline_rows(log_text: str) -> list[dict]:
    rows = {}
    for line in log_text.splitlines():
        if "REQUEST-TIME-TRACE " in line:
            try:
                trace = json.loads(line.split("REQUEST-TIME-TRACE ", 1)[1])
            except json.JSONDecodeError:
                continue  # A live log may end in a partial record.
            row = {"req_id": trace["request_id"], **trace["timestamps_ns"]}
        elif "ENCODER-PIPELINE-TIME " in line:
            row = dict(
                re.findall(r"([a-z_]+)=([^\s]+)", line.split("ENCODER-PIPELINE-TIME ", 1)[1])
            )
        else:
            continue
        for old, new in (
            ("encode_start_ns", "encoder_dispatch_start_ns"),
            ("encode_done_ns", "encoder_dispatch_done_ns"),
        ):
            if old in row:
                row.setdefault(new, row[old])
        for name, (start, end) in _PHASES.items():
            if start in row and end in row:
                duration = int(row[end]) - int(row[start])
                if duration >= 0:
                    row[name] = duration / 1_000_000
        rows[row["req_id"]] = row
    return list(rows.values())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    args = parser.parse_args()
    rows = pipeline_rows(args.log.read_text())
    phases = {}
    for phase in _PHASES:
        values = [float(row[phase]) for row in rows if phase in row]
        if values:
            phases[phase] = {"n": len(values), "mean": statistics.fmean(values), "max": max(values)}
    print(json.dumps({"requests": len(rows), "phases_ms": phases}, indent=2))


if __name__ == "__main__":
    main()
