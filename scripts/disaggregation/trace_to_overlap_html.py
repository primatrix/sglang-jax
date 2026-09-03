#!/usr/bin/env python3
"""Render an aligned EPD overlap timeline from CPU-simulation wall-clock logs.

The regular flame graph aggregates self time and therefore cannot show overlap.
This script uses the request pipeline, simulated transfer, and simulated device
intervals emitted during one aligned benchmark window to produce:

* ``epd_overlap.html``: compact multi-request timeline.
* ``epd_overlap.trace.json``: the same spans for Perfetto/Chrome tracing.
* ``overlap-summary.json``: pairwise overlap durations and coverage ratios.

Only the Python standard library is required.
"""

from __future__ import annotations

import argparse
import html
import json
import re
from dataclasses import dataclass, replace
from pathlib import Path

_PIPELINE_RE = re.compile(r"ENCODER-PIPELINE-TIME (?P<body>[^\n]+)")
_TRANSFER_RE = re.compile(r"ENCODER-RAIDEN-INFLIGHT (?P<body>[^\n]+)")
_DEVICE_RE = re.compile(r"SIM-DEVICE-COMPUTE (?P<body>[^\n]+)")
_KEY_VALUE_RE = re.compile(r"([a-z_]+)=([^\s]+)")
_TRANSFER_TERMINALS = frozenset({"sent", "failed", "release", "close"})


@dataclass(frozen=True)
class Span:
    start_ns: int
    end_ns: int
    stage: str
    label: str
    lane: str
    detail: str = ""

    @property
    def duration_ns(self) -> int:
        return max(0, self.end_ns - self.start_ns)


def _rows(pattern: re.Pattern, text: str) -> list[dict[str, str]]:
    return [dict(_KEY_VALUE_RE.findall(match.group("body"))) for match in pattern.finditer(text)]


def _parse_pipeline(path: Path, start_ns: int, end_ns: int) -> list[dict[str, str]]:
    required = (
        "req_id",
        "enqueue_ns",
        "dequeue_ns",
        "preprocess_start_ns",
        "preprocess_done_ns",
        "encode_start_ns",
        "encode_done_ns",
        "publish_done_ns",
        "receive_done_ns",
        "language_ready_ns",
        "language_prefill_start_ns",
        "language_prefill_done_ns",
    )
    parsed = []
    for row in _rows(_PIPELINE_RE, path.read_text()):
        if not all(key in row for key in required):
            continue
        enqueue_ns = int(row["enqueue_ns"])
        if start_ns <= enqueue_ns <= end_ns:
            parsed.append(row)
    return sorted(parsed, key=lambda row: int(row["enqueue_ns"]))


def _parse_transfers(paths: list[Path], start_ns: int, end_ns: int) -> list[Span]:
    active: dict[tuple[str, str], tuple[int, dict[str, str]]] = {}
    spans = []
    for path in paths:
        for row in _rows(_TRANSFER_RE, path.read_text()):
            try:
                event_ns = int(row["time_ns"])
                transfer_id = row["transfer_id"]
            except (KeyError, ValueError):
                continue
            key = (path.name, transfer_id)
            event = row.get("event")
            if event == "start":
                active[key] = (event_ns, row)
            elif event in _TRANSFER_TERMINALS and key in active:
                transfer_start_ns, start_row = active.pop(key)
                if event_ns < start_ns or transfer_start_ns > end_ns:
                    continue
                spans.append(
                    Span(
                        start_ns=max(start_ns, transfer_start_ns),
                        end_ns=min(end_ns, event_ns),
                        stage="transfer",
                        label="transfer",
                        lane=path.stem,
                        detail=(f"{transfer_id} · group={start_row.get('group_size', '?')}"),
                    )
                )
    packed = []
    for base_lane in dict.fromkeys(span.lane for span in spans):
        lane_ends: list[int] = []
        lane_spans = sorted(
            (span for span in spans if span.lane == base_lane),
            key=lambda span: span.start_ns,
        )
        for span in lane_spans:
            lane_index = next(
                (
                    index
                    for index, lane_end_ns in enumerate(lane_ends)
                    if lane_end_ns <= span.start_ns
                ),
                len(lane_ends),
            )
            if lane_index == len(lane_ends):
                lane_ends.append(span.end_ns)
            else:
                lane_ends[lane_index] = span.end_ns
            packed.append(replace(span, lane=f"{base_lane} ch{lane_index + 1}"))
    return sorted(packed, key=lambda span: span.start_ns)


def _parse_device(path: Path, start_ns: int, end_ns: int) -> list[Span]:
    spans = []
    for row in _rows(_DEVICE_RE, path.read_text()):
        try:
            device_start_ns = int(row["start_ns"])
            device_end_ns = int(row["end_ns"])
        except (KeyError, ValueError):
            continue
        if device_end_ns < start_ns or device_start_ns > end_ns:
            continue
        kind = row.get("kind", "unknown")
        spans.append(
            Span(
                start_ns=max(start_ns, device_start_ns),
                end_ns=min(end_ns, device_end_ns),
                stage=kind,
                label=f"{kind} b{row.get('bid', '?')}",
                lane="language device",
                detail=f"batch={row.get('batch_size', '?')}",
            )
        )
    return sorted(spans, key=lambda span: span.start_ns)


def _request_spans(rows: list[dict[str, str]]) -> list[Span]:
    phase_specs = (
        ("queue", "enqueue_ns", "preprocess_start_ns", "queue / stage wait"),
        ("preprocess", "preprocess_start_ns", "preprocess_done_ns", "preprocess"),
        ("queue", "preprocess_done_ns", "encode_start_ns", "encoder wait"),
        ("encoder", "encode_start_ns", "encode_done_ns", "encoder"),
        ("publish", "encode_done_ns", "publish_done_ns", "publish"),
        ("pickup", "publish_done_ns", "receive_done_ns", "transfer + pickup wait"),
        ("prepare", "receive_done_ns", "language_ready_ns", "MM prepare"),
        ("queue", "language_ready_ns", "language_prefill_start_ns", "language queue"),
        (
            "prefill_host",
            "language_prefill_start_ns",
            "language_prefill_done_ns",
            "prefill host dispatch/result",
        ),
    )
    spans = []
    for index, row in enumerate(rows, start=1):
        req_id = row["req_id"]
        lane = f"R{index} {req_id[:8]}"
        for stage, start_key, end_key, label in phase_specs:
            phase_start_ns = int(row[start_key])
            phase_end_ns = int(row[end_key])
            if phase_end_ns <= phase_start_ns:
                continue
            spans.append(
                Span(
                    start_ns=phase_start_ns,
                    end_ns=phase_end_ns,
                    stage=stage,
                    label=label,
                    lane=lane,
                    detail=req_id,
                )
            )
    return spans


def _merge(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[list[int]] = []
    for start_ns, end_ns in sorted(intervals):
        if end_ns <= start_ns:
            continue
        if not merged or start_ns > merged[-1][1]:
            merged.append([start_ns, end_ns])
        else:
            merged[-1][1] = max(merged[-1][1], end_ns)
    return [(start_ns, end_ns) for start_ns, end_ns in merged]


def _intersection(
    left: list[tuple[int, int]], right: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    left = _merge(left)
    right = _merge(right)
    intersections = []
    left_idx = right_idx = 0
    while left_idx < len(left) and right_idx < len(right):
        start_ns = max(left[left_idx][0], right[right_idx][0])
        end_ns = min(left[left_idx][1], right[right_idx][1])
        if end_ns > start_ns:
            intersections.append((start_ns, end_ns))
        if left[left_idx][1] <= right[right_idx][1]:
            left_idx += 1
        else:
            right_idx += 1
    return intersections


def _duration_ns(intervals: list[tuple[int, int]]) -> int:
    return sum(end_ns - start_ns for start_ns, end_ns in _merge(intervals))


def _analyse(request_spans: list[Span], transfers: list[Span], devices: list[Span]) -> dict:
    encoder = [(span.start_ns, span.end_ns) for span in request_spans if span.stage == "encoder"]
    transfer = [(span.start_ns, span.end_ns) for span in transfers]
    language = [(span.start_ns, span.end_ns) for span in devices]
    prefill = [(span.start_ns, span.end_ns) for span in devices if span.stage == "prefill"]
    decode = [(span.start_ns, span.end_ns) for span in devices if span.stage == "decode"]
    encoder_language = _intersection(encoder, language)
    transfer_language = _intersection(transfer, language)
    upstream_language = _intersection(_merge(encoder + transfer), language)
    any_cross_stage = _merge(encoder_language + transfer_language)

    encoder_ns = _duration_ns(encoder)
    transfer_ns = _duration_ns(transfer)
    language_ns = _duration_ns(language)
    encoder_language_ns = _duration_ns(encoder_language)
    transfer_language_ns = _duration_ns(transfer_language)
    upstream_language_ns = _duration_ns(upstream_language)

    def ms(value: int) -> float:
        return value / 1e6

    def pct(part: int, whole: int) -> float:
        return 100.0 * part / whole if whole else 0.0

    return {
        "schema_version": 1,
        "stage_time_ms": {
            "encoder": ms(encoder_ns),
            "transfer": ms(transfer_ns),
            "language_prefill": ms(_duration_ns(prefill)),
            "language_decode": ms(_duration_ns(decode)),
            "language_total": ms(language_ns),
        },
        "overlap_ms": {
            "encoder_language": ms(encoder_language_ns),
            "transfer_language": ms(transfer_language_ns),
            "upstream_language": ms(upstream_language_ns),
            "any_cross_stage": ms(_duration_ns(any_cross_stage)),
        },
        "coverage_pct": {
            "encoder_hidden_by_language": pct(encoder_language_ns, encoder_ns),
            "transfer_hidden_by_language": pct(transfer_language_ns, transfer_ns),
            "language_overlapped_by_upstream": pct(upstream_language_ns, language_ns),
        },
        "counts": {
            "requests": len({span.lane for span in request_spans}),
            "transfers": len(transfers),
            "prefill_batches": sum(span.stage == "prefill" for span in devices),
            "decode_batches": sum(span.stage == "decode" for span in devices),
        },
    }


def _trace_event(span: Span, *, origin_ns: int, pid: int, tid: int) -> dict:
    return {
        "name": span.label,
        "cat": span.stage,
        "ph": "X",
        "ts": (span.start_ns - origin_ns) / 1000,
        "dur": span.duration_ns / 1000,
        "pid": pid,
        "tid": tid,
        "args": {"detail": span.detail},
    }


def _write_trace(
    path: Path,
    request_spans: list[Span],
    transfers: list[Span],
    devices: list[Span],
    origin_ns: int,
) -> None:
    events = []
    request_lanes = list(dict.fromkeys(span.lane for span in request_spans))
    for index, lane in enumerate(request_lanes, start=1):
        events.append(
            {
                "ph": "M",
                "pid": 100,
                "tid": index,
                "name": "thread_name",
                "args": {"name": lane},
            }
        )
        events.extend(
            _trace_event(span, origin_ns=origin_ns, pid=100, tid=index)
            for span in request_spans
            if span.lane == lane
        )
    events.append(
        {
            "ph": "M",
            "pid": 100,
            "name": "process_name",
            "args": {"name": "EPD requests"},
        }
    )

    transfer_lanes = list(dict.fromkeys(span.lane for span in transfers))
    for index, lane in enumerate(transfer_lanes, start=1):
        events.append(
            {
                "ph": "M",
                "pid": 200,
                "tid": index,
                "name": "thread_name",
                "args": {"name": lane},
            }
        )
        events.extend(
            _trace_event(span, origin_ns=origin_ns, pid=200, tid=index)
            for span in transfers
            if span.lane == lane
        )
    events.append(
        {
            "ph": "M",
            "pid": 200,
            "name": "process_name",
            "args": {"name": "Raiden transfer"},
        }
    )

    events.append(
        {
            "ph": "M",
            "pid": 300,
            "tid": 1,
            "name": "thread_name",
            "args": {"name": "simulated device"},
        }
    )
    events.append({"ph": "M", "pid": 300, "name": "process_name", "args": {"name": "Language"}})
    events.extend(_trace_event(span, origin_ns=origin_ns, pid=300, tid=1) for span in devices)
    path.write_text(json.dumps({"displayTimeUnit": "ms", "traceEvents": events}))


def _write_html(
    path: Path,
    request_spans: list[Span],
    transfers: list[Span],
    devices: list[Span],
    analysis: dict,
    *,
    max_requests: int,
) -> None:
    request_lanes = list(dict.fromkeys(span.lane for span in request_spans))[:max_requests]
    visible_requests = [span for span in request_spans if span.lane in request_lanes]
    visible = visible_requests + transfers + devices
    if not visible:
        raise ValueError("no overlap spans found")
    origin_ns = min(span.start_ns for span in visible)
    end_ns = max(span.end_ns for span in visible)
    duration_ns = max(1, end_ns - origin_ns)

    stage_colors = {
        "queue": "var(--wait)",
        "preprocess": "var(--preprocess)",
        "encoder": "var(--encoder)",
        "publish": "var(--publish)",
        "pickup": "var(--pickup)",
        "prepare": "var(--prepare)",
        "prefill_host": "var(--host)",
        "transfer": "var(--transfer)",
        "prefill": "var(--prefill)",
        "decode": "var(--decode)",
    }

    def bars(spans: list[Span]) -> str:
        rendered = []
        for span in spans:
            left = 100 * (span.start_ns - origin_ns) / duration_ns
            width = max(0.12, 100 * span.duration_ns / duration_ns)
            tooltip = f"{span.label}: {span.duration_ns / 1e6:.3f} ms" + (
                f" · {span.detail}" if span.detail else ""
            )
            rendered.append(
                f'<span class="bar" style="left:{left:.5f}%;width:{width:.5f}%;'
                f'background:{stage_colors.get(span.stage, "var(--other)")}" '
                f'title="{html.escape(tooltip)}"><i>{html.escape(span.label)}</i></span>'
            )
        return "".join(rendered)

    rows = []
    rows.append(
        '<div class="lane resource"><strong>Language device</strong><div class="track">'
        + bars(devices)
        + "</div></div>"
    )
    transfer_lanes = list(dict.fromkeys(span.lane for span in transfers))
    for lane in transfer_lanes:
        rows.append(
            f'<div class="lane resource"><strong>{html.escape(lane)}</strong><div class="track">'
            + bars([span for span in transfers if span.lane == lane])
            + "</div></div>"
        )
    for lane in request_lanes:
        rows.append(
            f'<div class="lane"><strong>{html.escape(lane)}</strong><div class="track">'
            + bars([span for span in visible_requests if span.lane == lane])
            + "</div></div>"
        )

    ticks = "".join(
        f'<span style="left:{index * 10}%">{duration_ns / 1e6 * index / 10:.0f}</span>'
        for index in range(11)
    )
    coverage = analysis["coverage_pct"]
    overlap = analysis["overlap_ms"]
    legend = (
        ("Encoder", "encoder"),
        ("Actual transfer", "transfer"),
        ("Transfer + pickup wait", "pickup"),
        ("Language prefill", "prefill"),
        ("Language decode", "decode"),
        ("Host/preprocess/wait", "preprocess"),
    )
    legend_html = "".join(
        f'<span><i style="background:{stage_colors[stage]}"></i>{label}</span>'
        for label, stage in legend
    )
    document = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>EPD overlap timeline</title>
<style>
:root{{--bg:#fff;--fg:#172033;--muted:#667085;--line:#d7dce5;--track:#eef1f5;
--encoder:#d95f5f;--transfer:#e09b35;--pickup:#f1c36d;--prefill:#387bd3;
--decode:#2d9b69;--preprocess:#8a66c2;--wait:#a9b1bd;--publish:#d67f4a;
--prepare:#5f9ca8;--host:#7f91ab;--other:#777}}
@media(prefers-color-scheme:dark){{:root{{--bg:#111827;--fg:#edf2f7;--muted:#aab4c3;
--line:#374151;--track:#202938;--encoder:#e77474;--transfer:#edaa48;--pickup:#a9782f;
--prefill:#5b98e8;--decode:#48b882;--preprocess:#a783d7;--wait:#667085;
--publish:#dd8c5c;--prepare:#75b5bf;--host:#91a4c0;--other:#999}}}}
body{{margin:22px;background:var(--bg);color:var(--fg);font:13px/1.35 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
h1{{font-size:18px;font-weight:600;margin:0 0 12px}} .summary{{display:flex;gap:22px;flex-wrap:wrap;margin-bottom:14px}}
.metric b{{display:block;font-size:18px;font-weight:600}} .metric small{{color:var(--muted)}}
.legend{{display:flex;gap:14px;flex-wrap:wrap;margin:10px 0;color:var(--muted)}}
.legend i{{display:inline-block;width:11px;height:11px;margin-right:5px;vertical-align:-1px}}
.chart{{min-width:760px}} .lane{{display:grid;grid-template-columns:145px 1fr;align-items:center;min-height:28px}}
.lane strong{{font-weight:500;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;padding-right:8px}}
.lane.resource{{min-height:34px}} .track{{position:relative;height:22px;background:var(--track)}}
.resource .track{{height:28px}} .bar{{position:absolute;top:2px;height:18px;overflow:hidden;white-space:nowrap}}
.resource .bar{{height:24px}} .bar i{{font-style:normal;font-size:11px;color:#fff;padding:2px 4px;line-height:18px}}
.ruler{{position:relative;height:24px;margin-left:145px;width:calc(100% - 145px);border-top:1px solid var(--line);color:var(--muted)}}
.ruler span{{position:absolute;transform:translateX(-50%);padding-top:3px}} .note{{color:var(--muted);margin-top:12px}}
.legend span{{white-space:nowrap}}
@media(max-width:820px){{body{{margin:12px}}.chart{{min-width:0}}.lane{{grid-template-columns:110px 1fr}}
.ruler{{margin-left:110px;width:calc(100% - 110px)}}.bar i{{display:none}}.legend{{gap:8px 12px}}}}
</style></head><body>
<h1>EPD overlap timeline</h1>
<div class="summary">
 <div class="metric"><b>{overlap["encoder_language"]:.2f} ms</b><small>Encoder ↔ Language overlap</small></div>
 <div class="metric"><b>{coverage["encoder_hidden_by_language"]:.1f}%</b><small>Encoder hidden by Language</small></div>
 <div class="metric"><b>{coverage["language_overlapped_by_upstream"]:.1f}%</b><small>Language overlapped by E/T</small></div>
</div>
<div class="legend">{legend_html}</div>
<div class="chart-wrap"><div class="chart">{"".join(rows)}<div class="ruler">{ticks}</div></div></div>
<div class="note">Shared wall-clock axis, milliseconds from the first visible event. “Actual transfer” is sender start→completion; the wider request-level amber span also includes scheduler pickup delay. Showing {len(request_lanes)} request(s).</div>
</body></html>"""
    path.write_text(document)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiler-dir", default="/tmp/epd-sim-profile")
    parser.add_argument("--out", default=None, help="HTML output path")
    parser.add_argument("--max-requests", type=int, default=32)
    args = parser.parse_args()

    profiler_dir = Path(args.profiler_dir)
    aligned_path = profiler_dir / "aligned-summary.json"
    if not aligned_path.exists():
        parser.error(f"missing {aligned_path}; run the aligned EPD benchmark first")
    aligned = json.loads(aligned_path.read_text())
    start_ns = int(aligned["measurement_start_ns"])
    end_ns = int(aligned["measurement_end_ns"])
    pipeline = _parse_pipeline(profiler_dir / "language.log", start_ns, end_ns)
    request_spans = _request_spans(pipeline)
    transfers = _parse_transfers(sorted(profiler_dir.glob("encoder_*.log")), start_ns, end_ns)
    devices = _parse_device(profiler_dir / "language.log", start_ns, end_ns)
    if not devices:
        parser.error(
            "no SIM-DEVICE-COMPUTE intervals found; rerun CPU simulation with the updated model"
        )

    analysis = _analyse(request_spans, transfers, devices)
    analysis.update(
        {
            "measurement_start_ns": start_ns,
            "measurement_end_ns": end_ns,
            "workload": aligned.get("workload", {}),
        }
    )
    summary_path = profiler_dir / "overlap-summary.json"
    summary_path.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n")

    html_path = Path(args.out) if args.out else profiler_dir / "epd_overlap.html"
    _write_html(
        html_path,
        request_spans,
        transfers,
        devices,
        analysis,
        max_requests=max(1, args.max_requests),
    )
    trace_path = profiler_dir / "epd_overlap.trace.json"
    origin_ns = min(span.start_ns for span in request_spans + transfers + devices)
    _write_trace(trace_path, request_spans, transfers, devices, origin_ns)

    coverage = analysis["coverage_pct"]
    overlap = analysis["overlap_ms"]
    print(f"wrote {html_path}")
    print(f"wrote {trace_path}")
    print(f"wrote {summary_path}")
    print(
        "overlap: "
        f"encoder↔language={overlap['encoder_language']:.3f} ms "
        f"({coverage['encoder_hidden_by_language']:.1f}% of encoder), "
        f"transfer↔language={overlap['transfer_language']:.3f} ms "
        f"({coverage['transfer_hidden_by_language']:.1f}% of transfer), "
        f"language covered by upstream={coverage['language_overlapped_by_upstream']:.1f}%"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
