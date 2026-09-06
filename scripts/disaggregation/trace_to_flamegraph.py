#!/usr/bin/env python3
"""Fold sglang-jax jax.profiler traces into a single EPD flame graph.

Reads the per-tier ``*.trace.json.gz`` under a profiler dir (as produced by
scripts/disaggregation/profile_epd.py), builds a combined self-time
flame graph across tiers, and writes an SVG + a folded-stacks file. Also prints
a per-tier self-time breakdown so you can locate EPD latency from the terminal.

    python scripts/disaggregation/trace_to_flamegraph.py --profiler-dir /tmp/epd-sim-profile

The folded file works with Brendan Gregg's flamegraph.pl / inferno if you want
an alternative renderer.
"""

from __future__ import annotations

import argparse
import collections
import glob
import html
import os
from pathlib import Path

if __package__:
    from .trace_io import complete_events, read_trace
else:
    from trace_io import complete_events, read_trace


def frame_name(name: str) -> str:
    if name.startswith("$"):
        rest = name[1:]
        return rest.split(" ", 1)[1] if " " in rest else rest
    return name


# Leaf frames that are pure event-loop / queue / lock parking. Their self-time
# is idle wait, not work, and dominates a lightly-loaded async server — drop it
# so the flame graph highlights actual EPD work.
_IDLE_LEAVES = {
    "get",
    "acquire",
    "wait",
    "select",
    "poll",
    "kevent",
    "epoll_wait",
    "_run_once",
    "run_forever",
    "run_until_complete",
    "_worker",
    "_bootstrap",
    "_bootstrap_inner",
    "run",
    "wrapper",
}


def fold_trace(path: str, drop_idle: bool = True) -> collections.Counter:
    events = complete_events(read_trace(path))
    idle_leaves = _IDLE_LEAVES
    if any(e["name"].startswith("sim_device_compute:") for e in events):
        # New CPU-sim traces contain the actual background-device intervals.
        # The scheduler-side wait overlaps those intervals, so retaining both
        # double-counts modeled compute in a multi-thread self-time flame graph.
        idle_leaves = _IDLE_LEAVES | {"sim_device_wait"}
    # Keep python-tracer frames ($...) and named annotations; drop XLA op spam.
    events = [
        e
        for e in events
        if e["name"].startswith("$")
        or e["name"] in ("_forward_raw", "run_batch", "process_batch_result")
        or ":" not in e["name"][:3]
    ]
    events_by_thread = collections.defaultdict(list)
    for e in events:
        events_by_thread[(e["pid"], e["tid"])].append(e)
    folded: collections.Counter = collections.Counter()
    for thread_events in events_by_thread.values():
        thread_events.sort(key=lambda e: (e["ts"], -e["dur"]))
        nodes = []
        stack = []
        for e in thread_events:
            s = e["ts"]
            while stack and thread_events[stack[-1]]["ts"] + thread_events[stack[-1]]["dur"] <= s:
                stack.pop()
            nodes.append(
                {
                    "name": frame_name(e["name"]),
                    "dur": e["dur"],
                    "child": 0,
                    "parent": stack[-1] if stack else -1,
                }
            )
            stack.append(len(nodes) - 1)
        for n in nodes:
            if n["parent"] >= 0:
                nodes[n["parent"]]["child"] += n["dur"]
        for i, n in enumerate(nodes):
            self_time = n["dur"] - n["child"]
            if self_time <= 0:
                continue
            if drop_idle and n["name"] in idle_leaves:
                continue
            path = []
            j = i
            while j >= 0:
                path.append(nodes[j]["name"])
                j = nodes[j]["parent"]
            folded[";".join(reversed(path))] += self_time
    return folded


def _build_tree(folded):
    root = {"name": "root", "val": 0, "ch": {}}
    for stack, val in folded.items():
        node = root
        root["val"] += val
        for fr in stack.split(";"):
            node = node["ch"].setdefault(fr, {"name": fr, "val": 0, "ch": {}})
            node["val"] += val
    return root


TIER_COLORS = {"EPD": "#4477aa", "encoder": "#ee6677", "language": "#228833"}


def _color(name: str) -> str:
    if name in TIER_COLORS:
        return TIER_COLORS[name]
    h = hash(name) % 60
    return f"#cc{0x66 + h:02x}44"


def write_flamegraph(root, out: str, title: str, width: int = 1200, rowh: int = 16) -> float:
    total = root["val"]
    rows = []

    def layout(node, depth, x0):
        w = node["val"] / total * width
        if w >= 0.3:
            rows.append((x0, depth, w, node["name"], node["val"]))
        cx = x0
        for c in sorted(node["ch"].values(), key=lambda n: -n["val"]):
            layout(c, depth + 1, cx)
            cx += c["val"] / total * width

    layout(root, 0, 0)
    maxd = max((d for _, d, _, _, _ in rows), default=0) + 1
    height = maxd * rowh + 40
    p = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'font-family="monospace" font-size="10">',
        f'<text x="{width // 2}" y="14" text-anchor="middle" font-size="13" '
        f'font-weight="bold">{html.escape(title)}</text>',
    ]
    for x, d, w, name, val in rows:
        y = height - (d + 1) * rowh
        p.append(
            f'<rect x="{x:.1f}" y="{y}" width="{max(0.6, w - 0.6):.1f}" height="{rowh - 1}" '
            f'fill="{_color(name)}" stroke="#fff" stroke-width="0.3">'
            f"<title>{html.escape(name)} — {val / 1000:.1f} ms ({val / total * 100:.1f}%)</title></rect>"
        )
        if w > 34:
            p.append(
                f'<text x="{x + 2:.1f}" y="{y + rowh - 4}">{html.escape(name)[: int(w // 6)]}</text>'
            )
    p.append("</svg>")
    Path(out).write_text("\n".join(p))
    return total


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profiler-dir", default="/tmp/epd-sim-profile")
    ap.add_argument(
        "--out",
        default=None,
        help="SVG path (default <profiler-dir>/epd_flamegraph.svg)",
    )
    args = ap.parse_args()

    combined: collections.Counter = collections.Counter()
    per_tier = {}
    for tier_dir in sorted(glob.glob(os.path.join(args.profiler_dir, "*"))):
        if not os.path.isdir(tier_dir):
            continue
        traces = glob.glob(os.path.join(tier_dir, "plugins", "profile", "*", "*.trace.json.gz"))
        if not traces:
            continue
        tier = os.path.basename(tier_dir)
        folded = collections.Counter()
        for t in traces:
            folded.update(fold_trace(t))
        per_tier[tier] = folded
        for stack, us in folded.items():
            combined[f"EPD;{tier};{stack}"] += us

    if not combined:
        print(f"no traces found under {args.profiler_dir}")
        return 1

    out = args.out or os.path.join(args.profiler_dir, "epd_flamegraph.svg")
    total = write_flamegraph(_build_tree(combined), out, "EPD CPU-sim flame graph (self time)")
    folded_path = os.path.join(args.profiler_dir, "epd.folded")
    with open(folded_path, "w") as f:
        for s, v in sorted(combined.items(), key=lambda x: -x[1]):
            f.write(f"{s} {int(v)}\n")
    print(f"wrote {out}  (total {total / 1000:.0f} ms)")
    print(f"wrote {folded_path}")

    for tier, folded in per_tier.items():
        tot = sum(folded.values())
        agg = collections.Counter()
        for stack, us in folded.items():
            agg[stack.split(";")[-1]] += us
        print(f"\n{tier}: {tot / 1000:.0f} ms sampled self-time — top frames:")
        for name, us in agg.most_common(10):
            print(f"   {us / 1000:8.1f} ms  {name[:56]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
