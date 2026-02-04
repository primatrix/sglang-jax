#!/usr/bin/env python3
"""
Plot per-layer expert balance ratios over inference segments.

Input CSV is produced by expert balance debug (expert_balance_*.csv).
Outputs one PNG per layer with multiple lines: max/mean, min/mean, hot_topk, cold_topk, std/mean.
Also shades segments that look like Hotspot or Sparse Hotspot (heuristic).
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from typing import Iterable


def _parse_layers(spec: str, all_layers: Iterable[int]) -> list[int]:
    spec = spec.strip()
    if spec.lower() in ("all", "*", ""):
        return sorted(set(all_layers))
    layers: set[int] = set()
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    for part in parts:
        m = re.fullmatch(r"(\\d+)-(\\d+)", part)
        if m:
            start = int(m.group(1))
            end = int(m.group(2))
            if end < start:
                start, end = end, start
            layers.update(range(start, end + 1))
        else:
            layers.add(int(part))
    return sorted(layers)


def _read_rows(path: str):
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Plot per-layer expert balance metrics over segments."
    )
    parser.add_argument("csv_path", help="Path to expert_balance_*.csv")
    parser.add_argument(
        "--out-dir",
        default="expert_balance_plots",
        help="Output directory for PNGs.",
    )
    parser.add_argument(
        "--layers",
        default="all",
        help="Layer selection: 'all', '0,1,2', or '0-47'.",
    )
    parser.add_argument(
        "--x-axis",
        choices=["segment_idx", "tokens"],
        default="segment_idx",
        help="X axis uses segment index or cumulative tokens.",
    )
    parser.add_argument(
        "--skip-empty",
        action="store_true",
        help="Skip segments with has_data=0.",
    )
    parser.add_argument(
        "--hotspot-hot-multiple-threshold",
        type=float,
        default=3.0,
        help="hot_topk/mean >= threshold => Hotspot.",
    )
    parser.add_argument(
        "--sparse-cold-multiple-threshold",
        type=float,
        default=0.1,
        help="cold_topk/mean <= threshold => Sparse Hotspot (with active ratio check).",
    )
    parser.add_argument(
        "--sparse-active-ratio-threshold",
        type=float,
        default=0.9,
        help="active_experts/num_experts <= threshold => Sparse Hotspot.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=160,
        help="PNG DPI.",
    )
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - runtime import guard
        raise SystemExit(
            "matplotlib is required. Install it or run in an env that has it."
        ) from exc

    rows = _read_rows(args.csv_path)
    if not rows:
        raise SystemExit("No rows found in CSV.")

    all_layers = {int(r["layer"]) for r in rows}
    layers = _parse_layers(args.layers, all_layers)

    os.makedirs(args.out_dir, exist_ok=True)

    # Group rows by layer
    by_layer: dict[int, list[dict]] = {layer: [] for layer in layers}
    for r in rows:
        layer = int(r["layer"])
        if layer not in by_layer:
            continue
        if args.skip_empty and r.get("has_data", "1") in ("0", 0, "false", "False"):
            continue
        by_layer[layer].append(r)

    for layer, layer_rows in by_layer.items():
        if not layer_rows:
            continue
        layer_rows.sort(key=lambda x: int(x["segment_idx"]))

        seg_tokens = int(layer_rows[0]["segment_tokens"])
        seg_idx = [int(r["segment_idx"]) for r in layer_rows]
        x = [(idx * seg_tokens) if args.x_axis == "tokens" else idx for idx in seg_idx]
        mean_count = [float(r["mean_count"]) for r in layer_rows]
        min_count = [float(r["min_count"]) for r in layer_rows]
        max_count = [float(r["max_count"]) for r in layer_rows]
        hot_mult = [float(r["hot_topk_mean_multiple"]) for r in layer_rows]
        cold_mult = [float(r["cold_topk_mean_multiple"]) for r in layer_rows]
        active_experts = [int(r["active_experts"]) for r in layer_rows]
        num_experts = int(layer_rows[0]["num_experts"])

        fig, ax = plt.subplots(figsize=(12, 4))
        std_count = [float(r["std_count"]) for r in layer_rows]
        max_over_mean = [(mx / mu if mu > 0 else 0.0) for mx, mu in zip(max_count, mean_count)]
        min_over_mean = [(mn / mu if mu > 0 else 0.0) for mn, mu in zip(min_count, mean_count)]
        std_over_mean = [(sd / mu if mu > 0 else 0.0) for sd, mu in zip(std_count, mean_count)]
        ax.plot(x, max_over_mean, label="max/mean")
        ax.plot(x, min_over_mean, label="min/mean")
        ax.plot(x, hot_mult, label="hot_topk/mean")
        ax.plot(x, cold_mult, label="cold_topk/mean")
        ax.plot(x, std_over_mean, label="std/mean (cv)")

        active_ratio = [ae / num_experts for ae in active_experts]
        sparse_flags = [
            (ar <= args.sparse_active_ratio_threshold)
            and (cm <= args.sparse_cold_multiple_threshold)
            for ar, cm in zip(active_ratio, cold_mult)
        ]
        hotspot_flags = [(hm >= args.hotspot_hot_multiple_threshold) for hm in hot_mult]

        def _shade(flags, color, label):
            if not any(flags):
                return
            start = None
            for i, flag in enumerate(flags + [False]):
                if flag and start is None:
                    start = i
                if start is not None and not flag:
                    if args.x_axis == "tokens":
                        x0 = seg_idx[start] * seg_tokens
                        x1 = (seg_idx[i - 1] + 1) * seg_tokens
                    else:
                        x0 = seg_idx[start] - 0.5
                        x1 = seg_idx[i - 1] + 0.5
                    ax.axvspan(x0, x1, color=color, alpha=0.12, label=label)
                    start = None

        _shade(hotspot_flags, "#f6a600", "Hotspot")
        _shade(sparse_flags, "#d9534f", "Sparse Hotspot")

        ax.set_title(f"Layer {layer}")
        ax.set_xlabel("tokens" if args.x_axis == "tokens" else "segment_idx")
        ax.set_ylabel("multiple of mean")
        ax.grid(True, alpha=0.3)
        handles, labels = ax.get_legend_handles_labels()
        # de-duplicate legend labels
        seen = set()
        uniq_handles = []
        uniq_labels = []
        for h, l in zip(handles, labels):
            if l not in seen:
                uniq_handles.append(h)
                uniq_labels.append(l)
                seen.add(l)
        ax.legend(uniq_handles, uniq_labels, loc="upper right")
        fig.text(
            0.99,
            0.01,
            "Ratios: max/mean (busiest expert), min/mean (coldest expert), "
            "hot_topk/mean (avg of top-k experts), cold_topk/mean (avg of coldest top-k), "
            "std/mean (cv). Shading shows Hotspot/Sparse Hotspot heuristics.",
            ha="right",
            va="bottom",
            fontsize=8,
            wrap=True,
        )
        fig.tight_layout()

        out_path = os.path.join(args.out_dir, f"layer_{layer:03d}.png")
        fig.savefig(out_path, dpi=args.dpi)
        plt.close(fig)

    print(f"Saved plots to: {args.out_dir}")


if __name__ == "__main__":
    main()
