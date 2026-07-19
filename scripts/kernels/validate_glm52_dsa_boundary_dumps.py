#!/usr/bin/env python3
"""Validate real-weight GLM-5.2 DSA selection dumps at context boundaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _semantic_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["component"],
        row["layer"],
        row["forward_mode"],
        row["name"],
        row["occurrence"],
        row["process"],
    )


def _forward_key(key: tuple[Any, ...]) -> tuple[Any, ...]:
    return key[2], key[4], key[5]


def _read_rows(directory: Path) -> tuple[dict[tuple[Any, ...], Path], list[str]]:
    rows = {}
    failures = []
    for manifest in sorted(directory.rglob("manifest-*.jsonl")):
        for line_number, line in enumerate(
            manifest.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                key = _semantic_key(row)
                path = (manifest.parent / row["filename"]).resolve()
                path.relative_to(manifest.parent.resolve())
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                failures.append(f"{manifest}:{line_number}: invalid manifest row: {error}")
                continue
            if key in rows:
                failures.append(f"duplicate semantic key: {key}")
                continue
            rows[key] = path
    if not rows:
        failures.append("no debug tensor rows found")
    return rows, failures


def _tensor_key(
    forward: tuple[Any, ...], component: str, layer: int | None, name: str
) -> tuple[Any, ...]:
    mode, occurrence, process = forward
    return component, layer, mode, name, occurrence, process


def validate_boundary_dumps(
    directory: str | Path,
    *,
    index_topk: int,
    selection_layer: int,
    required_positions: set[int],
) -> dict[str, Any]:
    rows, failures = _read_rows(Path(directory))
    all_forwards = {_forward_key(key) for key in rows}
    complete_forwards = {
        _forward_key(key)
        for key in rows
        if key[0] == "debug_context" and key[3] == "forward_complete"
    }
    observed_positions: set[int] = set()
    active_query_count = 0
    clipped_query_count = 0
    validated_forward_count = 0

    required_tensors = (
        ("debug_context", None, "token_valid_mask"),
        ("debug_context", None, "token_positions"),
        ("dsa_selection", selection_layer, "logical_topk_ids"),
        ("dsa_selection", selection_layer, "selected_counts"),
        ("dsa_selection", selection_layer, "physical_slots"),
    )
    for forward in sorted(complete_forwards):
        paths = {}
        for component, layer, name in required_tensors:
            key = _tensor_key(forward, component, layer, name)
            if key not in rows:
                failures.append(f"forward {forward}: missing {component}:{layer}:{name}")
            else:
                paths[name] = rows[key]
        if len(paths) != len(required_tensors):
            continue

        try:
            mask = np.load(paths["token_valid_mask"], allow_pickle=False)
            positions = np.load(paths["token_positions"], allow_pickle=False)
            logical_ids = np.load(paths["logical_topk_ids"], allow_pickle=False)
            selected_counts = np.load(paths["selected_counts"], allow_pickle=False)
            physical_slots = np.load(paths["physical_slots"], allow_pickle=False)
        except (OSError, ValueError) as error:
            failures.append(f"forward {forward}: cannot load tensors: {error}")
            continue

        token_count = mask.size
        if mask.dtype != np.bool_ or mask.shape != (token_count,):
            failures.append(f"forward {forward}: token_valid_mask must be rank-1 bool")
            continue
        if positions.shape != (token_count,) or selected_counts.shape != (token_count,):
            failures.append(f"forward {forward}: position/count shapes do not match mask")
            continue
        if logical_ids.shape != (token_count, index_topk):
            failures.append(
                f"forward {forward}: logical_topk_ids shape {logical_ids.shape} is not "
                f"({token_count}, {index_topk})"
            )
            continue
        if physical_slots.shape != logical_ids.shape:
            failures.append(f"forward {forward}: physical slot shape does not match IDs")
            continue
        if not np.issubdtype(positions.dtype, np.integer) or not np.issubdtype(
            selected_counts.dtype, np.integer
        ):
            failures.append(f"forward {forward}: positions/counts must be integer tensors")
            continue

        validated_forward_count += 1
        for row_index in np.flatnonzero(mask):
            position = int(positions[row_index])
            count = int(selected_counts[row_index])
            expected_count = min(position + 1, index_topk)
            observed_positions.add(position)
            active_query_count += 1
            clipped_query_count += int(position + 1 > index_topk)
            prefix_ids = logical_ids[row_index, :count].astype(np.int64).tolist()
            prefix_slots = physical_slots[row_index, :count].astype(np.int64).tolist()
            suffix_ids = logical_ids[row_index, count:]
            suffix_slots = physical_slots[row_index, count:]

            label = f"forward {forward} row {row_index} position {position}"
            if count != expected_count:
                failures.append(f"{label}: selected_count {count} != {expected_count}")
            if len(set(prefix_ids)) != len(prefix_ids):
                failures.append(f"{label}: duplicate logical IDs in valid prefix")
            if any(token_id < 0 or token_id > position for token_id in prefix_ids):
                failures.append(f"{label}: logical ID is negative or points to a future token")
            if position + 1 <= index_topk and set(prefix_ids) != set(range(position + 1)):
                failures.append(f"{label}: non-truncated selection does not cover all causal IDs")
            if suffix_ids.size and not np.all(suffix_ids == -1):
                failures.append(f"{label}: logical ID suffix is not padded with -1")
            if len(set(prefix_slots)) != len(prefix_slots) or any(
                slot < 0 for slot in prefix_slots
            ):
                failures.append(f"{label}: physical slot prefix is negative or duplicated")
            if suffix_slots.size and not np.all(suffix_slots == 0):
                failures.append(f"{label}: physical slot suffix is not padded with 0")

    missing_positions = sorted(required_positions - observed_positions)
    if missing_positions:
        failures.append(f"required positions were not observed: {missing_positions}")
    report = {
        "passed": not failures and bool(complete_forwards),
        "index_topk": index_topk,
        "selection_layer": selection_layer,
        "tensor_count": len(rows),
        "complete_forward_count": len(complete_forwards),
        "dropped_incomplete_forward_count": len(all_forwards - complete_forwards),
        "validated_forward_count": validated_forward_count,
        "active_query_count": active_query_count,
        "clipped_query_count": clipped_query_count,
        "observed_position_min": min(observed_positions) if observed_positions else None,
        "observed_position_max": max(observed_positions) if observed_positions else None,
        "required_positions": sorted(required_positions),
        "missing_required_positions": missing_positions,
        "failures": failures,
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump-dir", type=Path, required=True)
    parser.add_argument("--index-topk", type=int, default=2048)
    parser.add_argument("--selection-layer", type=int, default=0)
    parser.add_argument("--require-positions", default="2046,2047,2048,3071")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    required_positions = {
        int(value.strip()) for value in args.require_positions.split(",") if value.strip()
    }
    report = validate_boundary_dumps(
        args.dump_dir,
        index_topk=args.index_topk,
        selection_layer=args.selection_layer,
        required_positions=required_positions,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
