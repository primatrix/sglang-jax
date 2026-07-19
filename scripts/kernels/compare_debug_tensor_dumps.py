#!/usr/bin/env python3
"""Compare semantic JAX tensor dumps from two debug artifact trees."""

from __future__ import annotations

import argparse
import json
import math
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

try:
    import ml_dtypes  # noqa: F401  # Register bfloat16 with NumPy.
except ImportError:  # pragma: no cover - JAX environments provide ml_dtypes.
    pass


KEY_FIELDS = (
    "component",
    "layer",
    "forward_mode",
    "name",
    "occurrence",
    "process",
)
EMPTY_METRICS = {
    "max_abs": None,
    "mean_abs": None,
    "p99_abs": None,
    "cosine": None,
    "topk_overlap": None,
}


def _key_dict(key: tuple[Any, ...]) -> dict[str, Any]:
    return dict(zip(KEY_FIELDS, key, strict=True))


def _key_sort_value(key: tuple[Any, ...]) -> tuple[Any, ...]:
    component, layer, forward_mode, name, occurrence, process = key
    return (
        component,
        layer is not None,
        -1 if layer is None else layer,
        forward_mode,
        name,
        occurrence,
        process,
    )


def _forward_key(key: tuple[Any, ...]) -> tuple[Any, ...]:
    return key[2], key[4], key[5]


def _filter_complete_forwards(
    rows: dict[tuple[Any, ...], dict[str, Any]], marker: tuple[str, str]
) -> tuple[dict[tuple[Any, ...], dict[str, Any]], dict[str, int]]:
    component, name = marker
    all_forwards = {_forward_key(key) for key in rows}
    completed_forwards = {
        _forward_key(key)
        for key in rows
        if key[0] == component and key[3] == name
    }
    filtered = {
        key: record
        for key, record in rows.items()
        if _forward_key(key) in completed_forwards
    }
    return filtered, {
        "completed_forward_count": len(completed_forwards),
        "dropped_incomplete_forward_count": len(all_forwards - completed_forwards),
        "dropped_tensor_count": len(rows) - len(filtered),
    }


def _validate_key(row: dict[str, Any]) -> tuple[Any, ...]:
    missing = [field for field in KEY_FIELDS if field not in row]
    if missing:
        raise ValueError(f"missing semantic key fields: {', '.join(missing)}")
    component = row["component"]
    layer = row["layer"]
    forward_mode = row["forward_mode"]
    name = row["name"]
    occurrence = row["occurrence"]
    process = row["process"]
    if not isinstance(component, str) or not component:
        raise ValueError("component must be a non-empty string")
    if layer is not None and (not isinstance(layer, int) or isinstance(layer, bool)):
        raise ValueError("layer must be an integer or null")
    if not isinstance(forward_mode, str) or not forward_mode:
        raise ValueError("forward_mode must be a non-empty string")
    if not isinstance(name, str) or not name:
        raise ValueError("name must be a non-empty string")
    if not isinstance(occurrence, int) or isinstance(occurrence, bool) or occurrence < 0:
        raise ValueError("occurrence must be a non-negative integer")
    if not isinstance(process, int) or isinstance(process, bool) or process < 0:
        raise ValueError("process must be a non-negative integer")
    return component, layer, forward_mode, name, occurrence, process


def _manifest_label(root: Path, manifest: Path) -> str:
    return manifest.relative_to(root).as_posix()


def _read_manifests(
    directory: Path, label: str
) -> tuple[dict[tuple[Any, ...], dict[str, Any]], list[str], list[tuple[Any, ...]]]:
    rows: dict[tuple[Any, ...], dict[str, Any]] = {}
    errors: list[str] = []
    duplicate_keys: list[tuple[Any, ...]] = []
    if not directory.is_dir():
        return rows, [f"{label}: dump directory does not exist or is not a directory"], []

    manifests = sorted(
        directory.rglob("manifest-*.jsonl"),
        key=lambda path: path.relative_to(directory).as_posix(),
    )
    if not manifests:
        return rows, [f"{label}: no manifest-*.jsonl files found"], []

    for manifest in manifests:
        manifest_name = _manifest_label(directory, manifest)
        if manifest.is_symlink():
            errors.append(
                f"{label}:{manifest_name}: manifest path must not be a symlink"
            )
            continue
        try:
            lines = manifest.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            errors.append(f"{label}:{manifest_name}: manifest is not valid UTF-8")
            continue
        except OSError as error:
            errors.append(f"{label}:{manifest_name}: cannot read manifest: {error}")
            continue
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            location = f"{label}:{manifest_name}:{line_number}"
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                errors.append(f"{location}: invalid JSON: {error.msg}")
                continue
            if not isinstance(row, dict):
                errors.append(f"{location}: manifest row must be a JSON object")
                continue
            try:
                key = _validate_key(row)
            except ValueError as error:
                errors.append(f"{location}: {error}")
                continue
            filename = row.get("filename")
            shape = row.get("shape")
            dtype = row.get("dtype")
            if not isinstance(filename, str) or not filename:
                errors.append(f"{location}: filename must be a non-empty string")
                continue
            if (
                not isinstance(shape, list)
                or not all(
                    isinstance(dimension, int)
                    and not isinstance(dimension, bool)
                    and dimension >= 0
                    for dimension in shape
                )
            ):
                errors.append(f"{location}: shape must be a list of non-negative integers")
                continue
            if not isinstance(dtype, str) or not dtype:
                errors.append(f"{location}: dtype must be a non-empty string")
                continue
            array_path = manifest.parent / filename
            if array_path.is_symlink():
                errors.append(f"{location}: tensor path must not be a symlink")
                continue
            try:
                resolved_array_path = array_path.resolve()
                resolved_array_path.relative_to(manifest.parent.resolve())
            except (OSError, ValueError):
                errors.append(f"{location}: filename escapes the manifest directory")
                continue
            if key in rows:
                rendered_key = json.dumps(_key_dict(key), sort_keys=True)
                errors.append(f"{label}: duplicate semantic key {rendered_key}")
                duplicate_keys.append(key)
                continue
            rows[key] = {
                "array_path": resolved_array_path,
                "declared_shape": tuple(shape),
                "declared_dtype": dtype,
                "location": location,
            }
    return rows, errors, duplicate_keys


def _load_array(record: dict[str, Any], label: str) -> tuple[np.ndarray | None, list[str]]:
    errors: list[str] = []
    try:
        array = np.load(record["array_path"], allow_pickle=False)
    except EOFError:
        return None, [f"{label}: cannot load tensor: unexpected end of file"]
    except zipfile.BadZipFile:
        return None, [f"{label}: cannot load tensor: invalid ZIP archive"]
    except (OSError, ValueError) as error:
        return None, [f"{label}: cannot load tensor: {error}"]
    if not isinstance(array, np.ndarray):
        loadable_type = type(array).__name__
        close = getattr(array, "close", None)
        if callable(close):
            try:
                close()
            except Exception as error:  # pragma: no cover - defensive for third-party loaders.
                return None, [
                    f"{label}: loaded object is not an ndarray: {loadable_type}; "
                    f"close failed: {type(error).__name__}"
                ]
        return None, [
            f"{label}: loaded object is not an ndarray: {loadable_type}"
        ]
    if tuple(array.shape) != record["declared_shape"]:
        errors.append(
            f"{label}: manifest shape {list(record['declared_shape'])} does not match "
            f"tensor shape {list(array.shape)}"
        )
    if str(array.dtype) != record["declared_dtype"]:
        errors.append(
            f"{label}: manifest dtype {record['declared_dtype']} does not match "
            f"tensor dtype {array.dtype}"
        )
    return array, errors


def _as_metric_values(array: np.ndarray) -> np.ndarray:
    dtype_name = str(array.dtype)
    if dtype_name == "bfloat16":
        return array.astype(np.float32)
    if np.issubdtype(array.dtype, np.complexfloating):
        raise TypeError("complex tensors are unsupported")
    try:
        numeric = np.issubdtype(array.dtype, np.number) or np.issubdtype(
            array.dtype, np.bool_
        )
    except TypeError:
        numeric = dtype_name == "bfloat16"
    if not numeric:
        raise TypeError(f"unsupported non-numeric dtype {array.dtype}")
    if np.issubdtype(array.dtype, np.floating) and array.dtype.itemsize < 4:
        return array.astype(np.float32)
    return array


def _topk_overlap(
    candidate: np.ndarray,
    baseline: np.ndarray,
    *,
    tensor_name: str,
    candidate_counts: np.ndarray | None = None,
    baseline_counts: np.ndarray | None = None,
) -> float | None:
    try:
        integer_ids = np.issubdtype(candidate.dtype, np.integer)
    except TypeError:
        integer_ids = False
    name = tensor_name.lower()
    if not integer_ids or candidate.ndim == 0 or not (
        "topk" in name or name.endswith("_ids")
    ):
        return None
    width = candidate.shape[-1]
    if width == 0:
        return None
    candidate_rows = candidate.reshape(-1, width)
    baseline_rows = baseline.reshape(-1, width)
    if tensor_name == "logical_topk_ids":
        if candidate_counts is None or baseline_counts is None:
            raise ValueError(
                "logical_topk_ids requires matching selected_counts tensors"
            )
        for label, counts in (
            ("candidate", candidate_counts),
            ("baseline", baseline_counts),
        ):
            try:
                integer_counts = np.issubdtype(counts.dtype, np.integer)
            except TypeError:
                integer_counts = False
            if not integer_counts:
                raise ValueError(f"{label} selected_counts must have integer dtype")
            if counts.size != candidate_rows.shape[0]:
                raise ValueError(
                    f"{label} selected_counts shape is incompatible with logical_topk_ids"
                )
        candidate_widths = candidate_counts.reshape(-1)
        baseline_widths = baseline_counts.reshape(-1)
    else:
        candidate_widths = np.full(candidate_rows.shape[0], width, dtype=np.int64)
        baseline_widths = np.full(baseline_rows.shape[0], width, dtype=np.int64)

    overlaps = []
    for candidate_row, baseline_row, candidate_width, baseline_width in zip(
        candidate_rows,
        baseline_rows,
        candidate_widths,
        baseline_widths,
        strict=True,
    ):
        candidate_width = int(candidate_width)
        baseline_width = int(baseline_width)
        if not 0 <= candidate_width <= width:
            raise ValueError(
                f"candidate selected_count {candidate_width} is outside [0, {width}]"
            )
        if not 0 <= baseline_width <= width:
            raise ValueError(
                f"baseline selected_count {baseline_width} is outside [0, {width}]"
            )
        candidate_ids = candidate_row[:candidate_width].tolist()
        baseline_ids = baseline_row[:baseline_width].tolist()
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("candidate valid Top-K prefix contains duplicate IDs")
        if len(set(baseline_ids)) != len(baseline_ids):
            raise ValueError("baseline valid Top-K prefix contains duplicate IDs")
        denominator = max(candidate_width, baseline_width)
        overlaps.append(
            1.0
            if denominator == 0
            else len(set(candidate_ids) & set(baseline_ids)) / denominator
        )
    return float(min(overlaps)) if overlaps else None


def _metrics(
    candidate: np.ndarray,
    baseline: np.ndarray,
    *,
    tensor_name: str,
    candidate_counts: np.ndarray | None = None,
    baseline_counts: np.ndarray | None = None,
) -> dict[str, float | None]:
    candidate_values = _as_metric_values(candidate)
    baseline_values = _as_metric_values(baseline)
    if not np.all(np.isfinite(candidate_values)) or not np.all(np.isfinite(baseline_values)):
        raise ValueError("tensor contains non-finite values")

    integer_values = np.issubdtype(candidate_values.dtype, np.integer) or np.issubdtype(
        candidate_values.dtype, np.bool_
    )
    if integer_values:
        difference = np.asarray(
            np.abs(
                candidate_values.astype(object) - baseline_values.astype(object)
            ),
            dtype=np.float64,
        )
    else:
        candidate_metric_values = candidate_values.astype(np.float64)
        baseline_metric_values = baseline_values.astype(np.float64)
        with np.errstate(over="ignore", invalid="ignore"):
            difference = np.abs(candidate_metric_values - baseline_metric_values)
    if not np.all(np.isfinite(difference)):
        raise ValueError("metric difference overflowed to a non-finite value")
    if difference.size:
        max_abs = float(np.max(difference))
        mean_abs = float(np.mean(difference, dtype=np.float64))
        p99_abs = float(np.percentile(difference, 99))
    else:
        max_abs = mean_abs = p99_abs = 0.0

    candidate_flat = candidate_values.astype(np.float64).reshape(-1)
    baseline_flat = baseline_values.astype(np.float64).reshape(-1)
    candidate_norm = float(np.linalg.norm(candidate_flat))
    baseline_norm = float(np.linalg.norm(baseline_flat))
    cosine = None
    if candidate_norm > 0.0 and baseline_norm > 0.0:
        cosine = float(
            np.dot(candidate_flat, baseline_flat) / (candidate_norm * baseline_norm)
        )
        cosine = max(-1.0, min(1.0, cosine))

    return {
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "p99_abs": p99_abs,
        "cosine": cosine,
        "topk_overlap": _topk_overlap(
            candidate,
            baseline,
            tensor_name=tensor_name,
            candidate_counts=candidate_counts,
            baseline_counts=baseline_counts,
        ),
    }


def _threshold_failures(
    metrics: dict[str, float | None], thresholds: dict[str, float | None]
) -> list[str]:
    failures = []
    for metric, threshold_name in (
        ("max_abs", "max_abs"),
        ("mean_abs", "max_mean_abs"),
        ("p99_abs", "max_p99_abs"),
    ):
        threshold = thresholds[threshold_name]
        value = metrics[metric]
        if threshold is not None and value is not None and value > threshold:
            failures.append(f"{metric} > {threshold:g}")
    minimum_cosine = thresholds["min_cosine"]
    if minimum_cosine is not None:
        cosine = metrics["cosine"]
        if cosine is None:
            failures.append("cosine unavailable")
        elif cosine < minimum_cosine:
            failures.append(f"cosine < {minimum_cosine:g}")
    minimum_overlap = thresholds["min_topk_overlap"]
    overlap = metrics["topk_overlap"]
    if minimum_overlap is not None and overlap is not None and overlap < minimum_overlap:
        failures.append(f"topk_overlap < {minimum_overlap:g}")
    return failures


def compare_dump_directories(
    candidate_dir: str | Path,
    baseline_dir: str | Path,
    *,
    max_abs: float | None = None,
    max_mean_abs: float | None = None,
    max_p99_abs: float | None = None,
    min_cosine: float | None = None,
    min_topk_overlap: float | None = None,
    complete_forward_marker: tuple[str, str] | None = None,
) -> dict[str, Any]:
    """Return a deterministic, JSON-compatible tensor comparison report."""
    candidate_rows, candidate_errors, candidate_duplicates = _read_manifests(
        Path(candidate_dir), "candidate"
    )
    baseline_rows, baseline_errors, baseline_duplicates = _read_manifests(
        Path(baseline_dir), "baseline"
    )
    candidate_completion = baseline_completion = {
        "completed_forward_count": None,
        "dropped_incomplete_forward_count": 0,
        "dropped_tensor_count": 0,
    }
    if complete_forward_marker is not None:
        candidate_rows, candidate_completion = _filter_complete_forwards(
            candidate_rows, complete_forward_marker
        )
        baseline_rows, baseline_completion = _filter_complete_forwards(
            baseline_rows, complete_forward_marker
        )
    candidate_keys = set(candidate_rows)
    baseline_keys = set(baseline_rows)
    missing_candidate_keys = sorted(
        baseline_keys - candidate_keys, key=_key_sort_value
    )
    missing_baseline_keys = sorted(
        candidate_keys - baseline_keys, key=_key_sort_value
    )
    common_keys = sorted(candidate_keys & baseline_keys, key=_key_sort_value)
    thresholds = {
        "max_abs": max_abs,
        "max_mean_abs": max_mean_abs,
        "max_p99_abs": max_p99_abs,
        "min_cosine": min_cosine,
        "min_topk_overlap": min_topk_overlap,
    }

    comparisons = []
    failing_keys = set(candidate_duplicates + baseline_duplicates)
    failing_keys.update(missing_candidate_keys)
    failing_keys.update(missing_baseline_keys)
    for key in common_keys:
        candidate, candidate_load_errors = _load_array(
            candidate_rows[key], "candidate"
        )
        baseline, baseline_load_errors = _load_array(baseline_rows[key], "baseline")
        failures = candidate_load_errors + baseline_load_errors
        shape_match = (
            candidate is not None
            and baseline is not None
            and candidate.shape == baseline.shape
        )
        dtype_match = (
            candidate is not None
            and baseline is not None
            and candidate.dtype == baseline.dtype
        )
        if candidate is not None and baseline is not None:
            if not shape_match:
                failures.append(
                    f"shape mismatch: {list(candidate.shape)} != {list(baseline.shape)}"
                )
            if not dtype_match:
                failures.append(f"dtype mismatch: {candidate.dtype} != {baseline.dtype}")

        metrics = dict(EMPTY_METRICS)
        if not failures and candidate is not None and baseline is not None:
            candidate_counts = None
            baseline_counts = None
            if key[3] == "logical_topk_ids":
                counts_key = (*key[:3], "selected_counts", *key[4:])
                if counts_key in candidate_rows:
                    candidate_counts, count_errors = _load_array(
                        candidate_rows[counts_key], "candidate selected_counts"
                    )
                    failures.extend(count_errors)
                if counts_key in baseline_rows:
                    baseline_counts, count_errors = _load_array(
                        baseline_rows[counts_key], "baseline selected_counts"
                    )
                    failures.extend(count_errors)
            try:
                if failures:
                    raise ValueError(failures[0])
                metrics = _metrics(
                    candidate,
                    baseline,
                    tensor_name=str(key[3]),
                    candidate_counts=candidate_counts,
                    baseline_counts=baseline_counts,
                )
            except (TypeError, ValueError) as error:
                if str(error) not in failures:
                    failures.append(str(error))
            else:
                failures.extend(_threshold_failures(metrics, thresholds))

        comparison = {
            "key": _key_dict(key),
            "candidate_shape": list(candidate.shape) if candidate is not None else None,
            "baseline_shape": list(baseline.shape) if baseline is not None else None,
            "candidate_dtype": str(candidate.dtype) if candidate is not None else None,
            "baseline_dtype": str(baseline.dtype) if baseline is not None else None,
            "shape_match": shape_match,
            "dtype_match": dtype_match,
            "metrics": metrics,
            "failures": failures,
            "passed": not failures,
        }
        comparisons.append(comparison)
        if failures:
            failing_keys.add(key)

    manifest_errors = candidate_errors + baseline_errors
    ordered_failing_keys = sorted(failing_keys, key=_key_sort_value)
    report = {
        "passed": False,
        "key_fields": list(KEY_FIELDS),
        "tensor_count": len(common_keys),
        "candidate_tensor_count": len(candidate_rows),
        "baseline_tensor_count": len(baseline_rows),
        "complete_forward_marker": (
            {
                "component": complete_forward_marker[0],
                "name": complete_forward_marker[1],
            }
            if complete_forward_marker is not None
            else None
        ),
        "candidate_completed_forward_count": candidate_completion[
            "completed_forward_count"
        ],
        "baseline_completed_forward_count": baseline_completion[
            "completed_forward_count"
        ],
        "candidate_dropped_incomplete_forward_count": candidate_completion[
            "dropped_incomplete_forward_count"
        ],
        "baseline_dropped_incomplete_forward_count": baseline_completion[
            "dropped_incomplete_forward_count"
        ],
        "candidate_dropped_tensor_count": candidate_completion[
            "dropped_tensor_count"
        ],
        "baseline_dropped_tensor_count": baseline_completion[
            "dropped_tensor_count"
        ],
        "manifest_errors": manifest_errors,
        "missing_from_candidate": [
            _key_dict(key) for key in missing_candidate_keys
        ],
        "missing_from_baseline": [_key_dict(key) for key in missing_baseline_keys],
        "thresholds": thresholds,
        "comparisons": comparisons,
        "first_failing_key": (
            _key_dict(ordered_failing_keys[0]) if ordered_failing_keys else None
        ),
    }
    report["passed"] = (
        not manifest_errors
        and not missing_candidate_keys
        and not missing_baseline_keys
        and all(comparison["passed"] for comparison in comparisons)
        and bool(common_keys)
    )
    return report


def _non_negative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError("must be a finite non-negative number")
    return parsed


def _unit_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be a finite number between 0 and 1")
    return parsed


def _component_name(value: str) -> tuple[str, str]:
    parts = value.split(":", 1)
    if len(parts) != 2 or not all(parts):
        raise argparse.ArgumentTypeError("must be COMPONENT:NAME")
    return parts[0], parts[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare JAX debug tensor manifests by semantic key."
    )
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--max-abs", "--max-abs-error", dest="max_abs", type=_non_negative_float
    )
    parser.add_argument(
        "--max-mean-abs",
        "--max-mean-abs-error",
        dest="max_mean_abs",
        type=_non_negative_float,
    )
    parser.add_argument(
        "--max-p99-abs",
        "--max-p99-abs-error",
        dest="max_p99_abs",
        type=_non_negative_float,
    )
    parser.add_argument("--min-cosine", type=_unit_float)
    parser.add_argument("--min-topk-overlap", type=_unit_float)
    parser.add_argument(
        "--complete-forward-marker",
        type=_component_name,
        help=(
            "Compare only forwards containing COMPONENT:NAME; useful for dropping "
            "partially executed callbacks from cancelled scheduler work."
        ),
    )
    args = parser.parse_args(argv)

    report = compare_dump_directories(
        args.candidate,
        args.baseline,
        max_abs=args.max_abs,
        max_mean_abs=args.max_mean_abs,
        max_p99_abs=args.max_p99_abs,
        min_cosine=args.min_cosine,
        min_topk_overlap=args.min_topk_overlap,
        complete_forward_marker=args.complete_forward_marker,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
