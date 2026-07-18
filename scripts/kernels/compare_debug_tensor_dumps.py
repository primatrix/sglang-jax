#!/usr/bin/env python3
"""Compare semantic JAX tensor dumps from two debug artifact trees."""

from __future__ import annotations

import argparse
import json
import math
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
        directory.rglob("manifest-p*.jsonl"),
        key=lambda path: path.relative_to(directory).as_posix(),
    )
    if not manifests:
        return rows, [f"{label}: no manifest-p*.jsonl files found"], []

    for manifest in manifests:
        manifest_name = _manifest_label(directory, manifest)
        try:
            lines = manifest.read_text(encoding="utf-8").splitlines()
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
            try:
                array_path.resolve().relative_to(directory.resolve())
            except (OSError, ValueError):
                errors.append(f"{location}: filename escapes the dump directory")
                continue
            if key in rows:
                rendered_key = json.dumps(_key_dict(key), sort_keys=True)
                errors.append(f"{label}: duplicate semantic key {rendered_key}")
                duplicate_keys.append(key)
                continue
            rows[key] = {
                "array_path": array_path,
                "declared_shape": tuple(shape),
                "declared_dtype": dtype,
                "location": location,
            }
    return rows, errors, duplicate_keys


def _load_array(record: dict[str, Any], label: str) -> tuple[np.ndarray | None, list[str]]:
    errors: list[str] = []
    try:
        array = np.load(record["array_path"], allow_pickle=False)
    except (OSError, ValueError) as error:
        return None, [f"{label}: cannot load tensor: {error}"]
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
    try:
        floating = np.issubdtype(array.dtype, np.floating)
    except TypeError:
        floating = dtype_name == "bfloat16"
    if floating or dtype_name == "bfloat16":
        return array.astype(np.float32)
    try:
        numeric = np.issubdtype(array.dtype, np.number) or np.issubdtype(
            array.dtype, np.bool_
        )
    except TypeError:
        numeric = False
    if not numeric:
        raise TypeError(f"unsupported non-numeric dtype {array.dtype}")
    return array.astype(np.float64)


def _topk_overlap(
    candidate: np.ndarray, baseline: np.ndarray, *, tensor_name: str
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
    overlaps = [
        len(set(candidate_row.tolist()) & set(baseline_row.tolist())) / width
        for candidate_row, baseline_row in zip(
            candidate_rows, baseline_rows, strict=True
        )
    ]
    return float(min(overlaps)) if overlaps else None


def _metrics(
    candidate: np.ndarray, baseline: np.ndarray, *, tensor_name: str
) -> dict[str, float | None]:
    candidate_values = _as_metric_values(candidate)
    baseline_values = _as_metric_values(baseline)
    if not np.all(np.isfinite(candidate_values)) or not np.all(np.isfinite(baseline_values)):
        raise ValueError("tensor contains non-finite values")

    difference = np.abs(candidate_values - baseline_values)
    if difference.size:
        max_abs = float(np.max(difference))
        mean_abs = float(np.mean(difference, dtype=np.float64))
        p99_abs = float(np.percentile(difference, 99))
    else:
        max_abs = mean_abs = p99_abs = 0.0

    candidate_flat = candidate_values.reshape(-1).astype(np.float64)
    baseline_flat = baseline_values.reshape(-1).astype(np.float64)
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
            candidate, baseline, tensor_name=tensor_name
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
) -> dict[str, Any]:
    """Return a deterministic, JSON-compatible tensor comparison report."""
    candidate_rows, candidate_errors, candidate_duplicates = _read_manifests(
        Path(candidate_dir), "candidate"
    )
    baseline_rows, baseline_errors, baseline_duplicates = _read_manifests(
        Path(baseline_dir), "baseline"
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
            try:
                metrics = _metrics(candidate, baseline, tensor_name=str(key[3]))
            except (TypeError, ValueError) as error:
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
    args = parser.parse_args(argv)

    report = compare_dump_directories(
        args.candidate,
        args.baseline,
        max_abs=args.max_abs,
        max_mean_abs=args.max_mean_abs,
        max_p99_abs=args.max_p99_abs,
        min_cosine=args.min_cosine,
        min_topk_overlap=args.min_topk_overlap,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
