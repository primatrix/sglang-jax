#!/usr/bin/env python3
"""Fit paper-faithful DSpark Sequential Temperature Scaling from capture JSONL."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

_EPS_PROB = 1e-8


def _probability_to_logit(confidence: np.ndarray) -> np.ndarray:
    """Convert legacy probability captures to logits for backward compatibility."""
    clipped = np.clip(confidence, _EPS_PROB, 1.0 - _EPS_PROB)
    return np.log(clipped) - np.log1p(-clipped)


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -60.0, 60.0)))


def load_capture(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load raw confidence logits and realized prefix-survival masks."""
    logits = []
    prefix_masks = []
    with path.open(encoding="utf-8") as capture:
        for line in capture:
            if not line.strip():
                continue
            record = json.loads(line)
            if "logits" in record:
                row_logits = np.asarray(record["logits"], dtype=np.float64)
            elif "confidence" in record:
                row_logits = _probability_to_logit(
                    np.asarray(record["confidence"], dtype=np.float64)
                )
            else:
                raise ValueError(
                    "STS capture row must contain 'logits' or legacy 'confidence'."
                )

            if "prefix_mask" in record:
                prefix_mask = np.asarray(record["prefix_mask"], dtype=np.float64)
            elif "accepted_draft" in record:
                prefix_mask = (
                    np.arange(row_logits.shape[0], dtype=np.int32)
                    < int(record["accepted_draft"])
                ).astype(np.float64)
            else:
                raise ValueError(
                    "STS capture row must contain 'prefix_mask' or 'accepted_draft'."
                )
            logits.append(row_logits)
            prefix_masks.append(prefix_mask)

    if not logits:
        raise ValueError(f"No DSpark STS samples found in {path}.")
    values = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(prefix_masks, dtype=np.float64)
    if values.ndim != 2 or targets.shape != values.shape:
        raise ValueError(
            f"Invalid capture shapes: logits={values.shape}, prefix_mask={targets.shape}."
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("STS capture logits must all be finite.")
    if not np.all((targets == 0.0) | (targets == 1.0)):
        raise ValueError("STS prefix_mask must contain only zero or one.")
    return values, targets


def default_temperature_grid() -> np.ndarray:
    return np.logspace(math.log10(0.1), math.log10(10.0), num=41, dtype=np.float64)


def expected_calibration_error(
    probability: np.ndarray,
    targets: np.ndarray,
    *,
    bins: int = 15,
) -> float:
    probability = np.asarray(probability, dtype=np.float64).reshape(-1)
    targets = np.asarray(targets, dtype=np.float64).reshape(-1)
    if probability.shape != targets.shape:
        raise ValueError(
            f"ECE probability/target shapes differ: {probability.shape} vs {targets.shape}."
        )
    if probability.size == 0:
        return float("nan")
    probability = np.clip(probability, _EPS_PROB, 1.0 - _EPS_PROB)
    bin_index = np.minimum((probability * bins).astype(np.int32), bins - 1)
    count = np.bincount(bin_index, minlength=bins).astype(np.float64)
    pred_sum = np.bincount(bin_index, weights=probability, minlength=bins)
    target_sum = np.bincount(bin_index, weights=targets, minlength=bins)
    denominator = np.maximum(count, 1.0)
    bin_error = np.abs(pred_sum / denominator - target_sum / denominator)
    return float(np.sum(bin_error * count) / probability.size)


def fit_sts(
    logits: np.ndarray,
    prefix_mask: np.ndarray,
    *,
    grid: np.ndarray | None = None,
    bins: int = 15,
) -> dict:
    """Fit cumulative prefix survival from left to right as defined by DSpark STS."""
    logits = np.asarray(logits, dtype=np.float64)
    prefix_mask = np.asarray(prefix_mask, dtype=np.float64)
    if logits.ndim != 2 or prefix_mask.shape != logits.shape:
        raise ValueError(
            f"STS logits/prefix_mask shapes differ: {logits.shape} vs {prefix_mask.shape}."
        )
    if logits.shape[0] == 0:
        raise ValueError("fit_sts requires at least one sample.")
    grid = (
        default_temperature_grid()
        if grid is None
        else np.asarray(grid, dtype=np.float64)
    )
    if (
        grid.ndim != 1
        or grid.size == 0
        or not np.all(np.isfinite(grid))
        or np.any(grid <= 0)
    ):
        raise ValueError(
            "STS temperature grid must be a non-empty vector of positive values."
        )

    survival_at_one = np.ones(logits.shape[0], dtype=np.float64)
    survival_fitted = np.ones(logits.shape[0], dtype=np.float64)
    temperatures = []
    positions = []
    for position in range(logits.shape[1]):
        position_logits = logits[:, position]
        position_target = prefix_mask[:, position]
        survival_at_one *= _sigmoid(position_logits)
        ece_before = expected_calibration_error(
            survival_at_one,
            position_target,
            bins=bins,
        )

        best_temperature = float(grid[0])
        best_survival = survival_fitted * _sigmoid(position_logits / best_temperature)
        best_ece = expected_calibration_error(best_survival, position_target, bins=bins)
        for temperature in grid[1:]:
            candidate_survival = survival_fitted * _sigmoid(
                position_logits / temperature
            )
            candidate_ece = expected_calibration_error(
                candidate_survival,
                position_target,
                bins=bins,
            )
            if candidate_ece < best_ece:
                best_temperature = float(temperature)
                best_survival = candidate_survival
                best_ece = candidate_ece

        temperatures.append(best_temperature)
        positions.append(
            {
                "position": position,
                "temperature": best_temperature,
                "ece_before": ece_before,
                "ece_after": best_ece,
                "empirical_prefix_survival": float(np.mean(position_target)),
            }
        )
        survival_fitted = best_survival

    return {
        "schema_version": 2,
        "method": "sequential_temperature_scaling",
        "objective": "cumulative_prefix_survival_ece",
        "rows": int(logits.shape[0]),
        "gamma": int(logits.shape[1]),
        "num_bins": int(bins),
        "temperature_grid": grid.tolist(),
        "temperatures": temperatures,
        "positions": positions,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("capture", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-bins", type=int, default=15)
    args = parser.parse_args()
    logits, prefix_mask = load_capture(args.capture)
    result = fit_sts(logits, prefix_mask, bins=args.num_bins)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
