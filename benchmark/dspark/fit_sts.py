#!/usr/bin/env python3
"""Fit per-position DSpark confidence temperatures from capture JSONL."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def load_capture(path: Path) -> tuple[np.ndarray, np.ndarray]:
    confidence = []
    accepted = []
    with path.open(encoding="utf-8") as capture:
        for line in capture:
            if not line.strip():
                continue
            record = json.loads(line)
            confidence.append(record["confidence"])
            accepted.append(record["accepted_draft"])
    if not confidence:
        raise ValueError(f"No DSpark STS samples found in {path}.")
    values = np.asarray(confidence, dtype=np.float64)
    accepted_draft = np.asarray(accepted, dtype=np.int32)
    if values.ndim != 2 or accepted_draft.shape != (values.shape[0],):
        raise ValueError(
            f"Invalid capture shapes: confidence={values.shape}, accepted={accepted_draft.shape}."
        )
    return values, accepted_draft


def _calibrated_probability(confidence: np.ndarray, temperature: float) -> np.ndarray:
    clipped = np.clip(confidence, 1e-6, 1.0 - 1e-6)
    logits = np.log(clipped) - np.log1p(-clipped)
    return 1.0 / (1.0 + np.exp(-logits / temperature))


def _nll(confidence: np.ndarray, labels: np.ndarray, temperature: float) -> float:
    probability = np.clip(
        _calibrated_probability(confidence, temperature), 1e-9, 1.0 - 1e-9
    )
    return float(
        -np.mean(labels * np.log(probability) + (1.0 - labels) * np.log1p(-probability))
    )


def fit_temperature(confidence: np.ndarray, labels: np.ndarray) -> float:
    """Minimize Bernoulli NLL over log-temperature with golden-section search."""
    left, right = math.log(0.05), math.log(20.0)
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    x1 = right - ratio * (right - left)
    x2 = left + ratio * (right - left)
    f1 = _nll(confidence, labels, math.exp(x1))
    f2 = _nll(confidence, labels, math.exp(x2))
    for _ in range(80):
        if f1 <= f2:
            right, x2, f2 = x2, x1, f1
            x1 = right - ratio * (right - left)
            f1 = _nll(confidence, labels, math.exp(x1))
        else:
            left, x1, f1 = x1, x2, f2
            x2 = left + ratio * (right - left)
            f2 = _nll(confidence, labels, math.exp(x2))
    return math.exp((left + right) / 2.0)


def _ece(probability: np.ndarray, labels: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(labels)
    value = 0.0
    for index in range(bins):
        upper_closed = index == bins - 1
        selected = (probability >= edges[index]) & (
            (probability <= edges[index + 1])
            if upper_closed
            else (probability < edges[index + 1])
        )
        if np.any(selected):
            value += float(np.sum(selected)) / total * abs(
                float(np.mean(probability[selected])) - float(np.mean(labels[selected]))
            )
    return value


def fit_sts(
    confidence: np.ndarray,
    accepted_draft: np.ndarray,
    *,
    seed: int = 980406,
    train_fraction: float = 0.8,
) -> dict:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between zero and one.")
    rng = np.random.default_rng(seed)
    train_rows = rng.random(confidence.shape[0]) < train_fraction
    positions = []
    temperatures = []
    for position in range(confidence.shape[1]):
        # Confidence is conditional: position k is observed only when all
        # positions before k were accepted.
        at_risk = accepted_draft >= position
        labels = (accepted_draft > position).astype(np.float64)
        train = at_risk & train_rows
        validation = at_risk & ~train_rows
        if not np.any(train):
            raise ValueError(f"No at-risk training samples for position {position}.")
        temperature = fit_temperature(confidence[train, position], labels[train])
        temperatures.append(temperature)
        before = _calibrated_probability(confidence[validation, position], 1.0)
        after = _calibrated_probability(confidence[validation, position], temperature)
        validation_labels = labels[validation]
        positions.append(
            {
                "position": position,
                "train_samples": int(np.sum(train)),
                "validation_samples": int(np.sum(validation)),
                "empirical_acceptance": float(np.mean(labels[at_risk])),
                "temperature": temperature,
                "validation_nll_before": _nll(
                    confidence[validation, position], validation_labels, 1.0
                ),
                "validation_nll_after": _nll(
                    confidence[validation, position], validation_labels, temperature
                ),
                "validation_ece_before": _ece(before, validation_labels),
                "validation_ece_after": _ece(after, validation_labels),
            }
        )
    return {
        "schema_version": 1,
        "method": "per_position_temperature_scaling",
        "conditional_label": "accepted_draft > position given accepted_draft >= position",
        "seed": seed,
        "train_fraction": train_fraction,
        "rows": int(confidence.shape[0]),
        "gamma": int(confidence.shape[1]),
        "temperatures": temperatures,
        "positions": positions,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("capture", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=980406)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    args = parser.parse_args()
    confidence, accepted_draft = load_capture(args.capture)
    result = fit_sts(
        confidence,
        accepted_draft,
        seed=args.seed,
        train_fraction=args.train_fraction,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
