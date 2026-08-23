import json

import numpy as np

from benchmark.dspark.fit_sts import (
    default_temperature_grid,
    expected_calibration_error,
    fit_sts,
    load_capture,
)


def test_fit_sts_recovers_scale_and_reduces_prefix_ece():
    rng = np.random.default_rng(7)
    num_samples, gamma, scale = 60_000, 4, 2.5
    base_logit = np.asarray([2.0, 1.2, 0.8, 0.4])
    true_logits = base_logit[None, :] + rng.normal(scale=0.5, size=(num_samples, gamma))
    true_probability = 1.0 / (1.0 + np.exp(-true_logits))
    accepted = rng.random(true_probability.shape) < true_probability
    prefix_mask = np.cumprod(accepted.astype(np.int32), axis=1).astype(np.float64)

    result = fit_sts(
        true_logits * scale,
        prefix_mask,
        grid=default_temperature_grid(),
    )

    assert result["method"] == "sequential_temperature_scaling"
    assert len(result["temperatures"]) == gamma
    for temperature in result["temperatures"]:
        assert scale / 1.5 < temperature < scale * 1.5
    mean_before = np.mean([row["ece_before"] for row in result["positions"]])
    mean_after = np.mean([row["ece_after"] for row in result["positions"]])
    assert mean_after < 0.25 * mean_before


def test_expected_calibration_error_distinguishes_calibrated_probability():
    rng = np.random.default_rng(11)
    calibrated = np.full(20_000, 0.3)
    targets = (rng.random(calibrated.shape) < 0.3).astype(np.float64)
    assert expected_calibration_error(calibrated, targets) < 0.02
    assert expected_calibration_error(np.full_like(calibrated, 0.95), targets) > 0.5


def test_load_capture_prefers_raw_logits_and_builds_legacy_prefix(tmp_path):
    capture = tmp_path / "capture.jsonl"
    records = [
        {"logits": [2.0, 1.0, -1.0], "prefix_mask": [1, 1, 0]},
        {"confidence": [0.5, 0.5, 0.5], "accepted_draft": 1},
    ]
    capture.write_text(
        "".join(json.dumps(row) + "\n" for row in records), encoding="utf-8"
    )

    logits, prefix_mask = load_capture(capture)

    np.testing.assert_allclose(logits[0], [2.0, 1.0, -1.0])
    np.testing.assert_allclose(logits[1], 0.0, atol=1e-12)
    np.testing.assert_array_equal(prefix_mask, [[1, 1, 0], [1, 0, 0]])
