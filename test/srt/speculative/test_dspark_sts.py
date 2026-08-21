import numpy as np

from benchmark.dspark.fit_sts import _calibrated_probability, fit_sts, fit_temperature


def test_fit_temperature_recovers_synthetic_scale():
    rng = np.random.default_rng(7)
    logits = rng.normal(size=200_000)
    true_temperature = 2.5
    labels = rng.random(logits.shape) < 1.0 / (1.0 + np.exp(-logits / true_temperature))
    uncalibrated = 1.0 / (1.0 + np.exp(-logits))

    fitted = fit_temperature(uncalibrated, labels.astype(np.float64))

    assert abs(fitted - true_temperature) < 0.08


def test_fit_sts_uses_conditional_at_risk_rows():
    confidence = np.full((10, 3), 0.5, dtype=np.float64)
    accepted = np.asarray([0, 0, 1, 1, 1, 2, 2, 3, 3, 3], dtype=np.int32)

    result = fit_sts(confidence, accepted, seed=1, train_fraction=0.5)

    assert result["gamma"] == 3
    assert result["positions"][0]["empirical_acceptance"] == 0.8
    assert result["positions"][1]["empirical_acceptance"] == 0.625
    assert result["positions"][2]["empirical_acceptance"] == 0.6
    assert np.all(np.isfinite(result["temperatures"]))


def test_temperature_one_preserves_probability():
    confidence = np.asarray([0.1, 0.5, 0.9])
    np.testing.assert_allclose(_calibrated_probability(confidence, 1.0), confidence)
