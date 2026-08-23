import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.speculative.dspark_planner import select_dspark_verify_budget
from sgl_jax.srt.speculative.dspark_tuned_config import (
    calibrate_dspark_confidence,
    get_tuned_dspark_config,
    make_dspark_tuned_key,
    normalize_dspark_model_id,
    select_dspark_sps_profile,
)


def _key(**overrides):
    values = dict(
        target_model="/models/Qwen3-8B",
        draft_model="deepseek-ai/dspark_qwen3_8b_block7",
        target_revision=None,
        draft_revision=None,
        device_name="TPU v7",
        device_count=8,
        dtype="bfloat16",
        quantization=None,
        tp_size=8,
        dp_size=2,
        gamma=7,
        page_size=64,
        attention_backend="fa",
        overlap=True,
    )
    values.update(overrides)
    return make_dspark_tuned_key(**values)


def test_normalize_model_id_unifies_mount_and_hf_basename():
    assert normalize_dspark_model_id("/models/Qwen3-8B") == normalize_dspark_model_id(
        "Qwen/Qwen3-8B"
    )


def test_qwen3_v7x8_tuned_config_exact_hit():
    config = get_tuned_dspark_config(_key())

    assert config is not None
    assert len(config.sts_temperatures) == 7
    points = config.sps_profiles[0].points
    assert {point.request_bucket_per_dp for point in points} == {32, 64}
    assert {
        point.verify_tokens_per_dp for point in points if point.request_bucket_per_dp == 32
    } == {32, 64, 128, 256}


def test_tuned_config_shape_or_revision_mismatch_is_a_miss():
    assert get_tuned_dspark_config(_key(tp_size=4)) is None
    assert get_tuned_dspark_config(_key(page_size=128)) is None
    assert get_tuned_dspark_config(_key(draft_revision="new-revision")) is None


def test_qwen3_v7x8_2d_profile_avoids_legacy_m64_cliff_at_r33():
    profile = get_tuned_dspark_config(_key()).sps_profiles[0]
    conditional_confidence = np.full((33, 7), 0.8, dtype=np.float32)

    decision = select_dspark_verify_budget(
        profile,
        np.cumprod(conditional_confidence, axis=-1),
    )

    assert decision.token_bucket == 256


def test_sps_context_bucket_never_extrapolates_upward():
    config = get_tuned_dspark_config(_key())

    assert select_dspark_sps_profile(config, 768).context_bucket == 1024
    assert select_dspark_sps_profile(config, 1025) is None


def test_sts_temperature_one_is_plain_sigmoid_and_width_is_checked():
    confidence_logits = jnp.asarray([[-2.0, 0.0, 2.0]], dtype=jnp.float32)
    np.testing.assert_allclose(
        np.asarray(calibrate_dspark_confidence(confidence_logits, (1.0, 1.0, 1.0))),
        np.asarray(jax.nn.sigmoid(confidence_logits)),
        rtol=1e-6,
    )
    try:
        calibrate_dspark_confidence(confidence_logits, (1.0, 1.0))
    except ValueError as exc:
        assert "STS width" in str(exc)
    else:
        raise AssertionError("STS width mismatch must be rejected")


def test_sts_scales_each_raw_logit_column_before_sigmoid():
    confidence_logits = jnp.full((2, 3), 2.0, dtype=jnp.float32)
    calibrated = calibrate_dspark_confidence(confidence_logits, (0.5, 1.0, 2.0))
    expected = jax.nn.sigmoid(confidence_logits / jnp.asarray([0.5, 1.0, 2.0], dtype=jnp.float32))
    np.testing.assert_allclose(np.asarray(calibrated), np.asarray(expected), rtol=1e-6)
