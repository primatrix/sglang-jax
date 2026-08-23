"""Built-in DSpark calibration and step-cost profiles.

The lookup intentionally follows the same shape as TPU kernel tune tables:
derive an exact deployment key, return a frozen config on hit, and return
``None`` on miss. A miss must never change target correctness; callers keep
the fixed verify-all path.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DSparkTunedKey:
    target_model: str
    draft_model: str
    target_revision: str
    draft_revision: str
    device_name: str
    device_count: int
    dtype: str
    quantization: str
    tp_size: int
    dp_size: int
    gamma: int
    page_size: int
    attention_backend: str
    overlap: bool


@dataclass(frozen=True)
class DSparkSPSPoint:
    verify_tokens_per_dp: int
    median_step_time_ms: float
    steps_per_second: float
    # None denotes the legacy one-dimensional T(M) bootstrap table.  A
    # measured ragged point records the per-DP request bucket as well, making
    # its cost T(R, M).
    request_bucket_per_dp: int | None = None


@dataclass(frozen=True)
class DSparkSPSProfile:
    context_bucket: int
    points: tuple[DSparkSPSPoint, ...]


@dataclass(frozen=True)
class DSparkTunedConfig:
    sts_temperatures: tuple[float, ...]
    sps_profiles: tuple[DSparkSPSProfile, ...]
    provenance: str


def normalize_dspark_model_id(model_path: str) -> str:
    """Normalize HF IDs and mounted checkpoint paths to one tune-table ID."""
    base = os.path.basename(str(model_path).rstrip("/"))
    return re.sub(r"[_-]+", "-", base.casefold())


def make_dspark_tuned_key(
    *,
    target_model: str,
    draft_model: str,
    target_revision: str | None,
    draft_revision: str | None,
    device_name: str,
    device_count: int,
    dtype: str,
    quantization: str | None,
    tp_size: int,
    dp_size: int,
    gamma: int,
    page_size: int,
    attention_backend: str,
    overlap: bool,
) -> DSparkTunedKey:
    return DSparkTunedKey(
        target_model=normalize_dspark_model_id(target_model),
        draft_model=normalize_dspark_model_id(draft_model),
        target_revision=target_revision or "default",
        draft_revision=draft_revision or "default",
        device_name=device_name,
        device_count=int(device_count),
        dtype=str(dtype).lower(),
        quantization=(quantization or "none").lower(),
        tp_size=int(tp_size),
        dp_size=int(dp_size),
        gamma=int(gamma),
        page_size=int(page_size),
        attention_backend=str(attention_backend).lower(),
        overlap=bool(overlap),
    )


_QWEN3_8B_BLOCK7_V7X8 = make_dspark_tuned_key(
    target_model="Qwen/Qwen3-8B",
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


# Two-dimensional ragged SPS values were measured by Falcon exp-525gl93cv9,
# with padded boundary points corrected by exp-xi9gjnux5z. Sequential STS
# values were regenerated from raw confidence logits by exp-ksphva699n. All
# runs used JAX 0.10.2/libtpu 0.0.42.1 on v7x-8.
# The SPS profile used random input=256/output=256, represented by the next
# power-of-two context bucket. Runtime selection must not use it above 1024.
TUNED_DSPARK_CONFIGS: dict[DSparkTunedKey, DSparkTunedConfig] = {
    _QWEN3_8B_BLOCK7_V7X8: DSparkTunedConfig(
        sts_temperatures=(
            1.9952623149688797,
            2.511886431509581,
            2.23872113856834,
            2.511886431509581,
            2.511886431509581,
            2.511886431509581,
            2.23872113856834,
        ),
        sps_profiles=(
            DSparkSPSProfile(
                context_bucket=1024,
                points=(
                    DSparkSPSPoint(32, 10.906374999649415, 91.68949353310747, 32),
                    DSparkSPSPoint(64, 11.128035500405531, 89.86312094021966, 32),
                    DSparkSPSPoint(128, 11.181399000633974, 89.43424699747331, 32),
                    DSparkSPSPoint(256, 11.40483300014239, 87.68212563809702, 32),
                    DSparkSPSPoint(64, 11.165372500272497, 89.56261871026645, 64),
                    DSparkSPSPoint(128, 12.116556500131992, 82.53169949639623, 64),
                    DSparkSPSPoint(256, 13.09894699988945, 76.3420143625621, 64),
                    DSparkSPSPoint(512, 16.603519999989658, 60.22819257606959, 64),
                ),
            ),
        ),
        provenance=(
            "Falcon exp-ksphva699n raw-logit sequential STS (GSM8K-500); "
            "exp-525gl93cv9 + exp-xi9gjnux5z random-256/256 ragged T(R,M) SPS"
        ),
    ),
}


def get_tuned_dspark_config(key: DSparkTunedKey) -> DSparkTunedConfig | None:
    return TUNED_DSPARK_CONFIGS.get(key)


def select_dspark_sps_profile(
    config: DSparkTunedConfig, context_length: int
) -> DSparkSPSProfile | None:
    """Select the smallest measured context bucket that covers the request."""
    for profile in sorted(config.sps_profiles, key=lambda item: item.context_bucket):
        if context_length <= profile.context_bucket:
            return profile
    return None


def calibrate_dspark_confidence(confidence_logits, temperatures):
    """Apply per-position STS directly to uncalibrated confidence logits."""
    import jax
    import jax.numpy as jnp

    temperature = jnp.asarray(temperatures, dtype=jnp.float32)
    if confidence_logits.shape[-1] != temperature.shape[0]:
        raise ValueError(
            "DSPARK STS width must match confidence width: "
            f"{temperature.shape[0]} vs {confidence_logits.shape[-1]}."
        )
    return jax.nn.sigmoid(confidence_logits.astype(jnp.float32) / temperature)


__all__ = [
    "DSparkSPSPoint",
    "DSparkSPSProfile",
    "DSparkTunedConfig",
    "DSparkTunedKey",
    "TUNED_DSPARK_CONFIGS",
    "calibrate_dspark_confidence",
    "get_tuned_dspark_config",
    "make_dspark_tuned_key",
    "normalize_dspark_model_id",
    "select_dspark_sps_profile",
]
