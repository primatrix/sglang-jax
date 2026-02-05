"""Tuning-related knobs and bucketing logic for fused MoE.

Keep this module dependency-light: it is imported from both runtime code paths
(ForwardBatch construction) and benchmarks.
"""

from __future__ import annotations

# Minimum bucket size for `valid_num_tokens` when selecting a tuned block config.
# The bucketing scheme maps (potentially many) `valid_num_tokens` values to a
# smaller set of keys to avoid excessive compile variants.
SGLANG_FUSED_MOE_TUNING_MIN_BUCKET: int = 64


def _round_up_to_multiple(x: int, multiple: int) -> int:
    if multiple <= 0:
        raise ValueError(f"Expected {multiple=} to be > 0.")
    return ((x + multiple - 1) // multiple) * multiple


def _next_power_of_two(x: int) -> int:
    if x <= 0:
        return 1
    return 1 << (x - 1).bit_length()


def bucket_valid_num_tokens_for_fused_moe(
    *,
    valid_num_tokens: int,
    padded_num_tokens: int,
    ep_size: int,
    min_bucket: int | None = None,
) -> int:
    """Bucket valid-token count for tuned block-config lookup.

    Note: this is used only for selecting a tuned block config. Kernel shapes
    must still be compiled for `padded_num_tokens`.
    """
    if ep_size <= 0:
        raise ValueError(f"Expected {ep_size=} to be > 0.")
    if padded_num_tokens <= 0:
        raise ValueError(f"Expected {padded_num_tokens=} to be > 0.")

    if min_bucket is None:
        min_bucket = SGLANG_FUSED_MOE_TUNING_MIN_BUCKET
    min_bucket = max(0, int(min_bucket))

    target = max(int(valid_num_tokens), min_bucket, ep_size)
    bucket = _next_power_of_two(target)
    bucket = _round_up_to_multiple(bucket, ep_size)
    bucket = min(bucket, int(padded_num_tokens))
    bucket = _round_up_to_multiple(bucket, ep_size)
    bucket = min(bucket, int(padded_num_tokens))
    return int(bucket)
