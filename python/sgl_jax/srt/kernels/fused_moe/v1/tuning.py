"""Tuning-related helpers for fused MoE.

Keep this module dependency-light: it is imported from both runtime code paths
(ForwardBatch construction) and benchmarks.
"""

from __future__ import annotations


def clamp_valid_num_tokens_for_fused_moe(
    *,
    valid_num_tokens: int,
    padded_num_tokens: int,
) -> int:
    """Clamp valid-token count for tuned block-config lookup.

    Note: this is used only for selecting a tuned block config. Kernel shapes
    must still be compiled for `padded_num_tokens`.
    """
    if padded_num_tokens <= 0:
        raise ValueError(f"Expected {padded_num_tokens=} to be > 0.")

    valid_num_tokens = int(valid_num_tokens)
    padded_num_tokens = int(padded_num_tokens)
    if valid_num_tokens <= 0:
        return padded_num_tokens
    if valid_num_tokens > padded_num_tokens:
        return padded_num_tokens
    return valid_num_tokens
