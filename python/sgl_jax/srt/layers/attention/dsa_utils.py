from __future__ import annotations

from collections.abc import Sequence


def normalize_dsa_context_buckets(
    buckets: Sequence[int] | None,
    *,
    page_size: int,
    max_context_len: int | None = None,
) -> tuple[int, ...] | None:
    """Normalize opt-in DSA JIT buckets to page-aligned context widths."""
    if buckets is None:
        return None
    if not buckets:
        raise ValueError("DSA context buckets must not be empty")
    if page_size <= 0:
        raise ValueError("DSA page_size must be positive")

    normalized = set()
    for bucket in buckets:
        if bucket <= 0:
            raise ValueError(f"DSA context bucket must be positive, got {bucket}")
        normalized.add((bucket + page_size - 1) // page_size * page_size)

    if max_context_len is not None:
        if max_context_len <= 0:
            raise ValueError(f"DSA max_context_len must be positive, got {max_context_len}")
        aligned_max = (max_context_len + page_size - 1) // page_size * page_size
        if max(normalized) < aligned_max:
            normalized.add(aligned_max)

    return tuple(sorted(normalized))
