"""Memory budget utilities for SWA dual-pool allocation."""

import logging

logger = logging.getLogger(__name__)


def compute_optimal_swa_ratio(
    user_R: float,
    sliding_window: int | None,
    context_len: int,
    fa_cell_per_layer: int,
    swa_cell_per_layer: int,
    fa_layers: int,
    swa_layers: int,
    default_R: float = 0.8,
    headroom: float = 1.25,
) -> float:
    """Compute optimal SWA-to-full token ratio for dual-pool memory allocation.

    The ratio R = swa_max_tokens / full_max_tokens determines how memory is
    split between the full-attention (FA) and sliding-window (SWA) pools.

    Key insight: SWA layers evict tokens beyond the sliding window, so each
    request only needs at most ``sliding_window`` tokens in the SWA pool,
    whereas it may need up to ``context_len`` tokens in the FA pool.  Setting R
    based on this bound (rather than a fixed 0.8) avoids over-allocating the
    SWA pool and frees memory for the FA pool, increasing throughput.

    Strategy:
    1. If the user explicitly set ``--swa-full-tokens-ratio`` to a non-default
       value, respect it.
    2. Otherwise, compute ``R_window = sliding_window * headroom / context_len``
       as an upper bound on the fraction of the full pool that the SWA pool
       needs.
    3. Apply a cost-weighting adjustment:
       ``R_cost = (fa_total_cost / swa_total_cost) * R_window``
       so that when SWA layers are more expensive per token (e.g. more KV
       heads), the ratio shrinks proportionally.
    4. Take the minimum of R_window and R_cost, then clamp to [0.01, 0.95].

    Args:
        user_R: The value of ``--swa-full-tokens-ratio`` from server args.
        sliding_window: Sliding window size (tokens).  ``None`` or ``<= 0``
            means unknown, in which case ``user_R`` is returned as-is.
        context_len: Maximum sequence length.
        fa_cell_per_layer: Per-token per-layer byte cost for FA layers.
        swa_cell_per_layer: Per-token per-layer byte cost for SWA layers.
        fa_layers: Number of full-attention layers.
        swa_layers: Number of sliding-window attention layers.
        default_R: The default ratio (0.8).  Used to detect explicit overrides.
        headroom: Multiplicative headroom on the window-based bound (default
            1.25, i.e. 25 % extra).

    Returns:
        Effective R value to use for ``swa_max_tokens = full_max_tokens * R``.
    """
    # If no sliding window info, fall back to user-provided R
    if sliding_window is None or sliding_window <= 0 or context_len <= 0:
        return user_R

    # Compute the window-bounded ratio with headroom
    R_window = min((sliding_window * headroom) / context_len, 0.95)

    # Cost-weighted adjustment
    fa_total_cost = fa_cell_per_layer * fa_layers
    swa_total_cost = swa_cell_per_layer * swa_layers

    if swa_total_cost > 0:
        cost_ratio = fa_total_cost / swa_total_cost
        R_cost = cost_ratio * R_window
    else:
        R_cost = R_window

    # Use the more conservative (smaller) of the two
    R_optimal = min(R_window, R_cost)

    # Clamp to safe range
    R_optimal = max(0.01, min(R_optimal, 0.95))

    # If user explicitly set a value different from default, prefer it
    if abs(user_R - default_R) > 1e-6:
        logger.info(
            "SWA ratio: using user-specified R=%.4f (computed optimal=%.4f)",
            user_R,
            R_optimal,
        )
        return user_R

    logger.info(
        "SWA ratio: computed R_optimal=%.4f (window=%d, context=%d, "
        "fa_cost=%d*%d, swa_cost=%d*%d, R_window=%.4f, R_cost=%.4f)",
        R_optimal,
        sliding_window,
        context_len,
        fa_cell_per_layer,
        fa_layers,
        swa_cell_per_layer,
        swa_layers,
        R_window,
        R_cost,
    )
    return R_optimal
