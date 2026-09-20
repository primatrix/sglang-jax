"""``searchsorted(side="right")`` for a short sorted table by compare-and-count.

``jnp.searchsorted`` lowers to an XLA ``while`` (binary search with select_n
bodies); for the HCA metadata the table is ``cu_q_lens`` / ``query_starts`` with
at most a few dozen entries, so a [N, B+1] compare + row sum is one fusion and
gives the identical integer result. On by default; ``DSV4_HCA_SEARCH_COMPARE=0``
falls back to ``jnp.searchsorted``.
"""

from __future__ import annotations

import os

import jax.numpy as jnp


def _compare_enabled() -> bool:
    return os.environ.get("DSV4_HCA_SEARCH_COMPARE", "1") == "1"


def searchsorted_right(table, values):
    """Number of ``table`` entries ``<= v`` for each ``v`` (``table`` sorted ascending)."""
    table = jnp.asarray(table)
    values = jnp.asarray(values)
    if not _compare_enabled():
        return jnp.searchsorted(table, values, side="right")
    return jnp.sum(
        (table.reshape((1,) * values.ndim + (-1,)) <= values[..., None]).astype(jnp.int32),
        axis=-1,
    )
