"""Shared fixtures for cache-miss MRU (minimal reproducible unit) tests.

See docs/fix_eagle_cache_miss.html section "最小可复现单元 (MRU)" for the
overall verification strategy.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Iterator

import jax

logger = logging.getLogger(__name__)


@contextmanager
def cache_miss_probe(label: str = "probe") -> Iterator[list[int]]:
    """Count pjit cpp cache misses and surface them via jax_explain_cache_misses.

    Usage::

        with cache_miss_probe("my_test") as misses:
            jitted_fn(x)
            jitted_fn(y)   # same shape -> miss should still be 0
        assert misses[0] == 0

    The probe yields a one-element list updated when the context exits; check
    `misses[0]` after the block, not inside.
    """
    from jax._src import (
        test_util as jtu,  # private but stable; same API used in tp_worker
    )

    misses: list[int] = [0]
    prev_explain = bool(getattr(jax.config, "jax_explain_cache_misses", False))
    jax.config.update("jax_explain_cache_misses", True)
    try:
        with jtu.count_pjit_cpp_cache_miss() as count:
            yield misses
            misses[0] = count()
    finally:
        jax.config.update("jax_explain_cache_misses", prev_explain)
    logger.info("[cache_miss_probe:%s] miss=%d", label, misses[0])
