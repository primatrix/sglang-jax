"""L0 unit tests for cache_miss_probe fixture.

Verifies:
- Same-shape repeated jit calls produce miss=0.
- Different-shape jit calls produce miss>0.
- Probe restores jax_explain_cache_misses to its previous value.

Run on niu-v6e4-sleep (per docs/fix_eagle_cache_miss.html):
    kubectl exec niu-v6e4-sleep -- bash -lc '
      cd /tmp/sglang-jax-baseline && \
      python3 -m pytest test/srt/speculative/test_cache_miss_probe.py -x -q
    '
"""

from __future__ import annotations

from test.srt.speculative.utils import cache_miss_probe

import jax
import jax.numpy as jnp
import pytest


@jax.jit
def _trivial_add(x, y):
    return x + y


def test_same_shape_no_miss():
    a = jnp.ones((4,), dtype=jnp.float32)
    b = jnp.ones((4,), dtype=jnp.float32)
    _trivial_add(a, b).block_until_ready()  # warm up
    with cache_miss_probe("same_shape") as misses:
        for _ in range(5):
            _trivial_add(a, b).block_until_ready()
    assert misses[0] == 0, f"expected 0 misses on repeated same-shape call, got {misses[0]}"


def test_different_shape_triggers_miss():
    _trivial_add(jnp.ones((4,)), jnp.ones((4,))).block_until_ready()  # warm up shape 4
    with cache_miss_probe("varied_shape") as misses:
        _trivial_add(jnp.ones((8,)), jnp.ones((8,))).block_until_ready()
        _trivial_add(jnp.ones((16,)), jnp.ones((16,))).block_until_ready()
    assert misses[0] >= 2, f"expected >=2 misses on new shapes, got {misses[0]}"


def test_probe_restores_explain_flag():
    jax.config.update("jax_explain_cache_misses", False)
    with cache_miss_probe("restore"):
        assert jax.config.jax_explain_cache_misses is True
    assert jax.config.jax_explain_cache_misses is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
