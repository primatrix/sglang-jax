"""Compacting the valid rows before the HCA cache scatter does not change the result."""

import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.kernels.hca.attention import _scatter_physical_rows


def test_compacted_scatter_matches_masked_scatter():
    rng = np.random.default_rng(0)
    rows, D, T = 512, 64, 300
    cache = jnp.asarray(rng.standard_normal((rows, D)), jnp.bfloat16)
    values = jnp.asarray(rng.standard_normal((T, D)), jnp.bfloat16)
    # distinct destinations for the valid rows (duplicates are undefined in both paths)
    locs = rng.permutation(rows)[:T].astype(np.int32)
    valid = rng.random(T) < 0.15
    locs[~valid] = -1
    want = _scatter_physical_rows(cache, jnp.asarray(locs), values, jnp.asarray(valid))
    for max_rows in (int(valid.sum()), int(valid.sum()) + 7, T - 1, T, 10_000):
        got = _scatter_physical_rows(
            cache, jnp.asarray(locs), values, jnp.asarray(valid), max_rows=max_rows
        )
        if max_rows >= int(valid.sum()):
            np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
    # no valid rows at all
    got = _scatter_physical_rows(cache, jnp.asarray(locs), values, jnp.zeros(T, bool), max_rows=8)
    np.testing.assert_array_equal(np.asarray(got), np.asarray(cache))
