"""searchsorted_right by compare-and-count equals jnp.searchsorted(side="right")."""

from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sgl_jax.srt.kernels.hca import search


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("n_table,n_vals", [(17, 128), (5, 70), (2, 9)])
def test_matches_searchsorted(seed, n_table, n_vals):
    key = jax.random.PRNGKey(seed)
    k1, k2 = jax.random.split(key)
    lens = jax.random.randint(k1, (n_table - 1,), 0, 50)
    table = jnp.concatenate([jnp.zeros((1,), jnp.int32), jnp.cumsum(lens)]).astype(jnp.int32)
    vals = jax.random.randint(k2, (n_vals,), -3, int(table[-1]) + 4).astype(jnp.int32)
    ref = np.asarray(jnp.searchsorted(table, vals, side="right"))
    with mock.patch.object(search, "_COMPARE", True):
        got = np.asarray(search.searchsorted_right(table, vals))
    np.testing.assert_array_equal(got, ref)
    # ties on table entries (duplicates from zero-length requests) behave the same
    vals2 = table
    with mock.patch.object(search, "_COMPARE", True):
        got2 = np.asarray(search.searchsorted_right(table, vals2))
    np.testing.assert_array_equal(got2, np.asarray(jnp.searchsorted(table, vals2, side="right")))
