"""DSV4_HCA_PAGED_ROW_WRITE: the paged writer lands the same rows as the XLA scatter."""

import os
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sgl_jax.srt.kernels.hca import attention as hca


@pytest.mark.parametrize("max_rows", [None, 40])
@pytest.mark.parametrize("seed", [0, 1])
def test_paged_writer_matches_scatter(seed, max_rows):
    key = jax.random.PRNGKey(seed)
    k1, k2, k3 = jax.random.split(key, 3)
    rows, width, head_dim, n = 512, 640, 512, 96
    cache = jax.random.normal(k1, (rows, width), jnp.float32).astype(jnp.bfloat16)
    values = jax.random.normal(k2, (n, head_dim), jnp.float32).astype(jnp.bfloat16)
    # a contiguous 16-aligned run, scattered rows, duplicates avoided, some invalid
    locs = np.concatenate(
        [np.arange(64, 80), np.random.RandomState(seed).permutation(rows)[: n - 16]]
    )
    valid = np.ones(n, bool)
    valid[::7] = False
    locs = locs.astype(np.int32)
    locs[5] = rows + 3  # out of range, must drop

    with mock.patch.object(hca, "_PAGED_ROW_WRITE", False):
        ref = hca._scatter_physical_rows(
            cache, jnp.asarray(locs), values, jnp.asarray(valid), max_rows=max_rows
        )
    with (
        mock.patch.object(hca, "_PAGED_ROW_WRITE", True),
        mock.patch.dict(os.environ, {"PALLAS_INTERPRET": "1"}),
    ):
        out = hca._scatter_physical_rows(
            cache, jnp.asarray(locs), values, jnp.asarray(valid), max_rows=max_rows
        )
    np.testing.assert_array_equal(np.asarray(out, np.float32), np.asarray(ref, np.float32))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_paged_writer_scattered_rows_with_tile_collisions(seed):
    """Decode-like commits: every row scattered, several rows per 16-row tile, duplicate
    destinations (last row wins), invalid rows interleaved. The batched read-modify-write
    must equal the serial scatter bit for bit."""
    from sgl_jax.srt.kernels.dsv4.paged_row_write import paged_row_write

    rng = np.random.RandomState(seed)
    rows, dim, n = 256, 512, 64
    cache = jnp.asarray(rng.standard_normal((rows, dim)), jnp.bfloat16)
    values = jnp.asarray(rng.standard_normal((n, dim)), jnp.bfloat16)
    # 64 rows into only 6 tiles -> heavy collisions inside and across 16-row groups
    locs = rng.randint(0, 6, size=n) * 16 + rng.randint(0, 16, size=n)
    locs[3] = locs[9]  # duplicate destination: row 9 must win
    locs[20] = locs[40]  # duplicate across groups: row 40 must win
    valid = rng.rand(n) > 0.2
    valid[9] = valid[40] = True
    safe = np.where(valid, locs, rows)
    ref = np.asarray(cache).copy()
    for i in range(n):  # serial scatter semantics
        if valid[i]:
            ref[locs[i]] = np.asarray(values)[i]
    out = paged_row_write(
        cache, values, jnp.asarray(locs, jnp.int32), jnp.asarray(valid), run=16, interpret=True
    )
    np.testing.assert_array_equal(np.asarray(out, np.float32), ref.astype(np.float32))
    out128 = paged_row_write(
        cache, values, jnp.asarray(locs, jnp.int32), jnp.asarray(valid), run=128, interpret=True
    )
    np.testing.assert_array_equal(np.asarray(out128, np.float32), ref.astype(np.float32))
    del safe
