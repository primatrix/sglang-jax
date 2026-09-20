"""silu_mul_rows matches the whole-buffer activation on the local rows and zeros the rest."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sgl_jax.srt.kernels.dsv4.moe_act import silu_mul_rows
from sgl_jax.srt.layers.activation import silu_and_mul_with_clamp


@pytest.mark.parametrize(
    "rows,start,end", [(4096, 700, 2900), (4096, 0, 4096), (4096, 1024, 1024), (4000, 3500, 3999)]
)
@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float32])
def test_rows_match(rows, start, end, dtype):
    k1, k2 = jax.random.split(jax.random.PRNGKey(rows + start))
    gate = (jax.random.normal(k1, (rows, 256), jnp.float32) * 8).astype(dtype)
    up = (jax.random.normal(k2, (rows, 256), jnp.float32) * 8).astype(dtype)
    ref = np.asarray(silu_and_mul_with_clamp(gate, up, 10.0), np.float32)
    out = np.asarray(
        jax.jit(lambda g, u, s, e: silu_mul_rows(g, u, s, e, limit=10.0, interpret=True))(
            gate, up, jnp.int32(start), jnp.int32(end)
        ),
        np.float32,
    )
    assert out.shape == ref.shape
    np.testing.assert_array_equal(out[start:end], ref[start:end])
    block = 512
    first, last = start // block, (end + block - 1) // block - 1
    for b in range(-(-rows // block)):
        if b < first or b > last:
            assert not out[b * block : (b + 1) * block].any()
