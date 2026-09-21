"""The fused CSA prefill attention kernel (interpret mode) matches a dense NumPy reference.

Semantics per query ``t`` and head ``h``: softmax over the admissible keys plus a
per-head sink logit that takes probability mass but contributes no value; queries
with no admissible key return exactly zero.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from sgl_jax.srt.kernels.dsv4.csa_flash_attention import csa_flash_attention


def _reference(q, keys, mask, sink, scale):
    q = np.asarray(q.astype(jnp.float32))
    k = np.asarray(keys.astype(jnp.float32))
    mask = np.asarray(mask, bool)
    sink = np.asarray(sink, np.float32)
    out = np.zeros(q.shape, np.float32)
    for t in range(q.shape[0]):
        if not mask[t].any():
            continue
        s = (q[t] @ k.T) * scale  # [H, N]
        s = np.where(mask[t][None], s, -np.inf)
        m = np.maximum(s.max(-1, keepdims=True), sink[:, None])
        p = np.exp(s - m)
        den = p.sum(-1, keepdims=True) + np.exp(sink[:, None] - m)
        out[t] = (p @ k) / den
    return out


@pytest.mark.parametrize(
    "T,N,H,D,block_q,block_k,seed",
    [
        (16, 300, 8, 128, 256, 512, 0),  # one q block, one padded k block
        (40, 1100, 4, 128, 16, 512, 1),  # several q blocks, several k blocks
        (9, 64, 2, 128, 8, 128, 2),  # tiny
    ],
)
def test_flash_matches_dense(T, N, H, D, block_q, block_k, seed):
    rng = np.random.default_rng(seed)
    q = jnp.asarray(rng.standard_normal((T, H, D)), jnp.bfloat16)
    keys = jnp.asarray(rng.standard_normal((N, D)), jnp.bfloat16)
    mask = rng.random((T, N)) > 0.4
    mask[0] = False  # a query admitting nothing -> exact zero
    mask[1, :] = False
    mask[1, 5] = True  # a single key
    sink = jnp.asarray(rng.standard_normal(H) * 2, jnp.float32)
    got = np.asarray(
        csa_flash_attention(
            q,
            keys,
            jnp.asarray(mask),
            sink,
            sm_scale=D**-0.5,
            block_q=block_q,
            block_k=block_k,
            interpret=True,
        )
    )
    want = _reference(q, keys, mask, sink, D**-0.5)
    assert got.shape == (T, H, D) and np.isfinite(got).all()
    np.testing.assert_allclose(got, want, rtol=1e-3, atol=1e-3)
    np.testing.assert_array_equal(got[0], 0.0)
