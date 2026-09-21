"""Batches smaller than one mHC token block run unpadded (``DSV4_MHC_NOPAD_SMALL=1``).

At bs=1 decode the pre/post launchers padded the streams to 8 rows and the gate
mixes to a lane block in every layer.  With the switch on, the block equals the
batch; the results must match the padded path.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sgl_jax.srt.kernels.mhc import mhc as mhc
from sgl_jax.srt.kernels.mhc.mhc import mhc_gates, mhc_post_fused, mhc_pre_fused

HC, D = 4, 512


def _interpret(monkeypatch):
    monkeypatch.setenv("PALLAS_INTERPRET", "1")
    monkeypatch.setattr(mhc, "_device_kind", lambda: "TPU7x")


def _pre(n, key):
    k1, k2, k3 = jax.random.split(key, 3)
    rows = (2 + HC) * HC
    x = jax.random.normal(k1, (n, HC, D), jnp.bfloat16)
    fn = jax.random.normal(k2, (rows, HC * D), jnp.float32) * 0.02
    scale = jnp.asarray([0.7, 1.1, 0.9], jnp.float32)
    base = jax.random.normal(k3, (rows,), jnp.float32) * 0.1
    return mhc_pre_fused(
        x, fn, scale, base, hc_mult=HC, sinkhorn_iters=20, norm_eps=1e-6, hc_eps=1e-6
    )


def _post(n, key):
    k1, k2, k3, k4 = jax.random.split(key, 4)
    x = jax.random.normal(k1, (n, D), jnp.bfloat16)
    res = jax.random.normal(k2, (n, HC, D), jnp.bfloat16)
    post = jax.random.normal(k3, (n, HC), jnp.float32)
    comb = jax.random.normal(k4, (n, HC, HC), jnp.float32)
    return mhc_post_fused(x, res, post, comb, backend="pallas")


def _gates(n, key):
    k1, k2 = jax.random.split(key, 2)
    mix_hc = mhc.mix_hc_width(HC)
    mixes = jax.random.normal(k1, (n, mix_hc), jnp.float32)
    base = jax.random.normal(k2, (mix_hc,), jnp.float32) * 0.1
    scale = jnp.asarray([0.7, 1.1, 0.9], jnp.float32)
    return mhc_gates(mixes, scale, base, hc_mult=HC, sinkhorn_iters=20, eps=1e-6)


def _leaves(out):
    return [np.asarray(jnp.asarray(a).astype(jnp.float32)) for a in jax.tree.leaves(out)]


@pytest.mark.parametrize("n", [1, 3])
@pytest.mark.parametrize("fn", [_pre, _post, _gates])
def test_small_batch_unpadded_matches_padded(monkeypatch, n, fn):
    _interpret(monkeypatch)
    key = jax.random.PRNGKey(n)
    monkeypatch.setenv("DSV4_MHC_NOPAD_SMALL", "0")
    padded = _leaves(fn(n, key))
    monkeypatch.setenv("DSV4_MHC_NOPAD_SMALL", "1")
    assert mhc.nopad_small_enabled()
    unpadded = _leaves(fn(n, key))
    assert len(padded) == len(unpadded)
    for a, b in zip(padded, unpadded, strict=True):
        assert a.shape == b.shape
        np.testing.assert_allclose(b, a, rtol=1e-2, atol=2e-2)
