"""Slice-free interleaved RoPE equals the reference slice/stack form (CPU-only)."""

import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.layers.attention.dsv4.compressor import interleaved_rope


def _reference(x, cos, sin, rope_head_dim):
    head_dim = x.shape[-1]
    start = head_dim - rope_head_dim
    head, tail = x[..., :start], x[..., start:]
    even, odd = tail[..., 0::2], tail[..., 1::2]
    rotated_even = even * cos - odd * sin
    rotated_odd = even * sin + odd * cos
    tail = jnp.stack((rotated_even, rotated_odd), axis=-1).reshape(tail.shape)
    return jnp.concatenate((head, tail), axis=-1)


def test_matches_reference_bitwise():
    rng = np.random.default_rng(0)
    for n, d, r in ((5, 128, 64), (3, 128, 128), (7, 64, 32), (2, 128, 0)):
        x = rng.standard_normal((n, d)).astype(np.float32)
        ang = rng.uniform(-3, 3, size=(n, max(r // 2, 1))).astype(np.float32)
        cos, sin = np.cos(ang), np.sin(ang)
        if r == 0:
            cos, sin = cos[:, :0], sin[:, :0]
        got = np.asarray(interleaved_rope(jnp.asarray(x), jnp.asarray(cos), jnp.asarray(sin), r))
        ref = np.asarray(_reference(jnp.asarray(x), jnp.asarray(cos), jnp.asarray(sin), r))
        np.testing.assert_array_equal(got, ref)


def test_rejects_bad_rope_dim():
    import pytest

    with pytest.raises(ValueError):
        interleaved_rope(jnp.zeros((2, 128)), jnp.zeros((2, 3)), jnp.zeros((2, 3)), 7)
