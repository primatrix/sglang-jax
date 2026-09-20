"""Fused q head-norm + rope kernel against the model's XLA formulation (CPU interpret)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sgl_jax.srt.kernels.dsv4.q_head_norm_rope import q_head_norm_rope, rope_tail_tables
from sgl_jax.srt.layers.attention.dsv4.rope import apply_dsv4_partial_rope

H, D, R = 8, 512, 64


def _reference(q2d, cos, sin, *, normalize, eps):
    q = q2d.reshape(-1, H, D)
    if normalize:
        q = (
            q.astype(jnp.float32)
            * jax.lax.rsqrt(
                jnp.mean(jnp.square(q.astype(jnp.float32)), axis=-1, keepdims=True) + eps
            )
        ).astype(jnp.bfloat16)
    q = apply_dsv4_partial_rope(q, cos[:, None, :], sin[:, None, :], rope_head_dim=R)
    return q.astype(jnp.bfloat16).reshape(-1, D)


@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("tokens", [16, 37])
def test_kernel_matches_reference(normalize, tokens):
    rng = np.random.default_rng(0)
    q = jnp.asarray(rng.standard_normal((tokens, H * D)) * 3, jnp.bfloat16)
    ang = rng.standard_normal((tokens, R // 2)).astype(np.float32)
    cos, sin = jnp.asarray(np.cos(ang)), jnp.asarray(np.sin(ang))
    got = q_head_norm_rope(
        q, cos, sin, heads=H, head_dim=D, rope_head_dim=R, normalize=normalize, eps=1e-6
    )
    want = _reference(q, cos, sin, normalize=normalize, eps=1e-6)
    assert got.shape == want.shape == (tokens * H, D)
    got32, want32 = np.asarray(got, np.float32), np.asarray(want, np.float32)
    # Non-rope lanes are untouched by the rotation: bit-identical when not normalising.
    if not normalize:
        np.testing.assert_array_equal(got32[:, : D - R], want32[:, : D - R])
    np.testing.assert_allclose(got32, want32, rtol=2e-2, atol=2e-2)
    # Rounding-level only: the fraction of differing bf16 values stays tiny.
    assert np.mean(got32 != want32) < 0.02


def test_rope_tail_tables_are_exact():
    rng = np.random.default_rng(1)
    ang = rng.standard_normal((5, R // 2)).astype(np.float32)
    cos_full, sin_full = rope_tail_tables(np.cos(ang), np.sin(ang), R)
    cos_full, sin_full = np.asarray(cos_full), np.asarray(sin_full)
    start = 128 - R
    np.testing.assert_array_equal(cos_full[:, :start], 1.0)
    np.testing.assert_array_equal(sin_full[:, :start], 0.0)
    for i in range(R // 2):
        for lane in (start + 2 * i, start + 2 * i + 1):
            np.testing.assert_array_equal(cos_full[:, lane], np.cos(ang)[:, i])
            np.testing.assert_array_equal(sin_full[:, lane], np.sin(ang)[:, i])
