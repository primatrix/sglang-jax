"""Fused inverse-RoPE + wo_a kernel (interpret) matches the model's XLA path."""

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.kernels.dsv4.wo_a_projection import widen_cos_sin, wo_a_projection
from sgl_jax.srt.layers.attention.dsv4.o_projection import group_wo_a
from sgl_jax.srt.layers.attention.dsv4.rope import apply_dsv4_partial_rope


def _reference(x, cos, sin, wo_a_ck, *, num_groups, rope_head_dim):
    # deepseek_v4.py: inverse partial rope -> regroup -> tgd,gdr->tgr (bf16 x, f32 acc)
    out = apply_dsv4_partial_rope(
        x, cos[:, None, :], sin[:, None, :], rope_head_dim=rope_head_dim, inverse=True
    ).astype(jnp.bfloat16)
    T = out.shape[0]
    grouped = out.reshape(T, num_groups, -1)
    weights = group_wo_a(wo_a_ck, num_groups=num_groups)  # [G, D, R]
    reduced = jnp.einsum(
        "tgd,gdr->tgr",
        grouped.astype(jnp.float32),
        weights.astype(jnp.float32),
        preferred_element_type=jnp.float32,
    )
    return reduced.reshape(T, -1).astype(jnp.bfloat16)


def test_fused_wo_a_matches_reference():
    rng = np.random.default_rng(0)
    T, G, D, R, RD = 16, 2, 256, 64, 64
    H = 8 * G
    x = jnp.asarray(rng.standard_normal((T, H, D)), jnp.bfloat16)
    ang = rng.uniform(0, 6.28, size=(T, RD // 2))
    cos, sin = jnp.asarray(np.cos(ang), jnp.float32), jnp.asarray(np.sin(ang), jnp.float32)
    wo_a_ck = jnp.asarray(rng.standard_normal((G * R, 8 * D)) * 0.05, jnp.bfloat16)  # [G*R, 8D]
    want = np.asarray(_reference(x, cos, sin, wo_a_ck, num_groups=G, rope_head_dim=RD), np.float32)
    cos_sin = widen_cos_sin(cos, sin, rope_head_dim=RD, inverse=True)
    got = np.asarray(wo_a_projection(x, wo_a_ck.T, cos_sin, tile_t=8, interpret=True), np.float32)
    assert got.shape == want.shape == (T, G * R)
    np.testing.assert_allclose(got, want, rtol=2e-2, atol=2e-2)


def test_widen_cos_sin_identity_on_nope_lanes():
    cos = jnp.ones((3, 32)) * 0.5
    sin = jnp.ones((3, 32)) * 0.25
    cs = np.asarray(widen_cos_sin(cos, sin, rope_head_dim=64, inverse=True))
    assert cs.shape == (3, 256)
    np.testing.assert_array_equal(cs[:, :64], 1.0)  # NoPE lanes: cos 1
    np.testing.assert_array_equal(cs[:, 128:192], 0.0)  # NoPE lanes: sin 0
    np.testing.assert_array_equal(cs[:, 64:128], 0.5)
    np.testing.assert_array_equal(cs[:, 192:], -0.25)  # inverse negates sin
