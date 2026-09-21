"""Fused compressor tail kernel (interpret) matches select_window_fields + pool_normalize_rope."""

import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.kernels.dsv4.compressor_tail import compressor_tail_pallas
from sgl_jax.srt.layers.attention.dsv4.compressor import (
    pool_normalize_rope,
    select_window_fields,
)


def _case(N, D, ratio, coff, seed):
    rng = np.random.default_rng(seed)
    W, width = coff * ratio, coff * D
    combined = jnp.asarray(rng.standard_normal((N, W, 2 * width)), jnp.float32)
    valid = rng.random((N, W)) > 0.3
    valid[:, -1] = True  # the record's own row is always in sequence
    ang = rng.uniform(0, 6.28, size=(N, 32))
    return dict(
        combined=combined,
        valid=jnp.asarray(valid),
        norm_weight=jnp.asarray(rng.uniform(0.5, 1.5, size=D), jnp.float32),
        cos=jnp.asarray(np.cos(ang), jnp.float32),
        sin=jnp.asarray(np.sin(ang), jnp.float32),
        ratio=ratio,
        coff=coff,
        head_dim=D,
        width=width,
    )


def test_tail_matches_reference():
    for N, D, ratio, coff, seed in ((5, 512, 4, 2, 0), (16, 128, 4, 2, 1), (3, 128, 8, 1, 2)):
        c = _case(N, D, ratio, coff, seed)
        offsets = jnp.arange(coff * ratio)
        kv_w, sc_w = select_window_fields(
            c["combined"], offsets, ratio=ratio, coff=coff, head_dim=D, width=c["width"]
        )
        want = pool_normalize_rope(
            kv_w,
            sc_w,
            c["valid"],
            c["norm_weight"],
            c["cos"],
            c["sin"],
            rope_head_dim=64,
            norm_eps=1e-6,
        )
        got = compressor_tail_pallas(
            c["combined"],
            c["valid"],
            c["norm_weight"],
            c["cos"],
            c["sin"],
            ratio=ratio,
            coff=coff,
            head_dim=D,
            width=c["width"],
            rope_head_dim=64,
            norm_eps=1e-6,
            interpret=True,
        )
        assert got.shape == (N, D)
        np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=1e-4, atol=1e-4)
