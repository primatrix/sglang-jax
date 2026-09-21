"""mHC fused seam (interpret) == post_reference followed by pre_reference."""

import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.kernels.mhc.seam import mhc_seam_fused
from sgl_jax.srt.layers.deepseek_v4_mhc import post_reference, pre_reference


def test_seam_matches_post_then_pre():
    rng = np.random.default_rng(0)
    T, hc, d = 21, 4, 256
    y = jnp.asarray(rng.standard_normal((T, d)), jnp.bfloat16)
    streams = jnp.asarray(rng.standard_normal((T, hc, d)), jnp.bfloat16)
    post = jnp.asarray(rng.uniform(0.5, 1.5, size=(T, hc)), jnp.float32)
    comb = jnp.asarray(rng.dirichlet(np.ones(hc), size=(T, hc)), jnp.float32)
    fn = jnp.asarray(rng.standard_normal((hc * hc + 2 * hc, hc * d)) * 0.02, jnp.float32)
    scale = jnp.asarray([0.7, 1.1, 0.9], jnp.float32)
    base = jnp.asarray(rng.standard_normal(hc * hc + 2 * hc) * 0.1, jnp.float32)
    kw = dict(hc_mult=hc, sinkhorn_iters=3, norm_eps=1e-6, hc_eps=1e-6)

    want_streams = post_reference(y, streams, post, comb).astype(jnp.bfloat16)
    want_hidden, want_post, want_comb = pre_reference(want_streams, fn, scale, base, **kw)

    got_streams, got_hidden, got_post, got_comb = mhc_seam_fused(
        y, streams, post, comb, fn, scale, base, interpret=True, **kw
    )
    f32 = lambda a: np.asarray(a, np.float32)  # noqa: E731
    # bf16 streams: one-ulp differences from summation order are expected
    np.testing.assert_allclose(f32(got_streams), f32(want_streams), rtol=1e-2, atol=1e-2)
    np.testing.assert_allclose(
        f32(got_hidden), f32(want_hidden.astype(jnp.bfloat16)), rtol=1e-2, atol=1e-2
    )
    np.testing.assert_allclose(f32(got_post), f32(want_post), rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(f32(got_comb), f32(want_comb), rtol=1e-4, atol=1e-5)
