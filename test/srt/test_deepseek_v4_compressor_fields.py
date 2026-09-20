"""select_window_fields (static slices + select) equals the take_along_axis form (CPU-only)."""

import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.layers.attention.dsv4.compressor import select_window_fields


def _reference(combined, offsets, ratio, coff, head_dim, width):
    window = offsets.shape[0]
    field = (
        (offsets >= ratio).astype(jnp.int32) * head_dim
        if coff == 2
        else jnp.zeros((window,), jnp.int32)
    )
    cols = field[None, :, None] + jnp.arange(head_dim)[None, None, :]
    return (
        jnp.take_along_axis(combined[..., :width], cols, axis=2),
        jnp.take_along_axis(combined[..., width:], cols, axis=2),
    )


def test_matches_reference():
    rng = np.random.default_rng(0)
    for ratio, coff in ((4, 2), (128, 1)):
        head_dim, width = 128, coff * 128
        window = coff * ratio if ratio == 4 else 8  # keep the ratio-128 case small
        n = 5
        combined = jnp.asarray(rng.standard_normal((n, window, 2 * width)).astype(np.float32))
        offsets = jnp.arange(window)
        got = select_window_fields(
            combined, offsets, ratio=ratio, coff=coff, head_dim=head_dim, width=width
        )
        ref = _reference(combined, offsets, ratio, coff, head_dim, width)
        for g, r in zip(got, ref):
            np.testing.assert_array_equal(np.asarray(g), np.asarray(r))
