"""The gathered-key decode attention kernel matches the XLA einsum path.

Long-path decode (capacity above the top-k budget): scorer -> exact top-k -> gather ->
attention; the kernel replaces only the last stage. Runs on CPU in interpret mode.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from sgl_jax.srt.kernels.dsv4.csa_decode_attention import gathered_decode_attention

NEG = float(jnp.finfo(jnp.float32).min)


def _reference(q, window, compressed, window_valid, selected_valid, sink, scale):
    q = np.asarray(q.astype(jnp.float32))
    keys = np.concatenate(
        (np.asarray(window.astype(jnp.float32)), np.asarray(compressed.astype(jnp.float32))), 1
    )
    mask = np.concatenate((np.asarray(window_valid), np.asarray(selected_valid)), 1)
    sink = np.asarray(sink, np.float32)
    out = np.zeros(q.shape, np.float32)
    for t in range(q.shape[0]):
        s = (q[t] @ keys[t].T) * scale  # [H, K]
        s = np.where(mask[t][None], s, NEG)
        shift = np.maximum(s.max(-1, keepdims=True), sink[:, None])
        p = np.where(mask[t][None], np.exp(s - shift), 0.0)
        den = p.sum(-1, keepdims=True) + np.exp(sink[:, None] - shift)
        out[t] = (p @ keys[t]) / den
    return out


@pytest.mark.parametrize("tokens,rows", [(3, 4), (9, 4), (5, 1)])
def test_kernel_matches_numpy_reference(tokens, rows):
    rng = np.random.default_rng(tokens)
    H, D, W, S = 8, 512, 128, 512
    q = jnp.asarray(rng.standard_normal((tokens, H, D)), jnp.bfloat16)
    window = jnp.asarray(rng.standard_normal((tokens, W, D)), jnp.bfloat16)
    compressed = jnp.asarray(rng.standard_normal((tokens, S, D)), jnp.bfloat16)
    window_valid = jnp.asarray(rng.random((tokens, W)) > 0.3)
    selected_valid = jnp.asarray(rng.random((tokens, S)) > 0.5)
    window_valid = window_valid.at[0].set(False)  # a padded row: sink only -> zeros
    selected_valid = selected_valid.at[0].set(False)
    sink = jnp.asarray(rng.standard_normal((H,)), jnp.float32)
    out = np.asarray(
        gathered_decode_attention(
            q,
            window,
            compressed,
            window_valid,
            selected_valid,
            sink,
            softmax_scale=D**-0.5,
            rows_per_step=rows,
            interpret=True,
        )
    )
    ref = _reference(q, window, compressed, window_valid, selected_valid, sink, D**-0.5)
    assert out.shape == (tokens, H, D)
    np.testing.assert_allclose(out, ref, rtol=2e-4, atol=2e-4)
    assert np.all(out[0] == 0)
