"""Decode buckets whose capacity fits the top-k budget skip scoring/selection with identical output."""

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.layers.attention.dsv4.decode import csa_decode_attention

B, H, D, DIDX, PAGE, NPAGES, W, RATIO, TOPK = 3, 4, 128, 128, 8, 16, 16, 4, 512


def _inputs(seed=0):
    k = jax.random.split(jax.random.PRNGKey(seed), 8)
    cap = NPAGES * PAGE  # 128 <= TOPK: the shortcut applies
    total_pages = 64
    q = jax.random.normal(k[0], (B, H, D), jnp.bfloat16)
    index_q = jax.random.normal(k[1], (B, H, DIDX), jnp.bfloat16)
    index_weights = jax.nn.softmax(jax.random.normal(k[2], (B, H), jnp.float32), axis=-1)
    index_cache = jax.random.normal(k[3], (total_pages, PAGE, DIDX), jnp.bfloat16)
    compressed_cache = jax.random.normal(k[4], (total_pages * PAGE, D), jnp.bfloat16)
    window_cache = jax.random.normal(k[5], (1024, D), jnp.bfloat16)
    rng = np.random.default_rng(seed)
    pages = jnp.asarray(
        rng.permutation(total_pages)[: B * NPAGES].reshape(B, NPAGES).astype(np.int32)
    )
    window_rows = jnp.asarray(rng.permutation(1024)[: B * W].reshape(B, W).astype(np.int32))
    positions = jnp.asarray([300, 40, 0], jnp.int32)  # lengths 75, 10, (padded)
    valid = jnp.asarray([True, True, False])
    sink = jax.random.normal(k[6], (H,), jnp.float32)
    assert cap <= TOPK
    return dict(
        q=q,
        index_q=index_q,
        index_weights=index_weights,
        index_cache=index_cache,
        compressed_cache=compressed_cache,
        window_cache=window_cache,
        pages=pages,
        window_rows=window_rows,
        query_positions=positions,
        valid_token_mask=valid,
        attention_sink=sink,
        softmax_scale=D**-0.5,
        compressed_page_size=PAGE,
        index_topk=TOPK,
        ratio=RATIO,
    )


def _run():
    return np.asarray(csa_decode_attention(**_inputs()).astype(jnp.float32))


def test_short_kv_kernel_matches_xla_bypass(monkeypatch):
    monkeypatch.setenv("DSV4_DECODE_SHORT_KV_KERNEL", "0")
    ref = _run()
    monkeypatch.setenv("DSV4_DECODE_SHORT_KV_KERNEL", "1")
    out = _run()
    assert np.isfinite(out).all()
    assert np.all(out[2] == 0)
    np.testing.assert_allclose(out, ref, rtol=2e-2, atol=2e-2)
