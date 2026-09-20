"""DSV4_HCA_BOUNDARY_NATIVE: boundary snapshots pooled from the ring + chunk rows in the
kernel match the gather-based snapshot path (windows straddling prefix / chunk, padded
boundaries)."""

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.kernels.hca import compressor as hca
from sgl_jax.srt.kernels.hca.tuned_block_sizes import get_hca_kernel_schedule


def _inputs(seed=0):
    k = jax.random.split(jax.random.PRNGKey(seed), 6)
    T, hidden, D, R = 1024, 4096, 512, 128
    x = jax.random.normal(k[0], (T, hidden), jnp.float32).astype(jnp.bfloat16)
    pool = jax.random.normal(k[1], (4, R, 2, D), jnp.float32)
    w = (0.02 * jax.random.normal(k[2], (hidden, 2 * D), jnp.float32)).astype(jnp.bfloat16)
    ape = 0.1 * jax.random.normal(k[3], (R, D), jnp.float32)
    nw = 1.0 + 0.1 * jax.random.normal(k[4], (D,), jnp.float32)
    ang = jax.random.uniform(k[5], (2048, 32), jnp.float32, 0, 6.28)
    cos, sin = jnp.cos(ang), jnp.sin(ang)
    # request 0: prefix 300, 640 new tokens (positions 300..939); request 1: prefix 0, 384 tokens
    query_starts = jnp.asarray([0, 640], jnp.int32)
    prefix_lens = jnp.asarray([300, 0], jnp.int32)
    seq_lens = jnp.asarray([940, 384], jnp.int32)
    positions = jnp.concatenate([300 + jnp.arange(640), jnp.arange(384)]).astype(jnp.int32)
    request_slots = jnp.concatenate([jnp.full((640,), 2), jnp.full((384,), 0)]).astype(jnp.int32)
    b0 = [p - 300 for p in (383, 511, 639, 767, 895)]
    b1 = [640 + p for p in (127, 255, 383)]
    boundaries = jnp.asarray(b0 + b1 + [T, T], jnp.int32)  # two sentinel pads
    sched = get_hca_kernel_schedule(
        "TPU7x", page_size=1, max_compressed_entries=64, local_heads=8, head_dim=D
    )
    return (
        dict(
            x=x,
            state_pool=pool,
            fused_weight=w,
            ape=ape,
            norm_weight=nw,
            cos=cos,
            sin=sin,
            positions=positions,
            request_slots=request_slots,
            query_starts=query_starts,
            prefix_lens=prefix_lens,
            seq_lens=seq_lens,
            boundary_token_indices=boundaries,
        ),
        sched,
    )


def _run(inputs, sched):
    hca.hca_state_pool_update_ragged_fused_pallas.clear_cache()
    emitted, emit_mask, pool = hca.hca_state_pool_update_ragged_fused_pallas(
        **inputs, schedule=sched, compress_ratio=128, head_dim=512, norm_eps=1e-6
    )
    return np.asarray(emitted.astype(jnp.float32)), np.asarray(emit_mask), np.asarray(pool)


def test_boundary_native_matches_gather_path(monkeypatch):
    # state_pool is donated by the update: build the inputs afresh for each run.
    monkeypatch.setenv("DSV4_HCA_BOUNDARY_NATIVE", "0")
    ref, ref_mask, ref_pool = _run(*_inputs())
    monkeypatch.setenv("DSV4_HCA_BOUNDARY_NATIVE", "1")
    assert hca._boundary_native_enabled()
    got, got_mask, got_pool = _run(*_inputs())
    np.testing.assert_array_equal(got_mask, ref_mask)
    assert ref_mask.sum() == 8
    np.testing.assert_array_equal(got_pool, ref_pool)
    assert np.isfinite(got).all()
    rows = np.nonzero(ref_mask)[0]
    assert np.abs(ref[rows]).max() > 0
    np.testing.assert_allclose(got[rows], ref[rows], rtol=1e-2, atol=1e-2)
    np.testing.assert_array_equal(got[~ref_mask], 0.0)
