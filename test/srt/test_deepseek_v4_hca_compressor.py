"""HCA compressor state boundaries, native emission and invalid rows."""

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.kernels.hca import compressor as hca
from sgl_jax.srt.kernels.hca import compressor as hca_compressor
from sgl_jax.srt.kernels.hca.compressor import hca_state_pool_emit_pallas
from sgl_jax.srt.kernels.hca.tuned_block_sizes import get_hca_kernel_schedule


def _boundary_inputs(seed=0):
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
    ref, ref_mask, ref_pool = _run(*_boundary_inputs())
    monkeypatch.setenv("DSV4_HCA_BOUNDARY_NATIVE", "1")
    assert hca._boundary_native_enabled()
    got, got_mask, got_pool = _run(*_boundary_inputs())
    np.testing.assert_array_equal(got_mask, ref_mask)
    assert ref_mask.sum() == 8
    np.testing.assert_array_equal(got_pool, ref_pool)
    assert np.isfinite(got).all()
    rows = np.nonzero(ref_mask)[0]
    assert np.abs(ref[rows]).max() > 0
    np.testing.assert_allclose(got[rows], ref[rows], rtol=1e-2, atol=1e-2)
    np.testing.assert_array_equal(got[~ref_mask], 0.0)


def _emit_inputs(key):
    k1, k2, k3, k4 = jax.random.split(key, 4)
    slots, packed = 6, 5
    pool = jax.random.normal(k1, (slots, 128, 2, 512), jnp.float32)
    request_slots = jnp.asarray([3, 0, 5, 1, 2], jnp.int32)
    valid = jnp.asarray([True, True, False, True, True])
    norm_weight = 1.0 + 0.1 * jax.random.normal(k2, (512,), jnp.float32)
    angle = jax.random.uniform(k3, (packed, 32), jnp.float32, 0, 6.28)
    cos, sin = jnp.cos(angle), jnp.sin(angle)
    sched = get_hca_kernel_schedule(
        "TPU7x", page_size=1, max_compressed_entries=64, local_heads=8, head_dim=512
    )
    return pool, request_slots, valid, norm_weight, cos, sin, sched


def _emit(inputs):
    pool, slots, valid, w, cos, sin, sched = inputs
    hca_compressor._hca_emit_pool_pallas.clear_cache()
    hca_state_pool_emit_pallas.clear_cache()
    out = hca_state_pool_emit_pallas(pool, slots, valid, w, cos, sin, schedule=sched)
    return np.asarray(out.astype(jnp.float32))


def test_native_layout_emit_matches_default(monkeypatch):
    inputs = _emit_inputs(jax.random.PRNGKey(5))
    monkeypatch.setenv("DSV4_HCA_EMIT_NATIVE", "0")
    default = _emit(inputs)
    monkeypatch.setenv("DSV4_HCA_EMIT_NATIVE", "1")
    assert hca_compressor._emit_native_layout()
    native = _emit(inputs)
    assert native.shape == default.shape == (5, 512)
    assert np.isfinite(native).all()
    # Invalid rows are zero on both paths; valid rows agree to bf16 rounding.
    np.testing.assert_array_equal(native[2], 0.0)
    np.testing.assert_allclose(native, default, rtol=1e-2, atol=1e-2)
    assert np.abs(default[[0, 1, 3, 4]]).max() > 0


def test_native_layout_emit_all_invalid_tile_is_zero(monkeypatch):
    monkeypatch.setenv("DSV4_HCA_EMIT_NATIVE", "1")
    pool, slots, _, w, cos, sin, sched = _emit_inputs(jax.random.PRNGKey(3))
    valid = jnp.zeros((slots.shape[0],), bool)
    out = _emit((pool, slots, valid, w, cos, sin, sched))
    assert out.shape == (slots.shape[0], 512)
    assert np.all(out == 0)
    # one live row in an otherwise empty batch still emits that row
    valid = valid.at[2].set(True)
    ref = _emit((pool, slots, valid, w, cos, sin, sched))
    monkeypatch.setenv("DSV4_HCA_EMIT_NATIVE", "0")
    default = _emit((pool, slots, valid, w, cos, sin, sched))
    np.testing.assert_allclose(ref, default, rtol=1e-2, atol=1e-2)
    assert np.all(ref[[0, 1, 3, 4]] == 0) and np.any(ref[2] != 0)
