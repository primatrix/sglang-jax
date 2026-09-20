"""The boundary-emit kernel reading the state pool in its own ``[slots,128,2,512]`` layout.

The default launcher reshapes the pool to ``[slots,128,2,4,128]``, which XLA turns
into a copy of the whole pool per HCA layer per step.  ``DSV4_HCA_EMIT_NATIVE=1``
DMAs the rows as stored; the emitted records must match the default path.
"""

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.kernels.hca import compressor as hca_compressor
from sgl_jax.srt.kernels.hca.compressor import hca_state_pool_emit_pallas
from sgl_jax.srt.kernels.hca.tuned_block_sizes import get_hca_kernel_schedule


def _inputs(key):
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
    inputs = _inputs(jax.random.PRNGKey(5))
    monkeypatch.setenv("DSV4_HCA_EMIT_NATIVE", "0")  # default on since pfbase14 (09-19)
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
    pool, slots, _, w, cos, sin, sched = _inputs(jax.random.PRNGKey(3))
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
