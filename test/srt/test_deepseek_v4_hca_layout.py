"""The ratio-128 compressor state lives in the HCA kernels' physical layout and the
kernels accept the SWA pool's flat rows, so no HCA buffer is relaid out per step."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sgl_jax.srt.kernels.hca.attention import _cache_layout
from sgl_jax.srt.mem_cache.deepseek_v4.pool import (
    DeepseekV4CacheSpec,
    DeepseekV4TokenToKVPool,
)
from sgl_jax.srt.mem_cache.deepseek_v4.state import (
    DeepseekV4CompressStatePool,
    native_hca_layout,
    score_slice,
)

D = 64


def _mesh():
    return jax.sharding.Mesh(
        np.asarray(jax.devices()[:1], object).reshape(1, 1),
        ("data", "tensor"),
        axis_types=(jax.sharding.AxisType.Explicit,) * 2,
    )


def _pool(monkeypatch, native):
    monkeypatch.setenv("DSV4_HCA_NATIVE_LAYOUT", "1" if native else "0")
    assert native_hca_layout() is native
    spec = DeepseekV4CacheSpec(compress_ratios=(4, 128), head_dim=D, index_head_dim=16)
    return DeepseekV4CompressStatePool(3, spec, _mesh(), dp_size=1)


def test_native_state_layout_matches_legacy_bitwise(monkeypatch):
    native = _pool(monkeypatch, True).get_buffer("c128", 1)
    legacy = _pool(monkeypatch, False).get_buffer("c128", 1)
    assert native.shape == (4, 128, 2, D) and legacy.shape == (4, 128, 2 * D)
    np.testing.assert_array_equal(np.asarray(native), np.asarray(legacy).reshape(native.shape))
    assert np.all(np.asarray(native)[..., 0, :] == 0)
    assert np.all(np.isneginf(np.asarray(native)[..., 1, :]))


def test_native_state_reset_restores_empty(monkeypatch):
    pool = _pool(monkeypatch, True)
    empty = np.asarray(pool.get_buffer("c128", 1)).copy()
    slots = jnp.asarray([1, 2], jnp.int32)
    junk = jnp.full((2, 128, 2, D), 7.0, jnp.float32)
    with jax.set_mesh(pool.mesh):
        pool.write("c128", 1, slots, junk, jnp.asarray([True, True]))
        assert np.all(np.asarray(pool.get_buffer("c128", 1))[1:3] == 7.0)
        pool.reset(slots, jnp.asarray([True, False]))
    after = np.asarray(pool.get_buffer("c128", 1))
    np.testing.assert_array_equal(after[1], empty[1])
    assert np.all(after[2] == 7.0)  # invalid mask entry untouched
    np.testing.assert_array_equal(after[0], empty[0])


def test_score_slice_selects_the_score_half():
    three = np.zeros((2, 8, 4 * D), np.float32)
    three[score_slice(three.shape)] = 1
    assert three[..., : 2 * D].sum() == 0 and three[..., 2 * D :].all()
    four = np.zeros((2, 128, 2, D), np.float32)
    four[score_slice(four.shape)] = 1
    assert four[..., 0, :].sum() == 0 and four[..., 1, :].all()


def test_cache_layout_accepts_flat_rows_with_explicit_page_size():
    flat = jnp.zeros((4 * 128, D), jnp.bfloat16)
    rows, page = _cache_layout(flat, D, 128)
    assert rows is flat and page == 128
    paged = flat.reshape(4, 64, 2, D)
    rows4, page4 = _cache_layout(paged, D)
    assert rows4.shape == flat.shape and page4 == 128
    assert _cache_layout(paged, D, 128)[1] == 128
    with pytest.raises(ValueError):
        _cache_layout(flat, D)  # flat rows need the page size
    with pytest.raises(ValueError):
        _cache_layout(paged, D, 256)  # shape and argument disagree
    with pytest.raises(ValueError):
        _cache_layout(jnp.zeros((4 * 128 + 1, D), jnp.bfloat16), D, 128)


def test_native_kv_c128_family_is_paged_4d(monkeypatch):
    spec = DeepseekV4CacheSpec(compress_ratios=(4, 128), head_dim=D, index_head_dim=16)
    monkeypatch.setenv("DSV4_HCA_NATIVE_LAYOUT", "1")
    pool = DeepseekV4TokenToKVPool(4 * 128, 2 * 128, 128, spec, _mesh())
    assert pool.get_buffer("c128", 1).shape == (5, 1, 1, D)
    assert pool.get_buffer("swa", 1).ndim == 2 and pool.get_buffer("c4", 0).ndim == 3
    monkeypatch.setenv("DSV4_HCA_NATIVE_LAYOUT", "0")
    legacy = DeepseekV4TokenToKVPool(4 * 128, 2 * 128, 128, spec, _mesh())
    assert legacy.get_buffer("c128", 1).shape == (5, 1, D)
