"""DSV4_ROPE_CACHE_LANE_PAD: the [max_position, cos|sin] tables padded to 128 lanes must
agree with the unpadded tables on the rope lanes, and the split halves must be unchanged.
(64-lane tables get a column-major jit-parameter layout on TPU and a 2 x 268 MB relayout
copy per step; padding removes the copy without touching any value.)"""

import types

import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.models import deepseek_v4 as m


def _config():
    return types.SimpleNamespace(
        qk_rope_head_dim=64,
        max_position_embeddings=4096,
        rope_scaling={"factor": 4.0, "original_max_position_embeddings": 1024},
        rope_theta=10000.0,
        compress_rope_theta=10000.0,
        compress_ratios=(4, 128),
    )


def test_padded_cache_matches_on_rope_lanes(monkeypatch):
    cfg = _config()
    monkeypatch.setattr(m, "_ROPE_CACHE_LANE_PAD", False)
    plain = np.asarray(m._rope_cache(cfg, 4))
    monkeypatch.setattr(m, "_ROPE_CACHE_LANE_PAD", True)
    padded = np.asarray(m._rope_cache(cfg, 4))
    assert plain.shape == (4096, 64)
    assert padded.shape == (4096, 128)
    np.testing.assert_array_equal(padded[:, :64], plain)
    assert not np.any(padded[:, 64:])
    cos_a, sin_a = m._split_rope_cache(jnp.asarray(plain), 64)
    cos_b, sin_b = m._split_rope_cache(jnp.asarray(padded), 64)
    np.testing.assert_array_equal(np.asarray(cos_a), np.asarray(cos_b))
    np.testing.assert_array_equal(np.asarray(sin_a), np.asarray(sin_b))
    assert cos_b.shape == (4096, 32) and sin_b.shape == (4096, 32)
