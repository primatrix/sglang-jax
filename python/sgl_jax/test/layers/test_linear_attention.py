"""Tests for BailingMoeV2_5LinearAttention __init__ and build_slope_tensor.

Run with: pytest python/sgl_jax/test/layers/test_linear_attention.py -v
"""

import math
from types import SimpleNamespace

import jax
import numpy as np

from sgl_jax.srt.layers.attention.fla.linear_attention_backend import (
    LinearAttentionBackend,
)
from sgl_jax.srt.models.bailing_moe_v2_5_linear_attention import (
    BailingMoeV2_5LinearAttention,
)
from sgl_jax.srt.utils.mesh_utils import create_device_mesh

mesh = create_device_mesh(ici_parallelism=[1, -1], dcn_parallelism=[1, 1])


def _make_config(
    hidden_size=8192,
    num_attention_heads=64,
    head_dim=128,
    partial_rotary_factor=0.5,
    use_qk_norm=True,
    group_norm_size=8,
    rms_norm_eps=1e-6,
    use_qkv_bias=False,
    use_bias=False,
    rope_theta=6_000_000,
    max_position_embeddings=131072,
    num_hidden_layers=80,
):
    return SimpleNamespace(**locals())


def _hf_build_slope_tensor(num_heads):
    """Ground-truth ALiBi slope computation from HuggingFace."""

    def get_slopes_power_of_2(n):
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * ratio**i for i in range(n)]

    if math.log2(num_heads).is_integer():
        return get_slopes_power_of_2(num_heads)
    else:
        closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
        return (
            get_slopes_power_of_2(closest_power_of_2)
            + _hf_build_slope_tensor(2 * closest_power_of_2)[0::2][: num_heads - closest_power_of_2]
        )


def _expected_slope(num_heads, layer_idx, num_hidden_layers):
    """Expected slope tensor matching the _compute_slope formula."""
    base = np.array(_hf_build_slope_tensor(num_heads), dtype=np.float32)
    return -base * (1 - (layer_idx - 1) / (num_hidden_layers - 1) + 1e-5)


def _make_module(layer_idx=1, config=None):
    """Construct a BailingMoeV2_5LinearAttention on CPU under the global mesh."""
    if config is None:
        config = _make_config()
    backend = LinearAttentionBackend(mesh=mesh)
    with jax.default_device(jax.devices("cpu")[0]), jax.set_mesh(mesh):
        module = BailingMoeV2_5LinearAttention(
            config=config,
            layer_idx=layer_idx,
            mesh=mesh,
            backend=backend,
        )
    return module


# ---------------------------------------------------------------------------
# build_slope_tensor (static method) tests
# ---------------------------------------------------------------------------


class TestBuildSlopeTensor:
    def test_power_of_2_heads(self):
        """For power-of-2 head count, values match HF ground truth."""
        slopes = BailingMoeV2_5LinearAttention.build_slope_tensor(8)
        expected = _hf_build_slope_tensor(8)
        np.testing.assert_allclose(slopes, expected, rtol=1e-6)

    def test_non_power_of_2_heads(self):
        """For non-power-of-2 head count, values match HF ground truth."""
        slopes = BailingMoeV2_5LinearAttention.build_slope_tensor(48)
        expected = _hf_build_slope_tensor(48)
        np.testing.assert_allclose(slopes, expected, rtol=1e-6)

    def test_length_matches_num_heads(self):
        """Output list length equals num_heads."""
        for n in [8, 16, 32, 48, 64]:
            assert len(BailingMoeV2_5LinearAttention.build_slope_tensor(n)) == n

    def test_all_positive(self):
        """Raw slopes from build_slope_tensor are all positive."""
        slopes = BailingMoeV2_5LinearAttention.build_slope_tensor(64)
        assert all(s > 0 for s in slopes)


# ---------------------------------------------------------------------------
# ALiBi slope tensor tests (via module._compute_slope / module.slope)
# ---------------------------------------------------------------------------


class TestSlopes:
    def test_slopes_all_negative(self):
        """All slope values must be negative after applying the layer scaling."""
        module = _make_module(layer_idx=10)
        slope_np = np.asarray(module.slope)
        assert np.all(slope_np < 0), "Expected all slopes to be negative"

    def test_slopes_decrease_with_layer_idx(self):
        """Layer 5 should have larger magnitude slopes than layer 50."""
        config = _make_config()
        module_early = _make_module(layer_idx=5, config=config)
        module_late = _make_module(layer_idx=50, config=config)
        early_mag = np.abs(np.asarray(module_early.slope))
        late_mag = np.abs(np.asarray(module_late.slope))
        assert np.all(
            early_mag > late_mag
        ), "Expected layer 5 slope magnitude > layer 50 slope magnitude"

    def test_slopes_match_hf_formula(self):
        """Slope tensor must exactly match the reference formula for several layers."""
        config = _make_config()
        for layer_idx in [1, 10, 40, 79]:
            module = _make_module(layer_idx=layer_idx, config=config)
            actual = np.asarray(module.slope)
            expected = _expected_slope(
                config.num_attention_heads, layer_idx, config.num_hidden_layers
            )
            np.testing.assert_allclose(
                actual,
                expected,
                rtol=1e-5,
                atol=1e-7,
                err_msg=f"Slope mismatch at layer_idx={layer_idx}",
            )

    def test_slopes_shape(self):
        """Slope tensor shape must be (num_attention_heads,)."""
        config = _make_config()
        module = _make_module(layer_idx=1, config=config)
        assert module.slope.shape == (config.num_attention_heads,)


# ---------------------------------------------------------------------------
# Module structure test
# ---------------------------------------------------------------------------


class TestModuleStructure:
    def test_module_has_expected_submodules(self):
        """All expected submodules and attributes must be present."""
        module = _make_module(layer_idx=1)
        for attr in [
            "qkv_proj",
            "g_proj",
            "dense",
            "q_norm",
            "k_norm",
            "rotary_emb",
            "g_norm",
            "slope",
            "backend",
        ]:
            assert hasattr(module, attr), f"Missing attribute: {attr}"

    def test_call_raises_not_implemented(self):
        """__call__ must raise NotImplementedError."""
        module = _make_module(layer_idx=1)
        try:
            module(None, None, None, None)
        except NotImplementedError:
            pass
        else:
            raise AssertionError("Expected NotImplementedError from __call__")

    def test_stored_attributes(self):
        """Scalar attributes must be stored with correct values."""
        config = _make_config()
        module = _make_module(layer_idx=5, config=config)
        assert module.layer_idx == 5
        assert module.hidden_size == config.hidden_size
        assert module.num_heads == config.num_attention_heads
        assert module.head_dim == config.head_dim
        assert module.num_hidden_layers == config.num_hidden_layers

    def test_no_q_norm_when_disabled(self):
        """When use_qk_norm=False, q_norm and k_norm must be None."""
        config = _make_config(use_qk_norm=False)
        module = _make_module(layer_idx=1, config=config)
        assert module.q_norm is None
        assert module.k_norm is None
