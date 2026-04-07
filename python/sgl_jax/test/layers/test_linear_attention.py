"""Tests for BailingMoeV2_5LinearAttention __init__, build_slope_tensor, and forward pass.

Run with: pytest python/sgl_jax/test/layers/test_linear_attention.py -v
"""

import math
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sgl_jax.srt.layers.attention.fla.linear_attention_backend import (
    LinearAttentionBackend,
)
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode
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


# ---------------------------------------------------------------------------
# Forward pass tests (require tops library)
# ---------------------------------------------------------------------------

try:
    from tops.ops.simple_gla import simple_gla_fwd  # noqa: F401
    from tops.ops.simple_gla.fused_recurrent import (  # noqa: F401
        fused_recurrent_simple_gla,
    )

    HAS_TOPS = True
except ImportError:
    HAS_TOPS = False

requires_tops = pytest.mark.skipif(not HAS_TOPS, reason="tops library not available")


def _make_forward_batch(forward_mode):
    return SimpleNamespace(forward_mode=forward_mode)


class TestDecodeForward:
    @requires_tops
    def test_decode_output_shape(self):
        """Decode forward should return output [T, hidden_size] and new_state [T, H, K, V]."""
        config = _make_config(
            hidden_size=256,
            num_attention_heads=4,
            head_dim=64,
            num_hidden_layers=10,
            partial_rotary_factor=0.5,
            max_position_embeddings=1024,
            rope_theta=10000,
            group_norm_size=4,
        )
        backend = LinearAttentionBackend(mesh=mesh)
        T = 2
        H, K = 4, 64

        with jax.default_device(jax.devices("cpu")[0]), jax.set_mesh(mesh):
            module = BailingMoeV2_5LinearAttention(
                config=config, layer_idx=5, mesh=mesh, backend=backend
            )
            hidden = jax.random.normal(jax.random.PRNGKey(0), (T, 256), dtype=jnp.bfloat16)
            positions = jnp.arange(T, dtype=jnp.int32)
            recurrent_state = jnp.zeros((T, H, K, K), dtype=jnp.bfloat16)
            fb = _make_forward_batch(ForwardMode.DECODE)

            output, new_state = module(positions, hidden, fb, recurrent_state)

        assert output.shape == (T, 256), f"Expected ({T}, 256), got {output.shape}"
        assert new_state.shape == (
            T,
            H,
            K,
            K,
        ), f"Expected ({T}, {H}, {K}, {K}), got {new_state.shape}"

    @requires_tops
    def test_decode_state_updates(self):
        """Two decode steps should produce different states."""
        config = _make_config(
            hidden_size=256,
            num_attention_heads=4,
            head_dim=64,
            num_hidden_layers=10,
            partial_rotary_factor=0.5,
            max_position_embeddings=1024,
            rope_theta=10000,
            group_norm_size=4,
        )
        backend = LinearAttentionBackend(mesh=mesh)
        T = 1
        H, K = 4, 64

        with jax.default_device(jax.devices("cpu")[0]), jax.set_mesh(mesh):
            module = BailingMoeV2_5LinearAttention(
                config=config, layer_idx=5, mesh=mesh, backend=backend
            )
            hidden = jax.random.normal(jax.random.PRNGKey(42), (T, 256), dtype=jnp.bfloat16)
            state0 = jnp.zeros((T, H, K, K), dtype=jnp.bfloat16)
            fb = _make_forward_batch(ForwardMode.DECODE)

            _, state1 = module(jnp.array([0], dtype=jnp.int32), hidden, fb, state0)
            _, state2 = module(jnp.array([1], dtype=jnp.int32), hidden, fb, state1)

        assert not jnp.allclose(state1, state2), "States should differ after two steps"

    @requires_tops
    def test_decode_state_affects_output(self):
        """Different initial states should produce different outputs."""
        config = _make_config(
            hidden_size=256,
            num_attention_heads=4,
            head_dim=64,
            num_hidden_layers=10,
            partial_rotary_factor=0.5,
            max_position_embeddings=1024,
            rope_theta=10000,
            group_norm_size=4,
        )
        backend = LinearAttentionBackend(mesh=mesh)
        T = 1
        H, K = 4, 64

        with jax.default_device(jax.devices("cpu")[0]), jax.set_mesh(mesh):
            module = BailingMoeV2_5LinearAttention(
                config=config, layer_idx=5, mesh=mesh, backend=backend
            )
            hidden = jax.random.normal(jax.random.PRNGKey(42), (T, 256), dtype=jnp.bfloat16)
            fb = _make_forward_batch(ForwardMode.DECODE)

            state_zeros = jnp.zeros((T, H, K, K), dtype=jnp.bfloat16)
            _, state_nonzero = module(jnp.array([0], dtype=jnp.int32), hidden, fb, state_zeros)

            out_from_zeros, _ = module(jnp.array([1], dtype=jnp.int32), hidden, fb, state_zeros)
            out_from_nonzero, _ = module(jnp.array([1], dtype=jnp.int32), hidden, fb, state_nonzero)

        assert not jnp.allclose(
            out_from_zeros, out_from_nonzero
        ), "Different states should give different outputs"


class TestPrefillForward:
    @requires_tops
    def test_prefill_output_shape(self):
        """Prefill forward should return output [T, hidden_size] and new_state [N_padded, H, K, V]."""
        config = _make_config(
            hidden_size=256,
            num_attention_heads=4,
            head_dim=64,
            num_hidden_layers=10,
            partial_rotary_factor=0.5,
            max_position_embeddings=1024,
            rope_theta=10000,
            group_norm_size=4,
        )
        backend = LinearAttentionBackend(mesh=mesh)
        seq_len = 128
        N_padded = 1
        H, K = 4, 64

        with jax.default_device(jax.devices("cpu")[0]), jax.set_mesh(mesh):
            module = BailingMoeV2_5LinearAttention(
                config=config, layer_idx=5, mesh=mesh, backend=backend
            )

            batch_meta = SimpleNamespace(
                forward_mode=ForwardMode.EXTEND,
                extend_seq_lens=np.array([seq_len], dtype=np.int32),
                seq_lens=np.array([seq_len], dtype=np.int32),
                input_ids=np.zeros(seq_len, dtype=np.int32),
            )
            backend.get_forward_metadata(batch_meta)

            hidden = jax.random.normal(jax.random.PRNGKey(0), (seq_len, 256), dtype=jnp.bfloat16)
            positions = jnp.arange(seq_len, dtype=jnp.int32)
            recurrent_state = jnp.zeros((N_padded, H, K, K), dtype=jnp.bfloat16)
            fb = _make_forward_batch(ForwardMode.EXTEND)

            output, new_state = module(positions, hidden, fb, recurrent_state)

        assert output.shape == (seq_len, 256), f"Expected ({seq_len}, 256), got {output.shape}"
        assert new_state.shape == (N_padded, H, K, K)

    @requires_tops
    def test_prefill_non_chunk_aligned(self):
        """Prefill with non-chunk-aligned seq_len (e.g. 100) should work via scatter/gather."""
        config = _make_config(
            hidden_size=256,
            num_attention_heads=4,
            head_dim=64,
            num_hidden_layers=10,
            partial_rotary_factor=0.5,
            max_position_embeddings=1024,
            rope_theta=10000,
            group_norm_size=4,
        )
        backend = LinearAttentionBackend(mesh=mesh)
        seq_len = 100
        N_padded = 1
        H, K = 4, 64

        with jax.default_device(jax.devices("cpu")[0]), jax.set_mesh(mesh):
            module = BailingMoeV2_5LinearAttention(
                config=config, layer_idx=5, mesh=mesh, backend=backend
            )

            batch_meta = SimpleNamespace(
                forward_mode=ForwardMode.EXTEND,
                extend_seq_lens=np.array([seq_len], dtype=np.int32),
                seq_lens=np.array([seq_len], dtype=np.int32),
                input_ids=np.zeros(seq_len, dtype=np.int32),
            )
            backend.get_forward_metadata(batch_meta)

            hidden = jax.random.normal(jax.random.PRNGKey(0), (seq_len, 256), dtype=jnp.bfloat16)
            positions = jnp.arange(seq_len, dtype=jnp.int32)
            recurrent_state = jnp.zeros((N_padded, H, K, K), dtype=jnp.bfloat16)
            fb = _make_forward_batch(ForwardMode.EXTEND)

            output, new_state = module(positions, hidden, fb, recurrent_state)

        assert output.shape == (seq_len, 256)
        assert new_state.shape == (N_padded, H, K, K)

    @requires_tops
    def test_prefill_zeros_state_runs(self):
        """Prefill with all-zeros initial state should complete without error."""
        config = _make_config(
            hidden_size=256,
            num_attention_heads=4,
            head_dim=64,
            num_hidden_layers=10,
            partial_rotary_factor=0.5,
            max_position_embeddings=1024,
            rope_theta=10000,
            group_norm_size=4,
        )
        backend = LinearAttentionBackend(mesh=mesh)
        seq_len = 64
        H, K = 4, 64

        with jax.default_device(jax.devices("cpu")[0]), jax.set_mesh(mesh):
            module = BailingMoeV2_5LinearAttention(
                config=config, layer_idx=5, mesh=mesh, backend=backend
            )

            batch_meta = SimpleNamespace(
                forward_mode=ForwardMode.EXTEND,
                extend_seq_lens=np.array([seq_len], dtype=np.int32),
                seq_lens=np.array([seq_len], dtype=np.int32),
                input_ids=np.zeros(seq_len, dtype=np.int32),
            )
            backend.get_forward_metadata(batch_meta)

            hidden = jax.random.normal(jax.random.PRNGKey(0), (seq_len, 256), dtype=jnp.bfloat16)
            positions = jnp.arange(seq_len, dtype=jnp.int32)
            recurrent_state = jnp.zeros((1, H, K, K), dtype=jnp.bfloat16)
            fb = _make_forward_batch(ForwardMode.EXTEND)

            output, new_state = module(positions, hidden, fb, recurrent_state)

        assert output.shape == (seq_len, 256)
        assert new_state.shape == (1, H, K, K)


# ---------------------------------------------------------------------------
# White-box tests (require tops library)
# ---------------------------------------------------------------------------


class TestWhiteBox:
    def test_qkv_projection_shape(self):
        """QKV projection should produce [T, 3*H*head_dim]."""
        config = _make_config(
            hidden_size=256,
            num_attention_heads=4,
            head_dim=64,
            num_hidden_layers=10,
            partial_rotary_factor=0.5,
            max_position_embeddings=1024,
            rope_theta=10000,
            group_norm_size=4,
        )
        backend = LinearAttentionBackend(mesh=mesh)

        with jax.default_device(jax.devices("cpu")[0]), jax.set_mesh(mesh):
            module = BailingMoeV2_5LinearAttention(
                config=config, layer_idx=5, mesh=mesh, backend=backend
            )
            hidden = jax.random.normal(jax.random.PRNGKey(0), (4, 256), dtype=jnp.bfloat16)
            qkv, _ = module.qkv_proj(hidden)

        assert qkv.shape == (4, 3 * 4 * 64), f"Expected (4, 768), got {qkv.shape}"

    def test_gating_values_in_range(self):
        """Sigmoid gating values should be in [0, 1]."""
        config = _make_config(
            hidden_size=256,
            num_attention_heads=4,
            head_dim=64,
            num_hidden_layers=10,
            partial_rotary_factor=0.5,
            max_position_embeddings=1024,
            rope_theta=10000,
            group_norm_size=4,
        )
        backend = LinearAttentionBackend(mesh=mesh)

        with jax.default_device(jax.devices("cpu")[0]), jax.set_mesh(mesh):
            module = BailingMoeV2_5LinearAttention(
                config=config, layer_idx=5, mesh=mesh, backend=backend
            )
            hidden = jax.random.normal(jax.random.PRNGKey(0), (4, 256), dtype=jnp.bfloat16)
            g, _ = module.g_proj(hidden)
            gate = jax.nn.sigmoid(g)

        assert jnp.all(gate >= 0) and jnp.all(gate <= 1), "Gate values out of [0,1]"
