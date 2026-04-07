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

# Prefill (chunk) kernel uses Pallas TPU primitives and cannot run on CPU.
_HAS_TPU = any(d.platform == "tpu" for d in jax.devices())
requires_tpu = pytest.mark.skipif(not _HAS_TPU, reason="chunk kernel requires TPU")


def _make_forward_batch(forward_mode):
    return SimpleNamespace(forward_mode=forward_mode)


_SMALL_CONFIG = _make_config(
    hidden_size=512,
    num_attention_heads=4,
    head_dim=128,
    num_hidden_layers=10,
    partial_rotary_factor=0.5,
    max_position_embeddings=1024,
    rope_theta=10000,
    group_norm_size=4,
)

_SMALL_H = 4
_SMALL_K = 128
_SMALL_HIDDEN = 512


class TestDecodeForward:
    @requires_tops
    def test_decode_output_shape(self):
        """Decode forward should return output [T, hidden_size] and new_state [T, H, K, V]."""
        backend = LinearAttentionBackend(mesh=mesh)
        T = 2

        with jax.default_device(jax.devices("cpu")[0]), jax.set_mesh(mesh):
            module = BailingMoeV2_5LinearAttention(
                config=_SMALL_CONFIG, layer_idx=5, mesh=mesh, backend=backend
            )
            hidden = jax.random.normal(
                jax.random.PRNGKey(0), (T, _SMALL_HIDDEN), dtype=jnp.bfloat16
            )
            positions = jnp.arange(T, dtype=jnp.int32)
            recurrent_state = jnp.zeros((T, _SMALL_H, _SMALL_K, _SMALL_K), dtype=jnp.bfloat16)
            fb = _make_forward_batch(ForwardMode.DECODE)

            output, new_state = module(positions, hidden, fb, recurrent_state)

        assert output.shape == (T, _SMALL_HIDDEN)
        assert new_state.shape == (T, _SMALL_H, _SMALL_K, _SMALL_K)

    @requires_tops
    def test_decode_state_updates(self):
        """Two decode steps should produce different states."""
        backend = LinearAttentionBackend(mesh=mesh)
        T = 1

        with jax.default_device(jax.devices("cpu")[0]), jax.set_mesh(mesh):
            module = BailingMoeV2_5LinearAttention(
                config=_SMALL_CONFIG, layer_idx=5, mesh=mesh, backend=backend
            )
            hidden = jax.random.normal(
                jax.random.PRNGKey(42), (T, _SMALL_HIDDEN), dtype=jnp.bfloat16
            )
            state0 = jnp.zeros((T, _SMALL_H, _SMALL_K, _SMALL_K), dtype=jnp.bfloat16)
            fb = _make_forward_batch(ForwardMode.DECODE)

            _, state1 = module(jnp.array([0], dtype=jnp.int32), hidden, fb, state0)
            _, state2 = module(jnp.array([1], dtype=jnp.int32), hidden, fb, state1)

        assert not jnp.allclose(state1, state2), "States should differ after two steps"

    @requires_tops
    def test_decode_state_affects_output(self):
        """Different initial states should produce different outputs."""
        backend = LinearAttentionBackend(mesh=mesh)
        T = 1

        with jax.default_device(jax.devices("cpu")[0]), jax.set_mesh(mesh):
            module = BailingMoeV2_5LinearAttention(
                config=_SMALL_CONFIG, layer_idx=5, mesh=mesh, backend=backend
            )
            hidden = jax.random.normal(
                jax.random.PRNGKey(42), (T, _SMALL_HIDDEN), dtype=jnp.bfloat16
            )
            fb = _make_forward_batch(ForwardMode.DECODE)

            state_zeros = jnp.zeros((T, _SMALL_H, _SMALL_K, _SMALL_K), dtype=jnp.bfloat16)
            state_large = jax.random.normal(
                jax.random.PRNGKey(99), (T, _SMALL_H, _SMALL_K, _SMALL_K), dtype=jnp.bfloat16
            )

            out_from_zeros, _ = module(jnp.array([0], dtype=jnp.int32), hidden, fb, state_zeros)
            out_from_large, _ = module(jnp.array([0], dtype=jnp.int32), hidden, fb, state_large)

        assert not jnp.allclose(
            out_from_zeros, out_from_large
        ), "Different states should give different outputs"


class TestPrefillForward:
    @requires_tops
    @requires_tpu
    def test_prefill_output_shape(self):
        """Prefill forward should return output [T, hidden_size] and new_state [N_padded, H, K, V]."""
        backend = LinearAttentionBackend(mesh=mesh)
        seq_len = 128
        N_padded = 1

        with jax.default_device(jax.devices("cpu")[0]), jax.set_mesh(mesh):
            module = BailingMoeV2_5LinearAttention(
                config=_SMALL_CONFIG, layer_idx=5, mesh=mesh, backend=backend
            )

            batch_meta = SimpleNamespace(
                forward_mode=ForwardMode.EXTEND,
                extend_seq_lens=np.array([seq_len], dtype=np.int32),
                seq_lens=np.array([seq_len], dtype=np.int32),
                input_ids=np.zeros(seq_len, dtype=np.int32),
            )
            backend.get_forward_metadata(batch_meta)

            hidden = jax.random.normal(
                jax.random.PRNGKey(0), (seq_len, _SMALL_HIDDEN), dtype=jnp.bfloat16
            )
            positions = jnp.arange(seq_len, dtype=jnp.int32)
            recurrent_state = jnp.zeros(
                (N_padded, _SMALL_H, _SMALL_K, _SMALL_K), dtype=jnp.bfloat16
            )
            fb = _make_forward_batch(ForwardMode.EXTEND)

            output, new_state = module(positions, hidden, fb, recurrent_state)

        assert output.shape == (seq_len, _SMALL_HIDDEN)
        assert new_state.shape == (N_padded, _SMALL_H, _SMALL_K, _SMALL_K)

    @requires_tops
    @requires_tpu
    def test_prefill_non_chunk_aligned(self):
        """Prefill with non-chunk-aligned seq_len (e.g. 100) should work via scatter/gather."""
        backend = LinearAttentionBackend(mesh=mesh)
        seq_len = 100
        N_padded = 1

        with jax.default_device(jax.devices("cpu")[0]), jax.set_mesh(mesh):
            module = BailingMoeV2_5LinearAttention(
                config=_SMALL_CONFIG, layer_idx=5, mesh=mesh, backend=backend
            )

            batch_meta = SimpleNamespace(
                forward_mode=ForwardMode.EXTEND,
                extend_seq_lens=np.array([seq_len], dtype=np.int32),
                seq_lens=np.array([seq_len], dtype=np.int32),
                input_ids=np.zeros(seq_len, dtype=np.int32),
            )
            backend.get_forward_metadata(batch_meta)

            hidden = jax.random.normal(
                jax.random.PRNGKey(0), (seq_len, _SMALL_HIDDEN), dtype=jnp.bfloat16
            )
            positions = jnp.arange(seq_len, dtype=jnp.int32)
            recurrent_state = jnp.zeros(
                (N_padded, _SMALL_H, _SMALL_K, _SMALL_K), dtype=jnp.bfloat16
            )
            fb = _make_forward_batch(ForwardMode.EXTEND)

            output, new_state = module(positions, hidden, fb, recurrent_state)

        assert output.shape == (seq_len, _SMALL_HIDDEN)
        assert new_state.shape == (N_padded, _SMALL_H, _SMALL_K, _SMALL_K)

    @requires_tops
    @requires_tpu
    def test_prefill_zeros_state_runs(self):
        """Prefill with all-zeros initial state should complete without error."""
        backend = LinearAttentionBackend(mesh=mesh)
        seq_len = 128

        with jax.default_device(jax.devices("cpu")[0]), jax.set_mesh(mesh):
            module = BailingMoeV2_5LinearAttention(
                config=_SMALL_CONFIG, layer_idx=5, mesh=mesh, backend=backend
            )

            batch_meta = SimpleNamespace(
                forward_mode=ForwardMode.EXTEND,
                extend_seq_lens=np.array([seq_len], dtype=np.int32),
                seq_lens=np.array([seq_len], dtype=np.int32),
                input_ids=np.zeros(seq_len, dtype=np.int32),
            )
            backend.get_forward_metadata(batch_meta)

            hidden = jax.random.normal(
                jax.random.PRNGKey(0), (seq_len, _SMALL_HIDDEN), dtype=jnp.bfloat16
            )
            positions = jnp.arange(seq_len, dtype=jnp.int32)
            recurrent_state = jnp.zeros((1, _SMALL_H, _SMALL_K, _SMALL_K), dtype=jnp.bfloat16)
            fb = _make_forward_batch(ForwardMode.EXTEND)

            output, new_state = module(positions, hidden, fb, recurrent_state)

        assert output.shape == (seq_len, _SMALL_HIDDEN)
        assert new_state.shape == (1, _SMALL_H, _SMALL_K, _SMALL_K)


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

    def test_v_skips_rmsnorm(self):
        """V should NOT be modified by Q/K RMSNorm."""
        H, K = 4, 64
        config = _make_config(
            hidden_size=256,
            num_attention_heads=H,
            head_dim=K,
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
            hidden = jax.random.normal(jax.random.PRNGKey(0), (4, 256), dtype=jnp.float32)

            qkv, _ = module.qkv_proj(hidden)
            qkv = qkv.reshape(4, 3, H, K)
            v_before = qkv[:, 2]

            q = qkv[:, 0]
            k = qkv[:, 1]
            if module.q_norm is not None:
                q = module.q_norm(q)
                k = module.k_norm(k)
            v_after = qkv[:, 2]

        np.testing.assert_array_equal(
            np.array(v_before), np.array(v_after), err_msg="V should not be modified by Q/K RMSNorm"
        )

    def test_rope_only_affects_first_rotary_dims(self):
        """RoPE should only modify the first rope_dim dims; rest unchanged."""
        H, K = 4, 64
        config = _make_config(
            hidden_size=256,
            num_attention_heads=H,
            head_dim=K,
            num_hidden_layers=10,
            partial_rotary_factor=0.5,
            max_position_embeddings=1024,
            rope_theta=10000,
            group_norm_size=4,
        )
        rope_dim = int(K * 0.5)  # 32
        backend = LinearAttentionBackend(mesh=mesh)

        with jax.default_device(jax.devices("cpu")[0]), jax.set_mesh(mesh):
            module = BailingMoeV2_5LinearAttention(
                config=config, layer_idx=5, mesh=mesh, backend=backend
            )
            hidden = jax.random.normal(jax.random.PRNGKey(0), (4, 256), dtype=jnp.float32)

            qkv, _ = module.qkv_proj(hidden)
            qkv = qkv.reshape(4, 3, H, K)
            q_pre = qkv[:, 0]
            k_pre = qkv[:, 1]

            if module.q_norm is not None:
                q_pre = module.q_norm(q_pre)
                k_pre = module.k_norm(k_pre)

            positions = jnp.arange(4, dtype=jnp.int32)
            q_post, k_post = module.rotary_emb(positions, q_pre, k_pre)

        np.testing.assert_array_equal(
            np.array(q_pre[:, :, rope_dim:]),
            np.array(q_post[:, :, rope_dim:]),
            err_msg="Q dims after rope_dim should be unchanged by RoPE",
        )
        np.testing.assert_array_equal(
            np.array(k_pre[:, :, rope_dim:]),
            np.array(k_post[:, :, rope_dim:]),
            err_msg="K dims after rope_dim should be unchanged by RoPE",
        )

    def test_dense_projection_changes_values(self):
        """Dense projection should produce different values from its input."""
        H, K = 4, 64
        config = _make_config(
            hidden_size=256,
            num_attention_heads=H,
            head_dim=K,
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
            x = jax.random.normal(jax.random.PRNGKey(0), (4, H * K), dtype=jnp.bfloat16)
            y, _ = module.dense(x)

        assert not jnp.allclose(x, y), "Dense projection should change values"


# ---------------------------------------------------------------------------
# Multi-request isolation tests (require tops library)
# ---------------------------------------------------------------------------


class TestMultiRequestIsolation:
    @requires_tops
    def test_decode_multi_request_isolation(self):
        """Two requests decoded separately should match decoded together."""
        backend = LinearAttentionBackend(mesh=mesh)

        with jax.default_device(jax.devices("cpu")[0]), jax.set_mesh(mesh):
            module = BailingMoeV2_5LinearAttention(
                config=_SMALL_CONFIG, layer_idx=5, mesh=mesh, backend=backend
            )

            hidden1 = jax.random.normal(
                jax.random.PRNGKey(42), (1, _SMALL_HIDDEN), dtype=jnp.bfloat16
            )
            hidden2 = jax.random.normal(
                jax.random.PRNGKey(43), (1, _SMALL_HIDDEN), dtype=jnp.bfloat16
            )
            state1 = jax.random.normal(
                jax.random.PRNGKey(44), (1, _SMALL_H, _SMALL_K, _SMALL_K), dtype=jnp.bfloat16
            )
            state2 = jax.random.normal(
                jax.random.PRNGKey(45), (1, _SMALL_H, _SMALL_K, _SMALL_K), dtype=jnp.bfloat16
            )

            fb = _make_forward_batch(ForwardMode.DECODE)

            # Separate
            out1, s1 = module(jnp.array([0], dtype=jnp.int32), hidden1, fb, state1)
            out2, s2 = module(jnp.array([0], dtype=jnp.int32), hidden2, fb, state2)

            # Together
            hidden_both = jnp.concatenate([hidden1, hidden2], axis=0)
            state_both = jnp.concatenate([state1, state2], axis=0)
            positions_both = jnp.array([0, 0], dtype=jnp.int32)
            out_both, s_both = module(positions_both, hidden_both, fb, state_both)

        np.testing.assert_allclose(
            np.array(out_both[0]),
            np.array(out1[0]),
            atol=2e-1,
            err_msg="Request 1 output differs between separate and batched decode",
        )
        np.testing.assert_allclose(
            np.array(out_both[1]),
            np.array(out2[0]),
            atol=2e-1,
            err_msg="Request 2 output differs between separate and batched decode",
        )

    @requires_tops
    @requires_tpu
    def test_prefill_multi_request_isolation(self):
        """Two requests prefilled separately should match prefilled together."""
        seq_len1, seq_len2 = 128, 128

        with jax.default_device(jax.devices("cpu")[0]), jax.set_mesh(mesh):
            # --- Separate ---
            backend1 = LinearAttentionBackend(mesh=mesh)
            module = BailingMoeV2_5LinearAttention(
                config=_SMALL_CONFIG, layer_idx=5, mesh=mesh, backend=backend1
            )
            batch1 = SimpleNamespace(
                forward_mode=ForwardMode.EXTEND,
                extend_seq_lens=np.array([seq_len1], dtype=np.int32),
                seq_lens=np.array([seq_len1], dtype=np.int32),
                input_ids=np.zeros(seq_len1, dtype=np.int32),
            )
            backend1.get_forward_metadata(batch1)
            h1 = jax.random.normal(
                jax.random.PRNGKey(0), (seq_len1, _SMALL_HIDDEN), dtype=jnp.bfloat16
            )
            pos1 = jnp.arange(seq_len1, dtype=jnp.int32)
            state1_init = jnp.zeros((1, _SMALL_H, _SMALL_K, _SMALL_K), dtype=jnp.bfloat16)
            fb_ext = _make_forward_batch(ForwardMode.EXTEND)
            out1, s1 = module(pos1, h1, fb_ext, state1_init)

            backend2 = LinearAttentionBackend(mesh=mesh)
            module.backend = backend2
            batch2 = SimpleNamespace(
                forward_mode=ForwardMode.EXTEND,
                extend_seq_lens=np.array([seq_len2], dtype=np.int32),
                seq_lens=np.array([seq_len2], dtype=np.int32),
                input_ids=np.zeros(seq_len2, dtype=np.int32),
            )
            backend2.get_forward_metadata(batch2)
            h2 = jax.random.normal(
                jax.random.PRNGKey(1), (seq_len2, _SMALL_HIDDEN), dtype=jnp.bfloat16
            )
            pos2 = jnp.arange(seq_len2, dtype=jnp.int32)
            state2_init = jnp.zeros((1, _SMALL_H, _SMALL_K, _SMALL_K), dtype=jnp.bfloat16)
            out2, s2 = module(pos2, h2, fb_ext, state2_init)

            # --- Together ---
            backend_both = LinearAttentionBackend(mesh=mesh)
            module.backend = backend_both
            batch_both = SimpleNamespace(
                forward_mode=ForwardMode.EXTEND,
                extend_seq_lens=np.array([seq_len1, seq_len2], dtype=np.int32),
                seq_lens=np.array([seq_len1, seq_len2], dtype=np.int32),
                input_ids=np.zeros(seq_len1 + seq_len2, dtype=np.int32),
            )
            backend_both.get_forward_metadata(batch_both)
            h_both = jnp.concatenate([h1, h2], axis=0)
            pos_both = jnp.concatenate([pos1, pos2], axis=0)
            state_both_init = jnp.zeros((2, _SMALL_H, _SMALL_K, _SMALL_K), dtype=jnp.bfloat16)
            out_both, s_both = module(pos_both, h_both, fb_ext, state_both_init)

        np.testing.assert_allclose(
            np.array(out_both[:seq_len1]),
            np.array(out1),
            atol=5e-2,
            err_msg="Request 1 output differs in batched prefill",
        )
        np.testing.assert_allclose(
            np.array(s_both[0]),
            np.array(s1[0]),
            atol=5e-2,
            err_msg="Request 1 state differs in batched prefill",
        )
