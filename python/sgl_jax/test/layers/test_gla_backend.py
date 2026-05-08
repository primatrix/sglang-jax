"""Tests for GLA attention backends (GLAMetadataBackend metadata + LightningAttnBackend).

Run with: pytest python/sgl_jax/test/layers/test_gla_backend.py -v
"""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sgl_jax.srt.layers.attention.hybrid_linear_attn_backend import (
    HybridLinearAttnBackend,
    MockRecurrentStatePool,
)
from sgl_jax.srt.layers.attention.linear.lightning_backend import LightningAttnBackend
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode
from sgl_jax.srt.utils.mesh_utils import create_device_mesh

mesh = create_device_mesh(ici_parallelism=[1, -1], dcn_parallelism=[1, 1])

try:
    from sgl_jax.srt.kernels.simple_gla.simple_gla import fused_recurrent_simple_gla

    HAS_SIMPLE_GLA = True
except ImportError:
    HAS_SIMPLE_GLA = False

requires_simple_gla = pytest.mark.skipif(
    not HAS_SIMPLE_GLA, reason="simple_gla kernel not available"
)

_HAS_TPU = any(d.platform == "tpu" for d in jax.devices())
requires_tpu = pytest.mark.skipif(not _HAS_TPU, reason="chunk kernel requires TPU")

_H = 4
_K = 128

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_batch(forward_mode, extend_seq_lens=None, input_ids=None, recurrent_indices=None):
    batch = SimpleNamespace(forward_mode=forward_mode)
    if extend_seq_lens is not None:
        batch.extend_seq_lens = np.asarray(extend_seq_lens, dtype=np.int32)
    if input_ids is not None:
        batch.input_ids = np.asarray(input_ids, dtype=np.int32)
    if recurrent_indices is not None:
        batch.recurrent_indices = np.asarray(recurrent_indices, dtype=np.int32)
    if forward_mode == ForwardMode.DECODE:
        n_seqs = len(recurrent_indices) if recurrent_indices is not None else 1
        batch.seq_lens = np.ones(n_seqs, dtype=np.int32)
    return batch


def _make_mock_pool(layer_id, recurrent_state, recurrent_indices=None):
    B = recurrent_state.shape[0]
    if recurrent_indices is None:
        recurrent_indices = np.arange(1, B + 1, dtype=np.int32)
    N_plus_1 = int(max(recurrent_indices)) + 1
    buf = jnp.zeros((N_plus_1,) + recurrent_state.shape[1:], dtype=recurrent_state.dtype)
    buf = buf.at[jnp.array(recurrent_indices)].set(recurrent_state)
    return MockRecurrentStatePool(layer_caches={layer_id: (buf, [])}), recurrent_indices


def _make_fake_layer(layer_id=5):
    """Minimal layer stand-in for backend tests.

    The backend reads ``layer.layer_id``, ``layer.num_heads``, ``layer.head_dim``,
    ``layer.mesh``. Slope is no longer carried by the layer — it lives on
    ``backend.tp_slope[layer_id]`` (per upstream LightningAttention pattern).
    """
    return SimpleNamespace(
        layer_id=layer_id,
        mesh=mesh,
        num_heads=_H,
        head_dim=_K,
    )


def _extract_state(pool_updates, recurrent_indices):
    new_ssm_full, conv_list = pool_updates
    assert conv_list == [] or conv_list is None
    return new_ssm_full[jnp.array(recurrent_indices)]


# ---------------------------------------------------------------------------
# LightningAttnBackend tests
# ---------------------------------------------------------------------------


class TestForwardDecode:
    @requires_simple_gla
    def test_decode_matches_direct_kernel(self):
        """Backend decode should match direct fused_recurrent_simple_gla call."""
        layer_id = 5
        # Backend must be constructed with linear_recurrent_layer_ids so
        # tp_slope[layer_id] is populated; the test compares against a
        # direct kernel call using the same slope value.
        backend = LightningAttnBackend(
            mesh=mesh,
            linear_recurrent_layer_ids=[layer_id],
            num_hidden_layers=80,
            num_heads=_H,
        )

        with jax.set_mesh(mesh):
            state_init = jnp.zeros((1, _H, _K, _K), dtype=jnp.float32)
            pool, rec_indices = _make_mock_pool(layer_id, state_init)

            batch = _make_batch(
                ForwardMode.DECODE,
                recurrent_indices=rec_indices,
            )
            metadata = backend.get_forward_metadata(batch)
            backend.forward_metadata = metadata

            q = jax.random.normal(jax.random.PRNGKey(0), (1, _H, _K), dtype=jnp.bfloat16)
            k = jax.random.normal(jax.random.PRNGKey(1), (1, _H, _K), dtype=jnp.bfloat16)
            v = jax.random.normal(jax.random.PRNGKey(2), (1, _H, _K), dtype=jnp.bfloat16)
            slope = backend.tp_slope[layer_id]
            layer = _make_fake_layer(layer_id=layer_id)
            fb = SimpleNamespace(forward_mode=ForwardMode.DECODE)

            output, pool_updates = backend(
                q, k, v, layer=layer, forward_batch=fb, recurrent_state_pool=pool
            )
            backend_state = _extract_state(pool_updates, rec_indices)

            direct_out, direct_state = fused_recurrent_simple_gla(
                q[:, None, :, :],
                k[:, None, :, :],
                v[:, None, :, :],
                g_gamma=slope,
                initial_state=state_init.astype(jnp.float32),
                output_final_state=True,
                scale=None,
            )
            direct_out = direct_out[:, 0, :, :].reshape(1, -1)

        np.testing.assert_allclose(
            np.array(output),
            np.array(direct_out),
            atol=1e-5,
            err_msg="Backend decode output != direct kernel",
        )
        np.testing.assert_allclose(
            np.array(backend_state),
            np.array(direct_state),
            atol=1e-5,
            err_msg="Backend decode state != direct kernel state",
        )


class TestHybridIntegration:
    def test_dispatch_routes_to_lightning(self):
        from sgl_jax.srt.layers.attention.flashattention_backend import FlashAttention

        with jax.set_mesh(mesh):
            full_backend = FlashAttention(
                num_attn_heads=_H,
                num_kv_heads=_H,
                head_dim=_K,
                page_size=1,
                mesh=mesh,
            )
            lightning = LightningAttnBackend(mesh=mesh)
            hybrid = HybridLinearAttnBackend(full_backend, lightning, full_attn_layers=[0, 1])

        assert 5 not in hybrid.full_attn_layers
        assert isinstance(hybrid.linear_attn_backend, LightningAttnBackend)

    def test_attn_backend_wrapper_lightning_path(self):
        """attn_backend_wrapper factory builds LightningAttnBackend with the
        right tp_slope for a Lightning hybrid config.

        Covers the production wire-up that the model-skeleton PR will turn
        on. The factory must read linear_layer_ids / num_hidden_layers /
        num_attention_heads from runner.lightning_config and forward them
        to LightningAttnBackend so tp_slope[layer_id] is populated for
        every Lightning layer.
        """
        from sgl_jax.srt.layers.attention.flashattention_backend import FlashAttention
        from sgl_jax.srt.layers.attention.hybrid_linear_attn_backend import (
            attn_backend_wrapper,
        )

        # Mock the minimal runner / config shape that attn_backend_wrapper reads.
        full_attn_layer_ids = [0, 8, 16]
        linear_layer_ids = [i for i in range(20) if i not in full_attn_layer_ids]
        cfg_lightning = SimpleNamespace(
            linear_layer_ids=linear_layer_ids,
            num_hidden_layers=20,
            num_attention_heads=_H,
            full_attention_layer_ids=full_attn_layer_ids,
        )
        runner = SimpleNamespace(
            mesh=mesh,
            kimi_linear_config=None,
            linear_recurrent_config=cfg_lightning,
            lightning_config=cfg_lightning,
        )

        with jax.set_mesh(mesh):
            full_backend = FlashAttention(
                num_attn_heads=_H,
                num_kv_heads=_H,
                head_dim=_K,
                page_size=1,
                mesh=mesh,
            )
            hybrid = attn_backend_wrapper(runner, full_backend)

        assert isinstance(hybrid, HybridLinearAttnBackend)
        assert isinstance(hybrid.linear_attn_backend, LightningAttnBackend)
        assert hybrid.full_attn_layers == frozenset(full_attn_layer_ids)
        # tp_slope must cover every Lightning layer id (and only those).
        assert sorted(hybrid.linear_attn_backend.tp_slope.keys()) == sorted(linear_layer_ids)
        # Slope vector shape matches num_heads.
        assert hybrid.linear_attn_backend.tp_slope[linear_layer_ids[0]].shape == (_H,)


class TestRadixLightningAttention:
    """Direct unit tests for the dispatcher class."""

    def test_construct_carries_layer_metadata(self):
        from sgl_jax.srt.layers.radix_lightning_attention import RadixLightningAttention

        wrapper = RadixLightningAttention(layer_id=7, num_heads=4, head_dim=128)
        assert wrapper.layer_id == 7
        assert wrapper.num_heads == 4
        assert wrapper.head_dim == 128
        # Slope is owned by the backend, not the wrapper.
        assert not hasattr(wrapper, "slope")

    def test_call_forwards_to_attn_backend_with_pool_kw(self):
        """Dispatcher must call forward_batch.attn_backend with `pool=` keyword
        so HybridLinearAttnBackend can route via the pool/None fan-out.
        """
        from sgl_jax.srt.layers.radix_lightning_attention import RadixLightningAttention

        wrapper = RadixLightningAttention(layer_id=3, num_heads=4, head_dim=128)
        captured = {}

        def fake_attn_backend(*, q, k, v, layer, forward_batch, pool):
            captured["q"] = q
            captured["k"] = k
            captured["v"] = v
            captured["layer"] = layer
            captured["pool"] = pool
            captured["forward_batch"] = forward_batch
            # Mimic backend return: (output, new_pool)
            return (jnp.array([1.0]), pool)

        fb = SimpleNamespace(forward_mode=ForwardMode.DECODE, attn_backend=fake_attn_backend)
        q = jnp.array([10.0])
        k = jnp.array([20.0])
        v = jnp.array([30.0])
        sentinel_pool = object()

        out, returned_pool = wrapper(fb, q, k, v, sentinel_pool)

        assert captured["layer"] is wrapper
        assert captured["pool"] is sentinel_pool
        assert captured["forward_batch"] is fb
        assert captured["q"] is q and captured["k"] is k and captured["v"] is v
        assert returned_pool is sentinel_pool


class TestTpSlopeFailFast:
    """tp_slope[layer_id] missing must raise a contextual KeyError."""

    def test_keyerror_message_lists_registered_ids(self):
        # Backend with only layer_ids 1, 2 registered.
        backend = LightningAttnBackend(
            mesh=mesh,
            linear_recurrent_layer_ids=[1, 2],
            num_hidden_layers=80,
            num_heads=_H,
        )
        # Stub out forward_metadata so __call__ gets past the early lookup
        # and reaches the tp_slope access. We only need to exercise the
        # KeyError path; no actual kernel call should occur.
        backend.forward_metadata = SimpleNamespace(recurrent_indices=jnp.array([1]))

        layer = _make_fake_layer(layer_id=99)
        pool = SimpleNamespace()
        # We need pool.get_linear_recurrent_layer_cache to return something
        # so get_state doesn't crash before the KeyError.
        dummy_buf = jnp.zeros((2, _H, _K, _K), dtype=jnp.float32)
        pool.get_linear_recurrent_layer_cache = lambda lid: (dummy_buf, [])

        fb = SimpleNamespace(forward_mode=ForwardMode.DECODE)
        with pytest.raises(KeyError) as exc_info:
            backend(
                jnp.zeros((1, _H, _K)),
                jnp.zeros((1, _H, _K)),
                jnp.zeros((1, _H, _K)),
                layer=layer,
                forward_batch=fb,
                recurrent_state_pool=pool,
            )

        msg = str(exc_info.value)
        assert "layer_id=99" in msg
        assert "registered ids: [1, 2]" in msg
        assert "attn_backend_wrapper" in msg
