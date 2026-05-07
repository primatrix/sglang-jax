"""Tests for GLA attention backends (LinearAttentionBackend metadata + LightningAttnBackend).

Run with: pytest python/sgl_jax/test/layers/test_gla_backend.py -v
"""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.layers.attention.fla.linear_attention_backend import (
    LinearAttentionBackend,
    LinearAttentionMetadata,
    gather_from_packed,
    scatter_to_packed,
)
from sgl_jax.srt.layers.attention.hybrid_linear_attn_backend import (
    HybridLinearAttnBackend,
    HybridLinearAttnBackendMetadata,
    MockRecurrentStatePool,
)
from sgl_jax.srt.layers.attention.linear.lightning_backend import (
    LightningAttnBackend,
)
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode
from sgl_jax.srt.utils.mesh_utils import create_device_mesh

mesh = create_device_mesh(ici_parallelism=[1, -1], dcn_parallelism=[1, 1])

try:
    from sgl_jax.srt.kernels.simple_gla.simple_gla import (
        fused_recurrent_simple_gla,
        simple_gla_fwd,
    )

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


def _make_metadata_batch(forward_mode, extend_seq_lens, seq_lens, T_outer=None):
    esl = np.array(extend_seq_lens, dtype=np.int32) if extend_seq_lens is not None else None
    if T_outer is None and esl is not None:
        T_outer = int(np.sum(esl))
    return SimpleNamespace(
        forward_mode=forward_mode,
        extend_seq_lens=esl,
        seq_lens=np.array(seq_lens, dtype=np.int32),
        input_ids=np.zeros(T_outer or 0, dtype=np.int32),
    )


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


def _make_fake_layer(layer_id=5, slope=None):
    if slope is None:
        slope = jnp.array([-0.1, -0.2, -0.3, -0.4], dtype=jnp.float32)
    return SimpleNamespace(
        layer_id=layer_id,
        slope=slope,
        mesh=mesh,
        num_heads=_H,
        head_dim=_K,
    )


def _extract_state(pool_updates, recurrent_indices):
    new_ssm_full, conv_list = pool_updates
    assert conv_list == [] or conv_list is None
    return new_ssm_full[jnp.array(recurrent_indices)]


# ---------------------------------------------------------------------------
# cu_seqlens tests
# ---------------------------------------------------------------------------


class TestCuSeqlens:
    def test_single_aligned_request(self):
        backend = LinearAttentionBackend()
        batch = _make_metadata_batch(ForwardMode.EXTEND, [64], [64])
        metadata = backend.get_forward_metadata(batch)
        expected = np.array([0, 64], dtype=np.int32)
        np.testing.assert_array_equal(np.asarray(metadata.cu_seqlens_dev), expected)

    def test_single_unaligned_request(self):
        backend = LinearAttentionBackend()
        batch = _make_metadata_batch(ForwardMode.EXTEND, [100], [100])
        metadata = backend.get_forward_metadata(batch)
        # ceil(100/64)*64 = 128
        expected = np.array([0, 128], dtype=np.int32)
        np.testing.assert_array_equal(np.asarray(metadata.cu_seqlens_dev), expected)

    def test_multiple_requests(self):
        backend = LinearAttentionBackend()
        batch = _make_metadata_batch(ForwardMode.EXTEND, [30, 64, 50], [30, 64, 50])
        metadata = backend.get_forward_metadata(batch)
        # 30->64, 64->64, 50->64; cumsum: [0,64,128,192]
        expected = np.array([0, 64, 128, 192], dtype=np.int32)
        np.testing.assert_array_equal(np.asarray(metadata.cu_seqlens_dev), expected)

    def test_padded_batch_zero_length(self):
        backend = LinearAttentionBackend()
        # Third request has length 0 (padding slot)
        T_outer = 64 + 128  # tight sum of real requests
        batch = _make_metadata_batch(ForwardMode.EXTEND, [64, 128, 0], [64, 128, 0], T_outer=T_outer)
        metadata = backend.get_forward_metadata(batch)
        # 64->64, 128->128, 0->0; cumsum: [0,64,192,192]
        expected = np.array([0, 64, 192, 192], dtype=np.int32)
        np.testing.assert_array_equal(np.asarray(metadata.cu_seqlens_dev), expected)


# ---------------------------------------------------------------------------
# T_packed_bucket tests
# ---------------------------------------------------------------------------


class TestTPackedBucket:
    def test_two_unaligned_requests(self):
        backend = LinearAttentionBackend()
        batch = _make_metadata_batch(ForwardMode.EXTEND, [30, 50], [30, 50])
        backend.get_forward_metadata(batch)
        # 30->64, 50->64; total=128
        assert backend.T_packed_bucket == 128


# ---------------------------------------------------------------------------
# scatter_idx tests
# ---------------------------------------------------------------------------


class TestScatterIdx:
    def test_shape(self):
        backend = LinearAttentionBackend()
        batch = _make_metadata_batch(ForwardMode.EXTEND, [30, 50], [30, 50], T_outer=128)
        metadata = backend.get_forward_metadata(batch)
        idx = np.asarray(metadata.scatter_idx)
        assert idx.shape == (128,)

    def test_single_aligned_request(self):
        backend = LinearAttentionBackend()
        # T_outer == T_tight == 64, all tokens map to themselves
        batch = _make_metadata_batch(ForwardMode.EXTEND, [64], [64], T_outer=64)
        metadata = backend.get_forward_metadata(batch)
        idx = np.asarray(metadata.scatter_idx)
        np.testing.assert_array_equal(idx[:64], np.arange(64, dtype=np.int32))

    def test_two_requests_mapping(self):
        backend = LinearAttentionBackend()
        # extend=[30,50], T_outer=128
        # request 0: seq_len=30 -> chunk slot [0..63], real tokens [0..29]
        # request 1: seq_len=50 -> chunk slot [64..127], real tokens [0..49]
        batch = _make_metadata_batch(ForwardMode.EXTEND, [30, 50], [30, 50], T_outer=128)
        metadata = backend.get_forward_metadata(batch)
        idx = np.asarray(metadata.scatter_idx)
        T_pb = backend.T_packed_bucket  # 128

        # First 30 tokens map to packed positions 0..29
        np.testing.assert_array_equal(idx[:30], np.arange(0, 30, dtype=np.int32))
        # Next 50 tokens map to packed positions 64..113
        np.testing.assert_array_equal(idx[30:80], np.arange(64, 114, dtype=np.int32))
        # Remaining 48 outer positions map to dummy slot T_pb
        np.testing.assert_array_equal(idx[80:], np.full(48, T_pb, dtype=np.int32))

    def test_padding_slot_maps_to_dummy(self):
        backend = LinearAttentionBackend()
        # extend=[30, 0], T_outer=64 (outer bucket pads to 64)
        batch = _make_metadata_batch(ForwardMode.EXTEND, [30, 0], [30, 0], T_outer=64)
        metadata = backend.get_forward_metadata(batch)
        idx = np.asarray(metadata.scatter_idx)
        T_pb = backend.T_packed_bucket  # 64 (only one real chunk of 64)

        # First 30 tokens map to 0..29
        np.testing.assert_array_equal(idx[:30], np.arange(30, dtype=np.int32))
        # Remaining 34 outer positions map to dummy slot
        np.testing.assert_array_equal(idx[30:], np.full(34, T_pb, dtype=np.int32))


# ---------------------------------------------------------------------------
# Decode no-op test
# ---------------------------------------------------------------------------


class TestDecodeNoOp:
    def test_decode_does_not_crash(self):
        backend = LinearAttentionBackend()
        batch = _make_metadata_batch(ForwardMode.DECODE, None, [10, 20], T_outer=2)
        # Should return immediately without raising
        metadata = backend.get_forward_metadata(batch)
        # State unchanged from init
        assert backend.T_packed_bucket == 0
        # Decode returns an empty LinearAttentionMetadata with None fields.
        assert isinstance(metadata, LinearAttentionMetadata)
        assert metadata.cu_seqlens_dev is None
        assert metadata.scatter_idx is None


# ---------------------------------------------------------------------------
# scatter_to_packed / gather_from_packed tests
# ---------------------------------------------------------------------------


class TestScatterGather:
    def _make_backend_and_batch(self, extend_seq_lens, T_outer=None):
        backend = LinearAttentionBackend()
        batch = _make_metadata_batch(ForwardMode.EXTEND, extend_seq_lens, extend_seq_lens, T_outer=T_outer)
        metadata = backend.get_forward_metadata(batch)
        return backend, metadata

    def test_scatter_output_shape(self):
        backend, metadata = self._make_backend_and_batch([30, 50], T_outer=128)
        H, K = 4, 8
        x = jnp.ones((128, H, K), dtype=jnp.float32)
        scatter_idx = metadata.scatter_idx
        T_pb = backend.T_packed_bucket
        out = scatter_to_packed(x, scatter_idx, T_pb)
        assert out.shape == (1, T_pb, H, K)

    def test_gather_roundtrip_aligned(self):
        """When T_outer == T_tight and no padding, scatter then gather recovers original."""
        backend, metadata = self._make_backend_and_batch([64], T_outer=64)
        H, K = 2, 4
        rng = np.random.default_rng(0)
        x_np = rng.standard_normal((64, H, K)).astype(np.float32)
        x = jnp.array(x_np)
        scatter_idx = metadata.scatter_idx
        T_pb = backend.T_packed_bucket  # 64

        packed = scatter_to_packed(x, scatter_idx, T_pb)
        recovered = gather_from_packed(packed, scatter_idx)

        np.testing.assert_allclose(np.asarray(recovered[:64]), x_np, rtol=1e-6, atol=1e-6)

    def test_gather_roundtrip_multi_request(self):
        """Real tokens (first T_tight) roundtrip exactly; outer padding gathers zeros."""
        extend_seq_lens = [30, 50]
        T_tight = 80
        T_outer = 128
        backend, metadata = self._make_backend_and_batch(extend_seq_lens, T_outer=T_outer)
        H, K = 3, 6
        rng = np.random.default_rng(1)
        # Only set real token values; outer padding is zeros
        x_np = np.zeros((T_outer, H, K), dtype=np.float32)
        x_np[:T_tight] = rng.standard_normal((T_tight, H, K)).astype(np.float32)
        x = jnp.array(x_np)

        scatter_idx = metadata.scatter_idx
        T_pb = backend.T_packed_bucket

        packed = scatter_to_packed(x, scatter_idx, T_pb)
        recovered = gather_from_packed(packed, scatter_idx)

        # Real tokens roundtrip exactly
        np.testing.assert_allclose(
            np.asarray(recovered[:T_tight]), x_np[:T_tight], rtol=1e-6, atol=1e-6
        )
        # Outer padding positions gather zeros (dummy slot)
        np.testing.assert_allclose(
            np.asarray(recovered[T_tight:]), np.zeros((T_outer - T_tight, H, K)), atol=1e-7
        )

    def test_gather_roundtrip_outer_padding(self):
        """T_outer > sum(extend_seq_lens): tail padding tokens map to dummy slot and gather zeros."""
        extend_seq_lens = [50]
        T_tight = 50
        T_outer = 128  # scheduler bucket pads beyond real tokens
        backend, metadata = self._make_backend_and_batch(extend_seq_lens, T_outer=T_outer)
        H, K = 2, 4
        rng = np.random.default_rng(2)
        x_np = np.zeros((T_outer, H, K), dtype=np.float32)
        x_np[:T_tight] = rng.standard_normal((T_tight, H, K)).astype(np.float32)
        x = jnp.array(x_np)

        scatter_idx = metadata.scatter_idx
        T_pb = backend.T_packed_bucket  # ceil(50/64)*64 = 64

        # Verify tail positions in scatter_idx map to dummy slot
        idx_np = np.asarray(scatter_idx)
        np.testing.assert_array_equal(
            idx_np[T_tight:], np.full(T_outer - T_tight, T_pb, dtype=np.int32)
        )

        packed = scatter_to_packed(x, scatter_idx, T_pb)
        recovered = gather_from_packed(packed, scatter_idx)

        # Real tokens roundtrip exactly
        np.testing.assert_allclose(
            np.asarray(recovered[:T_tight]), x_np[:T_tight], rtol=1e-6, atol=1e-6
        )
        # Tail padding positions gather zeros from dummy slot
        np.testing.assert_allclose(
            np.asarray(recovered[T_tight:]), np.zeros((T_outer - T_tight, H, K)), atol=1e-7
        )


# ---------------------------------------------------------------------------
# JIT safety tests --- verify LinearAttentionMetadata survives jax.jit
# ---------------------------------------------------------------------------


class TestJitSafety:
    """Verify that LinearAttentionMetadata flows correctly through jax.jit.

    These tests catch the original bug where nnx.data fields accessed via
    `.value` would crash inside JIT with 'DynamicJaxprTracer has no attribute
    value'. The pytree-based metadata must survive flatten/unflatten and be
    accessible as normal traced arrays inside JIT.
    """

    def test_metadata_pytree_survives_jit(self):
        """LinearAttentionMetadata arrays are accessible inside jax.jit."""
        backend = LinearAttentionBackend()
        batch = _make_metadata_batch(ForwardMode.EXTEND, [30, 50], [30, 50], T_outer=128)
        metadata = backend.get_forward_metadata(batch)

        @jax.jit
        def use_metadata(meta):
            # Access fields --- this would crash with the old .value interface
            return meta.cu_seqlens_dev[0] + meta.scatter_idx[0]

        result = use_metadata(metadata)
        expected = np.asarray(metadata.cu_seqlens_dev)[0] + np.asarray(metadata.scatter_idx)[0]
        assert int(result) == int(expected)

    def test_metadata_in_container_survives_jit(self):
        """Metadata nested in a container (simulating ForwardBatch) works in JIT."""
        backend = LinearAttentionBackend()
        batch = _make_metadata_batch(ForwardMode.EXTEND, [64], [64], T_outer=64)
        metadata = backend.get_forward_metadata(batch)

        # Simulate ForwardBatch by nesting metadata in a tuple (pytree container)
        @jax.jit
        def scatter_inside_jit(x, meta):
            packed = scatter_to_packed(x, meta.scatter_idx, backend.T_packed_bucket)
            recovered = gather_from_packed(packed, meta.scatter_idx)
            return recovered

        H, K = 2, 4
        x = jax.random.normal(jax.random.PRNGKey(0), (64, H, K))
        recovered = scatter_inside_jit(x, metadata)

        np.testing.assert_allclose(np.asarray(recovered), np.asarray(x), rtol=1e-6, atol=1e-6)

    def test_decode_metadata_none_fields_in_jit(self):
        """DECODE metadata (None fields) can be passed through JIT without crash."""
        backend = LinearAttentionBackend()
        batch = _make_metadata_batch(ForwardMode.DECODE, None, [10, 20], T_outer=2)
        metadata = backend.get_forward_metadata(batch)

        @jax.jit
        def check_decode_meta(meta):
            # In decode, both fields are None --- verify JIT doesn't crash
            # Return a constant to prove the function executed
            return jnp.array(42)

        result = check_decode_meta(metadata)
        assert int(result) == 42


# ---------------------------------------------------------------------------
# LightningAttnBackend tests
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_basic_construction(self):
        backend = LightningAttnBackend(mesh=mesh)
        assert backend.mesh is mesh
        assert backend.chunk_size == 64
        assert backend.T_packed_bucket == 0
        assert backend.scatter_idx is None
        assert backend.cu_seqlens_aligned is None

    def test_custom_chunk_size(self):
        backend = LightningAttnBackend(mesh=mesh, chunk_size=128)
        assert backend.chunk_size == 128


class TestGetForwardMetadata:
    def test_decode_returns_base_metadata(self):
        backend = LightningAttnBackend(mesh=mesh)
        batch = _make_batch(
            ForwardMode.DECODE,
            recurrent_indices=np.array([1, 2]),
        )
        metadata = backend.get_forward_metadata(batch)
        assert metadata.cu_q_lens is not None
        assert metadata.recurrent_indices is not None
        assert backend.scatter_idx is None
        assert backend.cu_seqlens_aligned is None

    def test_extend_populates_scatter_metadata(self):
        backend = LightningAttnBackend(mesh=mesh)
        batch = _make_batch(
            ForwardMode.EXTEND,
            extend_seq_lens=np.array([100, 50]),
            input_ids=np.zeros(150),
            recurrent_indices=np.array([1, 2]),
        )
        metadata = backend.get_forward_metadata(batch)
        assert metadata.cu_q_lens is not None
        assert metadata.recurrent_indices is not None
        assert backend.scatter_idx is not None
        assert backend.cu_seqlens_aligned is not None
        assert backend.T_packed_bucket > 0

    def test_scatter_metadata_matches_old_backend(self):
        """Scatter/gather metadata should match LinearAttentionBackend output."""
        old_backend = LinearAttentionBackend(mesh=mesh)
        new_backend = LightningAttnBackend(mesh=mesh)

        batch = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            extend_seq_lens=np.array([100, 60, 128], dtype=np.int32),
            seq_lens=np.array([100, 60, 128], dtype=np.int32),
            input_ids=np.zeros(288, dtype=np.int32),
            recurrent_indices=np.array([1, 2, 3], dtype=np.int32),
        )
        old_meta = old_backend.get_forward_metadata(batch)
        new_backend.get_forward_metadata(batch)

        np.testing.assert_array_equal(
            np.array(old_meta.cu_seqlens_dev),
            np.array(new_backend.cu_seqlens_aligned),
        )
        np.testing.assert_array_equal(
            np.array(old_meta.scatter_idx),
            np.array(new_backend.scatter_idx),
        )
        assert old_backend.T_packed_bucket == new_backend.T_packed_bucket

    def test_chunk_alignment(self):
        backend = LightningAttnBackend(mesh=mesh, chunk_size=64)
        batch = _make_batch(
            ForwardMode.EXTEND,
            extend_seq_lens=np.array([100]),
            input_ids=np.zeros(100),
            recurrent_indices=np.array([1]),
        )
        backend.get_forward_metadata(batch)
        assert backend.T_packed_bucket == 128  # ceil(100/64)*64


class TestStateManagement:
    def test_get_state(self):
        backend = LightningAttnBackend(mesh=mesh)
        layer_id = 5
        state = jnp.ones((_H, _K, _K), dtype=jnp.float32)
        pool, indices = _make_mock_pool(layer_id, state[None])
        result = backend.get_state(pool, layer_id, jnp.array(indices))
        np.testing.assert_array_equal(np.array(result[0]), np.array(state))

    def test_set_ssm_state(self):
        backend = LightningAttnBackend(mesh=mesh)
        layer_id = 5
        state = jnp.zeros((1, _H, _K, _K), dtype=jnp.float32)
        pool, indices = _make_mock_pool(layer_id, state)

        new_state = jnp.ones((1, _H, _K, _K), dtype=jnp.float32)
        new_full = backend.set_ssm_state(pool, layer_id, jnp.array(indices), new_state)

        np.testing.assert_array_equal(
            np.array(new_full[indices[0]]),
            np.array(new_state[0]),
        )

    def test_no_conv_state(self):
        """GLA should return empty conv list in pool_updates."""
        backend = LightningAttnBackend(mesh=mesh)
        layer_id = 5
        state = jnp.zeros((1, _H, _K, _K), dtype=jnp.float32)
        pool, indices = _make_mock_pool(layer_id, state)

        _, conv = pool.get_linear_recurrent_layer_cache(layer_id)
        assert conv == []


class TestForwardDecode:
    @requires_simple_gla
    def test_decode_output_shape(self):
        backend = LightningAttnBackend(mesh=mesh)
        layer_id = 5
        T = 2

        with jax.set_mesh(mesh):
            state = jnp.zeros((T, _H, _K, _K), dtype=jnp.float32)
            pool, rec_indices = _make_mock_pool(layer_id, state)

            batch = _make_batch(
                ForwardMode.DECODE,
                recurrent_indices=rec_indices,
            )
            metadata = backend.get_forward_metadata(batch)
            backend.forward_metadata = metadata

            q = jax.random.normal(jax.random.PRNGKey(0), (T, _H, _K), dtype=jnp.bfloat16)
            k = jax.random.normal(jax.random.PRNGKey(1), (T, _H, _K), dtype=jnp.bfloat16)
            v = jax.random.normal(jax.random.PRNGKey(2), (T, _H, _K), dtype=jnp.bfloat16)
            layer = _make_fake_layer(layer_id=layer_id)
            fb = SimpleNamespace(forward_mode=ForwardMode.DECODE)

            output, pool_updates = backend(q, k, v, layer=layer, forward_batch=fb, recurrent_state_pool=pool)

        assert output.shape == (T, _H * _K)
        new_state = _extract_state(pool_updates, rec_indices)
        assert new_state.shape == (T, _H, _K, _K)

    @requires_simple_gla
    def test_decode_state_updates(self):
        backend = LightningAttnBackend(mesh=mesh)
        layer_id = 5

        with jax.set_mesh(mesh):
            state0 = jnp.zeros((1, _H, _K, _K), dtype=jnp.float32)
            pool, rec_indices = _make_mock_pool(layer_id, state0)

            batch = _make_batch(
                ForwardMode.DECODE,
                recurrent_indices=rec_indices,
            )
            metadata = backend.get_forward_metadata(batch)
            backend.forward_metadata = metadata

            q = jax.random.normal(jax.random.PRNGKey(0), (1, _H, _K), dtype=jnp.bfloat16)
            k = jax.random.normal(jax.random.PRNGKey(1), (1, _H, _K), dtype=jnp.bfloat16)
            v = jax.random.normal(jax.random.PRNGKey(2), (1, _H, _K), dtype=jnp.bfloat16)
            layer = _make_fake_layer(layer_id=layer_id)
            fb = SimpleNamespace(forward_mode=ForwardMode.DECODE)

            _, pool_updates = backend(q, k, v, layer=layer, forward_batch=fb, recurrent_state_pool=pool)
            new_state = _extract_state(pool_updates, rec_indices)

        assert not jnp.allclose(new_state, state0), "State should change after decode"

    @requires_simple_gla
    def test_decode_matches_direct_kernel(self):
        """Backend decode should match direct fused_recurrent_simple_gla call."""
        backend = LightningAttnBackend(mesh=mesh)
        layer_id = 5

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
            slope = jnp.array([-0.1, -0.2, -0.3, -0.4], dtype=jnp.float32)
            layer = _make_fake_layer(layer_id=layer_id, slope=slope)
            fb = SimpleNamespace(forward_mode=ForwardMode.DECODE)

            output, pool_updates = backend(q, k, v, layer=layer, forward_batch=fb, recurrent_state_pool=pool)
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
            np.array(output), np.array(direct_out), atol=1e-5,
            err_msg="Backend decode output != direct kernel",
        )
        np.testing.assert_allclose(
            np.array(backend_state), np.array(direct_state), atol=1e-5,
            err_msg="Backend decode state != direct kernel state",
        )


class TestForwardExtend:
    @requires_simple_gla
    @requires_tpu
    def test_extend_output_shape(self):
        backend = LightningAttnBackend(mesh=mesh)
        layer_id = 5
        seq_len = 128

        with jax.set_mesh(mesh):
            state = jnp.zeros((1, _H, _K, _K), dtype=jnp.float32)
            pool, rec_indices = _make_mock_pool(layer_id, state)

            batch = _make_batch(
                ForwardMode.EXTEND,
                extend_seq_lens=np.array([seq_len]),
                input_ids=np.zeros(seq_len),
                recurrent_indices=rec_indices,
            )
            metadata = backend.get_forward_metadata(batch)
            backend.forward_metadata = metadata

            q = jax.random.normal(jax.random.PRNGKey(0), (seq_len, _H, _K), dtype=jnp.bfloat16)
            k = jax.random.normal(jax.random.PRNGKey(1), (seq_len, _H, _K), dtype=jnp.bfloat16)
            v = jax.random.normal(jax.random.PRNGKey(2), (seq_len, _H, _K), dtype=jnp.bfloat16)
            layer = _make_fake_layer(layer_id=layer_id)
            fb = SimpleNamespace(forward_mode=ForwardMode.EXTEND)

            output, pool_updates = backend(q, k, v, layer=layer, forward_batch=fb, recurrent_state_pool=pool)

        assert output.shape == (seq_len, _H * _K)
        new_state = _extract_state(pool_updates, rec_indices)
        assert new_state.shape == (1, _H, _K, _K)


class TestHybridIntegration:
    def test_dispatch_routes_to_lightning(self):
        from sgl_jax.srt.layers.attention.flashattention_backend import FlashAttention

        with jax.set_mesh(mesh):
            full_backend = FlashAttention(
                num_attn_heads=_H, num_kv_heads=_H, head_dim=_K,
                page_size=1, mesh=mesh,
            )
            lightning = LightningAttnBackend(mesh=mesh)
            hybrid = HybridLinearAttnBackend(full_backend, lightning, full_attn_layers=[0, 1])

        assert 5 not in hybrid.full_attn_layers
        assert isinstance(hybrid.linear_attn_backend, LightningAttnBackend)

    def test_forward_metadata_setter_propagates(self):
        from sgl_jax.srt.layers.attention.flashattention_backend import FlashAttention

        with jax.set_mesh(mesh):
            full_backend = FlashAttention(
                num_attn_heads=_H, num_kv_heads=_H, head_dim=_K,
                page_size=1, mesh=mesh,
            )
            lightning = LightningAttnBackend(mesh=mesh)
            hybrid = HybridLinearAttnBackend(full_backend, lightning, full_attn_layers=[0])

            batch = SimpleNamespace(
                forward_mode=ForwardMode.DECODE,
                recurrent_indices=np.array([1]),
                seq_lens=np.array([1]),
                cache_loc=np.array([0]),
            )
            metadata = hybrid.get_forward_metadata(batch)
            hybrid.forward_metadata = metadata

        assert lightning.forward_metadata.recurrent_indices is not None
