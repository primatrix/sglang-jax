import unittest
from unittest import mock

import jax
import jax.experimental.pallas as pl
import jax.numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.multimodal.kernels.flash_attention import (
    DEFAULT_VMEM_LIMIT_BYTES,
    BlockSizes,
    SegmentIds,
    _select_default_block_sizes,
    flash_attention,
    mha_reference_no_custom_vjp,
)


@jax.jit
def jit_flash_attention(q, k, v):
    q_len = q.shape[2]
    kv_len = k.shape[2]
    align_q_len = align_to(q_len, 256)
    align_kv_len = align_to(kv_len, 256)
    seg_q = None
    seg_kv = None
    segment_ids = None
    if q_len != align_q_len:
        q = jnp.pad(q, ((0, 0), (0, 0), (0, align_q_len - q_len), (0, 0)))
        seg_q = jnp.concatenate(
            [jnp.ones((q.shape[0], q_len)), jnp.zeros((q.shape[0], align_q_len - q_len))], axis=1
        )
    if kv_len != align_kv_len:
        k = jnp.pad(k, ((0, 0), (0, 0), (0, align_kv_len - kv_len), (0, 0)))
        v = jnp.pad(v, ((0, 0), (0, 0), (0, align_kv_len - kv_len), (0, 0)))
        seg_kv = jnp.concatenate(
            [jnp.ones((k.shape[0], kv_len)), jnp.zeros((k.shape[0], align_kv_len - kv_len))], axis=1
        )
    if seg_q is not None and seg_kv is not None:
        segment_ids = SegmentIds(q=seg_q, kv=seg_kv)
    output = flash_attention(q, k, v, segment_ids=segment_ids, causal=False)
    output = output[:, :, :q_len, :]
    return output


def simple_attention(q, k, v, sm_scale=1.0):
    attn_weights = jnp.einsum("bhsd,bhtd->bhst", q, k)
    attn_weights *= sm_scale
    attn_weights = jax.nn.softmax(attn_weights, axis=-1)
    output = jnp.einsum("bhst,bhtd->bhsd", attn_weights, v)
    return output


def align_to(a, b):
    return pl.cdiv(a, b) * b


class TestFlashAttentionKernel(unittest.TestCase):
    """Test flash attention kernel"""

    @mock.patch(
        "sgl_jax.srt.multimodal.kernels.flash_attention.get_tuned_block_sizes",
        return_value=256,
    )
    def test_long_sequence_uses_online_softmax(self, _):
        """64K+ KV must stay on the bounded-VMEM, tiled-K path."""
        seq_len = 64 * 1024 + 128
        shape = (1, 1, seq_len, 128)
        q = jax.ShapeDtypeStruct(shape, jnp.bfloat16)
        k = jax.ShapeDtypeStruct(shape, jnp.bfloat16)
        v = jax.ShapeDtypeStruct(shape, jnp.bfloat16)
        segment_ids = SegmentIds(
            q=jax.ShapeDtypeStruct((1, seq_len), jnp.int32),
            kv=jax.ShapeDtypeStruct((1, seq_len), jnp.int32),
        )

        blocks = _select_default_block_sizes(
            q,
            k,
            v,
            None,
            segment_ids,
            vmem_limit_bytes=DEFAULT_VMEM_LIMIT_BYTES,
        )

        self.assertEqual(blocks.block_k_major, 128)
        self.assertEqual(blocks.block_k, 128)
        self.assertLess(blocks.block_k, seq_len)

    def test_online_softmax_accuracy(self):
        """The bounded-VMEM path matches the reference attention."""
        seq_len = 256
        key = jax.random.PRNGKey(11)
        q_key, k_key, v_key = jax.random.split(key, 3)
        shape = (1, 1, seq_len, 128)
        q = jax.random.normal(q_key, shape, dtype=jnp.bfloat16)
        k = jax.random.normal(k_key, shape, dtype=jnp.bfloat16)
        v = jax.random.normal(v_key, shape, dtype=jnp.bfloat16)
        sm_scale = shape[-1] ** -0.5
        segment_ids = SegmentIds(
            q=jnp.ones((1, seq_len), dtype=jnp.int32),
            kv=jnp.ones((1, seq_len), dtype=jnp.int32),
        )
        blocks = BlockSizes(
            block_q=128,
            block_b=1,
            block_k_major=128,
            block_k=128,
        )

        actual = flash_attention(
            q,
            k,
            v,
            segment_ids=segment_ids,
            block_sizes=blocks,
            sm_scale=sm_scale,
            interpret=True,
        )
        expected = simple_attention(q, k, v, sm_scale)

        np.testing.assert_allclose(
            np.asarray(actual, dtype=np.float32),
            np.asarray(expected, dtype=np.float32),
            rtol=2e-2,
            atol=2e-2,
        )

    def test_local_segment_grid_accuracy_across_k_tiles(self):
        """A bounded segment crossing a K tile matches dense segmented attention."""
        seq_len = 256
        head_dim = 80
        q_key, k_key, v_key = jax.random.split(jax.random.PRNGKey(19), 3)
        shape = (1, 1, seq_len, head_dim)
        q = jax.random.normal(q_key, shape, dtype=jnp.bfloat16)
        k = jax.random.normal(k_key, shape, dtype=jnp.bfloat16)
        v = jax.random.normal(v_key, shape, dtype=jnp.bfloat16)
        # The +32 offset makes 64-token segments straddle 128-token K tiles.
        segments = ((jnp.arange(seq_len, dtype=jnp.int32) + 32) // 64)[None, :]
        segment_ids = SegmentIds(q=segments, kv=segments)
        sm_scale = head_dim**-0.5
        blocks = BlockSizes(
            block_q=128,
            block_b=1,
            block_k_major=128,
            block_k=128,
        )

        actual = flash_attention(
            q,
            k,
            v,
            segment_ids=segment_ids,
            block_sizes=blocks,
            sm_scale=sm_scale,
            max_segment_len=64,
            interpret=True,
        )
        expected = mha_reference_no_custom_vjp(
            q,
            k,
            v,
            segment_ids=segment_ids,
            sm_scale=sm_scale,
        )

        np.testing.assert_allclose(
            np.asarray(actual, dtype=np.float32),
            np.asarray(expected, dtype=np.float32),
            rtol=2e-2,
            atol=2e-2,
        )

    def test_block_sparse_full_attention_accuracy(self):
        """Skipping cross-image blocks preserves exact segmented attention."""
        seq_len = 512
        valid_len = 448
        head_dim = 80
        q_key, k_key, v_key = jax.random.split(jax.random.PRNGKey(29), 3)
        shape = (1, 1, seq_len, head_dim)
        q = jax.random.normal(q_key, shape, dtype=jnp.bfloat16)
        k = jax.random.normal(k_key, shape, dtype=jnp.bfloat16)
        v = jax.random.normal(v_key, shape, dtype=jnp.bfloat16)
        lengths = [96, 160, 80, 112]
        real_segments = np.repeat(np.arange(len(lengths), dtype=np.int32), lengths)
        segments = np.pad(
            real_segments,
            (0, seq_len - real_segments.size),
            constant_values=-1,
        )[None, :]
        segment_ids = SegmentIds(q=jnp.asarray(segments), kv=jnp.asarray(segments))
        sm_scale = head_dim**-0.5
        blocks = BlockSizes(
            block_q=128,
            block_b=1,
            block_k_major=128,
            block_k=128,
        )

        actual = flash_attention(
            q,
            k,
            v,
            segment_ids=segment_ids,
            block_sizes=blocks,
            sm_scale=sm_scale,
            block_sparse_segments=True,
            interpret=True,
        )
        expected = mha_reference_no_custom_vjp(
            q,
            k,
            v,
            segment_ids=segment_ids,
            sm_scale=sm_scale,
        )

        np.testing.assert_allclose(
            np.asarray(actual[:, :, :valid_len], dtype=np.float32),
            np.asarray(expected[:, :, :valid_len], dtype=np.float32),
            rtol=2e-2,
            atol=2e-2,
        )

    def test_accuracy(self):
        """Test flash attention accuracy"""
        mesh = jax.make_mesh(
            (1, 1, 1, 1), axis_names=("x", "y", "z", "p"), devices=[jax.devices()[0]]
        )
        sharding = jax.sharding.NamedSharding(mesh, P(None, None, None, None))
        q_shape = (2, 12, 120, 128)
        kv_shape = (2, 12, 60, 128)
        key = jax.random.PRNGKey(1)
        key1, key2 = jax.random.split(key, num=2)
        q = jax.random.normal(key, q_shape)
        k = jax.random.normal(key1, kv_shape)
        v = jax.random.normal(key2, kv_shape)

        q = jax.device_put(q, sharding)
        k = jax.device_put(k, sharding)
        v = jax.device_put(v, sharding)

        flash_output = jit_flash_attention(q, k, v)
        simple_output = simple_attention(q, k, v)
        print(flash_output.shape, simple_output.shape)
        np.testing.assert_allclose(np.array(flash_output), np.array(simple_output), 1e-5, 1e-5)


if __name__ == "__main__":
    unittest.main()
