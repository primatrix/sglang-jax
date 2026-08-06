import unittest

import jax
import jax.experimental.pallas as pl
import jax.numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.multimodal.kernels.flash_attention import (
    BlockSizes,
    SegmentIds,
    _get_block_sparse_default_block_sizes,
    _segment_block_sparse_schedule,
    flash_attention,
    mha_reference_no_custom_vjp,
)
from sgl_jax.srt.multimodal.layers.attention.flash_attention_backend import (
    VisionAttentionMetadata,
    VisionFlashAttentionBackend,
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
    output = flash_attention(
        q,
        k,
        v,
        segment_ids=segment_ids,
        causal=False,
        interpret=jax.default_backend() == "cpu",
    )
    output = output[:, :, :q_len, :]
    return output


def simple_attention(q, k, v):
    attn_weights = jnp.einsum("bhsd,bhtd->bhst", q, k)
    attn_weights = jax.nn.softmax(attn_weights, axis=-1)
    output = jnp.einsum("bhst,bhtd->bhsd", attn_weights, v)
    return output


def align_to(a, b):
    return pl.cdiv(a, b) * b


class TestFlashAttentionKernel(unittest.TestCase):
    """Test flash attention kernel"""

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

    def test_block_sparse_schedule(self):
        segment_ids = jnp.asarray(
            [[0, 0, 0, 0, 1, 1, 1, 1], [2, 2, 3, 3, -1, -1, -1, -1]],
            dtype=jnp.int32,
        )

        block_mask = _segment_block_sparse_schedule(
            segment_ids,
            segment_ids,
            block_q=4,
            block_k_major=4,
        )

        np.testing.assert_array_equal(
            block_mask,
            np.asarray(
                [
                    [[1, 0], [0, 1]],
                    [[1, 0], [0, 0]],
                ],
                dtype=np.int32,
            ),
        )

    def test_block_sparse_32k_uses_v7x_safe_tiles(self):
        blocks = _get_block_sparse_default_block_sizes(32 * 1024, 32 * 1024)

        self.assertEqual(blocks.block_q, 512)
        self.assertEqual(blocks.block_k_major, 256)
        self.assertEqual(blocks.block_k, 128)

    def test_block_sparse_interpret_matches_reference(self):
        key = jax.random.key(17)
        keys = jax.random.split(key, 3)
        shape = (1, 2, 512, 128)
        q = jax.random.normal(keys[0], shape, dtype=jnp.float32)
        k = jax.random.normal(keys[1], shape, dtype=jnp.float32)
        v = jax.random.normal(keys[2], shape, dtype=jnp.float32)
        ids = jnp.concatenate(
            (
                jnp.full(96, 0, dtype=jnp.int32),
                jnp.full(160, 1, dtype=jnp.int32),
                jnp.full(128, 2, dtype=jnp.int32),
                jnp.full(128, -1, dtype=jnp.int32),
            )
        )[None]
        segments = SegmentIds(q=ids, kv=ids)
        blocks = BlockSizes(
            block_q=256,
            block_k_major=128,
            block_k=128,
            block_b=1,
        )

        sparse_output = flash_attention(
            q,
            k,
            v,
            segment_ids=segments,
            block_sizes=blocks,
            block_sparse_segments=True,
            interpret=True,
        )
        expected = mha_reference_no_custom_vjp(q, k, v, segment_ids=segments)
        tolerance = 1e-5 if jax.default_backend() == "cpu" else 1e-2

        np.testing.assert_allclose(
            np.asarray(sparse_output[:, :, :384]),
            np.asarray(expected[:, :, :384]),
            rtol=tolerance,
            atol=tolerance,
        )

    @unittest.skipUnless("TPU" in jax.devices()[0].device_kind, "Requires a TPU")
    def test_block_sparse_tpu_matches_dense_segmented(self):
        key = jax.random.key(29)
        keys = jax.random.split(key, 3)
        shape = (1, 2, 1024, 128)
        q = jax.random.normal(keys[0], shape, dtype=jnp.bfloat16)
        k = jax.random.normal(keys[1], shape, dtype=jnp.bfloat16)
        v = jax.random.normal(keys[2], shape, dtype=jnp.bfloat16)
        ids = jnp.repeat(
            jnp.arange(4, dtype=jnp.int32),
            jnp.asarray([128, 256, 384, 256], dtype=jnp.int32),
            total_repeat_length=1024,
        )[None]
        segments = SegmentIds(q=ids, kv=ids)
        blocks = BlockSizes(
            block_q=256,
            block_k_major=128,
            block_k=128,
            block_b=1,
        )

        sparse_output = flash_attention(
            q,
            k,
            v,
            segment_ids=segments,
            block_sizes=blocks,
            block_sparse_segments=True,
        )
        dense_output = flash_attention(
            q,
            k,
            v,
            segment_ids=segments,
            block_sizes=blocks,
        )
        sparse_dense_error = jnp.max(
            jnp.abs(sparse_output.astype(jnp.float32) - dense_output.astype(jnp.float32))
        )
        self.assertLessEqual(float(sparse_dense_error), 0.02)

    @unittest.skipUnless("TPU" in jax.devices()[0].device_kind, "Requires a TPU")
    def test_block_sparse_vision_backend_tpu_integration(self):
        mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:1]), ("data",))
        qkv = jnp.ones((1, 1024, 1, 80), dtype=jnp.bfloat16)  # [B, T, H, D]
        cu_seqlens = jnp.asarray([[0, 128, 384, 768, 1024]], dtype=jnp.int32)
        backend = VisionFlashAttentionBackend(
            mesh,
            block_sparse_segments=True,
        )

        output = backend(qkv, qkv, qkv, VisionAttentionMetadata(cu_seqlens))

        self.assertEqual(output.shape, qkv.shape)
        self.assertEqual(float(output[0, 0, 0, 0]), 1.0)

    def test_block_sparse_rejects_unsupported_combinations(self):
        qkv = jnp.ones((1, 1, 256, 128), dtype=jnp.float32)
        ids = jnp.zeros((1, 256), dtype=jnp.int32)
        segments = SegmentIds(q=ids, kv=ids)
        blocks = BlockSizes(128, 128, 128, 1)

        with self.assertRaisesRegex(ValueError, "requires segment_ids"):
            flash_attention(
                qkv,
                qkv,
                qkv,
                block_sizes=blocks,
                block_sparse_segments=True,
                interpret=True,
            )
        with self.assertRaisesRegex(ValueError, "non-causal"):
            flash_attention(
                qkv,
                qkv,
                qkv,
                segment_ids=segments,
                causal=True,
                block_sizes=blocks,
                block_sparse_segments=True,
                interpret=True,
            )


if __name__ == "__main__":
    unittest.main()
