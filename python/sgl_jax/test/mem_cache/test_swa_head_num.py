"""Tests for SWAKVPool swa_head_num parameter support."""

import unittest

import jax.numpy as jnp

from sgl_jax.srt.mem_cache.memory_pool import MHATokenToKVPool, SWAKVPool
from sgl_jax.srt.utils.mesh_utils import create_device_mesh

mesh = create_device_mesh(ici_parallelism=[1, -1], dcn_parallelism=[1, 1])


class TestSWAKVPoolHeadNum(unittest.TestCase):
    """Verify that SWAKVPool correctly applies swa_head_num to the SWA sub-pool."""

    def test_swa_head_num_override(self):
        """SWA sub-pool should use swa_head_num when provided."""
        fa_head_num = 4
        swa_head_num = 8

        pool = SWAKVPool(
            size=32,
            size_swa=16,
            page_size=1,
            swa_attention_layer_ids=[1, 3],
            full_attention_layer_ids=[0, 2],
            full_pool_class=MHATokenToKVPool,
            swa_pool_class=MHATokenToKVPool,
            dtype=jnp.bfloat16,
            head_num=fa_head_num,
            head_dim=128,
            mesh=mesh,
            swa_head_num=swa_head_num,
        )

        self.assertEqual(pool.full_kv_pool.head_num, fa_head_num)
        self.assertEqual(pool.swa_kv_pool.head_num, swa_head_num)

    def test_swa_head_num_default(self):
        """Without swa_head_num, SWA sub-pool should inherit head_num from FA."""
        head_num = 4

        pool = SWAKVPool(
            size=32,
            size_swa=16,
            page_size=1,
            swa_attention_layer_ids=[1, 3],
            full_attention_layer_ids=[0, 2],
            full_pool_class=MHATokenToKVPool,
            swa_pool_class=MHATokenToKVPool,
            dtype=jnp.bfloat16,
            head_num=head_num,
            head_dim=128,
            mesh=mesh,
        )

        self.assertEqual(pool.full_kv_pool.head_num, head_num)
        self.assertEqual(pool.swa_kv_pool.head_num, head_num)

    def test_swa_head_num_buffer_shape(self):
        """Verify the actual buffer shapes reflect the different head counts."""
        fa_head_num = 4
        swa_head_num = 8
        head_dim = 128

        pool = SWAKVPool(
            size=32,
            size_swa=16,
            page_size=1,
            swa_attention_layer_ids=[1],
            full_attention_layer_ids=[0],
            full_pool_class=MHATokenToKVPool,
            swa_pool_class=MHATokenToKVPool,
            dtype=jnp.bfloat16,
            head_num=fa_head_num,
            head_dim=head_dim,
            mesh=mesh,
            swa_head_num=swa_head_num,
        )

        # MHATokenToKVPool uses fused KV: shape = [size+page, head_num*2, head_dim]
        fa_buf = pool.full_kv_pool.kv_buffer[0]
        swa_buf = pool.swa_kv_pool.kv_buffer[0]

        self.assertEqual(fa_buf.shape[1], fa_head_num * 2)
        self.assertEqual(swa_buf.shape[1], swa_head_num * 2)


if __name__ == "__main__":
    unittest.main()
