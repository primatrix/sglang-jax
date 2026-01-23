import unittest

import jax
import jax.numpy as jnp

from sgl_jax.srt.multimodal.layers.attention.layer import ring_attention, simple_attention


class TestRingAttention(unittest.TestCase):
    def _run_case(self, causal: bool):
        key = jax.random.key(0)
        b, s, h, d = 2, 64, 4, 16
        q = jax.random.normal(key, (b, s, h, d), dtype=jnp.float32)
        k = jax.random.normal(jax.random.key(1), (b, s, h, d), dtype=jnp.float32)
        v = jax.random.normal(jax.random.key(2), (b, s, h, d), dtype=jnp.float32)

        out_simple = simple_attention(q, k, v, causal=causal)
        out_ring = ring_attention(q, k, v, causal=causal, block_q=16, block_k=16)

        self.assertTrue(jnp.allclose(out_ring, out_simple, rtol=1e-4, atol=1e-4))

    def test_ring_attention_non_causal(self):
        self._run_case(causal=False)

    def test_ring_attention_causal(self):
        self._run_case(causal=True)


if __name__ == "__main__":
    unittest.main()
