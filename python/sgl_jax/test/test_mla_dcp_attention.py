"""Two-device numerical test for the correctness-first MLA DCP path.

Run on CPU with:

  XLA_FLAGS=--xla_force_host_platform_device_count=2 \
    python -m unittest sgl_jax.test.test_mla_dcp_attention
"""

import unittest

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.layers.attention.mla_dcp import mla_dcp_attention_local


class TestMLADCPAttention(unittest.TestCase):
    PAGE_SIZE = 8
    DCP_SIZE = 2
    NUM_HEADS = 4
    LKV_DIM = 4
    ROPE_DIM = 4

    def setUp(self):
        if len(jax.devices()) < self.DCP_SIZE:
            self.skipTest("MLA DCP test requires at least two JAX devices")
        devices = np.asarray(jax.devices()[: self.DCP_SIZE], dtype=object).reshape(1, 1, 2)
        self.mesh = Mesh(
            devices,
            ("data", "tensor", "dcp"),
            axis_types=(jax.sharding.AxisType.Explicit,) * 3,
        )

    @staticmethod
    def _put(mesh, value, spec):
        return jax.device_put(jnp.asarray(value), NamedSharding(mesh, spec))

    def test_extend_matches_full_context_reference_and_stripes_cache(self):
        rng = np.random.default_rng(7)
        ql = rng.normal(size=(2, self.NUM_HEADS, self.LKV_DIM)).astype(np.float32)
        qpe = rng.normal(size=(2, self.NUM_HEADS, self.ROPE_DIM)).astype(np.float32)
        prefix_c = rng.normal(size=(4, self.LKV_DIM)).astype(np.float32)
        prefix_pe = rng.normal(size=(4, self.ROPE_DIM)).astype(np.float32)
        new_c = rng.normal(size=(2, self.LKV_DIM)).astype(np.float32)
        new_pe = rng.normal(size=(2, self.ROPE_DIM)).astype(np.float32)

        padded_lkv = 128
        padded_rope = 128
        kv_dim = padded_lkv + padded_rope
        packing = 1
        page_words = self.PAGE_SIZE // packing
        local_page_words = page_words // self.DCP_SIZE
        cache_np = np.zeros((4, page_words, packing, kv_dim), dtype=np.float32)

        # Page 1 owns this sequence. Store prefix positions in the opaque DCP
        # layout: a rank's contiguous word chunk contains its modulo stripe.
        for pos in range(4):
            owner = pos % self.DCP_SIZE
            local_pos = pos // self.DCP_SIZE
            word = owner * local_page_words + local_pos // packing
            lane = local_pos % packing
            cache_np[1, word, lane, : self.LKV_DIM] = prefix_c[pos]
            cache_np[1, word, lane, padded_lkv : padded_lkv + self.ROPE_DIM] = prefix_pe[pos]

        scale = float(1.0 / np.sqrt(self.LKV_DIM + self.ROPE_DIM))
        with jax.set_mesh(self.mesh):
            metadata_spec = P("data")
            seq_lens = self._put(self.mesh, np.array([6], np.int32), metadata_spec)
            page_indices = self._put(self.mesh, np.array([1], np.int32), metadata_spec)
            cu_q_lens = self._put(self.mesh, np.array([0, 2], np.int32), metadata_spec)
            cu_kv_lens = self._put(self.mesh, np.array([0, 8], np.int32), metadata_spec)
            distribution = self._put(self.mesh, np.array([0, 0, 1], np.int32), metadata_spec)

            ql_d = self._put(self.mesh, ql.astype(np.float32), P("data", "tensor", None))
            qpe_d = self._put(self.mesh, qpe.astype(np.float32), P("data", "tensor", None))
            new_c_d = self._put(self.mesh, new_c.astype(np.float32), P("data", None))
            new_pe_d = self._put(self.mesh, new_pe.astype(np.float32), P("data", None))
            cache_d = self._put(
                self.mesh, cache_np.astype(np.float32), P("data", "dcp", None, None)
            )
            out_cache_loc = self._put(
                self.mesh, np.array([self.PAGE_SIZE + 4, self.PAGE_SIZE + 5], np.int32), P("data")
            )

            def run(*args):
                return mla_dcp_attention_local(
                    *args,
                    page_size=self.PAGE_SIZE,
                    dcp_size=self.DCP_SIZE,
                    sm_scale=scale,
                    sliding_window=None,
                    soft_cap=None,
                )

            output, updated_cache = jax.shard_map(
                run,
                in_specs=(
                    P("data", "tensor", None),
                    P("data", "tensor", None),
                    P("data", None),
                    P("data", None),
                    P("data", "dcp", None, None),
                    P("data"),
                    P("data"),
                    P("data"),
                    P("data"),
                    P("data"),
                    P("data"),
                ),
                out_specs=(P("data", "tensor", None), P("data", "dcp", None, None)),
                check_vma=False,
            )(
                ql_d,
                qpe_d,
                new_c_d,
                new_pe_d,
                cache_d,
                seq_lens,
                page_indices,
                cu_q_lens,
                cu_kv_lens,
                distribution,
                out_cache_loc,
            )

        full_c = np.concatenate((prefix_c, new_c), axis=0)
        full_pe = np.concatenate((prefix_pe, new_pe), axis=0)
        full_q = np.concatenate((ql, qpe), axis=-1)
        full_k = np.concatenate((full_c, full_pe), axis=-1)
        scores = np.einsum("qhd,kd->qhk", full_q, full_k) * scale
        causal = np.arange(6)[None, :] <= np.array([4, 5])[:, None]
        scores = np.where(causal[:, None, :], scores, -np.inf)
        probs = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        probs /= np.sum(probs, axis=-1, keepdims=True)
        expected = np.einsum("qhk,kd->qhd", probs, full_c)

        np.testing.assert_allclose(
            np.asarray(output, dtype=np.float32), expected, rtol=2e-4, atol=2e-4
        )

        cache_after = np.asarray(updated_cache, dtype=np.float32)
        for pos, (expected_c, expected_pe) in enumerate(zip(full_c, full_pe, strict=True)):
            owner = pos % self.DCP_SIZE
            local_pos = pos // self.DCP_SIZE
            word = owner * local_page_words + local_pos // packing
            lane = local_pos % packing
            np.testing.assert_allclose(
                cache_after[1, word, lane, : self.LKV_DIM], expected_c, rtol=0, atol=0
            )
            np.testing.assert_allclose(
                cache_after[1, word, lane, padded_lkv : padded_lkv + self.ROPE_DIM],
                expected_pe,
                rtol=0,
                atol=0,
            )


if __name__ == "__main__":
    unittest.main()
