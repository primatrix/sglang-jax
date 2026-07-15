"""Host-side contract tests for the DSA decode MLA benchmark fixture."""

import unittest

import jax.numpy as jnp
import numpy as np

from benchmark.kernels.mla.bench_dsa_decode_mla import (
    GLM_ATTENTION_SCALE,
    dense_full_context_mla_attention,
    make_benchmark_inputs,
)
from sgl_jax.srt.kernels.mla.dsa.reference import reference_dsa_decode_mla_attention


class TestDSADecodeMLABenchmarkInputs(unittest.TestCase):
    def test_fixture_uses_packed_cache_and_valid_selected_physical_slots(self):
        inputs = make_benchmark_inputs(
            batch_size=2,
            context_length=32,
            top_k=8,
            num_heads=3,
            latent_dim=128,
            rope_dim=64,
            page_size=16,
            slot_order="unsorted",
            seed=0,
        )

        self.assertEqual(inputs.ql_nope.shape, (2, 3, 128))
        self.assertEqual(inputs.q_pe.shape, (2, 3, 64))
        self.assertEqual(inputs.cache_kv.shape, (2, 8, 2, 256))
        self.assertEqual(inputs.topk_slots.shape, (2, 8))
        self.assertTrue(np.array_equal(inputs.valid_counts, np.array([8, 8], dtype=np.int32)))
        self.assertTrue(np.all(inputs.topk_slots >= 0))
        self.assertTrue(np.all(inputs.topk_slots < 32))
        self.assertFalse(np.array_equal(inputs.topk_slots[0], np.sort(inputs.topk_slots[0])))

    def test_glm_benchmark_uses_unabsorbed_qk_scale(self):
        self.assertEqual(GLM_ATTENTION_SCALE, 256**-0.5)

    def test_sorted_fixture_has_the_same_selected_slot_multiset(self):
        common_kwargs = dict(
            batch_size=2,
            context_length=64,
            top_k=16,
            num_heads=2,
            latent_dim=96,
            rope_dim=64,
            page_size=16,
            seed=1,
        )
        unsorted = make_benchmark_inputs(slot_order="unsorted", **common_kwargs)
        sorted_inputs = make_benchmark_inputs(slot_order="page-sorted", **common_kwargs)

        for batch_index in range(common_kwargs["batch_size"]):
            np.testing.assert_array_equal(
                np.sort(unsorted.topk_slots[batch_index]),
                sorted_inputs.topk_slots[batch_index],
            )

    def test_fixture_rejects_invalid_top_k_and_alignment(self):
        common_kwargs = dict(
            batch_size=1,
            context_length=32,
            top_k=8,
            num_heads=1,
            latent_dim=128,
            rope_dim=64,
            page_size=16,
            slot_order="unsorted",
        )
        with self.assertRaisesRegex(ValueError, "top_k"):
            make_benchmark_inputs(**{**common_kwargs, "top_k": 33})
        with self.assertRaisesRegex(ValueError, "page_size"):
            make_benchmark_inputs(**{**common_kwargs, "page_size": 12})

    def test_dense_baseline_matches_full_selected_reference_and_preserves_bf16(self):
        inputs = make_benchmark_inputs(
            batch_size=1,
            context_length=32,
            top_k=32,
            num_heads=2,
            latent_dim=128,
            rope_dim=64,
            page_size=16,
            slot_order="page-sorted",
            seed=2,
        )
        ql_nope = jnp.asarray(inputs.ql_nope, dtype=jnp.bfloat16)
        q_pe = jnp.asarray(inputs.q_pe, dtype=jnp.bfloat16)
        cache_kv = jnp.asarray(inputs.cache_kv, dtype=jnp.bfloat16)
        sm_scale = (128 + 64) ** -0.5

        actual = dense_full_context_mla_attention(
            ql_nope, q_pe, cache_kv, context_length=32, sm_scale=sm_scale
        )
        expected = reference_dsa_decode_mla_attention(
            ql_nope,
            q_pe,
            cache_kv,
            inputs.topk_slots,
            inputs.valid_counts,
            sm_scale=sm_scale,
        )

        self.assertEqual(actual.dtype, jnp.bfloat16)
        np.testing.assert_allclose(np.asarray(actual), expected, rtol=2e-2, atol=1e-2)
