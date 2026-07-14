"""Tests for the host-side DSA decode MLA reference implementation."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.kernels.mla.dsa.kernel import dsa_decode_mla_attention
from sgl_jax.srt.kernels.mla.dsa.reference import (
    dense_selected_mla_attention,
    reference_dsa_decode_mla_attention,
)


class TestDSADecodeMLAReference(unittest.TestCase):
    """Exercise selected-slot gather semantics independently of a kernel."""

    def setUp(self):
        rng = np.random.default_rng(0)
        self.cache_kv = rng.standard_normal((3, 8, 1, 256), dtype=np.float32)
        self.ql_nope = rng.standard_normal((2, 2, 128), dtype=np.float32)
        self.q_pe = rng.standard_normal((2, 2, 128), dtype=np.float32)
        self.selected_slots = np.array(
            [[0, 7, 8, 19, -1], [23, 1, 8, -1, -1]], dtype=np.int32
        )
        self.valid_counts = np.array([4, 3], dtype=np.int32)

    def _inputs(self):
        return (
            self.ql_nope.copy(),
            self.q_pe.copy(),
            self.cache_kv.copy(),
            self.selected_slots.copy(),
            self.valid_counts.copy(),
        )

    def _scaled_attention_expected(self, sm_scale):
        """Compute scaled selected-slot attention without either reference helper."""
        latent_width = self.ql_nope.shape[-1]
        dense_cache = self.cache_kv.reshape(-1, self.cache_kv.shape[-1])
        expected = np.empty(self.ql_nope.shape, dtype=np.float32)
        for batch_index, valid_count in enumerate(self.valid_counts):
            gathered = dense_cache[self.selected_slots[batch_index, :valid_count]]
            query = np.concatenate((self.ql_nope[batch_index], self.q_pe[batch_index]), axis=-1)
            logits = query @ gathered.T
            logits *= np.float32(sm_scale)
            logits -= np.max(logits, axis=-1, keepdims=True)
            probabilities = np.exp(logits)
            probabilities /= np.sum(probabilities, axis=-1, keepdims=True)
            expected[batch_index] = probabilities @ gathered[:, :latent_width]
        return expected

    def _assert_validation_error(self, inputs):
        for function in (reference_dsa_decode_mla_attention, dense_selected_mla_attention):
            with self.subTest(function=function.__name__), self.assertRaises(ValueError):
                function(*inputs, sm_scale=1.0)

    def test_selected_slot_reference_matches_dense_gather(self):
        """Physical slot gather agrees bit-for-bit with a dense-cache oracle."""
        inputs = self._inputs()
        reference = reference_dsa_decode_mla_attention(*inputs, sm_scale=1.0)
        dense = dense_selected_mla_attention(*inputs, sm_scale=1.0)

        self.assertEqual(reference.dtype, np.dtype(np.float32))
        np.testing.assert_array_equal(reference, dense)

    def test_sm_scale_changes_attention_and_matches_independent_expected_value(self):
        """A non-unit QK scale changes attention and matches a direct FP32 calculation."""
        sm_scale = 0.25
        expected = self._scaled_attention_expected(sm_scale)
        unit_scale = reference_dsa_decode_mla_attention(*self._inputs(), sm_scale=1.0)
        reference = reference_dsa_decode_mla_attention(*self._inputs(), sm_scale=sm_scale)
        dense = dense_selected_mla_attention(*self._inputs(), sm_scale=sm_scale)

        self.assertFalse(np.allclose(unit_scale, reference))
        np.testing.assert_allclose(reference, expected, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(dense, expected, rtol=1e-6, atol=1e-6)

    def test_invalid_shape_or_dtype_raises(self):
        ql_nope, q_pe, cache_kv, selected_slots, valid_counts = self._inputs()
        self._assert_validation_error(
            (ql_nope[..., None], q_pe, cache_kv, selected_slots, valid_counts)
        )
        self._assert_validation_error(
            (ql_nope.astype(np.int32), q_pe, cache_kv, selected_slots, valid_counts)
        )

    def test_bfloat16_inputs_are_converted_to_fp32(self):
        """Both references accept JAX BF16 inputs and calculate in host FP32."""
        ql_nope = jnp.asarray(self.ql_nope, dtype=jnp.bfloat16)
        q_pe = jnp.asarray(self.q_pe, dtype=jnp.bfloat16)
        cache_kv = jnp.asarray(self.cache_kv, dtype=jnp.bfloat16)
        expected = reference_dsa_decode_mla_attention(
            np.asarray(ql_nope, dtype=np.float32),
            np.asarray(q_pe, dtype=np.float32),
            np.asarray(cache_kv, dtype=np.float32),
            self.selected_slots,
            self.valid_counts,
            sm_scale=1.0,
        )

        for function in (reference_dsa_decode_mla_attention, dense_selected_mla_attention):
            with self.subTest(function=function.__name__):
                output = function(
                    ql_nope,
                    q_pe,
                    cache_kv,
                    self.selected_slots,
                    self.valid_counts,
                    sm_scale=1.0,
                )
                self.assertEqual(output.dtype, np.dtype(np.float32))
                np.testing.assert_array_equal(output, expected)

    def test_invalid_count_range_raises(self):
        ql_nope, q_pe, cache_kv, selected_slots, valid_counts = self._inputs()
        valid_counts[0] = selected_slots.shape[1] + 1
        self._assert_validation_error((ql_nope, q_pe, cache_kv, selected_slots, valid_counts))

    def test_zero_count_raises(self):
        ql_nope, q_pe, cache_kv, selected_slots, valid_counts = self._inputs()
        valid_counts[0] = 0
        self._assert_validation_error((ql_nope, q_pe, cache_kv, selected_slots, valid_counts))

    def test_negative_valid_slot_raises(self):
        ql_nope, q_pe, cache_kv, selected_slots, valid_counts = self._inputs()
        selected_slots[0, 0] = -1
        self._assert_validation_error((ql_nope, q_pe, cache_kv, selected_slots, valid_counts))

    def test_out_of_capacity_valid_slot_raises(self):
        ql_nope, q_pe, cache_kv, selected_slots, valid_counts = self._inputs()
        selected_slots[0, 0] = np.prod(cache_kv.shape[:3], dtype=np.int32)
        self._assert_validation_error((ql_nope, q_pe, cache_kv, selected_slots, valid_counts))

    def test_padding_does_not_contribute(self):
        """A padded -1 must not read the final physical slot for batch item zero."""
        baseline = reference_dsa_decode_mla_attention(*self._inputs(), sm_scale=1.0)
        ql_nope, q_pe, cache_kv, selected_slots, valid_counts = self._inputs()
        cache_kv.reshape(-1, cache_kv.shape[-1])[-1] = 1_000_000.0

        padded = reference_dsa_decode_mla_attention(
            ql_nope, q_pe, cache_kv, selected_slots, valid_counts, sm_scale=1.0
        )

        np.testing.assert_array_equal(baseline[0], padded[0])

    def test_duplicate_and_nonmonotonic_valid_slots_work(self):
        """Valid selected slots preserve caller order and permit duplicates."""
        ql_nope, q_pe, cache_kv, _selected_slots, _valid_counts = self._inputs()
        selected_slots = np.array(
            [[19, 0, 19, 7, -1], [8, 1, 8, -1, -1]], dtype=np.int32
        )
        valid_counts = np.array([4, 3], dtype=np.int32)

        reference = reference_dsa_decode_mla_attention(
            ql_nope, q_pe, cache_kv, selected_slots, valid_counts, sm_scale=1.0
        )
        dense = dense_selected_mla_attention(
            ql_nope, q_pe, cache_kv, selected_slots, valid_counts, sm_scale=1.0
        )

        np.testing.assert_array_equal(reference, dense)


class TestDSADecodeMLAPallas(TestDSADecodeMLAReference):
    """Exercise the Pallas DSA decode kernel in local interpret mode."""

    def test_interpret_matches_reference(self):
        ql_nope, q_pe, cache_kv, topk_slots, valid_counts = self._inputs()
        ql_nope = jnp.asarray(ql_nope, dtype=jnp.bfloat16)
        q_pe = jnp.asarray(q_pe, dtype=jnp.bfloat16)
        cache_kv = jnp.asarray(cache_kv, dtype=jnp.bfloat16)

        actual = dsa_decode_mla_attention(
            ql_nope,
            q_pe,
            cache_kv,
            jnp.asarray(topk_slots),
            jnp.asarray(valid_counts),
            sm_scale=0.25,
            interpret=True,
        )
        expected = reference_dsa_decode_mla_attention(
            ql_nope, q_pe, cache_kv, topk_slots, valid_counts, sm_scale=0.25
        )

        self.assertEqual(actual.dtype, jnp.bfloat16)
        np.testing.assert_allclose(np.asarray(actual), expected, rtol=2e-2, atol=1e-2)

    def test_interpret_handles_page_sizes_duplicates_and_padding(self):
        rng = np.random.default_rng(1)
        ql_nope = rng.standard_normal((2, 2, 128), dtype=np.float32)
        q_pe = rng.standard_normal((2, 2, 128), dtype=np.float32)

        for page_size in (8, 16, 32, 64):
            with self.subTest(page_size=page_size):
                cache_kv = rng.standard_normal((2, page_size, 1, 256), dtype=np.float32)
                # A padded -1 would select this large value if it were read.
                cache_kv[-1, -1, 0] = 10_000.0
                topk_slots = np.array(
                    [
                        [page_size + 1, 0, page_size + 1, page_size - 1, -1],
                        [page_size + 2, 1, page_size + 2, -1, -1],
                    ],
                    dtype=np.int32,
                )
                valid_counts = np.array([4, 3], dtype=np.int32)
                expected = reference_dsa_decode_mla_attention(
                    ql_nope, q_pe, cache_kv, topk_slots, valid_counts, sm_scale=0.25
                )

                actual = dsa_decode_mla_attention(
                    jnp.asarray(ql_nope, dtype=jnp.bfloat16),
                    jnp.asarray(q_pe, dtype=jnp.bfloat16),
                    jnp.asarray(cache_kv, dtype=jnp.bfloat16),
                    jnp.asarray(topk_slots),
                    jnp.asarray(valid_counts),
                    sm_scale=0.25,
                    interpret=True,
                )

                np.testing.assert_allclose(np.asarray(actual), expected, rtol=2e-2, atol=1e-2)

    def test_interpret_pads_latent_and_rope_dimensions_independently(self):
        rng = np.random.default_rng(2)
        ql_nope = rng.standard_normal((2, 2, 96), dtype=np.float32)
        q_pe = rng.standard_normal((2, 2, 64), dtype=np.float32)
        cache_kv = rng.standard_normal((2, 8, 2, 256), dtype=np.float32)
        topk_slots = np.array([[17, 2, 31, -1], [16, 0, -1, -1]], dtype=np.int32)
        valid_counts = np.array([3, 2], dtype=np.int32)

        actual = dsa_decode_mla_attention(
            jnp.asarray(ql_nope, dtype=jnp.bfloat16),
            jnp.asarray(q_pe, dtype=jnp.bfloat16),
            jnp.asarray(cache_kv, dtype=jnp.bfloat16),
            jnp.asarray(topk_slots),
            jnp.asarray(valid_counts),
            sm_scale=0.25,
            interpret=True,
        )
        expected = reference_dsa_decode_mla_attention(
            ql_nope, q_pe, cache_kv, topk_slots, valid_counts, sm_scale=0.25
        )

        self.assertEqual(actual.shape, ql_nope.shape)
        self.assertEqual(actual.dtype, jnp.bfloat16)
        np.testing.assert_allclose(np.asarray(actual), expected, rtol=2e-2, atol=1e-2)

    def test_interpret_2048_selected_slots_matches_reference(self):
        """A fixed K=2048 must not unroll one Pallas conditional per slot."""
        rng = np.random.default_rng(3)
        ql_nope = jnp.asarray(
            rng.standard_normal((1, 1, 128), dtype=np.float32), dtype=jnp.bfloat16
        )
        q_pe = jnp.asarray(
            rng.standard_normal((1, 1, 128), dtype=np.float32), dtype=jnp.bfloat16
        )
        cache_kv = jnp.asarray(
            rng.standard_normal((1, 2048, 1, 256), dtype=np.float32), dtype=jnp.bfloat16
        )
        topk_slots = jnp.asarray((np.arange(2048, dtype=np.int32) * 17) % 2048)[None, :]
        valid_counts = jnp.asarray([2048], dtype=jnp.int32)

        actual = dsa_decode_mla_attention(
            ql_nope,
            q_pe,
            cache_kv,
            topk_slots,
            valid_counts,
            sm_scale=0.01,
            interpret=True,
        )
        # NumPy's macOS BLAS can emit spurious FP warnings for this large,
        # finite FP32 matmul; the oracle result itself remains finite.
        with np.errstate(all="ignore"):
            expected = reference_dsa_decode_mla_attention(
                ql_nope, q_pe, cache_kv, topk_slots, valid_counts, sm_scale=0.01
            )

        self.assertTrue(np.isfinite(expected).all())
        np.testing.assert_allclose(np.asarray(actual), expected, rtol=2e-2, atol=1e-2)

    def test_unchecked_launch_is_jittable_in_interpret_mode(self):
        from sgl_jax.srt.kernels.mla.dsa import dsa_decode_mla_attention_unchecked

        ql_nope, q_pe, cache_kv, topk_slots, valid_counts = self._inputs()
        ql_nope = jnp.asarray(ql_nope, dtype=jnp.bfloat16)
        q_pe = jnp.asarray(q_pe, dtype=jnp.bfloat16)
        cache_kv = jnp.asarray(cache_kv, dtype=jnp.bfloat16)
        topk_slots = jnp.asarray(topk_slots)
        valid_counts = jnp.asarray(valid_counts)

        @jax.jit
        def launch(ql_nope, q_pe, cache_kv, topk_slots, valid_counts):
            return dsa_decode_mla_attention_unchecked(
                ql_nope,
                q_pe,
                cache_kv,
                topk_slots,
                valid_counts,
                sm_scale=0.25,
                interpret=True,
            )

        actual = launch(ql_nope, q_pe, cache_kv, topk_slots, valid_counts)
        expected = reference_dsa_decode_mla_attention(
            ql_nope, q_pe, cache_kv, topk_slots, valid_counts, sm_scale=0.25
        )

        np.testing.assert_allclose(np.asarray(actual), expected, rtol=2e-2, atol=1e-2)

    def test_validated_wrapper_rejects_invalid_inputs_before_dispatch(self):
        ql_nope, q_pe, cache_kv, topk_slots, valid_counts = self._inputs()
        invalid_cases = (
            ("shape", ql_nope[..., None], q_pe, cache_kv, topk_slots, valid_counts),
            ("dtype", ql_nope.astype(np.int32), q_pe, cache_kv, topk_slots, valid_counts),
            (
                "count",
                ql_nope,
                q_pe,
                cache_kv,
                topk_slots,
                np.array([topk_slots.shape[1] + 1, valid_counts[1]], dtype=np.int32),
            ),
            (
                "bounds",
                ql_nope,
                q_pe,
                cache_kv,
                np.array(
                    [[np.prod(cache_kv.shape[:3]), *topk_slots[0, 1:]], topk_slots[1]],
                    dtype=np.int32,
                ),
                valid_counts,
            ),
        )

        for name, *inputs in invalid_cases:
            with self.subTest(name=name), self.assertRaises(ValueError):
                dsa_decode_mla_attention(*inputs, sm_scale=0.25, interpret=True, validate=True)

    def test_tpu_non_interpret_matches_reference_with_dynamic_slots(self):
        if jax.default_backend() != "tpu":
            self.skipTest("interpret=False Pallas lowering requires a TPU")

        ql_nope, q_pe, cache_kv, _topk_slots, _valid_counts = self._inputs()
        topk_slots = np.array([[19, 0, 19, 7, -1], [8, 1, 8, -1, -1]], dtype=np.int32)
        valid_counts = np.array([4, 3], dtype=np.int32)
        ql_nope = jnp.asarray(ql_nope, dtype=jnp.bfloat16)
        q_pe = jnp.asarray(q_pe, dtype=jnp.bfloat16)
        cache_kv = jnp.asarray(cache_kv, dtype=jnp.bfloat16)

        actual = dsa_decode_mla_attention(
            ql_nope,
            q_pe,
            cache_kv,
            jnp.asarray(topk_slots),
            jnp.asarray(valid_counts),
            sm_scale=0.25,
            interpret=False,
        )
        expected = reference_dsa_decode_mla_attention(
            ql_nope, q_pe, cache_kv, topk_slots, valid_counts, sm_scale=0.25
        )

        np.testing.assert_allclose(np.asarray(actual), expected, rtol=2e-2, atol=1e-2)

    def test_tpu_non_interpret_glm_shape_2048_matches_reference(self):
        """Exercise the Falcon gate at the GLM MLA widths and DSA Top-K."""
        if jax.default_backend() != "tpu":
            self.skipTest("interpret=False Pallas lowering requires a TPU")

        rng = np.random.default_rng(4)
        batch_size = 1
        num_heads = 8  # TP-local head count for the single-host Falcon gate.
        latent_dim = 512
        rope_dim = 64
        top_k = 2048
        page_size = 128
        padded_cache_width = 512 + 128
        cache_kv = jnp.asarray(
            rng.standard_normal(
                (top_k // page_size, page_size, 1, padded_cache_width),
                dtype=np.float32,
            ),
            dtype=jnp.bfloat16,
        )
        ql_nope = jnp.asarray(
            rng.standard_normal((batch_size, num_heads, latent_dim), dtype=np.float32),
            dtype=jnp.bfloat16,
        )
        q_pe = jnp.asarray(
            rng.standard_normal((batch_size, num_heads, rope_dim), dtype=np.float32),
            dtype=jnp.bfloat16,
        )
        topk_slots = jnp.asarray(
            ((np.arange(top_k, dtype=np.int32) * 17) % top_k)[None, :]
        )
        valid_counts = jnp.asarray([top_k], dtype=jnp.int32)

        actual = dsa_decode_mla_attention(
            ql_nope,
            q_pe,
            cache_kv,
            topk_slots,
            valid_counts,
            sm_scale=(latent_dim + rope_dim) ** -0.5,
            interpret=False,
        )
        expected = reference_dsa_decode_mla_attention(
            ql_nope,
            q_pe,
            cache_kv,
            topk_slots,
            valid_counts,
            sm_scale=(latent_dim + rope_dim) ** -0.5,
        )

        self.assertTrue(np.isfinite(np.asarray(actual)).all())
        np.testing.assert_allclose(np.asarray(actual), expected, rtol=2e-2, atol=1e-2)

    def test_non_interpret_requires_tpu_on_cpu(self):
        if jax.default_backend() == "tpu":
            self.skipTest("the non-interpret TPU guard applies only on CPU")

        ql_nope, q_pe, cache_kv, topk_slots, valid_counts = self._inputs()
        with self.assertRaisesRegex(RuntimeError, "requires a TPU"):
            dsa_decode_mla_attention(
                jnp.asarray(ql_nope, dtype=jnp.bfloat16),
                jnp.asarray(q_pe, dtype=jnp.bfloat16),
                jnp.asarray(cache_kv, dtype=jnp.bfloat16),
                jnp.asarray(topk_slots),
                jnp.asarray(valid_counts),
                sm_scale=0.25,
            )
