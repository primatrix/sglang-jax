"""Tests for the host-side DSA decode MLA reference implementation."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.kernels.mla.dsa.attention import selected_mla_attention
from sgl_jax.srt.kernels.mla.dsa.gather import (
    _active_sparsecore_cores,
    _plan_sparsecore_pipeline,
    materialize_selected_kv_sparsecore,
    materialize_selected_kv_sparsecore_pipeline,
    materialize_selected_kv_xla,
    prepare_safe_topk_slots,
)
from sgl_jax.srt.kernels.mla.dsa.kernel import dsa_decode_mla_attention
from sgl_jax.srt.kernels.mla.dsa.reference import (
    dense_selected_mla_attention,
    reference_dsa_decode_mla_attention,
    reference_selected_mla_attention,
)


class TestDSASelectedKVGather(unittest.TestCase):
    def test_sparsecore_pipeline_plan_uses_each_worker_once(self):
        self.assertEqual(_active_sparsecore_cores(4), 2)
        self.assertEqual(
            _plan_sparsecore_pipeline(
                batch_size=1,
                padded_selected=2048,
                gather_block=128,
                available_cores=2,
                num_subcores=16,
            ),
            (1, 16, 1),
        )
        self.assertEqual(
            _plan_sparsecore_pipeline(
                batch_size=32,
                padded_selected=2048,
                gather_block=128,
                available_cores=2,
                num_subcores=16,
            ),
            (2, 32, 16),
        )

    def test_safe_slots_are_padded_and_invalid_entries_use_slot_zero(self):
        topk_slots = jnp.asarray([[5, 3, -1], [2, -1, -1]], dtype=jnp.int32)
        valid_counts = jnp.asarray([2, 1], dtype=jnp.int32)

        actual = prepare_safe_topk_slots(
            topk_slots, valid_counts, gather_block=8
        )

        expected = np.array(
            [[5, 3, 0, 0, 0, 0, 0, 0], [2, 0, 0, 0, 0, 0, 0, 0]],
            dtype=np.int32,
        )
        np.testing.assert_array_equal(np.asarray(actual), expected)

    def test_safe_slot_gather_block_is_validated(self):
        topk_slots = jnp.asarray([[0]], dtype=jnp.int32)
        valid_counts = jnp.asarray([1], dtype=jnp.int32)

        for gather_block in (0, -8, 3):
            with self.subTest(gather_block=gather_block), self.assertRaises(ValueError):
                prepare_safe_topk_slots(
                    topk_slots, valid_counts, gather_block=gather_block
                )

    def test_xla_materialization_matches_explicit_packed_cache_mapping(self):
        page_size = 8
        width = 256
        cache_kv = np.empty((2, page_size // 2, 2, width), dtype=np.float32)
        for physical_slot in range(2 * page_size):
            page, offset = divmod(physical_slot, page_size)
            packed_row, lane = divmod(offset, 2)
            cache_kv[page, packed_row, lane] = physical_slot

        topk_slots = jnp.asarray(
            [[9, 0, 7, 9, -1], [15, 1, -1, -1, -1]], dtype=jnp.int32
        )
        valid_counts = jnp.asarray([4, 2], dtype=jnp.int32)

        actual = materialize_selected_kv_xla(
            jnp.asarray(cache_kv, dtype=jnp.bfloat16),
            topk_slots,
            valid_counts,
            gather_block=8,
        )

        expected_slots = np.array(
            [[9, 0, 7, 9, 0, 0, 0, 0], [15, 1, 0, 0, 0, 0, 0, 0]],
            dtype=np.int32,
        )
        expected = np.broadcast_to(expected_slots[..., None], (2, 8, width))
        self.assertEqual(actual.dtype, jnp.bfloat16)
        np.testing.assert_array_equal(np.asarray(actual), expected)

    def test_xla_materialization_is_jittable(self):
        cache_kv = jnp.arange(2 * 4 * 2 * 128, dtype=jnp.bfloat16).reshape(
            2, 4, 2, 128
        )
        topk_slots = jnp.asarray([[15, 0, -1]], dtype=jnp.int32)
        valid_counts = jnp.asarray([2], dtype=jnp.int32)

        materialize = jax.jit(
            lambda cache, slots, counts: materialize_selected_kv_xla(
                cache, slots, counts, gather_block=8
            )
        )
        actual = materialize(cache_kv, topk_slots, valid_counts)

        expected = cache_kv.reshape(-1, 128)[jnp.asarray([15, 0, 0, 0, 0, 0, 0, 0])]
        np.testing.assert_array_equal(np.asarray(actual[0]), np.asarray(expected))


@unittest.skipUnless(
    jax.default_backend() == "tpu", "SparseCore Pallas lowering requires a TPU"
)
class TestDSASparseCoreGather(unittest.TestCase):
    def test_pipelined_sparsecore_gather_matches_xla(self):
        rng = np.random.default_rng(9)
        cache_kv = jnp.asarray(
            rng.standard_normal((4, 64, 2, 640), dtype=np.float32),
            dtype=jnp.bfloat16,
        )
        slots = (np.arange(2048, dtype=np.int32) * 127) % 512
        topk_slots = jnp.asarray(slots[None, :], dtype=jnp.int32)
        valid_counts = jnp.asarray([2048], dtype=jnp.int32)

        expected = materialize_selected_kv_xla(
            cache_kv, topk_slots, valid_counts, gather_block=128
        )
        actual = materialize_selected_kv_sparsecore_pipeline(
            cache_kv, topk_slots, valid_counts, gather_block=128
        )

        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))

    def test_sparsecore_gather_matches_xla_for_packed_bfloat16_cache(self):
        rng = np.random.default_rng(10)
        cache_kv = jnp.asarray(
            rng.standard_normal((3, 32, 2, 128), dtype=np.float32),
            dtype=jnp.bfloat16,
        )
        topk_slots = jnp.asarray(
            [[129, 0, 63, 129, -1], [191, 64, -1, -1, -1]],
            dtype=jnp.int32,
        )
        valid_counts = jnp.asarray([4, 2], dtype=jnp.int32)

        expected = materialize_selected_kv_xla(
            cache_kv, topk_slots, valid_counts, gather_block=128
        )
        actual = materialize_selected_kv_sparsecore(
            cache_kv, topk_slots, valid_counts, gather_block=128
        )

        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))

    def test_sparsecore_gather_matches_xla_at_glm_width_and_topk(self):
        rng = np.random.default_rng(11)
        page_size = 128
        context_length = 4096
        top_k = 2048
        cache_kv = jnp.asarray(
            rng.standard_normal(
                (context_length // page_size, page_size // 2, 2, 640),
                dtype=np.float32,
            ),
            dtype=jnp.bfloat16,
        )
        slots = (np.arange(top_k, dtype=np.int32) * 1543) % context_length
        topk_slots = jnp.asarray(slots[None, :], dtype=jnp.int32)
        valid_counts = jnp.asarray([top_k], dtype=jnp.int32)

        expected = materialize_selected_kv_xla(
            cache_kv, topk_slots, valid_counts, gather_block=128
        )
        actual = materialize_selected_kv_sparsecore(
            cache_kv, topk_slots, valid_counts, gather_block=128
        )

        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


class TestDSASelectedMLAAttention(unittest.TestCase):
    def _assert_selected_attention_matches_reference(
        self,
        ql_nope,
        q_pe,
        selected_kv,
        valid_counts,
        *,
        sm_scale,
    ):
        actual = selected_mla_attention(
            jnp.asarray(ql_nope, dtype=jnp.bfloat16),
            jnp.asarray(q_pe, dtype=jnp.bfloat16),
            jnp.asarray(selected_kv, dtype=jnp.bfloat16),
            jnp.asarray(valid_counts, dtype=jnp.int32),
            sm_scale=sm_scale,
            interpret=True,
        )
        # macOS Accelerate can emit spurious FP warnings for the large finite
        # GLM FP32 matmuls; explicit finiteness checks remain authoritative.
        with np.errstate(all="ignore"):
            expected = reference_selected_mla_attention(
                jnp.asarray(ql_nope, dtype=jnp.bfloat16),
                jnp.asarray(q_pe, dtype=jnp.bfloat16),
                jnp.asarray(selected_kv, dtype=jnp.bfloat16),
                np.asarray(valid_counts, dtype=np.int32),
                sm_scale=sm_scale,
            )

        self.assertEqual(actual.shape, np.shape(ql_nope))
        self.assertEqual(actual.dtype, jnp.bfloat16)
        self.assertTrue(np.isfinite(np.asarray(actual)).all())
        self.assertTrue(np.isfinite(expected).all())
        np.testing.assert_allclose(np.asarray(actual), expected, rtol=2e-2, atol=1e-2)

    def test_interpret_matches_materialized_selected_reference(self):
        rng = np.random.default_rng(20)
        ql_nope = rng.standard_normal((2, 2, 128), dtype=np.float32)
        q_pe = rng.standard_normal((2, 2, 128), dtype=np.float32)
        selected_kv = rng.standard_normal((2, 8, 256), dtype=np.float32)
        valid_counts = np.array([5, 3], dtype=np.int32)
        selected_kv[0, 5:] = 10_000.0
        selected_kv[1, 3:] = -10_000.0

        self._assert_selected_attention_matches_reference(
            ql_nope,
            q_pe,
            selected_kv,
            valid_counts,
            sm_scale=0.25,
        )

    def test_interpret_pads_latent_and_rope_independently(self):
        rng = np.random.default_rng(21)
        ql_nope = rng.standard_normal((1, 2, 96), dtype=np.float32)
        q_pe = rng.standard_normal((1, 2, 64), dtype=np.float32)
        selected_kv = rng.standard_normal((1, 8, 256), dtype=np.float32)
        valid_counts = np.array([7], dtype=np.int32)

        self._assert_selected_attention_matches_reference(
            ql_nope,
            q_pe,
            selected_kv,
            valid_counts,
            sm_scale=0.25,
        )

    def test_interpret_retains_duplicate_mass_and_is_permutation_invariant(self):
        rng = np.random.default_rng(22)
        ql_nope = rng.standard_normal((1, 2, 128), dtype=np.float32)
        q_pe = rng.standard_normal((1, 2, 128), dtype=np.float32)
        unique_kv = rng.standard_normal((4, 256), dtype=np.float32)
        selected_kv = np.stack(
            [unique_kv[2], unique_kv[0], unique_kv[2], unique_kv[1]], axis=0
        )[None, ...]
        permuted_kv = selected_kv[:, [3, 2, 0, 1]]
        valid_counts = np.array([4], dtype=np.int32)

        original = selected_mla_attention(
            jnp.asarray(ql_nope, dtype=jnp.bfloat16),
            jnp.asarray(q_pe, dtype=jnp.bfloat16),
            jnp.asarray(selected_kv, dtype=jnp.bfloat16),
            jnp.asarray(valid_counts),
            sm_scale=0.25,
            interpret=True,
        )
        permuted = selected_mla_attention(
            jnp.asarray(ql_nope, dtype=jnp.bfloat16),
            jnp.asarray(q_pe, dtype=jnp.bfloat16),
            jnp.asarray(permuted_kv, dtype=jnp.bfloat16),
            jnp.asarray(valid_counts),
            sm_scale=0.25,
            interpret=True,
        )

        np.testing.assert_allclose(
            np.asarray(original), np.asarray(permuted), rtol=2e-2, atol=1e-2
        )
        self._assert_selected_attention_matches_reference(
            ql_nope,
            q_pe,
            selected_kv,
            valid_counts,
            sm_scale=0.25,
        )

    def test_interpret_glm_shape_2048_matches_reference(self):
        rng = np.random.default_rng(23)
        ql_nope = rng.standard_normal((1, 8, 512), dtype=np.float32)
        q_pe = rng.standard_normal((1, 8, 64), dtype=np.float32)
        selected_kv = rng.standard_normal((1, 2048, 640), dtype=np.float32)
        valid_counts = np.array([2048], dtype=np.int32)

        self._assert_selected_attention_matches_reference(
            ql_nope,
            q_pe,
            selected_kv,
            valid_counts,
            sm_scale=256**-0.5,
        )


@unittest.skipUnless(
    jax.default_backend() == "tpu", "TensorCore Pallas lowering requires a TPU"
)
class TestDSASelectedMLAAttentionTPU(unittest.TestCase):
    def _assert_tpu_case(self, *, latent_dim, rope_dim, top_k, seed):
        rng = np.random.default_rng(seed)
        ql_nope = jnp.asarray(
            rng.standard_normal((1, 8, latent_dim), dtype=np.float32),
            dtype=jnp.bfloat16,
        )
        q_pe = jnp.asarray(
            rng.standard_normal((1, 8, rope_dim), dtype=np.float32),
            dtype=jnp.bfloat16,
        )
        selected_kv = jnp.asarray(
            rng.standard_normal(
                (
                    1,
                    top_k,
                    ((latent_dim + 127) // 128) * 128
                    + ((rope_dim + 127) // 128) * 128,
                ),
                dtype=np.float32,
            ),
            dtype=jnp.bfloat16,
        )
        valid_counts = jnp.asarray([top_k], dtype=jnp.int32)

        actual = selected_mla_attention(
            ql_nope,
            q_pe,
            selected_kv,
            valid_counts,
            sm_scale=256**-0.5,
            interpret=False,
        )
        expected = reference_selected_mla_attention(
            ql_nope,
            q_pe,
            selected_kv,
            np.asarray(valid_counts),
            sm_scale=256**-0.5,
        )

        np.testing.assert_allclose(np.asarray(actual), expected, rtol=2e-2, atol=1e-2)

    def test_tpu_selected_attention_small_aligned_case(self):
        self._assert_tpu_case(latent_dim=128, rope_dim=128, top_k=128, seed=24)

    def test_tpu_selected_attention_glm_shape(self):
        self._assert_tpu_case(latent_dim=512, rope_dim=64, top_k=2048, seed=25)


class TestDSADecodeMLAReference(unittest.TestCase):
    """Exercise selected-slot gather semantics independently of a kernel."""

    def setUp(self):
        rng = np.random.default_rng(0)
        self.cache_kv = rng.standard_normal((3, 4, 2, 256), dtype=np.float32)
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

    def test_reference_decodes_production_bfloat16_packing_two(self):
        """Physical slots address page, packed row, then BF16 lane."""
        page_size = 8
        cache_kv = np.zeros((2, page_size // 2, 2, 256), dtype=np.float32)
        for physical_slot in range(2 * page_size):
            page, offset = divmod(physical_slot, page_size)
            packed_row, lane = divmod(offset, 2)
            cache_kv[page, packed_row, lane, :128] = physical_slot

        ql_nope = np.zeros((1, 1, 128), dtype=np.float32)
        q_pe = np.zeros((1, 1, 128), dtype=np.float32)
        selected_slots = np.array([[1, 6, 7, 8, 15]], dtype=np.int32)
        valid_counts = np.array([5], dtype=np.int32)

        actual = reference_dsa_decode_mla_attention(
            ql_nope,
            q_pe,
            cache_kv,
            selected_slots,
            valid_counts,
            sm_scale=256**-0.5,
        )
        dense = dense_selected_mla_attention(
            ql_nope,
            q_pe,
            cache_kv,
            selected_slots,
            valid_counts,
            sm_scale=256**-0.5,
        )

        expected_value = np.mean(selected_slots, dtype=np.float32)
        np.testing.assert_array_equal(actual, dense)
        np.testing.assert_array_equal(actual, np.full_like(actual, expected_value))


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
            gather_impl="xla",
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
                cache_kv = rng.standard_normal(
                    (2, page_size // 2, 2, 256), dtype=np.float32
                )
                # A padded -1 would select this large value if it were read.
                cache_kv[-1, -1, -1] = 10_000.0
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
            rng.standard_normal((1, 1024, 2, 256), dtype=np.float32), dtype=jnp.bfloat16
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

    def test_invalid_gather_implementation_is_rejected(self):
        ql_nope, q_pe, cache_kv, topk_slots, valid_counts = self._inputs()
        with self.assertRaisesRegex(ValueError, "gather_impl"):
            dsa_decode_mla_attention(
                ql_nope,
                q_pe,
                cache_kv,
                topk_slots,
                valid_counts,
                sm_scale=0.25,
                interpret=True,
                gather_impl="unknown",
            )

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

    def test_tpu_pipeline_glm_shape_2048_matches_reference(self):
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
                (top_k // page_size, page_size // 2, 2, padded_cache_width),
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
            sm_scale=256**-0.5,
            interpret=False,
            gather_impl="sparsecore-pipeline",
        )
        expected = reference_dsa_decode_mla_attention(
            ql_nope,
            q_pe,
            cache_kv,
            topk_slots,
            valid_counts,
            sm_scale=256**-0.5,
        )

        self.assertTrue(np.isfinite(np.asarray(actual)).all())
        np.testing.assert_allclose(np.asarray(actual), expected, rtol=2e-2, atol=1e-2)

    def test_tpu_pipeline_composed_batch_32_matches_reference(self):
        if jax.default_backend() != "tpu":
            self.skipTest("interpret=False Pallas lowering requires a TPU")

        rng = np.random.default_rng(5)
        batch_size = 32
        num_heads = 8
        latent_dim = rope_dim = 128
        top_k = 128
        page_size = 128
        context_length = 256
        cache_kv = jnp.asarray(
            rng.standard_normal(
                (context_length // page_size, page_size // 2, 2, 256),
                dtype=np.float32,
            ),
            dtype=jnp.bfloat16,
        )
        ql_nope = jnp.asarray(
            rng.standard_normal(
                (batch_size, num_heads, latent_dim), dtype=np.float32
            ),
            dtype=jnp.bfloat16,
        )
        q_pe = jnp.asarray(
            rng.standard_normal(
                (batch_size, num_heads, rope_dim), dtype=np.float32
            ),
            dtype=jnp.bfloat16,
        )
        base_slots = np.arange(top_k, dtype=np.int32) * 2
        topk_slots = jnp.asarray(
            np.stack(
                [np.roll(base_slots, batch_index) for batch_index in range(batch_size)]
            ),
            dtype=jnp.int32,
        )
        valid_counts = jnp.full((batch_size,), top_k, dtype=jnp.int32)

        actual = dsa_decode_mla_attention(
            ql_nope,
            q_pe,
            cache_kv,
            topk_slots,
            valid_counts,
            sm_scale=256**-0.5,
            interpret=False,
            gather_impl="sparsecore-pipeline",
        )
        expected = reference_dsa_decode_mla_attention(
            ql_nope,
            q_pe,
            cache_kv,
            np.asarray(topk_slots),
            np.asarray(valid_counts),
            sm_scale=256**-0.5,
        )

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
