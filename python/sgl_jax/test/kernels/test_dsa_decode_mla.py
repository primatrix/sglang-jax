"""Tests for the host-side DSA decode MLA reference implementation."""

import unittest

import numpy as np

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

    def _assert_validation_error(self, inputs):
        for function in (reference_dsa_decode_mla_attention, dense_selected_mla_attention):
            with self.subTest(function=function.__name__):
                with self.assertRaises(ValueError):
                    function(*inputs)

    def test_selected_slot_reference_matches_dense_gather(self):
        """Physical slot gather agrees bit-for-bit with a dense-cache oracle."""
        inputs = self._inputs()
        reference = reference_dsa_decode_mla_attention(*inputs)
        dense = dense_selected_mla_attention(*inputs)

        self.assertEqual(reference.dtype, np.dtype(np.float32))
        np.testing.assert_array_equal(reference, dense)

    def test_invalid_shape_or_dtype_raises(self):
        ql_nope, q_pe, cache_kv, selected_slots, valid_counts = self._inputs()
        self._assert_validation_error(
            (ql_nope[..., None], q_pe, cache_kv, selected_slots, valid_counts)
        )
        self._assert_validation_error(
            (ql_nope.astype(np.int32), q_pe, cache_kv, selected_slots, valid_counts)
        )

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
        baseline = reference_dsa_decode_mla_attention(*self._inputs())
        ql_nope, q_pe, cache_kv, selected_slots, valid_counts = self._inputs()
        cache_kv.reshape(-1, cache_kv.shape[-1])[-1] = 1_000_000.0

        padded = reference_dsa_decode_mla_attention(
            ql_nope, q_pe, cache_kv, selected_slots, valid_counts
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
            ql_nope, q_pe, cache_kv, selected_slots, valid_counts
        )
        dense = dense_selected_mla_attention(ql_nope, q_pe, cache_kv, selected_slots, valid_counts)

        np.testing.assert_array_equal(reference, dense)
