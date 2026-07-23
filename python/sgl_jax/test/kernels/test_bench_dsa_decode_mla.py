"""Host-side contract tests for the DSA decode MLA benchmark fixture."""

import inspect
import sys
import unittest
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np

import benchmark.kernels.mla.bench_dsa_decode_mla as benchmark_module
from benchmark.kernels.mla.bench_dsa_decode_mla import (
    GLM_ATTENTION_SCALE,
    dense_full_context_mla_attention,
    make_benchmark_inputs,
)
from sgl_jax.srt.kernels.mla.dsa.reference import reference_dsa_decode_mla_attention


class TestDSADecodeMLABenchmarkInputs(unittest.TestCase):
    def test_timing_excludes_explicit_compile_and_reports_p95(self):
        clock = iter(
            [
                0,
                50_000_000,
                100_000_000,
                101_000_000,
                200_000_000,
                202_000_000,
            ]
        )
        calls = 0

        def compute():
            nonlocal calls
            calls += 1
            return object()

        with (
            mock.patch.object(
                benchmark_module.time,
                "perf_counter_ns",
                side_effect=lambda: next(clock),
            ),
            mock.patch.object(
                benchmark_module.jax,
                "block_until_ready",
                side_effect=lambda value: value,
            ),
        ):
            metrics = benchmark_module._time_compiled(compute, warmup_iters=0, iters=2)

        self.assertEqual(calls, 3)
        self.assertEqual(metrics["compile_ms"], 50.0)
        self.assertEqual(metrics["median_ms"], 1.5)
        self.assertAlmostEqual(metrics["p95_ms"], 1.95)
        self.assertAlmostEqual(metrics["p99_ms"], 1.99)

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
        sorted_slots = np.sort(inputs.topk_slots[0])
        self.assertTrue(
            all(
                not np.array_equal(inputs.topk_slots[0], np.roll(sorted_slots, shift))
                for shift in range(sorted_slots.size)
            ),
            "unsorted slots must be a permutation, not a rotated sequential scan",
        )

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

    def test_fixture_rejects_invalid_dimensions_and_alignment(self):
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
        with self.assertRaisesRegex(ValueError, "dimensions"):
            make_benchmark_inputs(**{**common_kwargs, "top_k": 0})
        with self.assertRaisesRegex(ValueError, "page_size"):
            make_benchmark_inputs(**{**common_kwargs, "page_size": 12})
        parameters = inspect.signature(make_benchmark_inputs).parameters
        self.assertNotIn("active_batch_size", parameters)
        self.assertNotIn("cache_capacity", parameters)
        self.assertNotIn("request_layout", parameters)

    def test_cli_defaults_to_one_head_single_device_decode_case(self):
        with mock.patch.object(sys, "argv", ["bench_dsa_decode_mla.py"]):
            args = benchmark_module._parse_args()

        self.assertEqual(args.batch_size, 1)
        self.assertEqual(args.num_heads, 1)
        self.assertEqual(args.context_length, 8192)
        self.assertEqual(args.top_k, 2048)
        self.assertEqual(args.variant, "sparse")

    def test_cli_accepts_local_head_count_without_encoding_a_mesh_size(self):
        with mock.patch.object(
            sys, "argv", ["bench_dsa_decode_mla.py", "--num-heads", "8"]
        ):
            args = benchmark_module._parse_args()

        self.assertEqual(args.num_heads, 8)

    def test_cli_rejects_non_production_feature_dimensions(self):
        invalid_arguments = (
            ("--latent-dim", "128"),
            ("--rope-dim", "128"),
            ("--page-size", "64"),
        )
        for flag, value in invalid_arguments:
            with (
                self.subTest(flag=flag),
                mock.patch.object(sys, "argv", ["bench_dsa_decode_mla.py", flag, value]),
                self.assertRaises(SystemExit),
            ):
                benchmark_module._parse_args()

    def test_device_selection_uses_one_local_device_without_count_requirement(self):
        devices = [object() for _ in range(32)]
        self.assertIs(benchmark_module._select_benchmark_device(devices), devices[0])

    def test_fixture_builds_causal_prefill_counts_with_static_kmax(self):
        inputs = make_benchmark_inputs(
            batch_size=128,
            context_length=128,
            top_k=2048,
            num_heads=2,
            latent_dim=512,
            rope_dim=64,
            page_size=128,
            slot_order="unsorted",
            valid_count_pattern="causal",
            start_position=0,
            seed=5,
        )

        self.assertEqual(inputs.topk_slots.shape, (128, 2048))
        self.assertEqual(inputs.valid_counts.tolist(), list(range(1, 129)))
        for row, count in enumerate(inputs.valid_counts):
            counted = inputs.topk_slots[row, :count]
            self.assertTrue(np.all(counted >= 0))
            self.assertTrue(np.all(counted <= row))
            self.assertTrue(np.all(inputs.topk_slots[row, count:] == 0))

    def test_fixture_builds_saturated_prefill_counts_after_topk_boundary(self):
        inputs = make_benchmark_inputs(
            batch_size=128,
            context_length=2176,
            top_k=2048,
            num_heads=2,
            latent_dim=512,
            rope_dim=64,
            page_size=128,
            slot_order="page-sorted",
            valid_count_pattern="causal",
            start_position=2048,
            seed=6,
        )

        self.assertEqual(inputs.valid_counts.tolist(), [2048] * 128)
        self.assertTrue(np.all(inputs.topk_slots[:, :2048] < 2176))

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

    def test_compiled_variants_receive_runtime_arrays_instead_of_closed_constants(self):
        self.assertTrue(
            hasattr(benchmark_module, "build_benchmark_variants"),
            "benchmark variants must be built as argument-taking JIT functions",
        )
        inputs = make_benchmark_inputs(
            batch_size=1,
            context_length=32,
            top_k=8,
            num_heads=2,
            latent_dim=128,
            rope_dim=64,
            page_size=16,
            slot_order="unsorted",
            seed=3,
        )
        ql_nope = jnp.asarray(inputs.ql_nope, dtype=jnp.bfloat16)
        q_pe = jnp.asarray(inputs.q_pe, dtype=jnp.bfloat16)
        cache_kv = jnp.asarray(inputs.cache_kv, dtype=jnp.bfloat16)

        variants = benchmark_module.build_benchmark_variants(
            context_length=32,
            sm_scale=GLM_ATTENTION_SCALE,
        )

        dense_jaxpr = jax.make_jaxpr(variants["dense"])(ql_nope, q_pe, cache_kv).jaxpr
        self.assertEqual(len(inspect.signature(variants["sparse"]).parameters), 5)
        self.assertEqual(len(dense_jaxpr.invars), 3)

        def collect_non_scalar_const_shapes(jaxpr):
            shapes = [var.aval.shape for var in jaxpr.constvars if getattr(var.aval, "shape", ())]
            for equation in jaxpr.eqns:
                for parameter in equation.params.values():
                    nested = getattr(parameter, "jaxpr", None)
                    if nested is not None:
                        shapes.extend(collect_non_scalar_const_shapes(nested))
            return shapes

        self.assertEqual(collect_non_scalar_const_shapes(dense_jaxpr), [])
