"""
  Usage:
  1. For test benchmark in ci
  SGLANG_JAX_IS_IN_CI=true python benchmark/kernels/flash_attention/bench_flashattention.py
  2. For generic benchmark results
  python benchmark/kernels/flash_attention/bench_flashattention.py
"""

import functools

import jax
import numpy as np
from utils import create_decode_uniform_data, create_prefill_uniform_data

from sgl_jax.srt.kernels.ragged_paged_attention.ragged_paged_attention import (
    get_kernel_scope_name,
    ragged_paged_attention,
)
from sgl_jax.srt.kernels.utils.perf import multiple_iteration_timeit_from_trace
from sgl_jax.test.test_utils import CustomTestCase, is_in_ci


def benchmark_backend(
    mode,
    max_context_len,
    max_kv_cache_tokens,
    max_num_batched_tokens,
    q_head_num,
    kv_head_num,
    head_dim,
    page_size,
):
    scale = head_dim**-0.5

    if mode == "prefill":
        (
            q,
            k,
            v,
            kv_cache,
            kv_lens,
            page_indices,
            cu_q_lens,
            cu_kv_lens,
            _,
            _,
            _,
            distribution,
        ) = create_prefill_uniform_data(
            max_context_len,
            max_kv_cache_tokens,
            max_num_batched_tokens,
            q_head_num,
            kv_head_num,
            head_dim,
            page_size=page_size,
        )
    elif mode == "decode":
        (
            q,
            k,
            v,
            kv_cache,
            kv_lens,
            page_indices,
            cu_q_lens,
            cu_kv_lens,
            _,
            _,
            _,
            distribution,
        ) = create_decode_uniform_data(
            max_context_len,
            max_kv_cache_tokens,
            max_num_batched_tokens,
            q_head_num,
            kv_head_num,
            head_dim,
            page_size=page_size,
        )
    else:
        raise ValueError(f"Invalid mode: {mode=}")

    @functools.partial(
        jax.jit,
        static_argnames=["sm_scale"],
    )
    def jitted_attn(
        q,
        k,
        v,
        kv_cache,
        kv_lens,
        page_indices,
        cu_q_lens,
        cu_kv_lens,
        distribution,
        sm_scale,
    ):
        return ragged_paged_attention(
            q,
            k,
            v,
            kv_cache,
            kv_lens,
            page_indices,
            cu_q_lens,
            cu_kv_lens,
            distribution,
            custom_mask=None,
            causal=1,
            sm_scale=sm_scale,
        )

    attn = functools.partial(
        jitted_attn,
        q,
        k,
        v,
        kv_cache,
        kv_lens,
        page_indices,
        cu_q_lens,
        cu_kv_lens,
        distribution,
        scale,
    )

    # Warmup
    output = attn()
    jax.block_until_ready(output)

    # Benchmark
    times = multiple_iteration_timeit_from_trace(
        compute_func=lambda: attn(),
        data_generator=lambda: (),
        task=get_kernel_scope_name(16, 2, page_size),
        tries=3,
    )
    avg_time = float(np.mean(times)) if times else float("nan")

    # cal num_q_heads_per_blk, num_kv_heads_per_blk
    return avg_time


def full_benchmark():
    bench_modes = ["prefill", "decode"]
    page_size_config = [64, 128, 256]
    max_num_batched_tokens_config_for_decode = [
        1,
        2,
        4,
        8,
        16,
        32,
        64,
        128,
        256,
    ]
    max_num_batched_tokens_config_for_prefill = [
        512,
        1024,
        2048,
        4096,
        8192,
    ]
    q_head_num_config = [2, 4, 8, 16, 32, 64]
    kv_head_num_config = [2, 4, 8, 16, 32, 64]
    head_dim_config = [128]
    max_kv_cache_tokens_config = [600000]
    all_combinations = []
    config_of_modes = {}
    max_context_len = 40960
    for mode in bench_modes:
        for q_head_num in q_head_num_config:
            for kv_head_num in kv_head_num_config:
                for head_dim in head_dim_config:
                    for page_size in page_size_config:
                        for max_kv_cache_tokens in max_kv_cache_tokens_config:
                            if mode == "prefill":
                                max_num_batched_tokens_config = (
                                    max_num_batched_tokens_config_for_prefill
                                )
                            elif mode == "decode":
                                max_num_batched_tokens_config = (
                                    max_num_batched_tokens_config_for_decode
                                )

                            for max_num_batched_tokens in max_num_batched_tokens_config:
                                if q_head_num < kv_head_num or q_head_num % kv_head_num != 0:
                                    continue
                                all_combinations.append(
                                    (
                                        page_size,
                                        max_kv_cache_tokens,
                                        max_num_batched_tokens,
                                        q_head_num,
                                        kv_head_num,
                                        head_dim,
                                    )
                                )
        config_of_modes[mode] = all_combinations
        all_combinations = []

    for mode, configs in config_of_modes.items():
        print(f"[{mode.upper()}] BENCHMARK RESULTS SUMMARY")
        for _, (
            page_size,
            max_kv_cache_tokens,
            max_num_batched_tokens,
            q_head_num,
            kv_head_num,
            head_dim,
        ) in enumerate(configs):
            print(
                f"Config: q_head_num={q_head_num}, kv_head_num={kv_head_num}, head_dim={head_dim=}, page_size={page_size}, max_num_batched_tokens={max_num_batched_tokens}"
            )
            try:
                flash_time = benchmark_backend(
                    mode,
                    max_context_len,
                    max_kv_cache_tokens,
                    max_num_batched_tokens,
                    q_head_num,
                    kv_head_num,
                    head_dim,
                    page_size,
                )
            except Exception as e:
                raise ValueError(f"run failed: {e=}")

            print(f"cost: {flash_time * 1000}ms")


class TestPerformance(CustomTestCase):
    def test_ragged_paged_attention_performance(self, floating_threshold: int = 0.1):
        """
        Args:
            floating_threshold: the ratio of expected results
        """
        # Key: (mode, page_size, max_num_batched_tokens, q_head_num, kv_head_num, head_dim, max_kv_cache_tokens)
        # Value: expected cost-time (baseline) in ms
        test_cases = {
            ("prefill", 128, 1024, 4, 1, 128, 600000): 0.740023,
            ("prefill", 128, 1024, 4, 2, 128, 600000): 1.187553,
            ("prefill", 128, 1024, 8, 1, 128, 600000): 0.723803,
            ("prefill", 128, 1024, 8, 4, 128, 600000): 2.0092227,
            ("prefill", 128, 4096, 4, 1, 128, 600000): 0.80139633,
            ("prefill", 128, 4096, 4, 2, 128, 600000): 1.257796,
            ("prefill", 128, 4096, 8, 1, 128, 600000): 0.863873,
            ("prefill", 128, 4096, 8, 4, 128, 600000): 2.216679,
            ("prefill", 256, 1024, 4, 1, 128, 600000): 0.709483,
            ("prefill", 256, 1024, 4, 2, 128, 600000): 1.140886,
            ("prefill", 256, 1024, 8, 1, 128, 600000): 0.75215633,
            ("prefill", 256, 1024, 8, 4, 128, 600000): 1.9960757,
            ("prefill", 256, 4096, 4, 1, 128, 600000): 0.800553,
            ("prefill", 256, 4096, 4, 2, 128, 600000): 1.249716,
            ("prefill", 256, 4096, 8, 1, 128, 600000): 0.867703,
            ("prefill", 256, 4096, 8, 4, 128, 600000): 2.2203957,
            ("decode", 128, 128, 4, 1, 128, 600000): 0.925433,
            ("decode", 128, 128, 4, 2, 128, 600000): 1.431036,
            ("decode", 128, 128, 8, 1, 128, 600000): 0.92405633,
            ("decode", 128, 128, 8, 4, 128, 600000): 2.493542,
            ("decode", 128, 256, 4, 1, 128, 600000): 1.1431197,
            ("decode", 128, 256, 4, 2, 128, 600000): 1.735556,
            ("decode", 128, 256, 8, 1, 128, 600000): 1.137416,
            ("decode", 128, 256, 8, 4, 128, 600000): 3.010292,
            ("decode", 256, 128, 4, 1, 128, 600000): 0.87172933,
            ("decode", 256, 128, 4, 2, 128, 600000): 1.3990127,
            ("decode", 256, 128, 8, 1, 128, 600000): 0.856623,
            ("decode", 256, 128, 8, 4, 128, 600000): 2.3763857,
            ("decode", 256, 256, 4, 1, 128, 600000): 1.0207963,
            ("decode", 256, 256, 4, 2, 128, 600000): 1.6457827,
            ("decode", 256, 256, 8, 1, 128, 600000): 1.032563,
            ("decode", 256, 256, 8, 4, 128, 600000): 2.7910553,
        }
        max_context_len = 40960
        for case, baseline in test_cases.items():
            (
                mode,
                page_size,
                max_num_batched_tokens,
                q_head_num,
                kv_head_num,
                head_dim,
                max_kv_cache_tokens,
            ) = case
            res = benchmark_backend(
                mode,
                max_context_len,
                max_kv_cache_tokens,
                max_num_batched_tokens,
                q_head_num,
                kv_head_num,
                head_dim,
                page_size,
            )
            expected_result = baseline * (1 + floating_threshold)
            print(f"{case}, res={res:.4}ms, expected_result={expected_result:.4}ms")
            self.assertLess(
                res,
                expected_result,
                f"Run ragged_paged_attention performance test failed, {case=}",
            )


if __name__ == "__main__":
    if is_in_ci():
        print("Run Ragged Paged Attention Performance Test...")
        TestPerformance().test_ragged_paged_attention_performance()
    else:
        print("Run Ragged Paged Attention Full Benchmark...")
        full_benchmark()
