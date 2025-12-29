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
            ("prefill", 128, 1024, 4, 4, 128, 600000): 0.20109666666666667,
            ("prefill", 128, 1024, 4, 2, 128, 600000): 0.13419916666666667,
            ("prefill", 128, 1024, 8, 8, 128, 600000): 0.33863875000000004,
            ("prefill", 128, 1024, 8, 4, 128, 600000): 0.20110458333333334,
            ("prefill", 128, 4096, 4, 4, 128, 600000): 1.5589241666666667,
            ("prefill", 128, 4096, 4, 2, 128, 600000): 1.0281283333333333,
            ("prefill", 128, 4096, 8, 8, 128, 600000): 2.65723875,
            ("prefill", 128, 4096, 8, 4, 128, 600000): 1.5589158333333335,
            ("decode", 128, 128, 4, 4, 128, 600000): 0.7600533333333334,
            ("decode", 128, 128, 4, 2, 128, 600000): 0.5717766666666667,
            ("decode", 128, 128, 8, 8, 128, 600000): 1.16352,
            ("decode", 128, 128, 8, 4, 128, 600000): 0.7622062500000001,
            ("decode", 128, 256, 4, 4, 128, 600000): 1.492135,
            ("decode", 128, 256, 4, 2, 128, 600000): 1.130425,
            ("decode", 128, 256, 8, 8, 128, 600000): 2.2918925,
            ("decode", 128, 256, 8, 4, 128, 600000): 1.4930079166666665,
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
