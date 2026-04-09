"""
  Usage:
  1. For test benchmark in ci
  SGLANG_JAX_IS_IN_CI=true python benchmark/kernels/flash_attention/bench_flashattention.py
  2. For generic benchmark results (full sweep)
  python benchmark/kernels/flash_attention/bench_flashattention.py
  3. For custom benchmark with specific parameters
  python benchmark/kernels/flash_attention/bench_flashattention.py \
    --mode decode --q-head-num 4 --kv-head-num 4 --head-dim 128 \
    --page-size 128 --batch-sizes 1,4,16,64,128,256 \
    --sliding-windows 0,4096
"""

import argparse
import functools

import jax
import numpy as np
from utils import create_decode_uniform_data, create_prefill_uniform_data

from sgl_jax.srt.kernels.ragged_paged_attention.ragged_paged_attention import (
    get_kernel_scope_name,
    ragged_paged_attention,
)
from sgl_jax.srt.kernels.ragged_paged_attention.tuned_block_sizes import (
    get_tuned_block_sizes,
)
from sgl_jax.srt.kernels.utils.perf import multiple_iteration_timeit_from_trace
from sgl_jax.srt.utils.jax_utils import get_device_name
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
    sliding_window=None,
    kv_len=None,
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
            kv_len=kv_len,
        )
    else:
        raise ValueError(f"Invalid mode: {mode=}")

    @functools.partial(
        jax.jit,
        static_argnames=["sm_scale", "sliding_window"],
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
        sliding_window,
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
            decode_mode=0,
            causal=1,
            sm_scale=sm_scale,
            sliding_window=sliding_window,
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
        sliding_window,
    )

    # Warmup
    output = attn()
    jax.block_until_ready(output)

    # Benchmark
    best_bkv_p, best_bq_sz = get_tuned_block_sizes(
        q.dtype,
        k.dtype,
        q_head_num,
        kv_head_num,
        head_dim,
        page_size,
        max_num_batched_tokens,
        page_indices.shape[0] // kv_lens.shape[0],
        True,
    )
    times = multiple_iteration_timeit_from_trace(
        compute_func=lambda: attn(),
        data_generator=lambda: (),
        task=get_kernel_scope_name(best_bq_sz, best_bkv_p, page_size),
        tries=1,
    )
    avg_time = float(np.mean(times)) if times else float("nan")

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


def custom_benchmark(args):
    """Run benchmark with user-specified parameters via argparse."""
    print(f"JAX devices: {jax.device_count()} x {jax.devices()[0].platform}")
    print(f"Device: {jax.devices()[0].device_kind}")
    print()

    modes = args.mode
    q_head_nums = args.q_head_num
    kv_head_nums = args.kv_head_num
    head_dims = args.head_dim
    page_sizes = args.page_size
    sliding_windows = [None if sw == 0 else sw for sw in args.sliding_windows]
    kv_lens = args.kv_len if args.kv_len else [None]

    max_context_len = args.max_context_len
    max_kv_cache_tokens = args.max_kv_cache_tokens

    for mode in modes:
        bs_list = args.batch_sizes if mode == "decode" else args.prefill_tokens
        print("=" * 90)
        print(f"[{mode.upper()}] BENCHMARK RESULTS")
        print("=" * 90)
        has_kv_len = mode == "decode" and any(kl is not None for kl in kv_lens)
        header = (
            f"{'q_h':>4} {'kv_h':>4} {'dim':>4} {'ps':>4} "
            f"{'SW':>8} "
            + (f"{'kv_len':>8} " if has_kv_len else "")
            + f"{'BS/Tok':>8} {'Time(ms)':>10} {'us/token':>10}"
        )
        print(header)
        print("-" * len(header))

        for q_h in q_head_nums:
            for kv_h in kv_head_nums:
                if q_h < kv_h or q_h % kv_h != 0:
                    continue
                for hd in head_dims:
                    for ps in page_sizes:
                        for sw in sliding_windows:
                            sw_label = str(sw) if sw else "None"
                            for kl in kv_lens if mode == "decode" else [None]:
                                kl_label = str(kl) if kl else "rand"
                                for bs in bs_list:
                                    try:
                                        t = benchmark_backend(
                                            mode,
                                            max_context_len,
                                            max_kv_cache_tokens,
                                            bs,
                                            q_h,
                                            kv_h,
                                            hd,
                                            ps,
                                            sliding_window=sw,
                                            kv_len=kl,
                                        )
                                        us_per_token = t * 1000 / bs
                                        line = (
                                            f"{q_h:>4} {kv_h:>4} {hd:>4} {ps:>4} " f"{sw_label:>8} "
                                        )
                                        if has_kv_len:
                                            line += f"{kl_label:>8} "
                                        line += f"{bs:>8} {t * 1000:>10.3f} {us_per_token:>10.3f}"
                                        print(line)
                                    except Exception as e:
                                        line = (
                                            f"{q_h:>4} {kv_h:>4} {hd:>4} {ps:>4} " f"{sw_label:>8} "
                                        )
                                        if has_kv_len:
                                            line += f"{kl_label:>8} "
                                        line += f"{bs:>8} {'FAILED':>10} {str(e)[:30]}"
                                        print(line)
        print()


def parse_int_list(s):
    """Parse comma-separated int list: '1,4,16,64' -> [1, 4, 16, 64]."""
    return [int(x.strip()) for x in s.split(",")]


def build_parser():
    parser = argparse.ArgumentParser(
        description="Benchmark ragged_paged_attention kernel with configurable parameters."
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="decode",
        help="Comma-separated modes: decode,prefill (default: decode)",
    )
    parser.add_argument(
        "--q-head-num",
        type=parse_int_list,
        default=[4],
        help="Comma-separated q head nums (default: 4)",
    )
    parser.add_argument(
        "--kv-head-num",
        type=parse_int_list,
        default=[2],
        help="Comma-separated kv head nums (default: 2)",
    )
    parser.add_argument(
        "--head-dim",
        type=parse_int_list,
        default=[128],
        help="Comma-separated head dims (default: 128)",
    )
    parser.add_argument(
        "--page-size",
        type=parse_int_list,
        default=[128],
        help="Comma-separated page sizes (default: 128)",
    )
    parser.add_argument(
        "--batch-sizes",
        type=parse_int_list,
        default=[1, 4, 8, 16, 32, 64, 128, 256],
        help="Comma-separated decode batch sizes (default: 1,4,8,16,32,64,128,256)",
    )
    parser.add_argument(
        "--prefill-tokens",
        type=parse_int_list,
        default=[512, 1024, 2048, 4096],
        help="Comma-separated prefill token counts (default: 512,1024,2048,4096)",
    )
    parser.add_argument(
        "--sliding-windows",
        type=parse_int_list,
        default=[0],
        help="Comma-separated sliding window sizes, 0=None (default: 0)",
    )
    parser.add_argument(
        "--max-context-len",
        type=int,
        default=40960,
        help="Max context length for page indices allocation (default: 40960)",
    )
    parser.add_argument(
        "--max-kv-cache-tokens",
        type=int,
        default=600000,
        help="Max KV cache tokens (default: 600000)",
    )
    parser.add_argument(
        "--kv-len",
        type=parse_int_list,
        default=None,
        help="Comma-separated fixed KV lengths for decode mode (default: random 1024-2048). "
        "Use to test sliding_window with long contexts, e.g. --kv-len 2048,8192,32768",
    )
    return parser


class TestPerformance(CustomTestCase):
    def test_ragged_paged_attention_performance(self, floating_threshold: int = 0.1):
        """
        Args:
            floating_threshold: the ratio of expected results
        """
        # Key: (mode, page_size, max_num_batched_tokens, q_head_num, kv_head_num, head_dim, max_kv_cache_tokens)
        # Value: expected cost-time (baseline) in ms
        test_cases_for_different_devices = {
            "TPU v6e": {
                ("prefill", 128, 1024, 4, 1, 128, 600000): 0.01574625,
                ("prefill", 128, 1024, 4, 2, 128, 600000): 0.01525625,
                ("prefill", 128, 1024, 8, 1, 128, 600000): 0.02189,
                ("prefill", 128, 1024, 8, 4, 128, 600000): 0.0239625,
                ("prefill", 128, 4096, 4, 1, 128, 600000): 0.09134625,
                ("prefill", 128, 4096, 4, 2, 128, 600000): 0.0915075,
                ("prefill", 128, 4096, 8, 1, 128, 600000): 0.14679375,
                ("prefill", 128, 4096, 8, 4, 128, 600000): 0.162105,
                ("prefill", 256, 1024, 4, 1, 128, 600000): 0.0157125,
                ("prefill", 256, 1024, 4, 2, 128, 600000): 0.01491875,
                ("prefill", 256, 1024, 8, 1, 128, 600000): 0.02186125,
                ("prefill", 256, 1024, 8, 4, 128, 600000): 0.0237575,
                ("prefill", 256, 4096, 4, 1, 128, 600000): 0.09128125,
                ("prefill", 256, 4096, 4, 2, 128, 600000): 0.0919325,
                ("prefill", 256, 4096, 8, 1, 128, 600000): 0.14679125,
                ("prefill", 256, 4096, 8, 4, 128, 600000): 0.16232,
                ("decode", 128, 128, 4, 1, 128, 600000): 0.2329225,
                ("decode", 128, 128, 4, 2, 128, 600000): 0.29882625,
                ("decode", 128, 128, 8, 1, 128, 600000): 0.23746875,
                ("decode", 128, 128, 8, 4, 128, 600000): 0.5295875,
                ("decode", 128, 256, 4, 1, 128, 600000): 0.4627675,
                ("decode", 128, 256, 4, 2, 128, 600000): 0.5923825,
                ("decode", 128, 256, 8, 1, 128, 600000): 0.4630275,
                ("decode", 128, 256, 8, 4, 128, 600000): 1.05854875,
                ("decode", 256, 128, 4, 1, 128, 600000): 0.16898125,
                ("decode", 256, 128, 4, 2, 128, 600000): 0.25190875,
                ("decode", 256, 128, 8, 1, 128, 600000): 0.16870875,
                ("decode", 256, 128, 8, 4, 128, 600000): 0.3997425,
                ("decode", 256, 256, 4, 1, 128, 600000): 0.33464,
                ("decode", 256, 256, 4, 2, 128, 600000): 0.51019875,
                ("decode", 256, 256, 8, 1, 128, 600000): 0.33216375,
                ("decode", 256, 256, 8, 4, 128, 600000): 0.81125875,
            },
            "TPU v7": {
                ("prefill", 128, 1024, 4, 1, 128, 600000): 0.014767107,
                ("prefill", 128, 1024, 4, 2, 128, 600000): 0.015102041,
                ("prefill", 128, 1024, 8, 1, 128, 600000): 0.022470588,
                ("prefill", 128, 1024, 8, 4, 128, 600000): 0.023110444,
                ("prefill", 128, 4096, 4, 1, 128, 600000): 0.0794994,
                ("prefill", 128, 4096, 4, 2, 128, 600000): 0.084321729,
                ("prefill", 128, 4096, 8, 1, 128, 600000): 0.145336134,
                ("prefill", 128, 4096, 8, 4, 128, 600000): 0.145752701,
                ("prefill", 256, 1024, 4, 1, 128, 600000): 0.014686675,
                ("prefill", 256, 1024, 4, 2, 128, 600000): 0.015236495,
                ("prefill", 256, 1024, 8, 1, 128, 600000): 0.022472989,
                ("prefill", 256, 1024, 8, 4, 128, 600000): 0.023372149,
                ("prefill", 256, 4096, 4, 1, 128, 600000): 0.079831933,
                ("prefill", 256, 4096, 4, 2, 128, 600000): 0.084252101,
                ("prefill", 256, 4096, 8, 1, 128, 600000): 0.144997599,
                ("prefill", 256, 4096, 8, 4, 128, 600000): 0.145212485,
                ("decode", 128, 128, 4, 1, 128, 600000): 0.15942617,
                ("decode", 128, 128, 4, 2, 128, 600000): 0.226237695,
                ("decode", 128, 128, 8, 1, 128, 600000): 0.159394958,
                ("decode", 128, 128, 8, 4, 128, 600000): 0.331831933,
                ("decode", 128, 256, 4, 1, 128, 600000): 0.313444178,
                ("decode", 128, 256, 4, 2, 128, 600000): 0.452889556,
                ("decode", 128, 256, 8, 1, 128, 600000): 0.313991597,
                ("decode", 128, 256, 8, 4, 128, 600000): 0.649698679,
                ("decode", 256, 128, 4, 1, 128, 600000): 0.15329892,
                ("decode", 256, 128, 4, 2, 128, 600000): 0.207255702,
                ("decode", 256, 128, 8, 1, 128, 600000): 0.152806723,
                ("decode", 256, 128, 8, 4, 128, 600000): 0.313996399,
                ("decode", 256, 256, 4, 1, 128, 600000): 0.303966387,
                ("decode", 256, 256, 4, 2, 128, 600000): 0.415367347,
                ("decode", 256, 256, 8, 1, 128, 600000): 0.303222089,
                ("decode", 256, 256, 8, 4, 128, 600000): 0.610410564,
            },
        }
        test_cases = test_cases_for_different_devices[get_device_name()]
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
        parser = build_parser()
        args = parser.parse_args()
        args.mode = [m.strip() for m in args.mode.split(",")]
        args.mode_type = args.mode[0]
        print("Run Ragged Paged Attention Benchmark...")
        custom_benchmark(args)
