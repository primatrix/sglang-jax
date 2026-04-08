"""
SWA (Sliding Window Attention) kernel-level benchmark.

Compares ragged_paged_attention kernel latency with and without sliding_window,
using MiMo-V2-Flash model configurations.

Usage:
  python benchmark/kernels/flash_attention/bench_swa_flashattention.py

MiMo-V2-Flash config:
  - FA layers (9): q_head_num=4, kv_head_num=2, head_dim=192 (k) / 128 (v)
  - SWA layers (39): q_head_num=4, kv_head_num=4, head_dim=192 (k) / 128 (v)
  - sliding_window=4096
  - page_size=128
"""

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


def benchmark_swa_kernel(
    mode,
    max_context_len,
    max_kv_cache_tokens,
    max_num_batched_tokens,
    q_head_num,
    kv_head_num,
    head_dim,
    page_size,
    sliding_window=None,
):
    """Benchmark ragged_paged_attention with optional sliding_window."""
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

    @functools.partial(jax.jit, static_argnames=["sm_scale", "sliding_window"])
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
        task=get_kernel_scope_name(best_bkv_p, best_bkv_p, page_size),
        tries=1,
    )
    avg_time = float(np.mean(times)) if times else float("nan")
    return avg_time


def run_swa_benchmark():
    """Run SWA vs FA kernel comparison benchmark."""
    page_size = 128
    max_kv_cache_tokens = 600000
    max_context_len = 40960  # max context for page_indices allocation

    # MiMo-V2-Flash model configs
    configs = {
        "FA (q=4,kv=2,d=128)": dict(q_head_num=4, kv_head_num=2, head_dim=128),
        "SWA (q=4,kv=4,d=128)": dict(q_head_num=4, kv_head_num=4, head_dim=128),
    }

    sliding_windows = [None, 4096, 8192, 16384]
    decode_batch_sizes = [1, 4, 8, 16, 32, 64, 128, 256]
    prefill_token_counts = [512, 1024, 2048, 4096]

    # === DECODE benchmark ===
    print("=" * 80)
    print("DECODE BENCHMARK: SWA kernel latency vs batch size")
    print("=" * 80)

    header = f"{'Config':<25} {'SW':>8} {'BS':>6}"
    header += f" {'Time(ms)':>10} {'us/token':>10}"
    print(header)
    print("-" * len(header))

    for config_name, cfg in configs.items():
        for sw in sliding_windows:
            sw_label = str(sw) if sw else "None"
            for bs in decode_batch_sizes:
                try:
                    t = benchmark_swa_kernel(
                        "decode",
                        max_context_len,
                        max_kv_cache_tokens,
                        bs,
                        cfg["q_head_num"],
                        cfg["kv_head_num"],
                        cfg["head_dim"],
                        page_size,
                        sliding_window=sw,
                    )
                    us_per_token = t * 1000 / bs  # ms -> us, divide by BS
                    print(
                        f"{config_name:<25} {sw_label:>8} {bs:>6}"
                        f" {t*1000:>10.3f} {us_per_token:>10.3f}"
                    )
                except Exception as e:
                    print(
                        f"{config_name:<25} {sw_label:>8} {bs:>6}" f" {'FAILED':>10} {str(e)[:30]}"
                    )

    # === PREFILL benchmark ===
    print()
    print("=" * 80)
    print("PREFILL BENCHMARK: SWA kernel latency vs token count")
    print("=" * 80)

    header = f"{'Config':<25} {'SW':>8} {'Tokens':>8}"
    header += f" {'Time(ms)':>10} {'us/token':>10}"
    print(header)
    print("-" * len(header))

    for config_name, cfg in configs.items():
        for sw in [None, 4096]:
            sw_label = str(sw) if sw else "None"
            for tokens in prefill_token_counts:
                try:
                    t = benchmark_swa_kernel(
                        "prefill",
                        max_context_len,
                        max_kv_cache_tokens,
                        tokens,
                        cfg["q_head_num"],
                        cfg["kv_head_num"],
                        cfg["head_dim"],
                        page_size,
                        sliding_window=sw,
                    )
                    us_per_token = t * 1000 / tokens
                    print(
                        f"{config_name:<25} {sw_label:>8} {tokens:>8}"
                        f" {t*1000:>10.3f} {us_per_token:>10.3f}"
                    )
                except Exception as e:
                    print(
                        f"{config_name:<25} {sw_label:>8} {tokens:>8}"
                        f" {'FAILED':>10} {str(e)[:30]}"
                    )

    # === SWA speedup analysis ===
    print()
    print("=" * 80)
    print("SWA SPEEDUP: sliding_window=4096 vs None (decode, SWA config)")
    print("=" * 80)

    header = f"{'BS':>6} {'No SW (ms)':>12} {'SW=4096 (ms)':>12} {'Speedup':>8}"
    print(header)
    print("-" * len(header))

    cfg = configs["SWA (q=4,kv=4,d=128)"]
    for bs in decode_batch_sizes:
        try:
            t_none = benchmark_swa_kernel(
                "decode",
                max_context_len,
                max_kv_cache_tokens,
                bs,
                cfg["q_head_num"],
                cfg["kv_head_num"],
                cfg["head_dim"],
                page_size,
                sliding_window=None,
            )
            t_sw = benchmark_swa_kernel(
                "decode",
                max_context_len,
                max_kv_cache_tokens,
                bs,
                cfg["q_head_num"],
                cfg["kv_head_num"],
                cfg["head_dim"],
                page_size,
                sliding_window=4096,
            )
            speedup = t_none / t_sw if t_sw > 0 else float("nan")
            print(f"{bs:>6} {t_none*1000:>12.3f} {t_sw*1000:>12.3f} {speedup:>7.2f}x")
        except Exception as e:
            print(f"{bs:>6} {'FAILED':>12} {str(e)[:30]}")


if __name__ == "__main__":
    print(f"JAX devices: {jax.device_count()} x {jax.devices()[0].platform}")
    print(f"Device: {jax.devices()[0].device_kind}")
    print()
    run_swa_benchmark()
