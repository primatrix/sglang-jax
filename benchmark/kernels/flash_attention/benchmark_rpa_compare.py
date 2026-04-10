"""Focused fused-vs-split RPA benchmark for the v6e comparison cases.

Run from the repo root with:

    export PYTHONPATH="$PWD/python:$PWD/benchmark/kernels/flash_attention"
    python benchmark/kernels/flash_attention/benchmark_rpa_compare.py

This reproduces the 8-case comparison used in the 2026-04-08 delivery note.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from utils import (
    create_kv_cache_data,
    create_page_indices_data,
    create_prefill_uniform_data,
    create_qkv_data,
    create_split_kv_cache_data,
    create_split_prefill_uniform_data,
    create_split_qkv_data,
)

from sgl_jax.srt.kernels.ragged_paged_attention.ragged_paged_attention import (
    get_kernel_scope_name,
    ragged_paged_attention,
)
from sgl_jax.srt.kernels.ragged_paged_attention.tuned_block_sizes import (
    get_tuned_block_sizes,
)
from sgl_jax.srt.kernels.utils.perf import multiple_iteration_timeit_from_trace

TRACE_ROOT = "/tmp/sgl_rpa_compare_trace"
CASES = [
    (
        "decode",
        "split_192_128",
        dict(
            max_context_len=4096,
            max_kv_cache_tokens=524288,
            max_num_batched_tokens=128,
            q_head_num=4,
            kv_head_num=2,
            k_head_dim=192,
            v_head_dim=128,
            page_size=64,
        ),
    ),
    (
        "decode",
        "split_256_128",
        dict(
            max_context_len=4096,
            max_kv_cache_tokens=524288,
            max_num_batched_tokens=128,
            q_head_num=4,
            kv_head_num=2,
            k_head_dim=256,
            v_head_dim=128,
            page_size=64,
        ),
    ),
    (
        "decode",
        "split_128_128",
        dict(
            max_context_len=4096,
            max_kv_cache_tokens=524288,
            max_num_batched_tokens=128,
            q_head_num=4,
            kv_head_num=2,
            k_head_dim=128,
            v_head_dim=128,
            page_size=64,
        ),
    ),
    (
        "decode",
        "fused_128",
        dict(
            max_context_len=4096,
            max_kv_cache_tokens=524288,
            max_num_batched_tokens=128,
            q_head_num=4,
            kv_head_num=2,
            head_dim=128,
            page_size=64,
        ),
    ),
    (
        "decode",
        "fused_256",
        dict(
            max_context_len=4096,
            max_kv_cache_tokens=524288,
            max_num_batched_tokens=128,
            q_head_num=4,
            kv_head_num=2,
            head_dim=256,
            page_size=64,
        ),
    ),
    (
        "prefill",
        "split_192_128",
        dict(
            max_context_len=4096,
            max_kv_cache_tokens=524288,
            max_num_batched_tokens=8192,
            q_head_num=4,
            kv_head_num=2,
            k_head_dim=192,
            v_head_dim=128,
            page_size=64,
        ),
    ),
    (
        "prefill",
        "split_256_128",
        dict(
            max_context_len=4096,
            max_kv_cache_tokens=524288,
            max_num_batched_tokens=8192,
            q_head_num=4,
            kv_head_num=2,
            k_head_dim=256,
            v_head_dim=128,
            page_size=64,
        ),
    ),
    (
        "prefill",
        "split_128_128",
        dict(
            max_context_len=4096,
            max_kv_cache_tokens=524288,
            max_num_batched_tokens=8192,
            q_head_num=4,
            kv_head_num=2,
            k_head_dim=128,
            v_head_dim=128,
            page_size=64,
        ),
    ),
    (
        "prefill",
        "fused_128",
        dict(
            max_context_len=4096,
            max_kv_cache_tokens=524288,
            max_num_batched_tokens=8192,
            q_head_num=4,
            kv_head_num=2,
            head_dim=128,
            page_size=64,
        ),
    ),
    (
        "prefill",
        "fused_256",
        dict(
            max_context_len=4096,
            max_kv_cache_tokens=524288,
            max_num_batched_tokens=8192,
            q_head_num=4,
            kv_head_num=2,
            head_dim=256,
            page_size=64,
        ),
    ),
]


def fixed_decode_fused_data(
    *,
    max_context_len: int,
    max_kv_cache_tokens: int,
    batch_size: int,
    q_head_num: int,
    kv_head_num: int,
    head_dim: int,
    page_size: int,
    dtype=jnp.bfloat16,
    seed: int = 42,
):
    seq_lens = jnp.full((batch_size,), max_context_len, dtype=jnp.int32)
    cu_q_lens = jnp.arange(batch_size + 1, dtype=jnp.int32)
    cu_kv_lens = jnp.arange(batch_size + 1, dtype=jnp.int32) * max_context_len
    q, k, v = create_qkv_data(batch_size, q_head_num, kv_head_num, head_dim, dtype, seed)
    kv_cache = create_kv_cache_data(
        max_kv_cache_tokens,
        kv_head_num,
        head_dim,
        page_size=page_size,
        dtype=dtype,
        seed=seed,
    )
    total_kv_lens = int(batch_size * max_context_len)
    page_indices, _ = create_page_indices_data(
        batch_size,
        total_kv_lens,
        seq_lens,
        max_context_len,
        page_size=page_size,
    )
    distribution = jnp.array([0, 0, batch_size], dtype=jnp.int32)
    return q, k, v, kv_cache, seq_lens, page_indices, cu_q_lens, cu_kv_lens, distribution


def fixed_decode_split_data(
    *,
    max_context_len: int,
    max_kv_cache_tokens: int,
    batch_size: int,
    q_head_num: int,
    kv_head_num: int,
    k_head_dim: int,
    v_head_dim: int,
    page_size: int,
    dtype=jnp.bfloat16,
    seed: int = 42,
):
    seq_lens = jnp.full((batch_size,), max_context_len, dtype=jnp.int32)
    cu_q_lens = jnp.arange(batch_size + 1, dtype=jnp.int32)
    cu_kv_lens = jnp.arange(batch_size + 1, dtype=jnp.int32) * max_context_len
    q, k, v = create_split_qkv_data(
        batch_size, q_head_num, kv_head_num, k_head_dim, v_head_dim, dtype, seed
    )
    k_cache, v_cache = create_split_kv_cache_data(
        max_kv_cache_tokens,
        kv_head_num,
        k_head_dim,
        v_head_dim,
        page_size=page_size,
        dtype=dtype,
        seed=seed,
    )
    total_kv_lens = int(batch_size * max_context_len)
    page_indices, _ = create_page_indices_data(
        batch_size,
        total_kv_lens,
        seq_lens,
        max_context_len,
        page_size=page_size,
    )
    distribution = jnp.array([0, 0, batch_size], dtype=jnp.int32)
    return q, k, v, k_cache, v_cache, seq_lens, page_indices, cu_q_lens, cu_kv_lens, distribution


def bench_fused(
    mode: str,
    *,
    max_context_len: int,
    max_kv_cache_tokens: int,
    max_num_batched_tokens: int,
    q_head_num: int,
    kv_head_num: int,
    head_dim: int,
    page_size: int,
    tries: int = 3,
) -> dict:
    scale = head_dim**-0.5
    if mode == "decode":
        data = fixed_decode_fused_data(
            max_context_len=max_context_len,
            max_kv_cache_tokens=max_kv_cache_tokens,
            batch_size=max_num_batched_tokens,
            q_head_num=q_head_num,
            kv_head_num=kv_head_num,
            head_dim=head_dim,
            page_size=page_size,
        )
    elif mode == "prefill":
        data = create_prefill_uniform_data(
            max_context_len,
            max_kv_cache_tokens,
            max_num_batched_tokens,
            q_head_num,
            kv_head_num,
            head_dim,
            page_size=page_size,
        )
        data = (data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[11])
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    q, k, v, kv_cache, kv_lens, page_indices, cu_q_lens, cu_kv_lens, distribution = data

    @functools.partial(jax.jit, static_argnames=["sm_scale"])
    def run(
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

    fn = functools.partial(
        run,
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
    jax.block_until_ready(fn())

    pages_per_seq = page_indices.shape[0] // kv_lens.shape[0]
    bkv_p, bq = get_tuned_block_sizes(
        q.dtype,
        kv_cache.dtype,
        q_head_num,
        kv_head_num,
        head_dim,
        page_size,
        q.shape[0],
        pages_per_seq,
        True,
        kernel_type="fused",
    )
    task = get_kernel_scope_name(bq, bkv_p, page_size)
    times = multiple_iteration_timeit_from_trace(
        lambda: fn(),
        lambda: (),
        task=task,
        tries=tries,
        trace_root=TRACE_ROOT,
    )
    return {
        "mode": mode,
        "kind": "fused",
        "head_dim": head_dim,
        "page_size": page_size,
        "max_context_len": max_context_len,
        "max_num_batched_tokens": max_num_batched_tokens,
        "q_head_num": q_head_num,
        "kv_head_num": kv_head_num,
        "bkv_p": int(bkv_p),
        "bq": int(bq),
        "times_ms": [float(x) for x in times],
        "avg_ms": float(np.mean(times)),
    }


def bench_split(
    mode: str,
    *,
    max_context_len: int,
    max_kv_cache_tokens: int,
    max_num_batched_tokens: int,
    q_head_num: int,
    kv_head_num: int,
    k_head_dim: int,
    v_head_dim: int,
    page_size: int,
    tries: int = 3,
) -> dict:
    scale = k_head_dim**-0.5
    if mode == "decode":
        data = fixed_decode_split_data(
            max_context_len=max_context_len,
            max_kv_cache_tokens=max_kv_cache_tokens,
            batch_size=max_num_batched_tokens,
            q_head_num=q_head_num,
            kv_head_num=kv_head_num,
            k_head_dim=k_head_dim,
            v_head_dim=v_head_dim,
            page_size=page_size,
        )
    elif mode == "prefill":
        data = create_split_prefill_uniform_data(
            max_context_len,
            max_kv_cache_tokens,
            max_num_batched_tokens,
            q_head_num,
            kv_head_num,
            k_head_dim,
            v_head_dim,
            page_size=page_size,
        )
        data = (data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8], data[12])
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    q, k, v, k_cache, v_cache, kv_lens, page_indices, cu_q_lens, cu_kv_lens, distribution = data

    @functools.partial(jax.jit, static_argnames=["sm_scale"])
    def run(
        q,
        k,
        v,
        k_cache,
        v_cache,
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
            None,
            kv_lens,
            page_indices,
            cu_q_lens,
            cu_kv_lens,
            distribution,
            custom_mask=None,
            causal=1,
            sm_scale=sm_scale,
            k_cache=k_cache,
            v_cache=v_cache,
        )

    fn = functools.partial(
        run,
        q,
        k,
        v,
        k_cache,
        v_cache,
        kv_lens,
        page_indices,
        cu_q_lens,
        cu_kv_lens,
        distribution,
        scale,
    )
    jax.block_until_ready(fn())

    pages_per_seq = page_indices.shape[0] // kv_lens.shape[0]
    aligned_head_dim = ((k_head_dim + 127) // 128) * 128
    bkv_p, bq = get_tuned_block_sizes(
        q.dtype,
        k_cache.dtype,
        q_head_num,
        kv_head_num,
        aligned_head_dim,
        page_size,
        q.shape[0],
        pages_per_seq,
        True,
        kernel_type="split",
    )
    task = get_kernel_scope_name(bq, bkv_p, page_size)
    times = multiple_iteration_timeit_from_trace(
        lambda: fn(),
        lambda: (),
        task=task,
        tries=tries,
        trace_root=TRACE_ROOT,
    )
    return {
        "mode": mode,
        "kind": "split",
        "k_head_dim": k_head_dim,
        "v_head_dim": v_head_dim,
        "page_size": page_size,
        "max_context_len": max_context_len,
        "max_num_batched_tokens": max_num_batched_tokens,
        "q_head_num": q_head_num,
        "kv_head_num": kv_head_num,
        "bkv_p": int(bkv_p),
        "bq": int(bq),
        "times_ms": [float(x) for x in times],
        "avg_ms": float(np.mean(times)),
    }


def main() -> None:
    print("cwd", Path.cwd())
    print("devices", jax.devices())

    results = []
    for mode, name, kwargs in CASES:
        print(f"RUN {name} {mode} ...", flush=True)
        if name.startswith("split"):
            out = bench_split(mode, **kwargs)
        else:
            out = bench_fused(mode, **kwargs)
        out["name"] = name
        results.append(out)
        print(json.dumps(out), flush=True)

    print("RESULTS_JSON_START")
    print(json.dumps(results, indent=2))
    print("RESULTS_JSON_END")


if __name__ == "__main__":
    main()
