"""Accuracy test for split KV kernel with page_size=128.

Verifies that the split kernel produces correct results with page_size=128
before running performance benchmarks.

Usage:
    export PYTHONPATH="$PWD/python:$PWD/benchmark/kernels/flash_attention"
    python benchmark/kernels/flash_attention/test_split_ps128_accuracy.py
"""

from __future__ import annotations

import functools
import sys

import jax
import jax.numpy as jnp
import numpy as np
from utils import (
    create_page_indices_data,
    create_split_kv_cache_data,
    create_split_qkv_data,
)

from sgl_jax.srt.kernels.ragged_paged_attention.ragged_paged_attention import (
    ragged_paged_attention,
)


def ref_attention_split(
    queries, k_pages, v_pages, kv_lens, page_table, cu_q_lens,
    *, sm_scale, causal=True,
):
    """NumPy reference: multi-head attention with paged KV cache."""
    batch_size = kv_lens.shape[0]
    num_q_heads = queries.shape[1]
    num_kv_heads = k_pages.shape[2]
    v_head_dim = v_pages.shape[3]
    heads_per_group = num_q_heads // num_kv_heads
    page_size = k_pages.shape[1]

    outputs = []
    for b in range(batch_size):
        q_start = int(cu_q_lens[b])
        q_end = int(cu_q_lens[b + 1])
        q_len = q_end - q_start
        kv_len = int(kv_lens[b])
        q = queries[q_start:q_end]

        k_list, v_list = [], []
        for t in range(kv_len):
            page_idx = int(page_table[b, t // page_size])
            offset = t % page_size
            k_list.append(k_pages[page_idx, offset])
            v_list.append(v_pages[page_idx, offset])
        k = np.stack(k_list)
        v = np.stack(v_list)

        for h in range(num_q_heads):
            kv_h = h // heads_per_group
            q_h = np.array(q[:, h, :])
            k_h = np.array(k[:, kv_h, :])
            v_h = np.array(v[:, kv_h, :])

            common_dim = min(q_h.shape[-1], k_h.shape[-1])
            scores = q_h[:, :common_dim] @ k_h[:, :common_dim].T * sm_scale

            if causal:
                prefix_len = kv_len - q_len
                for qi in range(q_len):
                    for ki in range(kv_len):
                        if ki > prefix_len + qi:
                            scores[qi, ki] = -1e9

            weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
            weights = weights / weights.sum(axis=-1, keepdims=True)
            out_h = weights @ v_h
            outputs.append(out_h)

    total_tokens = int(cu_q_lens[-1])
    result = np.zeros((total_tokens, num_q_heads, v_head_dim))
    idx = 0
    for b in range(batch_size):
        q_start = int(cu_q_lens[b])
        q_end = int(cu_q_lens[b + 1])
        for h in range(num_q_heads):
            result[q_start:q_end, h, :] = outputs[idx]
            idx += 1
    return result


def run_test(mode, k_head_dim, v_head_dim, q_head_num=4, kv_head_num=2,
             page_size=128, max_context_len=512, batch_size=4,
             rtol=2e-2, atol=1e-2):
    print(f"  [{mode}] k_dim={k_head_dim}, v_dim={v_head_dim}, "
          f"q_heads={q_head_num}, kv_heads={kv_head_num}, ps={page_size} ... ",
          end="", flush=True)

    head_dim = k_head_dim
    sm_scale = head_dim ** -0.5
    max_kv_cache_tokens = batch_size * max_context_len + 1024
    seed = 42
    dtype = jnp.bfloat16

    if mode == "decode":
        num_tokens = batch_size
        seq_lens = jnp.full((batch_size,), max_context_len, dtype=jnp.int32)
        cu_q_lens = jnp.arange(batch_size + 1, dtype=jnp.int32)
        cu_kv_lens = jnp.arange(batch_size + 1, dtype=jnp.int32) * max_context_len
        distribution = jnp.array([0, 0, batch_size], dtype=jnp.int32)
    else:
        tokens_per_seq = max_context_len
        num_tokens = batch_size * tokens_per_seq
        seq_lens = jnp.full((batch_size,), max_context_len, dtype=jnp.int32)
        cu_q_lens = jnp.arange(batch_size + 1, dtype=jnp.int32) * tokens_per_seq
        cu_kv_lens = cu_q_lens
        distribution = jnp.array([0, batch_size, batch_size], dtype=jnp.int32)

    q, k, v = create_split_qkv_data(
        num_tokens, q_head_num, kv_head_num, k_head_dim, v_head_dim, dtype, seed
    )
    k_cache, v_cache = create_split_kv_cache_data(
        max_kv_cache_tokens, kv_head_num, k_head_dim, v_head_dim,
        page_size=page_size, dtype=dtype, seed=seed,
    )
    total_kv = int(batch_size * max_context_len)
    page_indices, cache_loc = create_page_indices_data(
        batch_size, total_kv, seq_lens, max_context_len, page_size
    )
    max_pages_per_seq = (max_context_len + page_size - 1) // page_size
    page_table_2d = np.array(page_indices).reshape(batch_size, max_pages_per_seq)

    @functools.partial(jax.jit, static_argnames=["sm_scale"])
    def run(q, k, v, k_cache, v_cache, kv_lens, page_indices,
            cu_q_lens, cu_kv_lens, distribution, sm_scale):
        return ragged_paged_attention(
            q, k, v, None, kv_lens, page_indices,
            cu_q_lens, cu_kv_lens, distribution,
            custom_mask=None, causal=1, sm_scale=sm_scale,
            k_cache=k_cache, v_cache=v_cache,
        )

    jax_out, updated_k, updated_v = run(
        q, k, v, k_cache, v_cache, seq_lens, page_indices,
        cu_q_lens, cu_kv_lens, distribution, sm_scale,
    )
    jax.block_until_ready(jax_out)
    jax_np = np.array(jax_out)

    q_np = np.array(q).reshape(num_tokens, q_head_num, head_dim)
    k_cache_np = np.array(k_cache)
    v_cache_np = np.array(v_cache)
    seq_lens_np = np.array(seq_lens)
    cu_q_lens_np = np.array(cu_q_lens)

    if mode == "decode":
        k_new_np = np.array(k).reshape(num_tokens, kv_head_num, k_head_dim)
        v_new_np = np.array(v).reshape(num_tokens, kv_head_num, v_head_dim)
        for b in range(batch_size):
            kv_len = int(seq_lens_np[b])
            last_page = page_table_2d[b, (kv_len - 1) // page_size]
            last_offset = (kv_len - 1) % page_size
            k_cache_np[last_page, last_offset] = k_new_np[b]
            v_cache_np[last_page, last_offset] = v_new_np[b]

    ref_out = ref_attention_split(
        q_np, k_cache_np, v_cache_np, seq_lens_np,
        page_table_2d, cu_q_lens_np,
        sm_scale=sm_scale, causal=True,
    )

    ref_flat = ref_out.reshape(num_tokens, -1)
    jax_flat = jax_np.reshape(num_tokens, -1)[:, :ref_flat.shape[1]]

    max_diff = np.max(np.abs(ref_flat - jax_flat))
    mean_diff = np.mean(np.abs(ref_flat - jax_flat))

    passed = np.allclose(ref_flat, jax_flat, rtol=rtol, atol=atol)
    status = "PASS" if passed else "FAIL"
    print(f"{status}  (max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f})")
    return passed


def main():
    print(f"Devices: {jax.devices()}")
    print(f"Backend: {jax.default_backend()}")
    print()
    print("=== page_size=128 accuracy tests ===")

    configs = [
        ("decode", 128, 128, 4, 2, 128),
        ("decode", 256, 128, 4, 2, 128),
        ("decode", 192, 128, 4, 2, 128),
        ("decode", 192, 128, 8, 4, 128),
        ("prefill", 128, 128, 4, 2, 128),
        ("prefill", 256, 128, 4, 2, 128),
        ("prefill", 192, 128, 4, 2, 128),
    ]

    results = []
    for mode, k_dim, v_dim, q_heads, kv_heads, ps in configs:
        ok = run_test(mode, k_dim, v_dim, q_heads, kv_heads, ps,
                      max_context_len=512, batch_size=4)
        results.append(ok)

    print()
    passed = sum(results)
    total = len(results)
    print(f"Results: {passed}/{total} passed")
    if passed < total:
        sys.exit(1)


if __name__ == "__main__":
    main()
