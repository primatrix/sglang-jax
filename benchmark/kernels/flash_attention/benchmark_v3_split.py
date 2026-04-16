"""v3 fused-vs-split KV benchmark with trace-based timing.

Compares:
  - v3 fused kernel (V padded to K's dim, simulating MiMo-V2 current approach)
  - v3 split kernel (native K/V dims)

Run from repo root:
    export PYTHONPATH="$PWD/python:$PWD/benchmark/kernels/flash_attention"
    python benchmark/kernels/flash_attention/benchmark_v3_split.py
"""

from __future__ import annotations

import json

import jax
import jax.numpy as jnp
import numpy as np
from utils import (
    create_page_indices_data,
    create_qkv_data,
    create_split_kv_cache_data,
    create_split_qkv_data,
)

from sgl_jax.srt.kernels.ragged_paged_attention.ragged_paged_attention_v3 import (
    get_kv_cache_shape,
    ragged_paged_attention,
)
from sgl_jax.srt.kernels.ragged_paged_attention.util import cdiv
from sgl_jax.srt.kernels.ragged_paged_attention.ragged_paged_attention_v3_split import (
    ragged_paged_attention_split_kv,
)
from sgl_jax.srt.kernels.utils.perf import multiple_iteration_timeit_from_trace

TRACE_ROOT = "/tmp/sgl_v3_split_trace"
# Same configs as the 04-10 delivery benchmark
CASES = [
    # --- Decode ---
    ("decode", "fused_128", dict(head_dim=128)),
    ("decode", "fused_256", dict(head_dim=256)),
    ("decode", "v3split_128_128", dict(k_head_dim=128, v_head_dim=128)),
    ("decode", "v3split_192_128", dict(k_head_dim=192, v_head_dim=128)),
    ("decode", "v3split_256_128", dict(k_head_dim=256, v_head_dim=128)),
    # --- Prefill ---
    ("prefill", "fused_128", dict(head_dim=128)),
    ("prefill", "fused_256", dict(head_dim=256)),
    ("prefill", "v3split_128_128", dict(k_head_dim=128, v_head_dim=128)),
    ("prefill", "v3split_192_128", dict(k_head_dim=192, v_head_dim=128)),
    ("prefill", "v3split_256_128", dict(k_head_dim=256, v_head_dim=128)),
]

# Shared params matching the 04-10 benchmark
COMMON = dict(
    max_context_len=4096,
    max_kv_cache_tokens=524288,
    q_head_num=4,
    kv_head_num=2,
    page_size=64,
)
DECODE_TOKENS = 128
PREFILL_TOKENS = 8192
TRIES = 5


def _create_fused_kv_cache(kv_head_num, head_dim, page_size, dtype=jnp.bfloat16, seed=42):
    """Create a 5D packed fused KV cache for v3 kernel."""
    shape = get_kv_cache_shape(
        cdiv(COMMON["max_kv_cache_tokens"], page_size),
        page_size,
        kv_head_num,
        head_dim,
        dtype,
    )
    return jax.random.normal(jax.random.PRNGKey(seed + 1), shape, dtype=dtype)


def _make_decode_data_fused(head_dim):
    bs = DECODE_TOKENS
    seq_lens = jnp.full((bs,), COMMON["max_context_len"], jnp.int32)
    cu_q = jnp.arange(bs + 1, dtype=jnp.int32)
    cu_kv = jnp.arange(bs + 1, dtype=jnp.int32) * COMMON["max_context_len"]
    q, k, v = create_qkv_data(bs, COMMON["q_head_num"], COMMON["kv_head_num"], head_dim)
    kv_cache = _create_fused_kv_cache(COMMON["kv_head_num"], head_dim, COMMON["page_size"])
    pi, _ = create_page_indices_data(
        bs,
        int(bs * COMMON["max_context_len"]),
        seq_lens,
        COMMON["max_context_len"],
        page_size=COMMON["page_size"],
    )
    dist = jnp.array([0, 0, bs], jnp.int32)
    return q, k, v, kv_cache, seq_lens, pi, cu_q, cu_kv, dist


def _make_decode_data_split(k_head_dim, v_head_dim):
    bs = DECODE_TOKENS
    seq_lens = jnp.full((bs,), COMMON["max_context_len"], jnp.int32)
    cu_q = jnp.arange(bs + 1, dtype=jnp.int32)
    cu_kv = jnp.arange(bs + 1, dtype=jnp.int32) * COMMON["max_context_len"]
    q, k, v = create_split_qkv_data(
        bs, COMMON["q_head_num"], COMMON["kv_head_num"], k_head_dim, v_head_dim
    )
    k_cache, v_cache = create_split_kv_cache_data(
        COMMON["max_kv_cache_tokens"],
        COMMON["kv_head_num"],
        k_head_dim,
        v_head_dim,
        page_size=COMMON["page_size"],
    )
    pi, _ = create_page_indices_data(
        bs,
        int(bs * COMMON["max_context_len"]),
        seq_lens,
        COMMON["max_context_len"],
        page_size=COMMON["page_size"],
    )
    dist = jnp.array([0, 0, bs], jnp.int32)
    return q, k, v, k_cache, v_cache, seq_lens, pi, cu_q, cu_kv, dist


def _make_prefill_data_fused(head_dim):
    bt = PREFILL_TOKENS
    if bt > 2048:
        bs = cdiv(bt, 2048)
        sl = [2048] * (bs - 1) + [bt - 2048 * (bs - 1)]
    else:
        bs = 1
        sl = [bt]
    seq_lens = jnp.array(sl, jnp.int32)
    cu_q = jnp.concatenate([jnp.array([0], jnp.int32), jnp.cumsum(seq_lens, dtype=jnp.int32)])
    cu_kv = jnp.concatenate([jnp.array([0], jnp.int32), jnp.cumsum(seq_lens, dtype=jnp.int32)])
    q, k, v = create_qkv_data(bt, COMMON["q_head_num"], COMMON["kv_head_num"], head_dim)
    kv_cache = _create_fused_kv_cache(COMMON["kv_head_num"], head_dim, COMMON["page_size"])
    pi, _ = create_page_indices_data(
        bs, bt, seq_lens, COMMON["max_context_len"], page_size=COMMON["page_size"]
    )
    dist = jnp.array([0, 0, bs], jnp.int32)
    return q, k, v, kv_cache, seq_lens, pi, cu_q, cu_kv, dist


def _make_prefill_data_split(k_head_dim, v_head_dim):
    bt = PREFILL_TOKENS
    if bt > 2048:
        bs = cdiv(bt, 2048)
        sl = [2048] * (bs - 1) + [bt - 2048 * (bs - 1)]
    else:
        bs = 1
        sl = [bt]
    seq_lens = jnp.array(sl, jnp.int32)
    cu_q = jnp.concatenate([jnp.array([0], jnp.int32), jnp.cumsum(seq_lens, dtype=jnp.int32)])
    cu_kv = jnp.concatenate([jnp.array([0], jnp.int32), jnp.cumsum(seq_lens, dtype=jnp.int32)])
    q, k, v = create_split_qkv_data(
        bt, COMMON["q_head_num"], COMMON["kv_head_num"], k_head_dim, v_head_dim
    )
    k_cache, v_cache = create_split_kv_cache_data(
        COMMON["max_kv_cache_tokens"],
        COMMON["kv_head_num"],
        k_head_dim,
        v_head_dim,
        page_size=COMMON["page_size"],
    )
    pi, _ = create_page_indices_data(
        bs, bt, seq_lens, COMMON["max_context_len"], page_size=COMMON["page_size"]
    )
    dist = jnp.array([0, 0, bs], jnp.int32)
    return q, k, v, k_cache, v_cache, seq_lens, pi, cu_q, cu_kv, dist


def bench_fused(mode, *, head_dim):
    scale = head_dim**-0.5
    if mode == "decode":
        q, k, v, kv_cache, kv_lens, pi, cu_q, cu_kv, dist = _make_decode_data_fused(head_dim)
    else:
        q, k, v, kv_cache, kv_lens, pi, cu_q, cu_kv, dist = _make_prefill_data_fused(head_dim)

    # Wrap to copy donated buffers each call
    def fn():
        return ragged_paged_attention(
            q.copy(), k.copy(), v.copy(), kv_cache.copy(),
            kv_lens, pi, cu_q, cu_kv, dist, None, sm_scale=scale,
        )

    jax.block_until_ready(fn())

    # v3 scope name: "RPAm-p_{ps}-bq_{bq}_{bq_csz}-bkv_{bkv}_{bkv_csz}"
    task = r"RPAm-p_\d+"
    times = multiple_iteration_timeit_from_trace(
        lambda: fn(), lambda: (), task=task, tries=TRIES, trace_root=TRACE_ROOT
    )
    return {
        "avg_ms": float(np.mean(times)),
        "med_ms": float(np.median(times)),
        "min_ms": float(np.min(times)),
        "std_ms": float(np.std(times)),
    }


def bench_split(mode, *, k_head_dim, v_head_dim):
    scale = k_head_dim**-0.5
    if mode == "decode":
        q, k, v, kc, vc, kv_lens, pi, cu_q, cu_kv, dist = _make_decode_data_split(
            k_head_dim, v_head_dim
        )
    else:
        q, k, v, kc, vc, kv_lens, pi, cu_q, cu_kv, dist = _make_prefill_data_split(
            k_head_dim, v_head_dim
        )

    def fn():
        return ragged_paged_attention_split_kv(
            q.copy(), k.copy(), v.copy(), kc.copy(), vc.copy(),
            kv_lens, pi, cu_q, cu_kv, dist, None, sm_scale=scale,
        )

    jax.block_until_ready(fn())

    # scope name: "RPA-bq_{bq}-bkvp_{bkv_p}-p_{ps}-split"
    task = r"RPA-bq_\d+.*split"
    times = multiple_iteration_timeit_from_trace(
        lambda: fn(), lambda: (), task=task, tries=TRIES, trace_root=TRACE_ROOT
    )
    return {
        "avg_ms": float(np.mean(times)),
        "med_ms": float(np.median(times)),
        "min_ms": float(np.min(times)),
        "std_ms": float(np.std(times)),
    }


def main():
    print(f"Devices: {jax.devices()}")
    print(f"Backend: {jax.default_backend()}")
    print()

    results = []
    for mode, name, kwargs in CASES:
        print(f"  {name:<22s} {mode:<8s} ...", end=" ", flush=True)
        if name.startswith("v3split"):
            r = bench_split(mode, **kwargs)
        else:
            r = bench_fused(mode, **kwargs)
        r["name"] = name
        r["mode"] = mode
        results.append(r)
        print(f"avg={r['avg_ms']:.3f}ms  med={r['med_ms']:.3f}ms  min={r['min_ms']:.3f}ms")

    # Summary table
    print()
    print("=" * 70)
    fused_128_d = next((r for r in results if r["name"] == "fused_128" and r["mode"] == "decode"), None)
    fused_128_p = next((r for r in results if r["name"] == "fused_128" and r["mode"] == "prefill"), None)
    print(f"{'Config':<22s} {'Mode':<8s} {'Avg(ms)':>8s} {'vs fused_128':>12s}")
    print("-" * 52)
    for r in results:
        base = fused_128_d if r["mode"] == "decode" else fused_128_p
        ratio = r["avg_ms"] / base["avg_ms"] if base else 0
        print(f"  {r['name']:<20s} {r['mode']:<8s} {r['avg_ms']:>8.3f} {ratio:>11.2f}x")

    print()
    print("RESULTS_JSON_START")
    print(json.dumps(results, indent=2))
    print("RESULTS_JSON_END")


if __name__ == "__main__":
    main()
