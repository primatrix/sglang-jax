"""shard_map test for split KV packed kernel with tp=4.

Tests two MiMo-V2-Flash production configs:
  - SWA layers:      kv_heads=8, tp=4 → per-device=2
  - Full Attn layers: kv_heads=4, tp=4 → per-device=1

Both must compile AND produce correct output vs the reference implementation.
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from sgl_jax.srt.kernels.ragged_paged_attention.ragged_paged_attention_v3_split import (
    ragged_paged_attention_split_kv,
    ref_ragged_paged_attention_split_kv,
)
from sgl_jax.srt.kernels.ragged_paged_attention.util import align_to, cdiv

# --- Constants ---
K_HEAD_DIM = 192
V_HEAD_DIM = 128
PAGE_SIZE = 64
TP = 4

CONFIGS = {
    "SWA (kv_heads=8, per-device=2)": {
        "num_q_heads": 64,
        "num_kv_heads": 8,
        "sliding_window": 128,
    },
    "Full Attn (kv_heads=4, per-device=1)": {
        "num_q_heads": 64,
        "num_kv_heads": 4,
        "sliding_window": None,
    },
}

# Decode scenario
LENS = [(1, 128), (1, 256), (1, 384), (1, 512)]


def build_test_data(num_q_heads, num_kv_heads):
    """Build global (unsharded) test data.

    Generates per-sequence KV data, fills cache pages, then extracts the last
    q_len tokens as the "new token" arrays (keys/values). This ensures the
    kernel (which reads old tokens from cache + new tokens from keys/values)
    sees the same data as the reference (which reads everything from cache).
    """
    num_seqs = len(LENS)
    key = jax.random.PRNGKey(42)

    q_lens = [ql for ql, _ in LENS]
    kv_lens_list = [kvl for _, kvl in LENS]
    total_q = sum(q_lens)
    max_pages_per_seq = max(cdiv(kvl, PAGE_SIZE) for kvl in kv_lens_list)
    total_pages = num_seqs * max_pages_per_seq + 1

    k_hd_aligned = align_to(K_HEAD_DIM, 128)
    v_hd_aligned = align_to(V_HEAD_DIM, 128)

    # Allocate caches (zero-initialized, fill per-sequence below)
    k_cache = jnp.zeros((total_pages, PAGE_SIZE, num_kv_heads, k_hd_aligned), dtype=jnp.bfloat16)
    v_cache = jnp.zeros((total_pages, PAGE_SIZE, num_kv_heads, v_hd_aligned), dtype=jnp.bfloat16)

    # Page layout and metadata
    kv_lens = jnp.array(kv_lens_list, dtype=jnp.int32)
    cu_q_lens = jnp.array([0] + list(np.cumsum(q_lens)), dtype=jnp.int32)
    cu_kv_lens = jnp.array(
        [i * max_pages_per_seq * PAGE_SIZE for i in range(num_seqs + 1)], dtype=jnp.int32
    )

    page_indices_list = []
    for i, kvl in enumerate(kv_lens_list):
        n_pages = cdiv(kvl, PAGE_SIZE)
        base = i * max_pages_per_seq
        seq_pages = list(range(base, base + n_pages))
        seq_pages += [0] * (max_pages_per_seq - n_pages)
        page_indices_list.extend(seq_pages)
    page_indices = jnp.array(page_indices_list, dtype=jnp.int32)
    page_indices_2d = page_indices.reshape(num_seqs, max_pages_per_seq)

    # Generate per-sequence data: fill cache pages, extract new tokens
    extend_q_list, extend_k_list, extend_v_list = [], [], []
    for i, (q_len, kv_len) in enumerate(LENS):
        n_pages = cdiv(kv_len, PAGE_SIZE)
        key, k1, k2, k3 = jax.random.split(key, 4)
        seq_k = jax.random.normal(k1, (kv_len, num_kv_heads, K_HEAD_DIM), dtype=jnp.bfloat16)
        seq_v = jax.random.normal(k2, (kv_len, num_kv_heads, V_HEAD_DIM), dtype=jnp.bfloat16)
        seq_q = jax.random.normal(k3, (q_len, num_q_heads, K_HEAD_DIM), dtype=jnp.bfloat16)

        # Fill cache pages (pad head_dim to aligned size)
        base_page = i * max_pages_per_seq
        for p in range(n_pages):
            start = p * PAGE_SIZE
            end = min(start + PAGE_SIZE, kv_len)
            sz = end - start
            page_idx = base_page + p
            k_padded = jnp.pad(seq_k[start:end], ((0, 0), (0, 0), (0, k_hd_aligned - K_HEAD_DIM)))
            v_padded = jnp.pad(seq_v[start:end], ((0, 0), (0, 0), (0, v_hd_aligned - V_HEAD_DIM)))
            k_cache = k_cache.at[page_idx, :sz].set(k_padded)
            v_cache = v_cache.at[page_idx, :sz].set(v_padded)

        # New tokens = last q_len tokens of the KV sequence
        extend_q_list.append(seq_q)
        extend_k_list.append(seq_k[kv_len - q_len:])
        extend_v_list.append(seq_v[kv_len - q_len:])

    queries = jnp.concatenate(extend_q_list, axis=0)
    keys = jnp.concatenate(extend_k_list, axis=0)
    values = jnp.concatenate(extend_v_list, axis=0)

    distribution = jnp.array([num_seqs, num_seqs, num_seqs], dtype=jnp.int32)
    num_seqs_arr = jnp.array([num_seqs], dtype=jnp.int32)

    return {
        "queries": queries,
        "keys": keys,
        "values": values,
        "k_cache": k_cache,
        "v_cache": v_cache,
        "kv_lens": kv_lens,
        "page_indices": page_indices,
        "page_indices_2d": page_indices_2d,
        "cu_q_lens": cu_q_lens,
        "cu_kv_lens": cu_kv_lens,
        "distribution": distribution,
        "num_seqs": num_seqs_arr,
    }


def run_test(config_name, num_q_heads, num_kv_heads, sliding_window, mesh):
    """Run one shard_map test: compile + correctness check."""
    per_device = num_kv_heads // TP
    sm_scale = K_HEAD_DIM**-0.5

    print(f"\n{'='*60}")
    print(f"Config: {config_name}")
    print(f"  kv_heads={num_kv_heads}, tp={TP} → per-device kv_heads={per_device}")
    print(f"  sliding_window={sliding_window}")
    print(f"{'='*60}")

    data = build_test_data(num_q_heads, num_kv_heads)

    # --- Print per-device shapes for verification ---
    print(f"  Global k_cache shape:     {data['k_cache'].shape}")
    print(f"  Per-device k_cache shape: "
          f"({data['k_cache'].shape[0]}, {data['k_cache'].shape[1]}, "
          f"{per_device}, {data['k_cache'].shape[3]})")

    # --- Reference output (unsharded) ---
    ref_output = ref_ragged_paged_attention_split_kv(
        data["queries"],
        data["k_cache"][:, :, :num_kv_heads, :K_HEAD_DIM],
        data["v_cache"][:, :, :num_kv_heads, :V_HEAD_DIM],
        data["kv_lens"],
        data["page_indices_2d"],
        data["cu_q_lens"],
        data["num_seqs"],
        sm_scale=sm_scale,
        sliding_window=sliding_window,
    )

    # --- Sharding specs ---
    q_spec = P("data", "tensor", None)
    kv_new_spec = P("data", "tensor", None)
    kv_cache_spec = P("data", None, "tensor", None)
    scalar_spec = P("data")
    no_spec = P()

    def shard(arr, spec):
        return jax.device_put(arr, NamedSharding(mesh, spec))

    queries_s = shard(data["queries"], q_spec)
    keys_s = shard(data["keys"], kv_new_spec)
    values_s = shard(data["values"], kv_new_spec)
    k_cache_s = shard(data["k_cache"], kv_cache_spec)
    v_cache_s = shard(data["v_cache"], kv_cache_spec)
    kv_lens_s = shard(data["kv_lens"], scalar_spec)
    page_indices_s = shard(data["page_indices"], scalar_spec)
    cu_q_lens_s = shard(data["cu_q_lens"], scalar_spec)
    cu_kv_lens_s = shard(data["cu_kv_lens"], scalar_spec)
    distribution_s = shard(data["distribution"], no_spec)

    def fn(q, k, v, kc, vc, kl, pi, cq, ck, d, m):
        return ragged_paged_attention_split_kv(
            q, k, v, kc, vc, kl, pi, cq, ck, d, m,
            sm_scale=sm_scale, sliding_window=sliding_window,
        )

    # --- Compile + run ---
    print("  Compiling shard_map kernel...")
    sharded_fn = jax.jit(
        jax.experimental.shard_map.shard_map(
            fn,
            mesh=mesh,
            in_specs=(
                q_spec, kv_new_spec, kv_new_spec,
                kv_cache_spec, kv_cache_spec,
                scalar_spec, scalar_spec, scalar_spec, scalar_spec,
                no_spec, no_spec,
            ),
            out_specs=(q_spec, kv_cache_spec, kv_cache_spec),
            check_rep=False,
        )
    )

    out = sharded_fn(
        queries_s, keys_s, values_s, k_cache_s, v_cache_s,
        kv_lens_s, page_indices_s, cu_q_lens_s, cu_kv_lens_s,
        distribution_s, None,
    )
    print(f"  Compiled OK. Output shape: {out[0].shape}")

    # --- Correctness check ---
    output_np = np.asarray(out[0])
    ref_np = np.asarray(ref_output)

    # Trim to v_head_dim for comparison
    output_np = output_np[:, :, :V_HEAD_DIM]
    ref_np = ref_np[:, :, :V_HEAD_DIM]

    max_diff = np.max(np.abs(output_np - ref_np))
    mean_diff = np.mean(np.abs(output_np - ref_np))
    print(f"  max_diff={max_diff}, mean_diff={mean_diff}")

    try:
        np.testing.assert_allclose(output_np, ref_np, rtol=2e-2, atol=1e-2)
        print(f"  PASS ✓")
        return True
    except AssertionError as e:
        print(f"  FAIL ✗: {e}")
        return False


def main():
    devices = jax.devices()
    assert len(devices) >= TP, f"Need {TP} devices, got {len(devices)}"
    mesh = Mesh(np.array(devices[:TP]).reshape(1, TP), ("data", "tensor"))
    print(f"Mesh: {mesh}, devices: {len(devices)}")

    results = {}
    for name, cfg in CONFIGS.items():
        try:
            ok = run_test(
                name,
                cfg["num_q_heads"],
                cfg["num_kv_heads"],
                cfg["sliding_window"],
                mesh,
            )
            results[name] = "PASS" if ok else "FAIL (accuracy)"
        except Exception as e:
            print(f"  FAIL ✗ (compile): {type(e).__name__}")
            print(f"  {str(e)[:2000]}")
            results[name] = f"FAIL ({type(e).__name__})"

    print(f"\n{'='*60}")
    print("Summary:")
    for name, status in results.items():
        print(f"  {name}: {status}")
    print(f"{'='*60}")

    if any("FAIL" in s for s in results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
