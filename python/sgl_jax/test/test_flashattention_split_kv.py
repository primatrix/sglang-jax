"""Kernel-level correctness test for ragged_paged_attention_split_kv.

Compares the Pallas kernel output against the reference implementation
(ref_ragged_paged_attention_split_kv) for split K/V cache with different
head dimensions.  No ForwardBatch, RadixAttention, or MHATokenToKVPool
-- this is a pure kernel test.
"""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.kernels.ragged_paged_attention.ragged_paged_attention_v3_split import (
    ragged_paged_attention_split_kv,
    ref_ragged_paged_attention_split_kv,
)
from sgl_jax.srt.kernels.ragged_paged_attention.util import (
    align_to,
    cdiv,
    get_dtype_packing,
)


def create_split_kv_test_data(
    lens,
    num_q_heads,
    num_kv_heads,
    k_head_dim,
    v_head_dim,
    page_size,
    dtype=jnp.bfloat16,
):
    """Create test data for split KV kernel testing.

    Args:
        lens: list of (q_len, kv_len) tuples per sequence.
        num_q_heads: number of query heads.
        num_kv_heads: number of key/value heads.
        k_head_dim: head dimension for keys.
        v_head_dim: head dimension for values.
        page_size: KV cache page size.
        dtype: tensor dtype.

    Returns:
        dict with all arrays needed by kernel and reference function.
    """
    num_seqs = len(lens)
    kv_packing = get_dtype_packing(dtype)

    # --- Page layout ---
    pages_per_seq_list = [cdiv(kv_len, page_size) for _, kv_len in lens]
    max_pages_per_seq = max(pages_per_seq_list)

    # kv_lens
    kv_lens = jnp.array([kv_len for _, kv_len in lens], dtype=jnp.int32)

    # cu_q_lens
    q_lens = [q_len for q_len, _ in lens]
    cu_q_lens = jnp.array([0] + list(np.cumsum(q_lens)), dtype=jnp.int32)

    # cu_kv_lens: each seq gets max_pages_per_seq page slots so that
    # cdiv(cu_kv_lens[i], page_size) == i * max_pages_per_seq.
    cu_kv_lens = jnp.array(
        [i * max_pages_per_seq * page_size for i in range(num_seqs + 1)],
        dtype=jnp.int32,
    )

    # page_indices (flat): seq i -> pages [i*max_pages_per_seq, ...)
    page_indices_list = []
    for i, (_, kv_len) in enumerate(lens):
        n_pages = cdiv(kv_len, page_size)
        base = i * max_pages_per_seq
        seq_pages = list(range(base, base + n_pages))
        seq_pages += [0] * (max_pages_per_seq - n_pages)
        page_indices_list.extend(seq_pages)
    page_indices_flat = jnp.array(page_indices_list, dtype=jnp.int32)
    page_indices_2d = page_indices_flat.reshape(num_seqs, max_pages_per_seq)
    total_pages = num_seqs * max_pages_per_seq + 1  # +1 padding page

    # distribution
    is_decode = all(q_len == 1 for q_len, _ in lens)
    if is_decode:
        distribution = jnp.array([num_seqs, num_seqs, num_seqs], dtype=jnp.int32)
    else:
        distribution = jnp.array([0, 0, num_seqs], dtype=jnp.int32)

    # --- Generate random data and fill pages ---
    key = jax.random.PRNGKey(42)
    k_pages_4d = jnp.zeros((total_pages, page_size, num_kv_heads, k_head_dim), dtype=dtype)
    v_pages_4d = jnp.zeros((total_pages, page_size, num_kv_heads, v_head_dim), dtype=dtype)

    extend_k_list, extend_v_list, extend_q_list = [], [], []
    for i, (q_len, kv_len) in enumerate(lens):
        n_pages = cdiv(kv_len, page_size)
        key, k1, k2, k3 = jax.random.split(key, 4)
        seq_k = jax.random.normal(k1, (kv_len, num_kv_heads, k_head_dim), dtype=dtype)
        seq_v = jax.random.normal(k2, (kv_len, num_kv_heads, v_head_dim), dtype=dtype)
        seq_q = jax.random.normal(k3, (q_len, num_q_heads, k_head_dim), dtype=dtype)

        # Fill pages
        base_page = i * max_pages_per_seq
        for p in range(n_pages):
            start = p * page_size
            end = min(start + page_size, kv_len)
            sz = end - start
            page_idx = base_page + p
            k_pages_4d = k_pages_4d.at[page_idx, :sz].set(seq_k[start:end])
            v_pages_4d = v_pages_4d.at[page_idx, :sz].set(seq_v[start:end])

        # Extend tokens (last q_len of the KV sequence)
        extend_k_list.append(seq_k[kv_len - q_len :])
        extend_v_list.append(seq_v[kv_len - q_len :])
        extend_q_list.append(seq_q)

    queries = jnp.concatenate(extend_q_list, axis=0)
    keys = jnp.concatenate(extend_k_list, axis=0)
    values = jnp.concatenate(extend_v_list, axis=0)

    # --- 4D native caches for the kernel (no packing dimension) ---
    pad_heads = align_to(num_kv_heads, kv_packing) - num_kv_heads
    if pad_heads > 0:
        k_cache = jnp.pad(k_pages_4d, ((0, 0), (0, 0), (0, pad_heads), (0, 0)))
        v_cache = jnp.pad(v_pages_4d, ((0, 0), (0, 0), (0, pad_heads), (0, 0)))
    else:
        k_cache = k_pages_4d
        v_cache = v_pages_4d

    return {
        "queries": queries,
        "keys": keys,
        "values": values,
        "k_cache": k_cache,
        "v_cache": v_cache,
        "k_pages_4d": k_pages_4d,
        "v_pages_4d": v_pages_4d,
        "kv_lens": kv_lens,
        "page_indices_flat": page_indices_flat,
        "page_indices_2d": page_indices_2d,
        "cu_q_lens": cu_q_lens,
        "cu_kv_lens": cu_kv_lens,
        "distribution": distribution,
        "num_seqs": jnp.array([num_seqs], dtype=jnp.int32),
    }


class TestSplitKVAttention(unittest.TestCase):
    """Kernel-level tests for ragged_paged_attention_split_kv."""

    def _run_split_kv_test(
        self,
        lens,
        num_q_heads,
        num_kv_heads,
        k_head_dim,
        v_head_dim,
        page_size,
        sliding_window=None,
        attention_sink=None,
    ):
        data = create_split_kv_test_data(
            lens, num_q_heads, num_kv_heads, k_head_dim, v_head_dim, page_size
        )
        sm_scale = 1.0 / (k_head_dim**0.5)

        # --- Reference (call first, kernel donates arrays) ---
        ref_output = ref_ragged_paged_attention_split_kv(
            data["queries"],
            data["k_pages_4d"],
            data["v_pages_4d"],
            data["kv_lens"],
            data["page_indices_2d"],
            data["cu_q_lens"],
            data["num_seqs"],
            sm_scale=sm_scale,
            sliding_window=sliding_window,
            attention_sink=attention_sink,
        )

        # --- Pallas kernel ---
        output, _, _ = ragged_paged_attention_split_kv(
            data["queries"],
            data["keys"],
            data["values"],
            data["k_cache"],
            data["v_cache"],
            data["kv_lens"],
            data["page_indices_flat"],
            data["cu_q_lens"],
            data["cu_kv_lens"],
            data["distribution"],
            None,  # custom_mask
            sm_scale=sm_scale,
            sliding_window=sliding_window,
            attention_sink=attention_sink,
        )

        output_np = np.asarray(output)
        ref_np = np.asarray(ref_output)

        np.testing.assert_allclose(
            output_np,
            ref_np,
            rtol=2e-2,
            atol=1e-2,
            err_msg=(
                f"Split KV attention mismatch: "
                f"k_dim={k_head_dim}, v_dim={v_head_dim}, "
                f"q_heads={num_q_heads}, kv_heads={num_kv_heads}, "
                f"lens={lens}"
            ),
        )

    # ------------------------------------------------------------------
    # Same head_dim (k=128, v=128) -- baseline sanity
    # ------------------------------------------------------------------
    def test_split_kv_decode_128_128(self):
        self._run_split_kv_test(
            lens=[(1, 128), (1, 256), (1, 512)],
            num_q_heads=32,
            num_kv_heads=8,
            k_head_dim=128,
            v_head_dim=128,
            page_size=64,
        )

    def test_split_kv_prefill_128_128(self):
        self._run_split_kv_test(
            lens=[(64, 64), (128, 128)],
            num_q_heads=32,
            num_kv_heads=8,
            k_head_dim=128,
            v_head_dim=128,
            page_size=64,
        )

    # ------------------------------------------------------------------
    # MiMo-V2-Flash case (k=192, v=128)
    # ------------------------------------------------------------------
    def test_split_kv_decode_192_128(self):
        self._run_split_kv_test(
            lens=[(1, 128), (1, 256), (1, 512)],
            num_q_heads=32,
            num_kv_heads=8,
            k_head_dim=192,
            v_head_dim=128,
            page_size=64,
        )

    def test_split_kv_prefill_192_128(self):
        self._run_split_kv_test(
            lens=[(64, 64), (128, 128)],
            num_q_heads=32,
            num_kv_heads=8,
            k_head_dim=192,
            v_head_dim=128,
            page_size=64,
        )

    # ------------------------------------------------------------------
    # Larger K dim (k=256, v=128)
    # ------------------------------------------------------------------
    def test_split_kv_decode_256_128(self):
        self._run_split_kv_test(
            lens=[(1, 128), (1, 256), (1, 512)],
            num_q_heads=32,
            num_kv_heads=8,
            k_head_dim=256,
            v_head_dim=128,
            page_size=64,
        )

    def test_split_kv_prefill_256_128(self):
        self._run_split_kv_test(
            lens=[(64, 64), (128, 128)],
            num_q_heads=32,
            num_kv_heads=8,
            k_head_dim=256,
            v_head_dim=128,
            page_size=64,
        )

    # ------------------------------------------------------------------
    # MHA variant (num_q_heads == num_kv_heads)
    # ------------------------------------------------------------------
    def test_split_kv_decode_192_128_mha(self):
        self._run_split_kv_test(
            lens=[(1, 128), (1, 256), (1, 512)],
            num_q_heads=32,
            num_kv_heads=32,
            k_head_dim=192,
            v_head_dim=128,
            page_size=64,
        )

    # ------------------------------------------------------------------
    # Multiple sequences with varying kv_lens
    # ------------------------------------------------------------------
    def test_split_kv_decode_192_128_multi_seq(self):
        self._run_split_kv_test(
            lens=[(1, 64), (1, 192), (1, 320), (1, 448)],
            num_q_heads=32,
            num_kv_heads=8,
            k_head_dim=192,
            v_head_dim=128,
            page_size=64,
        )

    # ------------------------------------------------------------------
    # SWA tests: sliding_window + attention_sink (MiMo-V2-Flash SWA config)
    # ------------------------------------------------------------------
    def test_split_kv_swa_decode_192_128(self):
        """SWA decode: kv_heads=8, sliding_window=128, with attention_sink."""
        num_q_heads = 64
        num_kv_heads = 8
        attention_sink = jnp.ones((num_q_heads,), dtype=jnp.float32) * 0.5
        self._run_split_kv_test(
            lens=[(1, 256), (1, 512), (1, 384)],
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            k_head_dim=192,
            v_head_dim=128,
            page_size=64,
            sliding_window=128,
            attention_sink=attention_sink,
        )

    def test_split_kv_swa_prefill_192_128(self):
        """SWA prefill: kv_heads=8, sliding_window=128, with attention_sink."""
        num_q_heads = 64
        num_kv_heads = 8
        attention_sink = jnp.ones((num_q_heads,), dtype=jnp.float32) * 0.5
        self._run_split_kv_test(
            lens=[(64, 256), (128, 384)],
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            k_head_dim=192,
            v_head_dim=128,
            page_size=64,
            sliding_window=128,
            attention_sink=attention_sink,
        )

    def test_split_kv_swa_decode_no_sink(self):
        """SWA decode with sliding_window only (no attention_sink)."""
        self._run_split_kv_test(
            lens=[(1, 256), (1, 512)],
            num_q_heads=64,
            num_kv_heads=8,
            k_head_dim=192,
            v_head_dim=128,
            page_size=64,
            sliding_window=128,
        )


if __name__ == "__main__":
    unittest.main()
