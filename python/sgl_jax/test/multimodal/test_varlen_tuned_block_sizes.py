"""Tests for the varlen_attention tuned block-size table and its wiring."""

import math

import jax
import jax.numpy as jnp
import pytest

from sgl_jax.srt.multimodal.kernels.varlen_attention import (
    DEFAULT_KV_BLOCK,
    DEFAULT_Q_BLOCK,
    ref_varlen_attention,
    varlen_attention,
)
from sgl_jax.srt.multimodal.kernels.varlen_tuned_block_sizes import (
    TUNED_VARLEN_BLOCK_SIZES,
    get_varlen_tuned_block_sizes,
)

HEADS = 16
HEAD_DIM = 80
SM_SCALE = 1.0 / math.sqrt(HEAD_DIM)


def test_v7_table_entries_present():
    v7 = TUNED_VARLEN_BLOCK_SIZES["TPU v7"]
    # Representative full-attention entry (large bucket) and window entry.
    assert v7[("bfloat16", "bfloat16", "bfloat16", 16, 8192, 8192, 80, "full")] == (256, 1024)
    assert v7[("bfloat16", "bfloat16", "bfloat16", 16, 2048, 2048, 80, "window")] == (128, 256)
    # Every full entry has a matching window entry (broadest table).
    full_keys = {k[:-1] for k in v7 if k[-1] == "full"}
    window_keys = {k[:-1] for k in v7 if k[-1] == "window"}
    assert full_keys == window_keys


def test_off_tpu_fallback_returns_defaults():
    # On CPU (the test platform) the lookup must return the kernel defaults.
    assert jax.devices()[0].device_kind == "cpu" or "TPU" not in jax.devices()[0].device_kind
    bq, bkv = get_varlen_tuned_block_sizes(
        jnp.bfloat16, jnp.bfloat16, jnp.bfloat16, HEADS, 8192, 8192, HEAD_DIM, "full"
    )
    assert (bq, bkv) == (DEFAULT_Q_BLOCK, DEFAULT_KV_BLOCK)


def _build_cu(total, layout):
    if layout == "full":
        bounds = [0, total]
    else:
        bounds = list(range(0, total, 64))
        if bounds[-1] != total:
            bounds.append(total)
    return (
        jnp.asarray(bounds, dtype=jnp.int32),
        jnp.asarray([len(bounds) - 1], dtype=jnp.int32),
    )


@pytest.mark.parametrize("layout", ["full", "window"])
def test_auto_block_sizes_match_reference_cpu(layout):
    # The None-sentinel (auto) path must produce the same result as the
    # reference on CPU interpret (fallback block sizes, correctness unchanged).
    total = 256
    keys = jax.random.split(jax.random.key(0), 3)
    q = jax.random.normal(keys[0], (total, HEADS, HEAD_DIM), jnp.bfloat16)
    k = jax.random.normal(keys[1], (total, HEADS, HEAD_DIM), jnp.bfloat16)
    v = jax.random.normal(keys[2], (total, HEADS, HEAD_DIM), jnp.bfloat16)
    cu, num_seqs = _build_cu(total, layout)
    ref = ref_varlen_attention(q, k, v, cu, num_seqs, sm_scale=SM_SCALE).astype(jnp.float32)
    out = varlen_attention(q, k, v, cu, num_seqs, sm_scale=SM_SCALE, interpret=True).astype(
        jnp.float32
    )
    assert float(jnp.max(jnp.abs(out[:total] - ref[:total]))) <= 2e-2
