"""Tests for the BF16 MHA padding removal path."""

import math

import jax
import jax.numpy as jnp

from sgl_jax.srt.multimodal.kernels.varlen_attention import (
    _prepare_mha_layout,
    ref_varlen_attention,
    varlen_attention,
)


def test_mha_layout_keeps_exact_capacities_and_native_head_dim():
    q = jnp.zeros((17, 16, 80), dtype=jnp.bfloat16)

    q_internal, k_internal, v_internal, *_ = _prepare_mha_layout(
        q,
        q,
        q,
        None,
        bq_size=32,
        bkv_size=32,
        interpret=False,
    )

    # No concat, no head-dim pad: Q/K/V stay token-major at native head_dim.
    assert q_internal.shape == (17, 16, 80)
    assert k_internal.shape == (17, 16, 80)
    assert v_internal.shape == (17, 16, 80)


def test_mha_interpreter_retains_only_token_dma_slack():
    q = jnp.zeros((17, 16, 80), dtype=jnp.bfloat16)

    q_internal, k_internal, v_internal, *_ = _prepare_mha_layout(
        q,
        q,
        q,
        None,
        bq_size=32,
        bkv_size=32,
        interpret=True,
    )

    assert q_internal.shape == (48, 16, 80)
    assert k_internal.shape == (48, 16, 80)
    assert v_internal.shape == (48, 16, 80)


def test_mha_padding_removal_matches_reference_in_interpreter():
    total_tokens = 32
    num_heads = 2
    head_dim = 80
    keys = jax.random.split(jax.random.key(7), 3)
    q = jax.random.normal(keys[0], (total_tokens, num_heads, head_dim), jnp.bfloat16)
    k = jax.random.normal(keys[1], (total_tokens, num_heads, head_dim), jnp.bfloat16)
    v = jax.random.normal(keys[2], (total_tokens, num_heads, head_dim), jnp.bfloat16)
    cu_seqlens = jnp.asarray([0, 13, total_tokens], dtype=jnp.int32)
    num_seqs = jnp.asarray([2], dtype=jnp.int32)
    sm_scale = 1.0 / math.sqrt(head_dim)

    output = varlen_attention(
        q,
        k,
        v,
        cu_seqlens,
        num_seqs,
        window_size=(-1, 0),
        sm_scale=sm_scale,
        num_queries_per_block=32,
        num_kv_per_block=32,
        interpret=True,
    )
    reference = ref_varlen_attention(
        q,
        k,
        v,
        cu_seqlens,
        num_seqs,
        window_size=(-1, 0),
        sm_scale=sm_scale,
    )

    max_error = jnp.max(jnp.abs(output.astype(jnp.float32) - reference.astype(jnp.float32)))
    assert float(max_error) <= 2e-2
