# Split K/V variant: supports separate K and V caches with different head_dim.
"""Split KV ragged paged attention — public API and reference implementation.

Supports separate K and V caches with potentially different head dimensions
(e.g., k_dim=192, v_dim=128 for MiMo-V2-Flash).

The actual Pallas kernel lives in ``ragged_paged_attention_v3_split_packed``
which uses a 5D packed VMEM layout to satisfy Mosaic bf16 tiling constraints.
This module re-exports the kernel entry point and provides a pure-JAX reference
implementation for correctness testing.
"""

import functools
import logging

import jax
import jax.numpy as jnp

from sgl_jax.srt.kernels.ragged_paged_attention.ragged_paged_attention_v3_split_packed import (
    DEFAULT_MASK_VALUE,
    ragged_paged_attention_split as _ragged_paged_attention_split_packed,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reference implementation
# ---------------------------------------------------------------------------
def ref_ragged_paged_attention_split_kv(
    queries: jax.Array,  # [padded_num_tokens, num_q_heads, head_dim]
    k_pages: jax.Array,  # [total_num_pages, page_size, num_kv_heads, k_head_dim]
    v_pages: jax.Array,  # [total_num_pages, page_size, num_kv_heads, v_head_dim]
    kv_lens: jax.Array,  # i32[padded_batch_size]
    page_indices: jax.Array,  # i32[padded_batch_size, max_pages_per_seq]
    cu_q_lens: jax.Array,  # i32[padded_batch_size + 1]
    num_seqs: jax.Array,  # i32[1]
    *,
    custom_mask: jax.Array = None,
    causal: bool = True,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = DEFAULT_MASK_VALUE,
    k_scale: float | None = None,
    v_scale: float | None = None,
    xai_temperature_len: float | None = None,
    attention_sink: jax.Array | float | None = None,
):
    """Reference implementation for split KV ragged paged attention.

    K and V pages may have different head_dim. The QK computation uses
    q.shape[-1] dimensions (which equals k_head_dim), and the PV computation
    uses v_head_dim. The output has shape [num_tokens, num_q_heads, v_head_dim].
    """
    if not causal:
        assert (
            custom_mask is not None and custom_mask.size > jnp.cumsum(kv_lens)[-1]
        ), "use custom_mask for non-causal attention"
    if mask_value is None:
        mask_value = DEFAULT_MASK_VALUE

    _, _, num_kv_heads, k_head_dim = k_pages.shape
    v_head_dim = v_pages.shape[-1]
    num_q_heads = queries.shape[1]
    q_head_dim = queries.shape[2]
    assert num_q_heads % num_kv_heads == 0
    num_query_per_kv = num_q_heads // num_kv_heads

    outputs = []
    mask_len_list = []
    for i in range(num_seqs[0]):
        kv_len = kv_lens[i]
        q_len = cu_q_lens[i + 1] - cu_q_lens[i]
        mask_len_list.append(q_len * kv_len)
    mask_lens = jnp.array(mask_len_list, dtype=jnp.int32)
    cu_mask_lens = jnp.concatenate([jnp.array([0], dtype=jnp.int32), jnp.cumsum(mask_lens)])

    for i in range(num_seqs[0]):
        q_start = cu_q_lens[i]
        q_end = cu_q_lens[i + 1]
        q_len = q_end - q_start
        kv_len = kv_lens[i]
        indices = page_indices[i]
        q = queries[q_start:q_end]
        k = k_pages[indices, :, :, :].reshape(-1, num_kv_heads, k_head_dim)[:kv_len]
        v = v_pages[indices, :, :, :].reshape(-1, num_kv_heads, v_head_dim)[:kv_len]

        if k_scale is not None:
            k = (k.astype(jnp.float32) * k_scale).astype(q.dtype)
        if v_scale is not None:
            v = (v.astype(jnp.float32) * v_scale).astype(q.dtype)

        k = jnp.repeat(k, num_query_per_kv, axis=1)
        v = jnp.repeat(v, num_query_per_kv, axis=1)

        # QK: use min(q_head_dim, k_head_dim) dimensions
        common_dim = min(q_head_dim, k_head_dim)
        attn = jnp.einsum(
            "qhd,khd->hqk",
            q[:, :, :common_dim],
            k[:, :, :common_dim],
            preferred_element_type=jnp.float32,
        )
        attn *= sm_scale

        if causal:
            q_span = (kv_len - q_len) + jax.lax.broadcasted_iota(jnp.int32, attn.shape, 1)
            kv_span = jax.lax.broadcasted_iota(jnp.int32, attn.shape, 2)
            mask = q_span < kv_span
        else:
            mask_start = cu_mask_lens[i]
            mask_end = cu_mask_lens[i + 1]
            mask = custom_mask[mask_start:mask_end]
            mask = (
                jnp.repeat(jnp.expand_dims(mask, axis=0), num_q_heads, axis=0).reshape(
                    num_q_heads, q_len, kv_len
                )
                < 1
            )

        if sliding_window is not None:
            mask = jnp.logical_or(mask, q_span - sliding_window >= kv_span)
        if soft_cap is not None:
            attn = soft_cap * jnp.tanh(attn / soft_cap)

        if xai_temperature_len is not None:
            prefix_len = kv_len - q_len
            qidx = jnp.arange(prefix_len, kv_len)
            xai_temperature_scale = 1.0 / jnp.log2(float(xai_temperature_len))
            _qtemp = jnp.log2(qidx.astype(jnp.float32)) * xai_temperature_scale
            xai_temperature_reg = jnp.where(qidx > xai_temperature_len, _qtemp, 1.0)
            attn = attn * xai_temperature_reg[None, :, None]

        attn += jnp.where(mask, mask_value, 0.0)

        if attention_sink is not None:
            sink = jnp.asarray(attention_sink, dtype=jnp.float32)
            if sink.ndim == 0:
                sink = jnp.full((num_q_heads,), sink)
            sink_logits = jnp.broadcast_to(
                sink.reshape(num_q_heads, 1, 1),
                (num_q_heads, q_len, 1),
            )
            attn = jnp.concatenate([sink_logits, attn], axis=-1)
            attn = jax.nn.softmax(attn, axis=-1).astype(v.dtype)
            attn = attn[..., 1:]
        else:
            attn = jax.nn.softmax(attn, axis=-1).astype(v.dtype)

        out = jnp.einsum("hqk,khd->qhd", attn, v).astype(queries.dtype)
        outputs.append(out)

    return jnp.concatenate(outputs, axis=0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
@functools.partial(
    jax.jit,
    static_argnames=(
        "causal",
        "sm_scale",
        "sliding_window",
        "soft_cap",
        "mask_value",
        "q_scale",
        "k_scale",
        "v_scale",
        "xai_temperature_len",
        "chunk_prefill_size",
        "d_block_sizes",
        "p_block_sizes",
        "m_block_sizes",
        "vmem_limit_bytes",
        "out_dtype",
        "skip_kv_mask",
        "disable_semaphore_checks",
        "debug_mode",
    ),
    donate_argnames=("queries", "keys", "values", "k_cache", "v_cache"),
)
def ragged_paged_attention_split_kv(
    queries: jax.Array,  # [max_num_tokens, actual_num_q_heads, actual_head_dim]
    keys: jax.Array,  # [max_num_tokens, actual_num_kv_heads, actual_k_head_dim]
    values: jax.Array,  # [max_num_tokens, actual_num_kv_heads, actual_v_head_dim]
    k_cache: jax.Array,  # [total_pages, page_size, num_kv_heads, k_head_dim]
    v_cache: jax.Array,  # [total_pages, page_size, num_kv_heads, v_head_dim]
    kv_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[flat]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    cu_kv_lens: jax.Array,  # i32[max_num_seqs + 1]
    distribution: jax.Array,  # i32[3]
    custom_mask: jax.Array | None,
    attention_sink: jax.Array | None = None,
    *,
    causal: int = 1,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = DEFAULT_MASK_VALUE,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    xai_temperature_len: float | None = None,
    chunk_prefill_size: int | None = None,
    d_block_sizes: tuple[int, int, int, int] | None = None,
    p_block_sizes: tuple[int, int, int, int] | None = None,
    m_block_sizes: tuple[int, int, int, int] | None = None,
    vmem_limit_bytes: int | None = None,
    out_dtype=None,
    skip_kv_mask: bool = False,
    disable_semaphore_checks: bool = True,
    debug_mode: bool = False,
):
    """Dispatch split-KV attention through the packed 5D TPU kernel.

    Extra v3-only tuning/debug kwargs are accepted for API compatibility but
    intentionally ignored by the packed kernel.
    """
    del d_block_sizes, p_block_sizes, m_block_sizes, out_dtype, skip_kv_mask
    del disable_semaphore_checks, debug_mode
    output, updated_k, updated_v = _ragged_paged_attention_split_packed(
        queries,
        keys,
        values,
        k_cache,
        v_cache,
        kv_lens,
        page_indices,
        cu_q_lens,
        cu_kv_lens,
        distribution,
        custom_mask,
        causal=causal,
        sm_scale=sm_scale,
        sliding_window=sliding_window,
        attention_sink=attention_sink,
        soft_cap=soft_cap,
        mask_value=mask_value,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        xai_temperature_len=xai_temperature_len,
        chunk_prefill_size=chunk_prefill_size,
        vmem_limit_bytes=vmem_limit_bytes,
    )
    updated_k = updated_k.reshape(k_cache.shape[0], k_cache.shape[1], *updated_k.shape[1:])
    updated_v = updated_v.reshape(v_cache.shape[0], v_cache.shape[1], *updated_v.shape[1:])
    return output, updated_k, updated_v
