import math

import jax
import jax.numpy as jnp
from flax import nnx
from jax import lax


def _default_compute_dtype(dtype: jnp.dtype) -> jnp.dtype:
    if dtype in (jnp.float16, jnp.bfloat16):
        return jnp.float32
    return dtype


def ring_attention(
    query,
    key,
    value,
    scale=None,
    causal=False,
    block_q=128,
    block_k=128,
    compute_dtype=None,
):
    """Blockwise ring attention for diffusion models (no KV cache).

    Args:
        query: [B, S_q, H, D]
        key: [B, S_k, H, D]
        value: [B, S_k, H, D]
        scale: softmax scale, default 1/sqrt(D)
        causal: whether to apply causal mask
        block_q: query block size
        block_k: key/value block size
        compute_dtype: optional dtype for attention math
    Returns:
        output: [B, S_q, H, D]
    """
    if block_q <= 0 or block_k <= 0:
        raise ValueError("block_q and block_k must be positive.")
    if scale is None:
        scale = 1.0 / math.sqrt(query.shape[-1])
    if query.shape[1] == 0 or key.shape[1] == 0:
        return jnp.zeros_like(query)

    if query.shape[2] != key.shape[2]:
        if query.shape[2] % key.shape[2] != 0:
            raise ValueError("Query heads must be a multiple of key heads.")
        repeats = query.shape[2] // key.shape[2]
        key = jnp.repeat(key, repeats, axis=2)
        value = jnp.repeat(value, repeats, axis=2)

    q_len = query.shape[1]
    kv_len = key.shape[1]

    # [B, H, S, D]
    q = jnp.transpose(query, (0, 2, 1, 3))
    k = jnp.transpose(key, (0, 2, 1, 3))
    v = jnp.transpose(value, (0, 2, 1, 3))

    q_pad = (-q_len) % block_q
    kv_pad = (-kv_len) % block_k
    if q_pad:
        q = jnp.pad(q, ((0, 0), (0, 0), (0, q_pad), (0, 0)))
    if kv_pad:
        k = jnp.pad(k, ((0, 0), (0, 0), (0, kv_pad), (0, 0)))
        v = jnp.pad(v, ((0, 0), (0, 0), (0, kv_pad), (0, 0)))

    batch_size, num_heads, q_len_pad, head_dim = q.shape
    kv_len_pad = k.shape[2]
    num_q_blocks = q_len_pad // block_q
    num_k_blocks = kv_len_pad // block_k

    compute_dtype = compute_dtype or _default_compute_dtype(query.dtype)
    scale = jnp.asarray(scale, dtype=compute_dtype)
    neg_inf = jnp.array(-jnp.inf, dtype=compute_dtype)
    q_idx_base = jnp.arange(block_q, dtype=jnp.int32)
    k_idx_base = jnp.arange(block_k, dtype=jnp.int32)
    mask_needed = causal or kv_pad > 0

    out = jnp.zeros((batch_size, num_heads, q_len_pad, head_dim), dtype=value.dtype)

    def q_loop(q_block_idx, out):
        q_start = q_block_idx * block_q
        q_block = lax.dynamic_slice(
            q, (0, 0, q_start, 0), (batch_size, num_heads, block_q, head_dim)
        )
        q_block = q_block.astype(compute_dtype)
        m = jnp.full((batch_size, num_heads, block_q), neg_inf, dtype=compute_dtype)
        l = jnp.zeros((batch_size, num_heads, block_q), dtype=compute_dtype)
        o = jnp.zeros((batch_size, num_heads, block_q, head_dim), dtype=compute_dtype)
        q_idx = q_start + q_idx_base if causal else None

        def k_loop(k_block_idx, state):
            m, l, o = state
            k_start = k_block_idx * block_k
            k_block = lax.dynamic_slice(
                k, (0, 0, k_start, 0), (batch_size, num_heads, block_k, head_dim)
            )
            v_block = lax.dynamic_slice(
                v, (0, 0, k_start, 0), (batch_size, num_heads, block_k, head_dim)
            )
            k_block = k_block.astype(compute_dtype)
            v_block = v_block.astype(compute_dtype)
            scores = jnp.einsum("bhqd,bhkd->bhqk", q_block, k_block) * scale
            if mask_needed:
                k_idx = k_start + k_idx_base
                if causal:
                    mask = q_idx[:, None] >= k_idx[None, :]
                else:
                    mask = jnp.ones((block_q, block_k), dtype=bool)
                if kv_pad > 0:
                    mask = mask & (k_idx < kv_len)[None, :]
                scores = jnp.where(mask[None, None, :, :], scores, neg_inf)
            m_new = jnp.maximum(m, jnp.max(scores, axis=-1))
            exp_m = jnp.exp(m - m_new)
            exp_scores = jnp.exp(scores - m_new[..., None])
            l = l * exp_m + jnp.sum(exp_scores, axis=-1)
            o = o * exp_m[..., None] + jnp.einsum("bhqk,bhkd->bhqd", exp_scores, v_block)
            return m_new, l, o

        m, l, o = lax.fori_loop(0, num_k_blocks, k_loop, (m, l, o))
        o = o / l[..., None]
        o = o.astype(value.dtype)
        out = lax.dynamic_update_slice(out, o, (0, 0, q_start, 0))
        return out

    out = lax.fori_loop(0, num_q_blocks, q_loop, out)
    out = out[:, :, :q_len, :]
    return jnp.transpose(out, (0, 2, 1, 3))


def simple_attention(query, key, value, scale=None, causal=False):
    """Legacy alias for ring_attention."""
    return ring_attention(query, key, value, scale, causal)


class RingAttention(nnx.Module):
    """Ring attention with blockwise streaming softmax."""

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        num_kv_heads: int | None = None,
        softmax_scale: float | None = None,
        causal: bool = False,
        dropout_rate: float = 0.0,
        layer_id: int = 0,
        logit_cap: float | None = None,
        scaling: float | None = None,
        **extra_impl_args,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.softmax_scale = softmax_scale or 1.0 / math.sqrt(head_size)
        self.causal = causal
        self.dropout_rate = dropout_rate
        self.head_dim = head_size
        self.layer_id = layer_id
        self.logit_cap = logit_cap or None
        self.scaling = scaling
        self.block_q = extra_impl_args.pop("block_q", 128)
        self.block_k = extra_impl_args.pop("block_k", 128)
        self.compute_dtype = extra_impl_args.pop("compute_dtype", None)

    def __call__(
        self,
        query: jax.Array,
        key: jax.Array,
        value: jax.Array,
        req=None,
    ) -> jax.Array:
        """
        Forward pass for RingAttention.

            q, k, v: [B, S_local, H, D]

        Note: Replicated tensors are not supported in this implementation.
        """
        # Use ring attention for diffusion (no KV cache needed)
        if req is None:
            scale = self.scaling if self.scaling is not None else self.softmax_scale
            return ring_attention(
                query,
                key,
                value,
                scale=scale,
                causal=self.causal,
                block_q=self.block_q,
                block_k=self.block_k,
                compute_dtype=self.compute_dtype,
            )

        # TODO refactor flashattention backend
        return req.attention_backend(query, key, value, self, None, None, 0)
