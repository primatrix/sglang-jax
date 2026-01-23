import math

import jax
import jax.numpy as jnp
from flax import nnx


def simple_attention(query, key, value, scale=None, causal=False):
    """Simple dot-product attention for diffusion models (no KV cache).

    Args:
        query: [B, S, H, D]
        key: [B, S, H, D]
        value: [B, S, H, D]
        scale: softmax scale, default 1/sqrt(D)
        causal: whether to apply causal mask
    Returns:
        output: [B, S, H, D]
    """
    if scale is None:
        scale = 1.0 / math.sqrt(query.shape[-1])

    # [B, H, S, D]
    q = jnp.transpose(query, (0, 2, 1, 3))
    k = jnp.transpose(key, (0, 2, 1, 3))
    v = jnp.transpose(value, (0, 2, 1, 3))

    # [B, H, S, S]
    attn_weights = jnp.einsum("bhsd,bhtd->bhst", q, k) * scale

    if causal:
        seq_len = query.shape[1]
        mask = jnp.tril(jnp.ones((seq_len, seq_len)))
        attn_weights = jnp.where(mask == 0, float("-inf"), attn_weights)

    attn_weights = jax.nn.softmax(attn_weights, axis=-1)

    # [B, H, S, D]
    output = jnp.einsum("bhst,bhtd->bhsd", attn_weights, v)

    # [B, S, H, D]
    return jnp.transpose(output, (0, 2, 1, 3))


def ring_attention(
    query,
    key,
    value,
    scale=None,
    causal=False,
    *,
    block_q: int = 128,
    block_k: int = 128,
):
    """Block-wise attention with online softmax (FlashAttention-style).

    Args:
        query: [B, S, H, D]
        key: [B, S, H, D]
        value: [B, S, H, D]
        scale: softmax scale, default 1/sqrt(D)
        causal: whether to apply causal mask
        block_q: query block size
        block_k: key/value block size
    Returns:
        output: [B, S, H, D]
    """
    if scale is None:
        scale = 1.0 / math.sqrt(query.shape[-1])

    # [B, H, S, D]
    q = jnp.transpose(query, (0, 2, 1, 3))
    k = jnp.transpose(key, (0, 2, 1, 3))
    v = jnp.transpose(value, (0, 2, 1, 3))

    b, h, s, d = q.shape
    num_q = (s + block_q - 1) // block_q
    num_k = (s + block_k - 1) // block_k
    s_q = num_q * block_q
    s_k = num_k * block_k

    # Pad to block multiples for safe slicing.
    if s_q != s:
        q = jnp.pad(q, ((0, 0), (0, 0), (0, s_q - s), (0, 0)))
    if s_k != s:
        k = jnp.pad(k, ((0, 0), (0, 0), (0, s_k - s), (0, 0)))
        v = jnp.pad(v, ((0, 0), (0, 0), (0, s_k - s), (0, 0)))

    q = q.astype(jnp.float32)
    k = k.astype(jnp.float32)
    v = v.astype(jnp.float32)
    scale = jnp.asarray(scale, dtype=jnp.float32)

    arange_q = jnp.arange(block_q)
    arange_k = jnp.arange(block_k)

    def q_loop(q_idx):
        q_start = q_idx * block_q
        q_blk = jax.lax.dynamic_slice(q, (0, 0, q_start, 0), (b, h, block_q, d))
        q_pos = q_start + arange_q
        q_valid = q_pos < s

        # Online softmax accumulators.
        m = jnp.full((b, h, block_q), -1.0e9, dtype=jnp.float32)
        l = jnp.zeros((b, h, block_q), dtype=jnp.float32)
        acc = jnp.zeros((b, h, block_q, d), dtype=jnp.float32)

        def k_loop(k_idx, carry):
            m_blk, l_blk, acc_blk = carry
            k_start = k_idx * block_k
            k_blk = jax.lax.dynamic_slice(k, (0, 0, k_start, 0), (b, h, block_k, d))
            v_blk = jax.lax.dynamic_slice(v, (0, 0, k_start, 0), (b, h, block_k, d))
            k_pos = k_start + arange_k
            k_valid = k_pos < s

            # [B, H, Q, K]
            scores = jnp.einsum("bhqd,bhkd->bhqk", q_blk, k_blk) * scale

            # Build mask: valid kv + optional causal.
            mask = (q_valid[:, None] & k_valid[None, :])  # [Q, K]
            if causal:
                causal_mask = k_pos[None, :] <= q_pos[:, None]
                mask = mask & causal_mask

            mask = mask[None, None, :, :]  # [1, 1, Q, K]
            scores = jnp.where(mask, scores, -1.0e9)

            block_max = jnp.max(scores, axis=-1)
            m_new = jnp.maximum(m_blk, block_max)
            exp_m = jnp.exp(m_blk - m_new)
            exp_scores = jnp.exp(scores - m_new[..., None]) * mask.astype(jnp.float32)

            l_new = l_blk * exp_m + jnp.sum(exp_scores, axis=-1)
            acc_new = acc_blk * exp_m[..., None] + jnp.einsum(
                "bhqk,bhkd->bhqd", exp_scores, v_blk
            )
            return (m_new, l_new, acc_new)

        m, l, acc = jax.lax.fori_loop(0, num_k, k_loop, (m, l, acc))

        l_safe = jnp.where(q_valid[None, None, :], l, 1.0)
        out_blk = acc / l_safe[..., None]
        out_blk = out_blk * q_valid[None, None, :, None]
        return out_blk

    out_blocks = jax.lax.map(q_loop, jnp.arange(num_q))
    # [num_q, B, H, block_q, D] -> [B, H, num_q, block_q, D]
    out = jnp.transpose(out_blocks, (1, 2, 0, 3, 4))
    out = out.reshape(b, h, s_q, d)
    out = out[:, :, :s, :]
    out = jnp.transpose(out, (0, 2, 1, 3))
    return out.astype(query.dtype)


class RingAttention(nnx.Module):
    """
    Ring Attention for Ulysses Sequence Parallelism.

    This class implements the Ring Attention for fine-grained sequence parallelism within subgroups.

    # FIXME(pc) we will implement above features later. For now, this is a naive attentnion implementation.
    """

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
        self.use_ring_attention = extra_impl_args.get("use_ring_attention", True)
        self.ring_block_q = extra_impl_args.get("ring_block_q", 128)
        self.ring_block_k = extra_impl_args.get("ring_block_k", 128)

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
        # Use ring/block attention for diffusion (no KV cache needed)
        if req is None:
            q_sharding = getattr(query, "sharding", None)
            if q_sharding is not None and getattr(q_sharding, "mesh", None) is not None:
                # Ensure replicated sharding to avoid dynamic_update_slice mismatches.
                no_shard = jax.sharding.NamedSharding(
                    q_sharding.mesh, jax.sharding.PartitionSpec()
                )
                query = jax.lax.with_sharding_constraint(query, no_shard)
                key = jax.lax.with_sharding_constraint(key, no_shard)
                value = jax.lax.with_sharding_constraint(value, no_shard)
            if self.use_ring_attention:
                return ring_attention(
                    query,
                    key,
                    value,
                    self.softmax_scale,
                    self.causal,
                    block_q=self.ring_block_q,
                    block_k=self.ring_block_k,
                )
            return simple_attention(query, key, value, self.softmax_scale, self.causal)

        # TODO refactor flashattention backend
        return req.attention_backend(query, key, value, self, None, None, 0)
