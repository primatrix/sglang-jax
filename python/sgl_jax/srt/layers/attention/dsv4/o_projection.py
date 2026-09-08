"""A1 -- DeepSeek-V4's grouped output projection (``wo_a`` / ``wo_b``).

V4's attention output does not go through a single ``o_proj``. It goes through an
**inverse RoPE, then a per-group LoRA down-projection, then one shared
up-projection**:

    x   : [T, n_heads, head_dim]        attention output
    x   = inverse_rope(x, positions)    # trailing rope_head_dim, GPT-J interleaved
    x   = x.reshape(T, G, H * head_dim) # G groups of H heads
    y   = einsum("tgd,gdr->tgr", x, wo_a)   # per-group, no cross-group mixing
    out = y.reshape(T, G * R) @ wo_b.T      # [T, hidden]

Shapes for Flash 0731, read off the checkpoint rather than inferred::

    wo_a.weight  [8192, 4096]  = [G*R, H*head_dim] = [8*1024, 8*512]
    wo_b.weight  [4096, 8192]  = [hidden, G*R]

so ``o_lora_rank`` is 1024 **per group**, not 1024 in total, and ``o_groups=8``
partitions the 64 heads into 8 groups of 8. There is no cross-group mixing until
``wo_b``.

Sources: upstream vLLM `vllm/models/deepseek_v4/attention.py` builds ``wo_a`` with
``is_bmm=True`` and ``bmm_batch_size = n_local_groups``, i.e. a batched per-group
matmul, and comments the step as "Inverse-RoPE + wo_a + wo_b output projection".
tpu-inference's `kernels/experimental/deepseek_v4/o_projection.py` states the same
contraction as ``einsum("tgd,dgr->tgr", ...)`` and fuses the inverse RoPE into it.

DeepSeek-V3's plain ``o_proj`` cannot be reused for any of this, which is why
nothing in this repo covered it.

This is native JAX. The fused Pallas version is a later optimisation; the fusion in
tpu-inference exists to avoid materialising the un-rotated activation and to keep
both MXU operands fp8, neither of which changes the arithmetic.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from sgl_jax.srt.layers.attention.dsv4.rope import apply_dsv4_partial_rope

__all__ = [
    "STANDARD_HEADS_PER_GROUP",
    "group_wo_a",
    "grouped_output_projection",
]

# tpu-inference's kernel asserts heads_per_group == 8 because it equals the TPU
# sublane count, which is what makes its tiling simple. Flash 0731 satisfies it
# (64 heads / 8 groups). Not required by the arithmetic here, but worth naming so a
# config that violates it is recognised as a kernel problem rather than a maths one.
STANDARD_HEADS_PER_GROUP = 8


def group_wo_a(wo_a, *, num_groups: int, out_sharding=None):
    """Reshape checkpoint ``wo_a`` ``[G*R, D]`` into ``[G, D, R]``.

    The checkpoint's output index is ``g * R + r`` -- group-major -- which is what
    lets the reshape be a plain view plus one transpose. Getting this backwards
    (``[R, G, D]``) silently mixes groups and still produces the right *shape*, so it
    is worth having in one named place with a test.
    """
    wo_a = jnp.asarray(wo_a)
    if wo_a.ndim != 2:
        raise ValueError(f"wo_a must be 2-D [G*R, D], got {wo_a.shape}")
    out_features, reduction = wo_a.shape
    if out_features % num_groups:
        raise ValueError(
            f"wo_a output width {out_features} is not divisible by o_groups {num_groups}"
        )
    lora_rank = out_features // num_groups
    grouped = jax.lax.reshape(wo_a, (num_groups, lora_rank, reduction), out_sharding=out_sharding)
    return jnp.transpose(grouped, (0, 2, 1))


def grouped_output_projection(
    attn_out,
    *,
    wo_a,
    wo_b,
    cos,
    sin,
    num_groups: int,
    rope_head_dim: int,
    apply_inverse_rope: bool = True,
):
    """Project attention output back to hidden width.

    Args:
      attn_out: ``[T, n_heads, head_dim]`` float array, the attention output.
      wo_a: ``[G*R, H*head_dim]`` in checkpoint layout.
      wo_b: ``[hidden, G*R]`` in checkpoint layout.
      cos, sin: ``[T, rope_head_dim // 2]`` for the inverse rotation.
      num_groups: ``o_groups``.
      apply_inverse_rope: leave False only to inspect the projection in isolation;
        production always inverts, because the attention output is still rotated.

    Returns:
      ``[T, hidden]`` float32.
    """
    attn_out = jnp.asarray(attn_out, jnp.float32)
    if attn_out.ndim != 3:
        raise ValueError(f"attn_out must be [T, n_heads, head_dim], got {attn_out.shape}")
    num_tokens, num_heads, head_dim = attn_out.shape
    if num_heads % num_groups:
        raise ValueError(f"{num_heads} heads do not split into {num_groups} groups")
    heads_per_group = num_heads // num_groups

    if apply_inverse_rope:
        attn_out = apply_dsv4_partial_rope(
            attn_out, cos[:, None, :], sin[:, None, :], rope_head_dim=rope_head_dim, inverse=True
        )

    grouped = group_wo_a(wo_a, num_groups=num_groups)  # [G, D, R]
    reduction = heads_per_group * head_dim
    if grouped.shape[1] != reduction:
        raise ValueError(
            f"wo_a reduction {grouped.shape[1]} != heads_per_group*head_dim {reduction}"
        )

    x = attn_out.reshape(num_tokens, num_groups, reduction)
    # Per group, no cross-group term: group g sees only its own heads and its own
    # slice of wo_a. Mixing here would be invisible in the output shape.
    y = jnp.einsum("tgd,gdr->tgr", x, grouped.astype(jnp.float32))

    wo_b = jnp.asarray(wo_b, jnp.float32)
    flat = y.reshape(num_tokens, -1)
    if wo_b.shape[1] != flat.shape[1]:
        raise ValueError(f"wo_b input width {wo_b.shape[1]} != G*R {flat.shape[1]}")
    return flat @ wo_b.T
