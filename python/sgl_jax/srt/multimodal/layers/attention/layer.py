import math

import jax
import jax.experimental.pallas as pl
import jax.numpy as jnp
from flax import nnx

from sgl_jax.srt.multimodal.kernels.flash_attention import SegmentIds
from sgl_jax.srt.multimodal.layers.attention.flash_attention_backend import (
    FlashAttentionBackend,
)


def align_to(x, a):
    return pl.cdiv(x, a) * a


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


class USPAttention(nnx.Module):
    """
    TPU-optimized Parallel Attention with SPMD sharding.

    This class implements efficient attention for TPU using JAX SPMD (Single Program Multiple Data)
    with the following parallelism strategies:

    1. Tensor Parallelism (TP): Shards attention heads across TPU chips
       - Q/K/V are sharded along head dimension: PartitionSpec("data", "tensor", None, None)
       - Each chip computes attention for a subset of heads
       - All-reduce after attention to gather results

    2. Data Parallelism: Shards batch dimension across TPU chips
       - Batch dimension uses "data" axis in PartitionSpec

    3. Flash Attention: Uses Pallas kernels optimized for TPU
       - Tiled computation with block size 128x128 (aligned to TPU MXU)
       - Supports dynamic sequence lengths via padding + segment_ids masking
       - Memory-efficient online softmax computation

    The implementation automatically detects mesh configuration and applies appropriate
    sharding constraints. When mesh=None, falls back to replicated computation.

    Args:
        num_heads: Number of attention heads
        head_size: Dimension of each attention head
        num_kv_heads: Number of key/value heads (for GQA, defaults to num_heads)
        softmax_scale: Scale factor for attention scores (defaults to 1/sqrt(head_size))
        causal: Whether to apply causal masking
        dropout_rate: Dropout rate (not used in inference)
        layer_id: Layer index for debugging
        logit_cap: Optional logit capping value
        scaling: Optional additional scaling factor
        mesh: JAX device mesh for SPMD sharding. If None, uses replicated computation.
        **extra_impl_args: Additional implementation-specific arguments

    Shape conventions:
        - Input Q/K/V: [B, S, H, D] (batch, sequence, heads, head_dim)
        - After transpose for backend: [B, H, S, D]
        - Output: [B, S, H, D]

    Performance notes:
        - On TPU v5e (8 chips): Expect 2-4x speedup vs single chip
        - Sequence length must be padded to multiples of 128 for optimal performance
        - Head dimension should be 64 or 128 for best MXU utilization
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
        mesh: jax.sharding.Mesh | None = None,
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
        self.mesh = mesh

        # Initialize Flash Attention backend with SPMD sharding
        # The backend already handles sharding via shard_map with PartitionSpec
        self.attention_backend = FlashAttentionBackend(
            mesh=self.mesh, sm_scale=self.softmax_scale, causal=False
        )

    def _apply_tensor_parallel_sharding(
        self, tensor: jax.Array, axis_name: str = "tensor"
    ) -> jax.Array:
        """
        Apply tensor parallel sharding constraint to Q/K/V tensors.

        Shards along the head dimension (axis 1 after transpose to [B, H, S, D]).
        This enables each TPU chip to compute attention for a subset of heads.

        Args:
            tensor: Input tensor of shape [B, H, S, D]
            axis_name: Mesh axis name for sharding (default: "tensor")

        Returns:
            Tensor with sharding constraint applied
        """
        if self.mesh is None:
            return tensor

        # Shard along head dimension: PartitionSpec("data", "tensor", None, None)
        # - "data": batch dimension (for data parallelism)
        # - "tensor": head dimension (for tensor parallelism)
        # - None: sequence and head_dim dimensions are replicated within each chip
        sharding_spec = jax.sharding.PartitionSpec("data", axis_name, None, None)
        sharding = jax.sharding.NamedSharding(self.mesh, sharding_spec)
        return jax.lax.with_sharding_constraint(tensor, sharding)

    def _apply_output_sharding(self, tensor: jax.Array) -> jax.Array:
        """
        Apply sharding constraint to attention output.

        After attention computation, output is sharded as [B, H, S, D] with heads distributed.
        This maintains the same sharding as input for efficient downstream processing.

        Args:
            tensor: Attention output of shape [B, H, S, D]

        Returns:
            Tensor with output sharding constraint applied
        """
        if self.mesh is None:
            return tensor

        # Keep same sharding as input: heads are distributed
        sharding_spec = jax.sharding.PartitionSpec("data", "tensor", None, None)
        sharding = jax.sharding.NamedSharding(self.mesh, sharding_spec)
        return jax.lax.with_sharding_constraint(tensor, sharding)

    def _pad_to_block_size(
        self, tensor: jax.Array, seq_axis: int, block_size: int = 128
    ) -> tuple[jax.Array, int]:
        """
        Pad sequence dimension to multiple of block_size for optimal TPU performance.

        TPU MXU (Matrix Multiply Unit) operates on 128x128 tiles. Padding ensures
        efficient utilization of hardware resources.

        Args:
            tensor: Input tensor with sequence dimension
            seq_axis: Axis index of sequence dimension
            block_size: Target block size (default: 128 for TPU)

        Returns:
            Tuple of (padded_tensor, original_length)
        """
        seq_len = tensor.shape[seq_axis]
        aligned_len = align_to(seq_len, block_size)

        if seq_len == aligned_len:
            return tensor, seq_len

        # Create padding specification
        pad_width = [(0, 0)] * tensor.ndim
        pad_width[seq_axis] = (0, aligned_len - seq_len)

        padded = jnp.pad(tensor, pad_width)
        return padded, seq_len

    def _create_segment_ids(
        self, batch_size: int, seq_len: int, padded_len: int
    ) -> jax.Array:
        """
        Create segment IDs for masking padded tokens.

        Segment IDs mark valid tokens (1) vs padding (0). The Flash Attention kernel
        uses these to prevent attention to padding tokens.

        Args:
            batch_size: Batch size
            seq_len: Original sequence length (before padding)
            padded_len: Padded sequence length

        Returns:
            Segment IDs of shape [batch_size, padded_len]
        """
        # Valid tokens: 1, padding tokens: 0
        valid_mask = jnp.ones((batch_size, seq_len))
        padding_mask = jnp.zeros((batch_size, padded_len - seq_len))
        return jnp.concatenate([valid_mask, padding_mask], axis=1)

    def __call__(
        self,
        query: jax.Array,
        key: jax.Array,
        value: jax.Array,
        req=None,
    ) -> jax.Array:
        """
        Forward pass for TPU-optimized parallel attention.

        This method implements the following pipeline:
        1. Transpose Q/K/V from [B, S, H, D] to [B, H, S, D] for backend
        2. Pad sequence length to multiples of 128 for TPU efficiency
        3. Create segment IDs to mask padding tokens
        4. Apply SPMD sharding constraints for tensor parallelism
        5. Call Flash Attention backend (Pallas kernel)
        6. Unpad and transpose output back to [B, S, H, D]

        Args:
            query: Query tensor of shape [B, S, H, D]
            key: Key tensor of shape [B, S, H, D]
            value: Value tensor of shape [B, S, H, D]
            req: Optional request object (unused, for API compatibility)

        Returns:
            Attention output of shape [B, S, H, D]

        Shape transformations:
            Input:  [B, S, H, D]
            Transpose: [B, H, S, D]
            Pad: [B, H, S_aligned, D]
            Attention: [B, H, S_aligned, D]
            Unpad: [B, H, S, D]
            Transpose: [B, S, H, D]
        """
        batch_size = query.shape[0]
        original_q_len = query.shape[1]
        original_kv_len = key.shape[1]

        # Step 1: Transpose to [B, H, S, D] format expected by backend
        query = jnp.transpose(query, (0, 2, 1, 3))  # [B, S, H, D] -> [B, H, S, D]
        key = jnp.transpose(key, (0, 2, 1, 3))
        value = jnp.transpose(value, (0, 2, 1, 3))

        # Step 2: Pad sequence dimensions to multiples of 128
        # This ensures optimal TPU MXU utilization
        query, q_len = self._pad_to_block_size(query, seq_axis=2, block_size=128)
        key, kv_len = self._pad_to_block_size(key, seq_axis=2, block_size=128)
        value, _ = self._pad_to_block_size(value, seq_axis=2, block_size=128)

        # Step 3: Create segment IDs for masking padded tokens
        segment_ids = None
        if q_len != query.shape[2] or kv_len != key.shape[2]:
            seg_q = self._create_segment_ids(batch_size, q_len, query.shape[2])
            seg_kv = self._create_segment_ids(batch_size, kv_len, key.shape[2])
            segment_ids = SegmentIds(q=seg_q, kv=seg_kv)

        # Step 4: Apply SPMD sharding constraints for tensor parallelism
        # This distributes attention heads across TPU chips
        if self.mesh is not None:
            query = self._apply_tensor_parallel_sharding(query)
            key = self._apply_tensor_parallel_sharding(key)
            value = self._apply_tensor_parallel_sharding(value)

        # Step 5: Call Flash Attention backend
        # The backend uses jax.shard_map with Pallas kernels for efficient computation
        output = self.attention_backend(query, key, value, segment_ids)

        # Step 6: Apply output sharding constraint
        if self.mesh is not None:
            output = self._apply_output_sharding(output)

        # Step 7: Unpad to original sequence length
        output = output[:, :, :original_q_len, :]

        # Step 8: Transpose back to [B, S, H, D]
        return jnp.transpose(output, (0, 2, 1, 3))
