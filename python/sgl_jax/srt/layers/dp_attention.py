"""
Data Parallel Attention for SGLang-JAX

This module implements data parallel attention mechanism optimized for JAX/TPU,
inspired by the PyTorch implementation but redesigned for JAX's distributed computing model.
"""

from __future__ import annotations

import functools
import logging
from dataclasses import dataclass
from enum import IntEnum, auto
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jax.tree_util import register_pytree_node_class

from sgl_jax.srt.layers.attention.base_attn_backend import AttentionBackend
from sgl_jax.srt.layers.radix_attention import RadixAttention
from sgl_jax.srt.utils.jax_utils import device_array

if TYPE_CHECKING:
    from sgl_jax.srt.managers.schedule_batch import ModelWorkerBatch
    from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)

# Global state for DP attention configuration
_DP_ATTENTION_CONFIG: Optional[DpAttentionConfig] = None


class DpPaddingMode(IntEnum):
    """Mode for handling data gathering in DP attention."""

    # Pad tokens to max length and use all_gather
    MAX_LEN = auto()
    # Pad tokens to sum length and use all_reduce
    SUM_LEN = auto()

    def is_max_len(self) -> bool:
        return self == DpPaddingMode.MAX_LEN

    def is_sum_len(self) -> bool:
        return self == DpPaddingMode.SUM_LEN

    @classmethod
    def get_dp_padding_mode(
        cls, global_num_tokens: List[int], dp_size: int
    ) -> DpPaddingMode:
        """Choose the optimal mode based on communication cost."""
        max_len = max(global_num_tokens)
        sum_len = sum(global_num_tokens)
        # Use all_reduce (SUM_LEN) if it's more efficient
        if sum_len * 2 > max_len * dp_size:
            return cls.MAX_LEN
        else:
            return cls.SUM_LEN

    @classmethod
    def get_default_mode_for_cuda_graph(cls) -> DpPaddingMode:
        """Default mode for CUDA graph compatibility."""
        return cls.MAX_LEN


@register_pytree_node_class
@dataclass
class DpAttentionConfig:
    """Configuration for data parallel attention."""

    # Data parallel size
    dp_size: int
    # Tensor parallel size within each DP group
    tp_size: int
    # Current DP rank
    dp_rank: int
    # Current TP rank within DP group
    tp_rank: int
    # Hidden size
    hidden_size: int
    # Device mesh for sharding
    mesh: Mesh
    # Whether DP attention is enabled
    enabled: bool = True

    def tree_flatten(self):
        children = ()
        aux_data = {
            "dp_size": self.dp_size,
            "tp_size": self.tp_size,
            "dp_rank": self.dp_rank,
            "tp_rank": self.tp_rank,
            "hidden_size": self.hidden_size,
            "enabled": self.enabled,
            "mesh": self.mesh,
        }
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(**aux_data)


@register_pytree_node_class
@dataclass
class DpAttentionMetadata:
    """Metadata for DP attention forward pass."""

    # Number of tokens on each DP rank
    global_num_tokens: jax.Array
    # Cumulative token counts for indexing
    cumulative_tokens: jax.Array
    # Current batch's padding mode
    padding_mode: DpPaddingMode
    # Local token range
    local_start_pos: int
    local_num_tokens: int
    # Buffer lengths
    global_buffer_len: int
    local_buffer_len: int

    def tree_flatten(self):
        children = (
            self.global_num_tokens,
            self.cumulative_tokens,
        )
        aux_data = {
            "padding_mode": self.padding_mode,
            "local_start_pos": self.local_start_pos,
            "local_num_tokens": self.local_num_tokens,
            "global_buffer_len": self.global_buffer_len,
            "local_buffer_len": self.local_buffer_len,
        }
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(
            global_num_tokens=children[0], cumulative_tokens=children[1], **aux_data
        )


def initialize_dp_attention(
    dp_size: int,
    tp_size: int,
    dp_rank: int,
    tp_rank: int,
    hidden_size: int,
    mesh: Mesh,
) -> None:
    """Initialize global DP attention configuration."""
    global _DP_ATTENTION_CONFIG
    _DP_ATTENTION_CONFIG = DpAttentionConfig(
        dp_size=dp_size,
        tp_size=tp_size,
        dp_rank=dp_rank,
        tp_rank=tp_rank,
        hidden_size=hidden_size,
        mesh=mesh,
        enabled=True,
    )
    logger.info(f"DP Attention initialized: dp_size={dp_size}, tp_size={tp_size}")


def get_dp_attention_config() -> DpAttentionConfig:
    """Get the global DP attention configuration."""
    if _DP_ATTENTION_CONFIG is None:
        raise RuntimeError(
            "DP attention not initialized. Call initialize_dp_attention() first."
        )
    return _DP_ATTENTION_CONFIG


def is_dp_attention_enabled() -> bool:
    """Check if DP attention is enabled."""
    return _DP_ATTENTION_CONFIG is not None and _DP_ATTENTION_CONFIG.enabled


@functools.partial(jax.jit, static_argnames=["padding_mode"])
def dp_gather_tokens(
    global_buffer: jax.Array,
    local_tokens: jax.Array,
    metadata: DpAttentionMetadata,
    padding_mode: DpPaddingMode,
    is_partial: bool = False,
) -> jax.Array:
    """Gather tokens from all DP ranks into global buffer."""
    config = get_dp_attention_config()

    if padding_mode.is_max_len():
        return _dp_gather_via_all_gather(
            global_buffer, local_tokens, metadata, is_partial
        )
    else:
        return _dp_gather_via_all_reduce(
            global_buffer, local_tokens, metadata, is_partial
        )


def _dp_gather_via_all_gather(
    global_buffer: jax.Array,
    local_tokens: jax.Array,
    metadata: DpAttentionMetadata,
    is_partial: bool,
) -> jax.Array:
    """Gather using all_gather for MAX_LEN mode."""
    config = get_dp_attention_config()

    # For JAX, we use shard_map for efficient gathering
    mesh = config.mesh
    gather_spec = P("data", None)  # Gather along data parallel axis

    def gather_fn(local_data):
        return lax.all_gather(local_data, axis_name="data", axis=0)

    # Use shard_map for zero-copy gathering
    sharded_fn = jax.shard_map(
        gather_fn,
        mesh=mesh,
        in_specs=gather_spec,
        out_specs=P(None, None),
        check_rep=False,
    )

    return sharded_fn(local_tokens)


def _dp_gather_via_all_reduce(
    global_buffer: jax.Array,
    local_tokens: jax.Array,
    metadata: DpAttentionMetadata,
    is_partial: bool,
) -> jax.Array:
    """Gather using all_reduce for SUM_LEN mode."""
    config = get_dp_attention_config()

    # Zero out global buffer
    global_buffer = jnp.zeros_like(global_buffer)

    # Place local tokens at correct positions
    start_pos = metadata.local_start_pos
    num_tokens = metadata.local_num_tokens

    if local_tokens.shape[0] > 0 and (is_partial or config.tp_rank == 0):
        global_buffer = global_buffer.at[start_pos : start_pos + num_tokens].set(
            local_tokens
        )

    # All-reduce across data parallel group
    mesh = config.mesh
    reduce_spec = P("data", None)

    def reduce_fn(data):
        return lax.psum(data, axis_name="data")

    sharded_fn = jax.shard_map(
        reduce_fn,
        mesh=mesh,
        in_specs=reduce_spec,
        out_specs=reduce_spec,
        check_rep=False,
    )

    return sharded_fn(global_buffer)


@functools.partial(jax.jit, static_argnames=[])
def dp_scatter_tokens(
    global_tokens: jax.Array,
    metadata: DpAttentionMetadata,
) -> jax.Array:
    """Scatter global tokens back to local buffer."""
    start_pos = metadata.local_start_pos
    num_tokens = metadata.local_num_tokens

    # Extract local portion
    return global_tokens[start_pos : start_pos + num_tokens]


def compute_dp_local_info(
    global_num_tokens: jax.Array,
    dp_rank: int,
) -> Tuple[int, int]:
    """Compute local start position and number of tokens."""
    cumtokens = jnp.cumsum(global_num_tokens, axis=0)

    if dp_rank == 0:
        local_start_pos = 0
    else:
        local_start_pos = cumtokens[dp_rank - 1]

    local_num_tokens = global_num_tokens[dp_rank]

    return int(local_start_pos), int(local_num_tokens)


@register_pytree_node_class
class DpAttentionBackend(AttentionBackend):
    """Data Parallel Attention Backend for JAX/TPU."""

    def __init__(
        self,
        base_backend: AttentionBackend,
        config: DpAttentionConfig,
    ):
        self.base_backend = base_backend
        self.config = config
        self.global_buffer = None
        self.local_buffer = None

    def tree_flatten(self):
        children = (self.base_backend, self.config)
        aux_data = {}
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(children[0], children[1])

    def get_forward_metadata(self, batch: ModelWorkerBatch) -> DpAttentionMetadata:
        """Prepare metadata for DP attention forward pass."""
        # Get global token counts from all DP ranks
        local_num_tokens = len(batch.input_ids) if hasattr(batch, "input_ids") else 0

        # Gather token counts from all DP ranks
        mesh = self.config.mesh
        global_num_tokens = jnp.array([local_num_tokens], dtype=jnp.int32)

        def gather_token_counts(local_count):
            return lax.all_gather(local_count, axis_name="data")

        gather_fn = jax.shard_map(
            gather_token_counts,
            mesh=mesh,
            in_specs=P("data"),
            out_specs=P(None),
            check_rep=False,
        )

        global_num_tokens = gather_fn(global_num_tokens)

        # Choose padding mode
        padding_mode = DpPaddingMode.get_dp_padding_mode(
            global_num_tokens.tolist(), self.config.dp_size
        )

        # Compute local info
        local_start_pos, local_num_tokens = compute_dp_local_info(
            global_num_tokens, self.config.dp_rank
        )

        # Compute buffer lengths
        if padding_mode.is_max_len():
            global_buffer_len = int(jnp.max(global_num_tokens) * self.config.dp_size)
        else:
            global_buffer_len = int(jnp.sum(global_num_tokens))

        local_buffer_len = int(global_num_tokens[self.config.dp_rank])

        cumulative_tokens = jnp.cumsum(global_num_tokens)

        return DpAttentionMetadata(
            global_num_tokens=global_num_tokens,
            cumulative_tokens=cumulative_tokens,
            padding_mode=padding_mode,
            local_start_pos=local_start_pos,
            local_num_tokens=local_num_tokens,
            global_buffer_len=global_buffer_len,
            local_buffer_len=local_buffer_len,
        )

    def __call__(
        self,
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        **kwargs,
    ) -> Tuple[jax.Array, jax.Array]:
        """Forward pass with data parallel attention."""

        if not self.config.enabled:
            # Fall back to base backend
            return self.base_backend(q, k, v, layer, forward_batch, **kwargs)

        # Get DP metadata
        batch = kwargs.get("batch")  # Assuming batch is passed in kwargs
        if batch is None:
            # Extract from forward_batch if available
            metadata = self._create_default_metadata(q.shape[0])
        else:
            metadata = self.get_forward_metadata(batch)

        # Prepare global buffers
        global_q = self._get_global_buffer(q.shape, metadata.global_buffer_len)
        global_k = self._get_global_buffer(k.shape, metadata.global_buffer_len)
        global_v = self._get_global_buffer(v.shape, metadata.global_buffer_len)

        # Gather Q, K, V from all DP ranks
        global_q = dp_gather_tokens(
            global_q, q, metadata, metadata.padding_mode, is_partial=True
        )
        global_k = dp_gather_tokens(
            global_k, k, metadata, metadata.padding_mode, is_partial=False
        )
        global_v = dp_gather_tokens(
            global_v, v, metadata, metadata.padding_mode, is_partial=False
        )

        # Run attention on gathered tokens
        # Create a modified forward_batch for global attention
        global_forward_batch = self._create_global_forward_batch(
            forward_batch, metadata
        )

        global_output, kv_fused = self.base_backend(
            global_q, global_k, global_v, layer, global_forward_batch, **kwargs
        )

        # Scatter output back to local
        local_output = dp_scatter_tokens(global_output, metadata)

        return local_output, kv_fused

    def _get_global_buffer(
        self, local_shape: Tuple[int, ...], global_len: int
    ) -> jax.Array:
        """Get or create global buffer with appropriate shape."""
        global_shape = (global_len,) + local_shape[1:]
        return jnp.zeros(
            global_shape,
            dtype=local_shape.dtype if hasattr(local_shape, "dtype") else jnp.float32,
        )

    def _create_default_metadata(self, num_tokens: int) -> DpAttentionMetadata:
        """Create default metadata when batch info is not available."""
        global_num_tokens = jnp.full(self.config.dp_size, num_tokens, dtype=jnp.int32)
        cumulative_tokens = jnp.cumsum(global_num_tokens)

        local_start_pos, local_num_tokens = compute_dp_local_info(
            global_num_tokens, self.config.dp_rank
        )

        return DpAttentionMetadata(
            global_num_tokens=global_num_tokens,
            cumulative_tokens=cumulative_tokens,
            padding_mode=DpPaddingMode.MAX_LEN,
            local_start_pos=local_start_pos,
            local_num_tokens=local_num_tokens,
            global_buffer_len=num_tokens * self.config.dp_size,
            local_buffer_len=num_tokens,
        )

    def _create_global_forward_batch(
        self, local_batch: ForwardBatch, metadata: DpAttentionMetadata
    ) -> ForwardBatch:
        """Create global forward batch for attention computation."""
        # This would need to be implemented based on the specific
        # requirements of how ForwardBatch handles global vs local data
        # For now, we'll return the local batch as a placeholder
        return local_batch

    @staticmethod
    def get_max_running_reqests(max_context_len: int, page_size: int) -> int:
        """Get maximum running requests for DP attention backend."""
        # Delegate to base backend implementation
        # This would need to be adjusted based on DP constraints
        return max_context_len // page_size


def create_dp_attention_backend(
    base_backend: AttentionBackend,
    dp_size: int = 1,
    tp_size: int = 1,
    dp_rank: int = 0,
    tp_rank: int = 0,
    hidden_size: int = 4096,
    mesh: Optional[Mesh] = None,
) -> AttentionBackend:
    """Factory function to create DP attention backend."""

    if dp_size <= 1 or mesh is None:
        # No DP needed, return base backend
        return base_backend

    # Initialize DP attention if not already done
    if not is_dp_attention_enabled():
        initialize_dp_attention(dp_size, tp_size, dp_rank, tp_rank, hidden_size, mesh)

    config = get_dp_attention_config()
    return DpAttentionBackend(base_backend, config)
