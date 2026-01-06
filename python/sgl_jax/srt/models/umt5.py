# Copyright 2026 SII Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Inference-only UMT5 model compatible with HuggingFace weights."""

import copy
import logging
import math
from typing import Any, Optional, Dict

import jax
import jax.numpy as jnp
from flax import nnx
from transformers import UMT5Config

from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.layers.embeddings import Embed, ParallelLMHead
from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.layers.layernorm import RMSNorm
from sgl_jax.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sgl_jax.srt.layers.radix_attention import RadixAttention
from sgl_jax.srt.mem_cache.memory_pool import KVCache
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
from sgl_jax.srt.utils.weight_utils import WeightLoader, WeightMapping

logger = logging.getLogger(__name__)


def fp16_clamp(x: jax.Array):
    """
    Prevents overflow in float16 by clamping values to the max representable range.
    Standard practice for T5/UMT5 models in lower precision.
    """
    if x.dtype == jnp.float16 and jnp.isinf(x).any():
        clamp = jnp.finfo(x.dtype).max - 1000
        x = jax.lax.clamp(x=x, min=-clamp, max=clamp)
    return x


def gelu_new(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Also see the paper: https://arxiv.org/abs/1606.08415
    
    Note: PyTorch's `func.gelu(tanh_approx=True)` matches this implementation:
    0.5 * x * (1.0 + tanh(sqrt(2.0 / pi) * (x + 0.044715 * x^3)))
    
    This is critical for exact numerical alignment with HuggingFace T5/UMT5 models.
    """
    return 0.5 * x * (1.0 + jnp.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * jnp.power(x, 3))))


ACT_FN = {
    "gelu": jax.nn.gelu,
    "gelu_new": gelu_new,
    "relu": jax.nn.relu,
}


class UMT5DenseGatedActDense(nnx.Module):
    """
    Gated-GELU Feed Forward Network used in newer T5 variants (e.g. UMT5, Flan-T5).
    Structure: (GeLU(x @ wi_0.T) * (x @ wi_1.T)) @ wo.T
    """
    def __init__(
        self,
        config: UMT5Config,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.wi_0 = LinearBase(
            input_size=config.d_model,
            output_size=config.d_ff,
            use_bias=False,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            mesh=mesh,
        )
        self.wi_1 = LinearBase(
            input_size=config.d_model,
            output_size=config.d_ff,
            use_bias=False,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            mesh=mesh,
        )
        self.wo = LinearBase(
            input_size=config.d_ff,
            output_size=config.d_model,
            use_bias=False,
            kernel_axes=("tensor", None),
            params_dtype=dtype,
            mesh=mesh,
        )
        self.dropout = nnx.Dropout(config.dropout_rate)
        self.act = ACT_FN.get(config.dense_act_fn, jax.nn.gelu)

    def __call__(self, hidden_states: jax.Array, deterministic: bool = False) -> jax.Array:
        hidden_gelu = self.act(self.wi_0(hidden_states)[0])
        hidden_linear = self.wi_1(hidden_states)[0]
        hidden_states = hidden_gelu * hidden_linear
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        hidden_states, _ = self.wo(hidden_states)
        return hidden_states


class UMT5DenseActDense(nnx.Module):
    """
    Standard Feed Forward Network (Linear -> Act -> Linear).
    Used in original T5 models.
    """
    def __init__(
        self,
        config: UMT5Config,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.wi = LinearBase(
            input_size=config.d_model,
            output_size=config.d_ff,
            use_bias=False,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            mesh=mesh,
        )
        self.wo = LinearBase(
            input_size=config.d_ff,
            output_size=config.d_model,
            use_bias=False,
            kernel_axes=("tensor", None),
            params_dtype=dtype,
            mesh=mesh,
        )
        self.dropout = nnx.Dropout(config.dropout_rate)
        self.act = ACT_FN.get(config.dense_act_fn, jax.nn.relu)

    def __call__(self, hidden_states: jax.Array, deterministic: bool = False) -> jax.Array:
        hidden_states, _ = self.wi(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        hidden_states, _ = self.wo(hidden_states)
        return hidden_states


class UMT5Attention(nnx.Module):
    """
    Multi-head attention module for UMT5 with support for:
    1. Self-Attention (Encoder/Decoder)
    2. Cross-Attention (Decoder)
    3. Relative Position Bias (T5-style)
    4. KV Cache optimization via RadixAttention (Decoder Self-Attention only)
    """
    def __init__(
        self,
        config: UMT5Config,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
        layer_idx: int = 0,
        is_cross_attention: bool = False,
        is_decoder: bool = False,
    ):
        self.is_decoder = is_decoder
        self.is_cross_attention = is_cross_attention
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.relative_attention_max_distance = getattr(config, "relative_attention_max_distance", 128)
        
        self.d_model = config.d_model
        self.key_value_proj_dim = config.d_kv
        self.n_heads = config.num_heads
        self.inner_dim = self.n_heads * self.key_value_proj_dim
        self.layer_idx = layer_idx

        # QKV projections
        self.q = LinearBase(
            input_size=self.d_model,
            output_size=self.inner_dim,
            use_bias=False,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            mesh=mesh,
        )
        self.k = LinearBase(
            input_size=self.d_model,
            output_size=self.inner_dim,
            use_bias=False,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            mesh=mesh,
        )
        self.v = LinearBase(
            input_size=self.d_model,
            output_size=self.inner_dim,
            use_bias=False,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            mesh=mesh,
        )
        self.o = LinearBase(
            input_size=self.inner_dim,
            output_size=self.d_model,
            use_bias=False,
            kernel_axes=("tensor", None),
            params_dtype=dtype,
            mesh=mesh,
        )

        # Relative Attention Bias
        # In T5/UMT5, partial position information is learned via these bias buckets.
        # This is strictly applied in Self-Attention layers.
        if not self.is_cross_attention:
            self.relative_attention_bias = Embed(
                self.relative_attention_num_buckets,
                self.n_heads,
                dtype=dtype,
                param_dtype=dtype,
                mesh=mesh,
                kernel_axes=(None, "tensor"),
            )
        
        self.dropout = nnx.Dropout(config.dropout_rate)

        # RadixAttention for Decoder Self-Attention with KV Cache optimization via PagedAttention
        if self.is_decoder and not self.is_cross_attention:
            self.radix_attn = RadixAttention(
                num_heads=self.n_heads,
                head_dim=self.key_value_proj_dim,
                scaling=1.0,  # T5 uses unscaled attention
                num_kv_heads=self.n_heads,
                layer_id=layer_idx,
            )

    def _relative_position_bucket(
        self, relative_position, bidirectional=True, num_buckets=32, max_distance=128
    ):
        """
        Adapted from HuggingFace T5 implementation.
        Computes bucket indices for relative positions.
        
        Key properties:
        1. Logarithmic bucketing for larger distances to capture long-range dependencies efficiently.
        2. Bidirectional (Encoder) uses separate buckets for past/future.
        3. Unidirectional (Decoder) only considers past.
        """
        ret = 0
        n = -relative_position
        if bidirectional:
            num_buckets //= 2
            ret += (n < 0).astype(jnp.int32) * num_buckets
            n = jnp.abs(n)
        else:
            n = jnp.maximum(n, 0)
        
        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
            jnp.log(n.astype(jnp.float32) / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).astype(jnp.int32)
        
        val_if_large = jnp.minimum(val_if_large, num_buckets - 1)
        return jnp.where(is_small, n, val_if_large) + ret

    def compute_bias(self, query_length, key_length, bidirectional=True):
        """
        Computes the relative attention bias matrix [1, n_heads, query_len, key_len].
        """
        context_position = jnp.arange(query_length, dtype=jnp.int32)[:, None]
        memory_position = jnp.arange(key_length, dtype=jnp.int32)[None, :]
        relative_position = memory_position - context_position
        
        rp_bucket = self._relative_position_bucket(
            relative_position,
            bidirectional=bidirectional,
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        
        values = self.relative_attention_bias(rp_bucket) 
        values = jnp.transpose(values, (2, 0, 1))[None, :, :, :]
        return values

    def __call__(
        self,
        hidden_states: jax.Array,
        mask: jax.Array | None = None,
        deterministic: bool = False,
        encoder_hidden_states: jax.Array | None = None,
        encoder_mask: jax.Array | None = None,
        forward_batch: ForwardBatch | None = None,
        token_to_kv_pool: KVCache | None = None,
    ) -> jax.Array:
        
        batch_size, seq_length = hidden_states.shape[:2]
        
        # Project queries
        q, _ = self.q(hidden_states)

        # Determine Key/Value sources based on attention type
        if self.is_cross_attention:
            # Cross-Attention: keys/values come from the Encoder Output
            if encoder_hidden_states is None:
                raise ValueError("encoder_hidden_states must be provided for cross attention")
            k, _ = self.k(encoder_hidden_states)
            v, _ = self.v(encoder_hidden_states)
            key_length = encoder_hidden_states.shape[1]
        else:
            # Self-Attention: keys/values come from the Input itself
            k, _ = self.k(hidden_states)
            v, _ = self.v(hidden_states)
            key_length = seq_length

        kv_fused = None
        
        # CASE 1: Decoder Self-Attention with KV Cache Optimization (RadixAttention)
        # This path is used during highly-optimized autoregressive decoding via SGLang runtime.
        if self.is_decoder and not self.is_cross_attention and forward_batch is not None and token_to_kv_pool is not None:
            q_flat = q.reshape(-1, self.n_heads, self.key_value_proj_dim)
            k_flat = k.reshape(-1, self.n_heads, self.key_value_proj_dim)
            v_flat = v.reshape(-1, self.n_heads, self.key_value_proj_dim)
            
            # Note: Relative Position Bias support in RadixAttention is handled implicitly 
            # or requires specific kernel support for T5-style bias.
            # Currently assuming basic masking and standard attention in fast path.
            attn_output, kv_fused = self.radix_attn(
                q_flat, k_flat, v_flat, 
                forward_batch, 
                token_to_kv_pool,
            )
            
            attn_output = attn_output.reshape(batch_size, seq_length, self.inner_dim)
            output, _ = self.o(attn_output)
            return output

        # CASE 2: Manual Attention Implementation (Standard / Training / Eager Mode)
        # Used for Encoder, Cross-Attention, or fallback when RadixAttention is not applicable.

        # Reshape for multi-head attention
        q = q.reshape(batch_size, seq_length, self.n_heads, self.key_value_proj_dim)
        k = k.reshape(batch_size, -1, self.n_heads, self.key_value_proj_dim)
        v = v.reshape(batch_size, -1, self.n_heads, self.key_value_proj_dim)
        
        q = jnp.transpose(q, (0, 2, 1, 3))  # [batch, n_heads, seq_q, head_dim]
        k = jnp.transpose(k, (0, 2, 1, 3))  # [batch, n_heads, seq_k, head_dim]
        v = jnp.transpose(v, (0, 2, 1, 3))  # [batch, n_heads, seq_k, head_dim]

        # Compute attention scores (T5 uses unscaled dot-product attention)
        q_f32 = q.astype(jnp.float32)
        k_f32 = k.astype(jnp.float32)
        scores = jnp.matmul(q_f32, jnp.swapaxes(k_f32, -1, -2))
        
        # Add Relative Position Bias (Self-Attention Only)
        if not self.is_cross_attention:
            bidirectional = not self.is_decoder
            position_bias = self.compute_bias(seq_length, k.shape[2], bidirectional=bidirectional)
            scores += position_bias.astype(jnp.float32)

        # Construct and apply attention masks
        active_mask = mask if not self.is_cross_attention else encoder_mask
        
        if self.is_decoder and not self.is_cross_attention:
             # Decoder Self-Attention requires a Causal Mask (lower triangular)
             causal_mask = jnp.tril(jnp.ones((seq_length, seq_length), dtype=jnp.bool_))
             causal_mask = causal_mask[None, None, :, :]  # [1, 1, seq, seq]
             
             if active_mask is not None:
                 if active_mask.ndim == 2:
                     # Combine Causal Mask with Padding Mask
                     active_mask = active_mask[:, None, None, :]
                     combined_mask = jnp.logical_and(causal_mask, active_mask)
                     active_mask = combined_mask
                 elif active_mask.ndim == 3:
                     active_mask = active_mask[:, None, :, :]
                     active_mask = jnp.logical_and(causal_mask, active_mask)
             else:
                 active_mask = causal_mask
        
        # Apply the final mask to scores
        if active_mask is not None:
            if active_mask.ndim == 2:
                active_mask = active_mask[:, None, None, :]
            elif active_mask.ndim == 3 and self.is_cross_attention:
                active_mask = active_mask[:, None, :, :]
            
            mask_value = jnp.finfo(scores.dtype).min
            scores = jnp.where(active_mask == 0, mask_value, scores)

        # Compute attention weights and output
        orig_dtype = scores.dtype
        attn_weights = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(orig_dtype)
        attn_weights = self.dropout(attn_weights, deterministic=deterministic)
        
        v_f32 = v.astype(jnp.float32)
        attn_output = jnp.matmul(attn_weights, v_f32)
        
        attn_output = jnp.transpose(attn_output, (0, 2, 1, 3))
        attn_output = attn_output.reshape(batch_size, seq_length, self.inner_dim)
        
        output, _ = self.o(attn_output)
        return output


class UMT5Block(nnx.Module):
    """
    Transformer block for UMT5 model.
    
    Structure:
    1. Self-Attention + LayerNorm + Residual
    2. Cross-Attention + LayerNorm + Residual (Decoder only)
    3. Feed-Forward + LayerNorm + Residual
    """
    def __init__(
        self,
        config: UMT5Config,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
        layer_idx: int = 0,
        is_decoder: bool = False,
    ):
        self.is_decoder = is_decoder
        
        # Self Attention sublayer
        self.layer0_LayerNorm = RMSNorm(
            config.d_model, 
            epsilon=config.layer_norm_epsilon, 
            dtype=dtype,
            param_dtype=dtype,
            use_scale=True,
        )
        self.layer0_SelfAttention = UMT5Attention(
            config, 
            mesh, 
            dtype=dtype,
            layer_idx=layer_idx,
            is_cross_attention=False,
            is_decoder=is_decoder,
        )
        self.layer0_dropout = nnx.Dropout(config.dropout_rate)
        
        # Cross Attention sublayer (Decoder only)
        if self.is_decoder:
            self.layer1_LayerNorm = RMSNorm(
                config.d_model, 
                epsilon=config.layer_norm_epsilon, 
                dtype=dtype,
                param_dtype=dtype,
                use_scale=True,
            )
            self.layer1_EncDecAttention = UMT5Attention(
                config,
                mesh,
                dtype=dtype,
                layer_idx=layer_idx,
                is_cross_attention=True,
                is_decoder=is_decoder,
            )
            self.layer1_dropout = nnx.Dropout(config.dropout_rate)
        
        # Feed Forward sublayer
        self.layer_FF_LayerNorm = RMSNorm(
            config.d_model, 
            epsilon=config.layer_norm_epsilon, 
            dtype=dtype,
            param_dtype=dtype,
            use_scale=True,
        )
        
        if config.is_gated_act:
            self.layer_FF_DenseReluDense = UMT5DenseGatedActDense(config, mesh, dtype=dtype)
        else:
            self.layer_FF_DenseReluDense = UMT5DenseActDense(config, mesh, dtype=dtype)
            
        self.layer_FF_dropout = nnx.Dropout(config.dropout_rate)
        
    def __call__(
        self, 
        hidden_states: jax.Array,
        mask: jax.Array | None = None,
        deterministic: bool = False,
        encoder_hidden_states: jax.Array | None = None,
        encoder_mask: jax.Array | None = None,
        forward_batch: ForwardBatch | None = None,
        token_to_kv_pool: KVCache | None = None,
    ) -> jax.Array:
        
        # Self Attention block
        normed_hidden_states = self.layer0_LayerNorm(hidden_states)
        attn_output = self.layer0_SelfAttention(
            normed_hidden_states, 
            mask=mask, 
            deterministic=deterministic,
            forward_batch=forward_batch,
            token_to_kv_pool=token_to_kv_pool,
        )
        attn_output = self.layer0_dropout(attn_output, deterministic=deterministic)
        hidden_states = hidden_states + attn_output
        hidden_states = fp16_clamp(hidden_states)
        
        # Cross Attention block (Decoder Only)
        if self.is_decoder and encoder_hidden_states is not None:
            normed_hidden_states = self.layer1_LayerNorm(hidden_states)
            attn_output = self.layer1_EncDecAttention(
                normed_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_mask=encoder_mask,
                deterministic=deterministic
            )
            attn_output = self.layer1_dropout(attn_output, deterministic=deterministic)
            hidden_states = hidden_states + attn_output
            hidden_states = fp16_clamp(hidden_states)

        # Feed Forward block
        normed_hidden_states = self.layer_FF_LayerNorm(hidden_states)
        mlp_output = self.layer_FF_DenseReluDense(normed_hidden_states, deterministic=deterministic)
        mlp_output = self.layer_FF_dropout(mlp_output, deterministic=deterministic)
        hidden_states = hidden_states + mlp_output
        hidden_states = fp16_clamp(hidden_states)
        
        return hidden_states


class UMT5Stack(nnx.Module):
    """
    Stack of UMT5 transformer blocks (Encoder or Decoder).
    """
    def __init__(
        self,
        config: UMT5Config,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.is_decoder = config.is_decoder
        
        self.block = nnx.List([
            UMT5Block(
                config, 
                mesh, 
                dtype=dtype,
                layer_idx=i,
                is_decoder=self.is_decoder,
            )
            for i in range(config.num_layers)
        ])
        
        self.final_layer_norm = RMSNorm(
            config.d_model, 
            epsilon=config.layer_norm_epsilon, 
            dtype=dtype,
            param_dtype=dtype,
            use_scale=True,
        )
        self.dropout = nnx.Dropout(config.dropout_rate)
    
    def share_relative_attention_bias(self):
        """
        Share relative_attention_bias from layer 0 to all other layers.
        
        In PyTorch/HuggingFace T5/UMT5, only block.0 has relative_attention_bias weights.
        In our JAX implementation, each layer has its own Embed layer for better parallelization.
        This method copies layer 0's bias to all other layers after loading weights.
        """
        if len(self.block) == 0:
            return
        
        # Get the bias from layer 0
        layer0_bias = self.block[0].layer0_SelfAttention.relative_attention_bias.embedding[...]
        
        # Copy to all other layers
        for i in range(1, len(self.block)):
            self.block[i].layer0_SelfAttention.relative_attention_bias.embedding[...] = layer0_bias
        
        logger.debug(f"Shared relative_attention_bias from layer 0 to {len(self.block)-1} other layers")

    def __call__(
        self, 
        hidden_states: jax.Array,
        mask: jax.Array | None = None,
        deterministic: bool = False,
        encoder_hidden_states: jax.Array | None = None,
        encoder_mask: jax.Array | None = None,
        forward_batch: ForwardBatch | None = None,
        token_to_kv_pool: KVCache | None = None,
    ) -> jax.Array:
        # Initial dropout on embeddings
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        
        # Pass through all transformer blocks
        for block in self.block:
            hidden_states = block(
                hidden_states, 
                mask=mask, 
                deterministic=deterministic,
                encoder_hidden_states=encoder_hidden_states,
                encoder_mask=encoder_mask,
                forward_batch=forward_batch,
                token_to_kv_pool=token_to_kv_pool,
            )
        
        # Final layer norm and dropout
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = fp16_clamp(hidden_states)
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        return hidden_states


class UMT5EncoderModel(nnx.Module):
    """
    UMT5 Encoder-only model.
    """
    def __init__(
        self,
        config: UMT5Config,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.config = config
        self.mesh = mesh
        self.dtype = dtype
        
        self.shared = Embed(
            config.vocab_size,
            config.d_model,
            dtype=dtype,
            param_dtype=dtype,
            mesh=mesh,
            kernel_axes=("tensor", None),
        )
        
        self.encoder = UMT5Stack(config, mesh, dtype=dtype)

    def load_weights(self, model_config: ModelConfig):
        """Load weights from HuggingFace checkpoint."""
        loader = WeightLoader(
            model=self,
            model_config=model_config,
            mesh=self.mesh,
            dtype=self.dtype,
        )
        weight_mappings = self._create_weight_mappings()
        loader.load_weights_from_safetensors(weight_mappings)
        
        # No need to share relative_attention_bias - each layer loads its own weights
        
        logger.info("UMT5Encoder weights loaded successfully!")
    
    def _create_weight_mappings(self) -> dict:
        """Create mappings from HuggingFace weight names to JAX model parameters."""
        mappings = {
            "shared.weight": WeightMapping(
                target_path="shared.embedding",
                sharding=("tensor", None),
                transpose=False,
            ),
            "encoder.final_layer_norm.weight": WeightMapping(
                target_path="encoder.final_layer_norm.scale",
                sharding=(None,),
                transpose=False,
            ),
        }

        for layer_idx in range(self.config.num_layers):
            mappings.update(_create_block_mapping_helper(self.config, layer_idx, is_decoder=False, prefix="encoder.block"))
        return mappings

    def __call__(
        self,
        input_ids: jax.Array,
        forward_batch: ForwardBatch | None = None, 
        token_to_kv_pool: Any | None = None,
        attention_mask: jax.Array | None = None,
        **kwargs
    ):
        x = forward_batch.input_ids if forward_batch is not None else input_ids
        mask = attention_mask if attention_mask is not None else kwargs.get("mask", None)
        deterministic = kwargs.get("deterministic", False)
            
        hidden_states = self.shared(x)
        hidden_states = self.encoder(hidden_states, mask=mask, deterministic=deterministic)
        return hidden_states


class UMT5DecoderModel(nnx.Module):
    """
    UMT5 Decoder-only model.
    """
    def __init__(
        self,
        config: UMT5Config,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.config = config
        self.mesh = mesh
        self.dtype = dtype
        
        self.shared = Embed(
            config.vocab_size,
            config.d_model,
            dtype=dtype,
            param_dtype=dtype,
            mesh=mesh,
            kernel_axes=("tensor", None),
        )
        
        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = UMT5Stack(decoder_config, mesh, dtype=dtype)
    
    def load_weights(self, model_config: ModelConfig):
        """Load weights from HuggingFace checkpoint."""
        loader = WeightLoader(
            model=self,
            model_config=model_config,
            mesh=self.mesh,
            dtype=self.dtype,
        )
        weight_mappings = self._create_weight_mappings()
        loader.load_weights_from_safetensors(weight_mappings)
        
        # No need to share relative_attention_bias - each layer loads its own weights
        
        logger.info("UMT5Decoder weights loaded successfully!")
    
    def _create_weight_mappings(self) -> dict:
        """Create weight mappings for decoder model."""
        mappings = {
            "shared.weight": WeightMapping(
                target_path="shared.embedding",
                sharding=("tensor", None),
                transpose=False,
            ),
            "decoder.final_layer_norm.weight": WeightMapping(
                target_path="decoder.final_layer_norm.scale",
                sharding=(None,),
                transpose=False,
            ),
        }
        
        for layer_idx in range(self.config.num_decoder_layers):
            mappings.update(_create_block_mapping_helper(self.config, layer_idx, is_decoder=True, prefix="decoder.block"))

        return mappings

    def __call__(
        self,
        input_ids: jax.Array | None = None,
        decoder_input_ids: jax.Array | None = None,
        forward_batch: ForwardBatch | None = None,
        token_to_kv_pool: KVCache | None = None,
        encoder_hidden_states: jax.Array | None = None,
        encoder_mask: jax.Array | None = None,
        decoder_attention_mask: jax.Array | None = None,
        **kwargs
    ):
        # Support multiple input conventions
        if decoder_input_ids is not None:
            x = decoder_input_ids
        elif input_ids is not None:
            x = input_ids
        elif forward_batch is not None:
            x = forward_batch.input_ids
        else:
            raise ValueError("Must provide either input_ids, decoder_input_ids, or forward_batch")
            
        mask = decoder_attention_mask if decoder_attention_mask is not None else kwargs.get("attention_mask")
            
        hidden_states = self.shared(x)
        hidden_states = self.decoder(
            hidden_states,
            mask=mask,
            forward_batch=forward_batch,
            token_to_kv_pool=token_to_kv_pool,
            encoder_hidden_states=encoder_hidden_states,
            encoder_mask=encoder_mask,
            deterministic=kwargs.get("deterministic", False)
        )
        return hidden_states


class UMT5Model(nnx.Module):
    """
    Full UMT5 Encoder-Decoder model (without LM head).
    """
    def __init__(
        self,
        config: UMT5Config,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.config = config
        self.mesh = mesh
        self.dtype = dtype
        
        self.shared = Embed(
            config.vocab_size,
            config.d_model,
            dtype=dtype,
            param_dtype=dtype,
            mesh=mesh,
            kernel_axes=("tensor", None),
        )
        
        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        self.encoder = UMT5Stack(encoder_config, mesh, dtype=dtype)
        
        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = UMT5Stack(decoder_config, mesh, dtype=dtype)
    
    def load_weights(self, model_config: ModelConfig):
        """Load weights from HuggingFace checkpoint."""
        loader = WeightLoader(
            model=self,
            model_config=model_config,
            mesh=self.mesh,
            dtype=self.dtype,
        )
        weight_mappings = self._create_weight_mappings()
        loader.load_weights_from_safetensors(weight_mappings)
        
        # No need to share relative_attention_bias - each layer loads its own weights
        
        logger.info("UMT5Model (Encoder-Decoder) weights loaded successfully!")
    
    def _create_weight_mappings(self) -> dict:
        """Create mappings from HuggingFace weight names to JAX model parameters."""
        mappings = {
            "shared.weight": WeightMapping(
                target_path="shared.embedding",
                sharding=("tensor", None),
                transpose=False,
            ),
            "encoder.final_layer_norm.weight": WeightMapping(
                target_path="encoder.final_layer_norm.scale",
                sharding=(None,),
                transpose=False,
            ),
            "decoder.final_layer_norm.weight": WeightMapping(
                target_path="decoder.final_layer_norm.scale",
                sharding=(None,),
                transpose=False,
            ),
        }
        
        for layer_idx in range(self.config.num_layers):
            mappings.update(_create_block_mapping_helper(self.config, layer_idx, is_decoder=False, prefix="encoder.block"))
        
        for layer_idx in range(self.config.num_decoder_layers):
            mappings.update(_create_block_mapping_helper(self.config, layer_idx, is_decoder=True, prefix="decoder.block"))
        
        return mappings
    
    def __call__(
        self,
        input_ids: jax.Array,
        decoder_input_ids: jax.Array,
        attention_mask: jax.Array | None = None,
        decoder_attention_mask: jax.Array | None = None,
        forward_batch: ForwardBatch | None = None,
        token_to_kv_pool: KVCache | None = None,
        **kwargs
    ):
        deterministic = kwargs.get("deterministic", False)
        
        # Encoder pass
        encoder_hidden_states = self.shared(input_ids)
        encoder_hidden_states = self.encoder(
            encoder_hidden_states, 
            mask=attention_mask,
            deterministic=deterministic
        )
        
        # Decoder pass
        decoder_hidden_states = self.shared(decoder_input_ids)
        decoder_hidden_states = self.decoder(
            decoder_hidden_states,
            mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_mask=attention_mask,
            forward_batch=forward_batch,
            token_to_kv_pool=token_to_kv_pool,
            deterministic=deterministic
        )
        
        return decoder_hidden_states


class UMT5ForConditionalGeneration(nnx.Module):
    """
    UMT5 model for conditional generation with LM head.
    Supports both testing mode (direct parameter passing) and SGLang inference mode.
    """
    def __init__(
        self,
        config: UMT5Config,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.config = config
        self.mesh = mesh
        self.dtype = dtype
        
        self.shared = Embed(
            config.vocab_size,
            config.d_model,
            dtype=dtype,
            param_dtype=dtype,
            mesh=mesh,
            kernel_axes=("tensor", None),
        )
        
        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        self.encoder = UMT5Stack(encoder_config, mesh, dtype=dtype)
        
        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = UMT5Stack(decoder_config, mesh, dtype=dtype)
        
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.d_model,
            dtype=dtype,
            param_dtype=dtype,
            kernel_axes=("tensor", None),
            mesh=mesh,
        )
        
        self.logits_processor = LogitsProcessor(config.vocab_size, mesh=mesh)

    def load_weights(self, model_config: ModelConfig):
        """Load weights from HuggingFace checkpoint."""
        loader = WeightLoader(
            model=self,
            model_config=model_config,
            mesh=self.mesh,
            dtype=self.dtype,
        )
        weight_mappings = self._create_weight_mappings()
        loader.load_weights_from_safetensors(weight_mappings)
        
        # No need to share relative_attention_bias - each layer loads its own weights
        
        logger.info("UMT5 weights loaded successfully!")

    def _create_weight_mappings(self) -> dict:
        """Create mappings from HuggingFace weight names to JAX model parameters."""
        mappings = {
            "shared.weight": WeightMapping(
                target_path="shared.embedding",
                sharding=("tensor", None),
                transpose=False,
            ),
            "encoder.final_layer_norm.weight": WeightMapping(
                target_path="encoder.final_layer_norm.scale",
                sharding=(None,),
                transpose=False,
            ),
            "decoder.final_layer_norm.weight": WeightMapping(
                target_path="decoder.final_layer_norm.scale",
                sharding=(None,),
                transpose=False,
            ),
            "lm_head.weight": WeightMapping(
                target_path="lm_head.embedding",
                sharding=("tensor", None),
                transpose=False,
            )
        }

        for layer_idx in range(self.config.num_layers):
            mappings.update(_create_block_mapping_helper(self.config, layer_idx, is_decoder=False, prefix="encoder.block"))

        for layer_idx in range(self.config.num_decoder_layers):
            mappings.update(_create_block_mapping_helper(self.config, layer_idx, is_decoder=True, prefix="decoder.block"))

        return mappings

    def __call__(
        self,
        forward_batch: ForwardBatch | None = None,
        token_to_kv_pool: KVCache | None = None,
        logits_metadata: LogitsMetadata | None = None,
        input_ids: jax.Array | None = None,
        decoder_input_ids: jax.Array | None = None,
        attention_mask: jax.Array | None = None,
        decoder_attention_mask: jax.Array | None = None,
        encoder_outputs: jax.Array | None = None,
        **kwargs
    ):
        """
        Forward pass supporting two modes:
        1. Testing mode: Direct parameter passing (input_ids, decoder_input_ids, etc.)
        2. SGLang mode: Using forward_batch, token_to_kv_pool, logits_metadata
        """
        deterministic = kwargs.get("deterministic", False)
            
        # Case 1: Testing mode (Direct parameters)
        if (input_ids is not None or encoder_outputs is not None) and decoder_input_ids is not None:
            # Encoder pass (compute only if encoder_outputs not provided)
            encoder_hidden_states = encoder_outputs
            if encoder_hidden_states is None:
                if input_ids is None:
                    raise ValueError("Must provide input_ids if encoder_outputs is not provided")
                encoder_embeds = self.shared(input_ids)
                encoder_hidden_states = self.encoder(
                    encoder_embeds,
                    mask=attention_mask,
                    deterministic=deterministic
                )
            
            # Decoder pass
            decoder_embeds = self.shared(decoder_input_ids)
            decoder_hidden_states = self.decoder(
                decoder_embeds,
                mask=decoder_attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_mask=attention_mask,
                deterministic=deterministic,
                forward_batch=None,
                token_to_kv_pool=None
            )
            
            # LM Head - compute logits using weights
            lm_head_weights = self.lm_head.embedding[...]
            logits = jnp.matmul(decoder_hidden_states, lm_head_weights.T)
            return logits
            
        # Case 2: SGLang inference mode
        elif forward_batch is not None:
            batch_input_ids = forward_batch.input_ids
            
            # Check if encoder_hidden_states are cached in forward_batch
            encoder_hidden_states = getattr(forward_batch, "encoder_hidden_states", None)
            encoder_mask = getattr(forward_batch, "encoder_mask", None)
            
            # Decoder embeddings
            decoder_embeds = self.shared(batch_input_ids)
            
            # Decoder Stack
            decoder_hidden_states = self.decoder(
                decoder_embeds, 
                mask=None,  # Mask handled by RadixAttention
                forward_batch=forward_batch,
                token_to_kv_pool=token_to_kv_pool,
                encoder_hidden_states=encoder_hidden_states,
                encoder_mask=encoder_mask,
                deterministic=deterministic
            )
            
            # LM Head
            logits = self.logits_processor(decoder_hidden_states, self.lm_head, logits_metadata)
            
            # Return format compatible with SGLang runtime
            layers_kv_fused = []  # Placeholder for KV cache updates
            layers_callback_flag = []  # Placeholder for callback flags
            return logits, layers_kv_fused, layers_callback_flag
        
        else:
            raise ValueError(
                "Either provide (input_ids, decoder_input_ids) for testing mode "
                "or forward_batch for SGLang inference mode"
            )


def _create_block_mapping_helper(config: UMT5Config, layer_idx: int, is_decoder: bool, prefix: str) -> dict:
    """
    Helper function to create weight mappings for a single transformer block.
    Reduces code duplication across different model classes.
    
    Args:
        config: UMT5 configuration
        layer_idx: Index of the layer/block
        is_decoder: Whether this is a decoder block (vs encoder)
        prefix: Prefix for weight keys (e.g., "encoder.block" or "decoder.block")
    
    Returns:
        Dictionary of weight mappings for this block
    """
    target_prefix = prefix + f".{layer_idx}"
    source_prefix = prefix + f".{layer_idx}"
    
    mappings = {}
    
    # Layer 0: Self Attention
    mappings.update({
        f"{source_prefix}.layer.0.layer_norm.weight": WeightMapping(
            target_path=f"{target_prefix}.layer0_LayerNorm.scale",
            sharding=(None,),
            transpose=False,
        ),
        f"{source_prefix}.layer.0.SelfAttention.q.weight": WeightMapping(
            target_path=f"{target_prefix}.layer0_SelfAttention.q.weight",
            sharding=(None, "tensor"),
            transpose=True,
        ),
        f"{source_prefix}.layer.0.SelfAttention.k.weight": WeightMapping(
            target_path=f"{target_prefix}.layer0_SelfAttention.k.weight",
            sharding=(None, "tensor"),
            transpose=True,
        ),
        f"{source_prefix}.layer.0.SelfAttention.v.weight": WeightMapping(
            target_path=f"{target_prefix}.layer0_SelfAttention.v.weight",
            sharding=(None, "tensor"),
            transpose=True,
        ),
        f"{source_prefix}.layer.0.SelfAttention.o.weight": WeightMapping(
            target_path=f"{target_prefix}.layer0_SelfAttention.o.weight",
            sharding=("tensor", None),
            transpose=True,
        ),
    })
    
    # Relative Attention Bias: Load for every layer
    # In HuggingFace UMT5, every layer has its own relative_attention_bias weights
    mappings[f"{source_prefix}.layer.0.SelfAttention.relative_attention_bias.weight"] = WeightMapping(
        target_path=f"{target_prefix}.layer0_SelfAttention.relative_attention_bias.embedding",
        sharding=(None, "tensor"),
        transpose=False,
    )
    
    if is_decoder:
        # Layer 1: Cross Attention (Decoder only)
        mappings.update({
            f"{source_prefix}.layer.1.layer_norm.weight": WeightMapping(
                target_path=f"{target_prefix}.layer1_LayerNorm.scale",
                sharding=(None,),
                transpose=False,
            ),
            f"{source_prefix}.layer.1.EncDecAttention.q.weight": WeightMapping(
                target_path=f"{target_prefix}.layer1_EncDecAttention.q.weight",
                sharding=(None, "tensor"),
                transpose=True,
            ),
            f"{source_prefix}.layer.1.EncDecAttention.k.weight": WeightMapping(
                target_path=f"{target_prefix}.layer1_EncDecAttention.k.weight",
                sharding=(None, "tensor"),
                transpose=True,
            ),
            f"{source_prefix}.layer.1.EncDecAttention.v.weight": WeightMapping(
                target_path=f"{target_prefix}.layer1_EncDecAttention.v.weight",
                sharding=(None, "tensor"),
                transpose=True,
            ),
            f"{source_prefix}.layer.1.EncDecAttention.o.weight": WeightMapping(
                target_path=f"{target_prefix}.layer1_EncDecAttention.o.weight",
                sharding=("tensor", None),
                transpose=True,
            ),
        })
        ffn_layer_idx = 2
    else:
        ffn_layer_idx = 1
    
    # FFN Layer
    mappings.update({
        f"{source_prefix}.layer.{ffn_layer_idx}.layer_norm.weight": WeightMapping(
            target_path=f"{target_prefix}.layer_FF_LayerNorm.scale",
            sharding=(None,),
            transpose=False,
        ),
    })

    # FFN weights (gated vs non-gated)
    dense_cls_name = "layer_FF_DenseReluDense"
    if config.is_gated_act:
        mappings.update({
            f"{source_prefix}.layer.{ffn_layer_idx}.DenseReluDense.wi_0.weight": WeightMapping(
                target_path=f"{target_prefix}.{dense_cls_name}.wi_0.weight",
                sharding=(None, "tensor"),
                transpose=True,
            ),
            f"{source_prefix}.layer.{ffn_layer_idx}.DenseReluDense.wi_1.weight": WeightMapping(
                target_path=f"{target_prefix}.{dense_cls_name}.wi_1.weight",
                sharding=(None, "tensor"),
                transpose=True,
            ),
            f"{source_prefix}.layer.{ffn_layer_idx}.DenseReluDense.wo.weight": WeightMapping(
                target_path=f"{target_prefix}.{dense_cls_name}.wo.weight",
                sharding=("tensor", None),
                transpose=True,
            ),
        })
    else:
        mappings.update({
            f"{source_prefix}.layer.{ffn_layer_idx}.DenseReluDense.wi.weight": WeightMapping(
                target_path=f"{target_prefix}.{dense_cls_name}.wi.weight",
                sharding=(None, "tensor"),
                transpose=True,
            ),
            f"{source_prefix}.layer.{ffn_layer_idx}.DenseReluDense.wo.weight": WeightMapping(
                target_path=f"{target_prefix}.{dense_cls_name}.wo.weight",
                sharding=("tensor", None),
                transpose=True,
            ),
        })

    return mappings


EntryClass = [UMT5EncoderModel, UMT5DecoderModel, UMT5Model, UMT5ForConditionalGeneration]

