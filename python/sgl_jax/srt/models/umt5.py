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
import math
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from transformers import UMT5Config

from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.layers.embeddings import Embed, ParallelLMHead
from sgl_jax.srt.layers.layernorm import RMSNorm
from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.layers.logits_processor import (
    LogitsMetadata,
    LogitsProcessor,
    LogitsProcessorOutput,
)
from sgl_jax.srt.layers.radix_attention import RadixAttention
from sgl_jax.srt.mem_cache.memory_pool import KVCache
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
from sgl_jax.srt.utils.weight_utils import WeightLoader, WeightMapping


def fp16_clamp(x: jax.Array):
    """Clamps values to prevent float16 overflow."""
    if x.dtype == jnp.float16 and jnp.isinf(x).any():
        clamp = jnp.finfo(x.dtype).max - 1000
        x = jax.lax.clamp(x=x, min=-clamp, max=clamp)
    return x


def gelu_new(x):
    """
    GELU activation with tanh approximation (matches HF T5/UMT5).
    Equivalent to PyTorch's F.gelu(x, approximate='tanh').
    """
    return 0.5 * x * (1.0 + jnp.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * jnp.power(x, 3))))


ACT_FN = {
    "gelu": jax.nn.gelu,
    "gelu_new": gelu_new,
    "relu": jax.nn.relu,
}


class UMT5DenseGatedActDense(nnx.Module):
    """
    Gated-GELU FFN (used in UMT5, Flan-T5).
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
            mesh=mesh,
            use_bias=False,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
        )
        self.wi_1 = LinearBase(
            input_size=config.d_model,
            output_size=config.d_ff,
            mesh=mesh,
            use_bias=False,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
        )
        self.wo = LinearBase(
            input_size=config.d_ff,
            output_size=config.d_model,
            mesh=mesh,
            use_bias=False,
            kernel_axes=("tensor", None),
            params_dtype=dtype,
        )
        self.dropout = nnx.Dropout(config.dropout_rate)
        self.act = ACT_FN.get(config.dense_act_fn, jax.nn.gelu)

    def __call__(self, hidden_states: jax.Array, deterministic: bool = True) -> jax.Array:
        hidden_gelu = self.act(self.wi_0(hidden_states)[0])
        hidden_linear = self.wi_1(hidden_states)[0]
        hidden_states = hidden_gelu * hidden_linear
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        hidden_states, _ = self.wo(hidden_states)
        return hidden_states


class UMT5DenseActDense(nnx.Module):
    """Standard FFN (used in original T5): Linear -> Act -> Linear."""

    def __init__(
        self,
        config: UMT5Config,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.wi = LinearBase(
            input_size=config.d_model,
            output_size=config.d_ff,
            mesh=mesh,
            use_bias=False,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
        )
        self.wo = LinearBase(
            input_size=config.d_ff,
            output_size=config.d_model,
            mesh=mesh,
            use_bias=False,
            kernel_axes=("tensor", None),
            params_dtype=dtype,
        )
        self.dropout = nnx.Dropout(config.dropout_rate)
        self.act = ACT_FN.get(config.dense_act_fn, jax.nn.relu)

    def __call__(self, hidden_states: jax.Array, deterministic: bool = True) -> jax.Array:
        hidden_states, _ = self.wi(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        hidden_states, _ = self.wo(hidden_states)
        return hidden_states


class UMT5Attention(nnx.Module):
    """
    Multi-head attention for UMT5.
    Supports self-attention, cross-attention, relative position bias, and KV cache.
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
        self.relative_attention_max_distance = getattr(
            config, "relative_attention_max_distance", 128
        )

        self.d_model = config.d_model
        self.key_value_proj_dim = config.d_kv
        self.n_heads = config.num_heads
        self.inner_dim = self.n_heads * self.key_value_proj_dim
        self.layer_idx = layer_idx

        # QKV projections
        self.q = LinearBase(
            input_size=self.d_model,
            output_size=self.inner_dim,
            mesh=mesh,
            use_bias=False,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
        )
        self.k = LinearBase(
            input_size=self.d_model,
            output_size=self.inner_dim,
            mesh=mesh,
            use_bias=False,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
        )
        self.v = LinearBase(
            input_size=self.d_model,
            output_size=self.inner_dim,
            mesh=mesh,
            use_bias=False,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
        )
        self.o = LinearBase(
            input_size=self.inner_dim,
            output_size=self.d_model,
            mesh=mesh,
            use_bias=False,
            kernel_axes=("tensor", None),
            params_dtype=dtype,
        )

        # Relative position bias (T5-style, self-attention only)
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

        # RadixAttention with KV cache (decoder self-attention only)
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
        Compute bucket indices for relative positions (adapted from HF T5).

        Bidirectional (encoder): separate buckets for past/future.
        Unidirectional (decoder): only considers past positions.
        Uses logarithmic bucketing for longer distances.
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
        """Compute relative attention bias matrix [1, n_heads, query_len, key_len]."""
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
        values = jnp.transpose(values, (2, 0, 1))
        return values

    def __call__(
        self,
        hidden_states: jax.Array,
        mask: jax.Array | None = None,
        deterministic: bool = True,
        encoder_hidden_states: jax.Array | None = None,
        encoder_mask: jax.Array | None = None,
        forward_batch: ForwardBatch | None = None,
        token_to_kv_pool: KVCache | None = None,
    ) -> jax.Array:

        seq_length, dim = hidden_states.shape[:2]

        # Project queries
        q, _ = self.q(hidden_states)

        # Get K/V from encoder (cross-attn) or input (self-attn)
        if self.is_cross_attention:
            # Cross-attention: K/V from encoder output
            if encoder_hidden_states is None:
                raise ValueError("encoder_hidden_states must be provided for cross attention")
            k, _ = self.k(encoder_hidden_states)
            v, _ = self.v(encoder_hidden_states)
        else:
            # Self-attention: K/V from input
            k, _ = self.k(hidden_states)
            v, _ = self.v(hidden_states)

        kv_fused = None

        # CASE 1: Decoder self-attention with KV cache (RadixAttention)
        # Used during autoregressive decoding in SGLang runtime.
        if (
            self.is_decoder
            and not self.is_cross_attention
            and forward_batch is not None
            and token_to_kv_pool is not None
        ):
            q_flat = q.reshape(-1, self.n_heads, self.key_value_proj_dim)
            k_flat = k.reshape(-1, self.n_heads, self.key_value_proj_dim)
            v_flat = v.reshape(-1, self.n_heads, self.key_value_proj_dim)

            # Note: Relative position bias not currently supported in RadixAttention fast path
            attn_output, kv_fused = self.radix_attn(
                q_flat,
                k_flat,
                v_flat,
                forward_batch,
                token_to_kv_pool,
            )

            attn_output = attn_output.reshape(seq_length, self.inner_dim)
            output, _ = self.o(attn_output)
            return output

        # CASE 2: Standard attention (encoder, cross-attention, or fallback)
        # Reshape for multi-head attention
        q = q.reshape(seq_length, self.n_heads, self.key_value_proj_dim)
        k = k.reshape(-1, self.n_heads, self.key_value_proj_dim)
        v = v.reshape(-1, self.n_heads, self.key_value_proj_dim)

        q = jnp.transpose(q, (1, 0, 2))  # [n_heads, seq_q, head_dim]
        k = jnp.transpose(k, (1, 0, 2))  # [n_heads, seq_k, head_dim]
        v = jnp.transpose(v, (1, 0, 2))  # [n_heads, seq_k, head_dim]

        # Compute attention scores (unscaled for T5)
        q_f32 = q.astype(jnp.float32)
        k_f32 = k.astype(jnp.float32)
        scores = jnp.matmul(q_f32, jnp.swapaxes(k_f32, -1, -2))

        # Add relative position bias (self-attention only)
        if not self.is_cross_attention:
            bidirectional = not self.is_decoder
            position_bias = self.compute_bias(seq_length, k.shape[1], bidirectional=bidirectional)
            scores += position_bias.astype(jnp.float32)

        # Apply attention masks
        active_mask = mask if not self.is_cross_attention else encoder_mask

        if self.is_decoder and not self.is_cross_attention:
            # Causal mask for decoder self-attention
            causal_mask = jnp.tril(jnp.ones((seq_length, seq_length), dtype=jnp.bool_))
            causal_mask = causal_mask[None, None, :, :]  # [1, 1, seq, seq]

            if active_mask is not None:
                if active_mask.ndim == 2:
                    # Combine causal mask with padding mask
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

        attn_output = jnp.transpose(attn_output, (1, 0, 2))
        attn_output = attn_output.reshape(seq_length, self.inner_dim)

        output, _ = self.o(attn_output)
        return output


class UMT5Block(nnx.Module):
    """
    Transformer block for UMT5.
    Includes self-attention, cross-attention (decoder only), and feed-forward sublayers.
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
        self.input_layernorm = RMSNorm(
            config.d_model,
            epsilon=config.layer_norm_epsilon,
            dtype=dtype,
            param_dtype=dtype,
            use_scale=True,
        )
        self.self_attn = UMT5Attention(
            config,
            mesh,
            dtype=dtype,
            layer_idx=layer_idx,
            is_cross_attention=False,
            is_decoder=is_decoder,
        )
        self.self_attn_dropout = nnx.Dropout(config.dropout_rate)

        # Cross Attention sublayer (Decoder only)
        if self.is_decoder:
            self.cross_attention_layernorm = RMSNorm(
                config.d_model,
                epsilon=config.layer_norm_epsilon,
                dtype=dtype,
                param_dtype=dtype,
                use_scale=True,
            )
            self.cross_attn = UMT5Attention(
                config,
                mesh,
                dtype=dtype,
                layer_idx=layer_idx,
                is_cross_attention=True,
                is_decoder=is_decoder,
            )
            self.cross_attn_dropout = nnx.Dropout(config.dropout_rate)

        # Feed Forward sublayer
        self.post_attention_layernorm = RMSNorm(
            config.d_model,
            epsilon=config.layer_norm_epsilon,
            dtype=dtype,
            param_dtype=dtype,
            use_scale=True,
        )

        if config.is_gated_act:
            self.mlp = UMT5DenseGatedActDense(config, mesh, dtype=dtype)
        else:
            self.mlp = UMT5DenseActDense(config, mesh, dtype=dtype)

        self.mlp_dropout = nnx.Dropout(config.dropout_rate)

    def __call__(
        self,
        hidden_states: jax.Array,
        mask: jax.Array | None = None,
        deterministic: bool = True,
        forward_batch: ForwardBatch | None = None,
        token_to_kv_pool: KVCache | None = None,
    ) -> jax.Array:

        # Self Attention block
        normed_hidden_states = self.input_layernorm(hidden_states)
        attn_output = self.self_attn(
            normed_hidden_states,
            mask=mask,
            deterministic=deterministic,
            forward_batch=forward_batch,
            token_to_kv_pool=token_to_kv_pool,
        )
        attn_output = self.self_attn_dropout(attn_output, deterministic=deterministic)
        hidden_states = hidden_states + attn_output
        hidden_states = fp16_clamp(hidden_states)

        # Cross Attention block (Decoder Only)
        if self.is_decoder:
            encoder_hidden_states = None
            encoder_mask = None
            if forward_batch is not None:
                encoder_hidden_states = getattr(forward_batch, "encoder_hidden_states", None)
                encoder_mask = getattr(forward_batch, "encoder_mask", None)

            if encoder_hidden_states is not None:
                normed_hidden_states = self.cross_attention_layernorm(hidden_states)
                attn_output = self.cross_attn(
                    normed_hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_mask=encoder_mask,
                    deterministic=deterministic,
                )
                attn_output = self.cross_attn_dropout(attn_output, deterministic=deterministic)
                hidden_states = hidden_states + attn_output
                hidden_states = fp16_clamp(hidden_states)

        # Feed Forward block
        normed_hidden_states = self.post_attention_layernorm(hidden_states)
        mlp_output = self.mlp(normed_hidden_states, deterministic=deterministic)
        mlp_output = self.mlp_dropout(mlp_output, deterministic=deterministic)
        hidden_states = hidden_states + mlp_output
        hidden_states = fp16_clamp(hidden_states)

        return hidden_states


class UMT5Stack(nnx.Module):
    """Stack of UMT5 transformer blocks (encoder or decoder)."""

    def __init__(
        self,
        config: UMT5Config,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.is_decoder = config.is_decoder

        self.block = nnx.List(
            [
                UMT5Block(
                    config,
                    mesh,
                    dtype=dtype,
                    layer_idx=i,
                    is_decoder=self.is_decoder,
                )
                for i in range(config.num_layers)
            ]
        )

        self.final_layer_norm = RMSNorm(
            config.d_model,
            epsilon=config.layer_norm_epsilon,
            dtype=dtype,
            param_dtype=dtype,
            use_scale=True,
        )
        self.dropout = nnx.Dropout(config.dropout_rate)

    def __call__(
        self,
        hidden_states: jax.Array,
        mask: jax.Array | None = None,
        deterministic: bool = True,
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
                forward_batch=forward_batch,
                token_to_kv_pool=token_to_kv_pool,
            )

        # Final layer norm and dropout
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = fp16_clamp(hidden_states)
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        return hidden_states


class UMT5EncoderModel(nnx.Module):
    """UMT5 encoder-only model."""

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
        import glob
        import os

        original_path = model_config.model_path
        is_local_path = os.path.isabs(original_path)

        if is_local_path and os.path.exists(original_path):
            has_safetensors = len(glob.glob(os.path.join(original_path, "*.safetensors"))) > 0
            has_text_encoder = os.path.exists(os.path.join(original_path, "text_encoder"))

            if not has_safetensors and has_text_encoder:
                model_config.model_path = os.path.join(original_path, "text_encoder")

        loader = WeightLoader(
            model=self,
            model_config=model_config,
            mesh=self.mesh,
            dtype=self.dtype,
        )
        weight_mappings = self._create_weight_mappings()
        loader.load_weights_from_safetensors(weight_mappings)

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
            mappings.update(
                _create_block_mapping_helper(
                    self.config, layer_idx, is_decoder=False, prefix="encoder.block"
                )
            )
        return mappings

    def __call__(
        self,
        forward_batch: ForwardBatch,
        token_to_kv_pool: Any | None = None,
        logits_metadata: LogitsMetadata | None = None,
    ):
        x = forward_batch.input_ids
        mask = getattr(forward_batch, "attention_mask", None)

        # Reshape 1D input_ids to (batch_size, seq_len) for Encoder
        if x.ndim == 1 and hasattr(forward_batch, "req_pool_indices"):
            bs = forward_batch.req_pool_indices.shape[0]
            total_tokens = x.shape[0]
            if total_tokens % bs == 0:
                seq_len = total_tokens // bs
                x = x.reshape(bs, seq_len)
                # Reshape mask to match input_ids if it exists
                if mask is not None and mask.ndim == 1:
                    mask = mask.reshape(bs, seq_len)

        deterministic = getattr(forward_batch, "deterministic", True)

        hidden_states = self.shared(x)

        hidden_states = self.encoder(
            hidden_states,
            mask=mask,
            deterministic=deterministic,
            forward_batch=forward_batch,
            token_to_kv_pool=token_to_kv_pool,
        )

        # Create dummy logits to satisfy sampler interface
        bs = (
            forward_batch.req_pool_indices.shape[0]
            if hasattr(forward_batch, "req_pool_indices")
            else x.shape[0]
        )
        dummy_logits = jnp.zeros((bs, self.config.vocab_size), dtype=self.dtype)

        # Apply tensor-parallel sharding
        sharding = NamedSharding(self.mesh, P(None, "tensor"))
        dummy_logits = jax.device_put(dummy_logits, sharding)

        return (
            LogitsProcessorOutput(
                next_token_logits=dummy_logits,
                hidden_states=hidden_states,
            ),
            [],
            [],
        )


class UMT5DecoderModel(nnx.Module):
    """UMT5 decoder-only model."""

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
            mappings.update(
                _create_block_mapping_helper(
                    self.config, layer_idx, is_decoder=True, prefix="decoder.block"
                )
            )

        return mappings

    def __call__(
        self,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache | None = None,
    ):
        x = forward_batch.input_ids
        deterministic = getattr(forward_batch, "deterministic", True)

        hidden_states = self.shared(x)

        mask = getattr(forward_batch, "attention_mask", None)

        hidden_states = self.decoder(
            hidden_states,
            mask=mask,
            forward_batch=forward_batch,
            token_to_kv_pool=token_to_kv_pool,
            deterministic=deterministic,
        )
        return hidden_states


class UMT5Model(nnx.Module):
    """UMT5 encoder-decoder model (without LM head)."""

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
            mappings.update(
                _create_block_mapping_helper(
                    self.config, layer_idx, is_decoder=False, prefix="encoder.block"
                )
            )

        for layer_idx in range(self.config.num_decoder_layers):
            mappings.update(
                _create_block_mapping_helper(
                    self.config, layer_idx, is_decoder=True, prefix="decoder.block"
                )
            )

        return mappings

    def __call__(
        self,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache | None = None,
    ):
        input_ids = forward_batch.input_ids
        attention_mask = getattr(forward_batch, "attention_mask", None)
        deterministic = getattr(forward_batch, "deterministic", True)

        # Encoder pass
        encoder_hidden_states = self.shared(input_ids)
        encoder_hidden_states = self.encoder(
            encoder_hidden_states,
            mask=attention_mask,
            deterministic=deterministic,
            forward_batch=forward_batch,
        )

        # Decoder pass
        decoder_input_ids = getattr(forward_batch, "decoder_input_ids", input_ids)

        decoder_hidden_states = self.shared(decoder_input_ids)

        decoder_attention_mask = getattr(forward_batch, "decoder_attention_mask", attention_mask)

        decoder_hidden_states = self.decoder(
            decoder_hidden_states,
            mask=decoder_attention_mask,
            forward_batch=forward_batch,
            token_to_kv_pool=token_to_kv_pool,
            deterministic=deterministic,
        )

        return decoder_hidden_states


class UMT5ForConditionalGeneration(nnx.Module):
    """
    UMT5 for conditional generation with LM head.
    Supports SGLang inference mode.
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
            ),
        }

        for layer_idx in range(self.config.num_layers):
            mappings.update(
                _create_block_mapping_helper(
                    self.config, layer_idx, is_decoder=False, prefix="encoder.block"
                )
            )

        for layer_idx in range(self.config.num_decoder_layers):
            mappings.update(
                _create_block_mapping_helper(
                    self.config, layer_idx, is_decoder=True, prefix="decoder.block"
                )
            )

        return mappings

    def __call__(
        self,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache | None = None,
        logits_metadata: LogitsMetadata | None = None,
    ):
        """
        Forward pass for SGLang inference mode.
        """
        batch_input_ids = forward_batch.input_ids
        deterministic = getattr(forward_batch, "deterministic", True)

        # Decoder forward pass
        decoder_embeds = self.shared(batch_input_ids)

        mask = getattr(forward_batch, "attention_mask", None)

        decoder_hidden_states = self.decoder(
            decoder_embeds,
            mask=mask,
            forward_batch=forward_batch,
            token_to_kv_pool=token_to_kv_pool,
            deterministic=deterministic,
        )

        # Compute logits
        if logits_metadata is not None:
            logits = self.logits_processor(decoder_hidden_states, self.lm_head, logits_metadata)
        else:
            lm_head_weights = self.lm_head.embedding[...]
            logits = jnp.matmul(decoder_hidden_states, lm_head_weights.T)

        # Return format for SGLang runtime
        layers_kv_fused = []
        layers_callback_flag = []
        return logits, layers_kv_fused, layers_callback_flag


def _create_block_mapping_helper(
    config: UMT5Config, layer_idx: int, is_decoder: bool, prefix: str
) -> dict:
    """
    Create weight mappings for a single transformer block.

    Args:
        config: UMT5 config
        layer_idx: Layer index
        is_decoder: True for decoder blocks
        prefix: Weight name prefix (e.g., "encoder.block")

    Returns:
        Weight mapping dict
    """
    target_prefix = prefix + f".{layer_idx}"
    source_prefix = prefix + f".{layer_idx}"

    mappings = {}

    # Layer 0: Self Attention
    mappings.update(
        {
            f"{source_prefix}.layer.0.layer_norm.weight": WeightMapping(
                target_path=f"{target_prefix}.input_layernorm.scale",
                sharding=(None,),
                transpose=False,
            ),
            f"{source_prefix}.layer.0.SelfAttention.q.weight": WeightMapping(
                target_path=f"{target_prefix}.self_attn.q.weight",
                sharding=(None, "tensor"),
                transpose=True,
            ),
            f"{source_prefix}.layer.0.SelfAttention.k.weight": WeightMapping(
                target_path=f"{target_prefix}.self_attn.k.weight",
                sharding=(None, "tensor"),
                transpose=True,
            ),
            f"{source_prefix}.layer.0.SelfAttention.v.weight": WeightMapping(
                target_path=f"{target_prefix}.self_attn.v.weight",
                sharding=(None, "tensor"),
                transpose=True,
            ),
            f"{source_prefix}.layer.0.SelfAttention.o.weight": WeightMapping(
                target_path=f"{target_prefix}.self_attn.o.weight",
                sharding=("tensor", None),
                transpose=True,
            ),
        }
    )

    # Relative attention bias (loaded for all layers from HF checkpoint)
    mappings[f"{source_prefix}.layer.0.SelfAttention.relative_attention_bias.weight"] = (
        WeightMapping(
            target_path=f"{target_prefix}.self_attn.relative_attention_bias.embedding",
            sharding=(None, "tensor"),
            transpose=False,
        )
    )

    if is_decoder:
        # Layer 1: Cross Attention (Decoder only)
        mappings.update(
            {
                f"{source_prefix}.layer.1.layer_norm.weight": WeightMapping(
                    target_path=f"{target_prefix}.cross_attention_layernorm.scale",
                    sharding=(None,),
                    transpose=False,
                ),
                f"{source_prefix}.layer.1.EncDecAttention.q.weight": WeightMapping(
                    target_path=f"{target_prefix}.cross_attn.q.weight",
                    sharding=(None, "tensor"),
                    transpose=True,
                ),
                f"{source_prefix}.layer.1.EncDecAttention.k.weight": WeightMapping(
                    target_path=f"{target_prefix}.cross_attn.k.weight",
                    sharding=(None, "tensor"),
                    transpose=True,
                ),
                f"{source_prefix}.layer.1.EncDecAttention.v.weight": WeightMapping(
                    target_path=f"{target_prefix}.cross_attn.v.weight",
                    sharding=(None, "tensor"),
                    transpose=True,
                ),
                f"{source_prefix}.layer.1.EncDecAttention.o.weight": WeightMapping(
                    target_path=f"{target_prefix}.cross_attn.o.weight",
                    sharding=("tensor", None),
                    transpose=True,
                ),
            }
        )
        ffn_layer_idx = 2
    else:
        ffn_layer_idx = 1

    # FFN Layer
    mappings.update(
        {
            f"{source_prefix}.layer.{ffn_layer_idx}.layer_norm.weight": WeightMapping(
                target_path=f"{target_prefix}.post_attention_layernorm.scale",
                sharding=(None,),
                transpose=False,
            ),
        }
    )

    # FFN weights (gated vs non-gated)
    dense_cls_name = "mlp"
    if config.is_gated_act:
        mappings.update(
            {
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
            }
        )
    else:
        mappings.update(
            {
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
            }
        )

    return mappings


EntryClass = [UMT5EncoderModel, UMT5DecoderModel, UMT5Model, UMT5ForConditionalGeneration]
