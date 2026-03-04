"""Qwen3-VL Generation Model for SGLang-JAX.

Extends QWen3Model with M-RoPE support for multimodal position encoding.
Follows the same pattern as Qwen2.5-VL (which extends Qwen2Model).
"""

import logging

import jax
import jax.numpy as jnp
from flax import nnx

from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.hf_transformers_utils import get_hf_text_config
from sgl_jax.srt.layers.embeddings import ParallelLMHead
from sgl_jax.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sgl_jax.srt.mem_cache.memory_pool import KVCache
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
from sgl_jax.srt.models.qwen3 import QWen3Model
from sgl_jax.srt.multimodal.models.qwen_vl_utils import MRotaryEmbedding
from sgl_jax.srt.utils.weight_utils import WeightLoader, WeightMapping

logger = logging.getLogger(__name__)


# =============================================================================
# Qwen3-VL Model (extends QWen3Model with M-RoPE)
# =============================================================================


class Qwen3_VL_Model(QWen3Model):
    """QWen3Model with MRoPE support for Qwen3-VL."""

    def __init__(self, config, mesh, dtype=jnp.bfloat16):
        super().__init__(config=config, mesh=mesh, dtype=dtype)
        rope_scaling = getattr(config, "rope_scaling", None) or {}
        self._mrope_section = rope_scaling.get("mrope_section")
        self._mrope_interleaved = rope_scaling.get("mrope_interleaved", True)
        if self._mrope_section:
            rope_theta = getattr(config, "rope_theta", 5_000_000)
            max_position_embeddings = getattr(config, "max_position_embeddings", 32768)
            for layer in self.layers:
                head_dim = layer.self_attn.head_dim
                layer.self_attn.rotary_emb = MRotaryEmbedding(
                    head_size=head_dim,
                    rotary_dim=head_dim,
                    max_position_embeddings=max_position_embeddings,
                    base=rope_theta,
                    is_neox_style=True,
                    dtype=dtype,
                    mrope_section=self._mrope_section,
                    mrope_interleaved=self._mrope_interleaved,
                )

    def __call__(
        self,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
    ):
        residual = None
        input_embeds = (
            forward_batch.input_embedding
            if forward_batch.forward_mode.is_extend_or_draft_extend_or_mixed()
            else None
        )
        hidden_states = (
            self.embed_tokens(forward_batch.input_ids) if input_embeds is None else input_embeds
        )
        rope_positions = (
            forward_batch.mrope_positions
            if self._mrope_section and forward_batch.mrope_positions is not None
            else forward_batch.positions
        )
        layers_kv_fused = []
        layers_callback_flag = []
        for layer in self.layers:
            hidden_states, residual, kv_fused, callback_flag = layer(
                rope_positions,
                hidden_states,
                forward_batch,
                token_to_kv_pool,
                residual,
            )
            layers_kv_fused.append(kv_fused)
            layers_callback_flag.extend(callback_flag)

        if residual is not None:
            hidden_states += residual
        hidden_states = self.norm(hidden_states)

        return hidden_states, layers_kv_fused, layers_callback_flag


# =============================================================================
# Top-level generation model
# =============================================================================


class Qwen3_VL_Generation(nnx.Module):
    """Qwen3-VL model for conditional generation.

    Usage Pattern:
    1. PREFILL: Process vision → merge embeddings → __call__()
    2. DECODE: __call__() without embeddings
    """

    def __init__(self, config=None, dtype=None, mesh=None):
        super().__init__()
        self.mesh = mesh
        self.config = config
        self.dtype = dtype or jnp.bfloat16

        self.text_config = get_hf_text_config(config) or config

        self.model = Qwen3_VL_Model(self.text_config, mesh=mesh, dtype=self.dtype)

        if not getattr(self.text_config, "tie_word_embeddings", False):
            self.lm_head = ParallelLMHead(
                self.text_config.vocab_size,
                self.text_config.hidden_size,
                dtype=self.dtype,
                param_dtype=self.dtype,
                kernel_axes=("tensor", None),
            )

        self.logits_processor = LogitsProcessor(self.text_config.vocab_size, mesh=self.mesh)

        self.image_token_id = getattr(self.config, "image_token_id", 151655)
        self.video_token_id = getattr(self.config, "video_token_id", 151656)

    def load_weights(self, model_config: ModelConfig):
        loader = WeightLoader(
            model=self,
            model_config=model_config,
            mesh=self.mesh,
            dtype=self.dtype,
        )
        weight_mappings = self._create_weight_mappings()
        loader.load_weights_from_safetensors(weight_mappings)
        logger.info("Qwen3-VL (LLM) weights loaded successfully!")

    def _create_weight_mappings(self) -> dict:
        """Create weight mappings for text decoder.

        HF safetensors keys use ``model.language_model.`` prefix;
        our JAX model paths use ``model.`` (no language_model).
        """
        mappings = {
            "model.language_model.embed_tokens.weight": WeightMapping(
                target_path="model.embed_tokens.embedding",
                sharding=("tensor", None),
                transpose=False,
            ),
            "model.language_model.norm.weight": WeightMapping(
                target_path="model.norm.scale",
                sharding=(None,),
                transpose=False,
            ),
        }

        if not getattr(self.text_config, "tie_word_embeddings", False):
            mappings["lm_head.weight"] = WeightMapping(
                target_path="lm_head.embedding",
                sharding=("tensor", None),
                transpose=False,
            )

        num_layers = self.text_config.num_hidden_layers
        for layer_idx in range(num_layers):
            mappings.update(self._create_layer_mappings(layer_idx))

        return mappings

    def _create_layer_mappings(self, layer_idx: int) -> dict:
        prefix = f"model.language_model.layers.{layer_idx}"
        target_prefix = f"model.layers.{layer_idx}"

        mappings = {
            f"{prefix}.input_layernorm.weight": WeightMapping(
                target_path=f"{target_prefix}.input_layernorm.scale",
                sharding=(None,),
                transpose=False,
            ),
            f"{prefix}.post_attention_layernorm.weight": WeightMapping(
                target_path=f"{target_prefix}.post_attention_layernorm.scale",
                sharding=(None,),
                transpose=False,
            ),
            f"{prefix}.self_attn.q_proj.weight": WeightMapping(
                target_path=f"{target_prefix}.self_attn.q_proj.weight",
                sharding=(None, "tensor"),
                transpose=True,
                head_dim_padding=True,
                kv_head_padding=False,
            ),
            f"{prefix}.self_attn.k_proj.weight": WeightMapping(
                target_path=f"{target_prefix}.self_attn.k_proj.weight",
                sharding=(None, "tensor"),
                transpose=True,
                head_dim_padding=True,
                kv_head_padding=True,
            ),
            f"{prefix}.self_attn.v_proj.weight": WeightMapping(
                target_path=f"{target_prefix}.self_attn.v_proj.weight",
                sharding=(None, "tensor"),
                transpose=True,
                head_dim_padding=True,
                kv_head_padding=True,
            ),
            f"{prefix}.self_attn.o_proj.weight": WeightMapping(
                target_path=f"{target_prefix}.self_attn.o_proj.weight",
                sharding=("tensor", None),
                transpose=True,
                head_dim_padding=True,
                kv_head_padding=False,
            ),
            # Q/K norms (Qwen3 specific)
            f"{prefix}.self_attn.q_norm.weight": WeightMapping(
                target_path=f"{target_prefix}.self_attn.q_norm.scale",
                sharding=(None,),
                transpose=False,
            ),
            f"{prefix}.self_attn.k_norm.weight": WeightMapping(
                target_path=f"{target_prefix}.self_attn.k_norm.scale",
                sharding=(None,),
                transpose=False,
            ),
            # MLP
            f"{prefix}.mlp.gate_proj.weight": WeightMapping(
                target_path=f"{target_prefix}.mlp.gate_proj.weight",
                sharding=(None, "tensor"),
                transpose=True,
            ),
            f"{prefix}.mlp.up_proj.weight": WeightMapping(
                target_path=f"{target_prefix}.mlp.up_proj.weight",
                sharding=(None, "tensor"),
                transpose=True,
            ),
            f"{prefix}.mlp.down_proj.weight": WeightMapping(
                target_path=f"{target_prefix}.mlp.down_proj.weight",
                sharding=("tensor", None),
                transpose=True,
            ),
        }

        if getattr(self.text_config, "attention_bias", False):
            mappings.update(
                {
                    f"{prefix}.self_attn.q_proj.bias": WeightMapping(
                        target_path=f"{target_prefix}.self_attn.q_proj.bias",
                        sharding=(None,),
                        transpose=False,
                        head_dim_padding=True,
                        kv_head_padding=False,
                    ),
                    f"{prefix}.self_attn.k_proj.bias": WeightMapping(
                        target_path=f"{target_prefix}.self_attn.k_proj.bias",
                        sharding=(None,),
                        transpose=False,
                        head_dim_padding=True,
                        kv_head_padding=True,
                    ),
                    f"{prefix}.self_attn.v_proj.bias": WeightMapping(
                        target_path=f"{target_prefix}.self_attn.v_proj.bias",
                        sharding=(None,),
                        transpose=False,
                        head_dim_padding=True,
                        kv_head_padding=True,
                    ),
                }
            )

        return mappings

    def get_embed_and_head(self):
        if getattr(self.text_config, "tie_word_embeddings", False):
            weight = self.model.embed_tokens.embedding.value
            return (weight, weight)
        return (self.model.embed_tokens.embedding.value, self.lm_head.embedding.value)

    def set_embed_and_head(
        self,
        embed_weight: jax.Array | None = None,
        head_weight: jax.Array | None = None,
    ) -> None:
        if embed_weight is not None:
            self.model.embed_tokens.embedding.value = embed_weight
        if head_weight is not None:
            self.lm_head.embedding.value = head_weight

    def __call__(
        self,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
        logits_metadata: LogitsMetadata,
    ):
        hidden_states, layers_kv_fused, layers_callback_flag = self.model(
            forward_batch, token_to_kv_pool
        )

        if not getattr(self.text_config, "tie_word_embeddings", False):
            output = self.logits_processor(hidden_states, self.lm_head, logits_metadata)
        else:
            output = self.logits_processor(hidden_states, self.model.embed_tokens, logits_metadata)

        return output, layers_kv_fused, layers_callback_flag, None
