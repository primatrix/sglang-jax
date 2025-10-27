import logging
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx
from transformers import PretrainedConfig

from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.layers.embeddings import Embed, ParallelLMHead, get_rope
from sgl_jax.srt.layers.layernorm import RMSNorm
from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sgl_jax.srt.layers.moe import EPMoE, GateLogit, TopK
from sgl_jax.srt.layers.radix_attention import RadixAttention
from sgl_jax.srt.mem_cache.memory_pool import KVCache
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
from sgl_jax.srt.utils.weight_utils import WeightLoader, WeightMapping

logger = logging.getLogger(__name__)
init_fn = nnx.initializers.uniform()


class GptOssAttention(nnx.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        layer_id: int,
        rope_theta: float,
        rope_scaling: dict[str, Any] | None,
        head_dim: int | None,
        rope_is_neox_style: bool,
        max_position_embeddings: int,
        sliding_window_size: int,
        dtype: jnp.dtype,
        attention_bias: bool,
    ) -> None:
        self.hidden_size = hidden_size
        self.q_head_num = num_heads
        self.kv_head_num = num_kv_heads
        self.head_dim = head_dim or (hidden_size // num_heads)

        self.rotary_dim = self.head_dim
        self.q_size = num_heads * self.head_dim
        self.kv_size = num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.q_proj = LinearBase(
            input_size=hidden_size,
            output_size=self.q_size,
            use_bias=attention_bias,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
        )
        self.k_proj = LinearBase(
            input_size=hidden_size,
            output_size=self.kv_size,
            use_bias=attention_bias,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
        )
        self.v_proj = LinearBase(
            input_size=hidden_size,
            output_size=self.kv_size,
            use_bias=attention_bias,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
        )
        self.o_proj = LinearBase(
            input_size=self.q_size,
            output_size=hidden_size,
            use_bias=attention_bias,
            kernel_axes=("tensor", None),
            params_dtype=dtype,
        )

        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            rotary_dim=self.rotary_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            is_neox_style=rope_is_neox_style,
            rope_scaling=rope_scaling,
            dtype=dtype,
        )

        self.attn = RadixAttention(
            num_heads=num_heads,
            head_dim=self.head_dim,
            scaling=self.scaling,
            num_kv_heads=num_kv_heads,
            layer_id=layer_id,
            sliding_window_size=sliding_window_size,
        )

    def __call__(
        self,
        positions: jax.Array,
        hidden_states: jax.Array,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
    ) -> jax.Array:
        q, _ = self.q_proj(hidden_states)
        k, _ = self.k_proj(hidden_states)
        v, _ = self.v_proj(hidden_states)

        q = q.reshape(-1, self.q_head_num, self.head_dim)
        k = k.reshape(-1, self.kv_head_num, self.head_dim)
        v = v.reshape(-1, self.kv_head_num, self.head_dim)

        q, k = self.rotary_emb(positions, q, k)
        attn_output, kv_fused = self.attn(
            q, k, v, forward_batch=forward_batch, token_to_kv_pool=token_to_kv_pool
        )
        jax.debug.print(
            "attn_output shape {}, self.o_proj shape {}",
            attn_output.shape,
            self.o_proj.weight.value.shape,
        )
        output, _ = self.o_proj(attn_output)
        return output, kv_fused


class GptOssDecoderLayer(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        dtype: jnp.dtype,
        mesh: jax.sharding.Mesh = None,
    ) -> None:
        self.layer_id = layer_id
        hidden_size = config.hidden_size
        expert_parallel_size = mesh.shape.get("data", 1) * mesh.shape.get("tensor", 1)
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        rope_is_neox_style = getattr(config, "rope_is_neox_style", True)
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        attention_bias = getattr(config, "attention_bias", False) or getattr(config, "bias", False)
        head_dim = getattr(config, "head_dim", None)
        use_sliding_window = config.layer_types[layer_id] == "sliding_attention"
        self.self_attn = GptOssAttention(
            hidden_size=hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=head_dim,
            layer_id=layer_id,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            rope_is_neox_style=rope_is_neox_style,
            max_position_embeddings=max_position_embeddings,
            sliding_window_size=config.sliding_window if use_sliding_window else 0,
            attention_bias=attention_bias,
            dtype=dtype,
        )

        self.router = GateLogit(
            input_size=hidden_size,
            num_experts=config.num_local_experts,
            weight_dtype=dtype,
            enable_expert_bias=True,
            score_func="softmax",
        )
        self.topk = TopK(
            topk=config.num_experts_per_tok,
            renormalize=True,
        )
        self.moe = EPMoE(
            config,
            num_experts=config.num_local_experts,
            num_experts_per_tok=config.num_experts_per_tok,
            expert_parallel_size=expert_parallel_size,
            intermediate_dim=config.intermediate_size,
            layer_id=layer_id,
            weight_dtype=dtype,
            dtype=dtype,
            mesh=mesh,
        )

        self.input_layernorm = RMSNorm(
            hidden_size,
            epsilon=config.rms_norm_eps,
            param_dtype=dtype,
            scale_init=nnx.with_partitioning(init_fn, (None,)),
            dtype=dtype,
        )
        self.post_attention_layernorm = RMSNorm(
            hidden_size,
            epsilon=config.rms_norm_eps,
            param_dtype=dtype,
            scale_init=nnx.with_partitioning(init_fn, (None,)),
            dtype=dtype,
        )

    def __call__(
        self,
        positions: jax.Array,
        hidden_states: jax.Array,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
        residual: jax.Array | None,
    ):
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states += residual
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)

        hidden_states, kv_fused = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
            token_to_kv_pool=token_to_kv_pool,
        )

        hidden_states += residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        router_logits = self.router(hidden_states)
        topk_weights, topk_ids = self.topk(router_logits)
        hidden_states = self.moe(hidden_states, topk_weights, topk_ids)

        return hidden_states, residual, kv_fused


class GptOssModel(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        mesh: jax.sharding.Mesh = None,
    ) -> None:
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = Embed(
            config.vocab_size,
            config.hidden_size,
            dtype=dtype,
            param_dtype=dtype,
        )

        self.layers = nnx.data(
            [
                GptOssDecoderLayer(
                    config=config,
                    layer_id=i,
                    dtype=dtype,
                    mesh=mesh,
                )
                for i in range(config.num_hidden_layers)
            ]
        )

        self.norm = RMSNorm(
            config.hidden_size,
            epsilon=config.rms_norm_eps,
            param_dtype=dtype,
            scale_init=nnx.with_partitioning(init_fn, (None,)),
        )

    def __call__(
        self,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
    ):
        residual = None
        inputs = forward_batch.input_ids
        hidden_states = self.embed_tokens(inputs.reshape(-1)).reshape(-1, self.config.hidden_size)

        for i in range(self.config.num_hidden_layers):
            hidden_states, residual, kv_fused = self.layers[i](
                positions=forward_batch.positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
                token_to_kv_pool=token_to_kv_pool,
                residual=residual,
            )

        hidden_states = self.norm(hidden_states)
        return hidden_states, kv_fused


class GptOssForCausalLM(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        mesh: jax.sharding.Mesh = None,
    ) -> None:
        self.dtype = dtype
        self.config = config
        self.model = GptOssModel(config, dtype=dtype, mesh=mesh)
        if not getattr(self.config, "tie_word_embeddings", False):
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                embedding_init=nnx.with_partitioning(init_fn, ("tensor", None)),
            )
        self.mesh = mesh
        self.logits_processor = LogitsProcessor(config.vocab_size, mesh=mesh)

    def __call__(
        self,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
        logits_metadata: LogitsMetadata,
    ) -> jax.Array:
        hidden_states, kv_fused = self.model(forward_batch, token_to_kv_pool)
        if not getattr(self.config, "tie_word_embeddings", False):
            output = self.logits_processor(hidden_states, self.lm_head, logits_metadata)
        else:
            output = self.logits_processor(hidden_states, self.model.embed_tokens, logits_metadata)
        return output, kv_fused

    def load_weights(self, model_config: ModelConfig):
        loader = WeightLoader(
            model=self,
            model_config=model_config,
            mesh=self.mesh,
            dtype=self.dtype,
        )
        weight_mappings = self._create_weight_mappings()
        loader.load_weights_from_safetensors(weight_mappings)
        logger.info("gpt-oss weights loaded successfully!")

    def _create_weight_mappings(self) -> dict:
        mappings = {
            "model.embed_tokens.weight": WeightMapping(
                target_path="model.embed_tokens.embedding", sharding=(None, None), transpose=False
            ),
            "model.norm.weight": WeightMapping(
                target_path="model.norm.scale", sharding=(None,), transpose=False
            ),
        }
        if not getattr(self.config, "tie_word_embeddings", False):
            mappings["lm_head.weight"] = WeightMapping(
                target_path="lm_head.embedding", sharding=(None, None), transpose=False
            )

        for i in range(self.config.num_hidden_layers):
            prefix = f"model.layers.{i}"
            target = f"model.layers.{i}"
            mappings.update(
                {
                    f"{prefix}.input_layernorm.weight": WeightMapping(
                        target_path=f"{target}.input_layernorm.scale",
                        sharding=(None,),
                        transpose=False,
                    ),
                    f"{prefix}.post_attention_layernorm.weight": WeightMapping(
                        target_path=f"{target}.post_attention_layernorm.scale",
                        sharding=(None,),
                        transpose=False,
                    ),
                    f"{prefix}.self_attn.q_proj.weight": WeightMapping(
                        target_path=f"{target}.self_attn.q_proj.weight",
                        sharding=(None, "tensor"),
                        transpose=True,
                    ),
                    f"{prefix}.self_attn.k_proj.weight": WeightMapping(
                        target_path=f"{target}.self_attn.k_proj.weight",
                        sharding=(None, "tensor"),
                        transpose=True,
                    ),
                    f"{prefix}.self_attn.v_proj.weight": WeightMapping(
                        target_path=f"{target}.self_attn.v_proj.weight",
                        sharding=(None, "tensor"),
                        transpose=True,
                    ),
                    f"{prefix}.self_attn.o_proj.weight": WeightMapping(
                        target_path=f"{target}.self_attn.o_proj.weight",
                        sharding=(None, "tensor"),
                        transpose=True,
                    ),
                }
            )
            if getattr(self.config, "attention_bias", True):
                mappings.update(
                    {
                        f"{prefix}.self_attn.q_proj.bias": WeightMapping(
                            target_path=f"{target}.self_attn.q_proj.bias",
                            sharding=(None,),
                            transpose=False,
                            head_dim_padding=True,
                        ),
                        f"{prefix}.self_attn.k_proj.bias": WeightMapping(
                            target_path=f"{target}.self_attn.k_proj.bias",
                            sharding=(None,),
                            transpose=False,
                            head_dim_padding=True,
                            kv_head_padding=True,
                        ),
                        f"{prefix}.self_attn.v_proj.bias": WeightMapping(
                            target_path=f"{target}.self_attn.v_proj.bias",
                            sharding=(None,),
                            transpose=False,
                            head_dim_padding=True,
                            kv_head_padding=True,
                        ),
                        f"{prefix}.self_attn.o_proj.bias": WeightMapping(
                            target_path=f"{target}.self_attn.o_proj.bias",
                            sharding=(None,),
                            transpose=False,
                        ),
                    }
                )
            mappings.update(
                {
                    f"{prefix}.mlp.router.weight": WeightMapping(
                        target_path=f"{target}.router.kernel", sharding=(None, None), transpose=True
                    ),
                    f"{prefix}.mlp.router.bias": WeightMapping(
                        target_path=f"{target}.router.bias", sharding=(None,), transpose=False
                    ),
                }
            )

        return mappings


EntryClass = GptOssForCausalLM
