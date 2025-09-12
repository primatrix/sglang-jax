import functools
import logging
from typing import Any, Dict, Iterable, Optional, Tuple

from flax import nnx
from jax import jax
from jax import numpy as jnp
from jax.sharding import get_abstract_mesh
from transformers import PretrainedConfig

from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.layers.activation import GeluAndMul
from sgl_jax.srt.layers.embeddings import (
    Embed,
    ParallelLMHead,
    RotaryEmbedding,
    ScalingRotaryEmbedding,
)
from sgl_jax.srt.layers.layernorm import RMSNorm
from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sgl_jax.srt.layers.moe import EPMoE, GateLogit
from sgl_jax.srt.layers.radix_attention import RadixAttention
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
from sgl_jax.srt.utils.weight_utils import WeightLoader, WeightMapping

logger = logging.getLogger(__name__)

init_fn = nnx.initializers.uniform()


def get_rope_scaling(config):
    rope_type = getattr(config, "rope_type", None)
    if rope_type:
        original_max_position_embeddings = getattr(
            config, "original_max_position_embeddings", None
        )
        scaling_factor = getattr(config, "scaling_factor", None)
        extrapolation_factor = getattr(config, "extrapolation_factor", 1.0)
        attn_factor = getattr(config, "attn_factor", 1.0)
        beta_fast = getattr(config, "beta_fast", 32)
        beta_slow = getattr(config, "beta_slow", 1)
        rope_scaling = {
            "extra_method": rope_type,
            "max_position_embeddings": original_max_position_embeddings,
            "scaling_factor": scaling_factor,
            "extrapolation_factor": extrapolation_factor,
            "attn_factor": attn_factor,
            "beta_fast": beta_fast,
            "beta_slow": beta_slow,
            "dtype": jnp.float32,
        }
        return rope_scaling
    else:
        return None


class Grok1MLP(nnx.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        layer_id: int,
        rngs: nnx.Rngs = None,
        dtype: jnp.dtype = jnp.bfloat16,
    ) -> None:
        super().__init__()

        self.gate_up_proj = LinearBase(
            input_size=hidden_size,
            output_size=[intermediate_size] * 2,
            use_bias=False,
            params_dtype=dtype,
            kernel_axes=(None, "tensor"),
            rngs=rngs,
        )
        self.down_proj = LinearBase(
            input_size=intermediate_size,
            output_size=hidden_size,
            use_bias=False,
            params_dtype=dtype,
            kernel_axes=("tensor", None),
            rngs=rngs,
        )
        self.act_fn = GeluAndMul(approximate="tanh")
        self.layer_id = layer_id

    def __call__(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x, _ = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class Grok1MoE(nnx.Module):
    """A tensor-parallel MoE implementation for Grok1 that shards each expert
    across all ranks.

    Each expert's weights are sharded across all ranks and a fused MoE
    kernel is used for the forward pass, and finally we reduce the outputs
    across ranks.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        rngs: Optional[nnx.Rngs] = None,
        dtype: jnp.dtype = jnp.bfloat16,
        mesh: Optional[jax.sharding.Mesh] = None,
    ):
        super().__init__()

        # Gate always runs at full precision for stability (see https://arxiv.org/pdf/2101.03961)
        self.gate = LinearBase(
            input_size=config.hidden_size,
            output_size=num_experts,
            bias=False,
            params_dtype=jnp.float32,
        )

        self.router_logit_softcapping = getattr(config, "router_logit_softcapping", 30)

        expert_parallel_size = mesh.shape.get("data", 1) * mesh.shape.get("tensor", 1)
        num_experts = getattr(config, "num_experts", 128)
        with mesh:
            self.experts = EPMoE(
                config=config,
                num_experts=config.num_experts,
                num_experts_per_tok=config.num_experts_per_tok,
                expert_parallel_size=expert_parallel_size,
                use_moe_router_tok=True,
                mesh=mesh,
                intermediate_dim=config.moe_intermediate_size,
                dtype=dtype,
                activation="gelu",
                layer_id=layer_id,
                rngs=rngs,
            )

    def __call__(self, hidden_states: jax.Array) -> jax.Array:
        router_logits = self.gate(hidden_states)
        if self.router_logit_softcapping != 0:
            router_logits = router_logits / self.router_logit_softcapping
            router_logits = jax.nn.tanh(router_logits)
        return self.experts(hidden_states, router_logits)


class Grok1Attention(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        layer_id: int = 0,
        max_position: int = 4096 * 32,
        rope_theta: float = 10000,
        reduce_results: bool = True,
        load_presharded_attn: bool = False,
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.hidden_size = hidden_size
        self.total_num_heads = num_heads
        self.num_heads = self.total_num_heads
        self.total_num_kv_heads = num_kv_heads
        self.num_kv_heads = self.total_num_kv_heads
        self.head_dim = getattr(config, "head_dim", 128)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        rope_scaling = get_rope_scaling(config)
        self.load_presharded_attn = load_presharded_attn

        num_heads = self.total_num_heads + self.total_num_kv_heads
        self.qkv_proj = LinearBase(
            input_size=hidden_size,
            output_size=num_heads * self.head_dim,
            bias=False,
            load_presharded_attn=self.load_presharded_attn,
            kernel_axes=(None, "tensor"),
        )
        self.o_proj = LinearBase(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            reduce_results=reduce_results,
            use_presharded_weights=self.load_presharded_attn,
            kernel_axes=("tensor", None),
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=int(self.rope_theta),
            is_neox_style=True,
        )

        self.rope_rotate_half_dims = getattr(config, "rope_rotate_half_dims", False)

        if rope_scaling is not None:
            self.rotary_emb = ScalingRotaryEmbedding(
                self.head_dim,
                rotary_dim=(
                    self.head_dim
                    if not self.rope_rotate_half_dims
                    else self.head_dim // 2
                ),
                base=int(self.rope_theta),
                is_neox_style=True,
                **rope_scaling,
            )
            pos_encoding_mode = "NONE"
        else:
            self.rotary_emb = get_rope(
                self.head_dim,
                rotary_dim=(
                    self.head_dim
                    if not self.rope_rotate_half_dims
                    else self.head_dim // 2
                ),
                max_position=max_position,
                base=int(self.rope_theta),
                is_neox_style=True,
            )
            pos_encoding_mode = "NONE"

        logit_cap = max(getattr(config, "attn_logit_softcapping", 30.0), 0.0)
        logit_capping_method = getattr(config, "attn_logit_softcapping_method", "tanh")

        self.attn = RadixAttention(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            scaling=self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            logit_cap=logit_cap,
            pos_encoding_mode=pos_encoding_mode,
            logit_capping_method=logit_capping_method,
        )
        self.attn.xai_temperature_len = getattr(self.config, "attn_temperature_len", -1)

    def forward(
        self,
        positions: jax.Array,
        hidden_states: jax.Array,
        forward_batch: ForwardBatch,
    ) -> jax.Array:
        if hidden_states.shape[0] == 0:
            assert (
                not self.o_proj.reduce_results
            ), "short-circuiting allreduce will lead to hangs"
            return hidden_states

        qkv, _ = self.qkv_proj(hidden_states)

        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)

        attn_output = self.attn(q, k, v, forward_batch)

        output, _ = self.o_proj(attn_output)
        return output


class Grok1DecoderLayer(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int = 0,
        load_presharded_moe: bool = False,
        load_presharded_attn: bool = False,
        load_presharded_mlp: bool = False,
    ) -> None:
        super().__init__()
        self.num_experts = config.num_local_experts
        self.hidden_size = config.hidden_size
        self.residual_moe = getattr(config, "residual_moe", False)
        self.layer_id = layer_id

        rope_theta = getattr(config, "rope_theta", 10000)
        self.self_attn = Grok1Attention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            max_position=(
                config.context_len
                if hasattr(config, "context_len")
                else config.max_position_embeddings
            ),
            num_kv_heads=config.num_key_value_heads,
            layer_id=layer_id,
            rope_theta=rope_theta,
            reduce_results=False,
            alt_stream=self.alt_stream,
            load_presharded_attn=load_presharded_attn,
        )

        split_gate_up = not getattr(config, "merge_gate_up", True)
        if self.num_experts > 0:
            self.block_sparse_moe = Grok1MoE(
                config=config,
                layer_id=layer_id,
                reduce_results=not self.residual_moe,
                use_presharded_weights=load_presharded_moe,
                inplace=False,  # not self.residual_moe,
                no_combine=False,  # self.residual_moe,  # just a suggestion to not combine topk
            )
            if self.residual_moe:
                self.mlp = Grok1MLP(
                    hidden_size=config.hidden_size,
                    intermediate_size=config.intermediate_size,
                    reduce_results=False,
                    use_presharded_weights=load_presharded_mlp,
                    layer_id=layer_id,
                    split_gate_up=split_gate_up,
                )
        else:
            raise NotImplementedError()

        self.pre_attn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_moe_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_moe_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        if self.num_experts > 0:
            if self.residual_moe:
                # NOTE: self.block_sparse_moe modifies the input in-place,
                # so we have to call it later. Be aware of any possible related errors.
                if get_tensor_model_parallel_world_size() > 1:
                    self.ffn = lambda x: tensor_model_parallel_all_reduce(
                        self.moe_with_rmoe(x)
                    )
                else:
                    self.ffn = self.moe_with_rmoe
            else:
                self.ffn = self.block_sparse_moe
        else:
            raise NotImplementedError()

    def forward(
        self,
        positions: jax.Array,
        hidden_states: jax.Array,
        forward_batch: ForwardBatch,
        residual: Optional[jax.Array] = None,
        deferred_norm: Optional[RMSNorm] = None,
    ) -> Tuple[jax.Array, jax.Array, RMSNorm]:

        hidden_states_original = hidden_states
        residual_original = residual

        # Self Attention
        if deferred_norm is not None:
            assert residual is not None
            # here hidden_states is output of ffn, residual is residual from after previous attn layer
            hidden_states, residual = fused_dual_residual_rmsnorm(
                hidden_states,
                residual,
                deferred_norm.weight,
                self.pre_attn_norm.weight,
                deferred_norm.variance_epsilon,
            )
        else:
            # here hidden_states is the residual
            hidden_states, residual = self.pre_attn_norm(hidden_states), hidden_states

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )

        hidden_states = jax.lax.psum(hidden_states, axis_name="tensor")

        hidden_states, residual = fused_dual_residual_rmsnorm(
            hidden_states,
            residual,
            self.post_attn_norm.weight,
            self.pre_moe_norm.weight,
            self.post_attn_norm.variance_epsilon,
        )

        # Fully Connected
        hidden_states = self.ffn(hidden_states)
        return hidden_states, residual, self.post_moe_norm  # defer layernorm

    def moe_with_rmoe(self, x):
        mlp_result = self.mlp(x)
        moe_result = self.block_sparse_moe(x)
        return (mlp_result + moe_result) / 1.4142135623730951


class Grok1Model(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        load_presharded_moe: bool = False,
        load_presharded_embedding: bool = False,
        load_presharded_attn: bool = False,
        load_presharded_mlp: bool = False,
        replicate_embedding: bool = False,
    ) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            use_presharded_weights=load_presharded_embedding,
            enable_tp=not replicate_embedding,
        )

        self.layers = [
            Grok1DecoderLayer(
                config,
                i,
                load_presharded_moe=load_presharded_moe,
                load_presharded_attn=load_presharded_attn,
                load_presharded_mlp=load_presharded_mlp,
            )
            for i in range(config.num_hidden_layers)
        ]
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: jax.Array,
        positions: jax.Array,
        forward_batch: ForwardBatch,
        input_embeds: jax.Array = None,
    ) -> jax.Array:
        if input_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
            hidden_states.mul_(self.config.embedding_multiplier_scale)
        else:
            hidden_states = input_embeds

        residual, deferred_norm = None, None
        for i in range(len(self.layers)):
            hidden_states, residual, deferred_norm = self.layers[i](
                positions, hidden_states, forward_batch, residual, deferred_norm
            )

        hidden_states, _ = fused_dual_residual_rmsnorm(
            hidden_states,
            residual,
            deferred_norm.weight,
            self.norm.weight,
            deferred_norm.variance_epsilon,
        )

        return hidden_states


class Grok1ForCausalLM(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config

        # Get presharded weights.
        self.load_presharded_mlp = getattr(config, "load_presharded_mlp", False)
        self.load_presharded_moe = (
            getattr(config, "load_presharded_moe", True)
            and self.config.num_local_experts > 0
            and get_tensor_model_parallel_world_size() > 1
        )
        self.load_presharded_attn = getattr(config, "load_presharded_attn", False)
        self.load_presharded_embedding = getattr(
            config, "load_presharded_embedding", False
        )

        self.is_weights_presharded = (
            self.load_presharded_mlp
            or self.load_presharded_moe
            or self.load_presharded_attn
            or self.load_presharded_embedding
        )

        default_replicate_lm_head = False
        self.replicate_lm_head = getattr(
            config, "replicate_lm_head", default_replicate_lm_head
        )

        if self.is_weights_presharded:
            setattr(DefaultModelLoader, "_prepare_weights", _prepare_presharded_weights)

        self.replicate_embedding = getattr(config, "replicate_embedding", False)

        self.model = Grok1Model(
            config,
            load_presharded_moe=self.load_presharded_moe,
            load_presharded_embedding=self.load_presharded_embedding,
            load_presharded_attn=self.load_presharded_attn,
            load_presharded_mlp=self.load_presharded_mlp,
            replicate_embedding=self.replicate_embedding,
        )

        lm_head_params_dtype = None
        if self.replicate_lm_head:
            self.lm_head = ReplicatedLinear(
                config.hidden_size,
                config.vocab_size,
                bias=False,
                params_dtype=lm_head_params_dtype,
            )
            self.logits_processor = LogitsProcessor(config, skip_all_gather=True)
        else:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                use_presharded_weights=self.load_presharded_embedding,
                params_dtype=lm_head_params_dtype,
            )
            self.logits_processor = LogitsProcessor(config)

        self.loaded_param_names = set()

    def forward(
        self,
        input_ids: jax.Array,
        positions: jax.Array,
        forward_batch: ForwardBatch,
        input_embeds: Optional[jax.Array] = None,
    ) -> jax.Array:
        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    def load_weights(
        self,
        weights: Iterable[Tuple[str, jax.Array]],
        ignore_parent_name: bool = False,
        check_hit_names: bool = True,
        model_config: PretrainedConfig | None = None,
    ) -> dict[str, jax.Array]:
        if model_config is None:
            model_config = self.config

        stacked_params_mapping = []
        stacked_params_mapping += [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]
        stacked_params_mapping += [
            # (param_name, shard_name, shard_id)
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        num_experts = model_config.num_local_experts
        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="w1",
            ckpt_down_proj_name="w2",
            ckpt_up_proj_name="w3",
            num_experts=num_experts,
        )

        params_dict = dict(self.named_parameters())
        all_names = set(params_dict.keys())
        hit_names = set()

        def load_weight_wrapper(name: str, loaded_weight: jax.Array, *args, **kwargs):
            # Fuse constant multipliers into the weights
            if "lm_head" in name:
                loaded_weight = (
                    loaded_weight.to(jnp.float32) * model_config.output_multiplier_scale
                )

            original_name = name
            if ignore_parent_name:
                name = name.split(".")[-1]

            if name not in params_dict:
                logger.info(f"Skipping {name=} in load_weights_wrapper")
                return

            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight, *args, **kwargs)
            hit_names.add(name)
            self.loaded_param_names.add(original_name)

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                load_weight_wrapper(name, loaded_weight, shard_id)
                break
            else:
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)

                    load_weight_wrapper(
                        name,
                        loaded_weight,
                        name,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )
                    break
                else:
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    if name is None:
                        continue

                    load_weight_wrapper(name=name, loaded_weight=loaded_weight)

        if check_hit_names:
            if len(hit_names) > 5:
                missing = all_names - hit_names
                missing_exclude_scales = {x for x in missing if "scale" not in x}
                logger.info(
                    f"#all_names: {len(all_names)}, #hit_names: {len(hit_names)}, #missing_exclude_scales: {len(missing_exclude_scales)}",
                )
                if len(missing_exclude_scales) > 0:
                    raise ValueError(
                        f"load_weights failed because some weights are missing: {missing_exclude_scales=}."
                    )

            elif len(hit_names) == 0:
                raise ValueError(
                    f"load_weights failed because it did not hit any names. {all_names=} {hit_names=}"
                )

        return hit_names

    def get_num_params_analytical(self):
        cfg = self.config
        moe_intermediate_size = getattr(
            cfg,
            "moe_intermediate_size",
            getattr(cfg, "intermediate_size", None),
        )
        residual_moe = getattr(cfg, "residual_moe", False)
        if cfg.num_local_experts > 0:
            num_experts = cfg.num_local_experts + (1 if residual_moe else 0)
        else:
            num_experts = 1

        wq = (
            cfg.num_hidden_layers
            * cfg.hidden_size
            * cfg.num_attention_heads
            * cfg.head_dim
        )
        wkv = (
            cfg.num_hidden_layers
            * cfg.hidden_size
            * cfg.num_key_value_heads
            * cfg.head_dim
            * 2
        )
        out = (
            cfg.num_hidden_layers
            * cfg.hidden_size
            * cfg.num_attention_heads
            * cfg.head_dim
        )
        ffn1 = (
            cfg.num_hidden_layers
            * num_experts
            * cfg.hidden_size
            * moe_intermediate_size
            * 2
        )
        ffn2 = (
            cfg.num_hidden_layers
            * num_experts
            * cfg.hidden_size
            * moe_intermediate_size
        )
        embed = cfg.hidden_size * cfg.vocab_size * 2
        return wq + wkv + out + ffn1 + ffn2 + embed
