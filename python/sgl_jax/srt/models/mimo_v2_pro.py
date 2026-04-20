"""MiMo-V2-Pro model implementation for SGLang-JAX.

Inherits from MiMoV2FlashForCausalLM. The only difference is the weight format:
Pro uses a fused qkv_proj instead of separate q/k/v_proj weights.
"""

import logging

import jax
import jax.numpy as jnp
from transformers import PretrainedConfig

from sgl_jax.srt.layers.moe import create_moe_weights_mapping
from sgl_jax.srt.models.mimo_v2_flash import MiMoV2FlashForCausalLM
from sgl_jax.srt.utils.weight_utils import WeightMapping

logger = logging.getLogger(__name__)


class MiMoV2ProForCausalLM(MiMoV2FlashForCausalLM):

    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh | None = None,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        super().__init__(config, mesh, dtype)

    def _create_layer_mappings(self, layer_idx: int) -> dict:
        """Override to handle fused qkv_proj weights in MiMo-V2-Pro checkpoints."""
        prefix = f"model.layers.{layer_idx}"
        target = prefix

        mappings = {}
        is_fp8 = self._is_static_quant

        # --- Fused QKV projection ---
        hf_qkv_key = f"{prefix}.self_attn.qkv_proj"
        qkv_ignored = self._is_quant_ignored(hf_qkv_key)
        qkv_weight_suffix = "weight" if (not is_fp8 or qkv_ignored) else "weight_q"

        mappings[f"{hf_qkv_key}.weight"] = WeightMapping(
            target_path=[
                f"{target}.self_attn.q_proj.{qkv_weight_suffix}",
                f"{target}.self_attn.k_proj.{qkv_weight_suffix}",
                f"{target}.self_attn.v_proj.{qkv_weight_suffix}",
            ],
            sharding=(None, "tensor"),
            transpose=True,
            head_dim_padding=False,
            kv_head_padding=not is_fp8,
        )

        if is_fp8 and not qkv_ignored:
            mappings[f"{hf_qkv_key}.weight_scale_inv"] = WeightMapping(
                target_path=[
                    f"{target}.self_attn.q_proj.weight_scale",
                    f"{target}.self_attn.k_proj.weight_scale",
                    f"{target}.self_attn.v_proj.weight_scale",
                ],
                sharding=(None, None),
                transpose=False,
                head_dim_padding=False,
            )

        # --- o_proj (separate, same as Flash) ---
        hf_o_key = f"{prefix}.self_attn.o_proj"
        o_ignored = self._is_quant_ignored(hf_o_key)
        o_weight_suffix = "weight" if (not is_fp8 or o_ignored) else "weight_q"

        mappings[f"{hf_o_key}.weight"] = WeightMapping(
            target_path=f"{target}.self_attn.o_proj.{o_weight_suffix}",
            sharding=("tensor", None),
            transpose=True,
            head_dim_padding=True,
        )

        if is_fp8 and not o_ignored:
            mappings[f"{hf_o_key}.weight_scale_inv"] = WeightMapping(
                target_path=f"{target}.self_attn.o_proj.weight_scale",
                sharding=(None, None),
                transpose=False,
            )

        # --- Attention sink bias (same as Flash) ---
        is_swa = (
            hasattr(self.config, "hybrid_layer_pattern")
            and 0 <= layer_idx < len(self.config.hybrid_layer_pattern)
            and self.config.hybrid_layer_pattern[layer_idx] == 1
        )
        has_sink_bias = (is_swa and getattr(self.config, "add_swa_attention_sink_bias", False)) or (
            not is_swa and getattr(self.config, "add_full_attention_sink_bias", False)
        )
        if has_sink_bias:
            mappings[f"{prefix}.self_attn.attention_sink_bias"] = WeightMapping(
                target_path=f"{target}.self_attn.attention_sink_bias",
                sharding=("tensor",),
                transpose=False,
            )

        # --- Layernorms (same as Flash) ---
        mappings[f"{prefix}.input_layernorm.weight"] = WeightMapping(
            target_path=f"{target}.input_layernorm.scale",
            sharding=(None,),
            transpose=False,
        )
        mappings[f"{prefix}.post_attention_layernorm.weight"] = WeightMapping(
            target_path=f"{target}.post_attention_layernorm.scale",
            sharding=(None,),
            transpose=False,
        )

        # --- MLP / MoE (same as Flash) ---
        is_sparse = (
            hasattr(self.config, "moe_layer_freq")
            and 0 <= layer_idx < len(self.config.moe_layer_freq)
            and not isinstance(self.config.moe_layer_freq, int)
            and self.config.moe_layer_freq[layer_idx]
        )

        if is_sparse:
            mappings[f"{prefix}.mlp.gate.weight"] = WeightMapping(
                target_path=f"{target}.mlp.moe_gate.kernel",
                sharding=(None, None),
                transpose=True,
            )

            if getattr(self.config, "topk_method", "greedy") == "noaux_tc":
                mappings[f"{prefix}.mlp.gate.e_score_correction_bias"] = WeightMapping(
                    target_path=f"{target}.mlp.correction_bias",
                    sharding=(None,),
                    transpose=False,
                )

            num_experts = getattr(
                self.config,
                "n_routed_experts",
                getattr(self.config, "num_experts", 8),
            )
            moe_backend = getattr(self.config, "moe_backend", "epmoe")

            moe_mappings = create_moe_weights_mapping(
                prefix=prefix,
                target_prefix=target,
                num_experts=num_experts,
                moe_backend=moe_backend,
                moe_path="mlp.experts",
                source_expert_pattern="{i}",
            )

            if is_fp8:
                augmented = {}
                use_model_mesh_for_scale = moe_backend == "fused"
                for key, mapping in moe_mappings.items():
                    augmented[key] = mapping
                    target_param = mapping.target_path[0]
                    src_paths = mapping.target_path[1:]
                    scale_key = key + "_scale"
                    scale_target = target_param + "_scale"
                    scale_srcs = [p.replace(".weight", ".weight_scale_inv") for p in src_paths]
                    scale_sharding = (
                        (("data", "tensor"), None, None)
                        if use_model_mesh_for_scale
                        else ("expert", None, None)
                    )
                    augmented[scale_key] = WeightMapping(
                        target_path=[scale_target] + scale_srcs,
                        sharding=scale_sharding,
                        transpose=use_model_mesh_for_scale,
                        concat_axis=mapping.concat_axis,
                        physical_to_logical_map=mapping.physical_to_logical_map,
                    )
                moe_mappings = augmented

            mappings.update(moe_mappings)
        else:
            for proj, sharding in [
                ("gate_proj", (None, "tensor")),
                ("up_proj", (None, "tensor")),
                ("down_proj", ("tensor", None)),
            ]:
                hf_key = f"{prefix}.mlp.{proj}"
                weight_suffix = "weight_q" if is_fp8 else "weight"
                mappings[f"{hf_key}.weight"] = WeightMapping(
                    target_path=f"{target}.mlp.{proj}.{weight_suffix}",
                    sharding=sharding,
                    transpose=True,
                )
                if is_fp8:
                    mappings[f"{hf_key}.weight_scale_inv"] = WeightMapping(
                        target_path=f"{target}.mlp.{proj}.weight_scale",
                        sharding=(None, None),
                        transpose=False,
                    )

        return mappings


EntryClass = MiMoV2ProForCausalLM
