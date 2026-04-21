"""MiMo-V2-Pro model implementation for SGLang-JAX.

Inherits from MiMoV2FlashForCausalLM. The main difference is the weight format:
Pro uses a fused qkv_proj instead of separate q/k/v_proj weights. The FP8
checkpoint was quantized per-TP-shard and concatenated, so the fused QKV has
a per-shard-interleaved layout that requires special dequantization handling.
"""

import logging
import math

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from transformers import PretrainedConfig

from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.layers.moe import create_moe_weights_mapping
from sgl_jax.srt.models.mimo_v2_flash import MiMoV2FlashForCausalLM
from sgl_jax.srt.utils.weight_utils import WeightLoader, WeightMapping

logger = logging.getLogger(__name__)


class MiMoV2ProForCausalLM(MiMoV2FlashForCausalLM):

    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh | None = None,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        super().__init__(config, mesh, dtype)
        # Buffer to hold fused QKV FP8 weights/scales before per-shard dequant.
        # Populated during weight loading, consumed by _dequant_fused_qkv.
        self._fused_qkv_buffers: dict[int, dict] = {}

    def load_weights(self, model_config):
        """Load weights with special handling for per-shard-quantized fused QKV."""
        loader = WeightLoader(
            model=self,
            model_config=model_config,
            mesh=self.mesh,
            dtype=self.dtype,
        )
        self._quant_config = model_config.quantization_config
        weight_mappings = self._create_weight_mappings()
        loader.load_weights_from_safetensors(weight_mappings)
        logger.info("MiMoV2Pro weights loaded successfully!")

        if self._is_static_quant:
            # Dequantize fused QKV per-shard, then split into Q/K/V bf16.
            self._dequant_fused_qkv()
            # Dequantize remaining FP8 weights (layer 0 MLP, etc).
            self._dequantize_fp8_to_bf16()

    def _dequant_fused_qkv(self):
        """Dequantize per-shard-interleaved fused QKV FP8 weights.

        The FP8 checkpoint was quantized per-TP-shard and concatenated:
          weight: [shard0_QKV | shard1_QKV | ... | shardN_QKV], shape [total_qkv, hidden]
          scale:  [shard0_scale | shard1_scale | ... | shardN_scale], shape [total_blocks, in_blocks]

        Each shard's QKV dim may not be a multiple of block_size, so scale blocks
        within a shard can span K/V boundaries. We must dequantize per-shard first,
        then extract Q/K/V from each shard.
        """
        if not self._fused_qkv_buffers:
            return

        head_dim = self.config.head_dim
        v_head_dim = getattr(self.config, "v_head_dim", head_dim)
        num_heads = self.config.num_attention_heads
        num_kv_heads = self.config.num_key_value_heads

        quant_cfg = getattr(self.config, "quantization_config", None)
        block_size = int(quant_cfg.weight_block_size[0]) if quant_cfg else 128

        for layer_idx in sorted(self._fused_qkv_buffers.keys()):
            buf = self._fused_qkv_buffers[layer_idx]
            fused_weight = buf["weight"]  # [total_qkv, hidden], FP8, HF layout
            fused_scale = buf["scale"]  # [total_blocks, in_blocks], f32

            in_dim = fused_weight.shape[1]  # hidden_size

            # Infer n_shards: find TP size where per-shard blocking matches scale shape
            n_shards, orig_kv_heads = self._infer_qkv_shards(
                fused_weight.shape[0],
                fused_scale.shape[0],
                num_heads,
                num_kv_heads,
                head_dim,
                v_head_dim,
                block_size,
            )

            per_shard_q = (num_heads // n_shards) * head_dim
            per_shard_k = (orig_kv_heads // n_shards) * head_dim
            per_shard_v = (orig_kv_heads // n_shards) * v_head_dim
            per_shard_total = per_shard_q + per_shard_k + per_shard_v
            per_shard_blocks = math.ceil(per_shard_total / block_size)
            padded_rows = per_shard_blocks * block_size
            in_blocks = in_dim // block_size

            if layer_idx % 10 == 0:
                logger.info(
                    "Layer %d: dequant fused QKV FP8, n_shards=%d, "
                    "per_shard=%d (Q=%d K=%d V=%d), blocks=%d",
                    layer_idx,
                    n_shards,
                    per_shard_total,
                    per_shard_q,
                    per_shard_k,
                    per_shard_v,
                    per_shard_blocks,
                )

            q_parts, k_parts, v_parts = [], [], []

            for shard_idx in range(n_shards):
                # Extract this shard's weight and scale
                w_start = shard_idx * per_shard_total
                shard_w = fused_weight[w_start : w_start + per_shard_total, :]

                s_start = shard_idx * per_shard_blocks
                shard_s = fused_scale[s_start : s_start + per_shard_blocks, :]

                # Pad weight rows to block boundary for dequant
                if per_shard_total < padded_rows:
                    shard_w = jnp.pad(shard_w, ((0, padded_rows - per_shard_total), (0, 0)))

                # Block dequantize: [padded_rows, in_dim] × [blocks, in_blocks]
                shard_f = shard_w.astype(jnp.float32).reshape(
                    per_shard_blocks, block_size, in_blocks, block_size
                )
                shard_s_4d = shard_s[:, None, :, None]
                shard_bf16 = (
                    (shard_f * shard_s_4d)
                    .reshape(padded_rows, in_dim)[:per_shard_total, :]
                    .astype(jnp.bfloat16)
                )

                # Split shard into Q, K, V (contiguous within each shard)
                q_parts.append(shard_bf16[:per_shard_q, :])
                k_parts.append(shard_bf16[per_shard_q : per_shard_q + per_shard_k, :])
                v_parts.append(shard_bf16[per_shard_q + per_shard_k :, :])

            # Concatenate across shards → full Q/K/V in HF layout [out, in]
            q_weight = jnp.concatenate(q_parts, axis=0)
            k_weight = jnp.concatenate(k_parts, axis=0)
            v_weight = jnp.concatenate(v_parts, axis=0)

            # Transpose to model layout [in, out] and shard
            q_weight = jnp.transpose(q_weight)
            k_weight = jnp.transpose(k_weight)
            v_weight = jnp.transpose(v_weight)

            tp_sharding = NamedSharding(self.mesh, P(None, "tensor"))
            q_weight = jax.device_put(q_weight, tp_sharding)
            k_weight = jax.device_put(k_weight, tp_sharding)
            v_weight = jax.device_put(v_weight, tp_sharding)

            # Replace QuantizedLinear with bf16 LinearBase
            attn = self.model.layers[layer_idx].self_attn
            for proj_name, w in [
                ("q_proj", q_weight),
                ("k_proj", k_weight),
                ("v_proj", v_weight),
            ]:
                with jax.set_mesh(self.mesh):
                    new_linear = LinearBase(
                        input_size=in_dim,
                        output_size=w.shape[1],
                        kernel_axes=(None, "tensor"),
                        use_bias=False,
                        params_dtype=jnp.bfloat16,
                        mesh=self.mesh,
                    )
                    new_linear.weight = nnx.Param(w)
                setattr(attn, proj_name, new_linear)

            if layer_idx == 0:
                logger.info(
                    "Layer 0 dequant result: Q=%s K=%s V=%s",
                    q_weight.shape,
                    k_weight.shape,
                    v_weight.shape,
                )

        # Clean up buffers
        self._fused_qkv_buffers.clear()
        logger.info("Fused QKV FP8 dequantization complete for all layers.")

        # Replicate KV heads for TP alignment
        self._ensure_kv_head_replication()

    @staticmethod
    def _infer_qkv_shards(
        total_out_dim: int,
        total_scale_blocks: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        v_head_dim: int,
        block_size: int,
    ) -> tuple[int, int]:
        """Infer the number of TP shards used during FP8 quantization.

        The config's num_kv_heads may be GQA-replicated (e.g. 32 instead of
        the original 8), so we also try divisors of num_kv_heads to find the
        original value used during per-shard quantization.

        Returns (tp_shards, original_num_kv_heads).
        """
        # Collect candidate kv_heads values: config value and its divisors
        kv_candidates = []
        for d in range(1, num_kv_heads + 1):
            if num_kv_heads % d == 0:
                kv_candidates.append(d)
        # Try config value first, then smaller divisors (descending)
        kv_candidates = sorted(set(kv_candidates), reverse=True)

        # TP candidates: all divisors of num_heads (covers any quantization-time TP)
        tp_candidates = sorted(d for d in range(1, num_heads + 1) if num_heads % d == 0)

        for orig_kv in kv_candidates:
            for tp in tp_candidates:
                if orig_kv % tp != 0:
                    continue
                per_shard = (
                    (num_heads // tp) * head_dim
                    + (orig_kv // tp) * head_dim
                    + (orig_kv // tp) * v_head_dim
                )
                per_shard_blocks = math.ceil(per_shard / block_size)
                if per_shard_blocks * tp == total_scale_blocks and per_shard * tp == total_out_dim:
                    if orig_kv != num_kv_heads:
                        logger.info(
                            "Inferred original num_kv_heads=%d (config has %d), tp=%d",
                            orig_kv,
                            num_kv_heads,
                            tp,
                        )
                    return tp, orig_kv
        raise ValueError(
            f"Cannot infer QKV shard count: out_dim={total_out_dim}, "
            f"scale_blocks={total_scale_blocks}, num_heads={num_heads}, "
            f"num_kv_heads={num_kv_heads}, head_dim={head_dim}, "
            f"v_head_dim={v_head_dim}, block_size={block_size}"
        )

    def _create_layer_mappings(self, layer_idx: int) -> dict:
        """Override to handle fused qkv_proj weights in MiMo-V2-Pro checkpoints."""
        prefix = f"model.layers.{layer_idx}"
        target = prefix

        mappings = {}
        is_fp8 = self._is_static_quant

        # --- Fused QKV projection ---
        hf_qkv_key = f"{prefix}.self_attn.qkv_proj"
        qkv_ignored = self._is_quant_ignored(hf_qkv_key)

        if is_fp8 and not qkv_ignored:
            # FP8 fused QKV: store raw weight+scale in buffer, dequant post-load.
            # Use a callback mapping that stores into _fused_qkv_buffers instead of
            # trying to split the per-shard-interleaved data.
            mappings[f"{hf_qkv_key}.weight"] = WeightMapping(
                target_path=f"__FUSED_QKV_WEIGHT__{layer_idx}",
                sharding=(None, None),
                transpose=False,
            )
            mappings[f"{hf_qkv_key}.weight_scale_inv"] = WeightMapping(
                target_path=f"__FUSED_QKV_SCALE__{layer_idx}",
                sharding=(None, None),
                transpose=False,
            )
        else:
            # BF16 or ignored: split normally (contiguous Q/K/V layout is fine)
            qkv_weight_suffix = "weight"
            mappings[f"{hf_qkv_key}.weight"] = WeightMapping(
                target_path=[
                    f"{target}.self_attn.q_proj.{qkv_weight_suffix}",
                    f"{target}.self_attn.k_proj.{qkv_weight_suffix}",
                    f"{target}.self_attn.v_proj.{qkv_weight_suffix}",
                ],
                sharding=(None, "tensor"),
                transpose=True,
                head_dim_padding=False,
                kv_head_padding=True,
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
