"""MiMo-V2-Pro model implementation for SGLang-JAX.

Architecture is identical to MiMo-V2-Flash (separated q/k/v/o projections,
hybrid SWA/FA layers, MoE). The key difference is weight loading: Pro uses
a fused ``qkv_proj`` in the HuggingFace checkpoint which must be split into
separate q/k/v during loading.

Critical: the FP8 checkpoint was quantized per TP-shard (TP=8) before merging.
This means the weight and scale are stored in shard-interleaved order, NOT
projection-grouped. We must dequant the fused weight/scale as a unit using
per-shard block mapping, then split into Q/K/V and regroup.
"""

import glob
import logging
import math
import os

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.layers.linear import LinearBase, QuantizedLinear
from sgl_jax.srt.models.mimo_v2_flash import MiMoV2FlashForCausalLM
from sgl_jax.srt.utils.weight_utils import WeightLoader, WeightMapping

logger = logging.getLogger(__name__)


class MiMoV2ProForCausalLM(MiMoV2FlashForCausalLM):
    """MiMo-V2-Pro: same architecture as Flash, fused QKV weight loading."""

    def load_weights(self, model_config: ModelConfig):
        self._quant_config = model_config.quantization_config
        self._original_num_kv_heads = model_config.get_total_num_kv_heads()

        # Load non-QKV weights via standard pipeline
        weight_mappings = self._create_weight_mappings()
        loader = WeightLoader(
            model=self,
            model_config=model_config,
            mesh=self.mesh,
            dtype=self.dtype,
        )
        loader.load_weights_from_safetensors(weight_mappings)
        logger.info("Standard (non-QKV) weights loaded.")

        # Load fused QKV: dequant with per-shard block mapping, then split
        if self._is_static_quant:
            self._load_and_dequant_fused_qkv(model_config)
            # Dequant layer 0 dense MLP (same as Flash)
            self._dequantize_layer0_mlp()
            self._ensure_kv_head_replication()

        logger.info("MiMoV2Pro weights loaded successfully!")

    def _create_layer_mappings(self, layer_idx: int) -> dict:
        # Get Flash mappings (MoE, layernorms, attention_sink_bias, q/k/v/o)
        mappings = super()._create_layer_mappings(layer_idx)

        prefix = f"model.layers.{layer_idx}"

        # Remove Q/K/V mappings — they are loaded manually via fused dequant
        for proj in ("q_proj", "k_proj", "v_proj"):
            mappings.pop(f"{prefix}.self_attn.{proj}.weight", None)
            mappings.pop(f"{prefix}.self_attn.{proj}.weight_scale_inv", None)

        # Override o_proj mapping for Pro (bf16, in ignored_layers)
        mappings.pop(f"{prefix}.self_attn.o_proj.weight", None)
        mappings.pop(f"{prefix}.self_attn.o_proj.weight_scale_inv", None)

        o_hf_key = f"{prefix}.self_attn.o_proj"
        o_ignored = self._is_quant_ignored(o_hf_key)
        o_weight_suffix = "weight" if (not self._is_static_quant or o_ignored) else "weight_q"
        mappings[f"{o_hf_key}.weight"] = WeightMapping(
            target_path=f"{prefix}.self_attn.o_proj.{o_weight_suffix}",
            sharding=("tensor", None),
            transpose=True,
            head_dim_padding=True,
        )
        if self._is_static_quant and not o_ignored:
            mappings[f"{o_hf_key}.weight_scale_inv"] = WeightMapping(
                target_path=f"{prefix}.self_attn.o_proj.weight_scale",
                sharding=(None, None),
                transpose=False,
            )

        return mappings

    def _load_and_dequant_fused_qkv(self, model_config: ModelConfig):
        """Load fused QKV from safetensors, dequant with per-shard block mapping, split and assign."""
        import ml_dtypes

        model_dir = model_config.model_path
        safetensors_files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))

        # Per-shard layout parameters (quantized with TP=8)
        num_quant_shards = 8
        original_kv_heads = self._original_num_kv_heads  # 8
        num_q_heads = self.config.num_attention_heads  # 128
        head_dim = self.config.head_dim  # 192
        v_head_dim = getattr(self.config, "v_head_dim", head_dim)  # 128
        hidden_size = self.config.hidden_size  # 6144

        q_heads_per_shard = num_q_heads // num_quant_shards  # 16
        kv_heads_per_shard = original_kv_heads // num_quant_shards  # 1

        q_per_shard = q_heads_per_shard * head_dim  # 3072
        k_per_shard = kv_heads_per_shard * head_dim  # 192
        v_per_shard = kv_heads_per_shard * v_head_dim  # 128
        shard_rows = q_per_shard + k_per_shard + v_per_shard  # 3392
        total_rows = shard_rows * num_quant_shards  # 27136

        block_size = 128
        blocks_per_shard = math.ceil(shard_rows / block_size)  # 27
        total_scale_blocks = blocks_per_shard * num_quant_shards  # 216

        logger.info(
            "Fused QKV layout: %d quant shards, shard_rows=%d (Q=%d K=%d V=%d), "
            "blocks_per_shard=%d, total_scale_blocks=%d",
            num_quant_shards, shard_rows, q_per_shard, k_per_shard, v_per_shard,
            blocks_per_shard, total_scale_blocks,
        )

        # Pre-compute per-shard row→block mapping (numpy, reused for all layers)
        row_indices = np.arange(total_rows)
        shard_of_row = row_indices // shard_rows
        local_row = row_indices % shard_rows
        local_block = local_row // block_size
        global_block = shard_of_row * blocks_per_shard + local_block  # [27136]

        col_block_indices = np.arange(hidden_size) // block_size  # [6144] -> [48 unique values]

        # Build key→file index for fast lookup
        key_to_file = {}
        for f in safetensors_files:
            from safetensors import safe_open
            with safe_open(f, framework="numpy") as sf:
                for k in sf.keys():
                    key_to_file[k] = f

        for layer_idx in range(self.config.num_hidden_layers):
            w_key = f"model.layers.{layer_idx}.self_attn.qkv_proj.weight"
            s_key = f"model.layers.{layer_idx}.self_attn.qkv_proj.weight_scale_inv"

            # Load raw tensors
            with safe_open(key_to_file[w_key], framework="numpy") as sf:
                weight_raw = sf.get_tensor(w_key)
            with safe_open(key_to_file[s_key], framework="numpy") as sf:
                scale_raw = sf.get_tensor(s_key)

            # Convert to float for dequant
            weight_fp8 = weight_raw.view(ml_dtypes.float8_e4m3fn)
            assert weight_fp8.shape == (total_rows, hidden_size), \
                f"Expected ({total_rows}, {hidden_size}), got {weight_fp8.shape}"
            assert scale_raw.shape == (total_scale_blocks, hidden_size // block_size), \
                f"Expected ({total_scale_blocks}, {hidden_size // block_size}), got {scale_raw.shape}"

            weight_f32 = weight_fp8.astype(np.float32)

            # Per-shard block dequant (vectorized)
            scale_per_row = scale_raw[global_block]  # [27136, 48]
            scale_expanded = scale_per_row[:, col_block_indices]  # [27136, 6144]
            weight_dequant = weight_f32 * scale_expanded  # [27136, 6144]

            # Split per-shard, regroup by projection
            weight_reshaped = weight_dequant.reshape(num_quant_shards, shard_rows, hidden_size)
            q_all = weight_reshaped[:, :q_per_shard, :].reshape(-1, hidden_size)
            k_all = weight_reshaped[:, q_per_shard:q_per_shard + k_per_shard, :].reshape(-1, hidden_size)
            v_all = weight_reshaped[:, q_per_shard + k_per_shard:, :].reshape(-1, hidden_size)

            # Transpose to model layout [in_dim, out_dim]
            q_weight = q_all.T  # [6144, 24576]
            k_weight = k_all.T  # [6144, 1536]
            v_weight = v_all.T  # [6144, 1024]

            if layer_idx == 0:
                logger.info(
                    "Layer 0 fused QKV dequant: q=%s k=%s v=%s "
                    "q_mean=%.8f k_mean=%.8f v_mean=%.8f",
                    q_weight.shape, k_weight.shape, v_weight.shape,
                    q_weight.mean(), k_weight.mean(), v_weight.mean(),
                )

            # Convert to JAX bfloat16, shard, and assign to model
            attn = self.model.layers[layer_idx].self_attn
            qkv_sharding = jax.sharding.NamedSharding(self.mesh, P(None, "tensor"))

            for name, w_np in [("q_proj", q_weight), ("k_proj", k_weight), ("v_proj", v_weight)]:
                w_jax = jnp.array(w_np, dtype=jnp.bfloat16)
                w_sharded = jax.device_put(w_jax, qkv_sharding)

                out_dim = w_np.shape[1]
                with jax.set_mesh(self.mesh):
                    new_linear = LinearBase(
                        input_size=hidden_size,
                        output_size=out_dim,
                        kernel_axes=(None, "tensor"),
                        use_bias=False,
                        params_dtype=jnp.bfloat16,
                        mesh=self.mesh,
                    )
                new_linear.weight = nnx.Param(w_sharded)
                setattr(attn, name, new_linear)

            if (layer_idx + 1) % 10 == 0 or layer_idx == 0:
                logger.info("Fused QKV dequant: layer %d/%d done", layer_idx + 1, self.config.num_hidden_layers)

        logger.info("All fused QKV weights dequantized and assigned.")

    def _dequantize_layer0_mlp(self):
        """Dequantize layer 0 dense MLP (same as Flash)."""
        layer = self.model.layers[0]
        if not layer.is_layer_sparse:
            for proj_name in ("gate_proj", "up_proj", "down_proj"):
                proj = getattr(layer.mlp, proj_name)
                if isinstance(proj, QuantizedLinear):
                    setattr(layer.mlp, proj_name, self._dequantize_quantized_linear(proj))
                    logger.info("Dequantized layer 0 MLP %s -> bf16", proj_name)

    def _dequantize_fp8_to_bf16(self):
        """Override Flash's dequant: Pro handles QKV separately in _load_and_dequant_fused_qkv.

        This method only dequants non-QKV FP8 projections (none for Pro currently).
        """
        # QKV dequant is handled by _load_and_dequant_fused_qkv
        # Layer 0 MLP dequant is handled by _dequantize_layer0_mlp
        # Just do kv_head_replication
        pass


EntryClass = [MiMoV2ProForCausalLM]
