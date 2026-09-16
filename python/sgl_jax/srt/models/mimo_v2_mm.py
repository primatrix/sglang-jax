"""Top-level MiMoV2 multimodal model composition."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import Mesh

from sgl_jax.srt.models.mimo_v2_audio import MiMoAudioEncoder
from sgl_jax.srt.models.mimo_v2_pro import MiMoV2ForCausalLM
from sgl_jax.srt.models.mimo_v2_vision import MiMoVisionTransformer
from sgl_jax.srt.multimodal.common.modality_enum import Modality, MultimodalDataItem
from sgl_jax.srt.multimodal.in_model.interface import InModelMultimodalContract
from sgl_jax.srt.multimodal.in_model.lane_packing import (
    encoder_num_lanes,
    pack_lanes,
    restore_encoder_output,
    run_mrope_vision_model,
)
from sgl_jax.srt.multimodal.layers.vision_sharding import resolve_encoder_tp
from sgl_jax.srt.utils.common_utils import resolve_vision_patch_buckets
from sgl_jax.srt.utils.weight_utils import WeightLoader, WeightMapping

if TYPE_CHECKING:
    from sgl_jax.srt.configs.model_config import ModelConfig

logger = logging.getLogger(__name__)


def _value(config: Mapping[str, Any] | object | None, name: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


# ---------------------------------------------------------------------------
# Top-level VLM (backbone + vision tower + in-model contract)
# ---------------------------------------------------------------------------


class MiMoV2ForConditionalGeneration(InModelMultimodalContract, MiMoV2ForCausalLM):
    """Wire the MiMoV2 vision + audio towers onto the in-model multimodal contract."""

    def __init__(self, config, mesh: Mesh | None = None, dtype: jnp.dtype = jnp.bfloat16):
        if mesh is None:
            raise ValueError("MiMoV2 multimodal models require a device mesh.")
        super().__init__(config, mesh, dtype)

        vision_config = _value(config, "vision_config", None)
        audio_config = _value(config, "audio_config", None)
        if vision_config is None and audio_config is None:
            raise ValueError("MiMoV2 VLM requires config.vision_config or config.audio_config.")

        self.encoder_tp = resolve_encoder_tp(mesh, getattr(config, "vision_encoder_parallel", "dp"))
        self.vision_tp = self.encoder_tp
        if vision_config is None:
            self.visual = None
        else:
            input_buckets = tuple(
                resolve_vision_patch_buckets(
                    getattr(config, "precompile_vision_patch_paddings", None)
                )
            )
            self.visual = MiMoVisionTransformer(
                vision_config,
                self.dtype,
                nnx.Rngs(0),
                mesh,
                self.encoder_tp,
                input_buckets,
            )
        self.audio_encoder = (
            MiMoAudioEncoder(audio_config, self.dtype, mesh, self.encoder_tp)
            if audio_config is not None
            else None
        )

    # -- contract ---------------------------------------------------------

    def get_input_embeddings(self) -> Callable[[jax.Array], jax.Array]:
        return self.model.embed_tokens

    def precompile_multimodal(self) -> None:
        if self.visual is not None:
            self.visual.precompile()
        if self.audio_encoder is not None:
            self.audio_encoder.precompile()

    def get_multimodal_embedding_packed_capacities(self) -> tuple[int, ...]:
        capacities: list[int] = []
        if self.visual is not None:
            rows = encoder_num_lanes(self.mesh, self.visual.vision_tp)
            unit = self.visual.spatial_merge_unit
            capacities.extend(rows * bucket // unit for bucket in self.visual.input_buckets)
        if self.audio_encoder is not None:
            rows = encoder_num_lanes(self.mesh, self.audio_encoder.encoder_tp)
            group = self.audio_encoder.group_size
            capacities.extend(rows * bucket // group for bucket in self.audio_encoder.input_buckets)
        return tuple(sorted(set(capacities)))

    def get_image_feature(self, items: list[MultimodalDataItem]) -> jax.Array:
        visual = self.visual
        if visual is None:
            raise ValueError("This MiMoV2 checkpoint has no vision encoder.")
        num_lanes = encoder_num_lanes(self.mesh, visual.vision_tp)
        return run_mrope_vision_model(
            visual,
            items,
            mesh=self.mesh,
            num_lanes=num_lanes,
            buckets=visual.input_buckets,
            merge_unit=visual.spatial_merge_unit,
            rope_type="rope_3d",
            input_sharding=visual.specs.sharding(visual.specs.batch_axis),
            output_sharding=visual.specs.sharding(),
        )

    def get_video_feature(self, items: list[MultimodalDataItem]) -> jax.Array:
        return self.get_image_feature(items)

    def get_audio_feature(self, items: list[MultimodalDataItem]) -> jax.Array:
        encoder = self.audio_encoder
        if encoder is None:
            raise ValueError("This MiMoV2 checkpoint has no audio encoder.")
        num_lanes = encoder_num_lanes(self.mesh, encoder.encoder_tp)
        batch_sharding = encoder.specs.sharding(encoder.specs.batch_axis)
        codes, output_indices, lanes = pack_lanes(
            items,
            num_lanes,
            buckets=encoder.input_buckets,
            merge_unit=encoder.group_size,
            input_sharding=batch_sharding,
            dtype=np.int32,
        )
        valid = np.asarray(
            [
                sum(int(np.asarray(items[item_index].feature).shape[0]) for item_index in lane)
                for lane in lanes
            ],
            dtype=np.int32,
        )
        channels = items[0].feature.shape[-1]
        with jax.set_mesh(self.mesh):
            output = encoder.encode(
                codes.reshape(num_lanes, -1, channels, out_sharding=batch_sharding),
                jax.device_put(valid, batch_sharding),
            )
            output = output.reshape(-1, output.shape[-1])
            return restore_encoder_output(output, output_indices, encoder.specs.sharding())

    def get_multimodal_encode_funcs(self):
        funcs = {}
        if self.visual is not None:
            funcs[Modality.IMAGE] = self.get_image_feature
            funcs[Modality.MULTI_IMAGES] = self.get_image_feature
            funcs[Modality.VIDEO] = self.get_video_feature
        if self.audio_encoder is not None:
            funcs[Modality.AUDIO] = self.get_audio_feature
        return funcs

    # -- weights ----------------------------------------------------------

    def load_weights(self, model_config: ModelConfig) -> None:
        """Load the text backbone, then load each modality tower.

        The text loader replaces graph state, so towers must be detached during
        that step and restored even if loading fails.
        """
        visual = self.visual
        audio_encoder = self.audio_encoder
        if visual is not None:
            del self.visual
        if audio_encoder is not None:
            del self.audio_encoder
        try:
            super().load_weights(model_config)
        finally:
            if visual is not None:
                self.visual = visual
            if audio_encoder is not None:
                self.audio_encoder = audio_encoder

        vision = _value(self.config, "vision_config", None)
        heads = int(_value(vision, "num_heads", 1))
        kv_heads = int(_value(vision, "num_key_value_heads", heads) or heads)
        head_dim = int(_value(vision, "qk_channels", 64))
        tower_config = SimpleNamespace(
            model_path=model_config.model_path,
            quantization_config=None,
            hf_config=self.config,
            hf_text_config=SimpleNamespace(head_dim=head_dim, v_head_dim=head_dim),
            num_attention_heads=heads,
            hidden_size=heads * head_dim,
            get_total_num_kv_heads=lambda: kv_heads,
            _dummy_mode=getattr(model_config, "_dummy_mode", False),
        )
        if self.visual is not None:
            loader = WeightLoader(self.visual, tower_config, self.mesh, self.dtype)
            with self.mesh:
                loader.load_weights_from_safetensors(self._vision_weight_mappings())
            logger.info("MiMoV2 vision tower weights loaded.")
        if self.audio_encoder is not None:
            audio_config = SimpleNamespace(
                model_path=model_config.model_path,
                quantization_config=None,
                _dummy_mode=getattr(model_config, "_dummy_mode", False),
            )
            loader = WeightLoader(self.audio_encoder, audio_config, self.mesh, self.dtype)
            with self.mesh:
                loader.load_weights_from_safetensors(self._audio_weight_mappings())
            logger.info("MiMoV2 audio tower weights loaded.")

    @staticmethod
    def _linear_mappings(source, target, sharding) -> dict[str, WeightMapping]:
        return {
            f"{source}.weight": WeightMapping(
                target_path=f"{target}.weight", sharding=sharding, transpose=True
            ),
            f"{source}.bias": WeightMapping(
                target_path=f"{target}.bias", sharding=(sharding[-1],), transpose=False
            ),
        }

    def _vision_weight_mappings(self) -> dict[str, WeightMapping]:
        specs = self.visual.specs
        col, row = specs.col_kernel_axes, specs.row_kernel_axes
        mappings: dict[str, WeightMapping] = {
            "visual.patch_embed.proj.weight": WeightMapping(
                target_path="patch_embed.proj.kernel",
                sharding=(None, None, None, None, None),
                transpose_axes=(2, 3, 4, 1, 0),
            ),
            "visual.merger.ln_q.weight": WeightMapping(
                target_path="merger.ln_q.scale", sharding=(None,), transpose=False
            ),
            "visual.merger.mlp.0.weight": WeightMapping(
                target_path="merger.mlp_fc1.weight", sharding=col, transpose=True
            ),
            "visual.merger.mlp.2.weight": WeightMapping(
                target_path="merger.mlp_fc2.weight", sharding=row, transpose=True
            ),
        }
        for index, block in enumerate(self.visual.blocks):
            src = f"visual.blocks.{index}"
            tgt = f"blocks.{index}"
            for norm in ("norm1", "norm2"):
                mappings[f"{src}.{norm}.weight"] = WeightMapping(
                    target_path=f"{tgt}.{norm}.scale", sharding=(None,), transpose=False
                )
            mappings[f"{src}.attn.qkv.weight"] = WeightMapping(
                target_path=[f"{tgt}.attn.{n}_proj.weight" for n in ("q", "k", "v")],
                sharding=col,
                transpose=True,
            )
            mappings[f"{src}.attn.qkv.bias"] = WeightMapping(
                target_path=[f"{tgt}.attn.{n}_proj.bias" for n in ("q", "k", "v")],
                sharding=(col[-1],),
                transpose=False,
            )
            mappings.update(self._linear_mappings(f"{src}.attn.proj", f"{tgt}.attn.proj", row))
            for name in ("gate_proj", "up_proj"):
                mappings.update(
                    self._linear_mappings(f"{src}.mlp.{name}", f"{tgt}.mlp.{name}", col)
                )
            mappings.update(
                self._linear_mappings(f"{src}.mlp.down_proj", f"{tgt}.mlp.down_proj", row)
            )
            if block.attn.sinks is not None:
                mappings[f"{src}.attn.sinks"] = WeightMapping(
                    target_path=f"{tgt}.attn.sinks",
                    sharding=(specs.tensor_axis,),
                    transpose=False,
                )
        return mappings

    def _audio_weight_mappings(self) -> dict[str, WeightMapping]:
        encoder = self.audio_encoder
        mappings: dict[str, WeightMapping] = {}
        # Per-channel speech code embeddings live at top level in the checkpoint.
        for index in range(encoder.channels):
            mappings[f"speech_embeddings.{index}.weight"] = WeightMapping(
                target_path=f"speech_embeddings.{index}.embedding",
                sharding=(None, None),
                transpose=False,
            )
        src_root = "audio_encoder.input_local_transformer"
        tgt_root = "transformer"
        if encoder.transformer.norm is not None:
            mappings[f"{src_root}.norm.weight"] = WeightMapping(
                target_path=f"{tgt_root}.norm.scale", sharding=(None,), transpose=False
            )
        for index in range(len(encoder.transformer.layers)):
            src = f"{src_root}.layers.{index}"
            tgt = f"{tgt_root}.layers.{index}"
            for norm in ("input_layernorm", "post_attention_layernorm"):
                mappings[f"{src}.{norm}.weight"] = WeightMapping(
                    target_path=f"{tgt}.{norm}.scale", sharding=(None,), transpose=False
                )
            for name in ("q_proj", "k_proj", "v_proj"):
                mappings.update(
                    self._linear_mappings(
                        f"{src}.self_attn.{name}", f"{tgt}.self_attn.{name}", (None, None)
                    )
                )
            mappings[f"{src}.self_attn.o_proj.weight"] = WeightMapping(
                target_path=f"{tgt}.self_attn.o_proj.weight", sharding=(None, None), transpose=True
            )
            for name in ("gate_proj", "up_proj", "down_proj"):
                mappings[f"{src}.mlp.{name}.weight"] = WeightMapping(
                    target_path=f"{tgt}.mlp.{name}.weight", sharding=(None, None), transpose=True
                )
        if encoder.proj_fc2 is None:
            mappings["audio_encoder.projection.weight"] = WeightMapping(
                target_path="proj_fc1.weight", sharding=(None, None), transpose=True
            )
        else:
            mappings["audio_encoder.projection.mlp.0.weight"] = WeightMapping(
                target_path="proj_fc1.weight", sharding=(None, None), transpose=True
            )
            mappings["audio_encoder.projection.mlp.2.weight"] = WeightMapping(
                target_path="proj_fc2.weight", sharding=(None, None), transpose=True
            )
        return mappings


EntryClass = MiMoV2ForConditionalGeneration
