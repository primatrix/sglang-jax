"""MiMoV2 image and video processor with optional audio composition."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any

from transformers import PretrainedConfig
from transformers.processing_utils import ProcessorMixin

from sgl_jax.srt.managers.io_struct import EmbeddingReqInput, GenerateReqInput
from sgl_jax.srt.multimodal.common.modality_enum import Modality, MultimodalInputs
from sgl_jax.srt.multimodal.processors.mimo_v2_audio import MiMoV2AudioProcessorMixin
from sgl_jax.srt.multimodal.processors.qwen_vl import QwenVLProcessor, preprocess_video
from sgl_jax.srt.server_args import ServerArgs
from sgl_jax.srt.utils.common_utils import resolve_vision_patch_buckets


def _value(config: Mapping[str, Any] | object | None, name: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def _config_value(config: object, name: str, default: Any = None) -> Any:
    value = getattr(config, name, None)
    return (
        value
        if value is not None
        else _value(getattr(config, "processor_config", None), name, default)
    )


class MiMoV2Processor(MiMoV2AudioProcessorMixin, QwenVLProcessor):
    auto_mm_processor_worker_num = 1
    supports_mm_processor_concurrency = False
    models = ("MiMoV2ForCausalLM", "MiMoV2ForConditionalGeneration")

    def __init__(
        self,
        hf_config: PretrainedConfig,
        server_args: ServerArgs,
        processor: ProcessorMixin,
    ) -> None:
        # Base helpers reach into ``hf_config.vision_config`` with attribute
        # access; normalize a dict sub-config so those keep working.
        if isinstance(getattr(hf_config, "vision_config", None), dict):
            hf_config = copy.copy(hf_config)
            hf_config.vision_config = SimpleNamespace(**hf_config.vision_config)
        super().__init__(hf_config, server_args, processor)

        self._init_audio_processor(hf_config)

    async def process_mm_data_async(
        self,
        image_data,
        input_text,
        request_obj: GenerateReqInput | EmbeddingReqInput,
        **kwargs,
    ) -> MultimodalInputs:
        if isinstance(input_text, list):
            raise ValueError("MiMoV2 multimodal requests require text input, not input_ids.")

        has_vision = getattr(self.hf_config, "vision_config", None) is not None
        sources = self._audio_sources(getattr(request_obj, "audio_data", None))
        if sources and not self._has_audio:
            raise ValueError("This MiMoV2 checkpoint has no audio encoder.")

        image_sources = self.normalize_data(image_data)
        video_data = self.normalize_data(getattr(request_obj, "video_data", None))
        if not has_vision and (image_sources or video_data):
            raise ValueError("This MiMoV2 checkpoint has no vision encoder.")
        return await self.mm_processor_executor.run(
            self._process_mm_data,
            input_text,
            image_sources,
            video_data,
            self._build_video_config(request_obj),
            sources,
        )

    def _process_mm_data(
        self, input_text, image_sources, video_data, video_config, audio_sources, *, processor
    ) -> MultimodalInputs:
        if getattr(self.hf_config, "vision_config", None) is not None:
            output = self._process_vision(
                input_text, image_sources, video_data, video_config, processor=processor
            )
        else:
            output = self.process_and_combine_mm_data(input_text, processor=processor)
        if audio_sources:
            self._merge_audio(output, [self._encode_audio(source) for source in audio_sources])
        return output

    # -- vision -----------------------------------------------------------

    def _process_vision(
        self, input_text, image_sources, video_data, video_config, *, processor
    ) -> MultimodalInputs:
        images = [self.load_image(source) for source in image_sources]
        videos = [
            preprocess_video(self.unwrap_source(source), video_config) for source in video_data
        ]
        processor_kwargs = {}
        if videos:
            processor_kwargs["videos_kwargs"] = {
                "do_sample_frames": False,
                "fps": video_config.get(
                    "fps",
                    float(_value(getattr(self.hf_config, "processor_config", None), "fps", 1.0)),
                ),
            }
        if images:
            buckets = resolve_vision_patch_buckets(
                getattr(self.server_args, "precompile_vision_patch_paddings", None)
            )
            patch_budget = max(buckets)
            vision_config = self.hf_config.vision_config
            merge_unit = int(_value(vision_config, "spatial_merge_size", 2)) ** 2
            per_image_patches = patch_budget // len(images) // merge_unit * merge_unit
            if per_image_patches <= 0:
                raise ValueError(
                    f"MiMoV2 received {len(images)} images, but the largest compiled vision "
                    f"bucket ({patch_budget}) cannot fit one merge unit per image."
                )
            patch_size = int(_value(vision_config, "patch_size", 16))
            compiled_max_pixels = per_image_patches * patch_size**2
            configured_max_pixels = int(
                _config_value(self.hf_config, "image_max_pixels", compiled_max_pixels)
            )
            processor_kwargs["images_kwargs"] = {
                "max_pixels": min(compiled_max_pixels, configured_max_pixels)
            }

        return self.process_and_combine_mm_data(
            input_text,
            images=images,
            videos=videos,
            processor=processor,
            **processor_kwargs,
        )

    def collect_mm_items_from_processor_output(
        self,
        processor_output,
        images: list | None = None,
        videos: list | None = None,
        **kwargs,
    ) -> MultimodalInputs:
        del images, videos, kwargs
        input_ids_array = self._to_numpy(processor_output.get("input_ids"))
        if input_ids_array is None:
            raise ValueError("MiMoV2 HF processor did not return input_ids.")
        input_ids = input_ids_array.reshape(-1).tolist()
        spatial_merge_size = int(_value(self.hf_config.vision_config, "spatial_merge_size", 2))
        image_token_id = self.hf_config.image_token_id
        video_token_id = getattr(self.hf_config, "video_token_id", None)
        mm_items = []
        for modality, token_id, pixel_key, grid_key in (
            (Modality.IMAGE, image_token_id, "pixel_values", "image_grid_thw"),
            (Modality.VIDEO, video_token_id, "pixel_values_videos", "video_grid_thw"),
        ):
            pixels = self._to_numpy(processor_output.get(pixel_key))
            grids = self._to_grid_list(processor_output.get(grid_key))
            ranges = self._compute_placeholder_ranges(
                input_ids, grids, token_id, spatial_merge_size, modality.name
            )
            mm_items.extend(
                self._build_items(pixels, grids, ranges, modality, grid_key)
            )
        for item in mm_items:
            item.set_pad_value()

        # No mrope_positions/mrope_position_delta: MiMoV2 uses standard 1-D RoPE.
        return MultimodalInputs(
            mm_items=mm_items,
            input_ids=input_ids,
            im_start_id=getattr(self.hf_config, "vision_start_token_id", None),
            im_end_id=getattr(self.hf_config, "vision_end_token_id", None),
            im_token_id=image_token_id,
            video_token_id=video_token_id,
        )

    def _build_video_config(self, request_obj) -> dict[str, int | float]:
        vision = self.hf_config.vision_config
        factor = int(_value(vision, "patch_size", 16)) * int(
            _value(vision, "spatial_merge_size", 2)
        )
        processor = getattr(self.hf_config, "processor_config", None)
        video_config: dict[str, int | float] = {
            "factor": factor,
            "fps": float(_value(processor, "fps", 1.0)),
            "min_pixels": int(_value(processor, "video_min_pixels", 8192)),
            "max_pixels": int(_value(processor, "video_max_pixels", 8388608)),
            "total_pixels": int(_value(processor, "video_total_max_pixels", 268435456)),
            "max_frames": int(_value(processor, "max_frames", 3600)),
        }
        min_frames = _value(processor, "min_frames", None)
        if min_frames is not None:
            video_config["min_frames"] = int(min_frames)

        requested_frames = getattr(request_obj, "num_frames", None)
        requested_fps = getattr(request_obj, "fps", None)
        if requested_frames is not None:
            video_config.pop("fps")
            video_config["nframes"] = int(requested_frames)
        elif requested_fps is not None:
            video_config["fps"] = float(requested_fps)
        return video_config
