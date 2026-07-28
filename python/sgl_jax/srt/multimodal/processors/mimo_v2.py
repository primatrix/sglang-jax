from __future__ import annotations

import asyncio
import base64
import copy
import io
import json
import os
from types import SimpleNamespace
from urllib.parse import unquote, urlparse

import numpy as np
import requests

from sgl_jax.srt.multimodal.common.modality_enum import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sgl_jax.srt.multimodal.processors.qwen_vl import QwenVLProcessor


def _value(config, name, default=None):
    if config is None:
        return default
    return config.get(name, default) if isinstance(config, dict) else getattr(config, name, default)


def _config_value(config, name, default=None):
    value = getattr(config, name, None)
    return (
        value
        if value is not None
        else _value(getattr(config, "processor_config", None), name, default)
    )


class _MiMoAudioCodec:
    def __init__(self, model_path):
        import torch
        from transformers import AutoModel

        path = os.path.join(model_path, "audio_tokenizer")
        try:
            self.model = AutoModel.from_pretrained(path, trust_remote_code=True)
        except (KeyError, ValueError):
            from transformers.dynamic_module_utils import get_class_from_dynamic_module

            config_type = get_class_from_dynamic_module(
                "modeling_mimo_v2.MiMoAudioTokenizerConfig",
                model_path,
                trust_remote_code=True,
            )
            model_type = get_class_from_dynamic_module(
                "modeling_mimo_v2.MiMoAudioTokenizer",
                model_path,
                trust_remote_code=True,
            )
            with open(os.path.join(path, "config.json")) as config_file:
                config = config_type(**json.load(config_file))
            self.model = model_type.from_pretrained(path, config=config)
        self.model.eval()
        self.torch = torch
        from sgl_jax.srt.multimodal.manager.multimodal_tokenizer import (
            MiMoAudioProcessor,
        )

        self.processor = MiMoAudioProcessor()

    @staticmethod
    def _waveform(source):
        import soundfile as sf

        if isinstance(source, dict):
            source = source.get("url", source.get("audio_url"))
        if isinstance(source, tuple) and len(source) == 2:
            waveform, sampling_rate = source
            return np.asarray(waveform, dtype=np.float32), int(sampling_rate)
        if isinstance(source, np.ndarray):
            return source.astype(np.float32), 24000
        if isinstance(source, bytes):
            return sf.read(io.BytesIO(source), dtype="float32")
        if isinstance(source, os.PathLike):
            source = os.fspath(source)
        if not isinstance(source, str):
            raise ValueError(f"Unsupported MiMoV2 audio source: {type(source).__name__}.")
        if source.startswith(("http://", "https://")):
            response = requests.get(source, timeout=30)
            response.raise_for_status()
            return sf.read(io.BytesIO(response.content), dtype="float32")
        if source.startswith("data:") and "base64," in source:
            payload = base64.b64decode(source.split("base64,", 1)[1])
            return sf.read(io.BytesIO(payload), dtype="float32")
        if source.startswith("file://"):
            source = unquote(urlparse(source).path)
        if os.path.isfile(source):
            return sf.read(source, dtype="float32")
        try:
            payload = base64.b64decode(source, validate=True)
        except ValueError as error:
            raise ValueError("Unsupported MiMoV2 audio source.") from error
        return sf.read(io.BytesIO(payload), dtype="float32")

    def encode(self, source):
        waveform, sampling_rate = self._waveform(source)
        if waveform.ndim == 2:
            axis = 0 if waveform.shape[0] <= 8 < waveform.shape[1] else 1
            waveform = waveform.mean(axis=axis)
        mels, _ = self.processor(waveform, sampling_rate)
        encoder = getattr(self.model, "encoder", self.model)
        parameter = next(encoder.parameters())
        parts = []
        with self.torch.no_grad():
            for start in range(0, mels.shape[1], 6000):
                features = self.torch.from_numpy(mels[:, start : start + 6000]).to(
                    device=parameter.device,
                    dtype=parameter.dtype,
                )
                lengths = self.torch.tensor(
                    [features.shape[1]],
                    dtype=self.torch.long,
                    device=parameter.device,
                )
                codes, _ = encoder.encode(
                    input_features=features,
                    input_lens=lengths,
                    return_codes_only=True,
                )
                parts.append(codes)
        return self.torch.cat(parts, dim=-1).transpose(0, 1).cpu().numpy()


class MiMoV2Processor(QwenVLProcessor):
    uses_mrope = False
    models = (
        "MiMoV2ForConditionalGeneration",
        "MiMoV2FlashForConditionalGeneration",
    )

    def __init__(self, hf_config, server_args, processor):
        if isinstance(getattr(hf_config, "vision_config", None), dict):
            hf_config = copy.copy(hf_config)
            hf_config.vision_config = SimpleNamespace(**hf_config.vision_config)
        super().__init__(hf_config, server_args, processor)
        audio_token_id = _config_value(hf_config, "audio_token_id", 151669)
        if audio_token_id is None:
            raise ValueError("MiMoV2 audio_token_id is missing from the model config.")
        self.audio_token_id = int(audio_token_id)
        audio_config = getattr(hf_config, "audio_config", None)
        self.audio_channels = int(_value(audio_config, "audio_channels", 20))
        self.group_size = int(_value(audio_config, "group_size", 4))
        if self.audio_channels <= 0 or self.group_size <= 0:
            raise ValueError("MiMoV2 audio_channels and group_size must be positive.")
        self.vocab_sizes = self._int_list(
            _value(audio_config, "speech_vocab_size", 1280),
            self.audio_channels,
        )
        self._audio_codec = None

    async def process_mm_data_async(self, image_data, input_text, request_obj, **kwargs):
        if isinstance(input_text, list):
            raise ValueError("MiMoV2 multimodal requests require text input.")
        has_vision = getattr(self.hf_config, "vision_config", None) is not None
        has_audio = getattr(self.hf_config, "audio_config", None) is not None
        if not has_vision and (
            self.normalize_data(image_data)
            or self.normalize_data(getattr(request_obj, "video_data", None))
        ):
            raise ValueError("This MiMoV2 checkpoint has no vision encoder.")
        if not has_audio and self._audio_sources(getattr(request_obj, "audio_data", None)):
            raise ValueError("This MiMoV2 checkpoint has no audio encoder.")
        if has_vision:
            output = await super().process_mm_data_async(
                image_data, input_text, request_obj, **kwargs
            )
        else:
            processed = self.processor(
                text=[input_text],
                padding=True,
                return_tensors="pt",
            )
            input_ids = self._to_numpy(processed.get("input_ids"))
            if input_ids is None:
                raise ValueError("HF processor did not return input_ids.")
            output = MultimodalInputs(mm_items=[], input_ids=input_ids.reshape(-1).tolist())

        sources = self._audio_sources(getattr(request_obj, "audio_data", None))
        if not sources:
            return output
        codes = await asyncio.to_thread(lambda: [self._encode_audio(source) for source in sources])
        self._merge_audio(output, codes)
        return output

    def _encode_audio(self, source):
        if isinstance(source, dict) and "codes" in source:
            source = source["codes"]
        array = np.asarray(source) if isinstance(source, (list, np.ndarray)) else None
        if array is not None and array.ndim == 2 and np.issubdtype(array.dtype, np.integer):
            return self._normalize_codes(array)
        if array is not None:
            source = array
        if self._audio_codec is None:
            self._audio_codec = _MiMoAudioCodec(self.server_args.model_path)
        return self._normalize_codes(self._audio_codec.encode(source))

    def _normalize_codes(self, values):
        values = np.asarray(values)
        if values.ndim != 2:
            raise ValueError(f"MiMoV2 audio codes must be 2D, got {values.shape}.")
        if values.shape[1] != self.audio_channels:
            if values.shape[0] == self.audio_channels:
                values = values.T
            else:
                raise ValueError(
                    f"MiMoV2 audio codes require {self.audio_channels} channels, "
                    f"got {values.shape}."
                )
        if not np.issubdtype(values.dtype, np.integer) or np.any(values < 0):
            raise ValueError("MiMoV2 audio codes must be non-negative integers.")
        for channel, size in enumerate(self.vocab_sizes):
            if np.any(values[:, channel] >= size):
                raise ValueError(
                    f"MiMoV2 audio code on channel {channel} exceeds vocab size {size}."
                )
        return values.astype(np.int32, copy=False)

    @staticmethod
    def _audio_sources(data):
        if data is None:
            return []
        if isinstance(data, list) and data and isinstance(data[0], (int, float, np.number)):
            return [data]
        try:
            array = np.asarray(data)
        except ValueError:
            array = None
        if (
            isinstance(data, list)
            and array is not None
            and array.ndim == 2
            and np.issubdtype(array.dtype, np.integer)
        ):
            return [data]
        return data if isinstance(data, list) else [data]

    @staticmethod
    def _int_list(value, length):
        values = value.split("-") if isinstance(value, str) else value
        values = list(values) if isinstance(values, (list, tuple)) else [values]
        values = [int(item) for item in values]
        if len(values) == 1:
            values *= length
        if len(values) != length:
            raise ValueError(f"Expected {length} values, got {len(values)}.")
        return values

    def _merge_audio(self, output, code_arrays):
        input_ids = list(output.input_ids)
        items = []
        cursor = 0
        for values in code_arrays:
            values = np.asarray(values)
            if values.ndim != 2 or not values.shape[0]:
                raise ValueError(
                    f"MiMoV2 audio codes must be non-empty [T, C], got {values.shape}."
                )
            pad = (-values.shape[0]) % self.group_size
            if pad:
                values = np.concatenate((values, np.repeat(values[-1:], pad, axis=0)))
            tokens = values.shape[0] // self.group_size
            try:
                start = input_ids.index(self.audio_token_id, cursor)
            except ValueError as error:
                raise ValueError("MiMoV2 prompt is missing an audio placeholder.") from error
            end = start + 1
            while end < len(input_ids) and input_ids[end] == self.audio_token_id:
                end += 1
            if end - start not in (1, tokens):
                raise ValueError(
                    f"MiMoV2 audio placeholder span has {end - start} tokens, expected 1 or {tokens}."
                )
            input_ids[start:end] = [self.audio_token_id] * tokens
            item = MultimodalDataItem(
                modality=Modality.AUDIO,
                feature=values,
                placeholder_ranges=[(start, start + tokens)],
            )
            item.set_pad_value()
            items.append(item)
            cursor = start + tokens

        if self.audio_token_id in input_ids[cursor:]:
            raise ValueError("MiMoV2 prompt has more audio placeholders than audio inputs.")
        output.input_ids = input_ids
        output.audio_token_id = self.audio_token_id
        output.mm_items.extend(items)
        self._refresh_vision_ranges(output)

    def _refresh_vision_ranges(self, output):
        for modality, token_id, grid_key in (
            (Modality.IMAGE, output.im_token_id, "image_grid_thw"),
            (Modality.VIDEO, output.video_token_id, "video_grid_thw"),
        ):
            items = [item for item in output.mm_items if item.modality is modality]
            if not items:
                continue
            grids = [tuple(np.asarray(item.get(grid_key)).reshape(-1)) for item in items]
            ranges = self._compute_placeholder_ranges(
                output.input_ids,
                grids,
                token_id,
                self.hf_config.vision_config.spatial_merge_size,
                modality.name,
            )
            for item, placeholder_range in zip(items, ranges):
                item.placeholder_ranges = [placeholder_range]
