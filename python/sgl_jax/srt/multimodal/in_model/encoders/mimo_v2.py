from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np
from jax.tree_util import register_pytree_node_class

from sgl_jax.srt.multimodal.common.modality_enum import Modality, MultimodalDataItem
from sgl_jax.srt.multimodal.in_model.encoder_planning import EncoderPlanBuilder


def _value(config, name, default=None):
    return config.get(name, default) if isinstance(config, dict) else getattr(config, name, default)


def _grid(item: MultimodalDataItem) -> tuple[int, int, int]:
    value = item.get("image_grid_thw")
    if value is None:
        value = item.get("video_grid_thw")
    rows = np.asarray(value)
    if rows.size != 3:
        raise ValueError(f"MiMoV2 vision item requires one (t, h, w) grid, got {rows.shape}.")
    return tuple(int(x) for x in rows.reshape(-1))


def _placeholder_rows(item: MultimodalDataItem) -> int:
    return sum(end - start for start, end in item.placeholder_ranges or ())


def _int_list(value, length):
    values = value.split("-") if isinstance(value, str) else value
    values = list(values) if isinstance(values, (list, tuple)) else [values]
    values = [int(item) for item in values]
    if len(values) == 1:
        values *= length
    if len(values) != length:
        raise ValueError(f"Expected {length} values, got {len(values)}.")
    return values


@register_pytree_node_class
@dataclass
class MiMoV2VisionMetadata:
    col_index: Any
    rotary_freqs: Any
    segment_ids: Any

    def tree_flatten(self):
        return (self.col_index, self.rotary_freqs, self.segment_ids), None

    @classmethod
    def tree_unflatten(cls, _, children):
        return cls(*children)


class MiMoV2VisionPlanBuilder(EncoderPlanBuilder):
    input_modalities = (Modality.IMAGE, Modality.MULTI_IMAGES, Modality.VIDEO)
    output_modality = Modality.IMAGE

    def __init__(self, model_config, input_buckets=None):
        super().__init__(input_buckets)
        config = getattr(model_config.hf_config, "vision_config", None)
        if config is None:
            raise ValueError("MiMoV2 vision planning requires vision_config.")
        self.config = config
        self.patch_size = int(_value(config, "patch_size", 16))
        self.temporal_patch_size = int(_value(config, "temporal_patch_size", 2))
        self.in_channels = int(_value(config, "in_channels", None) or _value(config, "in_chans", 3))
        self.spatial_merge_size = int(_value(config, "spatial_merge_size", 2))
        self.spatial_merge_unit = self.spatial_merge_size**2
        self.head_dim = int(_value(config, "qk_channels", 64))
        if self.head_dim % 4:
            raise ValueError("MiMoV2 vision head dimension must be divisible by four.")
        self.theta = float(_value(config, "rope_theta", 10000.0))
        self.feature_dim = (
            self.in_channels * self.temporal_patch_size * self.patch_size * self.patch_size
        )

    def _metadata_for_grid(self, grid):
        t, h, w = grid
        merge = self.spatial_merge_size
        if min(grid) <= 0 or h % merge or w % merge:
            raise ValueError(
                f"MiMoV2 vision grid {grid} must be positive and spatially divisible by {merge}."
            )

        h_pos, w_pos = np.indices((h, w))
        shape = (h // merge, merge, w // merge, merge)
        h_pos = h_pos.reshape(shape).transpose(0, 2, 1, 3).reshape(-1)
        w_pos = w_pos.reshape(shape).transpose(0, 2, 1, 3).reshape(-1)
        pos = np.tile(np.stack((h_pos, w_pos), axis=-1), (t, 1))
        inv = 1.0 / (
            self.theta
            ** (np.arange(0, self.head_dim // 2, 2, dtype=np.float32) / (self.head_dim // 2))
        )
        table = np.outer(np.arange(max(h, w), dtype=np.float32), inv)
        freqs = table[pos].reshape(pos.shape[0], -1)
        freqs = np.concatenate((freqs, freqs), axis=-1).astype(np.float32)

        units = np.arange(t * (h // merge) * (w // merge), dtype=np.int32)
        col_index = units.reshape(t, h // merge, w // merge).transpose(0, 2, 1).reshape(-1)
        segments = np.repeat(np.arange(t, dtype=np.int32), h * w)
        return col_index, freqs, segments

    def get_metadata(self, items):
        col_indices = []
        freqs = []
        segments = []
        unit_offset = 0
        segment_offset = 0
        for item in items:
            grid = _grid(item)
            feature = np.asarray(item.feature)
            expected_inputs = int(np.prod(grid))
            expected_outputs = expected_inputs // self.spatial_merge_unit
            if feature.shape != (expected_inputs, self.feature_dim):
                raise ValueError(
                    f"MiMoV2 vision feature shape {feature.shape} does not match "
                    f"grid {grid} and patch width {self.feature_dim}."
                )
            if _placeholder_rows(item) != expected_outputs:
                raise ValueError(
                    "MiMoV2 vision placeholder rows do not match the spatially merged grid."
                )
            index, item_freqs, item_segments = self._metadata_for_grid(grid)
            col_indices.append(index + unit_offset)
            freqs.append(item_freqs)
            segments.append(item_segments + segment_offset)
            unit_offset += index.size
            segment_offset += grid[0]
        return MiMoV2VisionMetadata(
            np.concatenate(col_indices),
            np.concatenate(freqs),
            np.concatenate(segments),
        )

    def dummy_metadata(self, input_capacity):
        self._validate_capacity(input_capacity)
        merge = self.spatial_merge_size
        index, freqs, segments = self._metadata_for_grid((1, merge, input_capacity // merge))
        return MiMoV2VisionMetadata(index, freqs, segments)

    def pad_metadata(self, meta, input_capacity):
        self._validate_capacity(input_capacity)
        units = input_capacity // self.spatial_merge_unit
        if meta.col_index.size > units or meta.rotary_freqs.shape[0] > input_capacity:
            raise ValueError("MiMoV2 vision metadata exceeds the selected input bucket.")
        col_index = np.arange(units, dtype=np.int32)
        col_index[: meta.col_index.size] = meta.col_index
        freqs = np.zeros((input_capacity, self.head_dim), dtype=np.float32)
        freqs[: meta.rotary_freqs.shape[0]] = meta.rotary_freqs
        segments = np.full(input_capacity, -1, dtype=np.int32)
        segments[: meta.segment_ids.size] = meta.segment_ids
        return MiMoV2VisionMetadata(col_index, freqs, segments)

    def get_num_output_tokens(self, input_len):
        self._validate_capacity(input_len)
        return input_len // self.spatial_merge_unit

    def _validate_capacity(self, capacity):
        if capacity <= 0 or capacity % self.spatial_merge_unit:
            raise ValueError(
                f"MiMoV2 vision input rows must be a positive multiple of "
                f"{self.spatial_merge_unit}, got {capacity}."
            )


@register_pytree_node_class
@dataclass
class MiMoV2AudioMetadata:
    marker: Any

    def tree_flatten(self):
        return (self.marker,), None

    @classmethod
    def tree_unflatten(cls, _, children):
        return cls(*children)


class MiMoV2AudioPlanBuilder(EncoderPlanBuilder):
    input_modalities = (Modality.AUDIO,)
    output_modality = Modality.AUDIO

    def __init__(self, model_config, input_buckets=None):
        super().__init__(input_buckets)
        config = getattr(model_config.hf_config, "audio_config", None)
        if config is None:
            raise ValueError("MiMoV2 audio planning requires audio_config.")
        self.channels = int(_value(config, "audio_channels"))
        self.group_size = int(_value(config, "group_size"))
        if self.channels <= 0 or self.group_size <= 0:
            raise ValueError("MiMoV2 audio_channels and group_size must be positive.")
        self.vocab_sizes = _int_list(_value(config, "speech_vocab_size"), self.channels)
        self.feature_dim = self.channels

    def select_items(self, req):
        items = super().select_items(req)
        padded = []
        for item in items:
            codes = np.asarray(item.feature)
            if codes.ndim != 2 or codes.shape[1] < self.channels or not codes.shape[0]:
                raise ValueError(
                    f"MiMoV2 audio codes must be [T, C] with C >= {self.channels}, "
                    f"got {codes.shape}."
                )
            codes = codes[:, : self.channels]
            if not np.issubdtype(codes.dtype, np.integer) or np.any(codes < 0):
                raise ValueError("MiMoV2 audio codes must be non-negative integers.")
            for channel, size in enumerate(self.vocab_sizes):
                if np.any(codes[:, channel] >= size):
                    raise ValueError(
                        f"MiMoV2 audio code on channel {channel} exceeds vocab size {size}."
                    )
            pad = (-codes.shape[0]) % self.group_size
            if pad:
                codes = np.concatenate((codes, np.repeat(codes[-1:], pad, axis=0)))
            clone = copy.copy(item)
            clone.feature = codes
            padded.append(clone)
        return padded

    def get_metadata(self, items):
        return MiMoV2AudioMetadata(np.asarray([len(items)], dtype=np.int32))

    def dummy_metadata(self, input_capacity):
        return MiMoV2AudioMetadata(np.asarray([0], dtype=np.int32))

    def pad_metadata(self, meta, input_capacity):
        self.get_num_output_tokens(input_capacity)
        return meta

    def get_num_output_tokens(self, input_len):
        if input_len <= 0 or input_len % self.group_size:
            raise ValueError(
                f"MiMoV2 audio code rows must be a positive multiple of "
                f"group_size={self.group_size}, got {input_len}."
            )
        return input_len // self.group_size


class MiMoV2PlanBuilder:
    def __init__(self, model_config):
        config = model_config.hf_config
        self.builders = []
        if getattr(config, "vision_config", None) is not None:
            self.builders.append(MiMoV2VisionPlanBuilder(model_config))
        if getattr(config, "audio_config", None) is not None:
            self.builders.append(MiMoV2AudioPlanBuilder(model_config))
        if not self.builders:
            raise ValueError("MiMoV2 multimodal planning requires vision_config or audio_config.")

    @property
    def input_buckets(self):
        return self.builders[0].input_buckets

    @input_buckets.setter
    def input_buckets(self, value):
        for builder in self.builders:
            builder.input_buckets = value

    def build(self, *args, **kwargs):
        plan = {}
        for builder in self.builders:
            if batch := builder.build(*args, **kwargs):
                plan.update(batch)
        return plan or None

    def dummy_plan(self, *args, **kwargs):
        plan = {}
        for builder in self.builders:
            plan.update(builder.dummy_plan(*args, **kwargs))
        return plan

    def get_num_output_tokens(self, input_len):
        lengths = {builder.get_num_output_tokens(input_len) for builder in self.builders}
        if len(lengths) != 1:
            raise ValueError("MiMoV2 encoder towers must use the same input/output ratio.")
        return lengths.pop()
