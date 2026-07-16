"""Architecture-agnostic in-model vision encode/merge planning.

A model plugs into this framework by registering a :class:`VisionEncoderPlugin`
under its HF architecture name (:func:`register_vision_encoder`). This builder
turns scheduled requests into fixed-shape ``[dp, tp]`` encoder batches and
token-merge routes, delegating only the model-specific metadata math and
encode-input pytree assembly to the plugin.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

import jax
import numpy as np

from sgl_jax.srt.multimodal.common.in_model_plan_builder import (
    register_in_model_plan_builder,
)
from sgl_jax.srt.multimodal.common.mm_plan import (
    DeviceMergePlan,
    ModalityEmbedBatch,
    ModalityEncodeInputs,
    MultimodalEmbedPlan,
)
from sgl_jax.srt.multimodal.common.modality_enum import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from sgl_jax.srt.configs.model_config import ModelConfig
    from sgl_jax.srt.managers.schedule_batch import ScheduleReqsInfo


class VisionEncoderPlugin(Protocol):
    """Model-specific hooks the generic vision plan builder calls into.

    The builder owns all request/lane/merge orchestration; a plugin only
    contributes its metadata math and the shape of its encode-input pytree.
    """

    # Modality under which merged encoder rows land (e.g. ``Modality.IMAGE``).
    output_modality: Modality

    def get_metadata(self, items: Sequence[MultimodalDataItem]) -> Any:
        """Per-lane request metadata (no dp/tp axis) for ``items`` in order."""
        ...

    def stack_metadata(self, lane_metadata: Sequence[Any | None], patch_k: int) -> Any:
        """Pad and stack per-lane metadata into leading ``[dp*tp, ...]`` arrays."""
        ...

    def make_encode_inputs(
        self, patches: np.ndarray, valid: np.ndarray, meta: Any
    ) -> ModalityEncodeInputs:
        """Assemble the model's encode-input pytree from stacked lane arrays."""
        ...


@dataclass(frozen=True)
class MergeSlice:
    """Host-only mapping from part of one task's output to current token rows."""

    task_id: int
    feature_start: int
    token_start: int
    length: int


@dataclass(frozen=True)
class _EncodeTask:
    task_id: int
    item: MultimodalDataItem
    encoded_rows: int
    merge_slices: tuple[MergeSlice, ...]

    @property
    def patch_rows(self) -> int:
        return int(np.asarray(self.item.feature).shape[0])

    @property
    def merge_rows(self) -> int:
        return sum(part.length for part in self.merge_slices)


def _assign_tp_lanes(tasks: list[_EncodeTask], tp_size: int) -> list[list[_EncodeTask]]:
    if tp_size <= 0:
        raise ValueError(f"tp_size must be positive, got {tp_size}.")
    if tp_size == 1:
        return [list(tasks)]

    lanes: list[list[_EncodeTask]] = [[] for _ in range(tp_size)]
    loads = [0] * tp_size
    for task in sorted(tasks, key=lambda task: (-task.patch_rows, task.task_id)):
        rank = min(range(tp_size), key=lambda candidate: (loads[candidate], candidate))
        lanes[rank].append(task)
        loads[rank] += task.patch_rows
    return lanes


def _vision_items(req) -> list[MultimodalDataItem]:
    mm_inputs = req.mm_inputs
    if mm_inputs is None or isinstance(mm_inputs, dict) and "mm_items" not in mm_inputs:
        return []
    if not isinstance(mm_inputs, MultimodalInputs):
        raise TypeError(
            "vision plan builder expects req.mm_inputs to be MultimodalInputs, "
            f"got {type(mm_inputs).__name__}."
        )
    if any(not isinstance(item, MultimodalDataItem) for item in mm_inputs.mm_items):
        bad = next(item for item in mm_inputs.mm_items if not isinstance(item, MultimodalDataItem))
        raise TypeError(
            "vision plan builder expects mm_items to contain "
            f"MultimodalDataItem, got {type(bad).__name__}."
        )
    return [item for item in mm_inputs.mm_items if item.is_image() or item.is_video()]


def _build_task(
    item: MultimodalDataItem,
    task_id: int,
    dp_rank: int,
    req_base: int,
    chunk_start: int,
    chunk_end: int,
    per_dp_token: int,
) -> _EncodeTask | None:
    feature = None if item.feature is None else np.asarray(item.feature)
    if feature is None:
        raise ValueError(f"Vision item in dp_rank {dp_rank} is missing feature.")
    if feature.ndim != 2 or not feature.shape[0]:
        raise ValueError(
            f"Vision item feature must be a non-empty 2D patch array, got shape={feature.shape} "
            f"in dp_rank {dp_rank}."
        )
    if not item.placeholder_ranges:
        raise ValueError(f"Vision item in dp_rank {dp_rank} has no placeholder ranges.")

    slices = []
    encoded_rows = 0
    previous_end = -1
    for start, end in item.placeholder_ranges:
        length = end - start + 1
        if length <= 0:
            raise ValueError(
                f"Vision placeholder range must be non-empty, got start={start}, end={end}."
            )
        if start <= previous_end:
            raise ValueError(
                "Vision placeholder token is assigned more than once: "
                f"previous_end={previous_end}, start={start}."
            )
        previous_end = end

        overlap_start = max(start, chunk_start)
        overlap_end = min(end + 1, chunk_end)
        if overlap_start < overlap_end:
            token_start = req_base + overlap_start - chunk_start
            merge_length = overlap_end - overlap_start
            if token_start < 0 or token_start + merge_length > per_dp_token:
                raise ValueError(
                    "Vision placeholder chunk is outside its packed rank slot: "
                    f"dp_rank={dp_rank}, req_base={req_base}, "
                    f"chunk=({chunk_start}, {chunk_end}), range=({start}, {end}), "
                    f"per_dp_token={per_dp_token}."
                )
            slices.append(
                MergeSlice(
                    task_id,
                    encoded_rows + overlap_start - start,
                    token_start,
                    merge_length,
                )
            )
        encoded_rows += length

    return _EncodeTask(task_id, item, encoded_rows, tuple(slices)) if slices else None


def _collect_encode_tasks(
    reqs_info: list[ScheduleReqsInfo] | None,
    dp_size: int,
    per_dp_token: int,
) -> list[list[_EncodeTask]]:
    tasks_by_dp: list[list[_EncodeTask]] = [[] for _ in range(dp_size)]
    task_id = 0
    for dp_rank, info in enumerate((reqs_info or [])[:dp_size]):
        req_base = 0
        for req_index, req in enumerate(info.reqs or []):
            prefix_len = (
                info.prefix_lens[req_index]
                if info.prefix_lens is not None and req_index < len(info.prefix_lens)
                else len(getattr(req, "prefix_indices", []) or [])
            )
            extend_len = (
                info.extend_lens[req_index]
                if info.extend_lens is not None and req_index < len(info.extend_lens)
                else int(getattr(req, "extend_input_len", 0) or 0)
            )
            if prefix_len < 0 or extend_len < 0:
                raise ValueError(
                    "Vision chunk bounds must be non-negative, "
                    f"got prefix_len={prefix_len}, extend_len={extend_len}."
                )

            for item in _vision_items(req):
                task = _build_task(
                    item,
                    task_id,
                    dp_rank,
                    req_base,
                    prefix_len,
                    prefix_len + extend_len,
                    per_dp_token,
                )
                if task is not None:
                    tasks_by_dp[dp_rank].append(task)
                    task_id += 1
            req_base += extend_len
    return tasks_by_dp


def _reshape_metadata(meta, dp_size: int, tp_size: int):
    def reshape(value):
        value = np.asarray(value)
        if value.ndim == 0 or value.shape[0] != dp_size * tp_size:
            raise ValueError(
                "Vision metadata leaves must be lane-leading after stack_metadata: "
                f"expected first dimension={dp_size * tp_size}, got shape={value.shape}."
            )
        return value.reshape(dp_size, tp_size, *value.shape[1:])

    return jax.tree.map(reshape, meta)


def _build_modality_batch(
    tasks_by_dp: list[list[_EncodeTask]],
    plugin: VisionEncoderPlugin,
    dp_size: int,
    tp_size: int,
) -> ModalityEmbedBatch:
    lanes_by_dp = [_assign_tp_lanes(tasks, tp_size) for tasks in tasks_by_dp]
    lane_features: list[list[np.ndarray | None]] = [[None] * tp_size for _ in range(dp_size)]
    lane_metadata: list[Any] = []
    placements: dict[int, tuple[int, int, int]] = {}
    merge_counts = np.zeros((dp_size, tp_size), dtype=np.int32)

    for dp_rank, lanes in enumerate(lanes_by_dp):
        for tp_rank, tasks in enumerate(lanes):
            if not tasks:
                lane_metadata.append(None)
                continue
            base = 0
            for task in tasks:
                placements[task.task_id] = dp_rank, tp_rank, base
                base += task.encoded_rows
                merge_counts[dp_rank, tp_rank] += task.merge_rows
            lane_features[dp_rank][tp_rank] = np.concatenate(
                [np.asarray(task.item.feature) for task in tasks]
            )
            lane_metadata.append(plugin.get_metadata([task.item for task in tasks]))

    real_features = [feature for lanes in lane_features for feature in lanes if feature is not None]
    patch_k = max(feature.shape[0] for feature in real_features)
    patches = np.zeros(
        (dp_size, tp_size, patch_k, real_features[0].shape[1]),
        dtype=np.float32,
    )
    valid = np.zeros((dp_size, tp_size), dtype=np.int32)
    for dp_rank, feature_lanes in enumerate(lane_features):
        for tp_rank, feature in enumerate(feature_lanes):
            if feature is not None:
                valid[dp_rank, tp_rank] = feature.shape[0]
                patches[dp_rank, tp_rank, : feature.shape[0]] = feature

    meta = plugin.stack_metadata(lane_metadata, patch_k)
    meta = _reshape_metadata(meta, dp_size, tp_size)

    merge_bucket = int(merge_counts.max())
    src_idx = np.zeros((dp_size, tp_size, merge_bucket), dtype=np.int32)
    dst_idx = np.zeros_like(src_idx)
    mask = np.zeros_like(src_idx, dtype=np.bool_)
    cursors = np.zeros((dp_size, tp_size), dtype=np.int32)
    for task in (task for tasks in tasks_by_dp for task in tasks):
        dp_rank, tp_rank, feature_base = placements[task.task_id]
        for part in task.merge_slices:
            begin = cursors[dp_rank, tp_rank]
            end = begin + part.length
            src_idx[dp_rank, tp_rank, begin:end] = np.arange(
                feature_base + part.feature_start,
                feature_base + part.feature_start + part.length,
            )
            dst_idx[dp_rank, tp_rank, begin:end] = np.arange(
                part.token_start, part.token_start + part.length
            )
            mask[dp_rank, tp_rank, begin:end] = True
            cursors[dp_rank, tp_rank] = end

    return ModalityEmbedBatch(
        plugin.make_encode_inputs(patches, valid, meta),
        DeviceMergePlan(src_idx, dst_idx, mask),
    )


class InModelVisionPlanBuilder:
    """Drive a :class:`VisionEncoderPlugin` into a fixed-shape encode/merge plan."""

    def __init__(self, plugin: VisionEncoderPlugin) -> None:
        self.plugin = plugin

    def build(
        self,
        reqs_info: list[ScheduleReqsInfo] | None,
        dp_size: int,
        per_dp_token: int,
        tp_size: int,
    ) -> MultimodalEmbedPlan | None:
        tasks = _collect_encode_tasks(reqs_info, dp_size, per_dp_token)
        if not any(tasks):
            return None
        batch = _build_modality_batch(tasks, self.plugin, dp_size, tp_size)
        return {self.plugin.output_modality: batch}


def register_vision_encoder(
    arch: str,
    plugin_factory: Callable[[ModelConfig], VisionEncoderPlugin],
) -> None:
    """Register ``plugin_factory`` for ``arch`` behind an ``InModelVisionPlanBuilder``."""

    def builder_factory(model_config: ModelConfig) -> InModelVisionPlanBuilder:
        return InModelVisionPlanBuilder(plugin_factory(model_config))

    register_in_model_plan_builder(arch, builder_factory)
