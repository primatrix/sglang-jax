from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.multimodal.common.modality_enum import Modality

# Adapted for JAX from SGLang's encoder receiver data structures:
# https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/disaggregation/encode_receiver.py


_MODALITY_GRID_KEYS = {
    Modality.IMAGE: ("img_grid_thw", False),
    Modality.VIDEO: ("video_grid_thw", False),
    Modality.AUDIO: ("audio_feature_lens", True),
}


@dataclass(frozen=True, slots=True)
class PooledEmbedding:
    """A row view into a registered receive pool without a device-side slice."""

    buffer: jax.Array
    slot: int
    block_shape: tuple[int, ...]
    shape: tuple[int, int]
    lease: Any
    row_offset: int = 0

    @property
    def dtype(self):
        return self.buffer.dtype

    @property
    def ndim(self) -> int:
        return 2

    @property
    def flat_row_start(self) -> int:
        return self.slot * self.block_shape[0] + self.row_offset

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, index: slice) -> PooledEmbedding:
        if not isinstance(index, slice) or index.step not in (None, 1):
            raise TypeError("pooled embeddings support contiguous row slices only")
        start, stop, step = index.indices(self.shape[0])
        if step != 1:
            raise TypeError("pooled embeddings support contiguous row slices only")
        return PooledEmbedding(
            self.buffer,
            self.slot,
            self.block_shape,
            (stop - start, self.shape[1]),
            self.lease,
            self.row_offset + start,
        )

    def materialize(self) -> jax.Array:
        source = self.buffer.reshape(
            self.buffer.shape[0] * self.block_shape[0],
            -1,
        )
        return jax.lax.dynamic_slice(
            source,
            (self.flat_row_start, 0),
            self.shape,
        )


class EmbeddingData:
    """Adapted from sglang.srt.disaggregation.encode_receiver.EmbeddingData for JAX."""

    def __init__(
        self,
        req_id: str,
        num_parts: int,
        part_idx: int,
        grid_dim: Any,
        modality: Modality,
        embedding_shape: list[int] | tuple[int, ...] | None = None,
        dtype: Any = None,
        error_msg: str | None = None,
        dispatch_start_ns: int | None = None,
        enqueue_ns: int | None = None,
        dequeue_ns: int | None = None,
        preprocess_start_ns: int | None = None,
        preprocess_done_ns: int | None = None,
        encode_start_ns: int | None = None,
        encode_done_ns: int | None = None,
        transfer_enqueue_ns: int | None = None,
        transfer_start_ns: int | None = None,
        transfer_reserve_start_ns: int | None = None,
        transfer_pool_ready_ns: int | None = None,
        transfer_reserve_done_ns: int | None = None,
        transfer_copy_start_ns: int | None = None,
        transfer_copy_submit_ns: int | None = None,
        transfer_copy_done_ns: int | None = None,
        transfer_register_start_ns: int | None = None,
        transfer_register_done_ns: int | None = None,
        transfer_publish_ready_ns: int | None = None,
        transfer_stage_done_ns: int | None = None,
        publish_done_ns: int | None = None,
        receive_metadata_ns: int | None = None,
        receive_setup_done_ns: int | None = None,
        receive_transfer_done_ns: int | None = None,
        receive_materialize_start_ns: int | None = None,
        receive_materialize_done_ns: int | None = None,
        receive_embedding_ns: int | None = None,
        queue_duration_ns: int | None = None,
        queue_ms: float | None = None,
        **kwargs: Any,
    ) -> None:
        self.req_id = req_id
        self.num_parts = num_parts
        self.part_idx = part_idx
        self.grid_dim = grid_dim
        self.modality = modality
        self.dtype = dtype
        self.shape = embedding_shape
        self.error_msg = error_msg
        # Encoder scheduler application-level timing. enqueue_ns/dequeue_ns
        # use Unix epoch time for cross-process correlation; queue_duration_ns
        # and queue_ms are calculated from a monotonic clock.
        self.dispatch_start_ns = dispatch_start_ns
        self.enqueue_ns = enqueue_ns
        self.dequeue_ns = dequeue_ns
        self.preprocess_start_ns = preprocess_start_ns
        self.preprocess_done_ns = preprocess_done_ns
        self.encode_start_ns = encode_start_ns
        self.encode_done_ns = encode_done_ns
        self.transfer_enqueue_ns = transfer_enqueue_ns
        self.transfer_start_ns = transfer_start_ns
        self.transfer_reserve_start_ns = transfer_reserve_start_ns
        self.transfer_pool_ready_ns = transfer_pool_ready_ns
        self.transfer_reserve_done_ns = transfer_reserve_done_ns
        self.transfer_copy_start_ns = transfer_copy_start_ns
        self.transfer_copy_submit_ns = transfer_copy_submit_ns
        self.transfer_copy_done_ns = transfer_copy_done_ns
        self.transfer_register_start_ns = transfer_register_start_ns
        self.transfer_register_done_ns = transfer_register_done_ns
        self.transfer_publish_ready_ns = transfer_publish_ready_ns
        self.transfer_stage_done_ns = transfer_stage_done_ns
        self.publish_done_ns = publish_done_ns
        self.receive_metadata_ns = receive_metadata_ns
        self.receive_setup_done_ns = receive_setup_done_ns
        self.receive_transfer_done_ns = receive_transfer_done_ns
        self.receive_materialize_start_ns = receive_materialize_start_ns
        self.receive_materialize_done_ns = receive_materialize_done_ns
        self.receive_embedding_ns = receive_embedding_ns
        self.queue_duration_ns = queue_duration_ns
        self.queue_ms = queue_ms
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __repr__(self) -> str:
        return (
            f"EmbeddingData(req_id={self.req_id}, num_parts={self.num_parts}, "
            f"part_idx={self.part_idx}, error_msg={self.error_msg})"
        )


class MultiModalEmbeddingData:
    def __init__(self, num_parts: int) -> None:
        if num_parts <= 0:
            raise ValueError("num_parts must be positive")
        self.num_parts = num_parts
        self._parts: list[tuple[EmbeddingData, jax.Array | PooledEmbedding] | None] = [
            None
        ] * num_parts

    def add(self, data: EmbeddingData, embedding: jax.Array | PooledEmbedding) -> None:
        if data.num_parts != self.num_parts:
            raise ValueError("inconsistent num_parts")
        if not 0 <= data.part_idx < self.num_parts:
            raise ValueError(f"invalid part_idx: {data.part_idx}")
        if self._parts[data.part_idx] is not None:
            raise ValueError(f"duplicate part_idx: {data.part_idx}")
        self._parts[data.part_idx] = (data, embedding)

    @property
    def ready(self) -> bool:
        return all(part is not None for part in self._parts)

    def has_part(self, part_idx: int) -> bool:
        return 0 <= part_idx < self.num_parts and self._parts[part_idx] is not None

    def get_embedding(self, is_concat: bool = False):
        if not self.ready:
            raise RuntimeError("embedding parts are incomplete")
        parts = [part for part in self._parts if part is not None]
        if not is_concat:
            return [embedding for _, embedding in parts]

        grouped: dict[Modality, list[jax.Array]] = {}
        for data, embedding in parts:
            grouped.setdefault(data.modality, []).append(embedding)
        result = {}
        for modality, embeddings in grouped.items():
            if len(embeddings) == 1:
                result[modality] = embeddings[0]
                continue
            materialized = [
                embedding.materialize() if isinstance(embedding, PooledEmbedding) else embedding
                for embedding in embeddings
            ]
            combined = jnp.concatenate(materialized, axis=0)
            leases = {
                id(embedding.lease): embedding.lease
                for embedding in embeddings
                if isinstance(embedding, PooledEmbedding)
            }
            for lease in leases.values():
                lease.release_after(combined)
            result[modality] = combined
        return result

    def get_mm_extra_meta(self) -> dict[str, Any]:
        result = {}
        parts = [part for part in self._parts if part is not None]
        for modality, (key, flatten) in _MODALITY_GRID_KEYS.items():
            values = []
            for data, _ in parts:
                if data.modality != modality or data.grid_dim is None:
                    continue
                value = np.asarray(data.grid_dim)
                if flatten:
                    value = value.reshape(-1)
                elif value.ndim == 0:
                    value = value.reshape(1)
                values.append(value)
            if values:
                result[key] = values[0] if len(values) == 1 else np.concatenate(values)

        item_hashes: dict[Modality, list[int]] = {}
        for data, _ in parts:
            values = getattr(data, "item_hashes", None)
            if values:
                item_hashes.setdefault(data.modality, []).extend(map(int, values))
        if item_hashes:
            result["item_hashes"] = item_hashes

        second_per_grid_ts = []
        for data, _ in parts:
            if data.modality == Modality.VIDEO:
                values = getattr(data, "second_per_grid_ts", None)
                if values is not None:
                    second_per_grid_ts.extend(np.asarray(values).ravel().tolist())
        if second_per_grid_ts:
            result["second_per_grid_ts"] = second_per_grid_ts
        return result

    def get_timing_meta(self) -> dict[str, int]:
        parts = [part for part in self._parts if part is not None]
        fields = (
            "dispatch_start_ns",
            "enqueue_ns",
            "dequeue_ns",
            "preprocess_start_ns",
            "preprocess_done_ns",
            "encode_start_ns",
            "encode_done_ns",
            "encode_server_postprocess_done_ns",
            "encode_server_postprocess_duration_ns",
            "encode_token_count_duration_ns",
            "encode_embedding_slice_duration_ns",
            "encode_split_compile_wait_duration_ns",
            "encode_split_dispatch_duration_ns",
            "encode_metadata_duration_ns",
            "encode_result_pack_duration_ns",
            "encode_server_postprocess_residual_ns",
            "runtime_encode_return_ns",
            "runtime_postprocess_done_ns",
            "runtime_postprocess_duration_ns",
            "runtime_metadata_prepare_duration_ns",
            "runtime_embedding_data_duration_ns",
            "runtime_result_pack_duration_ns",
            "runtime_postprocess_residual_ns",
            "runtime_timing_attach_duration_ns",
            "transfer_enqueue_ns",
            "transfer_start_ns",
            "transfer_reserve_start_ns",
            "transfer_pool_ready_ns",
            "transfer_reserve_done_ns",
            "transfer_copy_start_ns",
            "transfer_copy_submit_ns",
            "transfer_copy_done_ns",
            "transfer_register_start_ns",
            "transfer_register_done_ns",
            "transfer_publish_ready_ns",
            "transfer_stage_done_ns",
            "publish_done_ns",
            "receive_metadata_ns",
            "receive_setup_done_ns",
            "receive_transfer_done_ns",
            "receive_materialize_start_ns",
            "receive_materialize_done_ns",
            "receive_embedding_ns",
            "preprocess_request_start_ns",
            "image_load_start_ns",
            "image_load_done_ns",
            "processor_submit_ns",
            "processor_start_ns",
            "processor_done_ns",
            "preprocess_request_done_ns",
        )
        start_fields = {
            "dispatch_start_ns",
            "enqueue_ns",
            "preprocess_request_start_ns",
            "image_load_start_ns",
            "processor_submit_ns",
            "processor_start_ns",
            "transfer_enqueue_ns",
            "transfer_start_ns",
            "transfer_reserve_start_ns",
            "transfer_copy_start_ns",
            "transfer_register_start_ns",
            "receive_metadata_ns",
            "receive_materialize_start_ns",
        }
        timing = {}
        for field in fields:
            values = [getattr(data, field, None) for data, _ in parts]
            values = [int(value) for value in values if value is not None]
            if values:
                timing[field] = min(values) if field in start_fields else max(values)
        return timing
