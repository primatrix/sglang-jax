from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.multimodal.common.modality_enum import Modality
from sgl_jax.srt.multimodal.in_model.embedding_view import PooledEmbedding
from sgl_jax.srt.request_time_stats import merge_part_time_stats

# Adapted for JAX from SGLang's encoder receiver data structures:
# https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/disaggregation/encode_receiver.py


_MODALITY_GRID_KEYS = {
    Modality.IMAGE: ("image_grid_thw", False),
    Modality.VIDEO: ("video_grid_thw", False),
    Modality.AUDIO: ("audio_feature_lens", True),
}


@dataclass(slots=True)
class EmbeddingData:
    """One encoder part: identity, reconstruction metadata, and transfer descriptor."""

    req_id: str
    num_parts: int
    part_idx: int
    modality: Modality
    grid_dim: Any = None
    shape: tuple[int, ...] | None = None
    dtype: str | None = None
    error_msg: str | None = None
    item_hashes: list[int] | None = None
    second_per_grid_ts: list[float] | None = None
    transfer: dict[str, Any] = field(default_factory=dict)
    timing: dict[str, int] | None = None

    @property
    def transfer_id(self) -> str:
        return self.transfer["transfer_id"]


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

        grouped: dict[Modality, list[jax.Array | PooledEmbedding]] = {}
        for data, embedding in parts:
            grouped.setdefault(data.modality, []).append(embedding)
        result = {}
        for modality, embeddings in grouped.items():
            if len(embeddings) == 1:
                result[modality] = embeddings[0]
                continue
            if all(isinstance(embedding, PooledEmbedding) for embedding in embeddings):
                result[modality] = PooledEmbedding.concatenate(embeddings)
            elif any(isinstance(embedding, PooledEmbedding) for embedding in embeddings):
                raise ValueError("Cannot mix pooled and materialized embedding parts")
            else:
                result[modality] = jnp.concatenate(embeddings, axis=0)
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

    def get_timing_meta(self) -> dict[str, int] | None:
        return (
            merge_part_time_stats(
                part[0].timing
                for part in self._parts
                if part is not None and part[0].timing is not None
            )
            or None
        )

    def release(self) -> None:
        """Release received parts when their request is cancelled before admission."""
        for part in self._parts:
            if part is not None and isinstance(part[1], PooledEmbedding):
                part[1].release()
