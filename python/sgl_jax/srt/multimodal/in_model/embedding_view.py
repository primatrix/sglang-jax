"""Host-only views over a fixed device pool; slicing never moves embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import jax
import numpy as np


class EmbeddingLease(Protocol):
    def record_read(self, result: jax.Array) -> None: ...
    def release(self) -> None: ...


@dataclass(frozen=True, slots=True)
class PooledEmbedding:
    buffer: jax.Array
    row_indices: np.ndarray
    width: int
    leases: tuple[EmbeddingLease, ...]

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.row_indices), self.width

    @property
    def dtype(self):
        return self.buffer.dtype

    @property
    def ndim(self) -> int:
        return 2

    @property
    def flat_buffer(self) -> jax.Array:
        return self.buffer.reshape(self.buffer.shape[0] * self.buffer.shape[1], -1)

    def __len__(self) -> int:
        return len(self.row_indices)

    def __getitem__(self, index: slice) -> PooledEmbedding:
        if not isinstance(index, slice) or index.step not in (None, 1):
            raise TypeError("pooled embeddings support contiguous row slices only")
        return PooledEmbedding(self.buffer, self.row_indices[index], self.width, self.leases)

    @staticmethod
    def concatenate(embeddings: list[PooledEmbedding]) -> PooledEmbedding:
        first = embeddings[0]
        if len(embeddings) == 1:
            return first
        if any(e.buffer is not first.buffer or e.width != first.width for e in embeddings):
            raise ValueError("Received embeddings must share one pool and width")
        leases = {id(lease): lease for e in embeddings for lease in e.leases}
        return PooledEmbedding(
            first.buffer,
            np.concatenate([e.row_indices for e in embeddings]),
            first.width,
            tuple(leases.values()),
        )

    def record_read(self, result: jax.Array) -> None:
        for lease in self.leases:
            lease.record_read(result)

    def release(self) -> None:
        for lease in self.leases:
            lease.release()


def release_received_embeddings(mm_inputs) -> None:
    """End request ownership; leases still wait for outstanding device reads."""
    for item in getattr(mm_inputs, "mm_items", ()):
        if isinstance(item.precomputed_embeddings, PooledEmbedding):
            item.precomputed_embeddings.release()
            item.precomputed_embeddings = None
