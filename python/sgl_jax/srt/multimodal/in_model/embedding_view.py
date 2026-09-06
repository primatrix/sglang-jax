"""A zero-copy view whose lease protects an asynchronous embedding buffer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import jax


class EmbeddingLease(Protocol):
    def release_after(self, dependency: Any) -> None: ...
    def release(self) -> None: ...


@dataclass(frozen=True, slots=True)
class PooledEmbedding:
    """A row view into a registered receive pool without a device-side slice."""

    buffer: jax.Array
    slot: int
    block_shape: tuple[int, ...]
    shape: tuple[int, int]
    lease: EmbeddingLease
    row_offset: int = 0
    # Preserve the full transfer extent when individual items take row slices.
    total_rows: int | None = None

    @property
    def dtype(self):
        return self.buffer.dtype

    @property
    def ndim(self) -> int:
        return 2

    @property
    def flat_row_start(self) -> int:
        return self.slot * self.block_shape[0] + self.row_offset

    @property
    def flat_buffer(self) -> jax.Array:
        return self.buffer.reshape(self.buffer.shape[0] * self.block_shape[0], -1)

    @property
    def is_last_slice(self) -> bool:
        return self.total_rows is None or self.row_offset + self.shape[0] == self.total_rows

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
            (max(0, stop - start), self.shape[1]),
            self.lease,
            self.row_offset + start,
            self.total_rows if self.total_rows is not None else self.row_offset + self.shape[0],
        )

    def materialize(self) -> jax.Array:
        return jax.lax.dynamic_slice(
            self.flat_buffer,
            (self.flat_row_start, 0),
            self.shape,
        )
