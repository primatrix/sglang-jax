from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp

TPU_TILE_SHAPE = (8, 128)


@dataclass(frozen=True, slots=True)
class PackedEmbeddingSlice:
    """A logical request embedding backed by one packed encoder output."""

    packed: jax.Array
    offset: int
    rows: int
    max_batch_size: int
    packed_capacities: tuple[int, ...] = ()

    @property
    def shape(self) -> tuple[int, int]:
        return self.rows, int(self.packed.shape[1])

    @property
    def ndim(self) -> int:
        return 2

    @property
    def dtype(self):
        return self.packed.dtype

    @property
    def sharding(self):
        return self.packed.sharding


def encoder_pool_block_shape(shape: Sequence[int]) -> tuple[int, int, int, int]:
    """Return the request-slot shape registered with Raiden."""

    if len(shape) != 2:
        raise ValueError("encoder embedding must be a matrix")
    rows, width = (int(dim) for dim in shape)
    if rows <= 0 or width <= 0:
        raise ValueError("encoder embedding dimensions must be positive")
    tile_width = math.prod(TPU_TILE_SHAPE)
    return max(2, rows), max(2, math.ceil(width / tile_width)), *TPU_TILE_SHAPE


def encoder_transfer_nbytes(shape: Sequence[int], dtype: object) -> int:
    """Return bytes transferred for one padded Raiden request slot."""

    return math.prod(encoder_pool_block_shape(shape)) * jnp.dtype(dtype).itemsize
