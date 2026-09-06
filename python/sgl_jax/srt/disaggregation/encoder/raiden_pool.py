"""Donated packed-output copies into the registered Raiden source buffer."""

from __future__ import annotations

import logging
import math
import time
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.disaggregation.encoder.transfer_layout import encoder_pool_block_shape

logger = logging.getLogger(__name__)


def _pool_sharding(sharding: jax.sharding.Sharding) -> jax.sharding.Sharding:
    if isinstance(sharding, jax.sharding.NamedSharding):
        spec = jax.sharding.PartitionSpec(None, *tuple(sharding.spec))
        return jax.sharding.NamedSharding(sharding.mesh, spec)
    return sharding


@partial(jax.jit, donate_argnums=(0,), static_argnames=("token_counts",))
def _copy_packed_batch_into_slots(
    pool: jax.Array,
    packed: jax.Array,
    slots: jax.Array,
    *,
    token_counts: tuple[int, ...],
) -> tuple[jax.Array, tuple[jax.Array, ...]]:
    """Split one packed output directly into donated Raiden pool slots."""

    updated = pool
    block_shape = pool.shape[1:]
    padded_shape = (block_shape[0], math.prod(block_shape[1:]))
    offset = 0
    for index, token_count in enumerate(token_counts):
        value = jax.lax.dynamic_slice_in_dim(packed, offset, token_count, axis=0)
        padding = tuple((0, padded - size) for padded, size in zip(padded_shape, value.shape))
        block = jnp.pad(value, padding).reshape(block_shape)
        updated = jax.lax.dynamic_update_slice_in_dim(
            updated,
            block[None],
            slots[index],
            axis=0,
        )
        offset += token_count

    ready = tuple(
        jax.lax.dynamic_index_in_dim(
            updated,
            slots[index],
            axis=0,
            keepdims=False,
        ).reshape(
            -1
        )[0]
        for index in range(len(token_counts))
    )
    return updated, ready


@partial(jax.jit, donate_argnums=(0,), static_argnames=("token_counts",))
def _copy_contiguous_packed_batch_into_slots(
    pool: jax.Array,
    packed: jax.Array,
    start_slot: jax.Array,
    *,
    token_counts: tuple[int, ...],
) -> tuple[jax.Array, tuple[jax.Array, ...]]:
    """Write one packed batch into a contiguous pool extent with one DUS."""

    rows = token_counts[0]
    batch_size = len(token_counts)
    block_shape = pool.shape[1:]
    padded_width = math.prod(block_shape[1:])
    values = jax.lax.dynamic_slice_in_dim(
        packed,
        0,
        batch_size * rows,
        axis=0,
    )
    if rows < block_shape[0]:
        blocks = jnp.pad(
            values.reshape(batch_size, rows, packed.shape[1]),
            ((0, 0), (0, block_shape[0] - rows), (0, padded_width - packed.shape[1])),
        ).reshape((batch_size, *block_shape))
    else:
        blocks = jnp.pad(
            values,
            ((0, 0), (0, padded_width - packed.shape[1])),
        ).reshape((batch_size, *block_shape))
    updated = jax.lax.dynamic_update_slice_in_dim(
        pool,
        blocks,
        start_slot,
        axis=0,
    )
    ready = tuple(
        jax.lax.dynamic_index_in_dim(
            updated,
            start_slot + index,
            axis=0,
            keepdims=False,
        ).reshape(
            -1
        )[0]
        for index in range(batch_size)
    )
    return updated, ready


def _compile_donated_packed_copy(
    pool: jax.Array,
    packed: jax.Array,
    token_counts: tuple[int, ...],
    *,
    contiguous: bool,
) -> Any:
    copy_fn = (
        _copy_contiguous_packed_batch_into_slots if contiguous else _copy_packed_batch_into_slots
    )
    slot_spec = jax.ShapeDtypeStruct(() if contiguous else (len(token_counts),), np.int32)
    compiled = copy_fn.lower(pool, packed, slot_spec, token_counts=token_counts).compile()
    stats = compiled.memory_analysis()
    stats = stats if isinstance(stats, (list, tuple)) else (stats,)
    if not stats or any(
        stat is None
        or int(getattr(stat, "alias_size_in_bytes", 0)) <= 0
        or int(getattr(stat, "alias_size_in_bytes", 0)) * 100
        < int(getattr(stat, "output_size_in_bytes", 0)) * 99
        for stat in stats
    ):
        raise RuntimeError("Raiden packed pool update did not alias its donated input")
    return compiled


def compile_packed_pool_copy(
    packed: jax.Array | jax.ShapeDtypeStruct,
    request_shape: tuple[int, int],
    *,
    capacity: int,
    token_counts: tuple[int, ...],
    contiguous: bool = False,
) -> Any:
    block_shape = encoder_pool_block_shape(request_shape)
    pool = jax.ShapeDtypeStruct(
        (capacity, *block_shape),
        packed.dtype,
        sharding=_pool_sharding(packed.sharding),
    )
    start_ns = time.perf_counter_ns()
    compiled = _compile_donated_packed_copy(
        pool,
        packed,
        token_counts,
        contiguous=contiguous,
    )
    logger.info(
        "ENCODER-POOL-WRITE-PRECOMPILE capacity=%d batch_size=%d duration_ms=%.3f contiguous=%s",
        packed.shape[0],
        len(token_counts),
        (time.perf_counter_ns() - start_ns) / 1_000_000,
        contiguous,
    )
    return compiled


class RaidenSendPool:
    """Reusable source buffer with bounded, request-sized slots."""

    def __init__(
        self,
        shape: tuple[int, int],
        dtype: jnp.dtype,
        sharding: jax.sharding.Sharding,
        *,
        capacity: int,
    ) -> None:
        self.shape = tuple(int(dim) for dim in shape)
        self.dtype = jnp.dtype(dtype)
        self.sharding = sharding
        pool_sharding = _pool_sharding(sharding)
        self._block_shape = encoder_pool_block_shape(self.shape)
        self._buffer = jnp.zeros(
            (capacity, *self._block_shape),
            dtype=self.dtype,
            device=pool_sharding,
        )
        jax.block_until_ready(self._buffer)

    @property
    def buffer(self) -> jax.Array:
        return self._buffer

    def copy_packed_batch_async(
        self,
        packed: jax.Array,
        slots: list[int],
        token_counts: tuple[int, ...],
        executable: Any,
        *,
        contiguous: bool,
    ) -> tuple[jax.Array, ...]:
        if len(slots) != len(token_counts):
            raise ValueError("Raiden slot and packed item counts differ")
        if not token_counts:
            return ()
        if any(token_count != self.shape[0] for token_count in token_counts):
            raise ValueError("Raiden packed output contains incompatible item shapes")
        if (
            packed.ndim != 2
            or packed.shape[1] != self.shape[1]
            or packed.dtype != self.dtype
            or packed.sharding != self.sharding
            or sum(token_counts) > packed.shape[0]
        ):
            raise ValueError("Raiden packed output does not match the source pool")

        inferred_contiguous = slots == list(range(slots[0], slots[0] + len(slots)))
        if contiguous != inferred_contiguous:
            raise ValueError("Raiden contiguous pool-write mode does not match slots")
        # Place slot IDs directly on the executable's mesh. An uncommitted
        # default-device array forces cpp_pjit_shard_arg_fallback on every call.
        slot_indices = jax.device_put(
            np.asarray(slots[0] if contiguous else slots, dtype=np.int32),
            executable.input_shardings[0][2],
        )
        self._buffer, ready = executable(self._buffer, packed, slot_indices)
        return ready
