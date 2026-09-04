from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass, field
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp

from sgl_jax.srt.disaggregation.encoder.embedding_data import PooledEmbedding
from sgl_jax.srt.disaggregation.encoder.transfer_layout import encoder_pool_block_shape
from sgl_jax.srt.disaggregation.raiden_transfer.wrapper import RaidenTransferWrapper

logger = logging.getLogger(__name__)


def _pool_sharding(sharding: jax.sharding.Sharding) -> jax.sharding.Sharding:
    if isinstance(sharding, jax.sharding.NamedSharding):
        spec = jax.sharding.PartitionSpec(None, *tuple(sharding.spec))
        return jax.sharding.NamedSharding(sharding.mesh, spec)
    return sharding


@partial(jax.jit, donate_argnums=(0,))
def _copy_into_slot(pool: jax.Array, value: jax.Array, slot: jax.Array) -> jax.Array:
    block_shape = pool.shape[1:]
    padded_shape = (block_shape[0], math.prod(block_shape[1:]))
    padding = tuple((0, padded - size) for padded, size in zip(padded_shape, value.shape))
    block = jnp.pad(value, padding).reshape(block_shape)
    return jax.lax.dynamic_update_slice_in_dim(pool, block[None], slot, axis=0)


@partial(jax.jit, donate_argnums=(0,))
def _copy_into_slot_with_token(
    pool: jax.Array,
    value: jax.Array,
    slot: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    updated = _copy_into_slot(pool, value, slot)
    block_size = math.prod(updated.shape[1:])
    ready = updated.reshape(-1)[slot * block_size]
    return updated, ready


def _compile_donated_copy(
    pool: jax.Array,
    value: jax.Array,
) -> Any:
    compiled = _copy_into_slot_with_token.lower(
        pool,
        value,
        jnp.asarray(0, dtype=jnp.int32),
    ).compile()
    stats = compiled.memory_analysis()
    stats = stats if isinstance(stats, (list, tuple)) else (stats,)
    if not stats or any(
        stat is None
        or int(getattr(stat, "alias_size_in_bytes", 0)) <= 0
        or int(getattr(stat, "alias_size_in_bytes", 0)) * 100
        < int(getattr(stat, "output_size_in_bytes", 0)) * 99
        for stat in stats
    ):
        raise RuntimeError("Raiden encoder pool update did not fully alias its donated input")
    return compiled


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


def _compile_donated_packed_copy(
    pool: jax.Array,
    packed: jax.Array,
    token_counts: tuple[int, ...],
) -> Any:
    compiled = _copy_packed_batch_into_slots.lower(
        pool,
        packed,
        jnp.zeros((len(token_counts),), dtype=jnp.int32),
        token_counts=token_counts,
    ).compile()
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
) -> Any:
    block_shape = encoder_pool_block_shape(request_shape)
    pool = jax.ShapeDtypeStruct(
        (capacity, *block_shape),
        packed.dtype,
        sharding=_pool_sharding(packed.sharding),
    )
    start_ns = time.perf_counter_ns()
    compiled = _compile_donated_packed_copy(pool, packed, token_counts)
    logger.info(
        "ENCODER-POOL-WRITE-PRECOMPILE capacity=%d batch_size=%d duration_ms=%.3f",
        packed.shape[0],
        len(token_counts),
        (time.perf_counter_ns() - start_ns) / 1_000_000,
    )
    return compiled


class RaidenSendPool:
    """Reusable source buffer with bounded, request-sized slots."""

    def __init__(
        self,
        sample: jax.Array,
        *,
        capacity: int,
    ) -> None:
        self._initialize(
            tuple(int(dim) for dim in sample.shape),
            sample.dtype,
            sample.sharding,
            capacity,
        )
        self._copy = _compile_donated_copy(self._buffer, sample)

    @classmethod
    def for_shape(
        cls,
        shape: tuple[int, int],
        dtype: jnp.dtype,
        sharding: jax.sharding.Sharding,
        *,
        capacity: int,
    ) -> RaidenSendPool:
        pool = cls.__new__(cls)
        pool._initialize(shape, dtype, sharding, capacity)
        pool._copy = None
        return pool

    def _initialize(
        self,
        shape: tuple[int, int],
        dtype: jnp.dtype,
        sharding: jax.sharding.Sharding,
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
        self._packed_copies: dict[tuple[tuple[int, ...], tuple[int, ...]], Any] = {}

    @property
    def buffer(self) -> jax.Array:
        return self._buffer

    def matches(self, value: jax.Array) -> bool:
        return (
            tuple(value.shape) == self.shape
            and value.dtype == self.dtype
            and value.sharding == self.sharding
        )

    def copy_async(self, value: jax.Array, slot: int) -> jax.Array:
        if not self.matches(value):
            raise ValueError("Raiden pool contains an incompatible embedding")
        if self._copy is None:
            self._copy = _compile_donated_copy(self._buffer, value)
        self._buffer, ready = self._copy(
            self._buffer,
            value,
            jnp.asarray(slot, dtype=jnp.int32),
        )
        return ready

    def copy_packed_batch_async(
        self,
        packed: jax.Array,
        slots: list[int],
        token_counts: tuple[int, ...],
        executable: Any | None = None,
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

        key = (tuple(int(dim) for dim in packed.shape), token_counts)
        copy = executable or self._packed_copies.get(key)
        if copy is None:
            copy = compile_packed_pool_copy(
                packed,
                self.shape,
                capacity=self._buffer.shape[0],
                token_counts=token_counts,
            )
        self._packed_copies[key] = copy
        self._buffer, ready = copy(
            self._buffer,
            packed,
            jnp.asarray(slots, dtype=jnp.int32),
        )
        return ready

    def copy_sync(self, value: jax.Array, slot: int) -> None:
        self.copy_async(value, slot).block_until_ready()


@dataclass(slots=True)
class RaidenReceiveSession:
    transfer_id: str
    lane_id: int
    pool: RaidenReceivePool
    _done: bool = False
    timing_meta: dict[str, int] = field(default_factory=dict)

    def poll(self, *, refresh_backend: bool = True) -> PooledEmbedding | None:
        if self._done:
            return None
        result = self.pool.poll(
            self.transfer_id,
            self.lane_id,
            refresh_backend=refresh_backend,
        )
        self._done = result is not None
        if result is None:
            return None
        embedding, self.timing_meta = result
        return embedding

    def close(self) -> None:
        if not self._done:
            self.pool.abandon(self.transfer_id)


class RaidenReceiveLease:
    """Release a receive slot only after its Language-side readers finish."""

    def __init__(self, pool: RaidenReceivePool, transfer_id: str, lane_id: int) -> None:
        self._pool = pool
        self._transfer_id = transfer_id
        self._lane_id = lane_id
        self._lock = threading.Lock()
        self._released = False

    def release_after(self, dependency: Any) -> None:
        with self._lock:
            if self._released:
                return
            self._released = True
        self._pool.release_after(
            self._transfer_id,
            self._lane_id,
            dependency,
        )

    def release(self) -> None:
        with self._lock:
            if self._released:
                return
            self._released = True
        self._pool.release(self._transfer_id, self._lane_id)


class RaidenReceivePool:
    """One registered destination buffer with bounded request slots."""

    def __init__(
        self,
        host: str,
        shape: tuple[int, int],
        dtype: jnp.dtype,
        sharding: jax.sharding.Sharding,
        *,
        parallelism: int,
        capacity: int,
        timeout_s: float,
    ) -> None:
        self.shape = shape
        self.dtype = jnp.dtype(dtype)
        self._timeout_s = timeout_s
        self._block_shape = encoder_pool_block_shape(shape)
        self._buffer = jnp.zeros(
            (capacity, *self._block_shape),
            dtype=dtype,
            device=_pool_sharding(sharding),
        )
        jax.block_until_ready(self._buffer)
        self._transfer = RaidenTransferWrapper(host, 0, parallelism=parallelism)
        self._transfer.start(
            [self._buffer],
            max_blocks=1,
            num_slots=capacity,
            timeout_s=timeout_s,
        )
        self._condition = threading.Condition()
        self._free = list(range(capacity - 1, -1, -1))
        self._active: dict[str, int] = {}
        self._abandoned: set[str] = set()
        self._deferred_releases: dict[str, tuple[jax.Array, ...]] = {}
        self._received_ns: dict[str, int] = {}
        self._received: set[str] = set()
        self._failed: set[str] = set()
        self._closed = False

    def start(
        self,
        transfer_id: str,
        transfer_uuid: int,
        remote_endpoints: list[dict[str, Any]],
        remote_block_ids: list[int],
    ) -> RaidenReceiveSession:
        deadline = time.monotonic() + self._timeout_s
        with self._condition:
            while not self._free and not self._closed:
                self._reap_deferred_locked()
                self._reap_abandoned_locked()
                if self._free:
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("timed out waiting for a Raiden embedding buffer")
                self._condition.wait(min(remaining, 0.01))
            if self._closed:
                raise RuntimeError("Raiden receiver is closed")
            if transfer_id in self._active:
                raise ValueError(f"duplicate Raiden transfer_id: {transfer_id}")
            lane_id = self._free.pop()
            self._active[transfer_id] = lane_id
            try:
                self._transfer.start_read(
                    transfer_id,
                    transfer_uuid,
                    remote_endpoints,
                    remote_block_ids,
                    [lane_id],
                )
            except Exception:
                self._release_locked(transfer_id)
                raise
        return RaidenReceiveSession(
            transfer_id=transfer_id,
            lane_id=lane_id,
            pool=self,
        )

    def poll(
        self,
        transfer_id: str,
        lane_id: int,
        *,
        refresh_backend: bool = True,
    ) -> tuple[PooledEmbedding, dict[str, int]] | None:
        with self._condition:
            if self._active.get(transfer_id) != lane_id:
                raise RuntimeError(f"Raiden embedding lane changed: {transfer_id}")
            if refresh_backend:
                self._drain_stats_locked()
            if transfer_id in self._failed:
                self._release_locked(transfer_id)
                raise RuntimeError(f"Raiden embedding transfer failed: {transfer_id}")
            if transfer_id not in self._received:
                return None
            materialize_ns = time.time_ns()
            lease = RaidenReceiveLease(self, transfer_id, lane_id)
            embedding = PooledEmbedding(
                self._buffer,
                lane_id,
                self._block_shape,
                self.shape,
                lease,
            )
            timing = {
                "receive_transfer_done_ns": self._received_ns.get(
                    transfer_id,
                    materialize_ns,
                ),
                "receive_materialize_start_ns": materialize_ns,
                "receive_materialize_done_ns": materialize_ns,
            }
            return embedding, timing

    def progress(self) -> None:
        """Refresh shared transfer state once for all receive sessions."""
        with self._condition:
            if self._closed:
                return
            self._drain_stats_locked()
            self._reap_deferred_locked()
            self._release_abandoned_locked()

    def release_after(
        self,
        transfer_id: str,
        lane_id: int,
        dependency: Any,
    ) -> None:
        leaves = tuple(
            leaf for leaf in jax.tree_util.tree_leaves(dependency) if isinstance(leaf, jax.Array)
        )
        with self._condition:
            if self._active.get(transfer_id) != lane_id:
                return
            if not leaves or all(leaf.is_ready() for leaf in leaves):
                self._release_locked(transfer_id)
            else:
                self._deferred_releases[transfer_id] = leaves

    def release(self, transfer_id: str, lane_id: int) -> None:
        with self._condition:
            if self._active.get(transfer_id) == lane_id:
                self._release_locked(transfer_id)

    def abandon(self, transfer_id: str) -> None:
        with self._condition:
            if transfer_id not in self._active:
                return
            self._abandoned.add(transfer_id)
            try:
                self._reap_abandoned_locked()
            except Exception:
                logger.exception("Raiden receiver poll failed while abandoning %s", transfer_id)

    def _reap_abandoned_locked(self) -> None:
        self._drain_stats_locked()
        self._release_abandoned_locked()

    def _release_abandoned_locked(self) -> None:
        for transfer_id in list(self._abandoned):
            if transfer_id not in self._active:
                continue
            if transfer_id in self._received or transfer_id in self._failed:
                self._release_locked(transfer_id)

    def _reap_deferred_locked(self) -> None:
        for transfer_id, dependencies in list(self._deferred_releases.items()):
            if all(dependency.is_ready() for dependency in dependencies):
                self._release_locked(transfer_id)

    def _drain_stats_locked(self) -> None:
        _, received, failed = self._transfer.poll_stats()
        received_ns = time.time_ns()
        for transfer_id in received:
            self._received_ns.setdefault(transfer_id, received_ns)
        self._received.update(received)
        self._failed.update(failed)

    def _release_locked(self, transfer_id: str) -> None:
        lane_id = self._active.pop(transfer_id, None)
        self._abandoned.discard(transfer_id)
        self._deferred_releases.pop(transfer_id, None)
        self._received_ns.pop(transfer_id, None)
        self._received.discard(transfer_id)
        self._failed.discard(transfer_id)
        if lane_id is not None:
            self._free.append(lane_id)
            self._condition.notify()

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()
