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


class RaidenSendPool:
    """Reusable source buffer with bounded, request-sized slots."""

    def __init__(
        self,
        sample: jax.Array,
        *,
        capacity: int,
    ) -> None:
        self.shape = tuple(int(dim) for dim in sample.shape)
        self.dtype = sample.dtype
        self.sharding = sample.sharding
        pool_sharding = _pool_sharding(sample.sharding)
        self._block_shape = encoder_pool_block_shape(self.shape)
        self._buffer = jnp.zeros(
            (capacity, *self._block_shape),
            dtype=self.dtype,
            device=pool_sharding,
        )
        jax.block_until_ready(self._buffer)
        self._copy = _compile_donated_copy(self._buffer, sample)

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
        self._buffer, ready = self._copy(
            self._buffer,
            value,
            jnp.asarray(slot, dtype=jnp.int32),
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

    def poll(self, *, refresh_backend: bool = True) -> jax.Array | None:
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
        self._sharding = sharding
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
        self._materializing: dict[str, jax.Array] = {}
        self._received_ns: dict[str, int] = {}
        self._materialize_start_ns: dict[str, int] = {}
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
    ) -> tuple[jax.Array, dict[str, int]] | None:
        with self._condition:
            if self._active.get(transfer_id) != lane_id:
                raise RuntimeError(f"Raiden embedding lane changed: {transfer_id}")
            if refresh_backend:
                self._drain_stats_locked()
            if transfer_id in self._failed:
                self._release_locked(transfer_id)
                raise RuntimeError(f"Raiden embedding transfer failed: {transfer_id}")
            embedding = self._materializing.get(transfer_id)
            try:
                if embedding is None:
                    if transfer_id not in self._received:
                        return None

                    # Raiden writes outside JAX's dependency graph. Submit a
                    # non-aliasing copy, then let later poll calls observe its
                    # completion without blocking the scheduler event loop.
                    block = self._buffer[lane_id].reshape(self._block_shape[0], -1)
                    embedding = jax.device_put(
                        block[: self.shape[0], : self.shape[1]],
                        self._sharding,
                        may_alias=False,
                    )
                    self._materializing[transfer_id] = embedding
                    self._materialize_start_ns[transfer_id] = time.time_ns()

                if not embedding.is_ready():
                    return None
            except Exception:
                self._release_locked(transfer_id)
                raise

            # The source slot cannot be reused until the copy is complete.
            timing = {
                "receive_transfer_done_ns": self._received_ns.get(
                    transfer_id,
                    self._materialize_start_ns[transfer_id],
                ),
                "receive_materialize_start_ns": self._materialize_start_ns[transfer_id],
                "receive_materialize_done_ns": time.time_ns(),
            }
            self._release_locked(transfer_id)
            return embedding, timing

    def progress(self) -> None:
        """Refresh shared transfer state once for all receive sessions."""
        with self._condition:
            if self._closed:
                return
            self._drain_stats_locked()
            self._release_abandoned_locked()

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
            embedding = self._materializing.get(transfer_id)
            if embedding is not None:
                if embedding.is_ready():
                    self._release_locked(transfer_id)
                continue
            if transfer_id in self._received or transfer_id in self._failed:
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
        self._materializing.pop(transfer_id, None)
        self._received_ns.pop(transfer_id, None)
        self._materialize_start_ns.pop(transfer_id, None)
        self._received.discard(transfer_id)
        self._failed.discard(transfer_id)
        if lane_id is not None:
            self._free.append(lane_id)
            self._condition.notify()

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()
