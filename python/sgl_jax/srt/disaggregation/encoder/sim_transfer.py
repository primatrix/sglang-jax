from __future__ import annotations

import asyncio
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.disaggregation.encoder.client import DeferredReceiveSession
from sgl_jax.srt.disaggregation.encoder.embedding_data import EmbeddingData
from sgl_jax.srt.disaggregation.encoder.transfer_layout import encoder_transfer_nbytes

logger = logging.getLogger(__name__)


def _transfer_duration_ns(shape, dtype, ms_per_mb: float, rtt_ms: float) -> int:
    payload_mib = encoder_transfer_nbytes(shape, dtype) / (2**20)
    return max(0, int((float(rtt_ms) + float(ms_per_mb) * payload_mib) * 1_000_000))


class _SimSendPool:
    """Shape-specific sender pool with bounded slots and transfer channels."""

    def __init__(
        self,
        sample: jax.Array,
        *,
        capacity: int,
        parallelism: int,
        timeout_s: float,
        ms_per_mb: float,
        rtt_ms: float,
    ) -> None:
        self.shape = tuple(int(dim) for dim in sample.shape)
        self.dtype = sample.dtype
        self.sharding = sample.sharding
        self._timeout_s = float(timeout_s)
        self._duration_ns = _transfer_duration_ns(
            self.shape,
            self.dtype,
            ms_per_mb,
            rtt_ms,
        )
        self._channel_ready_ns = [0] * max(1, int(parallelism))
        self._condition = threading.Condition()
        self._free = list(range(max(1, int(capacity)) - 1, -1, -1))
        self._active: dict[str, tuple[int, int]] = {}
        self._closed = False

    def matches(self, value: jax.Array) -> bool:
        return (
            tuple(value.shape) == self.shape
            and value.dtype == self.dtype
            and value.sharding == self.sharding
        )

    def reserve_sync(self, transfer_id: str) -> int:
        deadline = time.monotonic() + self._timeout_s
        with self._condition:
            if transfer_id in self._active:
                raise ValueError(f"duplicate simulated transfer_id: {transfer_id}")
            while not self._free and not self._closed:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("timed out waiting for a simulated encoder pool slot")
                self._condition.wait(remaining)
            if self._closed:
                raise RuntimeError("simulated encoder pool is closed")

            slot = self._free.pop()
            self._active[transfer_id] = (slot, 0)
            return slot

    async def reserve(self, transfer_id: str) -> int:
        try:
            return await asyncio.to_thread(self.reserve_sync, transfer_id)
        except TimeoutError:
            # Keep the async API's historical error message.
            if not self._closed:
                raise TimeoutError("timed out waiting for a simulated encoder pool slot") from None
            raise

    def schedule(self, transfer_id: str, slot: int) -> int:
        with self._condition:
            if self._active.get(transfer_id, (None, None))[0] != slot:
                raise RuntimeError(f"simulated encoder slot changed: {transfer_id}")
            channel = min(
                range(len(self._channel_ready_ns)),
                key=self._channel_ready_ns.__getitem__,
            )
            ready_ns = max(time.monotonic_ns(), self._channel_ready_ns[channel]) + self._duration_ns
            self._channel_ready_ns[channel] = ready_ns
            self._active[transfer_id] = (slot, ready_ns)
            return ready_ns

    def poll(self) -> list[str]:
        with self._condition:
            now_ns = time.monotonic_ns()
            completed = [
                transfer_id
                for transfer_id, (_, ready_ns) in self._active.items()
                if ready_ns and ready_ns <= now_ns
            ]
            for transfer_id in completed:
                self._release_locked(transfer_id)
            return completed

    def release(self, transfer_id: str) -> None:
        with self._condition:
            self._release_locked(transfer_id)

    def _release_locked(self, transfer_id: str) -> None:
        active = self._active.pop(transfer_id, None)
        if active is not None:
            self._free.append(active[0])
            self._condition.notify()

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()


class SimEncoderServerTransfer:
    """Resource-aware stand-in for ``RaidenEncoderServerTransfer``.

    No embedding is sent over the wire. The model preserves Raiden's
    shape-specific pool capacity, channel contention, padded payload size, and
    asynchronous sender completion lifecycle.
    """

    def __init__(
        self,
        *,
        setup_ms: float = 0.0,
        parallelism: int = 1,
        pool_size: int = 32,
        timeout_s: float = 300.0,
        ms_per_mb: float = 0.0,
        rtt_ms: float = 0.0,
        poll_interval_s: float = 0.001,
        log_inflight: bool = False,
    ) -> None:
        self._setup_ms = max(0.0, float(setup_ms))
        self._parallelism = max(1, int(parallelism))
        self._pool_size = max(1, int(pool_size))
        self._timeout_s = float(timeout_s)
        self._ms_per_mb = float(ms_per_mb)
        self._rtt_ms = float(rtt_ms)
        self._poll_interval_s = float(poll_interval_s)
        self._log_inflight = bool(log_inflight)
        self._pools: list[_SimSendPool] = []
        self._active: dict[str, _SimSendPool] = {}
        self._pending: dict[str, _SimSendPool | None] = {}
        self._closed = False
        self._lock = threading.Lock()

    def stage_sync(self, transfer_id: str, embedding: jax.Array) -> Any:
        if embedding.ndim != 2 or embedding.shape[0] <= 0:
            raise ValueError("Sim embedding must be a non-empty matrix")

        with self._lock:
            if self._closed:
                raise RuntimeError("simulated encoder transfer is closed")
            if transfer_id in self._active or transfer_id in self._pending:
                raise ValueError(f"duplicate simulated transfer_id: {transfer_id}")
            self._pending[transfer_id] = None
            pool = next(
                (pool for pool in self._pools if pool.matches(embedding)),
                None,
            )
            if pool is None:
                pool = _SimSendPool(
                    embedding,
                    capacity=self._pool_size,
                    parallelism=self._parallelism,
                    timeout_s=self._timeout_s,
                    ms_per_mb=self._ms_per_mb,
                    rtt_ms=self._rtt_ms,
                )
                self._pools.append(pool)

        try:
            slot = pool.reserve_sync(transfer_id)
        except BaseException:
            with self._lock:
                self._pending.pop(transfer_id, None)
            if pool is not None:
                pool.release(transfer_id)
            raise

        with self._lock:
            self._pending[transfer_id] = pool
        return transfer_id, pool, slot

    async def stage(self, transfer_id: str, embedding: jax.Array) -> Any:
        return await asyncio.to_thread(self.stage_sync, transfer_id, embedding)

    def publish_sync(self, staged_transfer: Any) -> dict[str, Any]:
        transfer_id, pool, slot = staged_transfer
        try:
            if self._setup_ms:
                time.sleep(self._setup_ms / 1000.0)
            pool.schedule(transfer_id, slot)
        except BaseException:
            with self._lock:
                pending = self._pending.pop(transfer_id, None)
            if pending is not None:
                pool.release(transfer_id)
            raise

        with self._lock:
            self._pending.pop(transfer_id, None)
            self._active[transfer_id] = pool
        self._log_inflight_event("start", transfer_id)
        return {"transfer_id": transfer_id}

    async def publish(self, staged_transfer: Any) -> dict[str, Any]:
        return await asyncio.to_thread(self.publish_sync, staged_transfer)

    def poll_completed(self) -> None:
        with self._lock:
            pools = list(self._pools)
        for pool in pools:
            for transfer_id in pool.poll():
                self._discard_active(transfer_id, event="sent")

    async def release_completed(self) -> None:
        while True:
            self.poll_completed()
            await asyncio.sleep(self._poll_interval_s)

    def release(self, transfer_id: str) -> None:
        with self._lock:
            pool = self._active.pop(transfer_id, None)
            if pool is None:
                pool = self._pending.pop(transfer_id, None)
        if pool is not None:
            pool.release(transfer_id)
            self._log_inflight_event("release", transfer_id)

    def close(self) -> None:
        with self._lock:
            self._closed = True
            pools = list(self._pools)
            active = list(self._active)
            pending = list(self._pending.values())
            self._active.clear()
            self._pending.clear()
        for pool in pools:
            pool.close()
        for transfer_id in active:
            self._log_inflight_event("close", transfer_id)
        for pool in pending:
            if pool is not None:
                pool.close()

    def _discard_active(self, transfer_id: str, *, event: str) -> None:
        with self._lock:
            removed = self._active.pop(transfer_id, None)
        if removed is not None:
            self._log_inflight_event(event, transfer_id)

    def _log_inflight_event(self, event: str, transfer_id: str) -> None:
        if not self._log_inflight:
            return
        with self._lock:
            inflight = len(self._active)
        logger.info(
            "ENCODER-RAIDEN-INFLIGHT time_ns=%d event=%s transfer_id=%s "
            "group_size=1 inflight_groups=%d inflight_requests=%d",
            time.time_ns(),
            event,
            transfer_id,
            inflight,
            inflight,
        )


@dataclass(slots=True)
class SimReceiveSession:
    transfer_id: str
    buffer: jax.Array
    ready_at_ns: int
    lane_id: int
    pool: _SimReceivePool
    _done: bool = False

    def poll(self) -> jax.Array | None:
        if self._done or time.monotonic_ns() < self.ready_at_ns:
            return None
        self.pool.complete(self.transfer_id, self.lane_id)
        self._done = True
        return self.buffer

    def close(self) -> None:
        if not self._done:
            self.pool.abandon(self.transfer_id)


class _SimReceivePool:
    """Reusable zero embedding with bounded receiver slots and transfer channels."""

    def __init__(
        self,
        shape: tuple[int, int],
        dtype: jnp.dtype,
        sharding: jax.sharding.Sharding,
        *,
        capacity: int,
        parallelism: int,
        timeout_s: float,
        ms_per_mb: float,
        rtt_ms: float,
    ) -> None:
        self._timeout_s = float(timeout_s)
        self._duration_ns = _transfer_duration_ns(shape, dtype, ms_per_mb, rtt_ms)
        self._buffer = jax.device_put(np.zeros(shape, dtype=dtype), sharding)
        jax.block_until_ready(self._buffer)
        self._channel_ready_ns = [0] * max(1, int(parallelism))
        self._condition = threading.Condition()
        self._free = list(range(max(1, int(capacity)) - 1, -1, -1))
        self._active: dict[str, tuple[int, int]] = {}
        self._abandoned: set[str] = set()
        self._closed = False

    def start(self, transfer_id: str) -> SimReceiveSession:
        deadline = time.monotonic() + self._timeout_s
        with self._condition:
            while not self._free and not self._closed:
                self._reap_abandoned_locked()
                if self._free:
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("timed out waiting for a simulated embedding buffer")
                self._condition.wait(min(remaining, 0.01))
            if self._closed:
                raise RuntimeError("simulated receiver is closed")
            if transfer_id in self._active:
                raise ValueError(f"duplicate simulated transfer_id: {transfer_id}")

            lane_id = self._free.pop()
            channel = min(
                range(len(self._channel_ready_ns)),
                key=self._channel_ready_ns.__getitem__,
            )
            ready_at_ns = (
                max(time.monotonic_ns(), self._channel_ready_ns[channel]) + self._duration_ns
            )
            self._channel_ready_ns[channel] = ready_at_ns
            self._active[transfer_id] = (lane_id, ready_at_ns)
        return SimReceiveSession(
            transfer_id=transfer_id,
            buffer=self._buffer,
            ready_at_ns=ready_at_ns,
            lane_id=lane_id,
            pool=self,
        )

    def complete(self, transfer_id: str, lane_id: int) -> None:
        with self._condition:
            if self._active.get(transfer_id, (None, None))[0] != lane_id:
                raise RuntimeError(f"simulated embedding lane changed: {transfer_id}")
            self._release_locked(transfer_id)

    def abandon(self, transfer_id: str) -> None:
        with self._condition:
            if transfer_id not in self._active:
                return
            self._abandoned.add(transfer_id)
            self._reap_abandoned_locked()

    def _reap_abandoned_locked(self) -> None:
        now_ns = time.monotonic_ns()
        for transfer_id in list(self._abandoned):
            active = self._active.get(transfer_id)
            if active is not None and active[1] <= now_ns:
                self._release_locked(transfer_id)

    def _release_locked(self, transfer_id: str) -> None:
        active = self._active.pop(transfer_id, None)
        self._abandoned.discard(transfer_id)
        if active is not None:
            self._free.append(active[0])
            self._condition.notify()

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()


class SimReceiverBackend:
    """Rebuild zero embeddings while preserving receiver resource limits."""

    def __init__(
        self,
        sharding: jax.sharding.Sharding,
        ms_per_mb: float,
        rtt_ms: float = 0.0,
        *,
        parallelism: int = 1,
        pool_size: int = 32,
        transfer_timeout_s: float = 300.0,
    ) -> None:
        self._sharding = sharding
        self._ms_per_mb = float(ms_per_mb)
        self._rtt_ms = float(rtt_ms)
        self._parallelism = max(1, int(parallelism))
        self._pool_size = max(1, int(pool_size))
        self._transfer_timeout_s = float(transfer_timeout_s)
        self._pools: dict[tuple[tuple[int, int], jnp.dtype], _SimReceivePool] = {}
        self._pool_lock = threading.Lock()
        self._closed = False
        self._executor = ThreadPoolExecutor(max_workers=1)

    def start(self, data: EmbeddingData) -> DeferredReceiveSession:
        return DeferredReceiveSession(self._executor.submit(self._start, data))

    def _start(self, data: EmbeddingData) -> SimReceiveSession:
        if data.shape is None or data.dtype is None:
            raise ValueError("embedding shape and dtype are required")
        shape = tuple(int(dim) for dim in data.shape)
        if len(shape) != 2 or shape[0] <= 0 or shape[1] <= 0:
            raise ValueError("Sim embedding must be a non-empty matrix")
        transfer_id = getattr(data, "transfer_id", None)
        if not transfer_id:
            raise ValueError("simulated transfer_id is required")

        dtype = jnp.dtype(data.dtype)
        key = (shape, dtype)
        with self._pool_lock:
            if self._closed:
                raise RuntimeError("simulated receiver is closed")
            pool = self._pools.get(key)
            if pool is None:
                pool = _SimReceivePool(
                    shape,
                    dtype,
                    self._sharding,
                    capacity=self._pool_size,
                    parallelism=self._parallelism,
                    timeout_s=self._transfer_timeout_s,
                    ms_per_mb=self._ms_per_mb,
                    rtt_ms=self._rtt_ms,
                )
                self._pools[key] = pool
        return pool.start(str(transfer_id))

    def close(self) -> None:
        with self._pool_lock:
            self._closed = True
            pools = list(self._pools.values())
        for pool in pools:
            pool.close()
        self._executor.shutdown(cancel_futures=True)
