from __future__ import annotations

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
    """Sender timing model; slot ownership lives in the transfer backend."""

    def __init__(
        self,
        sample: jax.Array,
        *,
        parallelism: int,
        ms_per_mb: float,
        rtt_ms: float,
    ) -> None:
        self.shape = tuple(int(dim) for dim in sample.shape)
        self.dtype = sample.dtype
        self.sharding = sample.sharding
        self._duration_ns = _transfer_duration_ns(
            self.shape,
            self.dtype,
            ms_per_mb,
            rtt_ms,
        )
        self._channel_ready_ns = [0] * max(1, int(parallelism))
        self._active: dict[str, tuple[int, int]] = {}
        self._lock = threading.Lock()

    def matches(self, value: jax.Array) -> bool:
        return (
            tuple(value.shape) == self.shape
            and value.dtype == self.dtype
            and value.sharding == self.sharding
        )

    def schedule(self, transfer_id: str, slot: int) -> int:
        with self._lock:
            channel = min(
                range(len(self._channel_ready_ns)),
                key=self._channel_ready_ns.__getitem__,
            )
            ready_ns = max(time.monotonic_ns(), self._channel_ready_ns[channel]) + self._duration_ns
            self._channel_ready_ns[channel] = ready_ns
            self._active[transfer_id] = (slot, ready_ns)
            return ready_ns

    def poll(self) -> list[str]:
        with self._lock:
            now_ns = time.monotonic_ns()
            completed = [
                transfer_id
                for transfer_id, (_, ready_ns) in self._active.items()
                if ready_ns <= now_ns
            ]
            for transfer_id in completed:
                self._active.pop(transfer_id, None)
            return completed

    def close(self) -> None:
        with self._lock:
            self._active.clear()


@dataclass(frozen=True, slots=True)
class _SimReservation:
    transfer_id: str
    slot: int
    reserve_start_ns: int
    reserve_done_ns: int


@dataclass(frozen=True, slots=True)
class _SimStagedTransfer:
    reservation: _SimReservation
    pool_ready_ns: int
    copy_submit_ns: int


class SimEncoderServerTransfer:
    """Resource-aware stand-in for ``RaidenEncoderServerTransfer``.

    No embedding is sent over the wire. The model preserves Raiden's
    single-pool capacity, channel contention, padded payload size, and
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
        self._poll_interval_s = max(0.0001, float(poll_interval_s))
        self._log_inflight = bool(log_inflight)
        self._pool: _SimSendPool | None = None
        self._free = list(range(self._pool_size - 1, -1, -1))
        self._slots: dict[str, int] = {}
        self._active: set[str] = set()
        self._pending: set[str] = set()
        self._closed = False
        self._lock = threading.Lock()

    def reserve_batch_sync(self, transfer_ids: list[str]) -> list[_SimReservation]:
        transfer_ids = list(transfer_ids)
        if not transfer_ids:
            return []
        if len(transfer_ids) > self._pool_size:
            raise ValueError("encoder batch exceeds simulated pool capacity")
        if len(set(transfer_ids)) != len(transfer_ids):
            raise ValueError("duplicate simulated transfer_id in encoder batch")

        reserve_start_ns = time.time_ns()
        deadline = time.monotonic() + self._timeout_s
        while True:
            with self._lock:
                if self._closed:
                    raise RuntimeError("simulated encoder transfer is closed")
                duplicate = next(
                    (transfer_id for transfer_id in transfer_ids if transfer_id in self._slots),
                    None,
                )
                if duplicate is not None:
                    raise ValueError(f"duplicate simulated transfer_id: {duplicate}")
                if len(self._free) >= len(transfer_ids):
                    slots = [self._free.pop() for _ in transfer_ids]
                    self._slots.update(zip(transfer_ids, slots))
                    self._pending.update(transfer_ids)
                    reserve_done_ns = time.time_ns()
                    return [
                        _SimReservation(
                            transfer_id,
                            slot,
                            reserve_start_ns,
                            reserve_done_ns,
                        )
                        for transfer_id, slot in zip(transfer_ids, slots)
                    ]

            self._reap_completed()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("timed out waiting for simulated encoder pool slots")
            time.sleep(min(remaining, self._poll_interval_s))

    def stage_batch_sync(
        self,
        reservations: list[_SimReservation],
        embeddings: list[jax.Array],
    ) -> list[_SimStagedTransfer]:
        if len(reservations) != len(embeddings):
            raise ValueError("simulated reservation and embedding counts differ")
        if not reservations:
            return []
        if any(embedding.ndim != 2 or embedding.shape[0] <= 0 for embedding in embeddings):
            raise ValueError("Sim embedding must be a non-empty matrix")

        with self._lock:
            if self._closed:
                raise RuntimeError("simulated encoder transfer is closed")
            pool = self._pool
            if pool is None:
                pool = _SimSendPool(
                    embeddings[0],
                    parallelism=self._parallelism,
                    ms_per_mb=self._ms_per_mb,
                    rtt_ms=self._rtt_ms,
                )
                self._pool = pool

        for embedding in embeddings:
            if not pool.matches(embedding):
                raise ValueError(
                    "simulated encoder pool embedding mismatch: "
                    f"expected shape={pool.shape}, dtype={pool.dtype}, "
                    f"sharding={pool.sharding}; got shape={tuple(embedding.shape)}, "
                    f"dtype={embedding.dtype}, sharding={embedding.sharding}"
                )
        pool_ready_ns = time.time_ns()
        return [
            _SimStagedTransfer(
                reservation,
                pool_ready_ns,
                time.time_ns(),
            )
            for reservation in reservations
        ]

    def publish_sync(self, staged_transfer: Any) -> dict[str, Any]:
        reservation = staged_transfer.reservation
        transfer_id = reservation.transfer_id
        slot = reservation.slot
        copy_done_ns = time.time_ns()
        register_start_ns = copy_done_ns
        pool = self._pool
        if pool is None:
            raise RuntimeError("simulated encoder pool is not initialized")
        try:
            with self._lock:
                if transfer_id not in self._pending or self._slots.get(transfer_id) != slot:
                    raise RuntimeError(f"simulated reservation was cancelled: {transfer_id}")
                self._pending.remove(transfer_id)
                self._active.add(transfer_id)
            if self._setup_ms:
                time.sleep(self._setup_ms / 1000.0)
            pool.schedule(transfer_id, slot)
            register_done_ns = time.time_ns()
        except BaseException:
            with self._lock:
                self._release_locked(transfer_id)
            raise

        self._log_inflight_event("start", transfer_id)
        return {
            "transfer_id": transfer_id,
            "transfer_reserve_start_ns": reservation.reserve_start_ns,
            "transfer_pool_ready_ns": staged_transfer.pool_ready_ns,
            "transfer_reserve_done_ns": reservation.reserve_done_ns,
            "transfer_copy_submit_ns": staged_transfer.copy_submit_ns,
            "transfer_copy_done_ns": copy_done_ns,
            "transfer_register_start_ns": register_start_ns,
            "transfer_register_done_ns": register_done_ns,
            "transfer_publish_ready_ns": max(copy_done_ns, register_done_ns),
        }

    def _reap_completed(self) -> None:
        with self._lock:
            pool = self._pool
        if pool is None:
            return
        for transfer_id in pool.poll():
            self._discard_active(transfer_id, event="sent")

    def cancel_batch(self, reservations: list[_SimReservation]) -> None:
        with self._lock:
            for reservation in reservations:
                if reservation.transfer_id in self._pending:
                    self._release_locked(reservation.transfer_id)

    def release(self, transfer_id: str) -> None:
        event = None
        with self._lock:
            if transfer_id in self._pending:
                self._release_locked(transfer_id)
                event = "release"
            elif transfer_id in self._active:
                event = "defer"
        if event is not None:
            self._log_inflight_event(event, transfer_id)

    def close(self) -> None:
        with self._lock:
            self._closed = True
            pool = self._pool
            active = list(self._active)
            self._free.clear()
            self._slots.clear()
            self._active.clear()
            self._pending.clear()
        if pool is not None:
            pool.close()
        for transfer_id in active:
            self._log_inflight_event("close", transfer_id)

    def _discard_active(self, transfer_id: str, *, event: str) -> None:
        with self._lock:
            removed = transfer_id in self._active
            if removed:
                self._release_locked(transfer_id)
        if removed:
            self._log_inflight_event(event, transfer_id)

    def _release_locked(self, transfer_id: str) -> None:
        slot = self._slots.pop(transfer_id, None)
        self._pending.discard(transfer_id)
        self._active.discard(transfer_id)
        if slot is not None:
            self._free.append(slot)

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

    def poll(self, *, refresh_backend: bool = True) -> jax.Array | None:
        del refresh_backend
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
        self.shape = shape
        self.dtype = jnp.dtype(dtype)
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
        self._pool: _SimReceivePool | None = None
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
        with self._pool_lock:
            if self._closed:
                raise RuntimeError("simulated receiver is closed")
            pool = self._pool
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
                self._pool = pool
            elif shape != pool.shape or dtype != pool.dtype:
                raise ValueError(
                    "simulated receiver pool embedding mismatch: "
                    f"expected shape={pool.shape}, dtype={pool.dtype}; "
                    f"got shape={shape}, dtype={dtype}"
                )
        return pool.start(str(transfer_id))

    def close(self) -> None:
        with self._pool_lock:
            self._closed = True
            pool = self._pool
        if pool is not None:
            pool.close()
        self._executor.shutdown(cancel_futures=True)
