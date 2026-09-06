from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.disaggregation.encoder.client import DeferredReceiveSession
from sgl_jax.srt.disaggregation.encoder.embedding_data import EmbeddingData
from sgl_jax.srt.disaggregation.encoder.transfer_layout import (
    encoder_pool_block_shape,
    encoder_transfer_nbytes,
)
from sgl_jax.srt.multimodal.in_model.embedding_view import PooledEmbedding


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
    ) -> None:
        self._setup_ms = max(0.0, float(setup_ms))
        self._parallelism = max(1, int(parallelism))
        self._pool_size = max(1, int(pool_size))
        self._timeout_s = float(timeout_s)
        self._ms_per_mb = float(ms_per_mb)
        self._rtt_ms = float(rtt_ms)
        self._poll_interval_s = max(0.0001, float(poll_interval_s))
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
                    return [
                        _SimReservation(
                            transfer_id,
                            slot,
                        )
                        for transfer_id, slot in zip(transfer_ids, slots)
                    ]

            self._reap_completed()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("timed out waiting for simulated encoder pool slots")
            time.sleep(min(remaining, self._poll_interval_s))

    def precompile_packed_batches(self, specs) -> None:
        pass

    def stage_packed_batch_sync(
        self,
        reservations: list[_SimReservation],
        packed: jax.Array,
        token_counts: tuple[int, ...],
    ) -> list[_SimReservation]:
        if len(reservations) != len(token_counts):
            raise ValueError("simulated reservation and embedding counts differ")
        if not reservations:
            return []
        if packed.ndim != 2 or any(count <= 0 for count in token_counts):
            raise ValueError("Sim packed output must contain non-empty matrices")
        if any(count != token_counts[0] for count in token_counts):
            raise ValueError("simulated source pool requires one embedding shape")
        if sum(token_counts) > packed.shape[0]:
            raise ValueError("incomplete simulated packed output")
        sample = jax.ShapeDtypeStruct(
            (token_counts[0], packed.shape[1]), packed.dtype, sharding=packed.sharding
        )

        with self._lock:
            if self._closed:
                raise RuntimeError("simulated encoder transfer is closed")
            pool = self._pool
            if pool is None:
                pool = _SimSendPool(
                    sample,
                    parallelism=self._parallelism,
                    ms_per_mb=self._ms_per_mb,
                    rtt_ms=self._rtt_ms,
                )
                self._pool = pool

        if not pool.matches(sample):
            raise ValueError(
                "simulated encoder pool embedding mismatch: "
                f"expected shape={pool.shape}, dtype={pool.dtype}, "
                f"sharding={pool.sharding}; got shape={sample.shape}, "
                f"dtype={sample.dtype}, sharding={sample.sharding}"
            )
        return reservations

    def publish_batch_sync(self, reservations: list[_SimReservation]) -> list[dict[str, Any]]:
        return [self._publish(reservation) for reservation in reservations]

    def _publish(self, reservation: _SimReservation) -> dict[str, Any]:
        transfer_id = reservation.transfer_id
        slot = reservation.slot
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
        except BaseException:
            with self._lock:
                self._release_locked(transfer_id)
            raise

        return {"transfer_id": transfer_id}

    def _reap_completed(self) -> None:
        with self._lock:
            pool = self._pool
        if pool is None:
            return
        completed = pool.poll()
        with self._lock:
            for transfer_id in completed:
                self._release_locked(transfer_id)

    def cancel_batch(self, reservations: list[_SimReservation]) -> None:
        with self._lock:
            for reservation in reservations:
                if reservation.transfer_id in self._pending:
                    self._release_locked(reservation.transfer_id)

    def release(self, transfer_id: str) -> None:
        with self._lock:
            if transfer_id in self._pending:
                self._release_locked(transfer_id)

    def close(self) -> None:
        with self._lock:
            self._closed = True
            pool = self._pool
            self._free.clear()
            self._slots.clear()
            self._active.clear()
            self._pending.clear()
        if pool is not None:
            pool.close()

    def _release_locked(self, transfer_id: str) -> None:
        slot = self._slots.pop(transfer_id, None)
        self._pending.discard(transfer_id)
        self._active.discard(transfer_id)
        if slot is not None:
            self._free.append(slot)


@dataclass(slots=True)
class SimReceiveSession:
    transfer_id: str
    buffer: jax.Array
    ready_at_ns: int
    slot: int
    pool: _SimReceivePool
    _done: bool = False

    def poll(self, *, refresh_backend: bool = True) -> PooledEmbedding | None:
        del refresh_backend
        if self._done or time.monotonic_ns() < self.ready_at_ns:
            return None
        self._done = True
        return PooledEmbedding(
            self.buffer,
            0,
            self.pool.block_shape,
            self.pool.shape,
            _SimReceiveLease(self.pool, self.transfer_id),
        )

    def close(self) -> None:
        if not self._done:
            self.pool.abandon(self.transfer_id)


@dataclass(slots=True)
class _SimReceiveLease:
    pool: _SimReceivePool
    transfer_id: str
    _released: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def release_after(self, dependency: Any) -> None:
        with self._lock:
            if self._released:
                return
            self._released = True
        self.pool.release_after(self.transfer_id, dependency)

    def release(self) -> None:
        self.release_after(())


@dataclass(slots=True)
class _SimReceiveState:
    slot: int
    ready_at_ns: int
    readers: tuple[jax.Array, ...] | None = None


class _SimReceivePool:
    """Model channel deadlines and leased slots using one immutable zero buffer."""

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
        self.block_shape = encoder_pool_block_shape(shape)
        self._buffer = jax.device_put(np.zeros((1, *self.block_shape), dtype=dtype), sharding)
        jax.block_until_ready(self._buffer)
        self._channel_ready_ns = [0] * max(1, int(parallelism))
        self._condition = threading.Condition()
        self._free = list(range(max(1, int(capacity)) - 1, -1, -1))
        self._active: dict[str, _SimReceiveState] = {}
        self._closed = False

    def start(self, transfer_id: str) -> SimReceiveSession:
        deadline = time.monotonic() + self._timeout_s
        with self._condition:
            while not self._free and not self._closed:
                self._reap_released_locked()
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

            slot = self._free.pop()
            channel = min(
                range(len(self._channel_ready_ns)),
                key=self._channel_ready_ns.__getitem__,
            )
            ready_at_ns = (
                max(time.monotonic_ns(), self._channel_ready_ns[channel]) + self._duration_ns
            )
            self._channel_ready_ns[channel] = ready_at_ns
            self._active[transfer_id] = _SimReceiveState(slot, ready_at_ns)
            if jax.profiler.TraceAnnotation.is_enabled():
                clock_offset_ns = time.time_ns() - time.monotonic_ns()
                with jax.profiler.TraceAnnotation(
                    "sim_transfer",
                    transfer_id=transfer_id,
                    channel=channel,
                    start_ns=ready_at_ns - self._duration_ns + clock_offset_ns,
                    end_ns=ready_at_ns + clock_offset_ns,
                ):
                    pass
        return SimReceiveSession(
            transfer_id=transfer_id,
            buffer=self._buffer,
            ready_at_ns=ready_at_ns,
            slot=slot,
            pool=self,
        )

    def release_after(self, transfer_id: str, dependency: Any) -> None:
        readers = tuple(
            leaf for leaf in jax.tree_util.tree_leaves(dependency) if isinstance(leaf, jax.Array)
        )
        with self._condition:
            state = self._active.get(transfer_id)
            if state is not None and state.readers is None:
                state.readers = readers
                self._reap_released_locked()

    def abandon(self, transfer_id: str) -> None:
        self.release_after(transfer_id, ())

    def progress(self) -> None:
        with self._condition:
            self._reap_released_locked()

    def _reap_released_locked(self) -> None:
        now_ns = time.monotonic_ns()
        for transfer_id, state in list(self._active.items()):
            if (
                state.readers is not None
                and state.ready_at_ns <= now_ns
                and all(reader.is_ready() for reader in state.readers)
            ):
                self._release_locked(transfer_id)

    def _release_locked(self, transfer_id: str) -> None:
        active = self._active.pop(transfer_id, None)
        if active is not None:
            self._free.append(active.slot)
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
        transfer_id = data.transfer.get("transfer_id")
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

    def progress(self) -> bool:
        with self._pool_lock:
            pool = self._pool
        if pool is not None:
            pool.progress()
        return True

    def close(self) -> None:
        with self._pool_lock:
            self._closed = True
            pool = self._pool
        if pool is not None:
            pool.close()
        self._executor.shutdown(cancel_futures=True)
