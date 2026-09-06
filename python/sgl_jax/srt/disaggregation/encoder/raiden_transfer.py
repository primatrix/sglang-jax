"""Reserve source slots, overlap packed copies with Raiden registration, then publish."""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import jax

from sgl_jax.raiden import require_raiden_preloaded
from sgl_jax.srt.disaggregation.encoder.raiden_pool import (
    RaidenSendPool,
    compile_packed_pool_copy,
)
from sgl_jax.srt.disaggregation.raiden_transfer.wrapper import RaidenTransferWrapper

logger = logging.getLogger(__name__)


def _uuid_to_int(value: str) -> int:
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") & ((1 << 50) - 1)


@dataclass(slots=True)
class _SendReservation:
    transfer_id: str
    slot: int
    pool_copy: _PoolCopy | None = None
    registered: bool = False
    cancelled: bool = False


@dataclass(slots=True)
class _PoolCopy:
    ready_values: tuple[jax.Array, ...]
    completed: bool = False

    def is_ready(self) -> bool:
        return self.completed or all(ticket.is_ready() for ticket in self.ready_values)

    def wait(self) -> None:
        if not self.completed:
            for ticket in self.ready_values:
                if not ticket.is_ready():
                    ticket.block_until_ready()
            self.completed = True


class RaidenEncoderServerTransfer:
    """Each reservation owns one slot until its copy and registered read finish."""

    def __init__(
        self,
        host_ip: str,
        *,
        parallelism: int = 1,
        pool_size: int = 32,
        timeout_s: float = 300.0,
        poll_interval_s: float = 0.001,
    ) -> None:
        require_raiden_preloaded()
        self._pool_size = max(1, int(pool_size))
        self._timeout_s = float(timeout_s)
        self._poll_interval_s = max(0.0001, float(poll_interval_s))
        self._raiden = RaidenTransferWrapper(host_ip, 0, parallelism=max(1, int(parallelism)))
        self._pool: RaidenSendPool | None = None
        self._free_slots = list(range(self._pool_size - 1, -1, -1))
        self._reservations: dict[str, _SendReservation] = {}
        self._lock = threading.Lock()
        self._compile_pool = ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="encoder-pool-copy"
        )
        # Only the encode worker accesses the cache; compilation runs in parallel.
        self._copy_executables: dict[tuple[Any, ...], Future[Any]] = {}
        self._closed = False

    def _copy_executable(
        self,
        packed: jax.Array | jax.ShapeDtypeStruct,
        token_counts: tuple[int, ...],
        *,
        contiguous: bool,
    ) -> Future[Any]:
        if not token_counts or any(count != token_counts[0] for count in token_counts):
            raise ValueError("Raiden source pool requires one embedding shape")
        sharding = packed.sharding
        if isinstance(sharding, jax.sharding.NamedSharding) and sharding.is_fully_replicated:
            sharding = sharding.update(spec=jax.sharding.PartitionSpec())
        key = (packed.shape, str(packed.dtype), sharding, token_counts, contiguous)
        if key not in self._copy_executables:
            self._copy_executables[key] = self._compile_pool.submit(
                compile_packed_pool_copy,
                packed,
                (token_counts[0], int(packed.shape[1])),
                capacity=self._pool_size,
                token_counts=token_counts,
                contiguous=contiguous,
            )
        return self._copy_executables[key]

    def precompile_packed_batches(
        self, specs: tuple[tuple[jax.ShapeDtypeStruct, tuple[int, ...]], ...]
    ) -> None:
        for packed, token_counts in specs:
            self._copy_executable(packed, token_counts, contiguous=True)
            self._copy_executable(packed, token_counts, contiguous=False)

    def _take_free_slots_locked(self, count: int) -> list[int]:
        """Prefer contiguous slots so a batch needs only one device update."""
        available = sorted(self._free_slots)
        for index in range(count - 1, len(available)):
            start = available[index - count + 1]
            if available[index] - start == count - 1:
                slots = list(range(start, start + count))
                self._free_slots = [
                    slot for slot in self._free_slots if not start <= slot < start + count
                ]
                return slots
        return [self._free_slots.pop() for _ in range(count)]

    def reserve_batch_sync(self, transfer_ids: list[str]) -> list[_SendReservation]:
        if not transfer_ids:
            return []
        if len(transfer_ids) > self._pool_size:
            raise ValueError("encoder batch exceeds Raiden pool capacity")
        if len(set(transfer_ids)) != len(transfer_ids):
            raise ValueError("duplicate Raiden transfer_id in encoder batch")
        deadline = time.monotonic() + self._timeout_s
        while True:
            with self._lock:
                if self._closed:
                    raise RuntimeError("Raiden encoder transfer is closed")
                if any(transfer_id in self._reservations for transfer_id in transfer_ids):
                    raise ValueError("duplicate Raiden transfer_id")
                if len(self._free_slots) >= len(transfer_ids):
                    slots = self._take_free_slots_locked(len(transfer_ids))
                    reservations = [
                        _SendReservation(transfer_id, slot)
                        for transfer_id, slot in zip(transfer_ids, slots)
                    ]
                    self._reservations.update((item.transfer_id, item) for item in reservations)
                    return reservations
            self._reap_completed()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("timed out waiting for Raiden encoder pool slots")
            time.sleep(min(remaining, self._poll_interval_s))

    def _check_reservation_locked(self, reservation: _SendReservation) -> None:
        if (
            self._closed
            or reservation.registered
            or reservation.cancelled
            or self._reservations.get(reservation.transfer_id) is not reservation
        ):
            raise RuntimeError(f"Raiden reservation is no longer active: {reservation.transfer_id}")

    def _source_pool(self, packed: jax.Array, rows: int) -> RaidenSendPool:
        shape = (rows, int(packed.shape[1]))
        if self._pool is None:
            pool = RaidenSendPool(shape, packed.dtype, packed.sharding, capacity=self._pool_size)
            self._raiden.start(
                [pool.buffer], max_blocks=1, num_slots=self._pool_size, timeout_s=self._timeout_s
            )
            self._pool = pool
        return self._pool

    def stage_packed_batch_sync(
        self, reservations: list[_SendReservation], packed: jax.Array, token_counts: tuple[int, ...]
    ) -> list[_SendReservation]:
        try:
            if len(reservations) != len(token_counts):
                raise ValueError("Raiden reservation and packed item counts differ")
            if not reservations:
                return []
            if packed.ndim != 2 or any(count <= 0 for count in token_counts):
                raise ValueError("Raiden packed output must contain non-empty matrices")
            with self._lock:
                for reservation in reservations:
                    self._check_reservation_locked(reservation)
            pool = self._source_pool(packed, token_counts[0])
            slots = [reservation.slot for reservation in reservations]
            contiguous = slots == list(range(slots[0], slots[0] + len(slots)))
            executable = self._copy_executable(packed, token_counts, contiguous=contiguous).result()
            ready_values = pool.copy_packed_batch_async(
                packed, slots, token_counts, executable, contiguous=contiguous
            )
            copy = _PoolCopy(tuple(ready_values))
            for reservation in reservations:
                reservation.pool_copy = copy
            return reservations
        except BaseException:
            self.cancel_batch(reservations)
            raise

    def publish_batch_sync(self, reservations: list[_SendReservation]) -> list[dict[str, Any]]:
        """Register reads during the device copy; expose metadata only after it finishes."""
        if not reservations:
            return []
        metadata = []
        try:
            for reservation in reservations:
                if reservation.pool_copy is None:
                    raise RuntimeError("Raiden reservation has no staged copy")
                with self._lock:
                    self._check_reservation_locked(reservation)
                    reservation.registered = True
                transfer_id = reservation.transfer_id
                transfer_uuid = _uuid_to_int(transfer_id)
                if not self._raiden.register_read(transfer_id, transfer_uuid, [reservation.slot]):
                    with self._lock:
                        reservation.registered = False
                    raise RuntimeError(f"Raiden rejected encoder transfer {transfer_id!r}")
                metadata.append(
                    {
                        "transfer_id": transfer_id,
                        "transfer_uuid": transfer_uuid,
                        "transfer_address": self._raiden.endpoints,
                        "transfer_host": self._raiden.host_ip,
                        "transfer_block_ids": [reservation.slot],
                    }
                )
            for reservation in reservations:
                reservation.pool_copy.wait()
            return metadata
        except BaseException:
            # Accepted registrations remain owned by Raiden until poll_stats
            # reports sent/failed, even if another registration in the batch fails.
            self.cancel_batch(reservations)
            raise

    def _reap_completed(self) -> None:
        if self._pool is None:
            return
        try:
            sent, _, failed = self._raiden.poll_stats()
        except Exception:
            logger.exception("Raiden encoder sender poll failed")
            return
        with self._lock:
            for transfer_id in sent + failed:
                reservation = self._reservations.get(transfer_id)
                if reservation is not None and reservation.registered:
                    self._release_locked(reservation)
            for reservation in list(self._reservations.values()):
                if reservation.cancelled and not reservation.registered:
                    self._cancel_locked(reservation)

    def _cancel_locked(self, reservation: _SendReservation) -> None:
        reservation.cancelled = True
        if (
            not reservation.registered
            and self._reservations.get(reservation.transfer_id) is reservation
            and (reservation.pool_copy is None or reservation.pool_copy.is_ready())
        ):
            self._release_locked(reservation)

    def cancel_batch(self, reservations: list[_SendReservation]) -> None:
        with self._lock:
            for reservation in reservations:
                self._cancel_locked(reservation)

    def release(self, transfer_id: str) -> None:
        with self._lock:
            reservation = self._reservations.get(transfer_id)
            if reservation is None:
                return
            self._cancel_locked(reservation)

    def _release_locked(self, reservation: _SendReservation) -> None:
        self._reservations.pop(reservation.transfer_id)
        self._free_slots.append(reservation.slot)

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._free_slots.clear()
            self._reservations.clear()
        self._compile_pool.shutdown(cancel_futures=True)
