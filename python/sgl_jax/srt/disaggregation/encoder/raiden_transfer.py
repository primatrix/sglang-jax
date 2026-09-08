"""Write encoder pages, publish registered reads, and retain pages until completion."""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any

import jax

from sgl_jax.raiden import require_raiden_preloaded
from sgl_jax.srt.disaggregation.encoder.raiden_pool import RaidenPool
from sgl_jax.srt.disaggregation.raiden_transfer.wrapper import RaidenTransferWrapper

logger = logging.getLogger(__name__)


def _uuid_to_int(value: str) -> int:
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") & ((1 << 50) - 1)


@dataclass(slots=True)
class _SendReservation:
    transfer_id: str
    page_ids: tuple[int, ...]
    token_count: int
    write: jax.Array | None = None
    registered: bool = False
    cancelled: bool = False
    transfer_done: bool = False


class RaidenEncoderServerTransfer:
    """Each reservation owns pages until its write and registered read finish."""

    def __init__(
        self,
        host_ip: str,
        pool: RaidenPool,
        *,
        parallelism: int = 1,
        pool_size: int = 32,
        timeout_s: float = 300.0,
        poll_interval_s: float = 0.001,
    ) -> None:
        require_raiden_preloaded()
        self._max_inflight = max(1, int(pool_size))
        self._timeout_s = float(timeout_s)
        self._poll_interval_s = max(0.0001, float(poll_interval_s))
        self._raiden = RaidenTransferWrapper(host_ip, 0, parallelism=max(1, int(parallelism)))
        self._pool = pool
        self._reservations: dict[str, _SendReservation] = {}
        self._lock = threading.Lock()
        self._closed = False
        self._raiden.start(
            [pool.buffer],
            max_blocks=pool.num_pages,
            num_slots=self._max_inflight,
            timeout_s=self._timeout_s,
        )

    def batch_capacity(self, token_count: int) -> int:
        pages = max(1, (token_count + self._pool.page_size - 1) // self._pool.page_size)
        return max(1, min(self._max_inflight, self._pool.num_pages // pages))

    def reserve_batch_sync(
        self, transfer_ids: list[str], token_counts: tuple[int, ...]
    ) -> list[_SendReservation]:
        if len(transfer_ids) != len(token_counts):
            raise ValueError("Encoder transfer IDs and token counts differ")
        if not transfer_ids:
            return []
        needed = sum(self._pool.pages_needed(count) for count in token_counts)
        if len(transfer_ids) > self._max_inflight or needed > self._pool.num_pages:
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
                if (
                    needed <= self._pool.available_pages
                    and len(self._reservations) + len(transfer_ids) <= self._max_inflight
                ):
                    reservations = [
                        _SendReservation(transfer_id, self._pool.allocate(count), count)
                        for transfer_id, count in zip(transfer_ids, token_counts, strict=True)
                    ]
                    self._reservations.update((item.transfer_id, item) for item in reservations)
                    return reservations
            self._reap_completed()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("timed out waiting for Raiden encoder transfer capacity")
            time.sleep(min(remaining, self._poll_interval_s))

    def _check_reservation_locked(self, reservation: _SendReservation) -> None:
        if (
            self._closed
            or reservation.registered
            or reservation.cancelled
            or self._reservations.get(reservation.transfer_id) is not reservation
        ):
            raise RuntimeError(f"Raiden reservation is no longer active: {reservation.transfer_id}")

    def stage_packed_batch_sync(
        self,
        reservations: list[_SendReservation],
        packed: jax.Array,
    ) -> list[_SendReservation]:
        try:
            if not reservations:
                return []
            with self._lock:
                for reservation in reservations:
                    self._check_reservation_locked(reservation)
                write = self._pool.write_packed(
                    packed,
                    [item.page_ids for item in reservations],
                    tuple(item.token_count for item in reservations),
                )
                for reservation in reservations:
                    reservation.write = write
            return reservations
        except BaseException:
            self.cancel_batch(reservations)
            raise

    def publish_batch_sync(self, reservations: list[_SendReservation]) -> list[dict[str, Any]]:
        """Like PD, finish donated writes before handing their pages to Raiden."""
        if not reservations:
            return []
        metadata = []
        try:
            for reservation in reservations:
                if reservation.write is None:
                    raise RuntimeError("Raiden reservation has no staged copy")
                reservation.write.block_until_ready()
                with self._lock:
                    self._check_reservation_locked(reservation)
                    transfer_id = reservation.transfer_id
                    transfer_uuid = _uuid_to_int(transfer_id)
                    if not self._raiden.register_read(
                        transfer_id, transfer_uuid, list(reservation.page_ids)
                    ):
                        raise RuntimeError(
                            "Raiden reported no transfer for non-empty encoder pages"
                        )
                    reservation.registered = True
                metadata.append(
                    {
                        "transfer_id": transfer_id,
                        "transfer_uuid": transfer_uuid,
                        "transfer_address": self._raiden.endpoints,
                        "transfer_host": self._raiden.host_ip,
                        "transfer_block_ids": list(reservation.page_ids),
                        "transfer_page_size": self._pool.page_size,
                    }
                )
            return metadata
        except BaseException:
            # Accepted registrations remain owned by Raiden until poll_stats
            # reports send completion, even if another registration in the batch fails.
            self.cancel_batch(reservations)
            raise

    def _reap_completed(self) -> None:
        try:
            sent, _, _ = self._raiden.poll_stats()
        except Exception:
            logger.exception("Raiden encoder sender poll failed")
            return
        with self._lock:
            for transfer_id in sent:
                reservation = self._reservations.get(transfer_id)
                if reservation is not None and reservation.registered:
                    reservation.transfer_done = True
            for reservation in list(self._reservations.values()):
                if reservation.transfer_done or reservation.cancelled:
                    self._release_if_ready_locked(reservation)

    def _release_if_ready_locked(self, reservation: _SendReservation) -> None:
        if (
            (not reservation.registered or reservation.transfer_done)
            and self._reservations.get(reservation.transfer_id) is reservation
            and (reservation.write is None or reservation.write.is_ready())
        ):
            self._release_locked(reservation)

    def cancel_batch(self, reservations: list[_SendReservation]) -> None:
        with self._lock:
            for reservation in reservations:
                reservation.cancelled = True
                self._release_if_ready_locked(reservation)

    def release(self, transfer_id: str) -> None:
        with self._lock:
            reservation = self._reservations.get(transfer_id)
            if reservation is None:
                return
            reservation.cancelled = True
            self._release_if_ready_locked(reservation)

    def _release_locked(self, reservation: _SendReservation) -> None:
        self._reservations.pop(reservation.transfer_id)
        self._pool.release(reservation.page_ids)

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._reservations.clear()
