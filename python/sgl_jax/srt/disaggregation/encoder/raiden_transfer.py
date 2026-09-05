"""Raiden-backed encoder transfer implementations."""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from sgl_jax.raiden import require_raiden_preloaded
from sgl_jax.srt.disaggregation.encoder.client import DeferredReceiveSession
from sgl_jax.srt.disaggregation.encoder.embedding_data import EmbeddingData
from sgl_jax.srt.disaggregation.encoder.raiden_pool import (
    RaidenReceivePool,
    RaidenReceiveSession,
    RaidenSendPool,
    compile_packed_pool_copy,
)
from sgl_jax.srt.disaggregation.raiden_transfer.wrapper import RaidenTransferWrapper

logger = logging.getLogger(__name__)
_LOCAL_ENDPOINT_HOSTS = {"", "0.0.0.0", "127.0.0.1", "::", "::1", "localhost"}


def _uuid_to_int(value: str) -> int:
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") & ((1 << 50) - 1)


def _normalize_endpoint(endpoint: object, peer_host: str) -> str:
    value = str(endpoint)
    host, port_text = value.rsplit(":", 1)
    port = int(port_text)
    host = host.strip("[]")
    if host in _LOCAL_ENDPOINT_HOSTS:
        host = peer_host
    if ":" in host:
        host = f"[{host}]"
    return f"{host}:{port}"


def _normalize_endpoints(endpoints: object, peer_host: str) -> list[dict[str, Any]]:
    if not isinstance(endpoints, list) or not endpoints:
        raise ValueError("Raiden encoder did not publish endpoint descriptors")
    result = []
    for item in endpoints:
        shards = item.get("shards", [])
        result.append(
            {
                "endpoint": _normalize_endpoint(item.get("endpoint", ""), peer_host),
                "shards": [int(shard) for shard in shards],
            }
        )
    return result


@dataclass(frozen=True, slots=True)
class _Reservation:
    transfer_id: str
    slot: int
    reserve_start_ns: int
    reserve_done_ns: int


@dataclass(frozen=True, slots=True)
class _StagedTransfer:
    reservation: _Reservation
    pool_ready_ns: int
    copy_submit_ns: int
    ready_group: _ReadyGroup


@dataclass(slots=True)
class _ReadyGroup:
    """Copy completion cached by the single transfer worker."""

    tickets: tuple[jax.Array, ...]
    _copy_done_ns: int | None = None

    def wait(self) -> int:
        if self._copy_done_ns is None:
            for ticket in self.tickets:
                if not ticket.is_ready():
                    ticket.block_until_ready()
            self._copy_done_ns = time.time_ns()
        return self._copy_done_ns


class RaidenEncoderServerTransfer:
    """Publish encoder outputs from one registered source pool."""

    def __init__(
        self,
        host_ip: str,
        *,
        parallelism: int = 1,
        pool_size: int = 32,
        timeout_s: float = 300.0,
        poll_interval_s: float = 0.001,
        log_inflight: bool = False,
    ) -> None:
        require_raiden_preloaded()
        self._pool_size = max(1, int(pool_size))
        self._timeout_s = float(timeout_s)
        self._poll_interval_s = max(0.0001, float(poll_interval_s))
        self._log_inflight = bool(log_inflight)
        self._raiden = RaidenTransferWrapper(
            host_ip,
            0,
            parallelism=max(1, int(parallelism)),
        )
        self._pool: RaidenSendPool | None = None
        self._free = list(range(self._pool_size - 1, -1, -1))
        self._slots: dict[str, int] = {}
        self._active: set[str] = set()
        self._pending: set[str] = set()
        self._group_by_transfer: dict[str, str] = {}
        self._group_members: dict[str, set[str]] = {}
        self._group_sizes: dict[str, int] = {}
        self._lock = threading.Lock()
        self._compile_pool = ThreadPoolExecutor(
            max_workers=4,
            thread_name_prefix="encoder-pool-write-compile",
        )
        # Only the ViT worker accesses this cache; compile workers return via futures.
        self._packed_executables: dict[tuple[Any, ...], Future[Any]] = {}
        self._closed = False

    @staticmethod
    def _packed_key(
        packed: jax.Array | jax.ShapeDtypeStruct,
        token_counts: tuple[int, ...],
        contiguous: bool,
    ) -> tuple[Any, ...]:
        return (
            tuple(int(dim) for dim in packed.shape),
            str(packed.dtype),
            repr(packed.sharding),
            token_counts,
            contiguous,
        )

    def _packed_executable(
        self,
        packed: jax.Array | jax.ShapeDtypeStruct,
        token_counts: tuple[int, ...],
        *,
        contiguous: bool,
    ) -> Future[Any]:
        if not token_counts or any(token_count != token_counts[0] for token_count in token_counts):
            raise ValueError("Raiden source pool requires one embedding shape")
        key = self._packed_key(packed, token_counts, contiguous)
        future = self._packed_executables.get(key)
        if future is None:
            future = self._compile_pool.submit(
                compile_packed_pool_copy,
                packed,
                (token_counts[0], int(packed.shape[1])),
                capacity=self._pool_size,
                token_counts=token_counts,
                contiguous=contiguous,
            )
            self._packed_executables[key] = future
        return future

    def precompile_packed_batches(
        self,
        specs: tuple[
            tuple[jax.ShapeDtypeStruct, tuple[int, ...]],
            ...,
        ],
    ) -> None:
        for packed, token_counts in specs:
            self._packed_executable(packed, token_counts, contiguous=True)
            self._packed_executable(packed, token_counts, contiguous=False)

    def _reserve_slots_locked(self, count: int) -> list[int]:
        available = set(self._free)
        for start in sorted(available):
            slots = list(range(start, start + count))
            if all(slot in available for slot in slots):
                reserved = set(slots)
                self._free = [slot for slot in self._free if slot not in reserved]
                return slots
        return [self._free.pop() for _ in range(count)]

    def reserve_batch_sync(self, transfer_ids: list[str]) -> list[_Reservation]:
        transfer_ids = list(transfer_ids)
        if not transfer_ids:
            return []
        if len(transfer_ids) > self._pool_size:
            raise ValueError("encoder batch exceeds Raiden pool capacity")
        if len(set(transfer_ids)) != len(transfer_ids):
            raise ValueError("duplicate Raiden transfer_id in encoder batch")

        reserve_start_ns = time.time_ns()
        deadline = time.monotonic() + self._timeout_s
        while True:
            with self._lock:
                if self._closed:
                    raise RuntimeError("Raiden encoder transfer is closed")
                duplicate = next(
                    (transfer_id for transfer_id in transfer_ids if transfer_id in self._slots),
                    None,
                )
                if duplicate is not None:
                    raise ValueError(f"duplicate Raiden transfer_id: {duplicate}")
                if len(self._free) >= len(transfer_ids):
                    slots = self._reserve_slots_locked(len(transfer_ids))
                    self._slots.update(zip(transfer_ids, slots))
                    self._pending.update(transfer_ids)
                    reserve_done_ns = time.time_ns()
                    return [
                        _Reservation(
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
                raise TimeoutError("timed out waiting for Raiden encoder pool slots")
            time.sleep(min(remaining, self._poll_interval_s))

    def stage_packed_batch_sync(
        self,
        reservations: list[_Reservation],
        packed: jax.Array,
        token_counts: tuple[int, ...],
    ) -> list[_StagedTransfer]:
        if len(reservations) != len(token_counts):
            raise ValueError("Raiden reservation and packed item counts differ")
        if not reservations:
            return []
        if (
            packed.ndim != 2
            or not token_counts
            or any(token_count <= 0 for token_count in token_counts)
        ):
            raise ValueError("Raiden packed output must contain non-empty matrices")
        if any(token_count != token_counts[0] for token_count in token_counts):
            raise ValueError("Raiden source pool requires one embedding shape")

        with self._lock:
            if self._closed:
                raise RuntimeError("Raiden encoder transfer is closed")
            for reservation in reservations:
                if (
                    reservation.transfer_id not in self._pending
                    or self._slots.get(reservation.transfer_id) != reservation.slot
                ):
                    raise RuntimeError(
                        f"Raiden reservation is no longer active: {reservation.transfer_id}"
                    )
            pool = self._pool

        shape = (token_counts[0], int(packed.shape[1]))
        if pool is None:
            pool = RaidenSendPool(
                shape,
                packed.dtype,
                packed.sharding,
                capacity=self._pool_size,
            )
            self._raiden.start(
                [pool.buffer],
                max_blocks=1,
                num_slots=self._pool_size,
                timeout_s=self._timeout_s,
            )
            with self._lock:
                self._pool = pool
        elif pool.shape != shape or pool.dtype != packed.dtype or pool.sharding != packed.sharding:
            raise ValueError(
                "Raiden encoder pool packed output mismatch: "
                f"expected shape={pool.shape}, dtype={pool.dtype}, "
                f"sharding={pool.sharding}; got shape={shape}, "
                f"dtype={packed.dtype}, sharding={packed.sharding}"
            )
        pool_ready_ns = time.time_ns()
        slots = [reservation.slot for reservation in reservations]
        contiguous = slots == list(range(slots[0], slots[0] + len(slots)))

        try:
            executable = self._packed_executable(
                packed,
                token_counts,
                contiguous=contiguous,
            ).result()
            ready = pool.copy_packed_batch_async(
                packed,
                slots,
                token_counts,
                executable=executable,
                contiguous=contiguous,
            )
        except BaseException:
            self.cancel_batch(reservations)
            raise
        copy_submit_ns = time.time_ns()
        ready_group = _ReadyGroup(tuple(ready))
        return [
            _StagedTransfer(
                reservation,
                pool_ready_ns,
                copy_submit_ns,
                ready_group,
            )
            for reservation in reservations
        ]

    def publish_batch_sync(
        self,
        staged_transfers: list[_StagedTransfer],
    ) -> list[dict[str, Any]]:
        """Overlap source registration with the in-flight packed pool write."""
        if not staged_transfers:
            return []
        registrations = []
        try:
            for staged_transfer in staged_transfers:
                reservation = staged_transfer.reservation
                transfer_id = reservation.transfer_id
                slot = reservation.slot
                register_start_ns = time.time_ns()
                with self._lock:
                    if transfer_id not in self._pending or self._slots.get(transfer_id) != slot:
                        raise RuntimeError(f"Raiden reservation was cancelled: {transfer_id}")
                    self._pending.remove(transfer_id)
                    self._active.add(transfer_id)
                transfer_uuid = _uuid_to_int(transfer_id)
                if not self._raiden.register_read(transfer_id, transfer_uuid, [slot]):
                    raise RuntimeError(f"Raiden rejected encoder transfer {transfer_id!r}")
                registrations.append(
                    (staged_transfer, transfer_uuid, register_start_ns, time.time_ns())
                )

            metadata = []
            for (
                staged_transfer,
                transfer_uuid,
                register_start_ns,
                register_done_ns,
            ) in registrations:
                copy_done_ns = staged_transfer.ready_group.wait()
                metadata.append(
                    self._transfer_metadata(
                        staged_transfer,
                        transfer_uuid,
                        copy_done_ns=copy_done_ns,
                        register_start_ns=register_start_ns,
                        register_done_ns=register_done_ns,
                    )
                )
            group_id = staged_transfers[0].reservation.transfer_id
            with self._lock:
                self._track_group_locked(
                    [
                        staged_transfer.reservation.transfer_id
                        for staged_transfer in staged_transfers
                    ]
                )
            self._log_inflight_event(
                "start",
                group_id,
                group_size=len(staged_transfers),
            )
            return metadata
        except BaseException:
            with self._lock:
                for staged_transfer in staged_transfers:
                    self._release_locked(staged_transfer.reservation.transfer_id)
            raise

    def _transfer_metadata(
        self,
        staged_transfer: _StagedTransfer,
        transfer_uuid: int,
        *,
        copy_done_ns: int,
        register_start_ns: int,
        register_done_ns: int,
    ) -> dict[str, Any]:
        reservation = staged_transfer.reservation
        transfer_id = reservation.transfer_id
        slot = reservation.slot
        metadata = {
            "transfer_id": transfer_id,
            "transfer_uuid": transfer_uuid,
            "transfer_address": self._raiden.endpoints,
            "transfer_host": self._raiden.host_ip,
            "transfer_block_ids": [slot],
        }
        metadata.update(
            transfer_reserve_start_ns=reservation.reserve_start_ns,
            transfer_pool_ready_ns=staged_transfer.pool_ready_ns,
            transfer_reserve_done_ns=reservation.reserve_done_ns,
            transfer_copy_submit_ns=staged_transfer.copy_submit_ns,
            transfer_copy_done_ns=copy_done_ns,
            transfer_register_start_ns=register_start_ns,
            transfer_register_done_ns=register_done_ns,
            transfer_publish_ready_ns=max(copy_done_ns, register_done_ns),
        )
        return metadata

    def _reap_completed(self) -> None:
        with self._lock:
            started = self._pool is not None
        if not started:
            return
        try:
            sent, _, failed = self._raiden.poll_stats()
        except Exception:
            logger.exception("Raiden encoder sender poll failed")
            return
        self._discard_active_many(sent, event="sent")
        self._discard_active_many(failed, event="failed")

    def cancel_batch(self, reservations: list[_Reservation]) -> None:
        with self._lock:
            for reservation in reservations:
                transfer_id = reservation.transfer_id
                if transfer_id in self._pending:
                    self._release_locked(transfer_id)

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
            groups = [(group_id, self._group_sizes[group_id]) for group_id in self._group_members]
            self._free.clear()
            self._slots.clear()
            self._active.clear()
            self._pending.clear()
            self._group_by_transfer.clear()
            self._group_members.clear()
            self._group_sizes.clear()
        for group_id, group_size in groups:
            self._log_inflight_event("close", group_id, group_size=group_size)
        self._compile_pool.shutdown(cancel_futures=True)

    def _discard_active(self, transfer_id: str, *, event: str) -> None:
        self._discard_active_many([transfer_id], event=event)

    def _discard_active_many(self, transfer_ids: list[str], *, event: str) -> None:
        completed_groups = []
        removed = []
        with self._lock:
            for transfer_id in transfer_ids:
                if transfer_id in self._active:
                    removed.append(transfer_id)
                    completed = self._release_locked(transfer_id)
                    if completed is not None:
                        completed_groups.append(completed)
        for group_id, group_size in completed_groups:
            self._log_inflight_event(event, group_id, group_size=group_size)
        if removed and not completed_groups:
            self._log_inflight_event("progress", removed[0], group_size=len(removed))

    def _track_group_locked(self, transfer_ids: list[str]) -> None:
        group_id = transfer_ids[0]
        members = set(transfer_ids)
        self._group_members[group_id] = members
        self._group_sizes[group_id] = len(members)
        self._group_by_transfer.update((transfer_id, group_id) for transfer_id in members)

    def _release_locked(self, transfer_id: str) -> tuple[str, int] | None:
        slot = self._slots.pop(transfer_id, None)
        self._pending.discard(transfer_id)
        self._active.discard(transfer_id)
        if slot is not None:
            self._free.append(slot)
        group_id = self._group_by_transfer.pop(transfer_id, None)
        if group_id is None:
            return None
        members = self._group_members[group_id]
        members.discard(transfer_id)
        if members:
            return None
        self._group_members.pop(group_id, None)
        return group_id, self._group_sizes.pop(group_id)

    def _log_inflight_event(
        self,
        event: str,
        transfer_id: str,
        *,
        group_size: int = 1,
    ) -> None:
        if not self._log_inflight:
            return
        with self._lock:
            inflight_groups = len(self._group_members)
            inflight_requests = len(self._active)
        logger.info(
            "ENCODER-RAIDEN-INFLIGHT time_ns=%d event=%s transfer_id=%s "
            "group_size=%d inflight_groups=%d inflight_requests=%d",
            time.time_ns(),
            event,
            transfer_id,
            group_size,
            inflight_groups,
            inflight_requests,
        )


class RaidenReceiverBackend:
    def __init__(
        self,
        host: str,
        sharding: jax.sharding.Sharding,
        parallelism: int,
        pool_size: int,
        transfer_timeout_s: float,
    ) -> None:
        self._host = host
        self._sharding = sharding
        self._parallelism = max(1, int(parallelism))
        self._pool_size = max(1, int(pool_size))
        self._transfer_timeout_s = float(transfer_timeout_s)
        self._pool: RaidenReceivePool | None = None
        self._pool_lock = threading.Lock()
        self._closed = False
        # Pool creation and Raiden control-plane calls stay off the event loop.
        self._executor = ThreadPoolExecutor(max_workers=1)

    def start(self, data: EmbeddingData) -> DeferredReceiveSession:
        return DeferredReceiveSession(self._executor.submit(self._start, data))

    def _start(self, data: EmbeddingData) -> RaidenReceiveSession:
        if data.shape is None or data.dtype is None:
            raise ValueError("embedding shape and dtype are required")
        shape = tuple(int(dim) for dim in data.shape)
        if len(shape) != 2 or shape[0] <= 0:
            raise ValueError("Raiden embedding must be a non-empty matrix")
        transfer_id = getattr(data, "transfer_id", None)
        transfer_uuid = getattr(data, "transfer_uuid", None)
        remote_block_ids = getattr(data, "transfer_block_ids", None)
        endpoints = getattr(data, "transfer_address", None)
        if not transfer_id or not isinstance(transfer_uuid, int):
            raise ValueError("Raiden transfer identity is incomplete")
        if not isinstance(remote_block_ids, list) or len(remote_block_ids) != 1:
            raise ValueError("Raiden block metadata does not match embedding shape")
        remote_block_ids = [int(block_id) for block_id in remote_block_ids]
        if len(set(remote_block_ids)) != len(remote_block_ids) or any(
            block_id < 0 for block_id in remote_block_ids
        ):
            raise ValueError("Raiden remote block IDs must be unique and non-negative")

        transfer_host = getattr(data, "transfer_host", None)
        if str(transfer_host).strip("[]") in _LOCAL_ENDPOINT_HOSTS:
            transfer_host = None
        if not transfer_host:
            raise ValueError("Raiden transfer_host is required")
        remote_endpoints = _normalize_endpoints(endpoints, transfer_host)

        dtype = jnp.dtype(data.dtype)
        with self._pool_lock:
            if self._closed:
                raise RuntimeError("Raiden receiver is closed")
            pool = self._pool
            if pool is None:
                pool = RaidenReceivePool(
                    self._host,
                    shape,
                    dtype,
                    self._sharding,
                    parallelism=self._parallelism,
                    capacity=self._pool_size,
                    timeout_s=self._transfer_timeout_s,
                )
                self._pool = pool
            elif shape != pool.shape or dtype != pool.dtype:
                raise ValueError(
                    "Raiden receiver pool embedding mismatch: "
                    f"expected shape={pool.shape}, dtype={pool.dtype}; "
                    f"got shape={shape}, dtype={dtype}"
                )
        return pool.start(
            transfer_id,
            transfer_uuid,
            remote_endpoints,
            remote_block_ids,
        )

    def progress(self) -> bool:
        """Refresh the shared Raiden completion queue once per client tick."""
        with self._pool_lock:
            pool = self._pool
        if pool is None:
            return False
        pool.progress()
        return True

    def close(self) -> None:
        with self._pool_lock:
            self._closed = True
            pool = self._pool
        if pool is not None:
            pool.close()
        self._executor.shutdown(cancel_futures=True)
