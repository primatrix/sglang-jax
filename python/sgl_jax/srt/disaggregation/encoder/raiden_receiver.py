"""Raiden receives directly into request-owned embedding pages."""

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
from sgl_jax.srt.disaggregation.encoder.raiden_pool import RaidenPool
from sgl_jax.srt.disaggregation.raiden_transfer.wrapper import RaidenTransferWrapper
from sgl_jax.srt.multimodal.in_model.embedding_view import PooledEmbedding

_LOCAL_ENDPOINT_HOSTS = {"", "0.0.0.0", "127.0.0.1", "::", "::1", "localhost"}


def _normalize_endpoint(endpoint: object, peer_host: str) -> str:
    host, port_text = str(endpoint).rsplit(":", 1)
    host = host.strip("[]")
    if host in _LOCAL_ENDPOINT_HOSTS:
        host = peer_host
    if ":" in host:
        host = f"[{host}]"
    return f"{host}:{int(port_text)}"


def _normalize_endpoints(endpoints: object, peer_host: str) -> list[dict[str, Any]]:
    if not isinstance(endpoints, list) or not endpoints:
        raise ValueError("Raiden encoder did not publish endpoint descriptors")
    return [
        {
            "endpoint": _normalize_endpoint(item.get("endpoint", ""), peer_host),
            "shards": [int(shard) for shard in item.get("shards", [])],
        }
        for item in endpoints
    ]


@dataclass(eq=False, slots=True)
class RaidenReceiveSession:
    """The transfer and all its row views share this one page owner."""

    receiver: RaidenReceiverBackend
    transfer_id: str
    page_ids: tuple[int, ...]
    token_count: int
    received: bool = False
    failed: bool = False
    delivered: bool = False
    released: bool = False
    readers: list[jax.Array] = field(default_factory=list)
    rows: np.ndarray | None = None
    pending_transfer_ids: set[str] = field(default_factory=set)

    def poll(self, *, refresh_backend: bool = True) -> PooledEmbedding | None:
        with self.receiver._condition:
            if self.delivered or self.released:
                return None
            if refresh_backend:
                self.receiver._progress_locked()
            if self.failed:
                self.released = True
                self.receiver._reclaim_locked(self)
                raise RuntimeError(f"Raiden embedding transfer failed: {self.transfer_id}")
            if not self.received:
                return None
            self.delivered = True
            pool = self.receiver.pool
            return PooledEmbedding(pool.buffer, self.rows, pool.width, (self,))

    def record_read(self, result: jax.Array) -> None:
        with self.receiver._condition:
            if self.released:
                raise RuntimeError("Cannot read released encoder pages")
            self.readers = [reader for reader in self.readers if not reader.is_ready()]
            self.readers.append(result)

    def release(self) -> None:
        with self.receiver._condition:
            self.released = True
            self.receiver._reclaim_locked(self)

    def close(self) -> None:
        # After delivery the request, not the receive session, owns the pages.
        if not self.delivered:
            self.release()


class RaidenReceiverBackend:
    def __init__(self, host, pool: RaidenPool, parallelism, pool_size, transfer_timeout_s):
        self.pool = pool
        self._max_inflight = pool_size
        self._timeout_s = transfer_timeout_s
        self._condition = threading.Condition()
        self._receives: dict[str, RaidenReceiveSession] = {}
        self._pending_parts: dict[str, RaidenReceiveSession] = {}
        self._pending_ranks: dict[str, int] = {}
        self._closed = False
        self._sharded = not pool.sharding.is_fully_replicated
        buffers = (
            [s.data for s in sorted(pool.buffer.addressable_shards, key=lambda s: s.index[0].start)]
            if self._sharded
            else [pool.buffer]
        )
        self._transfers = []
        for buffer in buffers:
            transfer = RaidenTransferWrapper(host, 0, parallelism=parallelism)
            transfer.start(
                [buffer],
                max_blocks=buffer.shape[0],
                num_slots=pool_size,
                timeout_s=transfer_timeout_s,
            )
            self._transfers.append(transfer)
        self._setup_executor = ThreadPoolExecutor(max_workers=1)

    def _place_parts(self, plans):
        available = (
            self.pool.available_pages_by_shard() if self._sharded else [self.pool.available_pages]
        )
        inflight = [0] * len(available)
        for rank in self._pending_ranks.values():
            inflight[rank] += 1
        ranks = [0] * len(plans)
        for index in sorted(range(len(plans)), key=lambda index: -plans[index][-1]):
            count = plans[index][-1]
            candidates = [
                rank
                for rank, pages in enumerate(available)
                if pages >= count and inflight[rank] < self._max_inflight
            ]
            if not candidates:
                return None
            rank = max(candidates, key=lambda rank: available[rank])
            ranks[index] = rank
            available[rank] -= count
            inflight[rank] += 1
        return ranks

    def start(self, data: EmbeddingData) -> DeferredReceiveSession:
        return DeferredReceiveSession(self._setup_executor.submit(self._start, data))

    def _start(self, data: EmbeddingData) -> RaidenReceiveSession:
        if data.shape is None or data.dtype is None:
            raise ValueError("embedding shape and dtype are required")
        shape = tuple(map(int, data.shape))
        if (
            len(shape) != 2
            or min(shape) <= 0
            or shape[1] != self.pool.width
            or jnp.dtype(data.dtype) != self.pool.dtype
            or data.transfer.get("transfer_page_size") != self.pool.page_size
        ):
            raise ValueError("Raiden embedding layout does not match the receive pool")
        transfer_id = data.transfer.get("transfer_id")
        if not transfer_id:
            raise ValueError("Raiden transfer identity is incomplete")
        parts = data.transfer.get("transfer_parts")
        if parts is None:
            parts = [data.transfer | {"token_indices": list(range(shape[0]))}]
        if not isinstance(parts, list) or not parts:
            raise ValueError("Raiden transfer has no parts")
        plans = []
        positions = []
        for part in parts:
            indices = np.asarray(part.get("token_indices"), dtype=np.int32)
            if indices.ndim != 1 or not indices.size:
                raise ValueError("Raiden part has no logical token indices")
            count = self.pool.pages_needed(indices.size)
            child_id = part.get("transfer_id")
            uuid = part.get("transfer_uuid")
            blocks = part.get("transfer_block_ids")
            if not child_id or not isinstance(uuid, int):
                raise ValueError("Raiden transfer identity is incomplete")
            if (
                not isinstance(blocks, list)
                or len(blocks) != count
                or any(not isinstance(page, int) or page < 0 for page in blocks)
                or len(set(blocks)) != count
            ):
                raise ValueError("Raiden page metadata does not match the embedding length")
            if part.get("transfer_page_size") != self.pool.page_size:
                raise ValueError("Raiden part page size does not match the receive pool")
            host = part.get("transfer_host")
            if not host or str(host).strip("[]") in _LOCAL_ENDPOINT_HOSTS:
                raise ValueError("Raiden transfer_host is required")
            endpoints = _normalize_endpoints(part.get("transfer_address"), host)
            plans.append((child_id, uuid, endpoints, blocks, indices, count))
            positions.append(indices)
        if not np.array_equal(np.sort(np.concatenate(positions)), np.arange(shape[0])):
            raise ValueError("Raiden parts must cover each logical token exactly once")
        if len({part[0] for part in plans}) != len(plans):
            raise ValueError("duplicate Raiden part transfer_id")
        needed = sum(part[-1] for part in plans)
        if needed > self.pool.num_pages:
            raise ValueError("Raiden parts exceed the receive pool capacity")
        if any(part[-1] > self.pool.num_pages // len(self._transfers) for part in plans):
            raise ValueError("Raiden part exceeds one receive shard's page capacity")
        deadline = time.monotonic() + self._timeout_s
        with self._condition:
            if transfer_id in self._receives or any(
                part[0] in self._pending_parts for part in plans
            ):
                raise ValueError(f"duplicate Raiden transfer_id: {transfer_id}")
            while not self._closed:
                self._progress_locked()
                ranks = self._place_parts(plans)
                if ranks is not None:
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("timed out waiting for encoder embedding pages")
                self._condition.wait(min(remaining, 0.01))
            if self._closed:
                raise RuntimeError("Raiden receiver is closed")
            allocations = [
                self.pool.allocate(
                    plan[-1] * self.pool.page_size, shard=rank if self._sharded else None
                )
                for plan, rank in zip(plans, ranks, strict=True)
            ]
            pages = tuple(page for allocation in allocations for page in allocation)
            session = RaidenReceiveSession(self, transfer_id, pages, shape[0])
            session.rows = np.empty(shape[0], dtype=np.int32)
            self._receives[transfer_id] = session
            try:
                for plan, rank, local_pages in zip(plans, ranks, allocations, strict=True):
                    child_id, uuid, endpoints, blocks, indices, count = plan
                    rows = (
                        np.asarray(local_pages, np.int32)[:, None] * self.pool.page_size
                        + np.arange(self.pool.page_size, dtype=np.int32)
                    ).reshape(-1)
                    session.rows[indices] = rows[: indices.size]
                    page_base = rank * (self.pool.num_pages // len(self._transfers))
                    self._transfers[rank].start_read(
                        child_id,
                        uuid,
                        endpoints,
                        blocks,
                        [page - page_base for page in local_pages],
                    )
                    session.pending_transfer_ids.add(child_id)
                    self._pending_parts[child_id] = session
                    self._pending_ranks[child_id] = rank
            except Exception:
                session.failed = session.released = True
                self._reclaim_locked(session)
                raise
            return session

    def progress(self) -> bool:
        with self._condition:
            if not self._closed:
                self._progress_locked()
        return True

    def _progress_locked(self) -> None:
        for transfer in self._transfers:
            _, received, failed = transfer.poll_stats()
            for transfer_id in received + failed:
                if session := self._pending_parts.pop(transfer_id, None):
                    self._pending_ranks.pop(transfer_id)
                    session.pending_transfer_ids.remove(transfer_id)
                    session.failed |= transfer_id in failed
                    session.received = not session.pending_transfer_ids and not session.failed
        for session in list(self._receives.values()):
            self._reclaim_locked(session)

    def _reclaim_locked(self, session: RaidenReceiveSession) -> None:
        if (
            session.released
            and not session.pending_transfer_ids
            and (session.received or session.failed)
            and self._receives.get(session.transfer_id) is session
            and all(reader.is_ready() for reader in session.readers)
        ):
            self._receives.pop(session.transfer_id)
            self.pool.release(session.page_ids)
            session.readers.clear()
            self._condition.notify()

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        self._setup_executor.shutdown(cancel_futures=True)
