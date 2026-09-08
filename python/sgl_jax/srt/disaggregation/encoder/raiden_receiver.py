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
            rows = (
                np.asarray(self.page_ids, np.int32)[:, None] * pool.page_size
                + np.arange(pool.page_size, dtype=np.int32)
            ).reshape(-1)[: self.token_count]
            return PooledEmbedding(pool.buffer, rows, pool.width, (self,))

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
        self._closed = False
        self._transfer = RaidenTransferWrapper(host, 0, parallelism=parallelism)
        self._transfer.start(
            [pool.buffer],
            max_blocks=pool.num_pages,
            num_slots=pool_size,
            timeout_s=transfer_timeout_s,
        )
        self._setup_executor = ThreadPoolExecutor(max_workers=1)

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
        transfer_uuid = data.transfer.get("transfer_uuid")
        remote_blocks = data.transfer.get("transfer_block_ids")
        needed = self.pool.pages_needed(shape[0])
        if not transfer_id or not isinstance(transfer_uuid, int):
            raise ValueError("Raiden transfer identity is incomplete")
        if (
            not isinstance(remote_blocks, list)
            or len(remote_blocks) != needed
            or any(not isinstance(page, int) or page < 0 for page in remote_blocks)
            or len(set(remote_blocks)) != needed
        ):
            raise ValueError("Raiden page metadata does not match the embedding length")
        host = data.transfer.get("transfer_host")
        if not host or str(host).strip("[]") in _LOCAL_ENDPOINT_HOSTS:
            raise ValueError("Raiden transfer_host is required")
        endpoints = _normalize_endpoints(data.transfer.get("transfer_address"), host)
        deadline = time.monotonic() + self._timeout_s
        with self._condition:
            if transfer_id in self._receives:
                raise ValueError(f"duplicate Raiden transfer_id: {transfer_id}")
            while not self._closed:
                self._progress_locked()
                inflight = sum(not (s.received or s.failed) for s in self._receives.values())
                if needed <= self.pool.available_pages and inflight < self._max_inflight:
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("timed out waiting for encoder embedding pages")
                self._condition.wait(min(remaining, 0.01))
            if self._closed:
                raise RuntimeError("Raiden receiver is closed")
            pages = self.pool.allocate(shape[0])
            session = RaidenReceiveSession(self, transfer_id, pages, shape[0])
            self._receives[transfer_id] = session
            try:
                self._transfer.start_read(
                    transfer_id, transfer_uuid, endpoints, remote_blocks, list(pages)
                )
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
        _, received, failed = self._transfer.poll_stats()
        for transfer_id in received:
            if session := self._receives.get(transfer_id):
                session.received = True
        for transfer_id in failed:
            if session := self._receives.get(transfer_id):
                session.failed = True
        for session in list(self._receives.values()):
            self._reclaim_locked(session)

    def _reclaim_locked(self, session: RaidenReceiveSession) -> None:
        if (
            session.released
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
