from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
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
)

logger = logging.getLogger(__name__)
_LOCAL_ENDPOINT_HOSTS = {"", "0.0.0.0", "127.0.0.1", "::", "::1", "localhost"}


def _normalize_endpoint(endpoint: object, peer_host: str) -> str:
    value = str(endpoint)
    try:
        host, port_text = value.rsplit(":", 1)
        port = int(port_text)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid Raiden endpoint: {value!r}") from exc
    if not 0 < port <= 65535:
        raise ValueError(f"invalid Raiden endpoint port: {port}")
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
        if not isinstance(item, Mapping):
            raise TypeError("Raiden endpoint descriptor must be a mapping")
        shards = item.get("shards", [])
        if not isinstance(shards, list):
            raise TypeError("Raiden endpoint shards must be a list")
        result.append(
            {
                "endpoint": _normalize_endpoint(item.get("endpoint", ""), peer_host),
                "shards": [int(shard) for shard in shards],
            }
        )
    return result


class RaidenEncoderServerTransfer:
    """Publish request-sized encoder outputs from registered source pools."""

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
        self._host_ip = host_ip
        self._parallelism = max(1, int(parallelism))
        self._pool_size = max(1, int(pool_size))
        self._timeout_s = float(timeout_s)
        self._poll_interval_s = float(poll_interval_s)
        self._log_inflight = bool(log_inflight)
        self._pools: list[RaidenSendPool] = []
        self._active: dict[str, RaidenSendPool] = {}
        self._pending: dict[str, RaidenSendPool | None] = {}
        self._lock = threading.Lock()
        self._closed = False

    def stage_sync(self, transfer_id: str, embedding: jax.Array) -> Any:
        if embedding.ndim != 2 or embedding.shape[0] <= 0:
            raise ValueError("Raiden embedding must be a non-empty matrix")

        with self._lock:
            if self._closed:
                raise RuntimeError("Raiden encoder transfer is closed")
            if transfer_id in self._active or transfer_id in self._pending:
                raise ValueError(f"duplicate Raiden transfer_id: {transfer_id}")
            self._pending[transfer_id] = None
            pool = next(
                (pool for pool in self._pools if pool.matches(embedding)),
                None,
            )
            if pool is None:
                pool = RaidenSendPool(
                    self._host_ip,
                    embedding,
                    capacity=self._pool_size,
                    parallelism=self._parallelism,
                    timeout_s=self._timeout_s,
                )
                self._pools.append(pool)

        try:
            slot = pool.reserve_sync(transfer_id)
            pool.copy_sync(embedding, slot)
        except BaseException:
            with self._lock:
                self._pending.pop(transfer_id, None)
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
            metadata = pool.register(transfer_id, slot)
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
        return metadata

    async def publish(self, staged_transfer: Any) -> dict[str, Any]:
        return self.publish_sync(staged_transfer)

    def poll_completed(self) -> None:
        with self._lock:
            pools = list(self._pools)
        for pool in pools:
            try:
                sent, failed = pool.poll()
            except Exception:
                logger.exception("Raiden encoder sender pool poll failed")
                continue
            for transfer_id in sent:
                self._discard_active(transfer_id, event="sent")
            for transfer_id in failed:
                self._discard_active(transfer_id, event="failed")

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
            self._active.clear()
            self._pending.clear()
        for pool in pools:
            pool.close()
        for transfer_id in active:
            self._log_inflight_event("close", transfer_id)

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
        self._pools: dict[tuple[tuple[int, int], jnp.dtype], RaidenReceivePool] = {}
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
        key = (shape, dtype)
        with self._pool_lock:
            if self._closed:
                raise RuntimeError("Raiden receiver is closed")
            pool = self._pools.get(key)
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
                self._pools[key] = pool
        return pool.start(
            transfer_id,
            transfer_uuid,
            remote_endpoints,
            remote_block_ids,
        )

    def close(self) -> None:
        with self._pool_lock:
            self._closed = True
            pools = list(self._pools.values())
        for pool in pools:
            pool.close()
        self._executor.shutdown(cancel_futures=True)
