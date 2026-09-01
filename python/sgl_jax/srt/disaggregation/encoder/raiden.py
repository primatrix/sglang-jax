from __future__ import annotations

import asyncio
import hashlib
import logging
import threading
import time
from collections.abc import Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from sgl_jax.raiden import require_raiden_preloaded
from sgl_jax.srt.disaggregation.encoder.client import EncoderClient
from sgl_jax.srt.disaggregation.encoder.embedding_data import EmbeddingData
from sgl_jax.srt.disaggregation.raiden_transfer.wrapper import RaidenTransferWrapper

logger = logging.getLogger(__name__)
_LOCAL_ENDPOINT_HOSTS = {"", "0.0.0.0", "127.0.0.1", "::", "::1", "localhost"}


def _uuid_to_int(value: str) -> int:
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") & ((1 << 50) - 1)


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
    """Binds each produced embedding to its own Raiden transfer session."""

    def __init__(
        self,
        host_ip: str,
        *,
        parallelism: int = 1,
        setup_parallelism: int | None = None,
        timeout_s: float = 300.0,
        poll_interval_s: float = 0.001,
    ) -> None:
        require_raiden_preloaded()
        self._host_ip = host_ip
        self._parallelism = max(1, int(parallelism))
        self._setup_parallelism = max(
            1,
            int(setup_parallelism if setup_parallelism is not None else parallelism),
        )
        self._timeout_s = float(timeout_s)
        self._poll_interval_s = float(poll_interval_s)
        self._sessions: dict[str, RaidenTransferWrapper] = {}
        self._preparing: set[str] = set()
        # Starting a manager is control-plane work. Do not serialize a request
        # batch behind Raiden's per-transfer data-plane channel count.
        self._executor = ThreadPoolExecutor(max_workers=self._setup_parallelism)

    async def publish(self, transfer_id: str, embedding: jax.Array) -> dict[str, Any]:
        if transfer_id in self._sessions or transfer_id in self._preparing:
            raise ValueError(f"duplicate Raiden transfer_id: {transfer_id}")
        if embedding.ndim != 2 or embedding.shape[0] <= 0:
            raise ValueError("Raiden embedding must be a non-empty matrix")

        self._preparing.add(transfer_id)
        try:
            session, metadata = await asyncio.get_running_loop().run_in_executor(
                self._executor,
                self._prepare,
                transfer_id,
                embedding,
            )
        finally:
            self._preparing.discard(transfer_id)
        self._sessions[transfer_id] = session
        return metadata

    def _prepare(
        self,
        transfer_id: str,
        embedding: jax.Array,
    ) -> tuple[RaidenTransferWrapper, dict[str, Any]]:
        # Treat one embedding as one physical major slice. The leading transfer
        # axis makes TPU tile padding part of the slice instead of row stride.
        buffer = embedding[jnp.newaxis, ...]
        block_ids = [0]
        transfer_uuid = _uuid_to_int(transfer_id)
        session = RaidenTransferWrapper(
            self._host_ip,
            0,
            parallelism=self._parallelism,
        )
        session.start(
            [buffer],
            max_blocks=1,
            num_slots=1,
            timeout_s=self._timeout_s,
        )
        if not session.register_read(transfer_id, transfer_uuid, block_ids):
            raise RuntimeError(f"Raiden rejected encoder transfer {transfer_id!r}")
        return (
            session,
            {
                "transfer_id": transfer_id,
                "transfer_uuid": transfer_uuid,
                "transfer_address": session.endpoints,
                "transfer_host": self._host_ip,
                "transfer_block_ids": block_ids,
            },
        )

    async def release_completed(self) -> None:
        while True:
            for transfer_id, session in list(self._sessions.items()):
                try:
                    sent, _, _ = session.poll_stats()
                except Exception:
                    logger.exception("Raiden encoder sender poll failed for %s", transfer_id)
                    self._sessions.pop(transfer_id, None)
                    continue
                if transfer_id in sent:
                    self._sessions.pop(transfer_id, None)
            await asyncio.sleep(self._poll_interval_s)

    def release(self, transfer_id: str) -> None:
        self._sessions.pop(transfer_id, None)

    def close(self) -> None:
        self._sessions.clear()
        self._executor.shutdown(cancel_futures=True)


@dataclass(slots=True)
class RaidenReceiveSession:
    transfer_id: str
    buffer: jax.Array
    transfer: RaidenTransferWrapper
    block_id: int = 0
    pool: _RaidenReceivePool | None = None
    _done: bool = False

    def poll(self) -> jax.Array | None:
        if self._done:
            return None
        if self.pool is not None:
            result = self.pool.poll(self.transfer_id, self.block_id)
            self._done = result is not None
            return result
        _, received, failed = self.transfer.poll_stats()
        if self.transfer_id in failed:
            raise RuntimeError(f"Raiden embedding transfer failed: {self.transfer_id}")
        if self.transfer_id in received:
            self._done = True
            return self.buffer[self.block_id]
        return None

    def close(self) -> None:
        if self.pool is not None and not self._done:
            self.pool.abandon(self.transfer_id)


class _RaidenReceivePool:
    """One reusable Raiden manager for equal-shaped embedding buffers."""

    def __init__(
        self,
        host: str,
        shape: tuple[int, int],
        dtype: jnp.dtype,
        sharding: jax.sharding.Sharding,
        parallelism: int,
        capacity: int,
        timeout_s: float,
    ) -> None:
        self._sharding = sharding
        self._timeout_s = timeout_s
        self.buffer = jnp.zeros((capacity, *shape), dtype=dtype, device=sharding)
        jax.block_until_ready(self.buffer)
        self.transfer = RaidenTransferWrapper(host, 0, parallelism=parallelism)
        self.transfer.start(
            [self.buffer],
            max_blocks=capacity,
            num_slots=min(capacity, parallelism),
            timeout_s=timeout_s,
        )
        self._condition = threading.Condition()
        self._free = list(range(capacity - 1, -1, -1))
        self._active: dict[str, int] = {}
        self._completed: set[str] = set()
        self._failed: set[str] = set()
        self._abandoned: set[str] = set()
        self._materializing: set[str] = set()
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
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("timed out waiting for a Raiden embedding buffer")
                self._condition.wait(remaining)
            if self._closed:
                raise RuntimeError("Raiden receiver is closed")
            if transfer_id in self._active:
                raise ValueError(f"duplicate Raiden transfer_id: {transfer_id}")
            block_id = self._free.pop()
            self._active[transfer_id] = block_id
            try:
                self.transfer.start_read(
                    transfer_id,
                    transfer_uuid,
                    remote_endpoints,
                    remote_block_ids,
                    [block_id],
                )
            except Exception:
                self._release_locked(transfer_id)
                raise
        return RaidenReceiveSession(
            transfer_id,
            self.buffer,
            self.transfer,
            block_id=block_id,
            pool=self,
        )

    def poll(self, transfer_id: str, block_id: int) -> jax.Array | None:
        with self._condition:
            self._drain_locked()
            if transfer_id in self._failed:
                self._release_locked(transfer_id)
                raise RuntimeError(f"Raiden embedding transfer failed: {transfer_id}")
            if transfer_id not in self._completed:
                return None
            if self._active.get(transfer_id) != block_id:
                raise RuntimeError(f"Raiden embedding block changed: {transfer_id}")
            if transfer_id in self._materializing:
                return None
            self._materializing.add(transfer_id)

        try:
            # Raiden writes outside JAX's dependency graph. Copy the completed
            # block and synchronize it before making the pool slot reusable.
            embedding = jax.device_put(
                self.buffer[block_id],
                self._sharding,
                may_alias=False,
            )
            jax.block_until_ready(embedding)
            return embedding
        finally:
            with self._condition:
                self._release_locked(transfer_id)

    def abandon(self, transfer_id: str) -> None:
        with self._condition:
            if transfer_id not in self._active:
                return
            self._abandoned.add(transfer_id)
            try:
                self._drain_locked()
            except Exception:
                logger.exception("Raiden receiver poll failed while abandoning %s", transfer_id)

    def _drain_locked(self) -> None:
        _, received, failed = self.transfer.poll_stats()
        self._completed.update(received)
        self._failed.update(failed)
        for transfer_id in self._abandoned & (self._completed | self._failed):
            if transfer_id not in self._materializing:
                self._release_locked(transfer_id)

    def _release_locked(self, transfer_id: str) -> None:
        block_id = self._active.pop(transfer_id, None)
        self._completed.discard(transfer_id)
        self._failed.discard(transfer_id)
        self._abandoned.discard(transfer_id)
        self._materializing.discard(transfer_id)
        if block_id is not None:
            self._free.append(block_id)
            self._condition.notify()

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()


class DeferredRaidenReceiveSession:
    """Expose a non-blocking session while Raiden setup runs off-loop."""

    def __init__(self, future: Future[RaidenReceiveSession]) -> None:
        self._future = future
        self._session: RaidenReceiveSession | None = None
        self._closed = False

    def poll(self) -> jax.Array | None:
        if self._closed:
            return None
        if self._session is None:
            if not self._future.done():
                return None
            self._session = self._future.result()
        return self._session.poll()

    def close(self) -> None:
        self._closed = True
        if self._session is not None:
            self._session.close()
        elif not self._future.cancel():
            self._future.add_done_callback(self._close_session)

    @staticmethod
    def _close_session(future: Future[RaidenReceiveSession]) -> None:
        if future.cancelled():
            return
        try:
            future.result().close()
        except Exception:
            logger.exception("Deferred Raiden receiver setup failed during cleanup")


class RaidenReceiverBackend:
    def __init__(
        self,
        host: str,
        sharding: jax.sharding.Sharding,
        parallelism: int,
        pool_capacity: int,
        transfer_timeout_s: float,
    ) -> None:
        self._host = host
        self._sharding = sharding
        self._parallelism = max(1, int(parallelism))
        self._pool_capacity = max(1, int(pool_capacity))
        self._transfer_timeout_s = float(transfer_timeout_s)
        self._pools: dict[tuple[tuple[int, int], jnp.dtype], _RaidenReceivePool] = {}
        self._pool_lock = threading.Lock()
        self._closed = False
        # Pool creation and Raiden control-plane calls stay off the event loop.
        self._executor = ThreadPoolExecutor(max_workers=1)

    def start(self, data: EmbeddingData) -> DeferredRaidenReceiveSession:
        return DeferredRaidenReceiveSession(self._executor.submit(self._start, data))

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
                pool = _RaidenReceivePool(
                    self._host,
                    shape,
                    dtype,
                    self._sharding,
                    self._parallelism,
                    self._pool_capacity,
                    self._transfer_timeout_s,
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


RaidenEncoderClient = EncoderClient


def create_raiden_client(
    server_args,
    mesh: jax.sharding.Mesh,
) -> EncoderClient:
    from sgl_jax.srt.disaggregation.host_ip import resolve_host_ip

    require_raiden_preloaded()
    host = resolve_host_ip(server_args.disaggregation_host_ip)
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    control_timeout = server_args.encoder_control_timeout_seconds
    transfer_timeout = server_args.encoder_request_timeout_seconds
    if transfer_timeout <= 0:
        raise ValueError("Raiden requires a positive encoder request timeout")
    executor = ThreadPoolExecutor(max_workers=server_args.disaggregation_channel_number)
    backend = RaidenReceiverBackend(
        host=host,
        sharding=sharding,
        parallelism=server_args.disaggregation_channel_number,
        pool_capacity=(
            max(1, int(server_args.encoder_max_batch_size))
            * max(1, int(server_args.encoder_max_inflight_batches))
        ),
        transfer_timeout_s=transfer_timeout,
    )
    return EncoderClient(
        host=host,
        backend=backend,
        encoder_urls=server_args.encoder_urls,
        executor=executor,
        registration_timeout=None if control_timeout <= 0 else control_timeout,
    )
