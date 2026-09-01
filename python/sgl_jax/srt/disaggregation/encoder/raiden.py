from __future__ import annotations

import asyncio
import hashlib
import logging
import threading
import time
from collections.abc import Callable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Protocol

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
    """Publish encoder output through short-lived Raiden transfer sessions."""

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

    @property
    def publish_group_size(self) -> int:
        """Maximum group that can enter the data plane without queueing channels."""
        return self._parallelism

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

    async def publish_batch(
        self,
        items: list[tuple[str, jax.Array]],
    ) -> list[dict[str, Any]]:
        """Publish bounded contiguous groups without imposing a full-batch barrier."""
        if not items:
            return []
        if len(items) == 1:
            transfer_id, embedding = items[0]
            return [await self.publish(transfer_id, embedding)]

        transfer_ids = [transfer_id for transfer_id, _ in items]
        if len(set(transfer_ids)) != len(transfer_ids):
            raise ValueError("duplicate Raiden transfer_id in batch")
        embeddings = [embedding for _, embedding in items]
        hidden_size = embeddings[0].shape[-1] if embeddings[0].ndim == 2 else None
        dtype = embeddings[0].dtype
        if any(
            embedding.ndim != 2
            or embedding.shape[0] <= 0
            or embedding.shape[1] != hidden_size
            or embedding.dtype != dtype
            for embedding in embeddings
        ):
            raise ValueError("batched Raiden embeddings must have matching width and dtype")

        groups = [
            items[offset : offset + self._parallelism]
            for offset in range(0, len(items), self._parallelism)
        ]
        results = await asyncio.gather(
            *(self._publish_group(group) for group in groups),
            return_exceptions=True,
        )
        errors = [result for result in results if isinstance(result, BaseException)]
        if errors:
            for result in results:
                if isinstance(result, list) and result:
                    self.release(result[0]["transfer_id"])
            raise errors[0]
        return [item for group in results if isinstance(group, list) for item in group]

    async def _publish_group(
        self,
        items: list[tuple[str, jax.Array]],
    ) -> list[dict[str, Any]]:
        transfer_ids = [transfer_id for transfer_id, _ in items]
        digest = _uuid_to_int("\0".join(transfer_ids))
        group_id = f"{transfer_ids[0]}:batch:{digest}"
        if group_id in self._sessions or group_id in self._preparing:
            raise ValueError(f"duplicate Raiden transfer_id: {group_id}")

        offsets = []
        offset = 0
        embeddings = []
        for _, embedding in items:
            offsets.append(offset)
            offset += embedding.shape[0]
            embeddings.append(embedding)
        packed = jnp.concatenate(embeddings, axis=0)

        self._preparing.add(group_id)
        try:
            session, common = await asyncio.get_running_loop().run_in_executor(
                self._executor,
                self._prepare,
                group_id,
                packed,
            )
        finally:
            self._preparing.discard(group_id)
        self._sessions[group_id] = session

        transfer_shape = tuple(int(dim) for dim in packed.shape)
        return [
            {
                **common,
                "transfer_group_size": len(items),
                "transfer_shape": transfer_shape,
                "transfer_offset": item_offset,
            }
            for item_offset in offsets
        ]

    def _prepare(
        self,
        transfer_id: str,
        embedding: jax.Array,
    ) -> tuple[RaidenTransferWrapper, dict[str, Any]]:
        # Treat one embedding as one physical major slice. The leading transfer
        # axis makes TPU tile padding part of the slice instead of row stride.
        buffer = embedding[jnp.newaxis, ...]
        jax.block_until_ready(buffer)
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
    lane_id: int = 0
    pool: _RaidenReceivePool | None = None
    _done: bool = False

    def poll(self) -> jax.Array | None:
        if self._done:
            return None
        if self.pool is not None:
            result = self.pool.poll(self.transfer_id, self.lane_id)
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


class _RaidenPollingSession(Protocol):
    def poll(self) -> jax.Array | None: ...

    def close(self) -> None: ...


class _RaidenReceivePool:
    """Reusable single-block Raiden lanes for equal-shaped embeddings."""

    def __init__(
        self,
        host: str,
        shape: tuple[int, int],
        dtype: jnp.dtype,
        sharding: jax.sharding.Sharding,
        parallelism: int,
        timeout_s: float,
    ) -> None:
        self._sharding = sharding
        self._timeout_s = timeout_s
        buffers = [jnp.zeros((1, *shape), dtype=dtype, device=sharding) for _ in range(parallelism)]
        jax.block_until_ready(buffers)
        self._lanes: list[tuple[jax.Array, RaidenTransferWrapper]] = []
        for buffer in buffers:
            transfer = RaidenTransferWrapper(host, 0, parallelism=parallelism)
            transfer.start(
                [buffer],
                max_blocks=1,
                num_slots=1,
                timeout_s=timeout_s,
            )
            self._lanes.append((buffer, transfer))
        self._condition = threading.Condition()
        self._free = list(range(parallelism - 1, -1, -1))
        self._active: dict[str, int] = {}
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
                self._reap_abandoned_locked()
                if self._free:
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("timed out waiting for a Raiden embedding buffer")
                self._condition.wait(min(remaining, 0.01))
            if self._closed:
                raise RuntimeError("Raiden receiver is closed")
            if transfer_id in self._active:
                raise ValueError(f"duplicate Raiden transfer_id: {transfer_id}")
            lane_id = self._free.pop()
            self._active[transfer_id] = lane_id
            buffer, transfer = self._lanes[lane_id]
            try:
                transfer.start_read(
                    transfer_id,
                    transfer_uuid,
                    remote_endpoints,
                    remote_block_ids,
                    [0],
                )
            except Exception:
                self._release_locked(transfer_id)
                raise
        return RaidenReceiveSession(
            transfer_id,
            buffer,
            transfer,
            lane_id=lane_id,
            pool=self,
        )

    def poll(self, transfer_id: str, lane_id: int) -> jax.Array | None:
        with self._condition:
            if self._active.get(transfer_id) != lane_id:
                raise RuntimeError(f"Raiden embedding lane changed: {transfer_id}")
            buffer, transfer = self._lanes[lane_id]
            _, received, failed = transfer.poll_stats()
            if transfer_id in failed:
                self._release_locked(transfer_id)
                raise RuntimeError(f"Raiden embedding transfer failed: {transfer_id}")
            if transfer_id not in received:
                return None
            if transfer_id in self._materializing:
                return None
            self._materializing.add(transfer_id)

        try:
            # Raiden writes outside JAX's dependency graph. Copy the completed
            # block and synchronize it before making the pool slot reusable.
            embedding = jax.device_put(
                buffer[0],
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
                self._reap_abandoned_locked()
            except Exception:
                logger.exception("Raiden receiver poll failed while abandoning %s", transfer_id)

    def _reap_abandoned_locked(self) -> None:
        for transfer_id in list(self._abandoned):
            lane_id = self._active.get(transfer_id)
            if lane_id is None or transfer_id in self._materializing:
                continue
            _, received, failed = self._lanes[lane_id][1].poll_stats()
            if transfer_id in received or transfer_id in failed:
                self._release_locked(transfer_id)

    def _release_locked(self, transfer_id: str) -> None:
        lane_id = self._active.pop(transfer_id, None)
        self._abandoned.discard(transfer_id)
        self._materializing.discard(transfer_id)
        if lane_id is not None:
            self._free.append(lane_id)
            self._condition.notify()

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()


class DeferredRaidenReceiveSession:
    """Expose a non-blocking session while Raiden setup runs off-loop."""

    def __init__(self, future: Future[_RaidenPollingSession]) -> None:
        self._future = future
        self._session: _RaidenPollingSession | None = None
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
    def _close_session(future: Future[_RaidenPollingSession]) -> None:
        if future.cancelled():
            return
        try:
            future.result().close()
        except Exception:
            logger.exception("Deferred Raiden receiver setup failed during cleanup")


class _RaidenReceiveGroup:
    """Fan one physical transfer out to request-sized immutable JAX views."""

    def __init__(
        self,
        transfer_id: str,
        shape: tuple[int, int],
        size: int,
        session: RaidenReceiveSession,
        on_done: Callable[[str, _RaidenReceiveGroup], None],
    ) -> None:
        self.transfer_id = transfer_id
        self.shape = shape
        self.size = size
        self._session = session
        self._on_done = on_done
        self._members: set[str] = set()
        self._finished: set[str] = set()
        self._buffer: jax.Array | None = None
        self._lock = threading.Lock()

    def attach(
        self,
        member_id: str,
        offset: int,
        shape: tuple[int, int],
    ) -> _GroupedRaidenReceiveSession:
        if shape[1] != self.shape[1] or offset < 0 or offset + shape[0] > self.shape[0]:
            raise ValueError("Raiden transfer group slice is out of bounds")
        with self._lock:
            if member_id in self._members:
                raise ValueError(f"duplicate Raiden transfer group member: {member_id}")
            if len(self._members) >= self.size:
                raise ValueError("Raiden transfer group has too many members")
            self._members.add(member_id)
        return _GroupedRaidenReceiveSession(self, member_id, offset, shape[0])

    def poll(self, member_id: str, offset: int, rows: int) -> jax.Array | None:
        with self._lock:
            if member_id in self._finished:
                return None
            if self._buffer is None:
                self._buffer = self._session.poll()
                if self._buffer is None:
                    return None
            result = self._buffer[offset : offset + rows]
            self._finish_locked(member_id)
            return result

    def finish(self, member_id: str) -> None:
        with self._lock:
            self._finish_locked(member_id)

    def close(self) -> None:
        with self._lock:
            self._session.close()
            self._finished.update(self._members)

    def _finish_locked(self, member_id: str) -> None:
        if member_id in self._finished:
            return
        self._finished.add(member_id)
        if len(self._finished) == self.size:
            self._session.close()
            self._on_done(self.transfer_id, self)


@dataclass(slots=True)
class _GroupedRaidenReceiveSession:
    group: _RaidenReceiveGroup
    member_id: str
    offset: int
    rows: int
    _done: bool = False

    def poll(self) -> jax.Array | None:
        if self._done:
            return None
        result = self.group.poll(self.member_id, self.offset, self.rows)
        self._done = result is not None
        return result

    def close(self) -> None:
        if not self._done:
            self.group.finish(self.member_id)
        self._done = True


class RaidenReceiverBackend:
    def __init__(
        self,
        host: str,
        sharding: jax.sharding.Sharding,
        parallelism: int,
        transfer_timeout_s: float,
    ) -> None:
        self._host = host
        self._sharding = sharding
        self._parallelism = max(1, int(parallelism))
        self._transfer_timeout_s = float(transfer_timeout_s)
        self._pools: dict[tuple[tuple[int, int], jnp.dtype], _RaidenReceivePool] = {}
        self._pool_lock = threading.Lock()
        self._groups: dict[str, _RaidenReceiveGroup] = {}
        self._group_lock = threading.Lock()
        self._closed = False
        # Pool creation and Raiden control-plane calls stay off the event loop.
        self._executor = ThreadPoolExecutor(max_workers=1)

    def start(self, data: EmbeddingData) -> DeferredRaidenReceiveSession:
        return DeferredRaidenReceiveSession(self._executor.submit(self._start, data))

    def _start(self, data: EmbeddingData) -> _RaidenPollingSession:
        if data.shape is None or data.dtype is None:
            raise ValueError("embedding shape and dtype are required")
        shape = tuple(int(dim) for dim in data.shape)
        if len(shape) != 2 or shape[0] <= 0:
            raise ValueError("Raiden embedding must be a non-empty matrix")

        group_size = int(getattr(data, "transfer_group_size", 1))
        if group_size <= 1:
            return self._start_transfer(data, shape)
        transfer_shape = tuple(int(dim) for dim in getattr(data, "transfer_shape", ()))
        offset = int(getattr(data, "transfer_offset", -1))
        if (
            len(transfer_shape) != 2
            or transfer_shape[0] <= 0
            or transfer_shape[1] != shape[1]
            or offset < 0
            or offset + shape[0] > transfer_shape[0]
        ):
            raise ValueError("Raiden transfer group metadata does not match embedding shape")

        transfer_id = getattr(data, "transfer_id", None)
        if not transfer_id:
            raise ValueError("Raiden transfer identity is incomplete")
        with self._group_lock:
            group = self._groups.get(transfer_id)
        if group is None:
            candidate = _RaidenReceiveGroup(
                transfer_id,
                transfer_shape,
                group_size,
                self._start_transfer(data, transfer_shape),
                self._release_group,
            )
            with self._group_lock:
                if self._closed:
                    candidate.close()
                    raise RuntimeError("Raiden receiver is closed")
                group = self._groups.setdefault(transfer_id, candidate)
            if group is not candidate:
                candidate.close()
        elif group.shape != transfer_shape or group.size != group_size:
            raise ValueError("inconsistent Raiden transfer group metadata")

        member_id = f"{data.req_id}:{data.part_idx}"
        return group.attach(member_id, offset, shape)

    def _start_transfer(
        self,
        data: EmbeddingData,
        shape: tuple[int, int],
    ) -> RaidenReceiveSession:
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
                    self._transfer_timeout_s,
                )
                self._pools[key] = pool
        return pool.start(
            transfer_id,
            transfer_uuid,
            remote_endpoints,
            remote_block_ids,
        )

    def _release_group(self, transfer_id: str, group: _RaidenReceiveGroup) -> None:
        with self._group_lock:
            if self._groups.get(transfer_id) is group:
                self._groups.pop(transfer_id, None)

    def close(self) -> None:
        with self._group_lock:
            groups = list(self._groups.values())
            self._groups.clear()
        for group in groups:
            group.close()
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
        transfer_timeout_s=transfer_timeout,
    )
    return EncoderClient(
        host=host,
        backend=backend,
        encoder_urls=server_args.encoder_urls,
        executor=executor,
        registration_timeout=None if control_timeout <= 0 else control_timeout,
    )
