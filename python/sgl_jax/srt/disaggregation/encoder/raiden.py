from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import threading
import time
from collections.abc import Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from typing import Any, Protocol

import jax
import jax.numpy as jnp

from sgl_jax.raiden import require_raiden_preloaded
from sgl_jax.srt.disaggregation.encoder.client import EncoderClient
from sgl_jax.srt.disaggregation.encoder.embedding_data import EmbeddingData
from sgl_jax.srt.disaggregation.raiden_transfer.wrapper import RaidenTransferWrapper

logger = logging.getLogger(__name__)
_LOCAL_ENDPOINT_HOSTS = {"", "0.0.0.0", "127.0.0.1", "::", "::1", "localhost"}
_TPU_TILE_SHAPE = (8, 128)


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


def _pool_sharding(sharding: jax.sharding.Sharding) -> jax.sharding.Sharding:
    if isinstance(sharding, jax.sharding.NamedSharding):
        spec = jax.sharding.PartitionSpec(None, *tuple(sharding.spec))
        return jax.sharding.NamedSharding(sharding.mesh, spec)
    return sharding


def _pool_block_shape(shape: tuple[int, int]) -> tuple[int, int, int, int]:
    """Keep each request contiguous in TPU's canonical rank-5 KV layout."""

    rows, width = shape
    tile_width = math.prod(_TPU_TILE_SHAPE)
    return max(2, rows), max(2, math.ceil(width / tile_width)), *_TPU_TILE_SHAPE


def _pool_block(value: jax.Array, block_shape: tuple[int, ...]) -> jax.Array:
    padded_shape = (block_shape[0], math.prod(block_shape[1:]))
    padding = tuple((0, padded - size) for padded, size in zip(padded_shape, value.shape))
    return jnp.pad(value, padding).reshape(block_shape)


@partial(jax.jit, donate_argnums=(0,))
def _copy_into_slots(
    pool: jax.Array,
    slots: jax.Array,
    *values: jax.Array,
) -> jax.Array:
    for index, value in enumerate(values):
        block = _pool_block(value, pool.shape[1:])
        pool = jax.lax.dynamic_update_slice_in_dim(
            pool,
            block[None],
            slots[index],
            axis=0,
        )
    return pool


def _compile_donated_copy(
    pool: jax.Array,
    values: tuple[jax.Array, ...],
) -> Any:
    slots = jnp.arange(len(values), dtype=jnp.int32)
    compiled = _copy_into_slots.lower(pool, slots, *values).compile()
    stats = compiled.memory_analysis()
    stats = stats if isinstance(stats, (list, tuple)) else (stats,)
    if not stats or any(
        stat is None
        or int(getattr(stat, "alias_size_in_bytes", 0))
        < int(getattr(stat, "output_size_in_bytes", 0))
        for stat in stats
    ):
        raise RuntimeError("Raiden encoder pool update did not fully alias its donated input")
    return compiled


class _RaidenSendPool:
    """One registered source buffer with bounded, request-sized slots."""

    def __init__(
        self,
        host: str,
        sample: jax.Array,
        *,
        capacity: int,
        parallelism: int,
        timeout_s: float,
    ) -> None:
        self.shape = tuple(int(dim) for dim in sample.shape)
        self.dtype = sample.dtype
        self.sharding = sample.sharding
        self._timeout_s = timeout_s
        pool_sharding = _pool_sharding(sample.sharding)
        self._block_shape = _pool_block_shape(self.shape)
        self._buffer = jnp.zeros(
            (capacity, *self._block_shape),
            dtype=self.dtype,
            device=pool_sharding,
        )
        jax.block_until_ready(self._buffer)
        self._copies: dict[int, Any] = {}
        self.transfer = RaidenTransferWrapper(host, 0, parallelism=parallelism)
        self.transfer.start(
            [self._buffer],
            max_blocks=1,
            num_slots=capacity,
            timeout_s=timeout_s,
        )
        self._condition = threading.Condition()
        self._free = list(range(capacity - 1, -1, -1))
        self._active: dict[str, int] = {}
        self._closed = False

    def matches(self, value: jax.Array) -> bool:
        return (
            tuple(value.shape) == self.shape
            and value.dtype == self.dtype
            and value.sharding == self.sharding
        )

    def publish_batch(
        self,
        items: list[tuple[str, jax.Array]],
    ) -> list[dict[str, Any]]:
        if not items:
            return []
        transfer_ids = [transfer_id for transfer_id, _ in items]
        if len(set(transfer_ids)) != len(transfer_ids):
            raise ValueError("duplicate Raiden transfer_id in batch")
        if any(not self.matches(value) for _, value in items):
            raise ValueError("Raiden pool batch contains incompatible embeddings")

        deadline = time.monotonic() + self._timeout_s
        with self._condition:
            duplicate = next(
                (transfer_id for transfer_id in transfer_ids if transfer_id in self._active),
                None,
            )
            if duplicate is not None:
                raise ValueError(f"duplicate Raiden transfer_id: {duplicate}")
            if len(items) > len(self._buffer):
                raise ValueError("Raiden publish batch exceeds encoder pool capacity")
            while len(self._free) < len(items) and not self._closed:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("timed out waiting for Raiden encoder pool slots")
                self._condition.wait(remaining)
            if self._closed:
                raise RuntimeError("Raiden encoder pool is closed")
            slots = [self._free.pop() for _ in items]
            self._active.update(zip(transfer_ids, slots))

        registered = 0
        try:
            values = tuple(value for _, value in items)
            copy = self._copies.get(len(values))
            if copy is None:
                copy = _compile_donated_copy(self._buffer, values)
                self._copies[len(values)] = copy
            self._buffer = copy(
                self._buffer,
                jnp.asarray(slots, dtype=jnp.int32),
                *values,
            )
            jax.block_until_ready(self._buffer)

            metadata = []
            for (transfer_id, _), slot in zip(items, slots):
                transfer_uuid = _uuid_to_int(transfer_id)
                if not self.transfer.register_read(transfer_id, transfer_uuid, [slot]):
                    raise RuntimeError(f"Raiden rejected encoder transfer {transfer_id!r}")
                registered += 1
                metadata.append(
                    {
                        "transfer_id": transfer_id,
                        "transfer_uuid": transfer_uuid,
                        "transfer_address": self.transfer.endpoints,
                        "transfer_host": self.transfer.host_ip,
                        "transfer_block_ids": [slot],
                    }
                )
            return metadata
        except Exception:
            for transfer_id in transfer_ids[registered:]:
                self.release(transfer_id)
            raise

    def poll(self) -> tuple[list[str], list[str]]:
        sent, _, failed = self.transfer.poll_stats()
        for transfer_id in (*sent, *failed):
            self.release(transfer_id)
        return sent, failed

    def release(self, transfer_id: str) -> None:
        with self._condition:
            slot = self._active.pop(transfer_id, None)
            if slot is not None:
                self._free.append(slot)
                self._condition.notify()

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()


class RaidenEncoderServerTransfer:
    """Publish request-sized encoder outputs from registered source pools."""

    def __init__(
        self,
        host_ip: str,
        *,
        parallelism: int = 1,
        setup_parallelism: int | None = None,
        pool_size: int = 32,
        timeout_s: float = 300.0,
        poll_interval_s: float = 0.001,
        log_inflight: bool = False,
    ) -> None:
        require_raiden_preloaded()
        self._host_ip = host_ip
        self._parallelism = max(1, int(parallelism))
        self._setup_parallelism = max(
            1,
            int(setup_parallelism if setup_parallelism is not None else parallelism),
        )
        self._pool_size = max(1, int(pool_size))
        self._timeout_s = float(timeout_s)
        self._poll_interval_s = float(poll_interval_s)
        self._log_inflight = bool(log_inflight)
        self._pools: list[_RaidenSendPool] = []
        self._pool_lock = threading.Lock()
        self._active: dict[str, _RaidenSendPool] = {}
        self._preparing: set[str] = set()
        self._executor = ThreadPoolExecutor(max_workers=self._setup_parallelism)

    async def publish(self, transfer_id: str, embedding: jax.Array) -> dict[str, Any]:
        return (await self.publish_batch([(transfer_id, embedding)]))[0]

    async def publish_batch(
        self,
        items: list[tuple[str, jax.Array]],
    ) -> list[dict[str, Any]]:
        if not items:
            return []
        transfer_ids = [transfer_id for transfer_id, _ in items]
        if len(set(transfer_ids)) != len(transfer_ids):
            raise ValueError("duplicate Raiden transfer_id in batch")
        duplicate = next(
            (
                transfer_id
                for transfer_id in transfer_ids
                if transfer_id in self._active or transfer_id in self._preparing
            ),
            None,
        )
        if duplicate is not None:
            raise ValueError(f"duplicate Raiden transfer_id: {duplicate}")
        if any(embedding.ndim != 2 or embedding.shape[0] <= 0 for _, embedding in items):
            raise ValueError("Raiden embeddings must be non-empty matrices")

        self._preparing.update(transfer_ids)
        try:
            prepared = await asyncio.get_running_loop().run_in_executor(
                self._executor,
                self._prepare_batch,
                items,
            )
        finally:
            self._preparing.difference_update(transfer_ids)
        for transfer_id, (pool, _) in zip(transfer_ids, prepared):
            self._active[transfer_id] = pool
            self._log_inflight_event("start", transfer_id)
        return [metadata for _, metadata in prepared]

    def _prepare_batch(
        self,
        items: list[tuple[str, jax.Array]],
    ) -> list[tuple[_RaidenSendPool, dict[str, Any]]]:
        grouped: dict[_RaidenSendPool, list[tuple[int, str, jax.Array]]] = {}
        for index, (transfer_id, embedding) in enumerate(items):
            pool = self._pool_for(embedding)
            grouped.setdefault(pool, []).append((index, transfer_id, embedding))

        prepared: list[tuple[_RaidenSendPool, dict[str, Any]] | None] = [None] * len(items)
        for pool, group in grouped.items():
            metadata = pool.publish_batch(
                [(transfer_id, embedding) for _, transfer_id, embedding in group]
            )
            for (index, _, _), item_metadata in zip(group, metadata):
                prepared[index] = (pool, item_metadata)
        if any(item is None for item in prepared):
            raise RuntimeError("Raiden batch publication returned incomplete metadata")
        return [item for item in prepared if item is not None]

    def _pool_for(self, embedding: jax.Array) -> _RaidenSendPool:
        with self._pool_lock:
            pool = next((pool for pool in self._pools if pool.matches(embedding)), None)
            if pool is None:
                pool = _RaidenSendPool(
                    self._host_ip,
                    embedding,
                    capacity=self._pool_size,
                    parallelism=self._parallelism,
                    timeout_s=self._timeout_s,
                )
                self._pools.append(pool)
        return pool

    async def release_completed(self) -> None:
        while True:
            for pool in list(self._pools):
                try:
                    sent, failed = pool.poll()
                except Exception:
                    logger.exception("Raiden encoder sender pool poll failed")
                    continue
                for transfer_id in sent:
                    self._discard_active(transfer_id, event="sent")
                for transfer_id in failed:
                    self._discard_active(transfer_id, event="failed")
            await asyncio.sleep(self._poll_interval_s)

    def release(self, transfer_id: str) -> None:
        pool = self._active.pop(transfer_id, None)
        if pool is not None:
            pool.release(transfer_id)
            self._log_inflight_event("release", transfer_id)

    def close(self) -> None:
        for pool in self._pools:
            pool.close()
        for transfer_id in list(self._active):
            self._discard_active(transfer_id, event="close")
        self._executor.shutdown(cancel_futures=True)

    def _discard_active(self, transfer_id: str, *, event: str) -> None:
        if self._active.pop(transfer_id, None) is not None:
            self._log_inflight_event(event, transfer_id)

    def _log_inflight_event(self, event: str, transfer_id: str) -> None:
        if not self._log_inflight:
            return
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
    """One registered destination buffer with bounded request slots."""

    def __init__(
        self,
        host: str,
        shape: tuple[int, int],
        dtype: jnp.dtype,
        sharding: jax.sharding.Sharding,
        *,
        parallelism: int,
        capacity: int,
        timeout_s: float,
    ) -> None:
        self._sharding = sharding
        self._shape = shape
        self._timeout_s = timeout_s
        self._block_shape = _pool_block_shape(shape)
        self._buffer = jnp.zeros(
            (capacity, *self._block_shape),
            dtype=dtype,
            device=_pool_sharding(sharding),
        )
        jax.block_until_ready(self._buffer)
        self._transfer = RaidenTransferWrapper(host, 0, parallelism=parallelism)
        self._transfer.start(
            [self._buffer],
            max_blocks=1,
            num_slots=capacity,
            timeout_s=timeout_s,
        )
        self._condition = threading.Condition()
        self._free = list(range(capacity - 1, -1, -1))
        self._active: dict[str, int] = {}
        self._abandoned: set[str] = set()
        self._materializing: set[str] = set()
        self._received: set[str] = set()
        self._failed: set[str] = set()
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
            try:
                self._transfer.start_read(
                    transfer_id,
                    transfer_uuid,
                    remote_endpoints,
                    remote_block_ids,
                    [lane_id],
                )
            except Exception:
                self._release_locked(transfer_id)
                raise
        return RaidenReceiveSession(
            transfer_id,
            self._buffer,
            self._transfer,
            lane_id=lane_id,
            pool=self,
        )

    def poll(self, transfer_id: str, lane_id: int) -> jax.Array | None:
        with self._condition:
            if self._active.get(transfer_id) != lane_id:
                raise RuntimeError(f"Raiden embedding lane changed: {transfer_id}")
            self._drain_stats_locked()
            if transfer_id in self._failed:
                self._release_locked(transfer_id)
                raise RuntimeError(f"Raiden embedding transfer failed: {transfer_id}")
            if transfer_id not in self._received:
                return None
            if transfer_id in self._materializing:
                return None
            self._materializing.add(transfer_id)

        try:
            # Raiden writes outside JAX's dependency graph. Copy the completed
            # block and synchronize it before making the pool slot reusable.
            block = self._buffer[lane_id].reshape(self._block_shape[0], -1)
            embedding = jax.device_put(
                block[: self._shape[0], : self._shape[1]],
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
        self._drain_stats_locked()
        for transfer_id in list(self._abandoned):
            if transfer_id not in self._active or transfer_id in self._materializing:
                continue
            if transfer_id in self._received or transfer_id in self._failed:
                self._release_locked(transfer_id)

    def _drain_stats_locked(self) -> None:
        _, received, failed = self._transfer.poll_stats()
        self._received.update(received)
        self._failed.update(failed)

    def _release_locked(self, transfer_id: str) -> None:
        lane_id = self._active.pop(transfer_id, None)
        self._abandoned.discard(transfer_id)
        self._materializing.discard(transfer_id)
        self._received.discard(transfer_id)
        self._failed.discard(transfer_id)
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
        self._pools: dict[tuple[tuple[int, int], jnp.dtype], _RaidenReceivePool] = {}
        self._pool_lock = threading.Lock()
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
        return self._start_transfer(data, shape)

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
        pool_size=server_args.encoder_transfer_pool_size,
        transfer_timeout_s=transfer_timeout,
    )
    return EncoderClient(
        host=host,
        backend=backend,
        encoder_urls=server_args.encoder_urls,
        executor=executor,
        registration_timeout=None if control_timeout <= 0 else control_timeout,
    )
