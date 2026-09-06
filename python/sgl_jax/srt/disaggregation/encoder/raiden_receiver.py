"""Receive Raiden embeddings and retain slots until their readers finish."""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from sgl_jax.srt.disaggregation.encoder.client import DeferredReceiveSession
from sgl_jax.srt.disaggregation.encoder.embedding_data import EmbeddingData
from sgl_jax.srt.disaggregation.encoder.raiden_pool import _pool_sharding
from sgl_jax.srt.disaggregation.encoder.transfer_layout import encoder_pool_block_shape
from sgl_jax.srt.disaggregation.raiden_transfer.wrapper import RaidenTransferWrapper
from sgl_jax.srt.multimodal.in_model.embedding_view import PooledEmbedding

logger = logging.getLogger(__name__)
_LOCAL_ENDPOINT_HOSTS = {"", "0.0.0.0", "127.0.0.1", "::", "::1", "localhost"}


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
        if len(shape) != 2 or min(shape) <= 0:
            raise ValueError("Raiden embedding must be a non-empty matrix")
        transfer_id = data.transfer.get("transfer_id")
        transfer_uuid = data.transfer.get("transfer_uuid")
        remote_block_ids = data.transfer.get("transfer_block_ids")
        endpoints = data.transfer.get("transfer_address")
        if not transfer_id or not isinstance(transfer_uuid, int):
            raise ValueError("Raiden transfer identity is incomplete")
        if not isinstance(remote_block_ids, list) or len(remote_block_ids) != 1:
            raise ValueError("Raiden block metadata does not match embedding shape")
        remote_block_ids = [int(block_id) for block_id in remote_block_ids]
        if remote_block_ids[0] < 0:
            raise ValueError("Raiden remote block ID must be non-negative")

        transfer_host = data.transfer.get("transfer_host")
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


@dataclass(slots=True)
class RaidenReceiveSession:
    transfer_id: str
    slot: int
    pool: RaidenReceivePool
    _done: bool = False

    def poll(self, *, refresh_backend: bool = True) -> PooledEmbedding | None:
        if self._done:
            return None
        result = self.pool.poll(
            self.transfer_id,
            self.slot,
            refresh_backend=refresh_backend,
        )
        self._done = result is not None
        return result

    def close(self) -> None:
        if not self._done:
            self.pool.abandon(self.transfer_id)


class RaidenReceiveLease:
    """Release a receive slot only after its Language-side readers finish."""

    def __init__(self, pool: RaidenReceivePool, transfer_id: str, slot: int) -> None:
        self._pool = pool
        self._transfer_id = transfer_id
        self._slot = slot
        self._lock = threading.Lock()
        self._released = False

    def release_after(self, dependency: Any) -> None:
        with self._lock:
            if self._released:
                return
            self._released = True
        self._pool.release_after(
            self._transfer_id,
            self._slot,
            dependency,
        )

    def release(self) -> None:
        self.release_after(())


@dataclass(slots=True)
class _ReceiveState:
    slot: int
    received: bool = False
    failed: bool = False
    read_dependencies: tuple[jax.Array, ...] = ()


class RaidenReceivePool:
    """Keep a slot from start_read through the last Language-side consumer."""

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
        self.shape = shape
        self.dtype = jnp.dtype(dtype)
        self._timeout_s = timeout_s
        self._block_shape = encoder_pool_block_shape(shape)
        self._buffer = jnp.zeros(
            (capacity, *self._block_shape), dtype=dtype, device=_pool_sharding(sharding)
        )
        jax.block_until_ready(self._buffer)
        self._transfer = RaidenTransferWrapper(host, 0, parallelism=parallelism)
        self._transfer.start([self._buffer], max_blocks=1, num_slots=capacity, timeout_s=timeout_s)
        self._condition = threading.Condition()
        self._free_slots = list(range(capacity - 1, -1, -1))
        self._receives: dict[str, _ReceiveState] = {}
        self._pending_releases: dict[str, _ReceiveState] = {}
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
            if transfer_id in self._receives:
                raise ValueError(f"duplicate Raiden transfer_id: {transfer_id}")
            while not self._free_slots and not self._closed:
                self._progress_locked()
                if self._free_slots:
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("timed out waiting for a Raiden embedding buffer")
                self._condition.wait(min(remaining, 0.01))
            if self._closed:
                raise RuntimeError("Raiden receiver is closed")
            slot = self._free_slots.pop()
            self._receives[transfer_id] = _ReceiveState(slot)
            try:
                self._transfer.start_read(
                    transfer_id, transfer_uuid, remote_endpoints, remote_block_ids, [slot]
                )
            except Exception:
                self._release_locked(transfer_id)
                raise
        return RaidenReceiveSession(transfer_id, slot, self)

    def poll(
        self, transfer_id: str, slot: int, *, refresh_backend: bool = True
    ) -> PooledEmbedding | None:
        with self._condition:
            state = self._receives.get(transfer_id)
            if state is None or state.slot != slot:
                raise RuntimeError(f"Raiden embedding slot changed: {transfer_id}")
            if refresh_backend:
                self._drain_stats_locked()
            if state.failed:
                self._release_locked(transfer_id)
                raise RuntimeError(f"Raiden embedding transfer failed: {transfer_id}")
            if not state.received:
                return None
            return PooledEmbedding(
                self._buffer,
                slot,
                self._block_shape,
                self.shape,
                RaidenReceiveLease(self, transfer_id, slot),
            )

    def progress(self) -> None:
        """Drain completions once for all sessions and retire unused slots."""
        with self._condition:
            if not self._closed:
                self._progress_locked()

    def release_after(self, transfer_id: str, slot: int, dependency: Any) -> None:
        readers = tuple(
            leaf for leaf in jax.tree_util.tree_leaves(dependency) if isinstance(leaf, jax.Array)
        )
        with self._condition:
            state = self._receives.get(transfer_id)
            if state is None or state.slot != slot:
                return
            state.read_dependencies = readers
            if (state.received or state.failed) and all(reader.is_ready() for reader in readers):
                self._release_locked(transfer_id)
            else:
                self._pending_releases[transfer_id] = state

    def release(self, transfer_id: str, slot: int) -> None:
        self.release_after(transfer_id, slot, ())

    def abandon(self, transfer_id: str) -> None:
        with self._condition:
            state = self._receives.get(transfer_id)
            if state is None:
                return
            self._pending_releases[transfer_id] = state
            try:
                self._progress_locked()
            except Exception:
                logger.exception("Raiden receiver poll failed while abandoning %s", transfer_id)

    def _progress_locked(self) -> None:
        self._drain_stats_locked()
        self._release_completed_locked()

    def _drain_stats_locked(self) -> None:
        _, received, failed = self._transfer.poll_stats()
        for transfer_id in received:
            if state := self._receives.get(transfer_id):
                state.received = True
        for transfer_id in failed:
            if state := self._receives.get(transfer_id):
                state.failed = True

    def _release_completed_locked(self) -> None:
        for transfer_id, state in list(self._pending_releases.items()):
            if (state.received or state.failed) and all(
                reader.is_ready() for reader in state.read_dependencies
            ):
                self._release_locked(transfer_id)

    def _release_locked(self, transfer_id: str) -> None:
        state = self._receives.pop(transfer_id, None)
        self._pending_releases.pop(transfer_id, None)
        if state is not None:
            self._free_slots.append(state.slot)
            self._condition.notify()

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()
