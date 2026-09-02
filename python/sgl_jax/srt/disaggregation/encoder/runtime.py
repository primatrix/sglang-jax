from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from collections.abc import Awaitable, Callable
from contextlib import suppress
from typing import Any, Protocol

import jax
import zmq.asyncio
from zmq.constants import LINGER, PUSH

from sgl_jax.srt.disaggregation.encoder.embedding_data import EmbeddingData
from sgl_jax.srt.multimodal.common.modality_enum import Modality

EncodeResult = tuple[jax.Array, dict[str, Any]]
BatchEncodeFn = Callable[
    [list[dict[str, Any]]],
    Awaitable[list[EncodeResult]],
]
BatchPreprocessFn = Callable[[list[dict[str, Any]]], Awaitable[Any]]
BatchEncodePreprocessedFn = Callable[[Any], Awaitable[list[EncodeResult]]]
logger = logging.getLogger(__name__)


class PendingRequest:
    __slots__ = (
        "_enqueue_mono_ns",
        "dequeue_ns",
        "enqueue_ns",
        "future",
        "queue_duration_ns",
        "request",
    )

    def __init__(self, request: dict[str, Any]) -> None:
        self.request = request
        self.future: asyncio.Future[PublishedEmbedding] = asyncio.get_running_loop().create_future()
        self.enqueue_ns = time.time_ns()
        self._enqueue_mono_ns = time.monotonic_ns()
        self.dequeue_ns = 0
        self.queue_duration_ns = 0

    def mark_dequeued(self) -> None:
        self.dequeue_ns = time.time_ns()
        self.queue_duration_ns = max(0, time.monotonic_ns() - self._enqueue_mono_ns)


DispatchBatchFn = Callable[[list[PendingRequest]], Awaitable[None]]


class PublishedEmbedding:
    __slots__ = ("data", "req_id", "transfer_id")

    def __init__(self, req_id: str, transfer_id: str, data: EmbeddingData) -> None:
        self.req_id = req_id
        self.transfer_id = transfer_id
        self.data = data


class EncoderServerTransfer(Protocol):
    async def publish(self, transfer_id: str, embedding: jax.Array) -> dict[str, Any]: ...

    async def publish_batch(
        self,
        items: list[tuple[str, jax.Array]],
    ) -> list[dict[str, Any]]: ...

    async def release_completed(self) -> None: ...

    def release(self, transfer_id: str) -> None: ...

    def close(self) -> None: ...


class EncoderScheduler:
    """Collect queued requests and dispatch one modality group at a time."""

    def __init__(
        self,
        dispatch_batch: DispatchBatchFn,
        max_batch_size: int = 8,
        request_timeout: float | None = 300.0,
        log_queue_timing: bool = False,
        max_inflight_batches: int = 1,
    ) -> None:
        self._dispatch_batch = dispatch_batch
        self._max_batch_size = max(1, int(max_batch_size))
        self._max_inflight_batches = max(1, int(max_inflight_batches))
        self._request_timeout = request_timeout
        self._log_queue_timing = log_queue_timing
        self._pending_queue: asyncio.Queue[PendingRequest] = asyncio.Queue()
        self._inflight_tasks: set[asyncio.Task[None]] = set()
        self._worker_task: asyncio.Task[None] | None = None

    def start(self) -> None:
        if self._worker_task is None:
            self._worker_task = asyncio.create_task(self._batch_worker())

    async def stop(self) -> None:
        if self._worker_task is not None:
            self._worker_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._worker_task
            self._worker_task = None

        for task in self._inflight_tasks:
            task.cancel()
        if self._inflight_tasks:
            await asyncio.gather(*self._inflight_tasks, return_exceptions=True)
            self._inflight_tasks.clear()

        while True:
            try:
                pending = self._pending_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if not pending.future.done():
                pending.future.set_exception(RuntimeError("EncoderScheduler stopped"))
            self._pending_queue.task_done()

    async def submit(self, request: dict[str, Any]) -> PublishedEmbedding:
        if self._worker_task is None:
            raise RuntimeError("EncoderScheduler is not running")
        pending = PendingRequest(request)
        await self._pending_queue.put(pending)
        if self._request_timeout is None or self._request_timeout <= 0:
            return await pending.future
        return await asyncio.wait_for(pending.future, self._request_timeout)

    async def _collect_batch(self) -> list[PendingRequest]:
        first = await self._pending_queue.get()
        first.mark_dequeued()
        batch = [first]
        while len(batch) < self._max_batch_size:
            try:
                pending = self._pending_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            pending.mark_dequeued()
            batch.append(pending)

        if self._log_queue_timing:
            queue_depth = self._pending_queue.qsize()
            for pending in batch:
                queue_duration_ns = pending.queue_duration_ns
                logger.info(
                    "ENCODER-QUEUE-TIME req_id=%s part_idx=%s enqueue_ns=%d "
                    "dequeue_ns=%d queue_duration_ns=%d queue_ms=%.3f "
                    "batch_size=%d queue_depth=%d",
                    pending.request.get("req_id"),
                    pending.request.get("part_idx", 0),
                    pending.enqueue_ns,
                    pending.dequeue_ns,
                    queue_duration_ns,
                    queue_duration_ns / 1_000_000,
                    len(batch),
                    queue_depth,
                )
        return batch

    async def _batch_worker(self) -> None:
        while True:
            while len(self._inflight_tasks) >= self._max_inflight_batches:
                await asyncio.wait(
                    self._inflight_tasks,
                    return_when=asyncio.FIRST_COMPLETED,
                )

            batch = await self._collect_batch()
            task = asyncio.create_task(self._run_batch(batch))
            self._inflight_tasks.add(task)
            task.add_done_callback(self._inflight_tasks.discard)

    async def _run_batch(self, batch: list[PendingRequest]) -> None:
        try:
            groups: dict[Modality, list[PendingRequest]] = defaultdict(list)
            for pending in batch:
                modality = Modality.from_str(pending.request.get("modality", "image"))
                groups[modality].append(pending)
            for group in groups.values():
                await self._dispatch_batch(group)
        except asyncio.CancelledError:
            for pending in batch:
                if not pending.future.done():
                    pending.future.set_exception(RuntimeError("EncoderScheduler stopped"))
            raise
        except Exception as exc:
            logger.exception("Encoder batch failed")
            for pending in batch:
                if not pending.future.done():
                    pending.future.set_exception(exc)
        finally:
            for _ in batch:
                self._pending_queue.task_done()


class EncoderRuntime:
    """Owns Encoder execution state independently of the HTTP transport."""

    def __init__(
        self,
        batch_encode_fn: BatchEncodeFn,
        transfer: EncoderServerTransfer,
        *,
        batch_preprocess_fn: BatchPreprocessFn | None = None,
        batch_encode_preprocessed_fn: BatchEncodePreprocessedFn | None = None,
        receiver_timeout: float | None = 300.0,
        max_batch_size: int = 8,
        max_inflight_batches: int = 1,
        request_timeout: float | None = 300.0,
        log_queue_timing: bool = False,
    ) -> None:
        if (batch_preprocess_fn is None) != (batch_encode_preprocessed_fn is None):
            raise ValueError("preprocess and preprocessed encode functions must be paired")
        self._batch_encode_fn = batch_encode_fn
        self._batch_preprocess_fn = batch_preprocess_fn
        self._batch_encode_preprocessed_fn = batch_encode_preprocessed_fn
        self._transfer = transfer
        self._preprocess_lock = asyncio.Lock()
        self._encode_lock = asyncio.Lock()
        self._publish_lock = asyncio.Lock()

        self._zmq = zmq.asyncio.Context.instance()
        self._receiver_timeout = receiver_timeout
        self._receiver_addresses: dict[str, str] = {}
        self._receiver_events: dict[str, asyncio.Event] = {}
        self._receiver_sockets: dict[str, zmq.asyncio.Socket] = {}
        self._notify_lock = asyncio.Lock()
        self.scheduler = EncoderScheduler(
            self._dispatch_batch,
            max_batch_size=max_batch_size,
            max_inflight_batches=max_inflight_batches,
            request_timeout=request_timeout,
            log_queue_timing=log_queue_timing,
        )
        self._release_task: asyncio.Task[None] | None = None

    def start(self) -> None:
        self.scheduler.start()
        if self._release_task is None:
            self._release_task = asyncio.create_task(self._transfer.release_completed())

    async def stop(self) -> None:
        await self.scheduler.stop()

        if self._release_task is not None:
            self._release_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._release_task
            self._release_task = None
        for socket in self._receiver_sockets.values():
            socket.close()
        self._receiver_sockets.clear()
        self._transfer.close()

    async def register_scheduler_receiver(
        self,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        req_id = request["req_id"]
        self._receiver_addresses[req_id] = request["receive_url"]
        self._receiver_events.setdefault(req_id, asyncio.Event()).set()
        return {"req_id": req_id}

    async def submit(self, request: dict[str, Any]) -> dict[str, Any]:
        try:
            published = await self.scheduler.submit(request)
        except Exception as exc:
            try:
                await self._send_error(request, exc)
            except Exception:
                logger.exception(
                    "Encoder error delivery failed. req_id=%s",
                    request.get("req_id"),
                )
            raise

        try:
            await self.send_to_scheduler(published.req_id, published.data)
        except Exception:
            self._transfer.release(published.transfer_id)
            raise
        # The return value itself has no meaning; the client will not read it,
        # but it serves as an ACK, ensuring that the request is fully processed before returning.
        return {"req_id": request["req_id"]}

    async def _dispatch_batch(self, batch: list[PendingRequest]) -> None:
        pending_requests = [pending for pending in batch if not pending.future.done()]
        if not pending_requests:
            return

        try:
            requests = [pending.request for pending in pending_requests]
            if self._batch_preprocess_fn is None:
                async with self._encode_lock:
                    preprocess_start_ns = time.time_ns()
                    results = await self._encode_batch(requests)
            else:
                async with self._preprocess_lock:
                    preprocess_start_ns = time.time_ns()
                    prepared = await self._batch_preprocess_fn(requests)
                async with self._encode_lock:
                    assert self._batch_encode_preprocessed_fn is not None
                    results = await self._batch_encode_preprocessed_fn(prepared)
                    if len(results) != len(requests):
                        raise RuntimeError(
                            "batch_encode_preprocessed_fn returned "
                            f"{len(results)} results for {len(requests)} requests"
                        )
            encode_done_ns = time.time_ns()
        except Exception as exc:
            for pending in pending_requests:
                if not pending.future.done():
                    pending.future.set_exception(exc)
            return

        publish_items = [
            (pending, result)
            for pending, result in zip(pending_requests, results)
            if not pending.future.done()
        ]
        async with self._publish_lock:
            if len(publish_items) > 1:
                await self._publish_batch(
                    publish_items,
                    preprocess_start_ns,
                    encode_done_ns,
                )
            else:
                await asyncio.gather(
                    *(
                        self._publish_pending(
                            pending,
                            result,
                            preprocess_start_ns,
                            encode_done_ns,
                        )
                        for pending, result in publish_items
                    )
                )

    async def _publish_batch(
        self,
        items: list[tuple[PendingRequest, EncodeResult]],
        preprocess_start_ns: int,
        encode_done_ns: int,
    ) -> None:
        active = [(pending, result) for pending, result in items if not pending.future.done()]
        if not active:
            return
        try:
            await asyncio.gather(
                *(self._wait_for_receiver(pending.request["req_id"]) for pending, _ in active)
            )
            transfer_metadata = await self._transfer.publish_batch(
                [(self._transfer_id(pending.request), result[0]) for pending, result in active]
            )
            if len(transfer_metadata) != len(active):
                raise RuntimeError(
                    f"publish_batch returned {len(transfer_metadata)} results "
                    f"for {len(active)} embeddings"
                )
            publish_done_ns = time.time_ns()
            published = [
                self._build_published(
                    pending,
                    *result,
                    metadata,
                    preprocess_start_ns,
                    encode_done_ns,
                    publish_done_ns,
                )
                for (pending, result), metadata in zip(active, transfer_metadata)
            ]
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            for pending, _ in active:
                if not pending.future.done():
                    pending.future.set_exception(exc)
            return

        for (pending, _), item in zip(active, published):
            if pending.future.done():
                self._transfer.release(item.transfer_id)
            else:
                pending.future.set_result(item)

    async def _publish_pending(
        self,
        pending: PendingRequest,
        result: EncodeResult,
        preprocess_start_ns: int,
        encode_done_ns: int,
    ) -> None:
        try:
            published = await self._publish_result(
                pending,
                *result,
                preprocess_start_ns,
                encode_done_ns,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if not pending.future.done():
                pending.future.set_exception(exc)
            return

        if pending.future.done():
            self._transfer.release(published.transfer_id)
        else:
            pending.future.set_result(published)

    async def _encode_batch(self, requests: list[dict[str, Any]]) -> list[EncodeResult]:
        results = await self._batch_encode_fn(requests)
        # Preserve existing direct single-request callers while the public
        # runtime contract remains batch-only.
        if (
            len(requests) == 1
            and isinstance(results, tuple)
            and len(results) == 2
            and isinstance(results[1], dict)
        ):
            results = [results]
        if len(results) != len(requests):
            raise RuntimeError(
                f"batch_encode_fn returned {len(results)} results for {len(requests)} requests"
            )
        return results

    async def _publish_result(
        self,
        pending: PendingRequest,
        embedding: jax.Array,
        metadata: dict[str, Any],
        preprocess_start_ns: int,
        encode_done_ns: int,
    ) -> PublishedEmbedding:
        request = pending.request
        transfer_id = self._transfer_id(request)
        transfer_metadata = await self._transfer.publish(transfer_id, embedding)
        publish_done_ns = time.time_ns()

        return self._build_published(
            pending,
            embedding,
            metadata,
            transfer_metadata,
            preprocess_start_ns,
            encode_done_ns,
            publish_done_ns,
        )

    @staticmethod
    def _transfer_id(request: dict[str, Any]) -> str:
        return f"{request['req_id']}:{request.get('part_idx', 0)}:embedding"

    @staticmethod
    def _build_published(
        pending: PendingRequest,
        embedding: jax.Array,
        metadata: dict[str, Any],
        transfer_metadata: dict[str, Any],
        preprocess_start_ns: int,
        encode_done_ns: int,
        publish_done_ns: int,
    ) -> PublishedEmbedding:
        request = pending.request
        req_id = request["req_id"]
        modality = Modality.from_str(request["modality"])
        queue_duration_ns = pending.queue_duration_ns

        metadata = dict(metadata)
        encoder_timing = {
            "encode_done_ns": encode_done_ns,
            **metadata.pop("_encoder_timing", {}),
        }
        data = EmbeddingData(
            req_id=req_id,
            num_parts=request.get("num_parts", 1),
            part_idx=request.get("part_idx", 0),
            grid_dim=metadata.pop("grid_dim", None),
            modality=modality,
            embedding_shape=embedding.shape,
            dtype=str(embedding.dtype),
            dispatch_start_ns=request.get("dispatch_start_ns"),
            enqueue_ns=pending.enqueue_ns,
            dequeue_ns=pending.dequeue_ns,
            preprocess_start_ns=preprocess_start_ns,
            publish_done_ns=publish_done_ns,
            queue_duration_ns=queue_duration_ns,
            queue_ms=queue_duration_ns / 1_000_000,
            **transfer_metadata,
            **metadata,
            **encoder_timing,
        )
        transfer_id = str(
            transfer_metadata.get("transfer_id", EncoderRuntime._transfer_id(request))
        )
        return PublishedEmbedding(req_id, transfer_id, data)

    async def _send_error(
        self,
        request: dict[str, Any],
        exc: Exception,
    ) -> None:
        req_id = request["req_id"]
        await self.send_to_scheduler(
            req_id,
            EmbeddingData(
                req_id=req_id,
                num_parts=request.get("num_parts", 1),
                part_idx=request.get("part_idx", 0),
                grid_dim=None,
                modality=Modality.from_str(request["modality"]),
                error_msg=str(exc),
            ),
        )

    async def send_to_scheduler(self, req_id: str, data: EmbeddingData) -> None:
        try:
            await self._notify(await self._wait_for_receiver(req_id), data)
        finally:
            self._receiver_events.pop(req_id, None)
            self._receiver_addresses.pop(req_id, None)

    async def _wait_for_receiver(self, req_id: str) -> str:
        event = self._receiver_events.setdefault(req_id, asyncio.Event())
        if self._receiver_timeout is None or self._receiver_timeout <= 0:
            await event.wait()
        else:
            await asyncio.wait_for(event.wait(), self._receiver_timeout)
        return self._receiver_addresses[req_id]

    async def _notify(self, address: str, data: EmbeddingData) -> None:
        async with self._notify_lock:
            socket = self._receiver_sockets.get(address)
            if socket is None:
                socket = self._zmq.socket(PUSH)
                socket.setsockopt(LINGER, 1000)
                socket.connect(f"tcp://{address}")
                self._receiver_sockets[address] = socket
            await socket.send_pyobj(data)
