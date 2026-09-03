from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from contextlib import suppress

from sgl_jax.srt.disaggregation.encoder.embedding_data import EmbeddingData
from sgl_jax.srt.disaggregation.encoder.runtime import EncoderRuntime
from sgl_jax.srt.multimodal.common.modality_enum import Modality

logger = logging.getLogger(__name__)


class _PendingRequest:
    __slots__ = (
        "_enqueue_mono_ns",
        "dequeue_ns",
        "enqueue_ns",
        "future",
        "queue_duration_ns",
        "request",
    )

    def __init__(self, request: dict) -> None:
        self.request = request
        self.future: asyncio.Future[EmbeddingData] = asyncio.get_running_loop().create_future()
        self.enqueue_ns = time.time_ns()
        self._enqueue_mono_ns = time.monotonic_ns()
        self.dequeue_ns = 0
        self.queue_duration_ns = 0


class DisaggEncoderScheduler:
    """Own encoder admission, batching, request futures, and timeouts."""

    def __init__(
        self,
        runtime: EncoderRuntime,
        max_batch_size: int = 8,
        request_timeout: float | None = 300.0,
        log_queue_timing: bool = False,
        max_inflight_batches: int = 1,
    ) -> None:
        self._runtime = runtime
        self._max_batch_size = max(1, int(max_batch_size))
        self._max_inflight_batches = max(1, int(max_inflight_batches))
        self._request_timeout = request_timeout
        self._log_queue_timing = log_queue_timing
        self._pending_queue: asyncio.Queue[_PendingRequest] = asyncio.Queue()
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

        for task in list(self._inflight_tasks):
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
                pending.future.set_exception(RuntimeError("DisaggEncoderScheduler stopped"))
            self._pending_queue.task_done()

    async def submit(self, request: dict) -> EmbeddingData:
        if self._worker_task is None:
            raise RuntimeError("DisaggEncoderScheduler is not running")
        pending = _PendingRequest(request)
        await self._pending_queue.put(pending)
        if self._request_timeout is None or self._request_timeout <= 0:
            return await pending.future
        return await asyncio.wait_for(pending.future, self._request_timeout)

    async def _collect_batch(self) -> list[_PendingRequest]:
        first = await self._pending_queue.get()
        first.dequeue_ns = time.time_ns()
        first.queue_duration_ns = max(0, time.monotonic_ns() - first._enqueue_mono_ns)
        batch = [first]
        while len(batch) < self._max_batch_size:
            try:
                pending = self._pending_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            pending.dequeue_ns = time.time_ns()
            pending.queue_duration_ns = max(0, time.monotonic_ns() - pending._enqueue_mono_ns)
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

    async def _run_batch(self, batch: list[_PendingRequest]) -> None:
        try:
            groups: dict[Modality, list[_PendingRequest]] = defaultdict(list)
            for pending in batch:
                modality = Modality.from_str(pending.request.get("modality", "image"))
                groups[modality].append(pending)
            for group in groups.values():
                await self._run_group(group)
        except asyncio.CancelledError:
            self._fail_pending(batch, RuntimeError("DisaggEncoderScheduler stopped"))
            raise
        except Exception as exc:
            logger.exception("Encoder batch failed")
            self._fail_pending(batch, exc)
        finally:
            for _ in batch:
                self._pending_queue.task_done()

    async def _run_group(self, group: list[_PendingRequest]) -> None:
        active = [pending for pending in group if not pending.future.done()]
        if not active:
            return

        def complete(index: int, result: EmbeddingData | Exception) -> None:
            pending = active[index]
            if isinstance(result, Exception):
                if not pending.future.done():
                    pending.future.set_exception(result)
                return
            if pending.future.done():
                self._runtime.release(result.transfer_id)
                return
            result.enqueue_ns = pending.enqueue_ns
            result.dequeue_ns = pending.dequeue_ns
            result.queue_duration_ns = pending.queue_duration_ns
            result.queue_ms = pending.queue_duration_ns / 1_000_000
            pending.future.set_result(result)

        await self._runtime.execute_batch(
            [pending.request for pending in active],
            complete,
        )

    @staticmethod
    def _fail_pending(batch: list[_PendingRequest], exc: Exception) -> None:
        for pending in batch:
            if not pending.future.done():
                pending.future.set_exception(exc)
