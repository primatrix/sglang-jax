from __future__ import annotations

import asyncio
import logging

from sgl_jax.srt.disaggregation.encoder.embedding_data import EmbeddingData
from sgl_jax.srt.disaggregation.encoder.runtime import EncoderRuntime
from sgl_jax.srt.request_time_stats import mark_time_stats

logger = logging.getLogger(__name__)


class _PendingRequest:
    __slots__ = ("future", "request")

    def __init__(self, request: dict) -> None:
        self.request = request
        self.future: asyncio.Future[EmbeddingData] = asyncio.get_running_loop().create_future()


class DisaggEncoderScheduler:
    """Own request admission, preprocessing workers, futures, and timeouts."""

    def __init__(
        self,
        runtime: EncoderRuntime,
        request_timeout: float | None = 300.0,
        enable_time_stats: bool = False,
    ) -> None:
        self._runtime = runtime
        self._request_timeout = request_timeout
        self._enable_time_stats = enable_time_stats
        self._pending_queue: asyncio.Queue[_PendingRequest] = asyncio.Queue()
        self._workers: set[asyncio.Task[None]] = set()
        self._requests: set[_PendingRequest] = set()
        self._running = False

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        for index in range(self._runtime.preprocess_concurrency):
            task = asyncio.create_task(
                self._preprocess_worker(),
                name=f"encoder-preprocess-{index}",
            )
            self._workers.add(task)
            task.add_done_callback(self._workers.discard)

    async def stop(self) -> None:
        self._running = False
        workers = tuple(self._workers)
        for task in workers:
            task.cancel()
        if workers:
            await asyncio.gather(*workers, return_exceptions=True)
        self._workers.clear()

        error = RuntimeError("DisaggEncoderScheduler stopped")
        for pending in self._requests:
            if not pending.future.done():
                pending.future.set_exception(error)
        while True:
            try:
                self._pending_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            self._pending_queue.task_done()

    async def submit(self, request: dict) -> EmbeddingData:
        if not self._running:
            raise RuntimeError("DisaggEncoderScheduler is not running")
        timing = None
        if self._enable_time_stats and request.get("collect_request_time_stats", False):
            timing = {}
            if request.get("dispatch_start_ns") is not None:
                timing["dispatch_start_ns"] = request["dispatch_start_ns"]
        request["request_time_stats"] = timing
        mark_time_stats(timing, "enqueue_ns")
        pending = _PendingRequest(request)
        self._requests.add(pending)
        await self._pending_queue.put(pending)
        try:
            if self._request_timeout is None or self._request_timeout <= 0:
                return await pending.future
            return await asyncio.wait_for(pending.future, self._request_timeout)
        finally:
            self._requests.discard(pending)

    async def _preprocess_worker(self) -> None:
        while True:
            pending = await self._pending_queue.get()
            try:
                if pending.future.done():
                    continue
                prepared = await self._runtime.preprocess_request(pending.request)
                if pending.future.done():
                    continue

                def complete(
                    result: EmbeddingData | Exception,
                    pending: _PendingRequest = pending,
                ) -> None:
                    self._complete(pending, result)

                await self._runtime.enqueue_preprocessed(prepared, complete)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception("Encoder preprocessing failed")
                self._complete(pending, exc)
            finally:
                self._pending_queue.task_done()

    def _complete(
        self,
        pending: _PendingRequest,
        result: EmbeddingData | Exception,
    ) -> None:
        if isinstance(result, Exception):
            if not pending.future.done():
                pending.future.set_exception(result)
            return
        if pending.future.done():
            self._runtime.release(result.transfer_id)
            return
        pending.future.set_result(result)
