from __future__ import annotations

import asyncio
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from sgl_jax.srt.disaggregation.encoder.embedding_data import EmbeddingData
from sgl_jax.srt.multimodal.common.modality_enum import Modality

_ResultCallback = Callable[[int, EmbeddingData | Exception], None]
_STOP = object()


@dataclass(slots=True)
class _EncodeJob:
    requests: list[dict[str, Any]]
    prepared: Any
    preprocess_start_ns: int
    callback: _ResultCallback
    callback_loop: asyncio.AbstractEventLoop

    def deliver(self, index: int, result: EmbeddingData | Exception) -> None:
        self.callback_loop.call_soon_threadsafe(self.callback, index, result)

    def fail_batch(self, exc: Exception) -> None:
        for index in range(len(self.requests)):
            self.deliver(index, exc)


@dataclass(slots=True)
class _TransferJob:
    batch: _EncodeJob
    index: int
    transfer_id: str
    embedding: Any
    data: EmbeddingData


class EncoderRuntime:

    def __init__(
        self,
        encoder: Any,
        transfer: Any,
        *,
        pipeline_depth: int = 2,
        transfer_poll_interval_s: float = 0.001,
    ) -> None:
        self._encoder = encoder
        self._transfer = transfer
        depth = max(1, int(pipeline_depth))
        self._encode_queue: queue.Queue[_EncodeJob | object] = queue.Queue(depth)
        self._transfer_queue: queue.Queue[_TransferJob | object] = queue.Queue(depth)
        self._transfer_poll_interval_s = max(0.0001, float(transfer_poll_interval_s))
        self._stop_event = threading.Event()
        self._start_lock = threading.Lock()
        self._started = False
        self._accepting = True
        self._vit_thread: threading.Thread | None = None
        self._transfer_thread: threading.Thread | None = None
        self._progress_thread: threading.Thread | None = None

    def start(self) -> None:
        with self._start_lock:
            if self._started:
                return
            if not self._accepting:
                raise RuntimeError("EncoderRuntime cannot be restarted")
            self._started = True
            self._vit_thread = threading.Thread(
                target=self._vit_worker,
                name="sgl-jax-encoder-vit",
                daemon=True,
            )
            self._transfer_thread = threading.Thread(
                target=self._transfer_worker,
                name="sgl-jax-encoder-transfer",
                daemon=True,
            )
            self._vit_thread.start()
            self._transfer_thread.start()

            if callable(getattr(self._transfer, "poll_completed", None)):
                self._progress_thread = threading.Thread(
                    target=self._progress_worker,
                    name="sgl-jax-encoder-transfer-progress",
                    daemon=True,
                )
                self._progress_thread.start()

    async def stop(self) -> None:
        with self._start_lock:
            if not self._started:
                self._accepting = False
                self._transfer.close()
                return
            self._accepting = False
        await asyncio.to_thread(self._stop_workers)

    def _stop_workers(self) -> None:
        # Drain each stage before stopping the next one. The progress thread
        # remains alive while transfer staging drains so that pool credits can
        # still be reclaimed.
        self._encode_queue.put(_STOP)
        if self._vit_thread is not None:
            self._vit_thread.join()

        self._transfer_queue.put(_STOP)
        if self._transfer_thread is not None:
            self._transfer_thread.join()

        self._stop_event.set()
        if self._progress_thread is not None:
            self._progress_thread.join()
        self._transfer.close()

        with self._start_lock:
            self._started = False

    async def execute_batch(
        self,
        requests: list[dict[str, Any]],
        on_result_callback: _ResultCallback,
    ) -> None:
        if not requests:
            return
        if not self._accepting:
            raise RuntimeError("EncoderRuntime is stopped")
        if not self._started:
            self.start()

        # This coroutine performs orchestration only. MMEncoder.preprocess
        # delegates image loading and HF processing to their own executors.
        preprocess_start_ns = time.time_ns()
        prepared = await self._encoder.preprocess(requests)
        if not self._accepting:
            raise RuntimeError("EncoderRuntime stopped during preprocessing")
        job = _EncodeJob(
            requests=requests,
            prepared=prepared,
            preprocess_start_ns=preprocess_start_ns,
            callback=on_result_callback,
            callback_loop=asyncio.get_running_loop(),
        )
        try:
            self._encode_queue.put_nowait(job)
        except queue.Full:
            # Backpressure is intentionally offloaded instead of blocking the
            # server event loop. With pipeline_depth >= 2 this is a cold path.
            await asyncio.to_thread(self._encode_queue.put, job)

    def _vit_worker(self) -> None:
        while True:
            item = self._encode_queue.get()
            try:
                if item is _STOP:
                    return
                self._run_vit(item)
            finally:
                self._encode_queue.task_done()

    def _run_vit(self, job: _EncodeJob) -> None:
        try:
            results = self._encoder.encode(job.prepared)
            if len(results) != len(job.requests):
                raise RuntimeError(
                    f"encoder returned {len(results)} results for " f"{len(job.requests)} requests"
                )
            encode_done_ns = time.time_ns()
            transfer_jobs = []
            for index, (request, (embedding, metadata)) in enumerate(zip(job.requests, results)):
                transfer_id = f"{request['req_id']}:{request.get('part_idx', 0)}:embedding"
                metadata = dict(metadata)
                encoder_timing = {
                    "encode_done_ns": encode_done_ns,
                    **metadata.pop("_encoder_timing", {}),
                }
                data = EmbeddingData(
                    req_id=request["req_id"],
                    num_parts=request.get("num_parts", 1),
                    part_idx=request.get("part_idx", 0),
                    grid_dim=metadata.pop("grid_dim", None),
                    modality=Modality.from_str(request["modality"]),
                    embedding_shape=embedding.shape,
                    dtype=str(embedding.dtype),
                    dispatch_start_ns=request.get("dispatch_start_ns"),
                    preprocess_start_ns=job.preprocess_start_ns,
                    **metadata,
                    **encoder_timing,
                )
                transfer_jobs.append(_TransferJob(job, index, transfer_id, embedding, data))
            for transfer_job in transfer_jobs:
                self._transfer_queue.put(transfer_job)
        except Exception as exc:
            job.fail_batch(exc)

    def _transfer_worker(self) -> None:
        while True:
            item = self._transfer_queue.get()
            try:
                if item is _STOP:
                    return
                assert isinstance(item, _TransferJob)
                self._run_transfer(item)
            finally:
                self._transfer_queue.task_done()

    def _run_transfer(self, job: _TransferJob) -> None:
        try:
            staged_transfer = self._transfer.stage_sync(
                job.transfer_id,
                job.embedding,
            )
            transfer_metadata = self._transfer.publish_sync(staged_transfer)
            for key, value in transfer_metadata.items():
                setattr(job.data, key, value)
            job.data.transfer_id = str(transfer_metadata.get("transfer_id", job.transfer_id))
            job.data.publish_done_ns = time.time_ns()
        except Exception as exc:
            self._transfer.release(job.transfer_id)
            job.batch.deliver(job.index, exc)
        else:
            job.batch.deliver(job.index, job.data)

    def _progress_worker(self) -> None:
        while not self._stop_event.wait(self._transfer_poll_interval_s):
            self._transfer.poll_completed()
        self._transfer.poll_completed()

    def release(self, transfer_id: str) -> None:
        self._transfer.release(transfer_id)
