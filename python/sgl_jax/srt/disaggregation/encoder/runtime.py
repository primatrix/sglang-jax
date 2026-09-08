from __future__ import annotations

import asyncio
import queue
import threading
import time
from collections import deque
from collections.abc import Callable, Hashable
from dataclasses import dataclass
from typing import Any

from sgl_jax.srt.disaggregation.encoder.embedding_data import EmbeddingData
from sgl_jax.srt.multimodal.common.modality_enum import Modality
from sgl_jax.srt.request_time_stats import mark_request_time_stats, mark_time_stats

_ResultCallback = Callable[[EmbeddingData | Exception], None]
_STOP = object()


@dataclass(slots=True)
class PreprocessedRequest:
    request: dict[str, Any]
    model_input: Any
    batch_key: Hashable


@dataclass(slots=True)
class _EncodeRequest:
    preprocessed: PreprocessedRequest
    on_result: _ResultCallback
    callback_loop: asyncio.AbstractEventLoop


@dataclass(slots=True)
class _EncodeBatch:
    requests: list[_EncodeRequest]
    model_input: Any

    def mark(self, stage: str) -> None:
        mark_request_time_stats(
            (item.preprocessed.request.get("request_time_stats") for item in self.requests), stage
        )

    def deliver(self, results: list[EmbeddingData | Exception]) -> None:
        loop = self.requests[0].callback_loop
        deliveries = tuple(
            (item.on_result, result) for item, result in zip(self.requests, results, strict=True)
        )
        loop.call_soon_threadsafe(self._deliver_many_on_loop, deliveries)

    @staticmethod
    def _deliver_many_on_loop(
        deliveries: tuple[tuple[_ResultCallback, EmbeddingData | Exception], ...],
    ) -> None:
        for callback, result in deliveries:
            callback(result)

    def fail_batch(self, exc: Exception) -> None:
        self.deliver([exc] * len(self.requests))


@dataclass(slots=True)
class _TransferBatch:
    encode_batch: _EncodeBatch
    encoded_output: Any
    staged_transfers: list[Any]
    transfer_ids: list[str]


class EncoderRuntime:
    """Collect completed preprocessing and run ViT/transfer pipeline stages."""

    def __init__(
        self,
        encoder: Any,
        transfer: Any,
        *,
        pipeline_depth: int = 2,
        max_batch_size: int = 8,
        batch_coalesce_ms: float = 0.0,
    ) -> None:
        self._encoder = encoder
        self._transfer = transfer
        depth = max(1, int(pipeline_depth))
        self._max_batch_size = max(1, int(max_batch_size))
        self._batch_coalesce_s = max(0.0, float(batch_coalesce_ms)) / 1000.0
        # This is the completed-preprocess reservoir. The ViT thread drains it
        # when submitting the next batch, without waiting for device completion.
        self._encode_queue: queue.Queue[_EncodeRequest | object] = queue.Queue(
            depth * self._max_batch_size
        )
        # Reserve pool pages before encoding; they bound queued outputs until
        # transfer completion reclaims them. No second queue limit is needed.
        self._transfer_queue: queue.SimpleQueue[_TransferBatch | object] = queue.SimpleQueue()
        self._start_lock = threading.Lock()
        self._started = False
        self._accepting = True
        self._encode_thread: threading.Thread | None = None
        self._transfer_thread: threading.Thread | None = None

    @property
    def preprocess_concurrency(self) -> int:
        # CPU preprocessing and ViT batching have different concurrency needs.
        # Keep all configured processor workers available even for small ViT
        # batches; the bounded ready queue still applies backpressure.
        return max(1, int(self._encoder.preprocess_concurrency))

    def start(self) -> None:
        with self._start_lock:
            if self._started:
                return
            if not self._accepting:
                raise RuntimeError("EncoderRuntime cannot be restarted")
            self._started = True
            self._encode_thread = threading.Thread(
                target=self._encode_worker,
                name="sgl-jax-encoder-vit",
                daemon=True,
            )
            self._transfer_thread = threading.Thread(
                target=self._transfer_worker,
                name="sgl-jax-encoder-transfer",
                daemon=True,
            )
            self._encode_thread.start()
            self._transfer_thread.start()

    async def stop(self) -> None:
        with self._start_lock:
            if not self._started:
                self._accepting = False
                self._transfer.close()
                return
            self._accepting = False
        await asyncio.to_thread(self._stop_workers)

    def _stop_workers(self) -> None:
        # Drain each stage before stopping the next one.
        self._encode_queue.put(_STOP)
        if self._encode_thread is not None:
            self._encode_thread.join()

        self._transfer_queue.put(_STOP)
        if self._transfer_thread is not None:
            self._transfer_thread.join()
        self._transfer.close()

        with self._start_lock:
            self._started = False

    async def preprocess_request(self, request: dict[str, Any]) -> PreprocessedRequest:
        if not self._accepting:
            raise RuntimeError("EncoderRuntime is stopped")
        timing = request.get("request_time_stats")
        mark_time_stats(timing, "preprocess_start_ns")
        model_input = await self._encoder.preprocess_request(request)
        mark_time_stats(timing, "preprocess_done_ns")
        return PreprocessedRequest(
            request=request,
            model_input=model_input,
            batch_key=self._encoder.batch_key(model_input),
        )

    async def enqueue_preprocessed(
        self,
        prepared: PreprocessedRequest,
        on_result: _ResultCallback,
    ) -> None:
        if not self._accepting:
            raise RuntimeError("EncoderRuntime is stopped")
        if not self._started:
            self.start()
        job = _EncodeRequest(prepared, on_result, asyncio.get_running_loop())
        try:
            self._encode_queue.put_nowait(job)
        except queue.Full:
            await asyncio.to_thread(self._encode_queue.put, job)

    def _encode_worker(self) -> None:
        backlog: deque[_EncodeRequest] = deque()
        stopping = False
        while True:
            if backlog:
                item = backlog.popleft()
            elif stopping:
                return
            else:
                item = self._encode_queue.get()
                if item is _STOP:
                    return
            assert isinstance(item, _EncodeRequest)
            batch, saw_stop = self._collect_batch(item, backlog)
            stopping = stopping or saw_stop
            job = _EncodeBatch(batch, None)
            try:
                job.model_input = self._encoder.build_batch(
                    [item.preprocessed.model_input for item in batch]
                )
            except Exception as exc:
                job.fail_batch(exc)
            else:
                self._encode_batch(job)

    def _collect_batch(
        self,
        first: _EncodeRequest,
        backlog: deque[_EncodeRequest],
    ) -> tuple[list[_EncodeRequest], bool]:
        batch = [first]
        key = first.preprocessed.batch_key
        limit = min(
            self._max_batch_size,
            self._transfer.batch_capacity(first.preprocessed.model_input.token_count),
        )

        retained: deque[_EncodeRequest] = deque()
        while backlog:
            item = backlog.popleft()
            if (
                item.callback_loop is first.callback_loop
                and item.preprocessed.batch_key == key
                and len(batch) < limit
            ):
                batch.append(item)
            else:
                retained.append(item)
        backlog.extend(retained)

        deadline = time.monotonic() + self._batch_coalesce_s
        saw_stop = False
        while len(batch) < limit:
            try:
                item = self._encode_queue.get_nowait()
            except queue.Empty:
                # A ready incompatible request is immediately actionable; do
                # not leave the device idle waiting for this key to grow.
                if backlog:
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    item = self._encode_queue.get(timeout=remaining)
                except queue.Empty:
                    break

            if item is _STOP:
                saw_stop = True
                break
            assert isinstance(item, _EncodeRequest)
            if item.callback_loop is first.callback_loop and item.preprocessed.batch_key == key:
                batch.append(item)
            else:
                backlog.append(item)
        return batch, saw_stop

    def _encode_batch(self, job: _EncodeBatch) -> None:
        requests = [item.preprocessed.request for item in job.requests]
        transfer_ids = [
            f"{request['req_id']}:{request.get('part_idx', 0)}:embedding" for request in requests
        ]
        reservations = None
        try:
            job.mark("transfer_reserve_start_ns")
            reservations = self._transfer.reserve_batch_sync(
                transfer_ids, job.model_input.token_counts
            )
            job.mark("encoder_dispatch_start_ns")
            packed_output = self._encoder.encode_packed(job.model_input)
            job.mark("encoder_dispatch_done_ns")
            staged_transfers = self._transfer.stage_packed_batch_sync(
                reservations,
                packed_output.packed,
            )
            if len(staged_transfers) != len(requests):
                raise RuntimeError("transfer returned an incomplete staged batch")
            # The transfer worker builds host metadata while the next ViT batch
            # is dispatched. Pool reservations remain the only transfer limit.
            job.mark("transfer_enqueue_ns")
            self._transfer_queue.put(
                _TransferBatch(
                    job,
                    packed_output,
                    staged_transfers,
                    transfer_ids,
                )
            )
        except Exception as exc:
            if reservations is not None:
                self._transfer.cancel_batch(reservations)
            job.fail_batch(exc)

    def _prepare_transfer_metadata(self, batch: _TransferBatch) -> list[EmbeddingData]:
        output = batch.encoded_output
        metadata = self._encoder.metadata_for_packed(output)
        if len(metadata) != len(batch.encode_batch.requests):
            raise RuntimeError("encoder returned incomplete batch metadata")
        results = []
        for item, token_count, part_metadata in zip(
            batch.encode_batch.requests, output.batch.token_counts, metadata, strict=True
        ):
            request = item.preprocessed.request
            results.append(
                EmbeddingData(
                    req_id=request["req_id"],
                    num_parts=request.get("num_parts", 1),
                    part_idx=request.get("part_idx", 0),
                    modality=Modality.from_str(request["modality"]),
                    shape=(token_count, int(output.packed.shape[1])),
                    dtype=str(output.packed.dtype),
                    timing=request.get("request_time_stats"),
                    **part_metadata,
                )
            )
        return results

    def _transfer_worker(self) -> None:
        while True:
            item = self._transfer_queue.get()
            if item is _STOP:
                return
            assert isinstance(item, _TransferBatch)
            self._run_transfer_batch(item)

    def _run_transfer_batch(self, batch: _TransferBatch) -> None:
        batch.encode_batch.mark("transfer_start_ns")
        try:
            data_items = self._prepare_transfer_metadata(batch)
            metadata = self._transfer.publish_batch_sync(batch.staged_transfers)
            if len(metadata) != len(data_items):
                raise RuntimeError("transfer returned incomplete batch metadata")
        except Exception as exc:
            for transfer_id in batch.transfer_ids:
                self._transfer.release(transfer_id)
            batch.encode_batch.fail_batch(exc)
            return
        batch.encode_batch.mark("publish_done_ns")
        for data, item_metadata in zip(data_items, metadata):
            data.transfer = item_metadata
        batch.encode_batch.deliver(data_items)

    def release(self, transfer_id: str) -> None:
        self._transfer.release(transfer_id)
