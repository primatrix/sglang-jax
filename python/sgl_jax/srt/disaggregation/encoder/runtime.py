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

_ResultCallback = Callable[[EmbeddingData | Exception], None]
_STOP = object()


@dataclass(slots=True)
class PreprocessedRequest:
    request: dict[str, Any]
    value: Any
    batch_key: Hashable
    preprocess_start_ns: int


@dataclass(slots=True)
class _ReadyJob:
    prepared: PreprocessedRequest
    callback: _ResultCallback
    callback_loop: asyncio.AbstractEventLoop


@dataclass(slots=True)
class _EncodeJob:
    items: list[_ReadyJob]
    prepared: Any

    def deliver(self, index: int, result: EmbeddingData | Exception) -> None:
        item = self.items[index]
        item.callback_loop.call_soon_threadsafe(item.callback, result)

    def deliver_many(self, results: list[tuple[int, EmbeddingData | Exception]]) -> None:
        loop = self.items[0].callback_loop
        deliveries = tuple((self.items[index].callback, result) for index, result in results)
        loop.call_soon_threadsafe(self._deliver_many_on_loop, deliveries)

    @staticmethod
    def _deliver_many_on_loop(
        deliveries: tuple[tuple[_ResultCallback, EmbeddingData | Exception], ...],
    ) -> None:
        for callback, result in deliveries:
            callback(result)

    def fail_batch(self, exc: Exception) -> None:
        self.deliver_many([(index, exc) for index in range(len(self.items))])


@dataclass(slots=True)
class _TransferJob:
    batch: _EncodeJob
    index: int
    transfer_id: str
    staged_transfer: Any
    data: EmbeddingData


@dataclass(slots=True)
class _TransferBatchJob:
    jobs: list[_TransferJob]


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
        # only when the device is ready, so a future batch is never frozen early.
        self._vit_queue: queue.Queue[_ReadyJob | object] = queue.Queue(depth * self._max_batch_size)
        # Pool reservations provide backpressure. This queue carries only small
        # ready tickets, so it does not need a second capacity limit.
        self._transfer_queue: queue.SimpleQueue[_TransferBatchJob | object] = queue.SimpleQueue()
        self._start_lock = threading.Lock()
        self._started = False
        self._accepting = True
        self._vit_thread: threading.Thread | None = None
        self._transfer_thread: threading.Thread | None = None

    @property
    def preprocess_concurrency(self) -> int:
        return max(
            1,
            min(
                self._max_batch_size,
                int(getattr(self._encoder, "preprocess_concurrency", self._max_batch_size)),
            ),
        )

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
        self._vit_queue.put(_STOP)
        if self._vit_thread is not None:
            self._vit_thread.join()

        self._transfer_queue.put(_STOP)
        if self._transfer_thread is not None:
            self._transfer_thread.join()
        self._transfer.close()

        with self._start_lock:
            self._started = False

    async def preprocess_request(self, request: dict[str, Any]) -> PreprocessedRequest:
        if not self._accepting:
            raise RuntimeError("EncoderRuntime is stopped")
        preprocess_start_ns = time.time_ns()
        value = await self._encoder.preprocess_request(request)
        return PreprocessedRequest(
            request=request,
            value=value,
            batch_key=self._encoder.batch_key(value),
            preprocess_start_ns=preprocess_start_ns,
        )

    async def enqueue_preprocessed(
        self,
        prepared: PreprocessedRequest,
        on_result_callback: _ResultCallback,
    ) -> None:
        if not self._accepting:
            raise RuntimeError("EncoderRuntime is stopped")
        if not self._started:
            self.start()
        job = _ReadyJob(prepared, on_result_callback, asyncio.get_running_loop())
        try:
            self._vit_queue.put_nowait(job)
        except queue.Full:
            await asyncio.to_thread(self._vit_queue.put, job)

    def _vit_worker(self) -> None:
        backlog: deque[_ReadyJob] = deque()
        stopping = False
        while True:
            if backlog:
                item = backlog.popleft()
            elif stopping:
                return
            else:
                item = self._get_vit_queue()
                if item is _STOP:
                    return
            assert isinstance(item, _ReadyJob)
            batch, saw_stop = self._collect_ready(item, backlog)
            stopping = stopping or saw_stop
            job = _EncodeJob(batch, None)
            try:
                job.prepared = self._encoder.build_batch([item.prepared.value for item in batch])
                precompile_packed = getattr(self._transfer, "precompile_packed_batches", None)
                transfer_specs = getattr(job.prepared, "transfer_specs", ())
                if callable(precompile_packed) and transfer_specs:
                    precompile_packed(transfer_specs)
            except Exception as exc:
                job.fail_batch(exc)
            else:
                self._run_vit(job)

    def _get_vit_queue(self, timeout: float | None = None) -> Any:
        if timeout is None:
            item = self._vit_queue.get()
        elif timeout <= 0:
            item = self._vit_queue.get_nowait()
        else:
            item = self._vit_queue.get(timeout=timeout)
        self._vit_queue.task_done()
        return item

    def _collect_ready(
        self,
        first: _ReadyJob,
        backlog: deque[_ReadyJob],
    ) -> tuple[list[_ReadyJob], bool]:
        batch = [first]
        key = first.prepared.batch_key

        retained: deque[_ReadyJob] = deque()
        while backlog:
            item = backlog.popleft()
            if (
                item.callback_loop is first.callback_loop
                and item.prepared.batch_key == key
                and len(batch) < self._max_batch_size
            ):
                batch.append(item)
            else:
                retained.append(item)
        backlog.extend(retained)

        deadline = time.monotonic() + self._batch_coalesce_s
        saw_stop = False
        while len(batch) < self._max_batch_size:
            try:
                item = self._get_vit_queue(timeout=0)
            except queue.Empty:
                # A ready incompatible request is immediately actionable; do
                # not leave the device idle waiting for this key to grow.
                if backlog:
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    item = self._get_vit_queue(timeout=remaining)
                except queue.Empty:
                    break

            if item is _STOP:
                saw_stop = True
                break
            assert isinstance(item, _ReadyJob)
            if item.callback_loop is first.callback_loop and item.prepared.batch_key == key:
                batch.append(item)
            else:
                backlog.append(item)
        return batch, saw_stop

    def _run_vit(self, job: _EncodeJob) -> None:
        requests = [item.prepared.request for item in job.items]
        transfer_ids = [
            f"{request['req_id']}:{request.get('part_idx', 0)}:embedding" for request in requests
        ]
        reservations = None
        try:
            reservations = self._transfer.reserve_batch_sync(transfer_ids)
            encode_packed = getattr(self._encoder, "encode_packed", None)
            metadata_for_packed = getattr(self._encoder, "metadata_for_packed", None)
            stage_packed = getattr(self._transfer, "stage_packed_batch_sync", None)
            use_packed = all(
                callable(method) for method in (encode_packed, metadata_for_packed, stage_packed)
            )
            staged_transfers = None
            if use_packed:
                assert callable(encode_packed)
                assert callable(metadata_for_packed)
                assert callable(stage_packed)
                packed_output = encode_packed(job.prepared)
                runtime_encode_return_ns = time.time_ns()
                copy_start_ns = time.time_ns()
                staged_transfers = stage_packed(
                    reservations,
                    packed_output.packed,
                    packed_output.batch.token_counts,
                )
                runtime_postprocess_start_ns = time.perf_counter_ns()
                packed_metadata = metadata_for_packed(packed_output)
                results = [
                    (
                        (token_count, int(packed_output.packed.shape[1])),
                        packed_output.packed.dtype,
                        metadata,
                    )
                    for token_count, metadata in zip(
                        packed_output.batch.token_counts,
                        packed_metadata,
                    )
                ]
            else:
                encoded = self._encoder.encode(job.prepared)
                runtime_encode_return_ns = time.time_ns()
                runtime_postprocess_start_ns = time.perf_counter_ns()
                results = [
                    (embedding.shape, embedding.dtype, metadata) for embedding, metadata in encoded
                ]
            if len(results) != len(requests):
                raise RuntimeError(
                    f"encoder returned {len(results)} results for {len(requests)} requests"
                )
            encode_done_ns = time.time_ns()
            transfer_jobs = []
            metadata_prepare_duration_ns = 0
            embedding_data_duration_ns = 0
            result_pack_duration_ns = 0
            for index, (request, transfer_id, (embedding_shape, dtype, metadata)) in enumerate(
                zip(requests, transfer_ids, results)
            ):
                phase_start_ns = time.perf_counter_ns()
                metadata = dict(metadata)
                encoder_timing = {
                    "encode_done_ns": encode_done_ns,
                    **metadata.pop("_encoder_timing", {}),
                }
                metadata_prepare_duration_ns += time.perf_counter_ns() - phase_start_ns

                phase_start_ns = time.perf_counter_ns()
                data = EmbeddingData(
                    req_id=request["req_id"],
                    num_parts=request.get("num_parts", 1),
                    part_idx=request.get("part_idx", 0),
                    grid_dim=metadata.pop("grid_dim", None),
                    modality=Modality.from_str(request["modality"]),
                    embedding_shape=embedding_shape,
                    dtype=str(dtype),
                    dispatch_start_ns=request.get("dispatch_start_ns"),
                    preprocess_start_ns=job.items[index].prepared.preprocess_start_ns,
                    **metadata,
                    **encoder_timing,
                )
                embedding_data_duration_ns += time.perf_counter_ns() - phase_start_ns

                phase_start_ns = time.perf_counter_ns()
                transfer_jobs.append((index, transfer_id, data))
                result_pack_duration_ns += time.perf_counter_ns() - phase_start_ns

            runtime_postprocess_done_ns = time.time_ns()
            runtime_postprocess_duration_ns = time.perf_counter_ns() - runtime_postprocess_start_ns
            runtime_postprocess_residual_ns = max(
                0,
                runtime_postprocess_duration_ns
                - metadata_prepare_duration_ns
                - embedding_data_duration_ns
                - result_pack_duration_ns,
            )
            runtime_timing = {
                "runtime_encode_return_ns": runtime_encode_return_ns,
                "runtime_postprocess_done_ns": runtime_postprocess_done_ns,
                "runtime_postprocess_duration_ns": runtime_postprocess_duration_ns,
                "runtime_metadata_prepare_duration_ns": metadata_prepare_duration_ns,
                "runtime_embedding_data_duration_ns": embedding_data_duration_ns,
                "runtime_result_pack_duration_ns": result_pack_duration_ns,
                "runtime_postprocess_residual_ns": runtime_postprocess_residual_ns,
            }
            timing_attach_start_ns = time.perf_counter_ns()
            for _, _, data in transfer_jobs:
                for key, value in runtime_timing.items():
                    setattr(data, key, value)
            timing_attach_duration_ns = time.perf_counter_ns() - timing_attach_start_ns
            for _, _, data in transfer_jobs:
                data.runtime_timing_attach_duration_ns = timing_attach_duration_ns

            if not use_packed:
                copy_start_ns = time.time_ns()
            for _, _, data in transfer_jobs:
                data.transfer_copy_start_ns = copy_start_ns
            if staged_transfers is None:
                encoded_embeddings = [embedding for embedding, _ in encoded]
                staged_transfers = self._transfer.stage_batch_sync(
                    reservations,
                    encoded_embeddings,
                )
            if len(staged_transfers) != len(transfer_jobs):
                raise RuntimeError("transfer returned an incomplete staged batch")
            queued_jobs = []
            transfer_enqueue_ns = time.time_ns()
            for (index, transfer_id, data), staged_transfer in zip(
                transfer_jobs,
                staged_transfers,
            ):
                data.transfer_enqueue_ns = transfer_enqueue_ns
                queued_jobs.append(
                    _TransferJob(
                        job,
                        index,
                        transfer_id,
                        staged_transfer,
                        data,
                    )
                )
            self._transfer_queue.put(_TransferBatchJob(queued_jobs))
        except Exception as exc:
            if reservations is not None:
                self._transfer.cancel_batch(reservations)
            job.fail_batch(exc)

    def _transfer_worker(self) -> None:
        while True:
            item = self._transfer_queue.get()
            if item is _STOP:
                return
            assert isinstance(item, _TransferBatchJob)
            self._run_transfer_batch(item)

    def _run_transfer_batch(self, batch: _TransferBatchJob) -> None:
        publish_batch = getattr(self._transfer, "publish_batch_sync", None)
        if not callable(publish_batch):
            for job in batch.jobs:
                self._run_transfer(job)
            return

        transfer_start_ns = time.time_ns()
        for job in batch.jobs:
            job.data.transfer_start_ns = transfer_start_ns
        try:
            metadata = publish_batch([job.staged_transfer for job in batch.jobs])
            if len(metadata) != len(batch.jobs):
                raise RuntimeError("transfer returned incomplete batch metadata")
        except Exception as exc:
            deliveries: list[tuple[int, EmbeddingData | Exception]] = []
            for job in batch.jobs:
                self._transfer.release(job.transfer_id)
                deliveries.append((job.index, exc))
            if batch.jobs:
                batch.jobs[0].batch.deliver_many(deliveries)
            return
        deliveries = []
        for job, item_metadata in zip(batch.jobs, metadata):
            deliveries.append(
                (job.index, self._complete_transfer(job, item_metadata, deliver=False))
            )
        if batch.jobs:
            batch.jobs[0].batch.deliver_many(deliveries)

    def _run_transfer(self, job: _TransferJob) -> None:
        try:
            job.data.transfer_start_ns = time.time_ns()
            transfer_metadata = self._transfer.publish_sync(job.staged_transfer)
        except Exception as exc:
            self._transfer.release(job.transfer_id)
            job.batch.deliver(job.index, exc)
        else:
            self._complete_transfer(job, transfer_metadata)

    def _complete_transfer(
        self,
        job: _TransferJob,
        transfer_metadata: dict[str, Any],
        *,
        deliver: bool = True,
    ) -> EmbeddingData:
        for key, value in transfer_metadata.items():
            setattr(job.data, key, value)
        # Backends may optionally expose the pool/reservation/copy split.
        # Keep a complete timing chain for simpler downstream aggregation.
        job.data.transfer_pool_ready_ns = (
            getattr(job.data, "transfer_pool_ready_ns", None) or job.data.transfer_start_ns
        )
        job.data.transfer_reserve_done_ns = (
            getattr(job.data, "transfer_reserve_done_ns", None) or job.data.transfer_pool_ready_ns
        )
        job.data.transfer_copy_done_ns = (
            getattr(job.data, "transfer_copy_done_ns", None)
            or job.data.transfer_stage_done_ns
            or job.data.transfer_start_ns
        )
        job.data.transfer_register_start_ns = (
            getattr(job.data, "transfer_register_start_ns", None) or job.data.transfer_copy_done_ns
        )
        job.data.transfer_register_done_ns = (
            getattr(job.data, "transfer_register_done_ns", None) or time.time_ns()
        )
        assert job.data.transfer_copy_done_ns is not None
        assert job.data.transfer_register_done_ns is not None
        job.data.transfer_publish_ready_ns = getattr(
            job.data, "transfer_publish_ready_ns", None
        ) or max(job.data.transfer_copy_done_ns, job.data.transfer_register_done_ns)
        job.data.transfer_stage_done_ns = job.data.transfer_copy_done_ns
        job.data.transfer_id = str(transfer_metadata.get("transfer_id", job.transfer_id))
        job.data.publish_done_ns = time.time_ns()
        if deliver:
            job.batch.deliver(job.index, job.data)
        return job.data

    def release(self, transfer_id: str) -> None:
        self._transfer.release(transfer_id)
