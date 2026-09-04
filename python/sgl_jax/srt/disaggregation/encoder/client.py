from __future__ import annotations

import asyncio
import logging
import os
import random
import threading
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from queue import Empty, SimpleQueue
from typing import Any, Protocol

import httpx
import jax
import orjson
import zmq

from sgl_jax.srt.disaggregation.encoder.embedding_data import (
    EmbeddingData,
    MultiModalEmbeddingData,
    PooledEmbedding,
)
from sgl_jax.srt.managers.io_struct import (
    GenerateReqInput,
    ImageData,
    TokenizedGenerateReqInput,
)
from sgl_jax.srt.multimodal.common.modality_enum import Modality, flatten_nested_list

logger = logging.getLogger(__name__)
_DISPATCH_ORJSON = os.environ.get("SGLANG_ENCODER_DISPATCH_ORJSON", "1") != "0"


def create_part_req_id(req_id: str, part_idx: int) -> str:
    return f"{req_id}_local_part_{part_idx}"


def plan_encoder_registrations(
    request: TokenizedGenerateReqInput,
    default_encoder_urls: list[str],
) -> list[tuple[str, str, Modality | None]]:
    """Build the encoder receiver registrations for a tokenized request.

    Args:
        request: The request containing the request ID, optional encoder URLs, and
            the number of multimodal items assigned to each encoder.
        default_encoder_urls: Encoder URLs used when the request does not provide
            its own ``encoder_urls``.

    Returns:
        A list of ``(encoder_url, request_part_id, modality)`` tuples, one for
        each non-empty modality/encoder assignment. For a single encoder without
        explicit assignments, the original request ID and ``None`` modality are
        returned.
    """
    if not isinstance(request.rid, str):
        raise ValueError("encoder request requires a single rid")
    encoder_urls = request.encoder_urls or default_encoder_urls
    if not encoder_urls:
        raise ValueError("encoder_urls is required")

    if request.num_items_assigned is None:
        if len(encoder_urls) != 1:
            raise ValueError("num_items_assigned is required for multiple encoders")
        return [(encoder_urls[0], request.rid, None)]

    registrations: list[tuple[str, str, Modality | None]] = []
    for modality, assignments in request.num_items_assigned.items():
        if len(assignments) != len(encoder_urls):
            raise ValueError(
                f"{modality.name} has {len(assignments)} assignments for "
                f"{len(encoder_urls)} encoders"
            )
        for encoder_idx, count in enumerate(assignments):
            if count < 0:
                raise ValueError("num_items_assigned cannot contain negative values")
            if count == 0:
                continue
            part_idx = len(registrations)
            registrations.append(
                (
                    encoder_urls[encoder_idx],
                    create_part_req_id(request.rid, part_idx),
                    modality,
                )
            )

    if not registrations:
        raise ValueError("num_items_assigned does not assign any multimodal items")
    return registrations


def register_scheduler_receiver(
    registration: tuple[str, str, Modality | None],
    receive_url: str,
    client: httpx.Client,
) -> None:
    encoder_url, req_id, modality = registration
    payload = {
        "req_id": req_id,
        "receive_count": 1,
        "receive_url": receive_url,
    }
    if modality is not None:
        payload["modality"] = modality.name
    response = client.post(
        f"{encoder_url.rstrip('/')}/scheduler_receive_url",
        json=payload,
    )
    response.raise_for_status()


def validate_encoder_response(
    data: Any,
    expected_num_parts: int,
    active_part_indices: set[int],
    completed_part_indices: set[int],
) -> None:
    if not isinstance(data, EmbeddingData):
        raise TypeError(f"expected EmbeddingData, got {type(data).__name__}")
    if data.num_parts != expected_num_parts:
        raise ValueError("inconsistent encoder part metadata")
    if not 0 <= data.part_idx < expected_num_parts:
        raise ValueError(f"invalid part_idx: {data.part_idx}")
    if data.part_idx in active_part_indices or data.part_idx in completed_part_indices:
        raise ValueError(f"duplicate part_idx: {data.part_idx}")
    if data.error_msg is not None:
        raise RuntimeError(data.error_msg)


class EncoderReceiveSession(Protocol):
    def poll(self, *, refresh_backend: bool = True) -> jax.Array | PooledEmbedding | None: ...

    def close(self) -> None: ...


class DeferredReceiveSession:
    """Expose a non-blocking session while backend setup runs off-loop."""

    def __init__(self, future: Future[EncoderReceiveSession]) -> None:
        self._future = future
        self._session: EncoderReceiveSession | None = None
        self._closed = False
        self._setup_done_ns: int | None = None
        future.add_done_callback(self._mark_setup_done)

    def _mark_setup_done(self, _future: Future[EncoderReceiveSession]) -> None:
        self._setup_done_ns = time.time_ns()

    def poll(self, *, refresh_backend: bool = True) -> jax.Array | None:
        if self._closed:
            return None
        if self._session is None:
            if not self._future.done():
                return None
            self._session = self._future.result()
        return self._session.poll(refresh_backend=refresh_backend)

    @property
    def timing_meta(self) -> dict[str, int]:
        timing = {}
        if self._setup_done_ns is not None:
            timing["receive_setup_done_ns"] = self._setup_done_ns
        if self._session is not None:
            timing.update(getattr(self._session, "timing_meta", {}))
        return timing

    def close(self) -> None:
        self._closed = True
        if self._session is not None:
            self._session.close()
        elif not self._future.cancel():
            self._future.add_done_callback(self._close_session)

    @staticmethod
    def _close_session(future: Future[EncoderReceiveSession]) -> None:
        if future.cancelled():
            return
        try:
            future.result().close()
        except Exception:
            logger.exception("Deferred encoder receiver setup failed during cleanup")


class EncoderReceiverBackend(Protocol):
    def start(self, data: EmbeddingData) -> EncoderReceiveSession: ...

    def close(self) -> None: ...


class EncoderMetadataRouter:
    """Route one scheduler-wide metadata socket to pending encoder requests."""

    def __init__(self, host: str) -> None:
        self._receiver = zmq.Context.instance().socket(zmq.PULL)
        self._receiver.setsockopt(zmq.LINGER, 0)
        port = self._receiver.bind_to_random_port(f"tcp://{host}")
        self.receive_url = f"{host}:{port}"
        self._queues: dict[str, deque[Any]] = {}
        self._lock = threading.Lock()

    def register(self, req_ids: tuple[str, ...]) -> None:
        routes = set(req_ids)
        with self._lock:
            if len(routes) != len(req_ids) or not routes.isdisjoint(self._queues):
                raise ValueError(f"duplicate encoder metadata routes: {req_ids}")
            self._queues.update((req_id, deque()) for req_id in req_ids)

    def drain(self) -> None:
        while True:
            try:
                data = self._receiver.recv_pyobj(zmq.NOBLOCK)
            except zmq.Again:
                break
            with self._lock:
                queue = self._queues.get(getattr(data, "req_id", None))
                if queue is not None:
                    queue.append(data)

    def pop(self, req_ids: tuple[str, ...]) -> Any | None:
        with self._lock:
            for req_id in req_ids:
                queue = self._queues.get(req_id)
                if queue:
                    return queue.popleft()
        return None

    def poll(self, req_ids: tuple[str, ...]) -> Any | None:
        self.drain()
        return self.pop(req_ids)

    def unregister(self, req_ids: tuple[str, ...]) -> None:
        with self._lock:
            for req_id in req_ids:
                self._queues.pop(req_id, None)

    def close(self) -> None:
        with self._lock:
            self._queues.clear()
        self._receiver.close()


@dataclass(slots=True)
class PendingEncoderRequest:
    recv_req: TokenizedGenerateReqInput
    started_at: float
    metadata_router: EncoderMetadataRouter
    metadata_req_ids: tuple[str, ...]
    registration_futures: tuple[Future[None], ...]
    accumulator: MultiModalEmbeddingData
    backend: EncoderReceiverBackend
    result_preparer: Callable[[TokenizedGenerateReqInput, dict[str, Any]], None] | None = None
    # Keep each part's metadata alongside its in-flight transfer session so the
    # completed embedding can later be assembled with the correct modality and grid.
    sessions: dict[int, tuple[EmbeddingData, EncoderReceiveSession]] = field(default_factory=dict)
    background_progress: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _result: dict[str, Any] | None = field(default=None, init=False, repr=False)
    _error: Exception | None = field(default=None, init=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)
    _done: threading.Event = field(default_factory=threading.Event, init=False, repr=False)

    @property
    def done(self) -> bool:
        return self._done.is_set()

    def poll(self) -> dict[str, Any] | None:
        if not self.background_progress:
            return self._poll_once()
        with self._lock:
            if not self._done.is_set():
                return None
            if self._error is not None:
                raise self._error
            return self._result

    def progress(
        self,
        *,
        metadata_drained: bool = False,
        backend_progressed: bool = False,
    ) -> bool:
        """Advance one request from a dedicated receiver progress thread."""
        with self._lock:
            if self._closed or self._result is not None or self._error is not None:
                return True
            try:
                self._result = self._poll_once(
                    metadata_drained=metadata_drained,
                    backend_progressed=backend_progressed,
                )
            except Exception as exc:
                self._error = exc
            return self._result is not None or self._error is not None

    @property
    def prepared_in_background(self) -> bool:
        return self.background_progress and self.result_preparer is not None

    def prepare_result(self) -> None:
        """Finish pure CPU reconstruction without occupying the scheduler loop."""
        with self._lock:
            if self._closed:
                return
            result = self._result
            error = self._error
        if error is None and result is not None and self.result_preparer is not None:
            try:
                timing = result.get("encoder_timing")
                if timing is not None:
                    timing["language_prepare_start_ns"] = time.time_ns()
                self.result_preparer(self.recv_req, result)
            except Exception as exc:
                error = exc
        with self._lock:
            if error is not None:
                self._error = error
            self._done.set()

    def _poll_once(
        self,
        *,
        metadata_drained: bool = False,
        backend_progressed: bool = False,
    ) -> dict[str, Any] | None:
        for future in self.registration_futures:
            if future.done():
                future.result()  # error re-thrown to the scheduler main thread

        # The ZMQ message contains EmbeddingData metadata (part identity,
        # shape/dtype, and transfer endpoints); the backend pulls the actual
        # embedding separately through the receiver backend.
        data = (
            self.metadata_router.pop(self.metadata_req_ids)
            if metadata_drained
            else self.metadata_router.poll(self.metadata_req_ids)
        )
        if data is not None:
            data.receive_metadata_ns = time.time_ns()
            validate_encoder_response(
                data,
                self.accumulator.num_parts,
                set(self.sessions),
                {
                    part_idx
                    for part_idx in range(self.accumulator.num_parts)
                    if self.accumulator.has_part(part_idx)
                },
            )
            self.sessions[data.part_idx] = (data, self.backend.start(data))

        for part_idx, (part_data, session) in list(self.sessions.items()):
            embedding = (
                session.poll(refresh_backend=False) if backend_progressed else session.poll()
            )
            if embedding is None:
                continue
            for key, value in getattr(session, "timing_meta", {}).items():
                setattr(part_data, key, value)
            part_data.receive_embedding_ns = time.time_ns()
            self.accumulator.add(part_data, embedding)
            self.sessions.pop(part_idx)
            session.close()

        if not self.accumulator.ready:
            return None
        timing = self.accumulator.get_timing_meta()
        timing["receive_done_ns"] = time.time_ns()
        timing["receive_concat_start_ns"] = time.time_ns()
        embeddings = self.accumulator.get_embedding(is_concat=True)
        timing["receive_concat_done_ns"] = time.time_ns()
        timing["receive_extra_meta_start_ns"] = time.time_ns()
        extra_meta = self.accumulator.get_mm_extra_meta()
        timing["receive_extra_meta_done_ns"] = time.time_ns()
        result = {
            "embeddings": embeddings,
            **extra_meta,
        }
        timing["receive_result_ready_ns"] = time.time_ns()
        result["encoder_timing"] = timing
        return result

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            for future in self.registration_futures:
                future.cancel()
            for _, session in self.sessions.values():
                session.close()
            self.sessions.clear()
            self.metadata_router.unregister(self.metadata_req_ids)


class EncoderClient:
    def __init__(
        self,
        host: str,
        backend: EncoderReceiverBackend,
        encoder_urls: list[str],
        registration_workers: int,
        registration_timeout: float | None,
        background_progress: bool = False,
        progress_interval_s: float = 0.001,
    ) -> None:
        self._backend = backend
        self._encoder_urls = list(encoder_urls)
        self._executor = ThreadPoolExecutor(max_workers=max(1, registration_workers))
        self._registration_client = httpx.Client(timeout=registration_timeout)
        self._background_progress = bool(background_progress)
        self._result_preparer: (
            Callable[[TokenizedGenerateReqInput, dict[str, Any]], None] | None
        ) = None
        self._progress_interval_s = max(0.0001, float(progress_interval_s))
        self._pending: dict[int, PendingEncoderRequest] = {}
        self._preparing: dict[int, PendingEncoderRequest] = {}
        self._pending_lock = threading.Lock()
        self._completed: SimpleQueue[PendingEncoderRequest] = SimpleQueue()
        self._completed_ready = threading.Event()
        self._prepare_executor = (
            ThreadPoolExecutor(
                max_workers=min(2, max(1, registration_workers)),
                thread_name_prefix="encoder-language-prepare",
            )
            if self._background_progress
            else None
        )
        self._progress_stop = threading.Event()
        self._progress_ready = threading.Event()
        self._progress_error: Exception | None = None
        self._metadata_router: EncoderMetadataRouter | None = None
        self._progress_thread: threading.Thread | None = None
        if self._background_progress:
            self._progress_thread = threading.Thread(
                target=self._progress_loop,
                args=(host,),
                name="encoder-receiver-progress",
                daemon=True,
            )
            self._progress_thread.start()
            if not self._progress_ready.wait(30):
                raise TimeoutError("timed out starting encoder receiver progress thread")
            if self._progress_error is not None:
                raise RuntimeError("failed to start encoder receiver progress thread") from (
                    self._progress_error
                )
        else:
            self._metadata_router = EncoderMetadataRouter(host)

    @property
    def _router(self) -> EncoderMetadataRouter:
        if self._metadata_router is None:
            raise RuntimeError("encoder metadata router is not initialized")
        return self._metadata_router

    @property
    def background_progress(self) -> bool:
        return self._background_progress

    def set_result_preparer(
        self,
        preparer: Callable[[TokenizedGenerateReqInput, dict[str, Any]], None],
    ) -> None:
        self._result_preparer = preparer

    def drain_completed(self) -> list[PendingEncoderRequest]:
        completed = []
        while True:
            self._completed_ready.clear()
            while True:
                try:
                    completed.append(self._completed.get_nowait())
                except Empty:
                    break
            if not self._completed_ready.is_set():
                return completed

    def has_completed(self) -> bool:
        return self._completed_ready.is_set()

    def receive(self, request: TokenizedGenerateReqInput) -> PendingEncoderRequest:
        registrations = plan_encoder_registrations(request, self._encoder_urls)
        metadata_req_ids = tuple(registration[1] for registration in registrations)
        router = self._router
        router.register(metadata_req_ids)
        registration_futures = []
        try:
            for registration in registrations:
                registration_futures.append(
                    self._executor.submit(
                        register_scheduler_receiver,
                        registration,
                        self._metadata_router.receive_url,
                        self._registration_client,
                    )
                )
        except Exception:
            for future in registration_futures:
                future.cancel()
            router.unregister(metadata_req_ids)
            raise
        pending = PendingEncoderRequest(
            recv_req=request,
            started_at=time.monotonic(),
            metadata_router=router,
            metadata_req_ids=metadata_req_ids,
            registration_futures=tuple(registration_futures),
            accumulator=MultiModalEmbeddingData(len(registrations)),
            backend=self._backend,
            result_preparer=self._result_preparer,
            background_progress=self._background_progress,
        )
        if self._background_progress:
            with self._pending_lock:
                self._pending[id(pending)] = pending
        return pending

    def _progress_loop(self, host: str) -> None:
        try:
            self._metadata_router = EncoderMetadataRouter(host)
        except Exception as exc:
            self._progress_error = exc
            self._progress_ready.set()
            return
        self._progress_ready.set()
        try:
            while not self._progress_stop.wait(self._progress_interval_s):
                metadata_drained = False
                try:
                    self._router.drain()
                    metadata_drained = True
                except Exception:
                    logger.exception("Failed to drain encoder metadata")

                backend_progressed = False
                progress_backend = getattr(self._backend, "progress", None)
                if progress_backend is not None:
                    try:
                        backend_progressed = bool(progress_backend())
                    except Exception:
                        logger.exception("Failed to progress encoder receive backend")

                with self._pending_lock:
                    pending = list(self._pending.items())
                for key, request in pending:
                    if request.progress(
                        metadata_drained=metadata_drained,
                        backend_progressed=backend_progressed,
                    ):
                        self._submit_prepare(key, request)
        finally:
            self._router.close()

    def _submit_prepare(self, key: int, request: PendingEncoderRequest) -> None:
        with self._pending_lock:
            if self._pending.get(key) is not request:
                return
            self._pending.pop(key)
            self._preparing[key] = request
        result = request._result
        if result is not None:
            timing = result.get("encoder_timing")
            if timing is not None:
                timing["language_prepare_submit_ns"] = time.time_ns()
        executor = self._prepare_executor
        assert executor is not None
        future = executor.submit(request.prepare_result)
        future.add_done_callback(
            lambda completed_future: self._publish_completed(
                key,
                request,
                completed_future,
            )
        )

    def _publish_completed(
        self,
        key: int,
        request: PendingEncoderRequest,
        future: Future[None],
    ) -> None:
        try:
            future.result()
        except Exception as exc:
            with request._lock:
                request._error = exc
                request._done.set()
        with self._pending_lock:
            if self._preparing.get(key) is request:
                self._preparing.pop(key, None)
        self._completed.put(request)
        self._completed_ready.set()

    def close(self) -> None:
        self._progress_stop.set()
        if self._progress_thread is not None:
            self._progress_thread.join()
        with self._pending_lock:
            pending = [*self._pending.values(), *self._preparing.values()]
            self._pending.clear()
            self._preparing.clear()
        for request in pending:
            request.close()
        if self._prepare_executor is not None:
            self._prepare_executor.shutdown(cancel_futures=True)
        self._backend.close()
        self._executor.shutdown(cancel_futures=True)
        self._registration_client.close()
        if self._progress_thread is None:
            self._router.close()


class EncoderRequestDispatcher:
    """Dispatch encoder requests through a reusable HTTP client."""

    def __init__(self, timeout: float | None) -> None:
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None

    def dispatch(
        self,
        request: GenerateReqInput,
        encoder_urls: list[str],
    ) -> tuple[dict[Modality, list[int]], asyncio.Task[None]]:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout)
        client = self._client

        dispatch_start_ns = time.time_ns()
        items_by_modality = {}
        for name, modality in (
            ("image_data", Modality.IMAGE),
            ("video_data", Modality.VIDEO),
            ("audio_data", Modality.AUDIO),
        ):
            data = getattr(request, name, None)
            if data is None:
                continue
            items = []
            for item in flatten_nested_list(data):
                if item is None:
                    continue
                if isinstance(item, ImageData):
                    item = item.url
                elif isinstance(item, dict) and "url" in item:
                    item = item["url"]
                items.append(item)
            if items:
                items_by_modality[modality] = items

        assignments = {}
        encoder_indices = list(range(len(encoder_urls)))
        random.shuffle(encoder_indices)
        offset = 0
        for modality, items in items_by_modality.items():
            base, remainder = divmod(len(items), len(encoder_urls))
            counts = [base] * len(encoder_urls)
            for index in range(remainder):
                counts[encoder_indices[(offset + index) % len(encoder_urls)]] += 1
            assignments[modality] = counts
            offset = (offset + remainder) % len(encoder_urls)

        num_parts = sum(count > 0 for counts in assignments.values() for count in counts)
        encode_requests = []
        for modality, counts in assignments.items():
            items = items_by_modality[modality]
            item_offset = 0
            for encoder_idx, count in enumerate(counts):
                if count == 0:
                    continue
                part_idx = len(encode_requests)
                encode_requests.append(
                    (
                        encoder_urls[encoder_idx],
                        {
                            "req_id": create_part_req_id(request.rid, part_idx),
                            "dispatch_start_ns": dispatch_start_ns,
                            "mm_items": items[item_offset : item_offset + count],
                            "num_parts": num_parts,
                            "part_idx": part_idx,
                            "modality": modality.name,
                        },
                    )
                )
                item_offset += count

        async def send_encode_requests() -> None:
            async def send_one(encoder_url: str, payload: dict[str, Any]) -> None:
                url = f"{encoder_url.rstrip('/')}/encode"
                if _DISPATCH_ORJSON:
                    response = await client.post(
                        url,
                        content=orjson.dumps(payload),
                        headers={"content-type": "application/json"},
                    )
                else:
                    response = await client.post(url, json=payload)
                response.raise_for_status()

            results = await asyncio.gather(
                *(send_one(*encode_request) for encode_request in encode_requests),
                return_exceptions=True,
            )
            for result in results:
                if isinstance(result, Exception):
                    raise result

        task = asyncio.create_task(
            send_encode_requests(),
            name=f"encoder-dispatch-{request.rid}",
        )

        def finish(completed: asyncio.Task[None]) -> None:
            if completed.cancelled():
                return
            try:
                completed.result()
            except Exception:
                logger.exception("Encoder dispatch failed. rid=%s", request.rid)

        task.add_done_callback(finish)
        return assignments, task

    async def close(self) -> None:
        client, self._client = self._client, None
        if client is not None:
            await client.aclose()


def create_encoder_client(
    server_args,
    mesh: Any,
) -> EncoderClient:
    channel_number = max(1, int(server_args.disaggregation_channel_number))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    transfer_timeout = server_args.encoder_request_timeout_seconds

    if server_args.simulate_compute:
        from sgl_jax.srt.disaggregation.encoder.sim_transfer import SimReceiverBackend

        host = server_args.disaggregation_host_ip or "127.0.0.1"
        backend = SimReceiverBackend(
            sharding,
            server_args.simulate_transfer_ms_per_mb,
            server_args.simulate_network_rtt_ms,
            parallelism=channel_number,
            pool_size=server_args.encoder_transfer_pool_size,
            transfer_timeout_s=transfer_timeout,
        )
    else:
        from sgl_jax.raiden import require_raiden_preloaded
        from sgl_jax.srt.disaggregation.encoder.raiden_transfer import (
            RaidenReceiverBackend,
        )
        from sgl_jax.srt.disaggregation.host_ip import resolve_host_ip

        require_raiden_preloaded()
        if transfer_timeout <= 0:
            raise ValueError("Raiden requires a positive encoder request timeout")
        host = resolve_host_ip(server_args.disaggregation_host_ip)
        backend = RaidenReceiverBackend(
            host=host,
            sharding=sharding,
            parallelism=channel_number,
            pool_size=server_args.encoder_transfer_pool_size,
            transfer_timeout_s=transfer_timeout,
        )

    control_timeout = server_args.encoder_control_timeout_seconds
    return EncoderClient(
        host=host,
        backend=backend,
        encoder_urls=server_args.encoder_urls,
        registration_workers=channel_number,
        registration_timeout=None if control_timeout <= 0 else control_timeout,
        background_progress=getattr(
            server_args,
            "encoder_receiver_background_progress",
            False,
        ),
    )
