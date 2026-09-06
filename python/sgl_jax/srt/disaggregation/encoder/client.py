from __future__ import annotations

import logging
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
import zmq

from sgl_jax.srt.disaggregation.encoder.dispatcher import create_part_req_id
from sgl_jax.srt.disaggregation.encoder.embedding_data import (
    EmbeddingData,
    MultiModalEmbeddingData,
)
from sgl_jax.srt.managers.io_struct import TokenizedGenerateReqInput
from sgl_jax.srt.multimodal.common.modality_enum import Modality
from sgl_jax.srt.multimodal.in_model.embedding_view import PooledEmbedding
from sgl_jax.srt.request_time_stats import mark_time_stats

logger = logging.getLogger(__name__)


def plan_encoder_registrations(
    request: TokenizedGenerateReqInput,
) -> list[tuple[str, str, Modality]]:
    """Register exactly the modality parts dispatched by the tokenizer."""
    if not isinstance(request.rid, str):
        raise ValueError("encoder request requires a single rid")
    encoder_urls = request.encoder_urls
    if not encoder_urls:
        raise ValueError("encoder_urls is required")

    if request.num_items_assigned is None:
        raise ValueError("num_items_assigned is required")

    registrations: list[tuple[str, str, Modality]] = []
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
    registration: tuple[str, str, Modality],
    receive_url: str,
    client: httpx.Client,
) -> None:
    encoder_url, req_id, modality = registration
    payload = {
        "req_id": req_id,
        "receive_count": 1,
        "receive_url": receive_url,
        "modality": modality.name,
    }
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

    def poll(self, *, refresh_backend: bool = True) -> jax.Array | PooledEmbedding | None:
        if self._closed:
            return None
        if self._session is None:
            if not self._future.done():
                return None
            self._session = self._future.result()
        return self._session.poll(refresh_backend=refresh_backend)

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
    def progress(self) -> bool: ...

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
    result_preparer: Callable[[TokenizedGenerateReqInput, dict[str, Any]], None]
    # Keep each part's metadata alongside its in-flight transfer session so the
    # completed embedding can later be assembled with the correct modality and grid.
    sessions: dict[int, tuple[EmbeddingData, EncoderReceiveSession]] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _result: dict[str, Any] | None = field(default=None, init=False, repr=False)
    _error: Exception | None = field(default=None, init=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)
    _claimed: bool = field(default=False, init=False, repr=False)
    _done: threading.Event = field(default_factory=threading.Event, init=False, repr=False)

    @property
    def done(self) -> bool:
        return self._done.is_set()

    def poll(self) -> dict[str, Any] | None:
        with self._lock:
            if not self._done.is_set():
                return None
            if self._error is not None:
                raise self._error
            self._claimed = True
            return self._result

    def progress(
        self,
        *,
        backend_progressed: bool = False,
    ) -> bool:
        """Advance one request from a dedicated receiver progress thread."""
        with self._lock:
            if self._closed or self._result is not None or self._error is not None:
                return True
            try:
                self._result = self._poll_once(
                    backend_progressed=backend_progressed,
                )
            except Exception as exc:
                self._error = exc
            return self._result is not None or self._error is not None

    def prepare_result(self) -> None:
        """Finish pure CPU reconstruction without occupying the scheduler loop."""
        with self._lock:
            if self._closed:
                return
            result = self._result
            error = self._error
        if error is None and result is not None:
            try:
                timing = result.get("encoder_timing")
                mark_time_stats(timing, "language_prepare_start_ns")
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
        backend_progressed: bool = False,
    ) -> dict[str, Any] | None:
        for future in self.registration_futures:
            if future.done():
                future.result()  # error re-thrown to the scheduler main thread

        # The ZMQ message contains EmbeddingData metadata (part identity,
        # shape/dtype, and transfer endpoints); the backend pulls the actual
        # embedding separately through the receiver backend.
        data = self.metadata_router.pop(self.metadata_req_ids)
        if data is not None:
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
            mark_time_stats(data.timing, "receive_metadata_ns")
            self.sessions[data.part_idx] = (data, self.backend.start(data))

        for part_idx, (part_data, session) in list(self.sessions.items()):
            embedding = (
                session.poll(refresh_backend=False) if backend_progressed else session.poll()
            )
            if embedding is None:
                continue
            mark_time_stats(part_data.timing, "receive_done_ns")
            self.accumulator.add(part_data, embedding)
            self.sessions.pop(part_idx)
            session.close()

        if not self.accumulator.ready:
            return None
        return {
            "embeddings": self.accumulator.get_embedding(is_concat=True),
            **self.accumulator.get_mm_extra_meta(),
            "encoder_timing": self.accumulator.get_timing_meta(),
        }

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
            if not self._claimed:
                self.accumulator.release()
            self.metadata_router.unregister(self.metadata_req_ids)


class EncoderClient:
    def __init__(
        self,
        host: str,
        backend: EncoderReceiverBackend,
        result_preparer: Callable[[TokenizedGenerateReqInput, dict[str, Any]], None],
        registration_workers: int,
        registration_timeout: float | None,
        progress_interval_s: float = 0.001,
    ) -> None:
        self._backend = backend
        self._executor = ThreadPoolExecutor(max_workers=max(1, registration_workers))
        self._registration_client = httpx.Client(timeout=registration_timeout)
        self._result_preparer = result_preparer
        self._progress_interval_s = max(0.0001, float(progress_interval_s))
        self._pending: dict[int, PendingEncoderRequest] = {}
        self._preparing: dict[int, PendingEncoderRequest] = {}
        self._pending_lock = threading.Lock()
        self._completed: SimpleQueue[PendingEncoderRequest] = SimpleQueue()
        self._completed_ready = threading.Event()
        self._prepare_executor = ThreadPoolExecutor(
            max_workers=min(2, max(1, registration_workers)),
            thread_name_prefix="encoder-language-prepare",
        )
        self._progress_stop = threading.Event()
        self._progress_ready = threading.Event()
        self._progress_error: Exception | None = None
        self._metadata_router: EncoderMetadataRouter | None = None
        self._progress_thread = threading.Thread(
            target=self._progress_loop,
            args=(host,),
            name="encoder-receiver-progress",
            daemon=True,
        )
        self._progress_thread.start()
        if not self._progress_ready.wait(30):
            self.close()
            raise TimeoutError("timed out starting encoder receiver progress thread")
        if self._progress_error is not None:
            self.close()
            raise RuntimeError(
                "failed to start encoder receiver progress thread"
            ) from self._progress_error

    @property
    def _router(self) -> EncoderMetadataRouter:
        if self._metadata_router is None:
            raise RuntimeError("encoder metadata router is not initialized")
        return self._metadata_router

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
        registrations = plan_encoder_registrations(request)
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
        )
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
                try:
                    self._router.drain()
                except Exception:
                    logger.exception("Failed to drain encoder metadata")

                backend_progressed = False
                try:
                    backend_progressed = self._backend.progress()
                except Exception:
                    logger.exception("Failed to progress encoder receive backend")

                with self._pending_lock:
                    pending = list(self._pending.items())
                for key, request in pending:
                    if request.progress(
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
        future = self._prepare_executor.submit(request.prepare_result)
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
        self._progress_thread.join()
        with self._pending_lock:
            pending = [*self._pending.values(), *self._preparing.values()]
            self._pending.clear()
            self._preparing.clear()
        for request in pending:
            request.close()
        self._prepare_executor.shutdown(cancel_futures=True)
        self._backend.close()
        self._executor.shutdown(cancel_futures=True)
        self._registration_client.close()


def create_encoder_client(
    server_args,
    mesh: Any,
    result_preparer: Callable[[TokenizedGenerateReqInput, dict[str, Any]], None],
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
        from sgl_jax.srt.disaggregation.encoder.raiden_receiver import (
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
        result_preparer=result_preparer,
        registration_workers=channel_number,
        registration_timeout=None if control_timeout <= 0 else control_timeout,
    )
