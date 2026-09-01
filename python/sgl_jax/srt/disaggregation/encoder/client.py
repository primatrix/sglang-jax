from __future__ import annotations

import asyncio
import logging
import random
import threading
import time
from collections import deque
from concurrent.futures import Future, InvalidStateError, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx
import jax
import orjson
import zmq

from sgl_jax.srt.disaggregation.encoder.embedding_data import (
    EmbeddingData,
    MultiModalEmbeddingData,
)
from sgl_jax.srt.managers.io_struct import (
    GenerateReqInput,
    ImageData,
    TokenizedGenerateReqInput,
)
from sgl_jax.srt.multimodal.common.modality_enum import Modality, flatten_nested_list

logger = logging.getLogger(__name__)


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


def submit_scheduler_receiver_registrations(
    executor: ThreadPoolExecutor,
    registrations: list[tuple[str, str, Modality | None]],
    receive_url: str,
    client: httpx.Client,
) -> Future[None]:
    """Submit independent encoder registrations and aggregate their futures."""
    combined: Future[None] = Future()
    children: list[Future[None]] = []
    lock = threading.Lock()
    remaining = len(registrations)
    first_exception: BaseException | None = None
    child_cancelled = False

    if remaining == 0:
        combined.set_result(None)
        return combined

    def finish(child: Future[None]) -> None:
        nonlocal child_cancelled, first_exception, remaining
        action = None
        with lock:
            remaining -= 1
            if combined.done():
                return
            if child.cancelled():
                child_cancelled = True
            else:
                exception = child.exception()
                if exception is not None and first_exception is None:
                    first_exception = exception

            # Match the old gather(return_exceptions=True) behavior: let every
            # registration finish before surfacing the first failure.
            if remaining != 0:
                return
            if child_cancelled:
                action = ("cancel", None)
            elif first_exception is not None:
                action = ("exception", first_exception)
            else:
                action = ("result", None)

        # ``combined.cancel()`` invokes callbacks synchronously, so complete it
        # outside ``lock`` to avoid re-entering ``finish`` while cancelling peers.
        try:
            if action is None:
                return
            kind, value = action
            if kind == "cancel":
                combined.cancel()
            elif kind == "exception":
                combined.set_exception(value)
            else:
                combined.set_result(None)
        except InvalidStateError:
            # A caller may cancel the aggregate between the done check and the
            # completion above. Its cancellation already represents the result.
            pass

    def cancel_children(done: Future[None]) -> None:
        if done.cancelled():
            for child in children:
                child.cancel()

    combined.add_done_callback(cancel_children)
    try:
        for registration in registrations:
            child = executor.submit(
                register_scheduler_receiver,
                registration,
                receive_url,
                client,
            )
            children.append(child)
            child.add_done_callback(finish)
    except Exception:
        for child in children:
            child.cancel()
        raise
    return combined


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


def build_encoder_result(accumulator: MultiModalEmbeddingData) -> dict[str, Any]:
    timing = accumulator.get_timing_meta()
    timing["receive_done_ns"] = time.time_ns()
    return {
        "embeddings": accumulator.get_embedding(is_concat=True),
        "encoder_timing": timing,
        **accumulator.get_mm_extra_meta(),
    }


class EncoderReceiveSession(Protocol):
    def poll(self) -> jax.Array | None: ...

    def close(self) -> None: ...


class EncoderReceiverBackend(Protocol):
    def start(self, data: EmbeddingData) -> EncoderReceiveSession: ...

    def close(self) -> None: ...


class EncoderMetadataReceiver(Protocol):
    def poll(self, req_ids: tuple[str, ...]) -> Any | None: ...

    def unregister(self, req_ids: tuple[str, ...]) -> None: ...


class EncoderMetadataRouter:
    """Route one scheduler-wide metadata socket to pending encoder requests."""

    def __init__(self, host: str) -> None:
        self._receiver = zmq.Context.instance().socket(zmq.PULL)
        self._receiver.setsockopt(zmq.LINGER, 0)
        port = self._receiver.bind_to_random_port(f"tcp://{host}")
        self.receive_url = f"{host}:{port}"
        self._queues: dict[str, deque[Any]] = {}

    def register(self, req_ids: tuple[str, ...]) -> None:
        routes = set(req_ids)
        if len(routes) != len(req_ids) or not routes.isdisjoint(self._queues):
            raise ValueError(f"duplicate encoder metadata routes: {req_ids}")
        self._queues.update((req_id, deque()) for req_id in req_ids)

    def poll(self, req_ids: tuple[str, ...]) -> Any | None:
        while True:
            try:
                data = self._receiver.recv_pyobj(zmq.NOBLOCK)
            except zmq.Again:
                break
            queue = self._queues.get(getattr(data, "req_id", None))
            if queue is not None:
                queue.append(data)

        for req_id in req_ids:
            queue = self._queues.get(req_id)
            if queue:
                return queue.popleft()
        return None

    def unregister(self, req_ids: tuple[str, ...]) -> None:
        for req_id in req_ids:
            self._queues.pop(req_id, None)

    def close(self) -> None:
        self._queues.clear()
        self._receiver.close()


@dataclass(slots=True)
class PendingEncoderRequest:
    recv_req: TokenizedGenerateReqInput
    started_at: float
    metadata_router: EncoderMetadataReceiver
    metadata_req_ids: tuple[str, ...]
    register_future: Future[None]
    accumulator: MultiModalEmbeddingData
    backend: EncoderReceiverBackend
    # Keep each part's metadata alongside its in-flight transfer session so the
    # completed embedding can later be assembled with the correct modality and grid.
    sessions: dict[int, tuple[EmbeddingData, EncoderReceiveSession]] = field(default_factory=dict)

    def poll(self) -> dict[str, Any] | None:
        if self.register_future.done():
            self.register_future.result()  # error re-thrown to the scheduler main thread

        # The ZMQ message contains EmbeddingData metadata (part identity,
        # shape/dtype, and transfer endpoints); the backend pulls the actual
        # embedding separately in _start_receive().
        data = self.metadata_router.poll(self.metadata_req_ids)
        if data is not None:
            self._start_receive(data)

        for part_idx, (part_data, session) in list(self.sessions.items()):
            embedding = session.poll()
            if embedding is None:
                continue
            self.accumulator.add(part_data, embedding)
            self.sessions.pop(part_idx)
            session.close()

        if not self.accumulator.ready:
            return None
        return build_encoder_result(self.accumulator)

    def _start_receive(self, data: Any) -> None:
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

    def close(self) -> None:
        self.register_future.cancel()
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
        executor: ThreadPoolExecutor,
        registration_timeout: float | None,
    ) -> None:
        self._backend = backend
        self._encoder_urls = list(encoder_urls)
        self._executor = executor
        self._registration_client = httpx.Client(timeout=registration_timeout)
        self._metadata_router = EncoderMetadataRouter(host)

    def receive(self, request: TokenizedGenerateReqInput) -> PendingEncoderRequest:
        registrations = plan_encoder_registrations(request, self._encoder_urls)
        metadata_req_ids = tuple(registration[1] for registration in registrations)
        self._metadata_router.register(metadata_req_ids)
        try:
            register_future = submit_scheduler_receiver_registrations(
                self._executor,
                registrations,
                self._metadata_router.receive_url,
                self._registration_client,
            )
        except Exception:
            self._metadata_router.unregister(metadata_req_ids)
            raise
        return PendingEncoderRequest(
            recv_req=request,
            started_at=time.monotonic(),
            metadata_router=self._metadata_router,
            metadata_req_ids=metadata_req_ids,
            register_future=register_future,
            accumulator=MultiModalEmbeddingData(len(registrations)),
            backend=self._backend,
        )

    def close(self) -> None:
        self._backend.close()
        self._executor.shutdown(cancel_futures=True)
        self._registration_client.close()
        self._metadata_router.close()


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
        return dispatch_encoder_request(request, encoder_urls, self._client)

    async def close(self) -> None:
        client, self._client = self._client, None
        if client is not None:
            await client.aclose()


def dispatch_encoder_request(
    request: GenerateReqInput,
    encoder_urls: list[str],
    client: httpx.AsyncClient,
) -> tuple[dict[Modality, list[int]], asyncio.Task[None]]:
    if not isinstance(request.rid, str):
        raise ValueError("encoder request requires a single rid")
    if not encoder_urls:
        raise ValueError("encoder_urls is required")

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
            response = await client.post(
                f"{encoder_url.rstrip('/')}/encode",
                content=orjson.dumps(payload),
                headers={"Content-Type": "application/json"},
            )
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


def create_encoder_client(
    server_args,
    mesh: Any,
) -> EncoderClient:
    if server_args.simulate_compute:
        from sgl_jax.srt.disaggregation.encoder.sim_transfer import create_sim_client

        return create_sim_client(server_args, mesh)

    from sgl_jax.srt.disaggregation.encoder.raiden import create_raiden_client

    return create_raiden_client(server_args, mesh)
