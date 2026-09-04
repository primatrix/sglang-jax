from __future__ import annotations

import asyncio
import json
import threading
import time
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.disaggregation.encoder import client as encoder_client
from sgl_jax.srt.disaggregation.encoder.embedding_data import EmbeddingData
from sgl_jax.srt.managers.io_struct import GenerateReqInput
from sgl_jax.srt.multimodal.common.modality_enum import Modality


def test_encoder_metadata_router_routes_shared_socket_messages():
    class FakeReceiver:
        def __init__(self):
            self.messages = [
                SimpleNamespace(req_id="request-1"),
                SimpleNamespace(req_id="request-0"),
            ]

        def recv_pyobj(self, _flags):
            if not self.messages:
                raise encoder_client.zmq.Again()
            return self.messages.pop(0)

    router = object.__new__(encoder_client.EncoderMetadataRouter)
    router._receiver = FakeReceiver()
    router._queues = {}
    router._lock = threading.Lock()
    router.register(("request-0",))
    router.register(("request-1",))

    router.drain()
    assert router.pop(("request-0",)).req_id == "request-0"
    assert router.pop(("request-1",)).req_id == "request-1"


def test_encoder_receiver_reuses_http_client(monkeypatch):
    clients = []

    class FakeClient:
        def __init__(self, *, timeout) -> None:
            self.timeout = timeout
            self.closed = False
            clients.append(self)

        def close(self) -> None:
            self.closed = True

    class FakeBackend:
        def close(self) -> None:
            pass

    monkeypatch.setattr(encoder_client.httpx, "Client", FakeClient)
    client = encoder_client.EncoderClient(
        host="127.0.0.1",
        backend=FakeBackend(),
        encoder_urls=["http://encoder"],
        registration_workers=1,
        registration_timeout=12.0,
    )

    client.close()

    assert len(clients) == 1
    assert clients[0].timeout == 12.0
    assert clients[0].closed


def test_encoder_receiver_reuses_metadata_socket(monkeypatch):
    receive_urls = []

    class FakeClient:
        def __init__(self, *, timeout) -> None:
            pass

        def close(self) -> None:
            pass

    class FakeBackend:
        def close(self) -> None:
            pass

    def register(registration, receive_url, client):
        del registration, client
        receive_urls.append(receive_url)

    monkeypatch.setattr(encoder_client.httpx, "Client", FakeClient)
    monkeypatch.setattr(encoder_client, "register_scheduler_receiver", register)
    client = encoder_client.EncoderClient(
        host="127.0.0.1",
        backend=FakeBackend(),
        encoder_urls=["http://encoder"],
        registration_workers=1,
        registration_timeout=12.0,
    )
    requests = [
        SimpleNamespace(
            rid=f"request-{index}",
            encoder_urls=None,
            num_items_assigned=None,
        )
        for index in range(2)
    ]

    pending = [client.receive(request) for request in requests]
    for request in pending:
        request.registration_futures[0].result(timeout=1)
        request.close()
    client.close()

    assert len(receive_urls) == 2
    assert receive_urls[0] == receive_urls[1]


def test_encoder_receiver_registers_parts_concurrently(monkeypatch):
    barrier = threading.Barrier(2)
    posts = []

    class FakeResponse:
        def raise_for_status(self) -> None:
            pass

    class FakeClient:
        def __init__(self, *, timeout) -> None:
            pass

        def post(self, url, *, json):
            posts.append((url, json["req_id"]))
            barrier.wait(timeout=1)
            return FakeResponse()

        def close(self) -> None:
            pass

    class FakeBackend:
        def close(self) -> None:
            pass

    monkeypatch.setattr(encoder_client.httpx, "Client", FakeClient)
    client = encoder_client.EncoderClient(
        host="127.0.0.1",
        backend=FakeBackend(),
        encoder_urls=["http://encoder-0", "http://encoder-1"],
        registration_workers=2,
        registration_timeout=12.0,
    )
    pending = client.receive(
        SimpleNamespace(
            rid="request",
            encoder_urls=None,
            num_items_assigned={Modality.IMAGE: [1, 1]},
        )
    )
    try:
        for future in pending.registration_futures:
            future.result(timeout=2)
    finally:
        pending.close()
        client.close()

    assert sorted(posts) == [
        ("http://encoder-0/scheduler_receive_url", "request_local_part_0"),
        ("http://encoder-1/scheduler_receive_url", "request_local_part_1"),
    ]


def test_encoder_receiver_background_progresses_without_scheduler_poll(monkeypatch):
    class FakeResponse:
        def raise_for_status(self) -> None:
            pass

    class FakeClient:
        def __init__(self, *, timeout) -> None:
            del timeout

        def post(self, _url, *, json):
            del json
            return FakeResponse()

        def close(self) -> None:
            pass

    class FakeSession:
        timing_meta = {}

        def poll(self, *, refresh_backend=True):
            assert not refresh_backend
            return jnp.zeros((2, 3))

        def close(self) -> None:
            pass

    class FakeBackend:
        progress_calls = 0

        def start(self, _data):
            return FakeSession()

        def progress(self):
            self.progress_calls += 1
            return True

        def close(self) -> None:
            pass

    class FakeRouter:
        instance = None

        def __init__(self, host) -> None:
            self.receive_url = f"{host}:1234"
            self.message = None
            self.closed = False
            FakeRouter.instance = self

        def register(self, _req_ids) -> None:
            pass

        def poll(self, _req_ids):
            message, self.message = self.message, None
            return message

        def drain(self):
            pass

        def pop(self, _req_ids):
            message, self.message = self.message, None
            return message

        def unregister(self, _req_ids) -> None:
            pass

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(encoder_client.httpx, "Client", FakeClient)
    monkeypatch.setattr(encoder_client, "EncoderMetadataRouter", FakeRouter)
    client = encoder_client.EncoderClient(
        host="127.0.0.1",
        backend=FakeBackend(),
        encoder_urls=["http://encoder"],
        registration_workers=1,
        registration_timeout=12.0,
        background_progress=True,
    )
    prepare_threads = []

    def prepare_result(request, result):
        prepare_threads.append(threading.current_thread().name)
        request.prepared_shape = result["embeddings"][Modality.IMAGE].shape

    client.set_result_preparer(prepare_result)
    request = SimpleNamespace(rid="request-0", encoder_urls=None, num_items_assigned=None)
    pending = client.receive(request)
    FakeRouter.instance.message = EmbeddingData(
        req_id="request-0",
        num_parts=1,
        part_idx=0,
        grid_dim=None,
        modality=Modality.IMAGE,
    )
    result = None
    deadline = time.monotonic() + 1
    while result is None and time.monotonic() < deadline:
        result = pending.poll()
        time.sleep(0.001)
    try:
        assert result is not None
        assert pending.done
        assert client._backend.progress_calls > 0
        assert request.prepared_shape == (2, 3)
        assert prepare_threads[0].startswith("encoder-language-prepare")
        assert result["embeddings"][Modality.IMAGE].shape == (2, 3)
        timing = result["encoder_timing"]
        assert timing["receive_done_ns"] <= timing["receive_concat_start_ns"]
        assert timing["receive_concat_start_ns"] <= timing["receive_concat_done_ns"]
        assert timing["receive_concat_done_ns"] <= timing["receive_extra_meta_start_ns"]
        assert timing["receive_extra_meta_start_ns"] <= timing["receive_extra_meta_done_ns"]
        assert timing["receive_extra_meta_done_ns"] <= timing["receive_result_ready_ns"]
        assert timing["receive_result_ready_ns"] <= timing["language_prepare_submit_ns"]
        assert timing["language_prepare_submit_ns"] <= timing["language_prepare_start_ns"]
        completed = []
        deadline = time.monotonic() + 1
        while not completed and time.monotonic() < deadline:
            completed = client.drain_completed()
            time.sleep(0.001)
        assert completed == [pending]
        assert not client.has_completed()
    finally:
        pending.close()
        client.close()
    assert FakeRouter.instance.closed


def test_encoder_receiver_preserves_item_hashes_as_host_metadata():
    accumulator = encoder_client.MultiModalEmbeddingData(1)
    accumulator.add(
        EmbeddingData(
            req_id="request-0",
            num_parts=1,
            part_idx=0,
            grid_dim=np.asarray([[1, 2, 4]], dtype=np.int32),
            modality=Modality.IMAGE,
            item_hashes=[123],
        ),
        jnp.zeros((2, 3)),
    )

    metadata = accumulator.get_mm_extra_meta()

    assert metadata["item_hashes"] == {Modality.IMAGE: [123]}
    assert isinstance(metadata["img_grid_thw"], np.ndarray)


def test_encoder_request_dispatcher_reuses_http_client(monkeypatch):
    clients = []
    posts = []

    class FakeResponse:
        def raise_for_status(self) -> None:
            pass

    class FakeAsyncClient:
        def __init__(self, *, timeout, limits) -> None:
            self.timeout = timeout
            self.limits = limits
            self.closed = False
            clients.append(self)

        async def post(self, url, *, content, headers):
            assert headers == {"content-type": "application/json"}
            payload = json.loads(content)
            assert isinstance(payload["dispatch_start_ns"], int)
            posts.append((url, payload["req_id"]))
            return FakeResponse()

        async def aclose(self) -> None:
            self.closed = True

    monkeypatch.setattr(encoder_client.httpx, "AsyncClient", FakeAsyncClient)

    async def run() -> None:
        dispatcher = encoder_client.EncoderRequestDispatcher(timeout=12.0)
        tasks = []
        for rid in ("request-0", "request-1"):
            assignments, task = dispatcher.dispatch(
                GenerateReqInput(rid=rid, image_data="https://example.com/image.png"),
                ["http://encoder"],
            )
            assert sum(assignments.values(), []) == [1]
            tasks.append(task)

        await asyncio.gather(*tasks)
        await dispatcher.close()

    asyncio.run(run())

    assert len(clients) == 1
    assert clients[0].timeout == 12.0
    assert clients[0].limits.max_connections == 256
    assert clients[0].limits.max_keepalive_connections == 256
    assert clients[0].closed
    assert posts == [
        ("http://encoder/encode", "request-0_local_part_0"),
        ("http://encoder/encode", "request-1_local_part_0"),
    ]
