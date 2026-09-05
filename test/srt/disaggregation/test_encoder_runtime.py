from __future__ import annotations

import asyncio
import threading
import time
from contextlib import suppress
from types import SimpleNamespace
from typing import cast

import jax.numpy as jnp
import pytest

from sgl_jax.srt.disaggregation.encoder.embedding_data import EmbeddingData
from sgl_jax.srt.disaggregation.encoder.runtime import (
    EncoderRuntime,
    PreprocessedRequest,
    _EncodeJob,
    _ReadyJob,
)
from sgl_jax.srt.disaggregation.encoder.scheduler import DisaggEncoderScheduler
from sgl_jax.srt.disaggregation.encoder.server import EncoderServer
from sgl_jax.srt.disaggregation.encoder.sim_transfer import SimEncoderServerTransfer
from sgl_jax.srt.multimodal.common.modality_enum import Modality


class _TestEncoder:
    preprocess_concurrency = 8

    def __init__(self, encode, preprocess=None):
        self._encode = encode
        self._preprocess = preprocess

    async def preprocess_request(self, request):
        if self._preprocess is None:
            return request
        return await self._preprocess([request])

    @staticmethod
    def batch_key(prepared):
        request = prepared[0] if isinstance(prepared, list) else prepared
        if isinstance(request, dict):
            return request.get("modality"), request.get("token_count", 1)
        return request

    def _batch_requests(self, prepared):
        if self._preprocess is None:
            return prepared
        if all(isinstance(item, list) for item in prepared):
            return [request for item in prepared for request in item]
        return prepared[0] if len(prepared) == 1 else prepared

    def build_batch(self, prepared):
        return SimpleNamespace(inputs=self._batch_requests(prepared), transfer_specs=())

    def encode_packed(self, prepared):
        results = self._encode(prepared.inputs)
        return SimpleNamespace(
            batch=SimpleNamespace(token_counts=tuple(value.shape[0] for value, _ in results)),
            packed=jnp.concatenate([value for value, _ in results]),
            metadata=[metadata for _, metadata in results],
        )

    def metadata_for_packed(self, output):
        return output.metadata


class _FakeTransfer:
    def __init__(self):
        self.published = []
        self.released = []
        self.closed = False

    def reserve_batch_sync(self, transfer_ids):
        return list(transfer_ids)

    def precompile_packed_batches(self, specs):
        pass

    def stage_packed_batch_sync(self, reservations, packed, token_counts):
        offsets = [0]
        for count in token_counts:
            offsets.append(offsets[-1] + count)
        return [
            (reservation, packed[start:end])
            for reservation, start, end in zip(reservations, offsets, offsets[1:])
        ]

    def publish_batch_sync(self, staged_transfers):
        metadata = []
        for staged in staged_transfers:
            result = self._publish(staged)
            result["transfer_copy_done_ns"] = time.time_ns()
            metadata.append(result)
        return metadata

    def cancel_batch(self, reservations):
        self.released.extend(reservations)

    def _publish(self, staged_transfer):
        transfer_id, embedding = staged_transfer
        self.published.append((transfer_id, embedding))
        return {"transfer_id": transfer_id}

    def release(self, transfer_id) -> None:
        self.released.append(transfer_id)

    def close(self) -> None:
        self.closed = True


def _data(req_id: str, transfer_id: str = "transfer-0") -> EmbeddingData:
    return EmbeddingData(
        req_id=req_id,
        num_parts=1,
        part_idx=0,
        grid_dim=None,
        modality=Modality.IMAGE,
        transfer_id=transfer_id,
    )


async def _collect(runtime: EncoderRuntime, requests: list[dict]):
    results = []
    done = asyncio.Event()

    def complete(index, result):
        results.append((index, result))
        if len(results) == len(requests):
            done.set()

    await _enqueue(runtime, requests, complete)
    if requests:
        await asyncio.wait_for(done.wait(), 5)
    return results


async def _enqueue(runtime: EncoderRuntime, requests: list[dict], callback):
    prepared = await asyncio.gather(*(runtime.preprocess_request(request) for request in requests))
    for index, item in enumerate(prepared):

        def complete(result, index=index):
            callback(index, result)

        await runtime.enqueue_preprocessed(item, complete)


def test_encode_job_batches_cross_thread_deliveries_into_one_wakeup():
    class RecordingLoop:
        def __init__(self):
            self.calls = 0

        def call_soon_threadsafe(self, callback, *args):
            self.calls += 1
            callback(*args)

    delivered = []
    loop = RecordingLoop()
    job = _EncodeJob(
        items=[
            _ReadyJob(
                PreprocessedRequest({}, None, None, 0),
                lambda result, index=index: delivered.append((index, result)),
                cast(asyncio.AbstractEventLoop, loop),
            )
            for index in range(2)
        ],
        prepared=None,
    )
    first = _data("first")
    second = _data("second")
    job.deliver_many([(0, first), (1, second)])

    assert loop.calls == 1
    assert delivered == [(0, first), (1, second)]


def test_sim_server_transfer_reserves_stages_and_publishes():
    transfer = SimEncoderServerTransfer()
    reservations = transfer.reserve_batch_sync(["request-0:embedding"])
    staged = transfer.stage_packed_batch_sync(reservations, jnp.zeros((1, 2)), (1,))
    metadata = transfer.publish_batch_sync(staged)[0]
    assert metadata["transfer_id"] == "request-0:embedding"
    transfer.close()


@pytest.mark.parametrize("phase", ["stage_packed_batch_sync", "publish_batch_sync"])
def test_runtime_releases_entire_batch_when_transfer_fails(monkeypatch, phase):
    transfer = _FakeTransfer()

    def fail(*args):
        raise RuntimeError("transfer failed")

    monkeypatch.setattr(transfer, phase, fail)

    async def run():
        runtime = EncoderRuntime(
            _TestEncoder(lambda requests: [(jnp.zeros((1, 2)), {}) for _ in requests]),
            transfer,
            batch_coalesce_ms=20,
        )
        try:
            return await _collect(
                runtime,
                [
                    {"req_id": "request-0", "modality": "IMAGE"},
                    {"req_id": "request-1", "modality": "IMAGE"},
                ],
            )
        finally:
            await runtime.stop()

    results = asyncio.run(run())

    assert [index for index, _ in results] == [0, 1]
    assert all(isinstance(result, RuntimeError) for _, result in results)
    assert sorted(transfer.released) == ["request-0:0:embedding", "request-1:0:embedding"]


def test_server_owns_scheduler_and_runtime():
    server = EncoderServer(_TestEncoder(lambda _: None), _FakeTransfer())

    assert server.scheduler._runtime is server.runtime
    assert not hasattr(server.runtime, "scheduler")


def test_scheduler_records_enqueue_and_dequeue_timestamps(caplog):
    async def run():
        runtime = EncoderRuntime(
            _TestEncoder(lambda requests: [(jnp.zeros((1, 2)), {}) for _ in requests]),
            _FakeTransfer(),
        )
        scheduler = DisaggEncoderScheduler(runtime, log_queue_timing=True)
        scheduler.start()
        try:
            return await scheduler.submit(
                {
                    "req_id": "request-0",
                    "modality": "IMAGE",
                    "collect_request_time_stats": True,
                }
            )
        finally:
            await scheduler.stop()
            await runtime.stop()

    caplog.set_level("INFO")
    data = asyncio.run(run())

    assert isinstance(data.enqueue_ns, int)
    assert isinstance(data.dequeue_ns, int)
    assert data.queue_duration_ns >= 0
    assert data.queue_ms == data.queue_duration_ns / 1_000_000
    assert "ENCODER-QUEUE-TIME req_id=request-0" in caplog.text


def test_scheduler_preprocesses_next_batch_while_vit_is_busy():
    async def run() -> None:
        first_encode_started = asyncio.Event()
        second_preprocess_done = asyncio.Event()
        release_first_encode = threading.Event()
        loop = asyncio.get_running_loop()

        async def preprocess(requests):
            if requests[0]["req_id"] == "request-1":
                second_preprocess_done.set()
            return requests

        def encode(requests):
            if requests[0]["req_id"] == "request-0":
                loop.call_soon_threadsafe(first_encode_started.set)
                if not release_first_encode.wait(1):
                    raise TimeoutError("test did not release first encoder batch")
            return [(jnp.zeros((1, 2)), {})]

        runtime = EncoderRuntime(
            _TestEncoder(encode, preprocess),
            _FakeTransfer(),
            max_batch_size=1,
        )
        scheduler = DisaggEncoderScheduler(runtime)
        scheduler.start()
        first = asyncio.create_task(scheduler.submit({"req_id": "request-0", "modality": "IMAGE"}))
        try:
            await asyncio.wait_for(first_encode_started.wait(), 1)
            second = asyncio.create_task(
                scheduler.submit({"req_id": "request-1", "modality": "IMAGE"})
            )
            await asyncio.wait_for(second_preprocess_done.wait(), 1)
            release_first_encode.set()
            await asyncio.gather(first, second)
        finally:
            release_first_encode.set()
            await scheduler.stop()
            await runtime.stop()

    asyncio.run(run())


def test_scheduler_releases_result_cancelled_during_encode():
    async def run():
        encode_started = asyncio.Event()
        finish_encode = threading.Event()
        released = threading.Event()
        loop = asyncio.get_running_loop()

        class ControlledTransfer(_FakeTransfer):
            def release(self, transfer_id) -> None:
                super().release(transfer_id)
                released.set()

        def encode(_requests):
            loop.call_soon_threadsafe(encode_started.set)
            if not finish_encode.wait(1):
                raise TimeoutError("test did not release encoder")
            return [(jnp.zeros((1, 2)), {})]

        transfer = ControlledTransfer()
        runtime = EncoderRuntime(_TestEncoder(encode), transfer)
        scheduler = DisaggEncoderScheduler(runtime)
        scheduler.start()
        request = asyncio.create_task(
            scheduler.submit({"req_id": "request-0", "modality": "IMAGE"})
        )
        try:
            await asyncio.wait_for(encode_started.wait(), 1)
            request.cancel()
            with suppress(asyncio.CancelledError):
                await request
            finish_encode.set()
            assert await asyncio.to_thread(released.wait, 1)
        finally:
            finish_encode.set()
            await scheduler.stop()
            await runtime.stop()
        return transfer

    transfer = asyncio.run(run())
    assert [item[0] for item in transfer.published] == ["request-0:0:embedding"]
    assert transfer.released == ["request-0:0:embedding"]


def test_runtime_builds_transfer_metadata_without_scheduler_state():
    def encode(_requests):
        return [
            (
                jnp.zeros((1, 2)),
                {"_encoder_timing": {"processor_start_ns": 2}},
            )
        ]

    async def run() -> EmbeddingData:
        runtime = EncoderRuntime(_TestEncoder(encode), _FakeTransfer())
        results = await _collect(
            runtime,
            [{"req_id": "request-0", "modality": "IMAGE", "dispatch_start_ns": 1}],
        )
        index, result = results[0]
        assert index == 0
        assert isinstance(result, EmbeddingData)
        return result

    data = asyncio.run(run())
    assert data.enqueue_ns is None
    assert data.dequeue_ns is None
    assert data.queue_duration_ns is None
    assert data.dispatch_start_ns == 1
    assert data.processor_start_ns == 2
    assert data.encode_done_ns <= data.publish_done_ns
    assert data.runtime_encode_return_ns <= data.runtime_postprocess_done_ns
    assert data.runtime_postprocess_duration_ns >= 0
    assert data.runtime_metadata_prepare_duration_ns >= 0
    assert data.runtime_embedding_data_duration_ns >= 0
    assert data.runtime_result_pack_duration_ns >= 0
    assert data.runtime_postprocess_residual_ns >= 0
    assert data.runtime_timing_attach_duration_ns >= 0


def test_runtime_stages_packed_output_before_building_metadata():
    events = []

    class PackedEncoder:
        preprocess_concurrency = 8

        async def preprocess_request(self, _request):
            return None

        @staticmethod
        def batch_key(_prepared):
            return "image", 1

        @staticmethod
        def build_batch(prepared):
            return SimpleNamespace(token_counts=(1,) * len(prepared), transfer_specs=())

        def encode_packed(self, prepared):
            events.append("encode")
            return SimpleNamespace(
                batch=prepared,
                packed=jnp.arange(4, dtype=jnp.float32).reshape(2, 2),
            )

        def metadata_for_packed(self, _output):
            events.append("metadata")
            return [{}, {}]

    class PackedTransfer(_FakeTransfer):
        def stage_packed_batch_sync(self, reservations, packed, token_counts):
            events.append("stage")
            assert token_counts == (1, 1)
            return [(reservation, packed) for reservation in reservations]

    async def run():
        runtime = EncoderRuntime(
            PackedEncoder(),
            PackedTransfer(),
            batch_coalesce_ms=20,
        )
        requests = [
            {"req_id": "request-0", "modality": "IMAGE"},
            {"req_id": "request-1", "modality": "IMAGE"},
        ]
        try:
            return await _collect(runtime, requests)
        finally:
            await runtime.stop()

    results = asyncio.run(run())
    assert events == ["encode", "stage", "metadata"]
    assert [result.shape for _, result in results] == [(1, 2), (1, 2)]
    assert all(result.dtype == "float32" for _, result in results)


def test_runtime_uses_event_loop_for_preprocess_and_threads_for_data_path():
    async def run() -> tuple[str, str, list[str], list[str]]:
        preprocess_thread = ""
        encode_thread = ""

        async def preprocess(requests):
            nonlocal preprocess_thread
            preprocess_thread = threading.current_thread().name
            return requests

        def encode(requests):
            nonlocal encode_thread
            encode_thread = threading.current_thread().name
            return [(jnp.zeros((1, 2)), {}) for _ in requests]

        class RecordingTransfer(_FakeTransfer):
            def __init__(self):
                super().__init__()
                self.copy_threads = []
                self.publish_threads = []

            def stage_packed_batch_sync(self, reservations, packed, token_counts):
                self.copy_threads.append(threading.current_thread().name)
                return super().stage_packed_batch_sync(reservations, packed, token_counts)

            def _publish(self, staged_transfer):
                self.publish_threads.append(threading.current_thread().name)
                return super()._publish(staged_transfer)

        runtime = EncoderRuntime(
            _TestEncoder(encode, preprocess),
            RecordingTransfer(),
        )
        try:
            transfer = runtime._transfer
            await _collect(
                runtime,
                [{"req_id": "request-0", "modality": "IMAGE"}],
            )
            return (
                preprocess_thread,
                encode_thread,
                transfer.copy_threads,
                transfer.publish_threads,
            )
        finally:
            await runtime.stop()

    preprocess_thread, encode_thread, copy_threads, publish_threads = asyncio.run(run())

    assert preprocess_thread == "MainThread"
    assert encode_thread == "sgl-jax-encoder-vit"
    assert copy_threads == ["sgl-jax-encoder-vit"]
    assert publish_threads == ["sgl-jax-encoder-transfer"]


def test_runtime_reserves_before_forward():
    events = []

    class RecordingTransfer(_FakeTransfer):
        def reserve_batch_sync(self, transfer_ids):
            events.append("reserve")
            return super().reserve_batch_sync(transfer_ids)

        def stage_packed_batch_sync(self, reservations, packed, token_counts):
            events.append("copy")
            return super().stage_packed_batch_sync(reservations, packed, token_counts)

        def _publish(self, staged_transfer):
            events.append("publish")
            return super()._publish(staged_transfer)

    def encode(_requests):
        events.append("forward")
        return [(jnp.zeros((1, 2)), {})]

    async def run() -> None:
        runtime = EncoderRuntime(_TestEncoder(encode), RecordingTransfer())
        try:
            await _collect(runtime, [{"req_id": "request-0", "modality": "IMAGE"}])
            assert not hasattr(runtime, "_progress_thread")
        finally:
            await runtime.stop()

    asyncio.run(run())
    assert events == ["reserve", "forward", "copy", "publish"]


def test_runtime_delivers_batch_only_after_all_publishes():
    class ControlledTransfer(_FakeTransfer):
        def __init__(self):
            super().__init__()
            self.events = {
                "request-0:0:embedding": threading.Event(),
                "request-1:0:embedding": threading.Event(),
            }
            self.started = {
                "request-0:0:embedding": threading.Event(),
                "request-1:0:embedding": threading.Event(),
            }

        def _publish(self, staged_transfer):
            transfer_id, embedding = staged_transfer
            self.published.append((transfer_id, embedding))
            self.started[transfer_id].set()
            if not self.events[transfer_id].wait(1):
                raise TimeoutError("test did not release encoder publish")
            return {"transfer_id": transfer_id}

    async def run() -> None:
        transfer = ControlledTransfer()

        def encode(requests):
            return [(jnp.zeros((1, 2)), {}) for _ in requests]

        runtime = EncoderRuntime(_TestEncoder(encode), transfer)
        completed = [asyncio.Event(), asyncio.Event()]
        results = []

        def complete(index, result):
            results.append((index, result))
            completed[index].set()

        task = asyncio.create_task(
            _enqueue(
                runtime,
                [
                    {"req_id": "request-0", "modality": "IMAGE"},
                    {"req_id": "request-1", "modality": "IMAGE"},
                ],
                complete,
            )
        )
        assert await asyncio.to_thread(
            transfer.started["request-0:0:embedding"].wait,
            1,
        )
        assert [item[0] for item in transfer.published] == ["request-0:0:embedding"]
        await task
        transfer.events["request-0:0:embedding"].set()
        assert not completed[0].is_set()

        assert await asyncio.to_thread(
            transfer.started["request-1:0:embedding"].wait,
            1,
        )
        assert [item[0] for item in transfer.published] == [
            "request-0:0:embedding",
            "request-1:0:embedding",
        ]
        transfer.events["request-1:0:embedding"].set()
        await asyncio.wait_for(asyncio.gather(*(event.wait() for event in completed)), 1)
        assert [index for index, _ in results] == [0, 1]
        await runtime.stop()

    asyncio.run(run())


def test_runtime_publishes_requests_independently_across_receivers():
    def encode(_requests):
        return [
            (jnp.zeros((2, 3)), {}),
            (jnp.ones((4, 3)), {}),
        ]

    async def run():
        transfer = _FakeTransfer()
        runtime = EncoderRuntime(_TestEncoder(encode), transfer)
        results = await _collect(
            runtime,
            [
                {"req_id": "request-0", "modality": "IMAGE"},
                {"req_id": "request-1", "modality": "IMAGE"},
            ],
        )
        return transfer, results

    transfer, results = asyncio.run(run())
    assert [item[0] for item in transfer.published] == [
        "request-0:0:embedding",
        "request-1:0:embedding",
    ]
    assert [result.transfer_id for _, result in results] == [
        "request-0:0:embedding",
        "request-1:0:embedding",
    ]


def test_runtime_publishes_on_transfer_thread():
    class BatchTransfer(_FakeTransfer):
        def __init__(self):
            super().__init__()
            self.batch_threads = []

        def publish_batch_sync(self, staged_transfers):
            self.batch_threads.append(threading.current_thread().name)
            return super().publish_batch_sync(staged_transfers)

    def encode(requests):
        return [(jnp.zeros((1, 2)), {}) for _ in requests]

    async def run():
        transfer = BatchTransfer()
        runtime = EncoderRuntime(_TestEncoder(encode), transfer)
        try:
            results = await _collect(
                runtime,
                [
                    {"req_id": "request-0", "modality": "IMAGE"},
                    {"req_id": "request-1", "modality": "IMAGE"},
                ],
            )
            return transfer, results
        finally:
            await runtime.stop()

    transfer, results = asyncio.run(run())
    assert transfer.batch_threads == ["sgl-jax-encoder-transfer"]
    assert [result.transfer_id for _, result in results] == [
        "request-0:0:embedding",
        "request-1:0:embedding",
    ]


def test_server_reuses_metadata_sender_socket():
    class FakeSocket:
        def __init__(self):
            self.sent = []
            self.closed = False

        def setsockopt(self, option, value):
            pass

        def connect(self, address):
            self.address = address

        async def send_pyobj(self, data):
            self.sent.append(data)

        def close(self):
            self.closed = True

    class FakeContext:
        def __init__(self):
            self.sockets = []

        def socket(self, socket_type):
            socket = FakeSocket()
            self.sockets.append(socket)
            return socket

    async def run() -> None:
        context = FakeContext()
        transfer = _FakeTransfer()
        server = EncoderServer(_TestEncoder(lambda _: None), transfer)
        server._zmq = context
        first = _data("request-0")
        second = _data("request-1")

        for data in (first, second):
            await server.register_scheduler_receiver(
                {"req_id": data.req_id, "receive_url": "127.0.0.1:1234"}
            )
        await asyncio.gather(
            *(server.send_to_scheduler(data.req_id, data) for data in (first, second))
        )

        assert len(context.sockets) == 1
        assert context.sockets[0].sent == [first, second]
        await server.stop()
        assert context.sockets[0].closed
        assert transfer.closed

    asyncio.run(run())


def test_server_notification_backpressure_is_local_to_receiver():
    async def run() -> None:
        blocked = asyncio.Event()
        release = asyncio.Event()

        class FakeSocket:
            def __init__(self):
                self.sent = []

            async def send_pyobj(self, data):
                if data.req_id == "slow":
                    blocked.set()
                    await release.wait()
                self.sent.append(data)

            def close(self):
                pass

        server = EncoderServer(_TestEncoder(lambda _: None), _FakeTransfer())
        slow, healthy = _data("slow"), _data("healthy")
        sockets = {data.req_id: FakeSocket() for data in (slow, healthy)}
        server._receiver_sockets.update(sockets)
        for data in (slow, healthy):
            await server.register_scheduler_receiver(
                {"req_id": data.req_id, "receive_url": data.req_id}
            )

        slow_send = asyncio.create_task(server.send_to_scheduler(slow.req_id, slow))
        try:
            await asyncio.wait_for(blocked.wait(), 1)
            await asyncio.wait_for(server.send_to_scheduler(healthy.req_id, healthy), 1)
            assert sockets["healthy"].sent == [healthy]
            assert not slow_send.done()
            assert set(server._receiver_addresses) == {"slow"}
            assert set(server._receiver_events) == {"slow"}
        finally:
            release.set()
            await slow_send
            await server.stop()

    asyncio.run(run())


def test_server_routes_scheduler_result_to_registered_receiver():
    async def run():
        sent = []

        def encode(_requests):
            return [(jnp.zeros((1, 2)), {})]

        server = EncoderServer(_TestEncoder(encode), _FakeTransfer())

        async def send(req_id, data):
            sent.append((req_id, data))

        server.send_to_scheduler = send
        server.start()
        try:
            response = await server.encode({"req_id": "request-0", "modality": "IMAGE"})
        finally:
            await server.stop()
        return response, sent

    response, sent = asyncio.run(run())
    assert response == {"req_id": "request-0"}
    assert sent[0][0] == "request-0"
    assert sent[0][1].req_id == "request-0"


def test_runtime_overlaps_forward_with_background_publish():
    async def run() -> None:
        encode_started = []
        publish_started = []
        release_first_encode = threading.Event()
        release_second_encode = threading.Event()
        release_first_publish = threading.Event()
        first_publish_started = threading.Event()
        second_encode_started = asyncio.Event()
        second_publish_started = threading.Event()
        loop = asyncio.get_running_loop()

        class ControlledTransfer(_FakeTransfer):
            def _publish(self, staged_transfer):
                transfer_id, _ = staged_transfer
                publish_started.append(transfer_id)
                if transfer_id.startswith("request-0:"):
                    first_publish_started.set()
                    if not release_first_publish.wait(1):
                        raise TimeoutError("test did not release first publish")
                else:
                    second_publish_started.set()
                return {"transfer_id": transfer_id}

        def encode(requests):
            req_id = requests[0]["req_id"]
            encode_started.append(req_id)
            if req_id == "request-0":
                if not release_first_encode.wait(1):
                    raise TimeoutError("test did not release first encoder batch")
            else:
                loop.call_soon_threadsafe(second_encode_started.set)
                if not release_second_encode.wait(1):
                    raise TimeoutError("test did not release second encoder batch")
            return [(jnp.zeros((1, 2)), {})]

        runtime = EncoderRuntime(_TestEncoder(encode), ControlledTransfer())
        first_task = asyncio.create_task(
            _collect(runtime, [{"req_id": "request-0", "modality": "IMAGE"}])
        )
        for _ in range(100):
            if encode_started:
                break
            await asyncio.sleep(0.001)
        assert encode_started == ["request-0"]
        second_task = asyncio.create_task(
            _collect(runtime, [{"req_id": "request-1", "modality": "IMAGE"}])
        )
        release_first_encode.set()
        await asyncio.wait_for(second_encode_started.wait(), 1)
        assert await asyncio.to_thread(first_publish_started.wait, 1)
        assert publish_started == ["request-0:0:embedding"]

        release_second_encode.set()
        release_first_publish.set()
        assert await asyncio.to_thread(second_publish_started.wait, 1)
        await asyncio.gather(first_task, second_task)
        assert publish_started == [
            "request-0:0:embedding",
            "request-1:0:embedding",
        ]

    asyncio.run(run())


def test_runtime_does_not_block_vit_on_large_transfer_batch():
    async def run() -> None:
        first_transfer_started = threading.Event()
        release_first_transfer = threading.Event()
        second_encode_started = asyncio.Event()
        loop = asyncio.get_running_loop()

        class ControlledTransfer(_FakeTransfer):
            def _publish(self, staged_transfer):
                transfer_id, _ = staged_transfer
                if transfer_id.startswith("request-0:"):
                    first_transfer_started.set()
                    if not release_first_transfer.wait(1):
                        raise TimeoutError("test did not release first transfer")
                return super()._publish(staged_transfer)

        def encode(requests):
            if requests[0]["req_id"] == "request-next":
                loop.call_soon_threadsafe(second_encode_started.set)
            return [(jnp.zeros((1, 2)), {}) for _ in requests]

        runtime = EncoderRuntime(_TestEncoder(encode), ControlledTransfer())
        first_batch = [
            {
                "req_id": f"request-{index}",
                "part_idx": index,
                "modality": "IMAGE",
            }
            for index in range(8)
        ]
        first_task = asyncio.create_task(_collect(runtime, first_batch))
        second_task = asyncio.create_task(
            _collect(runtime, [{"req_id": "request-next", "modality": "IMAGE"}])
        )
        try:
            assert await asyncio.to_thread(first_transfer_started.wait, 1)
            await asyncio.wait_for(second_encode_started.wait(), 1)
        finally:
            release_first_transfer.set()
            await asyncio.gather(first_task, second_task)
            await runtime.stop()

    asyncio.run(run())


def test_runtime_forward_runs_with_four_transfers_inflight():
    async def run() -> list[tuple[str, int]]:
        class WindowedTransfer(_FakeTransfer):
            def __init__(self):
                super().__init__()
                self.active = set()

            def _publish(self, staged_transfer):
                transfer_id, _ = staged_transfer
                self.active.add(transfer_id)
                return super()._publish(staged_transfer)

            def release(self, transfer_id) -> None:
                self.active.discard(transfer_id)
                super().release(transfer_id)

        transfer = WindowedTransfer()
        observed = []

        def encode(requests):
            observed.append((requests[0]["req_id"], len(transfer.active)))
            return [(jnp.zeros((1, 2)), {})]

        runtime = EncoderRuntime(_TestEncoder(encode), transfer)
        try:
            for index in range(5):
                await _collect(
                    runtime,
                    [{"req_id": f"request-{index}", "modality": "IMAGE"}],
                )
        finally:
            await runtime.stop()
        return observed

    observed = asyncio.run(run())
    assert observed[-1] == ("request-4", 4)


def test_runtime_progress_thread_releases_transfer_backpressure():
    async def run() -> None:
        transfer = SimEncoderServerTransfer(
            pool_size=1,
            rtt_ms=5,
            poll_interval_s=0.0001,
        )

        def encode(requests):
            return [(jnp.zeros((1, 2)), {}) for _ in requests]

        runtime = EncoderRuntime(_TestEncoder(encode), transfer, max_batch_size=1)
        try:
            first = asyncio.create_task(
                _collect(runtime, [{"req_id": "request-0", "modality": "IMAGE"}])
            )
            second = asyncio.create_task(
                _collect(runtime, [{"req_id": "request-1", "modality": "IMAGE"}])
            )
            first_result, second_result = await asyncio.wait_for(
                asyncio.gather(first, second),
                1,
            )
            assert first_result[0][1].req_id == "request-0"
            assert second_result[0][1].req_id == "request-1"
        finally:
            await runtime.stop()

    asyncio.run(run())


def test_runtime_allows_concurrent_preprocess_before_serial_encode():
    async def run() -> None:
        first_preprocess_started = asyncio.Event()
        first_encode_started = asyncio.Event()
        second_preprocess_started = asyncio.Event()
        second_preprocess_done = asyncio.Event()
        release_first_preprocess = asyncio.Event()
        release_second_preprocess = asyncio.Event()
        release_first_encode = threading.Event()
        loop = asyncio.get_running_loop()

        async def preprocess(requests):
            req_id = requests[0]["req_id"]
            if req_id == "request-0":
                first_preprocess_started.set()
                await release_first_preprocess.wait()
            else:
                second_preprocess_started.set()
                await release_second_preprocess.wait()
                second_preprocess_done.set()
            return req_id

        def encode(req_id):
            if req_id == "request-0":
                loop.call_soon_threadsafe(first_encode_started.set)
                if not release_first_encode.wait(1):
                    raise TimeoutError("test did not release first encoder batch")
            return [(jnp.zeros((1, 2)), {})]

        runtime = EncoderRuntime(
            _TestEncoder(encode, preprocess),
            _FakeTransfer(),
        )
        first_task = asyncio.create_task(
            _collect(runtime, [{"req_id": "request-0", "modality": "IMAGE"}])
        )
        await asyncio.wait_for(first_preprocess_started.wait(), 1)
        second_task = asyncio.create_task(
            _collect(runtime, [{"req_id": "request-1", "modality": "IMAGE"}])
        )
        await asyncio.wait_for(second_preprocess_started.wait(), 1)

        release_first_preprocess.set()
        await asyncio.wait_for(first_encode_started.wait(), 1)
        release_second_preprocess.set()
        await asyncio.wait_for(second_preprocess_done.wait(), 1)
        assert not first_task.done()
        assert not second_task.done()

        release_first_encode.set()
        await asyncio.gather(first_task, second_task)

    asyncio.run(run())


def test_preprocessed_requests_batch_at_vit_boundary():
    async def run() -> list[list[str]]:
        first_encode_started = asyncio.Event()
        rest_preprocessed = asyncio.Event()
        release_first_encode = threading.Event()
        loop = asyncio.get_running_loop()
        batches = []
        prepared_count = 0

        class BufferedEncoder(_TestEncoder):
            preprocess_concurrency = 3

            def __init__(self):
                super().__init__(self._encode_requests)

            async def preprocess_request(self, request):
                nonlocal prepared_count
                if request["req_id"] != "request-0":
                    prepared_count += 1
                    if prepared_count == 3:
                        rest_preprocessed.set()
                return request

            @staticmethod
            def batch_key(request):
                return request["modality"], request["token_count"]

            def _encode_requests(self, requests):
                req_ids = [request["req_id"] for request in requests]
                batches.append(req_ids)
                if req_ids == ["request-0"]:
                    loop.call_soon_threadsafe(first_encode_started.set)
                    if not release_first_encode.wait(1):
                        raise TimeoutError("test did not release first encoder batch")
                return [(jnp.zeros((1, 2)), {}) for _ in requests]

        runtime = EncoderRuntime(
            BufferedEncoder(),
            _FakeTransfer(),
            pipeline_depth=1,
            max_batch_size=3,
        )
        scheduler = DisaggEncoderScheduler(runtime)
        scheduler.start()
        first = asyncio.create_task(
            scheduler.submit(
                {
                    "req_id": "request-0",
                    "modality": "IMAGE",
                    "token_count": 1,
                }
            )
        )
        try:
            await asyncio.wait_for(first_encode_started.wait(), 1)
            rest = [
                asyncio.create_task(
                    scheduler.submit(
                        {
                            "req_id": f"request-{index}",
                            "modality": "IMAGE",
                            "token_count": 1,
                        }
                    )
                )
                for index in range(1, 4)
            ]
            await asyncio.wait_for(rest_preprocessed.wait(), 1)
            assert batches == [["request-0"]]
            release_first_encode.set()
            await asyncio.gather(first, *rest)
            return batches
        finally:
            release_first_encode.set()
            await scheduler.stop()
            await runtime.stop()

    assert asyncio.run(run()) == [
        ["request-0"],
        ["request-1", "request-2", "request-3"],
    ]


def test_runtime_overlaps_next_vit_with_previous_metadata():
    metadata_started = threading.Event()
    release_metadata = threading.Event()
    second_encoded = threading.Event()

    class Encoder(_TestEncoder):
        def metadata_for_packed(self, output):
            assert threading.current_thread().name == "sgl-jax-encoder-transfer"
            if not metadata_started.is_set():
                metadata_started.set()
                assert release_metadata.wait(2)
            return super().metadata_for_packed(output)

    def encode(requests):
        if requests[0]["req_id"] == "second":
            second_encoded.set()
        return [(jnp.zeros((1, 2)), {}) for _ in requests]

    async def run():
        runtime = EncoderRuntime(Encoder(encode), _FakeTransfer(), max_batch_size=1)
        first = asyncio.create_task(_collect(runtime, [{"req_id": "first", "modality": "IMAGE"}]))
        second = None
        try:
            assert await asyncio.to_thread(metadata_started.wait, 1)
            second = asyncio.create_task(
                _collect(runtime, [{"req_id": "second", "modality": "IMAGE"}])
            )
            assert await asyncio.to_thread(second_encoded.wait, 1)
            release_metadata.set()
            results = await asyncio.gather(first, second)
            assert all(isinstance(rows[0][1], EmbeddingData) for rows in results)
        finally:
            release_metadata.set()
            await asyncio.gather(
                *(task for task in (first, second) if task), return_exceptions=True
            )
            await runtime.stop()

    asyncio.run(run())


def test_runtime_metadata_failure_releases_batch_and_keeps_transfer_worker_alive():
    class Encoder(_TestEncoder):
        fail = True

        def metadata_for_packed(self, output):
            if self.fail:
                self.fail = False
                raise ValueError("invalid metadata")
            return super().metadata_for_packed(output)

    async def run():
        transfer = _FakeTransfer()
        runtime = EncoderRuntime(
            Encoder(lambda requests: [(jnp.zeros((1, 2)), {}) for _ in requests]),
            transfer,
            max_batch_size=1,
        )
        try:
            failed = await _collect(runtime, [{"req_id": "failed", "modality": "IMAGE"}])
            assert isinstance(failed[0][1], ValueError)
            assert transfer.released == ["failed:0:embedding"]
            ready = await _collect(runtime, [{"req_id": "ready", "modality": "IMAGE"}])
            assert isinstance(ready[0][1], EmbeddingData)
        finally:
            await runtime.stop()

    asyncio.run(run())
