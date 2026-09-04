from __future__ import annotations

import asyncio
import threading
from contextlib import suppress

import jax.numpy as jnp

from sgl_jax.srt.disaggregation.encoder.embedding_data import EmbeddingData
from sgl_jax.srt.disaggregation.encoder.runtime import EncoderRuntime
from sgl_jax.srt.disaggregation.encoder.scheduler import DisaggEncoderScheduler
from sgl_jax.srt.disaggregation.encoder.server import EncoderServer
from sgl_jax.srt.disaggregation.encoder.sim_transfer import SimEncoderServerTransfer
from sgl_jax.srt.multimodal.common.modality_enum import Modality


class _TestEncoder:
    def __init__(self, encode, preprocess=None):
        self._encode = encode
        self._preprocess = preprocess

    async def preprocess(self, requests):
        if self._preprocess is None:
            return requests
        return await self._preprocess(requests)

    def encode(self, prepared):
        return self._encode(prepared)


class _FakeTransfer:
    def __init__(self):
        self.published = []
        self.released = []
        self.closed = False

    def reserve_batch_sync(self, transfer_ids):
        return list(transfer_ids)

    def stage_batch_sync(self, reservations, embeddings):
        return list(zip(reservations, embeddings))

    def cancel_batch(self, reservations):
        self.released.extend(reservations)

    def publish_sync(self, staged_transfer):
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

    await runtime.execute_batch(requests, complete)
    if requests:
        await done.wait()
    return results


def test_sim_server_transfer_reserves_stages_and_publishes():
    transfer = SimEncoderServerTransfer()
    reservations = transfer.reserve_batch_sync(["request-0:embedding"])
    staged = transfer.stage_batch_sync(reservations, [jnp.zeros((1, 2))])
    metadata = transfer.publish_sync(staged[0])
    assert metadata["transfer_id"] == "request-0:embedding"
    transfer.close()


def test_server_owns_scheduler_and_runtime():
    server = EncoderServer(_TestEncoder(lambda _: None), _FakeTransfer())

    assert server.scheduler._runtime is server.runtime
    assert not hasattr(server.runtime, "scheduler")


def test_scheduler_records_enqueue_and_dequeue_timestamps(caplog):
    class RecordingRuntime:
        def __init__(self):
            self.requests = []

        async def execute_batch(self, requests, on_result):
            self.requests.extend(requests)
            for index, request in enumerate(requests):
                on_result(index, _data(request["req_id"]))

        def release(self, transfer_id):
            pass

    async def run():
        runtime = RecordingRuntime()
        scheduler = DisaggEncoderScheduler(runtime, log_queue_timing=True)
        scheduler.start()
        try:
            data = await scheduler.submit({"req_id": "request-0", "modality": "IMAGE"})
            return runtime, data
        finally:
            await scheduler.stop()

    caplog.set_level("INFO")
    runtime, data = asyncio.run(run())

    assert [request["req_id"] for request in runtime.requests] == ["request-0"]
    assert isinstance(data.enqueue_ns, int)
    assert isinstance(data.dequeue_ns, int)
    assert data.queue_duration_ns >= 0
    assert data.queue_ms == data.queue_duration_ns / 1_000_000
    assert "ENCODER-QUEUE-TIME req_id=request-0" in caplog.text


def test_scheduler_coalesces_requests_with_one_bounded_deadline():
    batches = []

    async def run() -> None:
        class RecordingRuntime:
            async def execute_batch(self, requests, on_result):
                batches.append([request["req_id"] for request in requests])
                for index, request in enumerate(requests):
                    on_result(index, _data(request["req_id"]))

            def release(self, transfer_id):
                pass

        scheduler = DisaggEncoderScheduler(
            RecordingRuntime(),
            max_batch_size=4,
            batch_coalesce_ms=50,
        )
        scheduler.start()
        first = asyncio.create_task(scheduler.submit({"req_id": "request-0", "modality": "IMAGE"}))
        await asyncio.sleep(0.01)
        second = asyncio.create_task(scheduler.submit({"req_id": "request-1", "modality": "IMAGE"}))
        try:
            await asyncio.gather(first, second)
        finally:
            await scheduler.stop()

    asyncio.run(run())
    assert batches == [["request-0", "request-1"]]


def test_scheduler_dispatches_full_batch_without_waiting_for_deadline():
    batches = []

    async def run() -> None:
        dispatched = asyncio.Event()

        class RecordingRuntime:
            async def execute_batch(self, requests, on_result):
                batches.append([request["req_id"] for request in requests])
                dispatched.set()
                for index, request in enumerate(requests):
                    on_result(index, _data(request["req_id"]))

            def release(self, transfer_id):
                pass

        scheduler = DisaggEncoderScheduler(
            RecordingRuntime(),
            max_batch_size=2,
            batch_coalesce_ms=500,
        )
        first = asyncio.create_task(scheduler.submit({"req_id": "request-0", "modality": "IMAGE"}))
        second = asyncio.create_task(scheduler.submit({"req_id": "request-1", "modality": "IMAGE"}))
        scheduler.start()
        try:
            await asyncio.wait_for(dispatched.wait(), 0.1)
            await asyncio.gather(first, second)
        finally:
            await scheduler.stop()

    asyncio.run(run())
    assert batches == [["request-0", "request-1"]]


def test_scheduler_pipelines_bounded_inflight_batches():
    started = []

    async def run() -> None:
        both_started = asyncio.Event()
        release = asyncio.Event()

        class ControlledRuntime:
            async def execute_batch(self, requests, on_result):
                req_id = requests[0]["req_id"]
                started.append(req_id)
                if len(started) == 2:
                    both_started.set()
                await release.wait()
                on_result(0, _data(req_id))

            def release(self, transfer_id):
                pass

        scheduler = DisaggEncoderScheduler(
            ControlledRuntime(),
            max_batch_size=1,
            max_inflight_batches=2,
        )
        scheduler.start()
        first = asyncio.create_task(scheduler.submit({"req_id": "request-0", "modality": "IMAGE"}))
        second = asyncio.create_task(scheduler.submit({"req_id": "request-1", "modality": "IMAGE"}))
        try:
            await asyncio.wait_for(both_started.wait(), 1)
            release.set()
            await asyncio.gather(first, second)
        finally:
            release.set()
            await scheduler.stop()

    asyncio.run(run())
    assert started == ["request-0", "request-1"]


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

        runtime = EncoderRuntime(_TestEncoder(encode, preprocess), _FakeTransfer())
        scheduler = DisaggEncoderScheduler(runtime, max_batch_size=1)
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

            def stage_batch_sync(self, reservations, embeddings):
                self.copy_threads.append(threading.current_thread().name)
                return super().stage_batch_sync(reservations, embeddings)

            def publish_sync(self, staged_transfer):
                self.publish_threads.append(threading.current_thread().name)
                return super().publish_sync(staged_transfer)

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

        def stage_batch_sync(self, reservations, embeddings):
            events.append("copy")
            return super().stage_batch_sync(reservations, embeddings)

        def publish_sync(self, staged_transfer):
            events.append("publish")
            return super().publish_sync(staged_transfer)

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


def test_runtime_queues_results_and_completes_each_after_publish():
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

        def publish_sync(self, staged_transfer):
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
            runtime.execute_batch(
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
        await asyncio.wait_for(completed[0].wait(), 1)
        assert results[0][0] == 0

        assert await asyncio.to_thread(
            transfer.started["request-1:0:embedding"].wait,
            1,
        )
        assert [item[0] for item in transfer.published] == [
            "request-0:0:embedding",
            "request-1:0:embedding",
        ]
        transfer.events["request-1:0:embedding"].set()
        await asyncio.wait_for(completed[1].wait(), 1)
        assert results[1][0] == 1

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
            await server.send_to_scheduler(data.req_id, data)

        assert len(context.sockets) == 1
        assert context.sockets[0].sent == [first, second]
        await server.stop()
        assert context.sockets[0].closed
        assert transfer.closed

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
            def publish_sync(self, staged_transfer):
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
        await asyncio.sleep(0)
        second_task = asyncio.create_task(
            _collect(runtime, [{"req_id": "request-1", "modality": "IMAGE"}])
        )
        await asyncio.sleep(0)

        assert encode_started == ["request-0"]
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
            def publish_sync(self, staged_transfer):
                transfer_id, _ = staged_transfer
                if transfer_id.startswith("request-0:"):
                    first_transfer_started.set()
                    if not release_first_transfer.wait(1):
                        raise TimeoutError("test did not release first transfer")
                return super().publish_sync(staged_transfer)

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

            def publish_sync(self, staged_transfer):
                transfer_id, _ = staged_transfer
                self.active.add(transfer_id)
                return super().publish_sync(staged_transfer)

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

        runtime = EncoderRuntime(_TestEncoder(encode), transfer)
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
