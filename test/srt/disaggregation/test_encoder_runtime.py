from __future__ import annotations

import asyncio

import jax.numpy as jnp

from sgl_jax.srt.disaggregation.encoder.embedding_data import EmbeddingData
from sgl_jax.srt.disaggregation.encoder.runtime import (
    EncoderRuntime,
    EncoderScheduler,
    PendingRequest,
    PublishedEmbedding,
)
from sgl_jax.srt.disaggregation.encoder.sim_transfer import SimEncoderServerTransfer
from sgl_jax.srt.multimodal.common.modality_enum import Modality


def test_sim_server_transfer_publish_is_awaitable():
    async def run() -> None:
        metadata = await SimEncoderServerTransfer().publish(
            "request-0:embedding", jnp.zeros((1, 2))
        )
        assert metadata == {"transfer_id": "request-0:embedding"}

    asyncio.run(run())


def test_runtime_skips_transfer_for_request_cancelled_during_encode():
    published = []

    class FakeTransfer:
        async def publish(self, transfer_id, embedding):
            published.append(transfer_id)
            return {"transfer_id": transfer_id}

        async def release_completed(self) -> None:
            pass

        def release(self, transfer_id) -> None:
            pass

        def close(self) -> None:
            pass

    async def run() -> None:
        pending = PendingRequest({"req_id": "request-0", "modality": "IMAGE"})

        async def encode(_requests):
            pending.future.cancel()
            return [(jnp.zeros((1, 2)), {})]

        runtime = EncoderRuntime(encode, FakeTransfer())
        await runtime._dispatch_batch([pending])

    asyncio.run(run())
    assert published == []


def test_scheduler_records_enqueue_and_dequeue_timestamps(caplog):
    captured = []

    async def dispatch(batch):
        captured.extend(batch)
        for pending in batch:
            data = EmbeddingData(
                req_id=pending.request["req_id"],
                num_parts=1,
                part_idx=0,
                grid_dim=None,
                modality=Modality.IMAGE,
            )
            pending.future.set_result(
                PublishedEmbedding(pending.request["req_id"], "transfer-0", data)
            )

    async def run() -> None:
        scheduler = EncoderScheduler(dispatch, log_queue_timing=True)
        scheduler.start()
        try:
            await scheduler.submit({"req_id": "request-0", "modality": "IMAGE"})
        finally:
            await scheduler.stop()

    caplog.set_level("INFO")
    asyncio.run(run())

    assert len(captured) == 1
    pending = captured[0]
    assert isinstance(pending.enqueue_ns, int)
    assert isinstance(pending.dequeue_ns, int)
    assert pending.queue_duration_ns is not None
    assert pending.queue_duration_ns >= 0
    assert "ENCODER-QUEUE-TIME req_id=request-0" in caplog.text


def test_scheduler_pipelines_bounded_inflight_batches():
    started = []

    async def run() -> None:
        both_started = asyncio.Event()
        release = asyncio.Event()

        async def dispatch(batch):
            started.append(batch[0].request["req_id"])
            if len(started) == 2:
                both_started.set()
            await release.wait()
            for pending in batch:
                data = EmbeddingData(
                    req_id=pending.request["req_id"],
                    num_parts=1,
                    part_idx=0,
                    grid_dim=None,
                    modality=Modality.IMAGE,
                )
                pending.future.set_result(
                    PublishedEmbedding(pending.request["req_id"], "transfer-0", data)
                )

        scheduler = EncoderScheduler(
            dispatch,
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


def test_runtime_publishes_queue_timing_metadata():
    class FakeTransfer:
        async def publish(self, transfer_id, embedding):
            return {"transfer_id": transfer_id}

        async def release_completed(self) -> None:
            pass

        def release(self, transfer_id) -> None:
            pass

        def close(self) -> None:
            pass

    async def run() -> EmbeddingData:
        pending = PendingRequest(
            {
                "req_id": "request-0",
                "modality": "IMAGE",
                "dispatch_start_ns": 1,
            }
        )
        pending.mark_dequeued()

        async def encode(_requests):
            return [
                (
                    jnp.zeros((1, 2)),
                    {"_encoder_timing": {"processor_start_ns": 2}},
                )
            ]

        runtime = EncoderRuntime(encode, FakeTransfer())
        await runtime._dispatch_batch([pending])
        return pending.future.result().data

    data = asyncio.run(run())
    assert isinstance(data.enqueue_ns, int)
    assert isinstance(data.dequeue_ns, int)
    assert data.dequeue_ns <= data.encode_done_ns <= data.publish_done_ns
    assert data.queue_duration_ns >= 0
    assert data.queue_ms == data.queue_duration_ns / 1_000_000
    assert data.dispatch_start_ns == 1
    assert data.processor_start_ns == 2


def test_runtime_releases_each_request_when_its_publish_completes():
    class ControlledTransfer:
        def __init__(self):
            self.events = {
                "request-0:0:embedding": asyncio.Event(),
                "request-1:0:embedding": asyncio.Event(),
            }

        async def publish(self, transfer_id, embedding):
            await self.events[transfer_id].wait()
            return {"transfer_id": transfer_id}

        async def release_completed(self) -> None:
            pass

        def release(self, transfer_id) -> None:
            pass

        def close(self) -> None:
            pass

    async def run() -> None:
        transfer = ControlledTransfer()

        async def encode(requests):
            return [(jnp.zeros((1, 2)), {}) for _ in requests]

        runtime = EncoderRuntime(encode, transfer)
        pending = [
            PendingRequest({"req_id": f"request-{idx}", "modality": "IMAGE"}) for idx in range(2)
        ]
        for item in pending:
            item.mark_dequeued()

        dispatch = asyncio.create_task(runtime._dispatch_batch(pending))
        await asyncio.sleep(0)
        transfer.events["request-0:0:embedding"].set()
        await asyncio.wait_for(asyncio.shield(pending[0].future), 1)

        assert not pending[1].future.done()
        assert not dispatch.done()

        transfer.events["request-1:0:embedding"].set()
        await dispatch
        assert pending[1].future.done()

    asyncio.run(run())


def test_runtime_groups_batch_transfer_by_receiver():
    class BatchTransfer:
        def __init__(self):
            self.calls = []

        async def publish_batch(self, items):
            self.calls.append(items)
            return [
                {
                    "transfer_id": "group-0",
                    "transfer_offset": offset,
                }
                for offset in (0, items[0][1].shape[0])
            ]

        async def publish(self, transfer_id, embedding):
            raise AssertionError("multi-request batches must use publish_batch")

        async def release_completed(self) -> None:
            pass

        def release(self, transfer_id) -> None:
            pass

        def close(self) -> None:
            pass

    async def run() -> tuple[BatchTransfer, list[PendingRequest]]:
        transfer = BatchTransfer()

        async def encode(_requests):
            return [
                (jnp.zeros((2, 3)), {}),
                (jnp.ones((4, 3)), {}),
            ]

        runtime = EncoderRuntime(encode, transfer)
        pending = [
            PendingRequest({"req_id": f"request-{idx}", "modality": "IMAGE"}) for idx in range(2)
        ]
        for item in pending:
            item.mark_dequeued()
            await runtime.register_scheduler_receiver(
                {"req_id": item.request["req_id"], "receive_url": "127.0.0.1:1234"}
            )
        await runtime._dispatch_batch(pending)
        return transfer, pending

    transfer, pending = asyncio.run(run())
    assert len(transfer.calls) == 1
    assert [item[0] for item in transfer.calls[0]] == [
        "request-0:0:embedding",
        "request-1:0:embedding",
    ]
    assert [item.future.result().data.transfer_offset for item in pending] == [0, 2]
    assert {item.future.result().transfer_id for item in pending} == {"group-0"}


def test_runtime_reuses_metadata_sender_socket():
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

    class FakeTransfer:
        async def release_completed(self) -> None:
            pass

        def close(self) -> None:
            pass

    async def run() -> None:
        context = FakeContext()
        runtime = EncoderRuntime(lambda _: None, FakeTransfer())
        runtime._zmq = context
        data = EmbeddingData(
            req_id="request-0",
            num_parts=1,
            part_idx=0,
            grid_dim=None,
            modality=Modality.IMAGE,
        )

        await runtime._notify("127.0.0.1:1234", data)
        await runtime._notify("127.0.0.1:1234", data)

        assert len(context.sockets) == 1
        assert context.sockets[0].sent == [data, data]
        await runtime.stop()
        assert context.sockets[0].closed

    asyncio.run(run())


def test_runtime_pipelines_serial_encode_and_publish_stages():
    async def run() -> None:
        encode_started = []
        publish_started = []
        release_first_encode = asyncio.Event()
        release_first_publish = asyncio.Event()
        second_encode_done = asyncio.Event()
        second_publish_started = asyncio.Event()

        class ControlledTransfer:
            async def publish(self, transfer_id, embedding):
                publish_started.append(transfer_id)
                if transfer_id.startswith("request-0:"):
                    await release_first_publish.wait()
                else:
                    second_publish_started.set()
                return {"transfer_id": transfer_id}

            async def release_completed(self) -> None:
                pass

            def release(self, transfer_id) -> None:
                pass

            def close(self) -> None:
                pass

        async def encode(requests):
            req_id = requests[0]["req_id"]
            encode_started.append(req_id)
            if req_id == "request-0":
                await release_first_encode.wait()
            else:
                second_encode_done.set()
            return [(jnp.zeros((1, 2)), {})]

        runtime = EncoderRuntime(encode, ControlledTransfer())
        first = PendingRequest({"req_id": "request-0", "modality": "IMAGE"})
        second = PendingRequest({"req_id": "request-1", "modality": "IMAGE"})
        first.mark_dequeued()
        second.mark_dequeued()

        first_task = asyncio.create_task(runtime._dispatch_batch([first]))
        await asyncio.sleep(0)
        second_task = asyncio.create_task(runtime._dispatch_batch([second]))
        await asyncio.sleep(0)

        assert encode_started == ["request-0"]
        release_first_encode.set()
        await asyncio.wait_for(second_encode_done.wait(), 1)
        assert publish_started == ["request-0:0:embedding"]

        release_first_publish.set()
        await asyncio.wait_for(second_publish_started.wait(), 1)
        await asyncio.gather(first_task, second_task)
        assert publish_started == [
            "request-0:0:embedding",
            "request-1:0:embedding",
        ]

    asyncio.run(run())


def test_runtime_pipelines_preprocess_and_encode_stages():
    async def run() -> None:
        first_preprocess_started = asyncio.Event()
        first_encode_started = asyncio.Event()
        second_preprocess_started = asyncio.Event()
        second_preprocess_done = asyncio.Event()
        release_first_preprocess = asyncio.Event()
        release_first_encode = asyncio.Event()

        class FakeTransfer:
            async def publish(self, transfer_id, embedding):
                return {"transfer_id": transfer_id}

            async def release_completed(self) -> None:
                pass

            def release(self, transfer_id) -> None:
                pass

            def close(self) -> None:
                pass

        async def preprocess(requests):
            req_id = requests[0]["req_id"]
            if req_id == "request-0":
                first_preprocess_started.set()
                await release_first_preprocess.wait()
            else:
                second_preprocess_started.set()
                second_preprocess_done.set()
            return req_id

        async def encode_preprocessed(req_id):
            if req_id == "request-0":
                first_encode_started.set()
                await release_first_encode.wait()
            return [(jnp.zeros((1, 2)), {})]

        async def encode(_requests):
            raise AssertionError("staged encoder must use its preprocessed path")

        runtime = EncoderRuntime(
            encode,
            FakeTransfer(),
            batch_preprocess_fn=preprocess,
            batch_encode_preprocessed_fn=encode_preprocessed,
        )
        first = PendingRequest({"req_id": "request-0", "modality": "IMAGE"})
        second = PendingRequest({"req_id": "request-1", "modality": "IMAGE"})
        first.mark_dequeued()
        second.mark_dequeued()

        first_task = asyncio.create_task(runtime._dispatch_batch([first]))
        await asyncio.wait_for(first_preprocess_started.wait(), 1)
        second_task = asyncio.create_task(runtime._dispatch_batch([second]))
        await asyncio.sleep(0)
        assert not second_preprocess_started.is_set()

        release_first_preprocess.set()
        await asyncio.wait_for(first_encode_started.wait(), 1)
        await asyncio.wait_for(second_preprocess_done.wait(), 1)
        assert not first_task.done()
        assert not second_task.done()

        release_first_encode.set()
        await asyncio.gather(first_task, second_task)

    asyncio.run(run())
