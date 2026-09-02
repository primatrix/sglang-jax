from __future__ import annotations

import asyncio
import threading
from concurrent.futures import Future
from typing import ClassVar
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sgl_jax.raiden import raiden_requested
from sgl_jax.srt.disaggregation.encoder.client import PendingEncoderRequest
from sgl_jax.srt.disaggregation.encoder.embedding_data import (
    EmbeddingData,
    MultiModalEmbeddingData,
)
from sgl_jax.srt.disaggregation.encoder.raiden import (
    DeferredRaidenReceiveSession,
    RaidenEncoderServerTransfer,
    RaidenReceiverBackend,
    RaidenReceiveSession,
)
from sgl_jax.srt.managers.io_struct import TokenizedGenerateReqInput
from sgl_jax.srt.multimodal.common.modality_enum import Modality


@pytest.fixture(autouse=True)
def _pretend_raiden_is_preloaded(monkeypatch):
    monkeypatch.setattr(
        "sgl_jax.srt.disaggregation.encoder.raiden.require_raiden_preloaded",
        lambda: None,
    )


class _FakeRaidenWrapper:
    instances: ClassVar[list[_FakeRaidenWrapper]] = []
    start_barrier: ClassVar[threading.Barrier | None] = None

    def __init__(self, host: str, port: int, *, parallelism: int) -> None:
        self.host = host
        self.port = port
        self.parallelism = parallelism
        self.endpoints = [{"endpoint": "127.0.0.1:7788", "shards": [0]}]
        self.started = None
        self.registered = None
        self.read = None
        self.reads = []
        self.stats = ([], [], [])
        self.instances.append(self)

    def start(self, buffers, **kwargs) -> None:
        if self.start_barrier is not None:
            self.start_barrier.wait(timeout=1)
        self.started = (buffers, kwargs)

    def register_read(self, *args) -> bool:
        self.registered = args
        return True

    def start_read(self, *args) -> None:
        self.read = args
        self.reads.append(args)

    def poll_stats(self):
        return self.stats


class _NoMessageRouter:
    def __init__(self) -> None:
        self.unregistered = None

    def poll(self, _req_ids):
        return None

    def unregister(self, req_ids) -> None:
        self.unregistered = req_ids


def test_raiden_loader_recognizes_encoder_backend():
    assert raiden_requested(["--encoder-transfer-backend", "raiden"])
    assert raiden_requested(["--encoder-transfer-backend=raiden"])
    assert not raiden_requested(["--encoder-transfer-backend", "jax_pull"])


def test_raiden_server_binds_the_produced_embedding(monkeypatch):
    _FakeRaidenWrapper.instances.clear()
    monkeypatch.setattr(
        "sgl_jax.srt.disaggregation.encoder.raiden.RaidenTransferWrapper",
        _FakeRaidenWrapper,
    )
    transfer = RaidenEncoderServerTransfer(
        "10.0.0.4",
        parallelism=3,
        timeout_s=12.0,
    )
    embedding = jnp.arange(12, dtype=jnp.float32).reshape(4, 3)

    metadata = asyncio.run(transfer.publish("part-0:embedding", embedding))

    session = _FakeRaidenWrapper.instances[0]
    buffers, options = session.started
    assert len(buffers) == 1
    np.testing.assert_array_equal(buffers[0][0], embedding)
    assert buffers[0].shape == (1, 4, 3)
    assert options == {"max_blocks": 1, "num_slots": 1, "timeout_s": 12.0}
    assert session.registered == (
        "part-0:embedding",
        metadata["transfer_uuid"],
        [0],
    )
    assert metadata["transfer_address"] == session.endpoints
    assert metadata["transfer_host"] == "10.0.0.4"
    transfer.close()


def test_raiden_server_prepares_batch_transfers_concurrently(monkeypatch):
    _FakeRaidenWrapper.instances.clear()
    _FakeRaidenWrapper.start_barrier = threading.Barrier(2)
    monkeypatch.setattr(
        "sgl_jax.srt.disaggregation.encoder.raiden.RaidenTransferWrapper",
        _FakeRaidenWrapper,
    )
    transfer = RaidenEncoderServerTransfer("10.0.0.4", parallelism=2)

    async def publish_batch() -> None:
        await asyncio.gather(
            transfer.publish("part-0:embedding", jnp.zeros((2, 3))),
            transfer.publish("part-1:embedding", jnp.zeros((2, 3))),
        )

    try:
        asyncio.run(publish_batch())
    finally:
        _FakeRaidenWrapper.start_barrier = None
        transfer.close()

    assert len(_FakeRaidenWrapper.instances) == 2


def test_raiden_server_publishes_batch_as_one_contiguous_transfer(monkeypatch):
    _FakeRaidenWrapper.instances.clear()
    monkeypatch.setattr(
        "sgl_jax.srt.disaggregation.encoder.raiden.RaidenTransferWrapper",
        _FakeRaidenWrapper,
    )
    transfer = RaidenEncoderServerTransfer("10.0.0.4", parallelism=2)
    first = jnp.arange(6, dtype=jnp.float32).reshape(2, 3)
    second = jnp.arange(9, dtype=jnp.float32).reshape(3, 3) + 10

    metadata = asyncio.run(
        transfer.publish_batch(
            [
                ("part-0:embedding", first),
                ("part-1:embedding", second),
            ]
        )
    )

    assert len(_FakeRaidenWrapper.instances) == 1
    session = _FakeRaidenWrapper.instances[0]
    buffers, options = session.started
    np.testing.assert_array_equal(buffers[0][0], jnp.concatenate([first, second]))
    assert buffers[0].shape == (1, 5, 3)
    assert options == {"max_blocks": 1, "num_slots": 1, "timeout_s": 300.0}
    assert metadata[0]["transfer_id"] == metadata[1]["transfer_id"]
    assert [item["transfer_offset"] for item in metadata] == [0, 2]
    assert all(item["transfer_shape"] == (5, 3) for item in metadata)
    assert all(item["transfer_group_size"] == 2 for item in metadata)
    assert session.registered == (
        metadata[0]["transfer_id"],
        metadata[0]["transfer_uuid"],
        [0],
    )
    transfer.close()


def test_raiden_server_bounds_contiguous_groups_by_channel_parallelism(monkeypatch):
    _FakeRaidenWrapper.instances.clear()
    monkeypatch.setattr(
        "sgl_jax.srt.disaggregation.encoder.raiden.RaidenTransferWrapper",
        _FakeRaidenWrapper,
    )
    transfer = RaidenEncoderServerTransfer("10.0.0.4", parallelism=2)
    embeddings = [jnp.full((2, 3), index, dtype=jnp.float32) for index in range(5)]

    metadata = asyncio.run(
        transfer.publish_batch(
            [(f"part-{index}:embedding", embedding) for index, embedding in enumerate(embeddings)]
        )
    )

    assert len(_FakeRaidenWrapper.instances) == 3
    assert [session.started[0][0].shape for session in _FakeRaidenWrapper.instances] == [
        (1, 4, 3),
        (1, 4, 3),
        (1, 2, 3),
    ]
    assert [item["transfer_group_size"] for item in metadata] == [2, 2, 2, 2, 1]
    assert metadata[0]["transfer_id"] == metadata[1]["transfer_id"]
    assert metadata[2]["transfer_id"] == metadata[3]["transfer_id"]
    assert len({item["transfer_id"] for item in metadata}) == 3
    transfer.close()


def test_raiden_server_setup_concurrency_is_independent_from_transfer_channels(
    monkeypatch,
):
    _FakeRaidenWrapper.instances.clear()
    _FakeRaidenWrapper.start_barrier = threading.Barrier(2)
    monkeypatch.setattr(
        "sgl_jax.srt.disaggregation.encoder.raiden.RaidenTransferWrapper",
        _FakeRaidenWrapper,
    )
    transfer = RaidenEncoderServerTransfer(
        "10.0.0.4",
        parallelism=1,
        setup_parallelism=2,
    )

    async def publish_batch() -> None:
        await asyncio.gather(
            transfer.publish("part-0:embedding", jnp.zeros((2, 3))),
            transfer.publish("part-1:embedding", jnp.zeros((2, 3))),
        )

    try:
        asyncio.run(publish_batch())
    finally:
        _FakeRaidenWrapper.start_barrier = None
        transfer.close()

    assert [session.parallelism for session in _FakeRaidenWrapper.instances] == [1, 1]


def test_raiden_server_reaps_completed_sender(monkeypatch):
    _FakeRaidenWrapper.instances.clear()
    monkeypatch.setattr(
        "sgl_jax.srt.disaggregation.encoder.raiden.RaidenTransferWrapper",
        _FakeRaidenWrapper,
    )
    transfer = RaidenEncoderServerTransfer("10.0.0.4")
    asyncio.run(transfer.publish("part-0:embedding", jnp.zeros((2, 3))))
    _FakeRaidenWrapper.instances[0].stats = (["part-0:embedding"], [], [])

    async def stop_after_poll(_delay):
        raise asyncio.CancelledError

    monkeypatch.setattr(
        "sgl_jax.srt.disaggregation.encoder.raiden.asyncio.sleep",
        stop_after_poll,
    )
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(transfer.release_completed())

    assert not transfer._sessions
    transfer.close()


def test_raiden_server_logs_group_and_request_inflight(monkeypatch, caplog):
    transfer = RaidenEncoderServerTransfer(
        "10.0.0.4",
        parallelism=2,
        log_inflight=True,
    )
    first = mock.Mock()
    first.poll_stats.return_value = (["group-0"], [], [])
    second = mock.Mock()
    second.poll_stats.return_value = (["group-1"], [], [])

    caplog.set_level("INFO", logger="sgl_jax.srt.disaggregation.encoder.raiden")
    transfer._register_session("group-0", first, group_size=2)
    transfer._register_session("group-1", second, group_size=2)

    assert "event=start" in caplog.text
    assert "group_size=2 inflight_groups=2 inflight_requests=4" in caplog.text

    async def stop_after_poll(_delay):
        raise asyncio.CancelledError

    monkeypatch.setattr(
        "sgl_jax.srt.disaggregation.encoder.raiden.asyncio.sleep",
        stop_after_poll,
    )
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(transfer.release_completed())

    assert "event=sent" in caplog.text
    assert "group_size=2 inflight_groups=0 inflight_requests=0" in caplog.text
    transfer.close()


def test_raiden_request_receives_into_matching_jax_buffer(monkeypatch):
    _FakeRaidenWrapper.instances.clear()
    monkeypatch.setattr(
        "sgl_jax.srt.disaggregation.encoder.raiden.RaidenTransferWrapper",
        _FakeRaidenWrapper,
    )
    register_future: Future[None] = Future()
    register_future.set_result(None)
    metadata_router = _NoMessageRouter()
    backend = RaidenReceiverBackend(
        host="10.0.0.9",
        sharding=jax.sharding.SingleDeviceSharding(jax.local_devices()[0]),
        parallelism=2,
        transfer_timeout_s=30.0,
    )
    request = PendingEncoderRequest(
        recv_req=TokenizedGenerateReqInput(rid="request-0"),
        started_at=0.0,
        metadata_router=metadata_router,
        metadata_req_ids=("part-0",),
        register_future=register_future,
        accumulator=MultiModalEmbeddingData(1),
        backend=backend,
    )
    data = EmbeddingData(
        req_id="part-0",
        num_parts=1,
        part_idx=0,
        grid_dim=None,
        modality=Modality.IMAGE,
        embedding_shape=(2, 3),
        dtype="float32",
        transfer_id="part-0:embedding",
        transfer_uuid=17,
        transfer_address=[{"endpoint": "127.0.0.1:7788", "shards": [0]}],
        transfer_host="10.0.0.8",
        transfer_block_ids=[0],
    )

    request._start_receive(data)

    receive_session = request.sessions[0][1]
    assert isinstance(receive_session, DeferredRaidenReceiveSession)
    session = receive_session._future.result(timeout=1)
    transfer = session.transfer
    buffers, options = transfer.started
    assert buffers[0].shape == (1, 2, 3)
    assert buffers[0].dtype == jnp.float32
    assert session.buffer.shape == (1, 2, 3)
    assert options == {"max_blocks": 1, "num_slots": 1, "timeout_s": 30.0}
    assert transfer.read == (
        "part-0:embedding",
        17,
        [{"endpoint": "10.0.0.8:7788", "shards": [0]}],
        [0],
        [0],
    )

    transfer.stats = ([], ["part-0:embedding"], [])
    result = request.poll()

    np.testing.assert_array_equal(result["embeddings"][Modality.IMAGE], np.zeros((2, 3)))
    request.close()
    backend.close()
    assert metadata_router.unregistered == ("part-0",)


def test_raiden_receiver_reuses_manager_and_pool_blocks(monkeypatch):
    _FakeRaidenWrapper.instances.clear()
    monkeypatch.setattr(
        "sgl_jax.srt.disaggregation.encoder.raiden.RaidenTransferWrapper",
        _FakeRaidenWrapper,
    )
    backend = RaidenReceiverBackend(
        host="10.0.0.9",
        sharding=jax.sharding.SingleDeviceSharding(jax.local_devices()[0]),
        parallelism=2,
        transfer_timeout_s=30.0,
    )

    def metadata(index: int) -> EmbeddingData:
        return EmbeddingData(
            req_id=f"part-{index}",
            num_parts=1,
            part_idx=0,
            grid_dim=None,
            modality=Modality.IMAGE,
            embedding_shape=(2, 3),
            dtype="float32",
            transfer_id=f"part-{index}:embedding",
            transfer_uuid=index,
            transfer_address=[{"endpoint": "127.0.0.1:7788", "shards": [0]}],
            transfer_host="10.0.0.8",
            transfer_block_ids=[0],
        )

    first = backend.start(metadata(1))._future.result(timeout=1)
    second = backend.start(metadata(2))._future.result(timeout=1)

    assert len(_FakeRaidenWrapper.instances) == 2
    assert first.transfer is not second.transfer
    assert [first.lane_id, second.lane_id] == [0, 1]
    assert first.transfer.read[-1] == [0]
    assert second.transfer.read[-1] == [0]

    third_future = backend.start(metadata(3))._future
    assert not third_future.done()
    first.transfer.stats = ([], [first.transfer_id], [])
    assert first.poll().shape == (2, 3)
    third = third_future.result(timeout=1)
    second.transfer.stats = ([], [second.transfer_id], [])
    assert second.poll().shape == (2, 3)

    assert third.transfer is first.transfer
    assert third.lane_id == first.lane_id
    third.close()
    backend.close()


def test_raiden_receiver_pulls_group_once_and_returns_request_views(monkeypatch):
    _FakeRaidenWrapper.instances.clear()
    monkeypatch.setattr(
        "sgl_jax.srt.disaggregation.encoder.raiden.RaidenTransferWrapper",
        _FakeRaidenWrapper,
    )
    backend = RaidenReceiverBackend(
        host="10.0.0.9",
        sharding=jax.sharding.SingleDeviceSharding(jax.local_devices()[0]),
        parallelism=2,
        transfer_timeout_s=30.0,
    )

    def metadata(req_id: str, shape: tuple[int, int], offset: int) -> EmbeddingData:
        return EmbeddingData(
            req_id=req_id,
            num_parts=1,
            part_idx=0,
            grid_dim=None,
            modality=Modality.IMAGE,
            embedding_shape=shape,
            dtype="float32",
            transfer_id="batch-0:embedding",
            transfer_uuid=17,
            transfer_address=[{"endpoint": "127.0.0.1:7788", "shards": [0]}],
            transfer_host="10.0.0.8",
            transfer_block_ids=[0],
            transfer_group_size=2,
            transfer_shape=(5, 3),
            transfer_offset=offset,
        )

    first = backend.start(metadata("part-0", (2, 3), 0))._future.result(timeout=1)
    second = backend.start(metadata("part-1", (3, 3), 2))._future.result(timeout=1)

    assert first.group is second.group
    physical = first.group._session
    transfer = physical.transfer
    packed = jnp.arange(15, dtype=jnp.float32).reshape(1, 5, 3)
    physical.pool._lanes[physical.lane_id] = (packed, transfer)
    transfer.stats = ([], [physical.transfer_id], [])

    np.testing.assert_array_equal(first.poll(), packed[0, :2])
    np.testing.assert_array_equal(second.poll(), packed[0, 2:])
    assert "batch-0:embedding" not in backend._groups
    backend.close()


def test_raiden_request_surfaces_receive_failure():
    transfer = mock.Mock()
    transfer.poll_stats.return_value = ([], [], ["part-0:embedding"])

    session = RaidenReceiveSession(
        "part-0:embedding",
        jnp.zeros((1, 1)),
        transfer,
    )

    with pytest.raises(RuntimeError, match="Raiden embedding transfer failed"):
        session.poll()
