from __future__ import annotations

import asyncio
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

    def __init__(self, host: str, port: int, *, parallelism: int) -> None:
        self.host = host
        self.port = port
        self.parallelism = parallelism
        self.endpoints = [{"endpoint": "127.0.0.1:7788", "shards": [0]}]
        self.started = None
        self.registered = None
        self.registrations = []
        self.read = None
        self.reads = []
        self.stats = ([], [], [])
        self.instances.append(self)

    def start(self, buffers, **kwargs) -> None:
        self.started = (buffers, kwargs)

    def register_read(self, *args) -> bool:
        self.registered = args
        self.registrations.append(args)
        return True

    def start_read(self, *args) -> None:
        self.read = args
        self.reads.append(args)

    def poll_stats(self):
        stats, self.stats = self.stats, ([], [], [])
        return stats

    @property
    def host_ip(self) -> str:
        return self.host


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


def test_raiden_server_uses_donated_request_pool(monkeypatch):
    _FakeRaidenWrapper.instances.clear()
    monkeypatch.setattr(
        "sgl_jax.srt.disaggregation.encoder.raiden.RaidenTransferWrapper",
        _FakeRaidenWrapper,
    )
    transfer = RaidenEncoderServerTransfer(
        "10.0.0.4",
        parallelism=3,
        pool_size=2,
        timeout_s=12.0,
    )
    first = jnp.arange(12, dtype=jnp.float32).reshape(4, 3)
    second = first + 20

    first_metadata = asyncio.run(transfer.publish("part-0:embedding", first))
    pool_pointer = transfer._pools[0]._buffer.unsafe_buffer_pointer()
    second_metadata = asyncio.run(transfer.publish("part-1:embedding", second))

    session = _FakeRaidenWrapper.instances[0]
    buffers, options = session.started
    assert len(buffers) == 1
    assert buffers[0].shape == (2, 4, 2, 8, 128)
    assert options == {"max_blocks": 1, "num_slots": 2, "timeout_s": 12.0}
    buffer = transfer._pools[0]._buffer.reshape(2, 4, -1)
    np.testing.assert_array_equal(buffer[0, :, :3], first)
    np.testing.assert_array_equal(buffer[1, :, :3], second)
    assert transfer._pools[0]._buffer.unsafe_buffer_pointer() == pool_pointer
    assert session.registrations == [
        ("part-0:embedding", first_metadata["transfer_uuid"], [0]),
        ("part-1:embedding", second_metadata["transfer_uuid"], [1]),
    ]
    assert first_metadata["transfer_id"] != second_metadata["transfer_id"]
    assert first_metadata["transfer_address"] == session.endpoints
    assert first_metadata["transfer_host"] == "10.0.0.4"
    transfer.close()


def test_raiden_server_backpressures_when_pool_is_full(monkeypatch):
    _FakeRaidenWrapper.instances.clear()
    monkeypatch.setattr(
        "sgl_jax.srt.disaggregation.encoder.raiden.RaidenTransferWrapper",
        _FakeRaidenWrapper,
    )
    transfer = RaidenEncoderServerTransfer("10.0.0.4", pool_size=1)

    async def run() -> None:
        await transfer.publish("part-0:embedding", jnp.zeros((2, 3)))
        blocked = asyncio.create_task(transfer.publish("part-1:embedding", jnp.ones((2, 3))))
        await asyncio.sleep(0.05)
        assert not blocked.done()
        transfer.release("part-0:embedding")
        await asyncio.wait_for(blocked, 1)

    asyncio.run(run())
    transfer.close()


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

    assert not transfer._active
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
        pool_size=2,
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
    assert buffers[0].shape == (2, 2, 2, 8, 128)
    assert buffers[0].dtype == jnp.float32
    assert session.buffer.shape == (2, 2, 2, 8, 128)
    assert options == {"max_blocks": 1, "num_slots": 2, "timeout_s": 30.0}
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
        pool_size=2,
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

    assert len(_FakeRaidenWrapper.instances) == 1
    assert first.transfer is second.transfer
    assert [first.lane_id, second.lane_id] == [0, 1]
    assert [read[-1] for read in first.transfer.reads] == [[0], [1]]

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
