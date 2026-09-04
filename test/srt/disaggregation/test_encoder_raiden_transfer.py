from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import Future
from typing import ClassVar
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sgl_jax.raiden import raiden_requested
from sgl_jax.srt.disaggregation.encoder.client import (
    DeferredReceiveSession,
    PendingEncoderRequest,
)
from sgl_jax.srt.disaggregation.encoder.embedding_data import (
    EmbeddingData,
    MultiModalEmbeddingData,
)
from sgl_jax.srt.disaggregation.encoder.raiden_pool import (
    RaidenReceivePool,
    RaidenReceiveSession,
    RaidenSendPool,
)
from sgl_jax.srt.disaggregation.encoder.raiden_transfer import (
    RaidenEncoderServerTransfer,
    RaidenReceiverBackend,
)
from sgl_jax.srt.managers.io_struct import TokenizedGenerateReqInput
from sgl_jax.srt.multimodal.common.modality_enum import Modality


@pytest.fixture(autouse=True)
def _pretend_raiden_is_preloaded(monkeypatch):
    monkeypatch.setattr(
        "sgl_jax.srt.disaggregation.encoder.raiden_transfer.require_raiden_preloaded",
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


def _poll_until_ready(session):
    deadline = time.monotonic() + 1
    while time.monotonic() < deadline:
        result = session.poll()
        if result is not None:
            return result
        time.sleep(0.001)
    raise AssertionError("Raiden receive did not become ready")


async def _publish(transfer, transfer_id, embedding):
    reservations = await asyncio.to_thread(
        transfer.reserve_batch_sync,
        [transfer_id],
    )
    staged = transfer.stage_batch_sync(reservations, [embedding])
    return await asyncio.to_thread(transfer.publish_sync, staged[0])


def test_raiden_loader_recognizes_encoder_backend():
    assert raiden_requested(["--encoder-transfer-backend", "raiden"])
    assert raiden_requested(["--encoder-transfer-backend=raiden"])
    assert not raiden_requested(["--encoder-transfer-backend", "jax_pull"])


def test_raiden_server_uses_donated_request_pool(monkeypatch):
    _FakeRaidenWrapper.instances.clear()
    monkeypatch.setattr(
        "sgl_jax.srt.disaggregation.encoder.raiden_transfer.RaidenTransferWrapper",
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

    first_metadata = asyncio.run(_publish(transfer, "part-0:embedding", first))
    assert isinstance(transfer._pool, RaidenSendPool)
    pool_pointer = transfer._pool._buffer.unsafe_buffer_pointer()
    second_metadata = asyncio.run(_publish(transfer, "part-1:embedding", second))

    session = _FakeRaidenWrapper.instances[0]
    assert transfer._raiden is session
    assert not hasattr(transfer._pool, "transfer")
    buffers, options = session.started
    assert len(buffers) == 1
    assert buffers[0].shape == (2, 4, 2, 8, 128)
    assert options == {"max_blocks": 1, "num_slots": 2, "timeout_s": 12.0}
    buffer = transfer._pool._buffer.reshape(2, 4, -1)
    np.testing.assert_array_equal(buffer[0, :, :3], first)
    np.testing.assert_array_equal(buffer[1, :, :3], second)
    assert transfer._pool._buffer.unsafe_buffer_pointer() == pool_pointer
    assert session.registrations == [
        ("part-0:embedding", first_metadata["transfer_uuid"], [0]),
        ("part-1:embedding", second_metadata["transfer_uuid"], [1]),
    ]
    assert first_metadata["transfer_id"] != second_metadata["transfer_id"]
    assert first_metadata["transfer_address"] == session.endpoints
    assert first_metadata["transfer_host"] == "10.0.0.4"
    transfer.close()


def test_raiden_server_writes_packed_batch_directly_into_pool(monkeypatch):
    _FakeRaidenWrapper.instances.clear()
    monkeypatch.setattr(
        "sgl_jax.srt.disaggregation.encoder.raiden_transfer.RaidenTransferWrapper",
        _FakeRaidenWrapper,
    )
    transfer = RaidenEncoderServerTransfer("10.0.0.4", pool_size=4)
    reservations = transfer.reserve_batch_sync(
        ["part-0:embedding", "part-1:embedding", "part-2:embedding"]
    )
    packed = jnp.arange(24, dtype=jnp.float32).reshape(8, 3)

    staged = transfer.stage_packed_batch_sync(reservations, packed, (2, 2, 2))
    metadata = [transfer.publish_sync(item) for item in staged]

    assert isinstance(transfer._pool, RaidenSendPool)
    buffer = np.asarray(transfer._pool.buffer).reshape(4, 2, -1)
    np.testing.assert_array_equal(buffer[0, :, :3], packed[0:2])
    np.testing.assert_array_equal(buffer[1, :, :3], packed[2:4])
    np.testing.assert_array_equal(buffer[2, :, :3], packed[4:6])
    assert [item["transfer_block_ids"] for item in metadata] == [[0], [1], [2]]
    assert len(transfer._pool._packed_copies) == 1
    transfer.close()


def test_raiden_server_rejects_embedding_that_does_not_match_single_pool(monkeypatch):
    _FakeRaidenWrapper.instances.clear()
    monkeypatch.setattr(
        "sgl_jax.srt.disaggregation.encoder.raiden_transfer.RaidenTransferWrapper",
        _FakeRaidenWrapper,
    )
    transfer = RaidenEncoderServerTransfer("10.0.0.4")
    asyncio.run(_publish(transfer, "part-0:embedding", jnp.zeros((2, 3))))

    with pytest.raises(ValueError, match="pool embedding mismatch"):
        asyncio.run(_publish(transfer, "part-1:embedding", jnp.zeros((4, 3))))

    assert len(_FakeRaidenWrapper.instances) == 1
    transfer.close()


def test_raiden_server_backpressures_when_pool_is_full(monkeypatch):
    _FakeRaidenWrapper.instances.clear()
    monkeypatch.setattr(
        "sgl_jax.srt.disaggregation.encoder.raiden_transfer.RaidenTransferWrapper",
        _FakeRaidenWrapper,
    )
    transfer = RaidenEncoderServerTransfer("10.0.0.4", pool_size=1)

    async def run() -> None:
        await _publish(transfer, "part-0:embedding", jnp.zeros((2, 3)))
        blocked = asyncio.create_task(_publish(transfer, "part-1:embedding", jnp.ones((2, 3))))
        await asyncio.sleep(0.05)
        assert not blocked.done()
        _FakeRaidenWrapper.instances[0].stats = (["part-0:embedding"], [], [])
        await asyncio.wait_for(blocked, 1)

    asyncio.run(run())
    transfer.close()


def test_raiden_server_reaps_completed_sender(monkeypatch):
    _FakeRaidenWrapper.instances.clear()
    monkeypatch.setattr(
        "sgl_jax.srt.disaggregation.encoder.raiden_transfer.RaidenTransferWrapper",
        _FakeRaidenWrapper,
    )
    transfer = RaidenEncoderServerTransfer("10.0.0.4", pool_size=1)
    asyncio.run(_publish(transfer, "part-0:embedding", jnp.zeros((2, 3))))
    _FakeRaidenWrapper.instances[0].stats = (["part-0:embedding"], [], [])

    reservations = transfer.reserve_batch_sync(["part-1:embedding"])
    transfer.cancel_batch(reservations)

    assert not transfer._active
    transfer.close()


def test_raiden_reaps_only_when_reservation_needs_capacity(monkeypatch):
    _FakeRaidenWrapper.instances.clear()
    monkeypatch.setattr(
        "sgl_jax.srt.disaggregation.encoder.raiden_transfer.RaidenTransferWrapper",
        _FakeRaidenWrapper,
    )
    transfer = RaidenEncoderServerTransfer("10.0.0.4", pool_size=2)
    asyncio.run(_publish(transfer, "part-0:embedding", jnp.zeros((2, 3))))
    session = _FakeRaidenWrapper.instances[0]
    session.stats = (["part-0:embedding"], [], [])

    spare = transfer.reserve_batch_sync(["part-1:embedding"])
    assert "part-0:embedding" in transfer._active
    assert session.stats[0] == ["part-0:embedding"]
    transfer.cancel_batch(spare)

    full_batch = transfer.reserve_batch_sync(["part-2:embedding", "part-3:embedding"])
    assert "part-0:embedding" not in transfer._active
    transfer.cancel_batch(full_batch)
    transfer.close()


def test_raiden_register_waits_for_copy_ticket(monkeypatch):
    class PendingReady:
        def __init__(self):
            self.done = threading.Event()

        def block_until_ready(self):
            assert self.done.wait(1)

        def is_ready(self):
            return self.done.is_set()

    _FakeRaidenWrapper.instances.clear()
    monkeypatch.setattr(
        "sgl_jax.srt.disaggregation.encoder.raiden_transfer.RaidenTransferWrapper",
        _FakeRaidenWrapper,
    )
    ready = PendingReady()
    monkeypatch.setattr(RaidenSendPool, "copy_async", lambda *_args: ready)
    transfer = RaidenEncoderServerTransfer("10.0.0.4", pool_size=1)
    reservations = transfer.reserve_batch_sync(["part-0:embedding"])
    staged = transfer.stage_batch_sync(reservations, [jnp.zeros((2, 3))])
    published = threading.Event()

    def publish():
        transfer.publish_sync(staged[0])
        published.set()

    thread = threading.Thread(target=publish)
    thread.start()
    time.sleep(0.02)
    assert not _FakeRaidenWrapper.instances[0].registrations

    ready.done.set()
    assert published.wait(1)
    thread.join()
    assert _FakeRaidenWrapper.instances[0].registrations
    transfer.close()


def test_raiden_request_receives_into_matching_jax_buffer(monkeypatch):
    _FakeRaidenWrapper.instances.clear()
    monkeypatch.setattr(
        "sgl_jax.srt.disaggregation.encoder.raiden_pool.RaidenTransferWrapper",
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
        registration_futures=(register_future,),
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

    request.sessions[data.part_idx] = (data, backend.start(data))

    receive_session = request.sessions[0][1]
    assert isinstance(receive_session, DeferredReceiveSession)
    session = receive_session._future.result(timeout=1)
    transfer = session.pool._transfer
    buffers, options = transfer.started
    assert buffers[0].shape == (2, 2, 2, 8, 128)
    assert buffers[0].dtype == jnp.float32
    assert session.pool._buffer.shape == (2, 2, 2, 8, 128)
    assert options == {"max_blocks": 1, "num_slots": 2, "timeout_s": 30.0}
    assert transfer.read == (
        "part-0:embedding",
        17,
        [{"endpoint": "10.0.0.8:7788", "shards": [0]}],
        [0],
        [0],
    )

    transfer.stats = ([], ["part-0:embedding"], [])
    result = _poll_until_ready(request)

    np.testing.assert_array_equal(result["embeddings"][Modality.IMAGE], np.zeros((2, 3)))
    request.close()
    backend.close()
    assert metadata_router.unregistered == ("part-0",)


def test_raiden_receiver_reuses_manager_and_pool_blocks(monkeypatch):
    _FakeRaidenWrapper.instances.clear()
    monkeypatch.setattr(
        "sgl_jax.srt.disaggregation.encoder.raiden_pool.RaidenTransferWrapper",
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
    assert first.pool is second.pool
    assert [first.lane_id, second.lane_id] == [0, 1]
    assert [read[-1] for read in first.pool._transfer.reads] == [[0], [1]]

    third_future = backend.start(metadata(3))._future
    assert not third_future.done()
    first.pool._transfer.stats = ([], [first.transfer_id], [])
    assert _poll_until_ready(first).shape == (2, 3)
    third = third_future.result(timeout=1)
    second.pool._transfer.stats = ([], [second.transfer_id], [])
    assert _poll_until_ready(second).shape == (2, 3)

    assert third.pool is first.pool
    assert third.lane_id == first.lane_id
    third.close()
    backend.close()


def test_raiden_receiver_rejects_embedding_that_does_not_match_single_pool(monkeypatch):
    _FakeRaidenWrapper.instances.clear()
    monkeypatch.setattr(
        "sgl_jax.srt.disaggregation.encoder.raiden_pool.RaidenTransferWrapper",
        _FakeRaidenWrapper,
    )
    backend = RaidenReceiverBackend(
        host="10.0.0.9",
        sharding=jax.sharding.SingleDeviceSharding(jax.local_devices()[0]),
        parallelism=1,
        pool_size=2,
        transfer_timeout_s=30.0,
    )

    def metadata(transfer_id: str, shape: tuple[int, int]) -> EmbeddingData:
        return EmbeddingData(
            req_id=transfer_id,
            num_parts=1,
            part_idx=0,
            grid_dim=None,
            modality=Modality.IMAGE,
            embedding_shape=shape,
            dtype="float32",
            transfer_id=transfer_id,
            transfer_uuid=1,
            transfer_address=[{"endpoint": "127.0.0.1:7788", "shards": [0]}],
            transfer_host="10.0.0.8",
            transfer_block_ids=[0],
        )

    first = backend.start(metadata("part-0:embedding", (2, 3)))._future.result(timeout=1)
    with pytest.raises(ValueError, match="pool embedding mismatch"):
        backend.start(metadata("part-1:embedding", (4, 3)))._future.result(timeout=1)

    assert len(_FakeRaidenWrapper.instances) == 1
    first.close()
    backend.close()


def test_raiden_request_surfaces_receive_failure():
    pool = mock.Mock()
    pool.poll.side_effect = RuntimeError("Raiden embedding transfer failed: part-0:embedding")
    session = RaidenReceiveSession(
        transfer_id="part-0:embedding",
        lane_id=0,
        pool=pool,
    )

    with pytest.raises(RuntimeError, match="Raiden embedding transfer failed"):
        session.poll()


def test_raiden_receive_poll_does_not_wait_for_device_copy(monkeypatch):
    class PendingCopy:
        ready = False

        def is_ready(self) -> bool:
            return self.ready

    transfer_id = "part-0:embedding"
    pending_copy = PendingCopy()
    pool = object.__new__(RaidenReceivePool)
    pool._sharding = jax.sharding.SingleDeviceSharding(jax.local_devices()[0])
    pool.shape = (2, 3)
    pool._block_shape = (2, 2, 8, 128)
    pool._buffer = jnp.zeros((1, *pool._block_shape))
    pool._transfer = mock.Mock()
    pool._transfer.poll_stats.return_value = ([], [], [])
    pool._condition = threading.Condition()
    pool._free = []
    pool._active = {transfer_id: 0}
    pool._abandoned = set()
    pool._materializing = {}
    pool._received_ns = {transfer_id: time.time_ns()}
    pool._materialize_start_ns = {}
    pool._received = {transfer_id}
    pool._failed = set()

    monkeypatch.setattr(
        "sgl_jax.srt.disaggregation.encoder.raiden_pool.jax.device_put",
        lambda *_args, **_kwargs: pending_copy,
    )
    session = RaidenReceiveSession(transfer_id=transfer_id, lane_id=0, pool=pool)

    assert session.poll() is None
    assert pool._active == {transfer_id: 0}
    assert pool._free == []

    pending_copy.ready = True
    assert session.poll() is pending_copy
    assert session.timing_meta["receive_transfer_done_ns"] > 0
    assert (
        session.timing_meta["receive_materialize_done_ns"]
        >= session.timing_meta["receive_materialize_start_ns"]
    )
    assert pool._active == {}
    assert pool._free == [0]


def test_raiden_receive_pool_can_reuse_one_shared_stats_refresh():
    pool = object.__new__(RaidenReceivePool)
    pool._transfer = mock.Mock()
    pool._transfer.poll_stats.return_value = ([], [], [])
    pool._condition = threading.Condition()
    pool._closed = False
    pool._active = {"part-0:embedding": 0, "part-1:embedding": 1}
    pool._abandoned = set()
    pool._materializing = {}
    pool._received_ns = {}
    pool._received = set()
    pool._failed = set()

    pool.progress()
    assert pool.poll("part-0:embedding", 0, refresh_backend=False) is None
    assert pool.poll("part-1:embedding", 1, refresh_backend=False) is None

    pool._transfer.poll_stats.assert_called_once_with()
