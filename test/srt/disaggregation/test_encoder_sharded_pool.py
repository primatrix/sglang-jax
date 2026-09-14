"""Exercise sharded writes, logical row maps, and multipart page lifetimes."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from sgl_jax.srt.disaggregation.encoder import raiden_receiver, raiden_transfer
from sgl_jax.srt.disaggregation.encoder.embedding_data import EmbeddingData
from sgl_jax.srt.disaggregation.encoder.raiden_pool import RaidenPool
from sgl_jax.srt.disaggregation.encoder.sharded_transfer import ShardedEncoderTransfer
from sgl_jax.srt.multimodal.common.modality_enum import Modality
from sgl_jax.srt.multimodal.in_model.host_orchestration import _gather_overlay


class _Transport:
    peers = {}

    def __init__(self, *args, **kwargs):
        self.host_ip = "192.0.2.1"
        self.endpoint = f"192.0.2.1:{10000 + len(self.peers)}"
        self.endpoints = [{"endpoint": self.endpoint, "shards": [0]}]
        self.peers[self.endpoint] = self
        self.reads = {}
        self.sent, self.received, self.failed = [], [], []

    def start(self, *args, **kwargs):
        pass

    def register_read(self, key, uuid, pages):
        self.reads[key] = pages
        return True

    def start_read(self, key, uuid, endpoints, source_pages, target_pages):
        source = self.peers[endpoints[0]["endpoint"]]
        assert source.reads[key] == source_pages
        values = np.asarray(source.pool.buffer)[source_pages]
        target = np.array(self.pool.buffer)
        target[np.asarray(target_pages) + self.page_base] = values
        self.pool.buffer = jax.device_put(target, self.pool.sharding)

    def poll_stats(self):
        events = self.sent, self.received, self.failed
        self.sent, self.received, self.failed = [], [], []
        return events


@pytest.fixture
def sharded_transfer(monkeypatch):
    if len(jax.devices()) < 4:
        pytest.skip("requires four devices")
    _Transport.peers = {}
    monkeypatch.setattr(raiden_transfer, "require_raiden_preloaded", lambda: None)
    monkeypatch.setattr(raiden_transfer, "RaidenTransferWrapper", _Transport)
    monkeypatch.setattr(raiden_receiver, "RaidenTransferWrapper", _Transport)
    mesh = Mesh(np.asarray(jax.devices()[:4]), ("data",))
    args = SimpleNamespace(
        encoder_transfer_max_tokens=4096,
        encoder_max_batch_size=8,
        disable_precompile=False,
        disaggregation_channel_number=1,
        encoder_transfer_pool_size=8,
        encoder_request_timeout_seconds=1,
    )
    config = SimpleNamespace(
        hidden_size=3, dtype=jnp.float32, hf_config=SimpleNamespace(vision_config=SimpleNamespace())
    )
    model = SimpleNamespace(
        mesh=mesh,
        visual=SimpleNamespace(
            vision_tp=False,
            specs=SimpleNamespace(
                batch_axis="data",
                sharding=lambda axis: NamedSharding(mesh, PartitionSpec(axis)),
            ),
        ),
        get_multimodal_embedding_packed_capacities=lambda: (16,),
    )
    sender = ShardedEncoderTransfer("192.0.2.1", args, config, model)
    for part in sender.senders:
        part._raiden.pool = part._pool
    receiver = raiden_receiver.RaidenReceiverBackend(
        "192.0.2.2",
        RaidenPool((128, 3), jnp.float32, NamedSharding(mesh, PartitionSpec("data")), capacity=32),
        parallelism=1,
        pool_size=8,
        transfer_timeout_s=1,
    )
    for rank, transfer in enumerate(receiver._transfers):
        transfer.pool = receiver.pool
        transfer.page_base = rank * 8
    yield sender, receiver, mesh
    receiver.close()
    sender.close()


def _stage(sender, mesh):
    values = np.arange(48, dtype=np.float32).reshape(16, 3)
    packed = jax.device_put(values, NamedSharding(mesh, PartitionSpec("data")))
    indices = np.array([0, 8, 4, 12, 1, 9, 5, 13, 2, 10, 6, 14, -1, -1, -1, -1], np.int32)
    counts = (5, 4, 3)
    reservations = sender.reserve_batch_sync(["a", "b", "c"], counts, output_indices=indices)
    with jax.no_tracing(True):
        sender.stage_packed_batch_sync(reservations, packed)
    metadata = sender.publish_batch_sync(reservations)
    return values[indices[:12]], counts, metadata


def _receive(receiver, counts, metadata):
    return [
        receiver._start(
            EmbeddingData(
                str(i), 1, 0, Modality.IMAGE, shape=(count, 3), dtype="float32", transfer=m
            )
        )
        for i, (count, m) in enumerate(zip(counts, metadata, strict=True))
    ]


def _complete(receiver, children, *, failed=False):
    for child in children:
        transport = receiver._transfers[receiver._pending_ranks[child]]
        (transport.failed if failed else transport.received).append(child)


def test_sharded_transfer_restores_logical_order_without_retracing(sharded_transfer):
    sender, receiver, mesh = sharded_transfer
    expected, counts, metadata = _stage(sender, mesh)
    sessions = _receive(receiver, counts, metadata)
    offset = 0
    for session, count, item in zip(sessions, counts, metadata, strict=True):
        children = [part["transfer_id"] for part in item["transfer_parts"]]
        _complete(receiver, children[1:])
        assert session.poll() is None
        _complete(receiver, children[:1])
        view = session.poll()
        actual = np.asarray(view.buffer).reshape(32 * 128, -1)[view.row_indices, :3]
        np.testing.assert_array_equal(actual, expected[offset : offset + count])
        offset += count
        view.release()
    assert receiver.pool.available_pages == 32


def test_cancelled_multipart_receive_waits_for_all_parts(sharded_transfer):
    sender, receiver, mesh = sharded_transfer
    _, counts, metadata = _stage(sender, mesh)
    session = _receive(receiver, counts[:1], metadata[:1])[0]
    held = receiver.pool.available_pages
    children = [part["transfer_id"] for part in metadata[0]["transfer_parts"]]
    session.close()
    _complete(receiver, children[:1], failed=True)
    receiver.progress()
    assert receiver.pool.available_pages == held
    _complete(receiver, children[1:])
    receiver.progress()
    assert receiver.pool.available_pages == 32


def test_multipart_rejects_duplicate_logical_rows(sharded_transfer):
    sender, receiver, mesh = sharded_transfer
    _, counts, metadata = _stage(sender, mesh)
    parts = metadata[0]["transfer_parts"]
    parts[1]["token_indices"][0] = parts[0]["token_indices"][0]
    with pytest.raises(ValueError, match="each logical token exactly once"):
        _receive(receiver, counts[:1], metadata[:1])
    assert receiver.pool.available_pages == 32


def test_sharded_pool_gather_uses_runtime_rows(sharded_transfer):
    _, receiver, mesh = sharded_transfer
    pool = receiver.pool
    values = np.arange(pool.buffer.size, dtype=np.float32).reshape(pool.buffer.shape)
    source = jax.device_put(values, pool.sharding)
    rep = NamedSharding(mesh, PartitionSpec())
    running = jnp.zeros((16, 3), dtype=jnp.float32, device=rep)
    mask = jax.device_put(np.ones(16, dtype=np.bool_), rep)
    indices = jax.device_put(np.arange(16, dtype=np.int32), rep)
    with jax.set_mesh(mesh):
        _gather_overlay(running, source, indices, mask, out_sharding=rep).block_until_ready()
        rows = np.arange(16, dtype=np.int32) * 255
        indices = jax.device_put(rows, rep)
        with jax.no_tracing(True):
            result = _gather_overlay(running, source, indices, mask, out_sharding=rep)
        np.testing.assert_array_equal(np.asarray(result), values.reshape(4096, -1)[rows, :3])
