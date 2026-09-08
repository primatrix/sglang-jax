"""Exercise real compiled copies and receive-slot lifetime with a fake transport."""

import threading
from types import SimpleNamespace
from unittest.mock import Mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sgl_jax.srt.disaggregation.encoder.embedding_data import EmbeddingData
from sgl_jax.srt.disaggregation.encoder.raiden_pool import RaidenPool
from sgl_jax.srt.disaggregation.encoder.raiden_receiver import RaidenReceiverBackend
from sgl_jax.srt.disaggregation.encoder.raiden_transfer import (
    RaidenEncoderServerTransfer,
)
from sgl_jax.srt.multimodal.common.modality_enum import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sgl_jax.srt.multimodal.in_model.embedding_pool import EmbeddingPool
from sgl_jax.srt.multimodal.in_model.embedding_view import PooledEmbedding
from sgl_jax.srt.multimodal.in_model.host_orchestration import (
    build_multimodal_batch,
    embed_multimodal_inputs,
)


@pytest.mark.parametrize("rows", [1, 3])
@pytest.mark.parametrize("slots", [(0, 1), (0, 2)])
def test_packed_copy_preserves_rows_without_retracing(rows, slots):
    sharding = jax.sharding.NamedSharding(
        jax.sharding.Mesh(np.asarray(jax.devices()), ("device",)), jax.sharding.PartitionSpec()
    )
    counts = (rows, rows + 1)
    values = np.arange(8 * 3, dtype=np.float32).reshape(8, 3)
    packed = jax.device_put(values, sharding)
    pool = RaidenPool((2, 3), jnp.float32, sharding, capacity=4)
    pages = ((0, 1), (2, 3)) if slots == (0, 1) else ((0, 2), (3, 1))
    allocations = tuple(
        page_ids[: pool.pages_needed(count)] for page_ids, count in zip(pages, counts)
    )
    pool.warmup(packed.shape[0])
    with jax.no_tracing(True):
        ready = pool.write_packed(packed, allocations, counts)
        jax.block_until_ready(ready)
    actual = np.asarray(pool.buffer).reshape(4, 2, -1)
    offset = 0
    for page_ids, count in zip(allocations, counts):
        received = actual[list(page_ids)].reshape(-1, actual.shape[-1])
        np.testing.assert_array_equal(received[:count, :3], values[offset : offset + count])
        assert not received[count:].any()
        offset += count


def test_abandoned_receive_reuses_slot_only_after_transfer_finishes(monkeypatch):
    completed = []
    transport = SimpleNamespace(
        start=lambda *a, **kw: None,
        start_read=lambda *a: None,
        poll_stats=lambda: ([], completed[:], []),
    )
    monkeypatch.setattr(
        "sgl_jax.srt.disaggregation.encoder.raiden_receiver.RaidenTransferWrapper",
        lambda *a, **kw: transport,
    )
    sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    pool = RaidenReceiverBackend(
        "localhost",
        RaidenPool((2, 3), jnp.float32, sharding, capacity=1),
        parallelism=1,
        pool_size=1,
        transfer_timeout_s=0.01,
    )

    def start(request_id):
        return pool._start(
            EmbeddingData(
                request_id,
                1,
                0,
                Modality.IMAGE,
                shape=(1, 3),
                dtype="float32",
                transfer={
                    "transfer_id": request_id,
                    "transfer_uuid": 1,
                    "transfer_block_ids": [0],
                    "transfer_page_size": 2,
                    "transfer_host": "192.0.2.1",
                    "transfer_address": [{"endpoint": "192.0.2.1:1234", "shards": []}],
                },
            )
        )

    try:
        session = start("first")
        session.close()
        with pytest.raises(TimeoutError):
            start("second")
        completed.append("first")
        pool.progress()
        completed.clear()
        second = start("second")
        completed.append("second")
        embedding = second.poll()
        second.close()
        with pytest.raises(TimeoutError):
            start("third")
        earlier_read = Mock(spec=jax.Array, is_ready=Mock(return_value=False))
        embedding.record_read(earlier_read)
        embedding.record_read(jnp.zeros(()))
        embedding.release()
        with pytest.raises(TimeoutError):
            start("third")
        earlier_read.is_ready.return_value = True
        completed.clear()
        pool.progress()
        start("third").close()
    finally:
        pool.close()


@pytest.mark.parametrize("reject_second", [False, True])
def test_cancelled_send_waits_for_copy_and_accepted_registration(monkeypatch, reject_second):
    copy_ready = threading.Event()
    sent = []

    def register(transfer_id, *args):
        assert copy_ready.is_set()
        if transfer_id == "second":
            raise RuntimeError("registration failed")
        return True

    transport = SimpleNamespace(
        start=lambda *a, **kw: None,
        register_read=register,
        poll_stats=lambda: (sent[:], [], []),
        endpoints=[],
        host_ip="localhost",
    )
    module = "sgl_jax.srt.disaggregation.encoder.raiden_transfer"
    monkeypatch.setattr(f"{module}.require_raiden_preloaded", lambda: None)
    monkeypatch.setattr(f"{module}.RaidenTransferWrapper", lambda *a, **kw: transport)
    monkeypatch.setattr(
        RaidenPool,
        "write_packed",
        lambda *a, **kw: SimpleNamespace(
            is_ready=copy_ready.is_set, block_until_ready=lambda: copy_ready.wait(1)
        ),
    )
    pool = RaidenPool(
        (2, 3), jnp.float32, jax.sharding.SingleDeviceSharding(jax.devices()[0]), capacity=2
    )
    sender = RaidenEncoderServerTransfer("localhost", pool, pool_size=2, timeout_s=0.01)
    try:
        slots = sender.reserve_batch_sync(["first", "second"], (1, 1))
        staged = sender.stage_packed_batch_sync(slots, jnp.zeros((2, 3)))
        if reject_second:
            copy_ready.set()
            with pytest.raises(RuntimeError, match="registration failed"):
                sender.publish_batch_sync(staged)
        sender.cancel_batch(slots)
        if not reject_second:
            with pytest.raises(TimeoutError):
                sender.reserve_batch_sync(["replacement"], (1,))
        copy_ready.set()
        replacement = sender.reserve_batch_sync(["replacement"], (1,))
        if reject_second:
            assert replacement[0].page_ids == slots[1].page_ids
        sender.cancel_batch(replacement)
        if reject_second:
            with pytest.raises(TimeoutError):
                sender.reserve_batch_sync(["third", "fourth"], (1, 1))
            sent.append("first")
        sender.cancel_batch(sender.reserve_batch_sync(["third", "fourth"], (1, 1)))
    finally:
        sender.close()


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("chunks", [(4,), (2, 1, 1)])
def test_shared_receive_lease_keeps_pages_until_request_finishes(cached, chunks):
    values = np.arange(8, dtype=np.float32).reshape(4, 2)
    releases = []
    readers = []
    received = PooledEmbedding(
        jnp.asarray(np.stack([np.zeros_like(values[:2]), values[2:], values[:2]])),
        np.asarray([4, 5, 2, 3]),
        2,
        (SimpleNamespace(record_read=readers.append, release=lambda: releases.append(True)),),
    )
    items = [
        MultimodalDataItem(
            Modality.IMAGE,
            hash=start,
            placeholder_ranges=[(start, start + 2)],
            precomputed_embeddings=received[start : start + 2],
        )
        for start in (0, 2)
    ]
    request = SimpleNamespace(mm_inputs=MultimodalInputs(mm_items=items))
    config = SimpleNamespace(
        hf_config=SimpleNamespace(architectures=["Qwen2_5_VLForConditionalGeneration"])
    )
    text = jnp.full((4, 2), -1, dtype=jnp.float32)
    model = SimpleNamespace(
        mesh=None,
        deepstack_visual_layers=0,
        get_input_embeddings=lambda: lambda _: text,
        get_multimodal_encode_funcs=lambda: {},
    )
    pool = EmbeddingPool(1, 2, 2, jnp.float32) if cached else None
    prefix = 0
    for length in chunks:
        info = SimpleNamespace(reqs=[request], prefix_lens=[prefix], extend_lens=[length])
        batch = build_multimodal_batch([info], 1, config, 4)
        output, _, _ = embed_multimodal_inputs(batch, jnp.zeros(4, jnp.int32), model, pool)
        np.testing.assert_array_equal(output[:length], values[prefix : prefix + length])
        np.testing.assert_array_equal(output[length:], text[length:])
        prefix += length
        assert not releases
    received.release()
    assert len(releases) == 1
    assert any(reader is output for reader in readers)
    if cached:
        assert not pool._entries  # Received embeddings never enter a second cache.
