"""Exercise real compiled copies and receive-slot lifetime with a fake transport."""

import threading
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sgl_jax.srt.disaggregation.encoder.raiden_pool import (
    RaidenSendPool,
    compile_packed_pool_copy,
)
from sgl_jax.srt.disaggregation.encoder.raiden_receiver import RaidenReceivePool
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
    values = np.arange(rows * 2 * 3, dtype=np.float32).reshape(rows * 2, 3)
    packed = jax.device_put(values, sharding)
    pool = RaidenSendPool((rows, 3), jnp.float32, sharding, capacity=3)
    counts = (rows, rows)
    contiguous = slots == (0, 1)
    executable = compile_packed_pool_copy(
        jax.ShapeDtypeStruct(packed.shape, packed.dtype, sharding=sharding),
        (rows, 3),
        capacity=3,
        token_counts=counts,
        contiguous=contiguous,
    )
    with jax.no_tracing(True):
        ready = pool.copy_packed_batch_async(
            packed, list(slots), counts, executable, contiguous=contiguous
        )
        jax.block_until_ready(ready)
    actual = np.asarray(pool.buffer).reshape(3, max(2, rows), -1)
    np.testing.assert_array_equal(actual[list(slots), :rows, :3], values.reshape(2, rows, 3))


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
    pool = RaidenReceivePool(
        "localhost", (1, 3), jnp.float32, sharding, parallelism=1, capacity=1, timeout_s=0.01
    )

    def start(request_id):
        return pool.start(request_id, 1, [], [0])

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
        embedding.lease.release_after(jnp.zeros(()))
        completed.clear()
        pool.progress()
        start("third").close()
    finally:
        pool.close()


@pytest.mark.parametrize("reject_second", [False, True])
def test_cancelled_send_waits_for_copy_and_accepted_registration(monkeypatch, reject_second):
    copy_ready = threading.Event()
    failed = []
    transport = SimpleNamespace(
        start=lambda *a, **kw: None,
        register_read=lambda transfer_id, *a: transfer_id != "second",
        poll_stats=lambda: ([], [], failed[:]),
        endpoints=[],
        host_ip="localhost",
    )
    module = "sgl_jax.srt.disaggregation.encoder.raiden_transfer"
    monkeypatch.setattr(f"{module}.require_raiden_preloaded", lambda: None)
    monkeypatch.setattr(f"{module}.RaidenTransferWrapper", lambda *a, **kw: transport)
    monkeypatch.setattr(
        RaidenSendPool,
        "copy_packed_batch_async",
        lambda *a, **kw: (SimpleNamespace(is_ready=copy_ready.is_set),),
    )
    sender = RaidenEncoderServerTransfer("localhost", pool_size=2, timeout_s=0.01)
    try:
        slots = sender.reserve_batch_sync(["first", "second"])
        staged = sender.stage_packed_batch_sync(slots, jnp.zeros((2, 3)), (1, 1))
        if reject_second:
            with pytest.raises(RuntimeError, match="rejected"):
                sender.publish_batch_sync(staged)
        sender.cancel_batch(slots)
        with pytest.raises(TimeoutError):
            sender.reserve_batch_sync(["replacement"])
        copy_ready.set()
        replacement = sender.reserve_batch_sync(["replacement"])
        if reject_second:
            assert replacement[0].slot == slots[1].slot
        sender.cancel_batch(replacement)
        if reject_second:
            with pytest.raises(TimeoutError):
                sender.reserve_batch_sync(["third", "fourth"])
            failed.append("first")
        sender.cancel_batch(sender.reserve_batch_sync(["third", "fourth"]))
    finally:
        sender.close()


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("chunks", [(4,), (2, 1, 1)])
def test_shared_receive_lease_waits_for_last_image_and_chunk(monkeypatch, cached, chunks):
    values = np.arange(8, dtype=np.float32).reshape(4, 2)
    releases = []
    received = PooledEmbedding(
        jnp.asarray(np.stack([np.zeros_like(values), values])),
        1,
        (4, 2),
        (4, 2),
        SimpleNamespace(release_after=releases.append),
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
    monkeypatch.setattr(
        PooledEmbedding, "materialize", lambda _: pytest.fail("copied receive buffer")
    )
    prefix = 0
    for length in chunks:
        info = SimpleNamespace(reqs=[request], prefix_lens=[prefix], extend_lens=[length])
        batch = build_multimodal_batch([info], 1, config, 4)
        output, _ = embed_multimodal_inputs(batch, jnp.zeros(4, jnp.int32), model, pool)
        np.testing.assert_array_equal(output[:length], values[prefix : prefix + length])
        np.testing.assert_array_equal(output[length:], text[length:])
        prefix += length
        assert len(releases) == int(prefix == 4)
    readers = jax.tree_util.tree_leaves(releases[0])
    assert any(reader is output for reader in readers)
    if cached:
        assert any(reader is pool.pages for reader in readers)
