from __future__ import annotations

import asyncio
import time

import jax
import jax.numpy as jnp
import pytest

from sgl_jax.srt.disaggregation.encoder.embedding_data import EmbeddingData
from sgl_jax.srt.disaggregation.encoder.sim_transfer import (
    SimEncoderServerTransfer,
    SimReceiverBackend,
)
from sgl_jax.srt.disaggregation.encoder.transfer_layout import (
    encoder_pool_block_shape,
    encoder_transfer_nbytes,
)
from sgl_jax.srt.multimodal.common.modality_enum import Modality


def _metadata(transfer_id: str, shape: tuple[int, int] = (4, 3584)) -> EmbeddingData:
    return EmbeddingData(
        req_id=transfer_id,
        num_parts=1,
        part_idx=0,
        grid_dim=None,
        modality=Modality.IMAGE,
        embedding_shape=shape,
        dtype="bfloat16",
        transfer_id=transfer_id,
    )


async def _publish(transfer, transfer_id, embedding):
    reservations = await asyncio.to_thread(
        transfer.reserve_batch_sync,
        [transfer_id],
    )
    staged = transfer.stage_packed_batch_sync(reservations, embedding, (embedding.shape[0],))
    return (await asyncio.to_thread(transfer.publish_batch_sync, staged))[0]


def test_sim_transfer_uses_raiden_padded_payload_size():
    assert encoder_pool_block_shape((4, 3584)) == (4, 4, 8, 128)
    assert encoder_transfer_nbytes((4, 3584), jnp.bfloat16) == 4 * 4096 * 2


def test_sim_sender_backpressures_when_pool_is_full():
    transfer = SimEncoderServerTransfer(pool_size=1, rtt_ms=100)

    async def run() -> None:
        embedding = jnp.zeros((4, 3584), dtype=jnp.bfloat16)
        await _publish(transfer, "part-0:embedding", embedding)
        blocked = asyncio.create_task(_publish(transfer, "part-1:embedding", embedding))
        await asyncio.sleep(0.01)
        assert not blocked.done()
        await asyncio.wait_for(blocked, 1)

    asyncio.run(run())
    transfer.close()


def test_sim_sender_serializes_transfers_per_channel():
    transfer = SimEncoderServerTransfer(pool_size=2, parallelism=1, rtt_ms=20)

    async def run() -> None:
        embedding = jnp.zeros((4, 3584), dtype=jnp.bfloat16)
        await _publish(transfer, "part-0:embedding", embedding)
        await _publish(transfer, "part-1:embedding", embedding)

    asyncio.run(run())
    ready_times = [ready_ns for _, ready_ns in transfer._pool._active.values()]
    assert ready_times[1] - ready_times[0] >= 19_000_000
    transfer.close()


def test_sim_sender_rejects_embedding_that_does_not_match_single_pool():
    transfer = SimEncoderServerTransfer()
    asyncio.run(
        _publish(
            transfer,
            "part-0:embedding",
            jnp.zeros((4, 3584), dtype=jnp.bfloat16),
        )
    )

    with pytest.raises(ValueError, match="pool embedding mismatch"):
        asyncio.run(
            _publish(
                transfer,
                "part-1:embedding",
                jnp.zeros((8, 3584), dtype=jnp.bfloat16),
            )
        )

    transfer.close()


def test_sim_sender_reports_inflight_completion(caplog):
    transfer = SimEncoderServerTransfer(
        pool_size=1,
        rtt_ms=5,
        poll_interval_s=0.0001,
        log_inflight=True,
    )

    async def run() -> None:
        embedding = jnp.zeros((4, 3584), dtype=jnp.bfloat16)
        await _publish(transfer, "part-0:embedding", embedding)
        reservations = await asyncio.to_thread(
            transfer.reserve_batch_sync,
            ["part-1:embedding"],
        )
        transfer.cancel_batch(reservations)
        assert not transfer._active

    caplog.set_level("INFO")
    asyncio.run(run())
    transfer.close()

    assert "event=start transfer_id=part-0:embedding" in caplog.text
    assert "event=sent transfer_id=part-0:embedding" in caplog.text


def test_sim_receiver_backpressures_and_reuses_buffer():
    backend = SimReceiverBackend(
        jax.sharding.SingleDeviceSharding(jax.local_devices()[0]),
        ms_per_mb=0,
        pool_size=1,
        transfer_timeout_s=1,
    )
    first = backend.start(_metadata("part-0:embedding"))._future.result(timeout=1)
    second_future = backend.start(_metadata("part-1:embedding"))._future
    time.sleep(0.02)
    assert not second_future.done()

    first_buffer = first.poll()
    assert first_buffer is not None
    second = second_future.result(timeout=1)
    second_buffer = second.poll()
    assert second_buffer is not None
    assert first_buffer.unsafe_buffer_pointer() == second_buffer.unsafe_buffer_pointer()

    backend.close()


def test_sim_receiver_rejects_embedding_that_does_not_match_single_pool():
    backend = SimReceiverBackend(
        jax.sharding.SingleDeviceSharding(jax.local_devices()[0]),
        ms_per_mb=0,
    )
    first = backend.start(_metadata("part-0:embedding"))._future.result(timeout=1)

    with pytest.raises(ValueError, match="pool embedding mismatch"):
        backend.start(_metadata("part-1:embedding", shape=(8, 3584)))._future.result(timeout=1)

    first.close()
    backend.close()
