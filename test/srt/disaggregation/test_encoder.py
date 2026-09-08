"""Boundary contracts for encoder batching and multipart reconstruction."""

import asyncio
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from sgl_jax.srt.disaggregation.encoder.embedding_data import (
    EmbeddingData,
    MultiModalEmbeddingData,
)
from sgl_jax.srt.disaggregation.encoder.runtime import EncoderRuntime
from sgl_jax.srt.disaggregation.encoder.scheduler import DisaggEncoderScheduler
from sgl_jax.srt.multimodal.common.modality_enum import Modality
from sgl_jax.srt.multimodal.in_model.embedding_view import PooledEmbedding


class Encoder:
    preprocess_concurrency = 4

    def __init__(self):
        self.batches = []

    async def preprocess_request(self, request):
        return SimpleNamespace(**request, token_count=request["tokens"])

    def batch_key(self, request):
        return request.tokens

    def build_batch(self, requests):
        self.batches.append([request.req_id for request in requests])
        return SimpleNamespace(token_counts=tuple(request.tokens for request in requests))

    def encode_packed(self, batch):
        return SimpleNamespace(batch=batch, packed=np.zeros((sum(batch.token_counts), 2)))

    def metadata_for_packed(self, output):
        return [{} for _ in output.batch.token_counts]


class Transfer:
    def __init__(self, fail=None):
        self.fail = fail
        self.reserved = set()
        self.closed = False
        self.publishing = threading.Event()
        self.allow_publish = threading.Event()
        self.allow_publish.set()

    def batch_capacity(self, token_count):
        return 8

    def reserve_batch_sync(self, ids, token_counts):
        self.reserved.update(ids)
        return ids

    def stage_packed_batch_sync(self, reservations, packed):
        if self.fail == "copy":
            raise RuntimeError("copy failed")
        return reservations

    def publish_batch_sync(self, staged):
        self.publishing.set()
        if not self.allow_publish.wait(2):
            raise TimeoutError("test did not release transfer")
        if self.fail == "publish":
            raise RuntimeError("publish failed")
        return [{"transfer_id": key} for key in staged]

    def release(self, key):
        self.reserved.discard(key)

    def cancel_batch(self, reservations):
        self.reserved.difference_update(reservations)

    def close(self):
        self.closed = True


def request(index, tokens=1):
    return {"req_id": str(index), "modality": "IMAGE", "tokens": tokens}


@pytest.mark.parametrize("failure", [None, "copy", "publish"])
@pytest.mark.parametrize("sampled", [False, True])
def test_mixed_batches_drain_and_release_on_failure(monkeypatch, failure, sampled):
    if not sampled:
        monkeypatch.setattr("time.time_ns", lambda: pytest.fail("unsampled request read the clock"))

    async def run():
        encoder, transfer = Encoder(), Transfer(failure)
        runtime = EncoderRuntime(encoder, transfer, max_batch_size=2, batch_coalesce_ms=2)
        futures = []
        # Queue before starting the worker so grouping is deterministic.
        runtime._started = True
        for index, tokens in enumerate([1, 2, 1]):
            future = asyncio.get_running_loop().create_future()
            futures.append(future)
            payload = request(index, tokens)
            payload["request_time_stats"] = {} if sampled and index == 0 else None
            prepared = await runtime.preprocess_request(payload)
            await runtime.enqueue_preprocessed(prepared, future.set_result)
        runtime._started = False
        runtime.start()
        try:
            results = await asyncio.wait_for(asyncio.gather(*futures), 2)
            assert encoder.batches == [["0", "2"], ["1"]]
            if failure:
                assert all(isinstance(result, RuntimeError) for result in results)
                assert not transfer.reserved
            else:
                assert [result.req_id for result in results] == ["0", "1", "2"]
                assert [result.shape for result in results] == [(1, 2), (2, 2), (1, 2)]
                assert all(result.timing is None for result in results[1:])
                assert (results[0].timing is not None) == sampled
                if sampled:
                    timing = results[0].timing
                    assert (
                        timing["transfer_enqueue_ns"]
                        <= timing["transfer_start_ns"]
                        <= timing["publish_done_ns"]
                    )
        finally:
            await runtime.stop()
        assert transfer.closed

    asyncio.run(run())


def test_timed_out_request_releases_late_transfer():
    async def run():
        transfer = Transfer()
        transfer.allow_publish.clear()
        runtime = EncoderRuntime(Encoder(), transfer)
        scheduler = DisaggEncoderScheduler(runtime, request_timeout=0.05)
        scheduler.start()
        try:
            with pytest.raises(TimeoutError):
                await scheduler.submit(request(0))
            assert transfer.publishing.is_set()
        finally:
            transfer.allow_publish.set()
            await scheduler.stop()
            await runtime.stop()
        assert not transfer.reserved

    asyncio.run(run())


def test_out_of_order_parts_preserve_order_and_reject_duplicates():
    parts = MultiModalEmbeddingData(2)
    arrays = [np.full((1, 2), index) for index in range(2)]
    for index in (1, 0):
        data = EmbeddingData(str(index), 2, index, Modality.IMAGE, grid_dim=[[1, 2, 2]])
        parts.add(data, arrays[index])
        with pytest.raises(ValueError, match="duplicate"):
            parts.add(data, arrays[index])
    assert parts.ready
    assert all(actual is expected for actual, expected in zip(parts.get_embedding(), arrays))
    np.testing.assert_array_equal(parts.get_mm_extra_meta()["image_grid_thw"], [[1, 2, 2]] * 2)
    with pytest.raises(ValueError):
        MultiModalEmbeddingData(0)
    view = PooledEmbedding(np.zeros((2, 4, 3)), np.arange(4, 8), 3, (object(),))
    assert view[3:1].shape == (0, 3)
    np.testing.assert_array_equal(view[-2:].row_indices, [6, 7])


@pytest.mark.parametrize("claimed", [False, True])
def test_cancelled_completed_request_releases_only_unclaimed_embeddings(claimed):
    from unittest.mock import Mock

    from sgl_jax.srt.disaggregation.encoder.client import PendingEncoderRequest

    lease = Mock()
    parts = MultiModalEmbeddingData(1)
    parts.add(
        EmbeddingData("request", 1, 0, Modality.IMAGE),
        PooledEmbedding(np.zeros((1, 2, 3)), np.arange(2), 3, (lease,)),
    )
    pending = PendingEncoderRequest(SimpleNamespace(), 0, Mock(), (), (), parts, Mock(), Mock())
    pending._result = {}
    pending._done.set()
    if claimed:
        pending.poll()
    pending.close()
    pending.close()
    assert lease.release.call_count == int(not claimed)
