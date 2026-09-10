from dataclasses import dataclass
from queue import Queue
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from sgl_jax.srt.managers import tp_worker_overlap_thread as overlap
from sgl_jax.srt.managers.tp_worker_overlap_thread import ModelWorkerClient


def test_model_worker_client_exposes_page_size_from_wrapped_worker():
    client = object.__new__(ModelWorkerClient)
    client.worker = SimpleNamespace(page_size=128)

    assert client.page_size == 128


def test_model_worker_client_raises_when_wrapped_worker_lacks_page_size():
    client = object.__new__(ModelWorkerClient)
    client.worker = SimpleNamespace()

    with pytest.raises(AttributeError):
        _ = client.page_size


def test_overlap_queues_resource_aware_attention_metadata(monkeypatch):
    """The overlap client must use ModelRunner's backend-specific adapter."""

    @dataclass
    class SamplingInfo:
        sampling_info_done: object = None
        penalizer_orchestrator: object = None

        def update_penalties(self):
            pass

    request_pool, allocator, metadata = object(), object(), object()
    seen = []

    # V4 requires keyword-only resources; a direct backend call must fail.
    def get_forward_metadata(batch, *, request_pool, allocator):
        seen.append((batch, request_pool, allocator))
        return metadata

    backend = SimpleNamespace(get_forward_metadata=get_forward_metadata)
    runner = SimpleNamespace(attn_backend=backend)
    runner.get_attention_metadata = Mock(
        side_effect=lambda batch: backend.get_forward_metadata(
            batch, request_pool=request_pool, allocator=allocator
        )
    )
    client = object.__new__(ModelWorkerClient)
    client.worker = SimpleNamespace(
        model_runner=runner,
        server_args=SimpleNamespace(enable_lora=False),
        get_model_runner=lambda: runner,
    )
    client.mesh = None
    client.future_map_size = 4
    client.input_queue = Queue()
    batch = SimpleNamespace(
        sampling_info=SamplingInfo(),
        seq_lens=np.array([1]),
        req_pool_indices=np.array([0]),
    )
    forward_batch, sampling_metadata = object(), object()
    monkeypatch.setattr(
        overlap.ForwardBatch, "init_new", lambda batch, runner: forward_batch
    )
    monkeypatch.setattr(
        overlap, "future_slot_indices", lambda *args: np.array([1])
    )

    _, future_ids, _ = client.forward_batch_generation(batch, sampling_metadata)

    runner.get_attention_metadata.assert_called_once_with(batch)
    assert seen == [(batch, request_pool, allocator)]
    queued_batch, _, queued_sampling, queued_metadata = client.input_queue.get_nowait()
    assert queued_batch is batch
    assert queued_batch.forward_batch is forward_batch
    assert queued_sampling is sampling_metadata
    assert queued_metadata is metadata
    np.testing.assert_array_equal(future_ids, [-1])
