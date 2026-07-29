from types import SimpleNamespace

import jax
import numpy as np
import pytest
from jax.sharding import Mesh

from sgl_jax.srt.managers.tp_worker_overlap_thread import (
    ModelWorkerClient,
    future_token_ids_from_req_pool_indices,
)
from sgl_jax.srt.managers.utils import resolve_future_token_ids, set_future_token_ids


def test_model_worker_client_exposes_page_size_from_wrapped_worker():
    client = object.__new__(ModelWorkerClient)
    client.worker = SimpleNamespace(page_size=128)

    assert client.page_size == 128


def test_model_worker_client_raises_when_wrapped_worker_lacks_page_size():
    client = object.__new__(ModelWorkerClient)
    client.worker = SimpleNamespace()

    with pytest.raises(AttributeError):
        _ = client.page_size


def test_future_token_placeholders_are_owned_by_request_pool_slot():
    req_pool_indices = np.array([8, -1, 2, -1], dtype=np.int32)

    assert future_token_ids_from_req_pool_indices(req_pool_indices).tolist() == [
        -9,
        0,
        -3,
        0,
    ]


def test_request_owned_relay_survives_many_padded_prefill_waves():
    mesh = Mesh(
        np.array(jax.devices()[:1]),
        ("data",),
        axis_types=(jax.sharding.AxisType.Explicit,),
    )
    relay = jax.numpy.zeros(65, dtype=jax.numpy.int32)
    request_slots = np.array([0, 63, *range(2, 32)], dtype=np.int32)
    request_waves = [
        request_slots[:1],
        request_slots[1:2],
        request_slots[2:14],
        request_slots[14:24],
        request_slots[24:],
    ]

    for request_indices in request_waves:
        padded_indices = np.full(32, -1, dtype=np.int32)
        padded_tokens = np.zeros(32, dtype=np.int32)
        padded_indices[: len(request_indices)] = request_indices
        padded_tokens[: len(request_indices)] = np.array(request_indices) + 1000
        relay = set_future_token_ids(relay, padded_indices, padded_tokens, mesh)

    placeholders = future_token_ids_from_req_pool_indices(request_slots)
    resolved = resolve_future_token_ids(placeholders, relay, mesh)

    assert np.asarray(resolved).tolist() == (request_slots + 1000).tolist()
