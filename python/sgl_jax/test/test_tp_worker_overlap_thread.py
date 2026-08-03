import queue
from types import SimpleNamespace

import jax
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from sgl_jax.srt.managers import tp_worker_overlap_thread
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


@pytest.mark.parametrize("materialize", [False, True])
def test_resolve_last_batch_result_optionally_materializes_next_token_ids(monkeypatch, materialize):
    client = object.__new__(ModelWorkerClient)
    client.output_queue = queue.Queue()
    logits_output = SimpleNamespace(
        next_token_logprobs=None,
        input_token_logprobs=None,
        hidden_states=None,
    )
    device_token_ids = object()
    client.output_queue.put((None, logits_output, device_token_ids, 0))

    device_get_calls = []

    def fake_device_get(value):
        device_get_calls.append(value)
        return np.array([7], dtype=np.int32)

    monkeypatch.setattr(tp_worker_overlap_thread.jax, "device_get", fake_device_get)

    _, next_token_ids, _ = client.resolve_last_batch_result(materialize_next_token_ids=materialize)

    assert device_get_calls == ([device_token_ids] if materialize else [])
    assert next_token_ids == ([7] if materialize else None)


def test_resolve_keeps_prefill_logprobs_when_token_ids_stay_on_device(monkeypatch):
    client = object.__new__(ModelWorkerClient)
    client.output_queue = queue.Queue()
    input_token_logprobs = object()
    logits_output = SimpleNamespace(
        next_token_logprobs=None,
        input_token_logprobs=input_token_logprobs,
        hidden_states=None,
    )
    client.output_queue.put((None, logits_output, object(), 0))

    copy_calls = []

    def fake_copy_to_host_async(value):
        copy_calls.append(value)
        return np.array([-1.25], dtype=np.float32)

    monkeypatch.setattr(
        tp_worker_overlap_thread.jax,
        "copy_to_host_async",
        fake_copy_to_host_async,
    )
    monkeypatch.setattr(
        tp_worker_overlap_thread.jax,
        "device_get",
        lambda _: pytest.fail("next_token_ids should not be materialized"),
    )

    resolved_logits, next_token_ids, _ = client.resolve_last_batch_result(
        materialize_next_token_ids=False
    )

    assert copy_calls == [input_token_logprobs]
    assert resolved_logits.input_token_logprobs == [-1.25]
    assert next_token_ids is None


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


def test_overlap_precompile_warms_replicated_relay_path(monkeypatch):
    mesh = Mesh(
        np.array(jax.devices()[:1]),
        ("data",),
        axis_types=(jax.sharding.AxisType.Explicit,),
    )
    client = object.__new__(ModelWorkerClient)
    worker_calls = []
    client.worker = SimpleNamespace(
        run_precompile=lambda relay: worker_calls.append(relay),
        get_precompile_paddings=lambda: ([64], [1, 4], [128, 512]),
    )
    client.mesh = mesh
    client.future_token_ids_map = jax.device_put(
        jax.numpy.zeros(65, dtype=jax.numpy.int32),
        NamedSharding(mesh, PartitionSpec(None)),
    )

    gather_shapes = []
    replicated = NamedSharding(mesh, PartitionSpec())

    def gather(value):
        gather_shapes.append(value.shape)
        return jax.device_put(value, replicated)

    relay_shapes = []

    def update_relay(relay, req_pool_indices, next_token_ids, update_mesh):
        relay_shapes.append(
            (
                req_pool_indices.shape,
                next_token_ids.shape,
                next_token_ids.sharding,
                update_mesh,
            )
        )
        return relay

    client.async_gather_fn = gather
    monkeypatch.setattr(tp_worker_overlap_thread, "set_future_token_ids", update_relay)

    client.run_precompile()

    assert len(worker_calls) == 1
    assert worker_calls[0] is client.future_token_ids_map
    assert gather_shapes == [(1,), (4,)]
    assert [(req_shape, token_shape) for req_shape, token_shape, _, _ in relay_shapes] == [
        ((1,), (1,)),
        ((4,), (4,)),
    ]
    assert all(sharding == replicated for _, _, sharding, _ in relay_shapes)
    assert all(update_mesh is mesh for _, _, _, update_mesh in relay_shapes)
