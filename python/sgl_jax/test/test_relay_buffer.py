import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from functools import partial
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.managers.schedule_batch import ScheduleBatch
from sgl_jax.srt.utils.overlap_utils import (
    DecodeWorkspace,
    create_relay_buffers,
    gather_decode_batch_inputs,
    resolve_decode_inputs,
    resolve_relay_inputs,
    update_decode_result,
    update_relay_buffers,
)


def test_decode_page_indices_match_packed_cache_layout():
    req_to_token = np.zeros((3, 16), dtype=np.int32)
    req_to_token[0, :8] = np.r_[40:44, 80:84]
    req_to_token[1, :4] = np.r_[120:124]
    req_to_token[2, :4] = np.r_[160:164]
    batch = SimpleNamespace(
        dp_size=2,
        req_to_token_pool=SimpleNamespace(req_to_token=req_to_token),
        reqs_info=[
            SimpleNamespace(
                seq_lens=np.array([5], dtype=np.int32),
                req_pool_indices=np.array([0], dtype=np.int32),
            ),
            SimpleNamespace(
                seq_lens=np.array([3, 4], dtype=np.int32),
                req_pool_indices=np.array([1, 2], dtype=np.int32),
            ),
        ],
    )

    actual = ScheduleBatch._merge_decode_page_indices(
        batch,
        bs_paddings=[4],
        cache_loc_paddings=[32],
        page_size=4,
        per_dp_bs_size=2,
    )

    np.testing.assert_array_equal(actual, [10, 20, 0, 0, 30, 40, 0, 0])


def test_relay_buffer_updates_valid_request_slots():
    mesh = Mesh(
        np.asarray(jax.devices()[:1]).reshape((1, 1)),
        ("data", "tensor"),
        axis_types=(jax.sharding.AxisType.Explicit,) * 2,
    )
    pool = SimpleNamespace(req_to_token=np.zeros((8, 16), dtype=np.int32))
    sharding = NamedSharding(mesh, P("data"))
    relay_sharding = NamedSharding(mesh, P("data", None))
    indices = jax.device_put(jnp.array([2, 5, 0, 0]), sharding)
    valid = jax.device_put(jnp.array([True, True, False, False]), sharding)
    tokens = jax.device_put(jnp.array([11, 22, 99, 99]), sharding)
    input_ids = jax.device_put(jnp.array([1, 2, 3, 4]), sharding)

    buffers = create_relay_buffers(mesh, pool, dp_size=1)
    buffers = jax.jit(
        lambda b, i, m, t: update_relay_buffers(
            b,
            i,
            m,
            t,
            dp_size=1,
            output_sharding=relay_sharding,
        )
    )(buffers, indices, valid, tokens)
    actual = jax.jit(
        lambda b, i, m, x: resolve_relay_inputs(
            b,
            i,
            m,
            x,
            dp_size=1,
            relay_sharding=relay_sharding,
            output_sharding=sharding,
        ),
        out_shardings=sharding,
    )(buffers, indices, valid, input_ids)

    np.testing.assert_array_equal(np.asarray(actual), [11, 22, 3, 4])


def test_decode_relay_uses_request_indices_and_ignores_padding():
    mesh = Mesh(
        np.asarray(jax.devices()[:1]).reshape((1, 1)),
        ("data", "tensor"),
        axis_types=(jax.sharding.AxisType.Explicit,) * 2,
    )
    pool = SimpleNamespace(req_to_token=np.zeros((8, 16), dtype=np.int32))
    sharding = NamedSharding(mesh, P("data"))
    relay_sharding = NamedSharding(mesh, P("data", None))
    indices = jax.device_put(jnp.array([2, 5, -1, -1]), sharding)
    tokens = jax.device_put(jnp.array([11, 22, 99, 99]), sharding)
    input_ids = jax.device_put(jnp.array([1, 2, 0, 0]), sharding)
    seq_lens = jax.device_put(jnp.array([8, 13, 0, 0]), sharding)

    buffers = create_relay_buffers(mesh, pool, dp_size=1)
    buffers = jax.jit(
        lambda b, i, t: update_relay_buffers(
            b,
            jnp.where(i >= 0, i, 0),
            i >= 0,
            t,
            dp_size=1,
            output_sharding=relay_sharding,
        )
    )(buffers, indices, tokens)
    actual_ids, actual_positions = jax.jit(
        lambda b, i, x, s: resolve_decode_inputs(
            b,
            i,
            x,
            s,
            dp_size=1,
            relay_sharding=relay_sharding,
            output_sharding=sharding,
        ),
        out_shardings=(sharding, sharding),
    )(buffers, indices, input_ids, seq_lens)

    np.testing.assert_array_equal(np.asarray(actual_ids), [11, 22, 0, 0])
    np.testing.assert_array_equal(np.asarray(actual_positions), [7, 12, 0, 0])


def test_decode_workspace_persists_request_state_and_reuses_descriptor():
    mesh = Mesh(
        np.asarray(jax.devices()[:1]).reshape((1, 1)),
        ("data", "tensor"),
        axis_types=(jax.sharding.AxisType.Explicit,) * 2,
    )
    pool = SimpleNamespace(req_to_token=np.zeros((8, 16), dtype=np.int32))
    workspace = DecodeWorkspace(mesh, pool, dp_size=1)
    indices = np.array([2, 5, -1, -1], dtype=np.int32)
    page_indices = np.array([3, 4, 9, 0], dtype=np.int32)
    descriptor = workspace.get_descriptor(indices, page_indices)

    workspace.publish_request_state(
        descriptor.req_pool_indices,
        jnp.array([11, 22, 99, 99], dtype=jnp.int32),
        jnp.array([8, 13, 1, 1], dtype=jnp.int32),
        jnp.array([[0.5], [0.7], [1.0], [1.0]], dtype=jnp.float32),
        jnp.array([0.8, 0.9, 1.0, 1.0], dtype=jnp.float32),
        jnp.array([20, 30, 1, 1], dtype=jnp.int32),
        jnp.array([0.1, 0.2, 0.0, 0.0], dtype=jnp.float32),
        jnp.array([3, 4, 0, 0], dtype=jnp.int32),
    )
    workspace.mark_initialized(indices, [2], 4)
    gather = jax.jit(
        partial(
            gather_decode_batch_inputs,
            dp_size=1,
            page_size=1,
            relay_sharding=workspace.relay_sharding,
            state_sharding=workspace.state_sharding,
            output_sharding=workspace.input_sharding,
        )
    )
    actual = gather(workspace.request_state, descriptor)

    np.testing.assert_array_equal(np.asarray(actual.input_ids), [11, 22, 0, 0])
    np.testing.assert_array_equal(np.asarray(actual.seq_lens), [8, 13, 0, 0])
    np.testing.assert_array_equal(np.asarray(actual.positions), [7, 12, 0, 0])
    np.testing.assert_array_equal(np.asarray(actual.distribution), [2, 2, 2])
    np.testing.assert_allclose(np.asarray(actual.temperatures), [[0.5], [0.7], [1.0], [1.0]])
    assert workspace.contains_request_slots([np.array([2, 5], dtype=np.int32)])
    assert workspace.get_descriptor(indices, page_indices) is descriptor
    assert workspace.get_descriptor(indices, page_indices + 1) is not descriptor

    workspace.request_state = jax.jit(
        partial(
            update_decode_result,
            dp_size=1,
            relay_sharding=workspace.relay_sharding,
        )
    )(
        workspace.request_state,
        descriptor.req_pool_indices,
        jnp.array([33, 44, 0, 0], dtype=jnp.int32),
        jnp.array([8, 13, 0, 0], dtype=jnp.int32),
    )
    updated = gather(workspace.request_state, descriptor)
    np.testing.assert_array_equal(np.asarray(updated.input_ids), [33, 44, 0, 0])
    np.testing.assert_array_equal(np.asarray(updated.seq_lens), [9, 14, 0, 0])
    np.testing.assert_allclose(
        np.asarray(updated.temperatures),
        [[0.5], [0.7], [1.0], [1.0]],
    )
