"""DMA reset of compressor-state slots (``kernels/dsv4/state_init``) vs the XLA scatter.

Covers the raw kernel (valid / out-of-range / sentinel / duplicate slots, the padding
slot kept by ``capacity``), the CSA ``_reset_state`` switch and the HCA per-layer init
under a one-device data mesh. Runs on CPU in interpret mode.
"""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.kernels.dsv4 import state_init
from sgl_jax.srt.kernels.dsv4.state_init import init_state_slots
from sgl_jax.srt.layers.attention.deepseek_v4_csa_backend import _reset_state
from sgl_jax.srt.mem_cache.deepseek_v4.state import score_slice


@pytest.mark.parametrize("slot_shape", [(128, 2, 64), (8, 256), (2, 8, 128)])
def test_init_state_slots_matches_scatter(slot_shape):
    rng = np.random.default_rng(0)
    capacity = 9
    state = jnp.asarray(rng.standard_normal((capacity + 1, *slot_shape)), jnp.float32)
    template = jnp.zeros(slot_shape, jnp.float32).at[score_slice((1, *slot_shape))].set(-jnp.inf)
    slots = jnp.asarray([3, capacity, -1, 3, 0, capacity + 5, 8, capacity], jnp.int32)
    out = init_state_slots(state, slots, template, capacity=capacity, interpret=True)
    expected = np.asarray(state).copy()
    for slot in (3, 0, 8):
        expected[slot] = np.asarray(template)
    np.testing.assert_array_equal(np.asarray(out), expected)
    # the padding slot ``capacity`` and the out-of-range entries were left alone
    np.testing.assert_array_equal(np.asarray(out)[capacity], np.asarray(state)[capacity])


def test_init_state_slots_no_valid_slot_is_identity():
    state = jnp.arange(4 * 8 * 128, dtype=jnp.float32).reshape(4, 8, 128)
    out = init_state_slots(
        state, jnp.full((16,), 4, jnp.int32), jnp.zeros((8, 128)), interpret=True
    )
    np.testing.assert_array_equal(np.asarray(out), np.asarray(state))
    with pytest.raises(ValueError):
        init_state_slots(state, jnp.zeros((2,), jnp.int32), jnp.zeros((8, 64)), interpret=True)
    with pytest.raises(ValueError):
        init_state_slots(
            state, jnp.zeros((2,), jnp.int32), jnp.zeros((8, 128)), capacity=5, interpret=True
        )


def _csa_case(seed=1):
    rng = np.random.default_rng(seed)
    state = jnp.asarray(rng.standard_normal((6, 8, 512)), jnp.float32)  # 5 slots + padding
    metadata = SimpleNamespace(
        request_slots=jnp.asarray([2, 5, 0, 4, -1, 1], jnp.int32),
        state_init_mask=jnp.asarray([1, 1, 0, 1, 1, 1], bool),
        request_valid_mask=jnp.asarray([1, 1, 1, 1, 1, 0], bool),
    )
    return state, metadata


def test_csa_reset_state_kernel_matches_scatter(monkeypatch):
    state, metadata = _csa_case()
    monkeypatch.setenv(state_init.STATE_INIT_KERNEL_ENV, "0")
    ref = np.asarray(_reset_state(state, metadata))
    monkeypatch.setenv(state_init.STATE_INIT_KERNEL_ENV, "1")
    got = np.asarray(_reset_state(state, metadata))
    np.testing.assert_array_equal(got, ref)
    # slots 2 and 4 reset (mask & valid & in range); 0 (mask off), 5 (padding), -1, 1 (invalid) kept
    for slot in (2, 4):
        assert np.all(got[slot, :, :256] == 0) and np.all(got[slot, :, 256:] == -np.inf)
    for slot in (0, 1, 3, 5):
        np.testing.assert_array_equal(got[slot], np.asarray(state)[slot])


def test_hca_layer_init_under_data_mesh():
    from sgl_jax.srt.layers.attention.hca_backend import _data_spec

    mesh = jax.make_mesh((1, 1), ("data", "tensor"))
    capacity = 4
    rng = np.random.default_rng(2)
    state = jnp.asarray(rng.standard_normal((capacity + 1, 128, 2, 32)), jnp.float32)
    init_slots = jnp.asarray([capacity, 1, capacity, capacity, 3, capacity, capacity, capacity])
    template = jnp.zeros(state.shape[1:], state.dtype).at[score_slice(state.shape)].set(-jnp.inf)
    state = jax.device_put(state, jax.sharding.NamedSharding(mesh, _data_spec(state)))
    init_slots = jax.device_put(init_slots, jax.sharding.NamedSharding(mesh, P("data")))
    template = jax.device_put(template, jax.sharding.NamedSharding(mesh, P(None, None, None)))
    out = jax.shard_map(
        lambda s, i, t: init_state_slots(s, i, t, capacity=capacity, interpret=True),
        mesh=mesh,
        in_specs=(_data_spec(state), P("data"), P(None, None, None)),
        out_specs=_data_spec(state),
        check_vma=False,
    )(state, init_slots.astype(jnp.int32), template)
    out = np.asarray(out)
    for slot in (1, 3):
        assert np.all(out[slot, :, 0, :] == 0) and np.all(out[slot, :, 1, :] == -np.inf)
    for slot in (0, 2, capacity):
        np.testing.assert_array_equal(out[slot], np.asarray(state)[slot])
