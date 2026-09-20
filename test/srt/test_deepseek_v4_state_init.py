"""DMA reset of compressor-state slots (``kernels/dsv4/state_init``) vs the XLA scatter.

Covers valid / out-of-range / sentinel / duplicate slots and the padding slot kept by
``capacity``. Runs on CPU in interpret mode.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sgl_jax.srt.kernels.dsv4 import state_init
from sgl_jax.srt.kernels.dsv4.state_init import init_state_slots


@pytest.mark.parametrize("slot_shape", [(128, 2, 64), (8, 256), (2, 8, 128)])
def test_init_state_slots_matches_scatter(slot_shape):
    rng = np.random.default_rng(0)
    capacity = 9
    state = jnp.asarray(rng.standard_normal((capacity + 1, *slot_shape)), jnp.float32)
    # a template with a distinct pattern per element: the kernel must copy it whole
    template = -jnp.arange(np.prod(slot_shape), dtype=jnp.float32).reshape(slot_shape)
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
