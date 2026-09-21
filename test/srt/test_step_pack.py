"""Per-step batch arrays pack into one host vector and unpack (also under jit)."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.model_executor.step_pack import pack_step_arrays, unpack_step_arrays


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass
class _FB:
    input_ids: object = None
    seq_lens: object = None
    positions: object = None
    cache_loc: object = None
    extend_seq_lens: object = None
    step_packed: object = None
    step_layout: tuple | None = None

    def tree_flatten(self):
        return (self.input_ids, self.seq_lens, self.positions, self.cache_loc, self.step_packed), (
            self.step_layout,
            self.extend_seq_lens,
        )

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(children[0], children[1], children[2], children[3], aux[1], children[4], aux[0])


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass
class _SM:
    temperatures: object = None
    top_ks: object = None
    positions: object = None

    def tree_flatten(self):
        return (self.temperatures, self.top_ks, self.positions), ()

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)


def test_pack_unpack_round_trip_host_and_jit():
    fb = _FB(
        input_ids=np.array([5, 6, 7], np.int32),
        seq_lens=np.array([10, 11, 0], np.int32),
        positions=np.array([9, 10, 0], np.int64),
        cache_loc=np.zeros((8,), np.int32),
        extend_seq_lens=None,
    )
    sm = _SM(
        temperatures=np.array([0.7, 1.0, 0.0], np.float32),
        top_ks=np.array([40, 1, 1], np.int32),
        positions=jnp.array([9, 10, 0], jnp.int32),  # device array: left alone
    )
    pfb, psm = pack_step_arrays(fb, sm)
    assert pfb.step_packed.dtype == np.int32 and pfb.input_ids is None
    assert psm.temperatures is sm.temperatures  # host code reads its shape before the jit
    assert psm.positions is sm.positions and pfb.extend_seq_lens is None
    assert len(pfb.step_layout) == 5

    def check(ufb, usm):
        np.testing.assert_array_equal(np.asarray(ufb.input_ids), fb.input_ids)
        np.testing.assert_array_equal(np.asarray(ufb.seq_lens), fb.seq_lens)
        np.testing.assert_array_equal(np.asarray(ufb.positions), fb.positions)
        assert np.asarray(ufb.positions).dtype == np.int32  # int64 inputs pack to int32
        np.testing.assert_array_equal(np.asarray(ufb.cache_loc), fb.cache_loc)
        np.testing.assert_array_equal(np.asarray(usm.temperatures), sm.temperatures)
        np.testing.assert_array_equal(np.asarray(usm.top_ks), sm.top_ks)
        assert ufb.step_packed is None

    check(*unpack_step_arrays(pfb, psm))
    ufb, usm = jax.jit(unpack_step_arrays)(pfb, psm)
    check(ufb, usm)
