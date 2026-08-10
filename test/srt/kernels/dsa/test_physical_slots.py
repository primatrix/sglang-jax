import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.kernels.dsa.physical_slots import (
    logical_topk_to_physical_slots_pallas,
)


def _reference(topk, seq_lens, page_indices, cu_q_lens, cu_kv_lens, page_size):
    num_tokens = topk.shape[0]
    token_ids = jnp.arange(num_tokens, dtype=jnp.int32)
    seq_ids = jnp.searchsorted(cu_q_lens[1:], token_ids, side="right")
    seq_ids = jnp.clip(seq_ids, 0, seq_lens.shape[0] - 1)
    logical = jnp.maximum(topk, 0)
    page_ptr = cu_kv_lens[seq_ids, None] // page_size + logical // page_size
    ptr_in_bounds = (page_ptr >= 0) & (page_ptr < page_indices.shape[0])
    physical_pages = page_indices[jnp.clip(page_ptr, 0, page_indices.shape[0] - 1)]
    valid = (
        (token_ids < cu_q_lens[-1])[:, None]
        & (topk >= 0)
        & (logical < seq_lens[seq_ids, None])
        & ptr_in_bounds
        & (physical_pages >= 0)
    )
    slots = jnp.where(valid, physical_pages * page_size + logical % page_size, 0)
    return slots.astype(jnp.int32), jnp.sum(valid, axis=1, dtype=jnp.int32)


def test_pallas_mapper_matches_ragged_reference_in_interpret_mode():
    topk = jnp.asarray(
        [
            [0, 5, 6, -1, 99],
            [3, 4, -1, -1, -1],
            [0, 2, -1, -1, -1],
            [1, 3, -1, -1, -1],
            [0, 0, 0, 0, 0],  # padding query row
        ],
        jnp.int32,
    )
    seq_lens = jnp.asarray([7, 3], jnp.int32)
    cu_q_lens = jnp.asarray([0, 2, 4], jnp.int32)
    cu_kv_lens = jnp.asarray([0, 8, 12], jnp.int32)
    page_indices = jnp.asarray([5, 9, 3, -1], jnp.int32)

    expected = _reference(
        topk,
        seq_lens,
        page_indices,
        cu_q_lens,
        cu_kv_lens,
        page_size=4,
    )
    actual = jax.jit(
        lambda *args: logical_topk_to_physical_slots_pallas(
            *args,
            page_size=4,
            block_q=4,
            interpret=True,
        )
    )(topk, seq_lens, page_indices, cu_q_lens, cu_kv_lens)

    np.testing.assert_array_equal(np.asarray(actual[0]), np.asarray(expected[0]))
    np.testing.assert_array_equal(np.asarray(actual[1]), np.asarray(expected[1]))
