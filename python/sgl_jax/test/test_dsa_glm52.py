import jax
import jax.numpy as jnp
import numpy as np
import pytest


def test_dsa_selection_pytree_keeps_array_children_and_static_producer_layer():
    from sgl_jax.srt.layers.attention.dsa_types import DsaSelection

    physical_slots = jnp.array([[0, 8], [13, 21]], dtype=jnp.int32)
    selected_counts = jnp.array([1, 2], dtype=jnp.int32)
    logical_topk_ids = jnp.array([[0, 3], [5, 7]], dtype=jnp.int32)
    selection = DsaSelection(
        physical_slots=physical_slots,
        selected_counts=selected_counts,
        producer_layer=4,
        logical_topk_ids=logical_topk_ids,
    )

    leaves, tree_def = jax.tree_util.tree_flatten(selection)
    restored = jax.tree_util.tree_unflatten(tree_def, leaves)

    assert len(leaves) == 3
    np.testing.assert_array_equal(restored.physical_slots, physical_slots)
    np.testing.assert_array_equal(restored.selected_counts, selected_counts)
    np.testing.assert_array_equal(restored.logical_topk_ids, logical_topk_ids)
    assert restored.producer_layer == 4


def test_dsa_selection_validity_comes_only_from_selected_counts():
    from sgl_jax.srt.layers.attention.dsa_types import DsaSelection

    selection = DsaSelection(
        physical_slots=jnp.array([[0, 17], [9, 0]], dtype=jnp.int32),
        selected_counts=jnp.array([1, 1], dtype=jnp.int32),
        producer_layer=0,
    )

    np.testing.assert_array_equal(
        selection.valid_mask(),
        np.array([[True, False], [True, False]]),
    )


def test_dsa_selection_validation_accepts_decode_and_prefill_shapes():
    from sgl_jax.srt.layers.attention.dsa_types import DsaSelection

    decode = DsaSelection(
        physical_slots=jnp.zeros((2, 3), dtype=jnp.int32),
        selected_counts=jnp.array([3, 1], dtype=jnp.int32),
        producer_layer=1,
    )
    prefill = DsaSelection(
        physical_slots=jnp.zeros((5, 2), dtype=jnp.int32),
        selected_counts=jnp.array([2, 2, 1, 0, 2], dtype=jnp.int32),
        producer_layer=1,
    )

    decode.validate(mode="decode")
    prefill.validate(mode="prefill")


def test_dsa_selection_validation_rejects_structural_abi_mismatches():
    from sgl_jax.srt.layers.attention.dsa_types import DsaSelection

    for mode, rows in (("decode", 2), ("prefill", 5)):
        with pytest.raises(ValueError, match=rf"{mode} physical_slots must have rank 2"):
            DsaSelection(
                physical_slots=jnp.zeros((rows,), dtype=jnp.int32),
                selected_counts=jnp.zeros((rows,), dtype=jnp.int32),
                producer_layer=0,
            ).validate(mode=mode)

        with pytest.raises(TypeError, match=rf"{mode} physical_slots must have dtype int32"):
            DsaSelection(
                physical_slots=jnp.zeros((rows, 2), dtype=jnp.float32),
                selected_counts=jnp.zeros((rows,), dtype=jnp.int32),
                producer_layer=0,
            ).validate(mode=mode)

        with pytest.raises(ValueError, match=rf"{mode} selected_counts must have rank 1"):
            DsaSelection(
                physical_slots=jnp.zeros((rows, 2), dtype=jnp.int32),
                selected_counts=jnp.zeros((rows, 1), dtype=jnp.int32),
                producer_layer=0,
            ).validate(mode=mode)

        with pytest.raises(TypeError, match=rf"{mode} selected_counts must have dtype int32"):
            DsaSelection(
                physical_slots=jnp.zeros((rows, 2), dtype=jnp.int32),
                selected_counts=jnp.zeros((rows,), dtype=jnp.float32),
                producer_layer=0,
            ).validate(mode=mode)

        with pytest.raises(ValueError, match=rf"{mode} selected_counts must have shape"):
            DsaSelection(
                physical_slots=jnp.zeros((rows, 2), dtype=jnp.int32),
                selected_counts=jnp.zeros((rows + 1,), dtype=jnp.int32),
                producer_layer=0,
            ).validate(mode=mode)


def test_dsa_selection_validation_rejects_counts_outside_topk_width():
    from sgl_jax.srt.layers.attention.dsa_types import DsaSelection

    for selected_counts in (
        jnp.array([-1, 1], dtype=jnp.int32),
        jnp.array([1, 3], dtype=jnp.int32),
    ):
        with pytest.raises(ValueError, match="selected_counts entries must be in"):
            DsaSelection(
                physical_slots=jnp.zeros((2, 2), dtype=jnp.int32),
                selected_counts=selected_counts,
                producer_layer=0,
            ).validate(mode="decode")


def test_dsa_selection_validation_requires_static_nonnegative_producer_layer():
    from sgl_jax.srt.layers.attention.dsa_types import DsaSelection

    physical_slots = jnp.zeros((1, 2), dtype=jnp.int32)
    selected_counts = jnp.ones((1,), dtype=jnp.int32)

    with pytest.raises(TypeError, match="producer_layer must be a Python int"):
        DsaSelection(
            physical_slots=physical_slots,
            selected_counts=selected_counts,
            producer_layer=jnp.array(0, dtype=jnp.int32),
        ).validate(mode="decode")

    with pytest.raises(ValueError, match="producer_layer must be nonnegative"):
        DsaSelection(
            physical_slots=physical_slots,
            selected_counts=selected_counts,
            producer_layer=-1,
        ).validate(mode="decode")


def test_dsa_selection_validation_checks_optional_logical_topk_ids():
    from sgl_jax.srt.layers.attention.dsa_types import DsaSelection

    physical_slots = jnp.zeros((2, 3), dtype=jnp.int32)
    selected_counts = jnp.ones((2,), dtype=jnp.int32)

    with pytest.raises(TypeError, match="logical_topk_ids must have dtype int32"):
        DsaSelection(
            physical_slots=physical_slots,
            selected_counts=selected_counts,
            producer_layer=0,
            logical_topk_ids=jnp.zeros((2, 3), dtype=jnp.float32),
        ).validate(mode="prefill")

    with pytest.raises(ValueError, match="logical_topk_ids must match physical_slots shape"):
        DsaSelection(
            physical_slots=physical_slots,
            selected_counts=selected_counts,
            producer_layer=0,
            logical_topk_ids=jnp.zeros((2, 2), dtype=jnp.int32),
        ).validate(mode="prefill")


def test_dsa_selection_validation_accepts_rank2_int32_logical_topk_ids():
    from sgl_jax.srt.layers.attention.dsa_types import DsaSelection

    DsaSelection(
        physical_slots=jnp.zeros((2, 3), dtype=jnp.int32),
        selected_counts=jnp.ones((2,), dtype=jnp.int32),
        producer_layer=0,
        logical_topk_ids=jnp.zeros((2, 3), dtype=jnp.int32),
    ).validate(mode="prefill")


def test_dsa_selection_validation_rejects_wrong_rank_logical_topk_ids():
    from sgl_jax.srt.layers.attention.dsa_types import DsaSelection

    with pytest.raises(ValueError, match="logical_topk_ids must have rank 2"):
        DsaSelection(
            physical_slots=jnp.zeros((2, 3), dtype=jnp.int32),
            selected_counts=jnp.ones((2,), dtype=jnp.int32),
            producer_layer=0,
            logical_topk_ids=jnp.zeros((2, 3, 1), dtype=jnp.int32),
        ).validate(mode="prefill")


def test_dsa_topk_state_pytree_preserves_request_boundary_arrays():
    from sgl_jax.srt.layers.attention.dsa_types import DsaSelection, DsaTopKState

    selection = DsaSelection(
        physical_slots=jnp.array([[0, 4], [3, 0], [7, 8]], dtype=jnp.int32),
        selected_counts=jnp.array([2, 1, 2], dtype=jnp.int32),
        producer_layer=6,
    )
    state = DsaTopKState(
        selection=selection,
        query_offsets=jnp.array([0, 2, 3], dtype=jnp.int32),
        request_offsets=jnp.array([10, 21], dtype=jnp.int32),
    )

    leaves, tree_def = jax.tree_util.tree_flatten(state)
    restored = jax.tree_util.tree_unflatten(tree_def, leaves)

    assert len(leaves) == 4
    np.testing.assert_array_equal(
        restored.selection.physical_slots, selection.physical_slots
    )
    assert restored.selection.producer_layer == 6
    np.testing.assert_array_equal(restored.query_offsets, state.query_offsets)
    np.testing.assert_array_equal(restored.request_offsets, state.request_offsets)


def test_dsa_topk_state_validation_accepts_decode_and_ragged_prefill_boundaries():
    from sgl_jax.srt.layers.attention.dsa_types import DsaSelection, DsaTopKState

    decode = DsaTopKState(
        selection=DsaSelection(
            physical_slots=jnp.zeros((2, 3), dtype=jnp.int32),
            selected_counts=jnp.array([3, 2], dtype=jnp.int32),
            producer_layer=2,
        ),
        query_offsets=jnp.array([0, 1, 2], dtype=jnp.int32),
        request_offsets=jnp.array([11, 27], dtype=jnp.int32),
    )
    prefill = DsaTopKState(
        selection=DsaSelection(
            physical_slots=jnp.zeros((5, 3), dtype=jnp.int32),
            selected_counts=jnp.array([1, 2, 3, 0, 1], dtype=jnp.int32),
            producer_layer=2,
        ),
        query_offsets=jnp.array([0, 2, 5], dtype=jnp.int32),
        request_offsets=jnp.array([11, 27], dtype=jnp.int32),
    )

    decode.validate(mode="decode")
    prefill.validate(mode="prefill")


def test_dsa_topk_state_validation_rejects_offset_rank_and_dtype_mismatches():
    from sgl_jax.srt.layers.attention.dsa_types import DsaSelection, DsaTopKState

    selection = DsaSelection(
        physical_slots=jnp.zeros((3, 2), dtype=jnp.int32),
        selected_counts=jnp.ones((3,), dtype=jnp.int32),
        producer_layer=0,
    )

    for query_offsets, request_offsets, error_type, message in (
        (
            jnp.array([[0, 3]], dtype=jnp.int32),
            jnp.array([0], dtype=jnp.int32),
            ValueError,
            "query_offsets must have rank 1",
        ),
        (
            jnp.array([0, 3], dtype=jnp.float32),
            jnp.array([0], dtype=jnp.int32),
            TypeError,
            "query_offsets must have dtype int32",
        ),
        (
            jnp.array([0, 3], dtype=jnp.int32),
            jnp.array([[0]], dtype=jnp.int32),
            ValueError,
            "request_offsets must have rank 1",
        ),
        (
            jnp.array([0, 3], dtype=jnp.int32),
            jnp.array([0], dtype=jnp.float32),
            TypeError,
            "request_offsets must have dtype int32",
        ),
    ):
        with pytest.raises(error_type, match=message):
            DsaTopKState(
                selection=selection,
                query_offsets=query_offsets,
                request_offsets=request_offsets,
            ).validate(mode="prefill")
