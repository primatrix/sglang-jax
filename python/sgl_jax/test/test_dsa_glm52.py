from types import SimpleNamespace

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
    np.testing.assert_array_equal(restored.selection.physical_slots, selection.physical_slots)
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


def test_glm_dsa_select_topk_uses_configured_dimensions_and_returns_exact_topk():
    from sgl_jax.srt.models.glm5_moe import GlmDsaIndexer

    q_index = jnp.array(
        [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]],
        dtype=jnp.float32,
    )
    head_weights = jnp.array([[1.0, 2.0]], dtype=jnp.float32)
    k_index_cache = jnp.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [2.0, 1.0, 0.0, 0.0],
            [-1.0, 3.0, 0.0, 0.0],
        ],
        dtype=jnp.float32,
    )

    ids, selected_counts = GlmDsaIndexer.select_topk(
        q_index=q_index,
        head_weights=head_weights,
        k_index_cache=k_index_cache,
        candidate_slots=jnp.array([[0, 1, 2, 3]], dtype=jnp.int32),
        candidate_logical_ids=jnp.array([[10, 20, 30, 40]], dtype=jnp.int32),
        candidate_counts=jnp.array([4], dtype=jnp.int32),
        index_topk=3,
    )

    assert ids.dtype == jnp.int32
    assert selected_counts.dtype == jnp.int32
    np.testing.assert_array_equal(ids, np.array([[40, 30, 20]], dtype=np.int32))
    np.testing.assert_array_equal(selected_counts, np.array([3], dtype=np.int32))


def test_glm_dsa_score_candidates_applies_relu_head_gates_and_both_scales():
    from sgl_jax.srt.models.glm5_moe import GlmDsaIndexer

    scores = GlmDsaIndexer.score_candidates(
        q_index=jnp.array(
            [[[2.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0]]],
            dtype=jnp.float32,
        ),
        head_weights=jnp.array([[3.0, 5.0]], dtype=jnp.float32),
        k_index_cache=jnp.array(
            [[1.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0]],
            dtype=jnp.float32,
        ),
        candidate_slots=jnp.array([[0, 1]], dtype=jnp.int32),
    )

    scale = (4**-0.5) * (2**-0.5)
    np.testing.assert_allclose(
        scores,
        np.array([[6.0 * scale, 5.0 * scale]], dtype=np.float32),
        rtol=1e-6,
    )


def test_glm_dsa_indexer_rejects_top1_configuration():
    from sgl_jax.srt.models.glm5_moe import GlmDsaIndexer

    with pytest.raises(ValueError, match="greater than one"):
        GlmDsaIndexer.select_topk(
            q_index=jnp.ones((1, 1, 2), dtype=jnp.float32),
            head_weights=jnp.ones((1, 1), dtype=jnp.float32),
            k_index_cache=jnp.ones((1, 2), dtype=jnp.float32),
            candidate_slots=jnp.zeros((1, 1), dtype=jnp.int32),
            candidate_logical_ids=jnp.zeros((1, 1), dtype=jnp.int32),
            candidate_counts=jnp.ones((1,), dtype=jnp.int32),
            index_topk=1,
        )


def test_glm_dsa_select_topk_pads_when_candidates_are_fewer_than_topk():
    from sgl_jax.srt.models.glm5_moe import GlmDsaIndexer

    ids, selected_counts = GlmDsaIndexer.select_topk(
        q_index=jnp.array([[[1.0, 0.0, 0.0, 0.0]]], dtype=jnp.float32),
        head_weights=jnp.ones((1, 1), dtype=jnp.float32),
        k_index_cache=jnp.array(
            [[1.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0]],
            dtype=jnp.float32,
        ),
        candidate_slots=jnp.array([[0, 1]], dtype=jnp.int32),
        candidate_logical_ids=jnp.array([[0, 1]], dtype=jnp.int32),
        candidate_counts=jnp.array([2], dtype=jnp.int32),
        index_topk=3,
    )

    np.testing.assert_array_equal(ids, np.array([[1, 0, -1]], dtype=np.int32))
    np.testing.assert_array_equal(selected_counts, np.array([2], dtype=np.int32))


def test_glm_dsa_select_topk_masks_by_count_while_preserving_logical_zero():
    from sgl_jax.srt.models.glm5_moe import GlmDsaIndexer

    ids, selected_counts = GlmDsaIndexer.select_topk(
        q_index=jnp.array([[[1.0, 0.0]]], dtype=jnp.float32),
        head_weights=jnp.ones((1, 1), dtype=jnp.float32),
        k_index_cache=jnp.array([[1.0, 0.0], [2.0, 0.0], [100.0, 0.0]], dtype=jnp.float32),
        candidate_slots=jnp.array([[0, 1, 2]], dtype=jnp.int32),
        candidate_logical_ids=jnp.array([[0, 1, 2]], dtype=jnp.int32),
        candidate_counts=jnp.array([2], dtype=jnp.int32),
        index_topk=3,
    )

    np.testing.assert_array_equal(ids, np.array([[1, 0, -1]], dtype=np.int32))
    np.testing.assert_array_equal(selected_counts, np.array([2], dtype=np.int32))


def test_glm_dsa_select_topk_zero_count_returns_only_sentinels():
    from sgl_jax.srt.models.glm5_moe import GlmDsaIndexer

    ids, selected_counts = GlmDsaIndexer.select_topk(
        q_index=jnp.ones((1, 2, 4), dtype=jnp.float32),
        head_weights=jnp.ones((1, 2), dtype=jnp.float32),
        k_index_cache=jnp.ones((2, 4), dtype=jnp.float32),
        candidate_slots=jnp.array([[0, 1]], dtype=jnp.int32),
        candidate_logical_ids=jnp.array([[0, 1]], dtype=jnp.int32),
        candidate_counts=jnp.array([0], dtype=jnp.int32),
        index_topk=2,
    )

    np.testing.assert_array_equal(ids, np.array([[-1, -1]], dtype=np.int32))
    np.testing.assert_array_equal(selected_counts, np.array([0], dtype=np.int32))


def test_glm_dsa_select_topk_is_jittable_and_deterministic_for_ties():
    from sgl_jax.srt.models.glm5_moe import GlmDsaIndexer

    select = jax.jit(
        lambda q, weights, cache, slots, counts: GlmDsaIndexer.select_topk(
            q,
            weights,
            cache,
            slots,
            slots,
            counts,
            index_topk=3,
        )
    )
    inputs = (
        jnp.array([[[1.0, 0.0]]], dtype=jnp.float32),
        jnp.ones((1, 1), dtype=jnp.float32),
        jnp.array([[1.0, 0.0], [1.0, 0.0], [0.5, 0.0]], dtype=jnp.float32),
        jnp.array([[1, 0, 2]], dtype=jnp.int32),
        jnp.array([3], dtype=jnp.int32),
    )

    first_ids, first_counts = select(*inputs)
    second_ids, second_counts = select(*inputs)

    np.testing.assert_array_equal(first_ids, np.array([[1, 0, 2]], dtype=np.int32))
    np.testing.assert_array_equal(second_ids, first_ids)
    np.testing.assert_array_equal(first_counts, np.array([3], dtype=np.int32))
    np.testing.assert_array_equal(second_counts, first_counts)


def test_glm_dsa_indexer_projects_query_weights_and_key_with_configured_dimension(
    monkeypatch,
):
    import sgl_jax.srt.models.glm5_moe as glm5_moe

    class StubLinear:
        def __init__(self, *, scope_name, **kwargs):
            del kwargs
            self.scope_name = scope_name

        def __call__(self, inputs):
            token_count = inputs.shape[0]
            outputs = {
                "wq_b": jnp.array(
                    [[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]],
                    dtype=jnp.float32,
                ),
                "wk": jnp.array([[0.0, 0.0, 1.0, 0.0]], dtype=jnp.float32),
                "weights_proj": jnp.array([[2.0, -1.0]], dtype=jnp.float32),
            }
            return (
                jnp.broadcast_to(
                    outputs[self.scope_name], (token_count, *outputs[self.scope_name].shape[1:])
                ),
                None,
            )

    class TrackingNorm:
        def __init__(self, *args, **kwargs):
            del args, kwargs

        def __call__(self, inputs):
            return inputs * 2.0 + 1.0

    class TrackingRotary:
        calls = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def __call__(self, positions, query, key):
            self.calls.append(np.asarray(positions))
            return query + 1.0, key + 2.0

    monkeypatch.setattr(glm5_moe, "LinearBase", StubLinear)
    monkeypatch.setattr(glm5_moe, "GlmNorm", TrackingNorm)
    monkeypatch.setattr(glm5_moe, "RotaryEmbedding", TrackingRotary)
    indexer = glm5_moe.GlmDsaIndexer(
        hidden_size=3,
        q_lora_rank=2,
        index_head_dim=4,
        index_n_heads=2,
        index_topk=3,
        rope_head_dim=2,
        max_position_embeddings=256,
        rope_theta=1234.0,
        rope_scaling={"factor": 2.0},
        indexer_rope_interleave=False,
        mesh=None,
        dtype=jnp.float32,
    )
    hidden_states = jnp.zeros((1, 3), dtype=jnp.float32)
    q_lora = jnp.zeros((1, 2), dtype=jnp.float32)
    positions = jnp.array([7], dtype=jnp.int32)

    q_index, head_weights = indexer.project_query(hidden_states, q_lora, positions)
    k_index = indexer.project_key(hidden_states, positions)
    call_result = indexer(hidden_states, q_lora, positions)

    np.testing.assert_allclose(
        q_index,
        np.array(
            [[[2.5, 0.5, 0.5, 0.5], [2.5, -0.5, 0.5, -0.5]]],
            dtype=np.float32,
        ),
    )
    np.testing.assert_array_equal(head_weights, np.array([[2.0, -1.0]], dtype=np.float32))
    np.testing.assert_allclose(k_index, np.array([[7.0, 1.0, -1.0, -1.0]], dtype=np.float32))
    for actual, expected in zip(call_result, (q_index, head_weights, k_index), strict=True):
        np.testing.assert_array_equal(actual, expected)
    assert len(TrackingRotary.calls) == 4
    for seen_positions in TrackingRotary.calls:
        np.testing.assert_array_equal(seen_positions, np.array([7], dtype=np.int32))


@pytest.mark.parametrize(
    ("indexer_rope_interleave", "expected_neox_style"),
    [(False, True), (True, False)],
)
def test_glm_dsa_indexer_stores_config_and_owns_rotary_embedding(
    monkeypatch, indexer_rope_interleave, expected_neox_style
):
    import sgl_jax.srt.models.glm5_moe as glm5_moe

    class StubLinear:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(glm5_moe, "LinearBase", StubLinear)
    rope_scaling = {"rope_type": "linear", "factor": 2.0}
    indexer = glm5_moe.GlmDsaIndexer(
        hidden_size=3,
        q_lora_rank=2,
        index_head_dim=8,
        index_n_heads=5,
        index_topk=7,
        rope_head_dim=6,
        max_position_embeddings=4096,
        rope_theta=9876.0,
        rope_scaling=rope_scaling,
        indexer_rope_interleave=indexer_rope_interleave,
        mesh=None,
        dtype=jnp.float32,
    )

    assert indexer.index_head_dim == 8
    assert indexer.index_n_heads == 5
    assert indexer.index_topk == 7
    assert indexer.rope_head_dim == 6
    assert indexer.max_position_embeddings == 4096
    assert indexer.rope_theta == 9876.0
    assert indexer.rope_scaling is rope_scaling
    assert indexer.indexer_rope_interleave is indexer_rope_interleave
    assert isinstance(indexer.rotary_emb, glm5_moe.RotaryEmbedding)
    assert indexer.rotary_emb.head_size == 8
    assert indexer.rotary_emb.rotary_dim == 6
    assert indexer.rotary_emb.is_neox_style is expected_neox_style


def test_glm_dsa_indexer_requires_power_of_two_head_dimension():
    from sgl_jax.srt.models.glm5_moe import GlmDsaIndexer

    with pytest.raises(ValueError, match="positive power of two"):
        GlmDsaIndexer(
            hidden_size=3,
            q_lora_rank=2,
            index_head_dim=6,
            index_n_heads=2,
            index_topk=3,
            rope_head_dim=4,
            max_position_embeddings=128,
            rope_theta=10000.0,
            rope_scaling=None,
            indexer_rope_interleave=False,
            mesh=None,
            dtype=jnp.float32,
        )


def test_glm5_attention_passes_nondefault_indexer_config(monkeypatch):
    import sgl_jax.srt.models.glm5_moe as glm5_moe

    class StopConstruction(Exception):
        pass

    captured = {}

    class StubLayer:
        def __init__(self, *args, **kwargs):
            del args, kwargs

    class CapturingIndexer:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            raise StopConstruction

    monkeypatch.setattr(glm5_moe, "LinearBase", StubLayer)
    monkeypatch.setattr(glm5_moe, "RMSNorm", StubLayer)
    monkeypatch.setattr(glm5_moe, "GlmDsaIndexer", CapturingIndexer)

    with pytest.raises(StopConstruction):
        glm5_moe.Glm5Attention(
            hidden_size=16,
            num_heads=2,
            num_kv_heads=1,
            max_position_embeddings=8192,
            mesh=None,
            rope_theta=54321.0,
            rope_scaling={"factor": 4.0},
            index_head_dim=8,
            index_n_heads=5,
            index_topk=7,
            qk_rope_head_dim=6,
            indexer_rope_interleave=True,
            use_qk_norm=False,
        )

    assert captured["index_head_dim"] == 8
    assert captured["index_n_heads"] == 5
    assert captured["index_topk"] == 7
    assert captured["rope_head_dim"] == 6
    assert captured["max_position_embeddings"] == 8192
    assert captured["rope_theta"] == 54321.0
    assert captured["rope_scaling"] == {"factor": 4.0}
    assert captured["indexer_rope_interleave"] is True


def test_glm5_decoder_layer_propagates_nondefault_indexer_config(monkeypatch):
    import sgl_jax.srt.models.glm5_moe as glm5_moe

    class StopConstruction(Exception):
        pass

    captured = {}

    class CapturingAttention:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            raise StopConstruction

    monkeypatch.setattr(glm5_moe, "Glm5Attention", CapturingAttention)
    config = SimpleNamespace(
        hidden_size=16,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=8192,
        rope_theta=54321.0,
        rope_scaling={"factor": 4.0},
        head_dim=8,
        qk_rope_head_dim=6,
        index_head_dim=8,
        index_n_heads=5,
        index_topk=7,
        indexer_rope_interleave=True,
        rms_norm_eps=1e-5,
        use_qk_norm=False,
    )

    with pytest.raises(StopConstruction):
        glm5_moe.Glm5DecoderLayer(config=config, mesh=None, layer_id=0, dtype=jnp.float32)

    assert captured["index_head_dim"] == 8
    assert captured["index_n_heads"] == 5
    assert captured["index_topk"] == 7
    assert captured["qk_rope_head_dim"] == 6
    assert captured["indexer_rope_interleave"] is True
    assert captured["max_position_embeddings"] == 8192
    assert captured["rope_theta"] == 54321.0
    assert captured["rope_scaling"] == {"factor": 4.0}


def test_dsa_indexer_k_pool_uses_packed_paged_bf16_layout_and_is_a_pytree():
    from sgl_jax.srt.mem_cache.dsa_pool import DsaIndexerKPool

    pool = DsaIndexerKPool(
        size=8,
        page_size=4,
        index_head_dim=3,
        layer_num=2,
        mesh=None,
    )

    assert pool.dtype == jnp.bfloat16
    assert pool.packing == 2
    assert pool.aligned_head_dim == 128
    assert pool.get_buffer(0).shape == (3, 2, 2, 128)
    leaves, tree_def = jax.tree_util.tree_flatten(pool)
    restored = jax.tree_util.tree_unflatten(tree_def, leaves)
    assert len(leaves) == 2
    assert restored.get_buffer(1).shape == (3, 2, 2, 128)


def test_dsa_indexer_k_pool_replace_buffer_uses_local_layer_storage():
    from sgl_jax.srt.mem_cache.dsa_pool import DsaIndexerKPool

    pool = DsaIndexerKPool(
        size=8,
        page_size=4,
        index_head_dim=3,
        layer_num=2,
        mesh=None,
        start_layer=4,
    )
    first = jnp.ones_like(pool.get_buffer(4))
    second = jnp.full_like(pool.get_buffer(5), 2)
    pool.replace_buffer([first, second])

    assert pool.end_layer == 5
    np.testing.assert_array_equal(pool.get_buffer(4), first)
    np.testing.assert_array_equal(pool.get_buffer(5), second)


def test_dsa_indexer_k_pool_rejects_multi_dp_until_slots_are_globalized():
    from sgl_jax.srt.mem_cache.dsa_pool import DsaIndexerKPool

    with pytest.raises(NotImplementedError, match="single-DP"):
        DsaIndexerKPool(
            size=8,
            page_size=4,
            index_head_dim=3,
            layer_num=1,
            mesh=None,
            dp_size=2,
        )


def test_write_indexer_k_cache_handles_slot_zero_page_boundaries_and_padding():
    from sgl_jax.srt.kernels.dsa.reference import (
        gather_indexer_k_cache,
        write_indexer_k_cache,
    )
    from sgl_jax.srt.mem_cache.dsa_pool import DsaIndexerKPool

    pool = DsaIndexerKPool(
        size=8,
        page_size=4,
        index_head_dim=3,
        layer_num=1,
        mesh=None,
    )
    updated = write_indexer_k_cache(
        pool.get_buffer(0),
        index_k=jnp.array(
            [[1, 2, 3], [4, 5, 6], [7, 8, 9], [99, 99, 99]],
            dtype=jnp.bfloat16,
        ),
        write_slots=jnp.array([0, 3, 4, -1], dtype=jnp.int32),
        page_size=4,
        index_head_dim=3,
    )

    gathered = gather_indexer_k_cache(
        updated,
        physical_slots=jnp.array([[0, 3, 4, 1]], dtype=jnp.int32),
        page_size=4,
        index_head_dim=3,
    )
    np.testing.assert_array_equal(
        gathered,
        np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9], [0, 0, 0]]]),
    )


def test_logical_topk_to_physical_slots_compacts_causal_valid_unique_ids():
    from sgl_jax.srt.kernels.dsa.reference import logical_topk_to_physical_slots

    selection = logical_topk_to_physical_slots(
        logical_topk_ids=jnp.array(
            [[0, 2, 1, -1], [3, 1, 1, 0], [0, 2, 4, 1]],
            dtype=jnp.int32,
        ),
        selected_counts=jnp.array([3, 4, 4], dtype=jnp.int32),
        req_to_token_slots=jnp.array(
            [[0, 5, 9, 12, -1], [7, 8, -1, -1, -1]],
            dtype=jnp.int32,
        ),
        query_request_indices=jnp.array([0, 0, 1], dtype=jnp.int32),
        query_positions=jnp.array([2, 2, 2], dtype=jnp.int32),
        producer_layer=6,
    )

    np.testing.assert_array_equal(
        selection.logical_topk_ids,
        np.array([[0, 2, 1, -1], [1, 0, -1, -1], [0, 1, -1, -1]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        selection.physical_slots,
        np.array([[0, 9, 5, 0], [5, 0, 0, 0], [7, 8, 0, 0]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        selection.selected_counts,
        np.array([3, 2, 2], dtype=np.int32),
    )
    assert selection.producer_layer == 6
    selection.validate(mode="prefill")


def test_logical_topk_to_physical_slots_is_jittable():
    from sgl_jax.srt.kernels.dsa.reference import logical_topk_to_physical_slots

    transform = jax.jit(
        lambda ids, counts, mapping, reqs, positions: logical_topk_to_physical_slots(
            logical_topk_ids=ids,
            selected_counts=counts,
            req_to_token_slots=mapping,
            query_request_indices=reqs,
            query_positions=positions,
            producer_layer=2,
        )
    )
    selection = transform(
        jnp.array([[1, 0]], dtype=jnp.int32),
        jnp.array([2], dtype=jnp.int32),
        jnp.array([[0, 4]], dtype=jnp.int32),
        jnp.array([0], dtype=jnp.int32),
        jnp.array([1], dtype=jnp.int32),
    )

    np.testing.assert_array_equal(selection.physical_slots, np.array([[4, 0]], dtype=np.int32))
    np.testing.assert_array_equal(selection.selected_counts, np.array([2], dtype=np.int32))


def test_non_hybrid_memory_pools_register_indexer_k_pool_and_require_both_updates():
    from sgl_jax.srt.model_executor.model_runner_kv_cache_mixin import (
        _build_non_hybrid_memory_pools,
    )

    class FakePool:
        def __init__(self):
            self.replaced = None

        def replace_buffer(self, value):
            self.replaced = value

    token_pool = FakePool()
    indexer_pool = FakePool()
    dense_pool = FakePool()
    dense_pools = _build_non_hybrid_memory_pools(dense_pool)
    with pytest.raises(AttributeError, match="indexer_k_pool"):
        _ = dense_pools.indexer_k_pool
    dense_pools.replace_all(["dense-mla"])
    assert dense_pool.replaced == ["dense-mla"]

    pools = _build_non_hybrid_memory_pools(
        token_pool,
        indexer_k_pool=indexer_pool,
    )

    assert pools.token_to_kv_pool is token_pool
    assert pools.indexer_k_pool is indexer_pool
    pools.replace_all(
        {
            "token_to_kv_pool": ["mla"],
            "indexer_k_pool": ["index"],
        }
    )
    assert token_pool.replaced == ["mla"]
    assert indexer_pool.replaced == ["index"]
    with pytest.raises(ValueError, match="must exactly match"):
        pools.replace_all({"token_to_kv_pool": ["mla-only"]})
