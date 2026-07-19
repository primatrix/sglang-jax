from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def test_glm52_fused_moe_maps_shared_experts_into_fused_weight_slots():
    from sgl_jax.srt.configs.model_config import MoEBackend
    from sgl_jax.srt.models.glm5_moe import Glm5ForCausalLM

    model = SimpleNamespace(
        config=SimpleNamespace(
            n_routed_experts=256,
            n_shared_experts=1,
            moe_backend=MoEBackend.FUSED,
        )
    )

    mappings = Glm5ForCausalLM._create_moe_layer_mappings(
        model,
        layer_idx=3,
        target_idx=3,
        is_mlp_layer=False,
    )

    expected_targets = {
        "gate_proj": "model.layers.3.mlp.w1_shared",
        "up_proj": "model.layers.3.mlp.w3_shared",
        "down_proj": "model.layers.3.mlp.w2_shared",
    }
    for projection, target_path in expected_targets.items():
        mapping = mappings[
            f"model.layers.3.mlp.shared_experts.{projection}.weight"
        ]
        assert mapping.target_path == target_path
        assert mapping.sharding == (None, None)
        assert mapping.transpose is True


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


def test_glm_dsa_score_candidates_uses_slot_sharding_for_candidate_gather(monkeypatch):
    from sgl_jax.srt.models.glm5_moe import GlmDsaIndexer

    query_sharding = object()
    slot_sharding = object()
    captured = {}

    class FakeAt:
        def __getitem__(self, index):
            captured["index"] = index
            return self

        def get(self, **kwargs):
            captured["kwargs"] = kwargs
            return jnp.ones((1, 2, 4), dtype=jnp.float32)

    class FakeCache:
        ndim = 2
        shape = (4, 4)
        at = FakeAt()

    candidate_slots = jnp.array([[0, 1]], dtype=jnp.int32)

    def fake_typeof(array):
        sharding = slot_sharding if array is candidate_slots else query_sharding
        return SimpleNamespace(sharding=sharding)

    monkeypatch.setattr(jax, "typeof", fake_typeof)
    scores = GlmDsaIndexer.score_candidates(
        q_index=jnp.ones((1, 2, 4), dtype=jnp.float32),
        head_weights=jnp.ones((1, 2), dtype=jnp.float32),
        k_index_cache=FakeCache(),
        candidate_slots=candidate_slots,
    )

    assert scores.shape == (1, 2)
    assert captured["kwargs"]["out_sharding"] is slot_sharding


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


def test_write_indexer_k_cache_preserves_operand_sharding_on_scatter(monkeypatch):
    from sgl_jax.srt.kernels.dsa.reference import write_indexer_k_cache

    expected_sharding = object()
    captured = {}

    class FakeAt:
        def __getitem__(self, index):
            captured["index"] = index
            return self

        def set(self, values, **kwargs):
            captured["values"] = values
            captured["kwargs"] = kwargs
            return "updated"

    class FakeCache:
        ndim = 4
        shape = (2, 2, 2, 128)
        dtype = jnp.bfloat16
        at = FakeAt()

    monkeypatch.setattr(jax, "typeof", lambda _array: SimpleNamespace(sharding=expected_sharding))
    result = write_indexer_k_cache(
        FakeCache(),
        index_k=jnp.ones((1, 128), dtype=jnp.bfloat16),
        write_slots=jnp.array([0], dtype=jnp.int32),
        page_size=4,
        index_head_dim=128,
    )

    assert result == "updated"
    assert captured["kwargs"]["out_sharding"] is expected_sharding


def test_dsa_candidate_row_gather_preserves_mapping_sharding(monkeypatch):
    from sgl_jax.srt.layers.attention.dsa_backend import _gather_candidate_rows

    expected_sharding = object()
    captured = {}

    class FakeAt:
        def __getitem__(self, index):
            captured["index"] = index
            return self

        def get(self, **kwargs):
            captured["kwargs"] = kwargs
            return "candidate-rows"

    mapping = SimpleNamespace(at=FakeAt())
    monkeypatch.setattr(jax, "typeof", lambda _array: SimpleNamespace(sharding=expected_sharding))
    result = _gather_candidate_rows(mapping, "safe-requests")

    assert result == "candidate-rows"
    assert captured["index"] == "safe-requests"
    assert captured["kwargs"]["out_sharding"] is expected_sharding


def test_logical_slot_mapping_uses_logical_id_sharding_for_gather(monkeypatch):
    from sgl_jax.srt.kernels.dsa.reference import logical_topk_to_physical_slots

    expected_sharding = object()
    captured = {}

    class FakeAt:
        def __getitem__(self, index):
            captured["index"] = index
            return self

        def get(self, **kwargs):
            captured["kwargs"] = kwargs
            return jnp.array([[3, 4]], dtype=jnp.int32)

    class FakeMapping:
        ndim = 2
        dtype = jnp.int32
        shape = (1, 2)
        at = FakeAt()

    original_typeof = jax.typeof
    logical_ids = jnp.array([[0, 1]], dtype=jnp.int32)

    def fake_typeof(array):
        if array is logical_ids:
            return SimpleNamespace(sharding=expected_sharding)
        return original_typeof(array)

    monkeypatch.setattr(jax, "typeof", fake_typeof)
    selection = logical_topk_to_physical_slots(
        logical_topk_ids=logical_ids,
        selected_counts=jnp.array([2], dtype=jnp.int32),
        req_to_token_slots=FakeMapping(),
        query_request_indices=jnp.array([0], dtype=jnp.int32),
        query_positions=jnp.array([1], dtype=jnp.int32),
        producer_layer=0,
    )

    assert selection.physical_slots.shape == (1, 2)
    assert captured["kwargs"]["out_sharding"] is expected_sharding


def test_logical_slot_compaction_builds_sort_iota_with_validity_sharding(monkeypatch):
    from sgl_jax.srt.kernels.dsa.reference import logical_topk_to_physical_slots

    original_broadcasted_iota = jax.lax.broadcasted_iota
    captured = {}

    def capture_iota(*args, **kwargs):
        captured["kwargs"] = kwargs
        return original_broadcasted_iota(*args, **kwargs)

    monkeypatch.setattr(jax.lax, "broadcasted_iota", capture_iota)
    logical_ids = jnp.array([[2, 0, 1]], dtype=jnp.int32)
    selection = logical_topk_to_physical_slots(
        logical_topk_ids=logical_ids,
        selected_counts=jnp.array([3], dtype=jnp.int32),
        req_to_token_slots=jnp.array([[4, -1, 8]], dtype=jnp.int32),
        query_request_indices=jnp.array([0], dtype=jnp.int32),
        query_positions=jnp.array([2], dtype=jnp.int32),
        producer_layer=0,
    )

    np.testing.assert_array_equal(selection.logical_topk_ids, np.array([[2, 0, -1]]))
    assert captured["kwargs"]["out_sharding"] == jax.typeof(logical_ids).sharding


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


def test_glm_attention_full_builds_and_shared_reuses_indexshare_state():
    from types import SimpleNamespace

    from sgl_jax.srt.layers.attention.dsa_types import DsaSelection, DsaTopKState
    from sgl_jax.srt.models.glm5_moe import Glm5Attention

    selection = DsaSelection(
        logical_topk_ids=jnp.array([[1, 0]], dtype=jnp.int32),
        physical_slots=jnp.array([[9, 0]], dtype=jnp.int32),
        selected_counts=jnp.array([2], dtype=jnp.int32),
        producer_layer=0,
    )
    produced_state = DsaTopKState(
        selection=selection,
        query_offsets=jnp.array([0, 1], dtype=jnp.int32),
        request_offsets=jnp.array([0], dtype=jnp.int32),
    )

    class CountingIndexer:
        def __init__(self):
            self.calls = 0

        def __call__(self, hidden_states, q_lora, positions):
            self.calls += 1
            return hidden_states[:, None, :2], q_lora[:, :1], hidden_states[:, :2]

    class Backend:
        def __init__(self):
            self.calls = 0

        def build_dsa_state(self, **kwargs):
            self.calls += 1
            assert kwargs["layer_id"] == 0
            assert kwargs["prev_dsa_state"] is None
            return produced_state, "updated-full-index-cache"

    class IndexPool:
        def get_buffer(self, layer_id):
            return f"unchanged-index-cache-{layer_id}"

    backend = Backend()
    forward_batch = SimpleNamespace(attn_backend=backend)
    index_pool = IndexPool()
    indexer = CountingIndexer()
    full_attention = SimpleNamespace(layer_id=0, indexer=indexer)
    shared_attention = SimpleNamespace(layer_id=1, indexer=None)
    hidden = jnp.ones((1, 4), dtype=jnp.float32)
    q_lora = jnp.ones((1, 3), dtype=jnp.float32)
    positions = jnp.array([2], dtype=jnp.int32)

    full_state, full_update = Glm5Attention._build_or_share_dsa_state(
        full_attention,
        hidden_states=hidden,
        q_lora=q_lora,
        positions=positions,
        forward_batch=forward_batch,
        indexer_k_pool=index_pool,
        prev_dsa_state=None,
    )
    shared_state, shared_update = Glm5Attention._build_or_share_dsa_state(
        shared_attention,
        hidden_states=hidden,
        q_lora=q_lora,
        positions=positions,
        forward_batch=forward_batch,
        indexer_k_pool=index_pool,
        prev_dsa_state=full_state,
    )

    assert indexer.calls == 1
    assert backend.calls == 1
    assert full_state is produced_state
    assert shared_state is produced_state
    assert full_update == "updated-full-index-cache"
    assert shared_update is None


def test_indexer_k_pool_maps_sparse_full_layer_ids_to_compact_storage():
    from sgl_jax.srt.mem_cache.dsa_pool import DsaIndexerKPool

    pool = DsaIndexerKPool(
        size=4,
        page_size=2,
        index_head_dim=2,
        layer_num=2,
        layer_ids=(0, 2),
        mesh=None,
    )

    assert pool.get_buffer(0) is pool.k_buffer[0]
    assert pool.get_buffer(2) is pool.k_buffer[1]
    with pytest.raises(IndexError, match="no Index-K storage"):
        pool.get_buffer(1)

    leaves, tree_def = jax.tree_util.tree_flatten(pool)
    restored = jax.tree_util.tree_unflatten(tree_def, leaves)
    assert restored.layer_ids == (0, 2)
    assert restored.get_buffer(2) is restored.k_buffer[1]


def test_full_shared_full_schedule_uses_compact_indexer_k_pool_order():
    from types import SimpleNamespace

    from sgl_jax.srt.layers.attention.dsa_types import DsaSelection, DsaTopKState
    from sgl_jax.srt.mem_cache.dsa_pool import DsaIndexerKPool
    from sgl_jax.srt.models.glm5_moe import Glm5Attention

    pool = DsaIndexerKPool(
        size=4,
        page_size=2,
        index_head_dim=2,
        layer_num=2,
        layer_ids=(0, 2),
        mesh=None,
    )

    class Indexer:
        def __call__(self, hidden_states, q_lora, positions):
            del positions
            return hidden_states[:, None, :2], q_lora[:, :1], hidden_states[:, :2]

    class Backend:
        def __init__(self):
            self.layers = []

        def build_dsa_state(self, *, layer_id, indexer_k_pool, **kwargs):
            del kwargs
            self.layers.append(layer_id)
            state = DsaTopKState(
                selection=DsaSelection(
                    logical_topk_ids=jnp.array([[0]], dtype=jnp.int32),
                    physical_slots=jnp.array([[0]], dtype=jnp.int32),
                    selected_counts=jnp.array([1], dtype=jnp.int32),
                    producer_layer=layer_id,
                ),
                query_offsets=jnp.array([0, 1], dtype=jnp.int32),
                request_offsets=jnp.array([0], dtype=jnp.int32),
            )
            return state, indexer_k_pool.get_buffer(layer_id)

    backend = Backend()
    forward_batch = SimpleNamespace(attn_backend=backend)
    attentions = [
        SimpleNamespace(layer_id=0, indexer=Indexer()),
        SimpleNamespace(layer_id=1, indexer=None),
        SimpleNamespace(layer_id=2, indexer=Indexer()),
    ]
    hidden = jnp.ones((1, 4), dtype=jnp.float32)
    q_lora = jnp.ones((1, 3), dtype=jnp.float32)
    positions = jnp.array([0], dtype=jnp.int32)

    states = []
    updates = []
    previous = None
    for attention in attentions:
        previous, update = Glm5Attention._build_or_share_dsa_state(
            attention,
            hidden_states=hidden,
            q_lora=q_lora,
            positions=positions,
            forward_batch=forward_batch,
            indexer_k_pool=pool,
            prev_dsa_state=previous,
        )
        states.append(previous)
        if update is not None:
            updates.append(update)

    assert backend.layers == [0, 2]
    assert [state.selection.producer_layer for state in states] == [0, 0, 2]
    assert updates[0] is pool.k_buffer[0]
    assert updates[1] is pool.k_buffer[1]


def test_glm_attention_forwards_dsa_kwarg_only_for_nonempty_state(monkeypatch):
    from types import SimpleNamespace

    from sgl_jax.srt.layers.attention.dsa_types import DsaSelection, DsaTopKState
    from sgl_jax.srt.models import glm5_moe
    from sgl_jax.srt.models.glm5_moe import Glm5Attention

    broadcast_to = jnp.broadcast_to
    monkeypatch.setattr(
        glm5_moe.jnp,
        "broadcast_to",
        lambda array, shape, **_: broadcast_to(array, shape),
    )

    state = DsaTopKState(
        selection=DsaSelection(
            logical_topk_ids=jnp.array([[0]], dtype=jnp.int32),
            physical_slots=jnp.array([[0]], dtype=jnp.int32),
            selected_counts=jnp.array([1], dtype=jnp.int32),
            producer_layer=0,
        ),
        query_offsets=jnp.array([0, 1], dtype=jnp.int32),
        request_offsets=jnp.array([0], dtype=jnp.int32),
    )

    class CaptureAttention:
        def __init__(self):
            self.kwargs = []

        def __call__(self, q, k, v, **kwargs):
            del q, k
            self.kwargs.append(kwargs)
            return v, "kv-update"

    mqa_backend = CaptureAttention()
    mqa = SimpleNamespace(
        w_uk=SimpleNamespace(value=jnp.ones((1, 1, 1), dtype=jnp.float32)),
        w_uv=SimpleNamespace(value=jnp.ones((1, 1, 1), dtype=jnp.float32)),
        attn_mqa=mqa_backend,
        num_heads=1,
        v_head_dim=1,
    )
    mqa_args = (
        jnp.ones((1, 1, 1), dtype=jnp.float32),
        jnp.ones((1, 1, 1), dtype=jnp.float32),
        jnp.ones((1, 1), dtype=jnp.float32),
        jnp.ones((1, 1, 1), dtype=jnp.float32),
        "forward-batch",
        "mla-pool",
    )
    Glm5Attention._forward_mqa(mqa, *mqa_args, dsa_state=None)
    Glm5Attention._forward_mqa(mqa, *mqa_args, dsa_state=state)

    assert "dsa_state" not in mqa_backend.kwargs[0]
    assert mqa_backend.kwargs[1]["dsa_state"] is state

    mha_backend = CaptureAttention()
    mha = SimpleNamespace(
        kv_b_proj=lambda compressed: (
            jnp.ones((compressed.shape[0], 2), dtype=jnp.float32),
            None,
        ),
        num_heads=1,
        qk_nope_head_dim=1,
        v_head_dim=1,
        qk_rope_head_dim=1,
        attn_mha=mha_backend,
    )
    mha_args = (
        jnp.ones((1, 1, 1), dtype=jnp.float32),
        jnp.ones((1, 1, 1), dtype=jnp.float32),
        jnp.ones((1, 1), dtype=jnp.float32),
        jnp.ones((1, 1, 1), dtype=jnp.float32),
        "forward-batch",
        "mha-pool",
    )
    Glm5Attention._forward_mha(mha, *mha_args, dsa_state=None)
    Glm5Attention._forward_mha(mha, *mha_args, dsa_state=state)

    assert "dsa_state" not in mha_backend.kwargs[0]
    assert mha_backend.kwargs[1]["dsa_state"] is state


def test_glm_decoder_layer_returns_moe_dsa_and_index_updates_separately():
    from types import SimpleNamespace

    from sgl_jax.srt.layers.attention.dsa_types import DsaSelection, DsaTopKState
    from sgl_jax.srt.models.glm5_moe import Glm5DecoderLayer

    state = DsaTopKState(
        selection=DsaSelection(
            logical_topk_ids=jnp.array([[0]], dtype=jnp.int32),
            physical_slots=jnp.array([[0]], dtype=jnp.int32),
            selected_counts=jnp.array([1], dtype=jnp.int32),
            producer_layer=0,
        ),
        query_offsets=jnp.array([0, 1], dtype=jnp.int32),
        request_offsets=jnp.array([0], dtype=jnp.int32),
    )
    moe_ids = jnp.array([[4, 7]], dtype=jnp.int32)

    class SelfAttention:
        def __call__(self, **kwargs):
            assert kwargs["indexer_k_pool"] == "index-pool"
            assert kwargs["prev_dsa_state"] is state
            return jnp.zeros_like(kwargs["hidden_states"]), "mla-update", state, "index-update"

    class Gate:
        bias = None

        def __call__(self, hidden_states):
            return jnp.zeros((hidden_states.shape[0], 8), dtype=jnp.float32)

    class ExpertTopK:
        def __call__(self, router_logits, correction_bias, *, dispatch_info):
            del router_logits, correction_bias, dispatch_info
            return jnp.ones((1, 2), dtype=jnp.float32), moe_ids

    class Moe:
        def __call__(self, hidden_states, topk_weights, topk_ids):
            del topk_weights
            assert topk_ids is moe_ids
            return hidden_states + 10

    layer = SimpleNamespace(
        layer_id=0,
        input_layernorm=lambda value: value,
        self_attn=SelfAttention(),
        post_attention_layernorm=lambda value: value,
        is_moe_layer=True,
        shared_experts=None,
        moe_gate=Gate(),
        topk=ExpertTopK(),
        mlp=Moe(),
    )
    result = Glm5DecoderLayer.__call__(
        layer,
        positions=jnp.array([0], dtype=jnp.int32),
        hidden_states=jnp.array([[1.0, 2.0]], dtype=jnp.float32),
        forward_batch="forward-batch",
        token_to_kv_pool="mla-pool",
        indexer_k_pool="index-pool",
        residual=None,
        prev_dsa_state=state,
        dispatch_info="dispatch",
    )

    hidden, residual, mla_update, returned_moe_ids, dsa_state, index_update = result
    np.testing.assert_array_equal(hidden, np.array([[11.0, 12.0]], dtype=np.float32))
    np.testing.assert_array_equal(residual, np.array([[1.0, 2.0]], dtype=np.float32))
    assert mla_update == "mla-update"
    assert returned_moe_ids is moe_ids
    assert dsa_state is state
    assert index_update == "index-update"


def test_glm_decoder_layer_emits_attention_residual_and_mlp_debug_tensors(monkeypatch):
    from types import SimpleNamespace

    from sgl_jax.srt.models import glm5_moe
    from sgl_jax.srt.models.glm5_moe import Glm5DecoderLayer

    calls = []

    def capture(value, *, component, name, layer_id, forward_mode, **_kwargs):
        calls.append((component, name, layer_id, forward_mode, np.asarray(value)))
        return value

    monkeypatch.setattr(glm5_moe, "maybe_dump_jax_array", capture, raising=False)
    monkeypatch.setattr(
        glm5_moe,
        "maybe_dump_jax_array_sum",
        lambda left, right, **kwargs: capture(left + right, **kwargs),
        raising=False,
    )

    class SelfAttention:
        def __call__(self, **kwargs):
            return jnp.zeros_like(kwargs["hidden_states"]), "mla", None, None

    layer = SimpleNamespace(
        layer_id=4,
        input_layernorm=lambda value: value,
        self_attn=SelfAttention(),
        post_attention_layernorm=lambda value: value,
        is_moe_layer=False,
        mlp=lambda value: value + 10,
    )
    forward_batch = SimpleNamespace(forward_mode="decode")
    Glm5DecoderLayer.__call__(
        layer,
        positions=jnp.array([0], dtype=jnp.int32),
        hidden_states=jnp.array([[1.0, 2.0]], dtype=jnp.float32),
        forward_batch=forward_batch,
        token_to_kv_pool="mla-pool",
    )

    assert [(component, name, layer_id, mode) for component, name, layer_id, mode, _ in calls] == [
        ("decoder_layer", "attention_output", 4, "decode"),
        ("decoder_layer", "residual_post_attention", 4, "decode"),
        ("decoder_layer", "mlp_output", 4, "decode"),
        ("decoder_layer", "hidden_states_post_mlp", 4, "decode"),
    ]
    np.testing.assert_array_equal(calls[0][4], np.array([[0.0, 0.0]], dtype=np.float32))
    np.testing.assert_array_equal(calls[1][4], np.array([[1.0, 2.0]], dtype=np.float32))
    np.testing.assert_array_equal(calls[2][4], np.array([[11.0, 12.0]], dtype=np.float32))
    np.testing.assert_array_equal(calls[3][4], np.array([[12.0, 14.0]], dtype=np.float32))


def test_glm_decoder_mlp_debug_includes_shared_expert_and_delayed_residual(monkeypatch):
    from types import SimpleNamespace

    from sgl_jax.srt.models import glm5_moe
    from sgl_jax.srt.models.glm5_moe import Glm5DecoderLayer

    calls = {}

    def capture(value, *, name, **_kwargs):
        calls[name] = np.asarray(value)
        return value

    monkeypatch.setattr(glm5_moe, "maybe_dump_jax_array", capture)
    monkeypatch.setattr(
        glm5_moe,
        "maybe_dump_jax_array_sum",
        lambda left, right, **kwargs: capture(left + right, **kwargs),
        raising=False,
    )

    class SelfAttention:
        def __call__(self, **kwargs):
            return jnp.zeros_like(kwargs["hidden_states"]), "mla", None, None

    class Gate:
        bias = None

        def __call__(self, hidden_states):
            return jnp.zeros((hidden_states.shape[0], 2), dtype=jnp.float32)

    layer = SimpleNamespace(
        layer_id=5,
        input_layernorm=lambda value: value,
        self_attn=SelfAttention(),
        post_attention_layernorm=lambda value: value,
        is_moe_layer=True,
        shared_experts=lambda value: value + 100,
        moe_gate=Gate(),
        topk=lambda *_args, **_kwargs: (
            jnp.ones((1, 1), dtype=jnp.float32),
            jnp.zeros((1, 1), dtype=jnp.int32),
        ),
        mlp=lambda value, *_args: value + 10,
    )
    Glm5DecoderLayer.__call__(
        layer,
        positions=jnp.array([0], dtype=jnp.int32),
        hidden_states=jnp.array([[1.0, 2.0]], dtype=jnp.float32),
        forward_batch=SimpleNamespace(forward_mode="decode"),
        token_to_kv_pool="mla-pool",
    )

    np.testing.assert_array_equal(
        calls["mlp_output"], np.array([[112.0, 114.0]], dtype=np.float32)
    )
    np.testing.assert_array_equal(
        calls["hidden_states_post_mlp"], np.array([[113.0, 116.0]], dtype=np.float32)
    )


def test_glm_model_and_causal_lm_emit_global_debug_tensors(monkeypatch):
    from types import SimpleNamespace

    from sgl_jax.srt.models import glm5_moe
    from sgl_jax.srt.models.glm5_moe import Glm5ForCausalLM, Glm5Model

    calls = []

    def capture(value, *, component, name, layer_id, forward_mode, **_kwargs):
        calls.append((component, name, layer_id, forward_mode, np.asarray(value)))
        return value

    monkeypatch.setattr(glm5_moe, "maybe_dump_jax_array", capture, raising=False)
    forward_batch = SimpleNamespace(
        input_ids=jnp.array([3], dtype=jnp.int32),
        positions=jnp.array([0], dtype=jnp.int32),
        expert_location_metadata=None,
        forward_mode="extend",
        get_token_valid_mask=lambda num_tokens: jnp.ones(
            (num_tokens,), dtype=jnp.bool_
        ),
    )
    model = SimpleNamespace(
        embed_tokens=lambda input_ids: input_ids[:, None].astype(jnp.float32),
        layers=[],
        norm=lambda hidden_states: hidden_states + 1,
    )
    hidden_states, *_ = Glm5Model.__call__(
        model,
        forward_batch,
        token_to_kv_pool="mla-pool",
    )

    class StubModel:
        def __call__(self, *_args):
            return hidden_states, [], [], [], []

    logits = SimpleNamespace(next_token_logits=jnp.array([[0.25, -0.5]], dtype=jnp.float32))
    causal_lm = SimpleNamespace(
        model=StubModel(),
        config=SimpleNamespace(tie_word_embeddings=False),
        lm_head="lm-head",
        logits_processor=lambda *_args: logits,
    )
    Glm5ForCausalLM.__call__(
        causal_lm,
        forward_batch,
        SimpleNamespace(token_to_kv_pool="mla-pool"),
        logits_metadata="metadata",
    )

    assert [(component, name, layer_id, mode) for component, name, layer_id, mode, _ in calls] == [
        ("embed", "hidden_states", None, "extend"),
        ("debug_context", "token_valid_mask", None, "extend"),
        ("debug_context", "token_positions", None, "extend"),
        ("final", "normalized_hidden_states", None, "extend"),
        ("logits", "next_token_logits", None, "extend"),
        ("debug_context", "forward_complete", None, "extend"),
    ]
    np.testing.assert_array_equal(calls[0][4], np.array([[3.0]], dtype=np.float32))
    np.testing.assert_array_equal(calls[1][4], np.array([True]))
    np.testing.assert_array_equal(calls[2][4], np.array([0], dtype=np.int32))
    np.testing.assert_array_equal(calls[3][4], np.array([[4.0]], dtype=np.float32))
    np.testing.assert_array_equal(calls[4][4], np.array([[0.25, -0.5]], dtype=np.float32))
    np.testing.assert_array_equal(calls[5][4], np.array(1, dtype=np.int8))


def test_glm_model_threads_dsa_state_separately_from_moe_topk_ids():
    from types import SimpleNamespace

    from sgl_jax.srt.layers.attention.dsa_types import DsaSelection, DsaTopKState
    from sgl_jax.srt.models.glm5_moe import Glm5Model

    state = DsaTopKState(
        selection=DsaSelection(
            logical_topk_ids=jnp.array([[0]], dtype=jnp.int32),
            physical_slots=jnp.array([[0]], dtype=jnp.int32),
            selected_counts=jnp.array([1], dtype=jnp.int32),
            producer_layer=0,
        ),
        query_offsets=jnp.array([0, 1], dtype=jnp.int32),
        request_offsets=jnp.array([0], dtype=jnp.int32),
    )

    class StubLayer:
        def __init__(self, layer_id):
            self.layer_id = layer_id
            self.seen_prev = []

        def __call__(
            self,
            positions,
            hidden_states,
            forward_batch,
            token_to_kv_pool,
            indexer_k_pool,
            residual,
            *,
            prev_dsa_state,
            dispatch_info,
        ):
            del positions, forward_batch, token_to_kv_pool, indexer_k_pool, residual, dispatch_info
            self.seen_prev.append(prev_dsa_state)
            next_state = state if self.layer_id == 0 else prev_dsa_state
            index_update = f"index-{self.layer_id}" if self.layer_id == 0 else None
            return (
                hidden_states + 1,
                None,
                f"mla-{self.layer_id}",
                jnp.array([100 + self.layer_id], dtype=jnp.int32),
                next_state,
                index_update,
            )

    layers = [StubLayer(0), StubLayer(1)]
    model = SimpleNamespace(
        embed_tokens=lambda input_ids: input_ids[:, None].astype(jnp.float32),
        layers=layers,
        norm=lambda hidden_states: hidden_states,
    )
    forward_batch = SimpleNamespace(
        input_ids=jnp.array([3], dtype=jnp.int32),
        positions=jnp.array([0], dtype=jnp.int32),
        expert_location_metadata=None,
    )

    hidden, mla_updates, moe_topk_ids, dsa_states, index_updates = Glm5Model.__call__(
        model,
        forward_batch,
        token_to_kv_pool="mla-pool",
        indexer_k_pool="index-pool",
    )

    np.testing.assert_array_equal(hidden, np.array([[5.0]], dtype=np.float32))
    assert mla_updates == ["mla-0", "mla-1"]
    np.testing.assert_array_equal(moe_topk_ids[0], np.array([100], dtype=np.int32))
    np.testing.assert_array_equal(moe_topk_ids[1], np.array([101], dtype=np.int32))
    assert dsa_states == [state, state]
    assert index_updates == ["index-0"]
    assert layers[0].seen_prev == [None]
    assert layers[1].seen_prev == [state]


def test_glm_causal_lm_returns_dual_pool_updates_without_relabeling_moe_topk():
    from types import SimpleNamespace

    from sgl_jax.srt.models.glm5_moe import Glm5ForCausalLM

    moe_topk_ids = [jnp.array([[4, 7]], dtype=jnp.int32)]

    class StubModel:
        def __init__(self):
            self.args = None

        def __call__(self, forward_batch, token_to_kv_pool, indexer_k_pool):
            self.args = (forward_batch, token_to_kv_pool, indexer_k_pool)
            return (
                jnp.array([[3.0]], dtype=jnp.float32),
                ["mla-update"],
                moe_topk_ids,
                ["dsa-state"],
                ["index-update"],
            )

    stub_model = StubModel()
    causal_lm = SimpleNamespace(
        model=stub_model,
        config=SimpleNamespace(tie_word_embeddings=False),
        lm_head="lm-head",
        logits_processor=lambda hidden, head, metadata: (hidden, head, metadata),
    )
    forward_batch = object()
    memory_pools = SimpleNamespace(
        token_to_kv_pool="mla-pool",
        indexer_k_pool="index-pool",
    )

    output, pool_updates, callback_flag, returned_topk = Glm5ForCausalLM.__call__(
        causal_lm,
        forward_batch,
        memory_pools,
        logits_metadata="logits-metadata",
    )

    assert output[1:] == ("lm-head", "logits-metadata")
    assert pool_updates == {
        "token_to_kv_pool": ["mla-update"],
        "indexer_k_pool": ["index-update"],
    }
    assert callback_flag is True
    assert returned_topk is moe_topk_ids
    assert stub_model.args == (forward_batch, "mla-pool", "index-pool")
