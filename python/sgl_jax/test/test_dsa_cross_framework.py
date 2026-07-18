import jax.numpy as jnp
import numpy as np
import pytest
import torch


INDEX_TOPK = 2048
INDEX_N_HEADS = 32
INDEX_HEAD_DIM = 128
CANDIDATE_LENGTHS = (1, 127, 128, 129, 257, 2047, 2048, 2049, 3072, 4096)
MLA_LOCAL_HEADS = 2
MLA_LATENT_DIM = 512
MLA_ROPE_DIM = 64
MLA_PAGE_SIZE = 128
MLA_CACHE_WIDTH = MLA_LATENT_DIM + 128


def _selection_fixture(candidate_length: int):
    q_index = torch.zeros(
        (1, INDEX_N_HEADS, INDEX_HEAD_DIM), dtype=torch.float32
    )
    q_index[0, 0, :12] = 2.0 ** torch.arange(12, dtype=torch.float32)
    head_weights = torch.zeros((1, INDEX_N_HEADS), dtype=torch.float32)
    head_weights[0, 0] = 1.0

    row_ids = torch.arange(candidate_length, dtype=torch.int64)
    k_index_cache = torch.zeros(
        (candidate_length, INDEX_HEAD_DIM), dtype=torch.float32
    )
    for bit in range(12):
        k_index_cache[:, bit] = ((row_ids >> bit) & 1).to(torch.float32)

    generator = torch.Generator().manual_seed(0xD5A + candidate_length)
    candidate_slots = torch.randperm(candidate_length, generator=generator).to(
        torch.int32
    )[None, :]
    candidate_logical_ids = torch.arange(candidate_length, dtype=torch.int32)[
        None, :
    ]
    candidate_counts = torch.tensor([candidate_length], dtype=torch.int32)

    # Quantize every floating source through BF16 before either framework casts
    # it to FP32 for the score computation.
    return (
        q_index.to(torch.bfloat16),
        head_weights.to(torch.bfloat16),
        k_index_cache.to(torch.bfloat16),
        candidate_slots,
        candidate_logical_ids,
        candidate_counts,
    )


def _jax_float32(value: torch.Tensor):
    return jnp.asarray(value.float().numpy(), dtype=jnp.float32)


def _boundary_selection():
    from sgl_jax.srt.kernels.dsa.torch_reference import torch_glm_dsa_select

    counts = torch.tensor(
        [0, 1, 127, 128, 129, 2047, 2048, 3, 4, 12, 4],
        dtype=torch.int32,
    )
    token_count = counts.numel()
    (
        q_index,
        head_weights,
        k_index_cache,
        _,
        _,
        _,
    ) = _selection_fixture(INDEX_TOPK)
    candidate_slots = torch.arange(INDEX_TOPK, dtype=torch.int32).repeat(
        token_count, 1
    )
    candidate_logical_ids = candidate_slots.clone()
    # The two highest-scoring entries for the duplicate-compaction row map to the
    # same logical token, so mapping must preserve the first and remove the
    # later duplicate without disturbing score order.
    candidate_logical_ids[8, 2] = 3
    candidate_logical_ids[10, 3] = INDEX_TOPK

    return torch_glm_dsa_select(
        q_index=q_index.expand(token_count, -1, -1).contiguous(),
        head_weights=head_weights.expand(token_count, -1).contiguous(),
        k_index_cache=k_index_cache,
        candidate_slots=candidate_slots,
        candidate_logical_ids=candidate_logical_ids,
        candidate_counts=counts,
        index_topk=INDEX_TOPK,
    )


def _mapping_inputs():
    generator = torch.Generator().manual_seed(0x51A7)
    request_zero = torch.randperm(INDEX_TOPK, generator=generator).to(
        torch.int32
    )
    request_zero[11] = -1
    request_one = torch.randperm(INDEX_TOPK, generator=generator).to(
        torch.int32
    ) + INDEX_TOPK
    return {
        "req_to_token_slots": torch.stack((request_zero, request_one)),
        "query_request_indices": torch.tensor(
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1], dtype=torch.int32
        ),
        "query_positions": torch.tensor(
            [0, 0, 126, 127, 128, 2046, 2047, 0, 3, 11, 3],
            dtype=torch.int32,
        ),
        "producer_layer": 7,
    }


def _mla_fixture():
    capacity = 2 * INDEX_TOPK
    page_count = capacity // MLA_PAGE_SIZE
    generator = torch.Generator().manual_seed(0xA77E)
    cache = torch.randn(
        (
            page_count,
            MLA_PAGE_SIZE // 2,
            2,
            MLA_CACHE_WIDTH,
        ),
        generator=generator,
        dtype=torch.float32,
    ).to(torch.bfloat16)
    q_latent = torch.randn(
        (11, MLA_LOCAL_HEADS, MLA_LATENT_DIM),
        generator=generator,
        dtype=torch.float32,
    ).to(torch.bfloat16)
    q_rope = torch.randn(
        (11, MLA_LOCAL_HEADS, MLA_ROPE_DIM),
        generator=generator,
        dtype=torch.float32,
    ).to(torch.bfloat16)
    return q_latent, q_rope, cache


def _torch_dense_visible_mla(
    q_latent: torch.Tensor,
    q_rope: torch.Tensor,
    cache: torch.Tensor,
    visible_slots: torch.Tensor,
    sm_scale: float,
):
    token_rows = cache.reshape(-1, cache.shape[-1]).float()
    visible_rows = token_rows[visible_slots.long()]
    keys = torch.cat(
        (
            visible_rows[:, :MLA_LATENT_DIM],
            visible_rows[:, MLA_LATENT_DIM : MLA_LATENT_DIM + MLA_ROPE_DIM],
        ),
        dim=-1,
    )
    query = torch.cat((q_latent.float(), q_rope.float()), dim=-1)
    weights = torch.softmax(torch.einsum("hc,kc->hk", query, keys) * sm_scale, dim=-1)
    return torch.einsum(
        "hk,kc->hc", weights, visible_rows[:, :MLA_LATENT_DIM]
    )


def test_torch_cpu_selection_matches_jax_across_context_boundaries():
    from sgl_jax.srt.kernels.dsa.torch_reference import torch_glm_dsa_select
    from sgl_jax.srt.models.glm5_moe import GlmDsaIndexer

    for candidate_length in CANDIDATE_LENGTHS:
        (
            q_index,
            head_weights,
            k_index_cache,
            candidate_slots,
            candidate_logical_ids,
            candidate_counts,
        ) = _selection_fixture(candidate_length)

        torch_result = torch_glm_dsa_select(
            q_index=q_index,
            head_weights=head_weights,
            k_index_cache=k_index_cache,
            candidate_slots=candidate_slots,
            candidate_logical_ids=candidate_logical_ids,
            candidate_counts=candidate_counts,
            index_topk=INDEX_TOPK,
        )
        jax_scores = np.asarray(
            GlmDsaIndexer.score_candidates(
                q_index=_jax_float32(q_index),
                head_weights=_jax_float32(head_weights),
                k_index_cache=_jax_float32(k_index_cache),
                candidate_slots=jnp.asarray(
                    candidate_slots.numpy(), dtype=jnp.int32
                ),
            )
        )
        jax_ids, jax_counts = GlmDsaIndexer.select_topk(
            q_index=_jax_float32(q_index),
            head_weights=_jax_float32(head_weights),
            k_index_cache=_jax_float32(k_index_cache),
            candidate_slots=jnp.asarray(candidate_slots.numpy(), dtype=jnp.int32),
            candidate_logical_ids=jnp.asarray(
                candidate_logical_ids.numpy(), dtype=jnp.int32
            ),
            candidate_counts=jnp.asarray(candidate_counts.numpy(), dtype=jnp.int32),
            index_topk=INDEX_TOPK,
        )
        jax_ids = np.asarray(jax_ids)
        jax_counts = np.asarray(jax_counts)

        valid_count = min(candidate_length, INDEX_TOPK)
        expected_scores = np.full((1, INDEX_TOPK), -np.inf, dtype=np.float32)
        expected_scores[0, :valid_count] = jax_scores[
            0, jax_ids[0, :valid_count]
        ]

        np.testing.assert_allclose(
            torch_result.scores.numpy(), expected_scores, rtol=1e-6, atol=1e-6
        )
        np.testing.assert_array_equal(
            torch_result.selected_counts.numpy(), jax_counts
        )
        np.testing.assert_array_equal(
            torch_result.logical_topk_ids.numpy(), jax_ids
        )

        valid_scores = np.sort(jax_scores[0])[::-1]
        assert np.unique(valid_scores).size == candidate_length
        if candidate_length > INDEX_TOPK:
            boundary_margin = (
                valid_scores[INDEX_TOPK - 1] - valid_scores[INDEX_TOPK]
            )
            assert boundary_margin >= 1e-3


def test_torch_cpu_selection_validates_tensor_abi():
    from sgl_jax.srt.kernels.dsa.torch_reference import torch_glm_dsa_select

    args = dict(
        q_index=torch.ones((1, 2, 4), dtype=torch.bfloat16),
        head_weights=torch.ones((1, 2), dtype=torch.bfloat16),
        k_index_cache=torch.ones((2, 4), dtype=torch.bfloat16),
        candidate_slots=torch.tensor([[0, 1]], dtype=torch.int32),
        candidate_logical_ids=torch.tensor([[3, 4]], dtype=torch.int32),
        candidate_counts=torch.tensor([2], dtype=torch.int32),
        index_topk=2,
    )

    with pytest.raises(ValueError, match="q_index must have rank 3"):
        torch_glm_dsa_select(**{**args, "q_index": args["q_index"][0]})
    with pytest.raises(ValueError, match="head_weights must match"):
        torch_glm_dsa_select(
            **{**args, "head_weights": torch.ones((1, 1), dtype=torch.bfloat16)}
        )
    with pytest.raises(ValueError, match="k_index_cache must have shape"):
        torch_glm_dsa_select(
            **{**args, "k_index_cache": torch.ones((2, 3), dtype=torch.bfloat16)}
        )
    with pytest.raises(ValueError, match="at least one safe slot"):
        torch_glm_dsa_select(
            **{**args, "k_index_cache": torch.empty((0, 4), dtype=torch.bfloat16)}
        )
    with pytest.raises(TypeError, match="candidate_slots must have dtype int32"):
        torch_glm_dsa_select(
            **{**args, "candidate_slots": args["candidate_slots"].long()}
        )
    with pytest.raises(TypeError, match="candidate_logical_ids must have dtype int32"):
        torch_glm_dsa_select(
            **{
                **args,
                "candidate_logical_ids": args["candidate_logical_ids"].long(),
            }
        )
    with pytest.raises(TypeError, match="candidate_counts must have dtype int32"):
        torch_glm_dsa_select(
            **{**args, "candidate_counts": args["candidate_counts"].long()}
        )
    with pytest.raises(ValueError, match="greater than one"):
        torch_glm_dsa_select(**{**args, "index_topk": 1})


def test_torch_logical_mapping_matches_jax_with_selection_output_and_compaction():
    from sgl_jax.srt.kernels.dsa.reference import (
        logical_topk_to_physical_slots,
    )
    from sgl_jax.srt.kernels.dsa.torch_reference import (
        torch_logical_topk_to_physical_slots,
    )

    topk = _boundary_selection()
    mapping_inputs = _mapping_inputs()
    torch_mapping = torch_logical_topk_to_physical_slots(
        logical_topk_ids=topk.logical_topk_ids,
        selected_counts=topk.selected_counts,
        **mapping_inputs,
    )
    jax_mapping = logical_topk_to_physical_slots(
        logical_topk_ids=jnp.asarray(
            topk.logical_topk_ids.numpy(), dtype=jnp.int32
        ),
        selected_counts=jnp.asarray(topk.selected_counts.numpy(), dtype=jnp.int32),
        req_to_token_slots=jnp.asarray(
            mapping_inputs["req_to_token_slots"].numpy(), dtype=jnp.int32
        ),
        query_request_indices=jnp.asarray(
            mapping_inputs["query_request_indices"].numpy(), dtype=jnp.int32
        ),
        query_positions=jnp.asarray(
            mapping_inputs["query_positions"].numpy(), dtype=jnp.int32
        ),
        producer_layer=mapping_inputs["producer_layer"],
    )

    np.testing.assert_array_equal(
        torch_mapping.logical_topk_ids.numpy(),
        np.asarray(jax_mapping.logical_topk_ids),
    )
    np.testing.assert_array_equal(
        torch_mapping.physical_slots.numpy(), np.asarray(jax_mapping.physical_slots)
    )
    np.testing.assert_array_equal(
        torch_mapping.selected_counts.numpy(),
        np.asarray(jax_mapping.selected_counts),
    )
    np.testing.assert_array_equal(
        torch_mapping.selected_counts.numpy(),
        np.array(
            [0, 1, 127, 128, 129, 2047, 2048, 1, 3, 11, 3],
            dtype=np.int32,
        ),
    )
    assert torch_mapping.producer_layer == jax_mapping.producer_layer == 7


def test_torch_sparse_mla_matches_jax_and_dense_with_integrated_selection():
    from sgl_jax.srt.kernels.dsa.reference import dsa_sparse_mla_reference
    from sgl_jax.srt.kernels.dsa.torch_reference import (
        torch_dsa_sparse_mla,
        torch_logical_topk_to_physical_slots,
    )

    topk = _boundary_selection()
    mapping = torch_logical_topk_to_physical_slots(
        logical_topk_ids=topk.logical_topk_ids,
        selected_counts=topk.selected_counts,
        **_mapping_inputs(),
    )
    q_latent, q_rope, cache = _mla_fixture()
    sm_scale = (MLA_LATENT_DIM + MLA_ROPE_DIM) ** -0.5

    torch_output = torch_dsa_sparse_mla(
        q_latent,
        q_rope,
        cache,
        mapping.physical_slots,
        mapping.selected_counts,
        sm_scale=sm_scale,
        page_size=MLA_PAGE_SIZE,
        latent_dim=MLA_LATENT_DIM,
        rope_dim=MLA_ROPE_DIM,
    )
    jax_output = dsa_sparse_mla_reference(
        _jax_float32(q_latent),
        _jax_float32(q_rope),
        _jax_float32(cache),
        jnp.asarray(mapping.physical_slots.numpy(), dtype=jnp.int32),
        jnp.asarray(mapping.selected_counts.numpy(), dtype=jnp.int32),
        sm_scale=sm_scale,
        page_size=MLA_PAGE_SIZE,
        latent_dim=MLA_LATENT_DIM,
        rope_dim=MLA_ROPE_DIM,
    )

    assert torch_output.shape == (11, MLA_LOCAL_HEADS, MLA_LATENT_DIM)
    assert torch_output.dtype == torch.float32
    np.testing.assert_allclose(
        torch_output.numpy(), np.asarray(jax_output), rtol=2e-5, atol=2e-5
    )

    dense_visible = _torch_dense_visible_mla(
        q_latent[6],
        q_rope[6],
        cache,
        mapping.physical_slots[6],
        sm_scale,
    )
    torch.testing.assert_close(
        torch_output[6], dense_visible, rtol=1e-5, atol=1e-5
    )

    arbitrary_padding = mapping.physical_slots.clone()
    valid = (
        torch.arange(INDEX_TOPK)[None, :]
        < mapping.selected_counts[:, None]
    )
    arbitrary_padding[~valid] = -987654
    padding_output = torch_dsa_sparse_mla(
        q_latent,
        q_rope,
        cache,
        arbitrary_padding,
        mapping.selected_counts,
        sm_scale=sm_scale,
        page_size=MLA_PAGE_SIZE,
        latent_dim=MLA_LATENT_DIM,
        rope_dim=MLA_ROPE_DIM,
    )
    torch.testing.assert_close(torch_output, padding_output, rtol=0, atol=0)

    invalid_counted_slots = mapping.physical_slots.clone()
    invalid_counted_slots[1, 0] = cache.shape[0] * MLA_PAGE_SIZE
    with pytest.raises(ValueError, match="counted physical_slots"):
        torch_dsa_sparse_mla(
            q_latent,
            q_rope,
            cache,
            invalid_counted_slots,
            mapping.selected_counts,
            sm_scale=sm_scale,
            page_size=MLA_PAGE_SIZE,
            latent_dim=MLA_LATENT_DIM,
            rope_dim=MLA_ROPE_DIM,
        )
