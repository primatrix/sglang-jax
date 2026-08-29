from types import SimpleNamespace

import jax
import numpy as np

from sgl_jax.srt.layers.attention.flashattention_backend import _pad_page_indices
from sgl_jax.srt.managers.schedule_batch import ScheduleBatch
from sgl_jax.srt.speculative.dflash_info import DFlashDraftInput, _mask_draft_kv_writes
from sgl_jax.srt.speculative.dflash_worker import (
    _DFLASH_CONDITION_SOURCES,
    _DFLASH_FEEDBACK_ALL_SOURCES,
    _DFLASH_MARGIN_THRESHOLDS,
    _DFLASH_PREDICTOR_POLICIES,
    DFlashWorker,
)


def _bare_worker(**attrs):
    w = object.__new__(DFlashWorker)
    for k, v in attrs.items():
        object.__setattr__(w, k, v)
    return w


def _feedback_worker(block_size: int):
    width = block_size - 1
    return _bare_worker(
        block_size=block_size,
        _feedback_shadow_batches=0,
        _feedback_shadow_rounds=0,
        _feedback_shadow_stats={
            source: {
                metric: np.zeros((width,), dtype=np.int64)
                for metric in (
                    "valid",
                    "draft_reuse",
                    "target_match",
                    "target_novel",
                    "draft_target_match",
                    "accepted_chain",
                )
            }
            for source in _DFLASH_FEEDBACK_ALL_SOURCES
        },
        _feedback_margin_stats={
            source: {
                metric: np.zeros((len(_DFLASH_MARGIN_THRESHOLDS) + 1,), dtype=np.int64)
                for metric in (
                    "valid",
                    "alternative",
                    "target_match",
                    "target_novel",
                    "base_target",
                )
            }
            for source in _DFLASH_FEEDBACK_ALL_SOURCES
            if source != "target_correction"
        },
        _feedback_first_rejection_stats={
            source: {
                metric: np.zeros((len(_DFLASH_MARGIN_THRESHOLDS) + 1,), dtype=np.int64)
                for metric in (
                    "valid",
                    "alternative",
                    "candidate_target",
                    "base_target",
                )
            }
            for source in _DFLASH_FEEDBACK_ALL_SOURCES
            if source != "target_correction"
        },
        _feedback_condition_stats={
            source: {
                metric: np.zeros((block_size + 1, width), dtype=np.int64)
                for metric in ("valid", "target_match")
            }
            for source in _DFLASH_CONDITION_SOURCES
        },
        _feedback_oracle_rejected_rounds=0,
        _feedback_oracle_repair_rounds=0,
        _feedback_oracle_local_novel=0,
        _feedback_oracle_rejection_position=np.zeros((width,), dtype=np.int64),
        _feedback_oracle_repair_position=np.zeros((width,), dtype=np.int64),
        _feedback_oracle_source_repairs={
            source: 0 for source in ("rejected_draft", "stale_suffix", "historical_ngram")
        },
        _feedback_oracle_agreement_repairs=0,
        _feedback_predictor_stats={
            policy: {
                "predictions": 0,
                "rejected_predictions": 0,
                "position_hits": 0,
                "candidate_target": 0,
                "repairs": 0,
                "harms": 0,
                "neutral": 0,
                "accept_gain": 0,
                "accept_loss": 0,
                "accept_delta": 0,
                "selected_position": np.zeros((width,), dtype=np.int64),
                "hit_position": np.zeros((width,), dtype=np.int64),
            }
            for policy in _DFLASH_PREDICTOR_POLICIES
        },
    )


def _redenoise_worker(block_size: int):
    return _bare_worker(
        block_size=block_size,
        draft_block_size=block_size - 1,
        _redenoise_stats_batches=0,
        _redenoise_stats_rounds=0,
        _redenoise_stats_changed=0,
        _redenoise_stats_repairs=0,
        _redenoise_stats_harms=0,
        _redenoise_stats_base_accept=0,
        _redenoise_stats_final_accept=0,
        _redenoise_stats_accept_delta=0,
        _redenoise_stats_prefix_hist=np.zeros((block_size - 1,), dtype=np.int64),
    )


def test_redenoise_stats_separate_repairs_harms_and_accept_delta():
    worker = _redenoise_worker(block_size=4)
    worker._record_redenoise_stats(
        accept_lens=np.array([4, 2], dtype=np.int32),
        base_draft_token=np.array([[10, 1, 2, 3], [20, 4, 5, 6]], dtype=np.int32),
        final_draft_token=np.array([[10, 1, 9, 3], [20, 4, 8, 6]], dtype=np.int32),
        target_predict_flat=np.array([[1, 9, 3, 99], [4, 5, 6, 99]], dtype=np.int32),
        prefix_lens=np.array([1, 1], dtype=np.int32),
        selector=np.array([0, 1], dtype=np.int32),
    )

    assert worker._redenoise_stats_rounds == 2
    assert worker._redenoise_stats_changed == 2
    assert worker._redenoise_stats_repairs == 1
    assert worker._redenoise_stats_harms == 1
    assert worker._redenoise_stats_base_accept == 6
    assert worker._redenoise_stats_final_accept == 6
    assert worker._redenoise_stats_accept_delta == 0
    np.testing.assert_array_equal(worker._redenoise_stats_prefix_hist, [0, 2, 0])


def test_prefill_draft_extend_metadata_preserves_dp_rank_sections():
    # DP=2, four token rows per rank. Rank-local padding must stay between
    # rank 0's real rows and rank 1's real rows.
    mwb = SimpleNamespace(
        positions=np.array([5, 6, 0, 0, 9, 0, 0, 0], dtype=np.int32),
        out_cache_loc=np.array([20, 21, -1, -1, 40, -1, -1, -1], dtype=np.int32),
    )
    target_hidden = np.zeros((8, 16), dtype=np.float32)

    positions, cache_loc = DFlashWorker._prefill_draft_extend_metadata(mwb, target_hidden)

    np.testing.assert_array_equal(positions, mwb.positions)
    np.testing.assert_array_equal(cache_loc, mwb.out_cache_loc)


def test_prefill_draft_extend_metadata_rejects_bucket_mismatch():
    mwb = SimpleNamespace(
        positions=np.array([0, 1, 2], dtype=np.int32),
        out_cache_loc=np.array([10, 11, 12], dtype=np.int32),
    )
    target_hidden = np.zeros((2, 16), dtype=np.float32)

    with np.testing.assert_raises_regex(ValueError, "must match the target hidden bucket"):
        DFlashWorker._prefill_draft_extend_metadata(mwb, target_hidden)


def test_draft_extend_masks_unaccepted_and_padded_rows():
    cache_loc = np.arange(12, dtype=np.int32)
    masked = _mask_draft_kv_writes(
        jax.numpy.asarray(cache_loc),
        jax.numpy.asarray([2, 4, 3], dtype=jax.numpy.int32),
        jax.numpy.asarray([True, False, True]),
    )

    np.testing.assert_array_equal(
        np.asarray(masked),
        np.array([0, 1, -1, -1, -1, -1, -1, -1, 8, 9, 10, -1], dtype=np.int32),
    )


def test_verify_bucket_template_is_cached_by_active_slots():
    mesh = jax.sharding.Mesh(np.asarray(jax.devices()).reshape(1, 1), ("data", "tensor"))
    worker = _bare_worker(
        block_size=4,
        mesh=mesh,
        _verify_bucket_templates={},
    )
    mwb = SimpleNamespace(
        dp_size=1,
        per_dp_bs_size=4,
        real_bs=2,
        logits_indices_selector=np.array([0, 2], dtype=np.int32),
    )

    first = worker._get_verify_bucket_template(mwb, bs=4)
    second = worker._get_verify_bucket_template(mwb, bs=4)

    assert first is second
    np.testing.assert_array_equal(first.extend_seq_lens, np.array([4, 0, 4, 0]))
    np.testing.assert_array_equal(np.asarray(first.cu_q_lens), np.array([0, 4, 4, 8, 8]))
    np.testing.assert_array_equal(
        np.asarray(first.active_mask), np.array([True, False, True, False])
    )
    np.testing.assert_array_equal(np.asarray(first.distribution), np.array([0, 2, 2]))


def test_verify_bucket_template_distinguishes_draft_and_anchor_verify_widths():
    mesh = jax.sharding.Mesh(np.asarray(jax.devices()).reshape(1, 1), ("data", "tensor"))
    worker = _bare_worker(
        block_size=8,
        draft_block_size=7,
        mesh=mesh,
        _verify_bucket_templates={},
    )
    mwb = SimpleNamespace(
        dp_size=1,
        per_dp_bs_size=2,
        real_bs=2,
        logits_indices_selector=np.array([0, 1], dtype=np.int32),
    )

    draft = worker._get_verify_bucket_template(mwb, bs=2, token_num=7)
    target = worker._get_verify_bucket_template(mwb, bs=2, token_num=8)

    np.testing.assert_array_equal(draft.extend_seq_lens, np.array([7, 7]))
    np.testing.assert_array_equal(target.extend_seq_lens, np.array([8, 8]))
    assert draft is not target


def test_build_page_indices_preserves_dp_rank_sections():
    req_to_token = np.arange(200, dtype=np.int32).reshape(10, 20)
    w = _bare_worker(
        req_to_token_pool=SimpleNamespace(req_to_token=req_to_token),
        block_size=2,
        page_size=1,
        _page_indices_pool_capacity=16,
        _page_indices_per_seq_capacity=4,
    )
    mwb = SimpleNamespace(
        req_pool_indices=np.array([1, 2, 3, 4], dtype=np.int32),
        logits_indices_selector=np.arange(4, dtype=np.int32),
        dp_size=2,
        per_dp_bs_size=2,
    )

    page_indices = w._build_dflash_page_indices(
        mwb,
        np.array([1, 1, 1, 1], dtype=np.int32),
        bs=4,
    )

    np.testing.assert_array_equal(
        page_indices.reshape(2, 8),
        np.array(
            [
                [20, 21, 22, 40, 41, 42, 0, 0],
                [60, 61, 62, 80, 81, 82, 0, 0],
            ],
            dtype=np.int32,
        ),
    )


def test_build_page_indices_handles_uneven_dp_ranks():
    req_to_token = np.arange(160, dtype=np.int32).reshape(8, 20)
    w = _bare_worker(
        req_to_token_pool=SimpleNamespace(req_to_token=req_to_token),
        block_size=2,
        page_size=1,
        _page_indices_pool_capacity=16,
        _page_indices_per_seq_capacity=4,
    )
    mwb = SimpleNamespace(
        req_pool_indices=np.array([1, 2, 0, 4, 0, 0], dtype=np.int32),
        logits_indices_selector=np.array([0, 1, 3], dtype=np.int32),
        dp_size=2,
        per_dp_bs_size=3,
    )

    page_indices = w._build_dflash_page_indices(
        mwb,
        np.array([2, 1, 0, 3, 0, 0], dtype=np.int32),
        bs=6,
    )

    np.testing.assert_array_equal(
        page_indices.reshape(2, 8),
        np.array(
            [
                [20, 21, 22, 23, 40, 41, 42, 0],
                [80, 81, 82, 83, 84, 0, 0, 0],
            ],
            dtype=np.int32,
        ),
    )


def test_unpad_draft_state_removes_dp_padding_but_keeps_new_seq_lens():
    di = DFlashDraftInput(
        verified_id=np.array([10, 20, 0, 30, 0, 0], dtype=np.int32),
        target_hidden=None,
        ctx_lens=np.array([1, 2, 0, 3, 0, 0], dtype=np.int32),
        draft_seq_lens=np.array([5, 6, 0, 7, 0, 0], dtype=np.int32),
    )
    di.new_seq_lens = np.array([6, 8, 0, 10, 0, 0], dtype=np.int32)

    DFlashWorker._unpad_draft_state(
        di,
        np.array([0, 1, 3], dtype=np.int32),
    )

    np.testing.assert_array_equal(di.verified_id, np.array([10, 20, 30], dtype=np.int32))
    np.testing.assert_array_equal(di.ctx_lens, np.array([1, 2, 3], dtype=np.int32))
    np.testing.assert_array_equal(di.draft_seq_lens, np.array([5, 6, 7], dtype=np.int32))
    np.testing.assert_array_equal(di.new_seq_lens, np.array([6, 8, 0, 10, 0, 0]))


def test_ngram_metadata_scatter_preserves_dp_padded_slot_alignment():
    flat = DFlashDraftInput(
        verified_id=np.array([10, 20, 30], dtype=np.int32),
        ctx_lens=np.array([1, 2, 3], dtype=np.int32),
        draft_seq_lens=np.array([11, 12, 13], dtype=np.int32),
        ngram_token_ids=np.array([[101, 102], [201, 202], [301, 302]], dtype=np.int32),
        ngram_bonus=np.array([[1.0, 0.5], [2.0, 1.0], [3.0, 1.5]], dtype=np.float32),
        ngram_valid_mask=np.ones((3, 2), dtype=np.bool_),
        ngram_match_lens=np.array([3, 4, 5], dtype=np.int32),
        rejected_draft_token_ids=np.array([91, 92, 93], dtype=np.int32),
        rejection_valid_mask=np.array([True, False, True]),
        previous_accept_lens=np.array([2, 3, 1], dtype=np.int32),
        enable_ngram=True,
        block_size=3,
    )

    padded = ScheduleBatch._scatter_spec_info_to_dp_slots(
        flat,
        selector=np.array([0, 1, 3], dtype=np.int32),
        total_bs=6,
    )

    np.testing.assert_array_equal(
        padded.ngram_token_ids,
        np.array(
            [
                [101, 102],
                [201, 202],
                [0, 0],
                [301, 302],
                [0, 0],
                [0, 0],
            ],
            dtype=np.int32,
        ),
    )
    np.testing.assert_array_equal(padded.ngram_match_lens, np.array([3, 4, 0, 5, 0, 0]))
    np.testing.assert_array_equal(
        padded.rejected_draft_token_ids,
        np.array([91, 92, 0, 93, 0, 0], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        padded.rejection_valid_mask,
        np.array([True, False, False, True, False, False]),
    )
    np.testing.assert_array_equal(
        padded.previous_accept_lens,
        np.array([2, 3, 0, 1, 0, 0], dtype=np.int32),
    )
    assert padded.enable_ngram


def test_ngram_metadata_concat_preserves_rows_and_runtime_config():
    ranks = [
        DFlashDraftInput(
            verified_id=np.array([10], dtype=np.int32),
            ctx_lens=np.array([1], dtype=np.int32),
            draft_seq_lens=np.array([11], dtype=np.int32),
            ngram_token_ids=np.array([[101, 102]], dtype=np.int32),
            ngram_bonus=np.array([[1.0, 0.5]], dtype=np.float32),
            ngram_valid_mask=np.ones((1, 2), dtype=np.bool_),
            ngram_match_lens=np.array([3], dtype=np.int32),
            rejected_draft_token_ids=np.array([91], dtype=np.int32),
            rejection_valid_mask=np.array([True]),
            previous_accept_lens=np.array([2], dtype=np.int32),
            enable_ngram=True,
            ngram_min_match=2,
            ngram_max_match=6,
            ngram_base_bonus=1.5,
            block_size=3,
        ),
        DFlashDraftInput(
            verified_id=np.array([20], dtype=np.int32),
            ctx_lens=np.array([2], dtype=np.int32),
            draft_seq_lens=np.array([12], dtype=np.int32),
            ngram_token_ids=np.array([[201, 202]], dtype=np.int32),
            ngram_bonus=np.array([[2.0, 1.0]], dtype=np.float32),
            ngram_valid_mask=np.ones((1, 2), dtype=np.bool_),
            ngram_match_lens=np.array([4], dtype=np.int32),
            rejected_draft_token_ids=np.array([92], dtype=np.int32),
            rejection_valid_mask=np.array([False]),
            previous_accept_lens=np.array([3], dtype=np.int32),
            enable_ngram=True,
            ngram_min_match=2,
            ngram_max_match=6,
            ngram_base_bonus=1.5,
            block_size=3,
        ),
    ]

    flat = ScheduleBatch._concat_spec_info_per_rank(ranks)

    np.testing.assert_array_equal(flat.ngram_token_ids, np.array([[101, 102], [201, 202]]))
    np.testing.assert_array_equal(flat.ngram_match_lens, np.array([3, 4]))
    np.testing.assert_array_equal(flat.rejected_draft_token_ids, np.array([91, 92]))
    np.testing.assert_array_equal(flat.rejection_valid_mask, np.array([True, False]))
    np.testing.assert_array_equal(flat.previous_accept_lens, np.array([2, 3]))
    assert flat.enable_ngram
    assert flat.ngram_min_match == 2
    assert flat.ngram_max_match == 6
    assert flat.ngram_base_bonus == 1.5


def test_record_ngram_stats_separates_candidate_match_from_chain_acceptance():
    worker = _bare_worker(
        block_size=3,
        _ngram_stats_batches=0,
        _ngram_stats_rounds=0,
        _ngram_stats_covered_rounds=0,
        _ngram_stats_covered=0,
        _ngram_stats_selected=0,
        _ngram_stats_selected_accepted=0,
        _ngram_stats_candidate_matches=0,
        _ngram_stats_match_len_hist=np.zeros((9,), dtype=np.int64),
        _ngram_stats_position_covered=np.zeros((2,), dtype=np.int64),
        _ngram_stats_position_selected=np.zeros((2,), dtype=np.int64),
        _ngram_stats_position_accepted=np.zeros((2,), dtype=np.int64),
    )

    worker._record_ngram_stats(
        accept_lens=np.array([2, 0, 1], dtype=np.int32),
        selected_mask=np.array([[True, True], [False, False], [True, False]]),
        candidate_ids=np.array([[7, 8], [0, 0], [9, 0]], dtype=np.int32),
        valid_mask=np.array([[True, True], [False, False], [True, False]]),
        match_lens=np.array([3, 0, 4], dtype=np.int32),
        target_predict_flat=np.array([[7, 8, 0], [0, 0, 0], [9, 0, 0]], dtype=np.int32),
        selector=np.array([0, 2], dtype=np.int32),
    )

    assert worker._ngram_stats_rounds == 2
    assert worker._ngram_stats_covered_rounds == 2
    assert worker._ngram_stats_covered == 3
    assert worker._ngram_stats_selected == 3
    assert worker._ngram_stats_selected_accepted == 1
    assert worker._ngram_stats_candidate_matches == 3
    np.testing.assert_array_equal(
        worker._ngram_stats_match_len_hist,
        np.array([0, 0, 0, 1, 1, 0, 0, 0, 0], dtype=np.int64),
    )


def test_feedback_shadow_separates_reuse_novel_target_and_accepted_chain():
    worker = _feedback_worker(block_size=4)

    worker._record_feedback_shadow_stats(
        accept_lens=np.array([2, 0, 1], dtype=np.int32),
        draft_token=np.array(
            [[10, 11, 12, 13], [0, 0, 0, 0], [30, 31, 32, 33]],
            dtype=np.int32,
        ),
        target_predict_flat=np.array(
            [[11, 99, 13, 0], [0, 0, 0, 0], [40, 32, 33, 0]],
            dtype=np.int32,
        ),
        rejected_draft_token_ids=np.array([11, 0, 31], dtype=np.int32),
        rejection_valid_mask=np.array([True, False, True]),
        target_correction_token_ids=np.array([99, 0, 40], dtype=np.int32),
        stale_suffix_token_ids=np.array([[11, 99, 13], [0, 0, 0], [31, 32, 44]], dtype=np.int32),
        stale_suffix_valid_mask=np.array(
            [[True, True, True], [False, False, False], [True, True, True]]
        ),
        ngram_token_ids=np.array([[20, 12, 13], [0, 0, 0], [40, 32, 0]], dtype=np.int32),
        ngram_valid_mask=np.array([[True, True, True], [False, False, False], [True, True, False]]),
        ngram_match_lens=np.array([3, 0, 1], dtype=np.int32),
        previous_accept_lens=np.array([3, 0, 1], dtype=np.int32),
        candidate_margins=np.zeros((3, 3, 3), dtype=np.float32),
        selector=np.array([0, 2], dtype=np.int32),
    )

    rejected = worker._feedback_shadow_stats["rejected_draft"]
    assert rejected["valid"].sum() == 6
    assert rejected["draft_reuse"].sum() == 2
    assert rejected["target_match"].sum() == 1
    assert rejected["draft_target_match"].sum() == 1
    assert rejected["accepted_chain"].sum() == 1

    correction = worker._feedback_shadow_stats["target_correction"]
    assert correction["target_match"].sum() == 2
    assert correction["target_novel"].sum() == 2
    assert correction["draft_reuse"].sum() == 0

    stale = worker._feedback_shadow_stats["stale_suffix"]
    assert stale["target_match"].sum() == 4
    assert stale["draft_reuse"].sum() == 4
    assert stale["draft_target_match"].sum() == 3
    assert stale["accepted_chain"].sum() == 1

    assert worker._feedback_shadow_stats["ngram_len1"]["target_novel"].sum() == 1
    assert worker._feedback_shadow_stats["ngram_len3plus"]["target_match"].sum() == 1
    assert worker._feedback_shadow_stats["agree_rejected_stale"]["valid"].sum() == 2
    assert worker._feedback_shadow_stats["agree_stale_ngram"]["target_match"].sum() == 2
    assert worker._feedback_margin_stats["ngram_len3plus"]["target_match"][0] == 1
    assert worker._feedback_margin_stats["ngram_len3plus"]["alternative"][0] == 1
    assert worker._feedback_margin_stats["ngram_len3plus"]["base_target"][0] == 1
    assert worker._feedback_first_rejection_stats["ngram_len1"]["valid"][0] == 1
    assert worker._feedback_first_rejection_stats["ngram_len1"]["alternative"][0] == 1
    assert worker._feedback_first_rejection_stats["ngram_len1"]["candidate_target"][0] == 1
    assert worker._feedback_first_rejection_stats["ngram_len1"]["base_target"][0] == 0
    assert worker._feedback_first_rejection_stats["ngram_len3plus"]["valid"][0] == 1
    assert worker._feedback_first_rejection_stats["ngram_len3plus"]["alternative"][0] == 0
    assert worker._feedback_condition_stats["rejected_draft"]["valid"][3].sum() == 3
    assert worker._feedback_oracle_rejected_rounds == 2
    assert worker._feedback_oracle_repair_rounds == 2
    assert worker._feedback_oracle_local_novel == 2
    assert worker._feedback_oracle_source_repairs == {
        "rejected_draft": 0,
        "stale_suffix": 1,
        "historical_ngram": 1,
    }
    for counters in worker._feedback_predictor_stats.values():
        assert counters["predictions"] == 1
        assert counters["position_hits"] == 0
        assert counters["repairs"] == 0
        assert counters["harms"] == 1
        assert counters["accept_delta"] == -1


def test_feedback_predictor_compares_single_position_policies_counterfactually():
    worker = _feedback_worker(block_size=4)
    draft = np.array([[10, 20, 30]], dtype=np.int32)
    target = np.array([[10, 99, 30]], dtype=np.int32)
    ngram = np.array([[11, 99, 31]], dtype=np.int32)
    margins = np.zeros((1, 3, 3), dtype=np.float32)
    margins[..., 0] = np.array([[1.0, 0.1, 0.5]], dtype=np.float32)
    margins[..., 2] = np.array([[0.1, 0.4, 0.2]], dtype=np.float32)

    worker._record_feedback_predictor_stats(
        draft=draft,
        target=target,
        ngram_ids=ngram,
        ngram_valid=np.ones((1, 3), dtype=np.bool_),
        match_lens=np.array([3], dtype=np.int32),
        previous_accept_lens=np.array([2], dtype=np.int32),
        candidate_margins=margins,
        sparse_candidate_valid=np.array(
            [[[True, False], [True, False], [True, False]]], dtype=np.bool_
        ),
    )

    for policy in ("feedback_uncertainty", "combined_margin", "lagged_accept"):
        counters = worker._feedback_predictor_stats[policy]
        assert counters["position_hits"] == 1
        assert counters["repairs"] == 1
        assert counters["accept_gain"] == 2
        assert counters["accept_delta"] == 2
    for policy in ("earliest", "ngram_competition"):
        counters = worker._feedback_predictor_stats[policy]
        assert counters["position_hits"] == 0
        assert counters["harms"] == 1
        assert counters["accept_loss"] == 1
        assert counters["accept_delta"] == -1


def test_verify_write_cache_loc_selects_valid_half_per_dp_rank():
    w = _bare_worker(block_size=2)
    batch = SimpleNamespace(
        dp_size=2,
        per_dp_bs_size=2,
        out_cache_loc=np.array(
            [1, 2, 3, 4, -1, -1, -1, -1, 5, 6, 7, 8, -1, -1, -1, -1],
            dtype=np.int32,
        ),
    )

    np.testing.assert_array_equal(
        w._verify_write_cache_loc(batch),
        np.arange(1, 9, dtype=np.int32),
    )


def test_verify_write_cache_loc_can_keep_anchor_verify_width():
    w = _bare_worker(block_size=8)
    batch = SimpleNamespace(
        dp_size=1,
        per_dp_bs_size=2,
        out_cache_loc=np.concatenate(
            [np.arange(1, 17, dtype=np.int32), np.full(16, -1, dtype=np.int32)]
        ),
    )

    selected = w._verify_write_cache_loc(batch, token_num=8)

    np.testing.assert_array_equal(selected, np.arange(1, 17, dtype=np.int32))


def test_trim_draft_state_drops_stale_tail():
    w = _bare_worker()
    di = DFlashDraftInput(
        verified_id=np.array([10, 20, 30, 40], dtype=np.int32),
        target_hidden=None,
        ctx_lens=np.array([0, 0, 0, 0], dtype=np.int32),
        draft_seq_lens=np.array([5, 6, 7, 8], dtype=np.int32),
    )

    w._trim_draft_state_to_bs(di, bs=3)

    np.testing.assert_array_equal(di.verified_id, np.array([10, 20, 30], dtype=np.int32))
    np.testing.assert_array_equal(di.ctx_lens, np.array([0, 0, 0], dtype=np.int32))
    np.testing.assert_array_equal(di.draft_seq_lens, np.array([5, 6, 7], dtype=np.int32))


def test_page_indices_capacity_is_bounded_by_request_and_pool():
    w = _bare_worker(
        _page_indices_pool_capacity=8192,
        _page_indices_per_seq_capacity=1024,
    )

    assert w._page_indices_capacity(1) == 1024
    assert w._page_indices_capacity(4) == 4096
    assert w._page_indices_capacity(16) == 8192


def test_prefill_precompile_variants_use_runtime_extend_buckets():
    manager = SimpleNamespace(
        max_padded_batch_size=128,
        token_buckets=[64, 128, 256, 1024, 2048],
    )

    assert DFlashWorker._prefill_precompile_variants(manager) == [
        (128, 128),
        (128, 256),
        (128, 1024),
        (128, 2048),
    ]


def test_build_page_indices_reads_noncontiguous_physical_pages():
    req_to_token = np.array(
        [
            np.arange(8),
            [20, 21, 40, 41, 60, 61, 80, 81],
            [100, 101, 120, 121, 140, 141, 160, 161],
        ],
        dtype=np.int32,
    )
    w = _bare_worker(
        req_to_token_pool=SimpleNamespace(req_to_token=req_to_token),
        block_size=2,
        page_size=2,
        _page_indices_pool_capacity=8,
        _page_indices_per_seq_capacity=4,
    )
    mwb = SimpleNamespace(
        req_pool_indices=np.array([1, 2], dtype=np.int32),
        logits_indices_selector=np.array([0, 1], dtype=np.int32),
        dp_size=1,
        per_dp_bs_size=2,
    )

    page_indices = w._build_dflash_page_indices(
        mwb,
        np.array([3, 2], dtype=np.int32),
        bs=2,
    )

    np.testing.assert_array_equal(
        page_indices,
        np.array([10, 20, 30, 50, 60, 0, 0, 0], dtype=np.int32),
    )


def test_pad_page_indices_uses_fixed_dflash_capacity():
    page_indices = np.array([3, 5, 7], dtype=np.int32)

    padded = _pad_page_indices(page_indices, max_num_seqs=2, fixed_capacity=8)

    np.testing.assert_array_equal(
        padded,
        np.array([3, 5, 7, 0, 0, 0, 0, 0], dtype=np.int32),
    )


def test_pad_page_indices_rejects_fixed_capacity_overflow():
    with np.testing.assert_raises_regex(ValueError, "exceed fixed capacity"):
        _pad_page_indices(
            np.arange(9, dtype=np.int32),
            max_num_seqs=2,
            fixed_capacity=8,
        )
