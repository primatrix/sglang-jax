from __future__ import annotations

import contextlib
import copy
import logging
import os
from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.layers.logits_processor import LogitsMetadata
from sgl_jax.srt.managers.schedule_batch import ModelWorkerBatch
from sgl_jax.srt.managers.tp_worker import ModelWorker
from sgl_jax.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sgl_jax.srt.speculative.base_worker import BaseDraftWorker, BaseSpecWorker
from sgl_jax.srt.speculative.dflash_info import (
    DFlashDraftInput,
    DFlashVerifyInput,
    _gather_dflash_vocab_scores,
    _mask_draft_kv_writes,
    build_dflash_draft_block,
    build_dflash_flashback_feedback,
    build_dflash_redenoise_block,
    build_dflash_rejection_feedback,
    dflash_greedy_verify,
    dflash_sharded_top_k,
    dflash_top2_margins,
    merge_dflash_redenoise_tokens,
    select_dflash_redenoise_prefix_lens,
    select_dflash_proposal_hidden,
    select_dflash_ngram_tokens,
    select_dflash_flashback_tokens,
)
from sgl_jax.srt.speculative.dflash_util import (
    parse_dflash_draft_config,
    resolve_mask_token_id,
)
from sgl_jax.srt.speculative.relay_buffer import (
    create_dflash_relay_buffers,
    gather_dflash_relay_buffers,
    make_dp_valid_mask,
    update_dflash_relay_buffers,
)
from sgl_jax.srt.speculative.spec_info import SpeculativeAlgorithm

logger = logging.getLogger(__name__)

_DFLASH_FEEDBACK_SHADOW_SOURCES = (
    "rejected_draft",
    "target_correction",
    "stale_suffix",
    "ngram_len1",
    "ngram_len2",
    "ngram_len3plus",
)
_DFLASH_FEEDBACK_AGREEMENT_SOURCES = (
    "agree_rejected_stale",
    "agree_rejected_ngram",
    "agree_stale_ngram",
    "agree_all_three",
)
_DFLASH_FEEDBACK_ALL_SOURCES = _DFLASH_FEEDBACK_SHADOW_SOURCES + _DFLASH_FEEDBACK_AGREEMENT_SOURCES
_DFLASH_MARGIN_THRESHOLDS = np.asarray((0.25, 0.5, 1.0, 2.0, 4.0, 8.0), dtype=np.float32)
_DFLASH_CONDITION_SOURCES = ("rejected_draft", "stale_suffix", "ngram_len3plus")
_DFLASH_PREDICTOR_POLICIES = (
    "earliest",
    "feedback_uncertainty",
    "ngram_competition",
    "combined_margin",
    "lagged_accept",
)
_DFLASH_PREDICTOR_NGRAM_MARGIN = 0.5


@dataclass(frozen=True)
class DFlashVerifyBucketTemplate:
    extend_seq_lens: np.ndarray
    cu_q_lens: jax.Array
    active_mask: jax.Array
    distribution: jax.Array


@dataclass(frozen=True)
class DraftForwardPlan:
    forward_batch: ForwardBatch
    forward_metadata: object
    seq_lens: np.ndarray
    target_prefix_lens: np.ndarray
    positions_host: np.ndarray
    page_indices: np.ndarray
    allocated_lens: jax.Array
    reservation_base_lens: jax.Array
    relay_future_indices: jax.Array
    relay_valid_mask: jax.Array
    flashback_token_ids: jax.Array
    flashback_target_margins: jax.Array
    flashback_valid_mask: jax.Array
    rejected_draft_token_ids: jax.Array
    rejection_valid_mask: jax.Array
    previous_accept_lens: jax.Array
    target_correction_token_ids: jax.Array
    ngram_token_ids: jax.Array
    ngram_bonus: jax.Array
    ngram_valid_mask: jax.Array
    ngram_match_lens: jax.Array
    use_relay_state: bool
    dp_size: int
    bs: int


@dataclass(frozen=True)
class TargetVerifyPlan:
    model_worker_batch: ModelWorkerBatch
    forward_batch: ForwardBatch
    forward_metadata: object
    logits_metadata: LogitsMetadata
    seq_lens: np.ndarray
    target_prefix_lens: np.ndarray
    resolved_target_prefix_lens: jax.Array
    draft_extend_positions: jax.Array
    draft_extend_cache_loc: jax.Array
    active_mask: jax.Array
    allocated_lens: jax.Array
    relay_future_indices: jax.Array
    relay_valid_mask: jax.Array
    draft_token: jax.Array
    base_draft_token: jax.Array
    top2_draft_token: jax.Array
    top2_margins: jax.Array
    redenoise_candidate_token: jax.Array
    redenoise_prefix_lens: jax.Array
    flashback_token_ids: jax.Array
    flashback_valid_mask: jax.Array
    rejected_draft_token_ids: jax.Array
    rejection_valid_mask: jax.Array
    previous_accept_lens: jax.Array
    target_correction_token_ids: jax.Array
    candidate_margins: jax.Array
    ngram_selected_mask: jax.Array
    ngram_token_ids: jax.Array
    ngram_valid_mask: jax.Array
    ngram_match_lens: jax.Array
    update_relay: bool = False


class DFlashWorker(BaseSpecWorker, BaseDraftWorker):
    """DFlash draft/verify runtime worker (greedy, DP/TP aware)."""

    def __init__(self, server_args, target_worker: ModelWorker):
        super().__init__(
            server_args,
            target_worker,
            self,
        )
        self.draft_block_size = self.speculative_num_draft_tokens
        self.enable_anchor = bool(server_args.enable_dflash_anchor)
        self.block_size = self.draft_block_size + int(self.enable_anchor)
        self.speculative_verify_token_num = self.block_size
        self._ngram_enabled = bool(server_args.enable_dflash_ngram)
        self._ngram_max_rerank_positions = int(server_args.dflash_ngram_max_rerank_positions)
        self._feedback_shadow_enabled = bool(server_args.enable_dflash_feedback_shadow)
        self._top2_shadow_enabled = bool(server_args.enable_dflash_top2_shadow)
        self._redenoise_enabled = bool(server_args.enable_dflash_redenoise)
        self._redenoise_margin_threshold = float(server_args.dflash_redenoise_margin_threshold)
        self._redenoise_prefix_len = int(server_args.dflash_redenoise_prefix_len)
        self._redenoise_apply_start = int(server_args.dflash_redenoise_apply_start)
        self._redenoise_stats_batches = 0
        self._redenoise_stats_rounds = 0
        self._redenoise_stats_changed = 0
        self._redenoise_stats_repairs = 0
        self._redenoise_stats_harms = 0
        self._redenoise_stats_base_accept = 0
        self._redenoise_stats_final_accept = 0
        self._redenoise_stats_accept_delta = 0
        self._redenoise_stats_prefix_hist = np.zeros((self.draft_block_size,), dtype=np.int64)
        self._redenoise_stats_position_repairs = np.zeros((self.draft_block_size,), dtype=np.int64)
        self._redenoise_stats_position_harms = np.zeros((self.draft_block_size,), dtype=np.int64)
        self._redenoise_stats_start_accept_delta = np.zeros(
            (self.draft_block_size + 1,), dtype=np.int64
        )
        self._top2_shadow_batches = 0
        self._top2_shadow_rounds = 0
        self._top2_shadow_rejections = 0
        self._top2_shadow_hits = 0
        self._top2_shadow_width_hits = np.zeros((3,), dtype=np.int64)
        self._top2_shadow_reject_position = np.zeros((self.draft_block_size,), dtype=np.int64)
        self._top2_shadow_hit_position = np.zeros((self.draft_block_size,), dtype=np.int64)
        self._ngram_stats_batches = 0
        self._ngram_stats_rounds = 0
        self._ngram_stats_covered_rounds = 0
        self._ngram_stats_covered = 0
        self._ngram_stats_selected = 0
        self._ngram_stats_selected_accepted = 0
        self._ngram_stats_candidate_matches = 0
        self._ngram_stats_match_len_hist = np.zeros(
            (int(server_args.dflash_ngram_max_match) + 1,), dtype=np.int64
        )
        self._ngram_stats_position_covered = np.zeros((self.block_size - 1,), dtype=np.int64)
        self._ngram_stats_position_selected = np.zeros((self.block_size - 1,), dtype=np.int64)
        self._ngram_stats_position_accepted = np.zeros((self.block_size - 1,), dtype=np.int64)
        self._feedback_shadow_batches = 0
        self._feedback_shadow_rounds = 0
        self._feedback_shadow_stats = {
            source: {
                metric: np.zeros((self.block_size - 1,), dtype=np.int64)
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
        }
        margin_bin_count = len(_DFLASH_MARGIN_THRESHOLDS) + 1
        self._feedback_margin_stats = {
            source: {
                metric: np.zeros((margin_bin_count,), dtype=np.int64)
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
        }
        self._feedback_first_rejection_stats = {
            source: {
                metric: np.zeros((margin_bin_count,), dtype=np.int64)
                for metric in (
                    "valid",
                    "alternative",
                    "candidate_target",
                    "base_target",
                )
            }
            for source in _DFLASH_FEEDBACK_ALL_SOURCES
            if source != "target_correction"
        }
        self._feedback_condition_stats = {
            source: {
                metric: np.zeros((self.block_size + 1, self.block_size - 1), dtype=np.int64)
                for metric in ("valid", "target_match")
            }
            for source in _DFLASH_CONDITION_SOURCES
        }
        self._feedback_oracle_rejected_rounds = 0
        self._feedback_oracle_repair_rounds = 0
        self._feedback_oracle_local_novel = 0
        self._feedback_oracle_rejection_position = np.zeros((self.block_size - 1,), dtype=np.int64)
        self._feedback_oracle_repair_position = np.zeros((self.block_size - 1,), dtype=np.int64)
        self._feedback_oracle_source_repairs = {
            source: 0 for source in ("rejected_draft", "stale_suffix", "historical_ngram")
        }
        self._feedback_oracle_agreement_repairs = 0
        self._feedback_predictor_stats = {
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
                "selected_position": np.zeros((self.block_size - 1,), dtype=np.int64),
                "hit_position": np.zeros((self.block_size - 1,), dtype=np.int64),
            }
            for policy in _DFLASH_PREDICTOR_POLICIES
        }
        self._flashback_enabled = bool(server_args.enable_dflash_flashback)
        self._flashback_bonus = float(server_args.dflash_flashback_bonus)
        self._flashback_target_margin_weight = float(
            server_args.dflash_flashback_target_margin_weight
        )
        self._flashback_position_decay = float(server_args.dflash_flashback_position_decay)
        self._target_impl = getattr(target_worker, "worker", target_worker)
        self._target_compilation_manager = self._target_impl.compilation_manager

        draft_server_args = copy.deepcopy(server_args)
        draft_server_args.skip_tokenizer_init = True

        from sgl_jax.srt.models.dflash import DFlashDraftModel

        self._worker = ModelWorker(
            server_args=draft_server_args,
            mesh=self.mesh,
            req_to_token_pool=self.req_to_token_pool,
            is_draft_worker=True,
            model_class=DFlashDraftModel,
        )
        draft_model = self.draft_model_runner.model

        # Alias the KV allocator so draft block allocation draws from the same
        # free list the target uses (no collision with committed slots).
        self.draft_model_runner.token_to_kv_pool_allocator = self.token_to_kv_pool_allocator

        target_model = target_worker.model_runner.model
        embed_weight, head_weight = target_model.get_embed_and_head()
        self._target_lm_head = head_weight  # [vocab, hidden], for greedy head sampling
        self._target_embed = embed_weight  # [vocab, hidden]
        self._target_vocab_size = int(target_worker.model_runner.model_config.vocab_size)

        pool_pages = (
            int(target_worker.max_total_num_tokens) + self.page_size - 1
        ) // self.page_size
        max_req_pages = (
            int(self._target_compilation_manager.max_req_len) + self.page_size - 1
        ) // self.page_size
        self._page_indices_pool_capacity = 1 << max(0, pool_pages - 1).bit_length()
        self._page_indices_per_seq_capacity = max(
            16,
            1 << max(0, max_req_pages - 1).bit_length(),
        )
        self._verify_bucket_templates: dict[tuple, DFlashVerifyBucketTemplate] = {}

        dflash_config = parse_dflash_draft_config(
            server_args.speculative_draft_model_path,
            revision=server_args.speculative_draft_model_revision,
        )
        self._mask_token_id = resolve_mask_token_id(
            dflash_config,
            getattr(self._target_impl, "tokenizer", None),
            vocab_size=int(target_worker.model_runner.model_config.vocab_size),
        )

        draft_prefix_window = int(os.getenv("SGL_JAX_DFLASH_DRAFT_PREFIX_WINDOW", "0"))
        if draft_prefix_window > 0:
            for layer in draft_model.model.layers:
                layer.self_attn.attn.sliding_window_size = draft_prefix_window

        # Initialize JIT for the draft model runner (skipped during __init__
        # because is_draft_worker=True). The optional prefix window must be set
        # before nnx.split captures the model graph.
        self.draft_model_runner.initialize_jit()

        self.draft_layers = len(draft_model.model.layers)
        self._init_jit_target_verify()
        self._init_jit_kv_materialize()
        self._init_jit_draft_block()

        logger.info(
            "Initialized DFLASH worker: draft_block_size=%d, verify_block_size=%d, "
            "enable_anchor=%s, mask_token_id=%d, "
            "draft_layers=%d, ngram=%s, feedback_shadow=%s, top2_shadow=%s, "
            "flashback=%s, redenoise=%s, "
            "ngram_max_rerank_positions=%d, "
            "redenoise_margin_threshold=%.3f, redenoise_prefix_len=%d, "
            "redenoise_apply_start=%d, "
            "flashback_bonus=%.3f, "
            "flashback_target_margin_weight=%.3f, flashback_position_decay=%.3f, "
            "page_indices_pool_capacity=%d, "
            "page_indices_per_seq_capacity=%d",
            self.draft_block_size,
            self.block_size,
            self.enable_anchor,
            self._mask_token_id,
            self.draft_layers,
            self._ngram_enabled,
            self._feedback_shadow_enabled,
            self._top2_shadow_enabled,
            self._flashback_enabled,
            self._redenoise_enabled,
            self._ngram_max_rerank_positions,
            self._redenoise_margin_threshold,
            self._redenoise_prefix_len,
            self._redenoise_apply_start,
            self._flashback_bonus,
            self._flashback_target_margin_weight,
            self._flashback_position_decay,
            self._page_indices_pool_capacity,
            self._page_indices_per_seq_capacity,
        )

    @property
    def draft_model_runner(self):
        return self._worker.model_runner

    def __getattr__(self, name):
        target_worker = self.__dict__.get("_target_worker")
        if target_worker is None:
            raise AttributeError(name)
        return getattr(target_worker, name)

    def _draft_input_config(self) -> dict:
        return {
            "enable_ngram": self._ngram_enabled or self._feedback_shadow_enabled,
            "ngram_min_match": (
                1 if self._feedback_shadow_enabled else self.server_args.dflash_ngram_min_match
            ),
            "ngram_max_match": self.server_args.dflash_ngram_max_match,
            "ngram_base_bonus": self.server_args.dflash_ngram_bonus,
            "ngram_prompt_weight": self.server_args.dflash_ngram_prompt_weight,
            "ngram_output_weight": self.server_args.dflash_ngram_output_weight,
            "ngram_position_decay": self.server_args.dflash_ngram_position_decay,
        }

    def init_spec_relay_buffers(self):
        if self.spec_relay_buffers is None:
            self.spec_relay_buffers = create_dflash_relay_buffers(
                self.mesh,
                self.req_to_token_pool,
                dp_size=self.server_args.dp_size,
                feedback_width=self.block_size - 1,
            )

    def draft(self, model_worker_batch: ModelWorkerBatch) -> None:
        draft_input: DFlashDraftInput = model_worker_batch.spec_info_padded
        assert isinstance(
            draft_input, DFlashDraftInput
        ), "DFLASH decode requires DFlashDraftInput carried over from prefill."

        bs = int(model_worker_batch.seq_lens.shape[0])
        seq_lens = np.asarray(model_worker_batch.seq_lens, dtype=np.int32)
        target_prefix_lens = seq_lens - 1
        use_relay_state = draft_input.future_indices is not None
        if use_relay_state:
            draft_prefix_lens = target_prefix_lens
        else:
            self._trim_draft_state_to_bs(draft_input, bs)
            draft_prefix_lens = np.asarray(draft_input.draft_seq_lens, dtype=np.int32)

        draft_plan = self._build_draft_forward_plan(
            model_worker_batch,
            draft_input,
            target_prefix_lens,
            draft_prefix_lens,
            bs,
        )
        self.draft_model_runner.attn_backend.forward_metadata = draft_plan.forward_metadata
        (
            draft_token,
            base_draft_token,
            top2_draft_token,
            top2_margins,
            redenoise_candidate_token,
            redenoise_prefix_lens,
            resolved_target_prefix_lens,
            resolved_positions,
            resolved_cache_loc,
            ngram_selected_mask,
            candidate_margins,
        ) = self._run_jit_draft_block(draft_plan)

        # JAX dispatch is asynchronous. Bind the target model to the draft
        # proposal and shared device layout while jit_draft is executing.
        target_plan = self._build_target_verify_plan(
            model_worker_batch,
            draft_plan,
            draft_token,
            base_draft_token,
            top2_draft_token,
            top2_margins,
            redenoise_candidate_token,
            redenoise_prefix_lens,
            resolved_target_prefix_lens,
            resolved_positions,
            resolved_cache_loc,
            ngram_selected_mask,
            candidate_margins,
        )
        self.target_worker.model_runner.attn_backend.forward_metadata = target_plan.forward_metadata
        model_worker_batch._dflash_target_verify_plan = target_plan

    def verify(
        self,
        model_worker_batch: ModelWorkerBatch,
        cur_allocate_lens=None,
        *,
        update_relay: bool = False,
    ):
        from sgl_jax.srt.managers.scheduler import GenerationBatchResult

        plan = getattr(model_worker_batch, "_dflash_target_verify_plan", None)
        if not isinstance(plan, TargetVerifyPlan):
            raise RuntimeError("DFLASH target verify plan was not prepared by the draft phase.")
        if update_relay:
            plan = replace(plan, update_relay=True)
        (
            logits_output,
            cache_miss_count,
            accept_lens_out,
            next_token_ids_flat,
            new_verified_id,
            new_seq_lens,
            flashback_token_ids,
            flashback_target_margins,
            flashback_valid_mask,
            layers_topk_ids,
        ) = self._run_jit_target_verify(
            plan,
        )

        next_draft_input = DFlashDraftInput(
            verified_id=new_verified_id,
            target_hidden=logits_output.hidden_states,
            ctx_lens=accept_lens_out,
            draft_seq_lens=None,
            flashback_token_ids=flashback_token_ids,
            flashback_target_margins=flashback_target_margins,
            flashback_valid_mask=flashback_valid_mask,
            block_size=self.block_size,
            **self._draft_input_config(),
        )
        next_draft_input.new_seq_lens = new_seq_lens
        next_draft_input._target_verify_plan = plan

        # Start the small round-state copies as soon as target futures exist.
        # They can overlap draft KV materialization and are consumed only after
        # this method returns to draft_extend_for_decode.
        if not update_relay:
            jax.copy_to_host_async(
                (
                    accept_lens_out,
                    new_verified_id,
                    flashback_token_ids,
                    flashback_target_margins,
                    flashback_valid_mask,
                )
            )
        self._run_jit_draft_extend(
            logits_output.hidden_states,
            plan.draft_extend_positions,
            plan.draft_extend_cache_loc,
            accept_lens=accept_lens_out,
            active_mask=plan.active_mask,
        )
        self._target_impl.dump_topk_ids(layers_topk_ids, plan.model_worker_batch)
        self._target_impl.sync_queue.put((layers_topk_ids, plan.model_worker_batch))

        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=next_token_ids_flat,
            next_draft_input=next_draft_input,
            accept_lens=accept_lens_out,
            bid=model_worker_batch.bid,
            cache_miss_count=cache_miss_count,
            extend_input_len_per_req=None,
            extend_logprob_start_len_per_req=None,
        )

    def draft_extend_for_prefill(
        self,
        model_worker_batch: ModelWorkerBatch,
        target_hidden,
        next_token_ids,
    ) -> None:
        sel = np.asarray(model_worker_batch.logits_indices_selector)
        seq_lens = np.asarray(model_worker_batch.extend_seq_lens, dtype=np.int32)[sel]
        prefix_lens = np.asarray(model_worker_batch.extend_prefix_lens, dtype=np.int32)[sel]

        draft_input = DFlashDraftInput(
            verified_id=None,
            target_hidden=target_hidden,
            ctx_lens=seq_lens,
            draft_seq_lens=prefix_lens,
            block_size=self.block_size,
            **self._draft_input_config(),
        )

        # Materialization only depends on target hidden states. Dispatch it before
        # waiting for the sampled token so PJRT can run target prefill ->
        # jit_draft_extend without a host synchronization gap.
        self._append_target_hidden_to_draft_kv(model_worker_batch, draft_input)
        draft_input.verified_id = np.asarray(jax.device_get(next_token_ids))[sel].astype(np.int32)
        model_worker_batch.spec_info_padded = draft_input

    def draft_extend_for_decode(
        self,
        model_worker_batch: ModelWorkerBatch,
        batch_output,
    ) -> None:
        next_draft_input = batch_output.next_draft_input
        assert isinstance(next_draft_input, DFlashDraftInput)
        plan = getattr(next_draft_input, "_target_verify_plan", None)
        if not isinstance(plan, TargetVerifyPlan):
            raise RuntimeError("DFLASH draft extend is missing its target verify plan.")

        (
            accept_lens,
            verified_id,
            next_flashback_token_ids,
            next_flashback_target_margins,
            next_flashback_valid_mask,
            draft_token,
            base_draft_token,
            top2_draft_token,
            top2_margins,
            redenoise_candidate_token,
            redenoise_prefix_lens,
            prior_flashback_token_ids,
            prior_flashback_valid_mask,
            rejected_draft_token_ids,
            rejection_valid_mask,
            previous_accept_lens,
            target_correction_token_ids,
            candidate_margins,
            ngram_selected_mask,
            ngram_token_ids,
            ngram_valid_mask,
            ngram_match_lens,
            target_predict,
        ) = jax.device_get(
            (
                next_draft_input.ctx_lens,
                next_draft_input.verified_id,
                next_draft_input.flashback_token_ids,
                next_draft_input.flashback_target_margins,
                next_draft_input.flashback_valid_mask,
                plan.draft_token,
                plan.base_draft_token,
                plan.top2_draft_token,
                plan.top2_margins,
                plan.redenoise_candidate_token,
                plan.redenoise_prefix_lens,
                plan.flashback_token_ids,
                plan.flashback_valid_mask,
                plan.rejected_draft_token_ids,
                plan.rejection_valid_mask,
                plan.previous_accept_lens,
                plan.target_correction_token_ids,
                plan.candidate_margins,
                plan.ngram_selected_mask,
                plan.ngram_token_ids,
                plan.ngram_valid_mask,
                plan.ngram_match_lens,
                batch_output.next_token_ids,
            )
        )
        accept_lens = np.asarray(accept_lens, dtype=np.int32)
        verified_id = np.asarray(verified_id, dtype=np.int32)
        next_draft_input.verified_id = verified_id
        next_draft_input.flashback_token_ids = np.asarray(next_flashback_token_ids, dtype=np.int32)
        next_draft_input.flashback_target_margins = np.asarray(
            next_flashback_target_margins, dtype=np.float32
        )
        next_draft_input.flashback_valid_mask = np.asarray(
            next_flashback_valid_mask, dtype=np.bool_
        )
        selector = np.asarray(model_worker_batch.logits_indices_selector, dtype=np.int32)
        if self._ngram_enabled or self._feedback_shadow_enabled:
            self._record_ngram_stats(
                accept_lens,
                np.asarray(ngram_selected_mask, dtype=np.bool_),
                np.asarray(ngram_token_ids, dtype=np.int32),
                np.asarray(ngram_valid_mask, dtype=np.bool_),
                np.asarray(ngram_match_lens, dtype=np.int32),
                np.asarray(target_predict, dtype=np.int32),
                selector,
            )
        if self._feedback_shadow_enabled:
            self._record_feedback_shadow_stats(
                accept_lens=accept_lens,
                draft_token=np.asarray(draft_token, dtype=np.int32),
                target_predict_flat=np.asarray(target_predict, dtype=np.int32),
                rejected_draft_token_ids=np.asarray(rejected_draft_token_ids, dtype=np.int32),
                rejection_valid_mask=np.asarray(rejection_valid_mask, dtype=np.bool_),
                target_correction_token_ids=np.asarray(target_correction_token_ids, dtype=np.int32),
                stale_suffix_token_ids=np.asarray(prior_flashback_token_ids, dtype=np.int32),
                stale_suffix_valid_mask=np.asarray(prior_flashback_valid_mask, dtype=np.bool_),
                ngram_token_ids=np.asarray(ngram_token_ids, dtype=np.int32),
                ngram_valid_mask=np.asarray(ngram_valid_mask, dtype=np.bool_),
                ngram_match_lens=np.asarray(ngram_match_lens, dtype=np.int32),
                previous_accept_lens=np.asarray(previous_accept_lens, dtype=np.int32),
                candidate_margins=np.asarray(candidate_margins, dtype=np.float32),
                selector=selector,
            )
        if self._redenoise_enabled:
            self._record_redenoise_stats(
                accept_lens=accept_lens,
                base_draft_token=np.asarray(base_draft_token, dtype=np.int32),
                candidate_draft_token=np.asarray(redenoise_candidate_token, dtype=np.int32),
                final_draft_token=np.asarray(draft_token, dtype=np.int32),
                target_predict_flat=np.asarray(target_predict, dtype=np.int32),
                prefix_lens=np.asarray(redenoise_prefix_lens, dtype=np.int32),
                selector=selector,
            )
        if self._top2_shadow_enabled:
            self._record_top2_shadow_stats(
                accept_lens=accept_lens,
                base_draft_token=np.asarray(base_draft_token, dtype=np.int32),
                top2_draft_token=np.asarray(top2_draft_token, dtype=np.int32),
                top2_margins=np.asarray(top2_margins, dtype=np.float32),
                target_predict_flat=np.asarray(target_predict, dtype=np.int32),
                selector=selector,
            )
        active_mask = np.zeros((accept_lens.shape[0],), dtype=np.bool_)
        active_mask[selector] = True
        (
            next_draft_input.rejected_draft_token_ids,
            next_draft_input.rejection_valid_mask,
        ) = build_dflash_rejection_feedback(
            np.asarray(draft_token, dtype=np.int32),
            accept_lens,
            active_mask,
            block_size=self.block_size,
        )
        next_draft_input.previous_accept_lens = np.where(active_mask, accept_lens, 0).astype(
            np.int32
        )
        next_draft_input.ctx_lens = np.zeros_like(accept_lens)
        next_draft_input.draft_seq_lens = plan.target_prefix_lens + accept_lens
        next_draft_input.new_seq_lens = plan.seq_lens + accept_lens
        next_draft_input.target_hidden = None
        batch_output.accept_lens = accept_lens
        self._unpad_draft_state(
            next_draft_input,
            selector,
        )
        model_worker_batch.spec_info_padded = next_draft_input
        del next_draft_input._target_verify_plan
        del model_worker_batch._dflash_target_verify_plan

    def _record_redenoise_stats(
        self,
        *,
        accept_lens: np.ndarray,
        base_draft_token: np.ndarray,
        candidate_draft_token: np.ndarray,
        final_draft_token: np.ndarray,
        target_predict_flat: np.ndarray,
        prefix_lens: np.ndarray,
        selector: np.ndarray,
    ) -> None:
        padded_bs = int(accept_lens.shape[0])
        proposal_width = self.block_size - 1
        base = base_draft_token.reshape(padded_bs, self.block_size)[selector, 1:]
        candidate = candidate_draft_token.reshape(padded_bs, self.block_size)[selector, 1:]
        final = final_draft_token.reshape(padded_bs, self.block_size)[selector, 1:]
        target = target_predict_flat.reshape(padded_bs, self.block_size)[selector, :proposal_width]
        prefix_lens = prefix_lens.reshape(padded_bs)[selector]
        final_accept = accept_lens.reshape(padded_bs)[selector]

        changed = base != final
        base_correct = base == target
        final_correct = final == target
        repairs = changed & ~base_correct & final_correct
        harms = changed & base_correct & ~final_correct
        base_prefix_matches = np.logical_and.accumulate(base_correct, axis=1)
        base_accept = base_prefix_matches.sum(axis=1, dtype=np.int32) + 1

        self._redenoise_stats_batches += 1
        self._redenoise_stats_rounds += len(selector)
        self._redenoise_stats_changed += int(changed.sum())
        self._redenoise_stats_repairs += int(repairs.sum())
        self._redenoise_stats_harms += int(harms.sum())
        self._redenoise_stats_position_repairs += (~base_correct & (candidate == target)).sum(
            axis=0, dtype=np.int64
        )
        self._redenoise_stats_position_harms += (base_correct & (candidate != target)).sum(
            axis=0, dtype=np.int64
        )
        self._redenoise_stats_base_accept += int(base_accept.sum())
        self._redenoise_stats_final_accept += int(final_accept.sum())
        self._redenoise_stats_accept_delta += int((final_accept - base_accept).sum())
        positions = np.arange(proposal_width, dtype=np.int32)[None, :]
        for apply_start in range(proposal_width + 1):
            simulated = np.where(positions < apply_start, base, candidate)
            simulated_correct = simulated == target
            simulated_accept = (
                np.logical_and.accumulate(simulated_correct, axis=1).sum(axis=1, dtype=np.int32) + 1
            )
            self._redenoise_stats_start_accept_delta[apply_start] += int(
                (simulated_accept - base_accept).sum()
            )
        clipped_prefix_lens = np.clip(
            prefix_lens,
            0,
            len(self._redenoise_stats_prefix_hist) - 1,
        )
        self._redenoise_stats_prefix_hist += np.bincount(
            clipped_prefix_lens,
            minlength=len(self._redenoise_stats_prefix_hist),
        )
        if self._redenoise_stats_batches % 100 != 0:
            return

        rounds = max(1, self._redenoise_stats_rounds)
        changed_count = max(1, self._redenoise_stats_changed)
        logger.info(
            "[DFLASH-REDENOISE] batches=%d rounds=%d base_accept_len=%.6f "
            "final_accept_len=%.6f accept_delta=%.6f changed_per_round=%.6f "
            "repair_rate=%.6f harm_rate=%.6f prefix_hist=%s "
            "position_repairs=%s position_harms=%s start_accept_delta=%s",
            self._redenoise_stats_batches,
            self._redenoise_stats_rounds,
            self._redenoise_stats_base_accept / rounds,
            self._redenoise_stats_final_accept / rounds,
            self._redenoise_stats_accept_delta / rounds,
            self._redenoise_stats_changed / rounds,
            self._redenoise_stats_repairs / changed_count,
            self._redenoise_stats_harms / changed_count,
            self._redenoise_stats_prefix_hist.tolist(),
            self._redenoise_stats_position_repairs.tolist(),
            self._redenoise_stats_position_harms.tolist(),
            (self._redenoise_stats_start_accept_delta / rounds).tolist(),
        )

    def _record_top2_shadow_stats(
        self,
        *,
        accept_lens: np.ndarray,
        base_draft_token: np.ndarray,
        top2_draft_token: np.ndarray,
        top2_margins: np.ndarray,
        target_predict_flat: np.ndarray,
        selector: np.ndarray,
    ) -> None:
        """Measure fixed-width top-2 branch coverage at the first rejection.

        Only the token at the first rejected proposal has a target label that
        is valid under the verified prefix.  Consequently this reports a
        conservative one-token lower bound instead of pretending that logits
        after the rejection describe the corrected branch.
        """
        padded_bs = int(accept_lens.shape[0])
        proposal_width = self.block_size - 1
        base = base_draft_token.reshape(padded_bs, self.block_size)[selector, 1:]
        top2 = top2_draft_token.reshape(padded_bs, self.block_size)[selector, 1:]
        margins = top2_margins.reshape(padded_bs, proposal_width)[selector]
        target = target_predict_flat.reshape(padded_bs, self.block_size)[selector, :proposal_width]
        selected_accept = accept_lens.reshape(padded_bs)[selector]
        reject_positions = selected_accept - 1
        rejected = (reject_positions >= 0) & (reject_positions < proposal_width)
        rows = np.arange(len(selector), dtype=np.int32)
        safe_positions = np.clip(reject_positions, 0, proposal_width - 1)
        first_reject_consistent = base[rows, safe_positions] != target[rows, safe_positions]
        rejected &= first_reject_consistent
        hits = rejected & (top2[rows, safe_positions] == target[rows, safe_positions])

        self._top2_shadow_batches += 1
        self._top2_shadow_rounds += len(selector)
        self._top2_shadow_rejections += int(rejected.sum())
        self._top2_shadow_hits += int(hits.sum())
        self._top2_shadow_reject_position += np.bincount(
            safe_positions[rejected], minlength=proposal_width
        )[:proposal_width]
        self._top2_shadow_hit_position += np.bincount(
            safe_positions[hits], minlength=proposal_width
        )[:proposal_width]

        ranked_positions = np.argsort(margins, axis=1, kind="stable")
        for width_index, chain_width in enumerate((2, 4, 8)):
            alternative_count = min(chain_width - 1, proposal_width)
            selected_positions = ranked_positions[:, :alternative_count]
            covered = np.any(selected_positions == safe_positions[:, None], axis=1)
            self._top2_shadow_width_hits[width_index] += int((hits & covered).sum())

        if self._top2_shadow_batches % 100 != 0:
            return
        rounds = max(1, self._top2_shadow_rounds)
        rejections = max(1, self._top2_shadow_rejections)
        logger.info(
            "[DFLASH-TOP2-SHADOW] batches=%d rounds=%d rejection_rate=%.6f "
            "top2_first_reject_hit_rate=%.6f lower_bound_accept_gain=%.6f "
            "margin_width2_gain=%.6f margin_width4_gain=%.6f "
            "all_positions_gain=%.6f reject_position=%s hit_position=%s",
            self._top2_shadow_batches,
            self._top2_shadow_rounds,
            self._top2_shadow_rejections / rounds,
            self._top2_shadow_hits / rejections,
            self._top2_shadow_hits / rounds,
            self._top2_shadow_width_hits[0] / rounds,
            self._top2_shadow_width_hits[1] / rounds,
            self._top2_shadow_width_hits[2] / rounds,
            self._top2_shadow_reject_position.tolist(),
            self._top2_shadow_hit_position.tolist(),
        )

    def _record_ngram_stats(
        self,
        accept_lens: np.ndarray,
        selected_mask: np.ndarray,
        candidate_ids: np.ndarray,
        valid_mask: np.ndarray,
        match_lens: np.ndarray,
        target_predict_flat: np.ndarray,
        selector: np.ndarray,
    ) -> None:
        proposal_width = self.block_size - 1
        padded_bs = accept_lens.shape[0]
        selected_mask = selected_mask.reshape(padded_bs, proposal_width)[selector]
        candidate_ids = candidate_ids.reshape(padded_bs, proposal_width)[selector]
        valid_mask = valid_mask.reshape(padded_bs, proposal_width)[selector]
        match_lens = match_lens.reshape(padded_bs)[selector]
        accept_lens = accept_lens.reshape(padded_bs)[selector]
        target_predict = target_predict_flat.reshape(padded_bs, self.block_size)[selector]
        candidate_matches = valid_mask & (candidate_ids == target_predict[:, :proposal_width])
        offsets = np.arange(proposal_width, dtype=np.int32)[None, :]
        accepted_mask = offsets < np.maximum(accept_lens[:, None] - 1, 0)
        selected_accepted = selected_mask & accepted_mask

        self._ngram_stats_batches += 1
        self._ngram_stats_rounds += len(selector)
        self._ngram_stats_covered_rounds += int((match_lens > 0).sum())
        self._ngram_stats_covered += int(valid_mask.sum())
        self._ngram_stats_selected += int(selected_mask.sum())
        self._ngram_stats_selected_accepted += int(selected_accepted.sum())
        self._ngram_stats_candidate_matches += int(candidate_matches.sum())
        clipped_match_lens = np.clip(
            match_lens,
            0,
            len(self._ngram_stats_match_len_hist) - 1,
        )
        self._ngram_stats_match_len_hist += np.bincount(
            clipped_match_lens,
            minlength=len(self._ngram_stats_match_len_hist),
        )
        self._ngram_stats_position_covered += valid_mask.sum(axis=0, dtype=np.int64)
        self._ngram_stats_position_selected += selected_mask.sum(axis=0, dtype=np.int64)
        self._ngram_stats_position_accepted += selected_accepted.sum(axis=0, dtype=np.int64)
        if self._ngram_stats_batches % 100 != 0:
            return

        total_positions = max(1, self._ngram_stats_rounds * proposal_width)
        covered = max(1, self._ngram_stats_covered)
        selected = max(1, self._ngram_stats_selected)
        logger.info(
            "[DFLASH-NGRAM] batches=%d rounds=%d covered_rounds=%d "
            "coverage_rate=%.6f selected_rate=%.6f selected_accept_rate=%.6f "
            "candidate_match_rate=%.6f match_len_hist=%s position_covered=%s "
            "position_selected=%s position_accepted=%s",
            self._ngram_stats_batches,
            self._ngram_stats_rounds,
            self._ngram_stats_covered_rounds,
            self._ngram_stats_covered / total_positions,
            self._ngram_stats_selected / covered,
            self._ngram_stats_selected_accepted / selected,
            self._ngram_stats_candidate_matches / covered,
            self._ngram_stats_match_len_hist.tolist(),
            self._ngram_stats_position_covered.tolist(),
            self._ngram_stats_position_selected.tolist(),
            self._ngram_stats_position_accepted.tolist(),
        )

    def _record_feedback_shadow_stats(
        self,
        *,
        accept_lens: np.ndarray,
        draft_token: np.ndarray,
        target_predict_flat: np.ndarray,
        rejected_draft_token_ids: np.ndarray,
        rejection_valid_mask: np.ndarray,
        target_correction_token_ids: np.ndarray,
        stale_suffix_token_ids: np.ndarray,
        stale_suffix_valid_mask: np.ndarray,
        ngram_token_ids: np.ndarray,
        ngram_valid_mask: np.ndarray,
        ngram_match_lens: np.ndarray,
        previous_accept_lens: np.ndarray,
        candidate_margins: np.ndarray,
        selector: np.ndarray,
    ) -> None:
        """Measure prior-round signals against the untouched current DFlash block."""
        proposal_width = self.block_size - 1
        padded_bs = int(np.asarray(accept_lens).reshape(-1).shape[0])
        selector = np.asarray(selector, dtype=np.int32)
        accept_lens = np.asarray(accept_lens, dtype=np.int32).reshape(padded_bs)[selector]
        draft = np.asarray(draft_token, dtype=np.int32).reshape(padded_bs, self.block_size)[
            selector, 1:
        ]
        target = np.asarray(target_predict_flat, dtype=np.int32).reshape(
            padded_bs, self.block_size
        )[selector, :proposal_width]
        offsets = np.arange(proposal_width, dtype=np.int32)[None, :]
        accepted_mask = offsets < np.maximum(accept_lens[:, None] - 1, 0)
        base_target = draft == target

        rejection_valid = np.asarray(rejection_valid_mask, dtype=np.bool_).reshape(padded_bs)[
            selector, None
        ]
        rejected_ids = np.broadcast_to(
            np.asarray(rejected_draft_token_ids, dtype=np.int32).reshape(padded_bs)[selector, None],
            draft.shape,
        )
        correction_ids = np.broadcast_to(
            np.asarray(target_correction_token_ids, dtype=np.int32).reshape(padded_bs)[
                selector, None
            ],
            draft.shape,
        )
        scalar_valid = np.broadcast_to(rejection_valid, draft.shape)

        stale_ids = np.asarray(stale_suffix_token_ids, dtype=np.int32).reshape(
            padded_bs, proposal_width
        )[selector]
        stale_valid = np.asarray(stale_suffix_valid_mask, dtype=np.bool_).reshape(
            padded_bs, proposal_width
        )[selector]
        ngram_ids = np.asarray(ngram_token_ids, dtype=np.int32).reshape(padded_bs, proposal_width)[
            selector
        ]
        ngram_valid = np.asarray(ngram_valid_mask, dtype=np.bool_).reshape(
            padded_bs, proposal_width
        )[selector]
        match_lens = np.asarray(ngram_match_lens, dtype=np.int32).reshape(padded_bs)[selector]
        previous_accept_lens = np.asarray(previous_accept_lens, dtype=np.int32).reshape(padded_bs)[
            selector
        ]
        candidate_margins = np.asarray(candidate_margins, dtype=np.float32).reshape(
            padded_bs, proposal_width, 3
        )[selector]
        sources = {
            "rejected_draft": (rejected_ids, scalar_valid, 0),
            "target_correction": (correction_ids, scalar_valid, None),
            "stale_suffix": (stale_ids, stale_valid, 1),
            "ngram_len1": (
                ngram_ids,
                ngram_valid & (match_lens[:, None] == 1),
                2,
            ),
            "ngram_len2": (
                ngram_ids,
                ngram_valid & (match_lens[:, None] == 2),
                2,
            ),
            "ngram_len3plus": (
                ngram_ids,
                ngram_valid & (match_lens[:, None] >= 3),
                2,
            ),
        }
        rejected_stale = scalar_valid & stale_valid & (rejected_ids == stale_ids)
        rejected_ngram = scalar_valid & ngram_valid & (rejected_ids == ngram_ids)
        stale_ngram = stale_valid & ngram_valid & (stale_ids == ngram_ids)
        all_three = rejected_stale & ngram_valid & (rejected_ids == ngram_ids)
        sources.update(
            {
                "agree_rejected_stale": (stale_ids, rejected_stale, 1),
                "agree_rejected_ngram": (ngram_ids, rejected_ngram, 2),
                "agree_stale_ngram": (ngram_ids, stale_ngram, 2),
                "agree_all_three": (ngram_ids, all_three, 2),
            }
        )
        self._record_feedback_predictor_stats(
            draft=draft,
            target=target,
            ngram_ids=ngram_ids,
            ngram_valid=ngram_valid,
            match_lens=match_lens,
            previous_accept_lens=previous_accept_lens,
            candidate_margins=candidate_margins,
            sparse_candidate_valid=np.stack(
                [
                    scalar_valid & (rejected_ids != draft),
                    stale_valid & (stale_ids != draft),
                ],
                axis=-1,
            ),
        )
        current_rejection = (accept_lens > 0) & (accept_lens <= proposal_width)
        current_rejection_position = np.clip(accept_lens - 1, 0, proposal_width - 1)
        rows = np.arange(accept_lens.shape[0], dtype=np.int32)

        self._feedback_shadow_batches += 1
        self._feedback_shadow_rounds += int(selector.size)
        for source, (candidate, valid, margin_index) in sources.items():
            draft_reuse = valid & (candidate == draft)
            target_match = valid & (candidate == target)
            target_novel = target_match & ~draft_reuse
            counters = self._feedback_shadow_stats[source]
            counters["valid"] += valid.sum(axis=0, dtype=np.int64)
            counters["draft_reuse"] += draft_reuse.sum(axis=0, dtype=np.int64)
            counters["target_match"] += target_match.sum(axis=0, dtype=np.int64)
            counters["target_novel"] += target_novel.sum(axis=0, dtype=np.int64)
            counters["draft_target_match"] += (draft_reuse & target_match).sum(
                axis=0, dtype=np.int64
            )
            counters["accepted_chain"] += (draft_reuse & accepted_mask).sum(axis=0, dtype=np.int64)

            if margin_index is not None:
                margin_bins = np.digitize(
                    candidate_margins[..., margin_index],
                    _DFLASH_MARGIN_THRESHOLDS,
                    right=True,
                )
                margin_stats = self._feedback_margin_stats[source]
                first_stats = self._feedback_first_rejection_stats[source]
                first_valid = current_rejection & valid[rows, current_rejection_position]
                first_alternative = first_valid & ~draft_reuse[rows, current_rejection_position]
                first_candidate_target = (
                    first_alternative & target_match[rows, current_rejection_position]
                )
                first_base_target = (
                    first_alternative & base_target[rows, current_rejection_position]
                )
                first_margin_bins = margin_bins[rows, current_rejection_position]
                for bin_index in range(len(_DFLASH_MARGIN_THRESHOLDS) + 1):
                    in_bin = valid & (margin_bins == bin_index)
                    margin_stats["valid"][bin_index] += int(in_bin.sum())
                    margin_stats["alternative"][bin_index] += int((in_bin & ~draft_reuse).sum())
                    margin_stats["target_match"][bin_index] += int((in_bin & target_match).sum())
                    margin_stats["target_novel"][bin_index] += int((in_bin & target_novel).sum())
                    margin_stats["base_target"][bin_index] += int(
                        (in_bin & ~draft_reuse & base_target).sum()
                    )
                    first_in_bin = first_valid & (first_margin_bins == bin_index)
                    first_stats["valid"][bin_index] += int(first_in_bin.sum())
                    first_stats["alternative"][bin_index] += int(
                        (first_in_bin & first_alternative).sum()
                    )
                    first_stats["candidate_target"][bin_index] += int(
                        (first_in_bin & first_candidate_target).sum()
                    )
                    first_stats["base_target"][bin_index] += int(
                        (first_in_bin & first_base_target).sum()
                    )

            if source in self._feedback_condition_stats:
                condition = self._feedback_condition_stats[source]
                clipped_previous = np.clip(previous_accept_lens, 0, self.block_size)
                for previous_len in range(1, self.block_size + 1):
                    previous_rows = clipped_previous == previous_len
                    condition["valid"][previous_len] += valid[previous_rows].sum(
                        axis=0, dtype=np.int64
                    )
                    condition["target_match"][previous_len] += target_match[previous_rows].sum(
                        axis=0, dtype=np.int64
                    )

        algorithm_ids = np.stack([rejected_ids, stale_ids, ngram_ids], axis=-1)
        algorithm_valid = np.stack([scalar_valid, stale_valid, ngram_valid], axis=-1)
        algorithm_target = algorithm_valid & (algorithm_ids == target[..., None])
        self._feedback_oracle_local_novel += int(
            (algorithm_target.any(axis=-1) & (draft != target)).sum()
        )
        repair_matches = algorithm_target[rows, current_rejection_position]
        repair_matches &= current_rejection[:, None]
        repairable = repair_matches.any(axis=1)
        self._feedback_oracle_rejected_rounds += int(current_rejection.sum())
        self._feedback_oracle_repair_rounds += int(repairable.sum())
        self._feedback_oracle_rejection_position += np.bincount(
            current_rejection_position[current_rejection], minlength=proposal_width
        )
        self._feedback_oracle_repair_position += np.bincount(
            current_rejection_position[repairable], minlength=proposal_width
        )
        for source_index, source in enumerate(
            ("rejected_draft", "stale_suffix", "historical_ngram")
        ):
            self._feedback_oracle_source_repairs[source] += int(
                repair_matches[:, source_index].sum()
            )
        self._feedback_oracle_agreement_repairs += int((repair_matches.sum(axis=1) >= 2).sum())

        if self._feedback_shadow_batches % 100 != 0:
            return
        for source in _DFLASH_FEEDBACK_SHADOW_SOURCES:
            counters = self._feedback_shadow_stats[source]
            valid = max(1, int(counters["valid"].sum()))
            reuse = max(1, int(counters["draft_reuse"].sum()))
            logger.info(
                "[DFLASH-FEEDBACK-SHADOW] batches=%d rounds=%d source=%s "
                "valid=%d draft_reuse_rate=%.6f target_match_rate=%.6f "
                "target_novel_rate=%.6f reuse_target_precision=%.6f "
                "reuse_wrong_rate=%.6f accepted_chain_rate=%.6f "
                "accepted_given_reuse=%.6f position_valid=%s position_reuse=%s "
                "position_target=%s position_novel=%s position_reuse_target=%s "
                "position_chain=%s",
                self._feedback_shadow_batches,
                self._feedback_shadow_rounds,
                source,
                int(counters["valid"].sum()),
                counters["draft_reuse"].sum() / valid,
                counters["target_match"].sum() / valid,
                counters["target_novel"].sum() / valid,
                counters["draft_target_match"].sum() / reuse,
                (counters["draft_reuse"].sum() - counters["draft_target_match"].sum()) / valid,
                counters["accepted_chain"].sum() / valid,
                counters["accepted_chain"].sum() / reuse,
                counters["valid"].tolist(),
                counters["draft_reuse"].tolist(),
                counters["target_match"].tolist(),
                counters["target_novel"].tolist(),
                counters["draft_target_match"].tolist(),
                counters["accepted_chain"].tolist(),
            )
        rejected_rounds = max(1, self._feedback_oracle_rejected_rounds)
        total_positions = max(1, self._feedback_shadow_rounds * proposal_width)
        logger.info(
            "[DFLASH-FEEDBACK-ORACLE] batches=%d rounds=%d rejected_rounds=%d "
            "repair_rounds=%d repair_rate=%.6f local_novel_positions=%d "
            "local_novel_rate=%.6f source_repairs=%s agreement_repairs=%d "
            "rejection_position=%s repair_position=%s",
            self._feedback_shadow_batches,
            self._feedback_shadow_rounds,
            self._feedback_oracle_rejected_rounds,
            self._feedback_oracle_repair_rounds,
            self._feedback_oracle_repair_rounds / rejected_rounds,
            self._feedback_oracle_local_novel,
            self._feedback_oracle_local_novel / total_positions,
            self._feedback_oracle_source_repairs,
            self._feedback_oracle_agreement_repairs,
            self._feedback_oracle_rejection_position.tolist(),
            self._feedback_oracle_repair_position.tolist(),
        )
        if self._feedback_shadow_batches % 500 != 0:
            return
        for source in _DFLASH_FEEDBACK_AGREEMENT_SOURCES:
            counters = self._feedback_shadow_stats[source]
            valid = max(1, int(counters["valid"].sum()))
            logger.info(
                "[DFLASH-FEEDBACK-AGREEMENT] batches=%d source=%s valid=%d "
                "target_match_rate=%.6f target_novel_rate=%.6f position_valid=%s "
                "position_target=%s position_novel=%s",
                self._feedback_shadow_batches,
                source,
                int(counters["valid"].sum()),
                counters["target_match"].sum() / valid,
                counters["target_novel"].sum() / valid,
                counters["valid"].tolist(),
                counters["target_match"].tolist(),
                counters["target_novel"].tolist(),
            )
        for source, counters in self._feedback_margin_stats.items():
            logger.info(
                "[DFLASH-FEEDBACK-MARGIN] batches=%d source=%s thresholds=%s "
                "valid=%s alternative=%s target=%s novel=%s base_target=%s",
                self._feedback_shadow_batches,
                source,
                _DFLASH_MARGIN_THRESHOLDS.tolist(),
                counters["valid"].tolist(),
                counters["alternative"].tolist(),
                counters["target_match"].tolist(),
                counters["target_novel"].tolist(),
                counters["base_target"].tolist(),
            )
        for source, counters in self._feedback_first_rejection_stats.items():
            logger.info(
                "[DFLASH-FEEDBACK-FIRST-REJECTION] batches=%d source=%s "
                "thresholds=%s valid=%s alternative=%s candidate_target=%s "
                "base_target=%s net=%s",
                self._feedback_shadow_batches,
                source,
                _DFLASH_MARGIN_THRESHOLDS.tolist(),
                counters["valid"].tolist(),
                counters["alternative"].tolist(),
                counters["candidate_target"].tolist(),
                counters["base_target"].tolist(),
                (counters["candidate_target"] - counters["base_target"]).tolist(),
            )
        for source, counters in self._feedback_condition_stats.items():
            logger.info(
                "[DFLASH-FEEDBACK-CONDITION] batches=%d source=%s "
                "previous_accept_by_position_valid=%s "
                "previous_accept_by_position_target=%s",
                self._feedback_shadow_batches,
                source,
                counters["valid"].tolist(),
                counters["target_match"].tolist(),
            )
        for policy, counters in self._feedback_predictor_stats.items():
            predictions = max(1, counters["predictions"])
            rejected_predictions = max(1, counters["rejected_predictions"])
            logger.info(
                "[DFLASH-FEEDBACK-PREDICTOR] batches=%d policy=%s "
                "predictions=%d rejected_predictions=%d hit_at1=%.6f "
                "candidate_precision=%.6f repairs=%d harms=%d neutral=%d "
                "repair_rate=%.6f harm_rate=%.6f accept_gain=%d accept_loss=%d "
                "accept_delta=%d selected_position=%s hit_position=%s",
                self._feedback_shadow_batches,
                policy,
                counters["predictions"],
                counters["rejected_predictions"],
                counters["position_hits"] / rejected_predictions,
                counters["candidate_target"] / predictions,
                counters["repairs"],
                counters["harms"],
                counters["neutral"],
                counters["repairs"] / predictions,
                counters["harms"] / predictions,
                counters["accept_gain"],
                counters["accept_loss"],
                counters["accept_delta"],
                counters["selected_position"].tolist(),
                counters["hit_position"].tolist(),
            )

    def _record_feedback_predictor_stats(
        self,
        *,
        draft: np.ndarray,
        target: np.ndarray,
        ngram_ids: np.ndarray,
        ngram_valid: np.ndarray,
        match_lens: np.ndarray,
        previous_accept_lens: np.ndarray,
        candidate_margins: np.ndarray,
        sparse_candidate_valid: np.ndarray,
    ) -> None:
        """Evaluate single-position N-gram policies without changing proposals."""
        batch_size, proposal_width = draft.shape
        positions = np.arange(proposal_width, dtype=np.float32)[None, :]
        ngram_margins = candidate_margins[..., 2]
        sparse_margins = np.min(
            np.where(sparse_candidate_valid, candidate_margins[..., :2], np.inf),
            axis=-1,
        )
        sparse_margins = np.where(np.isfinite(sparse_margins), sparse_margins, ngram_margins)
        eligible = (
            ngram_valid
            & (match_lens[:, None] >= 3)
            & (ngram_ids != draft)
            & (ngram_margins <= _DFLASH_PREDICTOR_NGRAM_MARGIN)
        )
        combined_margins = sparse_margins + ngram_margins
        lagged_positions = np.clip(
            previous_accept_lens.astype(np.float32) - 1,
            0,
            proposal_width - 1,
        )[:, None]
        policy_scores = {
            "earliest": np.broadcast_to(positions, draft.shape),
            "feedback_uncertainty": sparse_margins,
            "ngram_competition": ngram_margins,
            "combined_margin": combined_margins,
            "lagged_accept": (np.abs(positions - lagged_positions) * 1024.0 + combined_margins),
        }

        base_matches = draft == target
        base_prefix = np.where(
            base_matches.all(axis=1),
            proposal_width,
            np.argmax(~base_matches, axis=1),
        )
        rows = np.arange(batch_size, dtype=np.int32)
        for policy, scores in policy_scores.items():
            predicted = eligible.any(axis=1)
            selected_position = np.argmin(np.where(eligible, scores, np.inf), axis=1).astype(
                np.int32
            )
            selected_candidate_target = predicted & (
                ngram_ids[rows, selected_position] == target[rows, selected_position]
            )
            modified_matches = base_matches.copy()
            modified_matches[rows[predicted], selected_position[predicted]] = (
                selected_candidate_target[predicted]
            )
            modified_prefix = np.where(
                modified_matches.all(axis=1),
                proposal_width,
                np.argmax(~modified_matches, axis=1),
            )
            accept_delta = np.where(predicted, modified_prefix - base_prefix, 0)
            rejected = base_prefix < proposal_width
            position_hit = predicted & rejected & (selected_position == base_prefix)
            repair = predicted & (accept_delta > 0)
            harm = predicted & (accept_delta < 0)
            neutral = predicted & (accept_delta == 0)

            counters = self._feedback_predictor_stats[policy]
            counters["predictions"] += int(predicted.sum())
            counters["rejected_predictions"] += int((predicted & rejected).sum())
            counters["position_hits"] += int(position_hit.sum())
            counters["candidate_target"] += int(selected_candidate_target.sum())
            counters["repairs"] += int(repair.sum())
            counters["harms"] += int(harm.sum())
            counters["neutral"] += int(neutral.sum())
            counters["accept_gain"] += int(accept_delta[repair].sum())
            counters["accept_loss"] += int(-accept_delta[harm].sum())
            counters["accept_delta"] += int(accept_delta.sum())
            counters["selected_position"] += np.bincount(
                selected_position[predicted], minlength=proposal_width
            )
            counters["hit_position"] += np.bincount(
                selected_position[position_hit], minlength=proposal_width
            )

    def forward_batch_speculative_prefill_overlap(self, model_worker_batch: ModelWorkerBatch):
        from jax.sharding import NamedSharding
        from jax.sharding import PartitionSpec as P

        from sgl_jax.srt.managers.scheduler import GenerationBatchResult
        from sgl_jax.srt.sampling.sampling_batch_info import SamplingMetadata
        from sgl_jax.srt.speculative.draft_extend_fused import (
            _prepare_spec_prefill_output_token_ids,
        )

        if not model_worker_batch.forward_mode.is_extend():
            raise NotImplementedError("DFLASH prefill overlap requires an extend batch.")
        if not self._can_use_fused_spec_prefill(model_worker_batch):
            raise NotImplementedError("DFLASH prefill overlap only supports greedy sampling.")

        self.init_spec_relay_buffers()
        if model_worker_batch.sampling_info.temperatures.ndim == 1:
            model_worker_batch.sampling_info.temperatures = (
                model_worker_batch.sampling_info.temperatures[:, None]
            )
        sampling_metadata = SamplingMetadata.from_model_worker_batch(
            model_worker_batch,
            len(model_worker_batch.seq_lens) - model_worker_batch.real_bs,
            self.mesh,
            vocab_size=self.target_worker.model_config.vocab_size,
        )
        logits_output, _, cache_miss_count, bid, seq_lens = self.forward_target_extend(
            model_worker_batch,
            sampling_metadata,
            skip_sample=True,
        )
        next_token_ids = jnp.argmax(logits_output.next_token_logits, axis=-1).astype(jnp.int32)

        sel = np.asarray(model_worker_batch.logits_indices_selector, dtype=np.int32)
        extend_seq_lens = np.asarray(model_worker_batch.extend_seq_lens, dtype=np.int32)[sel]
        extend_prefix_lens = np.asarray(model_worker_batch.extend_prefix_lens, dtype=np.int32)[sel]
        materialize_input = DFlashDraftInput(
            verified_id=None,
            target_hidden=logits_output.hidden_states,
            ctx_lens=extend_seq_lens,
            draft_seq_lens=extend_prefix_lens,
            block_size=self.block_size,
            **self._draft_input_config(),
        )
        self._append_target_hidden_to_draft_kv(model_worker_batch, materialize_input)

        total_bs = int(model_worker_batch.req_pool_indices.shape[0])
        valid_mask = make_dp_valid_mask(
            model_worker_batch.real_bs_per_dp,
            total_bs=total_bs,
            per_dp_bs=model_worker_batch.per_dp_bs_size,
        )
        safe_indices = np.where(
            valid_mask,
            np.asarray(model_worker_batch.req_pool_indices, dtype=np.int32),
            0,
        )
        data_sharding = NamedSharding(self.mesh, P("data"))
        relay_indices = jax.device_put(safe_indices, data_sharding)
        relay_valid_mask = jax.device_put(valid_mask, data_sharding)
        relay_new_seq_lens = jax.device_put(
            np.asarray(seq_lens, dtype=np.int32) + 1,
            data_sharding,
        )
        self._update_relay(
            next_token_ids,
            relay_new_seq_lens,
            relay_indices,
            relay_valid_mask,
            dp_size=model_worker_batch.dp_size,
        )

        output_token_ids = _prepare_spec_prefill_output_token_ids(self, next_token_ids)
        output_token_ids.copy_to_host_async()
        future_indices = np.asarray(model_worker_batch.req_pool_indices, dtype=np.int32)[sel]
        next_draft_input = DFlashDraftInput(
            future_indices=future_indices,
            block_size=self.block_size,
            **self._draft_input_config(),
        )
        model_worker_batch.spec_info_padded = next_draft_input
        launch_done = getattr(model_worker_batch, "launch_done", None)
        if launch_done is not None:
            launch_done.set()
        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=output_token_ids,
            next_draft_input=next_draft_input,
            spec_relay_buffers=self.spec_relay_buffers,
            prefill_relay_future_indices=relay_indices,
            bid=bid,
            cache_miss_count=cache_miss_count,
            extend_input_len_per_req=None,
            extend_logprob_start_len_per_req=None,
        )

    def forward_batch_speculative_decode_overlap(self, model_worker_batch: ModelWorkerBatch):
        if not model_worker_batch.forward_mode.is_decode():
            raise NotImplementedError("DFLASH decode overlap requires a decode batch.")

        self.init_spec_relay_buffers()
        input_state = model_worker_batch.spec_info_padded
        if not isinstance(input_state, DFlashDraftInput) or input_state.future_indices is None:
            raise RuntimeError("DFLASH overlap decode requires relay-backed draft state.")

        self.draft(model_worker_batch)
        batch_output = self.verify(model_worker_batch, update_relay=True)
        published_new_seq_lens = batch_output.next_draft_input.new_seq_lens

        sel = np.asarray(model_worker_batch.logits_indices_selector, dtype=np.int32)
        next_draft_input = DFlashDraftInput(
            future_indices=np.asarray(model_worker_batch.req_pool_indices, dtype=np.int32)[sel],
            block_size=self.block_size,
            **self._draft_input_config(),
        )
        batch_output.next_draft_input = next_draft_input
        batch_output.spec_relay_buffers = self.spec_relay_buffers
        model_worker_batch.spec_info_padded = next_draft_input

        launch_done = getattr(model_worker_batch, "launch_done", None)
        if launch_done is not None:
            launch_done.set()
        return batch_output, published_new_seq_lens

    def run_spec_decode_precompile(self):
        self._precompile_dflash_prefill()
        manager = self._target_compilation_manager
        dp_size = int(manager.dp_size)
        bs_buckets = [
            int(bs)
            for bs in manager.bs_buckets
            if int(bs) >= dp_size and int(bs) % dp_size == 0 and int(bs) & (int(bs) - 1) == 0
        ]
        if not bs_buckets:
            max_bs = int(manager.max_padded_batch_size)
            if max_bs % dp_size != 0:
                raise ValueError(
                    "DFLASH precompile batch size must be divisible by dp_size: "
                    f"max_padded_batch_size={max_bs}, dp_size={dp_size}."
                )
            bs_buckets = [max_bs]

        logger.info(
            "[DFLASH] Precompiling one fixed-page variant per bs: bs=%s, page_indices_capacity=%s",
            bs_buckets,
            [self._page_indices_capacity(bs) for bs in bs_buckets],
        )
        for bs in bs_buckets:
            self._precompile_dflash_variant(bs)

    def _build_draft_forward_plan(
        self,
        model_worker_batch: ModelWorkerBatch,
        draft_input: DFlashDraftInput,
        target_prefix_lens: np.ndarray,
        draft_prefix_lens: np.ndarray,
        bs: int,
    ) -> DraftForwardPlan:
        from jax.sharding import NamedSharding
        from jax.sharding import PartitionSpec as P

        use_relay_state = draft_input.future_indices is not None
        verified_id = (
            np.zeros((bs,), dtype=np.int32)
            if use_relay_state
            else np.asarray(draft_input.verified_id, dtype=np.int32)
        )
        block_ids, positions = build_dflash_draft_block(
            verified_id=verified_id,
            mask_token_id=self._mask_token_id,
            target_prefix_lens=target_prefix_lens,
            block_size=self.draft_block_size,
        )
        block_ids_flat = block_ids.reshape(-1)
        positions_flat = positions.reshape(-1)

        draft_mwb = self._make_draft_block_mwb(
            model_worker_batch,
            block_ids_flat,
            positions_flat,
            draft_prefix_lens,
            use_relay_state=use_relay_state,
        )
        forward_batch = ForwardBatch.init_new(draft_mwb, self.draft_model_runner)
        forward_batch.forward_mode = ForwardMode.TARGET_VERIFY
        # Reuse ForwardBatch's device token buffer.
        draft_mwb.spec_info_padded.draft_token = forward_batch.input_ids
        forward_batch.spec_info = draft_mwb.spec_info_padded
        active_mask = self._active_decode_slot_mask(model_worker_batch, bs)
        mismatched_prefix = active_mask & (draft_prefix_lens != target_prefix_lens)
        if not use_relay_state and np.any(mismatched_prefix):
            slots = np.flatnonzero(mismatched_prefix)
            raise RuntimeError(
                "DFLASH target/draft prefix layouts diverged for active slots: "
                f"slots={slots.tolist()}, "
                f"target={target_prefix_lens[slots].tolist()}, "
                f"draft={draft_prefix_lens[slots].tolist()}."
            )
        allocated_lens = draft_input.allocate_lens
        if allocated_lens is None:
            allocated_lens = target_prefix_lens + self.block_size
        allocated_lens = np.asarray(allocated_lens, dtype=np.int32)
        reservation_base_lens = draft_input.reservation_base_lens
        if reservation_base_lens is None:
            reservation_base_lens = target_prefix_lens
        reservation_base_lens = np.asarray(reservation_base_lens, dtype=np.int32)
        if allocated_lens.shape != (bs,) or reservation_base_lens.shape != (bs,):
            raise ValueError(
                "DFLASH overlap allocation state must match the padded batch: "
                f"allocated={allocated_lens.shape}, base={reservation_base_lens.shape}, bs={bs}."
            )

        page_indices = self._build_dflash_page_indices(
            draft_mwb,
            draft_prefix_lens,
            bs,
            allocated_lens=allocated_lens if use_relay_state else None,
        )
        template = self._get_verify_bucket_template(
            draft_mwb,
            bs,
            token_num=self.draft_block_size,
        )
        metadata = self.draft_model_runner.attn_backend.get_eagle_forward_metadata(
            draft_mwb,
            page_indices=page_indices,
            page_indices_capacity=self._page_indices_capacity(bs),
            extend_seq_lens=template.extend_seq_lens,
            cu_q_lens=template.cu_q_lens,
            distribution=template.distribution,
        )
        relay_future_indices = (
            np.asarray(draft_input.future_indices, dtype=np.int32)
            if use_relay_state
            else np.zeros((bs,), dtype=np.int32)
        )
        relay_future_indices = np.where(active_mask, relay_future_indices, 0)
        data_sharding = NamedSharding(self.mesh, P("data"))
        feedback_sharding = NamedSharding(self.mesh, P("data", None))
        if use_relay_state:
            feedback_shape = (bs, self.block_size - 1)
            flashback_token_ids = np.zeros(feedback_shape, dtype=np.int32)
            flashback_target_margins = np.zeros(feedback_shape, dtype=np.float32)
            flashback_valid_mask = np.zeros(feedback_shape, dtype=np.bool_)
            rejected_draft_token_ids = np.zeros((bs,), dtype=np.int32)
            rejection_valid_mask = np.zeros((bs,), dtype=np.bool_)
            previous_accept_lens = np.zeros((bs,), dtype=np.int32)
        else:
            (
                flashback_token_ids,
                flashback_target_margins,
                flashback_valid_mask,
            ) = draft_input._flashback_rows(bs)
            flashback_valid_mask = flashback_valid_mask & active_mask[:, None]
            rejected_draft_token_ids, rejection_valid_mask = draft_input._rejection_rows(bs)
            rejection_valid_mask = rejection_valid_mask & active_mask
            previous_accept_lens = draft_input._previous_accept_rows(bs)
            previous_accept_lens = np.where(active_mask, previous_accept_lens, 0)
        target_correction_token_ids = np.where(active_mask, verified_id, 0).astype(np.int32)
        ngram_token_ids, ngram_bonus, ngram_valid_mask, ngram_match_lens = draft_input._ngram_rows(
            bs
        )
        ngram_valid_mask = ngram_valid_mask & active_mask[:, None]
        return DraftForwardPlan(
            forward_batch=forward_batch,
            forward_metadata=metadata,
            seq_lens=np.asarray(model_worker_batch.seq_lens, dtype=np.int32),
            target_prefix_lens=np.asarray(target_prefix_lens, dtype=np.int32),
            positions_host=positions_flat,
            page_indices=page_indices,
            allocated_lens=jax.device_put(allocated_lens, data_sharding),
            reservation_base_lens=jax.device_put(reservation_base_lens, data_sharding),
            relay_future_indices=jax.device_put(relay_future_indices, data_sharding),
            relay_valid_mask=jax.device_put(active_mask, data_sharding),
            flashback_token_ids=jax.device_put(flashback_token_ids, feedback_sharding),
            flashback_target_margins=jax.device_put(flashback_target_margins, feedback_sharding),
            flashback_valid_mask=jax.device_put(flashback_valid_mask, feedback_sharding),
            rejected_draft_token_ids=jax.device_put(rejected_draft_token_ids, data_sharding),
            rejection_valid_mask=rejection_valid_mask,
            previous_accept_lens=previous_accept_lens,
            target_correction_token_ids=target_correction_token_ids,
            ngram_token_ids=jax.device_put(ngram_token_ids, feedback_sharding),
            ngram_bonus=jax.device_put(ngram_bonus, feedback_sharding),
            ngram_valid_mask=jax.device_put(ngram_valid_mask, feedback_sharding),
            ngram_match_lens=jax.device_put(ngram_match_lens, data_sharding),
            use_relay_state=use_relay_state,
            dp_size=int(model_worker_batch.dp_size),
            bs=bs,
        )

    def _build_target_verify_plan(
        self,
        model_worker_batch: ModelWorkerBatch,
        draft_plan: DraftForwardPlan,
        draft_token: jax.Array,
        base_draft_token: jax.Array,
        top2_draft_token: jax.Array,
        top2_margins: jax.Array,
        redenoise_candidate_token: jax.Array,
        redenoise_prefix_lens: jax.Array,
        resolved_target_prefix_lens: jax.Array,
        resolved_positions: jax.Array,
        resolved_cache_loc: jax.Array,
        ngram_selected_mask: jax.Array,
        candidate_margins: jax.Array,
    ) -> TargetVerifyPlan:
        bs = draft_plan.bs
        target_mwb = copy.copy(model_worker_batch)
        target_mwb.forward_mode = ForwardMode.TARGET_VERIFY
        target_mwb.input_ids = np.empty((0,), dtype=np.int32)
        target_mwb.positions = (
            draft_plan.target_prefix_lens[:, None]
            + np.arange(self.block_size, dtype=np.int32)[None, :]
        ).reshape(-1)
        target_mwb.seq_lens = draft_plan.target_prefix_lens
        target_mwb.cache_loc = np.zeros(
            int(getattr(model_worker_batch, "dp_size", 1)), dtype=np.int32
        )
        target_mwb.capture_hidden_mode = CaptureHiddenMode.FULL
        target_mwb.forward_batch = None

        verify_input = DFlashVerifyInput(
            draft_token=draft_token,
            draft_token_num=self.block_size,
        )
        target_mwb.spec_info_padded = verify_input

        template = self._get_verify_bucket_template(
            target_mwb,
            bs,
            token_num=self.block_size,
        )
        metadata = self.target_worker.model_runner.attn_backend.get_eagle_forward_metadata(
            target_mwb,
            page_indices=draft_plan.page_indices,
            page_indices_capacity=self._page_indices_capacity(bs),
            extend_seq_lens=template.extend_seq_lens,
            cu_q_lens=template.cu_q_lens,
            distribution=template.distribution,
        )

        # Reuse proposal positions, request indices, and other device buffers
        # from the draft plan. In particular, do not upload MASK ids for target
        # verify and overwrite them with draft_token afterwards.
        target_forward_batch = copy.copy(draft_plan.forward_batch)
        target_forward_batch.forward_mode = ForwardMode.TARGET_VERIFY
        target_forward_batch.input_ids = draft_token
        target_forward_batch.positions = resolved_positions
        target_forward_batch.seq_lens = resolved_target_prefix_lens
        target_forward_batch.out_cache_loc = resolved_cache_loc
        target_forward_batch.cache_loc = None
        target_forward_batch.attn_backend = self.target_worker.model_runner.attn_backend
        target_forward_batch.spec_info = verify_input
        target_forward_batch.capture_hidden_mode = CaptureHiddenMode.FULL
        target_forward_batch.input_embedding = None

        return TargetVerifyPlan(
            model_worker_batch=target_mwb,
            forward_batch=target_forward_batch,
            forward_metadata=metadata,
            logits_metadata=LogitsMetadata.from_model_worker_batch(target_mwb, self.mesh),
            seq_lens=draft_plan.seq_lens,
            target_prefix_lens=draft_plan.target_prefix_lens,
            resolved_target_prefix_lens=resolved_target_prefix_lens,
            draft_extend_positions=resolved_positions,
            draft_extend_cache_loc=resolved_cache_loc,
            active_mask=template.active_mask,
            allocated_lens=draft_plan.allocated_lens,
            relay_future_indices=draft_plan.relay_future_indices,
            relay_valid_mask=draft_plan.relay_valid_mask,
            draft_token=draft_token,
            base_draft_token=base_draft_token,
            top2_draft_token=top2_draft_token,
            top2_margins=top2_margins,
            redenoise_candidate_token=redenoise_candidate_token,
            redenoise_prefix_lens=redenoise_prefix_lens,
            flashback_token_ids=draft_plan.flashback_token_ids,
            flashback_valid_mask=draft_plan.flashback_valid_mask,
            rejected_draft_token_ids=draft_plan.rejected_draft_token_ids,
            rejection_valid_mask=draft_plan.rejection_valid_mask,
            previous_accept_lens=draft_plan.previous_accept_lens,
            target_correction_token_ids=draft_plan.target_correction_token_ids,
            candidate_margins=candidate_margins,
            ngram_selected_mask=ngram_selected_mask,
            ngram_token_ids=draft_plan.ngram_token_ids,
            ngram_valid_mask=draft_plan.ngram_valid_mask,
            ngram_match_lens=draft_plan.ngram_match_lens,
        )

    def _make_draft_block_mwb(
        self,
        base_mwb: ModelWorkerBatch,
        block_ids_flat: np.ndarray,
        positions_flat: np.ndarray,
        prefix_lens: np.ndarray,
        *,
        use_relay_state: bool,
    ) -> ModelWorkerBatch:
        mwb = copy.copy(base_mwb)
        mwb.forward_mode = ForwardMode.TARGET_VERIFY
        mwb.input_ids = np.asarray(block_ids_flat, dtype=np.int32)
        mwb.positions = np.asarray(positions_flat, dtype=np.int32)
        mwb.seq_lens = np.asarray(prefix_lens, dtype=np.int32)
        mwb.capture_hidden_mode = CaptureHiddenMode.NULL
        mwb.spec_algorithm = SpeculativeAlgorithm.DFLASH
        mwb.spec_info_padded = DFlashVerifyInput(
            draft_token=block_ids_flat,
            draft_token_num=self.draft_block_size,
        )
        # DFlashDraftInput.prepare_for_decode already reserved and packed one
        # block per active request into the first half of each DP rank section.
        # Reuse that scheduler output rather than gathering the same slots from
        # req_to_token_pool again on every decode round.
        mwb.out_cache_loc = (
            np.asarray(base_mwb.out_cache_loc, dtype=np.int32)
            if use_relay_state
            else self._verify_write_cache_loc(base_mwb, token_num=self.block_size)
        )
        mwb.cache_loc = None
        return mwb

    def _init_jit_draft_block(self):
        from functools import partial as _partial

        from flax import nnx
        from jax.sharding import NamedSharding
        from jax.sharding import PartitionSpec as P

        from sgl_jax.srt.lora.context_manager import LoraBatchContext
        from sgl_jax.srt.model_executor.model_runner import _maybe_apply_recurrent_cow
        from sgl_jax.srt.speculative.draft_extend_fused import (
            _make_target_verify_metadata,
        )

        runner = self.draft_model_runner
        model_def = runner._model_def
        model_state_def = runner._model_state_def
        draft_block_size = self.draft_block_size
        verify_block_size = self.block_size
        enable_anchor = self.enable_anchor
        redenoise_enabled = self._redenoise_enabled
        top2_shadow_enabled = self._top2_shadow_enabled
        redenoise_margin_threshold = self._redenoise_margin_threshold
        redenoise_prefix_len = self._redenoise_prefix_len
        redenoise_apply_start = self._redenoise_apply_start
        vocab_size = self._target_vocab_size
        embedding_sharding = NamedSharding(runner.mesh, P("data", "tensor"))
        logits_sharding = NamedSharding(runner.mesh, P("data", "tensor"))
        token_sharding = NamedSharding(runner.mesh, P("data"))
        cache_row_sharding = NamedSharding(runner.mesh, P("data", None))
        candidate_margin_sharding = NamedSharding(runner.mesh, P("data", None, None))

        @_partial(
            jax.jit,
            donate_argnames=["memory_pools"],
            static_argnames=[
                "model_state_def",
                "draft_block_size",
                "verify_block_size",
                "enable_anchor",
                "vocab_size",
                "use_relay_state",
                "ngram_enabled",
                "ngram_max_rerank_positions",
                "flashback_enabled",
                "feedback_shadow_enabled",
                "redenoise_enabled",
                "top2_shadow_enabled",
                "redenoise_apply_start",
                "dp_size",
            ],
        )
        def draft(
            model_def,
            model_state_def,
            model_state_leaves,
            forward_batch,
            memory_pools,
            embed,
            lm_head,
            relay_buffers,
            relay_future_indices,
            relay_valid_mask,
            allocated_lens,
            reservation_base_lens,
            flashback_token_ids,
            flashback_target_margins,
            flashback_valid_mask,
            rejected_draft_token_ids,
            ngram_token_ids,
            ngram_bonus,
            ngram_valid_mask,
            *,
            draft_block_size: int,
            verify_block_size: int,
            enable_anchor: bool,
            vocab_size: int,
            use_relay_state: bool,
            ngram_enabled: bool,
            flashback_enabled: bool,
            feedback_shadow_enabled: bool,
            redenoise_enabled: bool,
            top2_shadow_enabled: bool,
            redenoise_apply_start: int,
            ngram_max_rerank_positions: int,
            dp_size: int,
        ):
            target_prefix_lens = forward_batch.seq_lens
            if use_relay_state:
                (
                    verified_id,
                    relay_new_seq_lens,
                    flashback_token_ids,
                    flashback_target_margins,
                    flashback_valid_mask,
                ) = gather_dflash_relay_buffers(
                    relay_buffers,
                    relay_future_indices,
                    dp_size=dp_size,
                )
                target_prefix_lens = jnp.where(
                    relay_valid_mask,
                    relay_new_seq_lens - 1,
                    jnp.zeros_like(relay_new_seq_lens),
                )
                mask_rows = jnp.full(
                    (target_prefix_lens.shape[0], draft_block_size - 1),
                    jnp.int32(self._mask_token_id),
                )
                mask_rows = jax.sharding.reshard(mask_rows, cache_row_sharding)
                block_rows = jnp.concatenate([verified_id[:, None], mask_rows], axis=1)
                positions = (
                    target_prefix_lens[:, None]
                    + jnp.arange(draft_block_size, dtype=jnp.int32)[None, :]
                )
                forward_batch.input_ids = jax.sharding.reshard(
                    block_rows.reshape(-1), token_sharding
                )
                forward_batch.positions = jax.sharding.reshard(
                    positions.reshape(-1), token_sharding
                )
                forward_batch.seq_lens = target_prefix_lens
                forward_batch.attn_backend.forward_metadata = _make_target_verify_metadata(
                    forward_batch.attn_backend.forward_metadata,
                    target_prefix_lens,
                    allocated_lens,
                    speculative_num_draft_tokens=draft_block_size,
                    page_size=forward_batch.attn_backend.page_size,
                    dp_size=dp_size,
                )

            cache_rows = forward_batch.out_cache_loc.reshape((target_prefix_lens.shape[0], -1))
            cache_offsets = jnp.where(
                relay_valid_mask,
                target_prefix_lens - reservation_base_lens,
                jnp.zeros_like(target_prefix_lens),
            )
            cache_offsets = jnp.clip(
                cache_offsets,
                0,
                cache_rows.shape[1] - verify_block_size,
            )
            draft_cache_indices = (
                cache_offsets[:, None] + jnp.arange(draft_block_size, dtype=jnp.int32)[None, :]
            )
            verify_cache_indices = (
                cache_offsets[:, None] + jnp.arange(verify_block_size, dtype=jnp.int32)[None, :]
            )
            row_indices = jnp.arange(cache_rows.shape[0], dtype=jnp.int32)[:, None]
            selected_draft_cache_loc = cache_rows.at[row_indices, draft_cache_indices].get(
                out_sharding=cache_row_sharding,
            )
            selected_draft_cache_loc = jnp.where(
                relay_valid_mask[:, None],
                selected_draft_cache_loc,
                jnp.int32(-1),
            ).reshape(-1)
            selected_verify_cache_loc = cache_rows.at[row_indices, verify_cache_indices].get(
                out_sharding=cache_row_sharding,
            )
            selected_verify_cache_loc = jnp.where(
                relay_valid_mask[:, None],
                selected_verify_cache_loc,
                jnp.int32(-1),
            ).reshape(-1)
            forward_batch.out_cache_loc = selected_draft_cache_loc

            input_embedding = embed.at[forward_batch.input_ids].get(out_sharding=embedding_sharding)
            forward_batch.input_embedding = input_embedding
            model_state = jax.tree_util.tree_unflatten(model_state_def, model_state_leaves)
            model = nnx.merge(model_def, model_state)
            memory_pools = _maybe_apply_recurrent_cow(forward_batch, memory_pools)
            with LoraBatchContext.set_batch(forward_batch):
                output, pool_updates, _, _ = model(forward_batch, memory_pools, None)

            draft_hidden = output.hidden_states.reshape(
                (-1, draft_block_size, output.hidden_states.shape[-1])
            )
            proposal_hidden = select_dflash_proposal_hidden(
                draft_hidden,
                enable_anchor=enable_anchor,
            )
            proposal_flat = proposal_hidden.reshape((-1, proposal_hidden.shape[-1]))
            logits = jnp.dot(
                proposal_flat,
                lm_head.T,
                out_sharding=logits_sharding,
            )[:, :vocab_size]
            logits = logits.reshape(proposal_hidden.shape[:-1] + (vocab_size,))
            ngram_selected_mask = jnp.zeros(proposal_hidden.shape[:-1], dtype=jnp.bool_)
            if flashback_enabled:
                draft_next = select_dflash_flashback_tokens(
                    logits,
                    flashback_token_ids,
                    flashback_target_margins,
                    flashback_valid_mask,
                    bonus=self._flashback_bonus,
                    target_margin_weight=self._flashback_target_margin_weight,
                    position_decay=self._flashback_position_decay,
                )
            elif ngram_enabled:
                draft_next, ngram_selected_mask = select_dflash_ngram_tokens(
                    logits,
                    ngram_token_ids,
                    ngram_bonus,
                    ngram_valid_mask,
                    max_rerank_positions=ngram_max_rerank_positions,
                )
            else:
                draft_next = jnp.argmax(logits, axis=-1).astype(jnp.int32)
            base_draft_next = draft_next
            if top2_shadow_enabled:
                top2_values, top2_ids = dflash_sharded_top_k(logits, 2)
                top2_next = top2_ids[..., 1].astype(jnp.int32)
                top2_margins = (top2_values[..., 0] - top2_values[..., 1]).astype(jnp.float32)
            else:
                top2_next = draft_next
                top2_margins = jnp.zeros(logits.shape[:-1], dtype=jnp.float32)
            redenoise_candidate_next = draft_next
            prefix_lens = jnp.zeros((draft_next.shape[0],), dtype=jnp.int32)
            if redenoise_enabled:
                # The second pass is proposal-only.  Persisting its KV writes
                # would make even a shadow run change the next draft round,
                # because the re-denoise input contains a different mix of
                # visible and masked tokens than the canonical DFlash pass.
                # Keep the first-pass cache as the authoritative draft state
                # and discard the second-pass pool updates below.
                first_pass_pool_updates = pool_updates
                first_pass_tokens = draft_next
                if redenoise_prefix_len >= 0:
                    margins = jnp.zeros(logits.shape[:-1], dtype=jnp.float32)
                else:
                    margins = dflash_top2_margins(logits)
                prefix_lens = select_dflash_redenoise_prefix_lens(
                    margins,
                    margin_threshold=redenoise_margin_threshold,
                    fixed_prefix_len=redenoise_prefix_len,
                    max_prefix_len=draft_block_size - 1,
                )
                anchor_ids = forward_batch.input_ids.reshape((-1, draft_block_size))[:, 0]
                redenoise_rows = build_dflash_redenoise_block(
                    anchor_ids,
                    first_pass_tokens,
                    prefix_lens,
                    mask_token_id=self._mask_token_id,
                    draft_block_size=draft_block_size,
                )
                forward_batch.input_ids = jax.sharding.reshard(
                    redenoise_rows.reshape(-1), token_sharding
                )
                forward_batch.input_embedding = embed.at[forward_batch.input_ids].get(
                    out_sharding=embedding_sharding
                )
                memory_pools.replace_all(pool_updates)
                with LoraBatchContext.set_batch(forward_batch):
                    second_output, _, _, _ = model(forward_batch, memory_pools, None)
                second_hidden = second_output.hidden_states.reshape(
                    (-1, draft_block_size, second_output.hidden_states.shape[-1])
                )
                second_proposal_hidden = select_dflash_proposal_hidden(
                    second_hidden,
                    enable_anchor=enable_anchor,
                )
                second_flat = second_proposal_hidden.reshape((-1, second_proposal_hidden.shape[-1]))
                second_logits = jnp.dot(
                    second_flat,
                    lm_head.T,
                    out_sharding=logits_sharding,
                )[:, :vocab_size]
                second_logits = second_logits.reshape(
                    second_proposal_hidden.shape[:-1] + (vocab_size,)
                )
                second_tokens = jnp.argmax(second_logits, axis=-1).astype(jnp.int32)
                redenoise_candidate_next = second_tokens
                draft_next = merge_dflash_redenoise_tokens(
                    first_pass_tokens,
                    second_tokens,
                    prefix_lens,
                    apply_start=redenoise_apply_start,
                )
                pool_updates = first_pass_pool_updates
            if feedback_shadow_enabled:
                base_token_ids = jnp.argmax(logits, axis=-1).astype(jnp.int32)
                base_scores = _gather_dflash_vocab_scores(logits, base_token_ids)
                rejected_rows = jnp.broadcast_to(
                    rejected_draft_token_ids[:, None], base_token_ids.shape
                )
                candidate_scores = jnp.stack(
                    [
                        _gather_dflash_vocab_scores(logits, rejected_rows),
                        _gather_dflash_vocab_scores(logits, flashback_token_ids),
                        _gather_dflash_vocab_scores(logits, ngram_token_ids),
                    ],
                    axis=-1,
                )
                candidate_margins = (base_scores[..., None] - candidate_scores).astype(jnp.float32)
                candidate_margins = jax.sharding.reshard(
                    candidate_margins, candidate_margin_sharding
                )
            else:
                candidate_margins = jax.sharding.reshard(
                    jnp.zeros(logits.shape[:-1] + (3,), dtype=jnp.float32),
                    candidate_margin_sharding,
                )
            seed = forward_batch.input_ids.reshape((-1, draft_block_size))[:, :1]
            seed = jax.sharding.reshard(seed, cache_row_sharding)
            base_draft_next = jax.sharding.reshard(base_draft_next, cache_row_sharding)
            top2_next = jax.sharding.reshard(top2_next, cache_row_sharding)
            top2_margins = jax.sharding.reshard(top2_margins, cache_row_sharding)
            redenoise_candidate_next = jax.sharding.reshard(
                redenoise_candidate_next, cache_row_sharding
            )
            draft_next = jax.sharding.reshard(draft_next, cache_row_sharding)
            prefix_lens = jax.sharding.reshard(prefix_lens, token_sharding)
            base_draft_token = jnp.concatenate([seed, base_draft_next], axis=1).reshape(-1)
            base_draft_token = jax.sharding.reshard(base_draft_token, token_sharding)
            top2_draft_token = jnp.concatenate([seed, top2_next], axis=1).reshape(-1)
            top2_draft_token = jax.sharding.reshard(top2_draft_token, token_sharding)
            redenoise_candidate_token = jnp.concatenate(
                [seed, redenoise_candidate_next], axis=1
            ).reshape(-1)
            redenoise_candidate_token = jax.sharding.reshard(
                redenoise_candidate_token, token_sharding
            )
            draft_token = jnp.concatenate([seed, draft_next], axis=1).reshape(-1)
            draft_token = jax.sharding.reshard(draft_token, token_sharding)
            verify_positions = (
                target_prefix_lens[:, None]
                + jnp.arange(verify_block_size, dtype=jnp.int32)[None, :]
            ).reshape(-1)
            verify_positions = jax.sharding.reshard(verify_positions, token_sharding)
            return (
                pool_updates,
                draft_token,
                base_draft_token,
                top2_draft_token,
                top2_margins,
                redenoise_candidate_token,
                prefix_lens,
                target_prefix_lens,
                verify_positions,
                selected_verify_cache_loc,
                ngram_selected_mask,
                candidate_margins,
            )

        self._jit_draft_block = _partial(
            draft,
            model_def,
            model_state_def,
            draft_block_size=draft_block_size,
            verify_block_size=verify_block_size,
            enable_anchor=enable_anchor,
            vocab_size=vocab_size,
            ngram_enabled=self._ngram_enabled,
            flashback_enabled=self._flashback_enabled,
            feedback_shadow_enabled=self._feedback_shadow_enabled,
            redenoise_enabled=redenoise_enabled,
            top2_shadow_enabled=top2_shadow_enabled,
            redenoise_apply_start=redenoise_apply_start,
            ngram_max_rerank_positions=self._ngram_max_rerank_positions,
        )

    def _run_jit_draft_block(self, plan: DraftForwardPlan):
        runner = self.draft_model_runner
        forward_batch = plan.forward_batch
        forward_batch.cache_loc = None

        def _call_and_replace():
            (
                pool_updates,
                draft_token,
                base_draft_token,
                top2_draft_token,
                top2_margins,
                redenoise_candidate_token,
                prefix_lens,
                target_prefix_lens,
                positions,
                cache_loc,
                ngram_selected_mask,
                candidate_margins,
            ) = self._jit_draft_block(
                runner.model_state_leaves,
                forward_batch,
                runner.memory_pools,
                self._target_embed,
                self._target_lm_head,
                self.spec_relay_buffers if plan.use_relay_state else None,
                plan.relay_future_indices,
                plan.relay_valid_mask,
                plan.allocated_lens,
                plan.reservation_base_lens,
                plan.flashback_token_ids,
                plan.flashback_target_margins,
                plan.flashback_valid_mask,
                plan.rejected_draft_token_ids,
                plan.ngram_token_ids,
                plan.ngram_bonus,
                plan.ngram_valid_mask,
                use_relay_state=plan.use_relay_state,
                dp_size=plan.dp_size,
            )
            self._replace_memory_pools(runner, pool_updates)
            return (
                draft_token,
                base_draft_token,
                top2_draft_token,
                top2_margins,
                redenoise_candidate_token,
                prefix_lens,
                target_prefix_lens,
                positions,
                cache_loc,
                ngram_selected_mask,
                candidate_margins,
            )

        with self._dispatch_context(runner):
            return _call_and_replace()

    def _init_jit_target_verify(self):
        """Build target model forward + DFlash greedy verification as one JIT."""
        from functools import partial as _partial

        from flax import nnx
        from jax.sharding import NamedSharding
        from jax.sharding import PartitionSpec as P

        from sgl_jax.srt.lora.context_manager import LoraBatchContext
        from sgl_jax.srt.model_executor.model_runner import _maybe_apply_recurrent_cow
        from sgl_jax.srt.speculative.draft_extend_fused import (
            _make_target_verify_metadata,
        )

        runner = self.target_worker.model_runner
        model_def = runner._model_def
        model_state_def = runner._model_state_def
        draft_token_num = self.block_size
        token_sharding = NamedSharding(runner.mesh, P("data"))
        feedback_sharding = NamedSharding(runner.mesh, P("data", None))

        @_partial(
            jax.jit,
            donate_argnames=["memory_pools", "relay_buffers"],
            static_argnames=[
                "model_state_def",
                "draft_token_num",
                "use_relay_state",
                "update_relay",
                "flashback_enabled",
                "page_size",
                "dp_size",
            ],
        )
        def target_verify(
            model_def,
            model_state_def,
            model_state_leaves,
            forward_batch,
            memory_pools,
            logits_metadata,
            draft_token,
            target_prefix_lens,
            allocated_lens,
            relay_buffers,
            relay_future_indices,
            relay_valid_mask,
            *,
            draft_token_num: int,
            use_relay_state: bool,
            update_relay: bool,
            flashback_enabled: bool,
            page_size: int,
            dp_size: int,
        ):
            forward_batch.seq_lens = target_prefix_lens
            if use_relay_state:
                forward_batch.attn_backend.forward_metadata = _make_target_verify_metadata(
                    forward_batch.attn_backend.forward_metadata,
                    target_prefix_lens,
                    allocated_lens,
                    speculative_num_draft_tokens=draft_token_num,
                    page_size=page_size,
                    dp_size=dp_size,
                )
            model_state = jax.tree_util.tree_unflatten(model_state_def, model_state_leaves)
            model = nnx.merge(model_def, model_state)
            memory_pools = _maybe_apply_recurrent_cow(forward_batch, memory_pools)
            with LoraBatchContext.set_batch(forward_batch):
                output, pool_updates, _, layers_topk_ids = model(
                    forward_batch, memory_pools, logits_metadata
                )
            (
                accept_lens_out,
                next_token_ids_flat,
                new_verified_id,
                accept_len_draft,
            ) = dflash_greedy_verify(
                draft_token,
                output.next_token_logits,
                draft_token_num=draft_token_num,
            )
            if flashback_enabled:
                (
                    flashback_token_ids,
                    flashback_target_margins,
                    flashback_valid_mask,
                ) = build_dflash_flashback_feedback(
                    draft_token,
                    output.next_token_logits,
                    next_token_ids_flat,
                    accept_len_draft,
                    draft_token_num=draft_token_num,
                )
            else:
                feedback_shape = (accept_lens_out.shape[0], draft_token_num - 1)
                flashback_token_ids = jnp.zeros(feedback_shape, dtype=jnp.int32)
                flashback_target_margins = jnp.zeros(feedback_shape, dtype=jnp.float32)
                flashback_valid_mask = jnp.zeros(feedback_shape, dtype=jnp.bool_)
            flashback_token_ids = jax.sharding.reshard(flashback_token_ids, feedback_sharding)
            flashback_target_margins = jax.sharding.reshard(
                flashback_target_margins.astype(jnp.float32), feedback_sharding
            )
            flashback_valid_mask = jax.sharding.reshard(flashback_valid_mask, feedback_sharding)
            accept_lens_out = jax.sharding.reshard(accept_lens_out, token_sharding)
            next_token_ids_flat = jax.sharding.reshard(next_token_ids_flat, token_sharding)
            new_verified_id = jax.sharding.reshard(new_verified_id, token_sharding)
            new_seq_lens = (target_prefix_lens + 1 + accept_lens_out).astype(jnp.int32)
            new_seq_lens = jax.sharding.reshard(new_seq_lens, token_sharding)
            updated_relay_buffers = relay_buffers
            if update_relay:
                updated_relay_buffers = update_dflash_relay_buffers(
                    relay_buffers,
                    relay_future_indices,
                    relay_valid_mask,
                    new_verified_id,
                    new_seq_lens,
                    flashback_token_ids,
                    flashback_target_margins,
                    flashback_valid_mask,
                    dp_size=dp_size,
                )
            return (
                output,
                pool_updates,
                layers_topk_ids,
                accept_lens_out,
                next_token_ids_flat,
                new_verified_id,
                new_seq_lens,
                flashback_token_ids,
                flashback_target_margins,
                flashback_valid_mask,
                updated_relay_buffers,
            )

        self._jit_target_verify = _partial(
            target_verify,
            model_def,
            model_state_def,
            draft_token_num=draft_token_num,
            flashback_enabled=(self._flashback_enabled or self._feedback_shadow_enabled),
            page_size=self.page_size,
        )

    def _run_jit_target_verify(
        self,
        plan: TargetVerifyPlan,
    ):
        import jax._src.test_util as jtu

        target_worker = self.target_worker
        runner = target_worker.model_runner
        model_worker_batch = plan.model_worker_batch

        if target_worker.worker.server_args.enable_lora and target_worker.need_prepare_lora_batch:
            target_worker.prepare_lora_batch(model_worker_batch)

        def _call_and_replace():
            with jtu.count_pjit_cpp_cache_miss() as count:
                (
                    output,
                    pool_updates,
                    layers_topk_ids,
                    accept_lens_out,
                    next_token_ids_flat,
                    new_verified_id,
                    new_seq_lens,
                    flashback_token_ids,
                    flashback_target_margins,
                    flashback_valid_mask,
                    updated_relay_buffers,
                ) = self._jit_target_verify(
                    runner.model_state_leaves,
                    plan.forward_batch,
                    runner.memory_pools,
                    plan.logits_metadata,
                    plan.forward_batch.input_ids,
                    plan.resolved_target_prefix_lens,
                    plan.allocated_lens,
                    self.spec_relay_buffers if plan.update_relay else None,
                    plan.relay_future_indices,
                    plan.relay_valid_mask,
                    use_relay_state=plan.update_relay,
                    update_relay=plan.update_relay,
                    dp_size=plan.model_worker_batch.dp_size,
                )
                cache_miss_count = count()

            self._replace_memory_pools(runner, pool_updates)
            return (
                output,
                cache_miss_count,
                layers_topk_ids,
                accept_lens_out,
                next_token_ids_flat,
                new_verified_id,
                new_seq_lens,
                flashback_token_ids,
                flashback_target_margins,
                flashback_valid_mask,
                updated_relay_buffers,
            )

        with self._dispatch_context(runner):
            (
                output,
                cache_miss_count,
                layers_topk_ids,
                accept_lens_out,
                next_token_ids_flat,
                new_verified_id,
                new_seq_lens,
                flashback_token_ids,
                flashback_target_margins,
                flashback_valid_mask,
                updated_relay_buffers,
            ) = _call_and_replace()

        if plan.update_relay:
            self.spec_relay_buffers = updated_relay_buffers

        return (
            output,
            cache_miss_count,
            accept_lens_out,
            next_token_ids_flat,
            new_verified_id,
            new_seq_lens,
            flashback_token_ids,
            flashback_target_margins,
            flashback_valid_mask,
            layers_topk_ids,
        )

    def _init_jit_kv_materialize(self):
        """Fuse draft KV projection, merge, and cache writes into one JIT."""
        from functools import partial as _partial

        from flax import nnx
        from jax.sharding import NamedSharding
        from jax.sharding import PartitionSpec as P

        from sgl_jax.srt.mem_cache.memory_pool import _set_fused_kv_buffer, merge_kv

        runner = self.draft_model_runner
        pool = runner.token_to_kv_pool
        page_size = pool.page_size
        kv_part = pool.kv_partition_axis
        data_part = pool.attention_data_partition_axis
        mesh = pool.mesh
        n_layers = self.draft_layers
        vector_sharding = NamedSharding(mesh, P("data"))
        hidden_sharding = NamedSharding(mesh, P("data", None))

        model_def = runner._model_def
        model_state_def = runner._model_state_def

        @_partial(
            jax.jit,
            static_argnames=["model_state_def"],
            donate_argnames=["kv_buffers"],
        )
        def draft_extend(
            model_def,
            model_state_def,
            model_state_leaves,
            target_hidden,
            positions,
            cache_loc,
            accept_lens,
            active_mask,
            kv_buffers,
        ):
            positions = jax.sharding.reshard(positions.astype(jnp.int32), vector_sharding)
            cache_loc = jax.sharding.reshard(cache_loc.astype(jnp.int32), vector_sharding)
            accept_lens = jax.sharding.reshard(accept_lens.astype(jnp.int32), vector_sharding)
            active_mask = jax.sharding.reshard(active_mask.astype(jnp.bool_), vector_sharding)
            target_hidden = jax.sharding.reshard(target_hidden, hidden_sharding)

            cache_loc = _mask_draft_kv_writes(
                cache_loc,
                accept_lens,
                active_mask,
            )

            state = jax.tree_util.tree_unflatten(model_state_def, model_state_leaves)
            model = nnx.merge(model_def, state)
            kv_list = model.materialize_kv(target_hidden, positions)
            new_bufs = []
            for i in range(n_layers):
                k, v = kv_list[i]
                fused = merge_kv(k, v)
                new_bufs.append(
                    _set_fused_kv_buffer(
                        fused,
                        cache_loc,
                        kv_buffers[i],
                        page_size,
                        kv_part,
                        data_part,
                        mesh,
                    )
                )
            return new_bufs

        self._jit_materialize_write = _partial(draft_extend, model_def, model_state_def)

    def _append_target_hidden_to_draft_kv(
        self,
        model_worker_batch: ModelWorkerBatch,
        draft_input: DFlashDraftInput,
    ) -> None:
        target_hidden = draft_input.target_hidden
        if target_hidden is None or int(np.asarray(draft_input.ctx_lens).sum()) == 0:
            return

        positions, cache_loc = self._prefill_draft_extend_metadata(
            model_worker_batch,
            target_hidden,
        )

        self._run_jit_draft_extend(target_hidden, positions, cache_loc)

        draft_input.draft_seq_lens = np.asarray(
            draft_input.draft_seq_lens, dtype=np.int32
        ) + np.asarray(draft_input.ctx_lens, dtype=np.int32)
        draft_input.ctx_lens = np.zeros_like(np.asarray(draft_input.ctx_lens, dtype=np.int32))
        draft_input.target_hidden = None

    def _run_jit_draft_extend(
        self,
        target_hidden,
        positions,
        cache_loc,
        *,
        accept_lens=None,
        active_mask=None,
    ):
        pool = self.draft_model_runner.token_to_kv_pool
        if accept_lens is None:
            # Prefill metadata is already on the host. Build its fixed masks with
            # NumPy so they become inputs to jit_draft_extend instead of separate
            # broadcast/compare JAX launches.
            cache_loc = np.asarray(cache_loc, dtype=np.int32)
            accept_lens = np.ones((target_hidden.shape[0],), dtype=np.int32)
            if active_mask is None:
                active_mask = cache_loc >= 0
        cache_loc = jnp.asarray(cache_loc)
        accept_lens = jnp.asarray(accept_lens)
        active_mask = jnp.asarray(active_mask)
        if cache_loc.shape[0] % accept_lens.shape[0] != 0:
            raise ValueError(
                "DFLASH draft extend cache rows do not match accept_lens: "
                f"cache_loc={cache_loc.shape}, accept_lens={accept_lens.shape}."
            )
        new_buffers = self._jit_materialize_write(
            self.draft_model_runner.model_state_leaves,
            jnp.asarray(target_hidden),
            jnp.asarray(positions),
            cache_loc,
            accept_lens,
            active_mask,
            list(pool.kv_buffer[: self.draft_layers]),
        )
        for i, buf in enumerate(new_buffers):
            pool.kv_buffer[i] = buf

    @staticmethod
    def _prefill_draft_extend_metadata(model_worker_batch, target_hidden):
        """Reuse target-prefill's DP-segmented token metadata.

        Target hidden rows use ``[rank0 tokens + pad | rank1 tokens + pad | ...]``.
        Reusing the matching positions/out-cache buffers preserves those rank
        boundaries for the ``P("data")`` KV materialization JIT.
        """
        positions = np.asarray(model_worker_batch.positions, dtype=np.int32).reshape(-1)
        cache_loc = np.asarray(model_worker_batch.out_cache_loc, dtype=np.int32).reshape(-1)
        if positions.shape != cache_loc.shape:
            raise ValueError(
                "DFLASH prefill positions/cache_loc shape mismatch: "
                f"{positions.shape} vs {cache_loc.shape}."
            )

        bucket_tokens = int(target_hidden.shape[0])
        metadata_tokens = int(positions.shape[0])
        if metadata_tokens != bucket_tokens:
            raise ValueError(
                "DFLASH prefill metadata must match the target hidden bucket: "
                f"metadata_tokens={metadata_tokens}, bucket_tokens={bucket_tokens}."
            )
        return positions, cache_loc

    def _update_relay(
        self,
        verified_id,
        new_seq_lens,
        future_indices,
        valid_mask,
        flashback_token_ids=None,
        flashback_target_margins=None,
        flashback_valid_mask=None,
        *,
        dp_size: int,
    ):
        from functools import partial as _partial

        from jax.sharding import NamedSharding
        from jax.sharding import PartitionSpec as P

        data_sharding = NamedSharding(self.mesh, P("data"))
        feedback_sharding = NamedSharding(self.mesh, P("data", None))
        feedback_shape = (verified_id.shape[0], self.block_size - 1)
        if flashback_token_ids is None:
            flashback_token_ids = jnp.zeros(feedback_shape, dtype=jnp.int32)
            flashback_target_margins = jnp.zeros(feedback_shape, dtype=jnp.float32)
            flashback_valid_mask = jnp.zeros(feedback_shape, dtype=jnp.bool_)

        if not hasattr(self, "_jit_update_dflash_relay"):

            @_partial(jax.jit, donate_argnames=["buffers"], static_argnames=["dp_size"])
            def update(
                buffers,
                indices,
                mask,
                token_ids,
                seq_lens,
                feedback_ids,
                feedback_margins,
                feedback_valid,
                *,
                dp_size: int,
            ):
                indices = jax.sharding.reshard(indices, data_sharding)
                mask = jax.sharding.reshard(mask, data_sharding)
                token_ids = jax.sharding.reshard(token_ids, data_sharding)
                seq_lens = jax.sharding.reshard(seq_lens, data_sharding)
                feedback_ids = jax.sharding.reshard(feedback_ids, feedback_sharding)
                feedback_margins = jax.sharding.reshard(feedback_margins, feedback_sharding)
                feedback_valid = jax.sharding.reshard(feedback_valid, feedback_sharding)
                return update_dflash_relay_buffers(
                    buffers,
                    indices,
                    mask,
                    token_ids,
                    seq_lens,
                    feedback_ids,
                    feedback_margins,
                    feedback_valid,
                    dp_size=dp_size,
                )

            self._jit_update_dflash_relay = update

        with jax.set_mesh(self.mesh):
            self.spec_relay_buffers = self._jit_update_dflash_relay(
                self.spec_relay_buffers,
                future_indices,
                valid_mask,
                verified_id,
                new_seq_lens,
                flashback_token_ids,
                flashback_target_margins,
                flashback_valid_mask,
                dp_size=dp_size,
            )

    @staticmethod
    def _unpad_draft_state(
        draft_input: DFlashDraftInput,
        selector: np.ndarray,
    ) -> None:
        selector = np.asarray(selector, dtype=np.int32)
        for field in (
            "verified_id",
            "ctx_lens",
            "draft_seq_lens",
            "flashback_token_ids",
            "flashback_target_margins",
            "flashback_valid_mask",
            "rejected_draft_token_ids",
            "rejection_valid_mask",
            "previous_accept_lens",
            "ngram_token_ids",
            "ngram_bonus",
            "ngram_valid_mask",
            "ngram_match_lens",
        ):
            value = getattr(draft_input, field, None)
            if value is None:
                continue
            dtype = (
                np.float32
                if field in ("flashback_target_margins", "ngram_bonus")
                else (
                    np.bool_
                    if field
                    in (
                        "flashback_valid_mask",
                        "rejection_valid_mask",
                        "ngram_valid_mask",
                    )
                    else np.int32
                )
            )
            value = np.asarray(value, dtype=dtype)
            if selector.size and int(selector.max()) >= value.shape[0]:
                raise ValueError(
                    "DFLASH state selector is out of bounds: "
                    f"field={field}, shape={value.shape}, selector={selector}."
                )
            setattr(draft_input, field, value[selector])

    def _trim_draft_state_to_bs(
        self,
        draft_input: DFlashDraftInput,
        bs: int,
    ) -> None:
        draft_seq_lens = np.asarray(draft_input.draft_seq_lens, dtype=np.int32)
        state_bs = int(draft_seq_lens.shape[0])
        if state_bs == bs:
            return

        verified_id = np.asarray(draft_input.verified_id, dtype=np.int32)
        ctx_lens = np.asarray(draft_input.ctx_lens, dtype=np.int32)
        if state_bs > bs:
            draft_input.draft_seq_lens = draft_seq_lens[:bs]
            draft_input.verified_id = verified_id[:bs]
            draft_input.ctx_lens = ctx_lens[:bs]
            for field in (
                "flashback_token_ids",
                "flashback_target_margins",
                "flashback_valid_mask",
                "rejected_draft_token_ids",
                "rejection_valid_mask",
                "previous_accept_lens",
                "ngram_token_ids",
                "ngram_bonus",
                "ngram_valid_mask",
                "ngram_match_lens",
            ):
                value = getattr(draft_input, field, None)
                if value is not None:
                    setattr(draft_input, field, value[:bs])
            return

        raise ValueError(
            "DFLASH draft state is shorter than decode batch after prepare_for_decode: "
            f"state_bs={state_bs}, bs={bs}. Merged decode requests must be aligned "
            "from ScheduleBatch req state before entering the DFlash draft phase."
        )

    def _page_indices_capacity(self, bs: int) -> int:
        return min(
            self._page_indices_pool_capacity,
            max(int(bs), 1) * self._page_indices_per_seq_capacity,
        )

    def _build_dflash_page_indices(
        self,
        model_worker_batch: ModelWorkerBatch,
        prefix_lens: np.ndarray,
        bs: int,
        allocated_lens: np.ndarray | None = None,
    ) -> np.ndarray:
        """Build fixed-capacity, DP-segmented page indices from req_to_token."""
        dp_size = int(getattr(model_worker_batch, "dp_size", 1))
        per_dp_bs = int(getattr(model_worker_batch, "per_dp_bs_size", bs))
        if dp_size * per_dp_bs != bs:
            raise ValueError(
                "DFLASH page layout has inconsistent DP metadata: "
                f"dp_size={dp_size}, per_dp_bs={per_dp_bs}, bs={bs}."
            )

        capacity = self._page_indices_capacity(bs)
        if capacity % dp_size != 0:
            raise ValueError(
                "DFLASH page_indices capacity must be divisible by dp_size: "
                f"capacity={capacity}, dp_size={dp_size}."
            )
        per_rank_capacity = capacity // dp_size

        prefix_lens = np.asarray(prefix_lens, dtype=np.int32)
        if prefix_lens.shape != (bs,):
            raise ValueError(
                f"DFLASH prefix_lens must have shape ({bs},), got {prefix_lens.shape}."
            )
        req_pool_indices = np.asarray(model_worker_batch.req_pool_indices, dtype=np.int64)
        if req_pool_indices.shape != (bs,):
            raise ValueError(
                "DFLASH req_pool_indices must match the padded batch: "
                f"shape={req_pool_indices.shape}, bs={bs}."
            )

        selector = getattr(model_worker_batch, "logits_indices_selector", None)
        if selector is None:
            real_bs = int(getattr(model_worker_batch, "real_bs", bs))
            selector = np.arange(real_bs, dtype=np.int32)
        else:
            selector = np.asarray(selector, dtype=np.int32)
        if selector.size and (int(selector.min()) < 0 or int(selector.max()) >= bs):
            raise ValueError(f"DFLASH active-slot selector is out of bounds: {selector}.")
        active = np.zeros(bs, dtype=bool)
        active[selector] = True

        req_to_token = self.req_to_token_pool.req_to_token
        if allocated_lens is None:
            allocated_lens = prefix_lens + self.block_size
        allocated_lens = np.asarray(allocated_lens, dtype=np.int32)
        if allocated_lens.shape != (bs,):
            raise ValueError(
                f"DFLASH allocated_lens must have shape ({bs},), got {allocated_lens.shape}."
            )
        kv_lens = np.where(active, allocated_lens, 0).astype(np.int32)
        invalid_prefix = active & (prefix_lens < 0)
        if np.any(invalid_prefix):
            slot = int(np.flatnonzero(invalid_prefix)[0])
            raise ValueError(
                f"DFLASH active slot {slot} has invalid prefix_len={int(prefix_lens[slot])}."
            )
        overflow = kv_lens > req_to_token.shape[1]
        if np.any(overflow):
            slot = int(np.flatnonzero(overflow)[0])
            raise ValueError(
                "DFLASH KV length exceeds req_to_token capacity: "
                f"slot={slot}, kv_len={int(kv_lens[slot])}, "
                f"capacity={req_to_token.shape[1]}."
            )

        page_counts = (kv_lens + self.page_size - 1) // self.page_size
        max_pages = int(page_counts.max(initial=0))
        if max_pages:
            page_offsets = np.arange(max_pages, dtype=np.int64) * self.page_size
            valid_pages = page_offsets[None, :] < kv_lens[:, None]
            safe_req_indices = np.where(active, req_pool_indices, 0)
            page_locs = np.asarray(
                req_to_token[safe_req_indices[:, None], page_offsets[None, :]],
                dtype=np.int32,
            )
            incomplete = valid_pages & (page_locs < 0)
            if np.any(incomplete):
                slot = int(np.argwhere(incomplete)[0, 0])
                raise RuntimeError(
                    "DFLASH paged KV slots are incomplete: "
                    f"slot={slot}, req_pool_index={int(req_pool_indices[slot])}, "
                    f"kv_len={int(kv_lens[slot])}."
                )
            page_ids = page_locs // self.page_size
        else:
            valid_pages = np.zeros((bs, 0), dtype=bool)
            page_ids = np.zeros((bs, 0), dtype=np.int32)

        rank_chunks = []
        for dp_rank in range(dp_size):
            start = dp_rank * per_dp_bs
            end = start + per_dp_bs
            chunk = page_ids[start:end][valid_pages[start:end]].astype(np.int32, copy=False)
            if len(chunk) > per_rank_capacity:
                raise ValueError(
                    "DFLASH page_indices exceed the per-rank capacity: "
                    f"rank={dp_rank}, required={len(chunk)}, capacity={per_rank_capacity}."
                )
            rank_chunks.append(
                np.pad(chunk, (0, per_rank_capacity - len(chunk)), constant_values=0)
            )
        return np.concatenate(rank_chunks).astype(np.int32)

    def _get_verify_bucket_template(
        self,
        model_worker_batch: ModelWorkerBatch,
        bs: int,
        *,
        token_num: int | None = None,
    ) -> DFlashVerifyBucketTemplate:
        from jax.sharding import NamedSharding
        from jax.sharding import PartitionSpec as P

        dp_size = int(getattr(model_worker_batch, "dp_size", 1))
        per_dp_bs = int(getattr(model_worker_batch, "per_dp_bs_size", bs))
        selector = getattr(model_worker_batch, "logits_indices_selector", None)
        if selector is None:
            selector = np.arange(int(getattr(model_worker_batch, "real_bs", bs)), dtype=np.int32)
        else:
            selector = np.asarray(selector, dtype=np.int32)
        token_num = self.block_size if token_num is None else int(token_num)
        key = (dp_size, per_dp_bs, tuple(selector.tolist()), token_num)
        cached = self._verify_bucket_templates.get(key)
        if cached is not None:
            return cached

        active_host = np.zeros(bs, dtype=np.bool_)
        active_host[selector] = True
        extend_seq_lens = active_host.astype(np.int32) * token_num
        cu_q_lens = np.zeros((dp_size, per_dp_bs + 1), dtype=np.int32)
        cu_q_lens[:, 1:] = np.cumsum(
            extend_seq_lens.reshape(dp_size, per_dp_bs),
            axis=1,
            dtype=np.int32,
        )
        local_n = active_host.reshape(dp_size, per_dp_bs).sum(axis=1, dtype=np.int32)
        distribution = np.column_stack([np.zeros_like(local_n), local_n, local_n]).reshape(-1)
        data_sharding = NamedSharding(self.mesh, P("data"))
        cached = DFlashVerifyBucketTemplate(
            extend_seq_lens=extend_seq_lens,
            cu_q_lens=jax.device_put(cu_q_lens.reshape(-1), data_sharding),
            active_mask=jax.device_put(active_host, data_sharding),
            distribution=jax.device_put(distribution, data_sharding),
        )
        self._verify_bucket_templates[key] = cached
        return cached

    @staticmethod
    def _active_decode_slot_mask(model_worker_batch, total_bs: int) -> np.ndarray:
        mask = np.zeros(total_bs, dtype=bool)
        real_bs_per_dp = getattr(model_worker_batch, "real_bs_per_dp", None)
        if real_bs_per_dp is None:
            mask[: int(getattr(model_worker_batch, "real_bs", total_bs))] = True
            return mask

        per_dp_bs = int(getattr(model_worker_batch, "per_dp_bs_size", total_bs))
        for dp_rank, real_bs in enumerate(real_bs_per_dp):
            start = dp_rank * per_dp_bs
            end = min(start + int(real_bs), total_bs)
            mask[start:end] = True
        return mask

    def _can_use_fused_spec_prefill(self, model_worker_batch: ModelWorkerBatch) -> bool:
        if (
            self.server_args.disable_overlap_schedule
            or os.getenv("SGL_JAX_DISABLE_FUSED_SPEC_PREFILL") == "1"
        ):
            return False
        sampling_info = model_worker_batch.sampling_info
        penalizer = getattr(sampling_info, "penalizer_orchestrator", None)
        has_penalty = getattr(sampling_info, "linear_penalty", None) is not None or bool(
            getattr(penalizer, "is_required", False)
        )
        return (
            bool(getattr(sampling_info, "is_all_greedy", False))
            and not has_penalty
            and getattr(sampling_info, "vocab_mask", None) is None
            and not getattr(model_worker_batch, "return_logprob", False)
            and not getattr(model_worker_batch, "return_output_logprob_only", False)
        )

    def _prepare_overlap_sampling_info(self, model_worker_batch: ModelWorkerBatch):
        return

    @contextlib.contextmanager
    def _dispatch_context(self, runner):
        try:
            mesh_ctx = jax.sharding.use_mesh(self.mesh)
        except AttributeError:
            try:
                mesh_ctx = jax.set_mesh(self.mesh)
            except AttributeError:
                mesh_ctx = self.mesh
        kv_lock = getattr(runner.token_to_kv_pool, "_donate_lock", None)
        lock_ctx = kv_lock if kv_lock is not None else contextlib.nullcontext()
        with mesh_ctx, lock_ctx:
            yield

    @staticmethod
    def _replace_memory_pools(runner, pool_updates) -> None:
        if runner.tp_size == 1 and isinstance(pool_updates, list):
            target_sharding = runner.token_to_kv_pool.kv_sharding
            pool_updates = [jax.device_put(kv, target_sharding) for kv in pool_updates]
        runner.memory_pools.replace_all(pool_updates)

    @staticmethod
    def _prefill_precompile_variants(manager) -> list[tuple[int, int]]:
        bs = int(manager.max_padded_batch_size)
        return [(bs, int(tokens)) for tokens in manager.token_buckets if int(tokens) >= bs]

    def _precompile_dflash_prefill(self) -> None:
        import time

        from sgl_jax.srt.sampling.sampling_batch_info import SamplingMetadata

        manager = self._target_compilation_manager
        variants = self._prefill_precompile_variants(manager)
        logger.info("[DFLASH] Precompiling prefill variants: %s", variants)
        start = time.perf_counter()

        for bs, num_tokens in variants:
            t0 = time.perf_counter()
            batch = manager._make_dummy_batch(
                bs,
                num_tokens,
                ForwardMode.EXTEND,
                manager.cache_loc_buckets[-1],
                speculative_algorithm=SpeculativeAlgorithm.DFLASH,
                dp_size=manager.dp_size,
                per_dp_bs_size=bs // manager.dp_size,
            )
            batch.capture_hidden_mode = CaptureHiddenMode.FULL
            sampling_metadata = SamplingMetadata.from_model_worker_batch(
                batch,
                0,
                self.mesh,
                self._target_vocab_size,
            )
            batch.forward_batch = ForwardBatch.init_new(
                batch,
                self.target_worker.model_runner,
            )
            logits_output, *_ = self.forward_target_extend(
                batch,
                sampling_metadata,
                skip_sample=True,
            )

            # Consume the real target output so hidden-state sharding matches
            # the serving dependency exactly.
            self._run_jit_draft_extend(
                logits_output.hidden_states,
                batch.positions,
                batch.out_cache_loc,
            )
            logger.info(
                "[DFLASH] Prefill bs=%d tokens=%d compiled in %.1f secs",
                bs,
                num_tokens,
                time.perf_counter() - t0,
            )

        logger.info(
            "[DFLASH] Prefill precompile finished in %.0f secs",
            time.perf_counter() - start,
        )

    def _precompile_dflash_variant(self, bs: int) -> None:
        row_width = max(self.block_size, 16 * self.page_size)
        page_indices = np.zeros(self._page_indices_capacity(bs), dtype=np.int32)
        draft_batch = self._make_verify_dummy_batch(bs, row_width, is_draft=True)
        use_relay_state = not self.server_args.disable_overlap_schedule
        forward_batch = ForwardBatch.init_new(draft_batch, self.draft_model_runner)
        forward_batch.forward_mode = ForwardMode.TARGET_VERIFY
        template = self._get_verify_bucket_template(
            draft_batch,
            bs,
            token_num=self.draft_block_size,
        )
        draft_metadata = self.draft_model_runner.attn_backend.get_eagle_forward_metadata(
            draft_batch,
            page_indices=page_indices,
            page_indices_capacity=self._page_indices_capacity(bs),
            extend_seq_lens=template.extend_seq_lens,
            cu_q_lens=template.cu_q_lens,
            distribution=template.distribution,
        )
        self.draft_model_runner.attn_backend.forward_metadata = draft_metadata
        from jax.sharding import NamedSharding
        from jax.sharding import PartitionSpec as P

        data_sharding = NamedSharding(self.mesh, P("data"))
        feedback_sharding = NamedSharding(self.mesh, P("data", None))
        prefix_lens = np.asarray(draft_batch.seq_lens, dtype=np.int32)
        active_mask = template.active_mask
        if use_relay_state:
            per_dp_bs = bs // int(draft_batch.dp_size)
            future_indices = np.tile(
                np.arange(per_dp_bs, dtype=np.int32),
                int(draft_batch.dp_size),
            )
            relay_future_indices = jax.device_put(future_indices, data_sharding)
            allocated_lens = prefix_lens + 2 * self.block_size
            self.init_spec_relay_buffers()
            self._update_relay(
                jnp.ones((bs,), dtype=jnp.int32),
                jax.device_put(prefix_lens + 1, data_sharding),
                relay_future_indices,
                active_mask,
                dp_size=int(draft_batch.dp_size),
            )
        else:
            relay_future_indices = jax.device_put(np.zeros(bs, dtype=np.int32), data_sharding)
            allocated_lens = prefix_lens + self.block_size
        draft_plan = DraftForwardPlan(
            forward_batch=forward_batch,
            forward_metadata=draft_metadata,
            seq_lens=prefix_lens + 1,
            target_prefix_lens=prefix_lens,
            positions_host=np.asarray(draft_batch.positions, dtype=np.int32),
            page_indices=page_indices,
            allocated_lens=jax.device_put(allocated_lens, data_sharding),
            reservation_base_lens=jax.device_put(prefix_lens, data_sharding),
            relay_future_indices=relay_future_indices,
            relay_valid_mask=active_mask,
            flashback_token_ids=jax.device_put(
                np.zeros((bs, self.block_size - 1), dtype=np.int32),
                feedback_sharding,
            ),
            flashback_target_margins=jax.device_put(
                np.zeros((bs, self.block_size - 1), dtype=np.float32),
                feedback_sharding,
            ),
            flashback_valid_mask=jax.device_put(
                np.zeros((bs, self.block_size - 1), dtype=np.bool_),
                feedback_sharding,
            ),
            rejected_draft_token_ids=jax.device_put(np.zeros((bs,), dtype=np.int32), data_sharding),
            rejection_valid_mask=np.zeros((bs,), dtype=np.bool_),
            previous_accept_lens=np.zeros((bs,), dtype=np.int32),
            target_correction_token_ids=np.zeros((bs,), dtype=np.int32),
            ngram_token_ids=jax.device_put(
                np.zeros((bs, self.block_size - 1), dtype=np.int32),
                feedback_sharding,
            ),
            ngram_bonus=jax.device_put(
                np.zeros((bs, self.block_size - 1), dtype=np.float32),
                feedback_sharding,
            ),
            ngram_valid_mask=jax.device_put(
                np.zeros((bs, self.block_size - 1), dtype=np.bool_),
                feedback_sharding,
            ),
            ngram_match_lens=jax.device_put(np.zeros((bs,), dtype=np.int32), data_sharding),
            use_relay_state=use_relay_state,
            dp_size=int(draft_batch.dp_size),
            bs=bs,
        )
        (
            draft_token,
            base_draft_token,
            _top2_draft_token,
            _top2_margins,
            redenoise_candidate_token,
            redenoise_prefix_lens,
            resolved_prefix_lens,
            resolved_positions,
            resolved_cache_loc,
            ngram_selected_mask,
            candidate_margins,
        ) = self._run_jit_draft_block(draft_plan)

        # Match the serving dependency and sharding exactly: target verify
        # consumes the P("data") proposal produced by jit_draft.
        target_batch = self._make_verify_dummy_batch(bs, row_width)
        verify_input = DFlashVerifyInput(
            draft_token=draft_token,
            draft_token_num=self.block_size,
        )
        target_batch.spec_info_padded = verify_input
        target_forward_batch = copy.copy(forward_batch)
        target_forward_batch.input_ids = draft_token
        target_forward_batch.positions = resolved_positions
        target_forward_batch.seq_lens = resolved_prefix_lens
        target_forward_batch.out_cache_loc = resolved_cache_loc
        target_forward_batch.attn_backend = self.target_worker.model_runner.attn_backend
        target_forward_batch.spec_info = verify_input
        target_forward_batch.capture_hidden_mode = CaptureHiddenMode.FULL
        target_forward_batch.input_embedding = None
        target_template = self._get_verify_bucket_template(
            target_batch,
            bs,
            token_num=self.block_size,
        )
        target_metadata = self.target_worker.model_runner.attn_backend.get_eagle_forward_metadata(
            target_batch,
            page_indices=page_indices,
            page_indices_capacity=self._page_indices_capacity(bs),
            extend_seq_lens=target_template.extend_seq_lens,
            cu_q_lens=target_template.cu_q_lens,
            distribution=target_template.distribution,
        )
        target_plan = TargetVerifyPlan(
            model_worker_batch=target_batch,
            forward_batch=target_forward_batch,
            forward_metadata=target_metadata,
            logits_metadata=LogitsMetadata.from_model_worker_batch(target_batch, self.mesh),
            seq_lens=np.asarray(target_batch.seq_lens, dtype=np.int32) + 1,
            target_prefix_lens=np.asarray(target_batch.seq_lens, dtype=np.int32),
            resolved_target_prefix_lens=resolved_prefix_lens,
            draft_extend_positions=resolved_positions,
            draft_extend_cache_loc=resolved_cache_loc,
            active_mask=target_template.active_mask,
            allocated_lens=draft_plan.allocated_lens,
            relay_future_indices=draft_plan.relay_future_indices,
            relay_valid_mask=draft_plan.relay_valid_mask,
            draft_token=draft_token,
            base_draft_token=base_draft_token,
            redenoise_candidate_token=redenoise_candidate_token,
            redenoise_prefix_lens=redenoise_prefix_lens,
            flashback_token_ids=draft_plan.flashback_token_ids,
            flashback_valid_mask=draft_plan.flashback_valid_mask,
            rejected_draft_token_ids=draft_plan.rejected_draft_token_ids,
            rejection_valid_mask=draft_plan.rejection_valid_mask,
            previous_accept_lens=draft_plan.previous_accept_lens,
            target_correction_token_ids=draft_plan.target_correction_token_ids,
            candidate_margins=candidate_margins,
            ngram_selected_mask=ngram_selected_mask,
            ngram_token_ids=draft_plan.ngram_token_ids,
            ngram_valid_mask=draft_plan.ngram_valid_mask,
            ngram_match_lens=draft_plan.ngram_match_lens,
            update_relay=use_relay_state,
        )
        self.target_worker.model_runner.attn_backend.forward_metadata = target_metadata
        logits_output, _, accept_lens, *_ = self._run_jit_target_verify(target_plan)
        self._run_jit_draft_extend(
            logits_output.hidden_states,
            target_plan.draft_extend_positions,
            target_plan.draft_extend_cache_loc,
            accept_lens=accept_lens,
            active_mask=target_plan.active_mask,
        )

    def _verify_write_cache_loc(
        self,
        batch: ModelWorkerBatch,
        *,
        token_num: int | None = None,
    ) -> np.ndarray:
        dp_size = int(batch.dp_size)
        token_num = self.block_size if token_num is None else int(token_num)
        per_dp_tokens = int(batch.per_dp_bs_size) * token_num
        out_cache_loc = np.asarray(batch.out_cache_loc, dtype=np.int32)
        if out_cache_loc.shape[0] % dp_size != 0:
            raise ValueError(
                "DFLASH verify out_cache_loc is not divisible by dp_size: "
                f"shape={out_cache_loc.shape}, dp_size={dp_size}."
            )
        per_dp_ocl = out_cache_loc.shape[0] // dp_size
        if per_dp_ocl < per_dp_tokens:
            raise ValueError(
                "DFLASH verify out_cache_loc rank section is too short: "
                f"per_dp_ocl={per_dp_ocl}, verify_tokens={per_dp_tokens}."
            )
        return out_cache_loc.reshape(dp_size, per_dp_ocl)[:, :per_dp_tokens].reshape(-1)

    def _make_verify_dummy_batch(
        self, bs: int, row_width: int, is_draft: bool = False
    ) -> ModelWorkerBatch:
        block_size = self.draft_block_size if is_draft else self.block_size
        num_tokens = bs * block_size
        dp_size = int(self._target_compilation_manager.dp_size)
        if bs % dp_size != 0:
            raise ValueError(
                "DFLASH verify dummy batch must be divisible by dp_size: "
                f"bs={bs}, dp_size={dp_size}."
            )
        per_dp_bs = bs // dp_size
        per_dp_reserved_tokens = per_dp_bs * self.block_size
        kv_len = min(row_width, self._target_compilation_manager.max_req_len)
        prefix_len = max(0, kv_len - block_size)
        positions = np.tile(np.arange(prefix_len, prefix_len + block_size, dtype=np.int32), bs)
        batch = self._target_compilation_manager._make_dummy_batch(
            bs,
            num_tokens,
            ForwardMode.TARGET_VERIFY,
            bs * row_width,
            speculative_algorithm=SpeculativeAlgorithm.DFLASH,
            dp_size=dp_size,
            per_dp_bs_size=per_dp_bs,
        )
        capture_hidden = CaptureHiddenMode.NULL if is_draft else CaptureHiddenMode.FULL
        batch.input_ids = np.ones(num_tokens, dtype=np.int32)
        batch.real_input_ids_len = num_tokens
        batch.seq_lens = np.full(bs, prefix_len, dtype=np.int32)
        batch.out_cache_loc = np.concatenate(
            [
                np.concatenate(
                    [
                        np.arange(
                            dp_rank * per_dp_reserved_tokens + 1,
                            (dp_rank + 1) * per_dp_reserved_tokens + 1,
                            dtype=np.int32,
                        ),
                        np.full(per_dp_reserved_tokens, -1, dtype=np.int32),
                    ]
                )
                for dp_rank in range(dp_size)
            ]
        )
        batch.positions = positions
        batch.cache_loc = np.zeros(dp_size, dtype=np.int32)
        batch.capture_hidden_mode = capture_hidden
        batch.spec_info_padded = DFlashVerifyInput(
            draft_token=jnp.ones(num_tokens, dtype=jnp.int32),
            draft_token_num=block_size,
        )
        return batch
