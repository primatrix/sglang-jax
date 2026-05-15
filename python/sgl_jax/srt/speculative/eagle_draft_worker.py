import functools
import logging
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from tqdm import tqdm

from sgl_jax.srt.layers.logits_processor import LogitsMetadata, LogitsProcessorOutput
from sgl_jax.srt.managers.schedule_batch import ModelWorkerBatch
from sgl_jax.srt.managers.scheduler import GenerationBatchResult
from sgl_jax.srt.managers.tp_worker import ModelWorker
from sgl_jax.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sgl_jax.srt.speculative.base_worker import BaseDraftWorker, replicate_to_mesh
from sgl_jax.srt.speculative.eagle_util import (
    EagleDraftInput,
    EagleVerifyInput,
    build_tree_kernel_efficient,
    build_tree_mask_for_draft_decode,
)
from sgl_jax.srt.speculative.spec_info import SpeculativeAlgorithm
from sgl_jax.srt.utils.common_utils import get_bool_env_var
from sgl_jax.srt.utils.jax_utils import device_array

logger = logging.getLogger(__name__)
RETURN_ORIGINAL_LOGPROB = get_bool_env_var("RETURN_ORIGINAL_LOGPROB")


def _take_with_optional_out_sharding(array: jax.Array, index: jax.Array, trailing_slice=False):
    """Gather rows from ``array`` while preserving the source sharding when possible.

    Plain ``array[index]`` without ``out_sharding`` produces a fresh, often
    differently-sharded result for every distinct ``index.shape``. The
    persistent JIT cache keys gathers by their result sharding, so the
    EAGLE post-processing path otherwise compiles a new gather kernel for
    each real batch size at runtime. Reusing the source sharding makes
    those gathers stable across runtime batch sizes and lets the cache hit.
    """
    out_sharding = getattr(array, "sharding", None)
    # numpy arrays / non-jax inputs: plain indexing is fine
    if out_sharding is None:
        return array[index, :] if trailing_slice else array[index]
    # JAX 0.8.1+ rejects naked sharded gathers regardless of sharding class
    # (NamedSharding, GSPMDSharding, SingleDeviceSharding, ...). Always go
    # through .at[].get with an explicit out_sharding.
    if trailing_slice:
        return array.at[index, :].get(out_sharding=out_sharding)
    return array.at[index].get(out_sharding=out_sharding)


def _pad_1d_array(value, target_size: int, pad_value: int = -1) -> np.ndarray:
    value = np.asarray(value)
    pad_size = target_size - value.shape[0]
    if pad_size <= 0:
        return value
    return np.pad(value, (0, pad_size), constant_values=pad_value)


class EagleDraftWorker(BaseDraftWorker):
    """EAGLE draft model worker.

    Holds a ``ModelWorker`` (the draft model runner) via composition and
    implements draft-specific logic: multi-step decode, tree building,
    prefill extend, and decode extend.
    """

    def __init__(self, server_args, target_worker: ModelWorker):
        self.server_args = server_args
        self.target_worker_ref = target_worker
        self.topk = server_args.speculative_eagle_topk
        self.speculative_num_steps = server_args.speculative_num_steps
        self.speculative_num_draft_tokens = server_args.speculative_num_draft_tokens
        self.page_size = server_args.page_size
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )
        self.hot_token_ids = None

        req_to_token_pool, _ = target_worker.get_memory_pool()

        # Compose a ModelWorker for the draft model (instead of inheriting)
        # Must be created last to ensure model state is correct.
        self._worker = ModelWorker(
            server_args,
            target_worker.mesh,
            req_to_token_pool=req_to_token_pool,
            is_draft_worker=True,
        )

        EagleDraftInput.ALLOC_LEN_PER_DECODE = max(
            self.speculative_num_steps * self.topk, self.speculative_num_draft_tokens
        )

        self._share_embed_head(target_worker)

        target_slot_range = target_worker.model_runner.max_total_num_tokens
        draft_pool_size = self.draft_model_runner.max_total_num_tokens
        assert draft_pool_size >= target_slot_range, (
            f"draft KV pool ({draft_pool_size}) < target allocator slot range "
            f"({target_slot_range}); high-slot draft KV reads/writes will be "
            f"garbage. Hybrid target without the post-set_num_token_hybrid "
            f"draft_runner_cache_size overwrite hits this."
        )

        self._worker.model_runner.initialize_jit()

        (
            precompile_token_paddings,
            precompile_bs_paddings,
            precompile_cache_loc_paddings,
        ) = target_worker.get_precompile_paddings()
        self.precompile_bs_paddings = precompile_bs_paddings
        self.precompile_cache_loc_paddings = precompile_cache_loc_paddings
        self.precompile_token_paddings = precompile_token_paddings

    def _share_embed_head(self, target_worker: ModelWorker):
        embed, head = target_worker.model_runner.model.get_embed_and_head()

        if self.speculative_algorithm.is_eagle3():
            if (
                hasattr(self.draft_model_runner.model, "load_lm_head_from_target")
                and self.draft_model_runner.model.load_lm_head_from_target
            ):
                self.draft_model_runner.model.set_embed_and_head(embed, head)
            else:
                self.draft_model_runner.model.set_embed(embed)

            if self.draft_model_runner.model.hot_token_ids is not None:
                self.hot_token_ids = device_array(
                    self.draft_model_runner.model.hot_token_ids,
                    sharding=(NamedSharding(self._worker.mesh, P())),
                )
        else:
            if self.hot_token_ids is not None:
                head = head.clone()
                self.hot_token_ids = device_array(
                    self.draft_model_runner.model.hot_token_ids,
                    sharding=(NamedSharding(self._worker.mesh, P())),
                )
                head.data = head.data[self.hot_token_ids]

            self.draft_model_runner.model.set_embed_and_head(embed, head)

    @property
    def draft_model_runner(self):
        return self._worker.get_model_runner()

    @property
    def mesh(self):
        return self._worker.mesh

    @property
    def model_config(self):
        return self._worker.model_config

    @property
    def compilation_manager(self):
        return self._worker.compilation_manager

    @property
    def max_req_len(self):
        return self._worker.max_req_len

    def get_max_padded_size(self):
        return self._worker.get_max_padded_size()

    def _remap_hot_token_ids(self, token_ids):
        """Map draft-token ids back to full vocab ids while keeping sharding stable.

        Uses ``_take_with_optional_out_sharding`` so the gather kernel cache key
        does not depend on the runtime ``token_ids`` shape/sharding.
        """
        if not isinstance(token_ids, jax.Array):
            token_ids = jnp.asarray(token_ids)
        out_sharding = NamedSharding(
            self.mesh,
            P("data", None) if token_ids.ndim == 2 else P("data"),
        )
        return self.hot_token_ids.at[token_ids].get(out_sharding=out_sharding)

    # -- Phase-1 runtime padding helpers (cache-miss fixes) --

    def _get_phase1_runtime_bs_candidates(self) -> list[int]:
        max_bs = max(self.precompile_bs_paddings) if self.precompile_bs_paddings else 0
        if max_bs <= 0:
            return []
        return [bs for bs in (1, 2, 4, 8, 16) if bs <= max_bs]

    def _get_phase1_runtime_bs_padding(self, real_bs: int) -> int:
        for bs in self._get_phase1_runtime_bs_candidates():
            if bs >= real_bs:
                return bs
        return real_bs

    def _get_phase1_runtime_indices(self, real_bs: int) -> np.ndarray:
        """Indices of length ``padded_bs`` selecting the first ``real_bs`` rows.

        Padding entries point back at the last real row so downstream gathers
        produce well-defined values for unused slots while keeping the gather
        shape stable across requests with the same padded bucket.
        """
        padded_bs = self._get_phase1_runtime_bs_padding(real_bs)
        indices = np.arange(padded_bs, dtype=np.int32)
        if padded_bs > real_bs:
            indices[real_bs:] = max(real_bs - 1, 0)
        return indices

    def _get_padding_bs_for_real_bs(self, real_bs: int) -> int:
        for bs in sorted(self.precompile_bs_paddings):
            if bs >= real_bs:
                return bs
        raise RuntimeError("did not get comperate padding bs, it should not happened")

    def pad_out_cache_loc_for_verify(self, model_worker_batch: ModelWorkerBatch) -> None:
        """Pad ``out_cache_loc`` to ``bs * num_draft_tokens`` for the verify path.

        Real batches at runtime ship a tightly-packed ``out_cache_loc`` whose
        size depends on the live request count. The verify forward expects a
        bucketed shape; without this pad each fresh batch retraces.
        """
        target_size = model_worker_batch.seq_lens.shape[0] * self.speculative_num_draft_tokens
        model_worker_batch.out_cache_loc = _pad_1d_array(
            model_worker_batch.out_cache_loc,
            target_size,
            -1,
        )

    def _trim_prefill_spec_info_to_real_bs(
        self, draft_input: EagleDraftInput, real_bs: int
    ) -> None:
        """Trim padded prefill spec_info rows to the real batch via stable gather.

        The padded prefill output carries trailing rows from padding entries.
        We strip them with ``_take_with_optional_out_sharding`` so the trim
        gather kernel is itself shape-stable across runtime real_bs values.
        """
        keep_indices_host = np.arange(real_bs, dtype=np.int32)
        keep_indices = device_array(
            keep_indices_host,
            sharding=NamedSharding(self.mesh, P("data")),
        )

        def take_rows(value):
            if value is None:
                return None
            if isinstance(value, jax.Array):
                return _take_with_optional_out_sharding(
                    value, keep_indices, trailing_slice=value.ndim > 1
                )
            return value[:real_bs]

        draft_input.hidden_states = take_rows(draft_input.hidden_states)
        draft_input.verified_id = take_rows(draft_input.verified_id)
        draft_input.topk_p = take_rows(draft_input.topk_p)
        draft_input.topk_index = take_rows(draft_input.topk_index)
        draft_input.allocate_lens = take_rows(draft_input.allocate_lens)

    # -- BaseDraftWorker interface --

    def draft(self, model_worker_batch: ModelWorkerBatch) -> None:
        self.padding_for_decode(model_worker_batch)
        score_list, token_list, parents_list = self.draft_forward(model_worker_batch)
        verified_seq_lens = model_worker_batch.seq_lens - 1
        max_seq_len = int(np.max(verified_seq_lens)) if verified_seq_lens.size > 0 else 1
        max_context_len = self._pick_context_len(max_seq_len)
        (
            tree_mask,
            position,
            retrive_index,
            retrive_next_token,
            retrive_next_sibling,
            draft_tokens,
        ) = build_tree_kernel_efficient(
            model_worker_batch.spec_info.verified_id,
            score_list,
            token_list,
            parents_list,
            verified_seq_lens,
            np.sum(verified_seq_lens),
            self.topk,
            self.speculative_num_draft_tokens,
            max_context_len,
            model_worker_batch.seq_lens.shape[0],
            model_worker_batch.speculative_num_steps,
            self.mesh,
        )
        model_worker_batch.spec_info = EagleVerifyInput(
            draft_token=draft_tokens,
            custom_mask=tree_mask,
            positions=position,
            retrive_index=retrive_index,
            retrive_next_token=retrive_next_token,
            retrive_next_sibling=retrive_next_sibling,
            retrive_cum_len=None,
            spec_steps=self.speculative_num_steps,
            topk=self.topk,
            draft_token_num=self.speculative_num_draft_tokens,
            capture_hidden_mode=CaptureHiddenMode.LAST,
            seq_lens_sum=model_worker_batch.seq_lens_sum,
            seq_lens_cpu=model_worker_batch.seq_lens,
        )

    def draft_extend_for_prefill(
        self,
        model_worker_batch: ModelWorkerBatch,
        hidden_states: jax.Array,
        next_token_ids: jax.Array,
    ) -> None:
        # FIXME(pc) move this all prepare to prepare_for_extend_after_target_prefill
        index_sharding = NamedSharding(self.mesh, P("data"))
        padded_indices = device_array(
            np.arange(model_worker_batch.seq_lens.shape[0], dtype=np.int32),
            sharding=index_sharding,
        )
        model_worker_batch.spec_info = EagleDraftInput(
            hidden_states=hidden_states,
            verified_id=_take_with_optional_out_sharding(next_token_ids, padded_indices),
            num_tokens_per_batch=np.asarray(1, dtype=jnp.int32),
            num_tokens_for_logprob_per_batch=np.asarray(1, dtype=jnp.int32),
            allocate_lens=model_worker_batch.seq_lens,
        )
        model_worker_batch.return_hidden_states = False
        model_worker_batch.spec_info.prepare_for_extend_after_target_prefill(
            model_worker_batch=model_worker_batch
        )
        model_worker_batch.spec_info.capture_hidden_mode = CaptureHiddenMode.LAST
        model_worker_batch.capture_hidden_mode = CaptureHiddenMode.LAST
        forward_batch = ForwardBatch.init_new(model_worker_batch, self.draft_model_runner)
        forward_batch.return_logprob = False

        forward_metadata = self.draft_model_runner.attn_backend.get_eagle_forward_metadata(
            model_worker_batch
        )

        self.draft_model_runner.attn_backend.forward_metadata = forward_metadata
        forward_batch.forward_mode = ForwardMode.EXTEND

        logits_output, _, _ = self.draft_model_runner.forward(
            forward_batch,
            logits_metadata=LogitsMetadata.from_model_worker_batch(model_worker_batch, self.mesh),
        )
        real_indices = device_array(
            self._get_phase1_runtime_indices(model_worker_batch.real_bs),
            sharding=index_sharding,
        )
        logits_output.next_token_logits = _take_with_optional_out_sharding(
            logits_output.next_token_logits, real_indices, trailing_slice=True
        )
        if len(logits_output.hidden_states.shape) == 1:
            logits_output.hidden_states = jnp.expand_dims(logits_output.hidden_states, axis=0)
        logits_output.hidden_states = _take_with_optional_out_sharding(
            logits_output.hidden_states, real_indices, trailing_slice=True
        )
        assert isinstance(forward_batch.spec_info, EagleDraftInput)
        forward_batch.spec_info.verified_id = _take_with_optional_out_sharding(
            forward_batch.spec_info.verified_id, real_indices
        )
        runtime_bs = real_indices.shape[0]
        forward_batch.spec_info.allocate_lens = model_worker_batch.seq_lens[:runtime_bs]

        self.capture_for_decode(logits_output, forward_batch.spec_info)
        self._trim_prefill_spec_info_to_real_bs(forward_batch.spec_info, model_worker_batch.real_bs)

    def draft_extend_for_decode(
        self, model_worker_batch: ModelWorkerBatch, batch_output: GenerationBatchResult
    ) -> None:
        if batch_output.next_draft_input.verified_id.shape[0] <= 0:
            return
        draft_input = EagleDraftInput(
            hidden_states=batch_output.logits_output.hidden_states,
            allocate_lens=batch_output.allocate_lens,
        )
        model_worker_batch, logits_metadata = draft_input.prepare_for_extend_after_verify(
            model_worker_batch,
            self.draft_model_runner,
            batch_output,
            self.speculative_num_draft_tokens,
        )

        forward_batch = ForwardBatch.init_new(model_worker_batch, self.draft_model_runner)
        if forward_batch.input_ids.shape[0] <= 0:
            return
        draft_logits_output, _, _ = self.draft_model_runner.forward(
            forward_batch,
            logits_metadata=logits_metadata,
        )
        select_index = (
            np.arange(len(model_worker_batch.seq_lens[: model_worker_batch.real_bs]))
            * (self.speculative_num_steps + 1)
            + batch_output.accept_lens[: model_worker_batch.real_bs]
            - 1
        )
        rep_logits, rep_hidden = replicate_to_mesh(
            self.mesh, draft_logits_output.next_token_logits, draft_logits_output.hidden_states
        )
        draft_logits_output.next_token_logits = _take_with_optional_out_sharding(
            rep_logits, select_index, trailing_slice=True
        )
        draft_logits_output.hidden_states = _take_with_optional_out_sharding(
            rep_hidden, select_index, trailing_slice=True
        )
        topk_p, topk_index = topk_probs_from_logits(
            draft_logits_output.next_token_logits, self.topk
        )
        if self.hot_token_ids is not None:
            topk_index = self._remap_hot_token_ids(topk_index)
        topk_index = np.asarray(jax.device_get(topk_index))

        batch_output.next_draft_input.hidden_states = draft_logits_output.hidden_states
        batch_output.next_draft_input.topk_p = topk_p
        batch_output.next_draft_input.topk_index = topk_index
        batch_output.next_draft_input.verified_id = _take_with_optional_out_sharding(
            batch_output.next_draft_input.verified_id, select_index
        )
        batch_output.allocate_lens = batch_output.allocate_lens[: model_worker_batch.real_bs]
        batch_output.accept_lens = batch_output.accept_lens[: model_worker_batch.real_bs]

    # -- Internal draft helpers --

    def capture_for_decode(
        self, logits_output: LogitsProcessorOutput, draft_input: EagleDraftInput
    ):
        topk_p, topk_index = topk_probs_from_logits(logits_output.next_token_logits, self.topk)
        if self.hot_token_ids is not None:
            topk_index = self._remap_hot_token_ids(topk_index)
        topk_index = np.asarray(jax.device_get(topk_index))
        draft_input.topk_p = topk_p
        draft_input.topk_index = topk_index
        draft_input.hidden_states = replicate_to_mesh(self.mesh, logits_output.hidden_states)

    def padding_for_decode(self, model_worker_batch: ModelWorkerBatch):
        _, padding_bs_index = self.get_padding_bs(model_worker_batch.real_bs)
        self.copy_model_worker_batch_to_cpu(model_worker_batch)
        model_worker_batch.spec_info.prepare_for_draft_decode(
            model_worker_batch, self.topk, self.speculative_num_steps
        )
        model_worker_batch.seq_lens = model_worker_batch.seq_lens
        seq_lens_cpu = model_worker_batch.seq_lens
        page_size = self.page_size
        req_to_token_pool, _ = self.target_worker_ref.get_memory_pool()
        token_indices_with_all_reqs = req_to_token_pool.req_to_token[
            model_worker_batch.req_pool_indices
        ]
        spec_info = model_worker_batch.spec_info
        assert isinstance(spec_info, EagleDraftInput)
        cache_loc_flat = np.array([], dtype=np.int32)
        if len(seq_lens_cpu) > 0:
            valid_mask = seq_lens_cpu > 0
            if np.any(valid_mask):
                valid_indices = np.where(valid_mask)[0]
                valid_allocate_lens = spec_info.allocate_lens[valid_mask]
                aligned_lengths = ((valid_allocate_lens + page_size - 1) // page_size) * page_size
                total_aligned_length = np.sum(aligned_lengths)
                cache_loc_flat = np.zeros(total_aligned_length, dtype=np.int32)
                offset = 0
                for i, (seq_idx, allocate_len, aligned_len) in enumerate(
                    zip(valid_indices, valid_allocate_lens, aligned_lengths)
                ):
                    cache_loc_flat[offset : offset + allocate_len] = token_indices_with_all_reqs[
                        seq_idx, :allocate_len
                    ]
                    offset += aligned_len
        total_cache_loc_size = self.precompile_cache_loc_paddings[padding_bs_index]
        assert total_cache_loc_size >= len(cache_loc_flat)
        cache_loc_cpu = np.empty(total_cache_loc_size, dtype=np.int32)
        if len(cache_loc_flat) > 0:
            cache_loc_cpu[: len(cache_loc_flat)] = cache_loc_flat
        if len(cache_loc_flat) < total_cache_loc_size:
            cache_loc_cpu[len(cache_loc_flat) :] = 0

        model_worker_batch.cache_loc = cache_loc_cpu
        model_worker_batch.capture_hidden_mode = CaptureHiddenMode.LAST

        topk_index = model_worker_batch.spec_info.topk_index
        if self.topk > 1:
            self.draft_model_runner.attn_backend.forward_metadata.custom_mask = (
                build_tree_mask_for_draft_decode(
                    model_worker_batch.seq_lens,
                    topk=topk_index.shape[1],
                    speculative_step_id=0,
                    parents_list=None,
                )
            )
        bs = self.precompile_bs_paddings[padding_bs_index]
        if bs - model_worker_batch.spec_info.verified_id.shape[0] > 0:
            model_worker_batch.spec_info.verified_id = np.pad(
                model_worker_batch.spec_info.verified_id,
                ((0, bs - model_worker_batch.spec_info.verified_id.shape[0]),),
            )
        if bs - model_worker_batch.spec_info.topk_p.shape[0] > 0:
            model_worker_batch.spec_info.topk_p = np.pad(
                model_worker_batch.spec_info.topk_p,
                (
                    (0, bs - model_worker_batch.spec_info.topk_p.shape[0]),
                    (0, 0),
                ),
            )
        if bs - model_worker_batch.seq_lens.shape[0] > 0:
            model_worker_batch.seq_lens = np.pad(
                model_worker_batch.seq_lens, ((0, bs - model_worker_batch.seq_lens.shape[0]),)
            )
            if model_worker_batch.spec_info.allocate_lens is not None:
                model_worker_batch.spec_info.allocate_lens = np.pad(
                    model_worker_batch.spec_info.allocate_lens,
                    ((0, bs - model_worker_batch.spec_info.allocate_lens.shape[0]),),
                )
        if bs - model_worker_batch.spec_info.topk_index.shape[0] > 0:
            model_worker_batch.spec_info.topk_index = np.pad(
                model_worker_batch.spec_info.topk_index,
                (
                    (0, bs - model_worker_batch.spec_info.topk_index.shape[0]),
                    (0, 0),
                ),
            )
        if bs - model_worker_batch.spec_info.hidden_states.shape[0] > 0:
            model_worker_batch.spec_info.hidden_states = np.pad(
                model_worker_batch.spec_info.hidden_states,
                (
                    (0, bs - model_worker_batch.spec_info.hidden_states.shape[0]),
                    (0, 0),
                ),
            )
        model_worker_batch.speculative_eagle_topk = self.topk
        model_worker_batch.speculative_num_steps = self.speculative_num_steps
        model_worker_batch.speculative_num_draft_tokens = self.speculative_num_draft_tokens
        model_worker_batch.input_ids = np.empty(bs * self.topk, np.int32)
        model_worker_batch.positions = np.empty(bs * self.topk, np.int32)

    def draft_forward(self, model_worker_batch: ModelWorkerBatch):
        topk_p, topk_index, hidden_states = (
            model_worker_batch.spec_info.topk_p,
            model_worker_batch.spec_info.topk_index,
            model_worker_batch.spec_info.hidden_states,
        )
        bs = model_worker_batch.seq_lens.shape[0]
        step_min_1 = self.speculative_num_steps - 1
        score_list: jax.Array = jnp.empty((bs, 1 + step_min_1 * self.topk, self.topk))
        token_list: jax.Array = jnp.empty(
            (bs, self.topk + step_min_1 * self.topk * self.topk), dtype=jnp.int32
        )
        parents_list: jax.Array = jnp.empty((bs, self.topk + 1 + step_min_1 * self.topk))
        scores = None
        positions_base = device_array(
            np.repeat(model_worker_batch.seq_lens, self.topk),
            sharding=(NamedSharding(self.mesh, P())),
        )
        logits_metadata = None
        metadata_per_step = self.draft_model_runner.attn_backend.get_eagle_multi_step_metadata(
            model_worker_batch,
        )
        assert isinstance(metadata_per_step, list)
        logits_metadata = LogitsMetadata.from_model_worker_batch(
            model_worker_batch, self.draft_model_runner.mesh
        )
        forward_batch = ForwardBatch.init_new(model_worker_batch, self.draft_model_runner)
        forward_batch.out_cache_loc = np.empty((1,))
        forward_batch.cache_loc = np.empty((1,))
        forward_batch.spec_info = EagleDraftInput()
        forward_batch.spec_info.hidden_states = jnp.empty((bs * self.topk, hidden_states.shape[1]))
        for i in range(self.speculative_num_steps):

            input_ids, hidden_states, scores, tree_info = select_top_k_tokens(
                i, topk_p, topk_index, hidden_states, scores, self.topk
            )
            score_list, token_list, parents_list = update_eagle_lists(
                i, score_list, token_list, parents_list, tree_info, self.topk
            )
            if i == self.speculative_num_steps - 1:
                break

            forward_batch = update_forward_batch_info(
                forward_batch, i, input_ids, hidden_states, positions_base
            )
            self.draft_model_runner.attn_backend.forward_metadata = metadata_per_step[i]

            forward_batch.bid = model_worker_batch.bid
            logits_output, _, _ = self.draft_model_runner.forward(
                forward_batch,
                logits_metadata=logits_metadata,
            )

            topk_p, topk_index = topk_probs_from_logits(logits_output.next_token_logits, self.topk)

            if self.hot_token_ids is not None:
                topk_index = self.hot_token_ids[topk_index]
            hidden_states = replicate_to_mesh(self.mesh, logits_output.hidden_states)

        return score_list, token_list, parents_list

    def _pick_context_len(self, max_seq_len: int) -> int:
        max_seq_len = max(int(max_seq_len), 1)
        if self.precompile_token_paddings:
            for padding in self.precompile_token_paddings:
                if padding >= max_seq_len:
                    return padding
        return 1 << (max_seq_len - 1).bit_length()

    def copy_model_worker_batch_to_cpu(self, model_worker_batch: ModelWorkerBatch):
        model_worker_batch.input_ids = np.array(
            jax.device_get(model_worker_batch.input_ids), dtype=model_worker_batch.input_ids.dtype
        )
        model_worker_batch.seq_lens = np.array(
            jax.device_get(model_worker_batch.seq_lens), dtype=model_worker_batch.seq_lens.dtype
        )
        model_worker_batch.out_cache_loc = np.array(
            jax.device_get(model_worker_batch.out_cache_loc),
            dtype=model_worker_batch.out_cache_loc.dtype,
        )
        model_worker_batch.positions = np.array(
            jax.device_get(model_worker_batch.positions), dtype=model_worker_batch.positions.dtype
        )
        model_worker_batch.req_pool_indices = np.array(
            jax.device_get(model_worker_batch.req_pool_indices),
            dtype=model_worker_batch.req_pool_indices.dtype,
        )
        model_worker_batch.cache_loc = np.array(
            jax.device_get(model_worker_batch.cache_loc), dtype=model_worker_batch.cache_loc.dtype
        )
        model_worker_batch.extend_prefix_lens = (
            np.array(
                jax.device_get(model_worker_batch.extend_prefix_lens),
                dtype=model_worker_batch.extend_prefix_lens.dtype,
            )
            if model_worker_batch.extend_prefix_lens is not None
            else None
        )
        model_worker_batch.extend_seq_lens = (
            np.array(
                jax.device_get(model_worker_batch.extend_seq_lens),
                dtype=model_worker_batch.extend_seq_lens.dtype,
            )
            if model_worker_batch.extend_seq_lens is not None
            else None
        )

    def get_padding_bs(self, real_bs: int) -> int:
        self.precompile_bs_paddings.sort()
        select_bs_index = -1
        bs_padding_size = 0
        for i, size in enumerate(self.precompile_bs_paddings):
            if size >= real_bs:
                bs_padding_size = size - real_bs
                select_bs_index = i
                break
        if select_bs_index < 0:
            raise RuntimeError("did not get comperate padding bs, it should not happened")
        return bs_padding_size, select_bs_index

    def precompile_runtime_jax_helpers(self):
        """Warm EAGLE runtime helper ops whose shapes follow real batch size.

        The model forward itself is padded to the configured precompile buckets, but
        several EAGLE post-processing helpers intentionally operate on the real
        number of active requests. Without warming these shapes, the first request
        drain through batch sizes such as 15, 13, or 6 can trigger persistent-cache
        misses for small JAX gather/top-k/reshard kernels.
        """
        max_bs = max(self.precompile_bs_paddings) if self.precompile_bs_paddings else 0
        if max_bs <= 0:
            return

        start_time = time.perf_counter()
        # The 4K/1K Phase-1 cache gate exercises bsz up to 16.  Larger buckets
        # such as max-running-requests=256 are still covered by the normal
        # padded model-forward precompile, but warming every real drain size up
        # to 256 would add excessive startup work for tiny helper kernels.
        max_runtime_bs = min(max_bs, 16)
        bs_candidates = list(range(1, max_runtime_bs + 1))
        logger.info("[SPEC_RUNTIME] Begin to precompile real_bs=%s", bs_candidates)

        data_sharding = NamedSharding(self.mesh, P("data"))
        data_2d_sharding = NamedSharding(self.mesh, P("data", None))
        replicated_sharding = NamedSharding(self.mesh, P(None))
        replicated_2d_sharding = NamedSharding(self.mesh, P(None, None))
        logits_sharding = NamedSharding(self.mesh, P("data", "tensor"))

        dtype = jnp.bfloat16 if self.server_args.dtype == "bfloat16" else jnp.float32
        hidden_size = self.model_config.hidden_size
        vocab_size = self.model_config.vocab_size

        with tqdm(bs_candidates, desc="[SPEC_RUNTIME] PRECOMPILE", leave=False) as pbar:
            for bs in pbar:
                pbar.set_postfix(real_bs=bs)
                indices = device_array(
                    np.arange(bs, dtype=np.int32),
                    sharding=data_sharding,
                )

                logits = device_array(
                    np.zeros((bs, vocab_size), dtype=np.float32),
                    sharding=logits_sharding,
                ).astype(dtype)
                topk_p, topk_index = topk_probs_from_logits(logits, self.topk)
                topk_p.block_until_ready()
                topk_index.block_until_ready()
                if self.hot_token_ids is not None:
                    token_ids_2d_host = np.zeros((bs, self.topk), dtype=np.int32)
                    token_ids_1d_host = np.zeros((bs * self.topk,), dtype=np.int32)
                    token_ids_2d_data = device_array(token_ids_2d_host, sharding=data_2d_sharding)
                    token_ids_1d_data = device_array(token_ids_1d_host, sharding=data_sharding)
                    token_ids_2d_replicated = device_array(
                        token_ids_2d_host, sharding=replicated_2d_sharding
                    )
                    token_ids_1d_replicated = device_array(
                        token_ids_1d_host, sharding=replicated_sharding
                    )
                    for token_ids in (
                        topk_index,
                        topk_index.flatten(),
                        token_ids_2d_data,
                        token_ids_1d_data,
                        token_ids_2d_replicated,
                        token_ids_1d_replicated,
                        token_ids_2d_host,
                        token_ids_1d_host,
                    ):
                        remapped_token_ids = self._remap_hot_token_ids(token_ids)
                        remapped_token_ids.block_until_ready()
                        np.asarray(jax.device_get(remapped_token_ids))

                hidden = device_array(
                    np.zeros((bs, hidden_size), dtype=np.float32),
                    sharding=data_2d_sharding,
                ).astype(dtype)
                verified_id = device_array(
                    np.zeros((bs,), dtype=np.int32),
                    sharding=data_sharding,
                )
                replicated_hidden = device_array(
                    np.zeros((bs, hidden_size), dtype=np.float32),
                    sharding=replicated_2d_sharding,
                ).astype(dtype)
                replicated_verified_id = device_array(
                    np.zeros((bs,), dtype=np.int32),
                    sharding=replicated_sharding,
                )
                _take_with_optional_out_sharding(
                    logits, indices, trailing_slice=True
                ).block_until_ready()
                _take_with_optional_out_sharding(
                    hidden, indices, trailing_slice=True
                ).block_until_ready()
                _take_with_optional_out_sharding(verified_id, indices).block_until_ready()
                _take_with_optional_out_sharding(
                    replicated_hidden, indices, trailing_slice=True
                ).block_until_ready()
                _take_with_optional_out_sharding(
                    replicated_verified_id, indices
                ).block_until_ready()

                for keep_bs in range(1, bs + 1):
                    keep_indices = device_array(
                        np.arange(keep_bs, dtype=np.int32),
                        sharding=data_sharding,
                    )
                    keep_indices_replicated = device_array(
                        np.arange(keep_bs, dtype=np.int32),
                        sharding=replicated_sharding,
                    )
                    keep_indices_host = np.arange(keep_bs, dtype=np.int32)
                    for keep_index in (
                        keep_indices,
                        keep_indices_replicated,
                        keep_indices_host,
                    ):
                        _take_with_optional_out_sharding(
                            logits, keep_index, trailing_slice=True
                        ).block_until_ready()
                        _take_with_optional_out_sharding(
                            topk_p, keep_index, trailing_slice=True
                        ).block_until_ready()
                        _take_with_optional_out_sharding(
                            topk_index, keep_index, trailing_slice=True
                        ).block_until_ready()
                        _take_with_optional_out_sharding(
                            hidden, keep_index, trailing_slice=True
                        ).block_until_ready()
                        _take_with_optional_out_sharding(
                            verified_id, keep_index
                        ).block_until_ready()
                        _take_with_optional_out_sharding(
                            replicated_hidden, keep_index, trailing_slice=True
                        ).block_until_ready()
                        _take_with_optional_out_sharding(
                            replicated_verified_id, keep_index
                        ).block_until_ready()

                if bs == max_runtime_bs and max_bs > max_runtime_bs:
                    padded_bs = max_bs
                    padded_logits = device_array(
                        np.zeros((padded_bs, vocab_size), dtype=np.float32),
                        sharding=logits_sharding,
                    ).astype(dtype)
                    padded_hidden = device_array(
                        np.zeros((padded_bs, hidden_size), dtype=np.float32),
                        sharding=data_2d_sharding,
                    ).astype(dtype)
                    padded_ids = device_array(
                        np.zeros((padded_bs,), dtype=np.int32),
                        sharding=data_sharding,
                    )
                    padded_indices = device_array(
                        np.arange(padded_bs, dtype=np.int32),
                        sharding=data_sharding,
                    )
                    _take_with_optional_out_sharding(padded_ids, padded_indices).block_until_ready()
                    for keep_bs in bs_candidates:
                        keep_indices = device_array(
                            np.arange(keep_bs, dtype=np.int32),
                            sharding=data_sharding,
                        )
                        _take_with_optional_out_sharding(
                            padded_logits, keep_indices, trailing_slice=True
                        ).block_until_ready()
                        _take_with_optional_out_sharding(
                            padded_hidden, keep_indices, trailing_slice=True
                        ).block_until_ready()
                        _take_with_optional_out_sharding(
                            padded_ids, keep_indices
                        ).block_until_ready()

                draft_extend_slots = bs * (self.speculative_num_steps + 1)
                draft_extend_logits = device_array(
                    np.zeros((draft_extend_slots, vocab_size), dtype=np.float32),
                    sharding=logits_sharding,
                ).astype(dtype)
                draft_extend_hidden = device_array(
                    np.zeros((draft_extend_slots, hidden_size), dtype=np.float32),
                    sharding=data_2d_sharding,
                ).astype(dtype)
                draft_extend_ids = device_array(
                    np.zeros((draft_extend_slots,), dtype=np.int32),
                    sharding=data_sharding,
                )
                for keep_bs in range(1, bs + 1):
                    select_index_host = np.arange(keep_bs, dtype=np.int32) * (
                        self.speculative_num_steps + 1
                    )
                    select_index_data = device_array(
                        select_index_host,
                        sharding=data_sharding,
                    )
                    select_index_replicated = device_array(
                        select_index_host,
                        sharding=replicated_sharding,
                    )
                    for select_index in (
                        select_index_host,
                        select_index_data,
                        select_index_replicated,
                    ):
                        selected_logits = _take_with_optional_out_sharding(
                            draft_extend_logits, select_index, trailing_slice=True
                        )
                        selected_logits.block_until_ready()
                        _take_with_optional_out_sharding(
                            draft_extend_hidden, select_index, trailing_slice=True
                        ).block_until_ready()
                        _take_with_optional_out_sharding(
                            draft_extend_ids, select_index
                        ).block_until_ready()
                        selected_topk_p, selected_topk_index = topk_probs_from_logits(
                            selected_logits, self.topk
                        )
                        selected_topk_p.block_until_ready()
                        selected_topk_index.block_until_ready()
                        if self.hot_token_ids is not None:
                            selected_topk_index = self._remap_hot_token_ids(selected_topk_index)
                            selected_topk_index.block_until_ready()

                if self.topk == 1:
                    score_list = device_array(
                        np.zeros(
                            (bs, 1 + (self.speculative_num_steps - 1) * self.topk, self.topk),
                            dtype=np.float32,
                        ),
                        sharding=NamedSharding(self.mesh, P("data", None, None)),
                    )
                    token_list = device_array(
                        np.zeros(
                            (
                                bs,
                                self.topk
                                + (self.speculative_num_steps - 1) * self.topk * self.topk,
                            ),
                            dtype=np.int32,
                        ),
                        sharding=data_2d_sharding,
                    )
                    token_list_host = np.zeros(
                        (
                            bs,
                            self.topk + (self.speculative_num_steps - 1) * self.topk * self.topk,
                        ),
                        dtype=np.int32,
                    )
                    token_list_replicated = device_array(
                        token_list_host,
                        sharding=replicated_2d_sharding,
                    )
                    parents_list = device_array(
                        np.zeros(
                            (bs, self.topk + 1 + (self.speculative_num_steps - 1) * self.topk),
                            dtype=np.int32,
                        ),
                        sharding=data_2d_sharding,
                    )
                    select_hidden_states = hidden
                    select_scores = None
                    select_topk_p = topk_p
                    select_topk_index = np.zeros((bs, self.topk), dtype=np.int32)
                    for i in range(self.speculative_num_steps):
                        input_ids, select_hidden_states, select_scores, tree_info = (
                            select_top_k_tokens(
                                i,
                                select_topk_p,
                                select_topk_index,
                                select_hidden_states,
                                select_scores,
                                self.topk,
                            )
                        )
                        if isinstance(input_ids, jax.Array):
                            input_ids.block_until_ready()
                        if isinstance(select_hidden_states, jax.Array):
                            select_hidden_states.block_until_ready()
                        if isinstance(select_scores, jax.Array):
                            select_scores.block_until_ready()
                        for tree_item in tree_info:
                            if isinstance(tree_item, jax.Array):
                                tree_item.block_until_ready()
                        score_list, token_list, parents_list = update_eagle_lists(
                            i, score_list, token_list, parents_list, tree_info, self.topk
                        )
                        score_list.block_until_ready()
                        token_list.block_until_ready()
                        parents_list.block_until_ready()
                        if i == self.speculative_num_steps - 1:
                            break
                        select_topk_p = device_array(
                            np.zeros((bs * self.topk, self.topk), dtype=np.float32),
                            sharding=data_2d_sharding,
                        ).astype(dtype)
                        select_topk_index = device_array(
                            np.zeros((bs * self.topk, self.topk), dtype=np.int32),
                            sharding=data_2d_sharding,
                        )
                        select_hidden_states = device_array(
                            np.zeros((bs * self.topk, hidden_size), dtype=np.float32),
                            sharding=data_2d_sharding,
                        ).astype(dtype)
                    for context_len in self.precompile_token_paddings:
                        verified_id_host = np.zeros((bs,), dtype=np.int32)
                        seq_lens_host = np.full((bs,), context_len, dtype=np.int32)
                        seq_lens_device = device_array(seq_lens_host, sharding=data_sharding)
                        seq_lens_replicated = device_array(
                            seq_lens_host, sharding=replicated_sharding
                        )
                        for precompile_verified_id, precompile_token_list in (
                            (verified_id_host, token_list_host),
                            (verified_id, token_list),
                            (replicated_verified_id, token_list_replicated),
                        ):
                            for seq_lens in (
                                seq_lens_host,
                                seq_lens_device,
                                seq_lens_replicated,
                            ):
                                tree_outputs = build_tree_kernel_efficient(
                                    precompile_verified_id,
                                    score_list,
                                    precompile_token_list,
                                    parents_list,
                                    seq_lens,
                                    np.asarray(context_len * bs, dtype=np.int32),
                                    self.topk,
                                    self.speculative_num_draft_tokens,
                                    self._pick_context_len(context_len),
                                    bs,
                                    self.speculative_num_steps,
                                    self.mesh,
                                )
                                for output in tree_outputs:
                                    if output is not None:
                                        output.block_until_ready()
                    continue

                step_min_1 = self.speculative_num_steps - 1
                score_list = device_array(
                    np.zeros((bs, 1 + step_min_1 * self.topk, self.topk), dtype=np.float32),
                    sharding=NamedSharding(self.mesh, P("data", None, None)),
                )
                token_list = device_array(
                    np.zeros(
                        (bs, self.topk + step_min_1 * self.topk * self.topk),
                        dtype=np.int32,
                    ),
                    sharding=data_2d_sharding,
                )
                parents_list = device_array(
                    np.zeros((bs, self.topk + 1 + step_min_1 * self.topk), dtype=np.int32),
                    sharding=data_2d_sharding,
                )
                hidden_states = device_array(
                    np.zeros((bs, hidden_size), dtype=dtype),
                    sharding=data_2d_sharding,
                )
                scores = None
                for i in range(self.speculative_num_steps):
                    _, hidden_states, scores, tree_info = select_top_k_tokens(
                        i, topk_p, topk_index, hidden_states, scores, self.topk
                    )
                    score_list, token_list, parents_list = update_eagle_lists(
                        i, score_list, token_list, parents_list, tree_info, self.topk
                    )
                    score_list.block_until_ready()
                    token_list.block_until_ready()
                    parents_list.block_until_ready()

        end_time = time.perf_counter()
        logger.info("[SPEC_RUNTIME] Precompile finished in %.0f secs", end_time - start_time)


# ---------------------------------------------------------------------------
# Module-level JIT helpers (used exclusively by EagleDraftWorker)
# ---------------------------------------------------------------------------


@functools.partial(jax.jit, static_argnames=["topk"])
def topk_probs_from_logits(
    logits: jax.Array, topk: int, axis: int = -1
) -> tuple[jax.Array, jax.Array]:
    """Return top-k probabilities without materializing the full softmax tensor."""
    working_logits = jnp.moveaxis(logits, axis, -1) if axis != -1 else logits
    # TODO(#1053 Phase 2): replace this all-gather with a sharded top-k +
    # logsumexp over the vocab axis for DP/TP scalability.
    sh = jax.typeof(working_logits).sharding
    if isinstance(sh, NamedSharding):
        working_logits = jax.sharding.reshard(working_logits, NamedSharding(sh.mesh, P()))
    topk_logits, topk_index = jax.lax.top_k(working_logits, topk)
    logsumexp = jax.nn.logsumexp(working_logits, axis=-1, keepdims=True)
    topk_probs = jnp.exp(topk_logits - logsumexp)

    if axis != -1:
        topk_probs = jnp.moveaxis(topk_probs, -1, axis)
        topk_index = jnp.moveaxis(topk_index, -1, axis)

    return topk_probs, topk_index


def fast_topk(values, topk, axis=-1):
    working_values = jnp.moveaxis(values, axis, -1) if axis != -1 else values
    result_vals, result_indices = jax.lax.top_k(working_values, topk)

    if axis != -1:
        result_vals = jnp.moveaxis(result_vals, -1, axis)
        result_indices = jnp.moveaxis(result_indices, -1, axis)

    return result_vals, result_indices


@functools.partial(jax.jit, static_argnames=["i", "topk"])
def update_eagle_lists(
    i: int,
    score_list: jax.Array,
    token_list: jax.Array,
    parents_list: jax.Array,
    tree_info: tuple[jax.Array, jax.Array, jax.Array],
    topk: int,
):
    bs = score_list.shape[0]
    scores_update, tokens_update, parents_update = tree_info
    if i == 0:
        score_list = score_list.at[:bs, :1, :].set(scores_update[:bs])
        token_list = token_list.at[:bs, :topk].set(tokens_update[:bs])
        parents_list = parents_list.at[:bs, : topk + 1].set(parents_update[:bs])
    else:
        score_start = 1 + (i - 1) * topk
        token_start = topk + (i - 1) * topk * topk
        parent_start = topk + 1 + (i - 1) * topk

        score_list = score_list.at[:bs, score_start : score_start + topk, :].set(scores_update[:bs])
        token_list = token_list.at[:bs, token_start : token_start + topk * topk].set(
            tokens_update[:bs]
        )
        parents_list = parents_list.at[:bs, parent_start : parent_start + topk].set(
            parents_update[:bs]
        )
    return score_list, token_list, parents_list


# FIXME(pc) this should be jitted or convert as np.ndarray
# @functools.partial(jax.jit, static_argnames=["i"])
def update_forward_batch_info(
    forward_batch: ForwardBatch,
    i: int,
    input_ids: jax.Array,
    hidden_states: jax.Array,
    positions_base: jax.Array,
) -> ForwardBatch:
    input_ids = jax.device_get(input_ids)
    forward_batch.input_ids = input_ids
    # FIXME(pc) hiddenstate will become NAN when forward path is very long, we still have no reason for this
    forward_batch.spec_info.hidden_states = hidden_states
    forward_batch.positions = positions_base + i
    return forward_batch


def select_top_k_tokens(
    i: int,
    topk_p: jax.Array,
    topk_index: jax.Array,
    hidden_states: jax.Array,
    scores: jax.Array,
    topk: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    if i == 0:
        return select_top_k_tokens_step_0(topk_p, topk_index, hidden_states, scores, topk)
    else:
        return select_top_k_tokens_step_greater_0(
            jnp.asarray(i), topk_p, topk_index, hidden_states, scores, topk
        )


@functools.partial(jax.jit, static_argnames=["topk"])
def select_top_k_tokens_step_0(
    topk_p: jax.Array,
    topk_index: jax.Array,
    hidden_states: jax.Array,
    scores: jax.Array,
    topk: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    input_ids = topk_index.flatten()
    hidden_states = jnp.repeat(hidden_states, topk, axis=0)
    scores = topk_p
    tree_info = (
        jnp.expand_dims(topk_p, axis=1),
        topk_index,
        jnp.tile(
            jnp.expand_dims(jnp.arange(-1, topk, dtype=jnp.float32), axis=0),
            (topk_p.shape[0], 1),
        ),
    )
    return input_ids, hidden_states, scores, tree_info


# NOTE: not @jax.jit. Inside a JIT trace hidden_states.sharding is the
# tracer's abstract sharding (often not NamedSharding); the per-row gather
# below has to be done eagerly so we can branch on the concrete sharding
# class and supply an explicit out_sharding (JAX 0.8.1 requirement).
def select_top_k_tokens_step_greater_0(
    i: jax.Array,
    topk_p: jax.Array,
    topk_index: jax.Array,
    hidden_states: jax.Array,
    scores: jax.Array,
    topk: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    expand_scores = jax.lax.mul(jnp.expand_dims(scores, axis=2), topk_p.reshape(-1, topk, topk))
    topk_cs_p, topk_cs_index = fast_topk(
        expand_scores.reshape(expand_scores.shape[0], -1), topk, axis=-1
    )
    scores = topk_cs_p
    topk_index = topk_index.reshape(-1, topk**2)
    input_ids = jnp.take_along_axis(topk_index, topk_cs_index, axis=1).flatten()
    if hidden_states.shape[0] > 0:
        selected_input_index = topk_cs_index.flatten() // topk + jnp.repeat(
            jnp.arange(0, hidden_states.shape[0], topk), topk
        )
        if isinstance(hidden_states.sharding, NamedSharding):
            # Reuse the source sharding rather than forcing a fixed spec —
            # forcing P("data", None) on an input that is replicated or
            # sharded along tensor would silently reshard and shift data
            # across devices, breaking downstream draft consumers.
            hidden_states = hidden_states.at[selected_input_index, :].get(
                out_sharding=hidden_states.sharding
            )
        else:
            hidden_states = hidden_states[selected_input_index, :]
    tree_info = (
        expand_scores,
        topk_index,
        topk_cs_index + (topk**2 * (i - 1) + topk),
    )
    return input_ids, hidden_states, scores, tree_info
