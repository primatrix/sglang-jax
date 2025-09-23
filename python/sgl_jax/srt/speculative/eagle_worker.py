import logging
from typing import List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.layers.logits_processor import LogitsMetadata, LogitsProcessorOutput
from sgl_jax.srt.managers.schedule_batch import ModelWorkerBatch, ScheduleBatch
from sgl_jax.srt.managers.tp_worker import ModelWorker
from sgl_jax.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
)
from sgl_jax.srt.sampling.sampling_batch_info import SamplingMetadata
from sgl_jax.srt.speculative.eagle_util import (
    EagleDraftInput,
    EagleVerifyInput,
    build_tree_kernel_efficient,
    get_last_loc_large_page_size_large_top_k,
    get_last_loc_large_page_size_top_k_1,
)
from sgl_jax.srt.speculative.spec_info import SpeculativeAlgorithm

logger = logging.getLogger(__name__)


class EAGLEWorker(ModelWorker):
    def __init__(self, server_args, target_worker: ModelWorker):
        self.target_worker = target_worker
        self.speculative_num_steps = server_args.speculative_num_steps
        self.topk = server_args.speculative_eagle_topk
        self.speculative_num_draft_tokens = server_args.speculative_num_draft_tokens
        self.page_size = server_args.page_size
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )
        self.req_to_token_pool, self.token_to_kv_pool_allocator = (
            target_worker.get_memory_pool()
        )
        self.hot_token_id = None

        # Initialize dummy tensors for EAGLE operations
        self.num_new_pages_per_topk = None
        self.extend_lens = None

        super().__init__(server_args, target_worker.mesh, True, self.req_to_token_pool)

        embed, head = self.target_worker.model_runner.model.get_embed_and_head()

        if self.speculative_algorithm.is_eagle3():
            pass
        else:
            logger.info(
                f"{server_args.speculative_algorithm} set target model's embed and head for draft model"
            )
            self.target_worker.model_runner.model.set_embed_and_head(embed, head)

    def forward_batch_speculative_generation(
        self,
        batch: ScheduleBatch,
        model_worker_batch: ModelWorkerBatch,
        sample_meta_data: SamplingMetadata,
    ):
        # prefill : Target Extend -> Decode Extend for Update Draft State
        # Decode : Draft → Verify → Update Draft State → Draft → Verify → ...

        if batch.forward_mode.is_extend():
            # target extend
            logits_output, next_token_ids, cache_miss_count, bid, seq_lens = (
                self.forward_target_extend(model_worker_batch, sample_meta_data)
            )
            # draft extend for Update Draft State
            self.forward_draft_extend(
                batch, model_worker_batch, logits_output.hidden_states, next_token_ids
            )
            logger.info(
                f"-------------forward_draft_extend------------{logits_output.hidden_states.shape}"
            )
            return logits_output, next_token_ids, cache_miss_count, 0
        else:
            # draft
            spec_info = self.draft(batch)
            logger.info(f"-------------draft------------{spec_info}")
            # verify
            logits_output, verify_output, model_worker_batch, cache_hit = self.verify(
                batch, spec_info
            )
            self.forward_draft_extend_after_decode(batch)
            return (
                logits_output,
                next_token_ids,
                cache_miss_count,
                sum(verify_output.accept_length_per_req_cpu),
            )

    def forward_target_extend(
        self, model_worker_batch: ModelWorkerBatch, sample_meta_data: SamplingMetadata
    ) -> Tuple[LogitsProcessorOutput, jax.Array, int, int, np.ndarray]:
        model_worker_batch.capture_hidden_mode = CaptureHiddenMode.FULL
        logits_output, next_token_ids, cache_miss_count = (
            self.target_worker.forward_batch_generation(
                model_worker_batch, sampling_metadata=sample_meta_data
            )
        )
        return (
            logits_output,
            next_token_ids,
            cache_miss_count,
            model_worker_batch.bid,
            model_worker_batch.seq_lens,
        )

    def forward_draft_extend(
        self,
        batch: ScheduleBatch,
        model_worker_batch: ModelWorkerBatch,
        hidden_states: jax.Array,
        next_token_ids: jax.Array,
    ):
        batch.spec_info = EagleDraftInput(
            hidden_states=hidden_states,
            verified_id=next_token_ids[: model_worker_batch.real_bs],
            num_tokens_per_batch=1,
            num_tokens_for_logprob_per_batch=1,
        )
        batch.return_hidden_states = False
        batch.spec_info.prepare_for_extend(batch)
        batch.spec_info.capture_hidden_mode = CaptureHiddenMode.LAST
        #  this place we shift the input_ids, so we need re-get the model_worker_batch
        (
            precompile_token_paddings,
            precompile_bs_paddings,
            precompile_cache_loc_paddings,
        ) = self.target_worker.get_precompile_paddings()
        model_worker_batch = batch.get_model_worker_batch(
            precompile_token_paddings,
            precompile_bs_paddings,
            precompile_cache_loc_paddings,
            self.page_size,
        )
        forward_batch = ForwardBatch.init_new(
            model_worker_batch, self.draft_model_runner
        )
        forward_batch.return_logprob = False

        # Set forward_metadata for draft_model_runner's attention backend
        forward_metadata = self.draft_model_runner.attn_backend.get_forward_metadata(
            model_worker_batch, self.draft_model_runner.mesh
        )
        self.draft_model_runner.attn_backend.forward_metadata = forward_metadata

        logits_output, _ = self.draft_model_runner.forward(
            forward_batch,
            logits_metadata=LogitsMetadata.from_model_worker_batch(
                model_worker_batch, self.mesh
            ),
        )
        assert isinstance(forward_batch.spec_info, EagleDraftInput)
        assert forward_batch.spec_info is batch.spec_info
        self.capture_for_decode(logits_output, forward_batch.spec_info)
        has_finished, unfinished_req_index = False, []
        for i, req in enumerate(batch.reqs):
            if req.finished():
                has_finished = True
            else:
                unfinished_req_index.append(i)
        if has_finished:
            unfinished_index_device = jnp.array(
                unfinished_req_index,
                dtype=jnp.int64,
            )
            batch.spec_info.filter_batch(
                unfinished_index_device, has_been_filtered=False
            )
        logger.info(f"-------------extend------------{batch.spec_info}")

    @property
    def draft_model_runner(self):
        return self.get_model_runner()

    def capture_for_decode(
        self, logits_output: LogitsProcessorOutput, draft_input: EagleDraftInput
    ):
        probs = jax.nn.softmax(logits_output.next_token_logits, axis=-1)
        draft_input.topk_p, draft_input.topk_index = fast_topk(
            probs, self.topk, axis=-1
        )
        draft_input.hidden_states = logits_output.hidden_states

    def draft(self, batch: ScheduleBatch):
        if batch.forward_mode.is_idle():
            self._draft_preprocess_idle(batch)
        else:
            self._draft_preprocess_decode(batch)

        spec_info = batch.spec_info
        assert isinstance(spec_info, EagleDraftInput)
        logger.info(f"-------------draft------------{spec_info}")
        spec_info.capture_hidden_mode = CaptureHiddenMode.LAST
        spec_info.num_tokens_per_batch = self.topk
        spec_info.num_tokens_for_logprob_per_batch = self.topk
        batch.return_hidden_states = False

        # if not model_worker_batch.forward_mode.is_idle():
        #     forward_batch = ForwardBatch.init_new(
        #         model_worker_batch, self.draft_model_runner
        #     )
        #     # Initialize attention backend
        #     forward_metadata = (
        #         self.draft_model_runner.attn_backend.get_forward_metadata(
        #             model_worker_batch, self.mesh
        #         )
        #     )
        #     self.draft_model_runner.attn_backend.forward_metadata = forward_metadata

        # Run forward steps
        score_list, token_list, parents_list = self.draft_forward(batch)
        logger.info(f"-------------score_list------------{score_list}")
        logger.info(f"-------------token_list------------{token_list}")
        logger.info(f"-------------parents_list------------{parents_list}")
        (
            tree_mask,
            position,
            retrive_index,
            retrive_next_token,
            retrive_next_sibling,
            draft_tokens,
        ) = build_tree_kernel_efficient(
            spec_info.verified_id,
            score_list,
            token_list,
            parents_list,
            batch.seq_lens,
            batch.seq_lens_sum,
            self.topk,
            self.speculative_num_steps,
            self.speculative_num_draft_tokens,
        )
        # build tree
        logger.info(f"-------------tree_mask------------{tree_mask}")
        logger.info(f"-------------position------------{position}")
        logger.info(f"-------------retrive_index------------{retrive_index}")
        logger.info(f"-------------retrive_next_token------------{retrive_next_token}")
        logger.info(
            f"-------------retrive_next_sibling------------{retrive_next_sibling}"
        )
        logger.info(f"-------------draft_tokens------------{draft_tokens}")
        return EagleVerifyInput(
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
            seq_lens_sum=batch.seq_lens_sum,
            seq_lens_cpu=batch.seq_lens,
        )

    def verify(self, batch: ScheduleBatch, spec_info: EagleDraftInput):
        pass

    def forward_draft_extend_after_decode(self, batch: ScheduleBatch):
        pass

    def draft_forward(self, schedule_batch: ScheduleBatch):

        (
            precompile_token_paddings,
            precompile_bs_paddings,
            precompile_cache_loc_paddings,
        ) = self.target_worker.get_precompile_paddings()
        # Get forward batch
        model_worker_batch = schedule_batch.get_model_worker_batch(
            precompile_token_paddings,
            precompile_bs_paddings,
            precompile_cache_loc_paddings,
            self.page_size,
        )
        assert model_worker_batch.capture_hidden_mode == CaptureHiddenMode.LAST

        spec_info = model_worker_batch.spec_info
        assert isinstance(spec_info, EagleDraftInput)
        out_cache_loc = model_worker_batch.out_cache_loc
        topk_p, topk_index, hidden_states = (
            spec_info.topk_p,
            spec_info.topk_index,
            spec_info.hidden_states,
        )
        if self.hot_token_id is not None:
            topk_index = self.hot_token_id[topk_index]
        out_cache_loc = out_cache_loc.reshape(
            schedule_batch.batch_size(), self.topk, self.speculative_num_steps
        )
        out_cache_loc = jnp.transpose(out_cache_loc, (2, 0, 1)).reshape(
            self.speculative_num_steps, -1
        )
        # Return values
        score_list: List[jax.Array] = []
        token_list: List[jax.Array] = []
        parents_list: List[jax.Array] = []
        # Forward multiple steps
        scores = None
        for i in range(self.speculative_num_steps):
            input_ids, hidden_states, scores, tree_info = select_top_k_tokens(
                i, topk_p, topk_index, hidden_states, scores, self.topk
            )
            score_list.append(tree_info[0])
            token_list.append(tree_info[1])
            parents_list.append(tree_info[2])

            if i == self.speculative_num_steps - 1:
                break
            model_worker_batch.input_ids = input_ids
            model_worker_batch.out_cache_loc = out_cache_loc[i]
            model_worker_batch.positions = model_worker_batch.positions + 1
            self.draft_model_runner.attn_backend.forward_metadata = (
                self.draft_model_runner.attn_backend.get_forward_metadata(
                    model_worker_batch, self.draft_model_runner.mesh
                )
            )
            spec_info.hidden_states = hidden_states
            forward_batch = ForwardBatch.init_new(
                model_worker_batch, self.draft_model_runner
            )
            # Run forward
            logits_output, _ = self.draft_model_runner.forward(
                forward_batch,
                logits_metadata=LogitsMetadata.from_model_worker_batch(
                    model_worker_batch, self.draft_model_runner.mesh
                ),
            )
            # self._detect_nan_if_needed(logits_output)
            probs = jax.nn.softmax(logits_output.next_token_logits, axis=-1)
            topk_p, topk_index = fast_topk(probs, self.topk, axis=-1)
            if self.hot_token_id is not None:
                topk_index = self.hot_token_id[topk_index]
            hidden_states = logits_output.hidden_states

        return score_list, token_list, parents_list

    def _draft_preprocess_idle(self, batch: ScheduleBatch):
        pass

    def _draft_preprocess_decode(self, batch: ScheduleBatch):
        # Parse args
        num_seqs = batch.batch_size()
        spec_info = batch.spec_info

        # todo: add penalty

        if self.page_size == 1:
            out_cache_loc, token_to_kv_pool_state_backup = batch.alloc_token_slots(
                num_seqs * self.speculative_num_steps * self.topk, backup_state=True
            )
        else:
            if self.topk == 1:
                prefix_lens, seq_lens, last_loc = get_last_loc_large_page_size_top_k_1(
                    batch.req_to_token_pool.req_to_token,
                    batch.req_pool_indices,
                    batch.seq_lens,
                    self.speculative_num_steps,
                )
                extend_num_tokens = num_seqs * self.speculative_num_steps
            else:
                # In this case, the last partial page needs to be duplicated.
                # KV cache layout in batch.req_to_token_pool.req_to_token:
                #
                # | -------- | -- xxxx .. | -- xxxx .. | -- xxxx .. |
                #    prefix     top-k = 0    tok-k = 1    top-k = 2
                #
                #  "-" means prefix tokens
                #  "x" means speculative draft tokens
                #  "." means padded tokens

                # TODO(lmzheng): The current implementation is still a fake support
                # for page size > 1. In the `assign_draft_cache_locs` below,
                # we directly move the indices instead of the real kv cache.
                # This only works when the kernel backend runs with page size = 1.
                # If the kernel backend runs with page size > 1, we need to
                # duplicate the real KV cache. The overhead of duplicating KV
                # cache seems okay because the draft KV cache only has one layer.
                # see a related copy operation in MHATokenToKVPool::move_kv_cache.

                (
                    prefix_lens,
                    seq_lens,
                    last_loc,
                    num_new_pages_per_topk,
                    extend_lens,
                ) = get_last_loc_large_page_size_large_top_k(
                    batch.req_to_token_pool.req_to_token,
                    batch.req_pool_indices,
                    batch.seq_lens,
                    self.speculative_num_steps,
                    self.topk,
                    self.page_size,
                )

                # TODO(lmzheng): remove this device sync
                extend_num_tokens = int(jnp.sum(extend_lens))

                # Store in instance variables for later use
                self.num_new_pages_per_topk = num_new_pages_per_topk
                self.extend_lens = extend_lens

            out_cache_loc, token_to_kv_pool_state_backup = (
                batch.alloc_paged_token_slots_extend(
                    prefix_lens,
                    seq_lens,
                    last_loc,
                    extend_num_tokens,
                    backup_state=True,
                )
            )

        # [       topk 0         ] [       topk 1         ]
        # [iter=0, iter=1, iter=2] [iter=0, iter=1, iter=2]

        # Update req_to_token_pool with cache locations (no reshape needed)
        # Layout: [seq0_topk0_steps, seq0_topk1_steps, seq1_topk0_steps, ...]
        for i in range(num_seqs):
            req_idx = batch.req_pool_indices[i].item()
            start_pos = batch.seq_lens[i].item()

            # For each topk branch
            for k in range(self.topk):
                # For each speculative step
                for step in range(self.speculative_num_steps):
                    # Calculate flat index: i * (topk * steps) + k * steps + step
                    flat_idx = (
                        i * (self.topk * self.speculative_num_steps)
                        + k * self.speculative_num_steps
                        + step
                    )
                    token_pos = start_pos + step
                    cache_loc = out_cache_loc[flat_idx].item()

                    # Update req_to_token mapping
                    if token_pos < batch.req_to_token_pool.req_to_token.shape[1]:
                        batch.req_to_token_pool.write((req_idx, token_pos), cache_loc)

        if self.page_size > 1 and self.topk > 1:
            # Remove padded slots
            out_cache_loc = out_cache_loc[
                : num_seqs * self.topk * self.speculative_num_steps
            ]

        batch.out_cache_loc = out_cache_loc
        batch.seq_lens_sum = int(jnp.sum(batch.seq_lens))
        batch.return_hidden_states = False
        spec_info.positions = jnp.repeat(batch.seq_lens, self.topk)
        self.token_to_kv_pool_allocator.restore_state(token_to_kv_pool_state_backup)


def fast_topk(values, topk, axis=-1):
    if axis != -1:
        # Move target axis to last position
        values = jnp.moveaxis(values, axis, -1)

    if topk == 1:
        # Get max value and index for k=1 case
        max_vals = jnp.max(values, axis=-1, keepdims=True)
        max_indices = jnp.argmax(values, axis=-1, keepdims=True)
        result_vals, result_indices = max_vals, max_indices
    else:
        # Use top_k for k>1 case (operates on last axis)
        result_vals, result_indices = jax.lax.top_k(values, topk)

    if axis != -1:
        # Move axis back to original position
        result_vals = jnp.moveaxis(result_vals, -1, axis)
        result_indices = jnp.moveaxis(result_indices, -1, axis)

    return result_vals, result_indices


def select_top_k_tokens(
    i: int,
    topk_p: jax.Array,
    topk_index: jax.Array,
    hidden_states: jax.Array,
    scores: jax.Array,
    topk: int,
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    if i == 0:
        # The first step after extend
        input_ids = topk_index.flatten()
        hidden_states = jnp.repeat(hidden_states, topk, axis=0)
        scores = topk_p  # shape: (b, topk)

        tree_info = (
            jnp.expand_dims(topk_p, axis=1),  # shape: (b, 1, topk)
            topk_index,  # shape: (b, topk)
            jnp.tile(
                jnp.expand_dims(jnp.arange(-1, topk, dtype=jnp.float32), axis=0),
                (topk_p.shape[0], 1),
            ),  # shape: (b, topk + 1)
        )
    else:
        # The later decode steps
        expand_scores = jax.lax.mul(
            jnp.expand_dims(scores, axis=2), topk_p.reshape(-1, topk, topk)
        )  # (b, topk, 1) x (b, topk ,topk) -> (b, topk, topk)
        topk_cs_p, topk_cs_index = fast_topk(
            expand_scores.reshape(expand_scores.shape[0], -1), topk, axis=-1
        )  # (b, topk)
        scores = topk_cs_p  # shape: (b, topk)

        topk_index = topk_index.reshape(-1, topk**2)
        input_ids = jnp.take_along_axis(topk_index, topk_cs_index, axis=1).flatten()

        if hidden_states.shape[0] > 0:
            selected_input_index = topk_cs_index.flatten() // topk + jnp.repeat(
                jnp.arange(0, hidden_states.shape[0], topk), topk
            )
            hidden_states = hidden_states[selected_input_index, :]

        tree_info = (
            expand_scores,  # shape: (b, topk, topk)
            topk_index,  # shape: (b, topk * topk)
            topk_cs_index + (topk**2 * (i - 1) + topk),  # shape: (b, topk)
        )

    return input_ids, hidden_states, scores, tree_info
