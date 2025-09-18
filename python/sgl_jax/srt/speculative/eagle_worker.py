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
from sgl_jax.srt.speculative.eagle_util import EagleDraftInput, EagleVerifyInput
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
            unfinished_index_device = jax.numpy.array(
                unfinished_req_index,
                dtype=jax.numpy.int64,
                device=batch.spec_info.topk_p.device,
            )
            batch.spec_info.filter_batch(
                unfinished_index_device, has_been_filtered=False
            )

    @property
    def draft_model_runner(self):
        return self.get_model_runner()

    def capture_for_decode(
        self, logits_output: LogitsProcessorOutput, spec_info: EagleDraftInput
    ):
        pass

    def draft(self, batch: ScheduleBatch):
        if batch.forward_mode.is_idle():
            self._draft_preprocess_idle(batch)
        else:
            self._draft_preprocess_decode(batch)

        spec_info = batch.spec_info
        assert isinstance(spec_info, EagleDraftInput)

        spec_info.capture_hidden_mode = CaptureHiddenMode.LAST
        spec_info.num_tokens_per_batch = self.topk
        spec_info.num_tokens_for_logprob_per_batch = self.topk
        batch.return_hidden_states = False

        # Get forward batch
        model_worker_batch = batch.get_model_worker_batch()
        assert model_worker_batch.capture_hidden_mode == CaptureHiddenMode.LAST
        forward_batch = ForwardBatch.init_new(
            model_worker_batch, self.draft_model_runner
        )
        if not forward_batch.forward_mode.is_idle():
            # Initialize attention backend
            forward_metadata = (
                self.draft_model_runner.attn_backend.get_forward_metadata(
                    model_worker_batch, self.mesh
                )
            )
            self.draft_model_runner.attn_backend.forward_metadata = forward_metadata
        # Run forward steps
        score_list, token_list, parents_list = self.draft_forward(forward_batch)

        # build tree

        return EagleVerifyInput()

    def verify(self, batch: ScheduleBatch, spec_info: EagleDraftInput):
        pass

    def forward_draft_extend_after_decode(self, batch: ScheduleBatch):
        pass

    def draft_forward(self, forward_batch: ForwardBatch):
        spec_info = forward_batch.spec_info
        assert isinstance(spec_info, EagleDraftInput)
        out_cache_loc = forward_batch.out_cache_loc
        topk_p, topk_index, hidden_states = (
            spec_info.topk_p,
            spec_info.topk_index,
            spec_info.hidden_states,
        )
        if self.hot_token_id is not None:
            topk_index = self.hot_token_id[topk_index]
        out_cache_loc = out_cache_loc.reshape(
            forward_batch.batch_size, self.topk, self.speculative_num_steps
        )
        out_cache_loc = out_cache_loc.permute((2, 0, 1)).reshape(
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
            forward_batch.input_ids = input_ids
            forward_batch.out_cache_loc = out_cache_loc[i]
            forward_batch.positions.add(1)
            forward_batch.attn_backend = self.draft_attn_backend.attn_backends[i]
            spec_info.hidden_states = hidden_states

            # Run forward
            logits_output, _ = self.draft_model_runner.forward(
                forward_batch, skip_attn_backend_init=True
            )
            self._detect_nan_if_needed(logits_output)
            probs = jnp.softmax(logits_output.next_token_logits, dim=-1)
            topk_p, topk_index = fast_topk(probs, self.topk, dim=-1)
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
            # todo: page size > 1
            self.extend_lens = 1
            self.num_new_pages_per_topk = 1
            pass

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
                        batch.req_to_token_pool[req_idx, token_pos] = cache_loc

        if self.page_size > 1 and self.topk > 1:
            # Remove padded slots
            out_cache_loc = out_cache_loc[
                : num_seqs * self.topk * self.speculative_num_steps
            ]

        batch.out_cache_loc = out_cache_loc
        batch.seq_lens_sum = jnp.sum(batch.seq_lens).item()
        batch.return_hidden_states = False
        spec_info.positions = jnp.repeat(batch.seq_lens, self.topk)
        self.token_to_kv_pool_allocator.restore_state(token_to_kv_pool_state_backup)


def select_top_k_tokens(
    step: int,
    topk_p: jax.Array,
    topk_index: jax.Array,
    hidden_states: jax.Array,
    scores: jax.Array,
    topk: int,
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    pass
