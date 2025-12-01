import functools
import itertools
import logging
import time

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from tqdm import tqdm

from sgl_jax.srt.layers.logits_processor import LogitsMetadata, LogitsProcessorOutput
from sgl_jax.srt.layers.sampler import get_token_ids_logprobs, get_top_logprobs
from sgl_jax.srt.managers.schedule_batch import ModelWorkerBatch, ScheduleBatch
from sgl_jax.srt.managers.scheduler import GenerationBatchResult
from sgl_jax.srt.managers.tp_worker import ModelWorker
from sgl_jax.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sgl_jax.srt.sampling.sampling_batch_info import SamplingMetadata
from sgl_jax.srt.speculative.eagle_util import (
    EagleDraftInput,
    EagleVerifyInput,
    EagleVerifyOutput,
    build_tree_kernel_efficient,
    build_tree_mask_for_draft_decode,
)
from sgl_jax.srt.speculative.spec_info import SpeculativeAlgorithm
from sgl_jax.srt.utils.common_utils import get_bool_env_var
from sgl_jax.srt.utils.jax_utils import device_array

logger = logging.getLogger(__name__)
RETURN_ORIGINAL_LOGPROB = get_bool_env_var("RETURN_ORIGINAL_LOGPROB")


class EAGLEWorker(ModelWorker):
    def __init__(self, server_args, target_worker: ModelWorker):
        self.server_args = server_args
        self.target_worker = target_worker
        self.topk = server_args.speculative_eagle_topk
        self.speculative_num_steps = server_args.speculative_num_steps
        self.topk = server_args.speculative_eagle_topk
        self.speculative_num_draft_tokens = server_args.speculative_num_draft_tokens
        self.page_size = server_args.page_size
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )
        self.req_to_token_pool, self.token_to_kv_pool_allocator = target_worker.get_memory_pool()
        self.hot_token_ids = None

        # Initialize dummy tensors for EAGLE operations
        self.num_new_pages_per_topk = None
        self.extend_lens = None

        # this must be put at last to make sure model state is correct
        super().__init__(
            server_args,
            target_worker.mesh,
            req_to_token_pool=self.req_to_token_pool,
            is_draft_worker=True,
        )
        EagleDraftInput.ALLOC_LEN_PER_DECODE = max(
            self.speculative_num_steps * self.topk, self.speculative_num_draft_tokens
        )
        embed, head = self.target_worker.model_runner.model.get_embed_and_head()

        if self.speculative_algorithm.is_eagle3():
            # most cases EAGLE3 models don't share lm_head
            # but some models (e.g. nvidia/gpt-oss-120b-Eagle3) shares
            if (
                hasattr(self.draft_model_runner.model, "load_lm_head_from_target")
                and self.draft_model_runner.model.load_lm_head_from_target
            ):
                self.draft_model_runner.model.set_embed_and_head(embed, head)
            else:
                self.draft_model_runner.model.set_embed(embed)

            # grab hot token ids
            if self.draft_model_runner.model.hot_token_ids is not None:
                self.hot_token_ids = device_array(
                    self.draft_model_runner.model.hot_token_ids,
                    sharding=(
                        NamedSharding(self.model_runner.mesh, P())
                        if jax.process_count() == 1
                        else None
                    ),
                )
        else:
            if self.hot_token_ids is not None:
                head = head.clone()
                self.hot_token_ids = device_array(
                    self.draft_model_runner.model.hot_token_ids,
                    sharding=(
                        NamedSharding(self.model_runner.mesh, P())
                        if jax.process_count() == 1
                        else None
                    ),
                )
                head.data = head.data[self.hot_token_ids]

            # Share the embedding and lm_head
            self.draft_model_runner.model.set_embed_and_head(embed, head)

        self.model_runner.initialize_jit()
        (
            precompile_token_paddings,
            precompile_bs_paddings,
            precompile_cache_loc_paddings,
        ) = self.target_worker.get_precompile_paddings()
        self.precompile_bs_paddings = precompile_bs_paddings
        self.precompile_cache_loc_paddings = precompile_cache_loc_paddings
        self.precompile_token_paddings = precompile_token_paddings

    def forward_batch_speculative_generation(
        self,
        model_worker_batch: ModelWorkerBatch,
    ):
        if model_worker_batch.forward_mode.is_extend():
            model_worker_batch.padding_model_worker_batch(
                self.precompile_token_paddings,
                self.precompile_bs_paddings,
                self.precompile_cache_loc_paddings,
            )
            sampling_metadata = SamplingMetadata.from_model_worker_batch(
                model_worker_batch,
                len(model_worker_batch.seq_lens) - model_worker_batch.real_bs,
                self.mesh,
                vocab_size=self.model_config.vocab_size,
            )
            # target extend
            logits_output, next_token_ids, cache_miss_count, bid, seq_lens = (
                self.forward_target_extend(model_worker_batch, sampling_metadata)
            )
            # draft extend for Update Draft State
            self.draft_extend_for_prefill(
                model_worker_batch, logits_output.hidden_states, next_token_ids
            )
            # FIXME(pc) refactor this to batch output
            batch_output = GenerationBatchResult(
                logits_output=logits_output,
                next_token_ids=next_token_ids,
                next_draft_input=model_worker_batch.spec_info,
                allocate_lens=model_worker_batch.seq_lens[: model_worker_batch.real_bs],
                bid=bid,
                cache_miss_count=cache_miss_count,
                extend_input_len_per_req=None,
                extend_logprob_start_len_per_req=None,
            )
            return batch_output

        else:
            cur_allocate_lens = model_worker_batch.spec_info.allocate_lens
            self.draft(model_worker_batch)

            batch_output = self.verify(model_worker_batch, cur_allocate_lens)

            self.forward_draft_extend_after_decode(model_worker_batch, batch_output)

            return batch_output

    def forward_target_extend(
        self, model_worker_batch: ModelWorkerBatch, sample_meta_data: SamplingMetadata
    ) -> tuple[LogitsProcessorOutput, jax.Array, int, int, np.ndarray]:
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

    def draft_extend_for_prefill(
        self,
        model_worker_batch: ModelWorkerBatch,
        hidden_states: jax.Array,
        next_token_ids: jax.Array,
    ):
        # FIXME(pc) move this all prepare to prepare_for_extend_after_target_prefill
        model_worker_batch.spec_info = EagleDraftInput(
            hidden_states=hidden_states,
            verified_id=next_token_ids[: model_worker_batch.real_bs],
            num_tokens_per_batch=np.asarray(1, dtype=jnp.int32),
            num_tokens_for_logprob_per_batch=np.asarray(1, dtype=jnp.int32),
        )
        model_worker_batch.return_hidden_states = False
        model_worker_batch.spec_info.prepare_for_extend_after_target_prefill(
            model_worker_batch=model_worker_batch
        )
        model_worker_batch.spec_info.capture_hidden_mode = CaptureHiddenMode.LAST
        model_worker_batch.capture_hidden_mode = CaptureHiddenMode.LAST
        forward_batch = ForwardBatch.init_new(model_worker_batch, self.draft_model_runner)
        forward_batch.return_logprob = False

        # Set forward_metadata for draft_model_runner's attention backend
        forward_metadata = self.draft_model_runner.attn_backend.get_eagle_forward_metadata(
            model_worker_batch
        )
        self.draft_model_runner.attn_backend.forward_metadata = forward_metadata
        forward_batch.forward_mode = ForwardMode.EXTEND
        # last_idx = np.cumsum(model_worker_batch.extend_seq_lens, axis=0) - 1

        logits_output, _ = self.draft_model_runner.forward(
            forward_batch,
            logits_metadata=LogitsMetadata.from_model_worker_batch(model_worker_batch, self.mesh),
        )
        logits_output.truncate_logits_processor_output(model_worker_batch)
        assert isinstance(forward_batch.spec_info, EagleDraftInput)
        forward_batch.spec_info.allocate_lens = model_worker_batch.seq_lens[
            : model_worker_batch.real_bs
        ]

        self.capture_for_decode(logits_output, forward_batch.spec_info)

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
        model_worker_batch.extend_start_loc = np.array(
            jax.device_get(model_worker_batch.extend_start_loc),
            dtype=model_worker_batch.extend_start_loc.dtype,
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

    @property
    def draft_model_runner(self):
        return self.get_model_runner()

    def capture_for_decode(
        self, logits_output: LogitsProcessorOutput, draft_input: EagleDraftInput
    ):
        topk_p, topk_index = topk_probs_from_logits(logits_output.next_token_logits, self.topk)
        draft_input.topk_p = topk_p
        draft_input.topk_index = topk_index
        draft_input.hidden_states = logits_output.hidden_states

    def draft(self, model_worker_batch: ModelWorkerBatch):
        spec_info = model_worker_batch.spec_info
        assert isinstance(spec_info, EagleDraftInput)
        spec_info.prepare_for_draft_decode(
            model_worker_batch, self.topk, self.speculative_num_steps
        )
        model_worker_batch.padding_model_worker_batch(
            self.precompile_token_paddings,
            self.precompile_bs_paddings,
            self.precompile_cache_loc_paddings,
        )
        # Run forward steps
        score_list, token_list, parents_list = self.draft_forward(model_worker_batch)
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
            model_worker_batch.seq_lens[: model_worker_batch.real_bs],
            np.sum(model_worker_batch.seq_lens),
            self.topk,
            self.speculative_num_draft_tokens,
            int(self.req_to_token_pool.req_to_token.shape[1]),
            model_worker_batch.seq_lens.shape[0],
            model_worker_batch.speculative_num_steps,
            self.mesh,
        )
        # build tree
        spec_info = EagleVerifyInput(
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
        model_worker_batch.spec_info = spec_info
        return spec_info

    def verify(self, model_worker_batch: ModelWorkerBatch, cur_allocate_lens: jax.Array):
        spec_info: EagleVerifyInput = model_worker_batch.spec_info
        spec_info.prepare_for_verify(model_worker_batch, self.page_size, self.target_worker)
        model_worker_batch.padding_model_worker_batch(
            self.precompile_token_paddings,
            self.precompile_bs_paddings,
            self.precompile_cache_loc_paddings,
        )
        forward_metadata = self.target_worker.model_runner.attn_backend.get_eagle_forward_metadata(
            model_worker_batch
        )
        # custom_mask = forward_metadata.custom_mask
        self.copy_model_worker_batch_to_cpu(model_worker_batch)
        logits_output, _, cache_miss_count = self.target_worker.forward_batch_generation(
            model_worker_batch, skip_sample=True, forward_metadata=forward_metadata
        )
        logits_output.truncate_logits_processor_output(model_worker_batch)
        spec_info.hidden_states = logits_output.hidden_states
        (
            predict,
            verified_id,
            accept_length,
            accept_index,
        ) = spec_info.sample(
            model_worker_batch,
            logits_output,
            # self.token_to_kv_pool_allocator,
            # self.page_size,
            self.model_runner.rngs,
            self.mesh,
        )
        logits_output.next_token_logits = logits_output.next_token_logits[accept_index]
        logits_output.hidden_states = logits_output.hidden_states[accept_index]
        model_worker_batch.positions = model_worker_batch.positions[accept_index]
        new_seq_lens = model_worker_batch.seq_lens[: model_worker_batch.real_bs] + accept_length
        next_draft_input = EagleDraftInput(
            verified_id=verified_id,
            new_seq_lens=new_seq_lens,
            allocate_lens=cur_allocate_lens,
            hidden_states=logits_output.hidden_states,
        )
        model_worker_batch.spec_info = next_draft_input
        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=predict,
            next_draft_input=next_draft_input,
            accept_lens=accept_length,
            # FIXME(pc) this field is for overlap
            allocate_lens=cur_allocate_lens,
            bid=model_worker_batch.bid,
            cache_miss_count=cache_miss_count,
            extend_input_len_per_req=None,
            extend_logprob_start_len_per_req=None,
        )

    def add_logprob_values(
        self,
        batch: ScheduleBatch,
        res: EagleVerifyOutput,
        logits_output: LogitsProcessorOutput,
    ):
        # Extract args
        logits_output = res.logits_output
        top_logprobs_nums = batch.top_logprobs_nums
        token_ids_logprobs = batch.token_ids_logprobs
        accepted_indices = res.accepted_indices
        assert len(accepted_indices) == len(logits_output.next_token_logits)

        temperatures = batch.sampling_info.temperatures
        num_draft_tokens = batch.spec_info.draft_token_num
        # acceptance indices are the indices in a "flattened" batch.
        # dividing it to num_draft_tokens will yield the actual batch index.
        temperatures = temperatures[accepted_indices // num_draft_tokens]
        if RETURN_ORIGINAL_LOGPROB:
            logprobs = jax.nn.log_softmax(logits_output.next_token_logits, axis=-1)
        else:
            logprobs = jax.nn.log_softmax(logits_output.next_token_logits / temperatures, axis=-1)
        batch_next_token_ids = res.verified_id
        num_tokens_per_req = [accept + 1 for accept in res.accept_length_per_req_cpu]

        # We should repeat top_logprobs_nums to match num_tokens_per_req.
        top_logprobs_nums_repeat_interleaved = []
        token_ids_logprobs_repeat_interleaved = []
        for num, num_tokens in zip(top_logprobs_nums, num_tokens_per_req):
            top_logprobs_nums_repeat_interleaved.extend([num] * num_tokens)
        for token_ids, num_tokens in zip(token_ids_logprobs, num_tokens_per_req):
            token_ids_logprobs_repeat_interleaved.extend([token_ids] * num_tokens)

        # Extract logprobs
        if any(x > 0 for x in top_logprobs_nums):
            (
                logits_output.next_token_top_logprobs_val,
                logits_output.next_token_top_logprobs_idx,
            ) = get_top_logprobs(
                logprobs,
                top_logprobs_nums_repeat_interleaved,
            )

        if any(x is not None for x in token_ids_logprobs):
            (
                logits_output.next_token_token_ids_logprobs_val,
                logits_output.next_token_token_ids_logprobs_idx,
            ) = get_token_ids_logprobs(
                logprobs,
                token_ids_logprobs_repeat_interleaved,
            )

        logits_output.next_token_logprobs = logprobs[
            jnp.arange(len(batch_next_token_ids), device=batch.sampling_info.device),
            batch_next_token_ids,
        ]

        # Add output logprobs to the request
        pt = 0
        next_token_logprobs = logits_output.next_token_logprobs.tolist()
        verified_ids = batch_next_token_ids.tolist()
        for req, num_tokens in zip(batch.reqs, num_tokens_per_req, strict=True):
            for _ in range(num_tokens):
                if req.return_logprob:
                    req.output_token_logprobs_val.append(next_token_logprobs[pt])
                    req.output_token_logprobs_idx.append(verified_ids[pt])
                    if req.top_logprobs_num > 0:
                        req.output_top_logprobs_val.append(
                            res.logits_output.next_token_top_logprobs_val[pt]
                        )
                        req.output_top_logprobs_idx.append(
                            res.logits_output.next_token_top_logprobs_idx[pt]
                        )
                pt += 1

    def forward_draft_extend_after_decode(
        self, model_worker_batch: ModelWorkerBatch, batch_output: GenerationBatchResult
    ):
        if batch_output.next_draft_input.verified_id.shape[0] <= 0:
            return
        draft_input = EagleDraftInput(
            hidden_states=batch_output.logits_output.hidden_states,
        )
        forward_batch, logits_meatadata = draft_input.prepare_for_extend_after_verify(
            model_worker_batch,
            batch_output.next_token_ids,
            self.speculative_num_draft_tokens,
            self.draft_model_runner,
            batch_output,
            self.precompile_token_paddings,
            self.precompile_bs_paddings,
            self.precompile_cache_loc_paddings,
        )
        self.copy_model_worker_batch_to_cpu(model_worker_batch)
        forward_batch = ForwardBatch.init_new(model_worker_batch, self.draft_model_runner)
        if forward_batch.input_ids.shape[0] <= 0:
            return
        draft_logits_output, _ = self.draft_model_runner.forward(
            forward_batch,
            logits_metadata=logits_meatadata,
        )

        draft_logits_output.truncate_logits_processor_output(model_worker_batch)
        self.capture_for_decode(draft_logits_output, forward_batch.spec_info)
        select_index = (
            np.arange(len(model_worker_batch.seq_lens[: model_worker_batch.real_bs]))
            * self.speculative_num_draft_tokens
            + batch_output.accept_lens
            - 1
        )
        draft_logits_output.next_token_logits = draft_logits_output.next_token_logits[select_index]
        draft_logits_output.hidden_states = draft_logits_output.hidden_states[select_index]
        topk_p, topk_index = topk_probs_from_logits(
            draft_logits_output.next_token_logits, self.topk
        )

        # prepare for next draft decode
        batch_output.next_draft_input.hidden_states = draft_logits_output.hidden_states[
            : model_worker_batch.real_bs
        ]
        batch_output.next_draft_input.topk_p = topk_p[: model_worker_batch.real_bs]
        batch_output.next_draft_input.topk_index = topk_index[: model_worker_batch.real_bs]

        verified_id_idx = jnp.cumsum(batch_output.accept_lens) - 1
        batch_output.next_draft_input.verified_id = batch_output.next_draft_input.verified_id[
            verified_id_idx
        ]

    def draft_forward(self, model_worker_batch: ModelWorkerBatch):
        model_worker_batch.capture_hidden_mode = CaptureHiddenMode.LAST
        spec_info = model_worker_batch.spec_info
        assert isinstance(spec_info, EagleDraftInput)
        # out_cache_loc = model_worker_batch.out_cache_loc
        topk_p, topk_index, hidden_states = (
            spec_info.topk_p,
            spec_info.topk_index,
            spec_info.hidden_states,
        )
        if self.hot_token_ids is not None:
            topk_index = self.hot_token_ids[topk_index]
        # if we need custom mask, we should create for all at once and update it within loop
        # we should optimize build_tree_mask_for_draft_decode to a kernel
        if self.topk > 1:
            self.draft_model_runner.attn_backend.forward_metadata.custom_mask = (
                build_tree_mask_for_draft_decode(
                    model_worker_batch.seq_lens,
                    topk=topk_index.shape[1],
                    speculative_step_id=0,
                    parents_list=None,
                )
            )

        bs = model_worker_batch.real_bs
        if bs - hidden_states.shape[0] > 0:
            hidden_states = jnp.pad(
                hidden_states,
                (
                    (0, bs * self.topk - hidden_states.shape[0]),
                    0,
                    bs * self.topk - hidden_states.shape[0],
                    (0, 0),
                ),
            )

        # Force sharding to prevent cache miss (Python pad vs JIT output)
        replicated_sharding = (
            NamedSharding(self.model_runner.mesh, P()) if jax.process_count() == 1 else None
        )
        if replicated_sharding is not None:
            hidden_states = jax.device_put(hidden_states, replicated_sharding)
            topk_p = jax.device_put(topk_p, replicated_sharding)
            topk_index = jax.device_put(topk_index, replicated_sharding)
            if spec_info.verified_id is not None:
                spec_info.verified_id = jax.device_put(spec_info.verified_id, replicated_sharding)

        # Update spec_info with sharded arrays
        spec_info.hidden_states = hidden_states
        spec_info.topk_p = topk_p
        spec_info.topk_index = topk_index

        # --- Prepare Metadata for Scan ---
        # Ensure batch parameters are set BEFORE calling get_eagle_multi_step_metadata
        model_worker_batch.speculative_eagle_topk = self.topk
        model_worker_batch.speculative_num_steps = self.speculative_num_steps
        model_worker_batch.speculative_num_draft_tokens = self.speculative_num_draft_tokens

        self.copy_model_worker_batch_to_cpu(model_worker_batch)
        metadata_per_step = self.draft_model_runner.attn_backend.get_eagle_multi_step_metadata(
            model_worker_batch,
        )
        # Forward in scan corresponds to steps 0 to N-2.
        scan_metadata_list = metadata_per_step[:-1]
        if len(scan_metadata_list) > 0:
            stacked_metadata = jax.tree_util.tree_map(
                lambda *xs: jnp.stack(xs), *scan_metadata_list
            )
            self.draft_model_runner.attn_backend.multi_step_metadata = nnx.data(stacked_metadata)

        # --- Step 0 (Peel) ---
        # Initial selection (Tree Expansion Step 0)
        # Use a dummy scores array instead of None because jax.lax.cond traces both branches,
        # and select_top_k_tokens_step_greater_0 expects an array.
        scores = jnp.zeros((bs, self.topk), dtype=topk_p.dtype)
        input_ids, hidden_states, scores, tree_info_0 = select_top_k_tokens_step_0(
            topk_p, topk_index, hidden_states, scores, self.topk
        )

        score_list_0 = tree_info_0[0][:bs]  # (bs, 1, topk)
        token_list_0 = tree_info_0[1][:bs]  # (bs, topk)
        parents_list_0 = tree_info_0[2][:bs]  # (bs, topk + 1)

        if self.speculative_num_steps == 1:
            return score_list_0, token_list_0, parents_list_0

        # --- Prepare for Scan (Steps 1 to N-1) ---
        positions_base = device_array(
            np.repeat(model_worker_batch.seq_lens[: model_worker_batch.real_bs], self.topk),
            sharding=(
                NamedSharding(self.model_runner.mesh, P()) if jax.process_count() == 1 else None
            ),
        )

        model_worker_batch.input_ids = np.empty(bs * self.topk, np.int32)
        model_worker_batch.positions = np.empty(bs * self.topk, np.int32)
        model_worker_batch.spec_info.hidden_states = hidden_states

        logits_metadata = LogitsMetadata.from_model_worker_batch(
            model_worker_batch, self.draft_model_runner.mesh
        )
        forward_batch = ForwardBatch.init_new(model_worker_batch, self.draft_model_runner)
        forward_batch.out_cache_loc = jnp.empty((1,))
        forward_batch.cache_loc = jnp.empty((1,))

        # Set initial values for Step 0 Forward (which happens in first scan iter)
        forward_batch.input_ids = input_ids
        forward_batch.spec_info.hidden_states = hidden_states
        forward_batch.positions = positions_base
        # Initialize speculative_step_id to int32 array to match scan output structure
        forward_batch.speculative_step_id = jnp.array(0, dtype=jnp.int32)

        # Include KV cache in carry to handle state updates functionally
        curr_kv_cache = self.draft_model_runner.token_to_kv_pool.get_all_buffers()
        carry = (forward_batch, scores, curr_kv_cache)

        def scan_fn(carry, step_idx):
            batch, curr_scores, kv_cache = carry

            # HACK: Temporarily update the global KV cache with the current tracer
            # This ensures model.forward uses the state from the previous iteration
            self.draft_model_runner.token_to_kv_pool.replace_kv_buffer(kv_cache)

            # 1. Forward (Step `step_idx`)
            batch.speculative_step_id = step_idx
            logits_output, _ = self.draft_model_runner.forward(
                batch,
                logits_metadata=logits_metadata,
            )

            # Retrieve updated KV cache (new tracers) produced by forward
            new_kv_cache = self.draft_model_runner.token_to_kv_pool.get_all_buffers()

            # 2. Process Logits
            next_topk_p, next_topk_index = topk_probs_from_logits(
                logits_output.next_token_logits[: bs * self.topk], self.topk
            )
            if self.hot_token_ids is not None:
                next_topk_index = self.hot_token_ids[next_topk_index]
            next_hidden = logits_output.hidden_states[: bs * self.topk * self.topk, :]

            # 3. Select (Step `step_idx + 1`)
            next_i = step_idx + 1
            next_ids, next_h, next_scores, tree_info = select_top_k_tokens_step_greater_0(
                jnp.asarray(next_i),
                next_topk_p,
                next_topk_index,
                next_hidden,
                curr_scores,
                self.topk,
            )

            # 4. Update Batch for next Forward
            new_positions = positions_base + next_i

            # Create a new batch state for the next iteration
            batch.input_ids = next_ids
            batch.positions = new_positions
            batch.spec_info.hidden_states = next_h

            return (batch, next_scores, new_kv_cache), tree_info

        # Run Scan over steps 0 to N-2
        # This will produce results for Select(1) to Select(N-1)
        steps = jnp.arange(self.speculative_num_steps - 1)
        final_carry, stacked_tree_info = jax.lax.scan(scan_fn, carry, steps)

        # Update the global KV cache with the final state from scan
        _, _, final_kv_cache = final_carry
        self.draft_model_runner.token_to_kv_pool.replace_kv_buffer(final_kv_cache)

        # --- Combine Results ---        # stacked_tree_info[0] (scores): (num_steps-1, bs, topk, topk)
        # stacked_tree_info[1] (tokens): (num_steps-1, bs, topk^2)
        # stacked_tree_info[2] (parents): (num_steps-1, bs, topk)

        scores_rest = stacked_tree_info[0]
        tokens_rest = stacked_tree_info[1]
        parents_rest = stacked_tree_info[2]

        # Reshape and concatenate
        # score_list: [step 0 (bs, 1, topk), steps 1..N-1 (bs, (N-1)*topk, topk)]
        # We need to transpose stacked results from (steps, bs, ...) to (bs, steps, ...) and flatten

        scores_rest = jnp.swapaxes(scores_rest, 0, 1).reshape(bs, -1, self.topk)
        score_list = jnp.concatenate([score_list_0, scores_rest], axis=1)

        tokens_rest = jnp.swapaxes(tokens_rest, 0, 1).reshape(bs, -1)
        token_list = jnp.concatenate([token_list_0, tokens_rest], axis=1)

        parents_rest = jnp.swapaxes(parents_rest, 0, 1).reshape(bs, -1)
        parents_list = jnp.concatenate([parents_list_0, parents_rest], axis=1)

        return score_list, token_list, parents_list

    def run_spec_decode_precompile(self):
        self.precompile_spec_extend()
        self.precompile_spec_decode()
        # FIXME precompile some kernel

    def precompile_spec_extend(self):
        start_time = time.perf_counter()
        logger.info(
            "[SPEC_EXTEND] Begin to precompile bs_paddings=%s token_paddings=%s",
            self.precompile_bs_paddings[-1:],
            self.precompile_token_paddings,
        )

        bs, _ = self.get_max_padded_size()
        pairs = list(itertools.product([bs], self.precompile_token_paddings))

        with tqdm(pairs, desc="[SPEC_EXTEND] PRECOMPILE", leave=False) as pbar:
            for pair in pbar:
                pair = list(pair)
                bs, num_tokens = pair[0], pair[1]
                pbar.set_postfix(bs=bs, tokens=num_tokens)
                if bs > num_tokens:
                    logger.warning("bs=%s > num_tokens=%s, skip this pair", bs, num_tokens)
                    continue
                model_worker_batch = self.generate_model_worker_batch(
                    bs,
                    num_tokens,
                    ForwardMode.EXTEND,
                    self.precompile_cache_loc_paddings[-1],
                    do_penalties=False,
                    speculative_algotithm=self.speculative_algorithm,
                )
                self.forward_batch_speculative_generation(model_worker_batch)
        end_time = time.perf_counter()
        logger.info("[SPEC_EXTEND] Precompile finished in %.0f secs", end_time - start_time)

    def precompile_spec_decode(self):
        start_time = time.perf_counter()
        logger.info(
            "[SPEC_DECODE] Begin to precompile bs_paddings=%s",
            self.precompile_bs_paddings,
        )

        with tqdm(
            self.precompile_bs_paddings, desc="[SPEC_DECODE] PRECOMPILE", leave=False
        ) as pbar:
            for bs in pbar:
                pbar.set_postfix(bs=bs)
                # use same page aligned with precompile cache_loc_paddings
                aligned_cache_loc_size = (
                    (bs * self.max_req_len + self.page_size - 1) // self.page_size * self.page_size
                )
                model_worker_batch = self.generate_model_worker_batch(
                    bs,
                    bs,
                    ForwardMode.DECODE,
                    aligned_cache_loc_size,
                    do_penalties=False,
                    speculative_algotithm=self.speculative_algorithm,
                )
                spec_info = EagleDraftInput(
                    # FIXME(pc) dtype should according to serverargs
                    topk_p=jnp.ones(
                        (bs, self.topk),
                        dtype=jnp.bfloat16 if self.server_args.dtype == "bfloat16" else jnp.float32,
                    ),
                    topk_index=jnp.ones((bs, self.topk), dtype=jnp.int32),
                    hidden_states=jnp.ones(
                        (bs, self.model_config.hidden_size),
                        dtype=jnp.bfloat16 if self.server_args.dtype == "bfloat16" else jnp.float32,
                    ),
                    verified_id=jnp.ones((bs,), dtype=jnp.int32),
                    accept_length=jnp.ones((bs,), dtype=jnp.int32),
                    capture_hidden_mode=CaptureHiddenMode.LAST,
                    allocate_lens=model_worker_batch.seq_lens
                    + EagleDraftInput.ALLOC_LEN_PER_DECODE,
                )
                model_worker_batch.capture_hidden_mode = CaptureHiddenMode.LAST
                model_worker_batch.spec_info = spec_info
                model_worker_batch.speculative_eagle_topk = self.topk
                model_worker_batch.speculative_num_draft_tokens = self.speculative_num_draft_tokens
                model_worker_batch.speculative_num_steps = self.speculative_num_steps
                self.forward_batch_speculative_generation(model_worker_batch)

        end_time = time.perf_counter()
        logger.info("[SPEC_DECODE] Precompile finished in %.0f secs", end_time - start_time)


@functools.partial(jax.jit, static_argnames=["topk"])
def topk_probs_from_logits(
    logits: jax.Array, topk: int, axis: int = -1
) -> tuple[jax.Array, jax.Array]:
    """Return top-k probabilities without materializing the full softmax tensor."""
    working_logits = jnp.moveaxis(logits, axis, -1) if axis != -1 else logits
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


# FIXME(pc) this should be jitted or convert as np.ndarray
def select_top_k_tokens(
    i: int | jax.Array,
    topk_p: jax.Array,
    topk_index: jax.Array,
    hidden_states: jax.Array,
    scores: jax.Array,
    topk: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    # Use jax.lax.cond to handle both static int (for step 0) and traced array (for scan)
    return jax.lax.cond(
        i == 0,
        lambda _: select_top_k_tokens_step_0(topk_p, topk_index, hidden_states, scores, topk),
        lambda _: select_top_k_tokens_step_greater_0(
            jnp.asarray(i), topk_p, topk_index, hidden_states, scores, topk
        ),
        None,
    )


@functools.partial(jax.jit, static_argnames=["topk"])
def select_top_k_tokens_step_0(
    topk_p: jax.Array,
    topk_index: jax.Array,
    hidden_states: jax.Array,
    scores: jax.Array,
    topk: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
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
    return input_ids, hidden_states, scores, tree_info


@functools.partial(jax.jit, static_argnames=["topk"])
def select_top_k_tokens_step_greater_0(
    i: jax.Array,
    topk_p: jax.Array,
    topk_index: jax.Array,
    hidden_states: jax.Array,
    scores: jax.Array,
    topk: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
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
