import logging
import time

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from sgl_jax.srt.layers.logits_processor import LogitsProcessorOutput
from sgl_jax.srt.layers.sampler import get_token_ids_logprobs, get_top_logprobs
from sgl_jax.srt.managers.schedule_batch import ModelWorkerBatch, ScheduleBatch
from sgl_jax.srt.managers.scheduler import GenerationBatchResult
from sgl_jax.srt.managers.tp_worker import ModelWorker
from sgl_jax.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode
from sgl_jax.srt.sampling.sampling_batch_info import SamplingMetadata
from sgl_jax.srt.speculative.base_worker import BaseSpecWorker
from sgl_jax.srt.speculative.eagle_draft_worker import (
    EagleDraftWorker,
    _take_with_optional_out_sharding,
)
from sgl_jax.srt.speculative.eagle_util import (
    EagleDraftInput,
    EagleVerifyInput,
    EagleVerifyOutput,
)
from sgl_jax.srt.utils.common_utils import get_bool_env_var

logger = logging.getLogger(__name__)
RETURN_ORIGINAL_LOGPROB = get_bool_env_var("RETURN_ORIGINAL_LOGPROB")


class EAGLEWorker(BaseSpecWorker):
    """Standard EAGLE speculative decode orchestrator.

    Composes a ``target_worker`` (full model) with an ``EagleDraftWorker``
    (draft model).  Implements the ``BaseSpecWorker`` contract so the
    scheduler interface is unchanged.
    """

    def __init__(self, server_args, target_worker: ModelWorker):
        self.server_args = server_args
        self._target_worker = target_worker
        self._draft_worker = EagleDraftWorker(server_args, target_worker)

        self.topk = server_args.speculative_eagle_topk
        self.speculative_num_steps = server_args.speculative_num_steps
        self.speculative_num_draft_tokens = server_args.speculative_num_draft_tokens
        self.page_size = server_args.page_size
        self.mesh = target_worker.mesh

        from sgl_jax.srt.speculative.spec_info import SpeculativeAlgorithm

        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )

        self.req_to_token_pool, self.token_to_kv_pool_allocator = target_worker.get_memory_pool()

        (
            precompile_token_paddings,
            precompile_bs_paddings,
            precompile_cache_loc_paddings,
        ) = target_worker.get_precompile_paddings()
        self.precompile_bs_paddings = precompile_bs_paddings
        self.precompile_cache_loc_paddings = precompile_cache_loc_paddings
        self.precompile_token_paddings = precompile_token_paddings

    # -- BaseSpecWorker interface --

    @property
    def target_worker(self) -> ModelWorker:
        return self._target_worker

    @property
    def draft_worker(self) -> EagleDraftWorker:
        return self._draft_worker

    def forward_batch_speculative_generation(
        self,
        model_worker_batch: ModelWorkerBatch,
    ):
        if model_worker_batch.forward_mode.is_extend():
            # FIXME(pc) add padding logic here
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
            # target extend
            logits_output, next_token_ids, cache_miss_count, bid, seq_lens = (
                self.forward_target_extend(model_worker_batch, sampling_metadata)
            )
            # draft extend for Update Draft State
            self.draft_worker.draft_extend_for_prefill(
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
            self.draft_worker.draft(model_worker_batch)

            batch_output = self.verify(model_worker_batch, cur_allocate_lens)

            self.draft_worker.draft_extend_for_decode(model_worker_batch, batch_output)

            return batch_output

    # -- Target model methods --

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

    # -- Verify --

    def verify(self, model_worker_batch: ModelWorkerBatch, cur_allocate_lens: jax.Array):
        spec_info: EagleVerifyInput = model_worker_batch.spec_info
        spec_info.allocate_lens = cur_allocate_lens
        # Pad ``out_cache_loc`` to the bucketed ``bs * num_draft_tokens`` shape so
        # the verify forward sees the same pytree leaf size at runtime as it did
        # during precompile (cache-miss fix; see eagle_draft_worker.pad_out_cache_loc_for_verify).
        self.draft_worker.pad_out_cache_loc_for_verify(model_worker_batch)
        spec_info.prepare_for_verify(model_worker_batch, self.page_size, self.target_worker)
        forward_metadata = self.target_worker.model_runner.attn_backend.get_eagle_forward_metadata(
            model_worker_batch
        )

        logits_output, _, cache_miss_count = self.target_worker.forward_batch_generation(
            model_worker_batch, skip_sample=True, forward_metadata=forward_metadata
        )
        # Keep ``logits_output`` arrays at the JIT-output sharding (typically
        # ``Explicit('data', 'tensor')``) instead of replicate_to_mesh-ing them.
        # The downstream ``draft_extend_for_decode`` JIT call's cache key
        # depends on ``forward_batch.spec_info.hidden_states`` sharding;
        # matching the EXTEND-path sharding (also Explicit) keeps a single
        # cache entry per (bs, mode) instead of two.
        spec_info.hidden_states = logits_output.hidden_states
        (
            predict,
            verified_id,
            accept_length,
            accept_index,
        ) = spec_info.sample(
            model_worker_batch,
            logits_output,
            self.draft_worker.draft_model_runner.rngs,
            self.mesh,
        )
        # accept_index uses -1 for rejected slots; gathering with -1 picks the
        # global last element, so dext later writes rejected tokens' draft-KV at
        # a foreign position inside each req's page (corrupts prefix KV for all
        # but the last req at bs>1). Redirect -1 to each req's own last slot.
        # accept_index has length bs*(spec_steps+1); the gathered tensors have
        # length bs*draft_token_num — equal at topk=1, distinct at topk>1.
        draft_n = self.speculative_num_draft_tokens
        accept_width = self.speculative_num_steps + 1
        req_ids = np.arange(len(accept_index)) // accept_width
        per_req_last = req_ids * draft_n + draft_n - 1
        safe_index = np.where(accept_index >= 0, accept_index, per_req_last)
        # Gather via ``_take_with_optional_out_sharding`` so the gather kernel's
        # output sharding tracks the source array's sharding (cache-miss fix).
        logits_output.next_token_logits = _take_with_optional_out_sharding(
            logits_output.next_token_logits, safe_index, trailing_slice=True
        )
        logits_output.hidden_states = _take_with_optional_out_sharding(
            logits_output.hidden_states, safe_index, trailing_slice=True
        )
        model_worker_batch.positions = _take_with_optional_out_sharding(
            model_worker_batch.positions, safe_index
        )
        new_seq_lens = model_worker_batch.seq_lens + accept_length
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

    # -- Logprob post-processing --

    def add_logprob_values(
        self,
        batch: ScheduleBatch,
        res: EagleVerifyOutput,
        logits_output: LogitsProcessorOutput,
    ):
        logits_output = res.logits_output
        top_logprobs_nums = batch.top_logprobs_nums
        token_ids_logprobs = batch.token_ids_logprobs
        accepted_indices = res.accepted_indices
        assert len(accepted_indices) == len(logits_output.next_token_logits)

        temperatures = batch.sampling_info.temperatures
        num_draft_tokens = batch.spec_info.draft_token_num
        temperatures = temperatures[accepted_indices // num_draft_tokens]
        if RETURN_ORIGINAL_LOGPROB:
            logprobs = jax.nn.log_softmax(logits_output.next_token_logits, axis=-1)
        else:
            logprobs = jax.nn.log_softmax(logits_output.next_token_logits / temperatures, axis=-1)
        batch_next_token_ids = res.verified_id
        num_tokens_per_req = [accept + 1 for accept in res.accept_length_per_req_cpu]

        top_logprobs_nums_repeat_interleaved = []
        token_ids_logprobs_repeat_interleaved = []
        for num, num_tokens in zip(top_logprobs_nums, num_tokens_per_req):
            top_logprobs_nums_repeat_interleaved.extend([num] * num_tokens)
        for token_ids, num_tokens in zip(token_ids_logprobs, num_tokens_per_req):
            token_ids_logprobs_repeat_interleaved.extend([token_ids] * num_tokens)

        if any(x > 0 for x in top_logprobs_nums):
            (
                logits_output.next_token_top_logprobs_val,
                logits_output.next_token_top_logprobs_idx,
            ) = get_top_logprobs(
                logprobs,
                top_logprobs_nums_repeat_interleaved,
            )

        if any(x is not None for x in token_ids_logprobs):
            logits_output.next_token_token_ids_logprobs_val = get_token_ids_logprobs(
                logprobs,
                token_ids_logprobs_repeat_interleaved,
                None,
            )
            logits_output.next_token_token_ids_logprobs_idx = None

        logits_output.next_token_logprobs = logprobs[
            jnp.arange(len(batch_next_token_ids), device=batch.sampling_info.device),
            batch_next_token_ids,
        ]

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

    # -- Precompilation --

    def run_spec_decode_precompile(self):
        self.precompile_spec_extend()
        self.precompile_spec_decode()
        self.draft_worker.precompile_runtime_jax_helpers()

    def precompile_spec_extend(self):
        start_time = time.perf_counter()
        real_bs_candidates = self.draft_worker._get_phase1_runtime_bs_candidates()
        precompile_pairs = []
        for num_tokens in self.precompile_token_paddings:
            for real_bs in real_bs_candidates:
                if num_tokens % real_bs != 0:
                    continue
                precompile_pairs.append(
                    (
                        self.draft_worker._get_padding_bs_for_real_bs(real_bs),
                        real_bs,
                        num_tokens,
                    )
                )

        logger.info(
            "[SPEC_EXTEND] Begin to precompile bs_paddings=%s real_bs=%s token_paddings=%s",
            sorted(set(padded_bs for padded_bs, _, _ in precompile_pairs)),
            real_bs_candidates,
            self.precompile_token_paddings,
        )

        cache_loc_by_bs = dict(zip(self.precompile_bs_paddings, self.precompile_cache_loc_paddings))

        with tqdm(precompile_pairs, desc="[SPEC_EXTEND] PRECOMPILE", leave=False) as pbar:
            for padded_bs, real_bs, num_tokens in pbar:
                pbar.set_postfix(bs=padded_bs, real_bs=real_bs, tokens=num_tokens)
                tokens_per_req = num_tokens // real_bs
                if tokens_per_req <= 0:
                    continue
                if padded_bs > num_tokens:
                    logger.warning(
                        "bs=%s > num_tokens=%s, skip this pair",
                        padded_bs,
                        num_tokens,
                    )
                    continue
                model_worker_batch = self.draft_worker.compilation_manager._make_dummy_batch(
                    padded_bs,
                    num_tokens,
                    ForwardMode.EXTEND,
                    cache_loc_by_bs[padded_bs],
                    speculative_algorithm=self.speculative_algorithm,
                    dp_size=1,
                    per_dp_bs_size=padded_bs,
                )
                # Override the placeholder layout from _make_dummy_batch with a
                # real_bs-shaped layout so the EAGLE prefill path exercises the
                # same gather/extend kernels as a real batch.
                model_worker_batch.return_output_logprob_only = False
                model_worker_batch.real_bs = real_bs
                model_worker_batch.real_input_ids_len = num_tokens
                model_worker_batch.seq_lens = np.zeros(padded_bs, dtype=np.int32)
                model_worker_batch.seq_lens[:real_bs] = tokens_per_req
                model_worker_batch.req_pool_indices = np.full(padded_bs, -1, dtype=np.int32)
                model_worker_batch.req_pool_indices[:real_bs] = np.arange(real_bs, dtype=np.int32)
                model_worker_batch.extend_seq_lens = np.zeros(padded_bs, dtype=np.int32)
                model_worker_batch.extend_seq_lens[:real_bs] = tokens_per_req
                model_worker_batch.extend_prefix_lens = np.zeros(padded_bs, dtype=np.int32)
                model_worker_batch.logits_indices = (
                    np.cumsum(model_worker_batch.extend_seq_lens, dtype=np.int32) - 1
                )
                model_worker_batch.logits_indices[real_bs:] = 0
                model_worker_batch.positions = np.arange(num_tokens, dtype=np.int32)
                model_worker_batch.out_cache_loc = np.arange(1, num_tokens + 1, dtype=np.int32)
                self.forward_batch_speculative_generation(model_worker_batch)
                jax.device_get(model_worker_batch.input_ids)
        end_time = time.perf_counter()
        logger.info("[SPEC_EXTEND] Precompile finished in %.0f secs", end_time - start_time)

    def precompile_spec_decode(self):
        start_time = time.perf_counter()
        max_bs = max(self.precompile_bs_paddings) if self.precompile_bs_paddings else 0
        runtime_bs_candidates = list(range(1, min(max_bs, 16) + 1))
        decode_bs_candidates = sorted(
            set(runtime_bs_candidates + list(self.precompile_bs_paddings))
        )
        logger.info(
            "[SPEC_DECODE] Begin to precompile bs_paddings=%s",
            decode_bs_candidates,
        )

        with tqdm(decode_bs_candidates, desc="[SPEC_DECODE] PRECOMPILE", leave=False) as pbar:
            for bs in pbar:
                pbar.set_postfix(bs=bs)
                aligned_cache_loc_size = (
                    (bs * self.draft_worker.max_req_len + self.page_size - 1)
                    // self.page_size
                    * self.page_size
                )
                model_worker_batch = self.draft_worker.compilation_manager._make_dummy_batch(
                    bs,
                    bs,
                    ForwardMode.DECODE,
                    aligned_cache_loc_size,
                    speculative_algorithm=self.speculative_algorithm,
                    dp_size=1,
                    per_dp_bs_size=bs,
                )
                spec_info = EagleDraftInput(
                    # FIXME(pc) dtype should according to serverargs
                    topk_p=jnp.ones(
                        (bs, self.topk),
                        dtype=jnp.bfloat16 if self.server_args.dtype == "bfloat16" else jnp.float32,
                    ),
                    topk_index=jnp.ones((bs, self.topk), dtype=jnp.int32),
                    hidden_states=jnp.ones(
                        (bs, self.draft_worker.model_config.hidden_size),
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
