from __future__ import annotations

import copy
import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Optional

import jax
import jax._src.sharding as sharding
import jax.numpy as jnp
import numpy
from flax import nnx
from jax._src.lib import xla_client as xc

from sgl_jax.srt.layers.logits_processor import LogitsProcessorOutput
from sgl_jax.srt.managers.schedule_batch import (
    ScheduleBatch,
    get_last_loc,
    global_server_args_dict,
)
from sgl_jax.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sgl_jax.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sgl_jax.srt.speculative.pallas.kernel import (
    align_evict_mask_to_page_size,
    assign_req_to_token_pool,
    filter_finished_cache_loc_kernel,
    get_target_cache_loc,
    top_k_renorm_prob,
    top_p_renorm_prob,
    tree_speculative_sampling_target_only,
    verify_tree_greedy,
)
from sgl_jax.srt.utils.common_utils import next_power_of_2

logger = logging.getLogger(__name__)

SIMULATE_ACC_LEN = os.environ.get("SIMULATE_ACC_LEN")
SIMULATE_ACC_METHOD = os.environ.get("SIMULATE_ACC_METHOD", "multinomial")


@dataclass
class EagleDraftInput:
    # The inputs for decode
    # shape: (b, topk)
    topk_p: jax.Array = None
    topk_index: jax.Array = None
    # shape: (b, hidden_size)
    hidden_states: jax.Array = None
    capture_hidden_mode: CaptureHiddenMode = CaptureHiddenMode.FULL

    # Inputs for extend
    # shape: (b,)
    verified_id: jax.Array = None
    accept_length: jax.Array = None
    accept_length_cpu: List[int] = None

    # Inputs for the attention backends
    # shape: (b + 1,)
    kv_indptr: jax.Array = None
    kv_indices: jax.Array = None

    # Shape info for padding
    num_tokens_per_batch: int = -1
    num_tokens_for_logprob_per_batch: int = -1

    # Inputs for draft extend
    # shape: (b,)
    seq_lens_for_draft_extend: jax.Array = None
    req_pool_indices_for_draft_extend: jax.Array = None

    def prepare_for_extend(self, batch: ScheduleBatch):

        if batch.forward_mode.is_idle():
            return

        # Prefill only generate 1 token.
        assert len(self.verified_id) == len(batch.seq_lens)

        pt = 0
        for i, extend_len in enumerate(batch.extend_lens):
            input_ids = batch.input_ids[pt : pt + extend_len]
            batch.input_ids[pt : pt + extend_len] = jnp.concatenate(
                (input_ids[1:], self.verified_id[i].reshape(1))
            )
            pt += extend_len

    @classmethod
    def create_idle_input(
        cls,
        device: xc.Device | sharding.Sharding | None,
        hidden_size: int,
        dtype: jnp.dtype,
        topk: int,
        capture_hidden_mode: CaptureHiddenMode,
    ):
        return cls(
            verified_id=jnp.empty((0,), device=device, dtype=jnp.int32),
            hidden_states=jnp.empty((0, hidden_size), device=device, dtype=dtype),
            topk_p=jnp.empty((0, topk), device=device, dtype=jnp.float32),
            topk_index=jnp.empty((0, topk), device=device, dtype=jnp.int64),
            capture_hidden_mode=capture_hidden_mode,
            accept_length=jnp.empty((0,), device=device, dtype=jnp.int32),
            accept_length_cpu=[],
        )

    def prepare_extend_after_decode(
        self,
        batch: ScheduleBatch,
        speculative_num_steps: int,
    ):
        pass

    def generate_attn_arg_prefill(
        self,
        req_pool_indices: jax.Array,
        paged_kernel_lens: jax.Array,
        paged_kernel_lens_sum: int,
        req_to_token: jax.Array,
    ):
        pass

    def filter_batch(self, new_indices: jax.Array, has_been_filtered: bool = True):
        pass

    def merge_batch(self, spec_info: Any):
        pass


@dataclass
class EagleVerifyOutput:
    # Draft input batch
    draft_input: EagleDraftInput
    # Logit outputs from target worker
    logits_output: LogitsProcessorOutput
    # Accepted token ids including the bonus token
    verified_id: jax.Array
    # Accepted token length per sequence in a batch in CPU.
    accept_length_per_req_cpu: List[int]
    # Accepted indices from logits_output.next_token_logits
    accepted_indices: jax.Array


@dataclass
class EagleVerifyInput:
    draft_token: jax.Array
    custom_mask: jax.Array
    positions: jax.Array
    retrive_index: jax.Array
    retrive_next_token: jax.Array
    retrive_next_sibling: jax.Array
    retrive_cum_len: jax.Array
    spec_steps: int
    topk: int
    draft_token_num: int
    capture_hidden_mode: CaptureHiddenMode
    seq_lens_sum: int
    seq_lens_cpu: jax.Array
    # grammar: BaseGrammarObject = None

    def prepare_for_verify(self, batch: ScheduleBatch, page_size: int):
        if batch.forward_mode.is_idle():
            return

        batch.input_ids = self.draft_token

        if page_size == 1:
            batch.out_cache_loc = batch.alloc_token_slots(len(batch.input_ids))
            end_offset = batch.seq_lens + self.draft_token_num
        else:
            prefix_lens = batch.seq_lens
            end_offset = prefix_lens + self.draft_token_num
            last_loc = get_last_loc(
                batch.req_to_token_pool.req_to_token,
                batch.req_pool_indices,
                prefix_lens,
            )
            batch.out_cache_loc = batch.alloc_paged_token_slots_extend(
                prefix_lens, end_offset, last_loc, len(batch.input_ids)
            )
            self.last_loc = last_loc

        bs = batch.batch_size()
        assign_req_to_token_pool[(bs,)](
            batch.req_pool_indices,
            batch.req_to_token_pool.req_to_token,
            batch.seq_lens,
            end_offset,
            batch.out_cache_loc,
            batch.req_to_token_pool.req_to_token.shape[1],
            next_power_of_2(bs),
        )

    def verify(
        self,
        batch: ScheduleBatch,
        logits_output: LogitsProcessorOutput,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        page_size: int,
        vocab_mask: Optional[jax.Array] = None,  # For grammar
    ) -> jax.Array:
        """
        Verify and find accepted tokens based on logits output and batch
        (which contains spec decoding information).

        WARNING: This API in-place modifies the states of logits_output

        This API updates values inside logits_output based on the accepted
        tokens. I.e., logits_output.next_token_logits only contains
        accepted token logits.
        """
        if batch.forward_mode.is_idle():
            return EagleVerifyOutput(
                draft_input=EagleDraftInput.create_idle_input(
                    device=batch.device,
                    hidden_size=batch.model_config.hidden_size,
                    dtype=batch.model_config.dtype,
                    topk=self.topk,
                    capture_hidden_mode=CaptureHiddenMode.LAST,
                ),
                logits_output=logits_output,
                verified_id=jnp.empty(0, dtype=jnp.long, device=batch.device),
                accept_length_per_req_cpu=[],
                accepted_indices=jnp.full(
                    (0, self.spec_steps + 1),
                    -1,
                    dtype=jnp.int32,
                    device=batch.device,
                ),
            )

        bs = self.retrive_index.shape[0]
        candidates = self.draft_token.reshape(bs, self.draft_token_num)
        sampling_info = batch.sampling_info

        predict_shape = list(logits_output.next_token_logits.shape)[:-1]
        predict_shape[-1] += 1
        predict = jnp.empty(predict_shape, dtype=jnp.int32)
        accept_index = jnp.full((bs, self.spec_steps + 1), -1, dtype=jnp.int32)
        accept_length = jnp.empty((bs,), dtype=jnp.int32)

        if bs != len(sampling_info):
            sampling_info = copy.deepcopy(sampling_info)
            # NOTE: retrive_index are the indices of the requests that are kept.
            sampling_info.filter_batch(self.retrive_index.tolist(), self.retrive_index)

        # TODO: support custom sampler, apply the custom logit processors if registered in the sampling info.
        # if sampling_info.has_custom_logit_processor:
        #     pass

        # TODO: Apply penalty
        # if sampling_info.penalizer_orchestrator.is_required:
        #     pass

        # TODO: Apply grammar mask
        # if vocab_mask is not None:
        #     pass

        # Sample tokens. Force greedy sampling on AMD
        is_all_greedy = sampling_info.is_all_greedy

        if is_all_greedy:
            target_predict = jnp.argmax(logits_output.next_token_logits, axis=-1)
            target_predict = target_predict.reshape(bs, self.draft_token_num)

            verify_tree_greedy(
                predicts=predict,  # mutable
                accept_index=accept_index,  # mutable
                accept_token_num=accept_length,  # mutable
                candidates=candidates,
                retrive_index=self.retrive_index,
                retrive_next_token=self.retrive_next_token,
                retrive_next_sibling=self.retrive_next_sibling,
                target_predict=target_predict,
            )
        else:
            # apply temperature and get target probs
            expanded_temperature = jnp.repeat(
                sampling_info.temperatures, self.draft_token_num
            )  # (bs * draft_token_num, 1)

            target_probs = jax.nn.softmax(
                logits_output.next_token_logits / expanded_temperature, axis=-1
            )  # (bs * draft_token_num, vocab_size)
            target_probs = top_k_renorm_prob(
                target_probs,
                jnp.repeat(sampling_info.top_ks, self.draft_token_num),
            )  # (bs * draft_token_num, vocab_size)
            if not jnp.all(sampling_info.top_ps == 1.0):
                target_probs = top_p_renorm_prob(
                    target_probs,
                    jnp.repeat(sampling_info.top_ps, self.draft_token_num),
                )
            target_probs = target_probs.reshape(bs, self.draft_token_num, -1)

            draft_probs = jnp.zeros(target_probs.shape, dtype=jnp.float32)

            # coins for rejection sampling
            coins = jax.random.uniform(candidates.shape, dtype=jnp.float32)
            # coins for final sampling
            coins_for_final_sampling = jax.random.uniform((bs,), dtype=jnp.float32)
            tree_speculative_sampling_target_only(
                predicts=predict,  # mutable
                accept_index=accept_index,  # mutable
                accept_token_num=accept_length,  # mutable
                candidates=candidates,
                retrive_index=self.retrive_index,
                retrive_next_token=self.retrive_next_token,
                retrive_next_sibling=self.retrive_next_sibling,
                uniform_samples=coins,
                uniform_samples_for_final_sampling=coins_for_final_sampling,
                target_probs=target_probs,
                draft_probs=draft_probs,
                threshold_single=global_server_args_dict[
                    "speculative_accept_threshold_single"
                ],
                threshold_acc=global_server_args_dict[
                    "speculative_accept_threshold_acc"
                ],
                deterministic=True,
            )

        if SIMULATE_ACC_LEN:
            # Do simulation
            accept_index, accept_length, predict = _generate_simulated_accept_index(
                accept_index=accept_index,
                predict=predict,
                accept_length=accept_length,
                simulate_acc_len=SIMULATE_ACC_LEN,
                bs=bs,
                spec_steps=self.spec_steps,
            )

        unfinished_index = []
        unfinished_accept_index = []
        accept_index_cpu = accept_index.tolist()
        predict_cpu = predict.tolist()
        has_finished = False

        # Iterate every accepted token and check if req has finished after append the token
        # should be checked BEFORE free kv cache slots
        for i, (req, accept_index_row) in enumerate(zip(batch.reqs, accept_index_cpu)):
            for j, idx in enumerate(accept_index_row):
                if idx == -1:
                    break
                id = predict_cpu[idx]
                req.output_ids.append(id)
                req.check_finished()
                if req.finished():
                    has_finished = True
                    # set all tokens after finished token to -1 and break
                    accept_index[i, j + 1 :] = -1
                    break
            if not req.finished():
                unfinished_index.append(i)
                if idx == -1:
                    unfinished_accept_index.append(accept_index[i, :j])
                else:
                    unfinished_accept_index.append(accept_index[i])
            req.spec_verify_ct += 1

        if has_finished:
            accept_length = (accept_index != -1).sum(dim=1) - 1

        # Free the KV cache for unaccepted tokens
        # TODO: fuse them
        accept_index = accept_index[accept_index != -1]
        verified_id = predict[accept_index]
        evict_mask = jnp.full_like(self.draft_token, True, dtype=jnp.bool)
        evict_mask[accept_index] = False

        if page_size == 1:
            # TODO: boolean array index leads to a device sync. Remove it.
            token_to_kv_pool_allocator.free(batch.out_cache_loc[evict_mask])
        else:
            if self.topk == 1:
                # Only evict full empty page. Do not evict partial empty page
                align_evict_mask_to_page_size(
                    batch.seq_lens,
                    evict_mask,
                    page_size,
                    self.draft_token_num,
                    next_power_of_2(self.draft_token_num),
                )
                token_to_kv_pool_allocator.free(batch.out_cache_loc[evict_mask])
            else:
                # Shift the accepted tokens to the beginning.
                # Only evict the last part
                src_cache_loc, tgt_cache_loc, to_free_num_slots = get_src_tgt_cache_loc(
                    batch.seq_lens,
                    batch.out_cache_loc,
                    accept_index,
                    accept_length,
                    self.draft_token_num,
                    page_size,
                )
                to_free_slots = jnp.empty(
                    (to_free_num_slots.sum().item(),),
                    dtype=jnp.int64,
                    device=to_free_num_slots.device,
                )

                # out_cache_loc: [0  1  2,  3  4  5,  6  7  8]
                # accept_index:  [0 -1  2,  3  4 -1,  6 -1 -1]
                # tgt_cache_loc: [0  1   ,  3  4   ,  6      ]
                # to_free_slots: [      2,        5,     7  8]
                # to_free_slots also needs to be page-aligned without the first partial page
                #
                # split each row of out_cache_loc into two parts.
                # 1. the first part goes to tgt_cache_loc. length = accept_length[i] + 1
                # 2. the second part goes to to_free_slots.
                get_target_cache_loc(
                    tgt_cache_loc,
                    to_free_slots,
                    accept_length,
                    to_free_num_slots,
                    batch.out_cache_loc,
                    self.draft_token_num,
                    next_power_of_2(self.draft_token_num),
                    next_power_of_2(bs),
                )

                # Free the kv cache
                token_to_kv_pool_allocator.free(to_free_slots)

                # Copy the kv cache
                batch.token_to_kv_pool_allocator.get_kvcache().move_kv_cache(
                    tgt_cache_loc, src_cache_loc
                )

        # Construct EagleVerifyOutput
        if not has_finished:
            if page_size == 1 or self.topk == 1:
                batch.out_cache_loc = batch.out_cache_loc[accept_index]
                assign_req_to_token_pool[(bs,)](
                    batch.req_pool_indices,
                    batch.req_to_token_pool.req_to_token,
                    batch.seq_lens,
                    batch.seq_lens + accept_length + 1,
                    batch.out_cache_loc,
                    batch.req_to_token_pool.req_to_token.shape[1],
                    next_power_of_2(bs),
                )
            else:
                batch.out_cache_loc = tgt_cache_loc
            batch.seq_lens.add_(accept_length + 1)

            draft_input = EagleDraftInput(
                hidden_states=batch.spec_info.hidden_states[accept_index],
                verified_id=verified_id,
                accept_length=accept_length,
                accept_length_cpu=accept_length.tolist(),
                seq_lens_for_draft_extend=batch.seq_lens,
                req_pool_indices_for_draft_extend=batch.req_pool_indices,
            )

            return EagleVerifyOutput(
                draft_input=draft_input,
                logits_output=logits_output,
                verified_id=verified_id,
                accept_length_per_req_cpu=draft_input.accept_length_cpu,
                accepted_indices=accept_index,
            )
        else:
            if page_size == 1 or self.topk == 1:
                assign_req_to_token_pool[(bs,)](
                    batch.req_pool_indices,
                    batch.req_to_token_pool.req_to_token,
                    batch.seq_lens,
                    batch.seq_lens + accept_length + 1,
                    batch.out_cache_loc[accept_index],
                    batch.req_to_token_pool.req_to_token.shape[1],
                    next_power_of_2(bs),
                )
                batch.seq_lens.add_(accept_length + 1)

            accept_length_cpu = accept_length.tolist()
            if len(unfinished_accept_index) > 0:
                unfinished_accept_index = jnp.concatenate(unfinished_accept_index)
                unfinished_index_device = jnp.array(unfinished_index, dtype=jnp.int64)
                draft_input_accept_length_cpu = [
                    accept_length_cpu[i] for i in unfinished_index
                ]
                if page_size == 1 or self.topk == 1:
                    batch.out_cache_loc = batch.out_cache_loc[unfinished_accept_index]
                else:
                    batch.out_cache_loc = jnp.empty(
                        len(unfinished_index) + sum(draft_input_accept_length_cpu),
                        dtype=jnp.int64,
                    )
                    accept_length_filter = create_accept_length_filter(
                        accept_length,
                        unfinished_index_device,
                        batch.seq_lens,
                    )
                    filter_finished_cache_loc_kernel[(bs,)](
                        batch.out_cache_loc,
                        tgt_cache_loc,
                        accept_length,
                        accept_length_filter,
                        next_power_of_2(bs),
                        next_power_of_2(self.draft_token_num),
                    )

                draft_input = EagleDraftInput(
                    hidden_states=batch.spec_info.hidden_states[
                        unfinished_accept_index
                    ],
                    verified_id=predict[unfinished_accept_index],
                    accept_length_cpu=draft_input_accept_length_cpu,
                    accept_length=accept_length[unfinished_index_device],
                    seq_lens_for_draft_extend=batch.seq_lens[unfinished_index_device],
                    req_pool_indices_for_draft_extend=batch.req_pool_indices[
                        unfinished_index_device
                    ],
                )
            else:
                draft_input = EagleDraftInput.create_idle_input(
                    device=batch.device,
                    hidden_size=batch.model_config.hidden_size,
                    dtype=batch.model_config.dtype,
                    topk=self.topk,
                    capture_hidden_mode=CaptureHiddenMode.LAST,
                )

            return EagleVerifyOutput(
                draft_input=draft_input,
                logits_output=logits_output,
                verified_id=verified_id,
                accept_length_per_req_cpu=accept_length_cpu,
                accepted_indices=accept_index,
            )


def _generate_simulated_accept_index(
    accept_index: jax.Array,
    predict,
    accept_length,
    simulate_acc_len,
    bs,
    spec_steps,
    rng: nnx.Rngs,
):
    simulate_acc_len_float = float(simulate_acc_len)
    if SIMULATE_ACC_METHOD == "multinomial":
        # here data is on cpu
        simulated_values = numpy.random.normal(
            loc=simulate_acc_len_float,
            scale=1.0,
            size=(1,),
        )
        # clamp simulated values to be between 1 and self.spec_steps
        simulated_values = jnp.clip(simulated_values, min=1.0, max=spec_steps + 1)
        simulate_acc_len = int(simulated_values.round().item())
    elif SIMULATE_ACC_METHOD == "match-expected":
        # multinomial sampling does not match the expected length
        # we keep it for the sake of compatibility of existing tests
        # but it's better to use "match-expected" for the cases that need to
        # match the expected length, One caveat is that this will only sample
        # either round down or round up of the expected length
        simulate_acc_len_float = max(1.0, min(spec_steps + 1, simulate_acc_len_float))
        lower = int(simulate_acc_len_float // 1)
        upper = lower + 1 if lower < spec_steps + 1 else lower
        if lower == upper:
            simulate_acc_len = lower
        else:
            weight_upper = simulate_acc_len_float - lower
            weight_lower = 1.0 - weight_upper
            # here, data is on cpu
            probs = numpy.array([weight_lower, weight_upper])
            sampled_index = jax.random.multinomial(rng, probs, shape=(1,))
            simulate_acc_len = lower if sampled_index == 0 else upper
    else:
        raise ValueError(f"Invalid simulate_acc_method: {SIMULATE_ACC_METHOD}")

    accept_indx_first_col = accept_index[:, 0].view(-1, 1)
    sim_accept_index = jnp.full((bs, spec_steps + 1), -1, dtype=jnp.int32)
    sim_accept_index[:, :simulate_acc_len] = accept_indx_first_col + jnp.arange(
        simulate_acc_len
    )
    accept_length = accept_length.at[:].set(simulate_acc_len - 1)
    predict = predict.at[:].set(100)  # some legit token id
    return sim_accept_index, accept_length, predict


def get_src_tgt_cache_loc(
    seq_lens: jax.Array,
    out_cache_loc: jax.Array,
    accept_index: jax.Array,
    accept_length: jax.Array,
    draft_token_num: int,
    page_size: int,
):
    src_cache_loc = out_cache_loc[accept_index]
    tgt_cache_loc = jnp.empty_like(src_cache_loc)
    extended_len = seq_lens + draft_token_num
    keep_len = jnp.minimum(
        (seq_lens + accept_length + 1 + page_size - 1) // page_size * page_size,
        extended_len,
    )
    to_free_num_slots = extended_len - keep_len
    return src_cache_loc, tgt_cache_loc, to_free_num_slots


def create_accept_length_filter(
    accept_length: jax.Array,
    unfinished_index_device: jax.Array,
    seq_lens: jax.Array,
):
    accept_length_filter = jnp.zeros_like(accept_length)
    accept_length_filter[unfinished_index_device] = (
        accept_length[unfinished_index_device] + 1
    )
    seq_lens.add_(accept_length + 1)
    return accept_length_filter
