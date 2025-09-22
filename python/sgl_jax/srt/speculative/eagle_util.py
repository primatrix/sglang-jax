from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from sgl_jax.srt.layers.logits_processor import LogitsProcessorOutput
    from sgl_jax.srt.managers.schedule_batch import ScheduleBatch
    from sgl_jax.srt.model_executor.forward_batch_info import CaptureHiddenMode

logger = logging.getLogger(__name__)


@dataclass
class EagleDraftInput:
    # The inputs for decode
    # shape: (b, topk)
    topk_p: jax.Array = None
    topk_index: jax.Array = None
    # shape: (b, hidden_size)
    hidden_states: jax.Array = None
    capture_hidden_mode: "CaptureHiddenMode" = 2  # CaptureHiddenMode.FULL

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
        if has_been_filtered:
            # in eagle_utils.py:verify, we have already filtered the batch by `unfinished_index`
            # therefore, we don't need to filter the batch again in scheduler
            if len(new_indices) != len(self.topk_p):
                logger.warning(
                    f"length of new_indices: {len(new_indices)} != length of topk_p: {len(self.topk_p)}, this should not happen"
                )
            self.topk_p = self.topk_p[: len(new_indices)]
            self.topk_index = self.topk_index[: len(new_indices)]
            self.hidden_states = self.hidden_states[: len(new_indices)]
            self.verified_id = self.verified_id[: len(new_indices)]
        else:
            # in some cases(e.g draft_extend), we have not filtered the batch by `unfinished_index`
            self.topk_p = self.topk_p[new_indices]
            self.topk_index = self.topk_index[new_indices]
            self.hidden_states = self.hidden_states[new_indices]
            self.verified_id = self.verified_id[new_indices]

    def merge_batch(self, spec_info: EagleDraftInput):
        if self.hidden_states is None:
            self.hidden_states = spec_info.hidden_states
            self.verified_id = spec_info.verified_id
            self.topk_p = spec_info.topk_p
            self.topk_index = spec_info.topk_index
            return
        if spec_info.hidden_states is None:
            return
        self.hidden_states = jnp.concatenate(
            [self.hidden_states, spec_info.hidden_states], axis=0
        )
        self.verified_id = jnp.concatenate(
            [self.verified_id, spec_info.verified_id], axis=0
        )
        self.topk_p = jnp.concatenate([self.topk_p, spec_info.topk_p])
        self.topk_index = jnp.concatenate([self.topk_index, spec_info.topk_index])


@dataclass
class EagleVerifyOutput:
    # Draft input batch
    draft_input: EagleDraftInput
    # Logit outputs from target worker
    logits_output: "LogitsProcessorOutput"
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
    capture_hidden_mode: "CaptureHiddenMode"
    seq_lens_sum: int
    seq_lens_cpu: jax.Array
    # grammar: BaseGrammarObject = None
