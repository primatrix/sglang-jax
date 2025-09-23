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


def get_last_loc_jax_array(
    req_to_token: jax.Array,
    req_pool_indices: jax.Array,
    prefix_lens: jax.Array,
) -> jax.Array:
    """JAX version of get_last_loc that operates on JAX arrays.

    Args:
        req_to_token: Token mapping tensor of shape (num_reqs, max_seq_len)
        req_pool_indices: Request pool indices of shape (batch_size,)
        prefix_lens: Prefix lengths of shape (batch_size,)

    Returns:
        Last location tensor of shape (batch_size,)
    """
    return jnp.where(
        prefix_lens > 0,
        req_to_token[req_pool_indices, prefix_lens - 1],
        jnp.full_like(prefix_lens, -1),
    )


def get_last_loc_large_page_size_top_k_1(
    req_to_token: jax.Array,
    req_pool_indices: jax.Array,
    seq_lens: jax.Array,
    speculative_num_steps: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """JAX implementation of get_last_loc_large_page_size_top_k_1.

    This function is used in EAGLE speculative decoding to compute cache locations
    for large page sizes when top_k=1.

    Args:
        req_to_token: Request to token mapping tensor
        req_pool_indices: Request pool indices
        seq_lens: Current sequence lengths
        speculative_num_steps: Number of speculative decoding steps

    Returns:
        tuple of (prefix_lens, new_seq_lens, last_loc):
        - prefix_lens: Same as input seq_lens
        - new_seq_lens: Updated sequence lengths (prefix_lens + speculative_num_steps)
        - last_loc: Last cache locations computed using get_last_loc
    """
    prefix_lens = seq_lens
    new_seq_lens = prefix_lens + speculative_num_steps
    last_loc = get_last_loc_jax_array(
        req_to_token,
        req_pool_indices,
        prefix_lens,
    )
    return prefix_lens, new_seq_lens, last_loc


def get_last_loc_large_page_size_large_top_k(
    req_to_token: jax.Array,
    req_pool_indices: jax.Array,
    seq_lens: jax.Array,
    speculative_num_steps: int,
    topk: int,
    page_size: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """JAX implementation of get_last_loc_large_page_size_large_top_k.

    This function handles large page sizes with large top_k values in EAGLE speculative decoding.
    It computes cache locations and manages page allocation for multiple top-k branches.

    Args:
        req_to_token: Request to token mapping tensor
        req_pool_indices: Request pool indices
        seq_lens: Current sequence lengths
        speculative_num_steps: Number of speculative decoding steps
        topk: Number of top-k branches
        page_size: Size of each memory page

    Returns:
        tuple of (prefix_lens, new_seq_lens, last_loc, num_new_pages_per_topk, extend_lens):
        - prefix_lens: Same as input seq_lens
        - new_seq_lens: Updated sequence lengths considering page alignment
        - last_loc: Last cache locations
        - num_new_pages_per_topk: Number of new pages needed per top-k branch
        - extend_lens: Number of tokens to extend for each sequence
    """
    prefix_lens = seq_lens
    last_page_lens = prefix_lens % page_size
    num_new_pages_per_topk = (
        last_page_lens + speculative_num_steps + page_size - 1
    ) // page_size

    new_seq_lens = prefix_lens // page_size * page_size + num_new_pages_per_topk * (
        page_size * topk
    )
    extend_lens = new_seq_lens - prefix_lens

    last_loc = get_last_loc_jax_array(
        req_to_token,
        req_pool_indices,
        prefix_lens,
    )

    return prefix_lens, new_seq_lens, last_loc, num_new_pages_per_topk, extend_lens


def build_tree_kernel_efficient_preprocess(
    verified_id: jax.Array,
    score_list: List[jax.Array],
    token_list: List[jax.Array],
    parents_list: List[jax.Array],
    num_verify_tokens: int,
):
    """JAX implementation of build_tree_kernel_efficient_preprocess.

    This function matches the PyTorch preprocessing logic exactly.
    """
    logger.info(f"-------------verified_id------------{verified_id.shape}")
    logger.info(f"-------------score_list------------{score_list.shape}")
    logger.info(f"-------------token_list------------{token_list.shape}")
    logger.info(f"-------------parents_list------------{parents_list.shape}")
    logger.info(f"-------------num_verify_tokens------------{num_verify_tokens}")
    # Concatenate score_list along dim=1 and flatten from dim=1 onwards
    # b, n, topk; n = 1 + (num_steps-1) * self.topk
    score_tensor = jnp.concatenate(score_list, axis=1)
    batch_size = score_tensor.shape[0]
    score_tensor = score_tensor.reshape(batch_size, -1)

    # Concatenate token lists: b, (self.topk + (num_steps-1) * self.topk)
    ss_token_list = jnp.concatenate(token_list, axis=1)

    # Get top scores and indices
    _, top_scores_index = jax.lax.top_k(score_tensor, num_verify_tokens - 1)
    top_scores_index = jnp.sort(top_scores_index, axis=-1)

    # Gather draft tokens using the top indices
    draft_tokens = jnp.take_along_axis(ss_token_list, top_scores_index, axis=1)
    logger.info(
        f"-------------draft_tokens------------{verified_id.shape} {score_tensor.shape} {ss_token_list.shape} {draft_tokens.shape}"
    )
    assert draft_tokens.shape == (batch_size, verified_id.shape[0])
    draft_tokens = jnp.concatenate(
        [jnp.expand_dims(verified_id, axis=1), draft_tokens], axis=1
    ).flatten()

    # Build parent list
    if len(parents_list) > 1:
        parent_list = jnp.concatenate(parents_list[:-1], axis=1)
    else:
        batch_size = parents_list[0].shape[0]
        parent_list = jnp.empty((batch_size, 0), dtype=jnp.int32)

    return parent_list, top_scores_index, draft_tokens


def build_tree_kernel_efficient(
    verified_id: jax.Array,
    score_list: List[jax.Array],
    token_list: List[jax.Array],
    parents_list: List[jax.Array],
    seq_lens: jax.Array,
    seq_lens_sum: int,
    topk: int,
    spec_steps: int,
    num_verify_tokens: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """JAX implementation of build_tree_kernel_efficient.

    This implementation reconstructs the EAGLE tree building algorithm to match
    the PyTorch version's exact output patterns for positions, retrive_next_token,
    and retrive_next_sibling arrays.

    Args:
        verified_id: Verified token IDs from previous step
        score_list: List of score tensors from draft model
        token_list: List of token tensors from draft model
        parents_list: List of parent index tensors
        seq_lens: Sequence lengths
        seq_lens_sum: Sum of sequence lengths
        topk: Number of top-k candidates
        spec_steps: Number of speculative steps
        num_verify_tokens: Number of tokens to verify

    Returns:
        tuple of (tree_mask, positions, retrive_index, retrive_next_token,
                 retrive_next_sibling, draft_tokens)
    """
    # Use the preprocessing function exactly like PyTorch version
    parent_list, top_scores_index, draft_tokens = (
        build_tree_kernel_efficient_preprocess(
            verified_id, score_list, token_list, parents_list, num_verify_tokens
        )
    )

    # Get batch size
    bs = seq_lens.shape[0]

    # Create tree mask (simplified - full attention for now)
    total_tokens = seq_lens_sum + num_verify_tokens * bs
    tree_mask = jnp.ones((total_tokens,), dtype=jnp.bool_)

    positions, retrive_index, retrive_next_token, retrive_next_sibling = (
        build_eagle_tree_structure(
            parent_list=parent_list,
            selected_index=top_scores_index,
            verified_seq_len=seq_lens,
            bs=bs,
            draft_token_num=num_verify_tokens,
            topk=topk,
            depth=spec_steps,
            tree_mask_mode=0,  # FULL_MASK
        )
    )

    return (
        tree_mask,
        positions,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        draft_tokens,
    )


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


def build_eagle_tree_structure(
    parent_list: jax.Array,
    selected_index: jax.Array,
    verified_seq_len: jax.Array,
    bs: int,
    draft_token_num: int,
    topk: int,
    depth: int,
    tree_mask_mode: int = 0,  # FULL_MASK = 0
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """
    Args:
        parent_list: Parent indices array [bs, topk * (depth-1) + 1]
        selected_index: Selected token indices [bs, draft_token_num - 1]
        verified_seq_len: Sequence lengths [bs]
        bs: Batch size
        draft_token_num: Number of draft tokens
        topk: Top-k value
        depth: Tree depth
        tree_mask_mode: Tree mask mode (0=FULL_MASK)

    Returns:
        tuple of (positions, retrive_index, retrive_next_token, retrive_next_sibling)
    """

    # Initialize arrays
    positions = jnp.zeros((bs * draft_token_num,), dtype=jnp.int32)
    retrive_index = jnp.full((bs, draft_token_num), -1, dtype=jnp.int32)
    retrive_next_token = jnp.full((bs, draft_token_num), -1, dtype=jnp.int32)
    retrive_next_sibling = jnp.full((bs, draft_token_num), -1, dtype=jnp.int32)

    for bid in range(bs):
        seq_len = verified_seq_len[bid]

        # selected_index[bid * (draft_token_num - 1) + index]
        # parent_list[bid * (topk * (depth - 1) + 1) + parent_tb_idx]

        for tid in range(draft_token_num):
            global_token_idx = bid * draft_token_num + tid

            if tid == 0:
                # Verified token (tid == 0)
                positions = positions.at[global_token_idx].set(seq_len)
                retrive_index = retrive_index.at[bid, tid].set(global_token_idx)

                # Build retrive_next_token and retrive_next_sibling (backwards iteration)
                retrive_index_offset = bid * draft_token_num
                for i in range(
                    draft_token_num - 1, 0, -1
                ):  # i from draft_token_num-1 to 1
                    current_token_idx = retrive_index_offset + i
                    retrive_index = retrive_index.at[bid, i].set(current_token_idx)

                    selected_idx = bid * (draft_token_num - 1) + i - 1
                    parent_tb_idx = selected_index.flatten()[selected_idx] // topk
                    parent_position = 0

                    if parent_tb_idx > 0:
                        parent_list_idx = bid * (topk * (depth - 1) + 1) + parent_tb_idx
                        if parent_list_idx < parent_list.size:
                            parent_token_idx = parent_list.flatten()[parent_list_idx]

                            for parent_pos in range(draft_token_num - 1):
                                check_idx = bid * (draft_token_num - 1) + parent_pos
                                if (
                                    check_idx < selected_index.size
                                    and selected_index.flatten()[check_idx]
                                    == parent_token_idx
                                ):
                                    parent_position = (
                                        parent_pos + 1
                                    )  # +1 to convert to 1-indexed
                                    break
                            else:
                                parent_position = draft_token_num  # Not found
                        else:
                            parent_position = draft_token_num  # Invalid parent_list_idx
                    else:
                        parent_position = 0  # Root node

                    if parent_position >= draft_token_num:
                        # Invalid parent, skip
                        continue

                    next_token_idx = bid * draft_token_num + parent_position
                    if retrive_next_token.flatten()[next_token_idx] == -1:
                        retrive_next_token = retrive_next_token.at[
                            bid, parent_position
                        ].set(i)
                    else:
                        # There's already a next_token, so set sibling
                        origin_next_token = retrive_next_token.flatten()[next_token_idx]
                        retrive_next_token = retrive_next_token.at[
                            bid, parent_position
                        ].set(i)
                        retrive_next_sibling = retrive_next_sibling.at[bid, i].set(
                            origin_next_token
                        )

                retrive_index = retrive_index.at[bid, 0].set(bid * draft_token_num)

            else:
                # Draft token (tid > 0)
                # Calculate position by tracing back to root
                position = 0
                cur_position = tid - 1  # Convert to 0-indexed for selected_index

                while True:
                    position += 1
                    selected_idx = bid * (draft_token_num - 1) + cur_position
                    parent_tb_idx = selected_index.flatten()[selected_idx] // topk

                    if parent_tb_idx == 0:
                        # Reached root
                        break

                    parent_list_idx = bid * (topk * (depth - 1) + 1) + parent_tb_idx
                    if parent_list_idx < parent_list.size:
                        token_idx = parent_list.flatten()[parent_list_idx]

                        found = False
                        for cur_pos in range(draft_token_num - 1):
                            check_idx = bid * (draft_token_num - 1) + cur_pos
                            if (
                                check_idx < selected_index.size
                                and selected_index.flatten()[check_idx] == token_idx
                            ):
                                cur_position = cur_pos
                                found = True
                                break

                        if not found:
                            break  # Invalid tree structure
                    else:
                        break  # Invalid parent_list_idx

                positions = positions.at[global_token_idx].set(position + seq_len)
                retrive_index = retrive_index.at[bid, tid].set(global_token_idx)

    return positions, retrive_index, retrive_next_token, retrive_next_sibling
