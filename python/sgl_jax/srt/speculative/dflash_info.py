from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax.tree_util import register_pytree_node_class

from sgl_jax.srt.model_executor.forward_batch_info import CaptureHiddenMode


def _mask_draft_kv_writes(
    cache_loc: jax.Array,
    accept_lens: jax.Array,
    active_mask: jax.Array,
) -> jax.Array:
    """Mask unaccepted and padded draft-KV writes inside jit_draft_extend."""
    tokens_per_row = cache_loc.shape[0] // accept_lens.shape[0]
    cache_rows = cache_loc.reshape((-1, tokens_per_row))
    accept_rows = accept_lens[:, None]
    active_rows = active_mask[:, None]
    token_offsets = jnp.arange(tokens_per_row, dtype=jnp.int32)[None, :]
    mesh = getattr(jax.typeof(cache_loc).sharding, "mesh", None)
    if mesh is not None and not getattr(mesh, "empty", False):
        from jax.sharding import NamedSharding
        from jax.sharding import PartitionSpec as P

        row_sharding = NamedSharding(mesh, P("data", None))
        replicated_2d = NamedSharding(mesh, P(None, None))
        cache_rows = jax.sharding.reshard(cache_rows, row_sharding)
        accept_rows = jax.sharding.reshard(accept_rows, row_sharding)
        active_rows = jax.sharding.reshard(active_rows, row_sharding)
        token_offsets = jax.sharding.reshard(token_offsets, replicated_2d)
    write_mask = active_rows & (token_offsets < accept_rows)
    masked_cache_loc = jnp.where(
        write_mask,
        cache_rows,
        jnp.int32(-1),
    ).reshape(-1)
    cache_sharding = jax.typeof(cache_loc).sharding
    if isinstance(cache_sharding, jax.sharding.NamedSharding) and not cache_sharding.mesh.empty:
        masked_cache_loc = jax.sharding.reshard(masked_cache_loc, cache_sharding)
    return masked_cache_loc


def build_dflash_draft_block(
    verified_id: np.ndarray | jax.Array,
    mask_token_id: int,
    target_prefix_lens: np.ndarray | jax.Array,
    block_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the fixed-size DFlash draft block inputs for one decode step.

    - block: [verified_id, mask_token_id, mask_token_id, ...]
    - position: [target_prefix_lens, target_prefix_lens + 1, ..., target_prefix_lens + block_size - 1]
    """
    verified_id = np.asarray(verified_id, dtype=np.int32)
    target_prefix_lens = np.asarray(target_prefix_lens, dtype=np.int32)
    if verified_id.ndim != 1:
        raise ValueError(f"verified_id must be 1D, got shape={verified_id.shape}.")
    if target_prefix_lens.shape != verified_id.shape:
        raise ValueError(
            "target_prefix_lens must match verified_id, got "
            f"{target_prefix_lens.shape} vs {verified_id.shape}."
        )
    bs = int(verified_id.shape[0])
    block_size = int(block_size)

    block_ids = np.full((bs, block_size), int(mask_token_id), dtype=np.int32)
    block_ids[:, 0] = verified_id
    positions = target_prefix_lens[:, None] + np.arange(block_size, dtype=np.int32)[None, :]
    return block_ids, positions.astype(np.int32)


def select_dflash_proposal_hidden(
    draft_hidden: jax.Array,
    *,
    enable_anchor: bool,
) -> jax.Array:
    """Select hidden rows that predict DFlash proposals.

    DeepSpec's anchor layout feeds ``[anchor, mask, ...]`` but every hidden
    row predicts a future proposal, so a block_size-N draft produces N
    proposals.  The legacy layout treats row zero as the already-verified
    token and therefore produces only N-1 proposals.
    """
    if draft_hidden.ndim < 2:
        raise ValueError(f"DFLASH draft hidden must have a block axis, got {draft_hidden.shape}.")
    return draft_hidden if enable_anchor else draft_hidden[:, 1:, ...]


def build_dflash_ngram_continuation(
    token_ids: list[int] | np.ndarray,
    *,
    prompt_len: int,
    proposal_width: int,
    min_match: int,
    max_match: int,
    bonus: float,
    prompt_weight: float,
    output_weight: float,
    position_decay: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Build one fixed-width continuation from the longest suffix match.

    The lookup uses only tokens that precede the current suffix, so the
    returned continuation is historical evidence from the prompt or already
    target-verified output.  Among equal-length matches, prefer the occurrence
    with the longest available continuation and then the most recent one.
    """
    tokens = np.asarray(token_ids, dtype=np.int32)
    width = int(proposal_width)
    continuation = np.zeros((width,), dtype=np.int32)
    bonuses = np.zeros((width,), dtype=np.float32)
    valid = np.zeros((width,), dtype=np.bool_)
    if width <= 0 or tokens.ndim != 1:
        return continuation, bonuses, valid, 0

    upper = min(int(max_match), len(tokens) - 1)
    lower = max(1, int(min_match))
    for match_len in range(upper, lower - 1, -1):
        suffix = tokens[-match_len:]
        matches = []
        for start in range(0, len(tokens) - match_len):
            if np.array_equal(tokens[start : start + match_len], suffix):
                continuation_start = start + match_len
                available = min(width, len(tokens) - continuation_start)
                if available > 0:
                    matches.append((available, start, continuation_start))
        if not matches:
            continue

        available, _, continuation_start = max(matches)
        continuation[:available] = tokens[
            continuation_start : continuation_start + available
        ]
        valid[:available] = True
        source_weight = (
            float(prompt_weight)
            if continuation_start < int(prompt_len)
            else float(output_weight)
        )
        offsets = np.arange(available, dtype=np.float32)
        bonuses[:available] = (
            float(bonus) * source_weight * np.power(float(position_decay), offsets)
        )
        return continuation, bonuses, valid, match_len

    return continuation, bonuses, valid, 0


def select_dflash_ngram_tokens(
    draft_logits: jax.Array,
    ngram_token_ids: jax.Array,
    ngram_bonus: jax.Array,
    ngram_valid_mask: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Margin-gated sparse N-gram reranking with a fixed candidate shape."""
    expected = draft_logits.shape[:-1]
    for name, value in (
        ("ngram_token_ids", ngram_token_ids),
        ("ngram_bonus", ngram_bonus),
        ("ngram_valid_mask", ngram_valid_mask),
    ):
        if value.shape != expected:
            raise ValueError(f"{name} must have shape {expected}, got {value.shape}.")

    base_token_ids = jnp.argmax(draft_logits, axis=-1).astype(jnp.int32)
    base_scores = _gather_dflash_vocab_scores(draft_logits, base_token_ids)
    ngram_scores = _gather_dflash_vocab_scores(draft_logits, ngram_token_ids)
    selected = (
        ngram_valid_mask
        & (ngram_token_ids != base_token_ids)
        & (ngram_bonus > 0)
        & ((base_scores - ngram_scores) <= ngram_bonus.astype(base_scores.dtype))
    )
    return jnp.where(selected, ngram_token_ids, base_token_ids).astype(jnp.int32), selected


def build_dflash_rejection_feedback(
    draft_token: np.ndarray,
    accept_lens: np.ndarray,
    active_mask: np.ndarray,
    *,
    block_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract the first rejected proposal for use by the next decode round."""
    candidates = np.asarray(draft_token, dtype=np.int32).reshape((-1, int(block_size)))
    accept_lens = np.asarray(accept_lens, dtype=np.int32).reshape(-1)
    active_mask = np.asarray(active_mask, dtype=np.bool_).reshape(-1)
    if candidates.shape[0] != accept_lens.shape[0] or accept_lens.shape != active_mask.shape:
        raise ValueError(
            "DFLASH rejection feedback batch shapes differ: "
            f"candidates={candidates.shape}, accept_lens={accept_lens.shape}, "
            f"active_mask={active_mask.shape}."
        )

    proposal_width = int(block_size) - 1
    accepted_proposals = np.maximum(accept_lens - 1, 0)
    valid = active_mask & (accept_lens > 0) & (accepted_proposals < proposal_width)
    rejected_indices = np.clip(accepted_proposals + 1, 1, proposal_width)
    rejected = candidates[np.arange(candidates.shape[0]), rejected_indices]
    return np.where(valid, rejected, 0).astype(np.int32), valid


# TODO: Share greedy chain verification through common speculative helpers.
def dflash_greedy_verify(
    draft_token: jax.Array,
    target_logits: jax.Array,
    *,
    draft_token_num: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Pure JAX target-logits argmax and greedy DFlash verification."""
    candidates = draft_token.reshape((-1, int(draft_token_num)))
    target_predict_flat = jnp.argmax(target_logits, axis=-1).astype(jnp.int32)
    mesh = getattr(jax.typeof(target_predict_flat).sharding, "mesh", None)
    if mesh is not None and getattr(mesh, "empty", False):
        mesh = None
    target_predict = target_predict_flat.reshape(candidates.shape)
    if mesh is not None:
        from jax.sharding import NamedSharding
        from jax.sharding import PartitionSpec as P

        data_2d = NamedSharding(mesh, P("data", None))
        candidates = jax.sharding.reshard(candidates, data_2d)
        target_predict = jax.sharding.reshard(target_predict, data_2d)

    matches = candidates[:, 1:] == target_predict[:, :-1]
    accept_len_draft = jnp.sum(jnp.cumprod(matches.astype(jnp.int32), axis=1), axis=1)
    target_predict_flat = target_predict.reshape(-1).astype(jnp.int32)
    if mesh is None:
        bonus = jnp.take_along_axis(
            target_predict,
            accept_len_draft[:, None],
            axis=1,
        ).reshape(-1)
    else:

        def _select_local_bonus(local_predict, local_accept_len):
            return jnp.take_along_axis(
                local_predict,
                local_accept_len[:, None],
                axis=1,
            ).reshape(-1)

        bonus = jax.shard_map(
            _select_local_bonus,
            mesh=mesh,
            in_specs=(P("data", None), P("data")),
            out_specs=P("data"),
        )(target_predict, accept_len_draft)

    accept_lens_out = (accept_len_draft + 1).astype(jnp.int32)
    return accept_lens_out, target_predict_flat, bonus, accept_len_draft.astype(jnp.int32)


def _gather_dflash_vocab_scores(
    logits: jax.Array,
    token_ids: jax.Array,
) -> jax.Array:
    """Gather sparse token scores without replicating a TP-sharded vocabulary.

    DFlash logits shard their vocabulary dimension over ``tensor``.  A plain
    ``take_along_axis`` leaves GSPMD free to replicate that dimension before
    gathering, which is especially costly when this operation is fused into
    the target/draft model JIT.  Instead each TP rank reads only a candidate
    that falls inside its local vocabulary slice and the ranks combine those
    scalar scores with ``pmax``.
    """
    if logits.shape[:-1] != token_ids.shape:
        raise ValueError(
            "DFLASH sparse score ids must match logits leading shape: "
            f"{token_ids.shape} vs {logits.shape}."
        )

    logits_sharding = jax.typeof(logits).sharding
    mesh = getattr(logits_sharding, "mesh", None)
    if mesh is None or getattr(mesh, "empty", False) or "tensor" not in mesh.axis_names:
        return jnp.take_along_axis(logits, token_ids[..., None], axis=-1)[..., 0]

    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    logits_spec = P("data", *(None for _ in logits.shape[1:-1]), "tensor")
    ids_spec = P("data", *(None for _ in token_ids.shape[1:]))
    logits = jax.sharding.reshard(logits, NamedSharding(mesh, logits_spec))
    token_ids = jax.sharding.reshard(token_ids, NamedSharding(mesh, ids_spec))

    def _gather_local(local_logits, global_token_ids):
        local_vocab_size = local_logits.shape[-1]
        vocab_start = jax.lax.axis_index("tensor") * local_vocab_size
        local_token_ids = global_token_ids - vocab_start
        owns_token = (local_token_ids >= 0) & (local_token_ids < local_vocab_size)
        safe_token_ids = jnp.clip(local_token_ids, 0, local_vocab_size - 1)
        local_scores = jnp.take_along_axis(
            local_logits,
            safe_token_ids[..., None],
            axis=-1,
        )[..., 0]
        local_scores = jnp.where(
            owns_token,
            local_scores,
            jnp.asarray(-jnp.inf, dtype=local_logits.dtype),
        )
        return jax.lax.pmax(local_scores, "tensor")

    return jax.shard_map(
        _gather_local,
        mesh=mesh,
        in_specs=(logits_spec, ids_spec),
        out_specs=ids_spec,
    )(logits, token_ids)


def build_dflash_flashback_feedback(
    draft_token: jax.Array,
    target_logits: jax.Array,
    target_predict: jax.Array,
    accept_len_draft: jax.Array,
    *,
    draft_token_num: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Extract one-round counterfactual suffix feedback for FlashBack.

    ``draft_token[:, 0]`` is the verified anchor.  If ``r`` proposal tokens
    were accepted, proposal ``r + 1`` is the first rejected token and is
    replaced by the target bonus token.  Only the proposals after that first
    rejection are aligned with the next round, so this function carries
    ``draft_token[:, r + 2:]`` and their target-logit margins forward.

    The suffix logits were evaluated under a counterfactual (now stale)
    prefix.  Their margins are therefore evidence for sparse reranking, not
    verification results.  The returned validity mask keeps that distinction
    explicit and gives every request a fixed ``draft_token_num - 1`` shape.
    """
    block_size = int(draft_token_num)
    candidates = draft_token.reshape((-1, block_size))
    logits = target_logits.reshape((candidates.shape[0], block_size, -1))
    target_predict = target_predict.reshape(candidates.shape)
    proposal_width = block_size - 1

    # Logits row i predicts candidate i + 1.  Compute the target margin for
    # every proposal before dynamically shifting the stale suffix.
    proposal_logits = logits[:, :proposal_width, :]
    proposal_ids = candidates[:, 1:]
    target_top1_ids = target_predict[:, :proposal_width]
    proposal_scores = _gather_dflash_vocab_scores(proposal_logits, proposal_ids)
    target_top1_scores = _gather_dflash_vocab_scores(proposal_logits, target_top1_ids)
    proposal_margins = proposal_scores - target_top1_scores

    next_offsets = jnp.arange(proposal_width, dtype=jnp.int32)[None, :]
    source_candidate_indices = accept_len_draft[:, None] + 2 + next_offsets
    valid = source_candidate_indices < block_size
    source_proposal_indices = jnp.clip(
        source_candidate_indices - 1,
        0,
        proposal_width - 1,
    )
    stale_token_ids = jnp.take_along_axis(
        proposal_ids,
        source_proposal_indices,
        axis=1,
    )
    target_margins = jnp.take_along_axis(
        proposal_margins,
        source_proposal_indices,
        axis=1,
    )
    stale_token_ids = jnp.where(valid, stale_token_ids, jnp.int32(0))
    target_margins = jnp.where(valid, target_margins, jnp.asarray(0, target_margins.dtype))
    return stale_token_ids.astype(jnp.int32), target_margins, valid


def select_dflash_flashback_tokens(
    draft_logits: jax.Array,
    stale_token_ids: jax.Array,
    stale_target_margins: jax.Array,
    stale_valid_mask: jax.Array,
    *,
    bonus: float,
    target_margin_weight: float,
    position_decay: float,
) -> jax.Array:
    """Select DFlash proposals after training-free sparse FlashBack reranking.

    Adding a bonus to one sparse vocabulary entry is equivalent to comparing
    that entry against the row maximum.  Implementing the comparison directly
    avoids materializing or scattering a ``[batch, block, vocab]`` bias tensor.
    """
    if draft_logits.ndim != 3:
        raise ValueError(f"draft_logits must be [batch, block, vocab], got {draft_logits.shape}.")
    expected = draft_logits.shape[:-1]
    for name, value in (
        ("stale_token_ids", stale_token_ids),
        ("stale_target_margins", stale_target_margins),
        ("stale_valid_mask", stale_valid_mask),
    ):
        if value.shape != expected:
            raise ValueError(f"{name} must have shape {expected}, got {value.shape}.")

    base_token_ids = jnp.argmax(draft_logits, axis=-1).astype(jnp.int32)
    base_scores = _gather_dflash_vocab_scores(draft_logits, base_token_ids)
    stale_scores = _gather_dflash_vocab_scores(draft_logits, stale_token_ids)
    offsets = jnp.arange(draft_logits.shape[1], dtype=base_scores.dtype)[None, :]
    position_bonus = jnp.asarray(bonus, base_scores.dtype) * jnp.power(
        jnp.asarray(position_decay, base_scores.dtype),
        offsets,
    )
    effective_bonus = position_bonus + jnp.asarray(
        target_margin_weight, base_scores.dtype
    ) * stale_target_margins.astype(base_scores.dtype)
    use_stale = (
        stale_valid_mask & (effective_bonus > 0) & ((base_scores - stale_scores) <= effective_bonus)
    )
    return jnp.where(use_stale, stale_token_ids, base_token_ids).astype(jnp.int32)


@dataclass
class DFlashDraftInput:
    """Host-side DFlash state carried between decode iterations."""

    verified_id: jax.Array | np.ndarray = None
    target_hidden: jax.Array | None = None
    ctx_lens: np.ndarray = None
    draft_seq_lens: np.ndarray = None
    allocate_lens: np.ndarray = None
    reservation_base_lens: np.ndarray = None
    future_indices: np.ndarray = None
    flashback_token_ids: jax.Array | np.ndarray = None
    flashback_target_margins: jax.Array | np.ndarray = None
    flashback_valid_mask: jax.Array | np.ndarray = None
    rejected_draft_token_ids: jax.Array | np.ndarray = None
    rejection_valid_mask: jax.Array | np.ndarray = None
    ngram_token_ids: jax.Array | np.ndarray = None
    ngram_bonus: jax.Array | np.ndarray = None
    ngram_valid_mask: jax.Array | np.ndarray = None
    ngram_match_lens: jax.Array | np.ndarray = None
    enable_ngram: bool = False
    ngram_min_match: int = 3
    ngram_max_match: int = 8
    ngram_base_bonus: float = 1.0
    ngram_prompt_weight: float = 0.7
    ngram_output_weight: float = 1.0
    ngram_position_decay: float = 0.8
    block_size: int = 16
    capture_hidden_mode = CaptureHiddenMode.FULL

    def _ensure_host(self) -> None:
        int_fields = (
            "verified_id",
            "ctx_lens",
            "draft_seq_lens",
            "allocate_lens",
            "reservation_base_lens",
            "future_indices",
            "flashback_token_ids",
            "rejected_draft_token_ids",
            "ngram_token_ids",
            "ngram_match_lens",
        )
        fields = int_fields + (
            "flashback_target_margins",
            "flashback_valid_mask",
            "rejection_valid_mask",
            "ngram_bonus",
            "ngram_valid_mask",
        )
        for f in fields:
            v = getattr(self, f, None)
            if v is not None and hasattr(v, "copy_to_host_async"):
                v.copy_to_host_async()
        for f in int_fields:
            v = getattr(self, f, None)
            if v is not None:
                setattr(self, f, np.asarray(v, dtype=np.int32))
        if self.flashback_target_margins is not None:
            self.flashback_target_margins = np.asarray(
                self.flashback_target_margins, dtype=np.float32
            )
        if self.flashback_valid_mask is not None:
            self.flashback_valid_mask = np.asarray(self.flashback_valid_mask, dtype=np.bool_)
        if self.rejection_valid_mask is not None:
            self.rejection_valid_mask = np.asarray(self.rejection_valid_mask, dtype=np.bool_)
        if self.ngram_bonus is not None:
            self.ngram_bonus = np.asarray(self.ngram_bonus, dtype=np.float32)
        if self.ngram_valid_mask is not None:
            self.ngram_valid_mask = np.asarray(self.ngram_valid_mask, dtype=np.bool_)

    def new_tokens_required_next_decode(self, requests, page_size: int) -> int:
        total = 0
        block_size = int(self.block_size)
        reserve_tokens = block_size * (2 if self.future_indices is not None else 1)
        for req in requests:
            cur = int(req.kv_allocated_len)
            nxt = max(cur, int(req.kv_committed_len) + reserve_tokens)
            total += ((nxt + page_size - 1) // page_size) * page_size - (
                (cur + page_size - 1) // page_size
            ) * page_size
        return total

    def _flashback_rows(self, bs: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        width = int(self.block_size) - 1
        expected = (int(bs), width)
        token_ids = self.flashback_token_ids
        target_margins = self.flashback_target_margins
        valid_mask = self.flashback_valid_mask
        if token_ids is None and target_margins is None and valid_mask is None:
            return (
                np.zeros(expected, dtype=np.int32),
                np.zeros(expected, dtype=np.float32),
                np.zeros(expected, dtype=np.bool_),
            )
        if token_ids is None or target_margins is None or valid_mask is None:
            raise ValueError(
                "DFLASH FlashBack state must carry ids, margins, and validity together."
            )
        token_ids = np.asarray(token_ids, dtype=np.int32)
        target_margins = np.asarray(target_margins, dtype=np.float32)
        valid_mask = np.asarray(valid_mask, dtype=np.bool_)
        for name, value in (
            ("flashback_token_ids", token_ids),
            ("flashback_target_margins", target_margins),
            ("flashback_valid_mask", valid_mask),
        ):
            if value.shape != expected:
                raise ValueError(f"DFLASH {name} must have shape {expected}, got {value.shape}.")
        return token_ids, target_margins, valid_mask

    def _ngram_rows(
        self, bs: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        width = int(self.block_size) - 1
        expected = (int(bs), width)
        token_ids = self.ngram_token_ids
        bonuses = self.ngram_bonus
        valid_mask = self.ngram_valid_mask
        match_lens = self.ngram_match_lens
        if token_ids is None and bonuses is None and valid_mask is None and match_lens is None:
            return (
                np.zeros(expected, dtype=np.int32),
                np.zeros(expected, dtype=np.float32),
                np.zeros(expected, dtype=np.bool_),
                np.zeros((int(bs),), dtype=np.int32),
            )
        if token_ids is None or bonuses is None or valid_mask is None or match_lens is None:
            raise ValueError("DFLASH N-gram state must carry ids, bonus, validity, and match length.")
        token_ids = np.asarray(token_ids, dtype=np.int32)
        bonuses = np.asarray(bonuses, dtype=np.float32)
        valid_mask = np.asarray(valid_mask, dtype=np.bool_)
        match_lens = np.asarray(match_lens, dtype=np.int32)
        for name, value in (
            ("ngram_token_ids", token_ids),
            ("ngram_bonus", bonuses),
            ("ngram_valid_mask", valid_mask),
        ):
            if value.shape != expected:
                raise ValueError(f"DFLASH {name} must have shape {expected}, got {value.shape}.")
        if match_lens.shape != (int(bs),):
            raise ValueError(
                "DFLASH ngram_match_lens must have shape "
                f"({int(bs)},), got {match_lens.shape}."
            )
        return token_ids, bonuses, valid_mask, match_lens

    def _rejection_rows(self, bs: int) -> tuple[np.ndarray, np.ndarray]:
        token_ids = self.rejected_draft_token_ids
        valid_mask = self.rejection_valid_mask
        if token_ids is None and valid_mask is None:
            return (
                np.zeros((int(bs),), dtype=np.int32),
                np.zeros((int(bs),), dtype=np.bool_),
            )
        if token_ids is None or valid_mask is None:
            raise ValueError("DFLASH rejection feedback must carry ids and validity together.")
        token_ids = np.asarray(token_ids, dtype=np.int32)
        valid_mask = np.asarray(valid_mask, dtype=np.bool_)
        if token_ids.shape != (int(bs),) or valid_mask.shape != (int(bs),):
            raise ValueError(
                "DFLASH rejection feedback must have shape "
                f"({int(bs)},), got ids={token_ids.shape}, valid={valid_mask.shape}."
            )
        return token_ids, valid_mask

    def filter_batch(self, new_indices: np.ndarray, has_been_filtered: bool = True) -> None:
        self._ensure_host()
        new_indices = np.asarray(new_indices, dtype=np.int32)
        if self.future_indices is not None:
            old_bs = len(self.future_indices)
            selected = (
                np.arange(len(new_indices), dtype=np.int32)
                if has_been_filtered and len(new_indices) == old_bs
                else new_indices
            )
            for field in ("future_indices", "allocate_lens", "reservation_base_lens"):
                value = getattr(self, field, None)
                if value is not None:
                    setattr(self, field, np.asarray(value, dtype=np.int32)[selected])
            return

        old_verified_id = np.asarray(self.verified_id, dtype=np.int32)
        old_ctx_lens = np.asarray(self.ctx_lens, dtype=np.int32)
        old_draft_seq_lens = np.asarray(self.draft_seq_lens, dtype=np.int32)
        old_bs = len(old_verified_id)
        if has_been_filtered and len(new_indices) == old_bs:
            selected = np.arange(len(new_indices), dtype=np.int32)
        else:
            selected = new_indices

        self.verified_id = old_verified_id[selected]
        self.ctx_lens = old_ctx_lens[selected]
        self.draft_seq_lens = old_draft_seq_lens[selected]
        for field in ("allocate_lens", "reservation_base_lens"):
            value = getattr(self, field, None)
            if value is not None:
                setattr(self, field, np.asarray(value, dtype=np.int32)[selected])
        for field in (
            "flashback_token_ids",
            "flashback_target_margins",
            "flashback_valid_mask",
            "rejected_draft_token_ids",
            "rejection_valid_mask",
            "ngram_token_ids",
            "ngram_bonus",
            "ngram_valid_mask",
            "ngram_match_lens",
        ):
            value = getattr(self, field, None)
            if value is not None:
                setattr(self, field, np.asarray(value)[selected])

        if self.target_hidden is not None and self.target_hidden.shape[0] != 0:
            raise ValueError("DFLASH target_hidden must be materialized before filtering.")
        self.target_hidden = None

    def prepare_for_decode(self, schedule_batch) -> None:
        # TODO: Share KV slot reservation and req_to_token_pool updates
        # with EAGLE through common speculative helpers in the next PR.
        from sgl_jax.srt.managers.schedule_batch import get_last_loc
        from sgl_jax.srt.mem_cache.common import (
            alloc_paged_token_slots_extend,
            alloc_token_slots,
        )
        from sgl_jax.srt.speculative.eagle_util import assign_req_to_token_pool

        block_size = self.block_size
        page_size = schedule_batch.token_to_kv_pool_allocator.page_size
        reserve_tokens = block_size * (2 if schedule_batch.enable_overlap else 1)

        self._align_dp_state_to_reqs(schedule_batch)
        if self.enable_ngram:
            reqs = [req for info in schedule_batch.reqs_info for req in (info.reqs or [])]
            rows = [
                build_dflash_ngram_continuation(
                    req.origin_input_ids + req.output_ids,
                    prompt_len=len(req.origin_input_ids),
                    proposal_width=int(self.block_size) - 1,
                    min_match=self.ngram_min_match,
                    max_match=self.ngram_max_match,
                    bonus=self.ngram_base_bonus,
                    prompt_weight=self.ngram_prompt_weight,
                    output_weight=self.ngram_output_weight,
                    position_decay=self.ngram_position_decay,
                )
                for req in reqs
            ]
            if rows:
                self.ngram_token_ids = np.stack([row[0] for row in rows], axis=0)
                self.ngram_bonus = np.stack([row[1] for row in rows], axis=0)
                self.ngram_valid_mask = np.stack([row[2] for row in rows], axis=0)
                self.ngram_match_lens = np.asarray([row[3] for row in rows], dtype=np.int32)
            else:
                width = int(self.block_size) - 1
                self.ngram_token_ids = np.zeros((0, width), dtype=np.int32)
                self.ngram_bonus = np.zeros((0, width), dtype=np.float32)
                self.ngram_valid_mask = np.zeros((0, width), dtype=np.bool_)
                self.ngram_match_lens = np.zeros((0,), dtype=np.int32)
        allocate_lens = []
        reservation_base_lens = []

        for dp_rank, info in enumerate(schedule_batch.reqs_info):
            if info.seq_lens is None or len(info.seq_lens) == 0:
                continue

            reqs = info.reqs

            old_r = np.asarray([req.kv_allocated_len for req in reqs], dtype=np.int32)
            committed_r = np.asarray([req.kv_committed_len for req in reqs], dtype=np.int32)
            new_r = np.maximum(old_r, committed_r + reserve_tokens)
            ext_r = int((new_r - old_r).sum())

            if ext_r > 0 and page_size == 1:
                ocl_r = alloc_token_slots(schedule_batch.tree_cache, ext_r, dp_rank=dp_rank)
                assign_req_to_token_pool(
                    info.req_pool_indices,
                    schedule_batch.req_to_token_pool,
                    old_r,
                    new_r,
                    ocl_r,
                )
            elif ext_r > 0:
                last_loc_r = get_last_loc(
                    schedule_batch.req_to_token_pool.req_to_token,
                    info.req_pool_indices,
                    old_r,
                )
                ocl_r = alloc_paged_token_slots_extend(
                    schedule_batch.tree_cache,
                    old_r,
                    new_r,
                    last_loc_r,
                    int((new_r - old_r).sum()),
                    dp_rank=dp_rank,
                )
                assign_req_to_token_pool(
                    info.req_pool_indices,
                    schedule_batch.req_to_token_pool,
                    old_r,
                    new_r,
                    ocl_r,
                )

            req_to_token = schedule_batch.req_to_token_pool.req_to_token
            verify_locs = []
            for i, req in enumerate(reqs):
                rp = int(info.req_pool_indices[i])
                c = int(committed_r[i])
                verify_locs.append(
                    np.asarray(req_to_token[rp, c : c + reserve_tokens], dtype=np.int32)
                )
            info.out_cache_loc = (
                np.concatenate(verify_locs) if verify_locs else np.empty(0, dtype=np.int32)
            )
            allocate_lens.append(new_r)
            reservation_base_lens.append(committed_r)

            for req, allocated_len in zip(reqs, new_r):
                req.decode_batch_idx += 1
                req.kv_allocated_len = int(allocated_len)
                req.kv_committed_len += 1

            info.seq_lens_sum = np.sum(info.seq_lens).item()

        self.allocate_lens = (
            np.concatenate(allocate_lens) if allocate_lens else np.empty((0,), dtype=np.int32)
        )
        self.reservation_base_lens = (
            np.concatenate(reservation_base_lens)
            if reservation_base_lens
            else np.empty((0,), dtype=np.int32)
        )

    def _align_dp_state_to_reqs(self, schedule_batch) -> None:
        """Align each rank's state independently, then rebuild rank-major state."""
        if self.future_indices is not None:
            expected = sum(len(info.reqs or []) for info in schedule_batch.reqs_info)
            if len(self.future_indices) != expected:
                raise ValueError(
                    "DFLASH relay state does not match the decode requests: "
                    f"future_indices={len(self.future_indices)}, requests={expected}."
                )
            return

        rank_states = []
        for info in schedule_batch.reqs_info:
            reqs = info.reqs or []
            if not reqs:
                continue

            rank_state = info.spec_info
            if not isinstance(rank_state, DFlashDraftInput):
                rank_state = DFlashDraftInput(
                    verified_id=np.empty((0,), dtype=np.int32),
                    target_hidden=None,
                    ctx_lens=np.empty((0,), dtype=np.int32),
                    draft_seq_lens=np.empty((0,), dtype=np.int32),
                    block_size=self.block_size,
                    enable_ngram=self.enable_ngram,
                    ngram_min_match=self.ngram_min_match,
                    ngram_max_match=self.ngram_max_match,
                    ngram_base_bonus=self.ngram_base_bonus,
                    ngram_prompt_weight=self.ngram_prompt_weight,
                    ngram_output_weight=self.ngram_output_weight,
                    ngram_position_decay=self.ngram_position_decay,
                )
            committed_lens = np.asarray([req.kv_committed_len for req in reqs], dtype=np.int32)
            rank_state._align_to_reqs(reqs, committed_lens)
            rank_states.append(rank_state)

        if not rank_states:
            self.verified_id = np.empty((0,), dtype=np.int32)
            self.ctx_lens = np.empty((0,), dtype=np.int32)
            self.draft_seq_lens = np.empty((0,), dtype=np.int32)
            self.target_hidden = None
            self.flashback_token_ids = None
            self.flashback_target_margins = None
            self.flashback_valid_mask = None
            self.rejected_draft_token_ids = None
            self.rejection_valid_mask = None
            self.ngram_token_ids = None
            self.ngram_bonus = None
            self.ngram_valid_mask = None
            self.ngram_match_lens = None
            return

        self.verified_id = np.concatenate(
            [np.asarray(state.verified_id, dtype=np.int32) for state in rank_states]
        )
        self.ctx_lens = np.concatenate(
            [np.asarray(state.ctx_lens, dtype=np.int32) for state in rank_states]
        )
        self.draft_seq_lens = np.concatenate(
            [np.asarray(state.draft_seq_lens, dtype=np.int32) for state in rank_states]
        )
        feedback = [state._flashback_rows(len(state.draft_seq_lens)) for state in rank_states]
        self.flashback_token_ids = np.concatenate([rows[0] for rows in feedback], axis=0)
        self.flashback_target_margins = np.concatenate([rows[1] for rows in feedback], axis=0)
        self.flashback_valid_mask = np.concatenate([rows[2] for rows in feedback], axis=0)
        rejection = [state._rejection_rows(len(state.draft_seq_lens)) for state in rank_states]
        self.rejected_draft_token_ids = np.concatenate([rows[0] for rows in rejection], axis=0)
        self.rejection_valid_mask = np.concatenate([rows[1] for rows in rejection], axis=0)
        ngram = [state._ngram_rows(len(state.draft_seq_lens)) for state in rank_states]
        self.ngram_token_ids = np.concatenate([rows[0] for rows in ngram], axis=0)
        self.ngram_bonus = np.concatenate([rows[1] for rows in ngram], axis=0)
        self.ngram_valid_mask = np.concatenate([rows[2] for rows in ngram], axis=0)
        self.ngram_match_lens = np.concatenate([rows[3] for rows in ngram], axis=0)

        hidden_parts = [state.target_hidden for state in rank_states]
        if all(hidden is None for hidden in hidden_parts):
            self.target_hidden = None
        elif all(hidden is not None and hidden.shape[0] == 0 for hidden in hidden_parts):
            self.target_hidden = hidden_parts[0][:0]
        else:
            raise ValueError(
                "DFLASH target_hidden must be materialized before DP decode preparation."
            )

    def _align_to_reqs(self, reqs, committed_lens: np.ndarray) -> None:
        state_bs = int(np.asarray(self.draft_seq_lens, dtype=np.int32).shape[0])
        bs = len(reqs)
        if state_bs == bs:
            return

        verified_id = np.asarray(self.verified_id, dtype=np.int32)
        ctx_lens = np.asarray(self.ctx_lens, dtype=np.int32)
        draft_seq_lens = np.asarray(self.draft_seq_lens, dtype=np.int32)
        feedback_ids, feedback_margins, feedback_valid = self._flashback_rows(state_bs)
        rejected_ids, rejection_valid = self._rejection_rows(state_bs)
        ngram_ids, ngram_bonus, ngram_valid, ngram_match_lens = self._ngram_rows(state_bs)
        if state_bs > bs:
            self.verified_id = verified_id[:bs]
            self.ctx_lens = ctx_lens[:bs]
            self.draft_seq_lens = draft_seq_lens[:bs]
            self.flashback_token_ids = feedback_ids[:bs]
            self.flashback_target_margins = feedback_margins[:bs]
            self.flashback_valid_mask = feedback_valid[:bs]
            self.rejected_draft_token_ids = rejected_ids[:bs]
            self.rejection_valid_mask = rejection_valid[:bs]
            self.ngram_token_ids = ngram_ids[:bs]
            self.ngram_bonus = ngram_bonus[:bs]
            self.ngram_valid_mask = ngram_valid[:bs]
            self.ngram_match_lens = ngram_match_lens[:bs]
            return

        missing_reqs = reqs[state_bs:bs]
        missing_verified = np.asarray(
            [
                req.output_ids[-1] if len(req.output_ids) > 0 else req.origin_input_ids[-1]
                for req in missing_reqs
            ],
            dtype=np.int32,
        )
        self.verified_id = np.concatenate([verified_id, missing_verified], axis=0)
        self.ctx_lens = np.concatenate(
            [ctx_lens, np.zeros((bs - state_bs,), dtype=np.int32)], axis=0
        )
        self.draft_seq_lens = np.concatenate(
            [draft_seq_lens, committed_lens[state_bs:bs].astype(np.int32)], axis=0
        )
        missing = bs - state_bs
        width = int(self.block_size) - 1
        self.flashback_token_ids = np.concatenate(
            [feedback_ids, np.zeros((missing, width), dtype=np.int32)], axis=0
        )
        self.flashback_target_margins = np.concatenate(
            [feedback_margins, np.zeros((missing, width), dtype=np.float32)], axis=0
        )
        self.flashback_valid_mask = np.concatenate(
            [feedback_valid, np.zeros((missing, width), dtype=np.bool_)], axis=0
        )
        self.rejected_draft_token_ids = np.concatenate(
            [rejected_ids, np.zeros((missing,), dtype=np.int32)], axis=0
        )
        self.rejection_valid_mask = np.concatenate(
            [rejection_valid, np.zeros((missing,), dtype=np.bool_)], axis=0
        )
        self.ngram_token_ids = np.concatenate(
            [ngram_ids, np.zeros((missing, width), dtype=np.int32)], axis=0
        )
        self.ngram_bonus = np.concatenate(
            [ngram_bonus, np.zeros((missing, width), dtype=np.float32)], axis=0
        )
        self.ngram_valid_mask = np.concatenate(
            [ngram_valid, np.zeros((missing, width), dtype=np.bool_)], axis=0
        )
        self.ngram_match_lens = np.concatenate(
            [ngram_match_lens, np.zeros((missing,), dtype=np.int32)], axis=0
        )

    def merge_batch(self, other: DFlashDraftInput) -> None:
        self._ensure_host()
        other._ensure_host()
        if self.future_indices is not None or other.future_indices is not None:
            if self.future_indices is None or other.future_indices is None:
                raise ValueError("DFLASH overlap merge requires future_indices on both batches.")
            self.future_indices = np.concatenate(
                [self.future_indices, other.future_indices], axis=0
            )
            for field in ("allocate_lens", "reservation_base_lens"):
                lhs = getattr(self, field, None)
                rhs = getattr(other, field, None)
                setattr(
                    self,
                    field,
                    None if lhs is None or rhs is None else np.concatenate([lhs, rhs], axis=0),
                )
            self.verified_id = None
            self.ctx_lens = None
            self.draft_seq_lens = None
            self.target_hidden = None
            self.flashback_token_ids = None
            self.flashback_target_margins = None
            self.flashback_valid_mask = None
            self.rejected_draft_token_ids = None
            self.rejection_valid_mask = None
            self.ngram_token_ids = None
            self.ngram_bonus = None
            self.ngram_valid_mask = None
            self.ngram_match_lens = None
            return

        self.verified_id = np.concatenate(
            [np.asarray(self.verified_id), np.asarray(other.verified_id)], axis=0
        )
        self.ctx_lens = np.concatenate([self.ctx_lens, other.ctx_lens], axis=0)
        self.draft_seq_lens = np.concatenate([self.draft_seq_lens, other.draft_seq_lens], axis=0)
        self_feedback = self._flashback_rows(len(self.verified_id) - len(other.verified_id))
        other_feedback = other._flashback_rows(len(other.verified_id))
        self.flashback_token_ids = np.concatenate([self_feedback[0], other_feedback[0]], axis=0)
        self.flashback_target_margins = np.concatenate(
            [self_feedback[1], other_feedback[1]], axis=0
        )
        self.flashback_valid_mask = np.concatenate([self_feedback[2], other_feedback[2]], axis=0)
        self_rejection = self._rejection_rows(len(self.verified_id) - len(other.verified_id))
        other_rejection = other._rejection_rows(len(other.verified_id))
        self.rejected_draft_token_ids = np.concatenate(
            [self_rejection[0], other_rejection[0]], axis=0
        )
        self.rejection_valid_mask = np.concatenate([self_rejection[1], other_rejection[1]], axis=0)
        self_ngram = self._ngram_rows(len(self.verified_id) - len(other.verified_id))
        other_ngram = other._ngram_rows(len(other.verified_id))
        self.ngram_token_ids = np.concatenate([self_ngram[0], other_ngram[0]], axis=0)
        self.ngram_bonus = np.concatenate([self_ngram[1], other_ngram[1]], axis=0)
        self.ngram_valid_mask = np.concatenate([self_ngram[2], other_ngram[2]], axis=0)
        self.ngram_match_lens = np.concatenate([self_ngram[3], other_ngram[3]], axis=0)
        for field in ("allocate_lens", "reservation_base_lens"):
            lhs = getattr(self, field, None)
            rhs = getattr(other, field, None)
            setattr(
                self,
                field,
                None if lhs is None or rhs is None else np.concatenate([lhs, rhs], axis=0),
            )
        if self.target_hidden is None:
            self.target_hidden = other.target_hidden
        elif other.target_hidden is not None:
            self.target_hidden = jnp.concatenate([self.target_hidden, other.target_hidden], axis=0)


@register_pytree_node_class
@dataclass
class DFlashVerifyInput:
    """JIT-visible target verify input for a fixed DFlash block."""

    draft_token: jax.Array
    draft_token_num: int
    custom_mask = None

    def tree_flatten(self):
        return (self.draft_token,), {"draft_token_num": int(self.draft_token_num)}

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(
            draft_token=children[0],
            draft_token_num=aux_data["draft_token_num"],
        )
