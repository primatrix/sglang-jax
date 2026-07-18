"""Independent pure-PyTorch CPU references for DSA selection and sparse MLA."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TorchGlmDsaSelection:
    scores: torch.Tensor
    logical_topk_ids: torch.Tensor
    selected_counts: torch.Tensor


@dataclass(frozen=True)
class TorchDsaSelection:
    physical_slots: torch.Tensor
    selected_counts: torch.Tensor
    producer_layer: int
    logical_topk_ids: torch.Tensor


def _require_cpu(name: str, value: torch.Tensor) -> None:
    if value.device.type != "cpu":
        raise ValueError(f"{name} must be a CPU tensor; got {value.device}")


def torch_glm_dsa_select(
    q_index: torch.Tensor,
    head_weights: torch.Tensor,
    k_index_cache: torch.Tensor,
    candidate_slots: torch.Tensor,
    candidate_logical_ids: torch.Tensor,
    candidate_counts: torch.Tensor,
    *,
    index_topk: int,
) -> TorchGlmDsaSelection:
    """Score candidate Index-K rows and return sorted logical Top-K IDs."""
    for name, value in (
        ("q_index", q_index),
        ("head_weights", head_weights),
        ("k_index_cache", k_index_cache),
        ("candidate_slots", candidate_slots),
        ("candidate_logical_ids", candidate_logical_ids),
        ("candidate_counts", candidate_counts),
    ):
        _require_cpu(name, value)

    if q_index.ndim != 3:
        raise ValueError(f"q_index must have rank 3; got {q_index.ndim}")
    token_count, num_heads, head_dim = q_index.shape
    if head_weights.shape != (token_count, num_heads):
        raise ValueError(
            "head_weights must match q_index [tokens, heads]; got "
            f"{head_weights.shape} and {q_index.shape}"
        )
    if k_index_cache.ndim != 2 or k_index_cache.shape[1] != head_dim:
        raise ValueError(
            "k_index_cache must have shape [slots, q_index_dim]; got "
            f"{k_index_cache.shape} for q_index {q_index.shape}"
        )
    if k_index_cache.shape[0] == 0:
        raise ValueError("k_index_cache must contain at least one safe slot")
    if candidate_slots.ndim != 2 or candidate_slots.shape[0] != token_count:
        raise ValueError(
            "candidate_slots must have shape [tokens, candidates]; got "
            f"{candidate_slots.shape} for q_index {q_index.shape}"
        )
    if candidate_slots.dtype != torch.int32:
        raise TypeError(
            f"candidate_slots must have dtype int32; got {candidate_slots.dtype}"
        )
    if candidate_logical_ids.shape != candidate_slots.shape:
        raise ValueError(
            "candidate_logical_ids must match candidate_slots shape; got "
            f"{candidate_logical_ids.shape} and {candidate_slots.shape}"
        )
    if candidate_logical_ids.dtype != torch.int32:
        raise TypeError(
            "candidate_logical_ids must have dtype int32; got "
            f"{candidate_logical_ids.dtype}"
        )
    if candidate_counts.shape != (token_count,):
        raise ValueError(
            f"candidate_counts must have shape {(token_count,)}; "
            f"got {candidate_counts.shape}"
        )
    if candidate_counts.dtype != torch.int32:
        raise TypeError(
            f"candidate_counts must have dtype int32; got {candidate_counts.dtype}"
        )
    if index_topk <= 1:
        raise ValueError(f"index_topk must be greater than one; got {index_topk}")

    candidate_width = candidate_slots.shape[1]
    safe_slots = candidate_slots.clamp(0, k_index_cache.shape[0] - 1).long()
    candidate_keys = k_index_cache[safe_slots]
    logits = torch.einsum(
        "thd,tcd->tch", q_index.float(), candidate_keys.float()
    )
    scores = (
        torch.relu(logits) * head_weights.float()[:, None, :]
    ).sum(dim=-1)
    scores *= head_dim**-0.5 * num_heads**-0.5

    bounded_counts = candidate_counts.clamp(0, candidate_width)
    candidate_valid = (
        torch.arange(candidate_width)[None, :] < bounded_counts[:, None]
    )
    scores = scores.masked_fill(~candidate_valid, -torch.inf)

    pad_width = max(0, index_topk - candidate_width)
    if pad_width:
        scores = torch.nn.functional.pad(scores, (0, pad_width), value=-torch.inf)
        candidate_logical_ids = torch.nn.functional.pad(
            candidate_logical_ids, (0, pad_width), value=-1
        )

    topk_scores, topk_offsets = torch.topk(
        scores, index_topk, dim=1, largest=True, sorted=True
    )
    logical_topk_ids = torch.gather(
        candidate_logical_ids, 1, topk_offsets
    )
    selected_counts = torch.minimum(
        bounded_counts, torch.tensor(index_topk, dtype=torch.int32)
    )
    selected_valid = (
        torch.arange(index_topk)[None, :] < selected_counts[:, None]
    )
    logical_topk_ids = logical_topk_ids.masked_fill(~selected_valid, -1).to(
        torch.int32
    )

    return TorchGlmDsaSelection(
        scores=topk_scores,
        logical_topk_ids=logical_topk_ids,
        selected_counts=selected_counts,
    )


def torch_logical_topk_to_physical_slots(
    *,
    logical_topk_ids: torch.Tensor,
    selected_counts: torch.Tensor,
    req_to_token_slots: torch.Tensor,
    query_request_indices: torch.Tensor,
    query_positions: torch.Tensor,
    producer_layer: int,
) -> TorchDsaSelection:
    """Map logical IDs to physical slots and compact valid unique prefixes."""
    for name, value in (
        ("logical_topk_ids", logical_topk_ids),
        ("selected_counts", selected_counts),
        ("req_to_token_slots", req_to_token_slots),
        ("query_request_indices", query_request_indices),
        ("query_positions", query_positions),
    ):
        _require_cpu(name, value)

    if logical_topk_ids.ndim != 2 or logical_topk_ids.dtype != torch.int32:
        raise ValueError("logical_topk_ids must be a rank-2 int32 tensor")
    token_count, topk_width = logical_topk_ids.shape
    for name, value in (
        ("selected_counts", selected_counts),
        ("query_request_indices", query_request_indices),
        ("query_positions", query_positions),
    ):
        if value.shape != (token_count,) or value.dtype != torch.int32:
            raise ValueError(
                f"{name} must be int32 with shape {(token_count,)}"
            )
    if req_to_token_slots.ndim != 2 or req_to_token_slots.dtype != torch.int32:
        raise ValueError("req_to_token_slots must be a rank-2 int32 tensor")
    if type(producer_layer) is not int or producer_layer < 0:
        raise ValueError("producer_layer must be a nonnegative Python int")

    request_count, max_request_tokens = req_to_token_slots.shape
    compact_logical = torch.full_like(logical_topk_ids, -1)
    compact_physical = torch.zeros_like(logical_topk_ids)
    compact_counts = torch.zeros((token_count,), dtype=torch.int32)

    for token in range(token_count):
        count = min(max(int(selected_counts[token].item()), 0), topk_width)
        request_index = int(query_request_indices[token].item())
        query_position = int(query_positions[token].item())
        seen: set[int] = set()
        valid_logical: list[int] = []
        valid_physical: list[int] = []

        for rank in range(count):
            logical_id = int(logical_topk_ids[token, rank].item())
            duplicate = logical_id in seen
            seen.add(logical_id)
            if duplicate:
                continue
            if request_index < 0 or request_index >= request_count:
                continue
            if logical_id < 0 or logical_id >= max_request_tokens:
                continue
            if logical_id > query_position:
                continue
            physical_slot = int(
                req_to_token_slots[request_index, logical_id].item()
            )
            if physical_slot < 0:
                continue
            valid_logical.append(logical_id)
            valid_physical.append(physical_slot)

        compact_count = len(valid_logical)
        if compact_count:
            compact_logical[token, :compact_count] = torch.tensor(
                valid_logical, dtype=torch.int32
            )
            compact_physical[token, :compact_count] = torch.tensor(
                valid_physical, dtype=torch.int32
            )
        compact_counts[token] = compact_count

    return TorchDsaSelection(
        physical_slots=compact_physical,
        selected_counts=compact_counts,
        producer_layer=producer_layer,
        logical_topk_ids=compact_logical,
    )


def _align_to_128(dim: int) -> int:
    return ((dim + 127) // 128) * 128


def torch_dsa_sparse_mla(
    q_latent: torch.Tensor,
    q_rope: torch.Tensor,
    cache: torch.Tensor,
    physical_slots: torch.Tensor,
    selected_counts: torch.Tensor,
    *,
    sm_scale: float | torch.Tensor,
    page_size: int,
    latent_dim: int,
    rope_dim: int,
) -> torch.Tensor:
    """Gather counted packed-cache slots and compute FP32 sparse MLA."""
    for name, value in (
        ("q_latent", q_latent),
        ("q_rope", q_rope),
        ("cache", cache),
        ("physical_slots", physical_slots),
        ("selected_counts", selected_counts),
    ):
        _require_cpu(name, value)

    if cache.ndim != 4:
        raise ValueError(f"MLA cache must have rank 4; got {cache.ndim}")
    if page_size <= 0 or latent_dim <= 0 or rope_dim <= 0:
        raise ValueError("page_size, latent_dim, and rope_dim must be positive")
    packed_rows_per_page = cache.shape[1] * cache.shape[2]
    if packed_rows_per_page < page_size:
        raise ValueError(
            "MLA cache row/lane axes are too small for page_size; got "
            f"shape={cache.shape}, page_size={page_size}"
        )
    latent_aligned = _align_to_128(latent_dim)
    rope_aligned = _align_to_128(rope_dim)
    required_width = latent_aligned + rope_aligned
    if cache.shape[3] < required_width:
        raise ValueError(
            f"MLA cache width {cache.shape[3]} is smaller than required "
            f"{required_width}"
        )
    if not torch.is_floating_point(cache):
        raise TypeError(f"MLA cache must have floating dtype; got {cache.dtype}")

    if q_latent.ndim != 3 or q_latent.shape[-1] != latent_dim:
        raise ValueError(
            f"q_latent must have shape [tokens, heads, {latent_dim}]; "
            f"got {q_latent.shape}"
        )
    if q_rope.ndim != 3 or q_rope.shape[-1] != rope_dim:
        raise ValueError(
            f"q_rope must have shape [tokens, heads, {rope_dim}]; "
            f"got {q_rope.shape}"
        )
    if q_latent.shape[:2] != q_rope.shape[:2]:
        raise ValueError(
            "q_latent and q_rope must have matching token and head dimensions"
        )
    if not torch.is_floating_point(q_latent) or not torch.is_floating_point(
        q_rope
    ):
        raise TypeError("q_latent and q_rope must have floating dtypes")

    token_count, head_count, _ = q_latent.shape
    if physical_slots.ndim != 2 or physical_slots.shape[0] != token_count:
        raise ValueError("physical_slots must have shape [tokens, max_selected]")
    if physical_slots.dtype != torch.int32:
        raise TypeError(
            f"physical_slots must have dtype int32; got {physical_slots.dtype}"
        )
    if selected_counts.shape != (token_count,):
        raise ValueError(
            f"selected_counts must have shape {(token_count,)}"
        )
    if selected_counts.dtype != torch.int32:
        raise TypeError(
            f"selected_counts must have dtype int32; got {selected_counts.dtype}"
        )
    max_selected = physical_slots.shape[1]
    if max_selected == 0:
        raise ValueError("physical_slots must reserve at least one slot per token")

    scale = torch.as_tensor(sm_scale, dtype=torch.float32)
    if scale.ndim != 0:
        raise ValueError("sm_scale must be a scalar")
    capacity = cache.shape[0] * page_size
    concrete_counts = selected_counts.tolist()
    if any(count < 0 or count > max_selected for count in concrete_counts):
        raise ValueError(
            "selected_counts entries must be in [0, max_selected]"
        )
    for token, count in enumerate(concrete_counts):
        counted_slots = physical_slots[token, :count]
        if bool(((counted_slots < 0) | (counted_slots >= capacity)).any()):
            raise ValueError(
                "counted physical_slots must be valid cache addresses; "
                f"token={token}, count={count}"
            )

    token_rows = cache.reshape(
        cache.shape[0], packed_rows_per_page, cache.shape[3]
    )[:, :page_size].reshape(capacity, cache.shape[3]).float()
    output = torch.zeros(
        (token_count, head_count, latent_dim), dtype=torch.float32
    )
    for token, count in enumerate(concrete_counts):
        if count == 0:
            continue
        rows = token_rows.index_select(
            0, physical_slots[token, :count].long()
        )
        selected_latent = rows[:, :latent_dim]
        selected_rope = rows[
            :, latent_aligned : latent_aligned + rope_dim
        ]
        query = torch.cat(
            (q_latent[token].float(), q_rope[token].float()), dim=-1
        )
        keys = torch.cat((selected_latent, selected_rope), dim=-1)
        scores = torch.einsum("hc,kc->hk", query, keys) * scale
        weights = torch.softmax(scores, dim=-1)
        output[token] = torch.einsum(
            "hk,kc->hc", weights, selected_latent
        )
    return output
