"""Common host orchestration for in-model multimodal embedding."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec

from sgl_jax.srt.multimodal.common.in_model_plan_builder import (
    resolve_in_model_plan_builder,
)
from sgl_jax.srt.multimodal.common.mm_plan import MultimodalEmbedPlan
from sgl_jax.srt.multimodal.common.modality_enum import MultimodalInputs

if TYPE_CHECKING:
    from sgl_jax.srt.configs.model_config import ModelConfig
    from sgl_jax.srt.managers.schedule_batch import ScheduleReqsInfo

_MERGE_IN_SPECS = (
    PartitionSpec("data", None),  # running   [total_tok, H]
    PartitionSpec("data", "tensor", None, None),  # features [dp,tp,out_rows,H]
    PartitionSpec("data", "tensor", None),  # src_idx [dp,tp,merge_bucket]
    PartitionSpec("data", "tensor", None),  # dst_idx [dp,tp,merge_bucket]
    PartitionSpec("data", "tensor", None),  # mask [dp,tp,merge_bucket]
)


def _has_in_model_multimodal_inputs(reqs_info: list[ScheduleReqsInfo] | None) -> bool:
    for info in reqs_info or []:
        for req in info.reqs or []:
            mm_inputs = req.mm_inputs
            if isinstance(mm_inputs, MultimodalInputs):
                if mm_inputs.mm_items:
                    return True
            elif mm_inputs is not None and not (
                isinstance(mm_inputs, dict) and "mm_items" not in mm_inputs
            ):
                return True
    return False


def build_mm_embed_plan(
    reqs_info: list[ScheduleReqsInfo] | None,
    dp_size: int,
    model_config: ModelConfig,
    per_dp_token: int,
    tp_size: int = 1,
) -> MultimodalEmbedPlan | None:
    """Build an multimodal encode/merge plan."""
    if not _has_in_model_multimodal_inputs(reqs_info):
        return None
    builder = resolve_in_model_plan_builder(model_config)
    if builder is None:
        return None
    return builder.build(reqs_info, dp_size, per_dp_token, tp_size)


@functools.partial(jax.jit, static_argnames=("mesh",))
def merge_jit(mesh, running, features, src_idx, dst_idx, mask):
    """Scatter TP-lane features into the owning DP token embedding rows."""

    def merge_local(running, features, src_idx, dst_idx, mask):
        lane_features = features[0, 0]
        lane_src = src_idx[0, 0]
        lane_dst = dst_idx[0, 0]
        lane_mask = mask[0, 0]

        safe_src = jnp.where(lane_mask, lane_src, 0)
        safe_dst = jnp.where(lane_mask, lane_dst, 0)
        updates = jnp.where(lane_mask[:, None], lane_features[safe_src], 0)

        modality_rows = jnp.zeros_like(running).at[safe_dst].add(updates)
        modality_mask = (
            jnp.zeros(running.shape[0], dtype=jnp.int32)
            .at[safe_dst]
            .add(lane_mask.astype(jnp.int32))
        )
        modality_rows = jax.lax.psum(modality_rows, "tensor")
        modality_mask = jax.lax.psum(modality_mask, "tensor") > 0
        return jnp.where(modality_mask[:, None], modality_rows, running)

    return jax.shard_map(
        merge_local,
        mesh=mesh,
        in_specs=_MERGE_IN_SPECS,
        out_specs=PartitionSpec("data", None),
        check_vma=False,
    )(running, features, src_idx, dst_idx, mask)


def _flatten_device_batch(value, *, dp_size: int, tp_size: int):
    """Collapse runtime device axes into the model's neutral batch axis."""
    return value.reshape(dp_size * tp_size, *value.shape[2:])


def _encode_inputs_lane_shape(encode_inputs) -> tuple[int, int]:
    leaves = jax.tree.leaves(encode_inputs)
    if not leaves:
        raise ValueError("Multimodal encode inputs must contain at least one array leaf.")
    if any(value.ndim < 2 for value in leaves):
        raise ValueError("Multimodal encode input leaves must have leading [dp,tp] axes.")
    lane_shape = tuple(leaves[0].shape[:2])
    if any(tuple(value.shape[:2]) != lane_shape for value in leaves[1:]):
        raise ValueError("Multimodal encode input leaves must share leading [dp,tp] axes.")
    return int(lane_shape[0]), int(lane_shape[1])


def embed_mm_inputs(
    mm_embed_plan,
    input_ids,
    input_embedding,
    multimodal_model,
):
    """Encode each fixed-shape modality batch and merge it into token embeddings.

    ``running`` starts as the plain text embedding. Each batch encodes the items
    assigned to all ``[dp,tp]`` lanes, then replaces their owning placeholder
    rows. Returns the merged embedding ``[total_token, H]``.
    """
    mesh = multimodal_model.mesh
    running = input_embedding(input_ids)
    for modality, batch in mm_embed_plan.items():
        embedder = getattr(multimodal_model, f"get_{modality.name.lower()}_feature", None)
        assert embedder is not None, f"no embedding method for {modality}"
        device_inputs = batch.encode_inputs
        dp_size, tp_size = _encode_inputs_lane_shape(device_inputs)
        flatten_device_batch = functools.partial(
            _flatten_device_batch,
            dp_size=dp_size,
            tp_size=tp_size,
        )

        model_inputs = jax.tree.map(flatten_device_batch, device_inputs)
        features = embedder(model_inputs)
        features = features.reshape(dp_size, tp_size, *features.shape[1:])
        merge = batch.merge
        running = merge_jit(
            mesh,
            running,
            features,
            merge.src_idx,
            merge.dst_idx,
            merge.mask,
        )
    return running


def general_mm_embed_routine(
    input_ids,
    forward_batch,
    language_model,
    multimodal_model,
    mm_embed_plan,
):
    """Populate ``forward_batch.input_embedding`` for multimodal prefill.

    The language backbone consumes this fused embedding instead of re-embedding
    ``input_ids``.
    """
    embed_tokens = language_model.get_input_embeddings()
    input_embeds = embed_mm_inputs(
        mm_embed_plan,
        input_ids,
        embed_tokens,
        multimodal_model,
    )
    forward_batch.input_embedding = input_embeds
