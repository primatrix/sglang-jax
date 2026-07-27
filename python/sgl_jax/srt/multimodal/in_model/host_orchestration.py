"""Common host orchestration for in-model multimodal embedding."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any, Protocol

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec

from sgl_jax.srt.multimodal.common.modality_enum import Modality, MultimodalInputs
from sgl_jax.srt.multimodal.in_model.plan import MultimodalEmbedPlan
from sgl_jax.srt.multimodal.in_model.registry import resolve_encoder_plan_builder

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from jax.sharding import Mesh
    from jax.typing import ArrayLike

    from sgl_jax.srt.configs.model_config import ModelConfig
    from sgl_jax.srt.managers.schedule_batch import ScheduleReqsInfo


class _LanguageModel(Protocol):
    def get_input_embeddings(self) -> Callable[[jax.Array], jax.Array]: ...


class _MultimodalModel(Protocol):
    mesh: Mesh

    def get_multimodal_encoder(
        self, modality: Modality
    ) -> Callable[[Any], jax.Array | tuple[jax.Array, jax.Array]]: ...


class _ForwardBatch(Protocol):
    input_embedding: jax.Array | None


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
    input_buckets: Sequence[int] | None = None,
) -> MultimodalEmbedPlan | None:
    """Build a multimodal encode/merge plan."""
    if not _has_in_model_multimodal_inputs(reqs_info):
        return None
    builder = resolve_encoder_plan_builder(model_config, input_buckets=input_buckets)
    if builder is None:
        return None
    return builder.build(reqs_info, dp_size, per_dp_token, tp_size)


def _merge_in_specs(encoder_tp: bool) -> tuple[PartitionSpec, ...]:
    """Merge in-specs. DP-Encoder shards lanes over ``"tensor"``; TP-Encoder
    replicates them (one collaborative lane per DP rank)."""
    tp = None if encoder_tp else "tensor"
    return (
        PartitionSpec("data", None),  # running   [total_tok, H]
        PartitionSpec("data", tp, None, None),  # features [dp,tp,encoder_output_length,H]
        PartitionSpec("data", tp, None),  # src_idx [dp,tp,per_dp_token]
        PartitionSpec("data", tp, None),  # mask [dp,tp,per_dp_token]
    )


@functools.partial(jax.jit, static_argnames=("mesh", "encoder_tp"))
def merge_jit(
    mesh: Mesh,
    running: ArrayLike,
    features: ArrayLike,
    src_idx: ArrayLike,
    mask: ArrayLike,
    encoder_tp: bool = False,
) -> jax.Array:
    """Gather per-lane encoder features directly into DP token embedding rows.

    Routing is destination-driven: its final axis is the local token axis, so
    there is no independent merge shape. DP-Encoder fans requests across tensor
    lanes and combines their token-aligned updates with ``psum``. TP-Encoder has
    one tensor-replicated lane per DP rank and needs no final combine.
    """

    def merge_local(
        running: jax.Array,
        features: jax.Array,
        src_idx: jax.Array,
        mask: jax.Array,
    ) -> jax.Array:
        lane_features = features[0, 0]
        lane_src = src_idx[0, 0]
        lane_mask = mask[0, 0]

        safe_src = jnp.where(lane_mask, lane_src, 0)
        updates = jnp.where(lane_mask[:, None], lane_features[safe_src], 0)
        modality_mask = lane_mask.astype(jnp.int32)
        if not encoder_tp:
            updates = jnp.asarray(jax.lax.psum(updates, "tensor"))
            modality_mask = jnp.asarray(jax.lax.psum(modality_mask, "tensor"))
        return jnp.where((modality_mask > 0)[:, None], updates, running)

    return jax.shard_map(
        merge_local,
        mesh=mesh,
        in_specs=_merge_in_specs(encoder_tp),
        out_specs=PartitionSpec("data", None),
        check_vma=False,
    )(running, features, src_idx, mask)


def _deepstack_merge_in_specs(
    encoder_tp: bool, *, has_base: bool
) -> tuple[PartitionSpec | None, ...]:
    """Specs for one main-feature merge plus every deepstack layer."""
    tp = None if encoder_tp else "tensor"
    specs: list[PartitionSpec | None] = [
        PartitionSpec("data", None),  # running [total_tok,H]
    ]
    if has_base:
        specs.append(PartitionSpec(None, "data", None))  # [layers,total_tok,H]
    specs.extend(
        (
            PartitionSpec("data", tp, None, None),  # features [dp,tp,capacity,H]
            PartitionSpec("data", tp, None, None, None),  # [dp,tp,layers,capacity,H]
            PartitionSpec("data", tp, None),  # src_idx [dp,tp,per_dp_token]
            PartitionSpec("data", tp, None),  # mask [dp,tp,per_dp_token]
        )
    )
    return tuple(specs)


@functools.partial(jax.jit, static_argnames=("mesh", "encoder_tp"))
def merge_with_deepstack_jit(
    mesh: Mesh,
    running: ArrayLike,
    deepstack_running: ArrayLike | None,
    features: ArrayLike,
    deepstack_features: ArrayLike,
    src_idx: ArrayLike,
    mask: ArrayLike,
    encoder_tp: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Merge the main encoder output and every deepstack layer in one launch."""

    def merge_local(
        running: jax.Array,
        deepstack_running: jax.Array,
        features: jax.Array,
        deepstack_features: jax.Array,
        src_idx: jax.Array,
        mask: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        lane_features = features[0, 0]
        lane_deepstack = deepstack_features[0, 0]
        lane_src = src_idx[0, 0]
        lane_mask = mask[0, 0]

        safe_src = jnp.where(lane_mask, lane_src, 0)
        updates = jnp.where(
            lane_mask[:, None],
            lane_features[safe_src],
            0,
        )
        deepstack_updates = jnp.where(
            lane_mask[None, :, None],
            jnp.take(lane_deepstack, safe_src, axis=1),
            0,
        )
        modality_mask = lane_mask.astype(jnp.int32)
        if not encoder_tp:
            updates = jnp.asarray(jax.lax.psum(updates, "tensor"))
            deepstack_updates = jnp.asarray(jax.lax.psum(deepstack_updates, "tensor"))
            modality_mask = jnp.asarray(jax.lax.psum(modality_mask, "tensor"))

        active = modality_mask > 0
        return (
            jnp.where(active[:, None], updates, running),
            jnp.where(
                active[None, :, None],
                deepstack_updates,
                deepstack_running,
            ),
        )

    out_specs = (
        PartitionSpec("data", None),
        PartitionSpec(None, "data", None),
    )
    if deepstack_running is None:

        def merge_without_base(
            running: jax.Array,
            features: jax.Array,
            deepstack_features: jax.Array,
            src_idx: jax.Array,
            mask: jax.Array,
        ) -> tuple[jax.Array, jax.Array]:
            deepstack_base = jnp.zeros(
                (deepstack_features.shape[2], *running.shape),
                dtype=running.dtype,
            )
            return merge_local(
                running,
                deepstack_base,
                features,
                deepstack_features,
                src_idx,
                mask,
            )

        return jax.shard_map(
            merge_without_base,
            mesh=mesh,
            in_specs=_deepstack_merge_in_specs(encoder_tp, has_base=False),
            out_specs=out_specs,
            check_vma=False,
        )(running, features, deepstack_features, src_idx, mask)

    return jax.shard_map(
        merge_local,
        mesh=mesh,
        in_specs=_deepstack_merge_in_specs(encoder_tp, has_base=True),
        out_specs=out_specs,
        check_vma=False,
    )(
        running,
        deepstack_running,
        features,
        deepstack_features,
        src_idx,
        mask,
    )


def _flatten_device_batch(value: jax.Array, *, out_sharding: NamedSharding) -> jax.Array:
    return value.reshape(
        value.shape[0] * value.shape[1], *value.shape[2:], out_sharding=out_sharding
    )


def _encode_inputs_lane_shape(encode_inputs: Any) -> tuple[int, int]:
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
    mm_embed_plan: MultimodalEmbedPlan,
    input_ids: jax.Array,
    input_embedding: Callable[[jax.Array], jax.Array],
    multimodal_model: _MultimodalModel,
    *,
    return_deepstack: bool = False,
) -> jax.Array | tuple[jax.Array, jax.Array | None]:
    """Encode each fixed-shape modality batch and merge it into token embeddings.

    ``running`` starts as the plain text embedding. Each modality batch encodes
    and merges directly. Returns ``[total_token, H]``.

    ``multimodal_model.encoder_tp`` selects the merge/write shardings: DP-Encoder
    shards the ``tp`` lanes over ``"tensor"``; TP-Encoder keeps a single
    tensor-replicated lane per DP rank.
    """
    mesh = multimodal_model.mesh
    encoder_tp = getattr(multimodal_model, "encoder_tp", False)
    tp_axis = None if encoder_tp else "tensor"
    running = input_embedding(input_ids)
    deepstack_running = None
    for modality, batch in mm_embed_plan.items():
        encoder = multimodal_model.get_multimodal_encoder(modality)
        device_inputs = batch.encode_inputs
        dp_size, tp_size = _encode_inputs_lane_shape(device_inputs)
        flatten_device_batch = functools.partial(
            _flatten_device_batch,
            out_sharding=NamedSharding(
                mesh,
                PartitionSpec("data" if encoder_tp else ("data", "tensor")),
            ),
        )

        # [dp,tp,...] lanes -> flat [dp*tp,...] batch for the ViT
        model_inputs = jax.tree.map(flatten_device_batch, device_inputs)
        encoded = encoder(model_inputs)
        features, deepstack = encoded if isinstance(encoded, tuple) else (encoded, None)
        feature_shape = (dp_size, tp_size, *features.shape[1:])
        features = jax.lax.reshape(
            features,
            feature_shape,
            out_sharding=NamedSharding(
                mesh,
                PartitionSpec("data", tp_axis, *([None] * (len(feature_shape) - 2))),
            ),
        )

        if (
            batch.encoder_output_length is not None
            and features.shape[2] != batch.encoder_output_length
        ):
            raise ValueError(
                "Encoder output length does not match the plan: "
                f"got {features.shape[2]}, expected "
                f"{batch.encoder_output_length}."
            )

        if deepstack is not None:
            layer_shape = (dp_size, tp_size, *deepstack.shape[1:])
            deepstack = jax.lax.reshape(
                deepstack,
                layer_shape,
                out_sharding=NamedSharding(
                    mesh,
                    PartitionSpec("data", tp_axis, *([None] * (len(layer_shape) - 2))),
                ),
            )
            if (
                batch.encoder_output_length is not None
                and deepstack.shape[3] != batch.encoder_output_length
            ):
                raise ValueError(
                    "Deepstack output length does not match the plan: "
                    f"got {deepstack.shape[3]}, expected "
                    f"{batch.encoder_output_length}."
                )
            running, deepstack_running = merge_with_deepstack_jit(
                mesh,
                running,
                deepstack_running,
                features,
                deepstack,
                batch.merge.src_idx,
                batch.merge.mask,
                encoder_tp=encoder_tp,
            )
        else:
            running = merge_jit(
                mesh,
                running,
                features,
                batch.merge.src_idx,
                batch.merge.mask,
                encoder_tp=encoder_tp,
            )
    return (running, deepstack_running) if return_deepstack else running


def general_mm_embed_routine(
    input_ids: jax.Array,
    forward_batch: _ForwardBatch,
    language_model: _LanguageModel,
    multimodal_model: _MultimodalModel,
    mm_embed_plan: MultimodalEmbedPlan,
) -> None:
    """Populate ``forward_batch.input_embedding`` for multimodal prefill.

    The language backbone consumes this fused embedding instead of re-embedding
    ``input_ids``.
    """
    embed_tokens = language_model.get_input_embeddings()
    input_embeds, deepstack = embed_mm_inputs(
        mm_embed_plan,
        input_ids,
        embed_tokens,
        multimodal_model,
        return_deepstack=True,
    )
    forward_batch.input_embedding = input_embeds
    forward_batch.deepstack_visual_embedding = deepstack
    forward_batch.apply_for_deepstack = deepstack is not None
