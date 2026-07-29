"""Resolve multimodal items and merge them into their owning DP batch."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from functools import cache, partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from jax.typing import ArrayLike

from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.models.registry import ModelRegistry
from sgl_jax.srt.multimodal.common.modality_enum import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sgl_jax.srt.multimodal.in_model.interface import (
    InModelMultimodalContract,
    MultimodalEmbedding,
    MultimodalEmbeddingFunc,
    MultimodalEmbeddingOutput,
    MultimodalItemGroups,
    PackedMultimodalEmbeddingFunc,
)
from sgl_jax.srt.multimodal.in_model.placement import (
    dp_local_replicated_sharding,
    dp_submesh,
    place_on_dp,
)


@dataclass(frozen=True)
class _MergeMapping:
    source_start: int
    destination_start: int
    length: int


@dataclass(frozen=True)
class ItemTask:
    """Runtime-only item ownership and merge metadata."""

    item: MultimodalDataItem
    owner_dp: int
    output_len: int
    merge_mappings: tuple[_MergeMapping, ...]


_MultimodalBatch = dict[Modality, tuple[ItemTask, ...]]
_CacheKey = tuple[int, Modality, int]


@dataclass(frozen=True)
class _ItemEmbedding:
    embeddings: jax.Array
    deepstack: jax.Array | None

    @property
    def nbytes(self) -> int:
        return self.embeddings.nbytes + (self.deepstack.nbytes if self.deepstack is not None else 0)


@dataclass(frozen=True)
class ResolvedItem:
    task: ItemTask
    value: _ItemEmbedding


@dataclass(frozen=True)
class DPLocalMergeBatch:
    source: jax.Array
    src_idx: jax.Array
    mask: jax.Array
    deepstack: jax.Array | None = None


def encode_multimodal_items(
    items_by_owner: MultimodalItemGroups,
    encoder: MultimodalEmbeddingFunc,
) -> MultimodalEmbedding:
    return encoder([item for items in items_by_owner for item in items])


def _normalize_to_owner(
    value: _ItemEmbedding,
    mesh: Mesh | None,
    owner_dp: int,
) -> _ItemEmbedding:
    return _ItemEmbedding(
        place_on_dp(value.embeddings, mesh, owner_dp),
        (place_on_dp(value.deepstack, mesh, owner_dp) if value.deepstack is not None else None),
    )


class _MultimodalEmbeddingCache:
    """Byte-bounded LRU of complete owner-local encoder outputs."""

    def __init__(self, max_bytes: int) -> None:
        self.max_bytes = max_bytes
        self.size_bytes = 0
        self._entries: OrderedDict[_CacheKey, _ItemEmbedding] = OrderedDict()

    def get(self, key: _CacheKey) -> _ItemEmbedding | None:
        value = self._entries.pop(key, None)
        if value is not None:
            self._entries[key] = value
        return value

    def put(self, key: _CacheKey, value: _ItemEmbedding) -> None:
        if value.nbytes > self.max_bytes:
            return
        while self._entries and self.size_bytes + value.nbytes > self.max_bytes:
            _, evicted = self._entries.popitem(last=False)
            self.size_bytes -= evicted.nbytes
        self._entries[key] = value
        self.size_bytes += value.nbytes

    def clear(self) -> None:
        for item in self._entries.values():
            item.embeddings.delete()
            if item.deepstack is not None:
                item.deepstack.delete()
        self._entries.clear()
        self.size_bytes = 0


def _build_item_task(
    item: MultimodalDataItem,
    token_base: int,
    chunk_start: int,
    chunk_end: int,
    dp_rank: int,
) -> ItemTask | None:
    mappings: list[_MergeMapping] = []
    output_len = 0
    for start, end in item.placeholder_ranges or ():
        overlap_start = max(start, chunk_start)
        overlap_end = min(end, chunk_end)
        if overlap_start < overlap_end:
            mappings.append(
                _MergeMapping(
                    source_start=output_len + overlap_start - start,
                    destination_start=token_base + overlap_start - chunk_start,
                    length=overlap_end - overlap_start,
                )
            )
        output_len += end - start
    return ItemTask(item, dp_rank, output_len, tuple(mappings)) if mappings else None


def build_multimodal_batch(
    reqs_info: list | None,
    dp_size: int,
    model_config: ModelConfig,
    per_dp_token: int,
) -> _MultimodalBatch | None:
    """Build tasks for placeholders visible in this prefill chunk."""

    grouped: dict[Modality, list[ItemTask]] = {}
    for dp_rank, info in enumerate((reqs_info or ())[:dp_size]):
        request_base = dp_rank * per_dp_token
        for req_index, req in enumerate(info.reqs or ()):
            prefix_len = (
                info.prefix_lens[req_index]
                if info.prefix_lens is not None
                else len(getattr(req, "prefix_indices", ()))
            )
            extend_len = (
                info.extend_lens[req_index]
                if info.extend_lens is not None
                else getattr(req, "extend_input_len", 0)
            )
            if isinstance(req.mm_inputs, MultimodalInputs):
                for item in req.mm_inputs.mm_items:
                    task = _build_item_task(
                        item,
                        request_base,
                        prefix_len,
                        prefix_len + extend_len,
                        dp_rank,
                    )
                    if task is not None:
                        grouped.setdefault(item.modality, []).append(task)
            request_base += extend_len

    if not grouped or not ModelRegistry.is_in_model_multimodal(
        model_config.hf_config.architectures
    ):
        return None
    return {modality: tuple(entries) for modality, entries in grouped.items()}


def _split_embeddings(
    embeddings: ArrayLike | Sequence[ArrayLike],
    tasks: tuple[ItemTask, ...],
    axis: int,
) -> list[ArrayLike]:
    def replicated_get(value: ArrayLike, index: object = Ellipsis) -> ArrayLike:
        if (
            isinstance(value, jax.Array)
            and isinstance(value.sharding, NamedSharding)
            and not value.sharding.is_fully_replicated
        ):
            replicated = NamedSharding(value.sharding.mesh, PartitionSpec())
            return value.at[index].get(out_sharding=replicated)
        return value if index is Ellipsis else value[index]

    expected_lengths = [task.output_len for task in tasks]
    if not isinstance(embeddings, (jax.Array, np.ndarray)):
        outputs = list(embeddings)
        actual_lengths = [np.shape(value)[axis] for value in outputs]
        if len(outputs) != len(tasks) or actual_lengths != expected_lengths:
            raise ValueError(
                f"encoder output lengths {actual_lengths} do not match {expected_lengths}"
            )
        return [replicated_get(value) for value in outputs]
    expected_total = sum(expected_lengths)
    if embeddings.shape[axis] != expected_total:
        raise ValueError(
            f"encoder output length {embeddings.shape[axis]} does not match {expected_total}"
        )
    outputs = []
    offset = 0
    for task in tasks:
        end = offset + task.output_len
        index = [slice(None)] * embeddings.ndim
        index[axis] = slice(offset, end)
        outputs.append(replicated_get(embeddings, tuple(index)))
        offset = end
    return outputs


def resolve_items(
    tasks: tuple[ItemTask, ...],
    encoder: PackedMultimodalEmbeddingFunc,
    cache: _MultimodalEmbeddingCache | None,
    mesh: Mesh | None,
) -> tuple[ResolvedItem, ...]:
    """Resolve/cache complete items, issuing at most one encoder dispatch."""

    keys = []
    for task in tasks:
        item = task.item
        if item.hash is None:
            item.set_pad_value()
        keys.append(
            (
                task.owner_dp,
                item.modality,
                item.hash,
            )
        )
    values: dict[_CacheKey, _ItemEmbedding] = {}
    misses: dict[_CacheKey, ItemTask] = {}
    for key, task in zip(keys, tasks, strict=True):
        item = task.item
        if key in values or key in misses:
            continue
        if item.precomputed_embeddings is not None:
            (embedding,) = _split_embeddings(
                [item.precomputed_embeddings],
                (task,),
                0,
            )
            values[key] = _normalize_to_owner(
                _ItemEmbedding(jnp.asarray(embedding), None),
                mesh,
                task.owner_dp,
            )
            continue
        cached = cache.get(key) if cache is not None else None
        if cached is None:
            misses[key] = task
        else:
            values[key] = cached

    if misses:
        dp_size = (
            int(mesh.shape["data"])
            if mesh is not None and "data" in mesh.shape
            else max(task.owner_dp for task in misses.values()) + 1
        )
        misses_by_owner: list[list[tuple[_CacheKey, ItemTask]]] = [[] for _ in range(dp_size)]
        for key, task in misses.items():
            misses_by_owner[task.owner_dp].append((key, task))

        owner_misses = tuple(entry for group in misses_by_owner for entry in group)
        owner_tasks = tuple(task for _, task in owner_misses)
        output = encoder(tuple(tuple(task.item for _, task in group) for group in misses_by_owner))
        if not isinstance(output, MultimodalEmbeddingOutput):
            output = MultimodalEmbeddingOutput(output)
        embeddings = _split_embeddings(output.embeddings, owner_tasks, 0)
        deepstack = (
            _split_embeddings(output.deepstack, owner_tasks, 1)
            if output.deepstack is not None
            else None
        )
        stacked_values = deepstack or [None] * len(embeddings)
        for (key, task), embedding, stacked in zip(
            owner_misses,
            embeddings,
            stacked_values,
            strict=True,
        ):
            embedding = (
                jnp.asarray(embedding, copy=True)
                if cache is not None and isinstance(embedding, np.ndarray)
                else jnp.asarray(embedding)
            )
            stacked = (
                jnp.asarray(stacked, copy=True)
                if cache is not None and isinstance(stacked, np.ndarray)
                else jnp.asarray(stacked)
                if stacked is not None
                else None
            )
            value = _normalize_to_owner(
                _ItemEmbedding(embedding, stacked),
                mesh,
                task.owner_dp,
            )
            values[key] = value
            if cache is not None:
                cache.put(key, value)

    return tuple(ResolvedItem(task, values[key]) for key, task in zip(keys, tasks, strict=True))


def _merge_dp_lanes(
    target: jax.Array,
    source: jax.Array,
    src_idx: jax.Array,
    mask: jax.Array,
    token_axis: int,
) -> jax.Array:
    """Merge explicit DP source lanes into a flattened target token axis."""

    dp_size, per_dp_tokens = source.shape[token_axis : token_axis + 2]
    target_by_dp = target.reshape(
        (
            *target.shape[:token_axis],
            dp_size,
            per_dp_tokens,
            *target.shape[token_axis + 1 :],
        )
    )

    def merge_lane(lane_target, lane_source, lane_src_idx, lane_mask):
        gathered = jnp.take(lane_source, lane_src_idx, axis=token_axis)
        mask_shape = tuple(
            lane_mask.shape[0] if i == token_axis else 1 for i in range(lane_target.ndim)
        )
        return jnp.where(lane_mask.reshape(mask_shape), gathered, lane_target)

    merged = jax.vmap(merge_lane)(
        jnp.moveaxis(target_by_dp, token_axis, 0),
        jnp.moveaxis(source, token_axis, 0),
        src_idx,
        mask,
    )
    merged = jnp.moveaxis(merged, 0, token_axis)
    return merged.reshape(target.shape)


@cache
def _dp_local_merge_kernel(
    mesh: Mesh,
    target_spec: PartitionSpec,
    source_spec: PartitionSpec,
    token_axis: int,
):
    return jax.jit(
        jax.shard_map(
            lambda target, source, src_idx, mask: _merge_dp_lanes(
                target,
                source,
                src_idx,
                mask,
                token_axis,
            ),
            mesh=mesh,
            in_specs=(
                target_spec,
                source_spec,
                PartitionSpec("data", None),
                PartitionSpec("data", None),
            ),
            out_specs=target_spec,
        )
    )


def _assemble_dp_sources(
    pieces: list[list[jax.Array]],
    *,
    local_shape: tuple[int, ...],
    token_axis: int,
    dtype: jnp.dtype,
    mesh: Mesh | None,
) -> jax.Array:
    capacity = local_shape[token_axis]
    local_sources = []
    for rank, rank_pieces in enumerate(pieces):
        submesh = dp_submesh(mesh, rank) if mesh is not None else None
        with jax.set_mesh(submesh) if submesh is not None else nullcontext():
            if rank_pieces:
                source = jnp.concatenate(rank_pieces, axis=token_axis).astype(dtype)
                padding = [(0, 0)] * len(local_shape)
                padding[token_axis] = (0, capacity - source.shape[token_axis])
                source = jnp.pad(source, padding)
            else:
                source = jnp.zeros(local_shape, dtype)
            if mesh is not None:
                source = jax.device_put(
                    source,
                    dp_local_replicated_sharding(mesh, rank),
                )
        local_sources.append(source)

    if mesh is None or "data" not in mesh.shape:
        return jnp.stack(local_sources, axis=token_axis)

    global_shape = list(local_shape)
    global_shape.insert(token_axis, len(pieces))
    source_spec = [None] * len(global_shape)
    source_spec[token_axis] = "data"
    source_sharding = NamedSharding(mesh, PartitionSpec(*source_spec))
    buffers = {}
    for source in local_sources:
        for shard in source.addressable_shards:
            device_mesh = Mesh(np.asarray([shard.device]), ("device",))
            with jax.set_mesh(device_mesh):
                buffers[shard.device] = jnp.expand_dims(shard.data, token_axis)
    return jax.make_array_from_single_device_arrays(
        tuple(global_shape),
        source_sharding,
        [buffers[device] for device in source_sharding.addressable_devices],
        dtype=dtype,
    )


def lower_to_dp_merge_batch(
    items: Sequence[ResolvedItem],
    target: jax.Array,
    mesh: Mesh | None,
) -> DPLocalMergeBatch:
    dp_size = (
        int(mesh.shape["data"])
        if mesh is not None and "data" in mesh.shape
        else max((item.task.owner_dp for item in items), default=0) + 1
    )
    per_dp_tokens = target.shape[0] // dp_size
    pieces: list[list[jax.Array]] = [[] for _ in range(dp_size)]
    deepstack_pieces: list[list[jax.Array]] = [[] for _ in range(dp_size)]
    offsets = [0] * dp_size
    src_idx = jnp.zeros((dp_size, per_dp_tokens), dtype=jnp.int32)
    mask = jnp.zeros((dp_size, per_dp_tokens), dtype=jnp.bool_)
    has_deepstack = [item.value.deepstack is not None for item in items]
    if any(has_deepstack) and not all(has_deepstack):
        raise ValueError("encoder must return deepstack for either every item or no items")

    for item in items:
        task, value = item.task, item.value
        rank = task.owner_dp
        for mapping in task.merge_mappings:
            local_dst = mapping.destination_start - rank * per_dp_tokens
            source_slice = slice(
                mapping.source_start,
                mapping.source_start + mapping.length,
            )
            submesh = dp_submesh(mesh, rank) if mesh is not None else None
            with jax.set_mesh(submesh) if submesh is not None else nullcontext():
                pieces[rank].append(value.embeddings[source_slice])
                if value.deepstack is not None:
                    deepstack_pieces[rank].append(value.deepstack[:, source_slice])

            destination = slice(local_dst, local_dst + mapping.length)
            src_idx = src_idx.at[rank, destination].set(
                jnp.arange(
                    offsets[rank],
                    offsets[rank] + mapping.length,
                    dtype=jnp.int32,
                )
            )
            mask = mask.at[rank, destination].set(True)
            offsets[rank] += mapping.length

    source = _assemble_dp_sources(
        pieces,
        local_shape=(per_dp_tokens, *target.shape[1:]),
        token_axis=0,
        dtype=target.dtype,
        mesh=mesh,
    )
    if mesh is not None and "data" in mesh.shape:
        plan_sharding = NamedSharding(mesh, PartitionSpec("data", None))
        src_idx = jax.device_put(src_idx, plan_sharding)
        mask = jax.device_put(mask, plan_sharding)
    deepstack = None
    if any(has_deepstack):
        sample = items[0].value.deepstack
        assert sample is not None
        deepstack = _assemble_dp_sources(
            deepstack_pieces,
            local_shape=(sample.shape[0], per_dp_tokens, *target.shape[1:]),
            token_axis=1,
            dtype=sample.dtype,
            mesh=mesh,
        )
    return DPLocalMergeBatch(
        source,
        src_idx,
        mask,
        deepstack,
    )


def dp_local_merge(
    target: jax.Array,
    batch: DPLocalMergeBatch,
    *,
    deepstack: bool = False,
) -> jax.Array:
    source = batch.deepstack if deepstack else batch.source
    token_axis = 1 if deepstack else 0
    target_sharding = getattr(target, "sharding", None)
    source_sharding = getattr(source, "sharding", None)
    if (
        isinstance(target_sharding, NamedSharding)
        and target_sharding.spec[token_axis] == "data"
        and isinstance(source_sharding, NamedSharding)
    ):
        return _dp_local_merge_kernel(
            target_sharding.mesh,
            target_sharding.spec,
            source_sharding.spec,
            token_axis,
        )(target, source, batch.src_idx, batch.mask)
    return _merge_dp_lanes(
        target,
        source,
        batch.src_idx,
        batch.mask,
        token_axis,
    )


def general_mm_embed_routine(
    multimodal_batch: _MultimodalBatch,
    input_ids: jax.Array,
    input_embedding: Callable[[jax.Array], jax.Array],
    multimodal_model: InModelMultimodalContract,
    embedding_cache: _MultimodalEmbeddingCache | None = None,
) -> tuple[jax.Array, jax.Array | None]:
    mesh = multimodal_model.mesh
    with jax.set_mesh(mesh) if mesh is not None else nullcontext():
        running = input_embedding(input_ids)
        deepstack = None
        embedding_funcs = multimodal_model.get_multimodal_embedding_funcs()
        packed_embedding_funcs = multimodal_model.get_packed_multimodal_embedding_funcs()
        for modality, tasks in multimodal_batch.items():
            encoder = packed_embedding_funcs.get(modality)
            if encoder is None:
                encoder = partial(
                    encode_multimodal_items,
                    encoder=embedding_funcs[modality],
                )
            resolved = resolve_items(
                tasks,
                encoder,
                embedding_cache,
                mesh,
            )
            merge_batch = lower_to_dp_merge_batch(resolved, running, mesh)
            running = dp_local_merge(running, merge_batch)
            if merge_batch.deepstack is not None:
                if deepstack is None:
                    num_layers = merge_batch.deepstack.shape[0]
                    deepstack_sharding = (
                        NamedSharding(mesh, PartitionSpec(None, "data", None))
                        if mesh is not None and "data" in mesh.shape
                        else None
                    )
                    deepstack = jnp.zeros(
                        (num_layers, *running.shape),
                        dtype=merge_batch.deepstack.dtype,
                        out_sharding=deepstack_sharding,
                    )
                deepstack = dp_local_merge(deepstack, merge_batch, deepstack=True)
        return running, deepstack
