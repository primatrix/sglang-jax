from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.models.registry import ModelRegistry
from sgl_jax.srt.multimodal.common.modality_enum import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sgl_jax.srt.multimodal.in_model.embedding_pool import (
    EmbeddingLease,
    EmbeddingPool,
    EmbeddingPoolEntry,
)
from sgl_jax.srt.multimodal.in_model.interface import InModelMultimodalContract
from sgl_jax.srt.multimodal.in_model.lane_packing import balance_lanes


@dataclass(frozen=True)
class MergeMapping:
    source_start: int
    destination_start: int
    length: int


@dataclass(frozen=True)
class ItemTask:
    item: MultimodalDataItem
    output_len: int
    merge_mappings: list[MergeMapping]

    @property
    def has_unmerged_tail(self) -> bool:
        last = self.merge_mappings[-1]
        return last.source_start + last.length < self.output_len


@dataclass(frozen=True)
class ReadyEmbedding:
    task: ItemTask
    source: EmbeddingLease


@dataclass
class MultimodalBatch:
    per_lane_tasks: list[dict[Modality, list[ItemTask]]]
    ready_embeddings: list[ReadyEmbedding]

    def release(self):
        """Release reservations when a queued plan is cancelled or consumed."""
        for ready in self.ready_embeddings:
            ready.source.release()


def _build_item_task(
    item: MultimodalDataItem,
    token_base: int,
    chunk_start: int,
    chunk_end: int,
) -> ItemTask | None:
    mappings: list[MergeMapping] = []
    output_len = 0
    for start, end in item.placeholder_ranges or []:
        overlap_start = max(start, chunk_start)
        overlap_end = min(end, chunk_end)
        if overlap_start < overlap_end:
            mappings.append(
                MergeMapping(
                    source_start=output_len + overlap_start - start,
                    destination_start=token_base + overlap_start - chunk_start,
                    length=overlap_end - overlap_start,
                )
            )
        output_len += end - start
    return ItemTask(item, output_len, mappings) if mappings else None


def build_multimodal_batch(
    reqs_info: list | None,
    dp_size: int,
    model_config: ModelConfig,
    per_dp_token: int,
    embedding_pool: EmbeddingPool | None = None,
    num_encoder_lanes: int = 1,
) -> MultimodalBatch | None:
    """Reserve ready embeddings and balance cache misses for this prefill chunk."""
    if num_encoder_lanes < 1:
        raise ValueError("num_encoder_lanes must be positive")
    if reqs_info is None or not ModelRegistry.is_in_model_multimodal(
        model_config.hf_config.architectures
    ):
        return None

    grouped: dict[Modality, list[ItemTask]] = {}
    for dp_rank, info in enumerate(reqs_info[:dp_size]):
        request_base = dp_rank * per_dp_token
        for req_index, req in enumerate(info.reqs or []):
            prefix_len = (
                info.prefix_lens[req_index]
                if info.prefix_lens is not None
                else len(getattr(req, "prefix_indices", []))
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
                    )
                    if task is not None:
                        grouped.setdefault(item.modality, []).append(task)
            request_base += extend_len

    if not grouped:
        return None
    result = MultimodalBatch([{} for _ in range(num_encoder_lanes)], [])
    try:
        for modality, tasks in grouped.items():
            misses = []
            for task in tasks:
                item = task.item
                if item.hash is None:
                    item.set_pad_value()
                lease = embedding_pool.acquire(item.hash) if embedding_pool is not None else None
                if lease is not None:
                    result.ready_embeddings.append(ReadyEmbedding(task, lease))
                else:
                    misses.append(task)
            if misses:
                lengths = [int(task.item.feature.shape[0]) for task in misses]
                lanes = balance_lanes(lengths, num_encoder_lanes)
                for lane_id, indices in enumerate(lanes):
                    if indices:
                        result.per_lane_tasks[lane_id][modality] = [
                            misses[i] for i in sorted(indices)
                        ]
        return result
    except BaseException:
        result.release()
        raise


@partial(jax.jit, static_argnames="out_sharding")
def _gather_overlay(
    running: jax.Array,
    source: jax.Array,
    pos_idx: jax.Array,
    mask: jax.Array,
    *,
    out_sharding: NamedSharding | None,
) -> jax.Array:
    if out_sharding is None:
        gathered = source[pos_idx]
    else:
        gathered = source.at[pos_idx].get(out_sharding=out_sharding)
    return jnp.where(mask[:, None], gathered, running)


def _build_gather_indices(
    tasks: list[ItemTask],
    num_tokens: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Map each destination token to its item-ordered packed source row."""

    pos_idx = np.zeros(num_tokens, dtype=np.int32)
    mask = np.zeros(num_tokens, dtype=np.bool_)
    source_offset = 0
    for task in tasks:
        for mapping in task.merge_mappings:
            dst, length = mapping.destination_start, mapping.length
            if dst < 0 or dst + length > num_tokens:
                raise ValueError("multimodal merge slice exceeds the token batch")
            span = slice(dst, dst + length)
            pos_idx[span] = source_offset + mapping.source_start + np.arange(length, dtype=np.int32)
            mask[span] = True
        source_offset += task.output_len
    return pos_idx, mask


def _place_token_vector(vector: np.ndarray, running: jax.Array, mesh: Mesh | None) -> jax.Array:
    """Shard a ``[T]`` index/mask vector like ``running``'s token axis."""

    if mesh is not None and isinstance(running.sharding, NamedSharding):
        token_spec = running.sharding.spec[0] if running.sharding.spec else None
        return jax.device_put(vector, NamedSharding(mesh, PartitionSpec(token_spec)))
    return jnp.asarray(vector)


def _apply_gather(
    running: jax.Array,
    source: jax.Array,
    pos_idx: np.ndarray,
    mask: np.ndarray,
    mesh: Mesh | None,
) -> jax.Array:
    if not mask.any():
        return running

    pos_dev = _place_token_vector(pos_idx, running, mesh)
    mask_dev = _place_token_vector(mask, running, mesh)
    sharded = mesh is not None and isinstance(running.sharding, NamedSharding)
    out_sharding = running.sharding if sharded else None
    return _gather_overlay(
        running,
        source,
        pos_dev,
        mask_dev,
        out_sharding=out_sharding,
    )


def _gather_merge(
    running: jax.Array,
    packed: jax.Array,
    tasks: list[ItemTask],
    mesh: Mesh | None,
) -> jax.Array:
    expected_width = running.shape[-1]
    min_capacity = sum(task.output_len for task in tasks)
    if packed.ndim != 2 or packed.shape[1] != expected_width or packed.shape[0] < min_capacity:
        raise ValueError(
            f"packed embeddings must be [capacity, {expected_width}] with capacity >= "
            f"{min_capacity}, got {packed.shape}"
        )
    pos_idx, mask = _build_gather_indices(tasks, running.shape[0])
    return _apply_gather(running, packed, pos_idx, mask, mesh)


def _build_pool_gather_indices(
    tasks: list[ItemTask],
    entries: list[EmbeddingPoolEntry],
    page_size: int,
    num_tokens: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Map each destination token to a flat row in the embedding pool."""

    pos_idx = np.zeros(num_tokens, dtype=np.int32)
    mask = np.zeros(num_tokens, dtype=np.bool_)
    for task, entry in zip(tasks, entries, strict=True):
        for mapping in task.merge_mappings:
            dst, length = mapping.destination_start, mapping.length
            if dst < 0 or dst + length > num_tokens:
                raise ValueError("multimodal merge slice exceeds the token batch")
            token = mapping.source_start + np.arange(length, dtype=np.int32)
            span = slice(dst, dst + length)
            pos_idx[span] = entry.page_ids[token // page_size] * page_size + token % page_size
            mask[span] = True
    return pos_idx, mask


def _gather_from_pool(
    running: jax.Array,
    pool: EmbeddingPool,
    tasks: list[ItemTask],
    entries: list[EmbeddingPoolEntry],
    mesh: Mesh | None,
) -> jax.Array:
    """Overlay cache hits by gathering from the pool's paged buffers."""

    pos_idx, mask = _build_pool_gather_indices(tasks, entries, pool.page_size, running.shape[0])
    return _apply_gather(
        running,
        pool.pages.reshape(-1, pool.hidden),
        pos_idx,
        mask,
        mesh,
    )


def _cache_unfinished_items(
    pool: EmbeddingPool | None,
    packed: jax.Array,
    tasks: list[ItemTask],
) -> None:
    if pool is None:
        return
    write_mask = [task.has_unmerged_tail for task in tasks]
    if not any(write_mask):
        return
    pool.write_packed(
        [task.item.hash for task in tasks],
        packed,
        [task.output_len for task in tasks],
        write_mask=write_mask,
    )


def _split_embeddings(
    running: jax.Array,
    hidden: int,
    deepstack_dim: int,
    mesh: Mesh | None,
) -> tuple[jax.Array, jax.Array | None]:
    if not deepstack_dim:
        return running, None
    running, deepstack = jnp.split(running, [hidden], axis=-1)
    deepstack = deepstack.reshape(running.shape[0], deepstack_dim, hidden).transpose(1, 0, 2)
    if isinstance(running.sharding, NamedSharding):
        token_spec = running.sharding.spec[0] if running.sharding.spec else None
        deepstack = jax.sharding.reshard(
            deepstack,
            NamedSharding(mesh, PartitionSpec(None, token_spec, None)),
        )
    return running, deepstack


def precompile_multimodal_inputs(
    input_ids: jax.Array,
    multimodal_model: InModelMultimodalContract,
    embedding_pool: EmbeddingPool | None = None,
) -> tuple[jax.Array, jax.Array | None, bool]:
    """Warm merge kernels and return a multimodal-shaped forward input."""
    capacities = multimodal_model.get_multimodal_embedding_packed_capacities()
    if any(capacity <= 0 for capacity in capacities):
        raise ValueError(f"invalid multimodal packed capacities: {capacities}")

    mesh = multimodal_model.mesh
    with jax.set_mesh(mesh) if mesh is not None else nullcontext():
        running = multimodal_model.get_input_embeddings()(input_ids)
        num_tokens, hidden = running.shape
        deepstack_dim = multimodal_model.deepstack_visual_layers
        if deepstack_dim:
            running = jnp.pad(running, ((0, 0), (0, hidden * deepstack_dim)))

        item = MultimodalDataItem(modality=Modality.IMAGE)
        for capacity in capacities or [num_tokens]:
            length = min(num_tokens, capacity)
            task = ItemTask(item, length, [MergeMapping(0, 0, length)])
            packed = jnp.zeros((capacity, running.shape[-1]), running.dtype)
            if mesh is not None:
                packed = jax.device_put(packed, NamedSharding(mesh, PartitionSpec()))
            running = _gather_merge(running, packed, [task], mesh)
            jax.block_until_ready(running)

        if embedding_pool is not None:
            length = min(num_tokens, embedding_pool.page_size)
            task = ItemTask(item, length, [MergeMapping(0, 0, length)])
            entry = EmbeddingPoolEntry(np.asarray([0], dtype=np.int32), length)
            running = _gather_from_pool(running, embedding_pool, [task], [entry], mesh)
            jax.block_until_ready(running)

    input_embedding, deepstack = _split_embeddings(running, hidden, deepstack_dim, mesh)
    return input_embedding, deepstack, deepstack_dim > 0


def precompile_multimodal_components(
    multimodal_model: InModelMultimodalContract,
    embedding_pool: EmbeddingPool | None = None,
) -> None:
    multimodal_model.precompile_multimodal()
    if embedding_pool is not None:
        for capacity in multimodal_model.get_multimodal_embedding_packed_capacities():
            embedding_pool.precompile_packed_write(capacity)


def embed_multimodal_inputs(
    multimodal_batch: MultimodalBatch | None,
    input_ids: jax.Array,
    multimodal_model: InModelMultimodalContract,
    embedding_pool: EmbeddingPool | None = None,
) -> tuple[jax.Array, jax.Array | None, bool]:
    """Merge padded, item-ordered encoder outputs into the token stream."""
    mesh = multimodal_model.mesh
    with jax.set_mesh(mesh) if mesh is not None else nullcontext():
        running = multimodal_model.get_input_embeddings()(input_ids)
        hidden = running.shape[-1]
        deepstack_dim = multimodal_model.deepstack_visual_layers
        if deepstack_dim and multimodal_batch is None:
            deepstack = jnp.zeros(
                (deepstack_dim, running.shape[0], hidden),
                dtype=running.dtype,
                out_sharding=NamedSharding(
                    mesh,
                    PartitionSpec(None, "data", None),
                ),
            )
            return running, deepstack, False
        if deepstack_dim:
            running = jnp.pad(running, ((0, 0), (0, hidden * deepstack_dim)))

        if multimodal_batch is not None:
            try:
                cached = {}
                for ready in multimodal_batch.ready_embeddings:
                    cached.setdefault(ready.source.pool, []).append(ready)
                # Hold the buffer lock through dispatch: writes donate/rebind pool.pages.
                for pool, hits in cached.items():
                    with pool.lock:
                        running = _gather_from_pool(
                            running,
                            pool,
                            [hit.task for hit in hits],
                            [hit.source.entry for hit in hits],
                            mesh,
                        )
                        for hit in hits:
                            hit.source.release(running)
                encode_funcs = multimodal_model.get_multimodal_encode_funcs()
                modalities = dict.fromkeys(
                    modality for lane in multimodal_batch.per_lane_tasks for modality in lane
                )
                for modality in modalities:
                    tasks = []
                    items_per_lane = []
                    for lane in multimodal_batch.per_lane_tasks:
                        lane_tasks = lane.get(modality, [])
                        items_per_lane.append([task.item for task in lane_tasks])
                        tasks.extend(lane_tasks)
                    encode = encode_funcs.get(modality)
                    if encode is None:
                        raise ValueError(f"no embedding function for modality {modality}")
                    packed = encode(items_per_lane)
                    running = _gather_merge(running, packed, tasks, mesh)
                    _cache_unfinished_items(embedding_pool, packed, tasks)
            finally:
                multimodal_batch.release()

        input_embedding, deepstack = _split_embeddings(running, hidden, deepstack_dim, mesh)
        apply_for_deepstack = deepstack_dim > 0 and multimodal_batch is not None
        return input_embedding, deepstack, apply_for_deepstack
