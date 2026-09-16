from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.models.registry import ModelRegistry
from sgl_jax.srt.multimodal.common.modality_enum import Modality, MultimodalDataItem
from sgl_jax.srt.multimodal.in_model.embedding_pool import (
    EmbeddingLease,
    EmbeddingPool,
    EmbeddingPoolEntry,
)
from sgl_jax.srt.multimodal.in_model.interface import InModelMultimodalContract
from sgl_jax.srt.multimodal.in_model.lane_packing import balance_lanes
from sgl_jax.srt.multimodal.in_model.mm_utils import (
    ItemTask,
    MergeMapping,
    apply_gather,
    collect_item_tasks,
    gather_merge,
    prepare_input_embeddings,
    split_embeddings,
)


@dataclass(frozen=True)
class CachedEmbedding:
    task: ItemTask
    lease: EmbeddingLease


@dataclass
class MultimodalBatch:
    per_lane_tasks: list[dict[Modality, list[ItemTask]]]
    cached_embeddings: list[CachedEmbedding]

    def release(self):
        """Release reservations when a queued plan is cancelled or consumed."""
        for cached in self.cached_embeddings:
            cached.lease.release()


def build_multimodal_batch(
    reqs_info: list | None,
    dp_size: int,
    model_config: ModelConfig,
    per_dp_token: int,
    embedding_pool: EmbeddingPool | None = None,
    num_encoder_lanes: int = 1,
) -> MultimodalBatch | None:
    """Reserve local cache hits and balance local encoder work for this chunk."""
    if num_encoder_lanes < 1:
        raise ValueError("num_encoder_lanes must be positive")
    if reqs_info is None or not ModelRegistry.is_in_model_multimodal(
        model_config.hf_config.architectures
    ):
        return None

    grouped: dict[Modality, list[ItemTask]] = {}
    for task in collect_item_tasks(reqs_info, dp_size, per_dp_token):
        if task.item.precomputed_embeddings is not None:
            raise ValueError("Precomputed embeddings must use the disaggregated input path")
        grouped.setdefault(task.item.modality, []).append(task)

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
                    result.cached_embeddings.append(CachedEmbedding(task, lease))
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


def _gather_from_pool(
    running: jax.Array,
    pool: EmbeddingPool,
    tasks: list[ItemTask],
    entries: list[EmbeddingPoolEntry],
    mesh: Mesh | None,
) -> jax.Array:
    """Overlay cache hits by gathering from the pool's paged buffers."""

    pos_idx = np.zeros(running.shape[0], dtype=np.int32)
    mask = np.zeros(running.shape[0], dtype=np.bool_)
    for task, entry in zip(tasks, entries, strict=True):
        for mapping in task.merge_mappings:
            dst, length = mapping.destination_start, mapping.length
            if dst < 0 or dst + length > running.shape[0]:
                raise ValueError("multimodal merge slice exceeds the token batch")
            rows = mapping.source_start + np.arange(length, dtype=np.int32)
            span = slice(dst, dst + length)
            pos_idx[span] = (
                entry.page_ids[rows // pool.page_size] * pool.page_size + rows % pool.page_size
            )
            mask[span] = True
    return apply_gather(
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
            running = gather_merge(running, packed, [task], mesh)
            jax.block_until_ready(running)

        if embedding_pool is not None:
            length = min(num_tokens, embedding_pool.page_size)
            task = ItemTask(item, length, [MergeMapping(0, 0, length)])
            entry = EmbeddingPoolEntry(np.asarray([0], dtype=np.int32), length)
            running = _gather_from_pool(running, embedding_pool, [task], [entry], mesh)
            jax.block_until_ready(running)

    input_embedding, deepstack = split_embeddings(running, hidden, deepstack_dim, mesh)
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
        running, hidden = prepare_input_embeddings(input_ids, multimodal_model)
        deepstack_dim = multimodal_model.deepstack_visual_layers

        if multimodal_batch is not None:
            try:
                cached = {}
                for hit in multimodal_batch.cached_embeddings:
                    cached.setdefault(hit.lease.pool, []).append(hit)
                # Hold the buffer lock through dispatch: writes donate/rebind pool.pages.
                for pool, hits in cached.items():
                    with pool.lock:
                        running = _gather_from_pool(
                            running,
                            pool,
                            [hit.task for hit in hits],
                            [hit.lease.entry for hit in hits],
                            mesh,
                        )
                        for hit in hits:
                            hit.lease.release(running)
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
                    running = gather_merge(running, packed, tasks, mesh)
                    _cache_unfinished_items(embedding_pool, packed, tasks)
            finally:
                multimodal_batch.release()

        input_embedding, deepstack = split_embeddings(running, hidden, deepstack_dim, mesh)
        apply_for_deepstack = deepstack_dim > 0 and multimodal_batch is not None
        return input_embedding, deepstack, apply_for_deepstack
