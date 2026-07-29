import logging
import math
from collections.abc import Callable
from dataclasses import dataclass
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.tree_util import register_pytree_node_class

from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.hf_transformers_utils import get_hf_text_config
from sgl_jax.srt.layers.embeddings import Embed, ParallelLMHead
from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sgl_jax.srt.mem_cache.memory_pool import MemoryPools
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
from sgl_jax.srt.models.qwen3 import QWen3Model, create_qwen3_weight_mappings
from sgl_jax.srt.multimodal.common.modality_enum import Modality, MultimodalDataItem
from sgl_jax.srt.multimodal.in_model.interface import (
    InModelMultimodalContract,
    MultimodalEmbeddingOutput,
    MultimodalItemGroups,
)
from sgl_jax.srt.multimodal.kernels.flash_attention import SegmentIds
from sgl_jax.srt.multimodal.layers.vision_sharding import (
    VisionShardSpecs,
    apply_data_sharding,
    resolve_encoder_tp,
    schedule_vision_lanes,
    slice_owner_items,
)
from sgl_jax.srt.utils.common_utils import resolve_vision_patch_buckets
from sgl_jax.srt.utils.weight_utils import WeightLoader, WeightMapping

logger = logging.getLogger(__name__)


@register_pytree_node_class
@dataclass
class _Qwen3VisionMetadata:
    pos_indices: jax.Array | np.ndarray
    pos_weights: jax.Array | np.ndarray
    rotary_pos_emb: jax.Array | np.ndarray
    cu_seqlens: jax.Array | np.ndarray

    def tree_flatten(self):
        return (
            self.pos_indices,
            self.pos_weights,
            self.rotary_pos_emb,
            self.cu_seqlens,
        ), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


def _grid(item: MultimodalDataItem) -> tuple[int, int, int]:
    value = item.get("image_grid_thw")
    if value is None:
        value = item.get("video_grid_thw")
    return tuple(int(x) for x in np.asarray(value).reshape(3))


def _merge_order(x: np.ndarray, t: int, h: int, w: int, merge: int) -> np.ndarray:
    return (
        np.broadcast_to(x, (t, *x.shape))
        .reshape(t, h // merge, merge, w // merge, merge, *x.shape[2:])
        .transpose(0, 1, 3, 2, 4, *range(5, x.ndim + 3))
        .reshape(t * h * w, *x.shape[2:])
    )


def _rope(x: jax.Array, freqs: jax.Array) -> jax.Array:
    half = x.shape[-1] // 2
    left, right = x[..., :half], x[..., half:]
    cos, sin = jnp.cos(freqs)[:, :, None], jnp.sin(freqs)[:, :, None]
    return jnp.concatenate((left * cos - right * sin, left * sin + right * cos), axis=-1).astype(
        x.dtype
    )


def _segments(cu_seqlens: jax.Array, length: int) -> jax.Array:
    positions = jnp.arange(length, dtype=cu_seqlens.dtype)
    return jnp.sum(cu_seqlens[:, :, None] <= positions, axis=1).astype(jnp.int32)


def _attention(backend, q, k, v, segments):
    if backend is None:
        scores = jnp.einsum("bthd,bshd->bhts", q, k) / math.sqrt(q.shape[-1])
        mask = (segments[:, :, None] == segments[:, None, :]) & (segments[:, :, None] >= 0)
        probs = jax.nn.softmax(
            jnp.where(mask[:, None], scores, -jnp.inf).astype(jnp.float32), axis=-1
        ).astype(q.dtype)
        probs = jnp.where((segments >= 0)[:, None, :, None], probs, 0)
        return jnp.einsum("bhts,bshd->bthd", probs, v)

    length = q.shape[1]
    aligned = max(256, ((length + 127) // 128) * 128)
    pad = aligned - length
    q, k, v = (jnp.transpose(x, (0, 2, 1, 3)) for x in (q, k, v))
    if pad:
        padding = ((0, 0), (0, 0), (0, pad), (0, 0))
        q, k, v = (jnp.pad(x, padding) for x in (q, k, v))
        segments = jnp.pad(segments, ((0, 0), (0, pad)), constant_values=-1)
    output = backend(q, k, v, SegmentIds(q=segments, kv=segments))
    return jnp.transpose(output[:, :, :length], (0, 2, 1, 3))


class Qwen3VLPatchEmbed(nnx.Module):
    def __init__(self, config, dtype, rngs, mesh, tp):
        self.channels = config.in_channels
        self.temporal = config.temporal_patch_size
        self.patch = config.patch_size
        self.hidden = config.hidden_size
        self.mesh = mesh
        self.specs = VisionShardSpecs(mesh, tp)
        self.proj = nnx.Conv(
            self.channels,
            self.hidden,
            (self.temporal, self.patch, self.patch),
            strides=(self.temporal, self.patch, self.patch),
            use_bias=True,
            dtype=dtype,
            param_dtype=dtype,
            rngs=rngs,
        )

    def __call__(self, x):
        batch, length, _ = x.shape
        flat_sharding = self.specs.batch_sharding(None, None, None, None)
        output_sharding = self.specs.batch_sharding(None, None)
        x = x.reshape(
            batch * length,
            self.channels,
            self.temporal,
            self.patch,
            self.patch,
            out_sharding=flat_sharding,
        )
        x = jnp.transpose(x, (0, 2, 3, 4, 1))
        x = self.proj(x, out_sharding=flat_sharding).reshape(
            batch, length, self.hidden, out_sharding=output_sharding
        )
        if self.mesh is not None:
            x = apply_data_sharding(x, self.mesh, self.specs.batch_spec(None, None))
        return x


class Qwen3VLVisionMLP(nnx.Module):
    def __init__(self, config, dtype, mesh, specs):
        self.fc1 = LinearBase(
            config.hidden_size,
            config.intermediate_size,
            mesh,
            use_bias=True,
            kernel_axes=specs.col_kernel_axes,
            params_dtype=dtype,
        )
        self.fc2 = LinearBase(
            config.intermediate_size,
            config.hidden_size,
            mesh,
            use_bias=True,
            kernel_axes=specs.row_kernel_axes,
            params_dtype=dtype,
        )
        self.specs = specs
        self.approximate = config.hidden_act == "gelu_pytorch_tanh"

    def __call__(self, x):
        x, _ = self.fc1(x, out_sharding=self.specs.col_out(x.ndim))
        x = jax.nn.gelu(x, approximate=self.approximate)
        return self.fc2(x, out_sharding=self.specs.row_out(x.ndim))[0]


class Qwen3VLVisionAttention(nnx.Module):
    def __init__(self, config, dtype, mesh, specs):
        self.hidden = config.hidden_size
        self.heads = config.num_heads
        self.head_dim = self.hidden // self.heads
        self.specs = specs
        if specs.tp:
            assert (
                self.heads % int(mesh.shape["tensor"]) == 0
            ), f"vision num_heads={self.heads} must be divisible by tp={mesh.shape['tensor']}"
        linear = lambda: LinearBase(
            self.hidden,
            self.hidden,
            mesh,
            use_bias=True,
            kernel_axes=specs.col_kernel_axes,
            params_dtype=dtype,
        )
        self.q_proj, self.k_proj, self.v_proj = linear(), linear(), linear()
        self.proj = LinearBase(
            self.hidden,
            self.hidden,
            mesh,
            use_bias=True,
            kernel_axes=specs.row_kernel_axes,
            params_dtype=dtype,
        )
        if mesh is None or jax.default_backend() == "cpu":
            self.backend = None
        else:
            from sgl_jax.srt.multimodal.layers.attention.flash_attention_backend import (
                VisionFlashAttentionBackend,
            )

            self.backend = VisionFlashAttentionBackend(
                mesh,
                sm_scale=self.head_dim**-0.5,
                causal=False,
                head_tp=specs.tp,
            )

    def __call__(self, x, freqs, segments):
        batch, length, _ = x.shape
        q, k, v = (
            layer(x, out_sharding=self.specs.col_out(x.ndim))[0]
            for layer in (
                self.q_proj,
                self.k_proj,
                self.v_proj,
            )
        )
        sharding = self.specs.qkv_reshape_sharding()
        q, k, v = (
            value.reshape(batch, length, self.heads, self.head_dim, out_sharding=sharding)
            for value in (q, k, v)
        )
        output = _attention(self.backend, _rope(q, freqs), _rope(k, freqs), v, segments)
        output = output.reshape(batch, length, self.hidden, out_sharding=self.specs.col_out(3))
        return self.proj(output, out_sharding=self.specs.row_out(3))[0]


class Qwen3VLVisionBlock(nnx.Module):
    def __init__(self, config, dtype, rngs, mesh, tp):
        specs = VisionShardSpecs(mesh, tp)
        norm = lambda: nnx.LayerNorm(
            config.hidden_size,
            epsilon=1e-6,
            dtype=dtype,
            param_dtype=dtype,
            use_fast_variance=False,
            rngs=rngs,
        )
        self.norm1, self.norm2 = norm(), norm()
        self.attn = Qwen3VLVisionAttention(config, dtype, mesh, specs)
        self.mlp = Qwen3VLVisionMLP(config, dtype, mesh, specs)

    def __call__(self, x, freqs, segments):
        x = x + self.attn(self.norm1(x), freqs, segments)
        return x + self.mlp(self.norm2(x))


class Qwen3VLPatchMerger(nnx.Module):
    def __init__(self, config, dtype, rngs, mesh, tp, postshuffle):
        self.hidden = config.hidden_size * config.spatial_merge_size**2
        self.postshuffle = postshuffle
        self.specs = VisionShardSpecs(mesh, tp)
        self.norm = nnx.LayerNorm(
            self.hidden if postshuffle else config.hidden_size,
            epsilon=1e-6,
            dtype=dtype,
            param_dtype=dtype,
            use_fast_variance=False,
            rngs=rngs,
        )
        self.fc1 = LinearBase(
            self.hidden,
            self.hidden,
            mesh,
            use_bias=True,
            kernel_axes=self.specs.col_kernel_axes,
            params_dtype=dtype,
        )
        self.fc2 = LinearBase(
            self.hidden,
            config.out_hidden_size,
            mesh,
            use_bias=True,
            kernel_axes=self.specs.row_kernel_axes,
            params_dtype=dtype,
        )

    def __call__(self, x):
        sharding = self.specs.batch_sharding(None, None)
        if self.postshuffle:
            x = self.norm(x.reshape(x.shape[0], -1, self.hidden, out_sharding=sharding))
        else:
            x = self.norm(x).reshape(x.shape[0], -1, self.hidden, out_sharding=sharding)
        x, _ = self.fc1(x, out_sharding=self.specs.col_out(x.ndim))
        x = jax.nn.gelu(x, approximate=False)
        return self.fc2(x, out_sharding=self.specs.row_out(x.ndim))[0]


class Qwen3VLVisionModel(nnx.Module):
    def __init__(
        self,
        config,
        dtype,
        rngs=None,
        mesh=None,
        tp=False,
        input_buckets: tuple[int, ...] | None = None,
    ):
        rngs = rngs or nnx.Rngs(0)
        self.mesh = mesh
        self.tp = tp
        self.specs = VisionShardSpecs(mesh, tp)
        self.input_buckets = input_buckets or tuple(resolve_vision_patch_buckets(None))
        self.merge = int(config.spatial_merge_size)
        self.spatial_merge_unit = self.merge**2
        if any(bucket <= 0 or bucket % self.spatial_merge_unit for bucket in self.input_buckets):
            raise ValueError(
                f"vision patch buckets must be positive multiples of {self.spatial_merge_unit}"
            )
        self.num_grid = int(config.num_position_embeddings**0.5)
        rotary_dim = int(config.hidden_size) // int(config.num_heads) // 2
        self.inv_freq = 1.0 / (
            10000.0 ** (np.arange(0, rotary_dim, 2, dtype=np.float32) / rotary_dim)
        )
        self.patch_embed = Qwen3VLPatchEmbed(config, dtype, rngs, mesh, tp)
        self.pos_embed = Embed(
            config.num_position_embeddings,
            config.hidden_size,
            dtype=dtype,
            param_dtype=dtype,
            kernel_axes=(None, None),
            mesh=mesh,
        )
        self.blocks = nnx.List(
            [Qwen3VLVisionBlock(config, dtype, rngs, mesh, tp) for _ in range(config.depth)]
        )
        self.deepstack_indexes = tuple(config.deepstack_visual_indexes)
        self.deepstack_mergers = nnx.List(
            [
                Qwen3VLPatchMerger(config, dtype, rngs, mesh, tp, True)
                for _ in self.deepstack_indexes
            ]
        )
        self.merger = Qwen3VLPatchMerger(config, dtype, rngs, mesh, tp, False)
        self.patch_dim = config.in_channels * config.temporal_patch_size * config.patch_size**2

    def __call__(
        self,
        patches: jax.Array,
        meta: _Qwen3VisionMetadata,
        valid: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        length = patches.shape[1]
        segments = _segments(jnp.asarray(meta.cu_seqlens), length)
        if valid is not None:
            segments = jnp.where(jnp.arange(length)[None] < valid[:, None], segments, -1)
        x = self.patch_embed(patches)
        pos = self.pos_embed.embedding.at[meta.pos_indices].get(
            out_sharding=self.specs.batch_sharding(None, None, None)
        )
        x += jnp.sum(pos * meta.pos_weights[..., None].astype(pos.dtype), axis=1).astype(x.dtype)
        deepstack = []
        for index, block in enumerate(self.blocks):
            x = block(x, jnp.asarray(meta.rotary_pos_emb), segments)
            if index in self.deepstack_indexes:
                merger = self.deepstack_mergers[self.deepstack_indexes.index(index)]
                deepstack.append(merger(x))
        merged = self.merger(x)
        deepstack = (
            jnp.stack(deepstack, axis=1)
            if deepstack
            else jnp.empty((x.shape[0], 0, *merged.shape[1:]), x.dtype)
        )
        return merged, deepstack

    @jax.jit
    def encode(
        self,
        features: jax.Array,
        metadata: _Qwen3VisionMetadata,
        valid: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        output, deepstack = self(features, metadata, valid)
        if self.mesh is not None:
            output = apply_data_sharding(output, self.mesh, self.specs.output_spec(None, None))
            deepstack = apply_data_sharding(
                deepstack, self.mesh, self.specs.output_spec(None, None, None)
            )
        return output, deepstack

    def precompile(self) -> None:
        data_size = int(self.mesh.shape.get("data", 1)) if self.mesh is not None else 1
        owner_counts = (1,) + (0,) * (data_size - 1)
        for capacity in self.input_buckets:
            item = MultimodalDataItem(
                Modality.IMAGE,
                feature=np.zeros((capacity, self.patch_dim), dtype=np.float32),
                model_specific_data={
                    "image_grid_thw": np.asarray([[1, self.merge, capacity // self.merge]])
                },
            )
            features, metadata, valid, _ = self._batch_items(
                [item],
                owner_counts,
            )
            jax.block_until_ready(self.encode(features, metadata, valid))

    def encode_item_groups(
        self,
        items_by_owner: MultimodalItemGroups,
    ) -> MultimodalEmbeddingOutput:
        owner_counts = tuple(map(len, items_by_owner))
        items = [item for group in items_by_owner for item in group]
        if not items:
            return MultimodalEmbeddingOutput(())
        features, metadata, valid, placements = self._batch_items(
            items,
            owner_counts,
        )
        output, deepstack = self.encode(features, metadata, valid)
        lengths = [int(np.prod(_grid(item))) // self.spatial_merge_unit for item in items]
        if self.mesh is None:
            return MultimodalEmbeddingOutput(
                [
                    output[lane, start : start + length]
                    for (lane, start), length in zip(placements, lengths, strict=True)
                ],
                [
                    deepstack[lane, :, start : start + length]
                    for (lane, start), length in zip(placements, lengths, strict=True)
                ],
            )
        return MultimodalEmbeddingOutput(
            slice_owner_items(
                output,
                self.mesh,
                owner_counts,
                placements,
                lengths,
                token_axis=1,
            ),
            slice_owner_items(
                deepstack,
                self.mesh,
                owner_counts,
                placements,
                lengths,
                token_axis=2,
            ),
        )

    def _input_capacity(self, length: int) -> int:
        unit = self.spatial_merge_unit
        return next(
            (bucket for bucket in self.input_buckets if bucket >= length and bucket % unit == 0),
            ((1 << (length - 1).bit_length()) + unit - 1) // unit * unit,
        )

    def _pack_metadata(
        self,
        items: list[MultimodalDataItem],
    ) -> _Qwen3VisionMetadata:
        metas = [self._item_metadata(item) for item in items]
        patch_offset = 0
        pos_indices = []
        pos_weights = []
        rotary = []
        cu_seqlens = []
        for item, meta in zip(items, metas, strict=True):
            pos_indices.append(meta.pos_indices)
            pos_weights.append(meta.pos_weights)
            rotary.append(meta.rotary_pos_emb)
            cu_seqlens.append(meta.cu_seqlens + patch_offset)
            patch_offset += item.feature.shape[0]
        return _Qwen3VisionMetadata(
            np.concatenate(pos_indices, axis=1),
            np.concatenate(pos_weights, axis=1),
            np.concatenate(rotary),
            np.concatenate(cu_seqlens),
        )

    def _put_batch(self, value: np.ndarray) -> jax.Array:
        if self.mesh is None:
            return jnp.asarray(value)
        return jax.device_put(
            value,
            self.specs.batch_sharding(*([None] * (value.ndim - 1))),
        )

    def _batch_items(
        self,
        items: list[MultimodalDataItem],
        items_per_data_rank: tuple[int, ...],
    ):
        lengths = [item.feature.shape[0] for item in items]
        data_size = int(self.mesh.shape.get("data", 1)) if self.mesh is not None else 1
        tensor_size = int(self.mesh.shape.get("tensor", 1)) if self.mesh is not None else 1
        lanes = schedule_vision_lanes(
            lengths,
            data_size=data_size,
            tensor_size=tensor_size,
            vision_tp=self.tp,
            items_per_data_rank=items_per_data_rank,
        )
        lane_lengths = [sum(lengths[index] for index in lane) for lane in lanes]
        input_capacity = self._input_capacity(max(lane_lengths))
        feature_shape = items[0].feature.shape[1:]
        features = np.zeros(
            (len(lanes), input_capacity, *feature_shape),
            dtype=np.float32,
        )
        valid = np.zeros(len(lanes), dtype=np.int32)
        dummy_metadata = self._pad_metadata(self._empty_metadata(input_capacity), input_capacity)
        metadata = [dummy_metadata] * len(lanes)
        placements: list[tuple[int, int] | None] = [None] * len(items)
        for lane_index, lane in enumerate(lanes):
            if not lane:
                continue
            patch_offset = 0
            output_offset = 0
            lane_items = [items[index] for index in lane]
            for item_index, item in zip(lane, lane_items, strict=True):
                feature = np.asarray(item.feature)
                end = patch_offset + feature.shape[0]
                features[lane_index, patch_offset:end] = feature
                placements[item_index] = (lane_index, output_offset)
                patch_offset = end
                output_offset += feature.shape[0] // self.spatial_merge_unit
            valid[lane_index] = patch_offset
            metadata[lane_index] = self._pad_metadata(
                self._pack_metadata(lane_items),
                input_capacity,
            )
        metadata = jax.tree.map(lambda *values: np.stack(values), *metadata)
        assert all(placement is not None for placement in placements)
        return (
            self._put_batch(features),
            jax.tree.map(self._put_batch, metadata),
            self._put_batch(valid),
            tuple(placements),
        )

    def _grid_metadata(self, grid: tuple[int, int, int]) -> _Qwen3VisionMetadata:
        t, h, w = grid
        ys = np.linspace(0, self.num_grid - 1, h, dtype=np.float32)
        xs = np.linspace(0, self.num_grid - 1, w, dtype=np.float32)
        y0, x0 = ys.astype(np.int32), xs.astype(np.int32)
        y1 = np.minimum(y0 + 1, self.num_grid - 1)
        x1 = np.minimum(x0 + 1, self.num_grid - 1)
        dy, dx = ys - y0, xs - x0
        indices = np.stack(
            (
                y0[:, None] * self.num_grid + x0[None],
                y0[:, None] * self.num_grid + x1[None],
                y1[:, None] * self.num_grid + x0[None],
                y1[:, None] * self.num_grid + x1[None],
            )
        )
        weights = np.stack(
            (
                (1 - dy[:, None]) * (1 - dx[None]),
                (1 - dy[:, None]) * dx[None],
                dy[:, None] * (1 - dx[None]),
                dy[:, None] * dx[None],
            )
        )
        indices = np.stack(
            [_merge_order(value[..., None], t, h, w, self.merge)[:, 0] for value in indices]
        )
        weights = np.stack(
            [_merge_order(value[..., None], t, h, w, self.merge)[:, 0] for value in weights]
        )
        rows, cols = np.indices((h, w))
        coords = _merge_order(
            np.stack((rows, cols), axis=-1),
            t,
            h,
            w,
            self.merge,
        )
        rotary = np.concatenate(
            (
                coords[:, :1] * self.inv_freq,
                coords[:, 1:] * self.inv_freq,
            ),
            axis=-1,
        ).astype(np.float32)
        cu_seqlens = np.arange(h * w, t * h * w + 1, h * w, dtype=np.int32)
        return _Qwen3VisionMetadata(
            indices,
            weights.astype(np.float32),
            rotary,
            cu_seqlens,
        )

    def _item_metadata(self, item: MultimodalDataItem) -> _Qwen3VisionMetadata:
        grid = _grid(item)
        return self._grid_metadata(grid)

    def _empty_metadata(self, input_capacity: int) -> _Qwen3VisionMetadata:
        return self._grid_metadata((1, self.merge, input_capacity // self.merge))

    def _pad_metadata(
        self,
        metadata: _Qwen3VisionMetadata,
        input_capacity: int,
    ) -> _Qwen3VisionMetadata:
        boundary_capacity = input_capacity // self.spatial_merge_unit
        rows = metadata.rotary_pos_emb.shape[0]
        pos_indices = np.zeros((4, input_capacity), dtype=np.int32)
        pos_weights = np.zeros((4, input_capacity), dtype=np.float32)
        rotary = np.zeros(
            (input_capacity, metadata.rotary_pos_emb.shape[-1]),
            dtype=np.float32,
        )
        cu_seqlens = np.full(boundary_capacity, input_capacity, dtype=np.int32)
        pos_indices[:, :rows] = metadata.pos_indices
        pos_weights[:, :rows] = metadata.pos_weights
        rotary[:rows] = metadata.rotary_pos_emb
        cu_seqlens[: metadata.cu_seqlens.shape[0]] = metadata.cu_seqlens
        return _Qwen3VisionMetadata(
            pos_indices,
            pos_weights,
            rotary,
            cu_seqlens,
        )


class Qwen3VLForConditionalGeneration(nnx.Module, InModelMultimodalContract):
    mrope_position_axes = 3

    def __init__(self, config=None, dtype=None, mesh=None, rngs=None):
        self.mesh = mesh
        self.config = config
        self.text_config = get_hf_text_config(config) or config
        self.dtype = dtype or jnp.bfloat16
        rope = getattr(self.text_config, "rope_parameters", None)
        if rope:
            self.text_config.rope_theta = rope.get(
                "rope_theta", getattr(self.text_config, "rope_theta", 5_000_000)
            )
            self.text_config.rope_scaling = {
                "rope_type": rope.get("rope_type", "default"),
                "mrope_section": rope.get("mrope_section", [24, 20, 20]),
                "mrope_interleaved": True,
            }
        elif not getattr(self.text_config, "rope_scaling", None):
            self.text_config.rope_scaling = {
                "rope_type": "default",
                "mrope_section": [24, 20, 20],
                "mrope_interleaved": True,
            }
        self.model = QWen3Model(self.text_config, mesh=mesh, dtype=self.dtype)
        if not getattr(self.text_config, "tie_word_embeddings", False):
            self.lm_head = ParallelLMHead(
                self.text_config.vocab_size,
                self.text_config.hidden_size,
                dtype=self.dtype,
                param_dtype=self.dtype,
                kernel_axes=("tensor", None),
                mesh=mesh,
            )
        self.logits_processor = LogitsProcessor(self.text_config.vocab_size, mesh=mesh)
        from sgl_jax.srt.managers.schedule_batch import global_server_args_dict

        encoder_tp = resolve_encoder_tp(
            mesh, global_server_args_dict.get("vision_encoder_parallel", "dp")
        )
        self.visual = Qwen3VLVisionModel(
            config.vision_config,
            self.dtype,
            rngs,
            mesh,
            encoder_tp,
            tuple(
                resolve_vision_patch_buckets(
                    global_server_args_dict.get("precompile_vision_patch_paddings")
                )
            ),
        )
        self.deepstack_visual_layers = len(self.visual.deepstack_indexes)

    def get_input_embeddings(self) -> Callable[[jax.Array], jax.Array]:
        return self.model.get_input_embeddings()

    def precompile_multimodal(self) -> None:
        self.visual.precompile()

    def get_packed_multimodal_embedding_funcs(self):
        return dict.fromkeys(
            (Modality.IMAGE, Modality.MULTI_IMAGES, Modality.VIDEO),
            self.visual.encode_item_groups,
        )

    def load_weights(self, model_config: ModelConfig) -> None:
        text_loader = WeightLoader(self, model_config, self.mesh, self.dtype)
        text_loader.load_weights_from_safetensors(
            create_qwen3_weight_mappings(
                self.text_config, source_prefix="model.language_model", target_prefix="model"
            )
        )
        config = self.config.vision_config
        vision_config = SimpleNamespace(
            model_path=model_config.model_path,
            num_attention_heads=config.num_heads,
            hidden_size=config.hidden_size,
            get_total_num_kv_heads=lambda: config.num_heads,
        )
        WeightLoader(self, vision_config, self.mesh, self.dtype).load_weights_from_safetensors(
            self._vision_weight_mappings()
        )
        logger.info("Qwen3-VL weights loaded successfully")

    def _vision_weight_mappings(self):
        specs = self.visual.specs
        col, row = specs.col_kernel_axes, specs.row_kernel_axes
        mappings = {
            "model.visual.patch_embed.proj.weight": WeightMapping(
                "visual.patch_embed.proj.kernel",
                (None, None, None, None, None),
                transpose_axes=(2, 3, 4, 1, 0),
            ),
            "model.visual.patch_embed.proj.bias": WeightMapping(
                "visual.patch_embed.proj.bias", (None,), transpose=False
            ),
            "model.visual.pos_embed.weight": WeightMapping(
                "visual.pos_embed.embedding", (None, None), transpose=False
            ),
        }
        for index in range(self.config.vision_config.depth):
            source, target = f"model.visual.blocks.{index}", f"visual.blocks.{index}"
            mappings.update(self._block_mappings(source, target, col, row))
        mappings.update(self._merger_mappings("model.visual.merger", "visual.merger", col, row))
        for index, _ in enumerate(self.visual.deepstack_indexes):
            mappings.update(
                self._merger_mappings(
                    f"model.visual.deepstack_merger_list.{index}",
                    f"visual.deepstack_mergers.{index}",
                    col,
                    row,
                )
            )
        return mappings

    @staticmethod
    def _linear(source, target, sharding):
        return {
            f"{source}.weight": WeightMapping(target + ".weight", sharding, transpose=True),
            f"{source}.bias": WeightMapping(target + ".bias", (None,), transpose=False),
        }

    @classmethod
    def _block_mappings(cls, source, target, col, row):
        mappings = {}
        for name in ("norm1", "norm2"):
            mappings[f"{source}.{name}.weight"] = WeightMapping(
                f"{target}.{name}.scale", (None,), transpose=False
            )
            mappings[f"{source}.{name}.bias"] = WeightMapping(
                f"{target}.{name}.bias", (None,), transpose=False
            )
        mappings[f"{source}.attn.qkv.weight"] = WeightMapping(
            [f"{target}.attn.{name}_proj.weight" for name in "qkv"], col, transpose=True
        )
        mappings[f"{source}.attn.qkv.bias"] = WeightMapping(
            [f"{target}.attn.{name}_proj.bias" for name in "qkv"], (None,), transpose=False
        )
        mappings.update(cls._linear(f"{source}.attn.proj", f"{target}.attn.proj", row))
        mappings.update(cls._linear(f"{source}.mlp.linear_fc1", f"{target}.mlp.fc1", col))
        mappings.update(cls._linear(f"{source}.mlp.linear_fc2", f"{target}.mlp.fc2", row))
        return mappings

    @classmethod
    def _merger_mappings(cls, source, target, col, row):
        mappings = {
            f"{source}.norm.weight": WeightMapping(
                f"{target}.norm.scale", (None,), transpose=False
            ),
            f"{source}.norm.bias": WeightMapping(f"{target}.norm.bias", (None,), transpose=False),
        }
        mappings.update(cls._linear(f"{source}.linear_fc1", f"{target}.fc1", col))
        mappings.update(cls._linear(f"{source}.linear_fc2", f"{target}.fc2", row))
        return mappings

    def get_embed_and_head(self):
        embed = self.model.embed_tokens.embedding.value
        return (
            (embed, embed)
            if getattr(self.text_config, "tie_word_embeddings", False)
            else (
                embed,
                self.lm_head.embedding.value,
            )
        )

    def set_embed_and_head(self, embed_weight=None, head_weight=None):
        if embed_weight is not None:
            self.model.embed_tokens.embedding.value = embed_weight
        if head_weight is not None and not getattr(self.text_config, "tie_word_embeddings", False):
            self.lm_head.embedding.value = head_weight

    def __call__(
        self,
        forward_batch: ForwardBatch,
        memory_pools: MemoryPools,
        logits_metadata: LogitsMetadata,
    ):
        hidden, aux, kv, callbacks = self.model(forward_batch, memory_pools.token_to_kv_pool)
        head = (
            self.model.embed_tokens
            if getattr(self.text_config, "tie_word_embeddings", False)
            else self.lm_head
        )
        output = self.logits_processor(hidden, head, logits_metadata, aux_hidden_states=aux)
        return output, {"token_to_kv_pool": kv}, callbacks, None


EntryClass = Qwen3VLForConditionalGeneration
