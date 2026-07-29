import logging
import math
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import Mesh
from jax.tree_util import register_pytree_node_class
from transformers import modeling_flax_utils

from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.hf_transformers_utils import get_hf_text_config
from sgl_jax.srt.layers.embeddings import ParallelLMHead
from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sgl_jax.srt.mem_cache.memory_pool import MemoryPools
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
from sgl_jax.srt.models.qwen2 import Qwen2Model, create_qwen2_weight_mappings
from sgl_jax.srt.multimodal.common.modality_enum import Modality, MultimodalDataItem
from sgl_jax.srt.multimodal.configs.qwen_vl.qwen_2_5_vl_config import (
    QwenVLModelVitConfig,
)
from sgl_jax.srt.multimodal.in_model.interface import (
    InModelMultimodalContract,
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
logger.setLevel(logging.INFO)

_init_fn = nnx.initializers.uniform()


@register_pytree_node_class
@dataclass
class _Qwen25VisionMetadata:
    window_index: jax.Array | np.ndarray
    cu_window_seqlens: jax.Array | np.ndarray
    rotary_pos_emb: jax.Array | np.ndarray
    cu_image_seqlens: jax.Array | np.ndarray

    def tree_flatten(self):
        return (
            self.window_index,
            self.cu_window_seqlens,
            self.rotary_pos_emb,
            self.cu_image_seqlens,
        ), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


def _item_grid_thw(item: MultimodalDataItem) -> tuple[int, int, int]:
    grid = item.get("image_grid_thw")
    if grid is None:
        grid = item.get("video_grid_thw")
    return tuple(int(value) for value in np.asarray(grid).reshape(3))


def _apply_rotary_pos_emb_vision(x: jax.Array, rotary_pos_emb: jax.Array) -> jax.Array:
    """RoPE for vision: *x* is ``[B, T, heads, head_dim]``, *rotary_pos_emb* is ``[B, T, rot]``."""
    half_dim = x.shape[-1] // 2
    x_real, x_imag = x[..., :half_dim], x[..., half_dim:]
    cos = jnp.cos(rotary_pos_emb)[:, :, None, :]
    sin = jnp.sin(rotary_pos_emb)[:, :, None, :]
    return jnp.concatenate(
        [x_real * cos - x_imag * sin, x_real * sin + x_imag * cos],
        axis=-1,
    ).astype(x.dtype)


def _vision_attention(
    backend, q: jax.Array, k: jax.Array, v: jax.Array, seg: jax.Array
) -> jax.Array:
    """Batch-leading block-diagonal flash attention for ViT.

    Args:
        q, k, v: ``[B, T, heads, head_dim]`` (batch-leading).
        seg: ``[B, T]`` segment ids; ``-1`` rows are masked padding.
    Returns:
        ``[B, T, heads, head_dim]``.
    """
    _, T, _, _ = q.shape

    # [B, T, H, D] → [B, H, T, D] for the kernel.
    q = jnp.transpose(q, (0, 2, 1, 3))
    k = jnp.transpose(k, (0, 2, 1, 3))
    v = jnp.transpose(v, (0, 2, 1, 3))

    # Pad to the selected TPU tile. Window attention uses the measured
    # 512x256x128 local tile. Long global attention uses 2048x2048x2048 once
    # the single-step kernel approaches its VMEM limit. Backends used by unit
    # tests do not expose ``max_segment_len`` and retain the portable 128-row
    # alignment.
    max_segment_len = getattr(backend, "max_segment_len", None)
    if max_segment_len is not None:
        alignment = 256
    elif T >= 32 * 1024:
        alignment = 2048
    else:
        alignment = 128
    T_aligned = max(256, ((T + alignment - 1) // alignment) * alignment)
    pad = T_aligned - T
    if pad > 0:
        q = jnp.pad(q, ((0, 0), (0, 0), (0, pad), (0, 0)))
        k = jnp.pad(k, ((0, 0), (0, 0), (0, pad), (0, 0)))
        v = jnp.pad(v, ((0, 0), (0, 0), (0, pad), (0, 0)))
        seg = jnp.pad(seg, ((0, 0), (0, pad)), constant_values=-1)

    out = backend(q, k, v, SegmentIds(q=seg, kv=seg))  # [B, H, T_aligned, D]
    return jnp.transpose(out[:, :, :T, :], (0, 2, 1, 3))  # → [B, T, H, D]


def _segment_ids_from_cu_seqlens(
    cu_seqlens: jax.Array,
    sequence_length: int,
) -> jax.Array:
    """Convert padded cumulative boundaries to dense segment ids.

    ``cu_seqlens`` is ``[B, K]`` and uses ``sequence_length`` as its padding
    sentinel.  XLA fuses this comparison and reduction without materializing
    the logical ``[B, K, sequence_length]`` tensor on TPU.  Keeping it in one
    helper also lets the transformer build each attention layout only once.
    """
    positions = jnp.arange(sequence_length, dtype=cu_seqlens.dtype)
    return jnp.sum(cu_seqlens[:, :, None] <= positions[None, None, :], axis=1).astype(jnp.int32)


class Qwen2_5_VisionPatchEmbed(nnx.Module):
    """3D (temporal × spatial) patch embedding conv."""

    def __init__(
        self,
        rngs: nnx.Rngs = None,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        hidden_size: int = 1152,
        dtype: jnp.dtype = jnp.bfloat16,
        mesh: Mesh = None,
        vision_tp: bool = False,
    ) -> None:
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.hidden_size = hidden_size
        self.mesh = mesh
        self.specs = VisionShardSpecs(mesh, vision_tp)

        self.proj = nnx.Conv(
            in_features=in_channels,
            out_features=hidden_size,
            kernel_size=(temporal_patch_size, patch_size, patch_size),
            strides=(temporal_patch_size, patch_size, patch_size),
            use_bias=False,
            param_dtype=dtype,
            rngs=rngs or nnx.Rngs(0),
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        """*x*: ``[B, S, C·T·H·W]`` → ``[B, S, hidden_size]``."""
        B, S, D = x.shape
        C = D // (self.temporal_patch_size * self.patch_size * self.patch_size)
        x = x.reshape(B, S, C, self.temporal_patch_size, self.patch_size, self.patch_size)
        if self.mesh is not None:
            x = apply_data_sharding(
                x, self.mesh, self.specs.batch_spec(None, None, None, None, None)
            )

        # [B, S, C, T, H, W] → [B, S, T, H, W, C]
        x = jnp.transpose(x, (0, 1, 3, 4, 5, 2))

        flat_sh = out_sh = None
        if self.mesh is not None and "data" in self.mesh.abstract_mesh.explicit_axes:
            flat_sh = self.specs.batch_sharding(None, None, None, None)
            out_sh = self.specs.batch_sharding(None, None, None, None, None)

        x = x.reshape(
            B * S,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
            C,
            out_sharding=flat_sh,
        )
        x = self.proj(x, out_sharding=flat_sh)
        x = x.reshape(B, S, 1, 1, 1, self.hidden_size, out_sharding=out_sh)
        return jnp.squeeze(x, axis=(2, 3, 4))


class Qwen2_5_VLMLP(nnx.Module):
    """ViT MLP: gate/up → SiLU gate → down."""

    def __init__(
        self,
        config: QwenVLModelVitConfig,
        dtype: jnp.dtype,
        rngs: nnx.Rngs = None,
        mesh: Mesh = None,
        vision_tp: bool = False,
    ):
        self.specs = VisionShardSpecs(mesh, vision_tp)
        self.act_fn = modeling_flax_utils.ACT2FN[config.hidden_act]

        self.gate_proj = LinearBase(
            config.hidden_size,
            config.intermediate_size,
            mesh=mesh,
            use_bias=True,
            kernel_axes=self.specs.col_kernel_axes,
            params_dtype=dtype,
        )
        self.up_proj = LinearBase(
            config.hidden_size,
            config.intermediate_size,
            mesh=mesh,
            use_bias=True,
            kernel_axes=self.specs.col_kernel_axes,
            params_dtype=dtype,
        )
        self.down_proj = LinearBase(
            config.intermediate_size,
            config.hidden_size,
            mesh=mesh,
            use_bias=True,
            kernel_axes=self.specs.row_kernel_axes,
            params_dtype=dtype,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        gate, _ = self.gate_proj(x, out_sharding=self.specs.col_out(x.ndim))
        up, _ = self.up_proj(x, out_sharding=self.specs.col_out(x.ndim))
        out, _ = self.down_proj(self.act_fn(gate) * up, out_sharding=self.specs.row_out(x.ndim))
        return out


class Qwen2_5_VisionAttention(nnx.Module):
    """ViT self-attention with fused QKV, RoPE, and block-diagonal flash attn."""

    def __init__(
        self,
        config: QwenVLModelVitConfig,
        dtype: jnp.dtype,
        rngs: nnx.Rngs = None,
        mesh: Mesh = None,
        vision_tp: bool = False,
        max_segment_len: int | None = None,
        block_sparse_segments: bool = False,
    ):
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.mesh = mesh
        self.specs = VisionShardSpecs(mesh, vision_tp)

        if self.specs.tp:
            tp_size = int(mesh.shape["tensor"]) if mesh is not None else 1
            assert (
                self.num_heads % tp_size == 0
            ), f"vision num_heads={self.num_heads} must be divisible by tp={tp_size}"

        self.q_proj = LinearBase(
            self.hidden_size,
            self.hidden_size,
            mesh=mesh,
            use_bias=True,
            kernel_axes=self.specs.col_kernel_axes,
            params_dtype=dtype,
        )
        self.k_proj = LinearBase(
            self.hidden_size,
            self.hidden_size,
            mesh=mesh,
            use_bias=True,
            kernel_axes=self.specs.col_kernel_axes,
            params_dtype=dtype,
        )
        self.v_proj = LinearBase(
            self.hidden_size,
            self.hidden_size,
            mesh=mesh,
            use_bias=True,
            kernel_axes=self.specs.col_kernel_axes,
            params_dtype=dtype,
        )
        self.proj = LinearBase(
            self.hidden_size,
            self.hidden_size,
            mesh=mesh,
            use_bias=True,
            kernel_axes=self.specs.row_kernel_axes,
            params_dtype=dtype,
        )

        # Lazy import avoids cycle: flash_attention_backend → schedule_batch → models.
        if mesh is not None:
            from sgl_jax.srt.multimodal.layers.attention.flash_attention_backend import (
                VisionFlashAttentionBackend,
            )

            self.attn_backend = VisionFlashAttentionBackend(
                mesh,
                sm_scale=1.0 / math.sqrt(self.head_dim),
                causal=False,
                head_tp=self.specs.tp,
                max_segment_len=max_segment_len,
                block_sparse_segments=block_sparse_segments,
            )
        else:
            self.attn_backend = None

    def __call__(
        self,
        x: jax.Array,
        rotary_pos_emb: jax.Array,
        segment_ids: jax.Array,
    ) -> jax.Array:
        B, T, D = x.shape

        # Project Q, K, V separately (TP-safe: each is independently column-parallel).
        q, _ = self.q_proj(x, out_sharding=self.specs.col_out(x.ndim))
        k, _ = self.k_proj(x, out_sharding=self.specs.col_out(x.ndim))
        v, _ = self.v_proj(x, out_sharding=self.specs.col_out(x.ndim))

        hs = self.specs.qkv_reshape_sharding()
        q = q.reshape(B, T, self.num_heads, self.head_dim, out_sharding=hs)
        k = k.reshape(B, T, self.num_heads, self.head_dim, out_sharding=hs)
        v = v.reshape(B, T, self.num_heads, self.head_dim, out_sharding=hs)

        q = _apply_rotary_pos_emb_vision(q, rotary_pos_emb)
        k = _apply_rotary_pos_emb_vision(k, rotary_pos_emb)

        out = _vision_attention(self.attn_backend, q, k, v, segment_ids)
        out = out.reshape(B, T, D, out_sharding=self.specs.col_out(3))
        out, _ = self.proj(out, out_sharding=self.specs.row_out(out.ndim))
        return out


class Qwen2_5_VisionBlock(nnx.Module):
    """One ViT transformer block: attn (pre-norm) + MLP (pre-norm)."""

    def __init__(
        self,
        config: QwenVLModelVitConfig,
        dtype: jnp.dtype,
        rngs: nnx.Rngs = None,
        mesh: Mesh = None,
        norm_eps: float = 1e-6,
        vision_tp: bool = False,
        max_segment_len: int | None = None,
        block_sparse_segments: bool = False,
    ):
        _rngs = rngs or nnx.Rngs(0)
        norm = partial(
            nnx.RMSNorm, epsilon=norm_eps, scale_init=nnx.with_partitioning(_init_fn, (None,))
        )

        self.norm1 = norm(config.hidden_size, dtype=dtype, rngs=_rngs)
        self.norm2 = norm(config.hidden_size, dtype=dtype, rngs=_rngs)
        self.attn = Qwen2_5_VisionAttention(
            config,
            dtype,
            rngs=rngs,
            mesh=mesh,
            vision_tp=vision_tp,
            max_segment_len=max_segment_len,
            block_sparse_segments=block_sparse_segments,
        )
        self.mlp = Qwen2_5_VLMLP(config, dtype, rngs=rngs, mesh=mesh, vision_tp=vision_tp)

    def __call__(
        self,
        x: jax.Array,
        rotary_pos_emb: jax.Array,
        segment_ids: jax.Array,
    ) -> jax.Array:
        x = x + self.attn(self.norm1(x), rotary_pos_emb, segment_ids)
        x = x + self.mlp(self.norm2(x))
        return x


class Qwen2_5_VisionPatchMerger(nnx.Module):
    """Spatial merge: LN → reshape(sms²) → 2-layer MLP → [B, T/sms², d_model]."""

    def __init__(
        self,
        d_model: int,
        context_dim: int,
        norm_layer: Callable,
        spatial_merge_size: int,
        dtype: jnp.dtype,
        rngs: nnx.Rngs = None,
        mesh: Mesh = None,
        vision_tp: bool = False,
    ):
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.mesh = mesh
        self.specs = VisionShardSpecs(mesh, vision_tp)
        _rngs = rngs or nnx.Rngs(0)

        self.ln_q = norm_layer(
            context_dim,
            dtype=dtype,
            rngs=_rngs,
            scale_init=nnx.with_partitioning(_init_fn, (None,)),
        )
        self.mlp_fc1 = LinearBase(
            self.hidden_size,
            self.hidden_size,
            mesh=mesh,
            use_bias=True,
            kernel_axes=self.specs.col_kernel_axes,
            params_dtype=dtype,
        )
        self.mlp_act = modeling_flax_utils.ACT2FN["gelu"]
        self.mlp_fc2 = LinearBase(
            self.hidden_size,
            d_model,
            mesh=mesh,
            use_bias=True,
            kernel_axes=self.specs.row_kernel_axes,
            params_dtype=dtype,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.ln_q(x)
        B = x.shape[0]
        x = x.reshape(B, -1, self.hidden_size, out_sharding=self.specs.batch_sharding(None, None))
        x, _ = self.mlp_fc1(x, out_sharding=self.specs.col_out(x.ndim))
        x = self.mlp_act(x)
        x, _ = self.mlp_fc2(x, out_sharding=self.specs.row_out(x.ndim))
        return x


class Qwen2_5_VisionTransformer(nnx.Module):
    """Qwen2.5-VL ViT: patch embed → window / full-attn blocks → merge → reorder."""

    def __init__(
        self,
        config: QwenVLModelVitConfig,
        dtype: jnp.dtype,
        rngs: nnx.Rngs = None,
        mesh: Mesh = None,
        norm_eps: float = 1e-6,
        vision_tp: bool = False,
        input_buckets: tuple[int, ...] | None = None,
    ):
        self.mesh = mesh
        self.vision_tp = vision_tp
        self.specs = VisionShardSpecs(mesh, vision_tp)
        self.input_buckets = input_buckets or tuple(resolve_vision_patch_buckets(None))
        self.spatial_merge_size = config.spatial_merge_size
        self.spatial_merge_unit = self.spatial_merge_size**2
        if any(bucket <= 0 or bucket % self.spatial_merge_unit for bucket in self.input_buckets):
            raise ValueError(
                f"vision patch buckets must be positive multiples of {self.spatial_merge_unit}"
            )

        self.patch_embed = Qwen2_5_VisionPatchEmbed(
            patch_size=config.patch_size,
            temporal_patch_size=config.temporal_patch_size,
            in_channels=config.in_channels,
            hidden_size=config.hidden_size,
            dtype=dtype,
            rngs=rngs,
            mesh=mesh,
            vision_tp=vision_tp,
        )
        vit_merger_window_size = (
            config.window_size // config.spatial_merge_size // config.patch_size
        )
        max_window_tokens = vit_merger_window_size**2 * config.spatial_merge_size**2
        self.blocks = nnx.List(
            [
                Qwen2_5_VisionBlock(
                    config,
                    dtype,
                    rngs=rngs,
                    mesh=mesh,
                    norm_eps=norm_eps,
                    vision_tp=vision_tp,
                    max_segment_len=(
                        None if i in config.fullatt_block_indexes else max_window_tokens
                    ),
                    block_sparse_segments=i in config.fullatt_block_indexes,
                )
                for i in range(config.depth)
            ]
        )
        self.merger = Qwen2_5_VisionPatchMerger(
            d_model=config.out_hidden_size,
            context_dim=config.hidden_size,
            norm_layer=partial(nnx.RMSNorm, epsilon=norm_eps),
            spatial_merge_size=config.spatial_merge_size,
            dtype=dtype,
            rngs=rngs,
            mesh=mesh,
            vision_tp=vision_tp,
        )

        self.fullatt_block_indexes = config.fullatt_block_indexes
        self.patch_size = config.patch_size
        self.patch_dim = config.in_channels * config.temporal_patch_size * config.patch_size**2
        self.window_size = config.window_size
        self.rotary_dim = config.hidden_size // config.num_heads // 2
        self.theta = float(getattr(config, "rope_theta", 10000.0))
        self.rot_dim = 2 * len(range(0, self.rotary_dim, 2))

    def __call__(
        self,
        patches: jax.Array,
        meta: _Qwen25VisionMetadata,
        valid: jax.Array | None = None,
    ) -> jax.Array:
        return self._compute(
            patches,
            meta.window_index,
            meta.cu_window_seqlens,
            meta.rotary_pos_emb,
            meta.cu_image_seqlens,
            valid,
        )

    def encode(
        self,
        patches: jax.Array,
        meta: _Qwen25VisionMetadata,
        valid: jax.Array | None = None,
    ) -> jax.Array:
        if self.mesh is None:
            return self._encode_jit(patches, meta, valid)
        with self._mesh_ctx():
            return self._encode_jit(patches, meta, valid)

    def precompile(self) -> None:
        data_size = int(self.mesh.shape.get("data", 1)) if self.mesh is not None else 1
        owner_counts = (1,) + (0,) * (data_size - 1)
        for capacity in self.input_buckets:
            item = MultimodalDataItem(
                Modality.IMAGE,
                feature=np.zeros((capacity, self.patch_dim), dtype=np.float32),
                model_specific_data={
                    "image_grid_thw": np.asarray(
                        [[1, self.spatial_merge_size, capacity // self.spatial_merge_size]]
                    )
                },
            )
            patches, metadata, valid, _ = self._batch_items(
                [item],
                owner_counts,
            )
            jax.block_until_ready(self.encode(patches, metadata, valid))

    def encode_item_groups(
        self,
        items_by_owner: MultimodalItemGroups,
    ) -> list[jax.Array]:
        owner_counts = tuple(map(len, items_by_owner))
        items = [item for group in items_by_owner for item in group]
        if not items:
            return []
        patches, metadata, valid, placements = self._batch_items(
            items,
            owner_counts,
        )
        output = self.encode(patches, metadata, valid)
        lengths = [int(np.prod(_item_grid_thw(item))) // self.spatial_merge_unit for item in items]
        if self.mesh is None:
            return [
                output[lane, start : start + length]
                for (lane, start), length in zip(placements, lengths, strict=True)
            ]
        return slice_owner_items(
            output,
            self.mesh,
            owner_counts,
            placements,
            lengths,
            token_axis=1,
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
    ) -> _Qwen25VisionMetadata:
        metas = [self._item_metadata(item) for item in items]
        patch_offset = 0
        unit_offset = 0
        window_indices = []
        cu_window_seqlens = []
        rotary = []
        cu_image_seqlens = []
        for item, meta in zip(items, metas, strict=True):
            patch_rows = item.feature.shape[0]
            window_indices.append(meta.window_index + unit_offset)
            cu_window_seqlens.append(meta.cu_window_seqlens + patch_offset)
            rotary.append(meta.rotary_pos_emb)
            cu_image_seqlens.append(meta.cu_image_seqlens + patch_offset)
            patch_offset += patch_rows
            unit_offset += patch_rows // self.spatial_merge_unit
        return _Qwen25VisionMetadata(
            np.concatenate(window_indices),
            np.concatenate(cu_window_seqlens),
            np.concatenate(rotary),
            np.concatenate(cu_image_seqlens),
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
            vision_tp=self.vision_tp,
            items_per_data_rank=items_per_data_rank,
        )
        lane_lengths = [sum(lengths[index] for index in lane) for lane in lanes]
        input_capacity = self._input_capacity(max(lane_lengths))
        feature_shape = items[0].feature.shape[1:]
        patches = np.zeros(
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
                patches[lane_index, patch_offset:end] = feature
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
            self._put_batch(patches),
            jax.tree.map(self._put_batch, metadata),
            self._put_batch(valid),
            tuple(placements),
        )

    def _put_batch(self, value: np.ndarray) -> jax.Array:
        if self.mesh is None:
            return jnp.asarray(value)
        return jax.device_put(
            value,
            self.specs.batch_sharding(*([None] * (value.ndim - 1))),
        )

    def _rotary_pos_emb_full(self, seq_len: int) -> np.ndarray:
        inv_freq = 1.0 / (
            self.theta ** (np.arange(0, self.rotary_dim, 2, dtype=np.float32) / self.rotary_dim)
        )
        return np.outer(
            np.arange(seq_len, dtype=np.float32),
            inv_freq,
        )

    def _rotary_pos_emb_thw(self, t: int, h: int, w: int) -> np.ndarray:
        merge = self.spatial_merge_size
        hpos, wpos = np.indices((h, w))
        hpos = hpos.reshape(h // merge, merge, w // merge, merge)
        wpos = wpos.reshape(h // merge, merge, w // merge, merge)
        hpos = hpos.transpose(0, 2, 1, 3).flatten()
        wpos = wpos.transpose(0, 2, 1, 3).flatten()
        positions = np.tile(np.stack((hpos, wpos), axis=-1), (t, 1))
        rotary = self._rotary_pos_emb_full(max(h, w))[positions].reshape(
            positions.shape[0],
            -1,
        )
        return rotary.reshape(
            rotary.shape[0] // self.spatial_merge_unit,
            self.spatial_merge_unit,
            -1,
        )

    def _window_index_thw(self, t: int, h: int, w: int):
        merge = self.spatial_merge_size
        window = self.window_size // merge // self.patch_size
        grid_h, grid_w = h // merge, w // merge
        index = np.arange(t * grid_h * grid_w).reshape(t, grid_h, grid_w)
        pad_h = (-grid_h) % window
        pad_w = (-grid_w) % window
        windows_h = (grid_h + pad_h) // window
        windows_w = (grid_w + pad_w) // window
        index = np.pad(index, ((0, 0), (0, pad_h), (0, pad_w)), constant_values=-1)
        index = index.reshape(t, windows_h, window, windows_w, window)
        index = index.transpose(0, 1, 3, 2, 4).reshape(-1, window, window)
        seqlens = (index != -1).sum(axis=(1, 2))
        return (
            index.reshape(-1)[index.reshape(-1) != -1].astype(np.int32),
            (np.cumsum(seqlens) * self.spatial_merge_unit).astype(np.int32),
        )

    def _item_metadata(self, item: MultimodalDataItem) -> _Qwen25VisionMetadata:
        t, h, w = _item_grid_thw(item)
        patch_rows = t * h * w
        window_index, cu_window_seqlens = self._window_index_thw(t, h, w)
        rotary = self._rotary_pos_emb_thw(t, h, w)[window_index].reshape(
            patch_rows,
            -1,
        )
        return _Qwen25VisionMetadata(
            window_index,
            cu_window_seqlens,
            rotary.astype(np.float32),
            np.asarray([patch_rows], dtype=np.int32),
        )

    def _empty_metadata(self, input_capacity: int) -> _Qwen25VisionMetadata:
        merge = self.spatial_merge_size
        t, h, w = 1, merge, input_capacity // merge
        window_index, cu_window_seqlens = self._window_index_thw(t, h, w)
        rotary = self._rotary_pos_emb_thw(t, h, w)[window_index].reshape(
            input_capacity,
            self.rot_dim,
        )
        return _Qwen25VisionMetadata(
            window_index,
            cu_window_seqlens,
            rotary.astype(np.float32),
            np.asarray([input_capacity], dtype=np.int32),
        )

    def _pad_metadata(
        self,
        metadata: _Qwen25VisionMetadata,
        input_capacity: int,
    ) -> _Qwen25VisionMetadata:
        output_capacity = input_capacity // self.spatial_merge_unit
        window_index = np.arange(output_capacity, dtype=np.int32)
        window_index[: metadata.window_index.shape[0]] = metadata.window_index
        cu_window_seqlens = np.full(output_capacity, input_capacity, dtype=np.int32)
        cu_window_seqlens[: metadata.cu_window_seqlens.shape[0]] = metadata.cu_window_seqlens
        rotary = np.zeros((input_capacity, self.rot_dim), dtype=np.float32)
        rotary[: metadata.rotary_pos_emb.shape[0]] = metadata.rotary_pos_emb
        cu_image_seqlens = np.full(output_capacity, input_capacity, dtype=np.int32)
        cu_image_seqlens[: metadata.cu_image_seqlens.shape[0]] = metadata.cu_image_seqlens
        return _Qwen25VisionMetadata(
            window_index,
            cu_window_seqlens,
            rotary,
            cu_image_seqlens,
        )

    def _mesh_ctx(self):
        return jax.set_mesh(self.mesh)

    @jax.jit
    def _encode_jit(self, patches: jax.Array, meta, valid: jax.Array | None = None) -> jax.Array:
        features = self(patches, meta, valid)
        if self.mesh is None:
            return features
        return apply_data_sharding(features, self.mesh, self.specs.output_spec(None, None))

    def _pin(self, *arrays: jax.Array) -> tuple[jax.Array, ...]:
        """Reshard leading batch axes to the ViT batch spec."""
        if self.mesh is None:
            return tuple(jnp.asarray(array) for array in arrays)
        return tuple(
            apply_data_sharding(
                jnp.asarray(array),
                self.mesh,
                self.specs.batch_spec(*([None] * (array.ndim - 1))),
            )
            for array in arrays
        )

    def _compute(
        self,
        patches: jax.Array,
        window_index: jax.Array,
        cu_window_seqlens: jax.Array,
        rotary_pos_emb: jax.Array,
        cu_image_seqlens: jax.Array,
        valid: jax.Array | None = None,
    ) -> jax.Array:
        """Full ViT forward: patch embed → window-reorder → blocks → merge → un-reorder."""
        B, S = patches.shape[:2]
        u = self.spatial_merge_unit
        n_units = S // u

        window_index, cu_window_seqlens, rotary_pos_emb, cu_image_seqlens = self._pin(
            window_index, cu_window_seqlens, rotary_pos_emb, cu_image_seqlens
        )
        if valid is not None:
            (valid,) = self._pin(valid)

        x = self.patch_embed(patches)
        x = x.reshape(B, n_units, u, -1)

        # Window reorder (batch axis stays on 0).
        x = jnp.take_along_axis(x, window_index[:, :, None, None], axis=1)
        x = x.reshape(B, S, -1)

        # Boundary vectors are padded to a shape-stable capacity.  Convert each
        # attention layout to dense ids once per encode, rather than rebuilding
        # the same comparison/reduction inside every ViT block.
        attention_layouts = {i in self.fullatt_block_indexes for i in range(len(self.blocks))}
        segment_ids_by_layout = {}
        if False in attention_layouts:
            segment_ids_by_layout[False] = _segment_ids_from_cu_seqlens(cu_window_seqlens, S)
        if True in attention_layouts:
            segment_ids_by_layout[True] = _segment_ids_from_cu_seqlens(cu_image_seqlens, S)
        if self.mesh is not None:
            segment_ids_by_layout = {
                layout: self._pin(segment_ids)[0]
                for layout, segment_ids in segment_ids_by_layout.items()
            }
        if attention_layouts and valid is not None:
            positions = jnp.arange(S, dtype=valid.dtype)
            is_real = positions[None, :] < jnp.reshape(valid, (-1, 1))
            if self.mesh is not None:
                (is_real,) = self._pin(is_real)
            segment_ids_by_layout = {
                layout: jax.lax.select(is_real, segment_ids, jnp.full_like(segment_ids, -1))
                for layout, segment_ids in segment_ids_by_layout.items()
            }

        for i, blk in enumerate(self.blocks):
            segment_ids = segment_ids_by_layout[i in self.fullatt_block_indexes]
            x = blk(x, rotary_pos_emb, segment_ids)

        x = self.merger(x)

        reverse = jnp.argsort(window_index, axis=1)
        return jnp.take_along_axis(x, reverse[:, :, None], axis=1)


class Qwen2_5_VLForConditionalGeneration(nnx.Module, InModelMultimodalContract):
    """Qwen2.5-VL: vision tower + Qwen2 backbone (+ MRoPE) + lm_head.

    The visual encode stays outside the backbone JIT.  MRoPE is handled
    transparently by ``Qwen2Model`` (mrope-aware RoPE + 3-D positions).
    """

    mrope_position_axes = 3

    def __init__(self, config=None, dtype=None, mesh=None, rngs=None):
        super().__init__()
        self.mesh = mesh
        self.config = config
        self.text_config = get_hf_text_config(config) or config
        self.dtype = dtype or jnp.bfloat16

        # Language backbone.
        self.model = Qwen2Model(self.text_config, mesh=mesh, dtype=self.dtype)
        if not getattr(self.text_config, "tie_word_embeddings", False):
            self.lm_head = ParallelLMHead(
                self.text_config.vocab_size,
                self.text_config.hidden_size,
                dtype=self.dtype,
                param_dtype=self.dtype,
                kernel_axes=("tensor", None),
            )
        self.logits_processor = LogitsProcessor(self.text_config.vocab_size, mesh=self.mesh)
        self.image_token_id = getattr(self.config, "image_token_id", None)
        self.video_token_id = getattr(self.config, "video_token_id", None)

        # Vision tower.
        self.visual_config = config.vision_config

        # The host layout isolates DP groups. Inside each group, DP balances
        # images over tensor devices while TP replicates inputs and shards weights.
        from sgl_jax.srt.managers.schedule_batch import global_server_args_dict

        vision_tp = resolve_encoder_tp(
            mesh, global_server_args_dict.get("vision_encoder_parallel", "dp")
        )
        self.visual = Qwen2_5_VisionTransformer(
            config=self.visual_config,
            dtype=self.dtype,
            rngs=rngs,
            mesh=mesh,
            norm_eps=getattr(self.visual_config, "rms_norm_eps", 1e-6),
            vision_tp=vision_tp,
            input_buckets=tuple(
                resolve_vision_patch_buckets(
                    global_server_args_dict.get("precompile_vision_patch_paddings")
                )
            ),
        )

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
        # Text backbone + lm_head.
        loader = WeightLoader(
            model=self, model_config=model_config, mesh=self.mesh, dtype=self.dtype
        )
        loader.load_weights_from_safetensors(create_qwen2_weight_mappings(self.text_config))
        logger.info("Qwen2.5-VL (LLM) weights loaded.")
        # ViT weights — carry vision head info so _split_qkv_weight can slice the
        # fused ``qkv.weight`` / ``qkv.bias`` into q_proj, k_proj, v_proj.
        vc = self.visual_config
        vision_model_config = SimpleNamespace(
            model_path=model_config.model_path,
            num_attention_heads=vc.num_heads,
            hidden_size=vc.hidden_size,
            get_total_num_kv_heads=lambda: vc.num_heads,  # no GQA in ViT
        )
        self._load_vision_weights(vision_model_config)

    def _load_vision_weights(self, model_config) -> None:
        loader = WeightLoader(
            model=self, model_config=model_config, mesh=self.mesh, dtype=self.dtype
        )
        mappings = self._vision_weight_mappings()
        if self.mesh is not None:
            with self.mesh:
                loader.load_weights_from_safetensors(mappings)
        else:
            loader.load_weights_from_safetensors(mappings)
        logger.info("Qwen2.5-VL ViT weights loaded.")

    def _vision_weight_mappings(self) -> dict:
        tp = self.visual.specs.tp
        col = (None, "tensor") if tp else (None, None)
        row = ("tensor", None) if tp else (None, None)

        mappings = {
            # Patch embed Conv3D: PyTorch [out,in,kd,kh,kw] → JAX [kd,kh,kw,in,out].
            "visual.patch_embed.proj.weight": WeightMapping(
                target_path="visual.patch_embed.proj.kernel",
                sharding=(None, None, None, None, None),
                transpose_axes=(2, 3, 4, 1, 0),
            ),
            "visual.merger.ln_q.weight": WeightMapping(
                target_path="visual.merger.ln_q.scale",
                sharding=(None,),
                transpose=False,
            ),
            **self._merger_mlp_mappings(col, row),
        }
        for i in range(getattr(self.visual_config, "depth", 0)):
            mappings.update(self._block_mappings(i, col, row))
        return mappings

    @staticmethod
    def _merger_mlp_mappings(col, row) -> dict:
        """Weight mappings for the patch merger MLP (mlp.0 / mlp.2 in HF)."""
        return {
            "visual.merger.mlp.0.weight": WeightMapping(
                target_path="visual.merger.mlp_fc1.weight", sharding=col, transpose=True
            ),
            "visual.merger.mlp.0.bias": WeightMapping(
                target_path="visual.merger.mlp_fc1.bias", sharding=(None,), transpose=False
            ),
            "visual.merger.mlp.2.weight": WeightMapping(
                target_path="visual.merger.mlp_fc2.weight", sharding=row, transpose=True
            ),
            "visual.merger.mlp.2.bias": WeightMapping(
                target_path="visual.merger.mlp_fc2.bias", sharding=(None,), transpose=False
            ),
        }

    @staticmethod
    def _block_mappings(layer_idx: int, col, row) -> dict:
        """Weight mappings for one ViT block (``visual.blocks.{i}.*``).

        The fused ``qkv.weight`` / ``qkv.bias`` are split into separate
        q/k/v projections so column-parallel sharding is TP-safe (each
        projection independently stripe-interleaves its own output slice).
        """
        p = f"visual.blocks.{layer_idx}"
        return {
            f"{p}.norm1.weight": WeightMapping(
                target_path=f"{p}.norm1.scale", sharding=(None,), transpose=False
            ),
            f"{p}.norm2.weight": WeightMapping(
                target_path=f"{p}.norm2.scale", sharding=(None,), transpose=False
            ),
            f"{p}.attn.qkv.weight": WeightMapping(
                target_path=[
                    f"{p}.attn.q_proj.weight",
                    f"{p}.attn.k_proj.weight",
                    f"{p}.attn.v_proj.weight",
                ],
                sharding=col,
                transpose=True,
            ),
            f"{p}.attn.qkv.bias": WeightMapping(
                target_path=[
                    f"{p}.attn.q_proj.bias",
                    f"{p}.attn.k_proj.bias",
                    f"{p}.attn.v_proj.bias",
                ],
                sharding=(None,),
                transpose=False,
            ),
            f"{p}.attn.proj.weight": WeightMapping(
                target_path=f"{p}.attn.proj.weight", sharding=row, transpose=True
            ),
            f"{p}.attn.proj.bias": WeightMapping(
                target_path=f"{p}.attn.proj.bias", sharding=(None,), transpose=False
            ),
            f"{p}.mlp.gate_proj.weight": WeightMapping(
                target_path=f"{p}.mlp.gate_proj.weight", sharding=col, transpose=True
            ),
            f"{p}.mlp.gate_proj.bias": WeightMapping(
                target_path=f"{p}.mlp.gate_proj.bias", sharding=(None,), transpose=False
            ),
            f"{p}.mlp.up_proj.weight": WeightMapping(
                target_path=f"{p}.mlp.up_proj.weight", sharding=col, transpose=True
            ),
            f"{p}.mlp.up_proj.bias": WeightMapping(
                target_path=f"{p}.mlp.up_proj.bias", sharding=(None,), transpose=False
            ),
            f"{p}.mlp.down_proj.weight": WeightMapping(
                target_path=f"{p}.mlp.down_proj.weight", sharding=row, transpose=True
            ),
            f"{p}.mlp.down_proj.bias": WeightMapping(
                target_path=f"{p}.mlp.down_proj.bias", sharding=(None,), transpose=False
            ),
        }

    def get_embed_and_head(self):
        if getattr(self.text_config, "tie_word_embeddings", False):
            w = self.model.embed_tokens.embedding.value
            return (w, w)
        return (self.model.embed_tokens.embedding.value, self.lm_head.embedding.value)

    def set_embed_and_head(
        self, embed_weight: jax.Array | None = None, head_weight: jax.Array | None = None
    ) -> None:
        if embed_weight is not None:
            self.model.embed_tokens.embedding.value = embed_weight
        if head_weight is not None:
            self.lm_head.embedding.value = head_weight

    def __call__(
        self,
        forward_batch: ForwardBatch,
        memory_pools: MemoryPools,
        logits_metadata: LogitsMetadata,
    ):
        hidden_states, layers_kv_fused, layers_callback_flag = self.model(
            forward_batch, memory_pools.token_to_kv_pool
        )
        head = (
            self.model.embed_tokens
            if getattr(self.text_config, "tie_word_embeddings", False)
            else self.lm_head
        )
        output = self.logits_processor(hidden_states, head, logits_metadata)
        return output, layers_kv_fused, layers_callback_flag, None


EntryClass = Qwen2_5_VLForConditionalGeneration
