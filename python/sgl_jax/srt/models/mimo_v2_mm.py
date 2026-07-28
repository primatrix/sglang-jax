from __future__ import annotations

import logging
import math
from collections.abc import Callable
from types import SimpleNamespace

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.layers.layernorm import RMSNorm
from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.models.mimo_v2_flash import MiMoV2FlashForCausalLM
from sgl_jax.srt.models.mimo_v2_pro import MiMoV2ForCausalLM
from sgl_jax.srt.multimodal.common.modality_enum import Modality
from sgl_jax.srt.multimodal.in_model.encoder_planning import EncodeInputs
from sgl_jax.srt.multimodal.in_model.encoders.mimo_v2 import MiMoV2PlanBuilder
from sgl_jax.srt.multimodal.in_model.registry import register_encoder_plan_builder
from sgl_jax.srt.multimodal.kernels.flash_attention import SegmentIds
from sgl_jax.srt.multimodal.layers.vision_sharding import (
    VisionShardSpecs,
    apply_data_sharding,
    resolve_encoder_tp,
)
from sgl_jax.srt.utils.weight_utils import WeightLoader, WeightMapping

logger = logging.getLogger(__name__)

_ARCHITECTURES = (
    "MiMoV2ForConditionalGeneration",
    "MiMoV2FlashForConditionalGeneration",
)
for _architecture in _ARCHITECTURES:
    register_encoder_plan_builder(_architecture, MiMoV2PlanBuilder)


def _value(config, name, default=None):
    return config.get(name, default) if isinstance(config, dict) else getattr(config, name, default)


def _int_list(value, length):
    if isinstance(value, str):
        values = [int(item) for item in value.split("-")]
    elif isinstance(value, (list, tuple)):
        values = [int(item) for item in value]
    else:
        values = [int(value)]
    if len(values) == 1:
        values *= length
    if len(values) != length:
        raise ValueError(f"Expected {length} values, got {len(values)}.")
    return values


def _apply_rope(x, freqs):
    original_dtype = x.dtype
    x = x.astype(jnp.float32)
    half = x.shape[-1] // 2
    rotated = jnp.concatenate((-x[..., half:], x[..., :half]), axis=-1)
    cos, sin = jnp.cos(freqs)[:, :, None, :], jnp.sin(freqs)[:, :, None, :]
    return (x * cos + rotated * sin).astype(original_dtype)


def _take_units(x, index, unit):
    batch, length = x.shape[:2]
    tail = x.shape[2:]
    x = x.reshape(batch, length // unit, unit, *tail)
    gather = index.reshape(batch, index.shape[1], *([1] * (x.ndim - 2)))
    gather = jnp.broadcast_to(gather, (batch, index.shape[1], *x.shape[2:]))
    return jnp.take_along_axis(x, gather, axis=1).reshape(batch, length, *tail)


def _dense_vision_attention(q, k, v, segments, window_size, sinks):
    scores = jnp.einsum("bthd,bshd->bhts", q, k) / math.sqrt(q.shape[-1])
    valid = segments >= 0
    mask = valid[:, :, None] & valid[:, None, :]
    mask &= segments[:, :, None] == segments[:, None, :]
    if window_size > 0:
        positions = jnp.arange(q.shape[1])
        mask &= jnp.abs(positions[:, None] - positions[None, :]) <= window_size
    if sinks is not None:
        starts = valid & jnp.concatenate(
            (
                jnp.ones((segments.shape[0], 1), dtype=jnp.bool_),
                segments[:, 1:] != segments[:, :-1],
            ),
            axis=1,
        )
        sink_mask = mask & starts[:, None, :]
        scores += sinks[None, :, None, None].astype(scores.dtype) * sink_mask[:, None]
    scores = jnp.where(mask[:, None], scores, jnp.finfo(scores.dtype).min)
    probs = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(q.dtype)
    output = jnp.einsum("bhts,bshd->bthd", probs, v)
    return jnp.where(valid[:, :, None, None], output, 0)


def _flash_vision_attention(backend, q, k, v, segments):
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


class MiMoVisionPatchEmbed(nnx.Module):
    def __init__(self, config, dtype, rngs, mesh, tp):
        self.channels = int(_value(config, "in_channels", None) or _value(config, "in_chans", 3))
        self.temporal = int(_value(config, "temporal_patch_size", 2))
        self.patch = int(_value(config, "patch_size", 16))
        self.hidden = int(_value(config, "hidden_size"))
        self.specs = VisionShardSpecs(mesh, tp)
        self.proj = nnx.Conv(
            self.channels,
            self.hidden,
            (self.temporal, self.patch, self.patch),
            strides=(self.temporal, self.patch, self.patch),
            use_bias=False,
            dtype=dtype,
            param_dtype=dtype,
            rngs=rngs,
        )

    def __call__(self, patches):
        batch, length = patches.shape[:2]
        flat_sharding = self.specs.batch_sharding(None, None, None, None)
        patches = patches.reshape(
            batch * length,
            self.channels,
            self.temporal,
            self.patch,
            self.patch,
            out_sharding=flat_sharding,
        )
        patches = jnp.transpose(patches, (0, 2, 3, 4, 1))
        hidden = self.proj(patches, out_sharding=flat_sharding)
        return hidden.reshape(
            batch, length, self.hidden, out_sharding=self.specs.batch_sharding(None, None)
        )


class MiMoVisionMLP(nnx.Module):
    def __init__(self, config, dtype, mesh, specs):
        hidden = int(_value(config, "hidden_size"))
        intermediate = int(_value(config, "intermediate_size"))
        self.gate_proj = LinearBase(
            hidden,
            intermediate,
            mesh,
            use_bias=True,
            params_dtype=dtype,
            kernel_axes=specs.col_kernel_axes,
        )
        self.up_proj = LinearBase(
            hidden,
            intermediate,
            mesh,
            use_bias=True,
            params_dtype=dtype,
            kernel_axes=specs.col_kernel_axes,
        )
        self.down_proj = LinearBase(
            intermediate,
            hidden,
            mesh,
            use_bias=True,
            params_dtype=dtype,
            kernel_axes=specs.row_kernel_axes,
        )
        self.specs = specs
        self.activation = _value(config, "hidden_act", "silu")

    def __call__(self, hidden):
        gate, _ = self.gate_proj(hidden, out_sharding=self.specs.col_out(hidden.ndim))
        up, _ = self.up_proj(hidden, out_sharding=self.specs.col_out(hidden.ndim))
        activation = jax.nn.silu if self.activation == "silu" else jax.nn.gelu
        return self.down_proj(
            activation(gate) * up,
            out_sharding=self.specs.row_out(hidden.ndim),
        )[0]


class MiMoVisionAttention(nnx.Module):
    def __init__(self, config, dtype, mesh, specs, use_sinks):
        hidden = int(_value(config, "hidden_size"))
        self.heads = int(_value(config, "num_heads"))
        self.kv_heads = int(_value(config, "num_key_value_heads", self.heads) or self.heads)
        self.head_dim = int(_value(config, "qk_channels", 64))
        if self.heads % self.kv_heads:
            raise ValueError("MiMoV2 vision heads must be divisible by key/value heads.")
        if self.head_dim % 4:
            raise ValueError("MiMoV2 vision head dimension must be divisible by four.")
        self.specs = specs
        linear = lambda output: LinearBase(
            hidden,
            output,
            mesh,
            use_bias=True,
            params_dtype=dtype,
            kernel_axes=specs.col_kernel_axes,
        )
        self.q_proj = linear(self.heads * self.head_dim)
        self.k_proj = linear(self.kv_heads * self.head_dim)
        self.v_proj = linear(self.kv_heads * self.head_dim)
        self.proj = LinearBase(
            self.heads * self.head_dim,
            hidden,
            mesh,
            use_bias=True,
            params_dtype=dtype,
            kernel_axes=specs.row_kernel_axes,
        )
        sink_sharding = NamedSharding(mesh, P("tensor" if specs.tp else None))
        self.sinks = (
            nnx.Param(jnp.zeros((self.heads,), dtype=dtype, out_sharding=sink_sharding))
            if use_sinks
            else None
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

    def __call__(self, hidden, freqs, segments, window_size):
        batch, length = hidden.shape[:2]
        q, _ = self.q_proj(hidden, out_sharding=self.specs.col_out(hidden.ndim))
        k, _ = self.k_proj(hidden, out_sharding=self.specs.col_out(hidden.ndim))
        v, _ = self.v_proj(hidden, out_sharding=self.specs.col_out(hidden.ndim))
        sharding = self.specs.qkv_reshape_sharding()
        q = q.reshape(batch, length, self.heads, self.head_dim, out_sharding=sharding)
        k = k.reshape(batch, length, self.kv_heads, self.head_dim, out_sharding=sharding)
        v = v.reshape(batch, length, self.kv_heads, self.head_dim, out_sharding=sharding)
        q, k = _apply_rope(q, freqs), _apply_rope(k, freqs)
        if self.kv_heads != self.heads:
            groups = self.heads // self.kv_heads
            k, v = jnp.repeat(k, groups, axis=2), jnp.repeat(v, groups, axis=2)
        if self.backend is not None and window_size <= 0 and self.sinks is None:
            output = _flash_vision_attention(self.backend, q, k, v, segments)
        else:
            sinks = None if self.sinks is None else self.sinks[...]
            output = _dense_vision_attention(q, k, v, segments, window_size, sinks)
        output = output.reshape(
            batch,
            length,
            self.heads * self.head_dim,
            out_sharding=self.specs.col_out(3),
        )
        return self.proj(output, out_sharding=self.specs.row_out(3))[0]


class MiMoVisionBlock(nnx.Module):
    def __init__(self, config, dtype, rngs, mesh, tp, use_sinks):
        hidden = int(_value(config, "hidden_size"))
        epsilon = float(_value(config, "rms_norm_eps", 1e-6))
        self.norm1 = nnx.RMSNorm(hidden, epsilon=epsilon, dtype=dtype, param_dtype=dtype, rngs=rngs)
        self.norm2 = nnx.RMSNorm(hidden, epsilon=epsilon, dtype=dtype, param_dtype=dtype, rngs=rngs)
        specs = VisionShardSpecs(mesh, tp)
        self.attn = MiMoVisionAttention(config, dtype, mesh, specs, use_sinks)
        self.mlp = MiMoVisionMLP(config, dtype, mesh, specs)

    def __call__(self, hidden, freqs, segments, window_size):
        hidden += self.attn(self.norm1(hidden), freqs, segments, window_size)
        return hidden + self.mlp(self.norm2(hidden))


class MiMoVisionPatchMerger(nnx.Module):
    def __init__(self, config, dtype, rngs, mesh, tp):
        context = int(_value(config, "hidden_size"))
        unit = int(_value(config, "spatial_merge_size", 2)) ** 2
        hidden = context * unit
        self.unit = unit
        self.specs = VisionShardSpecs(mesh, tp)
        self.ln_q = nnx.LayerNorm(
            context,
            epsilon=1e-6,
            dtype=dtype,
            param_dtype=dtype,
            use_fast_variance=False,
            rngs=rngs,
        )
        self.mlp_fc1 = LinearBase(
            hidden,
            hidden,
            mesh,
            use_bias=True,
            params_dtype=dtype,
            kernel_axes=self.specs.col_kernel_axes,
        )
        self.mlp_fc2 = LinearBase(
            hidden,
            int(_value(config, "out_hidden_size")),
            mesh,
            use_bias=True,
            params_dtype=dtype,
            kernel_axes=self.specs.row_kernel_axes,
        )

    def __call__(self, hidden):
        hidden = self.ln_q(hidden).reshape(
            hidden.shape[0],
            -1,
            hidden.shape[-1] * self.unit,
            out_sharding=self.specs.batch_sharding(None, None),
        )
        hidden, _ = self.mlp_fc1(hidden, out_sharding=self.specs.col_out(hidden.ndim))
        hidden = jax.nn.gelu(hidden, approximate=False)
        return self.mlp_fc2(hidden, out_sharding=self.specs.row_out(hidden.ndim))[0]


class MiMoVisionTransformer(nnx.Module):
    def __init__(self, config, dtype, rngs, mesh, tp):
        self.config = config
        self.mesh = mesh
        self.specs = VisionShardSpecs(mesh, tp)
        self.unit = int(_value(config, "spatial_merge_size", 2)) ** 2
        depth = int(_value(config, "depth"))
        full = tuple(int(i) for i in _value(config, "fullatt_block_indexes", ()))
        self.full_blocks = frozenset(full)
        self.window_types = tuple(_value(config, "vit_window_attn_types", None) or [-1] * depth)
        if len(self.window_types) != depth:
            raise ValueError("vit_window_attn_types must have one entry per vision block.")
        use_sink = bool(_value(config, "use_sink", False))
        self.window_size = int(_value(config, "visual_token_window_size", -1))
        self.patch_embed = MiMoVisionPatchEmbed(config, dtype, rngs, mesh, tp)
        self.blocks = nnx.List(
            [
                MiMoVisionBlock(config, dtype, rngs, mesh, tp, use_sink and i not in full)
                for i in range(depth)
            ]
        )
        self.merger = MiMoVisionPatchMerger(config, dtype, rngs, mesh, tp)

    def __call__(self, patches, meta, valid):
        hidden = self.patch_embed(patches)
        segments = jnp.where(
            jnp.arange(hidden.shape[1])[None] < valid[:, None],
            meta.segment_ids,
            -1,
        )
        col_freqs = _take_units(meta.rotary_freqs, meta.col_index, self.unit)
        reverse_col_index = jnp.argsort(meta.col_index, axis=1)
        for index, block in enumerate(self.blocks):
            col = self.window_types[index] == 1
            previous_col = index > 0 and self.window_types[index - 1] == 1
            if col and not previous_col:
                hidden = _take_units(hidden, meta.col_index, self.unit)
            elif previous_col and not col:
                hidden = _take_units(hidden, reverse_col_index, self.unit)
            freqs = col_freqs if col else meta.rotary_freqs
            window = -1 if index in self.full_blocks else self.window_size
            hidden = block(hidden, freqs, segments, window)
        output = self.merger(hidden)
        output_valid = valid // self.unit
        return jnp.where(
            jnp.arange(output.shape[1])[None, :, None] < output_valid[:, None, None],
            output,
            0,
        )

    @jax.jit
    def encode(self, inputs):
        output = self(inputs.features, inputs.meta, inputs.valid)
        if self.mesh is not None:
            output = apply_data_sharding(output, self.mesh, self.specs.batch_spec(None, None))
        return output


class MiMoAudioCodeEmbedding(nnx.Module):
    def __init__(self, size, features, dtype, mesh, specs):
        self.embedding = nnx.Param(
            jax.random.normal(
                jax.random.PRNGKey(0),
                (size, features),
                dtype=dtype,
                out_sharding=NamedSharding(mesh, P(None, None)),
            )
        )
        self.mesh = mesh
        self.specs = specs

    def __call__(self, indices):
        return (
            self.embedding[...]
            .at[indices]
            .get(
                out_sharding=NamedSharding(
                    self.mesh,
                    self.specs.batch_spec(*([None] * indices.ndim)),
                )
            )
        )


class MiMoAudioAttention(nnx.Module):
    def __init__(self, config, dtype, mesh, specs):
        hidden = int(_value(config, "input_local_dim"))
        self.heads = int(_value(config, "input_local_attn_heads"))
        self.head_dim = int(_value(config, "input_local_head_dim", hidden // self.heads))
        self.rotary_dim = int(self.head_dim * float(_value(config, "partial_rotary_factor", 1.0)))
        if self.rotary_dim % 2:
            raise ValueError("MiMoV2 audio rotary dimension must be even.")
        self.theta = float(_value(config, "rope_theta", 640000.0))
        self.full_attention = bool(_value(config, "input_full_attention", True))
        self.specs = specs
        linear = lambda: LinearBase(
            hidden,
            self.heads * self.head_dim,
            mesh,
            use_bias=True,
            params_dtype=dtype,
            kernel_axes=(None, None),
        )
        self.q_proj, self.k_proj, self.v_proj = linear(), linear(), linear()
        self.o_proj = LinearBase(
            self.heads * self.head_dim,
            hidden,
            mesh,
            use_bias=False,
            params_dtype=dtype,
            kernel_axes=(None, None),
        )

    def __call__(self, hidden):
        batch, length = hidden.shape[:2]
        out_sharding = self.specs.row_out(hidden.ndim)
        q, _ = self.q_proj(hidden, out_sharding=out_sharding)
        k, _ = self.k_proj(hidden, out_sharding=out_sharding)
        v, _ = self.v_proj(hidden, out_sharding=out_sharding)
        q, k, v = (value.reshape(batch, length, self.heads, self.head_dim) for value in (q, k, v))
        positions = jnp.arange(length, dtype=jnp.float32)
        inv = 1.0 / (
            self.theta ** (jnp.arange(0, self.rotary_dim, 2, dtype=jnp.float32) / self.rotary_dim)
        )
        angles = jnp.outer(positions, inv)
        freqs = jnp.concatenate((angles, angles), axis=-1)[None]
        q_rot, k_rot = (
            _apply_rope(q[..., : self.rotary_dim], freqs),
            _apply_rope(k[..., : self.rotary_dim], freqs),
        )
        q = jnp.concatenate((q_rot, q[..., self.rotary_dim :]), axis=-1)
        k = jnp.concatenate((k_rot, k[..., self.rotary_dim :]), axis=-1)
        scores = jnp.einsum("bthd,bshd->bhts", q, k) / math.sqrt(self.head_dim)
        if not self.full_attention:
            scores = jnp.where(
                jnp.arange(length)[:, None] >= jnp.arange(length)[None, :],
                scores,
                jnp.finfo(scores.dtype).min,
            )
        probs = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(hidden.dtype)
        output = jnp.einsum("bhts,bshd->bthd", probs, v).reshape(
            batch, length, self.heads * self.head_dim
        )
        return self.o_proj(output, out_sharding=out_sharding)[0]


class MiMoAudioMLP(nnx.Module):
    def __init__(self, config, dtype, mesh, specs):
        hidden = int(_value(config, "input_local_dim"))
        intermediate = int(_value(config, "input_local_intermediate_size"))
        linear = lambda input_size, output_size: LinearBase(
            input_size,
            output_size,
            mesh,
            use_bias=False,
            params_dtype=dtype,
            kernel_axes=(None, None),
        )
        self.gate_proj = linear(hidden, intermediate)
        self.up_proj = linear(hidden, intermediate)
        self.down_proj = linear(intermediate, hidden)
        self.specs = specs

    def __call__(self, hidden):
        out_sharding = self.specs.row_out(hidden.ndim)
        gate, _ = self.gate_proj(hidden, out_sharding=out_sharding)
        up, _ = self.up_proj(hidden, out_sharding=out_sharding)
        return self.down_proj(jax.nn.silu(gate) * up, out_sharding=out_sharding)[0]


class MiMoAudioBlock(nnx.Module):
    def __init__(self, config, dtype, mesh, specs):
        hidden = int(_value(config, "input_local_dim"))
        epsilon = float(_value(config, "rms_norm_eps", 1e-6))
        self.input_layernorm = RMSNorm(hidden, epsilon=epsilon, param_dtype=dtype)
        self.post_attention_layernorm = RMSNorm(hidden, epsilon=epsilon, param_dtype=dtype)
        self.self_attn = MiMoAudioAttention(config, dtype, mesh, specs)
        self.mlp = MiMoAudioMLP(config, dtype, mesh, specs)

    def __call__(self, hidden):
        hidden += self.self_attn(self.input_layernorm(hidden))
        return hidden + self.mlp(self.post_attention_layernorm(hidden))


class MiMoAudioTransformer(nnx.Module):
    def __init__(self, config, dtype, mesh, specs):
        self.layers = nnx.List(
            [
                MiMoAudioBlock(config, dtype, mesh, specs)
                for _ in range(int(_value(config, "input_local_layers")))
            ]
        )
        self.norm = (
            RMSNorm(
                int(_value(config, "input_local_dim")),
                epsilon=float(_value(config, "rms_norm_eps", 1e-6)),
                param_dtype=dtype,
            )
            if bool(_value(config, "add_post_norm", True))
            else None
        )

    def __call__(self, hidden):
        for layer in self.layers:
            hidden = layer(hidden)
        return self.norm(hidden) if self.norm is not None else hidden


class MiMoAudioEncoder(nnx.Module):
    def __init__(self, config, dtype, mesh, encoder_tp):
        self.config = config
        self.mesh = mesh
        self.dtype = dtype
        self.specs = VisionShardSpecs(mesh, encoder_tp)
        self.channels = int(_value(config, "audio_channels"))
        self.group_size = int(_value(config, "group_size"))
        self.local_dim = int(_value(config, "input_local_dim"))
        self.output_size = int(_value(config, "out_hidden_size"))
        vocab_sizes = _int_list(_value(config, "speech_vocab_size"), self.channels)
        self.zero_ids = tuple(
            _int_list(
                _value(config, "speech_zeroemb_idx", _value(config, "zeroemb_idx")),
                self.channels,
            )
        )
        self.speech_embeddings = nnx.List(
            [
                MiMoAudioCodeEmbedding(size, self.local_dim, dtype, mesh, self.specs)
                for size in vocab_sizes
            ]
        )
        self.transformer = MiMoAudioTransformer(config, dtype, mesh, self.specs)
        projection_layers = int(_value(config, "projection_layers", 2))
        projection_input = self.local_dim * self.group_size
        if projection_layers == 1:
            self.proj_fc1 = LinearBase(
                projection_input,
                self.output_size,
                mesh,
                use_bias=False,
                params_dtype=dtype,
                kernel_axes=(None, None),
            )
            self.proj_fc2 = None
        elif projection_layers == 2:
            self.proj_fc1 = LinearBase(
                projection_input,
                projection_input * 4,
                mesh,
                use_bias=False,
                params_dtype=dtype,
                kernel_axes=(None, None),
            )
            self.proj_fc2 = LinearBase(
                projection_input * 4,
                self.output_size,
                mesh,
                use_bias=False,
                params_dtype=dtype,
                kernel_axes=(None, None),
            )
        else:
            raise ValueError(f"Unsupported MiMoV2 audio projection_layers={projection_layers}.")

    def __call__(self, codes, valid):
        codes = codes.astype(jnp.int32)
        position_valid = jnp.arange(codes.shape[1])[None] < valid[:, None]
        zero_ids = jnp.asarray(self.zero_ids, dtype=jnp.int32)
        codes = jnp.where(position_valid[:, :, None], codes, zero_ids)
        batch, length = codes.shape[:2]
        groups = length // self.group_size
        codes = codes.reshape(
            batch,
            groups,
            self.group_size,
            self.channels,
            out_sharding=self.specs.batch_sharding(None, None, None),
        )
        hidden = jnp.zeros(
            (batch, groups, self.group_size, self.local_dim),
            dtype=self.dtype,
        )
        for channel, embedding in enumerate(self.speech_embeddings):
            ids = codes[..., channel]
            hidden += embedding(ids)
        hidden = hidden.reshape(
            batch * groups,
            self.group_size,
            self.local_dim,
            out_sharding=self.specs.batch_sharding(None, None),
        )
        hidden = self.transformer(hidden)
        hidden = hidden.reshape(
            batch,
            groups,
            self.group_size * self.local_dim,
            out_sharding=self.specs.batch_sharding(None, None),
        )
        out_sharding = self.specs.row_out(hidden.ndim)
        hidden, _ = self.proj_fc1(hidden, out_sharding=out_sharding)
        if self.proj_fc2 is not None:
            hidden = jax.nn.gelu(hidden, approximate=False)
            hidden, _ = self.proj_fc2(hidden, out_sharding=out_sharding)
        output_valid = valid // self.group_size
        return jnp.where(
            jnp.arange(groups)[None, :, None] < output_valid[:, None, None],
            hidden,
            0,
        )

    @jax.jit
    def encode(self, inputs):
        output = self(inputs.features, inputs.valid)
        return apply_data_sharding(output, self.mesh, self.specs.batch_spec(None, None))


class _MiMoV2MultimodalMixin:
    materialize_input_embeddings = True

    def __init__(self, config=None, dtype=None, mesh=None, **kwargs):
        super().__init__(config=config, dtype=dtype, mesh=mesh, **kwargs)
        from sgl_jax.srt.managers.schedule_batch import global_server_args_dict

        self.encoder_tp = resolve_encoder_tp(
            mesh, global_server_args_dict.get("vision_encoder_parallel", "dp")
        )
        rngs = nnx.Rngs(0)
        vision_config = getattr(config, "vision_config", None)
        audio_config = getattr(config, "audio_config", None)
        self.visual = (
            MiMoVisionTransformer(
                vision_config,
                self.dtype,
                rngs,
                mesh,
                self.encoder_tp,
            )
            if vision_config is not None
            else None
        )
        self.audio_encoder = (
            MiMoAudioEncoder(audio_config, self.dtype, mesh, self.encoder_tp)
            if audio_config is not None
            else None
        )

    def get_multimodal_encoder(self, modality: Modality) -> Callable[[EncodeInputs], jax.Array]:
        if modality is Modality.IMAGE and self.visual is not None:
            return self.visual.encode
        if modality is Modality.AUDIO and self.audio_encoder is not None:
            return self.audio_encoder.encode
        raise ValueError(f"{type(self).__name__} does not support {modality.name} encoding.")

    def load_weights(self, model_config):
        super().load_weights(model_config)
        mappings = self._tower_weight_mappings()
        if not mappings:
            return
        vision = getattr(self.config, "vision_config", None)
        heads = int(_value(vision, "num_heads", 1))
        kv_heads = int(_value(vision, "num_key_value_heads", heads) or heads)
        head_dim = int(_value(vision, "qk_channels", 1))
        tower_config = SimpleNamespace(
            model_path=model_config.model_path,
            quantization_config=None,
            hf_config=self.config,
            hf_text_config=SimpleNamespace(head_dim=head_dim, v_head_dim=head_dim),
            num_attention_heads=heads,
            hidden_size=heads * head_dim,
            get_total_num_kv_heads=lambda: kv_heads,
            _dummy_mode=getattr(model_config, "_dummy_mode", False),
        )
        loader = WeightLoader(self, tower_config, self.mesh, self.dtype)
        loader.load_weights_from_safetensors(mappings)
        logger.info("MiMoV2 multimodal tower weights loaded successfully.")

    def _tower_weight_mappings(self):
        mappings = {}
        if self.visual is not None:
            mappings.update(self._vision_weight_mappings())
        if self.audio_encoder is not None:
            mappings.update(self._audio_weight_mappings())
        return mappings

    def _vision_weight_mappings(self):
        specs = self.visual.specs
        col, row = specs.col_kernel_axes, specs.row_kernel_axes
        mappings = {
            "visual.patch_embed.proj.weight": WeightMapping(
                "visual.patch_embed.proj.kernel",
                (None, None, None, None, None),
                transpose_axes=(2, 3, 4, 1, 0),
            ),
            "visual.merger.ln_q.weight": WeightMapping(
                "visual.merger.ln_q.scale", (None,), transpose=False
            ),
            "visual.merger.ln_q.bias": WeightMapping(
                "visual.merger.ln_q.bias", (None,), transpose=False
            ),
        }
        mappings.update(self._linear_mappings("visual.merger.mlp.0", "visual.merger.mlp_fc1", col))
        mappings.update(self._linear_mappings("visual.merger.mlp.2", "visual.merger.mlp_fc2", row))
        for index, block in enumerate(self.visual.blocks):
            source = target = f"visual.blocks.{index}"
            for norm in ("norm1", "norm2"):
                mappings[f"{source}.{norm}.weight"] = WeightMapping(
                    f"{target}.{norm}.scale", (None,), transpose=False
                )
            mappings[f"{source}.attn.qkv.weight"] = WeightMapping(
                [f"{target}.attn.{name}_proj.weight" for name in ("q", "k", "v")],
                col,
                transpose=True,
            )
            mappings[f"{source}.attn.qkv.bias"] = WeightMapping(
                [f"{target}.attn.{name}_proj.bias" for name in ("q", "k", "v")],
                (col[-1],),
                transpose=False,
            )
            mappings.update(
                self._linear_mappings(f"{source}.attn.proj", f"{target}.attn.proj", row)
            )
            for name in ("gate_proj", "up_proj"):
                mappings.update(
                    self._linear_mappings(f"{source}.mlp.{name}", f"{target}.mlp.{name}", col)
                )
            mappings.update(
                self._linear_mappings(f"{source}.mlp.down_proj", f"{target}.mlp.down_proj", row)
            )
            if block.attn.sinks is not None:
                mappings[f"{source}.attn.sinks"] = WeightMapping(
                    f"{target}.attn.sinks", ("tensor" if specs.tp else None,), transpose=False
                )
        return mappings

    def _audio_weight_mappings(self):
        encoder = self.audio_encoder
        mappings = {}
        for index in range(encoder.channels):
            mappings[f"speech_embeddings.{index}.weight"] = WeightMapping(
                f"audio_encoder.speech_embeddings.{index}.embedding",
                (None, None),
                transpose=False,
            )
        source_root = "audio_encoder.input_local_transformer"
        target_root = "audio_encoder.transformer"
        if encoder.transformer.norm is not None:
            mappings[f"{source_root}.norm.weight"] = WeightMapping(
                f"{target_root}.norm.scale", (None,), transpose=False
            )
        for index in range(len(encoder.transformer.layers)):
            source = f"{source_root}.layers.{index}"
            target = f"{target_root}.layers.{index}"
            for norm in ("input_layernorm", "post_attention_layernorm"):
                mappings[f"{source}.{norm}.weight"] = WeightMapping(
                    f"{target}.{norm}.scale", (None,), transpose=False
                )
            for name in ("q_proj", "k_proj", "v_proj"):
                mappings.update(
                    self._linear_mappings(
                        f"{source}.self_attn.{name}",
                        f"{target}.self_attn.{name}",
                        (None, None),
                    )
                )
            mappings[f"{source}.self_attn.o_proj.weight"] = WeightMapping(
                f"{target}.self_attn.o_proj.weight", (None, None), transpose=True
            )
            for name in ("gate_proj", "up_proj", "down_proj"):
                mappings[f"{source}.mlp.{name}.weight"] = WeightMapping(
                    f"{target}.mlp.{name}.weight", (None, None), transpose=True
                )
        if encoder.proj_fc2 is None:
            mappings["audio_encoder.projection.weight"] = WeightMapping(
                "audio_encoder.proj_fc1.weight", (None, None), transpose=True
            )
        else:
            mappings["audio_encoder.projection.mlp.0.weight"] = WeightMapping(
                "audio_encoder.proj_fc1.weight", (None, None), transpose=True
            )
            mappings["audio_encoder.projection.mlp.2.weight"] = WeightMapping(
                "audio_encoder.proj_fc2.weight", (None, None), transpose=True
            )
        return mappings

    @staticmethod
    def _linear_mappings(source, target, sharding):
        return {
            f"{source}.weight": WeightMapping(f"{target}.weight", sharding, transpose=True),
            f"{source}.bias": WeightMapping(f"{target}.bias", (sharding[-1],), transpose=False),
        }


class MiMoV2ForConditionalGeneration(_MiMoV2MultimodalMixin, MiMoV2ForCausalLM):
    pass


class MiMoV2FlashForConditionalGeneration(_MiMoV2MultimodalMixin, MiMoV2FlashForCausalLM):
    pass


EntryClass = [MiMoV2ForConditionalGeneration, MiMoV2FlashForConditionalGeneration]
