"""MiMoV2 audio tower."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import NamedSharding, PartitionSpec

from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.multimodal.in_model.lane_packing import encoder_num_lanes
from sgl_jax.srt.multimodal.layers.vision_sharding import VisionShardSpecs


def _value(config: Mapping[str, Any] | object | None, name: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def _apply_rope(x: jax.Array, freqs: jax.Array) -> jax.Array:
    original_dtype = x.dtype
    x = x.astype(jnp.float32)
    half = x.shape[-1] // 2
    rotated = jnp.concatenate((-x[..., half:], x[..., :half]), axis=-1)
    cos = jnp.cos(freqs)[..., None, :]
    sin = jnp.sin(freqs)[..., None, :]
    return (x * cos + rotated * sin).astype(original_dtype)

# ---------------------------------------------------------------------------
# Audio tower (speech codes → per-channel embed → local transformer → project)
# ---------------------------------------------------------------------------


def _int_list(value, length: int) -> list[int]:
    if isinstance(value, str):
        values = [int(x) for x in value.split("-")]
    elif isinstance(value, int):
        values = [value]
    else:
        values = [int(x) for x in value]
    if len(values) == 1:
        values *= length
    if len(values) != length:
        raise ValueError(f"Expected {length} audio values, got {len(values)}.")
    return values


class MiMoAudioCodeEmbedding(nnx.Module):
    """Per-channel speech-code embedding lookup (padding index handled upstream)."""

    def __init__(self, size, features, dtype, mesh, specs):
        self.mesh = mesh
        self.specs = specs
        self.embedding = nnx.Param(
            jnp.zeros(
                (size, features),
                dtype=dtype,
                out_sharding=(
                    NamedSharding(mesh, PartitionSpec(None, None)) if mesh is not None else None
                ),
            )
        )

    def __call__(self, indices: jax.Array) -> jax.Array:
        emb = self.embedding[...]
        if self.mesh is None:
            return emb[indices]
        sh = NamedSharding(
            self.mesh, PartitionSpec(self.specs.batch_axis, *([None] * (indices.ndim - 1)))
        )
        return emb.at[indices].get(out_sharding=sh)


class MiMoAudioAttention(nnx.Module):
    """Full or causal self-attention with partial RoPE (no GQA), audio-local."""

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
        proj = lambda bias: LinearBase(
            hidden,
            self.heads * self.head_dim,
            mesh=mesh,
            use_bias=bias,
            kernel_axes=(None, None),
            params_dtype=dtype,
        )
        self.q_proj, self.k_proj, self.v_proj = proj(True), proj(True), proj(True)
        self.o_proj = LinearBase(
            self.heads * self.head_dim,
            hidden,
            mesh=mesh,
            use_bias=False,
            kernel_axes=(None, None),
            params_dtype=dtype,
        )

    def __call__(self, hidden: jax.Array) -> jax.Array:
        B, T = hidden.shape[:2]
        row = self.specs.sharding(self.specs.batch_axis)
        q, _ = self.q_proj(hidden, out_sharding=row)
        k, _ = self.k_proj(hidden, out_sharding=row)
        v, _ = self.v_proj(hidden, out_sharding=row)
        q, k, v = (x.reshape(B, T, self.heads, self.head_dim) for x in (q, k, v))
        positions = jnp.arange(T, dtype=jnp.float32)
        inv = 1.0 / (
            self.theta ** (jnp.arange(0, self.rotary_dim, 2, dtype=jnp.float32) / self.rotary_dim)
        )
        angles = jnp.outer(positions, inv)
        freqs = jnp.concatenate((angles, angles), axis=-1)[None]
        q = jnp.concatenate(
            (_apply_rope(q[..., : self.rotary_dim], freqs), q[..., self.rotary_dim :]), axis=-1
        )
        k = jnp.concatenate(
            (_apply_rope(k[..., : self.rotary_dim], freqs), k[..., self.rotary_dim :]), axis=-1
        )
        scores = jnp.einsum("bthd,bshd->bhts", q, k) / math.sqrt(self.head_dim)
        if not self.full_attention:
            scores = jnp.where(
                jnp.arange(T)[:, None] >= jnp.arange(T)[None, :],
                scores,
                jnp.finfo(scores.dtype).min,
            )
        probs = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(hidden.dtype)
        out = jnp.einsum("bhts,bshd->bthd", probs, v).reshape(B, T, self.heads * self.head_dim)
        return self.o_proj(out, out_sharding=row)[0]


class MiMoAudioMLP(nnx.Module):
    """Audio-local SwiGLU MLP (no bias)."""

    def __init__(self, config, dtype, mesh, specs):
        hidden = int(_value(config, "input_local_dim"))
        intermediate = int(_value(config, "input_local_intermediate_size"))
        self.specs = specs
        linear = lambda i, o: LinearBase(
            i, o, mesh=mesh, use_bias=False, kernel_axes=(None, None), params_dtype=dtype
        )
        self.gate_proj = linear(hidden, intermediate)
        self.up_proj = linear(hidden, intermediate)
        self.down_proj = linear(intermediate, hidden)

    def __call__(self, hidden: jax.Array) -> jax.Array:
        row = self.specs.sharding(self.specs.batch_axis)
        gate, _ = self.gate_proj(hidden, out_sharding=row)
        up, _ = self.up_proj(hidden, out_sharding=row)
        return self.down_proj(jax.nn.silu(gate) * up, out_sharding=row)[0]


class MiMoAudioBlock(nnx.Module):
    def __init__(self, config, dtype, mesh, specs):
        hidden = int(_value(config, "input_local_dim"))
        eps = float(_value(config, "rms_norm_eps", 1e-6))
        self.input_layernorm = nnx.RMSNorm(hidden, epsilon=eps, param_dtype=dtype, rngs=nnx.Rngs(0))
        self.post_attention_layernorm = nnx.RMSNorm(
            hidden, epsilon=eps, param_dtype=dtype, rngs=nnx.Rngs(0)
        )
        self.self_attn = MiMoAudioAttention(config, dtype, mesh, specs)
        self.mlp = MiMoAudioMLP(config, dtype, mesh, specs)

    def __call__(self, hidden: jax.Array) -> jax.Array:
        hidden = hidden + self.self_attn(self.input_layernorm(hidden))
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
            nnx.RMSNorm(
                int(_value(config, "input_local_dim")),
                epsilon=float(_value(config, "rms_norm_eps", 1e-6)),
                param_dtype=dtype,
                rngs=nnx.Rngs(0),
            )
            if bool(_value(config, "add_post_norm", True))
            else None
        )

    def __call__(self, hidden: jax.Array) -> jax.Array:
        for layer in self.layers:
            hidden = layer(hidden)
        return self.norm(hidden) if self.norm is not None else hidden


class MiMoAudioEncoder(nnx.Module):
    """Speech codes ``[B, cap, C]`` → grouped embed → local transformer → project."""

    def __init__(self, config, dtype, mesh, encoder_tp):
        self.config = config
        self.mesh = mesh
        self.dtype = dtype
        self.encoder_tp = encoder_tp
        self.specs = VisionShardSpecs(mesh, encoder_tp)
        self.channels = int(_value(config, "audio_channels"))
        self.group_size = int(_value(config, "group_size"))
        self.local_dim = int(_value(config, "input_local_dim"))
        self.out_hidden_size = int(_value(config, "out_hidden_size"))
        vocab_sizes = _int_list(_value(config, "speech_vocab_size"), self.channels)
        self.zero_ids = tuple(
            _int_list(
                _value(config, "speech_zeroemb_idx", _value(config, "zeroemb_idx")),
                self.channels,
            )
        )
        # Bucket ladder (multiples of group_size); oversize falls back to pow2.
        self.input_buckets = tuple(self.group_size * n for n in (16, 64, 256, 1024))

        self.speech_embeddings = nnx.List(
            [
                MiMoAudioCodeEmbedding(size, self.local_dim, dtype, mesh, self.specs)
                for size in vocab_sizes
            ]
        )
        self.transformer = MiMoAudioTransformer(config, dtype, mesh, self.specs)
        projection_layers = int(_value(config, "projection_layers", 2))
        projection_input = self.local_dim * self.group_size
        linear = lambda i, o: LinearBase(
            i, o, mesh=mesh, use_bias=False, kernel_axes=(None, None), params_dtype=dtype
        )
        if projection_layers == 1:
            self.proj_fc1 = linear(projection_input, self.out_hidden_size)
            self.proj_fc2 = None
        elif projection_layers == 2:
            self.proj_fc1 = linear(projection_input, projection_input * 4)
            self.proj_fc2 = linear(projection_input * 4, self.out_hidden_size)
        else:
            raise ValueError(f"Unsupported MiMoV2 audio projection_layers={projection_layers}.")

    def __call__(self, codes: jax.Array, valid: jax.Array) -> jax.Array:
        codes = codes.astype(jnp.int32)
        position_valid = jnp.arange(codes.shape[1])[None] < valid[:, None]
        zero_ids = jnp.asarray(self.zero_ids, dtype=jnp.int32)
        codes = jnp.where(position_valid[:, :, None], codes, zero_ids)
        B, T = codes.shape[:2]
        groups = T // self.group_size
        codes = codes.reshape(
            B,
            groups,
            self.group_size,
            self.channels,
            out_sharding=self.specs.sharding(self.specs.batch_axis, None, None, None),
        )
        hidden = jnp.zeros((B, groups, self.group_size, self.local_dim), dtype=self.dtype)
        for channel, embedding in enumerate(self.speech_embeddings):
            hidden += embedding(codes[..., channel])
        hidden = hidden.reshape(
            B * groups,
            self.group_size,
            self.local_dim,
            out_sharding=self.specs.sharding(self.specs.batch_axis, None, None),
        )
        hidden = self.transformer(hidden)
        hidden = hidden.reshape(
            B,
            groups,
            self.group_size * self.local_dim,
            out_sharding=self.specs.sharding(self.specs.batch_axis, None, None),
        )
        row = self.specs.sharding(self.specs.batch_axis)
        hidden, _ = self.proj_fc1(hidden, out_sharding=row)
        if self.proj_fc2 is not None:
            hidden = jax.nn.gelu(hidden, approximate=False)
            hidden, _ = self.proj_fc2(hidden, out_sharding=row)
        output_valid = valid // self.group_size
        return jnp.where(
            jnp.arange(groups)[None, :, None] < output_valid[:, None, None],
            hidden,
            0,
        )

    @nnx.jit
    def _encode_jit(self, codes, valid) -> jax.Array:
        features = self(codes, valid)
        if self.mesh is None:
            return features
        return jax.sharding.reshard(
            features,
            NamedSharding(self.mesh, PartitionSpec(*([None] * features.ndim))),
        )

    def encode(self, codes, valid) -> jax.Array:
        if self.mesh is None:
            return self._encode_jit(codes, valid)
        with jax.set_mesh(self.mesh):
            return self._encode_jit(codes, valid)

    def precompile(self) -> None:
        num_lanes = encoder_num_lanes(self.mesh, self.encoder_tp)
        batch_sharding = self.specs.sharding(self.specs.batch_axis)
        for capacity in self.input_buckets:
            codes = jax.device_put(
                np.zeros((num_lanes, capacity, self.channels), dtype=np.int32),
                batch_sharding,
            )
            valid = np.zeros(num_lanes, dtype=np.int32)
            valid[0] = capacity
            valid = jax.device_put(valid, batch_sharding)
            jax.block_until_ready(self.encode(codes, valid))
