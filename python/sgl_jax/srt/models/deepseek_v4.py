"""DeepSeek V4 trunk: model parameters, mixed-checkpoint loading and forward graph.

MTP keys are explicitly excluded. Attention backends own cache execution; all
V4 weights, including compressors, indexer, mHC and routing, are owned here.
"""

from __future__ import annotations

import enum
import logging
import re
import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.configs.deepseek_v4 import (
    DeepseekV4LayerType,
    classify_layers,
    hash_moe_layer_flags,
)
from sgl_jax.srt.layers.activation import silu_and_mul_with_clamp
from sgl_jax.srt.layers.gate import GateLogit, TopK
from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.layers.moe import EPMoE
from sgl_jax.srt.utils.weight_utils import WeightMapping

logger = logging.getLogger(__name__)

__all__ = [
    "Disposition",
    "KeyFacts",
    "build_weight_mappings",
    "classify_checkpoint",
    "classify_key",
    "expected_trunk_keys",
]


class Disposition(enum.Enum):
    CLAIMED = "claimed"
    # Routed-expert weight or scale: M0.2 owns the resident layout.
    EXPERT_CONVERTED = "expert_converted"
    DROPPED = "dropped"


@dataclass(frozen=True)
class KeyFacts:
    disposition: Disposition
    reason: str
    layer: int | None = None


_LAYER = re.compile(r"^layers\.(\d+)\.(.+)$")
_MTP = re.compile(r"^mtp\.\d+\.")
_EXPERT = re.compile(r"^ffn\.experts\.(\d+)\.(w[123])\.(weight|scale)$")

# Root-level tensors, mapped to the model's own parameters.
_ROOT = {
    "embed.weight": "model.embed_tokens.embedding",
    "norm.weight": "model.norm.scale",
    "head.weight": "lm_head.embedding",
    "hc_head_fn": "model.hc_head_fn",
    "hc_head_base": "model.hc_head_base",
    "hc_head_scale": "model.hc_head_scale",
}

# Per-layer tensors present on every trunk layer.
_EVERY_LAYER = {
    "attn_norm.weight": "attn_norm.scale",
    "ffn_norm.weight": "ffn_norm.scale",
    "hc_attn_fn": "hc_attn_fn",
    "hc_attn_base": "hc_attn_base",
    "hc_attn_scale": "hc_attn_scale",
    "hc_ffn_fn": "hc_ffn_fn",
    "hc_ffn_base": "hc_ffn_base",
    "hc_ffn_scale": "hc_ffn_scale",
    "attn.attn_sink": "self_attn.attn_sink",
    "attn.q_norm.weight": "self_attn.q_norm.scale",
    "attn.kv_norm.weight": "self_attn.kv_norm.scale",
    "ffn.gate.weight": "mlp.gate.kernel",
}

# FP8 linears on every trunk layer: weight plus a block-scale sibling.
_EVERY_LAYER_FP8 = {
    "attn.wq_a": "self_attn.wq_a",
    "attn.wq_b": "self_attn.wq_b",
    "attn.wkv": "self_attn.wkv",
    "attn.wo_a": "self_attn.wo_a",
    "attn.wo_b": "self_attn.wo_b",
}

# Present only where the layer keeps compressed history (ratio > 0). Unquantised.
_COMPRESSOR = {
    "attn.compressor.ape": "self_attn.compressor.ape",
    "attn.compressor.norm.weight": "self_attn.compressor.norm.scale",
    "attn.compressor.wkv.weight": "self_attn.compressor.wkv",
    "attn.compressor.wgate.weight": "self_attn.compressor.wgate",
}

# Present only on CSA (ratio 4) layers.
_INDEXER = {
    "attn.indexer.compressor.ape": "self_attn.indexer.compressor.ape",
    "attn.indexer.compressor.norm.weight": "self_attn.indexer.compressor.norm.scale",
    "attn.indexer.compressor.wkv.weight": "self_attn.indexer.compressor.wkv",
    "attn.indexer.compressor.wgate.weight": "self_attn.indexer.compressor.wgate",
    "attn.indexer.weights_proj.weight": "self_attn.indexer.weights_proj",
}
_INDEXER_FP8 = {"attn.indexer.wq_b": "self_attn.indexer.wq_b"}

# Shared experts are FP8 like the other linears, not FP4 like the routed ones.
_SHARED_EXPERTS_FP8 = {
    "ffn.shared_experts.w1": "mlp.shared_experts.gate_proj",
    "ffn.shared_experts.w2": "mlp.shared_experts.down_proj",
    "ffn.shared_experts.w3": "mlp.shared_experts.up_proj",
}

# Sharding contract for the target tree. M1.4 must build parameters that accept
# these; they are recorded here because the mapping is what pins them.
_REPLICATED = (None,)
_COLUMN = (None, "tensor")
_ROW = ("tensor", None)


def _layer_families(config):
    """Which per-layer families exist on each trunk layer.

    Derived from `classify_layers` and `hash_moe_layer_flags` -- the single
    classification -- so this cannot drift from C1's or M2's view of the layers.
    """
    types = classify_layers(config)
    hashed = hash_moe_layer_flags(config)
    out = []
    for layer_type, is_hash in zip(types, hashed, strict=True):
        out.append(
            {
                "compressor": layer_type is not DeepseekV4LayerType.SWA_ONLY,
                "indexer": layer_type is DeepseekV4LayerType.C4A,
                "hash_gate": bool(is_hash),
            }
        )
    return out


def expected_trunk_keys(config) -> set[str]:
    """Every trunk key this config implies, so coverage can be checked both ways.

    A mapping is wrong if it misses a key the checkpoint has *and* if it asks for a
    key the checkpoint does not have -- e.g. `compressor` on a SWA-only layer.
    """
    keys = set(_ROOT)
    experts = int(config.n_routed_experts)
    for layer, families in enumerate(_layer_families(config)):
        prefix = f"layers.{layer}."
        keys |= {prefix + name for name in _EVERY_LAYER}
        for stem in _EVERY_LAYER_FP8:
            keys |= {f"{prefix}{stem}.weight", f"{prefix}{stem}.scale"}
        for stem in _SHARED_EXPERTS_FP8:
            keys |= {f"{prefix}{stem}.weight", f"{prefix}{stem}.scale"}
        keys.add(prefix + ("ffn.gate.tid2eid" if families["hash_gate"] else "ffn.gate.bias"))
        if families["compressor"]:
            keys |= {prefix + name for name in _COMPRESSOR}
        if families["indexer"]:
            keys |= {prefix + name for name in _INDEXER}
            for stem in _INDEXER_FP8:
                keys |= {f"{prefix}{stem}.weight", f"{prefix}{stem}.scale"}
        for expert in range(experts):
            for w in ("w1", "w2", "w3"):
                keys |= {
                    f"{prefix}ffn.experts.{expert}.{w}.weight",
                    f"{prefix}ffn.experts.{expert}.{w}.scale",
                }
    return keys


def classify_key(config, key: str) -> KeyFacts:
    """Disposition of one checkpoint key. Never returns "unknown" -- it raises."""
    if _MTP.match(key):
        return KeyFacts(
            Disposition.DROPPED,
            "MTP draft block; three are shipped despite num_nextn_predict_layers=1, and "
            "mtp.2 carries the DSpark heads (INFERENCE-84). Out of the first version.",
        )
    if key in _ROOT:
        return KeyFacts(Disposition.CLAIMED, "root tensor")

    match = _LAYER.match(key)
    if match is None:
        raise ValueError(f"unrecognised DeepSeek-V4 checkpoint key {key!r}")
    layer = int(match.group(1))
    tail = match.group(2)

    families = _layer_families(config)
    if not 0 <= layer < len(families):
        raise ValueError(
            f"key {key!r} names layer {layer}, outside the {len(families)}-layer trunk"
        )
    present = families[layer]

    expert = _EXPERT.match(tail)
    if expert is not None:
        if int(expert.group(1)) >= int(config.n_routed_experts):
            raise ValueError(f"key {key!r} names an expert beyond n_routed_experts")
        return KeyFacts(
            Disposition.EXPERT_CONVERTED,
            "routed expert; the resident layout is M0.2's power-of-two per-channel FP8 "
            "loaded by the paired MXFP4 converter",
            layer,
        )

    if tail in _EVERY_LAYER:
        return KeyFacts(Disposition.CLAIMED, "every-layer tensor", layer)
    for table, needed in (
        (_EVERY_LAYER_FP8, True),
        (_SHARED_EXPERTS_FP8, True),
        (_INDEXER_FP8, present["indexer"]),
    ):
        for stem in table:
            if tail in (f"{stem}.weight", f"{stem}.scale"):
                if not needed:
                    raise ValueError(f"key {key!r} is present but layer {layer} should not have it")
                return KeyFacts(Disposition.CLAIMED, "FP8 linear (block scale)", layer)
    if tail in _COMPRESSOR:
        if not present["compressor"]:
            raise ValueError(f"key {key!r} on a SWA-only layer, which has no compressor")
        return KeyFacts(Disposition.CLAIMED, "compressor (unquantised)", layer)
    if tail in _INDEXER:
        if not present["indexer"]:
            raise ValueError(f"key {key!r} on a layer that is not CSA")
        return KeyFacts(Disposition.CLAIMED, "indexer (unquantised)", layer)
    if tail == "ffn.gate.bias":
        if present["hash_gate"]:
            raise ValueError(
                f"key {key!r} on a hash-routed layer; bias and tid2eid are mutually exclusive"
            )
        return KeyFacts(Disposition.CLAIMED, "noaux_tc correction bias", layer)
    if tail == "ffn.gate.tid2eid":
        if not present["hash_gate"]:
            raise ValueError(f"key {key!r} on a layer that does not route by token id")
        return KeyFacts(Disposition.CLAIMED, "hash routing table", layer)

    raise ValueError(f"unrecognised DeepSeek-V4 checkpoint key {key!r}")


def classify_checkpoint(config, keys) -> dict[str, KeyFacts]:
    """Classify every key, and require the partition to be total.

    Raises on an unrecognised key rather than skipping it -- the whole point is that
    "not loaded" can never be silent.
    """
    return {key: classify_key(config, key) for key in keys}


def _add_fp8_linear(mappings, hf_stem, target, *, sharding):
    """An FP8 linear: the weight plus its ``[out/128, in/128]`` block scale.

    Checkpoint weights are ``[out, in]`` and load into ``weight_q`` without a
    transpose, with the block scale as a sidecar -- the same shape the existing
    static-FP8 path in `models/deepseek_v3.py` uses.
    """
    quant = (sharding[1], sharding[0])
    mappings[f"{hf_stem}.weight"] = WeightMapping(
        target_path=f"{target}.weight_q", sharding=quant, transpose=False
    )
    mappings[f"{hf_stem}.scale"] = WeightMapping(
        target_path=f"{target}.weight_scale", sharding=quant, transpose=False
    )


def build_weight_mappings(config) -> dict[str, WeightMapping]:
    """The `WeightMapping` table for every CLAIMED key.

    Routed-expert tensors are deliberately absent; see `Disposition.EXPERT_CONVERTED`.
    """
    mappings: dict[str, WeightMapping] = {}
    for key, target in _ROOT.items():
        if key.startswith("hc_head"):
            # mHC gates are float32 and not ``[out, in]`` projections: no transpose,
            # replicated, and the dtype must not follow the activation dtype.
            mappings[key] = WeightMapping(target_path=target, sharding=_REPLICATED, transpose=False)
        elif key in ("embed.weight", "head.weight"):
            mappings[key] = WeightMapping(
                target_path=target, sharding=("tensor", None), transpose=False
            )
        else:
            mappings[key] = WeightMapping(target_path=target, sharding=_REPLICATED, transpose=False)

    for layer, families in enumerate(_layer_families(config)):
        prefix = f"layers.{layer}."
        target = f"model.layers.{layer}."
        for name, suffix in _EVERY_LAYER.items():
            sharding = (None, None) if suffix == "mlp.gate.kernel" else _REPLICATED
            transpose = suffix == "mlp.gate.kernel"
            mappings[prefix + name] = WeightMapping(
                target_path=target + suffix, sharding=sharding, transpose=transpose
            )
        for stem, suffix in _EVERY_LAYER_FP8.items():
            # wo_b reduces G*R back to hidden, so it is the row-parallel one.
            sharding = (
                _ROW
                if stem == "attn.wo_b"
                else ((None, None) if stem in ("attn.wq_a", "attn.wkv") else _COLUMN)
            )
            _add_fp8_linear(mappings, prefix + stem, target + suffix, sharding=sharding)
        for stem, suffix in _SHARED_EXPERTS_FP8.items():
            sharding = _ROW if stem.endswith("w2") else _COLUMN
            _add_fp8_linear(mappings, prefix + stem, target + suffix, sharding=sharding)

        if families["hash_gate"]:
            mappings[prefix + "ffn.gate.tid2eid"] = WeightMapping(
                target_path=target + "mlp.gate.tid2eid", sharding=(None, None), transpose=False
            )
        else:
            mappings[prefix + "ffn.gate.bias"] = WeightMapping(
                target_path=target + "mlp.gate.bias",
                sharding=_REPLICATED,
                transpose=False,
            )

        if families["compressor"]:
            for name, suffix in _COMPRESSOR.items():
                mappings[prefix + name] = WeightMapping(
                    target_path=target + suffix,
                    sharding=_REPLICATED if "norm" in name else (None, None),
                    transpose=False,
                )
        if families["indexer"]:
            for name, suffix in _INDEXER.items():
                mappings[prefix + name] = WeightMapping(
                    target_path=target + suffix,
                    sharding=_REPLICATED if "norm" in name else (None, None),
                    transpose=False,
                )
            for stem, suffix in _INDEXER_FP8.items():
                _add_fp8_linear(mappings, prefix + stem, target + suffix, sharding=(None, None))
    return mappings


class DeepseekV4SharedMLP(nnx.Module):
    def __init__(self, hidden_size, intermediate_size, mesh, dtype, swiglu_limit, quantized=False):
        self.swiglu_limit = swiglu_limit
        for name in ("gate_proj", "up_proj", "down_proj"):
            down = name == "down_proj"
            setattr(
                self,
                name,
                _linear(
                    intermediate_size if down else hidden_size,
                    hidden_size if down else intermediate_size,
                    mesh,
                    dtype,
                    ("tensor", None) if down else (None, "tensor"),
                    name,
                    quantized,
                ),
            )

    def __call__(self, hidden_states):
        gate, _ = self.gate_proj(hidden_states)
        up, _ = self.up_proj(hidden_states)
        activated = (
            jax.nn.silu(gate) * up
            if self.swiglu_limit is None
            else silu_and_mul_with_clamp(gate, up, self.swiglu_limit)
        )
        output, _ = self.down_proj(activated)
        return output


class DeepseekV4MoE(nnx.Module):
    """Hash layers and learned routing share scoring, normalization and experts.

    ``load_hash_table`` must be called with the checkpoint's ``gate.tid2eid``
    before real inference; the deterministic initial table is for dummy loads.
    ``route`` exposes weights/IDs for routing diagnostics without running GMM.
    """

    def __init__(self, config, mesh, layer_id, dtype=jnp.bfloat16):
        self.mesh = mesh
        self.hidden_size = config.hidden_size
        self.num_experts = config.n_routed_experts
        self.top_k = config.num_experts_per_tok
        self.vocab_size = config.vocab_size
        self.is_hash_layer = layer_id < config.num_hash_layers
        if not 0 < self.top_k <= self.num_experts:
            raise ValueError("num_experts_per_tok must be in [1, n_routed_experts]")
        if self.vocab_size <= 0 or layer_id < 0:
            raise ValueError("vocab_size must be positive and layer_id nonnegative")
        self.gate = GateLogit(
            self.hidden_size,
            self.num_experts,
            weight_dtype=jnp.float32,
            enable_expert_bias=not self.is_hash_layer,
            score_func=getattr(config, "scoring_func", "sqrtsoftplus"),
        )
        if self.is_hash_layer:
            table = (np.arange(self.vocab_size)[:, None] + np.arange(self.top_k)) % self.num_experts
            self.gate.tid2eid = nnx.Param(
                jax.device_put(table.astype(np.int32), NamedSharding(mesh, P(None, None)))
            )
        self.topk = TopK(
            topk=self.top_k,
            renormalize=config.norm_topk_prob,
            num_expert_group=getattr(config, "n_group", 1),
            topk_group=getattr(config, "topk_group", 1),
            routed_scaling_factor=config.routed_scaling_factor,
            layer_id=layer_id,
            mesh=mesh,
        )
        self.experts = EPMoE(
            hidden_size=self.hidden_size,
            num_experts=self.num_experts,
            num_experts_per_tok=self.top_k,
            intermediate_dim=config.moe_intermediate_size,
            mesh=mesh,
            ep_size=getattr(config, "ep_size", 1),
            moe_dp_size=getattr(config, "moe_dp_size", 1),
            weight_dtype=dtype,
            dtype=dtype,
            layer_id=layer_id,
            quantization_config=getattr(config, "quantization_config", None),
            swiglu_limit=config.swiglu_limit,
        )
        if getattr(config, "expert_dtype", None) == "fp4":
            if self.experts.replicate_experts:
                raise ValueError("V4 resident FP8 experts require moe_dp_size=1")
            self.experts.quantized_dtype = jnp.float8_e4m3fn
            self.experts.weight_block_size = None
            with jax.sharding.use_abstract_mesh(self.experts.updated_mesh):
                for name in ("wi_0", "wi_1", "wo"):
                    old = getattr(self.experts, name).value
                    spec = jax.typeof(old).sharding.spec
                    setattr(
                        self.experts,
                        name,
                        nnx.Param(jnp.zeros(old.shape, jnp.float8_e4m3fn, out_sharding=spec)),
                    )
                    setattr(
                        self.experts,
                        name + "_scale",
                        nnx.data(
                            nnx.Param(
                                jnp.ones(
                                    (old.shape[0], 1, 1, old.shape[2]),
                                    jnp.float32,
                                    out_sharding=P(spec[0], None, None, spec[2]),
                                )
                            )
                        ),
                    )
        if getattr(config, "n_shared_experts", 0):
            self.shared_experts = DeepseekV4SharedMLP(
                self.hidden_size,
                config.moe_intermediate_size * config.n_shared_experts,
                mesh,
                dtype,
                config.swiglu_limit,
                quantized=_static_fp8(config),
            )
        else:
            self.shared_experts = None

    def load_hash_table(self, table):
        """Load a host checkpoint tensor without floating-point dtype conversion."""
        if not self.is_hash_layer:
            raise ValueError("only hash layers have gate.tid2eid")
        table = np.asarray(table)
        if table.shape != (self.vocab_size, self.top_k):
            raise ValueError("gate.tid2eid must have shape [vocab_size, top_k]")
        if not np.issubdtype(table.dtype, np.integer):
            raise ValueError("gate.tid2eid must contain integer expert IDs")
        if np.any(table < 0) or np.any(table >= self.num_experts):
            raise ValueError("gate.tid2eid expert IDs are outside the logical expert range")
        self.gate.tid2eid.value = jax.device_put(
            table.astype(np.int32), NamedSharding(self.mesh, P(None, None))
        )

    def route(
        self,
        hidden_states,
        input_ids=None,
        *,
        token_valid_mask=None,
        dispatch_info=None,
        routing_sharding=None,
    ):
        if hidden_states.ndim != 2 or hidden_states.shape[1] != self.hidden_size:
            raise ValueError("hidden_states must have shape [tokens, hidden_size]")
        if self.experts.replicate_experts and dispatch_info is not None:
            raise ValueError("replicated experts do not support EPLB dispatch metadata")
        routing_sharding = routing_sharding or NamedSharding(self.mesh, P("data", None))
        if len(routing_sharding.spec) != 2 or routing_sharding.spec[1] is not None:
            raise ValueError("routing sharding must partition tokens only")
        hidden_states = jax.sharding.reshard(hidden_states, routing_sharding)
        token_sharding = NamedSharding(self.mesh, P(routing_sharding.spec[0]))
        tokens = hidden_states.shape[0]
        valid = jnp.ones((tokens,), dtype=jnp.bool_)
        if token_valid_mask is not None:
            if token_valid_mask.shape != (tokens,):
                raise ValueError("token_valid_mask must have shape [tokens]")
            valid = jax.sharding.reshard(token_valid_mask.astype(jnp.bool_), token_sharding)
        selected = None
        if self.is_hash_layer:
            if input_ids is None or input_ids.shape != (tokens,):
                raise ValueError("hash routing requires input_ids with shape [tokens]")
            if not jnp.issubdtype(input_ids.dtype, jnp.integer):
                raise ValueError("input_ids must be integers")
            input_ids = jax.sharding.reshard(input_ids, token_sharding)
            valid = valid & (input_ids >= 0) & (input_ids < self.vocab_size)
            # Padding must never produce negative IDs in EPMoE's bincount/permutation.
            safe_ids = jnp.where(valid, input_ids, 0)
            selected = self.gate.tid2eid.value.at[safe_ids].get(out_sharding=routing_sharding)
        scores = self.gate(jnp.where(valid[:, None], hidden_states, 0))
        weights, ids = self.topk(
            scores,
            None if self.is_hash_layer else self.gate.bias.value,
            dispatch_info,
            routing_sharding,
            selected_experts=selected,
        )
        return jnp.where(valid[:, None], weights, 0), ids

    def __call__(
        self,
        hidden_states,
        input_ids=None,
        *,
        token_valid_mask=None,
        dispatch_info=None,
        out_sharding=None,
    ):
        weights, ids = self.route(
            hidden_states,
            input_ids,
            token_valid_mask=token_valid_mask,
            dispatch_info=dispatch_info,
            routing_sharding=out_sharding,
        )
        # Zero invalid activations too: zero routing weight alone cannot mask NaN.
        valid = jnp.ones(hidden_states.shape[:1], dtype=jnp.bool_)
        if token_valid_mask is not None:
            valid = valid & token_valid_mask.astype(jnp.bool_)
        if self.is_hash_layer:
            valid = valid & (input_ids >= 0) & (input_ids < self.vocab_size)
        hidden_states = jnp.where(valid[:, None], hidden_states, 0)
        output = self.experts(hidden_states, weights, ids, out_sharding=out_sharding)
        if self.shared_experts is not None:
            # routed_scaling_factor applies only to routed weights, exactly once.
            shared = self.shared_experts(hidden_states)
            if out_sharding is not None:
                shared = jax.sharding.reshard(shared, out_sharding)
            output = output + shared
        return jnp.where(valid[:, None], output, 0), ids


def _static_fp8(config):
    quant = getattr(config, "quantization_config", None)
    return quant is not None and getattr(quant, "is_static_checkpoint", False)


def _linear(input_size, output_size, mesh, dtype, axes, name, quantized=False):
    """Build the checkpoint's resident representation before eval_shape loading."""
    if not quantized:
        return LinearBase(
            input_size=input_size,
            output_size=output_size,
            mesh=mesh,
            params_dtype=dtype,
            kernel_axes=axes,
            use_bias=False,
            scope_name=name,
        )
    from sgl_jax.srt.layers.linear import QuantizedLinear

    # Non-expert FP8 uses K128 block scales; expert MXFP4 is converted separately.
    if input_size % 128 or output_size % 128:
        raise ValueError(f"static V4 FP8 linear {name} requires dimensions divisible by 128")
    if axes[0] is not None and (input_size // 128) % mesh.shape[axes[0]]:
        # A block cannot span two devices. Match QuantizedLinear.from_linear.
        axes = (None, axes[1])
    return QuantizedLinear(
        weight_q=jnp.zeros(
            (output_size, input_size), jnp.float8_e4m3fn, out_sharding=P(axes[1], axes[0])
        ),
        weight_scale=jnp.zeros(
            (input_size // 128, 1, output_size), jnp.float32, out_sharding=P(axes[0], None, axes[1])
        ),
        bias=None,
        activation_dtype=None,
        mesh=mesh,
        kernel_axes=axes,
        params_dtype=dtype,
        weight_block_size=(128, 128),
        # As in ModelConfig's static-FP8 loader: scales came from checkpoint,
        # not the online narrow-N quantizer the guard protects against.
        allow_narrow_n_blockwise=True,
        scope_name=name,
    )


def _checkpoint_matrix(linear):
    """Weight in [out,in] layout, used only by the grouped output contraction."""
    if hasattr(linear, "weight_q"):
        weight = linear.weight_q.value.astype(jnp.float32)
        # Kernel-ready [K/128,1,N] -> [N,K]. TP follows N for grouped wo_a.
        scale = jnp.repeat(linear.weight_scale.value[:, 0, :], 128, axis=0).T
        return weight * scale
    return linear.weight.value.T


def _rope_cache(config, ratio):
    from sgl_jax.srt.layers.attention.dsv4.rope import build_dsv4_rope

    rope = build_dsv4_rope(config, ratio, dtype=jnp.float32)
    cos, sin = rope._compute_cos_sin(jnp.arange(config.max_position_embeddings, dtype=jnp.int32))
    return jnp.concatenate((cos, sin), axis=-1)


class DeepseekV4Compressor(nnx.Module):
    """All compressor parameters stay in the modeling file, in checkpoint layout."""

    def __init__(self, config, head_dim, ratio, dtype):
        from sgl_jax.srt.layers.layernorm import RMSNorm

        width = head_dim * (2 if ratio == 4 else 1)
        self.wkv = nnx.Param(
            jnp.zeros((width, config.hidden_size), dtype, out_sharding=P(None, None))
        )
        self.wgate = nnx.Param(
            jnp.zeros((width, config.hidden_size), jnp.float32, out_sharding=P(None, None))
        )
        self.ape = nnx.Param(jnp.zeros((ratio, width), jnp.float32, out_sharding=P(None, None)))
        self.norm = RMSNorm(head_dim, epsilon=config.rms_norm_eps, param_dtype=jnp.float32)

    def weights(self, cache):
        from sgl_jax.srt.layers.attention.deepseek_v4_csa_backend import (
            CompressorWeights,
        )

        return CompressorWeights(
            self.wkv.value, self.wgate.value, self.ape.value, self.norm.scale.value, cache
        )


class DeepseekV4Indexer(nnx.Module):
    def __init__(self, config, mesh, dtype):
        self.head_dim = config.index_head_dim
        self.num_heads = config.index_n_heads
        self.rope_head_dim = config.qk_rope_head_dim
        self.weight_scale = (self.head_dim * self.num_heads) ** -0.5
        self.wq_b = _linear(
            config.q_lora_rank,
            self.num_heads * self.head_dim,
            mesh,
            dtype,
            (None, None),
            "indexer_wq_b",
            _static_fp8(config),
        )
        self.weights_proj = nnx.Param(
            jnp.zeros((self.num_heads, config.hidden_size), dtype, out_sharding=P(None, None))
        )
        self.compressor = DeepseekV4Compressor(config, self.head_dim, 4, dtype)

    def __call__(self, hidden, q_lora, cos, sin, cache):
        from sgl_jax.srt.layers.attention.deepseek_v4_csa_backend import IndexerInputs
        from sgl_jax.srt.layers.attention.dsv4.rope import apply_dsv4_partial_rope

        q, _ = self.wq_b(q_lora)
        q = q.reshape(-1, self.num_heads, self.head_dim)
        q = apply_dsv4_partial_rope(
            q, cos[:, None, :], sin[:, None, :], rope_head_dim=self.rope_head_dim
        ).astype(hidden.dtype)
        weights = jnp.dot(hidden.astype(jnp.float32), self.weights_proj.value.T.astype(jnp.float32))
        return IndexerInputs(q, weights * self.weight_scale, self.compressor.weights(cache))


class DeepseekV4Attention(nnx.Module):
    def __init__(self, config, mesh, layer_id, dtype):
        from sgl_jax.srt.layers.layernorm import RMSNorm

        self.mesh = mesh
        self.layer_id = layer_id
        self.ratio = config.compress_ratios[layer_id]
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.num_groups = config.o_groups
        self.scaling = config.head_dim**-0.5
        self.norm_eps = config.rms_norm_eps
        self.index_topk = config.index_topk
        self.dtype = dtype
        if self.num_heads % self.num_groups or self.num_groups % mesh.shape["tensor"]:
            raise ValueError(
                "V4 heads must divide into output groups, and groups must divide by TP"
            )
        self.wq_a = _linear(
            config.hidden_size,
            config.q_lora_rank,
            mesh,
            dtype,
            (None, None),
            "wq_a",
            _static_fp8(config),
        )
        self.q_norm = RMSNorm(config.q_lora_rank, epsilon=self.norm_eps, dtype=dtype)
        self.wq_b = _linear(
            config.q_lora_rank,
            self.num_heads * self.head_dim,
            mesh,
            dtype,
            (None, "tensor"),
            "wq_b",
            _static_fp8(config),
        )
        self.wkv = _linear(
            config.hidden_size, self.head_dim, mesh, dtype, (None, None), "wkv", _static_fp8(config)
        )
        self.kv_norm = RMSNorm(self.head_dim, epsilon=self.norm_eps, dtype=dtype)
        self.attn_sink = nnx.Param(
            jnp.zeros((self.num_heads,), jnp.float32, out_sharding=P("tensor"))
        )
        self.wo_a = _linear(
            self.num_heads // self.num_groups * self.head_dim,
            self.num_groups * config.o_lora_rank,
            mesh,
            dtype,
            (None, "tensor"),
            "wo_a",
            _static_fp8(config),
        )
        self.wo_b = _linear(
            self.num_groups * config.o_lora_rank,
            config.hidden_size,
            mesh,
            dtype,
            ("tensor", None),
            "wo_b",
            _static_fp8(config),
        )
        self.compressor = (
            DeepseekV4Compressor(config, self.head_dim, self.ratio, dtype) if self.ratio else None
        )
        self.indexer = DeepseekV4Indexer(config, mesh, dtype) if self.ratio == 4 else None

    def __call__(self, hidden, batch, pools, rope_cache):
        from sgl_jax.srt.layers.attention.dsv4.o_projection import group_wo_a
        from sgl_jax.srt.layers.attention.dsv4.rope import apply_dsv4_partial_rope

        q_lora, _ = self.wq_a(hidden)
        q_lora = self.q_norm(q_lora)
        q, _ = self.wq_b(q_lora)
        q = q.reshape(-1, self.num_heads, self.head_dim)
        # V4 normalizes q again per head after wq_b, with no learned weight.
        q = (
            q.astype(jnp.float32)
            * jax.lax.rsqrt(
                jnp.mean(jnp.square(q.astype(jnp.float32)), axis=-1, keepdims=True) + self.norm_eps
            )
        ).astype(self.dtype)
        kv, _ = self.wkv(hidden)
        kv = self.kv_norm(kv)
        positions = batch.positions
        selected = rope_cache.at[positions].get(
            out_sharding=NamedSharding(self.mesh, P("data", None))
        )
        cos, sin = jnp.split(selected, 2, axis=-1)
        q = apply_dsv4_partial_rope(
            q, cos[:, None, :], sin[:, None, :], rope_head_dim=self.rope_head_dim
        ).astype(self.dtype)
        kv = apply_dsv4_partial_rope(kv, cos, sin, rope_head_dim=self.rope_head_dim).astype(
            self.dtype
        )
        indexer = (
            None if self.indexer is None else self.indexer(hidden, q_lora, cos, sin, rope_cache)
        )
        output, updates = batch.attn_backend(
            q,
            kv,
            kv,
            self,
            batch,
            pools.token_to_kv_pool,
            compressor_state_pool=pools.compressor_state_pool,
            compressor_input=hidden,
            compressor=None if self.compressor is None else self.compressor.weights(rope_cache),
            indexer=indexer,
            attention_sink=self.attn_sink.value,
            rope_head_dim=self.rope_head_dim,
            norm_eps=self.norm_eps,
            index_topk=self.index_topk,
        )
        output = apply_dsv4_partial_rope(
            output, cos[:, None, :], sin[:, None, :], rope_head_dim=self.rope_head_dim, inverse=True
        )
        grouped = jax.lax.reshape(
            output,
            (output.shape[0], self.num_groups, self.num_heads // self.num_groups * self.head_dim),
            out_sharding=NamedSharding(self.mesh, P("data", "tensor", None)),
        )
        weights = group_wo_a(
            _checkpoint_matrix(self.wo_a),
            num_groups=self.num_groups,
            out_sharding=NamedSharding(self.mesh, P("tensor", None, None)),
        )
        reduced = jnp.einsum("tgd,gdr->tgr", grouped, weights, preferred_element_type=jnp.float32)
        reduced = reduced.reshape(reduced.shape[0], -1).astype(self.dtype)
        output, _ = self.wo_b(reduced)
        return output, updates


class DeepseekV4DecoderLayer(nnx.Module):
    def __init__(self, config, mesh, layer_id, dtype):
        from sgl_jax.srt.configs.deepseek_v4 import mhc_param_shapes
        from sgl_jax.srt.layers.deepseek_v4_mhc import DeepseekV4MHC
        from sgl_jax.srt.layers.layernorm import RMSNorm

        self.mhc = DeepseekV4MHC(config)
        self.mesh = mesh
        for kind in ("attn", "ffn"):
            for part in ("fn", "base", "scale"):
                shape = mhc_param_shapes(config)[part]
                setattr(
                    self,
                    f"hc_{kind}_{part}",
                    nnx.Param(
                        jnp.zeros(shape, jnp.float32, out_sharding=P(*([None] * len(shape))))
                    ),
                )
        self.attn_norm = RMSNorm(config.hidden_size, epsilon=config.rms_norm_eps, dtype=dtype)
        self.ffn_norm = RMSNorm(config.hidden_size, epsilon=config.rms_norm_eps, dtype=dtype)
        self.self_attn = DeepseekV4Attention(config, mesh, layer_id, dtype)
        self.mlp = DeepseekV4MoE(config, mesh, layer_id, dtype)
        self.dtype = dtype

    def _mhc_pre(self, streams, fn, base, scale):
        if self.mhc.backend != "pallas":
            return self.mhc.pre(streams, fn, base, scale)
        specs = (P("data", None), P("data", None), P("data", None, None))
        compute = jax.shard_map(
            self.mhc.pre,
            mesh=None,
            in_specs=(P("data", None, None), P(), P(), P()),
            out_specs=specs,
            check_vma=False,
        )
        compute = jax.sharding.auto_axes(
            compute,
            axes=self.mesh.axis_names,
            out_sharding=tuple(NamedSharding(self.mesh, spec) for spec in specs),
        )
        return compute(streams, fn, base, scale)

    def _mhc_post(self, output, residual, post, comb):
        if self.mhc.backend != "pallas":
            return self.mhc.post(output, residual, post, comb)
        spec = P("data", None, None)
        compute = jax.shard_map(
            self.mhc.post,
            mesh=None,
            in_specs=(P("data", None), spec, P("data", None), spec),
            out_specs=spec,
            check_vma=False,
        )
        compute = jax.sharding.auto_axes(
            compute, axes=self.mesh.axis_names, out_sharding=NamedSharding(self.mesh, spec)
        )
        return compute(output, residual, post, comb)

    def __call__(self, streams, batch, pools, rope_cache):
        hidden, post, comb = self._mhc_pre(
            streams, self.hc_attn_fn.value, self.hc_attn_base.value, self.hc_attn_scale.value
        )
        attn, updates = self.self_attn(
            self.attn_norm(hidden.astype(self.dtype)), batch, pools, rope_cache
        )
        streams = self._mhc_post(attn, streams, post, comb).astype(self.dtype)
        hidden, post, comb = self._mhc_pre(
            streams, self.hc_ffn_fn.value, self.hc_ffn_base.value, self.hc_ffn_scale.value
        )
        ffn, ids = self.mlp(
            self.ffn_norm(hidden.astype(self.dtype)),
            batch.input_ids,
            token_valid_mask=batch.get_token_valid_mask(hidden.shape[0]),
            dispatch_info=batch.expert_location_metadata,
        )
        streams = self._mhc_post(ffn, streams, post, comb).astype(self.dtype)
        return streams, updates, ids


class DeepseekV4Model(nnx.Module):
    def __init__(self, config, mesh, dtype):
        from sgl_jax.srt.configs.deepseek_v4 import mhc_param_shapes
        from sgl_jax.srt.layers.deepseek_v4_mhc import DeepseekV4MHC
        from sgl_jax.srt.layers.embeddings import Embed
        from sgl_jax.srt.layers.layernorm import RMSNorm

        self.mhc = DeepseekV4MHC(config)
        self.mesh = mesh
        self.dtype = dtype
        self.embed_tokens = Embed(
            config.vocab_size,
            config.hidden_size,
            dtype=dtype,
            param_dtype=dtype,
            kernel_axes=("tensor", None),
            mesh=mesh,
        )
        self.layers = nnx.List(
            [
                DeepseekV4DecoderLayer(config, mesh, i, dtype)
                for i in range(config.num_hidden_layers)
            ]
        )
        for part in ("fn", "base", "scale"):
            shape = mhc_param_shapes(config)["head_" + part]
            setattr(
                self,
                "hc_head_" + part,
                nnx.Param(jnp.zeros(shape, jnp.float32, out_sharding=P(*([None] * len(shape))))),
            )
        self.norm = RMSNorm(config.hidden_size, epsilon=config.rms_norm_eps, dtype=dtype)
        self.rope_plain = nnx.Variable(_rope_cache(config, 0))
        self.rope_compressed = nnx.Variable(_rope_cache(config, 4))

    def _collapse_head(self, streams):
        params = (self.hc_head_fn.value, self.hc_head_base.value, self.hc_head_scale.value)
        if self.mhc.backend != "pallas":
            return self.mhc.collapse_head(streams, *params)
        spec = P("data", None)
        compute = jax.shard_map(
            self.mhc.collapse_head,
            mesh=None,
            in_specs=(P("data", None, None), P(), P(), P()),
            out_specs=spec,
            check_vma=False,
        )
        compute = jax.sharding.auto_axes(
            compute, axes=self.mesh.axis_names, out_sharding=NamedSharding(self.mesh, spec)
        )
        return compute(streams, *params)

    def __call__(self, batch, pools):
        from sgl_jax.srt.layers.deepseek_v4_mhc import expand_streams

        hidden = self.embed_tokens(batch.input_ids)
        if batch.input_embedding is not None:
            hidden = batch.input_embedding
        streams = expand_streams(hidden, self.mhc.hc_mult).astype(hidden.dtype)
        updates, ids = {}, []
        for i, layer in enumerate(self.layers):
            cache = self.rope_compressed if layer.self_attn.ratio else self.rope_plain
            streams, updates[i], route_ids = layer(streams, batch, pools, cache.value)
            ids.append(route_ids)
        hidden = self._collapse_head(streams)
        return (
            self.norm(hidden.astype(self.dtype)),
            batch.attn_backend.pack_pool_updates(
                updates, pools.token_to_kv_pool, pools.compressor_state_pool
            ),
            ids,
        )


class DeepseekV4ForCausalLM(nnx.Module):
    owns_quantization_structure = True

    @classmethod
    def patch_model_config(cls, mc):
        from sgl_jax.srt.configs.model_config import AttentionArch

        mc.attention_arch = AttentionArch.MLA
        mc.head_dim = mc.hf_text_config.head_dim

    def __init__(self, config, mesh, dtype=jnp.bfloat16):
        from sgl_jax.srt.layers.embeddings import ParallelLMHead
        from sgl_jax.srt.layers.logits_processor import LogitsProcessor

        self.config = config
        self.mesh = mesh
        self.dtype = dtype
        self.model = DeepseekV4Model(config, mesh, dtype)
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            dtype=dtype,
            param_dtype=dtype,
            kernel_axes=("tensor", None),
        )
        self.logits_processor = LogitsProcessor(config.vocab_size, mesh=mesh)

    def __call__(self, forward_batch, memory_pools, logits_metadata):
        hidden, updates, ids = self.model(forward_batch, memory_pools)
        output = self.logits_processor(hidden, self.lm_head, logits_metadata)
        return output, updates, True, ids

    def load_weights(self, model_config):
        """Load all trunk tensors; source coverage and target shapes are mandatory."""
        from sgl_jax.srt.utils.weight_utils import WeightLoader

        if getattr(model_config, "_dummy_mode", False):
            state = nnx.state(self, nnx.Param)
            for i, (path, variable) in enumerate(state.flat_state()):
                old = variable.value
                mesh = self.mesh
                if "experts" in path:
                    mesh = self.model.layers[int(path[2])].mlp.experts.moe_mesh
                sharding = NamedSharding(mesh, old.sharding.spec)

                def initialize(index, shape=old.shape, dtype=old.dtype, seed=i):
                    dims = tuple(len(range(*sl.indices(n))) for sl, n in zip(index, shape))
                    rng = np.random.default_rng(seed)
                    return (rng.standard_normal(dims) * 0.02).astype(dtype)

                variable.value = jax.make_array_from_callback(old.shape, sharding, initialize)
            nnx.update(self, state)
            for layer in self.model.layers:
                if layer.mlp.is_hash_layer:
                    table = (
                        np.arange(self.config.vocab_size)[:, None]
                        + np.arange(self.config.num_experts_per_tok)
                    ) % self.config.n_routed_experts
                    layer.mlp.load_hash_table(table)
        else:
            from sgl_jax.srt.utils.quantization.deepseek_v4_static_fp8 import (
                CONFIG_KEY,
                FORMAT,
                validate_static_checkpoint,
            )

            expert_format = getattr(self.config, CONFIG_KEY, None)
            if expert_format is not None:
                if expert_format != FORMAT:
                    raise ValueError(f"Unsupported V4 expert format: {expert_format}")
                validate_static_checkpoint(model_config.model_path)
            loader = WeightLoader(self, model_config, self.mesh, self.dtype)
            info = loader._scan_weight_info()
            classify_checkpoint(self.config, info)
            missing = expected_trunk_keys(self.config) - info.keys()
            if missing:
                raise ValueError(
                    f"V4 checkpoint missing {len(missing)} trunk tensors: {sorted(missing)[:8]}"
                )
            if any(len(entries) != 1 for entries in info.values()):
                raise ValueError("V4 tensors must occur exactly once across checkpoint shards")
            started = time.monotonic()
            logger.info("Loading DeepSeek V4 non-expert parameters")
            self._load_regular_weights(info)
            logger.info(
                "Loaded DeepSeek V4 non-expert parameters in %.1fs", time.monotonic() - started
            )
            if expert_format == FORMAT:
                self._load_static_expert_weights(loader, info)
            else:
                self._load_expert_weights(info)
        # eval_shape creates placeholders for these non-parameter tables too.
        with jax.set_mesh(self.mesh):
            self.model.rope_plain.value = _rope_cache(self.config, 0)
            self.model.rope_compressed.value = _rope_cache(self.config, 4)

    def _load_regular_weights(self, info):
        from safetensors import safe_open

        def read(key):
            entry = info[key][0]
            if entry["dtype"] == "F8_E8M0":
                # safetensors' NumPy API cannot expose E8M0. Non-expert block
                # scales use this format too, not only the MXFP4 expert scales.
                with open(entry["file"], "rb") as source:
                    source.seek(entry["byte_offset"])
                    raw = source.read(entry["byte_size"])
                if len(raw) != entry["byte_size"] or len(raw) != int(np.prod(entry["shape"])):
                    raise ValueError(f"{key}: truncated E8M0 scale payload")
                codes = np.frombuffer(raw, np.uint8).reshape(entry["shape"])
                if np.any(codes == 255):
                    raise ValueError(f"{key}: reserved E8M0 scale code 255")
                return np.ldexp(np.ones(codes.shape, np.float32), codes.astype(np.int16) - 127)
            with safe_open(entry["file"], framework="numpy") as handle:
                return handle.get_tensor(key)

        def parameter(path):
            obj = self
            for component in path.split("."):
                obj = obj[int(component)] if component.isdigit() else getattr(obj, component)
            return obj

        def assign(param, array, key):
            old = param.value
            if array.shape != old.shape:
                raise ValueError(
                    f"{key}: checkpoint {array.shape} does not match model {old.shape}"
                )
            if not np.isfinite(array).all():
                raise ValueError(f"{key}: non-finite checkpoint tensor")
            param.value = jax.device_put(
                array.astype(old.dtype), NamedSharding(self.mesh, old.sharding.spec)
            )
            param.value.block_until_ready()

        for key, mapping in build_weight_mappings(self.config).items():
            path = mapping.target_path
            if path.endswith(".weight_scale"):
                continue  # Paired with its FP8 weight below.
            if path.endswith(".weight_q"):
                linear = parameter(path.rsplit(".", 1)[0])
                weight = read(key)
                scale = read(key.removesuffix(".weight") + ".scale").astype(np.float32)
                if scale.shape != ((weight.shape[0] + 127) // 128, (weight.shape[1] + 127) // 128):
                    raise ValueError(f"{key}: invalid K128/N128 FP8 block scales")
                if hasattr(linear, "weight_q"):
                    assign(linear.weight_q, weight, key)
                    expanded = np.repeat(scale, 128, axis=0)[: weight.shape[0], :].T[:, None, :]
                    assign(linear.weight_scale, expanded, key + " scale")
                else:
                    expanded = np.repeat(np.repeat(scale, 128, axis=0), 128, axis=1)
                    assign(
                        linear.weight,
                        (
                            weight.astype(np.float32)
                            * expanded[: weight.shape[0], : weight.shape[1]]
                        ).T,
                        key,
                    )
                continue
            value = read(key)
            if key.endswith("ffn.gate.tid2eid"):
                self.model.layers[int(key.split(".")[1])].mlp.load_hash_table(value)
            else:
                assign(parameter(path), value.T if mapping.transpose else value, key)

    def _load_static_expert_weights(self, loader, info):
        """Map published FP8 experts into the shared parallel weight loader.

        Export validates values and exact conversion; publication records hashes.
        Startup checks tensor metadata here, without rereading every value through
        the offline validation/conversion helpers.
        """
        for layer_id, layer in enumerate(self.model.layers):
            started = time.monotonic()
            experts = layer.mlp.experts
            if experts.num_experts != self.config.n_routed_experts:
                raise ValueError(
                    "V4 checkpoint loading currently requires identity expert placement"
                )
            mappings = {}
            for source, target in (("w1", "wi_0"), ("w3", "wi_1"), ("w2", "wo")):
                weight = getattr(experts, target).value
                scale_param = getattr(experts, target + "_scale")
                if weight.dtype != jnp.float8_e4m3fn or scale_param is None:
                    raise ValueError("Static V4 experts require resident FP8 weights and scales")
                scale = scale_param.value
                prefix = f"model.layers.{layer_id}.mlp.experts.{target}"
                keys = [
                    f"layers.{layer_id}.ffn.experts.{expert_id}.{source}"
                    for expert_id in range(experts.num_experts)
                ]
                for key in keys:
                    for suffix, dtype, shape, size in (
                        (".weight", "F8_E4M3", (weight.shape[2], weight.shape[1]), 1),
                        (".scale", "F32", (weight.shape[-1],), 4),
                    ):
                        entries = info.get(key + suffix, [])
                        if len(entries) != 1:
                            raise ValueError(f"{key + suffix}: expected exactly one tensor")
                        entry = entries[0]
                        if (
                            entry["dtype"] != dtype
                            or tuple(entry["shape"]) != shape
                            or entry["byte_size"] != int(np.prod(shape)) * size
                        ):
                            raise ValueError(
                                f"{key + suffix}: invalid static FP8 dtype/shape/bytes"
                            )
                mappings["__MOE_EXPERTS__" + prefix] = WeightMapping(
                    target_path=[prefix] + [key + ".weight" for key in keys],
                    transpose=True,
                    sharding=tuple(weight.sharding.spec),
                )
                mappings["__MOE_EXPERTS__" + prefix + "_scale"] = WeightMapping(
                    target_path=[prefix + "_scale"] + [key + ".scale" for key in keys],
                    transpose=False,
                    sharding=(scale.sharding.spec[0], scale.sharding.spec[-1]),
                )
            # Scale expansion performs JAX operations on the expert/tensor mesh.
            with jax.set_mesh(experts.moe_mesh):
                loader.load_weights_from_safetensors(mappings)
            for target in ("wi_0", "wi_1", "wo", "wi_0_scale", "wi_1_scale", "wo_scale"):
                getattr(experts, target).value.block_until_ready()
            logger.info(
                "Loaded DeepSeek V4 layer %d/%d static FP8 experts via WeightLoader in %.1fs",
                layer_id + 1,
                len(self.model.layers),
                time.monotonic() - started,
            )

    def _load_expert_weights(self, info):
        from jax.sharding import SingleDeviceSharding

        from sgl_jax.srt.utils.quantization.mxfp4_fp8_loader import (
            convert_mxfp4_pair_from_safetensors,
        )

        def on_device(call, device, *args):
            local_mesh = jax.sharding.Mesh(
                np.asarray([device]), ("loader",), axis_types=(jax.sharding.AxisType.Explicit,)
            )
            with jax.set_mesh(local_mesh):
                return call(*args)

        def empty_shards(param, mesh):
            old = param.value
            sharding = NamedSharding(mesh, old.sharding.spec)
            shards = {}
            for device, index in sharding.addressable_devices_indices_map(old.shape).items():
                local_shape = tuple(len(range(*sl.indices(n))) for sl, n in zip(index, old.shape))
                placement = SingleDeviceSharding(device)
                initialize = jax.jit(
                    lambda shape=local_shape, dtype=old.dtype: jnp.zeros(shape, dtype),
                    out_shardings=placement,
                )
                update = jax.jit(
                    lambda buffer, row, offset: jax.lax.dynamic_update_slice(
                        buffer, row[None], (offset,) + (0,) * (buffer.ndim - 1)
                    ),
                    donate_argnums=(0,),
                    out_shardings=placement,
                )
                shards[device] = [index, on_device(initialize, device), update]
            return sharding, shards

        for layer_id, layer in enumerate(self.model.layers):
            experts = layer.mlp.experts
            if experts.num_experts != self.config.n_routed_experts:
                raise ValueError(
                    "V4 checkpoint loading currently requires identity expert placement"
                )
            for source, target in (("w1", "wi_0"), ("w3", "wi_1"), ("w2", "wo")):
                started = time.monotonic()
                weight_param = getattr(experts, target)
                scale_param = getattr(experts, target + "_scale")
                parameters = [(weight_param, False)]
                if scale_param is not None:
                    parameters.append((scale_param, True))
                buffers = [
                    (param, scale, *empty_shards(param, experts.moe_mesh))
                    for param, scale in parameters
                ]
                # One conversion per local logical expert, shared by all its TP
                # slices and both weight/scale arrays. Only one expert's host
                # payload is live; final expert stacks are assembled on devices.
                for expert_id in range(experts.num_experts):
                    local = any(
                        expert_id in range(*index[0].indices(weight_param.value.shape[0]))
                        for index, _, _ in buffers[0][3].values()
                    )
                    if not local:
                        continue
                    stem = f"layers.{layer_id}.ffn.experts.{expert_id}.{source}"
                    wk, sk = stem + ".weight", stem + ".scale"
                    converted = convert_mxfp4_pair_from_safetensors(
                        info[wk][0]["file"], wk, sk, scale_file=info[sk][0]["file"], strict=True
                    )
                    weight, scale = converted.weight_fp8, converted.scale_fp32
                    for param, is_scale, _, shards in buffers:
                        if is_scale:
                            value = scale[None, None, :]
                        elif np.dtype(param.value.dtype) == np.dtype(jnp.float8_e4m3fn):
                            value = weight.T
                        else:
                            value = (weight.astype(np.float32) * scale[:, None]).T
                        if value.shape != param.value.shape[1:]:
                            raise ValueError(
                                f"{stem}: decoded expert shape {value.shape} != {param.value.shape[1:]}"
                            )
                        for device, shard in shards.items():
                            index, array, update = shard
                            owned = range(*index[0].indices(param.value.shape[0]))
                            if expert_id not in owned:
                                continue
                            placement = SingleDeviceSharding(device)
                            row = jax.device_put(
                                np.ascontiguousarray(value[index[1:]], dtype=param.value.dtype),
                                placement,
                            )
                            offset = jax.device_put(np.int32(owned.index(expert_id)), placement)
                            shard[1] = on_device(update, device, array, row, offset)
                    # Bound outstanding host transfers before releasing this expert.
                    for _, _, _, shards in buffers:
                        for _, array, _ in shards.values():
                            array.block_until_ready()
                for param, _, sharding, shards in buffers:
                    param.value = jax.make_array_from_single_device_arrays(
                        param.value.shape, sharding, [array for _, array, _ in shards.values()]
                    )
                logger.info(
                    "Loaded DeepSeek V4 layer %d/%d routed %s in %.1fs (%s)",
                    layer_id + 1,
                    len(self.model.layers),
                    source,
                    time.monotonic() - started,
                    "strict MXFP4 conversion",
                )


EntryClass = [DeepseekV4ForCausalLM]
