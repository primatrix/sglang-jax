"""DeepSeek V4 trunk: model parameters, mixed-checkpoint loading and forward graph.

MTP keys are explicitly excluded. Attention backends own cache execution; all
V4 weights, including compressors, indexer, mHC and routing, are owned here.
"""

from __future__ import annotations

import enum
import logging
import os
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
from sgl_jax.srt.kernels.dsv4.q_head_norm_rope import (
    kernel_enabled as q_norm_rope_kernel_enabled,
)
from sgl_jax.srt.kernels.dsv4.q_head_norm_rope import q_head_norm_rope
from sgl_jax.srt.kernels.sparse_core.moe_permute import moe_sc_permute_enabled_by_env
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
            # V4 defaults: SparseCore permute/combine (SGL_JAX_MOE_SC_PERMUTE=false
            # turns it off) and sort-free routing permutations for decode.
            use_sc_permute=moe_sc_permute_enabled_by_env("true"),
            sort_free_permute=True,
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

    def _fused_experts(self, hidden_states, topk_weights, topk_ids, out_sharding):
        """Route the routed experts through kernels/fused_moe v2 (one Pallas call per layer).

        Uses the EPMoE module's own expert-sharded FP8 weights and per-channel scales
        (``wi_0``/``wi_1``/``wo`` == kernel ``w1``/``w3``/``w2``); V4's biased grouped
        top-k and the shared experts stay exactly as they are. Requires expert-parallel
        weights (ep_size == number of devices), see ``_use_fused_moe``.
        """
        from sgl_jax.srt.kernels.fused_moe.v2.kernel import fused_ep_moe_v2
        from sgl_jax.srt.kernels.fused_moe.v2.tuned_block_configs import (
            get_tuned_fused_moe_v2_block_config,
        )

        ex = self.experts
        mesh = ex.mesh
        tok_sh = jax.sharding.NamedSharding(mesh, P(("data", "tensor"), None))
        w_sh = jax.sharding.NamedSharding(mesh, P(("data", "tensor"), None, None))
        s_sh = jax.sharding.NamedSharding(mesh, P(("data", "tensor"), None, None, None))
        # The kernel shards tokens over every device: pad the token axis to a multiple
        # of the device count (zero rows routed to expert 0 with weight 0), slice after.
        n_tokens = hidden_states.shape[0]
        n_dev = int(np.prod(list(mesh.shape.values())))
        pad = (-n_tokens) % n_dev
        if pad:
            hidden_states = jnp.pad(hidden_states, ((0, pad), (0, 0)))
            topk_weights = jnp.pad(topk_weights, ((0, pad), (0, 0)))
            topk_ids = jnp.pad(topk_ids, ((0, pad), (0, 0)))
        x = jax.sharding.reshard(hidden_states, tok_sh)
        tw = jax.sharding.reshard(topk_weights.astype(jnp.float32), tok_sh)
        ti = jax.sharding.reshard(topk_ids.astype(jnp.int32), tok_sh)
        w1 = jax.sharding.reshard(ex.wi_0.value, w_sh)
        w3 = jax.sharding.reshard(ex.wi_1.value, w_sh)
        w2 = jax.sharding.reshard(ex.wo.value, w_sh)
        scales = [
            (
                None
                if getattr(ex, n, None) is None
                else jax.sharding.reshard(getattr(ex, n).value, s_sh)
            )
            for n in ("wi_0_scale", "wi_1_scale", "wo_scale")
        ]
        quant_mode = "none" if scales[0] is None else "per_channel"
        # fp8 per-token activation quant inside the kernel (what EPMoE does via qmm);
        # opt-in while we measure it against the bf16-activation path.
        act_quant = os.environ.get("DSV4_FUSED_ACT_QUANT", "0") == "1" and scales[0] is not None
        block_config = get_tuned_fused_moe_v2_block_config(
            num_tokens=x.shape[0],
            num_experts=ex.num_experts,
            top_k=self.top_k,
            hidden_size=self.hidden_size,
            intermediate_size=ex.intermediate_dim,
            dtype=x.dtype,
            weight_dtype=w1.dtype,
            ep_size=ex.ep_size,
            use_shared_expert=False,
            use_grouped_topk=False,
            enable_act_quant=act_quant,
            quant_mode=quant_mode,
        )
        out = fused_ep_moe_v2(
            mesh,
            x,
            w1,
            w2,
            w3,
            tw,
            ti,
            self.top_k,
            act_fn="silu",
            swiglu_limit=ex.swiglu_limit,
            block_config=block_config,
            quant_block_k=None,
            w1_scale=scales[0],
            w2_scale=scales[2],
            w3_scale=scales[1],
            enable_act_quant=act_quant,
            direct_scaled_dot=scales[0] is not None,
            dp_axis_name="data",
            tp_axis_name="tensor",
        )
        # Reshard before dropping the pad rows: slicing the (data, tensor)-sharded
        # token axis down to a size the 8 devices cannot divide is rejected.
        target = out_sharding or jax.sharding.NamedSharding(mesh, P("data", None))
        out = jax.sharding.reshard(out, target)
        if pad:
            out = out[:n_tokens]
        return out

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
        output_sharding=None,
    ):
        # ``output_sharding``: sharding of the returned rows only (the routing keeps
        # ``out_sharding``); DSV4_SEQ_PARALLEL asks for P("tensor", None) so the
        # expert combine is a reduce-scatter and the shared-expert / mask terms
        # are brought onto the same rows.
        if output_sharding is None:
            output_sharding = out_sharding
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
        if _use_fused_moe(self.experts) and hidden_states.shape[0] >= _FUSED_MOE_MIN_TOKENS:
            # Auto-tuned v7x blocks: the fused kernel is -17% per layer on 8K prefill
            # chunks but +35% on decode buckets (~90 us fixed cost per call).
            output = self._fused_experts(hidden_states, weights, ids, output_sharding)
        else:
            output = self.experts(hidden_states, weights, ids, out_sharding=output_sharding)
        if self.shared_experts is not None:
            # routed_scaling_factor applies only to routed weights, exactly once.
            shared = self.shared_experts(hidden_states)
            if output_sharding is not None:
                shared = jax.sharding.reshard(shared, output_sharding)
            output = output + shared
        return jnp.where(self._rows_like(valid, output_sharding)[:, None], output, 0), ids

    def _rows_like(self, valid, output_sharding):
        """``valid`` [T] on the same row sharding as the output (a slice when sharded)."""
        if output_sharding is None:
            return valid
        return jax.sharding.reshard(valid, NamedSharding(self.mesh, P(output_sharding.spec[0])))


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
        # W8A8 when requested: the blockwise qmm quantizes activations per token
        # (GPU serving runs every dense fp8 GEMM w8a8; default here stays w8a16).
        activation_dtype=(
            jnp.float8_e4m3fn
            if _W8A8_DENSE and (_W8A8_DENSE_NAMES is None or name in _W8A8_DENSE_NAMES)
            else None
        ),
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


_FUSED_MOE_MIN_TOKENS = int(os.environ.get("DSV4_FUSED_MOE_MIN_TOKENS", "256"))
# ``DSV4_SP_NORM_BEFORE_GATHER=1``: under sequence parallelism apply the sublayer
# RMSNorm (per row) and the bf16 cast on the T/tp rows and all-gather the result,
# instead of gathering the raw stream and normalising all T rows on every device.
# Same values; the gather carries bf16 instead of the stream dtype.
_SP_NORM_BEFORE_GATHER = os.environ.get("DSV4_SP_NORM_BEFORE_GATHER", "1") == "1"
# ``DSV4_MOE_MERGED_GATE_UP=1``: run the routed experts' gate and up projections as one gmm.
_MERGED_GATE_UP = os.environ.get("DSV4_MOE_MERGED_GATE_UP", "0") == "1"
# ``DSV4_LOWRANK_AG=1``: on CSA layers under sequence parallelism, project q_lora / kv /
# indexer weights on the local T/tp rows and all-gather those (1024+512+64 columns)
# instead of the 4096-wide hidden; the compressors then run on local rows with a
# ppermute halo (needs DSV4_COMPRESSOR_ROW_SHARD=1). HCA layers keep the full gather.
_LOWRANK_AG = os.environ.get("DSV4_LOWRANK_AG", "1") == "1"


def _q_norm_rope(q, cos, sin, *, heads, head_dim, rope_head_dim, normalize, eps, out_dtype):
    """`q_head_norm_rope` on a ``[T, heads*head_dim]`` projection -> ``[T, heads, head_dim]``.

    A head-parallel projection (lanes sharded over ``tensor``) runs the kernel per
    shard on its local heads; a replicated one runs it directly.
    """
    sharding = jax.typeof(q).sharding
    spec = sharding.spec
    lane_axis = spec[1] if len(spec) > 1 else None
    # The step function traces without a mesh context (run_and_sample sets none), so the
    # projection's own sharding mesh is the only reliable source of axis names and types.
    # A lane-replicated q runs the kernel directly when that mesh is empty (CPU tests) or
    # when every axis is already Manual (inside the decode step's shard_map, where q is a
    # per-device value). Under explicit axes (the prefill program) it must go through
    # shard_map: pallas_call requires Manual axes, and a direct call traced the kernel
    # body with sharded block values and failed on the first iota-vs-block select.
    mesh = sharding.mesh
    manual = not mesh.shape or all(
        t == jax.sharding.AxisType.Manual for t in getattr(mesh, "axis_types", ())
    )
    if lane_axis is None and manual:
        return q_head_norm_rope(
            q,
            cos,
            sin,
            heads=heads,
            head_dim=head_dim,
            rope_head_dim=rope_head_dim,
            normalize=normalize,
            eps=eps,
            out_dtype=out_dtype,
        ).reshape(-1, heads, head_dim)
    shards = int(mesh.shape[lane_axis]) if isinstance(lane_axis, str) else 1
    local_heads = heads // shards
    row_axis = spec[0]  # tokens may be sharded over data; keep whatever the input has

    def per_shard(q_local, cos_local, sin_local):
        out = q_head_norm_rope(
            q_local,
            cos_local,
            sin_local,
            heads=local_heads,
            head_dim=head_dim,
            rope_head_dim=rope_head_dim,
            normalize=normalize,
            eps=eps,
            out_dtype=out_dtype,
        )
        return out.reshape(-1, local_heads, head_dim)

    cos_spec = jax.typeof(cos).sharding.spec
    sin_spec = jax.typeof(sin).sharding.spec
    return jax.shard_map(
        per_shard,
        mesh=mesh,
        in_specs=(P(row_axis, lane_axis), P(*cos_spec), P(*sin_spec)),
        out_specs=P(row_axis, lane_axis, None),
        check_vma=False,
    )(q, cos, sin)


# ``DSV4_HCA_FUSED_PROJ=1`` (default): build the HCA compressor's fused ``[Wkv|Wgate]^T`` bf16
# projection once after loading instead of converting the f32 gate weight and
# concatenating on every step in every HCA layer.
_HCA_FUSED_PROJ = os.environ.get("DSV4_HCA_FUSED_PROJ", "1") == "1"
_WGATE_F32 = os.environ.get("DSV4_COMPRESSOR_WGATE_F32", "0") == "1"
# ``DSV4_W8A8_DENSE=1``: fp8 activations for the dense fp8 linears (weights are
# already fp8); default keeps bf16 activations.
_W8A8_DENSE = os.environ.get("DSV4_W8A8_DENSE", "1") == "1"
# ``DSV4_W8A8_DENSE_NAMES=wq_b,wo_b``: restrict fp8 activations to these linears (an empty
# value means all of them).
_W8A8_DENSE_NAMES_ENV = os.environ.get(
    "DSV4_W8A8_DENSE_NAMES", "wq_a,wkv,wo_a,wo_b,indexer_wq_b,gate_proj,up_proj,down_proj"
)
_W8A8_DENSE_NAMES = (
    frozenset(x.strip() for x in _W8A8_DENSE_NAMES_ENV.split(",") if x.strip())
    if _W8A8_DENSE_NAMES_ENV
    else None
)


def _use_fused_moe(experts) -> bool:
    """``DSV4_MOE_BACKEND=fused`` routes routed experts through kernels/fused_moe v2.

    Only meaningful with expert-parallel weights (``--ep-size`` == device count);
    otherwise the EPMoE tensor-parallel path is kept regardless of the flag.
    """
    if os.environ.get("DSV4_MOE_BACKEND", "epmoe").lower() != "fused":
        return False
    return int(getattr(experts, "ep_size", 1)) == int(np.prod(list(experts.mesh.shape.values())))


# DSV4_ROPE_CACHE_LANE_PAD=1: store the [max_position, cos|sin] tables with the lane axis
# padded to a multiple of 128. With 64 lanes XLA picks a column-major layout for the jit
# parameter and inserts a full-table relayout copy (2 x 268 MB) at the top of every step
# (measured 0.71 ms/step on v7x). Consumers slice the first ``rope_head_dim`` lanes.
_ROPE_CACHE_LANE_PAD = os.environ.get("DSV4_ROPE_CACHE_LANE_PAD", "1") == "1"


def _split_rope_cache(cache, rope_dim):
    """``[N, >=rope_dim]`` cos|sin table -> (cos, sin) halves, materialised once (not per step)."""
    half = rope_dim // 2
    return cache[:, :half], cache[:, half : 2 * half]


def _rope_cache(config, ratio):
    from sgl_jax.srt.layers.attention.dsv4.rope import build_dsv4_rope

    rope = build_dsv4_rope(config, ratio, dtype=jnp.float32)
    cos, sin = rope._compute_cos_sin(jnp.arange(config.max_position_embeddings, dtype=jnp.int32))
    cache = jnp.concatenate((cos, sin), axis=-1)
    if _ROPE_CACHE_LANE_PAD and cache.shape[-1] % 128:
        cache = jnp.pad(cache, ((0, 0), (0, -cache.shape[-1] % 128)))
    return cache


class DeepseekV4Compressor(nnx.Module):
    """All compressor parameters stay in the modeling file, in checkpoint layout."""

    def __init__(self, config, head_dim, ratio, dtype):
        from sgl_jax.srt.layers.layernorm import RMSNorm

        width = head_dim * (2 if ratio == 4 else 1)
        self.wkv = nnx.Param(
            jnp.zeros((width, config.hidden_size), dtype, out_sharding=P(None, None))
        )
        # The checkpoint stores wgate in BF16; keeping the parameter in BF16 is exact
        # and halves the bytes every compress projection streams per layer per step
        # (the projections cast to f32 / HIGHEST themselves). DSV4_COMPRESSOR_WGATE_F32=1
        # restores the previous f32 storage for A/B.
        self.wgate = nnx.Param(
            jnp.zeros(
                (width, config.hidden_size),
                jnp.float32 if _WGATE_F32 else dtype,
                out_sharding=P(None, None),
            )
        )
        self.ape = nnx.Param(jnp.zeros((ratio, width), jnp.float32, out_sharding=P(None, None)))
        self.norm = RMSNorm(head_dim, epsilon=config.rms_norm_eps, param_dtype=jnp.float32)
        self.ratio = ratio

    def prepare_fused_projection(self, mesh):
        """Materialise the HCA kernels' fused ``[hidden, 2*D]`` bf16 projection once.

        Without it ``fused_projection_weight`` rebuilds it in every HCA layer on every
        step: an f32->bf16 convert of ``wgate`` (8 MB prefetched through VMEM), a
        concatenation and a transpose.
        """
        with jax.set_mesh(mesh):
            fused = jnp.concatenate(
                (self.wkv.value.astype(jnp.bfloat16), self.wgate.value.astype(jnp.bfloat16)),
                axis=0,
            ).T
        self.fused_proj = nnx.Variable(fused)

    def weights(self, cache, halves=None):
        from sgl_jax.srt.layers.attention.dsv4.execution import CompressorWeights

        cos_table, sin_table = halves if halves is not None else (None, None)
        fused = getattr(self, "fused_proj", None)
        return CompressorWeights(
            self.wkv.value,
            self.wgate.value,
            self.ape.value,
            self.norm.scale.value,
            cache,
            cos_table,
            sin_table,
            None if fused is None else fused.value,
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

    def weights_from_hidden(self, hidden):
        """``[rows, H_idx]`` f32 per-head indexer weights (row-local, no communication)."""
        return jnp.dot(hidden.astype(jnp.float32), self.weights_proj.value.T.astype(jnp.float32))

    def project(self, q_lora, weights, cos, sin, cache, dtype):
        """Indexer inputs from an already-gathered ``q_lora`` and the raw weights."""
        from sgl_jax.srt.layers.attention.dsv4.execution import IndexerInputs
        from sgl_jax.srt.layers.attention.dsv4.rope import apply_dsv4_partial_rope

        q, _ = self.wq_b(q_lora)
        if q_norm_rope_kernel_enabled():
            q = _q_norm_rope(
                q,
                cos,
                sin,
                heads=self.num_heads,
                head_dim=self.head_dim,
                rope_head_dim=self.rope_head_dim,
                normalize=False,
                eps=0.0,
                out_dtype=dtype,
            )
        else:
            q = q.reshape(-1, self.num_heads, self.head_dim)
            q = apply_dsv4_partial_rope(
                q, cos[:, None, :], sin[:, None, :], rope_head_dim=self.rope_head_dim
            ).astype(dtype)
        return IndexerInputs(q, weights * self.weight_scale, self.compressor.weights(cache))

    def __call__(self, hidden, q_lora, cos, sin, cache):
        return self.project(q_lora, self.weights_from_hidden(hidden), cos, sin, cache, hidden.dtype)


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

    def prepare_grouped_wo_a(self):
        """Materialise the grouped, dequantised wo_a once after loading.

        The forward used to dequantise (scale broadcast + multiply) and regroup wo_a on
        every step; at bs=1 that was ~0.9 ms per decode step across the layers.
        """
        from sgl_jax.srt.layers.attention.dsv4.o_projection import (
            fuse_wo_a_weights,
            group_wo_a,
            use_fused_wo_a,
        )

        with jax.set_mesh(self.mesh):
            weights = group_wo_a(
                _checkpoint_matrix(self.wo_a),
                num_groups=self.num_groups,
                out_sharding=NamedSharding(self.mesh, P("tensor", None, None)),
            )
            if use_fused_wo_a():
                # The fused kernel wants [8*head_dim, G*R]; keep only that copy.
                self.wo_a_fused = nnx.Param(fuse_wo_a_weights(weights, mesh=self.mesh))
                return
        self.wo_a_grouped = nnx.Param(weights)

    def __call__(
        self,
        hidden,
        batch,
        pools,
        rope_cache,
        rope_halves=None,
        wo_out_sharding=None,
        sp_local=False,
    ):
        from sgl_jax.srt.layers.attention.dsv4.o_projection import group_wo_a
        from sgl_jax.srt.layers.attention.dsv4.rope import apply_dsv4_partial_rope

        indexer_weights = None
        if sp_local:
            # ``hidden`` is this device's T/tp rows (DSV4_LOWRANK_AG): project locally,
            # gather the narrow results once. Row-wise norms commute with the gather.
            rows_sh = NamedSharding(self.mesh, P("tensor", None))
            q_lora_l, _ = self.wq_a(hidden, out_sharding=rows_sh)
            q_lora_l = self.q_norm(q_lora_l)
            kv_l, _ = self.wkv(hidden, out_sharding=rows_sh)
            kv_l = self.kv_norm(kv_l)
            parts = [q_lora_l.astype(self.dtype), kv_l.astype(self.dtype)]
            if self.indexer is not None:
                # f32 weights ride along as two bf16 halves (exact round trip)
                w32 = jax.lax.bitcast_convert_type(
                    self.indexer.weights_from_hidden(hidden), jnp.uint32
                )
                parts.append(
                    jax.lax.bitcast_convert_type((w32 >> 16).astype(jnp.uint16), jnp.bfloat16)
                )
                parts.append(
                    jax.lax.bitcast_convert_type((w32 & 0xFFFF).astype(jnp.uint16), jnp.bfloat16)
                )
            packed = _sp_gather(self.mesh, jnp.concatenate(parts, axis=-1))
            widths = [p.shape[-1] for p in parts]
            cuts = np.cumsum(widths)[:-1].tolist()
            pieces = jnp.split(packed, cuts, axis=-1)
            q_lora, kv = pieces[0], pieces[1]
            if self.indexer is not None:
                hi = jax.lax.bitcast_convert_type(pieces[2], jnp.uint16).astype(jnp.uint32)
                lo = jax.lax.bitcast_convert_type(pieces[3], jnp.uint16).astype(jnp.uint32)
                indexer_weights = jax.lax.bitcast_convert_type((hi << 16) | lo, jnp.float32)
        else:
            q_lora, _ = self.wq_a(hidden)
            q_lora = self.q_norm(q_lora)
        q, _ = self.wq_b(q_lora)
        if not sp_local:
            kv, _ = self.wkv(hidden)
            kv = self.kv_norm(kv)
        positions = batch.positions
        selected = rope_cache.at[positions].get(
            out_sharding=NamedSharding(self.mesh, P("data", None))
        )
        cos, sin = jnp.split(selected[:, : self.rope_head_dim], 2, axis=-1)
        if q_norm_rope_kernel_enabled():
            # One pass over the [T, H*D] projection: per-head norm + partial rope,
            # written straight in the attention kernels' [T*H, D] row order.
            q = _q_norm_rope(
                q,
                cos,
                sin,
                heads=self.num_heads,
                head_dim=self.head_dim,
                rope_head_dim=self.rope_head_dim,
                normalize=True,
                eps=self.norm_eps,
                out_dtype=self.dtype,
            )
        else:
            q = q.reshape(-1, self.num_heads, self.head_dim)
            # V4 normalizes q again per head after wq_b, with no learned weight.
            q = (
                q.astype(jnp.float32)
                * jax.lax.rsqrt(
                    jnp.mean(jnp.square(q.astype(jnp.float32)), axis=-1, keepdims=True)
                    + self.norm_eps
                )
            ).astype(self.dtype)
            q = apply_dsv4_partial_rope(
                q, cos[:, None, :], sin[:, None, :], rope_head_dim=self.rope_head_dim
            ).astype(self.dtype)
        kv = apply_dsv4_partial_rope(kv, cos, sin, rope_head_dim=self.rope_head_dim).astype(
            self.dtype
        )
        if self.indexer is None:
            indexer = None
        elif sp_local:
            indexer = self.indexer.project(
                q_lora, indexer_weights, cos, sin, rope_cache, self.dtype
            )
        else:
            indexer = self.indexer(hidden, q_lora, cos, sin, rope_cache)
        output, updates = batch.attn_backend(
            q,
            kv,
            kv,
            self,
            batch,
            pools.token_to_kv_pool,
            compressor_state_pool=pools.compressor_state_pool,
            compressor_input=hidden,
            compressor_input_local=sp_local,
            compressor=(
                None
                if self.compressor is None
                else self.compressor.weights(rope_cache, rope_halves)
            ),
            indexer=indexer,
            attention_sink=self.attn_sink.value,
            rope_head_dim=self.rope_head_dim,
            norm_eps=self.norm_eps,
            index_topk=self.index_topk,
        )
        if getattr(self, "wo_a_fused", None) is not None:
            from sgl_jax.srt.layers.attention.dsv4.o_projection import (
                fused_wo_a_projection,
            )

            reduced = fused_wo_a_projection(
                output,
                cos,
                sin,
                self.wo_a_fused.value,
                mesh=self.mesh,
                rope_head_dim=self.rope_head_dim,
                dtype=self.dtype,
            )
            output, _ = self.wo_b(reduced, out_sharding=wo_out_sharding)
            return output, updates
        output = apply_dsv4_partial_rope(
            output, cos[:, None, :], sin[:, None, :], rope_head_dim=self.rope_head_dim, inverse=True
        )
        grouped = jax.lax.reshape(
            output,
            (output.shape[0], self.num_groups, self.num_heads // self.num_groups * self.head_dim),
            out_sharding=NamedSharding(self.mesh, P("data", "tensor", None)),
        )
        if getattr(self, "wo_a_grouped", None) is not None:
            weights = self.wo_a_grouped.value
        else:
            weights = group_wo_a(
                _checkpoint_matrix(self.wo_a),
                num_groups=self.num_groups,
                out_sharding=NamedSharding(self.mesh, P("tensor", None, None)),
            )
        reduced = jnp.einsum("tgd,gdr->tgr", grouped, weights, preferred_element_type=jnp.float32)
        reduced = reduced.reshape(reduced.shape[0], -1).astype(self.dtype)
        output, _ = self.wo_b(reduced, out_sharding=wo_out_sharding)
        return output, updates


_MHC_SEAM_MIN_TOKENS = int(os.environ.get("DSV4_MHC_SEAM_MIN_TOKENS", "64"))


# ``DSV4_SEQ_PARALLEL=1``: sequence parallelism for the mHC / norm / residual work.
# Today every TP rank runs the mHC pre/post/seam kernels, the layer norms and the
# residual adds over all T rows (P("data", ...) = replicated across "tensor"):
# ~55 ms of an 8K prefill step on v7x that each of the 8 ranks repeats. With the
# flag the streams live row-sharded over the tensor axis (P("tensor", ...)), the
# row-parallel wo_b returns a reduce-scatter instead of an all-reduce, and the
# hidden states are all-gathered only where a full row set is needed (the
# attention projections, the MoE dispatch, the final norm). Same bytes on the
# ICI (RS + AG == AR); the replicated compute shrinks by the tensor size. Only
# prefill buckets take the path (rows >= DSV4_SEQ_PARALLEL_MIN_TOKENS and
# divisible by the tensor axis); decode keeps the replicated form.
_SEQ_PARALLEL = os.environ.get("DSV4_SEQ_PARALLEL", "1") == "1"
_SEQ_PARALLEL_MIN_TOKENS = int(os.environ.get("DSV4_SEQ_PARALLEL_MIN_TOKENS", "256"))


def _sp_active(mesh, rows: int) -> bool:
    if not _SEQ_PARALLEL:
        return False
    tp = int(mesh.shape.get("tensor", 1))
    return tp > 1 and rows >= _SEQ_PARALLEL_MIN_TOKENS and rows % tp == 0


def _sp_gather(mesh, x):
    """Row-sharded ``[T/tp, ...]`` -> replicated ``[T, ...]`` (all-gather over tensor)."""
    return jax.sharding.reshard(x, NamedSharding(mesh, P("data", *([None] * (x.ndim - 1)))))


def _sp_rows(mesh, x):
    """Replicated ``[T, ...]`` -> row-sharded over the tensor axis (a local slice)."""
    return jax.sharding.reshard(x, NamedSharding(mesh, P("tensor", *([None] * (x.ndim - 1)))))


def _use_mhc_seam() -> bool:
    """``DSV4_MHC_SEAM=1``: fuse each sublayer's mHC post with the next sublayer's pre."""
    return os.environ.get("DSV4_MHC_SEAM", "1") == "1"


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
        row = "tensor" if _sp_active(self.mesh, streams.shape[0] * self._sp_tp(streams)) else "data"
        specs = (P(row, None), P(row, None), P(row, None, None))
        compute = jax.shard_map(
            self.mhc.pre,
            mesh=None,
            in_specs=(P(row, None, None), P(), P(), P()),
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
        row = (
            "tensor" if _sp_active(self.mesh, residual.shape[0] * self._sp_tp(residual)) else "data"
        )
        spec = P(row, None, None)
        compute = jax.shard_map(
            self.mhc.post,
            mesh=None,
            in_specs=(P(row, None), spec, P(row, None), spec),
            out_specs=spec,
            check_vma=False,
        )
        compute = jax.sharding.auto_axes(
            compute, axes=self.mesh.axis_names, out_sharding=NamedSharding(self.mesh, spec)
        )
        return compute(output, residual, post, comb)

    def _mhc_seam(self, output, residual, post, comb, fn, base, scale):
        """post(output) fused with the next sublayer's pre: one kernel, streams stored once."""
        row = (
            "tensor" if _sp_active(self.mesh, residual.shape[0] * self._sp_tp(residual)) else "data"
        )
        stream_spec = P(row, None, None)
        out_specs = (stream_spec, P(row, None), P(row, None), stream_spec)
        compute = jax.shard_map(
            self.mhc.seam,
            mesh=None,
            in_specs=(P(row, None), stream_spec, P(row, None), stream_spec, P(), P(), P()),
            out_specs=out_specs,
            check_vma=False,
        )
        compute = jax.sharding.auto_axes(
            compute,
            axes=self.mesh.axis_names,
            out_sharding=tuple(NamedSharding(self.mesh, spec) for spec in out_specs),
        )
        return compute(output, residual, post, comb, fn, base, scale)

    def attn_params(self):
        return (self.hc_attn_fn.value, self.hc_attn_base.value, self.hc_attn_scale.value)

    def _sp_tp(self, x):
        # Global arrays carry the global row count whether replicated or sharded; the
        # gate below only needs T itself, so this is 1 (kept for readability).
        return 1

    def _lowrank_attn(self, hidden, batch) -> bool:
        """DSV4_LOWRANK_AG applies on CSA layers, under SP, for single-request batches
        (the row-local compressors have no multi-request path)."""
        if not _LOWRANK_AG or getattr(self.self_attn, "ratio", None) != 4:
            return False
        tp = int(self.mesh.shape.get("tensor", 1))
        if not _sp_active(self.mesh, hidden.shape[0] * tp):
            return False
        if os.environ.get("DSV4_COMPRESSOR_ROW_SHARD", "1") != "1":
            raise ValueError("DSV4_LOWRANK_AG needs DSV4_COMPRESSOR_ROW_SHARD=1")
        return int(batch.seq_lens.shape[0]) == 1

    def _sp_full(self, hidden):
        """All-gather a row-sharded pre-output before the attention / MoE projections."""
        if _sp_active(self.mesh, hidden.shape[0]):
            return _sp_gather(self.mesh, hidden)
        return hidden

    def _sp_shard(self, x):
        """Bring a sublayer output onto the streams' row sharding (no-op if already there)."""
        if _sp_active(self.mesh, x.shape[0]):
            return _sp_rows(self.mesh, x)
        return x

    def _sp_wo_sharding(self, rows: int):
        if _sp_active(self.mesh, rows):
            return NamedSharding(self.mesh, P("tensor", None))
        return None

    def call_seam(
        self, streams, hidden, post, comb, next_params, batch, pools, rope_cache, rope_halves
    ):
        """Seam-fused layer step: ``(streams, hidden, post, comb)`` in and out.

        ``hidden/post/comb`` are this layer's attention-side pre outputs (from the
        previous seam or the model's first pre); ``next_params`` are the next layer's
        attention hc params, or None for the last layer (plain post, hidden None).
        """
        lowrank = self._lowrank_attn(hidden, batch)
        if lowrank:
            # DSV4_LOWRANK_AG: hand the attention the local rows; it gathers q_lora/kv
            full_rows = hidden.shape[0] * int(self.mesh.shape["tensor"])
            attn_in = self.attn_norm(hidden.astype(self.dtype))
        elif _SP_NORM_BEFORE_GATHER:
            # RMSNorm is per row: normalise (and cast) the T/tp rows, then gather bf16.
            hidden_full = self._sp_full(self.attn_norm(hidden.astype(self.dtype)))
            attn_in = hidden_full
            full_rows = hidden_full.shape[0]
        else:
            hidden_full = self._sp_full(hidden)
            attn_in = self.attn_norm(hidden_full.astype(self.dtype))
            full_rows = hidden_full.shape[0]
        attn, updates = self.self_attn(
            attn_in,
            batch,
            pools,
            rope_cache,
            rope_halves,
            wo_out_sharding=self._sp_wo_sharding(full_rows),
            sp_local=lowrank,
        )
        attn = self._sp_shard(attn)
        streams, hidden, post, comb = self._mhc_seam(
            attn,
            streams,
            post,
            comb,
            self.hc_ffn_fn.value,
            self.hc_ffn_base.value,
            self.hc_ffn_scale.value,
        )
        streams = streams.astype(self.dtype)
        if _SP_NORM_BEFORE_GATHER:
            rows = self.ffn_norm(hidden.astype(self.dtype))
            hidden_full = self._sp_full(rows)
            ffn_in = hidden_full
        else:
            hidden_full = self._sp_full(hidden)
            ffn_in = self.ffn_norm(hidden_full.astype(self.dtype))
        ffn, ids = self.mlp(
            ffn_in,
            batch.input_ids,
            token_valid_mask=batch.get_token_valid_mask(hidden_full.shape[0]),
            dispatch_info=batch.expert_location_metadata,
        )
        ffn = self._sp_shard(ffn)
        if next_params is None:
            streams = self._mhc_post(ffn, streams, post, comb).astype(self.dtype)
            return streams, None, None, None, updates, ids
        fn, base, scale = next_params
        streams, hidden, post, comb = self._mhc_seam(ffn, streams, post, comb, fn, base, scale)
        return streams.astype(self.dtype), hidden, post, comb, updates, ids

    def __call__(self, streams, batch, pools, rope_cache, rope_halves=None):
        hidden, post, comb = self._mhc_pre(
            streams, self.hc_attn_fn.value, self.hc_attn_base.value, self.hc_attn_scale.value
        )
        lowrank = self._lowrank_attn(hidden, batch)
        if lowrank:
            # DSV4_LOWRANK_AG: hand the attention the local rows; it gathers q_lora/kv
            full_rows = hidden.shape[0] * int(self.mesh.shape["tensor"])
            attn_in = self.attn_norm(hidden.astype(self.dtype))
        elif _SP_NORM_BEFORE_GATHER:
            # RMSNorm is per row: normalise (and cast) the T/tp rows, then gather bf16.
            hidden_full = self._sp_full(self.attn_norm(hidden.astype(self.dtype)))
            attn_in = hidden_full
            full_rows = hidden_full.shape[0]
        else:
            hidden_full = self._sp_full(hidden)
            attn_in = self.attn_norm(hidden_full.astype(self.dtype))
            full_rows = hidden_full.shape[0]
        attn, updates = self.self_attn(
            attn_in,
            batch,
            pools,
            rope_cache,
            rope_halves,
            wo_out_sharding=self._sp_wo_sharding(full_rows),
            sp_local=lowrank,
        )
        attn = self._sp_shard(attn)
        streams = self._mhc_post(attn, streams, post, comb).astype(self.dtype)
        hidden, post, comb = self._mhc_pre(
            streams, self.hc_ffn_fn.value, self.hc_ffn_base.value, self.hc_ffn_scale.value
        )
        if _SP_NORM_BEFORE_GATHER:
            rows = self.ffn_norm(hidden.astype(self.dtype))
            hidden_full = self._sp_full(rows)
            ffn_in = hidden_full
        else:
            hidden_full = self._sp_full(hidden)
            ffn_in = self.ffn_norm(hidden_full.astype(self.dtype))
        ffn, ids = self.mlp(
            ffn_in,
            batch.input_ids,
            token_valid_mask=batch.get_token_valid_mask(hidden_full.shape[0]),
            dispatch_info=batch.expert_location_metadata,
        )
        ffn = self._sp_shard(ffn)
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
        cos, sin = _split_rope_cache(self.rope_compressed.value, config.qk_rope_head_dim)
        self.rope_compressed_cos = nnx.Variable(cos)
        self.rope_compressed_sin = nnx.Variable(sin)

    def _collapse_head(self, streams):
        params = (self.hc_head_fn.value, self.hc_head_base.value, self.hc_head_scale.value)
        if self.mhc.backend != "pallas":
            return self.mhc.collapse_head(streams, *params)
        row = "tensor" if _sp_active(self.mesh, streams.shape[0]) else "data"
        spec = P(row, None)
        compute = jax.shard_map(
            self.mhc.collapse_head,
            mesh=None,
            in_specs=(P(row, None, None), P(), P(), P()),
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
        if _sp_active(self.mesh, streams.shape[0]):
            streams = _sp_rows(self.mesh, streams)
        updates, ids = {}, []
        # The seam pays on prefill chunks (8K TTFT -10 ms, 32K -20 ms) and costs
        # +0.4 ms/step on decode buckets, so it is gated on the token count.
        seam = (
            _use_mhc_seam()
            and self.mhc.backend == "pallas"
            and streams.shape[0] >= _MHC_SEAM_MIN_TOKENS
        )
        if seam:
            first = self.layers[0]
            hidden, post, comb = first._mhc_pre(streams, *first.attn_params())
        for i, layer in enumerate(self.layers):
            cache = self.rope_compressed if layer.self_attn.ratio else self.rope_plain
            halves = (
                (self.rope_compressed_cos.value, self.rope_compressed_sin.value)
                if layer.self_attn.ratio
                else None
            )
            if seam:
                nxt = self.layers[i + 1].attn_params() if i + 1 < len(self.layers) else None
                streams, hidden, post, comb, updates[i], route_ids = layer.call_seam(
                    streams, hidden, post, comb, nxt, batch, pools, cache.value, halves
                )
            else:
                streams, updates[i], route_ids = layer(streams, batch, pools, cache.value, halves)
            ids.append(route_ids)
        hidden = self._collapse_head(streams)
        if _sp_active(self.mesh, hidden.shape[0]):
            hidden = _sp_gather(self.mesh, hidden)
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
            cos, sin = _split_rope_cache(
                self.model.rope_compressed.value, self.config.qk_rope_head_dim
            )
            self.model.rope_compressed_cos.value = cos
            self.model.rope_compressed_sin.value = sin
        # Per-step work that only depends on loaded weights is done once here.
        for layer in self.model.layers:
            layer.self_attn.prepare_grouped_wo_a()
            if _MERGED_GATE_UP and hasattr(layer.mlp, "experts"):
                layer.mlp.experts.prepare_merged_gate_up()
            compressor = getattr(layer.self_attn, "compressor", None)
            if _HCA_FUSED_PROJ and compressor is not None and compressor.ratio == 128:
                compressor.prepare_fused_projection(self.mesh)

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
