"""M1.1b -- DeepSeek-V4 checkpoint key mapping and disposition.

The Flash 0731 checkpoint has **72317 tensors across 48 shards**, and the risky part
is not any single mapping entry -- it is that a key can go unhandled without anything
saying so. A weight the loader never claims is silently left at its initial value,
which shows up as degraded output rather than an error. So the primary artifact here
is a **total classification**: every key in the checkpoint is exactly one of

    CLAIMED           mapped to a model parameter
    EXPERT_PENDING    a routed-expert tensor whose target layout is M0.2's to fix
    DROPPED           deliberately not loaded, with a reason

and `classify_checkpoint` raises if a key falls through. `build_weight_mappings`
produces the `WeightMapping` table for the claimed set.

What is dropped, and why
------------------------
``mtp.0`` / ``mtp.1`` / ``mtp.2`` -- **4705 tensors**, three whole draft blocks. The
config says ``num_nextn_predict_layers=1`` but the checkpoint ships three, and
``mtp.2`` additionally carries ``confidence_head`` / ``markov_head`` / its own
``hc_head_*`` and ``norm``, i.e. the DSpark stack (INFERENCE-84). MTP is out of the
first version, so these are dropped explicitly rather than left to loader tolerance:
"unclaimed" and "deliberately skipped" have to be distinguishable.

Per-layer families are gated on layer type
------------------------------------------
The checkpoint's own tensor counts encode the classification, and they line up with
`configs/deepseek_v4.classify_layers`:

    attn.compressor.*        41 layers  = 43 - 2 SWA-only  (ratio > 0)
    attn.indexer.*           21 layers  = the C4A layers   (ratio == 4)
    ffn.gate.tid2eid          3 layers  = num_hash_layers
    ffn.gate.bias            40 layers  = 43 - 3           (mutually exclusive with tid2eid)

So a mapping that asks for `compressor` on a SWA-only layer, or `gate.bias` on a hash
layer, is asking for a tensor that does not exist. Both directions are checked.

Mixed quantisation
------------------
``expert_dtype="fp4"`` but ``quantization_config`` is fp8/e4m3 with
``weight_block_size=[128,128]``, so the two families need different scale handling:

* non-expert ``.scale`` -- FP8 block scale, ``[out/128, in/128]``, expanded the way
  the existing loader does for GLM.
* routed-expert ``.scale`` -- FP4 source, and the resident representation is M0.2's
  power-of-two per-channel FP8. That target layout is not frozen, so those keys are
  classified ``EXPERT_PENDING`` rather than given a mapping that would have to be
  rewritten. Shared-expert and every non-expert tensor are mapped normally.

Tensors with **no** ``.scale`` sibling are not quantised and must not be pushed
through a quantised path: ``attn.compressor.*``, ``attn.indexer.compressor.*``,
``attn.indexer.weights_proj``, all norms, ``attn.attn_sink``, the mHC gates, and
``ffn.gate.weight``.
"""

from __future__ import annotations

import enum
import re
from dataclasses import dataclass

from sgl_jax.srt.configs.deepseek_v4 import (
    DeepseekV4LayerType,
    classify_layers,
    hash_moe_layer_flags,
)
from sgl_jax.srt.utils.weight_utils import WeightMapping

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
    EXPERT_PENDING = "expert_pending"
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
    "ffn.gate.weight": "mlp.gate.weight",
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
    "ffn.shared_experts.w1": "mlp.shared_experts.w1",
    "ffn.shared_experts.w2": "mlp.shared_experts.w2",
    "ffn.shared_experts.w3": "mlp.shared_experts.w3",
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
            Disposition.EXPERT_PENDING,
            "routed expert; the resident layout is M0.2's power-of-two per-channel FP8 "
            "and is not frozen, so no mapping is emitted yet",
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

    Routed-expert tensors are deliberately absent; see `Disposition.EXPERT_PENDING`.
    """
    mappings: dict[str, WeightMapping] = {}
    for key, target in _ROOT.items():
        if key.startswith("hc_head"):
            # mHC gates are float32 and not ``[out, in]`` projections: no transpose,
            # replicated, and the dtype must not follow the activation dtype.
            mappings[key] = WeightMapping(target_path=target, sharding=_REPLICATED, transpose=False)
        elif key in ("embed.weight", "head.weight"):
            mappings[key] = WeightMapping(target_path=target, sharding=_COLUMN, transpose=False)
        else:
            mappings[key] = WeightMapping(target_path=target, sharding=_REPLICATED, transpose=False)

    for layer, families in enumerate(_layer_families(config)):
        prefix = f"layers.{layer}."
        target = f"model.layers.{layer}."
        for name, suffix in _EVERY_LAYER.items():
            sharding = _COLUMN if suffix == "mlp.gate.weight" else _REPLICATED
            transpose = suffix == "mlp.gate.weight"
            mappings[prefix + name] = WeightMapping(
                target_path=target + suffix, sharding=sharding, transpose=transpose
            )
        for stem, suffix in _EVERY_LAYER_FP8.items():
            # wo_b reduces G*R back to hidden, so it is the row-parallel one.
            sharding = _ROW if stem == "attn.wo_b" else _COLUMN
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
                target_path=target + "mlp.gate.correction_bias",
                sharding=_REPLICATED,
                transpose=False,
            )

        if families["compressor"]:
            for name, suffix in _COMPRESSOR.items():
                mappings[prefix + name] = WeightMapping(
                    target_path=target + suffix,
                    sharding=_REPLICATED if "norm" in name or "ape" in name else _COLUMN,
                    transpose=not ("norm" in name or "ape" in name),
                )
        if families["indexer"]:
            for name, suffix in _INDEXER.items():
                mappings[prefix + name] = WeightMapping(
                    target_path=target + suffix,
                    sharding=_REPLICATED if "norm" in name or "ape" in name else _COLUMN,
                    transpose=not ("norm" in name or "ape" in name),
                )
            for stem, suffix in _INDEXER_FP8.items():
                _add_fp8_linear(mappings, prefix + stem, target + suffix, sharding=_COLUMN)
    return mappings
