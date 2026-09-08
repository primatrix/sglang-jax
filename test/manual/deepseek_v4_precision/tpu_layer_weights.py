"""Load a BF16 decoder baseline using production expert promotion as an oracle check."""

import numpy as np
import jax.numpy as jnp
from sgl_jax.srt.utils.quantization.mxfp4_fp8_loader import convert_mxfp4_pair_from_safetensors


def load_layer_weights(block, cp, cfg, layer, assign):
    stem = f"layers.{layer}"
    for family in ("attn", "ffn"):
        for part in ("fn", "base", "scale"):
            name = f"hc_{family}_{part}"
            assign(getattr(block, name), cp.read(stem + "." + name))
        assign(
            getattr(block, family + "_norm").scale, cp.read(stem + "." + family + "_norm.weight")
        )
    moe = block.mlp
    assign(moe.gate.kernel, cp.read(stem + ".ffn.gate.weight").T)
    if moe.is_hash_layer:
        moe.load_hash_table(cp.read(stem + ".ffn.gate.tid2eid"))
    else:
        assign(moe.gate.bias, cp.read(stem + ".ffn.gate.bias"))
    for source, target in (("w1", "gate_proj"), ("w3", "up_proj"), ("w2", "down_proj")):
        assign(
            getattr(moe.shared_experts, target).weight,
            cp.block_fp8(stem + ".ffn.shared_experts." + source).T,
        )
    for source, target in (("w1", "wi_0"), ("w3", "wi_1"), ("w2", "wo")):
        param = getattr(moe.experts, target)
        stack = np.empty(param.value.shape, dtype=jnp.bfloat16)
        for expert in range(cfg.n_routed_experts):
            prefix = stem + f".ffn.experts.{expert}.{source}"
            wp, _, _ = cp.entry(prefix + ".weight")
            sp, _, _ = cp.entry(prefix + ".scale")
            value = convert_mxfp4_pair_from_safetensors(
                wp, prefix + ".weight", prefix + ".scale", scale_file=sp, strict=True
            ).dequantize()
            assert np.array_equal(value, cp.expert_reference(prefix)), prefix
            stack[expert] = value.T
            if expert % 64 == 0:
                print("TPU_LAYER_EXPERT_LOAD", layer, source, expert, flush=True)
        assign(param, stack)
        param.value.block_until_ready()
        del stack
