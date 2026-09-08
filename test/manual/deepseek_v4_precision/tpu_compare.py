"""Compare actual JAX modules to a portable SGLang GPU capture, on one TPU device."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P

from common import Capture, Checkpoint, compare
from sgl_jax.srt.layers.deepseek_v4_mhc import DeepseekV4MHC, collapse_head_reference
from sgl_jax.srt.models.deepseek_v4 import DeepseekV4MoE, DeepseekV4SharedMLP
from sgl_jax.srt.utils.quantization.mxfp4_fp8_loader import convert_mxfp4_pair_from_safetensors


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--reference", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    reference = Path(args.reference)
    ref_info = json.loads((reference / "run.json").read_text())
    arrays = json.loads((reference / "arrays.json").read_text())
    checkpoint = Checkpoint(args.model)
    assert checkpoint.identity == ref_info["checkpoint"], "checkpoint identity differs"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cap = Capture(out / "actual")
    mesh = Mesh(
        np.array(jax.devices()[:1]).reshape(1, 1),
        ("data", "tensor"),
        axis_types=(AxisType.Explicit,) * 2,
    )
    cfg = SimpleNamespace(**checkpoint.config)
    cfg.quantization_config = None
    cfg.expert_dtype = "bf16"
    cfg.ep_size = cfg.moe_dp_size = 1
    cfg.n_shared_experts = 0
    rows, errors = [], []

    def ref(name):
        import hashlib

        info = arrays[name]
        path = reference / info["file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == info["sha256"]
        return np.load(path, allow_pickle=False)

    def put(a, dtype=jnp.bfloat16):
        return jax.device_put(
            np.asarray(a, dtype=dtype), NamedSharding(mesh, P(*([None] * np.ndim(a))))
        )

    def assign(param, value):
        assert param.value.shape == value.shape, (param.value.shape, value.shape)
        param.value = jax.device_put(
            np.asarray(value, dtype=param.value.dtype),
            param.value.sharding,
        )
        param.value.block_until_ready()

    def record(name, value):
        cap.save(name, value)
        if name not in arrays:
            rows.append({"case": name, "error": "GPU reference missing"})
        else:
            row = {"case": name, **compare(np.asarray(value), ref(name))}
            rows.append(row)
            print("MODULE_METRIC", json.dumps(row), flush=True)
        (out / "metrics.json").write_text(json.dumps(rows, indent=2))

    def run(name, fn):
        start = time.monotonic()
        try:
            with jax.set_mesh(mesh):
                fn()
            print("TPU_CASE_COMPLETE", name, time.monotonic() - start, flush=True)
        except Exception as e:
            import traceback

            traceback.print_exc()
            errors.append({"case": name, "type": type(e).__name__, "message": str(e)})
        finally:
            (out / "errors.json").write_text(json.dumps(errors, indent=2))
            gc.collect()

    hidden, ids, residual, branch = (
        put(ref("input/" + name), dtype)
        for name, dtype in (
            ("hidden", jnp.bfloat16),
            ("token_ids", jnp.int32),
            ("residual", jnp.bfloat16),
            ("branch", jnp.bfloat16),
        )
    )

    def converted(stem):
        wp, _, _ = checkpoint.entry(stem + ".weight")
        sp, _, _ = checkpoint.entry(stem + ".scale")
        # Use the production TPU conversion, separate from GPU source decoder.
        result = convert_mxfp4_pair_from_safetensors(
            wp, stem + ".weight", stem + ".scale", scale_file=sp, strict=True
        )
        decoded = result.dequantize()
        expected = checkpoint.expert_reference(stem)
        if not np.array_equal(decoded, expected):
            raise ValueError(f"weight reconstruction mismatch: {stem}")
        return decoded

    def mlp_case(layer_id=0, expert=None):
        name = f"mlp/l{layer_id}" if expert is None else f"expert/l{layer_id}/e{expert}"
        stem = (
            f"layers.{layer_id}.ffn.shared_experts"
            if expert is None
            else f"layers.{layer_id}.ffn.experts.{expert}"
        )
        reader = checkpoint.block_fp8 if expert is None else converted
        module = DeepseekV4SharedMLP(
            cfg.hidden_size,
            cfg.moe_intermediate_size,
            mesh,
            jnp.bfloat16,
            cfg.swiglu_limit,
            quantized=False,
        )
        for source, target in (("w1", "gate_proj"), ("w3", "up_proj"), ("w2", "down_proj")):
            assign(getattr(module, target).weight, reader(stem + "." + source).T)
        graph, state = nnx.split(module)

        def compute(state, x):
            m = nnx.merge(graph, state)
            return m(x), jnp.concatenate((m.gate_proj(x)[0], m.up_proj(x)[0]), axis=-1)

        y, gu = jax.jit(compute)(state, hidden)
        record(name + "/output", y)
        record(name + "/gate_up", gu)

    run("mlp/l0", lambda: mlp_case())
    run("expert/l0/e0", lambda: mlp_case(expert=0))

    def mhc_case():
        module = DeepseekV4MHC(cfg, backend="pallas")
        pre = jax.jit(
            jax.shard_map(
                module.pre,
                mesh=mesh,
                in_specs=(P(), P(), P(), P()),
                out_specs=(P(), P(), P()),
                check_vma=False,
            )
        )
        post = jax.jit(
            jax.shard_map(
                module.post,
                mesh=mesh,
                in_specs=(P(), P(), P(), P()),
                out_specs=P(),
                check_vma=False,
            )
        )
        for family in ("attn", "ffn"):
            stem = "layers.0.hc_" + family
            fn, scale, base = [
                put(checkpoint.read(stem + "_" + k), jnp.float32) for k in ("fn", "scale", "base")
            ]
            x, gate, comb = pre(residual, fn, base, scale)
            record(f"mhc/{family}/pre", x.astype(jnp.bfloat16))
            record(f"mhc/{family}/post_gate", gate)
            record(f"mhc/{family}/comb", comb)
            # Isolate post: BOTH sides consume the same GPU gate/comb arrays.
            canonical_gate, canonical_comb = put(ref(f"mhc/{family}/post_gate"), jnp.float32), put(
                ref(f"mhc/{family}/comb"), jnp.float32
            )
            record(
                f"mhc/{family}/post",
                post(branch, residual, canonical_gate, canonical_comb).astype(jnp.bfloat16),
            )
        fn, scale, base = [
            put(checkpoint.read("hc_head_" + k), jnp.float32) for k in ("fn", "scale", "base")
        ]
        head = jax.jit(
            lambda x, f, s, b: collapse_head_reference(
                x, f, s, b, norm_eps=cfg.rms_norm_eps, hc_eps=cfg.hc_eps
            )
        )
        record("mhc/head", head(residual, fn, scale, base).astype(jnp.bfloat16))

    run("mhc", mhc_case)

    def moe_case(layer_id):
        module = DeepseekV4MoE(cfg, mesh, layer_id, dtype=jnp.bfloat16)
        assign(module.gate.kernel, checkpoint.read(f"layers.{layer_id}.ffn.gate.weight").T)
        if module.is_hash_layer:
            module.load_hash_table(checkpoint.read(f"layers.{layer_id}.ffn.gate.tid2eid"))
        else:
            assign(module.gate.bias, checkpoint.read(f"layers.{layer_id}.ffn.gate.bias"))
        for source, target in (("w1", "wi_0"), ("w3", "wi_1"), ("w2", "wo")):
            param = getattr(module.experts, target)
            stack = np.empty(param.value.shape, dtype=jnp.bfloat16)
            for expert in range(cfg.n_routed_experts):
                stem = f"layers.{layer_id}.ffn.experts.{expert}.{source}"
                stack[expert] = converted(stem).T
                if expert % 64 == 0:
                    print("TPU_EXPERT_LOAD", layer_id, source, expert, flush=True)
            assign(param, stack)
            del stack
        graph, state = nnx.split(module)

        def compute(state, x, tids):
            m = nnx.merge(graph, state)
            weights, selected = m.route(x, tids)
            result, _ = m(x, tids)
            logits = jnp.dot(x, m.gate.kernel.value, precision=jax.lax.Precision.HIGHEST)
            forced = jnp.broadcast_to(
                jnp.arange(cfg.num_experts_per_tok, dtype=jnp.int32),
                (x.shape[0], cfg.num_experts_per_tok),
            )
            forced_weights = jnp.broadcast_to(
                (jnp.arange(cfg.num_experts_per_tok) == 0).astype(jnp.float32), forced.shape
            )
            isolated = m.experts(x, forced_weights, forced)
            return result, selected, weights, logits, isolated

        y, selected, weights, logits, isolated = jax.jit(compute)(state, hidden, ids)
        name = f"moe/l{layer_id}"
        for suffix, value in (
            ("output", y),
            ("ids", selected),
            ("weights", weights),
            ("logits", logits),
            ("expert0", isolated),
        ):
            record(name + "/" + suffix, value)

    run("moe/l0", lambda: moe_case(0))
    run(f"moe/l{cfg.num_hash_layers}", lambda: moe_case(cfg.num_hash_layers))
    weights_gpu = json.loads((reference / "weights.json").read_text())
    checked = 0
    for key, digest in checkpoint.digests.items():
        if key in weights_gpu:
            assert weights_gpu[key] == digest, f"source payload differs: {key}"
            checked += 1
    summary = {
        "jax": jax.__version__,
        "devices": [str(d) for d in mesh.devices.flat],
        "tp": 1,
        "ep": 1,
        "weight_path": "source-exact-dequantized-BF16",
        "activation_quantization": False,
        "checkpoint": checkpoint.identity,
        "matched_weight_payloads": checked,
        "metrics_count": len(rows),
        "execution_errors": errors,
        "threshold_policy": "report only; operator reviews metrics",
    }
    (out / "run.json").write_text(json.dumps(summary, indent=2))
    print("TPU_COMPARISON_COMPLETE", json.dumps(summary), flush=True)
    if errors:
        raise RuntimeError("Some TPU cases failed; inspect errors.json")


if __name__ == "__main__":
    main()
