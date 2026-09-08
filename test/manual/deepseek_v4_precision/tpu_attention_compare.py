"""Real TPU HCA/CSA module paths with source-weight baseline and native GPU reference."""

from __future__ import annotations

import argparse
import hashlib
import json
import traceback
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import NamedSharding, PartitionSpec as P
from common import Capture, Checkpoint, compare
from sgl_jax.srt.models.deepseek_v4 import DeepseekV4Attention, _rope_cache
from sgl_jax.srt.layers.attention.deepseek_v4_backend import DeepseekV4AttentionBackend
from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.mem_cache.deepseek_v4.pool import DeepseekV4CacheSpec
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode
from sgl_jax.test.model_executor.test_deepseek_v4_runtime import Harness


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--reference", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--schedule", choices=("whole128", "whole129", "split"))
    p.add_argument("--layer", type=int, choices=(2, 3))
    p.add_argument("--diagnostic-csa-token-rope", action="store_true")
    args = p.parse_args()
    refroot = Path(args.reference)
    refmeta = json.loads((refroot / "run.json").read_text())
    arrays = json.loads((refroot / "arrays.json").read_text())
    cp = Checkpoint(args.model)
    assert cp.identity == refmeta["checkpoint"]
    cfg = SimpleNamespace(**cp.config)
    cfg.quantization_config = None
    cfg.max_position_embeddings = 256
    cfg.num_hidden_layers = 4
    cfg.compress_ratios = cfg.compress_ratios[:4]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cap = Capture(out / "actual")
    rows, errors = [], []
    info = dict(
        jax=jax.__version__,
        devices=[str(d) for d in jax.devices()],
        checkpoint=cp.identity,
        reference=refmeta,
        weight_path="source-dequantized-BF16",
        kv_cache="BF16",
        diagnostic_csa_token_rope=args.diagnostic_csa_token_rope,
        cases=[],
    )

    def ref(name):
        item = arrays[name]
        path = refroot / item["file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
        return np.load(path, allow_pickle=False)

    hidden = ref("input/hidden")

    def assign(param, value):
        assert param.value.shape == value.shape, (param.value.shape, value.shape)
        param.value = jax.device_put(
            np.asarray(value, dtype=param.value.dtype), param.value.sharding
        )

    def compressor(module, stem):
        for field in ("wkv", "wgate", "ape"):
            assign(
                getattr(module, field),
                cp.read(stem + "." + field + ("" if field == "ape" else ".weight")),
            )
        assign(module.norm.scale, cp.read(stem + ".norm.weight"))

    def record(name, value):
        value = np.asarray(value)
        cap.save(name, value)
        if name in arrays:
            expected = ref(name)
            # HCA commits only the final live SWA span; expired physical rows are not comparable.
            if name.endswith("/swa_cache"):
                value = value[-cfg.sliding_window :]
                expected = expected[-cfg.sliding_window :]
            if value.size == expected.size:
                value = value.reshape(expected.shape)
            row = dict(case=name, **compare(value, expected))
            rows.append(row)
            print("ATTENTION_METRIC", json.dumps(row), flush=True)

    captures = {}
    from sgl_jax.srt.layers.attention.dsv4 import dispatch

    original_compress = dispatch.compress_chunk
    if args.diagnostic_csa_token_rope:

        def diagnostic_compress(*a, **kw):
            # Diagnostic intervention only. Baseline calls retain production semantics.
            if kw["ratio"] == 4:
                kw["boundary_compressed_pos"] = kw["boundary_compressed_pos"] * 4
            return original_compress(*a, **kw)

        dispatch.compress_chunk = diagnostic_compress
    original_backend = DeepseekV4AttentionBackend.__call__

    def backend_call(self, q, k, v, *a, **kw):
        captures["q"] = q
        captures["kv_before_cache"] = k
        result = original_backend(self, q, k, v, *a, **kw)
        captures["raw_attention"] = result[0]
        return result

    original_linear = LinearBase.__call__

    def linear_call(self, x, *a, **kw):
        result = original_linear(self, x, *a, **kw)
        captures[self.name] = result[0]
        if self.name == "wo_b":
            captures["wo_a"] = x
        return result

    DeepseekV4AttentionBackend.__call__ = backend_call
    LinearBase.__call__ = linear_call
    try:
        for layer in (2, 3):
            if args.layer is not None and layer != args.layer:
                continue
            for schedule, lengths in (
                ("whole128", [128]),
                ("whole129", [129]),
                ("split", [63, 65, 1]),
            ):
                if args.schedule and schedule != args.schedule:
                    continue
                name = f"attention/l{layer}/{schedule}"
                try:
                    h = Harness(spec=DeepseekV4CacheSpec.from_config(cfg))
                    mesh = h.mesh
                    with jax.set_mesh(mesh):
                        h.runner.attn_backend = DeepseekV4AttentionBackend(
                            mesh=mesh, page_size=128, max_context_len=256, config=cfg
                        )
                        h.runner.bind_attention_resources()
                        module = DeepseekV4Attention(cfg, mesh, layer, jnp.bfloat16)
                        stem = f"layers.{layer}.attn"
                        for field in ("wq_a", "wq_b", "wkv", "wo_a", "wo_b"):
                            assign(
                                getattr(module, field).weight, cp.block_fp8(stem + "." + field).T
                            )
                        for field in ("q_norm", "kv_norm"):
                            assign(
                                getattr(module, field).scale,
                                cp.read(stem + "." + field + ".weight"),
                            )
                        assign(module.attn_sink, cp.read(stem + ".attn_sink"))
                        compressor(module.compressor, stem + ".compressor")
                        if module.indexer is not None:
                            assign(
                                module.indexer.wq_b.weight, cp.block_fp8(stem + ".indexer.wq_b").T
                            )
                            assign(
                                module.indexer.weights_proj,
                                cp.read(stem + ".indexer.weights_proj.weight"),
                            )
                            compressor(module.indexer.compressor, stem + ".indexer.compressor")
                        rope = _rope_cache(cfg, cfg.compress_ratios[layer])
                        graph, state = nnx.split(module)

                        def forward(state, x, fb, pools, rope):
                            captures.clear()
                            mod = nnx.merge(graph, state)
                            y, update = mod(x, fb, pools, rope)
                            return y, update, dict(captures)

                        compiled = jax.jit(forward)
                        start = 0
                        for step, count in enumerate(lengths):
                            mode = (
                                ForwardMode.DECODE if count == 1 and start else ForwardMode.EXTEND
                            )
                            capacity = 2 if mode == ForwardMode.DECODE else 256
                            batch = h.batch([count], mode, capacity=capacity)
                            fb = h.forward_batch(batch)
                            x = np.zeros(
                                (batch.positions.size, cfg.hidden_size), dtype=jnp.bfloat16
                            )
                            x[:count] = hidden[start : start + count]
                            x = jax.device_put(x, NamedSharding(mesh, P("data", None)))
                            y, update, aux = compiled(state, x, fb, h.runner.memory_pools, rope)
                            jax.block_until_ready((y, update, aux))
                            prefix = f"{name}/step{step}"
                            record(prefix + "/output", np.asarray(y)[:count])
                            for field, value in aux.items():
                                record(prefix + "/" + field, np.asarray(value)[:count])
                            # Resolve logical cache rows through the same allocator ledger used by the backend.
                            end = start + count
                            locations = np.asarray(
                                h.runner.req_to_token_pool.req_to_token[h.slots[0], :end], np.int32
                            )
                            mapping = h.runner.token_to_kv_pool_allocator.full_to_swa_index_mapping
                            if isinstance(mapping, list):
                                mapping = mapping[0]
                            swa_rows = np.asarray(mapping)[locations]
                            swa = np.asarray(update["swa"]).reshape(-1, cfg.head_dim)[swa_rows]
                            record(prefix + "/swa_cache", swa[:, None, :])
                            ratio = cfg.compress_ratios[layer]
                            if end // ratio:
                                compressed_rows = (
                                    locations[np.arange(end // ratio) * ratio] // ratio
                                )
                                compressed_values = np.asarray(update[f"c{ratio}"]).reshape(
                                    -1, cfg.head_dim
                                )[compressed_rows]
                                record(prefix + "/compressed_cache", compressed_values[:, None, :])
                            packed = h.runner.attn_backend.pack_pool_updates(
                                {layer: update},
                                h.runner.memory_pools.token_to_kv_pool,
                                h.runner.memory_pools.compressor_state_pool,
                            )
                            h.runner.memory_pools.replace_all(packed)
                            start += count
                            print("TPU_ATTENTION_STEP_COMPLETE", prefix, flush=True)
                        info["cases"].append(
                            dict(
                                name=name,
                                chunks=lengths,
                                kernel=(
                                    "HCA Pallas"
                                    if cfg.compress_ratios[layer] == 128
                                    and h.runner.attn_backend.use_pallas_hca
                                    else "CSA XLA"
                                ),
                            )
                        )
                except Exception as e:
                    traceback.print_exc()
                    errors.append(dict(name=name, type=type(e).__name__, message=str(e)))
                finally:
                    (out / "metrics.json").write_text(json.dumps(rows, indent=2))
                    (out / "errors.json").write_text(json.dumps(errors, indent=2))
                    (out / "run.json").write_text(json.dumps(info, indent=2))
                    (out / "weights.json").write_text(json.dumps(cp.digests, indent=2))
    finally:
        dispatch.compress_chunk = original_compress
        DeepseekV4AttentionBackend.__call__ = original_backend
        LinearBase.__call__ = original_linear
    if errors:
        raise RuntimeError("TPU attention cases failed; inspect errors.json")


if __name__ == "__main__":
    main()
