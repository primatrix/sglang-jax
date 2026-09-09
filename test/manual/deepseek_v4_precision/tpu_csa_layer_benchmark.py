"""Production CSA attention-module A/B benchmark, with real FP8 weights and synthetic inputs.

Run in separate source-pinned processes for baseline/candidate. This is a one-layer
performance/numerical comparison, not a full-model quality test. Pool cloning is
outside measurement; the production update path donates its cloned buffers.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from common import Checkpoint
from flax import nnx
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from sgl_jax.srt.layers.attention.deepseek_v4_backend import DeepseekV4AttentionBackend
from sgl_jax.srt.mem_cache.deepseek_v4.allocator import DeepseekV4TokenToKVPoolAllocator
from sgl_jax.srt.mem_cache.deepseek_v4.pool import (
    DeepseekV4CacheSpec,
    DeepseekV4TokenToKVPool,
)
from sgl_jax.srt.mem_cache.deepseek_v4.state import DeepseekV4CompressStatePool
from sgl_jax.srt.mem_cache.memory_pool import MemoryPools, ReqToTokenPool
from sgl_jax.srt.model_executor.compilation_manager import CompilationManager
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sgl_jax.srt.model_executor.model_runner import ModelRunner
from sgl_jax.srt.models.deepseek_v4 import DeepseekV4Attention, _rope_cache
from sgl_jax.srt.server_args import ServerArgs

CASES = {
    # live requests, padded requests, prefix per request, query per request, token bucket
    "short18": (1, 1, 0, 18, 128),
    "prefill2k": (1, 64, 0, 2048, 2048),
    "extend2k": (1, 64, 6144, 2048, 2048),
    "decode32": (32, 32, 8192, 1, 32),
    "decode64": (64, 64, 8192, 1, 64),
    "decode32_steady": (32, 32, 8224, 1, 32),
    "decode64_steady": (64, 64, 8224, 1, 64),
    "decode32_complete": (32, 32, 8227, 1, 32),
    "decode29_padded": (29, 32, 8227, 1, 32),
    "decode1_empty": (1, 1, 2, 1, 1),
    "decode1_first": (1, 1, 3, 1, 1),
}


def digest(array):
    return hashlib.sha256(np.asarray(array).tobytes()).hexdigest()


def load_attention(cp, cfg, mesh, checkpoint_layer):
    module = DeepseekV4Attention(cfg, mesh, 0, jnp.bfloat16)
    stem = f"layers.{checkpoint_layer}.attn"

    def assign(param, value):
        assert param.value.shape == value.shape, (param.value.shape, value.shape)
        param.value = jax.device_put(
            np.asarray(value, dtype=param.value.dtype), param.value.sharding
        )

    def linear(mod, name):
        w = cp.read(name + ".weight")
        scale = cp.read(name + ".scale")
        if cp.entry(name + ".scale")[2]["dtype"] == "F8_E8M0":
            assert not np.any(scale == 255)
            scale = np.ldexp(
                np.ones(scale.shape, np.float32), scale.astype(np.int16) - 127
            )
        assert scale.shape == (w.shape[0] // 128, w.shape[1] // 128)
        assign(mod.weight_q, w)
        assign(mod.weight_scale, np.repeat(scale, 128, axis=0).T[:, None, :])

    def compressor(mod, name):
        for field in ("wkv", "wgate", "ape"):
            assign(
                getattr(mod, field),
                cp.read(name + "." + field + ("" if field == "ape" else ".weight")),
            )
        assign(mod.norm.scale, cp.read(name + ".norm.weight"))

    for field in ("wq_a", "wq_b", "wkv", "wo_a", "wo_b"):
        linear(getattr(module, field), stem + "." + field)
    for field in ("q_norm", "kv_norm"):
        assign(getattr(module, field).scale, cp.read(stem + "." + field + ".weight"))
    assign(module.attn_sink, cp.read(stem + ".attn_sink"))
    compressor(module.compressor, stem + ".compressor")
    linear(module.indexer.wq_b, stem + ".indexer.wq_b")
    assign(module.indexer.weights_proj, cp.read(stem + ".indexer.weights_proj.weight"))
    compressor(module.indexer.compressor, stem + ".indexer.compressor")
    return nnx.split(module)


def resources(cfg, mesh, case, seed):
    bs, padded_bs, prefix, count, capacity = case
    end = prefix + count
    pool_size = max(4096, bs * ((end + 127) // 128) * 128 + 128)
    r = object.__new__(ModelRunner)
    r.mesh, r.tp_size, r.page_size = mesh, mesh.shape["tensor"], 128
    r.server_args = ServerArgs(
        model_path="benchmark",
        disable_overlap_schedule=True,
        disable_radix_cache=True,
        page_size=128,
        dp_size=1,
    )
    r.model_config = SimpleNamespace(
        context_len=16384, vocab_size=32, is_embedding=False, hf_config=cfg
    )
    r.attn_backend = DeepseekV4AttentionBackend(
        mesh=mesh, page_size=128, max_context_len=16384, config=cfg
    )
    spec = DeepseekV4CacheSpec.from_config(cfg)
    r.req_to_token_pool = ReqToTokenPool(64, 16384)
    r.token_to_kv_pool = DeepseekV4TokenToKVPool(pool_size, pool_size, 128, spec, mesh)
    r.memory_pools = MemoryPools(
        token_to_kv_pool=r.token_to_kv_pool,
        compressor_state_pool=DeepseekV4CompressStatePool(64, spec, mesh),
    )
    r.token_to_kv_pool_allocator = DeepseekV4TokenToKVPoolAllocator(r.token_to_kv_pool)
    r.bind_attention_resources()
    compiler = CompilationManager(r.server_args, 64, 2048, 1, r.tp_size, 128, 16384, 32)
    mode = ForwardMode.DECODE if count == 1 else ForwardMode.EXTEND
    batch = compiler._make_dummy_batch(
        padded_bs,
        capacity,
        mode,
        padded_bs * 16384,
        dp_size=1,
        per_dp_bs_size=padded_bs,
    )
    batch.seq_lens = np.zeros(padded_bs, np.int32)
    batch.req_pool_indices = np.full(padded_bs, 64, np.int32)
    batch.positions.fill(0)
    batch.out_cache_loc.fill(-1)
    if mode == ForwardMode.EXTEND:
        batch.extend_seq_lens = np.zeros(padded_bs, np.int32)
        batch.extend_prefix_lens = np.zeros(padded_bs, np.int32)
    owners = [SimpleNamespace(req_pool_idx=None) for _ in range(bs)]
    slots = np.asarray(r.req_to_token_pool.alloc(owners), np.int32)
    cursor = 0
    for i, slot in enumerate(slots):
        # Allocate the historical page ledger directly. Long-history values below
        # are seeded inputs, not claimed to come from a preceding model prefill.
        loc = r.token_to_kv_pool_allocator.alloc_extend(
            np.array([0]), np.array([end]), np.array([-1]), end, dp_rank=0
        )
        assert loc is not None
        r.req_to_token_pool.req_to_token[slot, :end] = loc
        batch.seq_lens[i], batch.req_pool_indices[i] = end, slot
        batch.positions[cursor : cursor + count] = np.arange(prefix, end)
        batch.out_cache_loc[cursor : cursor + count] = loc[prefix:end]
        if mode == ForwardMode.EXTEND:
            batch.extend_seq_lens[i], batch.extend_prefix_lens[i] = count, prefix
        cursor += count
    batch.real_bs, batch.real_bs_per_dp, batch.real_input_ids_len = bs, [bs], cursor
    # Identical nonzero BF16 cache tensors in both variants. Keep empty state on
    # fresh prefills; use finite synthetic continuation state for long histories.
    key = jax.random.key(seed)
    for pool_index, pool in enumerate(
        (r.token_to_kv_pool, r.memory_pools.compressor_state_pool)
    ):
        for family_index, (family, arrays) in enumerate(pool.buffers.items()):
            if pool_index == 1 and prefix == 0:
                continue
            pool.buffers[family] = tuple(
                jax.random.normal(
                    jax.random.fold_in(key, pool_index * 100 + family_index * 10 + i),
                    a.shape,
                    dtype=a.dtype,
                    out_sharding=a.sharding,
                )
                * 0.1
                for i, a in enumerate(arrays)
            )
    rng = np.random.default_rng(seed)
    hidden = np.zeros((capacity, cfg.hidden_size), dtype=jnp.bfloat16)
    hidden[:cursor] = rng.normal(size=(cursor, cfg.hidden_size)).astype(jnp.bfloat16)
    x = jax.device_put(hidden, NamedSharding(mesh, P("data", None)))
    metadata_start = time.perf_counter()
    r.attn_backend.forward_metadata = r.get_attention_metadata(batch)
    fb = ForwardBatch.init_new(batch, r)
    jax.block_until_ready(fb)
    metadata_ms = (time.perf_counter() - metadata_start) * 1000
    return r, fb, x, slots, metadata_ms


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    p.add_argument("--tp", type=int, default=8)
    p.add_argument("--repeats", type=int, default=7)
    p.add_argument("--checkpoint-layer", type=int, default=2)
    p.add_argument("--profile", action="store_true")
    args = p.parse_args()
    if args.tp < 1 or args.repeats < 1:
        p.error("--tp and --repeats must be positive")
    if jax.default_backend() != "tpu":
        raise RuntimeError("This benchmark requires a TPU; CPU timing is not accepted.")
    assert jax.device_count() >= args.tp
    args.out.mkdir(parents=True, exist_ok=True)
    cp = Checkpoint(args.model)
    assert cp.config["compress_ratios"][args.checkpoint_layer] == 4
    cfg = SimpleNamespace(**cp.config)
    cfg.quantization_config = SimpleNamespace(is_static_checkpoint=True)
    cfg.max_position_embeddings = 16384
    cfg.num_hidden_layers, cfg.compress_ratios = 1, [4]
    mesh = jax.sharding.Mesh(
        np.array(jax.devices()[: args.tp]).reshape(1, args.tp),
        ("data", "tensor"),
        axis_types=(jax.sharding.AxisType.Explicit,) * 2,
    )
    info = {
        "source_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "checkpoint": cp.identity,
        "checkpoint_layer": args.checkpoint_layer,
        "jax": jax.__version__,
        "devices": [str(d) for d in mesh.devices.flat],
        "inputs": "seeded synthetic hidden/cache; real static FP8 attention weights",
        "timing": "synchronized host wall; donated pool clone excluded; no server/scheduler",
        "cases": [],
    }
    with jax.set_mesh(mesh):
        graph, params = load_attention(cp, cfg, mesh, args.checkpoint_layer)
        rope = _rope_cache(cfg, 4)
        jax.block_until_ready((params, rope))
        (args.out / "weights.json").write_text(json.dumps(cp.digests, indent=2))
        for case_index, name in enumerate(args.cases):
            print("LAYER_CASE_START", name, flush=True)
            r, fb, x, _slots, metadata_ms = resources(
                cfg, mesh, CASES[name], 1700 + list(CASES).index(name)
            )
            pools = r.memory_pools
            jax.block_until_ready((x, pools))
            input_hashes = [digest(a) for a in jax.tree.leaves((x, pools))]

            def forward(state, hidden, batch, memory, rope_cache):
                mod = nnx.merge(graph, state)
                y, update = mod(hidden, batch, memory, rope_cache)
                packed = batch.attn_backend.pack_pool_updates(
                    {0: update}, memory.token_to_kv_pool, memory.compressor_state_pool
                )
                memory.replace_all(packed)
                return y, memory

            compiled = jax.jit(forward, donate_argnums=(3,))
            clone = jax.jit(
                lambda memory: jax.tree.map(lambda leaf: leaf.copy(), memory)
            )

            def fresh(source=pools, copy_pools=clone):
                result = copy_pools(source)
                jax.block_until_ready(result)
                return result

            memory = fresh()
            started = time.perf_counter()
            executable = compiled.lower(params, x, fb, memory, rope).compile()
            compile_seconds = time.perf_counter() - started
            y, updated = executable(params, x, fb, memory, rope)
            jax.block_until_ready((y, updated))
            live = CASES[name][0] * CASES[name][3]
            output = np.asarray(y)[:live].astype(np.float32)
            assert np.isfinite(output).all(), name
            np.save(args.out / f"{name}-output.npy", output)
            output_hashes = [digest(a) for a in jax.tree.leaves(updated)]
            # Preserve small continuation arrays for numerical attribution when
            # a graph change alters FP32 compiler fusion or reduction order.
            for family, arrays in updated.compressor_state_pool.buffers.items():
                for index, array in enumerate(arrays):
                    np.save(args.out / f"{name}-state-{family}-{index}.npy", np.asarray(array))
            samples = []
            del updated, y
            # Separate first execution and one additional warmup from samples.
            for repeat in range(args.repeats + 1):
                memory = fresh()
                started = time.perf_counter()
                result = executable(params, x, fb, memory, rope)
                jax.block_until_ready(result)
                elapsed = (time.perf_counter() - started) * 1000
                if repeat:
                    samples.append(elapsed)
                del result
            if args.profile:
                # Prepare donated inputs before tracing: profiles contain only
                # production layer invocations and their synchronization.
                prepared = [fresh() for _ in range(3)]
                with jax.profiler.trace(
                    str(args.out / name), create_perfetto_link=False
                ):
                    for step, memory in enumerate(prepared):
                        with jax.profiler.StepTraceAnnotation(
                            "csa_attention_layer", step_num=step
                        ):
                            result = executable(params, x, fb, memory, rope)
                            jax.block_until_ready(result)
                            del result
                del prepared
            row = {
                "name": name,
                "geometry": CASES[name],
                "metadata_ms": metadata_ms,
                "compressed_capacity": int(
                    fb.attn_backend.forward_metadata.read_tables[1].compressed_rows.size
                ),
                "decode_compressed_capacity": (
                    fb.attn_backend.forward_metadata.read_tables[
                        1
                    ].decode_page_indices.shape[1]
                    * r.page_size
                    // 4
                    if getattr(
                        fb.attn_backend.forward_metadata.read_tables[1],
                        "decode_page_indices",
                        None,
                    )
                    is not None
                    else None
                ),
                "compile_seconds": compile_seconds,
                "warm_ms": samples,
                "median_ms": float(np.median(samples)),
                "input_hashes": input_hashes,
                "updated_pool_hashes": output_hashes,
            }
            info["cases"].append(row)
            (args.out / "result.json").write_text(json.dumps(info, indent=2))
            print("LAYER_CASE_RESULT", json.dumps(row), flush=True)
            del r, fb, x, pools, memory, executable, compiled, clone, fresh
            jax.clear_caches()
            gc.collect()
    print("TPU_CSA_LAYER_BENCHMARK_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
