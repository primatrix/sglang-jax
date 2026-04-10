#!/usr/bin/env python3
"""Round 2: combination configs based on deep tuning findings.

Best individual findings:
- btc=16: best at 512-8192 tokens (7-8% faster)
- bse=2048: best at 128 tokens, good at all sizes
- bd2c=2048: good at ≥1024 tokens

This script tests combinations of these winners.
"""
import os
import sys
import time

sys.path.insert(0, "/tmp/sglang-jax/python")
sys.path.insert(0, "/tmp/sglang-jax")
os.chdir("/tmp/sglang-jax")

import jax  # noqa: E402

jax.distributed.initialize()
proc = jax.process_index()
print(f"[Process {proc}] ready, {jax.device_count()} devices", flush=True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from flax import nnx  # noqa: E402

from benchmark.moe.utils import build_mesh, generate_router_logits  # noqa: E402
from benchmark.utils import multiple_iteration_timeit_from_trace  # noqa: E402
from sgl_jax.srt.configs.quantization_config import QuantizationConfig  # noqa: E402
from sgl_jax.srt.kernels.fused_moe.v1.kernel import FusedMoEBlockConfig  # noqa: E402
from sgl_jax.srt.layers.fused_moe import FusedEPMoE  # noqa: E402
from sgl_jax.srt.layers.moe import TopK  # noqa: E402

HIDDEN = 4096
INTER = 2048
NUM_EXPERTS = 256
TOP_K = 8
EP_SIZE = 16
ACTIVATION = "silu"
DTYPE = jnp.bfloat16
WEIGHT_DTYPE = jnp.float8_e4m3fn
ITERS = 5  # more iters for precision
WARMUP = 2

TOKEN_COUNTS = [128, 256, 512, 1024, 2048, 4096, 8192]

# FP8 overrides: bd1c=512, bfc=256
BASE = dict(bt=32, bf=2048, bd1=4096, bd2=4096, btc=32, bfc=256, bd1c=512, bd2c=4096, bse=512)


def cfg(name, **overrides):
    d = {**BASE, **overrides}
    return (name, FusedMoEBlockConfig(**d))


CONFIGS = [
    # Reference: deep tuning winner
    cfg("r1_btc16", btc=16),
    cfg("r1_bse2048", bse=2048),
    cfg(
        "r1_base",
    ),
    # Round 2 combinations
    cfg("r2_btc16_bse2048", btc=16, bse=2048),
    cfg("r2_btc16_bse128", btc=16, bse=128),
    cfg("r2_btc16_bd2c2048", btc=16, bd2c=2048),
    cfg("r2_btc16_bse2048_bd2c2048", btc=16, bse=2048, bd2c=2048),
    cfg("r2_btc16_bd2_2048", btc=16, bd2=2048, bd2c=2048),
    cfg("r2_btc16_bd1_2048", btc=16, bd1=2048),
    cfg("r2_btc16_bd1_2048_bd2_2048", btc=16, bd1=2048, bd2=2048, bd2c=2048),
    # Also try btc=8 with combinations
    cfg("r2_btc8_bse2048", btc=8, bse=2048),
]


def run_one_config(config_name, block_config, num_tokens, mesh, quantization_config):
    with jax.set_mesh(mesh):
        tokens = jnp.empty((num_tokens, HIDDEN), dtype=DTYPE)
        router_logits = generate_router_logits(
            num_tokens,
            NUM_EXPERTS,
            "balanced",
            num_experts_per_tok=TOP_K,
        ).astype(DTYPE)

        topk_layer = TopK(
            topk=TOP_K,
            renormalize=True,
            num_expert_group=0,
            topk_group=0,
            routed_scaling_factor=None,
        )
        fused_layer = FusedEPMoE(
            hidden_size=HIDDEN,
            num_experts=NUM_EXPERTS,
            num_experts_per_tok=TOP_K,
            ep_size=EP_SIZE,
            mesh=mesh,
            intermediate_dim=INTER,
            weight_dtype=DTYPE,
            dtype=DTYPE,
            activation=ACTIVATION,
            layer_id=0,
            renormalize_topk_logits=True,
            quantization_config=quantization_config,
        )
        fused_layer.quantize_weights()

        topk_def, topk_state = nnx.split(topk_layer)
        topk_leaves, topk_treedef = jax.tree_util.tree_flatten(topk_state)
        moe_def, moe_state = nnx.split(fused_layer)
        moe_leaves, moe_treedef = jax.tree_util.tree_flatten(moe_state)

        @jax.jit(static_argnames=("topk_treedef", "moe_treedef", "block_config"))
        def fused_fn(
            hidden_states,
            router_logits,
            *,
            topk_treedef,
            topk_leaves,
            moe_treedef,
            moe_leaves,
            block_config,
        ):
            topk_st = jax.tree_util.tree_unflatten(topk_treedef, topk_leaves)
            topk = nnx.merge(topk_def, topk_st)
            moe_st = jax.tree_util.tree_unflatten(moe_treedef, moe_leaves)
            moe = nnx.merge(moe_def, moe_st)
            topk_weights, topk_ids = topk(router_logits)
            return moe(hidden_states, topk_weights, topk_ids, block_config=block_config)

        times = multiple_iteration_timeit_from_trace(
            compute_func=lambda: fused_fn(
                tokens,
                router_logits,
                topk_treedef=topk_treedef,
                topk_leaves=topk_leaves,
                moe_treedef=moe_treedef,
                moe_leaves=moe_leaves,
                block_config=block_config,
            ),
            data_generator=lambda: (),
            task=f"r2_{config_name}_nt{num_tokens}",
            tries=ITERS,
            warmup=WARMUP,
        )
        if len(times) > 1:
            times = times[1:]
        return float(np.mean(times)) if times else float("nan")


def main():
    mesh = build_mesh(ep_size=EP_SIZE, tp_size=1)
    qconfig = QuantizationConfig(moe_weight_dtype=WEIGHT_DTYPE, moe_activation_dtype=None)

    all_results = {}
    total = len(CONFIGS) * len(TOKEN_COUNTS)
    run_idx = 0

    if proc == 0:
        print(
            f"\nRound 2 Tuning: {len(CONFIGS)} configs × {len(TOKEN_COUNTS)} tokens = {total} runs"
        )
        start = time.time()

    for nt in TOKEN_COUNTS:
        all_results[nt] = []
        if proc == 0:
            print(f"\n{'='*60}\n  num_tokens={nt} (local={nt//EP_SIZE})\n{'='*60}")

        for name, bc in CONFIGS:
            run_idx += 1
            if proc == 0:
                print(f"  [{run_idx}/{total}] {name}")
            try:
                ms = run_one_config(name, bc, nt, mesh, qconfig)
                all_results[nt].append((name, ms))
                if proc == 0:
                    print(f"    -> {ms:.3f} ms")
            except Exception as e:
                if proc == 0:
                    print(f"    -> FAILED: {e}")
                all_results[nt].append((name, float("nan")))

    if proc == 0:
        elapsed = time.time() - start
        print(f"\n{'#'*60}\n  ROUND 2 COMPLETE — {elapsed/60:.1f} minutes\n{'#'*60}")

        for nt in TOKEN_COUNTS:
            valid = [(n, ms) for n, ms in all_results[nt] if not np.isnan(ms)]
            valid.sort(key=lambda x: x[1])
            print(f"\n  num_tokens={nt} (local={nt//EP_SIZE})")
            for rank, (name, ms) in enumerate(valid, 1):
                ratio = ms / valid[0][1] if valid else 0
                marker = " ★" if rank == 1 else ""
                print(f"    {rank}. {name:<35s} {ms:.3f} ms  ({ratio:.2f}x){marker}")

        print("\n  BEST PER TOKEN COUNT:")
        for nt in TOKEN_COUNTS:
            valid = [(n, ms) for n, ms in all_results[nt] if not np.isnan(ms)]
            valid.sort(key=lambda x: x[1])
            if valid:
                print(f"    {nt:>5d} tokens: {valid[0][0]:<35s} {valid[0][1]:.3f} ms")


if __name__ == "__main__":
    main()
