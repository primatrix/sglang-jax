#!/usr/bin/env python3
"""Block-128 quantization: EPMoE vs FusedEPMoE — Balanced load.

Both backends use weight_block_size=[128, 128] (FP8 block-128 quantization).
Tests EPMoE baseline against FusedEPMoE with default and tuned block configs.
"""
import os
import sys
import time
import traceback

sys.path.insert(0, "/tmp/sglang-jax/python")
sys.path.insert(0, "/tmp/sglang-jax")
os.chdir("/tmp/sglang-jax")

import jax  # noqa: E402

jax.distributed.initialize()
proc = jax.process_index()
ndev = jax.device_count()
print(f"[Process {proc}] ready, {ndev} devices", flush=True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from flax import nnx  # noqa: E402

from benchmark.moe.utils import build_mesh, generate_router_logits  # noqa: E402
from benchmark.utils import multiple_iteration_timeit_from_trace  # noqa: E402
from sgl_jax.srt.configs.quantization_config import QuantizationConfig  # noqa: E402
from sgl_jax.srt.kernels.fused_moe.v1.kernel import FusedMoEBlockConfig  # noqa: E402
from sgl_jax.srt.layers.fused_moe import FusedEPMoE  # noqa: E402
from sgl_jax.srt.layers.moe import EPMoE, TopK  # noqa: E402

# ── Model shape (MiMo-V2-Flash) ──
HIDDEN = 4096
INTER = 2048
NUM_EXPERTS = 256
TOP_K = 8
EP_SIZE = 16
ACTIVATION = "silu"
DTYPE = jnp.bfloat16
WEIGHT_DTYPE = jnp.float8_e4m3fn
WEIGHT_BLOCK_SIZE = (128, 128)
ITERS = 5
WARMUP = 2

# ── Token counts ──
TOKEN_COUNTS = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]

# ── Tuned FusedEPMoE block configs (from v6e tuning, bd1c/bfc overridden by effective_for) ──
BASE = dict(bt=32, bf=2048, bd1=4096, bd2=4096, btc=32, bfc=256, bd1c=512, bd2c=4096, bse=512)


def cfg(name, **overrides):
    d = {**BASE, **overrides}
    return (name, FusedMoEBlockConfig(**d))


FUSED_CONFIGS = [
    cfg("tuned_small", btc=16, bse=2048, bd2c=2048),
    cfg("tuned_medium", btc=16, bse=128),
    cfg("tuned_large", btc=16),
    cfg("base_btc32"),
]


def run_epmoe(num_tokens, mesh, router_logits, quantization_config):
    """Benchmark EPMoE and return mean latency in ms."""
    with jax.set_mesh(mesh):
        tokens = jnp.empty((num_tokens, HIDDEN), dtype=DTYPE)

        topk_layer = TopK(
            topk=TOP_K,
            renormalize=True,
            num_expert_group=0,
            topk_group=0,
            routed_scaling_factor=None,
        )
        ep_moe_layer = EPMoE(
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
            quantization_config=quantization_config,
        )
        ep_moe_layer.quantize_weights()

        topk_def, topk_state = nnx.split(topk_layer)
        topk_leaves, topk_treedef = jax.tree_util.tree_flatten(topk_state)
        moe_def, moe_state = nnx.split(ep_moe_layer)
        moe_leaves, moe_treedef = jax.tree_util.tree_flatten(moe_state)

        @jax.jit(static_argnames=("topk_treedef", "moe_treedef"))
        def ep_moe_fn(
            hidden_states,
            router_logits,
            *,
            topk_treedef,
            topk_leaves,
            moe_treedef,
            moe_leaves,
        ):
            topk_st = jax.tree_util.tree_unflatten(topk_treedef, topk_leaves)
            topk = nnx.merge(topk_def, topk_st)
            moe_st = jax.tree_util.tree_unflatten(moe_treedef, moe_leaves)
            moe = nnx.merge(moe_def, moe_st)
            topk_weights, topk_ids = topk(router_logits)
            return moe(hidden_states, topk_weights, topk_ids)

        times = multiple_iteration_timeit_from_trace(
            compute_func=lambda: ep_moe_fn(
                tokens,
                router_logits,
                topk_treedef=topk_treedef,
                topk_leaves=topk_leaves,
                moe_treedef=moe_treedef,
                moe_leaves=moe_leaves,
            ),
            data_generator=lambda: (),
            task=f"epmoe_b128_nt{num_tokens}",
            tries=ITERS,
            warmup=WARMUP,
        )
        if len(times) > 1:
            times = times[1:]
        return float(np.mean(times)) if times else float("nan")


def run_fused(config_name, block_config, num_tokens, mesh, router_logits, quantization_config):
    """Benchmark FusedEPMoE with a specific block config and return mean latency in ms."""
    with jax.set_mesh(mesh):
        tokens = jnp.empty((num_tokens, HIDDEN), dtype=DTYPE)

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
            task=f"fused_b128_{config_name}_nt{num_tokens}",
            tries=ITERS,
            warmup=WARMUP,
        )
        if len(times) > 1:
            times = times[1:]
        return float(np.mean(times)) if times else float("nan")


def main():
    mesh = build_mesh(ep_size=EP_SIZE, tp_size=1)
    qconfig = QuantizationConfig(
        moe_weight_dtype=WEIGHT_DTYPE,
        moe_activation_dtype=None,
        weight_block_size=WEIGHT_BLOCK_SIZE,
    )

    all_results = {}
    total_runs = len(TOKEN_COUNTS) * (1 + len(FUSED_CONFIGS))
    run_idx = 0

    if proc == 0:
        print(f"\n{'#'*70}")
        print("  Block-128 Quantization: EPMoE vs FusedEPMoE — Balanced")
        print(f"  weight_block_size={WEIGHT_BLOCK_SIZE}")
        print(
            f"  MiMo-V2-Flash: {NUM_EXPERTS}E, top_k={TOP_K}, "
            f"{HIDDEN}x{INTER}, ep={EP_SIZE}, FP8"
        )
        print(
            f"  {1 + len(FUSED_CONFIGS)} backends x {len(TOKEN_COUNTS)} tokens = {total_runs} runs"
        )
        print(f"  iters={ITERS}, warmup={WARMUP}")
        print(f"{'#'*70}")
        start_time = time.time()

    for num_tokens in TOKEN_COUNTS:
        all_results[num_tokens] = {}

        if proc == 0:
            print(f"\n{'='*60}")
            print(f"  num_tokens={num_tokens} (local={num_tokens // EP_SIZE})")
            print(f"{'='*60}")

        # Generate balanced router logits
        logits = generate_router_logits(
            num_tokens, NUM_EXPERTS, "balanced", num_experts_per_tok=TOP_K
        ).astype(DTYPE)

        # --- EPMoE baseline ---
        run_idx += 1
        if proc == 0:
            print(f"\n  [{run_idx}/{total_runs}] EPMoE (block-128 baseline)")
        try:
            ms = run_epmoe(num_tokens, mesh, logits, qconfig)
            all_results[num_tokens]["epmoe"] = ms
            if proc == 0:
                print(f"    -> {ms:.3f} ms")
        except Exception as e:
            all_results[num_tokens]["epmoe"] = float("nan")
            if proc == 0:
                print(f"    -> FAILED: {e}")
                traceback.print_exc()

        # --- FusedEPMoE configs ---
        for config_name, block_config in FUSED_CONFIGS:
            run_idx += 1
            if proc == 0:
                print(
                    f"\n  [{run_idx}/{total_runs}] FusedEPMoE: {config_name} "
                    f"(btc={block_config.btc} bse={block_config.bse} bd2c={block_config.bd2c})"
                )
            try:
                ms = run_fused(config_name, block_config, num_tokens, mesh, logits, qconfig)
                all_results[num_tokens][config_name] = ms
                if proc == 0:
                    print(f"    -> {ms:.3f} ms")
            except Exception as e:
                all_results[num_tokens][config_name] = float("nan")
                if proc == 0:
                    print(f"    -> FAILED: {e}")
                    traceback.print_exc()

    # ── Summary ──
    if proc == 0:
        elapsed = time.time() - start_time
        print(f"\n{'#'*70}")
        print(f"  COMPLETE — {elapsed / 60:.1f} minutes")
        print(f"{'#'*70}")

        config_names = ["epmoe"] + [name for name, _ in FUSED_CONFIGS]
        header = f"  {'tokens':>6s} | {'local':>5s}"
        for name in config_names:
            header += f" | {name:>15s}"
        header += " |   best fused vs EPMoE"

        print(f"\n{header}")
        print(
            f"  {'-'*6}-+-{'-'*5}"
            + "".join(f"-+-{'-'*15}" for _ in config_names)
            + "-+-"
            + "-" * 22
        )

        for num_tokens in TOKEN_COUNTS:
            res = all_results[num_tokens]
            epmoe_ms = res.get("epmoe", float("nan"))
            line = f"  {num_tokens:>6d} | {num_tokens // EP_SIZE:>5d}"

            best_fused_name = None
            best_fused_ms = float("inf")

            for name in config_names:
                ms = res.get(name, float("nan"))
                line += f" | {ms:>15.3f}" if not np.isnan(ms) else f" | {'FAIL':>15s}"
                if name != "epmoe" and not np.isnan(ms) and ms < best_fused_ms:
                    best_fused_ms = ms
                    best_fused_name = name

            if best_fused_name and not np.isnan(epmoe_ms):
                speedup = epmoe_ms / best_fused_ms
                winner = "Fused" if speedup > 1.0 else "EPMoE"
                line += f" | {speedup:.2f}x ({winner})"
            else:
                line += " | N/A"

            print(line)

        # Per-token best summary
        print("\n  BEST FUSED CONFIG PER TOKEN COUNT (vs EPMoE):")
        geo_speedups = []
        for num_tokens in TOKEN_COUNTS:
            res = all_results[num_tokens]
            epmoe_ms = res.get("epmoe", float("nan"))

            best_name = None
            best_ms = float("inf")
            for name, _ in FUSED_CONFIGS:
                ms = res.get(name, float("nan"))
                if not np.isnan(ms) and ms < best_ms:
                    best_ms = ms
                    best_name = name

            if best_name and not np.isnan(epmoe_ms):
                speedup = epmoe_ms / best_ms
                geo_speedups.append(speedup)
                winner = "Fused" if speedup > 1.0 else "EPMoE"
                print(
                    f"    {num_tokens:>5d} tokens: {best_name:<20s} "
                    f"{best_ms:.3f} ms  (EPMoE={epmoe_ms:.3f}, {speedup:.2f}x {winner})"
                )

        if geo_speedups:
            geo_mean = np.exp(np.mean(np.log(geo_speedups)))
            winner = "Fused" if geo_mean > 1.0 else "EPMoE"
            print(f"\n  Geo-mean speedup (tuned Fused / EPMoE): {geo_mean:.2f}x ({winner} wins)")


if __name__ == "__main__":
    main()
