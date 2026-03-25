"""
Focused kernel-level benchmark for fused_ep_moe with timing and profiling.

Usage:
    python -m benchmark.moe.bench_fused_moe_kernel --iters 5
    python -m benchmark.moe.bench_fused_moe_kernel --profile --profile-dir ./profile_baseline

    LIBTPU_INIT_ARGS="--xla_enable_custom_call_region_trace=true --xla_xprof_register_llo_debug_info=true" \\
    python -m benchmark.moe.bench_fused_moe_kernel --profile --profile-dir ./profile_baseline
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import PartitionSpec as P

from benchmark.moe.utils import (
    MoEBenchmarkCase,
    MoEImbalanceSimulator,
    build_mesh,
    prepare_fused_moe_inputs,
)
from benchmark.utils import multiple_iteration_timeit_from_trace
from sgl_jax.srt.layers.moe import FusedEPMoE, TopK


def run_benchmark(
    *,
    num_tokens: int = 512,
    num_experts: int = 64,
    top_k: int = 8,
    hidden_size: int = 5120,
    intermediate_size: int = 2048,
    iters: int = 5,
    warmup_iters: int = 1,
    profile: bool = False,
    profile_dir: str = "profile_fused_moe_kernel",
    imbalance_mode: str = "balanced",
):
    ep_size = len(jax.devices())
    print(f"Fused MoE Kernel Benchmark")
    print(f"  tokens={num_tokens}, experts={num_experts}, top_k={top_k}")
    print(f"  hidden={hidden_size}, intermediate={intermediate_size}, ep_size={ep_size}")
    print(f"  iters={iters}, warmup={warmup_iters}, profile={profile}")

    case = MoEBenchmarkCase(
        name="kernel_bench",
        num_tokens=num_tokens,
        num_experts=num_experts,
        top_k=top_k,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        ep_size=ep_size,
        tp_size=1,
        activation="silu",
        renormalize_topk_logits=True,
    )

    mesh = build_mesh(ep_size=ep_size, tp_size=1)
    print(f"  mesh: {mesh.shape}")

    data = prepare_fused_moe_inputs(
        case, weight_dtype=jnp.bfloat16, mesh=mesh,
        include_weights=False, include_shared_expert=False,
    )

    # Generate balanced/imbalanced routing
    target_counts = MoEImbalanceSimulator.generate_counts(
        num_tokens, top_k, num_experts, mode=imbalance_mode,
    )
    custom_logits = MoEImbalanceSimulator.create_logits_from_counts(
        num_tokens, num_experts, top_k, target_counts,
    )
    data["router_logits"] = jax.device_put(
        custom_logits, jax.sharding.NamedSharding(mesh, P("tensor", None)),
    )

    # Build FusedEPMoE layer + TopK module (same pattern as bench_fused_moe.py)
    with jax.set_mesh(mesh):
        fused_layer = FusedEPMoE(
            hidden_size=hidden_size,
            num_experts=num_experts,
            num_experts_per_tok=top_k,
            ep_size=ep_size,
            mesh=mesh,
            intermediate_dim=intermediate_size,
            weight_dtype=jnp.bfloat16,
            dtype=jnp.bfloat16,
            activation="silu",
            layer_id=0,
            renormalize_topk_logits=True,
            use_grouped_topk=False,
            num_groups=1,
            top_k_groups=1,
            num_shared_experts=0,
        )

        topk_module = TopK(
            topk=top_k, renormalize=True,
            num_expert_group=0, topk_group=0,
        )

        moe_def, moe_state = nnx.split(fused_layer)
        moe_state_leaves, moe_state_def = jax.tree_util.tree_flatten(moe_state)
        topk_def, topk_state = nnx.split(topk_module)
        topk_state_leaves, topk_state_def = jax.tree_util.tree_flatten(topk_state)

        @partial(
            jax.jit,
            static_argnames=("moe_state_def", "topk_state_def"),
        )
        def run_fused(tokens, router_logits, *, moe_state_def, moe_state_leaves, topk_state_def, topk_state_leaves):
            moe_state = jax.tree_util.tree_unflatten(moe_state_def, moe_state_leaves)
            moe = nnx.merge(moe_def, moe_state)
            topk_state = jax.tree_util.tree_unflatten(topk_state_def, topk_state_leaves)
            topk = nnx.merge(topk_def, topk_state)
            topk_weights, topk_ids = topk(router_logits)
            return moe(tokens, topk_weights, topk_ids)

        def compute():
            return run_fused(
                data["tokens"], data["router_logits"],
                moe_state_def=moe_state_def,
                moe_state_leaves=moe_state_leaves,
                topk_state_def=topk_state_def,
                topk_state_leaves=topk_state_leaves,
            )

        task = "fused-moe-k_.*"

        if profile:
            os.makedirs(profile_dir, exist_ok=True)
            print(f"\nProfiling to: {profile_dir}")

            # Warmup
            for _ in range(warmup_iters):
                out = compute()
                jax.block_until_ready(out)
            print("Warmup done")

            with jax.profiler.trace(profile_dir):
                for step in range(iters):
                    with jax.profiler.StepTraceAnnotation(task, step_num=step):
                        out = compute()
                        jax.block_until_ready(out)
            print(f"Profile saved to: {profile_dir}")
        else:
            times = multiple_iteration_timeit_from_trace(
                compute_func=compute,
                data_generator=lambda: (),
                task=task,
                tries=iters,
                warmup=warmup_iters,
            )

            if len(times) > 1:
                times = times[1:]  # drop first (may include tracing overhead)
            mean_ms = float(np.mean(times)) if times else float("nan")
            min_ms = float(np.min(times)) if times else float("nan")
            max_ms = float(np.max(times)) if times else float("nan")

            print(f"\n{'='*60}")
            print(f"RESULTS: tokens={num_tokens}, experts={num_experts}, top_k={top_k}")
            print(f"  mean: {mean_ms:.3f} ms")
            print(f"  min:  {min_ms:.3f} ms")
            print(f"  max:  {max_ms:.3f} ms")
            print(f"  samples: {times}")
            print(f"{'='*60}")
            return mean_ms


def parse_args():
    parser = argparse.ArgumentParser(description="Fused MoE kernel benchmark")
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--warmup-iters", type=int, default=1)
    parser.add_argument("--num-tokens", type=int, default=512)
    parser.add_argument("--num-experts", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=5120)
    parser.add_argument("--intermediate-size", type=int, default=2048)
    parser.add_argument("--imbalance-mode", type=str, default="balanced",
                        choices=["balanced", "dirichlet", "zipf", "hotspot", "sparse_hotspot"])
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-dir", type=str, default="profile_fused_moe_kernel")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_benchmark(
        num_tokens=args.num_tokens,
        num_experts=args.num_experts,
        top_k=args.top_k,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        iters=args.iters,
        warmup_iters=args.warmup_iters,
        profile=args.profile,
        profile_dir=args.profile_dir,
        imbalance_mode=args.imbalance_mode,
    )
