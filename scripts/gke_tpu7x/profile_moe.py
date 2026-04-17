"""
Profile fused_moe kernels with xprof custom call tracing.

Sets LIBTPU_INIT_ARGS for LLO utilization visibility, then runs
both sglang-jax and tpu-inference fused_moe benchmarks with profiling.

Traces are saved to /tmp/moe_profile/{sglang_jax,tpu_inference}/.

Usage (via launcher on TPU pod):
    python3 -u /tmp/launcher.py scripts/gke_tpu7x/profile_moe.py \
        --num-experts 8 --top-k 2 --hidden-size 2048 --intermediate-size 512 \
        --num-tokens 128 --iters 3 --warmup-iters 1
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from functools import partial

from flax import nnx

# Set LIBTPU_INIT_ARGS for custom call profiling BEFORE importing JAX
_xla_flags = (
    "--xla_enable_custom_call_region_trace=true "
    "--xla_xprof_register_llo_debug_info=true"
)
existing = os.environ.get("LIBTPU_INIT_ARGS", "")
if existing:
    os.environ["LIBTPU_INIT_ARGS"] = existing + " " + _xla_flags
else:
    os.environ["LIBTPU_INIT_ARGS"] = _xla_flags

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

proc = jax.process_index()
is_main = proc == 0


def log(msg):
    if is_main:
        print(msg, flush=True)


PROFILE_ROOT = "/tmp/moe_profile"


def profile_sglang_jax(args):
    """Profile sglang-jax fused_moe."""
    from benchmark.moe.utils import (
        MoEBenchmarkCase,
        MoEImbalanceSimulator,
        build_mesh,
        make_moe_cases,
        prepare_fused_moe_inputs,
        select_cases,
    )
    from sgl_jax.srt.layers.moe import FusedEPMoE, TopK

    trace_dir = os.path.join(PROFILE_ROOT, "sglang_jax")
    os.makedirs(trace_dir, exist_ok=True)

    raw_cases = make_moe_cases(
        num_tokens=args.num_tokens,
        num_experts=args.num_experts,
        top_k=args.top_k,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        activation="silu",
        renormalize_topk_logits=True,
        name_prefix="profile_sglang",
    )
    cases = list(select_cases(raw_cases))
    cases = [c for c in cases if c.tp_size == 1]

    for case in cases:
        log(f"\n[sglang-jax profile] tokens={case.num_tokens}, ep_size={case.ep_size}")
        mesh = build_mesh(ep_size=case.ep_size, tp_size=case.tp_size)

        data = prepare_fused_moe_inputs(
            case, weight_dtype=jnp.bfloat16, mesh=mesh, include_weights=False
        )

        target_counts = MoEImbalanceSimulator.generate_counts(
            case.num_tokens, case.top_k, case.num_experts, mode="balanced"
        )
        custom_logits = MoEImbalanceSimulator.create_logits_from_counts(
            case.num_tokens, case.num_experts, case.top_k, target_counts
        )
        data["router_logits"] = jax.device_put(
            custom_logits, jax.sharding.NamedSharding(mesh, P("tensor", None))
        )

        with jax.set_mesh(mesh):
            fused_layer = FusedEPMoE(
                hidden_size=case.hidden_size,
                num_experts=case.num_experts,
                num_experts_per_tok=case.top_k,
                ep_size=case.ep_size,
                mesh=mesh,
                intermediate_dim=case.intermediate_size,
                weight_dtype=jnp.bfloat16,
                dtype=jnp.bfloat16,
                activation=case.activation,
                layer_id=0,
                renormalize_topk_logits=True,
            )

            topk_module = TopK(
                topk=case.top_k,
                renormalize=True,
                layer_id=0,
            )

            # Use nnx.split/merge pattern to avoid closing over non-local arrays
            moe_def, moe_state = nnx.split(fused_layer)
            moe_state_leaves, moe_state_def = jax.tree_util.tree_flatten(moe_state)

            topk_def, topk_state = nnx.split(topk_module)
            topk_state_leaves, topk_state_def = jax.tree_util.tree_flatten(topk_state)

            @partial(jax.jit, static_argnames=("moe_state_def", "topk_state_def"))
            def run_sglang(
                tokens,
                router_logits,
                *,
                moe_state_def,
                moe_state_leaves,
                topk_state_def,
                topk_state_leaves,
            ):
                moe_state = jax.tree_util.tree_unflatten(moe_state_def, moe_state_leaves)
                moe = nnx.merge(moe_def, moe_state)
                topk_state = jax.tree_util.tree_unflatten(topk_state_def, topk_state_leaves)
                topk = nnx.merge(topk_def, topk_state)
                topk_weights, topk_ids = topk(router_logits)
                return moe(tokens, topk_weights, topk_ids)

            run_kwargs = dict(
                moe_state_def=moe_state_def,
                moe_state_leaves=moe_state_leaves,
                topk_state_def=topk_state_def,
                topk_state_leaves=topk_state_leaves,
            )

            # Warmup
            for _ in range(args.warmup_iters):
                out = run_sglang(data["tokens"], data["router_logits"], **run_kwargs)
                jax.block_until_ready(out)
            log(f"  warmup done")

            # Profile
            case_trace_dir = os.path.join(
                trace_dir, f"nt{case.num_tokens}_ne{case.num_experts}"
            )
            with jax.profiler.trace(case_trace_dir):
                for i in range(args.iters):
                    with jax.profiler.StepTraceAnnotation("sglang_fused_moe", step_num=i):
                        out = run_sglang(data["tokens"], data["router_logits"], **run_kwargs)
                        jax.block_until_ready(out)
            log(f"  profile saved to {case_trace_dir}")


def profile_tpu_inference(args):
    """Profile tpu-inference fused_ep_moe."""
    from scripts.gke_tpu7x.tpu_inference_fused_moe.kernel import fused_ep_moe

    trace_dir = os.path.join(PROFILE_ROOT, "tpu_inference")
    os.makedirs(trace_dir, exist_ok=True)

    num_devices = len(jax.devices())
    ep_size = min(num_devices, args.num_experts)
    for ep in range(ep_size, 0, -1):
        if args.num_experts % ep == 0:
            ep_size = ep
            break

    devices = jax.devices()[:ep_size]
    mesh = Mesh(
        np.array(devices).reshape(1, ep_size),
        axis_names=("data", "model"),
    )
    ep_axis_name = "model"

    for num_tokens in args.num_tokens:
        if num_tokens % ep_size != 0:
            log(f"\n[SKIP] num_tokens={num_tokens} not divisible by ep_size={ep_size}")
            continue

        log(f"\n[tpu-inference profile] tokens={num_tokens}, ep_size={ep_size}")

        tokens_sharding = NamedSharding(mesh, P(ep_axis_name, None))
        w1_sharding = NamedSharding(mesh, P(ep_axis_name, None, None, None))
        w2_sharding = NamedSharding(mesh, P(ep_axis_name, None, None))
        logits_sharding = NamedSharding(mesh, P(ep_axis_name, None))

        tokens = jax.jit(
            lambda: jnp.zeros((num_tokens, args.hidden_size), dtype=jnp.bfloat16),
            out_shardings=tokens_sharding,
        )()
        w1 = jax.jit(
            lambda: jnp.zeros(
                (args.num_experts, 2, args.hidden_size, args.intermediate_size),
                dtype=jnp.bfloat16,
            ),
            out_shardings=w1_sharding,
        )()
        w2 = jax.jit(
            lambda: jnp.zeros(
                (args.num_experts, args.intermediate_size, args.hidden_size),
                dtype=jnp.bfloat16,
            ),
            out_shardings=w2_sharding,
        )()
        gating_output = jax.jit(
            lambda: jnp.zeros((num_tokens, args.num_experts), dtype=jnp.bfloat16),
            out_shardings=logits_sharding,
        )()

        @jax.jit
        def run_tpu_inf(tokens, w1, w2, gating_output):
            return fused_ep_moe(
                mesh=mesh,
                tokens=tokens,
                w1=w1,
                w2=w2,
                gating_output=gating_output,
                top_k=args.top_k,
                renormalize_topk_logits=True,
                act_fn="silu",
                scoring_fn="softmax",
                ep_axis_name=ep_axis_name,
            )

        # Warmup
        for _ in range(args.warmup_iters):
            out = run_tpu_inf(tokens, w1, w2, gating_output)
            jax.block_until_ready(out)
        log(f"  warmup done")

        # Profile
        case_trace_dir = os.path.join(
            trace_dir, f"nt{num_tokens}_ne{args.num_experts}"
        )
        with jax.profiler.trace(case_trace_dir):
            for i in range(args.iters):
                with jax.profiler.StepTraceAnnotation("tpu_inf_fused_moe", step_num=i):
                    out = run_tpu_inf(tokens, w1, w2, gating_output)
                    jax.block_until_ready(out)
        log(f"  profile saved to {case_trace_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Profile fused_moe kernels with xprof")
    parser.add_argument("--num-experts", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--intermediate-size", type=int, default=512)
    parser.add_argument("--num-tokens", type=int, nargs="+", default=[128])
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--warmup-iters", type=int, default=1)
    parser.add_argument(
        "--skip-sglang", action="store_true", help="Skip sglang-jax profiling"
    )
    parser.add_argument(
        "--skip-tpu-inference",
        action="store_true",
        help="Skip tpu-inference profiling",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    log(f"LIBTPU_INIT_ARGS={os.environ.get('LIBTPU_INIT_ARGS', '')}")
    log(f"Profile config: experts={args.num_experts}, top_k={args.top_k}, "
        f"hidden={args.hidden_size}, intermediate={args.intermediate_size}")
    log(f"Tokens: {args.num_tokens}, iters={args.iters}, warmup={args.warmup_iters}")
    log(f"Traces will be saved to {PROFILE_ROOT}/")

    if not args.skip_sglang:
        log("\n========== Profiling sglang-jax ==========")
        profile_sglang_jax(args)

    if not args.skip_tpu_inference:
        log("\n========== Profiling tpu-inference ==========")
        profile_tpu_inference(args)

    log(f"\n========== Done ==========")
    log(f"Traces saved to {PROFILE_ROOT}/")
    log(f"  sglang-jax:     {PROFILE_ROOT}/sglang_jax/")
    log(f"  tpu-inference:   {PROFILE_ROOT}/tpu_inference/")
