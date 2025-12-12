"""
Benchmark fused_moe kernel with grouped-GEMM-like MoE shapes.

Usage:
    python -m benchmark.moe.bench_fused_moe [--scenario random|balanced|imbalanced]
"""

from __future__ import annotations

import argparse
import functools
import time

import jax
import jax.numpy as jnp
import numpy as np

from benchmark.moe.utils import (
    MoEBenchmarkCase,
    multiple_iteration_timeit_from_trace,
    prepare_fused_moe_inputs,
    select_cases,
)
from sgl_jax.srt.kernels.fused_moe.v1.kernel import fused_ep_moe


def build_mesh(ep_size: int = 1):
    from sgl_jax.srt.utils.mesh_utils import create_device_mesh

    return create_device_mesh(
        ici_parallelism=[ep_size, 1],
        dcn_parallelism=[1, 1],
        mesh_axes=("tensor", "data"),
    )


def _round_block(value: int, cap: int, base: int) -> int:
    block = min(cap, value)
    block = block - (block % base)
    return max(base, block)


def choose_block_config(case: MoEBenchmarkCase, dtype: jnp.dtype) -> dict[str, int]:
    """Pick fused_moe block config that satisfies kernel constraints."""
    return {
        "bt": 64,
        "bf": 1024,
        "bd1": 2048,
        "bd2": 2048,
        "btc": 64,
        "bfc": 1024,
        "bd1c": 2048,
        "bd2c": 2048,
    }


def run_all(
    scenario: str,
    iters: int,
    dtype: jnp.dtype = jnp.bfloat16,
) -> None:
    cases = list(select_cases())
    print(f"Running fused_moe benchmarks with scenario='{scenario}', dtype={dtype}")
    for case in cases:
        print(
            f"\n[case={case.name}] tokens={case.num_tokens}, experts={case.num_experts}, "
            f"top_k={case.top_k}, hidden={case.hidden_size}, intermediate={case.intermediate_size}"
        )

        mesh = build_mesh(ep_size=case.ep_size)
        data = prepare_fused_moe_inputs(case, scenario, dtype=dtype)
        block_cfg = choose_block_config(case, dtype)
        print(f"  fused_moe blocks -> {block_cfg}")

        fused = functools.partial(
            fused_ep_moe,
            mesh=mesh,
            top_k=case.top_k,
            renormalize_topk_logits=case.renormalize_topk_logits,
            act_fn=case.activation,
            bt=block_cfg["bt"],
            bf=block_cfg["bf"],
            bd1=block_cfg["bd1"],
            bd2=block_cfg["bd2"],
            btc=block_cfg["btc"],
            bfc=block_cfg["bfc"],
            bd1c=block_cfg["bd1c"],
            bd2c=block_cfg["bd2c"],
            ep_axis_name="tensor",
        )

        @jax.jit
        def run(tokens, w1, w2, router_logits):
            return fused(
                tokens=tokens,
                w1=w1,
                w2=w2,
                gating_output=router_logits,
            )

        # warmup
        start = time.perf_counter()
        jax.block_until_ready(run(data["tokens"], data["w1"], data["w2"], data["router_logits"]))
        elapsed_ms = (time.perf_counter() - start) * 1000
        print(f"warmup in {elapsed_ms} ms")

        times = multiple_iteration_timeit_from_trace(
            compute_func=lambda: run(data["tokens"], data["w1"], data["w2"], data["router_logits"]),
            data_generator=lambda: (),
            task=f"fused_moe_{case.name}",
            tries=iters,
        )
        mean_ms = float(np.mean(times)) if times else float("nan")
        print(f"  fused_moe: {mean_ms:.3f} ms (trace) | samples={times}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark fused_moe.")
    parser.add_argument(
        "--scenario",
        choices=["random", "balanced", "imbalanced"],
        default="random",
        help="Router logits distribution pattern.",
    )
    parser.add_argument("--iters", type=int, default=3, help="Number of benchmark iterations.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_all(args.scenario, args.iters)
