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
    zero_crop,
)
from sgl_jax.srt.kernels.fused_moe.v1.kernel import fused_ep_moe, get_dtype_packing


def benchmark_fused_moe(
    tokens: jax.Array,
    w1: jax.Array,
    w2: jax.Array,
    router_logits: jax.Array,
    *,
    mesh,
    top_k: int,
    block_cfg: dict[str, int],
    renormalize_topk_logits: bool = True,
    warmup: int = 1,
    iters: int = 3,
) -> float:
    """Return average latency (ms) for fused_ep_moe."""

    fused = functools.partial(
        fused_ep_moe,
        mesh=mesh,
        top_k=top_k,
        renormalize_topk_logits=renormalize_topk_logits,
        act_fn="silu",
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

    for _ in range(warmup):
        jax.block_until_ready(run(tokens, w1, w2, router_logits))

    times = []
    for _ in range(iters):
        start = time.perf_counter()
        out = run(tokens, w1, w2, router_logits)
        jax.block_until_ready(out)
        times.append((time.perf_counter() - start) * 1000)

    return float(jnp.mean(jnp.array(times)))


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
    t_pack = get_dtype_packing(dtype)
    max_bt = min(128, case.num_tokens)
    bt_candidates = [b for b in range(max_bt, t_pack - 1, -t_pack) if case.num_tokens % b == 0]
    bt = bt_candidates[0] if bt_candidates else t_pack

    bf = _round_block(case.intermediate_size, cap=2048, base=256)
    bfc = min(256, bf)
    bd1 = _round_block(case.hidden_size, cap=1024, base=256)
    bd2 = _round_block(case.hidden_size, cap=1024, base=256)
    bd1c = min(256, bd1)
    bd2c = min(256, bd2)

    return {
        "bt": bt,
        "bf": bf,
        "bd1": bd1,
        "bd2": bd2,
        "btc": bt,
        "bfc": bfc,
        "bd1c": bd1c,
        "bd2c": bd2c,
    }


def run_all(
    scenario: str,
    warmup: int,
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
            out = fused(
                tokens=tokens,
                w1=w1,
                w2=w2,
                gating_output=router_logits,
            )
            return zero_crop(out)

        for _ in range(warmup):
            start = time.perf_counter()
            jax.block_until_ready(
                run(data["tokens"], data["w1"], data["w2"], data["router_logits"])
            )
            print(f"warmed up in {(time.perf_counter() - start) * 1000} ms")

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
    parser.add_argument("--warmup", type=int, default=1, help="Number of warmup iterations.")
    parser.add_argument("--iters", type=int, default=3, help="Number of benchmark iterations.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_all(args.scenario, args.warmup, args.iters)
