"""EPMoE GMM v2 tile_m sweep — wall-clock timing on TPU.

Sweeps tile_m ∈ {64, 128, 256, 512} across three scenarios:
  1. ep1_tp16, 16384 tokens (prefill)
  2. ep1_tp16, 1024 tokens  (decode)
  3. ep16_tp1, 16384 tokens  (EP mode)

Uses make_tile_m_fn() to override only tile_m while preserving
the auto-tiler's optimal tile_k/tile_n for each GEMM shape.

Results printed as a table with median step time and speedup vs baseline (tile_m=128).
"""

from __future__ import annotations

import os
import time

import jax
import jax.numpy as jnp
from flax import nnx

from benchmark.moe.utils import build_mesh, generate_router_logits
from sgl_jax.srt.kernels.gmm.megablox_gmm_kernel.gmm_v2 import (
    TileSizes,
    calculate_tiling,
)
from sgl_jax.srt.layers.moe import EPMoE, TopK

NUM_EXPERTS = 256
TOP_K = 8
HIDDEN_SIZE = 4096
INTERMEDIATE_SIZE = 2048

WARMUP = 3
BENCH_STEPS = 10

TILE_M_VALUES = [64, 128, 256, 512]

# (scenario_name, ep_size, tp_size, num_tokens)
SCENARIOS = [
    ("ep1_tp16_nt16384", 1, 16, 16384),
    ("ep1_tp16_nt1024", 1, 16, 1024),
    ("ep16_tp1_nt16384", 16, 1, 16384),
]


def make_tile_m_fn(target_tile_m: int):
    """Create a TileFn that uses auto-tiler for tile_k/tile_n but overrides tile_m."""

    def fn(lhs_dtype, rhs_dtype, dims, vmem_limit_bytes):
        tiles = calculate_tiling(lhs_dtype, rhs_dtype, dims, vmem_limit_bytes)
        actual_tile_m = min(target_tile_m, dims.size_m)
        return TileSizes(tile_m=actual_tile_m, tile_k=tiles.tile_k, tile_n=tiles.tile_n)

    return fn


def bench_config(
    ep_size: int,
    tp_size: int,
    num_tokens: int,
    tile_m: int | None,
) -> list[float]:
    """Run EPMoE with given tile_m and return per-step times in ms."""
    v2_tile_info = make_tile_m_fn(tile_m) if tile_m is not None else None

    mesh = build_mesh(ep_size=ep_size, tp_size=tp_size)
    tokens = jnp.empty((num_tokens, HIDDEN_SIZE), dtype=jnp.bfloat16)
    router_logits = generate_router_logits(
        num_tokens, NUM_EXPERTS, "balanced", num_experts_per_tok=TOP_K
    ).astype(jnp.bfloat16)

    with jax.set_mesh(mesh):
        topk_layer = TopK(topk=TOP_K, renormalize=True)
        moe_layer = EPMoE(
            hidden_size=HIDDEN_SIZE,
            num_experts=NUM_EXPERTS,
            num_experts_per_tok=TOP_K,
            ep_size=ep_size,
            mesh=mesh,
            intermediate_dim=INTERMEDIATE_SIZE,
            weight_dtype=jnp.bfloat16,
            dtype=jnp.bfloat16,
            activation="silu",
            layer_id=0,
            v2_tile_info=v2_tile_info,
        )

        topk_def, topk_state = nnx.split(topk_layer)
        topk_leaves, topk_treedef = jax.tree_util.tree_flatten(topk_state)
        moe_def, moe_state = nnx.split(moe_layer)
        moe_leaves, moe_treedef = jax.tree_util.tree_flatten(moe_state)

        @jax.jit(static_argnames=("topk_treedef", "moe_treedef"))
        def fn(hidden, logits, *, topk_treedef, topk_leaves, moe_treedef, moe_leaves):
            topk = nnx.merge(topk_def, jax.tree_util.tree_unflatten(topk_treedef, topk_leaves))
            moe = nnx.merge(moe_def, jax.tree_util.tree_unflatten(moe_treedef, moe_leaves))
            w, ids = topk(logits)
            return moe(hidden, w, ids)

        kwargs = dict(
            topk_treedef=topk_treedef,
            topk_leaves=topk_leaves,
            moe_treedef=moe_treedef,
            moe_leaves=moe_leaves,
        )

        # Warmup (includes JIT compile)
        out = fn(tokens, router_logits, **kwargs)
        jax.block_until_ready(out)
        for _ in range(WARMUP - 1):
            out = fn(tokens, router_logits, **kwargs)
            jax.block_until_ready(out)

        # Benchmark
        times = []
        for _ in range(BENCH_STEPS):
            t0 = time.perf_counter()
            out = fn(tokens, router_logits, **kwargs)
            jax.block_until_ready(out)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)

    return times


def main():
    num_devices = len(jax.devices())
    print("=" * 90)
    print("EPMoE GMM v2 tile_m Sweep")
    print("=" * 90)
    print(f"  JAX: {jax.__version__}, Devices: {num_devices} ({jax.default_backend()})")
    print(
        f"  Experts: {NUM_EXPERTS}, top_k: {TOP_K}, hidden: {HIDDEN_SIZE}, intermediate: {INTERMEDIATE_SIZE}"
    )
    print(f"  tile_m values: {TILE_M_VALUES}")
    print(f"  warmup: {WARMUP}, bench_steps: {BENCH_STEPS}")
    print("=" * 90)

    # Output directory for results summary
    output_dir = os.environ.get("PROFILE_DIR", "/tmp/tiling_sweep")
    os.makedirs(output_dir, exist_ok=True)
    result_file = os.path.join(output_dir, "results.txt")

    all_results = []

    for scenario_name, ep_size, tp_size, num_tokens in SCENARIOS:
        if ep_size * tp_size > num_devices:
            print(
                f"\n[SKIP] {scenario_name}: needs {ep_size * tp_size} devices, have {num_devices}"
            )
            continue

        print(f"\n{'─' * 90}")
        print(f"Scenario: {scenario_name} (ep={ep_size}, tp={tp_size}, tokens={num_tokens})")
        print(f"{'─' * 90}")
        print(
            f"  {'tile_m':<10} | {'median':>10} {'mean':>10} {'min':>10} {'max':>10} | {'vs_128':>10}"
        )
        print(f"  {'-' * 75}")

        baseline_median = None
        scenario_results = []

        for tile_m in TILE_M_VALUES:
            try:
                times = bench_config(ep_size, tp_size, num_tokens, tile_m)
                median = sorted(times)[len(times) // 2]
                mean = sum(times) / len(times)
                t_min = min(times)
                t_max = max(times)

                if tile_m == 128:
                    baseline_median = median

                speedup = ""
                if baseline_median is not None and baseline_median > 0:
                    ratio = (baseline_median - median) / baseline_median * 100
                    speedup = f"{ratio:+.2f}%"

                print(
                    f"  {tile_m:<10} | {median:>9.3f}ms {mean:>9.3f}ms {t_min:>9.3f}ms {t_max:>9.3f}ms | {speedup:>10}"
                )
                scenario_results.append((tile_m, median, mean, t_min, t_max))
                all_results.append((scenario_name, tile_m, median, mean, t_min, t_max))

            except Exception as e:
                print(f"  {tile_m:<10} | ERROR: {e}")

        # Best for this scenario
        if scenario_results:
            best = min(scenario_results, key=lambda x: x[1])
            print(f"  >>> Best tile_m={best[0]} ({best[1]:.3f}ms)")

    # Summary
    print("\n" + "=" * 90)
    print("SUMMARY — Best tile_m per scenario")
    print("=" * 90)
    scenarios_seen = {}
    for scenario_name, tile_m, median, mean, t_min, t_max in all_results:
        if scenario_name not in scenarios_seen or median < scenarios_seen[scenario_name][1]:
            scenarios_seen[scenario_name] = (tile_m, median)
    for scenario_name, (tile_m, median) in scenarios_seen.items():
        print(f"  {scenario_name:<30} → tile_m={tile_m:<4} ({median:.3f}ms)")

    # Write results to file
    with open(result_file, "w") as f:
        f.write("scenario,tile_m,median_ms,mean_ms,min_ms,max_ms\n")
        for scenario_name, tile_m, median, mean, t_min, t_max in all_results:
            f.write(f"{scenario_name},{tile_m},{median:.3f},{mean:.3f},{t_min:.3f},{t_max:.3f}\n")
    print(f"\nResults saved to {result_file}")


if __name__ == "__main__":
    main()
