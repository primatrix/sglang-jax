"""
Compare EPMoE vs FusedEPMoE kernel performance side-by-side.

Runs both MoE backends on identical benchmark cases (same tokens, experts,
routing distribution, quantization) and prints a comparison table with
latency and speedup ratios.

Key insight on imbalance sensitivity:
  - EPMoE communication is fixed (replicate all tokens + psum reduce),
    independent of routing distribution. It serves as a stable baseline.
  - FusedEPMoE all-to-all DMA and pipeline efficiency vary with imbalance.
    Sweeping imbalance modes quantifies its degradation under hotspot routing.

Usage:
    python -m benchmark.moe.bench_moe_compare \\
        --num-experts 256 --top-k 8 --hidden-size 2048 --intermediate-size 512

    # FP8 quantization:
    python -m benchmark.moe.bench_moe_compare --weight-dtype float8_e4m3fn

    # EPLB (redundant experts):
    python -m benchmark.moe.bench_moe_compare --ep-num-redundant-experts 16

    # Imbalance sweep:
    python -m benchmark.moe.bench_moe_compare --imbalance-mode hotspot --hotspot-ratio 0.5

    # EPMoE with TP (mixed ep+tp):
    python -m benchmark.moe.bench_moe_compare --epmoe-tp-size 2
"""

from __future__ import annotations

import argparse
import dataclasses
import faulthandler
import math
import sys
import traceback

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.experimental.compilation_cache import compilation_cache as _compilation_cache
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from benchmark.moe.utils import (
    DEFAULT_NUM_TOKENS,
    MoEBenchmarkCase,
    MoEImbalanceSimulator,
    build_mesh,
    make_moe_cases,
)
from benchmark.utils import multiple_iteration_timeit_from_trace
from sgl_jax.srt.configs.quantization_config import QuantizationConfig
from sgl_jax.srt.eplb.expert_location import (
    ExpertLocationMetadata,
    set_global_expert_location_metadata,
)
from sgl_jax.srt.layers.moe import EPMoE, FusedEPMoE, TopK

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class ComparisonResult:
    case_name: str
    num_tokens: int
    epmoe_ep_size: int
    epmoe_tp_size: int
    fused_ep_size: int
    epmoe_ms: float
    fused_ms: float
    speedup: float  # epmoe_ms / fused_ms


# ---------------------------------------------------------------------------
# EPLB helpers
# ---------------------------------------------------------------------------


def setup_eplb(
    num_experts: int,
    ep_num_redundant_experts: int,
) -> ExpertLocationMetadata | None:
    """Set up trivial EPLB metadata with redundant expert replicas."""
    if ep_num_redundant_experts <= 0:
        return None

    num_physical = num_experts + ep_num_redundant_experts
    num_layers = 1  # benchmark uses a single layer

    # physical_to_logical_map: shape [num_layers, num_physical]
    physical_to_logical_map = np.tile(np.arange(num_physical), (num_layers, 1)) % num_experts

    # Compute logical_to_all_physical_map by inverting the above.
    max_replicas = int(np.max(np.bincount(physical_to_logical_map[0], minlength=num_experts)))
    logical_to_all_physical_map = np.full(
        (num_layers, num_experts, max_replicas), -1, dtype=np.int32
    )
    for layer in range(num_layers):
        counts = np.zeros(num_experts, dtype=np.int32)
        for phys_id in range(num_physical):
            log_id = physical_to_logical_map[layer, phys_id]
            logical_to_all_physical_map[layer, log_id, counts[log_id]] = phys_id
            counts[log_id] += 1

    logical_to_all_physical_map_num_valid = np.sum(
        logical_to_all_physical_map != -1, axis=2
    ).astype(np.int32)
    logical_to_rank_dispatch_physical_map = logical_to_all_physical_map[:, :, 0]

    metadata = ExpertLocationMetadata(
        ep_dispatch_algorithm="static",
        logical_to_rank_dispatch_physical_map=logical_to_rank_dispatch_physical_map,
        logical_to_all_physical_map=logical_to_all_physical_map,
        logical_to_all_physical_map_num_valid=logical_to_all_physical_map_num_valid,
        physical_to_logical_map=physical_to_logical_map,
        num_physical_experts=num_physical,
    )
    set_global_expert_location_metadata(metadata)
    return metadata


def cleanup_eplb() -> None:
    set_global_expert_location_metadata(None)


# ---------------------------------------------------------------------------
# Per-backend benchmark runners
# ---------------------------------------------------------------------------


def run_epmoe_benchmark(
    case: MoEBenchmarkCase,
    mesh: jax.sharding.Mesh,
    tokens: jax.Array,
    router_logits: jax.Array,
    *,
    quantization_config: QuantizationConfig | None,
    use_grouped_topk: bool,
    warmup_iters: int,
    iters: int,
    eplb_metadata: ExpertLocationMetadata | None,
) -> float:
    """Benchmark EPMoE and return mean latency in ms."""
    with jax.set_mesh(mesh):
        topk_layer = TopK(
            topk=case.top_k,
            renormalize=case.renormalize_topk_logits,
            num_expert_group=case.num_expert_group if use_grouped_topk else 0,
            topk_group=case.topk_group if use_grouped_topk else 0,
            routed_scaling_factor=case.routed_scaling_factor,
            layer_id=0,
        )
        ep_moe_layer = EPMoE(
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
            quantization_config=quantization_config,
        )
        if quantization_config is not None:
            ep_moe_layer.quantize_weights()

        topk_def, topk_state = nnx.split(topk_layer)
        topk_state_leaves, topk_state_def = jax.tree_util.tree_flatten(topk_state)
        moe_def, moe_state = nnx.split(ep_moe_layer)
        moe_state_leaves, moe_state_def = jax.tree_util.tree_flatten(moe_state)

        @jax.jit(static_argnames=("topk_state_def", "moe_state_def"))
        def ep_moe_fn(
            hidden_states,
            router_logits,
            *,
            topk_state_def,
            topk_state_leaves,
            moe_state_def,
            moe_state_leaves,
        ):
            topk_state = jax.tree_util.tree_unflatten(topk_state_def, topk_state_leaves)
            topk = nnx.merge(topk_def, topk_state)
            moe_state = jax.tree_util.tree_unflatten(moe_state_def, moe_state_leaves)
            moe = nnx.merge(moe_def, moe_state)
            topk_weights, topk_ids = topk(router_logits)
            return moe(hidden_states, topk_weights, topk_ids)

        times = multiple_iteration_timeit_from_trace(
            compute_func=lambda: ep_moe_fn(
                tokens,
                router_logits,
                topk_state_def=topk_state_def,
                topk_state_leaves=topk_state_leaves,
                moe_state_def=moe_state_def,
                moe_state_leaves=moe_state_leaves,
            ),
            data_generator=lambda: (),
            task=f"epmoe_{case.name}",
            tries=iters,
            warmup=warmup_iters,
        )
        if len(times) > 1:
            times = times[1:]
        return float(np.mean(times)) if times else float("nan")


def run_fused_benchmark(
    case: MoEBenchmarkCase,
    mesh: jax.sharding.Mesh,
    tokens: jax.Array,
    router_logits: jax.Array,
    *,
    quantization_config: QuantizationConfig | None,
    use_grouped_topk: bool,
    use_shared_expert: bool,
    warmup_iters: int,
    iters: int,
    eplb_metadata: ExpertLocationMetadata | None,
) -> float:
    """Benchmark FusedEPMoE and return mean latency in ms."""
    with jax.set_mesh(mesh):
        topk_layer = TopK(
            topk=case.top_k,
            renormalize=case.renormalize_topk_logits,
            num_expert_group=case.num_expert_group if use_grouped_topk else 0,
            topk_group=case.topk_group if use_grouped_topk else 0,
            routed_scaling_factor=case.routed_scaling_factor,
            layer_id=0,
        )
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
            renormalize_topk_logits=case.renormalize_topk_logits,
            use_grouped_topk=use_grouped_topk,
            num_groups=case.num_expert_group if use_grouped_topk else 1,
            top_k_groups=case.topk_group if use_grouped_topk else 1,
            num_shared_experts=1 if use_shared_expert else 0,
            moe_shared_expert_intermediate_size=(
                case.intermediate_size if use_shared_expert else None
            ),
            quantization_config=quantization_config,
        )
        if quantization_config is not None:
            fused_layer.quantize_weights()

        topk_def, topk_state = nnx.split(topk_layer)
        topk_state_leaves, topk_state_def = jax.tree_util.tree_flatten(topk_state)
        moe_def, moe_state = nnx.split(fused_layer)
        moe_state_leaves, moe_state_def = jax.tree_util.tree_flatten(moe_state)

        @jax.jit(static_argnames=("topk_state_def", "moe_state_def"))
        def fused_fn(
            hidden_states,
            router_logits,
            *,
            topk_state_def,
            topk_state_leaves,
            moe_state_def,
            moe_state_leaves,
        ):
            topk_state = jax.tree_util.tree_unflatten(topk_state_def, topk_state_leaves)
            topk = nnx.merge(topk_def, topk_state)
            moe_state = jax.tree_util.tree_unflatten(moe_state_def, moe_state_leaves)
            moe = nnx.merge(moe_def, moe_state)
            topk_weights, topk_ids = topk(router_logits)
            return moe(hidden_states, topk_weights, topk_ids)

        times = multiple_iteration_timeit_from_trace(
            compute_func=lambda: fused_fn(
                tokens,
                router_logits,
                topk_state_def=topk_state_def,
                topk_state_leaves=topk_state_leaves,
                moe_state_def=moe_state_def,
                moe_state_leaves=moe_state_leaves,
            ),
            data_generator=lambda: (),
            task=f"fused_moe_{case.name}",
            tries=iters,
            warmup=warmup_iters,
        )
        if len(times) > 1:
            times = times[1:]
        return float(np.mean(times)) if times else float("nan")


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------


def print_comparison_table(
    results: list[ComparisonResult],
    *,
    num_experts: int,
    top_k: int,
    hidden_size: int,
    intermediate_size: int,
    weight_dtype: jnp.dtype,
    imbalance_mode: str,
    ep_num_redundant_experts: int,
    use_shared_expert: bool,
) -> None:
    if not results:
        print("No results to display.")
        return

    r0 = results[0]
    num_devices = r0.epmoe_ep_size * r0.epmoe_tp_size

    print()
    print("=" * 78)
    print("  EPMoE vs FusedEPMoE Comparison")
    print("=" * 78)
    print(
        f"  Config: experts={num_experts}, top_k={top_k}, "
        f"hidden={hidden_size}, intermediate={intermediate_size}"
    )
    epmoe_mesh_str = f"ep={r0.epmoe_ep_size} tp={r0.epmoe_tp_size}"
    fused_mesh_str = f"ep={r0.fused_ep_size} tp=1"
    print(
        f"  EPMoE mesh: {epmoe_mesh_str} | FusedEPMoE mesh: {fused_mesh_str} ({num_devices} devices)"
    )
    eplb_str = f"{ep_num_redundant_experts} redundant" if ep_num_redundant_experts > 0 else "off"
    se_str = " +shared_expert" if use_shared_expert else ""
    print(
        f"  Weight: {jnp.dtype(weight_dtype).name} | Imbalance: {imbalance_mode} | EPLB: {eplb_str}{se_str}"
    )
    print()

    # Table header
    hdr = f"{'num_tokens':>10} | {'EPMoE (ms)':>11} | {'Fused (ms)':>11} | {'Speedup':>8} | {'Winner':>6}"
    sep = "-" * len(hdr)
    print(hdr)
    print(sep)

    valid_speedups: list[float] = []
    for r in results:
        if math.isfinite(r.epmoe_ms) and math.isfinite(r.fused_ms) and r.fused_ms > 0:
            speedup_str = f"{r.speedup:.2f}x"
            winner = "Fused" if r.speedup > 1.0 else "EPMoE"
            valid_speedups.append(r.speedup)
        else:
            speedup_str = "N/A"
            winner = "N/A"

        epmoe_str = f"{r.epmoe_ms:.3f}" if math.isfinite(r.epmoe_ms) else "FAIL"
        fused_str = f"{r.fused_ms:.3f}" if math.isfinite(r.fused_ms) else "FAIL"
        print(
            f"{r.num_tokens:>10} | {epmoe_str:>11} | {fused_str:>11} | {speedup_str:>8} | {winner:>6}"
        )

    # Summary
    if valid_speedups:
        geo_mean = math.exp(sum(math.log(s) for s in valid_speedups) / len(valid_speedups))
        max_s = max(valid_speedups)
        min_s = min(valid_speedups)
        max_case = results[valid_speedups.index(max_s)]
        min_case = results[valid_speedups.index(min_s)]
        overall = "Fused wins" if geo_mean > 1.0 else "EPMoE wins"

        print()
        print(f"  Geo-mean speedup: {geo_mean:.2f}x ({overall})")
        print(f"  Max: {max_s:.2f}x at num_tokens={max_case.num_tokens}")
        print(f"  Min: {min_s:.2f}x at num_tokens={min_case.num_tokens}")
    print()


# ---------------------------------------------------------------------------
# Main comparison orchestrator
# ---------------------------------------------------------------------------


def run_comparison(
    iters: int,
    weight_dtype: jnp.dtype = jnp.bfloat16,
    *,
    warmup_iters: int = 1,
    epmoe_tp_size: int = 1,
    num_tokens: list[int] | None = None,
    num_experts: int = 256,
    top_k: int = 8,
    hidden_size: int = 2048,
    intermediate_size: int = 512,
    activation: str = "silu",
    renormalize_topk_logits: bool = True,
    num_expert_group: int = 0,
    topk_group: int = 0,
    imbalance_mode: str = "balanced",
    alpha: float = 1.0,
    zipf_s: float = 1.1,
    hotspot_ratio: float = 0.5,
    hotspot_count: int = 1,
    zero_expert_count: int = 0,
    non_hotspot_alpha: float = 100.0,
    ep_num_redundant_experts: int = 0,
    use_shared_expert: bool = False,
) -> None:
    use_grouped_topk = bool(num_expert_group or topk_group)

    num_devices = len(jax.devices())
    if num_devices % epmoe_tp_size != 0:
        raise ValueError(
            f"num_devices ({num_devices}) must be divisible by epmoe_tp_size ({epmoe_tp_size})"
        )
    epmoe_ep_size = num_devices // epmoe_tp_size
    fused_ep_size = num_devices  # fused uses all devices as EP

    token_list = DEFAULT_NUM_TOKENS if num_tokens is None else num_tokens

    # Generate cases. We use select_cases to filter token counts that aren't
    # divisible by ep_size, but we override ep_size/tp_size ourselves below.
    raw_cases = make_moe_cases(
        num_tokens=token_list,
        num_experts=num_experts,
        top_k=top_k,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation=activation,
        renormalize_topk_logits=renormalize_topk_logits,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        name_prefix="compare",
    )

    # Filter cases: num_tokens must be divisible by both ep_sizes and num_experts
    # must be divisible by both ep_sizes.
    cases: list[MoEBenchmarkCase] = []
    for case in raw_cases:
        skip_reasons: list[str] = []
        if case.num_tokens % fused_ep_size != 0:
            skip_reasons.append(f"num_tokens({case.num_tokens}) % fused_ep({fused_ep_size}) != 0")
        if case.num_tokens % epmoe_ep_size != 0:
            skip_reasons.append(f"num_tokens({case.num_tokens}) % epmoe_ep({epmoe_ep_size}) != 0")
        if num_experts % fused_ep_size != 0:
            skip_reasons.append(f"num_experts({num_experts}) % fused_ep({fused_ep_size}) != 0")
        if num_experts % epmoe_ep_size != 0:
            skip_reasons.append(f"num_experts({num_experts}) % epmoe_ep({epmoe_ep_size}) != 0")
        if skip_reasons:
            print(f"skip [case={case.name}]: {'; '.join(skip_reasons)}")
            continue
        cases.append(case)

    if not cases:
        print("No runnable cases after filtering.")
        return

    # Quantization
    if weight_dtype == jnp.float8_e4m3fn:
        quantization_config = QuantizationConfig(
            moe_weight_dtype=weight_dtype,
            moe_activation_dtype=None,
        )
    else:
        quantization_config = None

    # EPLB
    eplb_metadata = setup_eplb(num_experts, ep_num_redundant_experts)

    print("Running EPMoE vs FusedEPMoE comparison")
    print(f"  devices={num_devices}, weight_dtype={jnp.dtype(weight_dtype).name}")
    print(f"  EPMoE: ep={epmoe_ep_size} tp={epmoe_tp_size} | FusedEPMoE: ep={fused_ep_size} tp=1")
    print(
        f"  shape: experts={num_experts}, top_k={top_k}, hidden={hidden_size}, "
        f"intermediate={intermediate_size}, activation={activation}"
    )
    print(f"  imbalance_mode={imbalance_mode}, grouped_topk={use_grouped_topk}")
    if ep_num_redundant_experts > 0:
        print(f"  EPLB: {ep_num_redundant_experts} redundant experts (trivial mapping)")
    if use_shared_expert:
        print("  shared_expert: enabled (fused only)")

    results: list[ComparisonResult] = []

    for case in cases:
        print(
            f"\n[case={case.name}] tokens={case.num_tokens}, experts={case.num_experts}, "
            f"top_k={case.top_k}, hidden={case.hidden_size}, intermediate={case.intermediate_size}"
        )

        # Build separate meshes
        fused_mesh = build_mesh(ep_size=fused_ep_size, tp_size=1)
        epmoe_mesh = build_mesh(ep_size=epmoe_ep_size, tp_size=epmoe_tp_size)

        # Generate imbalanced router logits (numpy, then shard per-backend)
        target_counts = MoEImbalanceSimulator.generate_counts(
            case.num_tokens,
            case.top_k,
            case.num_experts,
            mode=imbalance_mode,
            alpha=alpha,
            zipf_s=zipf_s,
            hotspot_ratio=hotspot_ratio,
            hotspot_count=hotspot_count,
            zero_expert_count=zero_expert_count,
            non_hotspot_alpha=non_hotspot_alpha,
        )
        custom_logits = MoEImbalanceSimulator.create_logits_from_counts(
            case.num_tokens, case.num_experts, case.top_k, target_counts
        )

        # Prepare tokens (unsharded; each backend will reshard as needed)
        tokens_np = jnp.zeros((case.num_tokens, case.hidden_size), dtype=jnp.bfloat16)

        # --- EPMoE ---
        epmoe_ms = float("nan")
        try:
            # Shard inputs for epmoe mesh
            epmoe_tokens = jax.device_put(tokens_np, NamedSharding(epmoe_mesh, P("tensor", None)))
            epmoe_logits = jax.device_put(
                custom_logits, NamedSharding(epmoe_mesh, P("tensor", None))
            )

            epmoe_case = dataclasses.replace(case, ep_size=epmoe_ep_size, tp_size=epmoe_tp_size)
            epmoe_ms = run_epmoe_benchmark(
                epmoe_case,
                epmoe_mesh,
                epmoe_tokens,
                epmoe_logits,
                quantization_config=quantization_config,
                use_grouped_topk=use_grouped_topk,
                warmup_iters=warmup_iters,
                iters=iters,
                eplb_metadata=eplb_metadata,
            )
            print(f"  EPMoE:      {epmoe_ms:.3f} ms")
        except Exception as e:
            print(f"  EPMoE:      FAIL ({type(e).__name__}: {e})")
            print(traceback.format_exc())

        # --- FusedEPMoE ---
        fused_ms = float("nan")
        try:
            # Shard inputs for fused mesh
            fused_tokens = jax.device_put(tokens_np, NamedSharding(fused_mesh, P("tensor", None)))
            fused_logits = jax.device_put(
                custom_logits, NamedSharding(fused_mesh, P("tensor", None))
            )

            fused_case = dataclasses.replace(case, ep_size=fused_ep_size, tp_size=1)
            fused_ms = run_fused_benchmark(
                fused_case,
                fused_mesh,
                fused_tokens,
                fused_logits,
                quantization_config=quantization_config,
                use_grouped_topk=use_grouped_topk,
                use_shared_expert=use_shared_expert,
                warmup_iters=warmup_iters,
                iters=iters,
                eplb_metadata=eplb_metadata,
            )
            print(f"  FusedEPMoE: {fused_ms:.3f} ms")
        except Exception as e:
            print(f"  FusedEPMoE: FAIL ({type(e).__name__}: {e})")
            print(traceback.format_exc())

        # Compute speedup
        speedup = (
            epmoe_ms / fused_ms if (math.isfinite(fused_ms) and fused_ms > 0) else float("nan")
        )

        results.append(
            ComparisonResult(
                case_name=case.name,
                num_tokens=case.num_tokens,
                epmoe_ep_size=epmoe_ep_size,
                epmoe_tp_size=epmoe_tp_size,
                fused_ep_size=fused_ep_size,
                epmoe_ms=epmoe_ms,
                fused_ms=fused_ms,
                speedup=speedup,
            )
        )

    # Cleanup
    cleanup_eplb()

    # Print summary table
    print_comparison_table(
        results,
        num_experts=num_experts,
        top_k=top_k,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        weight_dtype=weight_dtype,
        imbalance_mode=imbalance_mode,
        ep_num_redundant_experts=ep_num_redundant_experts,
        use_shared_expert=use_shared_expert,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare EPMoE vs FusedEPMoE kernel performance.")
    parser.add_argument("--iters", type=int, default=3, help="Benchmark iterations.")
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=1,
        help="Warmup iterations before profiling (per case per backend).",
    )
    parser.add_argument(
        "--weight-dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float8_e4m3fn"],
        help="Weight data type.",
    )
    parser.add_argument(
        "--num-tokens",
        type=int,
        nargs="+",
        default=None,
        help="Token counts to sweep (e.g. --num-tokens 128 512 4096).",
    )
    parser.add_argument("--num-experts", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--intermediate-size", type=int, default=512)
    parser.add_argument("--activation", type=str, default="silu")
    parser.add_argument(
        "--renormalize-topk-logits",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Renormalize top-k routing weights.",
    )
    parser.add_argument("--num-expert-group", type=int, default=0)
    parser.add_argument("--topk-group", type=int, default=0)
    parser.add_argument(
        "--epmoe-tp-size",
        type=int,
        default=1,
        help="TP size for EPMoE (default 1 = full EP). FusedEPMoE always uses full EP.",
    )

    # Imbalance simulation
    parser.add_argument(
        "--imbalance-mode",
        type=str,
        default="balanced",
        choices=["balanced", "dirichlet", "zipf", "hotspot", "sparse_hotspot"],
        help="Router imbalance distribution.",
    )
    parser.add_argument("--alpha", type=float, default=1.0, help="Dirichlet concentration.")
    parser.add_argument("--zipf-s", type=float, default=1.1, help="Zipf exponent.")
    parser.add_argument(
        "--hotspot-ratio", type=float, default=0.5, help="Fraction of tokens to hotspot experts."
    )
    parser.add_argument("--hotspot-count", type=int, default=1, help="Number of hotspot experts.")
    parser.add_argument(
        "--zero-expert-count", type=int, default=0, help="Experts with zero load (sparse_hotspot)."
    )
    parser.add_argument(
        "--non-hotspot-alpha",
        type=float,
        default=100.0,
        help="Dirichlet alpha for non-hotspot experts.",
    )

    # EPLB
    parser.add_argument(
        "--ep-num-redundant-experts",
        type=int,
        default=0,
        help="Number of redundant expert replicas for EPLB.",
    )

    # Features
    parser.add_argument(
        "--use-shared-expert",
        action="store_true",
        help="Enable shared expert in FusedEPMoE (EPMoE skips).",
    )
    parser.add_argument(
        "--compilation-cache-dir",
        type=str,
        default=None,
        help="JAX compilation cache directory.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    try:
        faulthandler.enable(file=sys.stdout, all_threads=True)
    except Exception:
        pass

    args = parse_args()

    DTYPE_MAP = {
        "bfloat16": jnp.bfloat16,
        "float8_e4m3fn": jnp.float8_e4m3fn,
    }
    weight_dtype = DTYPE_MAP[args.weight_dtype]

    if args.compilation_cache_dir:
        _compilation_cache.set_cache_dir(args.compilation_cache_dir)

    run_comparison(
        args.iters,
        weight_dtype=weight_dtype,
        warmup_iters=args.warmup_iters,
        epmoe_tp_size=args.epmoe_tp_size,
        num_tokens=args.num_tokens,
        num_experts=args.num_experts,
        top_k=args.top_k,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        activation=args.activation,
        renormalize_topk_logits=args.renormalize_topk_logits,
        num_expert_group=args.num_expert_group,
        topk_group=args.topk_group,
        imbalance_mode=args.imbalance_mode,
        alpha=args.alpha,
        zipf_s=args.zipf_s,
        hotspot_ratio=args.hotspot_ratio,
        hotspot_count=args.hotspot_count,
        zero_expert_count=args.zero_expert_count,
        non_hotspot_alpha=args.non_hotspot_alpha,
        ep_num_redundant_experts=args.ep_num_redundant_experts,
        use_shared_expert=args.use_shared_expert,
    )
