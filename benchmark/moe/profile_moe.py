"""
Capture xprof traces for EPMoE optimization experiments on MiMoV2Flash MoE dimensions.

Experiments:
  1. baseline: indexed_gmm + gate-up fusion (73cbcece)
  2. tiling_256: custom tile_m=256 for GMM v2
  3. tiling_512: custom tile_m=512 for GMM v2
  4. segment_sum: segment_sum-based unpermute (replaces argsort+take+einsum)
  5. tiling_256_segment: tile_m=256 + segment_sum combined

Saves traces to PROFILE_DIR for xprof analysis.
"""

from __future__ import annotations

import os

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
PROFILE_DIR = os.environ.get("PROFILE_DIR", "/gcs/moe_profiles")


def make_tile_m_fn(target_tile_m: int):
    """Create a TileFn that uses auto-tiler for tile_k/tile_n but overrides tile_m."""

    def fn(lhs_dtype, rhs_dtype, dims, vmem_limit_bytes):
        tiles = calculate_tiling(lhs_dtype, rhs_dtype, dims, vmem_limit_bytes)
        actual_tile_m = min(target_tile_m, dims.size_m)
        return TileSizes(tile_m=actual_tile_m, tile_k=tiles.tile_k, tile_n=tiles.tile_n)

    return fn


# (tag, ep_size, tp_size, num_tokens, v2_tile_info, use_segment_sum)
PROFILE_CASES = [
    # Best config from last round: ep1_tp16 at 16k tokens
    # Baseline (auto-tiling tile_m=128, argsort+take+einsum unpermute)
    ("baseline_ep1_tp16_nt16384", 1, 16, 16384, None, False),
    # Tiling experiments: tile_m=256 (auto tile_k/tile_n preserved)
    ("tiling256_ep1_tp16_nt16384", 1, 16, 16384, make_tile_m_fn(256), False),
    # Tiling: tile_m=512
    ("tiling512_ep1_tp16_nt16384", 1, 16, 16384, make_tile_m_fn(512), False),
    # Segment-sum unpermute (eliminates argsort+take+reshape+einsum)
    ("segsum_ep1_tp16_nt16384", 1, 16, 16384, None, True),
    # Combined: tiling_256 + segment_sum
    ("tiling256_segsum_ep1_tp16_nt16384", 1, 16, 16384, make_tile_m_fn(256), True),
    # Also test decode scenario (1k tokens)
    ("baseline_ep1_tp16_nt1024", 1, 16, 1024, None, False),
    ("tiling256_ep1_tp16_nt1024", 1, 16, 1024, make_tile_m_fn(256), False),
    ("segsum_ep1_tp16_nt1024", 1, 16, 1024, None, True),
    # EP16 configs for comparison
    ("baseline_ep16_tp1_nt16384", 16, 1, 16384, None, False),
    ("tiling256_ep16_tp1_nt16384", 16, 1, 16384, make_tile_m_fn(256), False),
    ("segsum_ep16_tp1_nt16384", 16, 1, 16384, None, True),
]


def profile_case(
    ep_size: int,
    tp_size: int,
    num_tokens: int,
    trace_dir: str,
    v2_tile_info=None,
    use_segment_sum: bool = False,
) -> None:
    mesh = build_mesh(ep_size=ep_size, tp_size=tp_size)
    tokens = jnp.empty((num_tokens, HIDDEN_SIZE), dtype=jnp.bfloat16)
    router_logits = generate_router_logits(
        num_tokens,
        NUM_EXPERTS,
        "balanced",
        num_experts_per_tok=TOP_K,
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
            use_segment_sum_unpermute=use_segment_sum,
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

        # Warmup (compile)
        out = fn(tokens, router_logits, **kwargs)
        jax.block_until_ready(out)
        print("    warmup done")

        # Profile
        with jax.profiler.trace(trace_dir):
            for i in range(5):
                out = fn(tokens, router_logits, **kwargs)
                jax.block_until_ready(out)
        print(f"    trace saved to {trace_dir}")


def main():
    num_devices = len(jax.devices())
    print(f"MoE xprof profiling: {num_devices} x {jax.devices()[0].device_kind}")
    os.makedirs(PROFILE_DIR, exist_ok=True)

    for tag, ep, tp, nt, tile_info, seg_sum in PROFILE_CASES:
        if ep * tp != num_devices:
            print(f"\n[{tag}] SKIP: requires {ep*tp} devices, have {num_devices}")
            continue

        trace_dir = os.path.join(PROFILE_DIR, tag)
        opts = []
        if tile_info is not None:
            if callable(tile_info):
                opts.append("tile=custom_fn")
            else:
                opts.append(f"tile=({tile_info.tile_m},{tile_info.tile_k},{tile_info.tile_n})")
        if seg_sum:
            opts.append("segment_sum")
        opts_str = f" [{', '.join(opts)}]" if opts else ""
        print(f"\n[{tag}]{opts_str}")

        try:
            profile_case(ep, tp, nt, trace_dir, v2_tile_info=tile_info, use_segment_sum=seg_sum)
        except Exception as e:
            print(f"    FAILED: {e}")

    # List output
    print("\n=== Profiles saved ===")
    for d in sorted(os.listdir(PROFILE_DIR)):
        full = os.path.join(PROFILE_DIR, d)
        if os.path.isdir(full):
            files = []
            for root, _, fnames in os.walk(full):
                files.extend(fnames)
            total_mb = sum(
                os.path.getsize(os.path.join(root, f))
                for root, _, fnames in os.walk(full)
                for f in fnames
            ) / (1024 * 1024)
            print(f"  {d}: {len(files)} files, {total_mb:.1f} MB")

    print(f"\n=== Done. Traces saved to {PROFILE_DIR} ===")


if __name__ == "__main__":
    main()
