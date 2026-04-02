#!/usr/bin/env python3
"""Deep tuning: comprehensive FusedMoEBlockConfig sweep for v6e + MiMo-V2-Flash.

Systematically explores block config parameter space to find optimal configs
for each token count. Parameters swept (FP8 fixes bd1c=512, bfc=256):
  - bt: outer token tile (8, 16, 32, 64)
  - bf: intermediate dim tile (256, 512, 1024, 2048)
  - bd1: hidden dim tile for W1/W3 (512, 1024, 2048, 4096)
  - bd2: hidden dim tile for W2 (512, 1024, 2048, 4096)
  - bd2c: compute tile for W2 (256, 512, 1024, 2048, 4096)
  - btc: token compute tile (8, 16, 32)
  - bse: shared expert tile (128, 256, 512, 1024, 2048)
"""
import json
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
from sgl_jax.srt.layers.moe import TopK  # noqa: E402

# ── Model shape (MiMo-V2-Flash) ──
HIDDEN = 4096
INTER = 2048
NUM_EXPERTS = 256
TOP_K = 8
EP_SIZE = 16
ACTIVATION = "silu"
DTYPE = jnp.bfloat16
WEIGHT_DTYPE = jnp.float8_e4m3fn
ITERS = 3
WARMUP = 1

# ── Token counts to test ──
TOKEN_COUNTS = [128, 512, 1024, 2048, 4096, 8192]

# ── Sweep configs ──
# Organized by category. Each config is (name, FusedMoEBlockConfig).
# FP8 overrides: bd1c -> 512 (=256*t_packing), bfc -> 256 (=subc_quant_wsz)
# So we set bd1c=512, bfc=256 in all configs.

# Constraints for MiMo shape (hidden=4096, inter=2048, bfloat16):
#   bf: must divide 2048, multiple of 128, bf%256==0 → {256, 512, 1024, 2048}
#   bd1: must divide 4096, bd1%512==0 (FP8+bf16) → {512, 1024, 2048, 4096}
#   bd2: must divide 4096, multiple of 256 → {256, 512, 1024, 2048, 4096}
#   bd2c: must divide bd2, multiple of 256
#   bt: must divide local_num_tokens (clamped by effective_for)
#   btc: must divide bts (=bt by default), btc<=bts

BASE = dict(bt=32, bf=2048, bd1=4096, bd2=4096, btc=32, bfc=256, bd1c=512, bd2c=4096, bse=512)


def cfg(name, **overrides):
    """Create a named config from base with overrides."""
    d = {**BASE, **overrides}
    return (name, FusedMoEBlockConfig(**d))


CONFIGS = [
    # ── Baseline (known best) ──
    cfg("base_best"),
    # ── bt sweep (token tile size) ──
    cfg("bt8", bt=8, btc=8),
    cfg("bt16", bt=16, btc=16),
    cfg("bt64", bt=64, btc=32),
    cfg("bt64c64", bt=64, btc=64),
    cfg("bt128", bt=128, btc=32),
    cfg("bt128c64", bt=128, btc=64),
    cfg("bt128c128", bt=128, btc=128),
    # ── bf sweep (intermediate dim tile) ──
    cfg("bf256", bf=256),
    cfg("bf512", bf=512),
    cfg("bf1024", bf=1024),
    # bf=2048 is baseline
    # ── bd1 sweep (hidden dim for W1/W3) ──
    cfg("bd1_512", bd1=512),
    cfg("bd1_1024", bd1=1024),
    cfg("bd1_2048", bd1=2048),
    # bd1=4096 is baseline
    # ── bd2 + bd2c sweep (hidden dim for W2) ──
    cfg("bd2_1024", bd2=1024, bd2c=1024),
    cfg("bd2_2048", bd2=2048, bd2c=2048),
    cfg("bd2c_256", bd2c=256),
    cfg("bd2c_512", bd2c=512),
    cfg("bd2c_1024", bd2c=1024),
    cfg("bd2c_2048", bd2c=2048),
    # bd2=4096, bd2c=4096 is baseline
    # ── bse sweep (shared expert tile) ──
    cfg("bse128", bse=128),
    cfg("bse256", bse=256),
    cfg("bse1024", bse=1024),
    cfg("bse2048", bse=2048),
    # bse=512 is baseline
    # ── btc sweep with base bt=32 ──
    cfg("btc8", btc=8),
    cfg("btc16", btc=16),
    # btc=32 is baseline
    # ── Combined experiments ──
    # Large bt + large bf
    cfg("bt64_bf2048", bt=64, btc=64, bf=2048),
    cfg("bt128_bf2048", bt=128, btc=128, bf=2048),
    # Different bd2c with base bd2=4096
    cfg("bd2_4096_bd2c_512", bd2=4096, bd2c=512),
    # Smaller blocks all around
    cfg("small_all", bt=32, bf=1024, bd1=2048, bd2=2048, bd2c=2048, bse=512),
    # Previous known configs for comparison
    cfg("default_orig", bt=32, bf=512, bd1=1024, bd2=1024, btc=32, bd2c=1024, bse=512),
    cfg("large_bf_orig", bt=32, bf=2048, bd1=2048, bd2=2048, btc=32, bd2c=2048, bse=512),
]


def run_one_config(config_name, block_config, num_tokens, mesh, quantization_config):
    """Benchmark FusedEPMoE with a specific block config."""
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
            task=f"tune_{config_name}_nt{num_tokens}",
            tries=ITERS,
            warmup=WARMUP,
        )
        if len(times) > 1:
            times = times[1:]
        mean_ms = float(np.mean(times)) if times else float("nan")
        return mean_ms


def main():
    mesh = build_mesh(ep_size=EP_SIZE, tp_size=1)
    quantization_config = QuantizationConfig(
        moe_weight_dtype=WEIGHT_DTYPE,
        moe_activation_dtype=None,
    )

    # Store all results: {num_tokens: [(config_name, ms), ...]}
    all_results = {}

    total_configs = len(CONFIGS)
    total_tokens = len(TOKEN_COUNTS)
    total_runs = total_configs * total_tokens

    if proc == 0:
        print(f"\n{'#'*70}")
        print(
            f"  Deep Tuning: {total_configs} configs × {total_tokens} token counts = {total_runs} runs"
        )
        print(
            f"  MiMo-V2-Flash: {NUM_EXPERTS}E, top_k={TOP_K}, {HIDDEN}x{INTER}, ep={EP_SIZE}, FP8"
        )
        print(f"{'#'*70}")
        start_time = time.time()

    run_idx = 0
    for num_tokens in TOKEN_COUNTS:
        all_results[num_tokens] = []

        if proc == 0:
            print(f"\n{'='*70}")
            print(f"  Token count: {num_tokens} (local={num_tokens // EP_SIZE})")
            print(f"{'='*70}")

        for config_name, block_config in CONFIGS:
            run_idx += 1
            if proc == 0:
                print(
                    f"\n  [{run_idx}/{total_runs}] {config_name}: "
                    f"bt={block_config.bt} bf={block_config.bf} "
                    f"bd1={block_config.bd1} bd2={block_config.bd2} "
                    f"btc={block_config.btc} bd2c={block_config.bd2c} "
                    f"bse={block_config.bse}"
                )

            try:
                ms = run_one_config(
                    config_name, block_config, num_tokens, mesh, quantization_config
                )
                all_results[num_tokens].append((config_name, ms))
                if proc == 0:
                    print(f"    -> {ms:.3f} ms")
            except Exception as e:
                if proc == 0:
                    print(f"    -> FAILED: {e}")
                    traceback.print_exc()
                all_results[num_tokens].append((config_name, float("nan")))

    # ── Summary ──
    if proc == 0:
        elapsed = time.time() - start_time
        print(f"\n{'#'*70}")
        print(f"  DEEP TUNING COMPLETE — {elapsed/60:.1f} minutes")
        print(f"{'#'*70}")

        # Per-token-count summary sorted by performance
        for num_tokens in TOKEN_COUNTS:
            results = all_results[num_tokens]
            valid = [(n, ms) for n, ms in results if not np.isnan(ms)]
            valid.sort(key=lambda x: x[1])

            print(f"\n  num_tokens={num_tokens} (local={num_tokens // EP_SIZE})")
            print(f"  {'Rank':<5s} {'Config':<25s} | {'Time (ms)':>10s} | {'vs best':>8s}")
            print(f"  {'-'*5} {'-'*25}-+-{'-'*10}-+-{'-'*8}")

            best_ms = valid[0][1] if valid else 1.0
            for rank, (name, ms) in enumerate(valid, 1):
                ratio = ms / best_ms
                marker = " ★" if rank == 1 else ""
                print(f"  {rank:<5d} {name:<25s} | {ms:>10.3f} | {ratio:>7.2f}x{marker}")

        # Overall best per token count
        print(f"\n{'='*70}")
        print("  BEST CONFIG PER TOKEN COUNT")
        print(f"{'='*70}")
        print(f"  {'tokens':>8s} | {'local':>6s} | {'best config':<25s} | {'time (ms)':>10s}")
        print(f"  {'-'*8}-+-{'-'*6}-+-{'-'*25}-+-{'-'*10}")
        for num_tokens in TOKEN_COUNTS:
            results = all_results[num_tokens]
            valid = [(n, ms) for n, ms in results if not np.isnan(ms)]
            if valid:
                valid.sort(key=lambda x: x[1])
                name, ms = valid[0]
                print(f"  {num_tokens:>8d} | {num_tokens//EP_SIZE:>6d} | {name:<25s} | {ms:>10.3f}")
            else:
                print(
                    f"  {num_tokens:>8d} | {num_tokens//EP_SIZE:>6d} | {'ALL FAILED':<25s} | {'N/A':>10s}"
                )

        # Save results to JSON for later analysis
        json_results = {}
        for nt, res_list in all_results.items():
            json_results[str(nt)] = {name: ms for name, ms in res_list}

        results_file = "/tmp/deep_tune_results.json"
        with open(results_file, "w") as f:
            json.dump(json_results, f, indent=2)
        print(f"\n  Results saved to {results_file}")


if __name__ == "__main__":
    main()
