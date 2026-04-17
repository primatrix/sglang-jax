"""
Tune tpu-inference's fused_ep_moe kernel block configs for Ring-1T model shape.

Generates VMEM-safe block config candidates, runs each on the TPU pod, and
reports the best config per token count.

Usage (via launcher on TPU pod):
    python3 -u /tmp/launcher.py scripts/gke_tpu7x/tune_tpu_inference_moe.py \
        --num-experts 64 --top-k 8 --hidden-size 8192 --intermediate-size 2048 \
        --num-tokens 64 128 256 512 1024
"""

from __future__ import annotations

import argparse
import itertools
import traceback
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from benchmark.utils import multiple_iteration_timeit_from_trace
from scripts.gke_tpu7x.tpu_inference_fused_moe.kernel import fused_ep_moe
from scripts.gke_tpu7x.tpu_inference_fused_moe.tuned_block_sizes import align_to

proc = jax.process_index()
is_main = proc == 0


def log(msg):
    if is_main:
        print(msg, flush=True)


def estimate_vmem_bytes(
    bt: int,
    bf: int,
    bd1: int,
    bd2: int,
    hidden_size: int,
    num_experts: int,
    top_k: int,
    num_devices: int,
    t_packing: int = 2,
) -> int:
    """Estimate VMEM usage for tpu-inference fused_ep_moe scratch buffers."""
    padded_num_experts = align_to(num_experts, 128)
    elem = 2  # bf16 = 2 bytes

    vmem = 0
    # a2a_s_x2_vmem: (2, bt*num_devices, t_packing, hidden_size//t_packing)
    vmem += 2 * bt * num_devices * t_packing * (hidden_size // t_packing) * elem
    # a2a_s_acc_x2_vmem: same shape
    vmem += 2 * bt * num_devices * t_packing * (hidden_size // t_packing) * elem
    # a2a_g_acc_vmem: (top_k, bt, t_packing, hidden_size//t_packing)
    vmem += top_k * bt * t_packing * (hidden_size // t_packing) * elem
    # b_gating_x2_vmem: (2, bt, padded_num_experts)
    vmem += 2 * bt * padded_num_experts * elem
    # b_output_x2_vmem: (2, bt, hidden_size)
    vmem += 2 * bt * hidden_size * elem
    # b_w1_x2_vmem: (2, t_packing, bd1//t_packing, bf)
    vmem += 2 * t_packing * (bd1 // t_packing) * bf * elem
    # b_w3_x2_vmem: same as w1
    vmem += 2 * t_packing * (bd1 // t_packing) * bf * elem
    # b_w2_x2_vmem: (2, t_packing, bf, bd2//t_packing)
    vmem += 2 * t_packing * bf * (bd2 // t_packing) * elem
    # b_acc_vmem: (2, bt*num_devices, 1, bf) in float32
    vmem += 2 * bt * num_devices * 1 * bf * 4
    return vmem


def generate_valid_configs(
    local_num_tokens: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    top_k: int,
    num_devices: int,
    t_packing: int = 2,
    vmem_budget: int = 58 * 1024 * 1024,  # 58 MB conservative (actual TPU v7 VMEM = 64 MB)
) -> list[dict]:
    """Generate VMEM-safe block configs for the given shape.

    Strategy: enumerate all valid (bt, bf, bd1, bd2) combos that fit in VMEM,
    then for each combo generate compute tile variants (btc, bfc, bd1c, bd2c).
    """
    # bt candidates: must divide local_num_tokens, be multiple of t_packing
    bt_candidates = [bt for bt in [2, 4, 8, 16, 32, 64, 128, 256]
                     if bt <= local_num_tokens
                     and local_num_tokens % bt == 0
                     and bt % t_packing == 0]

    # bf candidates: must divide intermediate_size, be multiple of 128
    bf_candidates = [v for v in [128, 256, 512, 1024, 2048]
                     if intermediate_size % v == 0 and v % 128 == 0]

    # bd candidates: must divide hidden_size, be multiple of t_packing*128=256
    bd_candidates = [v for v in [256, 512, 1024, 2048, 4096, 8192]
                     if hidden_size % v == 0 and v % (t_packing * 128) == 0]

    configs = []
    for bt, bf, bd1, bd2 in itertools.product(
        bt_candidates, bf_candidates, bd_candidates, bd_candidates
    ):
        vmem = estimate_vmem_bytes(
            bt, bf, bd1, bd2, hidden_size, num_experts, top_k, num_devices, t_packing
        )
        if vmem > vmem_budget:
            continue

        # Generate a few representative compute tile combos:
        # btc: divisors of bt
        btc_all = [v for v in [2, 4, 8, 16, 32, 64, 128] if v <= bt and bt % v == 0]
        # Pick max, half, and min btc
        btc_set = {max(btc_all), min(btc_all)}
        mid = btc_all[len(btc_all) // 2]
        btc_set.add(mid)

        # bd1c/bd2c: just use bd1/bd2 (full tile) — the kernel already loops internally
        # bfc: just use bf
        for btc in sorted(btc_set, reverse=True):
            configs.append({
                "bt": bt, "bf": bf, "bd1": bd1, "bd2": bd2,
                "btc": btc, "bfc": bf, "bd1c": bd1, "bd2c": bd2,
                "vmem_mb": vmem / (1024 * 1024),
            })

    return configs


def pareto_filter(configs: list[dict], max_configs: int = 40) -> list[dict]:
    """Select a diverse set of configs that covers the search space well.

    Strategy:
    1. Deduplicate by all 8 block params
    2. Group by (bt, bf, bd1, bd2) — only keep the best btc per group (= max btc)
    3. Ensure diversity across bt, bf, and bd values
    """
    # Deduplicate
    seen = set()
    unique = []
    for c in configs:
        key = (c["bt"], c["bf"], c["bd1"], c["bd2"],
               c["btc"], c["bfc"], c["bd1c"], c["bd2c"])
        if key not in seen:
            seen.add(key)
            unique.append(c)

    # Group by (bt, bf, bd1, bd2), keep top 2 btc variants per group
    groups: dict[tuple, list[dict]] = {}
    for c in unique:
        gkey = (c["bt"], c["bf"], c["bd1"], c["bd2"])
        groups.setdefault(gkey, []).append(c)

    representatives = []
    for gkey, group in groups.items():
        group.sort(key=lambda x: -x["btc"])  # prefer larger btc
        representatives.extend(group[:2])  # keep top 2 btc per (bt,bf,bd1,bd2)

    if len(representatives) <= max_configs:
        return representatives

    # Stratified sampling: ensure we cover different bt and bf values
    # Sort by bt desc, then bf desc, then bd1*bd2 desc
    representatives.sort(key=lambda x: (-x["bt"], -x["bf"], -x["bd1"] * x["bd2"]))

    # Ensure at least 2 configs per unique bt value
    by_bt: dict[int, list[dict]] = {}
    for c in representatives:
        by_bt.setdefault(c["bt"], []).append(c)

    selected = []
    # First pass: take top configs per bt
    per_bt = max(2, max_configs // len(by_bt))
    for bt_val in sorted(by_bt.keys(), reverse=True):
        selected.extend(by_bt[bt_val][:per_bt])

    # Deduplicate and trim
    seen2 = set()
    final = []
    for c in selected:
        key = (c["bt"], c["bf"], c["bd1"], c["bd2"],
               c["btc"], c["bfc"], c["bd1c"], c["bd2c"])
        if key not in seen2:
            seen2.add(key)
            final.append(c)

    return final[:max_configs]


def build_mesh(ep_size: int) -> Mesh:
    devices = jax.devices()[:ep_size]
    device_array = np.array(devices).reshape(1, ep_size)
    return Mesh(device_array, axis_names=("data", "model"))


def run_tune(
    num_tokens_list: list[int],
    num_experts: int,
    top_k: int,
    hidden_size: int,
    intermediate_size: int,
    act_fn: str,
    iters: int,
    warmup_iters: int,
    max_configs_per_token: int,
):
    num_devices = len(jax.devices())
    ep_size = min(num_devices, num_experts)
    for ep in range(ep_size, 0, -1):
        if num_experts % ep == 0:
            ep_size = ep
            break

    log(f"=== tpu-inference fused_ep_moe tuning ===")
    log(f"  devices={num_devices}, ep_size={ep_size}")
    log(f"  num_experts={num_experts}, top_k={top_k}, "
        f"hidden_size={hidden_size}, intermediate_size={intermediate_size}")
    log(f"  iters={iters}, warmup={warmup_iters}, max_configs={max_configs_per_token}")

    mesh = build_mesh(ep_size)
    ep_axis_name = "model"

    all_best: list[tuple[int, float, dict | None]] = []

    for num_tokens in num_tokens_list:
        if num_tokens % ep_size != 0:
            log(f"\n[SKIP] num_tokens={num_tokens} not divisible by ep_size={ep_size}")
            continue

        local_num_tokens = num_tokens // ep_size
        log(f"\n{'='*60}")
        log(f"[TUNING] num_tokens={num_tokens} (local={local_num_tokens})")

        # Generate valid configs
        configs = generate_valid_configs(
            local_num_tokens=local_num_tokens,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            top_k=top_k,
            num_devices=num_devices,
        )
        log(f"  {len(configs)} valid configs before filtering")

        configs = pareto_filter(configs, max_configs=max_configs_per_token)
        log(f"  {len(configs)} configs after Pareto filtering")

        if not configs:
            log(f"  WARNING: no valid configs found!")
            all_best.append((num_tokens, float("inf"), None))
            continue

        # Prepare inputs once
        tokens_sharding = NamedSharding(mesh, P(ep_axis_name, None))
        w1_sharding = NamedSharding(mesh, P(ep_axis_name, None, None, None))
        w2_sharding = NamedSharding(mesh, P(ep_axis_name, None, None))
        logits_sharding = NamedSharding(mesh, P(ep_axis_name, None))

        tokens = jax.jit(
            lambda: jnp.zeros((num_tokens, hidden_size), dtype=jnp.bfloat16),
            out_shardings=tokens_sharding,
        )()
        w1 = jax.jit(
            lambda: jnp.zeros((num_experts, 2, hidden_size, intermediate_size), dtype=jnp.bfloat16),
            out_shardings=w1_sharding,
        )()
        w2 = jax.jit(
            lambda: jnp.zeros((num_experts, intermediate_size, hidden_size), dtype=jnp.bfloat16),
            out_shardings=w2_sharding,
        )()
        gating_output = jax.jit(
            lambda: jnp.zeros((num_tokens, num_experts), dtype=jnp.bfloat16),
            out_shardings=logits_sharding,
        )()

        best_ms = float("inf")
        best_cfg = None

        for i, cfg in enumerate(configs):
            tag = (f"bt={cfg['bt']},bf={cfg['bf']},bd1={cfg['bd1']},bd2={cfg['bd2']},"
                   f"btc={cfg['btc']} (VMEM={cfg['vmem_mb']:.1f}MB)")
            log(f"  [{i+1}/{len(configs)}] {tag}")

            try:
                @partial(jax.jit)
                def compute(tokens, w1, w2, gating_output,
                            _bt=cfg["bt"], _bf=cfg["bf"],
                            _bd1=cfg["bd1"], _bd2=cfg["bd2"],
                            _btc=cfg["btc"], _bfc=cfg["bfc"],
                            _bd1c=cfg["bd1c"], _bd2c=cfg["bd2c"]):
                    return fused_ep_moe(
                        mesh=mesh,
                        tokens=tokens,
                        w1=w1,
                        w2=w2,
                        gating_output=gating_output,
                        top_k=top_k,
                        renormalize_topk_logits=True,
                        act_fn=act_fn,
                        scoring_fn="softmax",
                        ep_axis_name=ep_axis_name,
                        bt=_bt, bf=_bf, bd1=_bd1, bd2=_bd2,
                        btc=_btc, bfc=_bfc, bd1c=_bd1c, bd2c=_bd2c,
                    )

                times = multiple_iteration_timeit_from_trace(
                    compute_func=lambda: compute(tokens, w1, w2, gating_output),
                    data_generator=lambda: (),
                    task="tpu-inference-tune",
                    tries=iters,
                    warmup=warmup_iters,
                )

                if len(times) > 1:
                    times = times[1:]
                mean_ms = float(np.mean(times)) if times else float("nan")
                log(f"    => {mean_ms:.3f} ms | samples={[f'{t:.3f}' for t in times]}")

                if mean_ms < best_ms:
                    best_ms = mean_ms
                    best_cfg = cfg.copy()

            except Exception as e:
                log(f"    => FAILED: {type(e).__name__}: {e}")
                # Print first 3 lines of traceback
                tb_lines = traceback.format_exc().strip().split("\n")
                for line in tb_lines[-3:]:
                    log(f"       {line}")
                continue

        all_best.append((num_tokens, best_ms, best_cfg))
        if best_cfg:
            log(f"  BEST for {num_tokens}t: {best_ms:.3f} ms | "
                f"bt={best_cfg['bt']},bf={best_cfg['bf']},"
                f"bd1={best_cfg['bd1']},bd2={best_cfg['bd2']},"
                f"btc={best_cfg['btc']}")

    # Final summary
    log(f"\n{'='*60}")
    log(f"=== TUNING RESULTS SUMMARY ===")
    log(f"Model: experts={num_experts}, top_k={top_k}, "
        f"hidden={hidden_size}, intermediate={intermediate_size}, ep={ep_size}")
    log(f"{'num_tokens':>12} | {'best_ms':>10} | {'config':>50}")
    log(f"{'-'*12}-+-{'-'*10}-+-{'-'*50}")
    for nt, ms, cfg in all_best:
        if cfg:
            cfg_str = (f"bt={cfg['bt']},bf={cfg['bf']},bd1={cfg['bd1']},"
                       f"bd2={cfg['bd2']},btc={cfg['btc']}")
        else:
            cfg_str = "NO VALID CONFIG"
        log(f"{nt:>12} | {ms:>10.3f} | {cfg_str:>50}")

    log(f"\n# Tuned block sizes for tuned_block_sizes.py:")
    log(f"# Key: (hidden, intermediate, experts, top_k, t_packing, w_packing, tokens, ep)")
    for nt, ms, cfg in all_best:
        if cfg:
            key = f"({hidden_size}, {intermediate_size}, {num_experts}, {top_k}, 2, 2, {nt}, {ep_size})"
            val = (f"({cfg['bt']}, {cfg['bf']}, {cfg['bd1']}, {cfg['bd2']}, "
                   f"{cfg['btc']}, {cfg['bfc']}, {cfg['bd1c']}, {cfg['bd2c']})")
            log(f"    {key}: {val},  # {ms:.3f} ms")


def parse_args():
    parser = argparse.ArgumentParser(description="Tune tpu-inference fused_ep_moe")
    parser.add_argument("--num-experts", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=8192)
    parser.add_argument("--intermediate-size", type=int, default=2048)
    parser.add_argument("--num-tokens", type=int, nargs="+", default=[64, 128, 256, 512, 1024])
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--warmup-iters", type=int, default=1)
    parser.add_argument("--act-fn", type=str, default="silu")
    parser.add_argument("--max-configs", type=int, default=15,
                        help="Max configs to try per token count")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_tune(
        num_tokens_list=args.num_tokens,
        num_experts=args.num_experts,
        top_k=args.top_k,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        act_fn=args.act_fn,
        iters=args.iters,
        warmup_iters=args.warmup_iters,
        max_configs_per_token=args.max_configs,
    )
