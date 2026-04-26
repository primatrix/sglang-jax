#!/usr/bin/env python3
"""Compare QKV weight dumps from per-head vs uniform FP8 dequantization.

Usage:
    python scripts/compare_dequant_weights.py \
        --per-head /models/weight_dump/per_head \
        --uniform  /models/weight_dump/uniform \
        --head-dim 192 --v-head-dim 128 --block-size 128
"""

import argparse
import os

import numpy as np


def load_weight(directory: str, layer: int, proj: str) -> np.ndarray:
    path = os.path.join(directory, f"layer_{layer}_{proj}.npy")
    return np.load(path)


def boundary_analysis(diff: np.ndarray, proj: str, head_dim: int, v_head_dim: int,
                       block_size: int, num_kv_heads: int):
    """Analyze whether diff concentrates near head_dim block boundaries."""
    if proj == "q_proj":
        return None

    hd = head_dim if proj == "k_proj" else v_head_dim
    out_dim = diff.shape[1]  # weight layout [in_dim, out_dim]
    if out_dim == 0:
        return None

    per_head_out = out_dim // num_kv_heads if num_kv_heads > 0 else out_dim

    # Identify boundary regions: last (hd % block_size) elements of each head's output chunk
    remainder = hd % block_size
    if remainder == 0:
        return {"boundary_fraction": 0.0, "note": "head_dim aligned to block_size, no boundary issue"}

    boundary_mask = np.zeros(out_dim, dtype=bool)
    for h in range(num_kv_heads):
        start = h * per_head_out
        # The boundary block starts at floor(hd/block_size)*block_size within each head
        boundary_start = start + (hd // block_size) * block_size
        boundary_end = start + hd
        if proj == "k_proj":
            # For K, the boundary block [boundary_start, boundary_start+block_size)
            # crosses into V territory in fused mode
            boundary_mask[boundary_start:min(boundary_end, out_dim)] = True
        else:
            # For V, the first block [0, block_size) of each head shares a scale
            # with the tail of K in fused mode
            v_start = start
            v_boundary_end = min(start + block_size - remainder, out_dim)
            boundary_mask[v_start:v_boundary_end] = True

    abs_diff = np.abs(diff)
    col_diff = abs_diff.mean(axis=0)  # [out_dim]

    boundary_mean = col_diff[boundary_mask].mean() if boundary_mask.any() else 0.0
    non_boundary_mean = col_diff[~boundary_mask].mean() if (~boundary_mask).any() else 0.0

    return {
        "boundary_cols": int(boundary_mask.sum()),
        "total_cols": out_dim,
        "boundary_mean_diff": float(boundary_mean),
        "non_boundary_mean_diff": float(non_boundary_mean),
        "ratio": float(boundary_mean / non_boundary_mean) if non_boundary_mean > 0 else float("inf"),
    }


def compare_one(per_head_dir: str, uniform_dir: str, layer: int, proj: str,
                head_dim: int, v_head_dim: int, block_size: int, num_kv_heads: int):
    w_ph = load_weight(per_head_dir, layer, proj)
    w_un = load_weight(uniform_dir, layer, proj)

    result = {"layer": layer, "proj": proj}

    if w_ph.shape != w_un.shape:
        result["error"] = f"shape mismatch: per_head={w_ph.shape} uniform={w_un.shape}"
        return result

    result["shape"] = w_ph.shape
    result["dtype"] = str(w_ph.dtype)

    diff = w_ph.astype(np.float32) - w_un.astype(np.float32)
    abs_diff = np.abs(diff)

    result["max_abs_diff"] = float(abs_diff.max())
    result["mean_abs_diff"] = float(abs_diff.mean())
    result["std_diff"] = float(diff.std())

    # Relative diff (avoid div-by-zero)
    denom = np.maximum(np.abs(w_un.astype(np.float32)), 1e-12)
    rel_diff = abs_diff / denom
    result["max_rel_diff"] = float(rel_diff.max())
    result["mean_rel_diff"] = float(rel_diff.mean())

    # Percentiles
    flat = abs_diff.ravel()
    for p in [50, 90, 95, 99, 99.9]:
        result[f"p{p}"] = float(np.percentile(flat, p))

    # Non-zero ratio
    result["nonzero_fraction"] = float((abs_diff > 0).mean())

    # Location of max diff
    idx = np.unravel_index(abs_diff.argmax(), abs_diff.shape)
    result["max_diff_location"] = idx

    # Boundary analysis for K/V
    if proj in ("k_proj", "v_proj"):
        result["boundary"] = boundary_analysis(
            diff, proj, head_dim, v_head_dim, block_size, num_kv_heads,
        )

    return result


def print_results(results: list):
    sep = "=" * 90
    for r in results:
        print(sep)
        print(f"Layer {r['layer']}  {r['proj']}")
        print("-" * 90)

        if "error" in r:
            print(f"  ERROR: {r['error']}")
            continue

        print(f"  shape: {r['shape']}   dtype: {r['dtype']}")
        print(f"  max_abs_diff:  {r['max_abs_diff']:.6e}")
        print(f"  mean_abs_diff: {r['mean_abs_diff']:.6e}")
        print(f"  std_diff:      {r['std_diff']:.6e}")
        print(f"  max_rel_diff:  {r['max_rel_diff']:.6e}")
        print(f"  mean_rel_diff: {r['mean_rel_diff']:.6e}")
        print(f"  nonzero_frac:  {r['nonzero_fraction']:.4f}")
        print(f"  percentiles:   p50={r['p50']:.2e}  p90={r['p90']:.2e}  "
              f"p95={r['p95']:.2e}  p99={r['p99']:.2e}  p99.9={r['p99.9']:.2e}")
        print(f"  max_diff_at:   {r['max_diff_location']}")

        if "boundary" in r and r["boundary"] is not None:
            b = r["boundary"]
            if "note" in b:
                print(f"  boundary:      {b['note']}")
            else:
                print(f"  boundary analysis:")
                print(f"    boundary cols:     {b['boundary_cols']}/{b['total_cols']}")
                print(f"    boundary mean:     {b['boundary_mean_diff']:.6e}")
                print(f"    non-boundary mean: {b['non_boundary_mean_diff']:.6e}")
                print(f"    ratio (boundary/non): {b['ratio']:.2f}x")

    print(sep)

    # Summary table
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("-" * 90)
    print(f"{'Layer':>5}  {'Proj':>6}  {'MaxAbsDiff':>12}  {'MeanAbsDiff':>12}  {'NonzeroFrac':>12}  {'Identical?':>10}")
    print("-" * 90)
    for r in results:
        if "error" in r:
            print(f"{r['layer']:>5}  {r['proj']:>6}  {'ERROR':>12}")
            continue
        identical = "YES" if r["max_abs_diff"] == 0.0 else "NO"
        print(f"{r['layer']:>5}  {r['proj']:>6}  {r['max_abs_diff']:>12.4e}  "
              f"{r['mean_abs_diff']:>12.4e}  {r['nonzero_fraction']:>12.4f}  {identical:>10}")


def main():
    parser = argparse.ArgumentParser(description="Compare per-head vs uniform FP8 dequant weights")
    parser.add_argument("--per-head", required=True, help="Directory with per-head dequant dumps")
    parser.add_argument("--uniform", required=True, help="Directory with uniform dequant dumps")
    parser.add_argument("--layers", type=int, default=3, help="Number of layers to compare")
    parser.add_argument("--head-dim", type=int, default=192)
    parser.add_argument("--v-head-dim", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--num-kv-heads", type=int, default=0,
                        help="Number of KV heads (0 = infer from weight shape)")
    args = parser.parse_args()

    results = []
    for layer in range(args.layers):
        for proj in ("q_proj", "k_proj", "v_proj"):
            num_kv_heads = args.num_kv_heads
            if num_kv_heads == 0:
                # Infer from weight shape
                w = load_weight(args.per_head, layer, proj)
                hd = args.head_dim if proj == "k_proj" else args.v_head_dim
                if proj in ("k_proj", "v_proj"):
                    num_kv_heads = w.shape[1] // hd  # [in_dim, out_dim]

            r = compare_one(
                args.per_head, args.uniform, layer, proj,
                args.head_dim, args.v_head_dim, args.block_size, num_kv_heads,
            )
            results.append(r)

    print_results(results)


if __name__ == "__main__":
    main()
