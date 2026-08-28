"""Tune GMM v2 tiles for replicated Ling-3.0-tiny experts on TPU.

The replicated EPMoE path runs one local GMM per data-parallel rank.  With
top-k=8, the important padded M sizes are 32 (decode BS=1), 64 (decode BS=8),
and roughly 2048 (a balanced share of a 2K global prefill chunk).  Ling Tiny
uses two expert matrix shapes: 1536x512 for wi_0/wi_1 and 512x1536 for wo.

This benchmark compares a small, aligned tile grid with the current GMM v2
auto-tiler and writes one JSON record per matrix/M combination.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.pallas import tpu as pltpu

from sgl_jax.srt.kernels.gmm.megablox_gmm_kernel.gmm_v2 import (
    Dimensions,
    TileSizes,
    calculate_tiling,
    gmm_v2,
    get_scope_name,
)
from sgl_jax.srt.kernels.utils.perf import multiple_iteration_timeit_from_trace


NUM_EXPERTS = 128
MATRIX_CASES = {
    "wi": (1536, 512, False),
    "wo": (512, 1536, True),
}


def _csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item)


def _group_sizes(size_m: int) -> jax.Array:
    quotient, remainder = divmod(size_m, NUM_EXPERTS)
    sizes = np.full(NUM_EXPERTS, quotient, dtype=np.int32)
    sizes[:remainder] += 1
    return jnp.asarray(sizes)


def _dimensions(size_m: int, size_k: int, size_n: int) -> Dimensions:
    sublane = min(pltpu.get_tpu_info().get_sublane_tiling(jnp.bfloat16), size_m)
    return Dimensions(
        size_m=size_m,
        size_k=size_k,
        size_n=size_n,
        size_group=NUM_EXPERTS,
        size_lhs_group=NUM_EXPERTS,
        size_lhs_sublane=sublane,
    )


def _auto_tiles(dims: Dimensions) -> TileSizes:
    vmem_limit = int(pltpu.get_tpu_info().vmem_capacity_bytes * 0.9)
    return calculate_tiling(jnp.bfloat16, jnp.bfloat16, dims, vmem_limit)


def _candidate_tiles(dims: Dimensions, auto: TileSizes) -> list[TileSizes]:
    tile_ms = (32, 64, 128)
    tile_ks = (256, 512, 768, 1536)
    tile_ns = (256, 512, 768, 1536)
    candidates = {auto}
    for tile_m in tile_ms:
        if tile_m > dims.size_m:
            continue
        for tile_k in tile_ks:
            if tile_k > dims.size_k:
                continue
            for tile_n in tile_ns:
                if tile_n > dims.size_n:
                    continue
                candidates.add(TileSizes(tile_m, tile_k, tile_n))
    others = sorted(
        candidates - {auto}, key=lambda tile: (tile.tile_m, tile.tile_k, tile.tile_n)
    )
    return [auto, *others]


def _bench(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    dims: Dimensions,
    tiles: TileSizes,
    zero_initialize: bool,
    tries: int,
) -> tuple[float, jax.Array]:
    def compute():
        return gmm_v2(
            lhs,
            rhs,
            group_sizes,
            tile_info=tiles,
            preferred_element_type=jnp.bfloat16,
            acc_dtype=jnp.float32,
            maybe_quantize_lhs=False,
            zero_initialize=zero_initialize,
        )

    output = compute()
    jax.block_until_ready(output)
    scope = get_scope_name(dims, tiles)
    times = multiple_iteration_timeit_from_trace(
        compute_func=lambda: compute(),
        data_generator=lambda: (),
        task=scope,
        tries=tries,
    )
    if not times:
        raise RuntimeError(f"No device duration found for {scope}")
    return float(np.mean(times)), output


def _run_case(matrix: str, size_m: int, tries: int) -> dict:
    size_k, size_n, zero_initialize = MATRIX_CASES[matrix]
    dims = _dimensions(size_m, size_k, size_n)
    auto = _auto_tiles(dims)
    candidates = _candidate_tiles(dims, auto)

    lhs = jax.random.normal(jax.random.key(size_m + size_k), (size_m, size_k), jnp.bfloat16)
    rhs = jax.random.normal(
        jax.random.key(size_m + size_n),
        (NUM_EXPERTS, size_k, size_n),
        jnp.bfloat16,
    )
    group_sizes = _group_sizes(size_m)

    best_tiles = None
    best_ms = float("inf")
    auto_ms = float("inf")
    reference = None
    attempted = 0
    failed = 0

    for tiles in candidates:
        attempted += 1
        try:
            latency_ms, output = _bench(
                lhs,
                rhs,
                group_sizes,
                dims,
                tiles,
                zero_initialize,
                tries,
            )
            if tiles == auto:
                auto_ms = latency_ms
                reference = np.asarray(output)
            elif reference is not None:
                np.testing.assert_allclose(
                    np.asarray(output), reference, rtol=2e-2, atol=2e-2
                )
            if latency_ms < best_ms:
                best_ms = latency_ms
                best_tiles = tiles
        except Exception as exc:  # Keep the sweep moving past invalid/VMEM tiles.
            failed += 1
            print(f"SKIP matrix={matrix} m={size_m} tiles={tiles}: {exc}", flush=True)

    if best_tiles is None or not np.isfinite(auto_ms):
        raise RuntimeError(f"No valid GMM result for matrix={matrix}, m={size_m}")

    record = {
        "case": f"{matrix}_m{size_m}",
        "matrix": matrix,
        "size_m": size_m,
        "size_k": size_k,
        "size_n": size_n,
        "num_experts": NUM_EXPERTS,
        "dtype": "bfloat16",
        "zero_initialize": zero_initialize,
        "auto_tiles": dataclasses.asdict(auto),
        "auto_latency_ms": auto_ms,
        "best_tiles": dataclasses.asdict(best_tiles),
        "best_latency_ms": best_ms,
        "speedup_over_auto": auto_ms / best_ms,
        "candidates_attempted": attempted,
        "candidates_failed": failed,
    }
    print(json.dumps(record, sort_keys=True), flush=True)
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m-sizes", type=_csv_ints, default=(32, 64, 2048))
    parser.add_argument("--matrices", type=str, default="wi,wo")
    parser.add_argument("--tries", type=int, default=3)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    args = parser.parse_args()

    matrices = tuple(item for item in args.matrices.split(",") if item)
    unknown = set(matrices) - MATRIX_CASES.keys()
    if unknown:
        raise ValueError(f"Unknown matrix cases: {sorted(unknown)}")
    if args.tries < 1:
        raise ValueError("--tries must be positive")

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as output:
        for size_m in args.m_sizes:
            if size_m < 1:
                raise ValueError("M sizes must be positive")
            for matrix in matrices:
                record = _run_case(matrix, size_m, args.tries)
                output.write(json.dumps(record, sort_keys=True) + "\n")
                output.flush()


if __name__ == "__main__":
    main()
