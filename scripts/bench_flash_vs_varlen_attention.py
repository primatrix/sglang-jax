#!/usr/bin/env python3
"""Tune FlashAttention and compare it with varlen_attention on TPU."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Callable
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.multimodal.kernels.flash_attention import (
    BlockSizes,
    SegmentIds,
    flash_attention,
)
from sgl_jax.srt.multimodal.kernels.varlen_attention import varlen_attention

VMEM_LIMIT_BYTES = 60 * 1024 * 1024

SCENARIOS: dict[str, tuple[int, ...]] = {
    "full_256": (256,),
    "full_1024": (1024,),
    "full_4096": (4096,),
    "full_8192": (8192,),
    "windows_8192x64": (64,) * 128,
    "windows_16384x64": (64,) * 256,
    "frames_8192x1024": (1024,) * 8,
    "mixed_8192": (256, 512, 1024, 2048, 4096, 256),
}

TUNE_SCENARIOS = ("full_4096", "windows_8192x64", "mixed_8192")
TUNE_CANDIDATES = (
    (128, 256, 128),
    (256, 256, 128),
    (256, 512, 128),
    (512, 512, 128),
    (256, 512, 256),
    (512, 1024, 256),
)


def _parse_blocks(value: str) -> tuple[int, int, int]:
    result = tuple(int(item) for item in value.split(","))
    if len(result) != 3:
        raise argparse.ArgumentTypeError("blocks must be BQ,BKV_MAJOR,BKV")
    return result


def _parse_names(value: str) -> tuple[str, ...]:
    names = tuple(item for item in value.split(",") if item)
    unknown = set(names) - SCENARIOS.keys()
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown scenarios: {sorted(unknown)}")
    return names


def _block_sizes(values: tuple[int, int, int]) -> BlockSizes:
    bq, bkv_major, bkv = values
    return BlockSizes(block_q=bq, block_k_major=bkv_major, block_k=bkv, block_b=1)


def _write_metric(path: Path, metric: dict) -> None:
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(metric, sort_keys=True) + "\n")


def _measure(
    fn: Callable[[], jax.Array], warmup: int, iterations: int
) -> tuple[float, float, float, jax.Array]:
    start = time.perf_counter()
    output = fn()
    jax.block_until_ready(output)
    compile_ms = (time.perf_counter() - start) * 1e3

    for _ in range(warmup):
        jax.block_until_ready(fn())

    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        output = fn()
        jax.block_until_ready(output)
        samples.append((time.perf_counter() - start) * 1e3)
    return compile_ms, statistics.median(samples), min(samples), output


def _inputs(
    seq_lens: tuple[int, ...], heads: int, head_dim: int, seed: int
) -> tuple[jax.Array, ...]:
    total_tokens = sum(seq_lens)
    keys = jax.random.split(jax.random.key(seed), 3)
    thd_shape = (total_tokens, heads, head_dim)
    q, k, v = (jax.random.normal(key, thd_shape, dtype=jnp.bfloat16) for key in keys)
    q_flash, k_flash, v_flash = (
        jnp.transpose(value[None, ...], (0, 2, 1, 3)) for value in (q, k, v)
    )
    ids = jnp.asarray(np.repeat(np.arange(len(seq_lens), dtype=np.int32), seq_lens)[None, :])
    segment_ids = SegmentIds(q=ids, kv=ids)
    cu_seqlens = jnp.asarray((0, *np.cumsum(seq_lens)), dtype=jnp.int32)
    num_seqs = jnp.asarray([len(seq_lens)], dtype=jnp.int32)
    jax.block_until_ready((q, k, v, q_flash, k_flash, v_flash, segment_ids, cu_seqlens, num_seqs))
    return (
        q,
        k,
        v,
        q_flash,
        k_flash,
        v_flash,
        segment_ids,
        cu_seqlens,
        num_seqs,
    )


def _base_metric(
    scenario: str,
    seq_lens: tuple[int, ...],
    heads: int,
    head_dim: int,
) -> dict:
    return {
        "scenario": scenario,
        "dtype": "bfloat16",
        "heads": heads,
        "head_dim": head_dim,
        "total_tokens": sum(seq_lens),
        "max_seq_len": max(seq_lens),
        "num_seqs": len(seq_lens),
    }


def _run_flash_tuning(args: argparse.Namespace, output: Path) -> None:
    scores: dict[tuple[int, int, int], list[float]] = {item: [] for item in TUNE_CANDIDATES}
    for scenario_index, scenario in enumerate(args.scenarios):
        seq_lens = SCENARIOS[scenario]
        inputs = _inputs(seq_lens, args.heads, args.head_dim, args.seed + scenario_index)
        _, _, _, q, k, v, segment_ids, _, _ = inputs

        for candidate in TUNE_CANDIDATES:
            blocks = _block_sizes(candidate)

            def run() -> jax.Array:
                return flash_attention(
                    q,
                    k,
                    v,
                    segment_ids=segment_ids,
                    sm_scale=args.head_dim**-0.5,
                    block_sizes=blocks,
                    vmem_limit_bytes=VMEM_LIMIT_BYTES,
                )

            metric = {
                **_base_metric(scenario, seq_lens, args.heads, args.head_dim),
                "variant": "flash_attention_tune",
                "block_q": candidate[0],
                "block_k_major": candidate[1],
                "block_k": candidate[2],
            }
            try:
                compile_ms, median_ms, min_ms, _ = _measure(run, args.warmup, args.iters)
                metric.update(
                    status="ok",
                    compile_time_ms=compile_ms,
                    latency_ms=median_ms,
                    min_latency_ms=min_ms,
                    tokens_per_sec=sum(seq_lens) * 1e3 / median_ms,
                )
                scores[candidate].append(median_ms)
                print(
                    f"TUNE {scenario} blocks={candidate} median_ms={median_ms:.4f}",
                    flush=True,
                )
            except Exception as error:  # Keep the small candidate sweep running.
                metric.update(status="error", error=f"{type(error).__name__}: {error}")
                print(f"TUNE {scenario} blocks={candidate} ERROR {error}", flush=True)
            _write_metric(output, metric)

    eligible = {
        candidate: statistics.geometric_mean(latencies)
        for candidate, latencies in scores.items()
        if len(latencies) == len(args.scenarios)
    }
    if not eligible:
        raise RuntimeError("no FlashAttention candidate completed every tuning scenario")
    best, score_ms = min(eligible.items(), key=lambda item: item[1])
    _write_metric(
        output,
        {
            "variant": "flash_attention_tune_summary",
            "scenario": "geomean",
            "block_q": best[0],
            "block_k_major": best[1],
            "block_k": best[2],
            "latency_ms": score_ms,
            "status": "ok",
        },
    )
    print(f"BEST_FLASH_BLOCKS={best[0]},{best[1]},{best[2]}", flush=True)


def _run_comparison(args: argparse.Namespace, output: Path) -> None:
    for scenario_index, scenario in enumerate(args.scenarios):
        seq_lens = SCENARIOS[scenario]
        total_tokens = sum(seq_lens)
        flash_blocks = args.flash_blocks
        if total_tokens < max(args.flash_blocks[0], args.flash_blocks[1]):
            max_block = 1 << (total_tokens.bit_length() - 1)
            block_q = min(args.flash_blocks[0], max_block)
            block_k_major = min(args.flash_blocks[1], max_block)
            block_k = min(args.flash_blocks[2], max(128, block_k_major // 2))
            flash_blocks = (block_q, block_k_major, block_k)
        blocks = _block_sizes(flash_blocks)
        (
            q,
            k,
            v,
            q_flash,
            k_flash,
            v_flash,
            segment_ids,
            cu_seqlens,
            num_seqs,
        ) = _inputs(seq_lens, args.heads, args.head_dim, args.seed + scenario_index)

        def run_flash() -> jax.Array:
            return flash_attention(
                q_flash,
                k_flash,
                v_flash,
                segment_ids=segment_ids,
                sm_scale=args.head_dim**-0.5,
                block_sizes=blocks,
                vmem_limit_bytes=VMEM_LIMIT_BYTES,
            )

        def run_varlen() -> jax.Array:
            return varlen_attention(
                q,
                k,
                v,
                cu_seqlens,
                num_seqs,
                sm_scale=args.head_dim**-0.5,
                max_seq_len=max(seq_lens),
                vmem_limit_bytes=VMEM_LIMIT_BYTES,
            )

        common = _base_metric(scenario, seq_lens, args.heads, args.head_dim)
        flash_compile, flash_median, flash_min, flash_output = _measure(
            run_flash, args.warmup, args.iters
        )
        varlen_compile, varlen_median, varlen_min, varlen_output = _measure(
            run_varlen, args.warmup, args.iters
        )
        flash_thd = jnp.transpose(flash_output[0], (1, 0, 2))
        max_abs_error = float(
            jnp.max(jnp.abs(flash_thd.astype(jnp.float32) - varlen_output.astype(jnp.float32)))
        )

        for variant, compile_ms, median_ms, min_ms in (
            ("flash_attention", flash_compile, flash_median, flash_min),
            ("varlen_attention", varlen_compile, varlen_median, varlen_min),
        ):
            metric = {
                **common,
                "variant": variant,
                "compile_time_ms": compile_ms,
                "latency_ms": median_ms,
                "min_latency_ms": min_ms,
                "tokens_per_sec": sum(seq_lens) * 1e3 / median_ms,
                "max_abs_error": max_abs_error,
                "status": "ok",
            }
            if variant == "flash_attention":
                metric.update(
                    block_q=flash_blocks[0],
                    block_k_major=flash_blocks[1],
                    block_k=flash_blocks[2],
                )
            _write_metric(output, metric)
        print(
            f"COMPARE {scenario} flash_ms={flash_median:.4f} "
            f"varlen_ms={varlen_median:.4f} speedup={flash_median / varlen_median:.3f} "
            f"max_abs_error={max_abs_error:.6g}",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("tune-flash", "compare"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--head-dim", type=int, default=80)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--flash-blocks", type=_parse_blocks)
    parser.add_argument("--scenarios", type=_parse_names)
    args = parser.parse_args()

    if "TPU" not in jax.devices()[0].device_kind:
        raise RuntimeError(f"this benchmark requires TPU, got {jax.devices()[0].device_kind}")
    if args.mode == "tune-flash":
        args.scenarios = args.scenarios or TUNE_SCENARIOS
    else:
        args.scenarios = args.scenarios or tuple(SCENARIOS)
        if args.flash_blocks is None:
            parser.error("--flash-blocks is required in compare mode")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("", encoding="utf-8")
    print(
        f"device={jax.devices()[0].device_kind} mode={args.mode} "
        f"scenarios={args.scenarios} heads={args.heads} head_dim={args.head_dim}",
        flush=True,
    )
    if args.mode == "tune-flash":
        _run_flash_tuning(args, args.output)
    else:
        _run_comparison(args, args.output)


if __name__ == "__main__":
    main()
