#!/usr/bin/env python3
"""Benchmark dense-segmented and block-sparse FlashAttention on TPU."""

from __future__ import annotations

import argparse
import statistics
import time

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.multimodal.kernels.flash_attention import (
    _MAX_BLOCK_SPARSE_PREFETCH_ENTRIES,
    BlockSizes,
    SegmentIds,
    _segment_block_sparse_schedule,
    flash_attention,
)


def _csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item)


def _blocks(value: str) -> BlockSizes:
    sizes = _csv_ints(value)
    if len(sizes) != 3:
        raise argparse.ArgumentTypeError("blocks must be BQ,BKV_MAJOR,BKV")
    return BlockSizes(
        block_q=sizes[0],
        block_k_major=sizes[1],
        block_k=sizes[2],
        block_b=1,
    )


def _measure(fn, warmup: int, iterations: int) -> tuple[float, float, float]:
    start = time.perf_counter()
    output = fn()
    jax.block_until_ready(output)
    compile_ms = (time.perf_counter() - start) * 1e3
    for _ in range(warmup):
        jax.block_until_ready(fn())
    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        jax.block_until_ready(fn())
        samples.append((time.perf_counter() - start) * 1e3)
    return compile_ms, statistics.median(samples), min(samples)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seq-lens",
        type=_csv_ints,
        default=(1024, 1536, 2048, 2560, 3072, 2048, 2048, 2048),
    )
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--head-dim", type=int, default=80)
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--blocks", type=_blocks, default=_blocks("1024,512,128"))
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    args = parser.parse_args()

    total_tokens = sum(args.seq_lens)
    if total_tokens % args.blocks.block_k_major:
        raise ValueError("total sequence length must be divisible by block_k_major")
    dtype = jnp.bfloat16 if args.dtype == "bfloat16" else jnp.float32
    keys = jax.random.split(jax.random.key(11), 3)
    shape = (1, args.heads, total_tokens, args.head_dim)
    q = jax.random.normal(keys[0], shape, dtype=dtype)
    k = jax.random.normal(keys[1], shape, dtype=dtype)
    v = jax.random.normal(keys[2], shape, dtype=dtype)
    ids = jnp.asarray(np.repeat(np.arange(len(args.seq_lens), dtype=np.int32), args.seq_lens)[None])
    segments = SegmentIds(q=ids, kv=ids)
    block_mask, _ = _segment_block_sparse_schedule(
        ids,
        ids,
        block_q=args.blocks.block_q,
        block_k_major=args.blocks.block_k_major,
    )
    schedule_entries = block_mask.size
    active_blocks = int(jnp.sum(block_mask))

    def run(block_sparse_segments: bool):
        return flash_attention(
            q,
            k,
            v,
            segment_ids=segments,
            sm_scale=args.head_dim**-0.5,
            block_sizes=args.blocks,
            block_sparse_segments=block_sparse_segments,
        )

    print(
        f"device={jax.devices()[0].device_kind} seq_lens={args.seq_lens} "
        f"shape={shape} dtype={args.dtype} blocks="
        f"{args.blocks.block_q},{args.blocks.block_k_major},{args.blocks.block_k}"
    )
    print(
        f"schedule_entries={schedule_entries} active_blocks={active_blocks} "
        f"eligible={schedule_entries <= _MAX_BLOCK_SPARSE_PREFETCH_ENTRIES}"
    )
    outputs = {}
    for name, enabled in (("dense", False), ("block_sparse", True)):
        compile_ms, median_ms, min_ms = _measure(
            lambda enabled=enabled: run(enabled),
            args.warmup,
            args.iters,
        )
        outputs[name] = run(enabled)
        print(
            f"{name:>12} compile_ms={compile_ms:.1f} "
            f"median_ms={median_ms:.3f} min_ms={min_ms:.3f}",
            flush=True,
        )

    error = jnp.max(
        jnp.abs(outputs["dense"].astype(jnp.float32) - outputs["block_sparse"].astype(jnp.float32))
    )
    print(f"max_abs_error={float(error):.6g}")


if __name__ == "__main__":
    main()
