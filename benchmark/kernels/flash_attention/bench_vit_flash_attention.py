"""Benchmark the multimodal Pallas FlashAttention kernel with ViT shapes.

Example (Qwen2.5-VL-32B, one v6e core):

  uv run --extra tpu python benchmark/kernels/flash_attention/bench_vit_flash_attention.py \
    --seq-lens 16384,32768,65536 \
    --configs auto,128-128-128,256-128-128,256-256-128,256-256-256

Block configurations use ``block_q-block_k_major-block_k``. ``full`` selects
the legacy untiled-K path and is expected to fail VMEM checks at long lengths.
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
import traceback

import jax
import jax.numpy as jnp

from sgl_jax.srt.multimodal.kernels.flash_attention import (
    DEFAULT_VMEM_LIMIT_BYTES,
    BlockSizes,
    SegmentIds,
    _select_default_block_sizes,
    _select_tiled_block_sizes,
    _single_step_vmem_estimate_bytes,
    flash_attention,
)
from sgl_jax.srt.multimodal.kernels.tuned_block_sizes import get_tuned_block_sizes


def _csv_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def _csv_strings(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_block_sizes(config: str, seq_len: int) -> BlockSizes | None:
    if config == "auto":
        return None
    if config == "full":
        return BlockSizes(
            block_q=256,
            block_b=1,
            block_k_major=seq_len,
            block_k=seq_len,
        )
    try:
        block_q, block_k_major, block_k = (int(item) for item in config.split("-"))
    except ValueError as exc:
        raise ValueError(
            f"Invalid config {config!r}; expected auto, full, or block_q-block_k_major-block_k"
        ) from exc
    return BlockSizes(
        block_q=block_q,
        block_b=1,
        block_k_major=block_k_major,
        block_k=block_k,
    )


def _make_segments(seq_len: int, mode: str, segment_size: int) -> jax.Array | None:
    if mode == "none":
        return None
    if mode == "single":
        return jnp.zeros((1, seq_len), dtype=jnp.int32)
    if mode == "window":
        return (jnp.arange(seq_len, dtype=jnp.int32) // segment_size)[None, :]
    raise ValueError(f"Unknown segment mode: {mode}")


def _selected_blocks(
    q,
    k,
    v,
    segment_ids: SegmentIds | None,
    requested: BlockSizes | None,
    max_segment_len: int | None,
) -> BlockSizes:
    if requested is not None:
        return requested
    if max_segment_len is not None:
        return _select_tiled_block_sizes(q, k, v, max_segment_len=max_segment_len)
    return _select_default_block_sizes(
        q,
        k,
        v,
        None,
        segment_ids,
        vmem_limit_bytes=DEFAULT_VMEM_LIMIT_BYTES,
    )


def _attention_flops(
    *,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    blocks: BlockSizes,
    max_segment_len: int | None,
    block_sparse_segments: bool,
    segment_mode: str,
    segment_size: int,
) -> int:
    if max_segment_len is None:
        if block_sparse_segments and segment_mode == "window":
            score_pairs = 0
            for q_start in range(0, seq_len, blocks.block_q):
                q_end = min(q_start + blocks.block_q, seq_len)
                q_first = q_start // segment_size
                q_last = (q_end - 1) // segment_size
                for k_start in range(0, seq_len, blocks.block_k_major):
                    k_end = min(k_start + blocks.block_k_major, seq_len)
                    k_first = k_start // segment_size
                    k_last = (k_end - 1) // segment_size
                    if q_first <= k_last and k_first <= q_last:
                        score_pairs += blocks.block_q * blocks.block_k_major
        else:
            score_pairs = seq_len * seq_len
    else:
        block_q = blocks.block_q
        block_k_major = blocks.block_k_major
        halo = (max_segment_len - 1 + block_k_major - 1) // block_k_major
        local_grid_size = block_q // block_k_major + 2 * halo
        num_q_blocks = (seq_len + block_q - 1) // block_q
        num_kv_blocks = seq_len // block_k_major
        q_to_k_ratio = block_q // block_k_major
        valid_k_blocks = 0
        for q_index in range(num_q_blocks):
            first_k_index = q_index * q_to_k_ratio - halo
            valid_k_blocks += max(
                0,
                min(first_k_index + local_grid_size, num_kv_blocks) - max(first_k_index, 0),
            )
        score_pairs = valid_k_blocks * block_q * block_k_major
    return 4 * num_heads * score_pairs * head_dim


def _useful_attention_flops(
    *,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    segment_mode: str,
    segment_size: int,
) -> int:
    if segment_mode != "window":
        score_pairs = seq_len * seq_len
    else:
        full_segments, remainder = divmod(seq_len, segment_size)
        score_pairs = full_segments * segment_size**2 + remainder**2
    return 4 * num_heads * score_pairs * head_dim


def _benchmark_case(
    *,
    seq_len: int,
    config_name: str,
    num_heads: int,
    head_dim: int,
    segment_mode: str,
    segment_size: int,
    max_segment_len: int | None,
    block_sparse_segments: bool,
    interpret: bool,
    warmup: int,
    iterations: int,
) -> dict:
    if max_segment_len is not None and (segment_mode != "window" or segment_size > max_segment_len):
        raise ValueError(
            "max_segment_len requires window segments no longer than the declared bound"
        )
    if block_sparse_segments and segment_mode == "none":
        raise ValueError("block_sparse_segments requires segment IDs")
    shape = (1, num_heads, seq_len, head_dim)
    q = jnp.ones(shape, dtype=jnp.bfloat16)
    k = jnp.ones(shape, dtype=jnp.bfloat16)
    v = jnp.ones(shape, dtype=jnp.bfloat16)
    segments = _make_segments(seq_len, segment_mode, segment_size)
    segment_ids = None if segments is None else SegmentIds(q=segments, kv=segments)
    requested_blocks = _parse_block_sizes(config_name, seq_len)
    selected_blocks = _selected_blocks(q, k, v, segment_ids, requested_blocks, max_segment_len)

    def run(q, k, v, segments):
        ids = None if segments is None else SegmentIds(q=segments, kv=segments)
        return flash_attention(
            q,
            k,
            v,
            segment_ids=ids,
            sm_scale=head_dim**-0.5,
            block_sizes=requested_blocks,
            max_segment_len=max_segment_len,
            block_sparse_segments=block_sparse_segments,
            interpret=interpret,
        )

    jitted_run = jax.jit(run)
    compile_start = time.perf_counter()
    output = jitted_run(q, k, v, segments)
    jax.block_until_ready(output)
    compile_seconds = time.perf_counter() - compile_start

    for _ in range(max(0, warmup - 1)):
        jax.block_until_ready(jitted_run(q, k, v, segments))

    samples_ms = []
    for _ in range(iterations):
        start = time.perf_counter()
        output = jitted_run(q, k, v, segments)
        jax.block_until_ready(output)
        samples_ms.append((time.perf_counter() - start) * 1e3)

    median_ms = statistics.median(samples_ms)
    # QK^T and PV each perform one multiply and one add.
    dense_flops = 4 * num_heads * seq_len * seq_len * head_dim
    executed_flops = _attention_flops(
        seq_len=seq_len,
        num_heads=num_heads,
        head_dim=head_dim,
        blocks=selected_blocks,
        max_segment_len=max_segment_len,
        block_sparse_segments=block_sparse_segments,
        segment_mode=segment_mode,
        segment_size=segment_size,
    )
    useful_flops = _useful_attention_flops(
        seq_len=seq_len,
        num_heads=num_heads,
        head_dim=head_dim,
        segment_mode=segment_mode,
        segment_size=segment_size,
    )

    single_step_block_q = get_tuned_block_sizes(
        q.dtype,
        k.dtype,
        v.dtype,
        q.shape[0],
        q.shape[1],
        q.shape[2],
        k.shape[2],
        q.shape[3],
    )
    full_k = BlockSizes(
        block_q=single_step_block_q,
        block_b=selected_blocks.block_b,
        block_k_major=seq_len,
        block_k=seq_len,
    )
    full_k_vmem_mib = _single_step_vmem_estimate_bytes(q, k, v, None, segment_ids, full_k) / 2**20
    result = {
        "status": "ok",
        "seq_len": seq_len,
        "config": config_name,
        "selected": {
            "block_q": selected_blocks.block_q,
            "block_k_major": selected_blocks.block_k_major,
            "block_k": selected_blocks.block_k,
        },
        "shape": list(shape),
        "segment_mode": segment_mode,
        "segment_size": segment_size,
        "max_segment_len": max_segment_len,
        "block_sparse_segments": block_sparse_segments,
        "interpret": interpret,
        "compile_s": round(compile_seconds, 3),
        "samples_ms": [round(sample, 3) for sample in samples_ms],
        "median_ms": round(median_ms, 3),
        "executed_tflops_per_s": round(executed_flops / (median_ms * 1e9), 3),
        "dense_equivalent_tflops_per_s": round(dense_flops / (median_ms * 1e9), 3),
        "useful_tflops_per_s": round(useful_flops / (median_ms * 1e9), 3),
        "executed_to_useful_flops": round(executed_flops / useful_flops, 3),
        "full_k_vmem_estimate_mib": round(full_k_vmem_mib, 2),
        "output_sample": float(output[0, 0, 0, 0]),
    }
    del output, jitted_run, q, k, v, segments, segment_ids
    gc.collect()
    jax.clear_caches()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-lens", type=_csv_ints, default=[16384, 32768, 65536])
    parser.add_argument(
        "--configs",
        type=_csv_strings,
        default=["auto", "128-128-128", "256-128-128", "256-256-128", "256-256-256"],
    )
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--head-dim", type=int, default=80)
    parser.add_argument("--segment-mode", choices=["none", "single", "window"], default="single")
    parser.add_argument("--segment-size", type=int, default=128)
    parser.add_argument(
        "--segment-sizes",
        type=_csv_ints,
        default=None,
        help="Optional comma-separated segment sizes to benchmark.",
    )
    parser.add_argument(
        "--max-segment-len",
        type=int,
        default=None,
        help="Enable the local K-grid optimization with this segment-length bound.",
    )
    parser.add_argument("--block-sparse-segments", action="store_true")
    parser.add_argument(
        "--compare-block-sparse",
        action="store_true",
        help="Run both dense and block-sparse segment grids.",
    )
    parser.add_argument("--interpret", action="store_true")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    args = parser.parse_args()

    print(
        json.dumps(
            {
                "device_kind": jax.devices()[0].device_kind,
                "device_count": jax.device_count(),
                "jax_version": jax.__version__,
                "num_heads": args.num_heads,
                "head_dim": args.head_dim,
                "segment_mode": args.segment_mode,
                "max_segment_len": args.max_segment_len,
                "block_sparse_segments": args.block_sparse_segments,
                "interpret": args.interpret,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    segment_sizes = args.segment_sizes or [args.segment_size]
    sparse_modes = [False, True] if args.compare_block_sparse else [args.block_sparse_segments]
    for block_sparse_segments in sparse_modes:
        for segment_size in segment_sizes:
            for seq_len in args.seq_lens:
                for config in args.configs:
                    try:
                        result = _benchmark_case(
                            seq_len=seq_len,
                            config_name=config,
                            num_heads=args.num_heads,
                            head_dim=args.head_dim,
                            segment_mode=args.segment_mode,
                            segment_size=segment_size,
                            max_segment_len=args.max_segment_len,
                            block_sparse_segments=block_sparse_segments,
                            interpret=args.interpret,
                            warmup=args.warmup,
                            iterations=args.iterations,
                        )
                    except Exception as exc:  # noqa: BLE001
                        result = {
                            "status": "error",
                            "seq_len": seq_len,
                            "config": config,
                            "segment_size": segment_size,
                            "block_sparse_segments": block_sparse_segments,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                            "traceback_tail": traceback.format_exc().splitlines()[-8:],
                        }
                        jax.clear_caches()
                    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
