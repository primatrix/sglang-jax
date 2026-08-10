"""Benchmark logical DSA top-k to physical cache-slot mapping.

The default shape matches the profiled GLM-5.2 extend shard:

* Q=2048 query rows
* K=2048 selected logical positions per row
* two ragged sequences
* page_size=64 and a 4224-entry packed page table

Example on a single TPU host::

    python -m benchmark.kernels.dsa.bench_physical_slots \
      --block-q 32,64,128,256 --warmup 5 --iters 30

Use ``--smoke --interpret`` for a CPU correctness/timing smoke test only.
"""

from __future__ import annotations

import argparse
import functools
import importlib.metadata as metadata
import json
import statistics
import time
from dataclasses import asdict, dataclass

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.kernels.dsa.physical_slots import (
    logical_topk_to_physical_slots_pallas,
)


def logical_topk_to_physical_slots_xla(
    topk,
    seq_lens,
    page_indices,
    cu_q_lens,
    cu_kv_lens,
    page_size,
):
    """The current production implementation, kept verbatim for the A/B."""

    num_tokens, topk_size = topk.shape
    token_ids = jnp.arange(num_tokens, dtype=jnp.int32)
    seq_ids = jnp.searchsorted(cu_q_lens[1:], token_ids, side="right")
    seq_ids = jnp.clip(seq_ids, 0, seq_lens.shape[0] - 1)

    logical = jnp.maximum(topk, 0)
    page_ptr = cu_kv_lens[seq_ids, None] // page_size + logical // page_size
    ptr_in_bounds = (page_ptr >= 0) & (page_ptr < page_indices.shape[0])
    safe_ptr = jnp.clip(page_ptr, 0, page_indices.shape[0] - 1)
    physical_pages = page_indices[safe_ptr]
    query_valid = token_ids < cu_q_lens[-1]
    valid = (
        query_valid[:, None]
        & (topk >= 0)
        & (logical < seq_lens[seq_ids, None])
        & ptr_in_bounds
        & (physical_pages >= 0)
    )
    physical_slots = physical_pages * page_size + logical % page_size
    physical_slots = jnp.where(valid, physical_slots, jnp.int32(0))
    selected_counts = jnp.sum(valid, axis=1, dtype=jnp.int32)

    del topk_size
    return physical_slots.astype(jnp.int32), selected_counts


@dataclass(frozen=True)
class Timing:
    variant: str
    block_q: int | None
    median_ms: float
    mean_ms: float
    minimum_ms: float
    p90_ms: float
    speedup_vs_xla: float | None = None


def _percentile(values: list[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _measure(fn, args, *, warmup: int, iters: int) -> list[float]:
    for _ in range(warmup):
        jax.block_until_ready(fn(*args))
    samples = []
    for _ in range(iters):
        start = time.perf_counter_ns()
        output = fn(*args)
        jax.block_until_ready(output)
        samples.append((time.perf_counter_ns() - start) / 1e6)
    return samples


def _summarize(variant, block_q, samples, baseline_ms=None):
    median_ms = statistics.median(samples)
    return Timing(
        variant=variant,
        block_q=block_q,
        median_ms=median_ms,
        mean_ms=statistics.fmean(samples),
        minimum_ms=min(samples),
        p90_ms=_percentile(samples, 90),
        speedup_vs_xla=None if baseline_ms is None else baseline_ms / median_ms,
    )


def _build_inputs(*, q, k, seq_lens, page_size, page_table_size, seed):
    rng = np.random.default_rng(seed)
    num_seqs = len(seq_lens)
    q_base, q_remainder = divmod(q, num_seqs)
    q_lens = np.full(num_seqs, q_base, dtype=np.int32)
    q_lens[:q_remainder] += 1
    cu_q_lens = np.concatenate(([0], np.cumsum(q_lens, dtype=np.int32))).astype(
        np.int32
    )

    seq_lens_np = np.asarray(seq_lens, dtype=np.int32)
    page_counts = (seq_lens_np + page_size - 1) // page_size
    aligned_kv_lens = page_counts * page_size
    cu_kv_lens = np.concatenate(
        ([0], np.cumsum(aligned_kv_lens, dtype=np.int32))
    ).astype(np.int32)
    used_pages = int(page_counts.sum())
    if used_pages > page_table_size:
        raise ValueError(
            f"page table has {page_table_size} entries but sequences need {used_pages}"
        )

    page_indices = np.full(page_table_size, -1, dtype=np.int32)
    page_indices[:used_pages] = rng.permutation(page_table_size)[:used_pages]
    topk = np.empty((q, k), dtype=np.int32)
    for seq_id in range(num_seqs):
        begin, end = int(cu_q_lens[seq_id]), int(cu_q_lens[seq_id + 1])
        topk[begin:end] = rng.integers(
            0,
            int(seq_lens_np[seq_id]),
            size=(end - begin, k),
            dtype=np.int32,
        )

    return tuple(
        jax.device_put(value)
        for value in (
            topk,
            seq_lens_np,
            page_indices,
            cu_q_lens,
            cu_kv_lens,
        )
    )


def _package_version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "MISSING"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--q", type=int, default=2048)
    parser.add_argument("--k", type=int, default=2048)
    parser.add_argument("--seq-lens", default="131584,132096")
    parser.add_argument("--page-size", type=int, default=64)
    parser.add_argument("--page-table-size", type=int, default=4224)
    parser.add_argument("--block-q", default="32,64,128,256")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--interpret", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.q = 8
        args.k = 128
        args.seq_lens = "253,317"
        args.page_size = 8
        args.page_table_size = 80
        args.block_q = "4,8"
        args.warmup = min(args.warmup, 1)
        args.iters = min(args.iters, 3)
    if args.interpret and jax.default_backend() == "tpu":
        raise SystemExit("--interpret is only intended for CPU smoke tests")
    if not args.interpret and jax.default_backend() != "tpu":
        raise SystemExit(
            "compiled Pallas benchmark requires TPU; use --smoke --interpret on CPU"
        )
    if args.warmup < 0 or args.iters <= 0:
        raise SystemExit("--warmup must be nonnegative and --iters must be positive")

    seq_lens = tuple(int(value) for value in args.seq_lens.split(","))
    block_q_values = tuple(int(value) for value in args.block_q.split(","))
    inputs = _build_inputs(
        q=args.q,
        k=args.k,
        seq_lens=seq_lens,
        page_size=args.page_size,
        page_table_size=args.page_table_size,
        seed=args.seed,
    )
    jax.block_until_ready(inputs)

    environment = {
        "event": "environment",
        "jax": _package_version("jax"),
        "jaxlib": _package_version("jaxlib"),
        "libtpu": _package_version("libtpu"),
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "q": args.q,
        "k": args.k,
        "seq_lens": seq_lens,
        "page_size": args.page_size,
        "page_table_size": args.page_table_size,
        "warmup": args.warmup,
        "iters": args.iters,
        "interpret": args.interpret,
    }
    print(json.dumps(environment, sort_keys=True), flush=True)

    xla_fn = jax.jit(
        functools.partial(logical_topk_to_physical_slots_xla, page_size=args.page_size)
    )
    xla_output = jax.block_until_ready(xla_fn(*inputs))
    xla_samples = _measure(xla_fn, inputs, warmup=args.warmup, iters=args.iters)
    xla_timing = _summarize("xla", None, xla_samples)
    print(
        json.dumps({"event": "timing", **asdict(xla_timing)}, sort_keys=True),
        flush=True,
    )

    for block_q in block_q_values:
        try:
            pallas_fn = jax.jit(
                functools.partial(
                    logical_topk_to_physical_slots_pallas,
                    page_size=args.page_size,
                    block_q=block_q,
                    interpret=args.interpret,
                )
            )
            pallas_output = jax.block_until_ready(pallas_fn(*inputs))
            slots_equal = bool(jnp.array_equal(xla_output[0], pallas_output[0]))
            counts_equal = bool(jnp.array_equal(xla_output[1], pallas_output[1]))
            correctness = {
                "event": "correctness",
                "variant": "pallas",
                "block_q": block_q,
                "slots_equal": slots_equal,
                "counts_equal": counts_equal,
            }
            print(json.dumps(correctness, sort_keys=True), flush=True)
            if not slots_equal or not counts_equal:
                raise AssertionError(f"Pallas output mismatch for block_q={block_q}")
            samples = _measure(pallas_fn, inputs, warmup=args.warmup, iters=args.iters)
            timing = _summarize(
                "pallas",
                block_q,
                samples,
                baseline_ms=xla_timing.median_ms,
            )
            print(
                json.dumps({"event": "timing", **asdict(timing)}, sort_keys=True),
                flush=True,
            )
        except Exception as error:  # Keep the sweep alive across invalid VMEM shapes.
            print(
                json.dumps(
                    {
                        "event": "candidate_error",
                        "variant": "pallas",
                        "block_q": block_q,
                        "error_type": type(error).__name__,
                        "error": str(error),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
