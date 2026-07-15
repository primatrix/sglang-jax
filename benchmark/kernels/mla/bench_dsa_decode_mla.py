"""TPU microbenchmark for the staged sparse DSA decode MLA kernel.

The dense variant intentionally scans every token in the same packed cache.
It is a workload baseline rather than the production MLA-v2 kernel: the two
variants have different attention domains (Top-K versus full context), but
identical cache, query, precision, and launch environment.

Examples:

  python benchmark/kernels/mla/bench_dsa_decode_mla.py \
    --batch-size 1 --context-length 160000 --top-k 2048

  # Capture an XProf trace after normal timing warm-ups.
  python benchmark/kernels/mla/bench_dsa_decode_mla.py \
    --batch-size 32 --context-length 32000 --top-k 2048 \
    --profile --profile-dir /tmp/dsa-profile-b32
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import time
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.kernels.mla.dsa import (
    SPARSECORE_COMPILER_OPTIONS,
    dsa_decode_mla_attention_unchecked,
    materialize_selected_kv_sparsecore_unchecked,
    materialize_selected_kv_xla,
    prepare_safe_topk_slots,
    selected_mla_attention_unchecked,
)


_ALIGNMENT = 128
GLM_ATTENTION_SCALE = 256**-0.5
BENCHMARK_VARIANTS = (
    "sparsecore-pipeline",
    "sparsecore",
    "xla-gather",
    "gather-only",
    "attention-only",
    "dense-jax-baseline",
)


@dataclasses.dataclass(frozen=True)
class BenchmarkInputs:
    """Host fixture shared by sparse and full-context benchmark variants."""

    ql_nope: np.ndarray
    q_pe: np.ndarray
    cache_kv: np.ndarray
    topk_slots: np.ndarray
    valid_counts: np.ndarray


def _align_to_128(dim: int) -> int:
    return ((dim + _ALIGNMENT - 1) // _ALIGNMENT) * _ALIGNMENT


def estimate_variant_kv_bytes(
    *,
    batch_size: int,
    context_length: int,
    top_k: int,
    cache_width: int,
    itemsize: int,
) -> dict[str, int]:
    """Estimate major selected/full-cache payload bytes for each variant."""
    selected_bytes = batch_size * top_k * cache_width * itemsize
    dense_bytes = batch_size * context_length * cache_width * itemsize
    return {
        "selected_tensor": selected_bytes,
        "sparsecore-pipeline": 3 * selected_bytes,
        "sparsecore": 3 * selected_bytes,
        "xla-gather": 3 * selected_bytes,
        "gather-only": 2 * selected_bytes,
        "attention-only": selected_bytes,
        "dense-jax-baseline": dense_bytes,
    }


def make_benchmark_inputs(
    *,
    batch_size: int,
    context_length: int,
    top_k: int,
    num_heads: int,
    latent_dim: int,
    rope_dim: int,
    page_size: int,
    slot_order: str,
    slot_distribution: str = "uniform",
    seed: int = 0,
) -> BenchmarkInputs:
    """Create deterministic packed-cache inputs with physical selected slots."""
    dimensions = {
        "batch_size": batch_size,
        "context_length": context_length,
        "top_k": top_k,
        "num_heads": num_heads,
        "latent_dim": latent_dim,
        "rope_dim": rope_dim,
        "page_size": page_size,
    }
    invalid = [name for name, value in dimensions.items() if value <= 0]
    if invalid:
        raise ValueError(f"all benchmark dimensions must be positive: {', '.join(invalid)}")
    if top_k > context_length:
        raise ValueError("top_k must not exceed context_length")
    if page_size % 2 or (page_size % _ALIGNMENT and _ALIGNMENT % page_size):
        raise ValueError("page_size must be even and divide 128 or be divisible by 128")
    if slot_order not in {"unsorted", "page-sorted"}:
        raise ValueError("slot_order must be 'unsorted' or 'page-sorted'")
    if slot_distribution not in {"uniform", "clustered"}:
        raise ValueError("slot_distribution must be 'uniform' or 'clustered'")

    padded_width = _align_to_128(latent_dim) + _align_to_128(rope_dim)
    num_pages = (context_length + page_size - 1) // page_size
    rng = np.random.default_rng(seed)
    ql_nope = rng.standard_normal((batch_size, num_heads, latent_dim), dtype=np.float32)
    q_pe = rng.standard_normal((batch_size, num_heads, rope_dim), dtype=np.float32)
    cache_kv = rng.standard_normal(
        (num_pages, page_size // 2, 2, padded_width), dtype=np.float32
    )

    # Each batch row has the same multiset; the distribution controls locality
    # and slot_order controls only the caller-visible order.
    if slot_distribution == "uniform":
        base_slots = (np.arange(top_k, dtype=np.int32) * context_length) // top_k
    else:
        base_slots = np.arange(top_k, dtype=np.int32)
    topk_slots = np.empty((batch_size, top_k), dtype=np.int32)
    for batch_index in range(batch_size):
        if slot_order == "page-sorted":
            topk_slots[batch_index] = base_slots
        else:
            topk_slots[batch_index] = np.roll(base_slots, batch_index + 1)

    return BenchmarkInputs(
        ql_nope=ql_nope,
        q_pe=q_pe,
        cache_kv=cache_kv,
        topk_slots=topk_slots,
        valid_counts=np.full((batch_size,), top_k, dtype=np.int32),
    )


def dense_full_context_mla_attention(
    ql_nope: jax.Array,
    q_pe: jax.Array,
    cache_kv: jax.Array,
    *,
    context_length: int,
    sm_scale: float,
) -> jax.Array:
    """Full-context dense MLA baseline over the same packed physical cache."""
    output_dtype = ql_nope.dtype
    latent_dim = ql_nope.shape[-1]
    rope_dim = q_pe.shape[-1]
    padded_latent_dim = _align_to_128(latent_dim)
    padded_rope_dim = _align_to_128(rope_dim)
    cache = cache_kv.reshape((-1, cache_kv.shape[-1]))[:context_length].astype(jnp.float32)
    ql_nope = ql_nope.astype(jnp.float32)
    q_pe = q_pe.astype(jnp.float32)
    if latent_dim != padded_latent_dim:
        ql_nope = jnp.pad(ql_nope, ((0, 0), (0, 0), (0, padded_latent_dim - latent_dim)))
    if rope_dim != padded_rope_dim:
        q_pe = jnp.pad(q_pe, ((0, 0), (0, 0), (0, padded_rope_dim - rope_dim)))

    logits = jnp.einsum("bhl,tl->bht", ql_nope, cache[:, :padded_latent_dim])
    logits += jnp.einsum("bhr,tr->bht", q_pe, cache[:, padded_latent_dim:])
    probabilities = jax.nn.softmax(logits * jnp.float32(sm_scale), axis=-1)
    output = jnp.einsum("bht,tl->bhl", probabilities, cache[:, :latent_dim])
    return output.astype(output_dtype)


def _time_compiled(
    compute: Callable[[], jax.Array], *, warmup_iters: int, iters: int
) -> dict[str, float]:
    for _ in range(warmup_iters):
        jax.block_until_ready(compute())

    latency_ms = []
    for _ in range(iters):
        start = time.perf_counter_ns()
        jax.block_until_ready(compute())
        latency_ms.append((time.perf_counter_ns() - start) / 1_000_000.0)

    return {
        "median_ms": float(np.median(latency_ms)),
        "p99_ms": float(np.percentile(latency_ms, 99)),
        "mean_ms": float(np.mean(latency_ms)),
        "min_ms": float(np.min(latency_ms)),
    }


def _compile_variant(
    compute: Callable[[], jax.Array],
    *,
    compiler_options: dict[str, str] | None = None,
) -> tuple[Callable[[], jax.Array], float]:
    start = time.perf_counter_ns()
    compiled = jax.jit(
        compute,
        compiler_options=compiler_options,
    ).lower().compile()
    compile_ms = (time.perf_counter_ns() - start) / 1_000_000.0
    return compiled, compile_ms


def _capture_profile(
    compute: Callable[[], jax.Array], *, profile_dir: str, iters: int, name: str) -> None:
    pathlib.Path(profile_dir).mkdir(parents=True, exist_ok=True)
    with jax.profiler.trace(profile_dir):
        for step in range(iters):
            with jax.profiler.StepTraceAnnotation(name, step_num=step):
                jax.block_until_ready(compute())


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--context-length", type=int, default=160_000)
    parser.add_argument("--top-k", type=int, default=2048)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--latent-dim", type=int, default=512)
    parser.add_argument("--rope-dim", type=int, default=64)
    parser.add_argument("--page-size", type=int, default=128)
    parser.add_argument(
        "--sm-scale",
        type=float,
        default=None,
        help="Attention scale; defaults to GLM's unabsorbed 256-d QK scale.",
    )
    parser.add_argument("--slot-order", choices=("unsorted", "page-sorted"), default="unsorted")
    parser.add_argument(
        "--slot-distribution",
        choices=("uniform", "clustered"),
        default="uniform",
    )
    parser.add_argument(
        "--variant", choices=("all", *BENCHMARK_VARIANTS), default="all"
    )
    parser.add_argument("--warmup-iters", type=int, default=50)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-iters", type=int, default=3)
    parser.add_argument(
        "--profile-variant", choices=BENCHMARK_VARIANTS, default="sparsecore"
    )
    parser.add_argument("--profile-dir", default="/tmp/dsa-decode-mla-profile")
    parser.add_argument("--output", help="Optional path for the JSON summary.")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if jax.default_backend() != "tpu":
        raise RuntimeError("bench_dsa_decode_mla.py must run on a TPU")
    if args.warmup_iters < 0 or args.iters <= 0 or args.profile_iters <= 0:
        raise ValueError("warmup-iters must be nonnegative; iters and profile-iters must be positive")

    host_inputs = make_benchmark_inputs(
        batch_size=args.batch_size,
        context_length=args.context_length,
        top_k=args.top_k,
        num_heads=args.num_heads,
        latent_dim=args.latent_dim,
        rope_dim=args.rope_dim,
        page_size=args.page_size,
        slot_order=args.slot_order,
        slot_distribution=args.slot_distribution,
        seed=args.seed,
    )
    ql_nope = jnp.asarray(host_inputs.ql_nope, dtype=jnp.bfloat16)
    q_pe = jnp.asarray(host_inputs.q_pe, dtype=jnp.bfloat16)
    cache_kv = jnp.asarray(host_inputs.cache_kv, dtype=jnp.bfloat16)
    topk_slots = jnp.asarray(host_inputs.topk_slots, dtype=jnp.int32)
    valid_counts = jnp.asarray(host_inputs.valid_counts, dtype=jnp.int32)
    sm_scale = GLM_ATTENTION_SCALE if args.sm_scale is None else args.sm_scale
    if not np.isfinite(sm_scale):
        raise ValueError("sm-scale must be finite")

    selected_kv = materialize_selected_kv_xla(
        cache_kv, topk_slots, valid_counts, gather_block=128
    )
    safe_topk_slots = prepare_safe_topk_slots(
        topk_slots, valid_counts, gather_block=128
    )
    jax.block_until_ready(selected_kv)

    variants: dict[str, Callable[[], jax.Array]] = {
        "sparsecore-pipeline": lambda: dsa_decode_mla_attention_unchecked(
            ql_nope,
            q_pe,
            cache_kv,
            topk_slots,
            valid_counts,
            sm_scale=sm_scale,
            gather_impl="sparsecore-pipeline",
        ),
        "sparsecore": lambda: dsa_decode_mla_attention_unchecked(
            ql_nope,
            q_pe,
            cache_kv,
            topk_slots,
            valid_counts,
            sm_scale=sm_scale,
            gather_impl="sparsecore",
        ),
        "xla-gather": lambda: dsa_decode_mla_attention_unchecked(
            ql_nope,
            q_pe,
            cache_kv,
            topk_slots,
            valid_counts,
            sm_scale=sm_scale,
            gather_impl="xla",
        ),
        "gather-only": lambda: materialize_selected_kv_sparsecore_unchecked(
            cache_kv,
            safe_topk_slots,
            gather_block=128,
        ),
        "attention-only": lambda: selected_mla_attention_unchecked(
            ql_nope,
            q_pe,
            selected_kv,
            valid_counts,
            sm_scale=sm_scale,
        ),
        "dense-jax-baseline": lambda: dense_full_context_mla_attention(
            ql_nope,
            q_pe,
            cache_kv,
            context_length=args.context_length,
            sm_scale=sm_scale,
        ),
    }
    chosen = BENCHMARK_VARIANTS if args.variant == "all" else (args.variant,)
    cache_width = cache_kv.shape[-1]
    kv_byte_estimates = estimate_variant_kv_bytes(
        batch_size=args.batch_size,
        context_length=args.context_length,
        top_k=args.top_k,
        cache_width=cache_width,
        itemsize=2,
    )

    summary: dict[str, object] = {
        "backend": jax.default_backend(),
        "jax_version": jax.__version__,
        "devices": [str(device) for device in jax.devices()],
        "input": {
            "batch_size": args.batch_size,
            "context_length": args.context_length,
            "top_k": args.top_k,
            "num_heads": args.num_heads,
            "latent_dim": args.latent_dim,
            "rope_dim": args.rope_dim,
            "page_size": args.page_size,
            "sm_scale": sm_scale,
            "slot_order": args.slot_order,
            "slot_distribution": args.slot_distribution,
            "dtype": "bfloat16",
        },
        "estimated_kv_bytes": kv_byte_estimates,
        "warmup_iters": args.warmup_iters,
        "timed_iters": args.iters,
        "results": {},
    }
    compiled_variants: dict[str, Callable[[], jax.Array]] = {}
    for variant in chosen:
        compiler_options = (
            SPARSECORE_COMPILER_OPTIONS
            if variant
            in {
                "sparsecore-pipeline",
                "sparsecore",
                "gather-only",
            }
            else None
        )
        compiled, compile_ms = _compile_variant(
            variants[variant], compiler_options=compiler_options
        )
        compiled_variants[variant] = compiled
        metrics = _time_compiled(
            compiled, warmup_iters=args.warmup_iters, iters=args.iters
        )
        metrics["compile_ms"] = compile_ms
        summary["results"][variant] = metrics
        print(
            f"{variant}: compile={compile_ms:.2f} ms "
            f"median={metrics['median_ms']:.4f} ms p99={metrics['p99_ms']:.4f} ms"
        )

    if args.profile:
        if args.profile_variant not in chosen:
            raise ValueError("profile-variant must be included in --variant")
        _capture_profile(
            compiled_variants[args.profile_variant],
            profile_dir=args.profile_dir,
            iters=args.profile_iters,
            name=f"dsa_decode_mla_{args.profile_variant}",
        )
        summary["profile"] = {
            "variant": args.profile_variant,
            "iters": args.profile_iters,
            "dir": str(pathlib.Path(args.profile_dir).resolve()),
        }

    encoded = json.dumps(summary, indent=2, sort_keys=True)
    print(encoded)
    if args.output:
        output_path = pathlib.Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
