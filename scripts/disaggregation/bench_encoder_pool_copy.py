#!/usr/bin/env python3
"""Benchmark the device copy used by ``RaidenSendPool.copy_sync``.

This benchmark intentionally excludes pool creation, slot reservation, Raiden
registration, and network transfer.  It compiles the production
``_copy_into_slot`` operation once, warms it up, then measures the same
dispatch + ``block_until_ready`` sequence used on the serving path.

Examples:

  python scripts/disaggregation/bench_encoder_pool_copy.py

  python scripts/disaggregation/bench_encoder_pool_copy.py \
      --rows 324 --width 3584 --dtype bfloat16 --iterations 500

  python scripts/disaggregation/bench_encoder_pool_copy.py \
      --mesh-shape 2,2 --mesh-axis-names data,tensor
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import statistics
import time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.disaggregation.encoder.raiden_pool import (
    RaidenSendPool,
    _compile_donated_copy,
    _copy_into_slot,
    _pool_sharding,
)
from sgl_jax.srt.disaggregation.encoder.transfer_layout import encoder_pool_block_shape

_DTYPES = {
    "bfloat16": jnp.bfloat16,
    "float16": jnp.float16,
    "float32": jnp.float32,
}


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _partition_entry(value: str):
    if value.lower() in {"none", "-"}:
        return None
    axes = tuple(axis.strip() for axis in value.split("+") if axis.strip())
    return axes[0] if len(axes) == 1 else axes


def _embedding_sharding(args: argparse.Namespace) -> jax.sharding.Sharding:
    devices = jax.devices()
    if not args.mesh_shape:
        if not 0 <= args.device_index < len(devices):
            raise ValueError(f"device-index {args.device_index} is outside [0, {len(devices)})")
        return jax.sharding.SingleDeviceSharding(devices[args.device_index])

    mesh_shape = tuple(int(dim) for dim in _csv(args.mesh_shape))
    axis_names = tuple(_csv(args.mesh_axis_names))
    if not mesh_shape or any(dim <= 0 for dim in mesh_shape):
        raise ValueError("mesh-shape must contain positive dimensions")
    if len(mesh_shape) != len(axis_names):
        raise ValueError("mesh-shape and mesh-axis-names must have the same rank")
    if math.prod(mesh_shape) != len(devices):
        raise ValueError(
            f"mesh-shape contains {math.prod(mesh_shape)} devices, " f"but JAX sees {len(devices)}"
        )

    mesh = jax.sharding.Mesh(
        np.asarray(devices).reshape(mesh_shape),
        axis_names,
        axis_types=(jax.sharding.AxisType.Explicit,) * len(axis_names),
    )
    spec = jax.sharding.PartitionSpec(
        *(_partition_entry(item) for item in _csv(args.embedding_spec))
    )
    return jax.sharding.NamedSharding(mesh, spec)


def _percentile(sorted_values: list[float], percentile: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = percentile / 100.0 * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _mib(num_bytes: int) -> float:
    return num_bytes / (2**20)


def _gib_per_second(num_bytes: int, duration_ms: float) -> float:
    if duration_ms <= 0:
        return math.inf
    return num_bytes / (2**30) / (duration_ms / 1000.0)


@partial(jax.jit, donate_argnums=(0,))
def _copy_many_slots(
    pool: jax.Array,
    value: jax.Array,
    slots: jax.Array,
) -> jax.Array:
    """Run production slot updates in one device executable."""

    def copy_one(index: int, current_pool: jax.Array) -> jax.Array:
        return _copy_into_slot(current_pool, value, slots[index])

    return jax.lax.fori_loop(0, slots.shape[0], copy_one, pool)


def _compile_donated_copy_many(
    pool: jax.Array,
    value: jax.Array,
    slots: jax.Array,
):
    compiled = _copy_many_slots.lower(pool, value, slots).compile()
    stats = compiled.memory_analysis()
    stats = stats if isinstance(stats, (list, tuple)) else (stats,)
    if not stats or any(
        stat is None
        or int(getattr(stat, "alias_size_in_bytes", 0))
        < int(getattr(stat, "output_size_in_bytes", 0))
        for stat in stats
    ):
        raise RuntimeError("batched Raiden pool update did not alias its donated input")
    return compiled


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure the steady-state copy into one Raiden encoder pool slot."
    )
    parser.add_argument("--rows", type=int, default=324, help="Embedding token count.")
    parser.add_argument("--width", type=int, default=3584, help="Embedding hidden size.")
    parser.add_argument("--dtype", choices=sorted(_DTYPES), default="bfloat16")
    parser.add_argument("--pool-size", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument(
        "--iterations",
        type=int,
        default=200,
        help="Measured copies per worker.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Concurrent callers sharing the production copy lock.",
    )
    parser.add_argument(
        "--device-loop-steps",
        type=int,
        default=1,
        help=(
            "Copies to distinct slots per JIT dispatch. Values above one measure "
            "amortized device execution instead of production copy_sync latency."
        ),
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Optional JSONL path for one latency record per measured copy.",
    )
    parser.add_argument(
        "--device-index",
        type=int,
        default=0,
        help="Device used when no mesh is specified.",
    )
    parser.add_argument(
        "--mesh-shape",
        default="",
        help="Optional JAX mesh shape, for example 2,2.",
    )
    parser.add_argument(
        "--mesh-axis-names",
        default="",
        help="Mesh axis names matching --mesh-shape, for example dp,tp.",
    )
    parser.add_argument(
        "--embedding-spec",
        default="",
        help=(
            "Embedding PartitionSpec, for example dp,None. Use '+' to assign "
            "multiple mesh axes to one array dimension. Empty means replicated."
        ),
    )
    args = parser.parse_args()
    if args.rows <= 0 or args.width <= 0:
        parser.error("rows and width must be positive")
    if args.pool_size <= 0:
        parser.error("pool-size must be positive")
    if args.warmup < 0 or args.iterations <= 0:
        parser.error("warmup must be non-negative and iterations must be positive")
    if args.workers <= 0:
        parser.error("workers must be positive")
    if args.device_loop_steps <= 0:
        parser.error("device-loop-steps must be positive")
    if args.device_loop_steps > args.pool_size:
        parser.error("device-loop-steps cannot exceed pool-size")
    if args.device_loop_steps > 1 and args.workers != 1:
        parser.error("device-loop-steps above one requires workers=1")
    if bool(args.mesh_shape) != bool(args.mesh_axis_names):
        parser.error("mesh-shape and mesh-axis-names must be specified together")
    if args.embedding_spec and not args.mesh_shape:
        parser.error("embedding-spec requires mesh-shape")
    return args


def main() -> None:
    args = parse_args()
    dtype = _DTYPES[args.dtype]
    embedding_shape = (args.rows, args.width)
    block_shape = encoder_pool_block_shape(embedding_shape)
    embedding_sharding = _embedding_sharding(args)
    pool_sharding = _pool_sharding(embedding_sharding)

    embedding = jnp.ones(embedding_shape, dtype=dtype, device=embedding_sharding)
    pool_buffer = jnp.zeros(
        (args.pool_size, *block_shape),
        dtype=dtype,
        device=pool_sharding,
    )
    jax.block_until_ready((embedding, pool_buffer))

    # Construct only the copy-related state, bypassing Raiden initialization.
    # The timed call is the production RaidenSendPool.copy_sync method itself.
    pool = object.__new__(RaidenSendPool)
    pool.shape = embedding_shape
    pool.dtype = embedding.dtype
    pool.sharding = embedding.sharding
    pool._block_shape = block_shape
    pool._buffer = pool_buffer
    pool._copy = _compile_donated_copy(pool_buffer, embedding)

    slots = jax.device_put(
        np.arange(args.device_loop_steps, dtype=np.int32),
        embedding_sharding,
    )
    copy_many = None
    if args.device_loop_steps == 1:
        for index in range(args.warmup):
            pool.copy_sync(embedding, index % args.pool_size)
    else:
        copy_many = _compile_donated_copy_many(pool._buffer, embedding, slots)
        for _ in range(args.warmup):
            pool._buffer = copy_many(pool._buffer, embedding, slots)
            jax.block_until_ready(pool._buffer)

    start_wall_ns = time.perf_counter_ns()

    def measure(worker: int) -> list[float]:
        worker_measurements: list[float] = []
        for index in range(args.iterations):
            slot = (worker * args.iterations + index) % args.pool_size
            start_ns = time.perf_counter_ns()
            if copy_many is None:
                pool.copy_sync(embedding, slot)
            else:
                pool._buffer = copy_many(pool._buffer, embedding, slots)
                jax.block_until_ready(pool._buffer)
            batch_latency_ms = (time.perf_counter_ns() - start_ns) / 1_000_000
            worker_measurements.append(batch_latency_ms / args.device_loop_steps)
        return worker_measurements

    if args.workers == 1:
        measurements = measure(0)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            worker_measurements = executor.map(measure, range(args.workers))
            measurements = [
                measurement
                for worker_result in worker_measurements
                for measurement in worker_result
            ]
    wall_ms = (time.perf_counter_ns() - start_wall_ns) / 1_000_000
    durations_ms = list(measurements)

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        with args.json_output.open("w", encoding="utf-8") as output:
            for latency_ms in measurements:
                measurement = {
                    "schema_version": 1,
                    "variant": (
                        "production_copy_sync" if args.device_loop_steps == 1 else "device_loop"
                    ),
                    "rows": args.rows,
                    "width": args.width,
                    "dtype": args.dtype,
                    "pool_size": args.pool_size,
                    "workers": args.workers,
                    "device_loop_steps": args.device_loop_steps,
                    "device_count": len(jax.devices()),
                    "latency_ms": latency_ms,
                }
                output.write(json.dumps(measurement, sort_keys=True) + "\n")

    durations_ms.sort()
    logical_bytes = args.rows * args.width * jnp.dtype(dtype).itemsize
    padded_bytes = math.prod(block_shape) * jnp.dtype(dtype).itemsize
    p50_ms = _percentile(durations_ms, 50)

    print("Raiden encoder pool-slot copy benchmark")
    print(f"  platform:          {jax.default_backend()}")
    print(f"  devices:           {len(jax.devices())}")
    print(f"  embedding shape:   {embedding_shape}")
    print(f"  pool block shape:  {block_shape}")
    print(f"  dtype:             {jnp.dtype(dtype)}")
    print(f"  embedding shard:   {embedding.sharding}")
    print(f"  pool shard:        {pool._buffer.sharding}")
    print(f"  logical payload:   {_mib(logical_bytes):.3f} MiB")
    print(f"  padded slot:       {_mib(padded_bytes):.3f} MiB")
    print(f"  workers:           {args.workers}")
    print(f"  copies/dispatch:   {args.device_loop_steps}")
    print(f"  warmup/iterations: {args.warmup}/{args.iterations} per worker")
    total_copies = len(durations_ms) * args.device_loop_steps
    print(f"  total copies:      {total_copies}")
    print(f"  aggregate rate:    {total_copies / (wall_ms / 1000.0):.2f} copies/s")
    print(f"  mean:              {statistics.fmean(durations_ms):.4f} ms")
    print(f"  p50:               {p50_ms:.4f} ms")
    print(f"  p90:               {_percentile(durations_ms, 90):.4f} ms")
    print(f"  p99:               {_percentile(durations_ms, 99):.4f} ms")
    print(f"  min/max:           {durations_ms[0]:.4f}/{durations_ms[-1]:.4f} ms")
    print(f"  p50 padded BW:     {_gib_per_second(padded_bytes, p50_ms):.3f} GiB/s")


if __name__ == "__main__":
    main()
