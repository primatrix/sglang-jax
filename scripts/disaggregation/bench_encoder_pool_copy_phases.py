#!/usr/bin/env python3
"""Decompose host-visible latency around a Raiden encoder pool-slot update."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections import defaultdict
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


def _elapsed_ms(start_ns: int, end_ns: int) -> float:
    return (end_ns - start_ns) / 1_000_000


def _percentile(sorted_values: list[float], percentile: float) -> float:
    position = percentile / 100.0 * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _record(
    records: list[dict[str, object]],
    *,
    variant: str,
    phase: str,
    latency_ms: float,
) -> None:
    records.append(
        {
            "schema_version": 1,
            "variant": variant,
            "phase": phase,
            "rows": 324,
            "width": 3584,
            "dtype": "bfloat16",
            "device_count": len(jax.devices()),
            "latency_ms": latency_ms,
        }
    )


@partial(jax.jit, donate_argnums=(0,))
def _copy_many_slots(
    pool: jax.Array,
    value: jax.Array,
    slots: jax.Array,
) -> jax.Array:
    def copy_one(index: int, current_pool: jax.Array) -> jax.Array:
        return _copy_into_slot(current_pool, value, slots[index])

    return jax.lax.fori_loop(0, slots.shape[0], copy_one, pool)


def _new_pool(
    embedding: jax.Array,
    capacity: int,
) -> RaidenSendPool:
    block_shape = encoder_pool_block_shape(embedding.shape)
    pool_buffer = jnp.zeros(
        (capacity, *block_shape),
        dtype=embedding.dtype,
        device=_pool_sharding(embedding.sharding),
    )
    jax.block_until_ready(pool_buffer)
    pool = object.__new__(RaidenSendPool)
    pool.shape = tuple(embedding.shape)
    pool.dtype = embedding.dtype
    pool.sharding = embedding.sharding
    pool._block_shape = block_shape
    pool._buffer = pool_buffer
    pool._copy = _compile_donated_copy(pool_buffer, embedding)
    return pool


def _measure_python_only(
    pool: RaidenSendPool,
    embedding: jax.Array,
    iterations: int,
    records: list[dict[str, object]],
) -> None:
    for _ in range(iterations):
        start_ns = time.perf_counter_ns()
        if not pool.matches(embedding):
            raise AssertionError("unexpected pool mismatch")
        match_done_ns = time.perf_counter_ns()
        _record(
            records,
            variant="python_only",
            phase="matches",
            latency_ms=_elapsed_ms(start_ns, match_done_ns),
        )
        _record(
            records,
            variant="python_only",
            phase="total",
            latency_ms=_elapsed_ms(start_ns, match_done_ns),
        )


def _measure_slot_scalar(
    iterations: int,
    records: list[dict[str, object]],
) -> None:
    for index in range(iterations):
        start_ns = time.perf_counter_ns()
        slot = jnp.asarray(index % 32, dtype=jnp.int32)
        call_done_ns = time.perf_counter_ns()
        jax.block_until_ready(slot)
        ready_ns = time.perf_counter_ns()
        _record(
            records,
            variant="slot_scalar_only",
            phase="jnp_asarray_host_call",
            latency_ms=_elapsed_ms(start_ns, call_done_ns),
        )
        _record(
            records,
            variant="slot_scalar_only",
            phase="ready_wait",
            latency_ms=_elapsed_ms(call_done_ns, ready_ns),
        )
        _record(
            records,
            variant="slot_scalar_only",
            phase="total",
            latency_ms=_elapsed_ms(start_ns, ready_ns),
        )


def _measure_instrumented_copy(
    pool: RaidenSendPool,
    embedding: jax.Array,
    iterations: int,
    records: list[dict[str, object]],
    *,
    precreated_slots: list[jax.Array] | None,
) -> None:
    variant = "precreated_slot" if precreated_slots is not None else "dynamic_slot"
    for index in range(iterations):
        start_ns = time.perf_counter_ns()
        if not pool.matches(embedding):
            raise AssertionError("unexpected pool mismatch")
        match_done_ns = time.perf_counter_ns()
        if precreated_slots is None:
            slot = jnp.asarray(index % 32, dtype=jnp.int32)
        else:
            slot = precreated_slots[index % len(precreated_slots)]
        slot_ready_ns = time.perf_counter_ns()
        pool._buffer, ready = pool._copy(pool._buffer, embedding, slot)
        dispatch_done_ns = time.perf_counter_ns()
        ready.block_until_ready()
        ready_ns = time.perf_counter_ns()

        phases = {
            "matches": (start_ns, match_done_ns),
            "slot_prepare": (match_done_ns, slot_ready_ns),
            "executable_call": (slot_ready_ns, dispatch_done_ns),
            "ready_wait": (dispatch_done_ns, ready_ns),
            "total": (start_ns, ready_ns),
        }
        for phase, (phase_start_ns, phase_end_ns) in phases.items():
            _record(
                records,
                variant=variant,
                phase=phase,
                latency_ms=_elapsed_ms(phase_start_ns, phase_end_ns),
            )


def _measure_production(
    pool: RaidenSendPool,
    embedding: jax.Array,
    iterations: int,
    records: list[dict[str, object]],
) -> None:
    for index in range(iterations):
        start_ns = time.perf_counter_ns()
        pool.copy_sync(embedding, index % 32)
        done_ns = time.perf_counter_ns()
        _record(
            records,
            variant="production_copy_sync",
            phase="total",
            latency_ms=_elapsed_ms(start_ns, done_ns),
        )


def _measure_dispatch_chain(
    pool: RaidenSendPool,
    embedding: jax.Array,
    slots: list[jax.Array],
    iterations: int,
    records: list[dict[str, object]],
) -> None:
    steps = len(slots)
    for _ in range(iterations):
        start_ns = time.perf_counter_ns()
        ready = None
        for slot in slots:
            pool._buffer, ready = pool._copy(pool._buffer, embedding, slot)
        dispatch_done_ns = time.perf_counter_ns()
        assert ready is not None
        ready.block_until_ready()
        ready_ns = time.perf_counter_ns()
        phases = {
            "executable_calls": _elapsed_ms(start_ns, dispatch_done_ns) / steps,
            "final_ready_wait": _elapsed_ms(dispatch_done_ns, ready_ns) / steps,
            "total": _elapsed_ms(start_ns, ready_ns) / steps,
        }
        for phase, latency_ms in phases.items():
            _record(
                records,
                variant="dispatch_chain_32",
                phase=phase,
                latency_ms=latency_ms,
            )


def _measure_device_loop(
    pool: RaidenSendPool,
    embedding: jax.Array,
    slots: jax.Array,
    warmup: int,
    iterations: int,
    records: list[dict[str, object]],
) -> None:
    compiled = _copy_many_slots.lower(pool._buffer, embedding, slots).compile()
    for _ in range(warmup):
        pool._buffer = compiled(pool._buffer, embedding, slots)
        jax.block_until_ready(pool._buffer)

    steps = int(slots.shape[0])
    for _ in range(iterations):
        start_ns = time.perf_counter_ns()
        pool._buffer = compiled(pool._buffer, embedding, slots)
        dispatch_done_ns = time.perf_counter_ns()
        jax.block_until_ready(pool._buffer)
        ready_ns = time.perf_counter_ns()
        phases = {
            "executable_call": _elapsed_ms(start_ns, dispatch_done_ns) / steps,
            "ready_wait": _elapsed_ms(dispatch_done_ns, ready_ns) / steps,
            "total": _elapsed_ms(start_ns, ready_ns) / steps,
        }
        for phase, latency_ms in phases.items():
            _record(
                records,
                variant="device_loop_32",
                phase=phase,
                latency_ms=latency_ms,
            )


def _print_summary(records: list[dict[str, object]]) -> None:
    grouped: defaultdict[tuple[str, str], list[float]] = defaultdict(list)
    for record in records:
        grouped[(str(record["variant"]), str(record["phase"]))].append(float(record["latency_ms"]))

    print("variant,phase,count,mean_ms,p50_ms,p90_ms,p99_ms")
    for (variant, phase), values in sorted(grouped.items()):
        values.sort()
        print(
            f"{variant},{phase},{len(values)},{statistics.fmean(values):.6f},"
            f"{_percentile(values, 50):.6f},{_percentile(values, 90):.6f},"
            f"{_percentile(values, 99):.6f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--batch-iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.iterations <= 0 or args.batch_iterations <= 0 or args.warmup < 0:
        parser.error("iterations must be positive and warmup non-negative")
    return args


def main() -> None:
    args = parse_args()
    devices = np.asarray(jax.devices()).reshape((1, len(jax.devices())))
    mesh = jax.sharding.Mesh(
        devices,
        ("data", "tensor"),
        axis_types=(jax.sharding.AxisType.Explicit,) * 2,
    )
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    embedding = jnp.ones((324, 3584), dtype=jnp.bfloat16, device=sharding)
    jax.block_until_ready(embedding)

    precreated_slots = [jnp.asarray(index, dtype=jnp.int32) for index in range(32)]
    jax.block_until_ready(precreated_slots)
    device_slots = jax.device_put(np.arange(32, dtype=np.int32), sharding)
    jax.block_until_ready(device_slots)

    records: list[dict[str, object]] = []
    python_pool = _new_pool(embedding, 32)
    _measure_python_only(python_pool, embedding, args.iterations, records)
    _measure_slot_scalar(args.iterations, records)

    dynamic_pool = _new_pool(embedding, 32)
    for index in range(args.warmup):
        dynamic_pool.copy_sync(embedding, index % 32)
    _measure_instrumented_copy(
        dynamic_pool,
        embedding,
        args.iterations,
        records,
        precreated_slots=None,
    )

    precreated_pool = _new_pool(embedding, 32)
    for index in range(args.warmup):
        precreated_pool._buffer, ready = precreated_pool._copy(
            precreated_pool._buffer,
            embedding,
            precreated_slots[index % 32],
        )
        ready.block_until_ready()
    _measure_instrumented_copy(
        precreated_pool,
        embedding,
        args.iterations,
        records,
        precreated_slots=precreated_slots,
    )

    production_pool = _new_pool(embedding, 32)
    for index in range(args.warmup):
        production_pool.copy_sync(embedding, index % 32)
    _measure_production(production_pool, embedding, args.iterations, records)

    chain_pool = _new_pool(embedding, 32)
    _measure_dispatch_chain(
        chain_pool,
        embedding,
        precreated_slots,
        args.batch_iterations,
        records,
    )

    device_loop_pool = _new_pool(embedding, 32)
    _measure_device_loop(
        device_loop_pool,
        embedding,
        device_slots,
        args.warmup,
        args.batch_iterations,
        records,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, sort_keys=True) + "\n")

    print(f"platform={jax.default_backend()} devices={len(jax.devices())}")
    print(f"embedding_shape={embedding.shape} pool_capacity=32")
    _print_summary(records)


if __name__ == "__main__":
    main()
