"""Per-token Pallas ICI scatter microbenchmark.

This benchmark compares two schedules with the same token routing and payload:

* ``legacy`` uses one send semaphore and one receive semaphore, drains every
  ``BENCH_DRAIN_CHUNK`` assignments, and waits for the whole receive buffer.
* ``v2`` mirrors fused MoE v2 more closely: one send/receive semaphore per
  local expert, all assignment DMAs are issued before receive waits, receive
  waits happen expert-by-expert, and sends are drained per expert at the end.

Both variants issue one ``make_async_remote_copy`` per token/expert assignment.
The deterministic route is balanced across all experts while preserving the
real ``tokens x top_k`` source reuse: assignment ``r`` reads token
``r // top_k`` and routes to ``(r * 17) % num_experts``.

Default model dimensions match GLM-5.2:

* hidden size: 6144
* routed experts: 256
* top-k: 8
* EP size: inferred from the JAX device mesh (intended EP32)

The benchmark deliberately isolates scatter. It does not include routing
metadata construction, expert compute, or the block-granular return gather.
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

START_TIME = time.time()


def log(message: str) -> None:
    print(f"[{time.time() - START_TIME:.1f}s][p{jax.process_index()}] {message}", flush=True)


if os.environ.get("BENCH_SINGLE_HOST", "0") != "1":
    jax.distributed.initialize()


P = jax.sharding.PartitionSpec
DP, TP = "data", "tensor"

TOKENS = int(os.environ.get("BENCH_TOKENS", "512"))
TOP_K = int(os.environ.get("BENCH_TOP_K", "8"))
N = TOKENS * TOP_K
H = int(os.environ.get("BENCH_H", "6144"))
NUM_EXPERTS = int(os.environ.get("BENCH_NUM_EXPERTS", "256"))
DRAIN_CHUNK = int(os.environ.get("BENCH_DRAIN_CHUNK", "128"))
WARMUP = int(os.environ.get("BENCH_WARMUP", "3"))
ITERS = int(os.environ.get("BENCH_ITERS", "20"))
DTYPE = os.environ.get("BENCH_DTYPE", "both").lower()
VARIANTS = tuple(
    value.strip() for value in os.environ.get("BENCH_VARIANTS", "legacy,v2").split(",") if value
)
CHECK = os.environ.get("BENCH_CHECK", "1") == "1"
OUTPUT = os.environ.get("BENCH_OUTPUT")
ROUTE_MULTIPLIER = 17

NDEV = jax.device_count()
if NDEV < 2:
    raise SystemExit("per-token scatter requires at least two devices")
if NUM_EXPERTS % NDEV != 0:
    raise SystemExit(f"BENCH_NUM_EXPERTS={NUM_EXPERTS} must be divisible by device_count={NDEV}")
if N % NUM_EXPERTS != 0:
    raise SystemExit(f"tokens*top_k={N} must be divisible by num_experts={NUM_EXPERTS}")
if N % DRAIN_CHUNK != 0:
    raise SystemExit(f"assignments={N} must be divisible by BENCH_DRAIN_CHUNK={DRAIN_CHUNK}")
if math.gcd(ROUTE_MULTIPLIER, NUM_EXPERTS) != 1:
    raise SystemExit(
        f"route multiplier {ROUTE_MULTIPLIER} must be coprime with num_experts={NUM_EXPERTS}"
    )
if any(variant not in {"legacy", "v2"} for variant in VARIANTS):
    raise SystemExit(f"BENCH_VARIANTS must contain only legacy or v2, got {VARIANTS}")
if DTYPE not in {"bf16", "fp8", "both"}:
    raise SystemExit(f"BENCH_DTYPE must be bf16, fp8, or both, got {DTYPE}")

LOCAL_EXPERTS = NUM_EXPERTS // NDEV
ASSIGNMENTS_PER_EXPERT_PER_SOURCE = N // NUM_EXPERTS
ROWS_PER_LOCAL_EXPERT = ASSIGNMENTS_PER_EXPERT_PER_SOURCE * NDEV

devices = list(jax.devices())
coords = [getattr(device, "coords", None) for device in devices]
if all(coord is not None for coord in coords):
    chips = {tuple(int(value) for value in coord) for coord in coords}
    CORES_PER_CHIP = NDEV // len(chips)
else:
    CORES_PER_CHIP = 1
ICI_PEERS = NDEV - CORES_PER_CHIP

mesh = jax.sharding.Mesh(np.array(devices).reshape(1, NDEV), (DP, TP))
row_spec = P((DP, TP))


def _make_kernel(inner: tuple[int, ...], dtype: jnp.dtype, variant: str):
    use_v2_schedule = variant == "v2"

    def _kernel(x_ref, recv_ref, send_sem, recv_sem, barrier_sem):
        tp_size = lax.axis_size(TP)
        dp_size = lax.axis_size(DP)
        num_devices = tp_size * dp_size
        my_id = lax.axis_index(DP) * tp_size + lax.axis_index(TP)

        def mesh_id(rank):
            return (rank // tp_size, rank % tp_size)

        for device_id in range(num_devices):
            pltpu.semaphore_signal(
                barrier_sem,
                device_id=mesh_id(device_id),
                device_id_type=pltpu.DeviceIdType.MESH,
            )
        pltpu.semaphore_wait(barrier_sem, num_devices)

        with jax.named_scope(f"scatter_transfer_{variant}"):

            def fire_assignment(assignment_id, _):
                token_id = assignment_id // TOP_K
                expert_id = (assignment_id * ROUTE_MULTIPLIER) % NUM_EXPERTS
                destination = expert_id // LOCAL_EXPERTS
                local_expert = expert_id % LOCAL_EXPERTS
                occurrence = assignment_id // NUM_EXPERTS
                slot = my_id * ASSIGNMENTS_PER_EXPERT_PER_SOURCE + occurrence

                current_send_sem = send_sem.at[local_expert] if use_v2_schedule else send_sem
                current_recv_sem = recv_sem.at[local_expert] if use_v2_schedule else recv_sem
                pltpu.make_async_remote_copy(
                    src_ref=x_ref.at[pl.ds(token_id, 1)],
                    dst_ref=recv_ref.at[local_expert, pl.ds(slot, 1)],
                    send_sem=current_send_sem,
                    recv_sem=current_recv_sem,
                    device_id=mesh_id(destination),
                    device_id_type=pltpu.DeviceIdType.MESH,
                ).start()
                return None

            if use_v2_schedule:
                lax.fori_loop(0, N, fire_assignment, None, unroll=False)

                # Fused MoE v2 waits only for the expert that is about to
                # compute, rather than draining one global receive semaphore.
                for local_expert in range(LOCAL_EXPERTS):
                    expert_ref = recv_ref.at[
                        local_expert,
                        pl.ds(0, ROWS_PER_LOCAL_EXPERT),
                    ]
                    pltpu.make_async_copy(
                        src_ref=expert_ref,
                        dst_ref=expert_ref,
                        sem=recv_sem.at[local_expert],
                    ).wait()

                # The real kernel drains each expert's sends after its expert
                # loop; it does not force a drain every fixed number of rows.
                for local_expert in range(LOCAL_EXPERTS):
                    source_ref = x_ref.at[pl.ds(0, ROWS_PER_LOCAL_EXPERT)]
                    pltpu.make_async_copy(
                        src_ref=source_ref,
                        dst_ref=source_ref,
                        sem=send_sem.at[local_expert],
                    ).wait()
            else:
                for chunk_start in range(0, N, DRAIN_CHUNK):
                    lax.fori_loop(
                        chunk_start,
                        chunk_start + DRAIN_CHUNK,
                        fire_assignment,
                        None,
                        unroll=False,
                    )
                    source_ref = x_ref.at[pl.ds(0, DRAIN_CHUNK)]
                    pltpu.make_async_copy(
                        src_ref=source_ref,
                        dst_ref=source_ref,
                        sem=send_sem,
                    ).wait()

                recv_all_ref = recv_ref.at[
                    pl.ds(0, LOCAL_EXPERTS),
                    pl.ds(0, ROWS_PER_LOCAL_EXPERT),
                ]
                pltpu.make_async_copy(
                    src_ref=recv_all_ref,
                    dst_ref=recv_all_ref,
                    sem=recv_sem,
                ).wait()

        for device_id in range(num_devices):
            pltpu.semaphore_signal(
                barrier_sem,
                device_id=mesh_id(device_id),
                device_id_type=pltpu.DeviceIdType.MESH,
            )
        pltpu.semaphore_wait(barrier_sem, num_devices)

    semaphore_shape = (
        pltpu.SemaphoreType.DMA((LOCAL_EXPERTS,)) if use_v2_schedule else pltpu.SemaphoreType.DMA
    )
    return pl.pallas_call(
        _kernel,
        out_shape=jax.ShapeDtypeStruct(
            (LOCAL_EXPERTS, ROWS_PER_LOCAL_EXPERT, *inner),
            dtype,
        ),
        in_specs=[pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)],
        out_specs=pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
        scratch_shapes=[
            semaphore_shape,
            semaphore_shape,
            pltpu.SemaphoreType.BARRIER,
        ],
        compiler_params=pltpu.CompilerParams(
            collective_id=710_001 if use_v2_schedule else 710_000,
            allow_collective_id_without_custom_barrier=True,
            has_side_effects=True,
        ),
        name=f"scatter_pertoken_{variant}",
    )


def _runner(inner: tuple[int, ...], dtype: jnp.dtype, variant: str):
    kernel = _make_kernel(inner, dtype, variant)

    @jax.jit
    @jax.shard_map(mesh=mesh, in_specs=(row_spec,), out_specs=row_spec, check_vma=False)
    def run(x):
        return kernel(x)

    return run


def _make_input(inner: tuple[int, ...], dtype: jnp.dtype) -> jax.Array:
    key = jax.random.key(0)
    shards = []
    for local_index, device in enumerate(jax.local_devices()):
        global_index = jax.process_index() * len(jax.local_devices()) + local_index
        shard_key = jax.random.fold_in(key, global_index)
        shards.append(
            jax.device_put(
                jax.random.normal(shard_key, (TOKENS, *inner), jnp.float32).astype(dtype),
                device,
            )
        )
    return jax.make_array_from_single_device_arrays(
        (NDEV * TOKENS, *inner),
        jax.sharding.NamedSharding(mesh, row_spec),
        shards,
    )


def _verify_variants() -> None:
    if set(VARIANTS) != {"legacy", "v2"}:
        log("correctness comparison skipped because both legacy and v2 were not requested")
        return

    hidden = 128
    dtype = jnp.int32
    inner = (hidden,)
    x = _make_input(inner, dtype)
    legacy = _runner(inner, dtype, "legacy")(x)
    candidate = _runner(inner, dtype, "v2")(x)

    @jax.jit
    def max_error(left, right):
        return jnp.max(jnp.abs(left.astype(jnp.int32) - right.astype(jnp.int32)))

    error = max_error(legacy, candidate)
    jax.block_until_ready(error)
    local_error = int(np.asarray(error.addressable_shards[0].data))
    log(f"correctness legacy_vs_v2 max_error={local_error}")
    if local_error != 0:
        raise AssertionError(f"legacy and v2 scatter outputs differ: max_error={local_error}")


def _percentile(samples: list[float], quantile: float) -> float:
    return float(np.percentile(np.asarray(samples, dtype=np.float64), quantile * 100.0))


def _benchmark(dtype_name: str, variant: str) -> dict[str, object]:
    dtype = jnp.bfloat16 if dtype_name == "bf16" else jnp.float8_e4m3fn
    bytes_per_element = jnp.dtype(dtype).itemsize
    token_packing = 32 // (bytes_per_element * 8)
    hidden_per_tile = H // token_packing
    inner = (token_packing, hidden_per_tile)
    run = _runner(inner, dtype, variant)
    x = _make_input(inner, dtype)

    for _ in range(WARMUP):
        jax.block_until_ready(run(x))

    samples_ms: list[float] = []
    for _ in range(ITERS):
        start = time.monotonic()
        output = run(x)
        jax.block_until_ready(output)
        samples_ms.append((time.monotonic() - start) * 1e3)

    p50_ms = _percentile(samples_ms, 0.50)
    p90_ms = _percentile(samples_ms, 0.90)
    payload_bytes = N * H * bytes_per_element
    off_diagonal_bytes = payload_bytes * (NDEV - 1) // NDEV
    ici_bytes = payload_bytes * ICI_PEERS // NDEV
    result: dict[str, object] = {
        "variant": variant,
        "dtype": dtype_name,
        "ep": NDEV,
        "chips": NDEV // CORES_PER_CHIP,
        "cores_per_chip": CORES_PER_CHIP,
        "tokens": TOKENS,
        "top_k": TOP_K,
        "assignments": N,
        "num_experts": NUM_EXPERTS,
        "local_experts": LOCAL_EXPERTS,
        "hidden": H,
        "message_bytes": H * bytes_per_element,
        "drain_chunk": DRAIN_CHUNK if variant == "legacy" else None,
        "latency_ms_mean": round(float(np.mean(samples_ms)), 6),
        "latency_ms_p50": round(p50_ms, 6),
        "latency_ms_p90": round(p90_ms, 6),
        "remote_payload_bytes_per_device": off_diagonal_bytes,
        "ici_bytes_per_device": ici_bytes,
        "remote_payload_gb_s_per_device": round(off_diagonal_bytes / (p50_ms * 1e6), 6),
        "ici_useful_gb_s_per_device": round(ici_bytes / (p50_ms * 1e6), 6),
        "samples_ms": [round(sample, 6) for sample in samples_ms],
        "process_index": jax.process_index(),
        "source_commit": os.environ.get("FALCON_SOURCE_COMMIT", ""),
        "run_id": os.environ.get("FALCON_EXP_ID", ""),
    }
    log(json.dumps(result, sort_keys=True))
    return result


def _write_results(results: list[dict[str, object]]) -> None:
    if not OUTPUT:
        return
    output_path = Path(OUTPUT)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as output_file:
        for result in results:
            output_file.write(json.dumps(result, sort_keys=True) + "\n")
    log(f"wrote metrics to {output_path}")


def main() -> None:
    log(
        f"device={devices[0].device_kind} ep={NDEV} processes={jax.process_count()} "
        f"tokens={TOKENS} top_k={TOP_K} assignments={N} H={H} "
        f"experts={NUM_EXPERTS} local_experts={LOCAL_EXPERTS} "
        f"variants={VARIANTS}"
    )
    if CHECK:
        _verify_variants()

    dtype_names = ("bf16", "fp8") if DTYPE == "both" else (DTYPE,)
    results = [
        _benchmark(dtype_name, variant) for dtype_name in dtype_names for variant in VARIANTS
    ]
    _write_results(results)
    log("done")


if __name__ == "__main__":
    main()
