"""Small-token fused-MoE v2 scatter/gather communication benchmark.

This benchmark isolates the production communication granularity:

* scatter issues one DMA per token/expert assignment;
* gather issues one variable-sized block DMA per (local expert, source device).

Routes are seeded and top-k unique, so small-token cases naturally contain empty
experts and mostly one-row gather blocks. The benchmark does not include routing
metadata construction, expert compute, or output accumulation.
"""

from __future__ import annotations

import functools
import json
import os
import time
from dataclasses import dataclass
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

TOKENS_CASES = tuple(
    int(value)
    for value in os.environ.get("BENCH_TOKENS_PER_DEVICE", "2,4,8,16,32").split(",")
    if value
)
ROUTE_SEEDS = tuple(
    int(value) for value in os.environ.get("BENCH_ROUTE_SEEDS", "0,1,2").split(",") if value
)
CHECK_TOKENS = tuple(
    int(value) for value in os.environ.get("BENCH_CHECK_TOKENS", "2,16").split(",") if value
)
TOP_K = int(os.environ.get("BENCH_TOP_K", "8"))
H = int(os.environ.get("BENCH_H", "6144"))
NUM_EXPERTS = int(os.environ.get("BENCH_NUM_EXPERTS", "256"))
WARMUP = int(os.environ.get("BENCH_WARMUP", "3"))
ITERS = int(os.environ.get("BENCH_ITERS", "50"))
DTYPE = os.environ.get("BENCH_DTYPE", "both").lower()
CHECK = os.environ.get("BENCH_CHECK", "1") == "1"
OUTPUT = os.environ.get("BENCH_OUTPUT")
PROFILE_DIR = os.environ.get("BENCH_PROFILE_DIR")
PROFILE_TOKENS = int(os.environ.get("BENCH_PROFILE_TOKENS", "16"))
PROFILE_ROUTE_SEED = int(os.environ.get("BENCH_PROFILE_ROUTE_SEED", "0"))
PROFILE_ITERS = int(os.environ.get("BENCH_PROFILE_ITERS", "20"))
ROUTE_BASE_SEED = int(os.environ.get("BENCH_ROUTE_BASE_SEED", "20260803"))
METADATA_TILE_TOKENS = int(os.environ.get("BENCH_METADATA_TILE_TOKENS", "128"))
METADATA_MINOR = 128

NDEV = jax.device_count()
if NDEV < 2:
    raise SystemExit("small-token A2A benchmark requires at least two devices")
if NUM_EXPERTS % NDEV != 0:
    raise SystemExit(f"num_experts={NUM_EXPERTS} must be divisible by device_count={NDEV}")
if TOP_K > NUM_EXPERTS:
    raise SystemExit(f"top_k={TOP_K} must not exceed num_experts={NUM_EXPERTS}")
if METADATA_TILE_TOKENS <= 0:
    raise SystemExit(f"invalid BENCH_METADATA_TILE_TOKENS={METADATA_TILE_TOKENS}")
if not TOKENS_CASES or min(TOKENS_CASES) <= 0:
    raise SystemExit(f"invalid token cases: {TOKENS_CASES}")
if not ROUTE_SEEDS:
    raise SystemExit("BENCH_ROUTE_SEEDS must not be empty")
if any(tokens not in TOKENS_CASES for tokens in CHECK_TOKENS):
    raise SystemExit("BENCH_CHECK_TOKENS must be a subset of BENCH_TOKENS_PER_DEVICE")
if DTYPE not in {"bf16", "fp8", "both"}:
    raise SystemExit(f"BENCH_DTYPE must be bf16, fp8, or both, got {DTYPE}")
if PROFILE_DIR and PROFILE_TOKENS not in TOKENS_CASES:
    raise SystemExit("BENCH_PROFILE_TOKENS must be in BENCH_TOKENS_PER_DEVICE")
if PROFILE_DIR and PROFILE_ROUTE_SEED not in ROUTE_SEEDS:
    raise SystemExit("BENCH_PROFILE_ROUTE_SEED must be in BENCH_ROUTE_SEEDS")

LOCAL_EXPERTS = NUM_EXPERTS // NDEV
devices = list(jax.devices())
local_devices = list(jax.local_devices())
device_to_index = {device: index for index, device in enumerate(devices)}

coords = [getattr(device, "coords", None) for device in devices]
if all(coord is not None for coord in coords):
    chip_by_coord: dict[tuple[int, ...], int] = {}
    chip_ids = []
    for coord in coords:
        coord_tuple = tuple(int(value) for value in coord)
        if coord_tuple not in chip_by_coord:
            chip_by_coord[coord_tuple] = len(chip_by_coord)
        chip_ids.append(chip_by_coord[coord_tuple])
    CHIP_IDS = np.asarray(chip_ids, dtype=np.int32)
    CORES_PER_CHIP = NDEV // len(chip_by_coord)
else:
    CORES_PER_CHIP = 1
    CHIP_IDS = np.arange(NDEV, dtype=np.int32)

mesh = jax.sharding.Mesh(np.asarray(devices, dtype=object).reshape(1, NDEV), (DP, TP))
row_spec = P((DP, TP))


@dataclass(frozen=True)
class RoutePlan:
    tokens: int
    seed: int
    routes: np.ndarray
    occurrences: np.ndarray
    owner_counts: np.ndarray
    source_counts: np.ndarray
    scatter_send_counts: np.ndarray
    scatter_recv_counts: np.ndarray
    gather_send_counts: np.ndarray
    capacity: int
    cross_chip_rows_by_source: np.ndarray
    cross_chip_gather_blocks_by_owner: np.ndarray
    cross_chip_gather_block_rows: np.ndarray


def _build_route_plan(tokens: int, seed: int) -> RoutePlan:
    routes = np.empty((NDEV, tokens, TOP_K), dtype=np.int32)
    occurrences = np.empty_like(routes)
    source_counts = np.zeros((NDEV, NUM_EXPERTS), dtype=np.int32)
    owner_counts = np.zeros((NDEV, NDEV, LOCAL_EXPERTS), dtype=np.int32)
    scatter_send_counts = np.zeros((NDEV, LOCAL_EXPERTS), dtype=np.int32)

    for source in range(NDEV):
        rng = np.random.default_rng(ROUTE_BASE_SEED + seed * 1009 + source)
        for token in range(tokens):
            token_experts = rng.choice(NUM_EXPERTS, size=TOP_K, replace=False)
            routes[source, token] = token_experts
            for k_id, expert in enumerate(token_experts):
                occurrence = source_counts[source, expert]
                occurrences[source, token, k_id] = occurrence
                source_counts[source, expert] += 1
                owner = expert // LOCAL_EXPERTS
                local_expert = expert % LOCAL_EXPERTS
                owner_counts[owner, source, local_expert] += 1
                if owner != source:
                    scatter_send_counts[source, local_expert] += 1

    scatter_recv_counts = owner_counts.sum(axis=1, dtype=np.int32)
    gather_send_counts = owner_counts.sum(axis=1, dtype=np.int32)
    for owner in range(NDEV):
        gather_send_counts[owner] -= owner_counts[owner, owner]

    cross_chip_rows_by_source = np.zeros((NDEV,), dtype=np.int32)
    cross_chip_gather_blocks_by_owner = np.zeros((NDEV,), dtype=np.int32)
    gather_block_rows: list[int] = []
    for source in range(NDEV):
        for expert in routes[source].reshape(-1):
            owner = expert // LOCAL_EXPERTS
            if CHIP_IDS[source] != CHIP_IDS[owner]:
                cross_chip_rows_by_source[source] += 1
    for owner in range(NDEV):
        for source in range(NDEV):
            if CHIP_IDS[source] == CHIP_IDS[owner]:
                continue
            for local_expert in range(LOCAL_EXPERTS):
                count = int(owner_counts[owner, source, local_expert])
                if count:
                    cross_chip_gather_blocks_by_owner[owner] += 1
                    gather_block_rows.append(count)

    capacity = int(source_counts.max())
    if capacity <= 0:
        raise AssertionError("route plan unexpectedly has no assignments")
    if not np.array_equal(
        owner_counts.transpose(1, 0, 2).reshape(NDEV, NUM_EXPERTS), source_counts
    ):
        raise AssertionError("owner/source count views disagree")

    return RoutePlan(
        tokens=tokens,
        seed=seed,
        routes=routes,
        occurrences=occurrences,
        owner_counts=owner_counts,
        source_counts=source_counts,
        scatter_send_counts=scatter_send_counts,
        scatter_recv_counts=scatter_recv_counts,
        gather_send_counts=gather_send_counts,
        capacity=capacity,
        cross_chip_rows_by_source=cross_chip_rows_by_source,
        cross_chip_gather_blocks_by_owner=cross_chip_gather_blocks_by_owner,
        cross_chip_gather_block_rows=np.asarray(gather_block_rows, dtype=np.int32),
    )


PLANS = {
    (tokens, seed): _build_route_plan(tokens, seed)
    for tokens in TOKENS_CASES
    for seed in ROUTE_SEEDS
}
CAPACITY_BY_TOKENS = {
    tokens: max(PLANS[tokens, seed].capacity for seed in ROUTE_SEEDS) for tokens in TOKENS_CASES
}


def _mesh_id(rank):
    tp_size = lax.axis_size(TP)
    return (rank // tp_size, rank % tp_size)


def _barrier(barrier_sem, num_devices):
    for device_id in range(num_devices):
        pltpu.semaphore_signal(
            barrier_sem,
            device_id=_mesh_id(device_id),
            device_id_type=pltpu.DeviceIdType.MESH,
        )
    pltpu.semaphore_wait(barrier_sem, num_devices)


def _make_scatter_kernel(
    tokens: int,
    capacity: int,
    inner: tuple[int, ...],
    dtype: jnp.dtype,
):
    metadata_tiles = (tokens + METADATA_TILE_TOKENS - 1) // METADATA_TILE_TOKENS

    def kernel(
        send_counts_ref,
        recv_counts_ref,
        routes_ref,
        occurrences_ref,
        tokens_ref,
        scatter_ref,
        send_sem,
        recv_sem,
        metadata_sem,
        barrier_sem,
        routes_vmem,
        occurrences_vmem,
        routes_smem,
        occurrences_smem,
    ):
        tp_size = lax.axis_size(TP)
        dp_size = lax.axis_size(DP)
        num_devices = tp_size * dp_size
        my_id = lax.axis_index(DP) * tp_size + lax.axis_index(TP)

        _barrier(barrier_sem, num_devices)

        with jax.named_scope(f"small_token_scatter_t{tokens}"):
            for metadata_tile in range(metadata_tiles):
                token_start = metadata_tile * METADATA_TILE_TOKENS
                tile_tokens = min(METADATA_TILE_TOKENS, tokens - token_start)
                tile_assignments = tile_tokens * TOP_K

                routes_copy = pltpu.make_async_copy(
                    src_ref=routes_ref.at[
                        pl.ds(token_start, METADATA_TILE_TOKENS),
                        pl.ds(0, METADATA_MINOR),
                    ],
                    dst_ref=routes_vmem,
                    sem=metadata_sem.at[0],
                )
                occurrences_copy = pltpu.make_async_copy(
                    src_ref=occurrences_ref.at[
                        pl.ds(token_start, METADATA_TILE_TOKENS),
                        pl.ds(0, METADATA_MINOR),
                    ],
                    dst_ref=occurrences_vmem,
                    sem=metadata_sem.at[1],
                )
                routes_copy.start()
                occurrences_copy.start()
                routes_copy.wait()
                occurrences_copy.wait()

                routes_to_smem = pltpu.async_copy(
                    src_ref=routes_vmem,
                    dst_ref=routes_smem,
                    sem=metadata_sem.at[0],
                )
                occurrences_to_smem = pltpu.async_copy(
                    src_ref=occurrences_vmem,
                    dst_ref=occurrences_smem,
                    sem=metadata_sem.at[1],
                )
                routes_to_smem.wait()
                occurrences_to_smem.wait()

                def fire_assignment(tile_assignment_id, _, token_start=token_start):
                    tile_token_id = tile_assignment_id // TOP_K
                    k_id = tile_assignment_id % TOP_K
                    token_id = token_start + tile_token_id
                    expert_id = routes_smem[tile_token_id, k_id]
                    occurrence = occurrences_smem[tile_token_id, k_id]
                    owner = expert_id // LOCAL_EXPERTS
                    local_expert = expert_id % LOCAL_EXPERTS
                    slot = my_id * capacity + occurrence
                    source = tokens_ref.at[pl.ds(token_id, 1)]
                    destination = scatter_ref.at[local_expert, pl.ds(slot, 1)]
                    is_local = owner == my_id

                    @pl.when(is_local)
                    def _local_copy():
                        pltpu.make_async_copy(
                            src_ref=source,
                            dst_ref=destination,
                            sem=recv_sem.at[local_expert],
                        ).start()

                    @pl.when(jnp.logical_not(is_local))
                    def _remote_copy():
                        pltpu.make_async_remote_copy(
                            src_ref=source,
                            dst_ref=destination,
                            send_sem=send_sem.at[local_expert],
                            recv_sem=recv_sem.at[local_expert],
                            device_id=_mesh_id(owner),
                            device_id_type=pltpu.DeviceIdType.MESH,
                        ).start()

                    return None

                lax.fori_loop(0, tile_assignments, fire_assignment, None, unroll=False)

            for local_expert in range(LOCAL_EXPERTS):
                recv_count = recv_counts_ref[local_expert]

                @pl.when(recv_count != 0)
                def _wait_recv(local_expert=local_expert, recv_count=recv_count):
                    ref = scatter_ref.at[local_expert, pl.ds(0, recv_count)]
                    pltpu.make_async_copy(
                        src_ref=ref,
                        dst_ref=ref,
                        sem=recv_sem.at[local_expert],
                    ).wait()

            for local_expert in range(LOCAL_EXPERTS):
                send_count = send_counts_ref[local_expert]

                @pl.when(send_count != 0)
                def _wait_send(local_expert=local_expert, send_count=send_count):
                    ref = scatter_ref.at[local_expert, pl.ds(0, send_count)]
                    pltpu.make_async_copy(
                        src_ref=ref,
                        dst_ref=ref,
                        sem=send_sem.at[local_expert],
                    ).wait()

        _barrier(barrier_sem, num_devices)

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(
            (LOCAL_EXPERTS, NDEV * capacity, *inner),
            dtype,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            # routes/occurrences scale with tokens and exceed SMEM at the
            # EP16 16K-global-token case. Keep only the small per-expert
            # count vectors in scalar-prefetch and stage route metadata via
            # HBM -> VMEM -> SMEM tiles, matching the production kernel.
            num_scalar_prefetch=2,
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
                pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
                pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
            ],
            out_specs=pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
            scratch_shapes=[
                pltpu.SemaphoreType.DMA((LOCAL_EXPERTS,)),
                pltpu.SemaphoreType.DMA((LOCAL_EXPERTS,)),
                pltpu.SemaphoreType.DMA((2,)),
                pltpu.SemaphoreType.BARRIER,
                pltpu.VMEM((METADATA_TILE_TOKENS, METADATA_MINOR), jnp.int32),
                pltpu.VMEM((METADATA_TILE_TOKENS, METADATA_MINOR), jnp.int32),
                pltpu.SMEM((METADATA_TILE_TOKENS, METADATA_MINOR), jnp.int32),
                pltpu.SMEM((METADATA_TILE_TOKENS, METADATA_MINOR), jnp.int32),
            ],
        ),
        compiler_params=pltpu.CompilerParams(
            collective_id=720_000 + tokens,
            allow_collective_id_without_custom_barrier=True,
            has_side_effects=True,
        ),
        name=f"small_token_scatter_t{tokens}",
    )


def _make_gather_kernel(
    tokens: int,
    capacity: int,
    inner: tuple[int, ...],
    dtype: jnp.dtype,
):
    def kernel(
        owner_counts_ref,
        source_counts_ref,
        send_counts_ref,
        expert_output_ref,
        gather_ref,
        send_sem,
        recv_sem,
        barrier_sem,
    ):
        tp_size = lax.axis_size(TP)
        dp_size = lax.axis_size(DP)
        num_devices = tp_size * dp_size
        my_id = lax.axis_index(DP) * tp_size + lax.axis_index(TP)

        _barrier(barrier_sem, num_devices)

        with jax.named_scope(f"small_token_gather_t{tokens}"):

            def start_expert(local_expert, _):
                global_expert = my_id * LOCAL_EXPERTS + local_expert
                for source_id in range(NDEV):
                    count = owner_counts_ref[source_id, local_expert]
                    source = expert_output_ref.at[
                        local_expert,
                        pl.ds(source_id * capacity, count),
                    ]
                    destination = gather_ref.at[global_expert, pl.ds(0, count)]
                    is_local = source_id == my_id

                    @pl.when(jnp.logical_and(is_local, count != 0))
                    def _local_copy(source=source, destination=destination):
                        pltpu.make_async_copy(
                            src_ref=source,
                            dst_ref=destination,
                            sem=recv_sem,
                        ).start()

                    @pl.when(jnp.logical_and(jnp.logical_not(is_local), count != 0))
                    def _remote_copy(
                        source=source,
                        destination=destination,
                        local_expert=local_expert,
                        source_id=source_id,
                    ):
                        pltpu.make_async_remote_copy(
                            src_ref=source,
                            dst_ref=destination,
                            send_sem=send_sem.at[local_expert],
                            recv_sem=recv_sem,
                            device_id=_mesh_id(source_id),
                            device_id_type=pltpu.DeviceIdType.MESH,
                        ).start()

                return None

            lax.fori_loop(0, LOCAL_EXPERTS, start_expert, None, unroll=False)

            def wait_expert(global_expert, _):
                count = source_counts_ref[global_expert]

                @pl.when(count != 0)
                def _wait():
                    ref = gather_ref.at[global_expert, pl.ds(0, count)]
                    pltpu.make_async_copy(
                        src_ref=ref,
                        dst_ref=ref,
                        sem=recv_sem,
                    ).wait()

                return None

            lax.fori_loop(0, NUM_EXPERTS, wait_expert, None, unroll=False)

            for local_expert in range(LOCAL_EXPERTS):
                send_count = send_counts_ref[local_expert]

                @pl.when(send_count != 0)
                def _wait_send(local_expert=local_expert, send_count=send_count):
                    ref = expert_output_ref.at[local_expert, pl.ds(0, send_count)]
                    pltpu.make_async_copy(
                        src_ref=ref,
                        dst_ref=ref,
                        sem=send_sem.at[local_expert],
                    ).wait()

        _barrier(barrier_sem, num_devices)

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((NUM_EXPERTS, capacity, *inner), dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=3,
            in_specs=[pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)],
            out_specs=pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
            scratch_shapes=[
                pltpu.SemaphoreType.DMA((LOCAL_EXPERTS,)),
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.BARRIER,
            ],
        ),
        compiler_params=pltpu.CompilerParams(
            collective_id=721_000 + tokens,
            allow_collective_id_without_custom_barrier=True,
            has_side_effects=True,
        ),
        name=f"small_token_gather_t{tokens}",
    )


def _make_scatter_baseline_kernel(
    tokens: int,
    capacity: int,
    inner: tuple[int, ...],
    dtype: jnp.dtype,
):
    def kernel(
        send_counts_ref,
        recv_counts_ref,
        routes_ref,
        occurrences_ref,
        tokens_ref,
        scatter_ref,
        send_sem,
        recv_sem,
        barrier_sem,
    ):
        del (
            routes_ref,
            occurrences_ref,
            send_counts_ref,
            recv_counts_ref,
            tokens_ref,
            scatter_ref,
            send_sem,
            recv_sem,
        )
        num_devices = lax.axis_size(TP) * lax.axis_size(DP)
        _barrier(barrier_sem, num_devices)
        with jax.named_scope(f"small_token_scatter_baseline_t{tokens}"):
            pass
        _barrier(barrier_sem, num_devices)

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(
            (LOCAL_EXPERTS, NDEV * capacity, *inner),
            dtype,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
                pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
                pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
            ],
            out_specs=pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
            scratch_shapes=[
                pltpu.SemaphoreType.DMA((LOCAL_EXPERTS,)),
                pltpu.SemaphoreType.DMA((LOCAL_EXPERTS,)),
                pltpu.SemaphoreType.BARRIER,
            ],
        ),
        compiler_params=pltpu.CompilerParams(
            collective_id=722_000 + tokens,
            allow_collective_id_without_custom_barrier=True,
            has_side_effects=True,
        ),
        name=f"small_token_scatter_baseline_t{tokens}",
    )


def _make_gather_baseline_kernel(
    tokens: int,
    capacity: int,
    inner: tuple[int, ...],
    dtype: jnp.dtype,
):
    def kernel(
        owner_counts_ref,
        source_counts_ref,
        send_counts_ref,
        expert_output_ref,
        gather_ref,
        send_sem,
        recv_sem,
        barrier_sem,
    ):
        del (
            owner_counts_ref,
            source_counts_ref,
            send_counts_ref,
            expert_output_ref,
            gather_ref,
            send_sem,
            recv_sem,
        )
        num_devices = lax.axis_size(TP) * lax.axis_size(DP)
        _barrier(barrier_sem, num_devices)
        with jax.named_scope(f"small_token_gather_baseline_t{tokens}"):
            pass
        _barrier(barrier_sem, num_devices)

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((NUM_EXPERTS, capacity, *inner), dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=3,
            in_specs=[pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)],
            out_specs=pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
            scratch_shapes=[
                pltpu.SemaphoreType.DMA((LOCAL_EXPERTS,)),
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.BARRIER,
            ],
        ),
        compiler_params=pltpu.CompilerParams(
            collective_id=723_000 + tokens,
            allow_collective_id_without_custom_barrier=True,
            has_side_effects=True,
        ),
        name=f"small_token_gather_baseline_t{tokens}",
    )


@functools.cache
def _scatter_runner(tokens: int, capacity: int, inner: tuple[int, ...], dtype: jnp.dtype):
    kernel = _make_scatter_kernel(tokens, capacity, inner, dtype)

    @jax.jit
    @jax.shard_map(
        mesh=mesh,
        in_specs=(row_spec, row_spec, row_spec, row_spec, row_spec),
        out_specs=row_spec,
        check_vma=False,
    )
    def run(routes, occurrences, send_counts, recv_counts, token_values):
        return kernel(send_counts, recv_counts, routes, occurrences, token_values)

    return run


@functools.cache
def _gather_runner(tokens: int, capacity: int, inner: tuple[int, ...], dtype: jnp.dtype):
    kernel = _make_gather_kernel(tokens, capacity, inner, dtype)

    @jax.jit
    @jax.shard_map(
        mesh=mesh,
        in_specs=(row_spec, row_spec, row_spec, row_spec),
        out_specs=row_spec,
        check_vma=False,
    )
    def run(owner_counts, source_counts, send_counts, expert_output):
        return kernel(owner_counts, source_counts, send_counts, expert_output)

    return run


@functools.cache
def _scatter_baseline_runner(
    tokens: int,
    capacity: int,
    inner: tuple[int, ...],
    dtype: jnp.dtype,
):
    kernel = _make_scatter_baseline_kernel(tokens, capacity, inner, dtype)

    @jax.jit
    @jax.shard_map(
        mesh=mesh,
        in_specs=(row_spec, row_spec, row_spec, row_spec, row_spec),
        out_specs=row_spec,
        check_vma=False,
    )
    def run(routes, occurrences, send_counts, recv_counts, token_values):
        return kernel(send_counts, recv_counts, routes, occurrences, token_values)

    return run


@functools.cache
def _gather_baseline_runner(
    tokens: int,
    capacity: int,
    inner: tuple[int, ...],
    dtype: jnp.dtype,
):
    kernel = _make_gather_baseline_kernel(tokens, capacity, inner, dtype)

    @jax.jit
    @jax.shard_map(
        mesh=mesh,
        in_specs=(row_spec, row_spec, row_spec, row_spec),
        out_specs=row_spec,
        check_vma=False,
    )
    def run(owner_counts, source_counts, send_counts, expert_output):
        return kernel(owner_counts, source_counts, send_counts, expert_output)

    return run


def _make_sharded_from_per_device(per_device: np.ndarray) -> jax.Array:
    if per_device.shape[0] != NDEV:
        raise ValueError(f"leading dimension must be {NDEV}, got {per_device.shape}")
    local_shape = per_device.shape[1:]
    shards = [
        jax.device_put(per_device[device_to_index[device]], device) for device in local_devices
    ]
    return jax.make_array_from_single_device_arrays(
        (NDEV * local_shape[0], *local_shape[1:]),
        jax.sharding.NamedSharding(mesh, row_spec),
        shards,
    )


def _make_zero_sharded(local_shape: tuple[int, ...], dtype: jnp.dtype) -> jax.Array:
    shards = []
    for device in local_devices:
        with jax.default_device(device):
            shards.append(jnp.zeros(local_shape, dtype=dtype))
    return jax.make_array_from_single_device_arrays(
        (NDEV * local_shape[0], *local_shape[1:]),
        jax.sharding.NamedSharding(mesh, row_spec),
        shards,
    )


def _metadata(plan: RoutePlan) -> tuple[jax.Array, ...]:
    metadata_tiles = (plan.tokens + METADATA_TILE_TOKENS - 1) // METADATA_TILE_TOKENS
    padded_tokens = metadata_tiles * METADATA_TILE_TOKENS
    routes = np.full((NDEV, padded_tokens, METADATA_MINOR), -1, dtype=np.int32)
    occurrences = np.zeros((NDEV, padded_tokens, METADATA_MINOR), dtype=np.int32)
    routes[:, : plan.tokens, :TOP_K] = plan.routes
    occurrences[:, : plan.tokens, :TOP_K] = plan.occurrences
    return (
        _make_sharded_from_per_device(routes),
        _make_sharded_from_per_device(occurrences),
        _make_sharded_from_per_device(plan.scatter_send_counts),
        _make_sharded_from_per_device(plan.scatter_recv_counts),
        _make_sharded_from_per_device(plan.owner_counts),
        _make_sharded_from_per_device(plan.source_counts),
        _make_sharded_from_per_device(plan.gather_send_counts),
    )


def _inner_shape(dtype: jnp.dtype, hidden: int) -> tuple[int, ...]:
    bytes_per_element = jnp.dtype(dtype).itemsize
    token_packing = 32 // (bytes_per_element * 8)
    if hidden % token_packing:
        raise ValueError(f"hidden={hidden} is not divisible by token_packing={token_packing}")
    return (token_packing, hidden // token_packing)


def _encoded_tokens(tokens: int, hidden: int) -> jax.Array:
    per_device = np.empty((NDEV, tokens, 1, hidden), dtype=np.int32)
    for source in range(NDEV):
        for token in range(tokens):
            per_device[source, token, 0] = source * 1000 + token + 1
    return _make_sharded_from_per_device(per_device)


def _expected_roundtrip(
    plan: RoutePlan,
    capacity: int,
    hidden: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    expected_scatter = np.zeros(
        (NDEV, LOCAL_EXPERTS, NDEV * capacity, 1, hidden),
        dtype=np.int32,
    )
    scatter_mask = np.zeros(
        (NDEV, LOCAL_EXPERTS, NDEV * capacity, 1, 1),
        dtype=np.bool_,
    )
    expected_gather = np.zeros(
        (NDEV, NUM_EXPERTS, capacity, 1, hidden),
        dtype=np.int32,
    )
    gather_mask = np.zeros(
        (NDEV, NUM_EXPERTS, capacity, 1, 1),
        dtype=np.bool_,
    )
    for source in range(NDEV):
        for token in range(plan.tokens):
            value = source * 1000 + token + 1
            for k_id in range(TOP_K):
                expert = int(plan.routes[source, token, k_id])
                occurrence = int(plan.occurrences[source, token, k_id])
                owner = expert // LOCAL_EXPERTS
                local_expert = expert % LOCAL_EXPERTS
                scatter_slot = source * capacity + occurrence
                expected_scatter[owner, local_expert, scatter_slot, 0] = value
                scatter_mask[owner, local_expert, scatter_slot, 0] = True
                expected_gather[source, expert, occurrence, 0] = value
                gather_mask[source, expert, occurrence, 0] = True
    return (
        _make_sharded_from_per_device(expected_scatter),
        _make_sharded_from_per_device(scatter_mask),
        _make_sharded_from_per_device(expected_gather),
        _make_sharded_from_per_device(gather_mask),
    )


def _masked_max_error(actual, expected, mask) -> int:
    @jax.jit
    def max_error(actual_value, expected_value, mask_value):
        difference = jnp.abs(actual_value.astype(jnp.int32) - expected_value.astype(jnp.int32))
        return jnp.max(jnp.where(mask_value, difference, 0))

    error = max_error(actual, expected, mask)
    jax.block_until_ready(error)
    return int(np.asarray(error.addressable_shards[0].data))


def _verify_roundtrip() -> None:
    hidden = 128
    dtype = jnp.int32
    inner = _inner_shape(dtype, hidden)
    for tokens in CHECK_TOKENS:
        plan = PLANS[tokens, ROUTE_SEEDS[0]]
        capacity = CAPACITY_BY_TOKENS[tokens]
        (
            routes,
            occurrences,
            scatter_send_counts,
            scatter_recv_counts,
            owner_counts,
            source_counts,
            gather_send_counts,
        ) = _metadata(plan)
        token_values = _encoded_tokens(tokens, hidden)
        scatter = _scatter_runner(tokens, capacity, inner, dtype)(
            routes,
            occurrences,
            scatter_send_counts,
            scatter_recv_counts,
            token_values,
        )
        gather = _gather_runner(tokens, capacity, inner, dtype)(
            owner_counts,
            source_counts,
            gather_send_counts,
            scatter,
        )
        expected_scatter, scatter_mask, expected_gather, gather_mask = _expected_roundtrip(
            plan,
            capacity,
            hidden,
        )
        scatter_error = _masked_max_error(scatter, expected_scatter, scatter_mask)
        gather_error = _masked_max_error(gather, expected_gather, gather_mask)
        log(
            f"correctness tokens={tokens} scatter_max_error={scatter_error} "
            f"gather_max_error={gather_error}"
        )
        if scatter_error or gather_error:
            raise AssertionError(
                f"roundtrip mismatch tokens={tokens}: scatter={scatter_error}, gather={gather_error}"
            )


def _percentile(samples: list[float], quantile: float) -> float:
    return float(np.percentile(np.asarray(samples, dtype=np.float64), quantile * 100.0))


def _timed(run, arguments: tuple[jax.Array, ...]) -> list[float]:
    for _ in range(WARMUP):
        jax.block_until_ready(run(*arguments))
    samples_ms: list[float] = []
    for _ in range(ITERS):
        start = time.monotonic()
        output = run(*arguments)
        jax.block_until_ready(output)
        samples_ms.append((time.monotonic() - start) * 1e3)
    return samples_ms


def _result(
    *,
    operation: str,
    dtype_name: str,
    plan: RoutePlan,
    capacity: int,
    samples_ms: list[float],
) -> dict[str, object]:
    bytes_per_element = 2 if dtype_name == "bf16" else 1
    p50_ms = _percentile(samples_ms, 0.50)
    p90_ms = _percentile(samples_ms, 0.90)
    is_baseline = operation.endswith("_baseline")
    ici_rows_mean = 0.0 if is_baseline else float(np.mean(plan.cross_chip_rows_by_source))
    ici_bytes_mean = ici_rows_mean * H * bytes_per_element
    if operation.startswith("scatter"):
        ici_messages_mean = ici_rows_mean
        block_rows = np.ones((1,), dtype=np.int32)
    else:
        ici_messages_mean = (
            0.0 if is_baseline else float(np.mean(plan.cross_chip_gather_blocks_by_owner))
        )
        block_rows = (
            np.zeros((1,), dtype=np.int32) if is_baseline else plan.cross_chip_gather_block_rows
        )
    return {
        "operation": operation,
        "dtype": dtype_name,
        "tokens_per_device": plan.tokens,
        "global_tokens": plan.tokens * NDEV,
        "top_k": TOP_K,
        "assignments_per_device": plan.tokens * TOP_K,
        "num_experts": NUM_EXPERTS,
        "local_experts": LOCAL_EXPERTS,
        "hidden": H,
        "route_seed": plan.seed,
        "route_capacity": capacity,
        "message_bytes_per_row": H * bytes_per_element,
        "cross_chip_rows_per_device_mean": round(ici_rows_mean, 6),
        "cross_chip_bytes_per_device_mean": round(ici_bytes_mean, 3),
        "cross_chip_dma_count_per_device_mean": round(ici_messages_mean, 6),
        "cross_chip_block_rows_mean": round(float(np.mean(block_rows)), 6),
        "cross_chip_block_rows_p50": round(_percentile(block_rows.tolist(), 0.50), 6),
        "cross_chip_block_rows_p90": round(_percentile(block_rows.tolist(), 0.90), 6),
        "cross_chip_block_rows_max": int(np.max(block_rows)),
        "latency_ms_mean": round(float(np.mean(samples_ms)), 6),
        "latency_ms_p50": round(p50_ms, 6),
        "latency_ms_p90": round(p90_ms, 6),
        "ici_useful_gb_s_per_device": round(ici_bytes_mean / (p50_ms * 1e6), 6),
        "samples_ms": [round(sample, 6) for sample in samples_ms],
        "process_index": jax.process_index(),
        "source_commit": os.environ.get("FALCON_SOURCE_COMMIT", ""),
        "run_id": os.environ.get("FALCON_EXP_ID", ""),
    }


def _benchmark_case(dtype_name: str, plan: RoutePlan, capacity: int) -> list[dict[str, object]]:
    dtype = jnp.bfloat16 if dtype_name == "bf16" else jnp.float8_e4m3fn
    inner = _inner_shape(dtype, H)
    (
        routes,
        occurrences,
        scatter_send_counts,
        scatter_recv_counts,
        owner_counts,
        source_counts,
        gather_send_counts,
    ) = _metadata(plan)
    token_values = _make_zero_sharded((plan.tokens, *inner), dtype)
    expert_output = _make_zero_sharded(
        (LOCAL_EXPERTS, NDEV * capacity, *inner),
        dtype,
    )

    scatter_samples = _timed(
        _scatter_runner(plan.tokens, capacity, inner, dtype),
        (
            routes,
            occurrences,
            scatter_send_counts,
            scatter_recv_counts,
            token_values,
        ),
    )
    scatter_baseline_samples = _timed(
        _scatter_baseline_runner(plan.tokens, capacity, inner, dtype),
        (
            routes,
            occurrences,
            scatter_send_counts,
            scatter_recv_counts,
            token_values,
        ),
    )
    gather_samples = _timed(
        _gather_runner(plan.tokens, capacity, inner, dtype),
        (owner_counts, source_counts, gather_send_counts, expert_output),
    )
    gather_baseline_samples = _timed(
        _gather_baseline_runner(plan.tokens, capacity, inner, dtype),
        (owner_counts, source_counts, gather_send_counts, expert_output),
    )
    return [
        _result(
            operation="scatter",
            dtype_name=dtype_name,
            plan=plan,
            capacity=capacity,
            samples_ms=scatter_samples,
        ),
        _result(
            operation="scatter_baseline",
            dtype_name=dtype_name,
            plan=plan,
            capacity=capacity,
            samples_ms=scatter_baseline_samples,
        ),
        _result(
            operation="gather",
            dtype_name=dtype_name,
            plan=plan,
            capacity=capacity,
            samples_ms=gather_samples,
        ),
        _result(
            operation="gather_baseline",
            dtype_name=dtype_name,
            plan=plan,
            capacity=capacity,
            samples_ms=gather_baseline_samples,
        ),
    ]


def _profile_representative(dtype_names: tuple[str, ...]) -> None:
    if not PROFILE_DIR:
        return

    plan = PLANS[PROFILE_TOKENS, PROFILE_ROUTE_SEED]
    capacity = CAPACITY_BY_TOKENS[PROFILE_TOKENS]
    for dtype_name in dtype_names:
        dtype = jnp.bfloat16 if dtype_name == "bf16" else jnp.float8_e4m3fn
        inner = _inner_shape(dtype, H)
        (
            routes,
            occurrences,
            scatter_send_counts,
            scatter_recv_counts,
            owner_counts,
            source_counts,
            gather_send_counts,
        ) = _metadata(plan)
        token_values = _make_zero_sharded((plan.tokens, *inner), dtype)
        expert_output = _make_zero_sharded(
            (LOCAL_EXPERTS, NDEV * capacity, *inner),
            dtype,
        )
        calls = (
            (
                "scatter",
                _scatter_runner(plan.tokens, capacity, inner, dtype),
                (
                    routes,
                    occurrences,
                    scatter_send_counts,
                    scatter_recv_counts,
                    token_values,
                ),
            ),
            (
                "scatter_baseline",
                _scatter_baseline_runner(plan.tokens, capacity, inner, dtype),
                (
                    routes,
                    occurrences,
                    scatter_send_counts,
                    scatter_recv_counts,
                    token_values,
                ),
            ),
            (
                "gather",
                _gather_runner(plan.tokens, capacity, inner, dtype),
                (owner_counts, source_counts, gather_send_counts, expert_output),
            ),
            (
                "gather_baseline",
                _gather_baseline_runner(plan.tokens, capacity, inner, dtype),
                (owner_counts, source_counts, gather_send_counts, expert_output),
            ),
        )
        for _, run, arguments in calls:
            jax.block_until_ready(run(*arguments))

        trace_dir = str(Path(PROFILE_DIR) / dtype_name)
        log(
            f"profiling tokens={PROFILE_TOKENS} seed={PROFILE_ROUTE_SEED} "
            f"dtype={dtype_name} iterations={PROFILE_ITERS} dir={trace_dir}"
        )
        jax.profiler.start_trace(trace_dir, create_perfetto_link=False)
        for operation, run, arguments in calls:
            for _ in range(PROFILE_ITERS):
                with jax.profiler.TraceAnnotation(
                    f"profile_{operation}_{dtype_name}_t{PROFILE_TOKENS}"
                ):
                    jax.block_until_ready(run(*arguments))
        jax.profiler.stop_trace()


def _write_results(results: list[dict[str, object]]) -> None:
    if not OUTPUT:
        return
    output_path = Path(OUTPUT)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as output_file:
        for result in results:
            output_file.write(json.dumps(result, sort_keys=True) + "\n")
    log(f"wrote {len(results)} metrics to {output_path}")


def main() -> None:
    log(
        f"device={devices[0].device_kind} ep={NDEV} chips={NDEV // CORES_PER_CHIP} "
        f"tokens={TOKENS_CASES} seeds={ROUTE_SEEDS} top_k={TOP_K} H={H} "
        f"experts={NUM_EXPERTS} local_experts={LOCAL_EXPERTS} capacities={CAPACITY_BY_TOKENS}"
    )
    if CHECK:
        _verify_roundtrip()

    dtype_names = ("bf16", "fp8") if DTYPE == "both" else (DTYPE,)
    results: list[dict[str, object]] = []
    for tokens in TOKENS_CASES:
        capacity = CAPACITY_BY_TOKENS[tokens]
        for seed in ROUTE_SEEDS:
            plan = PLANS[tokens, seed]
            for dtype_name in dtype_names:
                case_results = _benchmark_case(dtype_name, plan, capacity)
                results.extend(case_results)
                for result in case_results:
                    log(json.dumps(result, sort_keys=True))
    _profile_representative(dtype_names)
    _write_results(results)
    log("done")


if __name__ == "__main__":
    main()
