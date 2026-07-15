"""Selected-slot materialization helpers for sparse DSA MLA decode."""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc

_DEFAULT_GATHER_BLOCK = 128
_MIN_SC_VECTOR_WIDTH = 8
_JAX_081_MAX_ACTIVE_SC_CORES = 2
SPARSECORE_COMPILER_OPTIONS = {
    "xla_tpu_use_tc_device_shape_on_sc": "false"
}


def _active_sparsecore_cores(reported_cores: int) -> int:
    """Return the SC core count usable by the pinned Falcon compiler."""
    if not isinstance(reported_cores, int) or reported_cores <= 0:
        raise ValueError("reported SparseCore count must be a positive integer")
    return min(reported_cores, _JAX_081_MAX_ACTIVE_SC_CORES)


def _validate_gather_block(gather_block: int) -> None:
    if not isinstance(gather_block, int):
        raise ValueError("gather_block must be a Python integer")
    if gather_block <= 0 or gather_block % _MIN_SC_VECTOR_WIDTH:
        raise ValueError("gather_block must be a positive multiple of 8")


def prepare_safe_topk_slots(
    topk_slots: jax.Array,
    valid_counts: jax.Array,
    *,
    gather_block: int = _DEFAULT_GATHER_BLOCK,
) -> jax.Array:
    """Pad selected slots to a static gather block and make padding safe.

    Entries outside ``valid_counts`` and negative entries are replaced with
    physical slot zero. The validated public wrapper rejects negative valid
    entries; handling them here as padding additionally guarantees that an
    unchecked gather never uses a negative address.
    """
    _validate_gather_block(gather_block)
    topk_slots = jnp.asarray(topk_slots, dtype=jnp.int32)
    valid_counts = jnp.asarray(valid_counts, dtype=jnp.int32)

    max_selected = topk_slots.shape[1]
    padded_selected = (
        (max_selected + gather_block - 1) // gather_block
    ) * gather_block
    if padded_selected != max_selected:
        topk_slots = jnp.pad(
            topk_slots,
            ((0, 0), (0, padded_selected - max_selected)),
            constant_values=0,
        )

    positions = jnp.arange(padded_selected, dtype=jnp.int32)[None, :]
    is_valid = (positions < valid_counts[:, None]) & (topk_slots >= 0)
    return jnp.where(is_valid, topk_slots, jnp.int32(0))


def materialize_selected_kv_xla(
    cache_kv: jax.Array,
    topk_slots: jax.Array,
    valid_counts: jax.Array,
    *,
    gather_block: int = _DEFAULT_GATHER_BLOCK,
) -> jax.Array:
    """Materialize selected logical cache rows with a regular XLA gather."""
    cache_kv = jnp.asarray(cache_kv)
    safe_slots = prepare_safe_topk_slots(
        topk_slots, valid_counts, gather_block=gather_block
    )
    logical_cache = cache_kv.reshape((-1, cache_kv.shape[-1]))
    return logical_cache[safe_slots]


def _plan_sparsecore_pipeline(
    *,
    batch_size: int,
    padded_selected: int,
    gather_block: int,
    available_cores: int,
    num_subcores: int,
) -> tuple[int, int, int]:
    """Choose a duplicate-free SC worker layout for the gather pipeline.

    Returns ``(num_cores, num_workers, windows_per_worker)``. VectorSubcoreMesh
    always activates every subcore in a selected SparseCore, so the total
    number of gather windows must divide evenly across those workers. The GLM
    production shape has 16 windows per batch row and satisfies this exactly
    on v7x (16 subcores per SparseCore).
    """
    dimensions = {
        "batch_size": batch_size,
        "padded_selected": padded_selected,
        "gather_block": gather_block,
        "available_cores": available_cores,
        "num_subcores": num_subcores,
    }
    if any(not isinstance(value, int) or value <= 0 for value in dimensions.values()):
        raise ValueError("SparseCore pipeline dimensions must be positive integers")

    total_rows = batch_size * padded_selected
    if total_rows % gather_block:
        raise ValueError("gather rows must be divisible by gather_block")
    num_windows = total_rows // gather_block
    if num_windows < num_subcores or num_windows % num_subcores:
        raise ValueError(
            "SparseCore pipeline requires gather windows divisible by num_subcores"
        )

    num_cores = min(available_cores, num_windows // num_subcores)
    while num_windows % (num_cores * num_subcores):
        num_cores -= 1
    num_workers = num_cores * num_subcores
    return num_cores, num_workers, num_windows // num_workers


def _sparsecore_gather_kernel(cache_hbm_ref, slot_indices_ref, output_vmem_ref):
    """Gather major-dimension cache rows with one SparseCore indirect DMA."""
    pltpu.sync_copy(cache_hbm_ref.at[slot_indices_ref], output_vmem_ref)


def materialize_selected_kv_sparsecore_unchecked(
    cache_kv: jax.Array,
    safe_topk_slots: jax.Array,
    *,
    gather_block: int = _DEFAULT_GATHER_BLOCK,
) -> jax.Array:
    """Run SparseCore gather on already-safe, gather-block-padded slots."""
    _validate_gather_block(gather_block)
    if jax.default_backend() != "tpu":
        raise RuntimeError("SparseCore selected KV materialization requires a TPU")

    cache_kv = jnp.asarray(cache_kv)
    safe_topk_slots = jnp.asarray(safe_topk_slots, dtype=jnp.int32)
    batch_size, padded_selected = safe_topk_slots.shape
    if padded_selected % gather_block:
        raise ValueError("safe_topk_slots width must be divisible by gather_block")

    cache_width = cache_kv.shape[-1]
    logical_cache = cache_kv.reshape((-1, cache_width))
    flattened_slots = safe_topk_slots.reshape((-1,))
    out_shape = jax.ShapeDtypeStruct(
        (batch_size * padded_selected, cache_width), cache_kv.dtype
    )

    gather_call = pl.pallas_call(
        _sparsecore_gather_kernel,
        out_shape=out_shape,
        grid=(batch_size * padded_selected // gather_block,),
        in_specs=(
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(
                (gather_block,),
                lambda block_index: (block_index,),
                memory_space=pltpu.VMEM,
            ),
        ),
        out_specs=pl.BlockSpec(
            (gather_block, cache_width),
            lambda block_index: (block_index, 0),
            memory_space=pltpu.VMEM,
        ),
        compiler_params=pltpu.CompilerParams(
            kernel_type=pltpu.KernelType.SC_VECTOR_SUBCORE
        ),
        name="dsa-selected-kv-sparsecore-gather",
    )
    gathered = gather_call(logical_cache, flattened_slots)
    return gathered.reshape((batch_size, padded_selected, cache_width))


def materialize_selected_kv_sparsecore_pipeline_unchecked(
    cache_kv: jax.Array,
    safe_topk_slots: jax.Array,
    *,
    gather_block: int = _DEFAULT_GATHER_BLOCK,
) -> jax.Array:
    """Gather selected rows with a pipelined VectorSubcoreMesh kernel.

    Each SparseCore vector subcore receives a unique contiguous range of
    gather windows. Indices and output tiles are double-buffered by
    ``emit_pipeline`` while the body issues one indirect HBM-to-VMEM DMA per
    window.
    """
    _validate_gather_block(gather_block)
    if jax.default_backend() != "tpu":
        raise RuntimeError("SparseCore selected KV materialization requires a TPU")

    cache_kv = jnp.asarray(cache_kv)
    safe_topk_slots = jnp.asarray(safe_topk_slots, dtype=jnp.int32)
    batch_size, padded_selected = safe_topk_slots.shape
    if padded_selected % gather_block:
        raise ValueError("safe_topk_slots width must be divisible by gather_block")

    sparsecore_info = pltpu.get_tpu_info().sparse_core
    if sparsecore_info is None:
        raise RuntimeError("The current TPU does not expose SparseCores")
    num_cores, _, windows_per_worker = _plan_sparsecore_pipeline(
        batch_size=batch_size,
        padded_selected=padded_selected,
        gather_block=gather_block,
        available_cores=_active_sparsecore_cores(sparsecore_info.num_cores),
        num_subcores=sparsecore_info.num_subcores,
    )

    cache_width = cache_kv.shape[-1]
    logical_cache = cache_kv.reshape((-1, cache_width))
    flattened_slots = safe_topk_slots.reshape((1, -1))
    out_shape = jax.ShapeDtypeStruct(
        (batch_size * padded_selected, cache_width), cache_kv.dtype
    )
    # JAX 0.8.1's VectorSubcoreMesh exposes these literal axis names from its
    # ``shape`` property even though the constructor also carries name fields.
    core_axis_name = "core"
    subcore_axis_name = "subcore"
    mesh = plsc.VectorSubcoreMesh(
        core_axis_name=core_axis_name,
        subcore_axis_name=subcore_axis_name,
        num_cores=num_cores,
    )

    @pl.kernel(
        out_shape=out_shape,
        mesh=mesh,
        name="dsa-selected-kv-sparsecore-pipeline",
    )
    def gather_kernel(cache_hbm_ref, indices_hbm_ref, output_hbm_ref):
        core_id = jax.lax.axis_index(core_axis_name)
        subcore_id = jax.lax.axis_index(subcore_axis_name)
        worker_id = core_id * sparsecore_info.num_subcores + subcore_id
        first_window = worker_id * windows_per_worker

        def gather_window(indices_vmem_ref, output_vmem_ref):
            pltpu.sync_copy(
                cache_hbm_ref.at[indices_vmem_ref.at[0]], output_vmem_ref
            )

        pltpu.emit_pipeline(
            gather_window,
            grid=(windows_per_worker,),
            in_specs=(
                pl.BlockSpec(
                    (1, gather_block),
                    lambda step: (0, first_window + step),
                ),
            ),
            out_specs=(
                pl.BlockSpec(
                    (gather_block, cache_width),
                    lambda step: (first_window + step, 0),
                ),
            ),
            dimension_semantics=(pltpu.PARALLEL,),
        )(indices_hbm_ref, output_hbm_ref)

    gathered = gather_kernel(logical_cache, flattened_slots)
    return gathered.reshape((batch_size, padded_selected, cache_width))


@functools.cache
def _sparsecore_gather_launcher(gather_block: int):
    launch = functools.partial(
        materialize_selected_kv_sparsecore_unchecked,
        gather_block=gather_block,
    )
    return jax.jit(launch, compiler_options=SPARSECORE_COMPILER_OPTIONS)


@functools.cache
def _sparsecore_pipeline_launcher(gather_block: int):
    launch = functools.partial(
        materialize_selected_kv_sparsecore_pipeline_unchecked,
        gather_block=gather_block,
    )
    return jax.jit(launch, compiler_options=SPARSECORE_COMPILER_OPTIONS)


def materialize_selected_kv_sparsecore(
    cache_kv: jax.Array,
    topk_slots: jax.Array,
    valid_counts: jax.Array,
    *,
    gather_block: int = _DEFAULT_GATHER_BLOCK,
) -> jax.Array:
    """Make selected slots safe, then materialize them with SparseCore."""
    safe_slots = prepare_safe_topk_slots(
        topk_slots, valid_counts, gather_block=gather_block
    )
    return _sparsecore_gather_launcher(gather_block)(cache_kv, safe_slots)


def materialize_selected_kv_sparsecore_pipeline(
    cache_kv: jax.Array,
    topk_slots: jax.Array,
    valid_counts: jax.Array,
    *,
    gather_block: int = _DEFAULT_GATHER_BLOCK,
) -> jax.Array:
    """Materialize selected rows with the pipelined SparseCore launcher."""
    safe_slots = prepare_safe_topk_slots(
        topk_slots, valid_counts, gather_block=gather_block
    )
    return _sparsecore_pipeline_launcher(gather_block)(cache_kv, safe_slots)


__all__ = [
    "materialize_selected_kv_sparsecore",
    "materialize_selected_kv_sparsecore_pipeline",
    "materialize_selected_kv_sparsecore_pipeline_unchecked",
    "materialize_selected_kv_sparsecore_unchecked",
    "materialize_selected_kv_xla",
    "prepare_safe_topk_slots",
    "SPARSECORE_COMPILER_OPTIONS",
]
