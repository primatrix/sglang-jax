"""Experimental TPU mapper for logical DSA top-k indices.

This module intentionally keeps the production JAX implementation in
``dsa_sparse_backend`` unchanged.  It provides a standalone Pallas candidate
whose ABI can be benchmarked against that implementation before integration.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def logical_topk_to_physical_slots_pallas(
    topk: jax.Array,
    seq_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    cu_kv_lens: jax.Array,
    page_size: int,
    *,
    block_q: int = 128,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Map sequence-local top-k positions to flat cache slots with Pallas.

    The small per-query metadata gathers stay in JAX.  The Pallas kernel owns
    the production hotspot: the ``[Q, K]`` indirect lookup into the ragged page
    table and physical-slot materialization.

    ``interpret=True`` is for CPU correctness tests only.  TPU benchmarks must
    use the default compiled Mosaic path.
    """

    if topk.ndim != 2:
        raise ValueError(f"topk must have rank 2, got shape={topk.shape}")
    if topk.dtype != jnp.int32:
        raise TypeError(f"topk must have dtype int32, got {topk.dtype}")
    for name, value in (
        ("seq_lens", seq_lens),
        ("page_indices", page_indices),
        ("cu_q_lens", cu_q_lens),
        ("cu_kv_lens", cu_kv_lens),
    ):
        if value.dtype != jnp.int32:
            raise TypeError(f"{name} must have dtype int32, got {value.dtype}")
    if page_size <= 0:
        raise ValueError(f"page_size must be positive, got {page_size}")
    if block_q <= 0:
        raise ValueError(f"block_q must be positive, got {block_q}")
    if page_indices.shape[0] == 0:
        raise ValueError("page_indices must not be empty")

    num_queries, topk_size = topk.shape
    padded_queries = ((num_queries + block_q - 1) // block_q) * block_q
    pad_queries = padded_queries - num_queries

    token_ids = jnp.arange(padded_queries, dtype=jnp.int32)
    seq_ids = jnp.searchsorted(cu_q_lens[1:], token_ids, side="right")
    seq_ids = jnp.clip(seq_ids, 0, seq_lens.shape[0] - 1)
    seq_len_by_query = seq_lens[seq_ids]
    page_start_by_query = cu_kv_lens[seq_ids] // page_size
    query_valid = token_ids < cu_q_lens[-1]
    padded_topk = jnp.pad(topk, ((0, pad_queries), (0, 0)), constant_values=-1)

    # On TPU these blocks are staged in VMEM for vector arithmetic and the
    # indirect page-table lookup.  CPU interpret mode uses ANY memory because
    # TPU memory spaces are not available there.
    input_memory = pl.ANY if interpret else pltpu.VMEM
    output_memory = pl.ANY if interpret else pltpu.HBM

    def kernel(
        topk_ref,
        seq_len_ref,
        page_start_ref,
        query_valid_ref,
        page_indices_ref,
        slots_ref,
        counts_ref,
    ):
        logical_topk = topk_ref[...]
        logical = jnp.maximum(logical_topk, 0)
        page_ptr = page_start_ref[...][:, None] + logical // page_size
        ptr_in_bounds = (page_ptr >= 0) & (page_ptr < page_indices.shape[0])
        safe_ptr = jnp.clip(page_ptr, 0, page_indices.shape[0] - 1)
        physical_pages = page_indices_ref[safe_ptr]
        valid = (
            query_valid_ref[...][:, None]
            & (logical_topk >= 0)
            & (logical < seq_len_ref[...][:, None])
            & ptr_in_bounds
            & (physical_pages >= 0)
        )
        physical_slots = physical_pages * page_size + logical % page_size
        slots_ref[...] = jnp.where(valid, physical_slots, jnp.int32(0))
        counts_ref[...] = jnp.sum(valid, axis=1, dtype=jnp.int32)

    grid = (padded_queries // block_q,)
    slots, counts = pl.pallas_call(
        kernel,
        out_shape=(
            jax.ShapeDtypeStruct((padded_queries, topk_size), jnp.int32),
            jax.ShapeDtypeStruct((padded_queries,), jnp.int32),
        ),
        grid=grid,
        in_specs=(
            pl.BlockSpec(
                (block_q, topk_size), lambda q_block: (q_block, 0), input_memory
            ),
            pl.BlockSpec((block_q,), lambda q_block: (q_block,), input_memory),
            pl.BlockSpec((block_q,), lambda q_block: (q_block,), input_memory),
            pl.BlockSpec((block_q,), lambda q_block: (q_block,), input_memory),
            pl.BlockSpec(
                (page_indices.shape[0],),
                lambda _q_block: (0,),
                input_memory,
            ),
        ),
        out_specs=(
            pl.BlockSpec(
                (block_q, topk_size), lambda q_block: (q_block, 0), output_memory
            ),
            pl.BlockSpec((block_q,), lambda q_block: (q_block,), output_memory),
        ),
        interpret=interpret,
        name=f"dsa_logical_topk_to_physical_slots_bq{block_q}",
    )(
        padded_topk,
        seq_len_by_query,
        page_start_by_query,
        query_valid,
        page_indices,
    )
    return slots[:num_queries], counts[:num_queries]
