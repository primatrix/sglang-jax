"""Page-run DMA writer for flat ``[rows, D]`` KV caches (Pallas TPU).

Prefill writes thousands of rows whose destinations are contiguous within each
allocator page; XLA's scatter handles them one row at a time (about 7 GB/s on v7x:
1.2 ms per layer for an 8K chunk). This kernel splits the write into fixed
``run``-row segments, DMAs a whole segment when its destinations are one
tile-aligned contiguous range, and falls back to a read-modify-write of the
16-row tile for rows of segments that are not contiguous, so any layout stays
correct. The cache is input/output aliased; untouched rows are never copied.
"""

from __future__ import annotations

import functools
import os

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

_TILE = 16  # bf16 sublane tile


_GROUP = 16  # scattered rows handled together: all loads, then merges, then all stores


def _kernel(dst_ref, loc_ref, valid_ref, values_ref, _, cache_hbm_ref, tiles_ref, sems, *, run):
    # ``cache_hbm_ref`` is the aliased output buffer (same HBM as the input cache).
    seg = pl.program_id(0)
    dst = dst_ref[seg]

    @pl.when(dst >= 0)
    def _contiguous():
        start = pl.multiple_of(dst, _TILE)  # the wrapper only marks tile-aligned runs
        copy = pltpu.make_async_copy(values_ref, cache_hbm_ref.at[pl.ds(start, run)], sems.at[0])
        copy.start()
        copy.wait()

    @pl.when(dst < 0)
    def _rows():
        # Scattered rows: read-modify-write of each destination's 16-row tile. Rows are
        # taken in groups of ``_GROUP``; inside a group every distinct tile is loaded
        # once (the first row aiming at it is the tile's leader), all loads are in
        # flight together, every row is merged into its leader's copy in row order
        # (later duplicates win, as with a serial scatter), and the leaders' tiles
        # are stored together. Groups are sequential, so a tile touched by two groups
        # is re-read after the earlier store. (One serial RMW per row exposed two DMA
        # latencies per row: 91 us for the 64 window rows a bs=64 decode step commits
        # per HCA layer on v7x.)
        row_ids = jax.lax.broadcasted_iota(jnp.int32, (_TILE, tiles_ref.shape[-1]), 0)

        def write_group(rows):
            tokens = [seg * run + r for r in rows]
            valid = [valid_ref[token] != 0 for token in tokens]
            locs = [loc_ref[token] for token in tokens]
            # invalid rows get a tile id no valid row can share
            tiles = [jnp.where(valid[j], locs[j] // _TILE, -1 - j) for j in range(_GROUP)]
            leaders = []
            for j in range(_GROUP):
                leader = jnp.int32(j)
                for i in reversed(range(j)):
                    leader = jnp.where(tiles[i] == tiles[j], jnp.int32(i), leader)
                leaders.append(leader)
            is_leader = [valid[j] & (leaders[j] == j) for j in range(_GROUP)]

            def tile_start(j):
                return pl.multiple_of((locs[j] // _TILE) * _TILE, _TILE)

            def load_copy(j):
                return pltpu.make_async_copy(
                    cache_hbm_ref.at[pl.ds(tile_start(j), _TILE)], tiles_ref.at[j], sems.at[j]
                )

            def store_copy(j):
                return pltpu.make_async_copy(
                    tiles_ref.at[j], cache_hbm_ref.at[pl.ds(tile_start(j), _TILE)], sems.at[j]
                )

            for j in range(_GROUP):
                pl.when(is_leader[j])(lambda j=j: load_copy(j).start())
            for j in range(_GROUP):
                pl.when(is_leader[j])(lambda j=j: load_copy(j).wait())

            def merge(j):
                target = tiles_ref.at[leaders[j]]
                new_row = jnp.broadcast_to(
                    values_ref[pl.ds(rows[j], 1), :], (_TILE, tiles_ref.shape[-1])
                )
                target[...] = jnp.where(
                    row_ids == (locs[j] - (locs[j] // _TILE) * _TILE), new_row, target[...]
                )

            for j in range(_GROUP):
                pl.when(valid[j])(lambda j=j: merge(j))
            for j in range(_GROUP):
                pl.when(is_leader[j])(lambda j=j: store_copy(j).start())
            for j in range(_GROUP):
                pl.when(is_leader[j])(lambda j=j: store_copy(j).wait())

        for group in range(run // _GROUP):
            write_group(list(range(group * _GROUP, (group + 1) * _GROUP)))


def default_run() -> int:
    """``DSV4_PAGED_ROW_RUN`` (default 128): rows per DMA segment.

    Each grid step issues one segment DMA and waits for it, so an 8K prefill
    chunk's 8192 rows cost 64 serial DMAs at 128; longer runs amortise the DMA
    latency when the allocator hands out physically consecutive pages, and fall
    back to the tile read-modify-write path otherwise.
    """
    return int(os.environ.get("DSV4_PAGED_ROW_RUN", "128"))


def paged_row_write(cache, values, loc, valid, *, run: int | None = None, interpret: bool = False):
    """``cache.at[loc].set(values)`` for valid rows, by page-run DMA.

    ``cache`` [R, D] (bf16), ``values`` [T, D], ``loc`` [T] int32 destinations,
    ``valid`` [T] bool. Rows with ``valid`` False or out-of-range ``loc`` are dropped.
    """
    if run is None:
        run = default_run()
    cache = jnp.asarray(cache)
    rows, dim = cache.shape
    values = jnp.asarray(values, cache.dtype)
    loc = jnp.asarray(loc, jnp.int32)
    valid = jnp.asarray(valid, bool) & (loc >= 0) & (loc < rows)
    n = values.shape[0]
    if run % _GROUP or run % _TILE:
        raise ValueError("run must be a multiple of the 16-row tile")
    if rows % _TILE:
        # The row fallback rewrites whole 16-row tiles, so the last tile of a cache
        # whose row count is not tile-aligned would reach past the buffer (bounds
        # checks are off). Such caches (small test pools) take the XLA scatter.
        safe = jnp.where(valid, loc, rows)
        return cache.at[safe].set(values, mode="drop")
    n_pad = -(-n // run) * run
    if n_pad != n:
        values = jnp.pad(values, ((0, n_pad - n), (0, 0)))
        loc = jnp.pad(loc, (0, n_pad - n))
        valid = jnp.pad(valid, (0, n_pad - n))
    seg = n_pad // run
    l2 = loc.reshape(seg, run)
    v2 = valid.reshape(seg, run)
    base = l2[:, 0]
    contiguous = (
        jnp.all(v2, axis=1)
        & jnp.all(l2 == base[:, None] + jnp.arange(run, dtype=jnp.int32)[None, :], axis=1)
        & (base % _TILE == 0)
        & (base + run <= rows)
    )
    dst = jnp.where(contiguous, base, -1).astype(jnp.int32)
    return pl.pallas_call(
        functools.partial(_kernel, run=run),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=3,
            grid=(seg,),
            in_specs=(
                pl.BlockSpec((run, dim), lambda i, *_: (i, 0)),
                pl.BlockSpec(memory_space=pltpu.HBM),
            ),
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            scratch_shapes=(
                pltpu.VMEM((_GROUP, _TILE, dim), cache.dtype),
                pltpu.SemaphoreType.DMA((_GROUP,)),
            ),
        ),
        out_shape=jax.ShapeDtypeStruct(cache.shape, cache.dtype),
        input_output_aliases={4: 0},
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",), disable_bounds_checks=True
        ),
        interpret=interpret,
        name=f"dsv4-paged-row-write-r{run}-d{dim}",
    )(dst, loc, valid.astype(jnp.int32), values, cache)
