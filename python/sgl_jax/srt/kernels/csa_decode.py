"""Request-local paged CSA decode scores, with double-buffered key DMA.

The page/DMA schedule follows the private DSA paged scorer. V4 uses completed
compression-group lengths and an explicit FP32 weighted head reduction, matching
the production CSA indexer rather than DSA's second matrix multiplication.
"""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

_NEG_INF = jnp.finfo(jnp.float32).min


def _score_kernel(lengths, pages, q, weights, cache, output, keys, sems, *, page_size):
    row = pl.program_id(0)
    length = lengths[row]
    block_k = keys.shape[1]
    pages_per_block = block_k // page_size
    blocks = (length + block_k - 1) // block_k
    output[...] = jnp.full(output.shape, _NEG_INF, jnp.float32)

    def fetch(block, buffer):
        dst = keys.at[buffer]
        dst[...] = jnp.zeros(dst.shape, dst.dtype)
        count = jnp.minimum(
            pages_per_block, (length + page_size - 1) // page_size - block * pages_per_block
        )

        def page_copy(p, _):
            physical = pages[row, block * pages_per_block + p]
            pltpu.make_async_copy(
                cache.at[physical], dst.at[pl.ds(p * page_size, page_size)], sems.at[buffer]
            ).start()
            return None

        lax.fori_loop(0, count, page_copy, None)

    @pl.when(length > 0)
    def score_request():
        fetch(0, 0)

        def step(block, _):
            buffer = block % 2
            dst = keys.at[buffer]
            # A final partial tile issues fewer page DMAs. Wait for exactly the
            # bytes issued, not for a full block (which would never complete).
            count = jnp.minimum(
                pages_per_block,
                (length + page_size - 1) // page_size - block * pages_per_block,
            )

            def wait_page(p, _):
                page_dst = dst.at[pl.ds(p * page_size, page_size)]
                pltpu.make_async_copy(page_dst, page_dst, sems.at[buffer]).wait()
                return None

            lax.fori_loop(0, count, wait_page, None)

            @pl.when(block + 1 < blocks)
            def prefetch():
                fetch(block + 1, 1 - buffer)

            # Mixed FP32/BF16 dots are not supported by every Mosaic lowering.
            query = q[0]
            resident = keys[buffer].astype(query.dtype)
            similarities = lax.dot_general(
                query,
                resident,
                dimension_numbers=(((1,), (1,)), ((), ())),
                preferred_element_type=jnp.float32,
            )
            scores = jnp.sum(jax.nn.relu(similarities) * weights[0, 0][:, None], axis=0)
            positions = block * block_k + jnp.arange(block_k, dtype=jnp.int32)
            output[0, 0, pl.ds(block * block_k, block_k)] = jnp.where(
                positions < length, scores, _NEG_INF
            )
            return None

        lax.fori_loop(0, blocks, step, None)


def paged_csa_decode_scores(q, weights, cache, lengths, pages, *, interpret=False):
    """Score only each query's completed compressed entries.

    q [T,H,D], weights [T,H], cache [P,page_size,D], lengths [T], pages [T,N].
    The result [T,N*page_size] uses finite minimum FP32 for every invalid entry.
    Request isolation is encoded in the allocator-derived page table.
    """
    tokens, heads, dim = q.shape
    page_size = cache.shape[1]
    capacity = pages.shape[1] * page_size
    if weights.shape != (tokens, heads) or lengths.shape != (tokens,) or pages.shape[0] != tokens:
        raise ValueError("CSA decode query, weight, length and page-table shapes disagree")
    if cache.shape[-1] != dim or dim % 128 or page_size % 8 or capacity % 128:
        raise ValueError("CSA decode requires aligned cache pages and 128-wide key dimensions")
    block_k = min(2048, capacity)
    if capacity % block_k:
        raise ValueError("CSA decode capacity must be divisible by the key tile")
    return pl.pallas_call(
        functools.partial(_score_kernel, page_size=page_size),
        out_shape=jax.ShapeDtypeStruct((tokens, 1, capacity), jnp.float32),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            grid=(tokens,),
            in_specs=[
                pl.BlockSpec((1, heads, dim), lambda row, *_: (row, 0, 0)),
                pl.BlockSpec((1, 1, heads), lambda row, *_: (row, 0, 0)),
                pl.BlockSpec(memory_space=pltpu.HBM),
            ],
            out_specs=pl.BlockSpec((1, 1, capacity), lambda row, *_: (row, 0, 0)),
            scratch_shapes=[
                pltpu.VMEM((2, block_k, dim), cache.dtype),
                pltpu.SemaphoreType.DMA((2,)),
            ],
        ),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        interpret=interpret,
        name="csa_request_local_decode_scores",
    )(lengths, pages, q, weights[:, None, :], cache)[:, 0, :]
