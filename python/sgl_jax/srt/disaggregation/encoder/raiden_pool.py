"""A fixed paged embedding buffer; lengths and page locations are host data."""

from __future__ import annotations

import logging
import math
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.disaggregation.encoder.transfer_layout import (
    ENCODER_PAGE_SIZE,
    encoder_pool_block_shape,
)

logger = logging.getLogger(__name__)


@partial(jax.jit, donate_argnums=(0,))
def _write_rows(pool, packed, destination_rows, source_rows):
    rows = packed[jnp.maximum(source_rows, 0)]
    rows = jnp.where((source_rows >= 0)[:, None], rows, 0)
    flat = pool.reshape(pool.shape[0] * pool.shape[1], -1)
    rows = jnp.pad(rows, ((0, 0), (0, flat.shape[1] - packed.shape[1])))
    flat = flat.at[destination_rows].set(rows, mode="drop")
    # A separate completion value survives the next donation of the pool.
    ready = flat[jnp.minimum(destination_rows[0], flat.shape[0] - 1), 0]
    return flat.reshape(pool.shape), ready


class RaidenPool:
    """Host page allocation plus one registered buffer. Callers serialize ownership."""

    def __init__(self, shape, dtype, sharding, *, capacity: int, max_batch_size: int = 8):
        self.page_size, self.width = map(int, shape)
        if self.page_size < 2 or capacity <= 0:
            raise ValueError("Raiden needs page_size >= 2 and positive page capacity")
        if not sharding.is_fully_replicated:
            raise ValueError("Encoder pools require replicated embeddings")
        self.dtype = jnp.dtype(dtype)
        self.sharding = sharding
        self.num_pages = int(capacity)
        self.max_batch_size = max(1, int(max_batch_size))
        self._free_pages = list(range(self.num_pages - 1, -1, -1))
        self.buffer = jnp.zeros(
            (self.num_pages, *encoder_pool_block_shape(shape)),
            self.dtype,
            device=self._sharding(5),
        )
        jax.block_until_ready(self.buffer)

    def _sharding(self, ndim):
        if isinstance(self.sharding, jax.sharding.NamedSharding):
            return self.sharding.update(spec=jax.sharding.PartitionSpec(*([None] * ndim)))
        return self.sharding

    @property
    def available_pages(self) -> int:
        return len(self._free_pages)

    def pages_needed(self, tokens: int) -> int:
        if tokens <= 0 or tokens > self.num_pages * self.page_size:
            raise ValueError("Encoder request exceeds the pool token capacity")
        return (tokens + self.page_size - 1) // self.page_size

    def allocate(self, tokens: int) -> tuple[int, ...] | None:
        count = self.pages_needed(tokens)
        if count > self.available_pages:
            return None
        return tuple(self._free_pages.pop() for _ in range(count))

    def release(self, page_ids: tuple[int, ...]) -> None:
        self._free_pages.extend(reversed(page_ids))

    def _write_capacity(self, packed_capacity: int) -> int:
        # At most one partial tail per request. No independent page-count bucket.
        pages = (packed_capacity + self.page_size - 1) // self.page_size + self.max_batch_size - 1
        return min(self.num_pages, pages) * self.page_size

    def warmup(self, packed_capacity: int) -> None:
        """Warm one writer per ViT capacity, before registering the buffer with Raiden."""
        packed = jnp.zeros((packed_capacity, self.width), self.dtype, device=self._sharding(2))
        capacity = self._write_capacity(packed_capacity)
        destinations = jax.device_put(
            np.full(capacity, self.num_pages * self.page_size, np.int32), self._sharding(1)
        )
        sources = jax.device_put(np.full(capacity, -1, np.int32), self._sharding(1))
        compiled = _write_rows.lower(self.buffer, packed, destinations, sources).compile()
        stats = compiled.memory_analysis()
        stats = stats if isinstance(stats, (list, tuple)) else (stats,)
        if not stats or any(
            stat is None
            or stat.alias_size_in_bytes <= 0
            or stat.alias_size_in_bytes * 100 < stat.output_size_in_bytes * 99
            for stat in stats
        ):
            raise RuntimeError("Raiden embedding writer must alias its donated buffer")
        self.buffer, ready = _write_rows(self.buffer, packed, destinations, sources)
        jax.block_until_ready((self.buffer, ready))
        logger.info("Encoder pool warmed: packed_capacity=%d", packed_capacity)

    def write_packed(self, packed, allocations, token_counts) -> jax.Array:
        if (
            len(allocations) != len(token_counts)
            or not allocations
            or len(allocations) > self.max_batch_size
            or packed.ndim != 2
            or packed.shape[1] != self.width
            or packed.dtype != self.dtype
            or not packed.sharding.is_equivalent_to(self._sharding(2), 2)
            or sum(token_counts) > packed.shape[0]
        ):
            raise ValueError("Packed encoder output does not match its page allocations")
        capacity = self._write_capacity(packed.shape[0])
        destinations = np.full(capacity, self.num_pages * self.page_size, np.int32)
        sources = np.full(capacity, -1, np.int32)
        row_offset = token_offset = 0
        for pages, tokens in zip(allocations, token_counts, strict=True):
            if len(pages) != self.pages_needed(tokens):
                raise ValueError("Encoder allocation does not match its token count")
            end = row_offset + len(pages) * self.page_size
            destinations[row_offset:end] = (
                np.asarray(pages, np.int32)[:, None] * self.page_size + np.arange(self.page_size)
            ).reshape(-1)
            # Zero the page tail as well, so no stale rows are exposed by transport.
            sources[row_offset : row_offset + tokens] = np.arange(
                token_offset, token_offset + tokens, dtype=np.int32
            )
            row_offset = end
            token_offset += tokens
        self.buffer, ready = _write_rows(
            self.buffer,
            jax.device_put(packed, self._sharding(2)),
            jax.device_put(destinations, self._sharding(1)),
            jax.device_put(sources, self._sharding(1)),
        )
        return ready


def create_encoder_pool(server_args, model_config, mesh) -> RaidenPool:
    vision = getattr(model_config.hf_config, "vision_config", None)
    width = model_config.hidden_size * (1 + len(getattr(vision, "deepstack_visual_indexes", ())))
    return RaidenPool(
        (ENCODER_PAGE_SIZE, width),
        model_config.dtype,
        jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec()),
        capacity=math.ceil(server_args.encoder_transfer_max_tokens / ENCODER_PAGE_SIZE),
        max_batch_size=server_args.encoder_max_batch_size,
    )
