"""Paged BF16 key cache used by the GLM/DeepSeek DSA indexer."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jax.tree_util import register_pytree_node_class

from sgl_jax.srt.kernels.mla.v2.kernel import (
    align_to,
    get_dtype_packing,
    get_kv_cache_shape,
)


@register_pytree_node_class
class DsaIndexerKPool:
    """Layered Index-K pool sharing token slot addresses with the MLA cache."""

    def __init__(
        self,
        *,
        size: int,
        page_size: int,
        index_head_dim: int,
        layer_num: int,
        mesh: Mesh | None,
        dp_size: int = 1,
        start_layer: int | None = None,
        end_layer: int | None = None,
    ) -> None:
        if size <= 0 or page_size <= 0 or layer_num <= 0 or dp_size <= 0:
            raise ValueError("size, page_size, layer_num, and dp_size must be positive")
        if dp_size != 1:
            raise NotImplementedError(
                "DsaIndexerKPool currently supports only the single-DP reference path; "
                "rank-local allocator slots must be globalized before enabling multi-DP"
            )
        if size % page_size != 0 or size % dp_size != 0:
            raise ValueError("size must be divisible by page_size and dp_size")
        if index_head_dim <= 0:
            raise ValueError("index_head_dim must be positive")

        self.size = size
        self.page_size = page_size
        self.index_head_dim = index_head_dim
        self.aligned_head_dim = align_to(index_head_dim, 128)
        self.layer_num = layer_num
        self.mesh = mesh
        self.dp_size = dp_size
        self.start_layer = 0 if start_layer is None else start_layer
        self.end_layer = self.start_layer + layer_num - 1 if end_layer is None else end_layer
        if self.end_layer - self.start_layer + 1 != layer_num:
            raise ValueError("start_layer/end_layer span must match layer_num")
        self.dtype = jnp.bfloat16
        self.packing = get_dtype_packing(self.dtype)
        self.total_num_pages = (size + page_size * dp_size) // page_size
        self.buffer_shape = get_kv_cache_shape(
            total_num_pages=self.total_num_pages,
            page_size=page_size,
            kv_dim=self.aligned_head_dim,
            kv_dtype=self.dtype,
        )
        self.k_sharding = None if mesh is None else NamedSharding(mesh, P("data", None, None, None))
        self.k_buffer = self._create_buffers()
        self.mem_usage = (
            self.layer_num
            * math.prod(self.buffer_shape)
            * jnp.dtype(self.dtype).itemsize
            / (1 << 30)
        )

    def _create_buffers(self) -> list[jax.Array]:
        if self.mesh is None:
            return [jnp.zeros(self.buffer_shape, dtype=self.dtype) for _ in range(self.layer_num)]

        with self.mesh:
            create = jax.jit(
                lambda: jnp.zeros(self.buffer_shape, dtype=self.dtype),
                out_shardings=self.k_sharding,
            )
            return [create() for _ in range(self.layer_num)]

    def get_buffer(self, layer_id: int) -> jax.Array:
        offset = layer_id - self.start_layer
        if offset < 0 or offset >= self.layer_num:
            raise IndexError(
                f"layer_id {layer_id} is outside [{self.start_layer}, {self.end_layer}]"
            )
        return self.k_buffer[offset]

    def replace_buffer(self, buffers: list[jax.Array]) -> None:
        if len(buffers) > self.layer_num:
            raise ValueError(f"received {len(buffers)} buffers for {self.layer_num} Index-K layers")
        self.k_buffer[: len(buffers)] = buffers

    def get_kv_size_bytes(self) -> int:
        return self.layer_num * math.prod(self.buffer_shape) * jnp.dtype(self.dtype).itemsize

    def tree_flatten(self):
        children = (self.k_buffer,)
        aux_data = {
            "size": self.size,
            "page_size": self.page_size,
            "index_head_dim": self.index_head_dim,
            "aligned_head_dim": self.aligned_head_dim,
            "layer_num": self.layer_num,
            "mesh": self.mesh,
            "dp_size": self.dp_size,
            "start_layer": self.start_layer,
            "end_layer": self.end_layer,
            "dtype": self.dtype,
            "packing": self.packing,
            "total_num_pages": self.total_num_pages,
            "buffer_shape": self.buffer_shape,
            "k_sharding": self.k_sharding,
            "mem_usage": self.mem_usage,
        }
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = object.__new__(cls)
        for name, value in aux_data.items():
            setattr(obj, name, value)
        obj.k_buffer = children[0]
        return obj
