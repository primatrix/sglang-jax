"""BF16 V4 resources. Addresses are rank-local; page zero is always padding.

TP/EP replicate the single KV head; only the leading capacity axis is sharded
on the existing attention ``data`` mesh axis. No raw full-history KV is stored.
"""

from dataclasses import dataclass
from functools import lru_cache, partial

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P


@dataclass(frozen=True)
class DeepseekV4CacheSpec:
    compress_ratios: tuple[int, ...]
    head_dim: int = 512
    index_head_dim: int = 128

    def __post_init__(self):
        object.__setattr__(self, "compress_ratios", tuple(self.compress_ratios))
        if not self.compress_ratios or any(r not in (0, 4, 128) for r in self.compress_ratios):
            raise ValueError("V4 backbone ratios must be 0, 4 or 128")
        if self.head_dim <= 0 or self.index_head_dim <= 0:
            raise ValueError("V4 cache dimensions must be positive")

    @classmethod
    def from_config(cls, config):
        n = config.num_hidden_layers
        if n <= 0 or len(config.compress_ratios) < n:
            raise ValueError("compress_ratios must cover all backbone layers")
        return cls(tuple(config.compress_ratios[:n]), config.head_dim, config.index_head_dim)

    def layers(self, ratio):
        return tuple(i for i, r in enumerate(self.compress_ratios) if r == ratio)

    def history_bytes_per_page(self, page_size):
        # BF16, no additional value cache: the same representation serves K/V.
        return 2 * (
            len(self.layers(4)) * (self.head_dim + self.index_head_dim) * (page_size // 4)
            + len(self.layers(128)) * self.head_dim * (page_size // 128)
        )

    @property
    def swa_bytes_per_token(self):
        return 2 * len(self.compress_ratios) * self.head_dim

    @property
    def state_bytes_per_request(self):
        return 4 * (
            len(self.layers(4)) * 8 * 4 * (self.head_dim + self.index_head_dim)
            + len(self.layers(128)) * 128 * 2 * self.head_dim
        )


@lru_cache(maxsize=128)
def _zero_allocator(shape, dtype, mesh):
    # Cache the compiled constructor, never an array shared between layers.
    sharding = NamedSharding(mesh, P("data", *([None] * (len(shape) - 1))))
    return jax.jit(partial(jnp.zeros, shape, dtype), out_shardings=sharding)


def allocate_buffer(shape, dtype, mesh):
    with jax.set_mesh(mesh):
        return _zero_allocator(tuple(shape), dtype, mesh)()


def scatter_sharding(mesh, ndim):
    # Explicit meshes require a result sharding for scattered indices. Auto
    # meshes infer it from the operand and reject explicit scatter specs.
    data_axis = mesh.axis_names.index("data")
    if mesh.axis_types[data_axis] == jax.sharding.AxisType.Explicit:
        return NamedSharding(mesh, P("data", *([None] * (ndim - 1))))
    return None


class _V4Buffers:
    """Static metadata plus a dict of tuples of arrays, compatible with MemoryPools."""

    def tree_flatten(self):
        return (self.buffers,), self._metadata

    @classmethod
    def tree_unflatten(cls, metadata, children):
        obj = object.__new__(cls)
        obj._configure(*metadata)
        obj.buffers = children[0]
        return obj

    @property
    def nbytes(self):
        return sum(a.size * a.dtype.itemsize for arrays in self.buffers.values() for a in arrays)

    @property
    def mem_usage(self):
        return self.nbytes / 1024**3

    def validate_buffer_updates(self, buffers):
        if set(buffers) != set(self.buffers):
            raise ValueError("V4 update must contain every buffer family")
        for family, old in self.buffers.items():
            new = buffers[family]
            if len(new) != len(old) or any(
                a.shape != b.shape or a.dtype != b.dtype for a, b in zip(old, new)
            ):
                raise ValueError(f"V4 {family} update shape/dtype mismatch")

    def replace_buffer(self, buffers):
        self.validate_buffer_updates(buffers)
        self.buffers = {key: tuple(arrays) for key, arrays in buffers.items()}


@jax.tree_util.register_pytree_node_class
class DeepseekV4TokenToKVPool(_V4Buffers):
    def __init__(self, size, size_swa, page_size, spec, mesh, dp_size=1, dtype=jnp.bfloat16):
        if jnp.dtype(dtype) != jnp.dtype(jnp.bfloat16):
            raise ValueError("V4 initial KV cache requires BF16")
        if page_size not in (128, 256):
            raise ValueError("V4 history/SWA page_size must be 128 or 256")
        if dp_size <= 0 or mesh.shape.get("data") != dp_size:
            raise ValueError("dp_size must match the attention data mesh")
        if any(s <= 0 or s % (page_size * dp_size) for s in (size, size_swa)):
            raise ValueError("V4 capacities must be positive page/DP aligned token counts")
        self._configure(size, size_swa, page_size, spec, mesh, dp_size)
        self.buffers = {
            family: tuple(allocate_buffer(shape, self.dtype, mesh) for _ in layers)
            for family, (layers, shape) in self.layout.items()
        }

    def _configure(self, size, size_swa, page_size, spec, mesh, dp_size):
        self._metadata = (size, size_swa, page_size, spec, mesh, dp_size)
        self.size, self.size_swa, self.page_size = size, size_swa, page_size
        self.spec, self.mesh, self.dp_size = spec, mesh, dp_size
        self.dtype = jnp.bfloat16
        self.layer_num = len(spec.compress_ratios)
        self.pages_per_rank = size // dp_size // page_size
        self.swa_slots_per_rank = size_swa // dp_size + page_size
        pages = (self.pages_per_rank + 1) * dp_size
        self.layout = {
            "swa": (
                tuple(range(self.layer_num)),
                (self.swa_slots_per_rank * dp_size, spec.head_dim),
            ),
            "c4": (spec.layers(4), (pages, page_size // 4, spec.head_dim)),
            "c128": (spec.layers(128), (pages, page_size // 128, spec.head_dim)),
            "indexer": (spec.layers(4), (pages, page_size // 4, spec.index_head_dim)),
        }
        self.layer_to_buffer = {
            family: {layer: i for i, layer in enumerate(layers)}
            for family, (layers, _) in self.layout.items()
        }

    def get_buffer(self, family, layer_id):
        return self.buffers[family][self.layer_to_buffer[family][layer_id]]

    def set_buffer(self, family, layer_id, value):
        i = self.layer_to_buffer[family][layer_id]
        old = self.buffers[family][i]
        if old.shape != value.shape or old.dtype != value.dtype:
            raise ValueError("V4 buffer shape/dtype mismatch")
        arrays = list(self.buffers[family])
        arrays[i] = value
        self.buffers = {**self.buffers, family: tuple(arrays)}

    def write(self, family, layer_id, loc, values, valid_mask, dp_rank=0):
        """Masked reference scatter; compressed loc is in compressed-entry units.

        This operates on global arrays. A shard_map consumer uses the same local
        loc without the rank offset. Invalid entries are dropped, never sent to
        slot zero (duplicate padding writes must not race with real tokens).
        """
        array = self.get_buffer(family, layer_id)
        flat = array.reshape(-1, array.shape[-1])
        per_rank = flat.shape[0] // self.dp_size
        reserved = self.page_size // {"swa": 1, "c4": 4, "c128": 128, "indexer": 4}[family]
        valid = (
            valid_mask
            & (loc >= reserved)
            & (loc < per_rank)
            & (dp_rank >= 0)
            & (dp_rank < self.dp_size)
        )
        index = jnp.where(valid, loc + dp_rank * per_rank, flat.shape[0])
        self.set_buffer(
            family,
            layer_id,
            flat.at[index]
            .set(values, mode="drop", out_sharding=scatter_sharding(self.mesh, 2))
            .reshape(array.shape),
        )
