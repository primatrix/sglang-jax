"""C128 HCA consumer of the V4 C1 pools and original-token page ledger.

The scheduler owns allocation, request mappings and safe SWA release. This
adapter only derives metadata and returns replacement arrays; it introduces
neither a second allocator nor a recurrent-slot free list.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.kernels.hca.attention import INERT_QUERY_OFFSET
from sgl_jax.srt.kernels.hca.hca import HCAMetadata
from sgl_jax.srt.kernels.hca.tuned_block_sizes import get_hca_kernel_schedule
from sgl_jax.srt.layers.attention.hca_backend import (
    HCABackend,
    HCABackendMetadata,
    _bucket_capacity,
    _bucket_max_queries,
    _pad_capacity,
    _query_schedule,
)
from sgl_jax.srt.mem_cache.deepseek_v4_memory_pool import scatter_sharding
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode


@jax.tree_util.register_pytree_node_class
@dataclass
class DeepseekV4HCAMetadata(HCABackendMetadata):
    # Rank-local request slots, with request_capacity denoting no initialization.
    state_init_slots: jax.Array | None = None

    def tree_flatten(self):
        return (self.kernel, self.state_init_slots), (
            self.schedule,
            self.use_uniform_prefill_fast_path,
        )

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(children[0], aux[0], aux[1], children[1])


class DeepseekV4HCABackend(HCABackend):
    """HCA math with C1 resources supplied explicitly for host metadata."""

    def __init__(self, *, mesh, page_size=128, max_context_len, request_capacity):
        if page_size not in (128, 256):
            raise ValueError("V4 HCA requires original-token page size 128 or 256")
        if max_context_len <= 0 or request_capacity <= 0:
            raise ValueError("V4 HCA context and request capacities must be positive")
        # The standalone HCA backend uses equal-sized ring/record pages. C1's
        # two tiers instead have P and P/128 rows, without changing the math.
        super().__init__(mesh=mesh, page_size=128)
        self.page_size = page_size
        self.max_context_len = max_context_len
        self.request_capacity = request_capacity
        self.forward_metadata = nnx.data(DeepseekV4HCAMetadata())

    def tree_flatten(self):
        return (self.forward_metadata,), dict(
            mesh=self.mesh,
            page_size=self.page_size,
            max_context_len=self.max_context_len,
            request_capacity=self.request_capacity,
        )

    def _page_tables(self, slots, lengths, prefixes, rank, request_pool, allocator):
        """Read both physical tiers from the same original-token ownership map."""
        p = self.page_size
        cp = p // self.compress_ratio
        mapping = allocator.full_to_swa_index_mapping
        mapping = mapping[rank] if isinstance(mapping, list) else mapping
        window, compressed, window_cu, compressed_cu = [], [], [0], [0]
        for slot, length, prefix in zip(slots, lengths, prefixes, strict=True):
            if not length:
                window.append(0)
                compressed.append(0)
                window_cu.append(window_cu[-1] + p)
                compressed_cu.append(compressed_cu[-1] + cp)
                continue
            locations = np.asarray(request_pool.req_to_token[slot, :length], np.int32)
            # C1 owns within-page contiguity. Inspect page anchors and the
            # bounded live SWA span, not every historical token on each decode.
            anchors = locations[::p]
            if np.any((anchors < p) | (anchors >= mapping.size) | (anchors % p != 0)):
                raise ValueError("V4 request mapping must contain allocated page anchors")
            pages = anchors // p
            history_start = max(0, int(prefix) - self.window_size + 1)
            required_positions = np.concatenate(
                (
                    np.arange(history_start, prefix),
                    np.arange(max(prefix, length - self.window_size), length),
                )
            )
            required_locations = locations[required_positions]
            expected_locations = anchors[required_positions // p] + required_positions % p
            if not np.array_equal(required_locations, expected_locations):
                raise ValueError("V4 logical pages must preserve original-token offsets")
            if np.any(mapping[required_locations] == 0):
                raise ValueError(
                    "V4 SWA pages required by this HCA step were released or unallocated"
                )
            window.extend((mapping[anchors] // p).tolist())
            window_cu.append(window_cu[-1] + len(pages) * p)
            completed = int(length) // self.compress_ratio
            count = max(1, (completed + cp - 1) // cp)
            compressed.extend(pages[:count].tolist() if completed else [0])
            compressed_cu.append(compressed_cu[-1] + count * cp)
        return tuple(
            np.asarray(a, np.int32) for a in (window, window_cu, compressed, compressed_cu)
        )

    def get_forward_metadata(
        self, batch, *, request_pool, allocator, state_init_mask=None, fixed_bucket=False
    ):
        dp = int(self.mesh.shape["data"])
        if allocator.dp_size != dp or allocator.page_size != self.page_size:
            raise ValueError("HCA and C1 allocator page/DP geometry disagree")
        if request_pool.size != self.request_capacity:
            raise ValueError("HCA and C1 request capacities disagree")
        lengths = np.asarray(batch.seq_lens, np.int32)
        slots = np.asarray(batch.req_pool_indices, np.int32)
        positions = np.asarray(batch.positions, np.int32).reshape(-1)
        if lengths.ndim != 1 or slots.shape != lengths.shape or not lengths.size:
            raise ValueError("HCA requires one sequence length and slot per padded request")
        if lengths.size % dp or positions.size % dp or not positions.size:
            raise ValueError("HCA request/token buffers must contain equal padded DP sections")
        if getattr(batch, "dp_size", dp) != dp:
            raise ValueError("batch DP size disagrees with the HCA mesh")
        b, t = lengths.size // dp, positions.size // dp
        if getattr(batch, "per_dp_bs_size", b) != b:
            raise ValueError("batch per-DP request capacity disagrees with its arrays")
        if np.any((lengths < 0) | (lengths > self.max_context_len)):
            raise ValueError("HCA sequence length exceeds the configured context")
        active = lengths > 0
        if np.any((slots[active] < 0) | (slots[active] >= self.request_capacity)):
            raise ValueError("active HCA requests need a valid global request slot")
        if np.unique(slots[active]).size != int(active.sum()):
            raise ValueError("each active HCA request must own a distinct slot")
        if batch.forward_mode == ForwardMode.DECODE:
            q_lens = active.astype(np.int32)
        elif batch.forward_mode == ForwardMode.EXTEND:
            q_lens = np.asarray(batch.extend_seq_lens, np.int32)
        else:
            raise ValueError("V4 HCA supports ordinary EXTEND and DECODE only")
        if q_lens.shape != lengths.shape or np.any(q_lens < 0) or np.any(q_lens > lengths):
            raise ValueError("invalid HCA query lengths")
        if np.any((q_lens > 0) != active):
            raise ValueError("only active HCA request rows may have query tokens")
        prefixes = lengths - q_lens
        supplied_prefixes = getattr(batch, "extend_prefix_lens", None)
        if (
            batch.forward_mode == ForwardMode.EXTEND
            and supplied_prefixes is not None
            and not np.array_equal(supplied_prefixes, prefixes)
        ):
            raise ValueError("HCA prefix plus query length must equal sequence length")
        init = active & (prefixes == 0)
        if state_init_mask is not None:
            explicit_init = np.asarray(state_init_mask, bool)
            if explicit_init.shape != lengths.shape or np.any(explicit_init & ~init):
                raise ValueError("state initialization requires an active zero-prefix request")
            init |= explicit_init
        local_lengths = lengths.reshape(dp, b)
        local_queries = q_lens.reshape(dp, b)
        local_slots = slots.reshape(dp, b)
        local_prefixes = prefixes.reshape(dp, b)
        local_positions = positions.reshape(dp, t)
        if np.any(local_queries.sum(axis=1) > t):
            raise ValueError("HCA queries exceed the padded token capacity on a DP rank")
        tables = [
            self._page_tables(
                local_slots[r], local_lengths[r], local_prefixes[r], r, request_pool, allocator
            )
            for r in range(dp)
        ]
        window_capacity = _bucket_capacity(max(len(x[0]) for x in tables), 8)
        compressed_capacity = _bucket_capacity(max(len(x[2]) for x in tables), 8)
        # Runtime precompile keys use capacities, never the live sequence lengths.
        # Standalone numerical/benchmark callers retain their adaptive schedule.
        if fixed_bucket:
            cache_tokens = np.asarray(batch.cache_loc).size // dp
            if cache_tokens < self.page_size or cache_tokens % self.page_size:
                raise ValueError("V4 cache bucket must contain whole pages per DP rank")
            window_capacity = cache_tokens // self.page_size + b
            compressed_capacity = window_capacity
        schedule = get_hca_kernel_schedule(
            str(np.asarray(self.mesh.devices).reshape(-1)[0].device_kind),
            page_size=self.page_size // self.compress_ratio,
            max_compressed_entries=max(
                1,
                (self.max_context_len if fixed_bucket else int(lengths.max()))
                // self.compress_ratio,
            ),
            local_heads=self.num_heads // int(self.mesh.shape.get("tensor", 1)),
            head_dim=self.head_dim,
        )
        uniform = bool(
            not fixed_bucket
            and batch.forward_mode == ForwardMode.EXTEND
            and np.all(active)
            and np.all(prefixes == 0)
            and np.all(local_queries.sum(axis=1) == t)
            and np.all(q_lens == q_lens[0])
        )
        max_queries = max(1, _bucket_max_queries(int(q_lens.max()), schedule.query_block_size))
        if fixed_bucket:
            max_queries = (
                1
                if batch.forward_mode == ForwardMode.DECODE
                else max(1, _bucket_max_queries(t, schedule.query_block_size))
            )
        metadata = []
        for rank in range(dp):
            qs = local_queries[rank]
            cu = np.concatenate(([0], np.cumsum(qs))).astype(np.int32)
            seq_ids = np.repeat(np.arange(b, dtype=np.int32), qs)
            expected = np.concatenate(
                [
                    np.arange(pre, end, dtype=np.int32)
                    for pre, end in zip(local_prefixes[rank], local_lengths[rank], strict=True)
                ]
            )
            if not np.array_equal(local_positions[rank, : cu[-1]], expected):
                raise ValueError("HCA positions must be contiguous within each request's chunk")
            seq_ids = _pad_capacity(seq_ids, t, 0)
            valid = np.arange(t) < cu[-1]
            state_slots = np.where(valid, local_slots[rank, seq_ids], self.request_capacity).astype(
                np.int32
            )
            boundary = np.flatnonzero(valid & ((local_positions[rank] + 1) % 128 == 0)).astype(
                np.int32
            )
            blocks, offsets, decode = _query_schedule(cu, schedule.query_block_size)
            block_capacity = t // schedule.query_block_size + b
            win, win_cu, comp, comp_cu = tables[rank]
            metadata.append(
                HCAMetadata(
                    state_slots,
                    seq_ids,
                    cu,
                    valid,
                    _pad_capacity(boundary, t // 128 + b, t),
                    _pad_capacity(win, window_capacity, 0),
                    win_cu,
                    local_lengths[rank],
                    _pad_capacity(comp, compressed_capacity, 0),
                    comp_cu,
                    local_lengths[rank] // 128,
                    _pad_capacity(blocks, block_capacity, 0),
                    _pad_capacity(offsets, block_capacity, INERT_QUERY_OFFSET),
                    _pad_capacity(decode, b, -1),
                    max_queries_per_request=max_queries,
                )
            )
        sharding = NamedSharding(self.mesh, P("data"))
        combined = jax.tree.map(
            lambda *leaves: jax.device_put(np.concatenate(leaves), sharding), *metadata
        )
        init_slots = jax.device_put(
            np.where(init, slots, self.request_capacity).astype(np.int32), sharding
        )
        return DeepseekV4HCAMetadata(combined, schedule, uniform, init_slots)

    def __call__(
        self, q, k, v, layer, forward_batch, token_to_kv_pool, *, compressor_state_pool, **kwargs
    ):
        """Use reshape views of C1 buffers and return native C1-shaped updates."""
        layer_id = int(layer.layer_id)
        if (
            token_to_kv_pool.page_size != self.page_size
            or compressor_state_pool.size != self.request_capacity
        ):
            raise ValueError("HCA and C1 pool geometry disagree")
        state = compressor_state_pool.get_buffer("c128", layer_id)
        window = token_to_kv_pool.get_buffer("swa", layer_id)
        compressed = token_to_kv_pool.get_buffer("c128", layer_id)
        init_slots = self.forward_metadata.state_init_slots
        if init_slots is None:
            raise RuntimeError("V4 HCA metadata has not been prepared")
        dp = int(self.mesh.shape["data"])
        ranks = jnp.repeat(jnp.arange(dp, dtype=jnp.int32), init_slots.shape[0] // dp)
        destinations = jnp.where(
            init_slots < self.request_capacity,
            ranks * (self.request_capacity + 1) + init_slots,
            state.shape[0],
        )
        empty = (
            jnp.zeros((init_slots.shape[0], *state.shape[1:]), state.dtype)
            .at[..., self.head_dim :]
            .set(-jnp.inf)
        )
        state = state.at[destinations].set(
            empty, mode="drop", out_sharding=scatter_sharding(self.mesh, 3)
        )
        state_view = state.reshape(state.shape[0], 128, 2, self.head_dim)
        window_view = window.reshape(-1, self.page_size // 2, 2, self.head_dim)
        compressed_view = compressed.reshape(
            compressed.shape[0], 1, self.page_size // 128, self.head_dim
        )
        # These contain views only. Ownership, allocation and update validation
        # stay with C1; the standalone HCA allocator/pools are never constructed.
        kv_view = SimpleNamespace(
            max_context_len=self.max_context_len,
            _layer_index=lambda _: 0,
            window_buffer=(window_view,),
            compressed_buffer=(compressed_view,),
        )
        state_proxy = SimpleNamespace(_layer_index=lambda _: 0, get_hca_state=lambda _: state_view)
        output, (new_state, new_window, new_compressed) = super().__call__(
            q,
            k,
            v,
            layer,
            forward_batch,
            kv_view,
            recurrent_state_pool=state_proxy,
            **kwargs,
        )
        return output, (
            new_state.reshape(state.shape),
            new_window.reshape(window.shape),
            new_compressed.reshape(compressed.shape),
        )

    @staticmethod
    def pack_pool_updates(layer_updates, token_to_kv_pool, compressor_state_pool):
        """Merge {layer_id: (state, SWA, C128)} into complete C1 update families."""
        kv = {name: list(arrays) for name, arrays in token_to_kv_pool.buffers.items()}
        state = {name: list(arrays) for name, arrays in compressor_state_pool.buffers.items()}
        for layer_id, (s, w, c) in layer_updates.items():
            state["c128"][compressor_state_pool.layer_to_buffer["c128"][layer_id]] = s
            kv["swa"][token_to_kv_pool.layer_to_buffer["swa"][layer_id]] = w
            kv["c128"][token_to_kv_pool.layer_to_buffer["c128"][layer_id]] = c
        return {
            "token_to_kv_pool": {k: tuple(v) for k, v in kv.items()},
            "compressor_state_pool": {k: tuple(v) for k, v in state.items()},
        }
