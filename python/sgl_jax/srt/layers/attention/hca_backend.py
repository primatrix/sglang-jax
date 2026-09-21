"""SGLang-JAX attention backend for HCA."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import numpy as np
from flax import nnx
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.kernels.hca.attention import INERT_QUERY_OFFSET
from sgl_jax.srt.kernels.hca.hca import HCAMetadata
from sgl_jax.srt.kernels.hca.tuned_block_sizes import get_hca_kernel_schedule
from sgl_jax.srt.layers.attention.base_attn_backend import AttentionBackend
from sgl_jax.srt.layers.attention.hca_execution import run_hca
from sgl_jax.srt.layers.attention.hca_metadata import (
    _BOUNDARY_FLOOR,
    _COMPRESSED_TABLE_FLOOR,
    _DECODE_IDS_FLOOR,
    _WINDOW_TABLE_FLOOR,
    HCABackendMetadata,
    _bucket_capacity,
    _bucket_max_queries,
    _pad_capacity,
    _query_schedule,
)
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode
from sgl_jax.srt.utils.jax_utils import device_array

if TYPE_CHECKING:
    from sgl_jax.srt.managers.schedule_batch import ModelWorkerBatch
    from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch


@dataclass
class HCABackend(AttentionBackend):
    """Standalone HCA pool interface used by the HCA integration tests.

    DSV4 uses the shared run_hca executor directly, without inheriting this
    pool-specific adapter or maintaining a second forward-metadata owner.
    """

    def __init__(
        self,
        *,
        num_attn_heads: int = 64,
        head_dim: int = 512,
        compressor_hidden_size: int = 4096,
        page_size: int = 128,
        compress_ratio: int = 128,
        window_size: int = 128,
        mesh: jax.sharding.Mesh,
    ):
        if mesh is None:
            raise ValueError("production HCABackend requires the SGLang device mesh")
        if page_size < 2 or window_size % page_size:
            raise ValueError("HCA page_size must be >=2 and divide window_size")
        if (
            num_attn_heads != 64
            or head_dim != 512
            or compressor_hidden_size != 4096
            or compress_ratio != 128
            or window_size != 128
        ):
            raise ValueError(
                "production HCA requires H=64, D=512, hidden=4096, ratio=128, and window=128"
            )
        self.num_heads = num_attn_heads
        self.head_dim = head_dim
        self.compressor_hidden_size = compressor_hidden_size
        self.page_size = page_size
        self.compress_ratio = compress_ratio
        self.window_size = window_size
        self.mesh = mesh
        self.forward_metadata = nnx.data(HCABackendMetadata())
        # HCA page ownership: set by model_runner after pool creation, read on
        # host during metadata construction, like FlashAttention's swa_index_mapping.
        self.allocator = None

    def get_forward_metadata(self, batch: ModelWorkerBatch):
        """Derive HCA metadata from the worker batch and place it on the data mesh.

        Reads only standard batch fields plus the attached allocator's page
        tables; the scheduler must already have grown the compressed tier via
        ``ensure_compressed_capacity``.
        """
        if self.allocator is None:
            raise RuntimeError("model_runner must attach an HCAKVPoolAllocator first")
        req_pool_indices = np.asarray(batch.req_pool_indices, np.int32)
        seq_lens = np.asarray(batch.seq_lens, np.int32)
        positions = np.asarray(batch.positions, np.int32).reshape(-1)
        if req_pool_indices.shape != seq_lens.shape:
            raise ValueError("req_pool_indices and seq_lens must have the same shape")
        if np.any(seq_lens < 0):
            raise ValueError("HCA sequence lengths must be non-negative")
        active_requests = seq_lens > 0
        if np.any(req_pool_indices[active_requests] < 0):
            raise ValueError("active HCA requests require allocated request slots")
        if batch.forward_mode == ForwardMode.DECODE:
            q_lens = active_requests.astype(np.int32)
            uniform_prefill = False
        elif batch.forward_mode == ForwardMode.EXTEND:
            q_lens = np.asarray(batch.extend_seq_lens, np.int32)
            if q_lens.shape != seq_lens.shape:
                raise ValueError("extend_seq_lens must have one value per request")
            prefix_lens = (
                seq_lens - q_lens
                if batch.extend_prefix_lens is None
                else np.asarray(batch.extend_prefix_lens, np.int32)
            )
            if prefix_lens.shape != seq_lens.shape:
                raise ValueError("extend_prefix_lens must have one value per request")
            active_q_lens = q_lens[active_requests]
            uniform_prefill = bool(
                active_q_lens.size
                and int(active_q_lens.sum()) == positions.size
                and np.all(prefix_lens[active_requests] == 0)
                and np.all(active_q_lens == active_q_lens[0])
            )
        else:
            raise ValueError(f"HCA does not support {batch.forward_mode}")
        if np.any(q_lens < 0) or np.any((q_lens > 0) != active_requests):
            raise ValueError("only active HCA requests may contain query tokens")

        valid_token_count = int(q_lens.sum())
        if valid_token_count > positions.size:
            raise ValueError("positions does not contain every HCA query token")
        query_seq_ids = np.repeat(np.arange(q_lens.size, dtype=np.int32), q_lens)
        valid_token_mask = np.arange(positions.size) < valid_token_count
        if positions.size > valid_token_count:
            query_seq_ids = np.pad(query_seq_ids, (0, positions.size - valid_token_count))
        cu_q_lens = np.concatenate((np.zeros((1,), np.int32), np.cumsum(q_lens, dtype=np.int32)))
        emit_mask = valid_token_mask & (np.mod(positions + 1, self.compress_ratio) == 0)
        boundary_tokens = np.flatnonzero(emit_mask).astype(np.int32)

        # Recurrent slots ride the standard hybrid-recurrent batch field.
        if batch.recurrent_indices is None:
            raise ValueError("HCA requires batch.recurrent_indices from hybrid scheduling")
        state_by_request = np.asarray(batch.recurrent_indices, np.int32)
        if state_by_request.shape != seq_lens.shape:
            raise ValueError("HCA requires one recurrent slot per request")
        if np.any(state_by_request[active_requests] == 0):
            raise ValueError("HCA forward references an unallocated recurrent slot")
        safe_ids = np.where(valid_token_mask, query_seq_ids, 0)
        state_slots = np.where(valid_token_mask, state_by_request[safe_ids], 0).astype(np.int32)

        (
            window_page_indices,
            window_cu_kv_lens,
            compressed_page_indices,
            compressed_cu_kv_lens,
            compressed_kv_lens,
        ) = self.allocator.page_tables(req_pool_indices, seq_lens)

        tensor_shards = int(self.mesh.shape.get("tensor", 1))
        if self.num_heads % tensor_shards:
            raise ValueError("HCA attention heads must divide the tensor mesh")
        device_kind = str(np.asarray(self.mesh.devices).reshape(-1)[0].device_kind)
        schedule = get_hca_kernel_schedule(
            device_kind,
            page_size=self.page_size,
            max_compressed_entries=max(1, int(compressed_kv_lens.max(initial=0))),
            local_heads=self.num_heads // tensor_shards,
            head_dim=self.head_dim,
        )
        block_requests, block_offsets, decode_requests = _query_schedule(
            cu_q_lens, schedule.query_block_size
        )
        # Pad every batch-dependent length to a bucketed capacity so metadata
        # drift never recompiles; ``HCAMetadata`` documents the inert sentinels.
        tokens = int(positions.size)
        batch_size = int(seq_lens.shape[0])
        block_capacity = tokens // schedule.query_block_size + batch_size
        if block_requests.shape[0]:
            block_requests = _pad_capacity(block_requests, block_capacity, 0)
            block_offsets = _pad_capacity(block_offsets, block_capacity, INERT_QUERY_OFFSET)
        if decode_requests.shape[0]:
            decode_capacity = _bucket_capacity(
                decode_requests.shape[0], _DECODE_IDS_FLOOR, bound=batch_size
            )
            decode_requests = _pad_capacity(decode_requests, decode_capacity, -1)
        if boundary_tokens.shape[0]:
            boundary_bound = tokens // self.compress_ratio + batch_size
            boundary_capacity = _bucket_capacity(
                boundary_tokens.shape[0], _BOUNDARY_FLOOR, bound=boundary_bound
            )
            boundary_tokens = _pad_capacity(boundary_tokens, boundary_capacity, tokens)
        window_pages = _pad_capacity(
            window_page_indices,
            _bucket_capacity(window_page_indices.shape[0], _WINDOW_TABLE_FLOOR),
            0,
        )
        compressed_pages = _pad_capacity(
            compressed_page_indices,
            _bucket_capacity(compressed_page_indices.shape[0], _COMPRESSED_TABLE_FLOOR),
            0,
        )
        max_queries = _bucket_max_queries(int(q_lens.max()), schedule.query_block_size)
        arrays = device_array(
            (
                state_slots,
                query_seq_ids.astype(np.int32),
                cu_q_lens,
                valid_token_mask,
                boundary_tokens,
                window_pages,
                window_cu_kv_lens,
                seq_lens,
                compressed_pages,
                compressed_cu_kv_lens,
                compressed_kv_lens,
                block_requests,
                block_offsets,
                decode_requests,
            ),
            sharding=NamedSharding(self.mesh, P("data")),
        )
        kernel_metadata = HCAMetadata(
            *arrays,
            max_queries_per_request=max_queries,
        )
        return HCABackendMetadata(
            kernel=kernel_metadata,
            schedule=schedule,
            use_uniform_prefill_fast_path=uniform_prefill,
        )

    def tree_flatten(self):
        children = (self.forward_metadata,)
        aux = {
            "num_attn_heads": self.num_heads,
            "head_dim": self.head_dim,
            "compressor_hidden_size": self.compressor_hidden_size,
            "page_size": self.page_size,
            "compress_ratio": self.compress_ratio,
            "window_size": self.window_size,
            "mesh": self.mesh,
        }
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = cls(**aux)
        obj.forward_metadata = children[0]
        return obj

    def __call__(
        self,
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        layer,
        forward_batch: ForwardBatch,
        token_to_kv_pool,
        *,
        recurrent_state_pool,
        compressor_input: jax.Array,
        wkv: jax.Array,
        wgate: jax.Array,
        ape: jax.Array,
        norm_weight: jax.Array,
        cos: jax.Array,
        sin: jax.Array,
        attention_sink: jax.Array,
        fused_weight: jax.Array | None = None,
        metadata=None,
        **_kwargs,
    ) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array]]:
        """Run complete cache-aware HCA and return explicit pool updates."""
        metadata = self.forward_metadata if metadata is None else metadata
        layer_id = int(layer.layer_id)
        layer_index = token_to_kv_pool._layer_index(layer_id)
        recurrent_state_pool._layer_index(layer_id)
        return run_hca(
            q,
            k,
            v,
            mesh=self.mesh,
            page_size=self.page_size,
            max_context_len=token_to_kv_pool.max_context_len,
            positions=forward_batch.positions,
            forward_mode=forward_batch.forward_mode,
            state_arg=recurrent_state_pool.get_hca_state(layer_id),
            window_arg=token_to_kv_pool.window_buffer[layer_index],
            compressed_arg=token_to_kv_pool.compressed_buffer[layer_index],
            metadata=metadata,
            compressor_input=compressor_input,
            wkv=wkv,
            wgate=wgate,
            ape=ape,
            norm_weight=norm_weight,
            cos=cos,
            sin=sin,
            attention_sink=attention_sink,
            fused_weight=fused_weight,
            softmax_scale=getattr(layer, "scaling", None),
        )

    @staticmethod
    def pack_pool_updates(layer_updates) -> dict:
        """Regroup per-layer ``(state, window, compressed)`` for their owning pools."""
        states, windows, compressed = zip(*layer_updates, strict=True)
        return {
            "token_to_kv_pool": {
                "window_buffer": list(windows),
                "compressed_buffer": list(compressed),
            },
            "recurrent_state_pool": {"state_buffers": list(states)},
        }

    @staticmethod
    def get_max_running_reqests(max_context_len: int, page_size: int) -> int:
        pages_per_request = (max_context_len + page_size - 1) // page_size
        return max(1, 1024 * 1024 // 2 // pages_per_request // 4)


__all__ = ["HCABackend", "HCABackendMetadata"]
