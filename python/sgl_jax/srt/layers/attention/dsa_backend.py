"""Reference backend for GLM/DeepSeek sparse MLA with framework-owned writes."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from jax.tree_util import register_pytree_node_class

from sgl_jax.srt.kernels.dsa.reference import (
    dsa_sparse_mla_reference,
    logical_topk_to_physical_slots,
    write_indexer_k_cache,
    write_mla_kv_cache,
)
from sgl_jax.srt.kernels.mla.dsa.kernel import dsa_decode_mla_attention_unchecked
from sgl_jax.srt.layers.attention.base_attn_backend import AttentionBackend
from sgl_jax.srt.layers.attention.dsa_types import DsaTopKState
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode
from sgl_jax.srt.utils.jax_utils import device_array


@register_pytree_node_class
@dataclass
class DsaAttentionMetadata:
    """Static-shape request mapping used by Indexer selection for one forward."""

    req_to_token_slots: jax.Array = None
    query_request_indices: jax.Array = None
    query_positions: jax.Array = None
    query_offsets: jax.Array = None
    request_offsets: jax.Array = None

    def tree_flatten(self):
        return (
            (
                self.req_to_token_slots,
                self.query_request_indices,
                self.query_positions,
                self.query_offsets,
                self.request_offsets,
            ),
            None,
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del aux_data
        return cls(*children)


def _gather_candidate_rows(req_to_token_slots, safe_requests):
    """Gather request rows while keeping the mapping's data sharding explicit."""
    return req_to_token_slots.at[safe_requests].get(
        out_sharding=jax.typeof(req_to_token_slots).sharding,
    )


@dataclass
class DsaAttentionBackend(AttentionBackend):
    """Correctness-first DSA backend: write caches, then run sparse MLA reference."""

    def __init__(
        self,
        num_attn_heads: int,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        index_head_dim: int,
        index_topk: int,
        page_size: int,
        mesh: jax.sharding.Mesh | None,
        attention_data_partition_axis: str = "data",
        use_pallas_kernel: bool | None = None,
    ):
        if page_size <= 0:
            raise ValueError("DSA page_size must be positive")
        if index_head_dim <= 0 or index_topk <= 1:
            raise ValueError("DSA index_head_dim must be positive and index_topk > 1")
        self.num_heads = num_attn_heads
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.index_head_dim = index_head_dim
        self.index_topk = index_topk
        self.page_size = page_size
        self.mesh = mesh
        self.attention_data_partition_axis = attention_data_partition_axis
        self.use_pallas_kernel = (
            jax.default_backend() == "tpu" if use_pallas_kernel is None else use_pallas_kernel
        )
        self.forward_metadata = nnx.data(DsaAttentionMetadata())

    def _device_metadata(self, metadata: DsaAttentionMetadata) -> DsaAttentionMetadata:
        if self.mesh is None:
            return jax.tree.map(jnp.asarray, metadata)

        dpa = self.attention_data_partition_axis
        return DsaAttentionMetadata(
            req_to_token_slots=device_array(
                metadata.req_to_token_slots,
                sharding=NamedSharding(self.mesh, P(dpa, None)),
            ),
            query_request_indices=device_array(
                metadata.query_request_indices,
                sharding=NamedSharding(self.mesh, P(dpa)),
            ),
            query_positions=device_array(
                metadata.query_positions,
                sharding=NamedSharding(self.mesh, P(dpa)),
            ),
            query_offsets=device_array(
                metadata.query_offsets,
                sharding=NamedSharding(self.mesh, P(dpa)),
            ),
            request_offsets=device_array(
                metadata.request_offsets,
                sharding=NamedSharding(self.mesh, P(dpa)),
            ),
        )

    def get_forward_metadata(self, batch):
        """Build a single-DP causal request-to-slot table from scheduler metadata."""
        if getattr(batch, "dp_size", 1) != 1:
            raise NotImplementedError("DSA reference backend currently supports dp_size=1 only")

        seq_lens = np.asarray(batch.seq_lens, dtype=np.int32)
        cache_loc = np.asarray(batch.cache_loc, dtype=np.int32)
        positions = np.asarray(batch.positions, dtype=np.int32)
        token_count = len(batch.input_ids)
        request_count = len(seq_lens)
        candidate_width = max(1, int(seq_lens.max(initial=0)))
        candidate_width = (candidate_width + self.page_size - 1) // self.page_size * self.page_size

        aligned_lens = ((seq_lens + self.page_size - 1) // self.page_size * self.page_size).astype(
            np.int32
        )
        request_offsets = np.zeros(request_count + 1, dtype=np.int32)
        if request_count:
            request_offsets[1:] = np.cumsum(aligned_lens, dtype=np.int32)
        if request_offsets[-1] > len(cache_loc):
            raise ValueError(
                "DSA cache_loc is shorter than the page-aligned request mapping: "
                f"need {request_offsets[-1]}, got {len(cache_loc)}"
            )

        req_to_token_slots = np.zeros((request_count, candidate_width), dtype=np.int32)
        for request_index, seq_len in enumerate(seq_lens):
            start = int(request_offsets[request_index])
            req_to_token_slots[request_index, : int(seq_len)] = cache_loc[
                start : start + int(seq_len)
            ]

        query_request_indices = np.full(token_count, -1, dtype=np.int32)
        if batch.forward_mode == ForwardMode.DECODE:
            real_requests = min(request_count, token_count)
            valid_requests = seq_lens[:real_requests] > 0
            query_request_indices[:real_requests] = np.where(
                valid_requests,
                np.arange(real_requests, dtype=np.int32),
                -1,
            )
            query_offsets = np.arange(request_count + 1, dtype=np.int32)
        elif batch.forward_mode == ForwardMode.EXTEND:
            if batch.extend_seq_lens is None:
                raise ValueError("DSA prefill metadata requires extend_seq_lens")
            extend_lens = np.asarray(batch.extend_seq_lens, dtype=np.int32)
            if extend_lens.shape != seq_lens.shape:
                raise ValueError("extend_seq_lens must match seq_lens shape")
            query_offsets = np.zeros(request_count + 1, dtype=np.int32)
            if request_count:
                query_offsets[1:] = np.cumsum(extend_lens, dtype=np.int32)
            for request_index in range(request_count):
                start = int(query_offsets[request_index])
                end = min(int(query_offsets[request_index + 1]), token_count)
                query_request_indices[start:end] = request_index
        else:
            raise ValueError(
                "DSA reference backend supports only ordinary EXTEND and DECODE; "
                f"got {batch.forward_mode}"
            )

        if positions.shape != (token_count,):
            raise ValueError(
                f"DSA positions must have shape {(token_count,)}, got {positions.shape}"
            )
        return self._device_metadata(
            DsaAttentionMetadata(
                req_to_token_slots=req_to_token_slots,
                query_request_indices=query_request_indices,
                query_positions=positions,
                query_offsets=query_offsets,
                request_offsets=request_offsets,
            )
        )

    def tree_flatten(self):
        children = (self.forward_metadata,)
        aux_data = {
            "num_attn_heads": self.num_heads,
            "kv_lora_rank": self.kv_lora_rank,
            "qk_nope_head_dim": self.qk_nope_head_dim,
            "qk_rope_head_dim": self.qk_rope_head_dim,
            "v_head_dim": self.v_head_dim,
            "index_head_dim": self.index_head_dim,
            "index_topk": self.index_topk,
            "page_size": self.page_size,
            "mesh": self.mesh,
            "attention_data_partition_axis": self.attention_data_partition_axis,
            "use_pallas_kernel": self.use_pallas_kernel,
        }
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = cls(**aux_data)
        obj.forward_metadata = children[0]
        return obj

    def build_dsa_state(
        self,
        *,
        layer_id: int,
        q_index: jax.Array,
        head_weights: jax.Array,
        index_k: jax.Array,
        forward_batch,
        indexer_k_pool,
        prev_dsa_state: DsaTopKState | None,
    ):
        """Write current Index-K first, then select causal physical MLA slots."""
        del prev_dsa_state
        if indexer_k_pool.page_size != self.page_size:
            raise ValueError("Index-K and main MLA pools must use the same page_size")
        if indexer_k_pool.index_head_dim != self.index_head_dim:
            raise ValueError("Index-K pool and backend index_head_dim must match")

        current_cache = indexer_k_pool.get_buffer(layer_id)
        updated_cache = write_indexer_k_cache(
            current_cache,
            index_k=index_k,
            write_slots=forward_batch.out_cache_loc,
            page_size=self.page_size,
            index_head_dim=self.index_head_dim,
        )

        metadata = self.forward_metadata
        request_count, candidate_width = metadata.req_to_token_slots.shape
        request_valid = (metadata.query_request_indices >= 0) & (
            metadata.query_request_indices < request_count
        )
        safe_requests = jnp.clip(
            metadata.query_request_indices,
            0,
            max(request_count - 1, 0),
        )
        candidate_slots = _gather_candidate_rows(
            metadata.req_to_token_slots,
            safe_requests,
        )
        candidate_logical_ids = jnp.broadcast_to(
            jnp.arange(candidate_width, dtype=jnp.int32)[None, :],
            candidate_slots.shape,
        )
        candidate_counts = jnp.where(
            request_valid,
            jnp.clip(metadata.query_positions + 1, 0, candidate_width),
            0,
        ).astype(jnp.int32)

        pages = updated_cache.shape[0]
        token_order_cache = updated_cache.reshape(
            pages,
            -1,
            updated_cache.shape[-1],
        )[:, : self.page_size, : self.index_head_dim]
        flat_index_cache = token_order_cache.reshape(-1, self.index_head_dim)

        from sgl_jax.srt.models.glm5_moe import GlmDsaIndexer

        logical_topk_ids, selected_counts = GlmDsaIndexer.select_topk(
            q_index,
            head_weights,
            flat_index_cache,
            candidate_slots,
            candidate_logical_ids,
            candidate_counts,
            index_topk=self.index_topk,
        )
        selection = logical_topk_to_physical_slots(
            logical_topk_ids=logical_topk_ids,
            selected_counts=selected_counts,
            req_to_token_slots=metadata.req_to_token_slots,
            query_request_indices=metadata.query_request_indices,
            query_positions=metadata.query_positions,
            producer_layer=layer_id,
        )
        return (
            DsaTopKState(
                selection=selection,
                query_offsets=metadata.query_offsets,
                request_offsets=metadata.request_offsets,
            ),
            updated_cache,
        )

    def __call__(
        self,
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        layer,
        forward_batch,
        token_to_kv_pool,
        **kwargs,
    ):
        """Write current MLA KV, then read selected slots from the updated cache."""
        del v
        q_rope = kwargs.get("q_rope")
        k_rope = kwargs.get("k_rope")
        dsa_state = kwargs.get("dsa_state")
        if q_rope is None or k_rope is None:
            raise ValueError("DSA backend requires q_rope and k_rope")
        if not isinstance(dsa_state, DsaTopKState):
            raise TypeError("DSA backend requires a DsaTopKState")

        new_c_kv = k if k.ndim == 2 else jnp.squeeze(k, axis=1)
        new_k_pe = k_rope if k_rope.ndim == 2 else jnp.squeeze(k_rope, axis=1)
        cache = token_to_kv_pool.get_fused_kv_buffer(layer.layer_id)
        updated_cache = write_mla_kv_cache(
            cache,
            new_c_kv=new_c_kv,
            new_k_pe=new_k_pe,
            write_slots=forward_batch.out_cache_loc,
            page_size=self.page_size,
            latent_dim=self.kv_lora_rank,
            rope_dim=self.qk_rope_head_dim,
        )
        sm_scale = (
            (self.qk_nope_head_dim + self.qk_rope_head_dim) ** -0.5
            if layer is None or layer.scaling is None
            else layer.scaling
        )
        if self.use_pallas_kernel:

            def _run_pallas(q_, q_rope_, cache_, slots_, counts_):
                return dsa_decode_mla_attention_unchecked(
                    q_,
                    q_rope_,
                    cache_,
                    slots_,
                    counts_,
                    sm_scale=sm_scale,
                    interpret=False,
                )

            if self.mesh is None:
                output = _run_pallas(
                    q,
                    q_rope,
                    updated_cache,
                    dsa_state.selection.physical_slots,
                    dsa_state.selection.selected_counts,
                )
            else:
                dpa = self.attention_data_partition_axis
                output = jax.shard_map(
                    _run_pallas,
                    mesh=self.mesh,
                    in_specs=(
                        P(dpa, "tensor", None),
                        P(dpa, "tensor", None),
                        P(dpa, None, None, None),
                        P(dpa, None),
                        P(dpa),
                    ),
                    out_specs=P(dpa, "tensor", None),
                    check_vma=False,
                )(
                    q,
                    q_rope,
                    updated_cache,
                    dsa_state.selection.physical_slots,
                    dsa_state.selection.selected_counts,
                )
        else:
            output = dsa_sparse_mla_reference(
                q,
                q_rope,
                updated_cache,
                dsa_state.selection.physical_slots,
                dsa_state.selection.selected_counts,
                sm_scale=sm_scale,
                page_size=self.page_size,
                latent_dim=self.kv_lora_rank,
                rope_dim=self.qk_rope_head_dim,
            )
        return output.astype(jnp.bfloat16), updated_cache

    @staticmethod
    def get_max_running_reqests(max_context_len: int, page_size: int) -> int:
        from sgl_jax.srt.layers.attention.mla_backend import MLAAttentionBackend

        return MLAAttentionBackend.get_max_running_reqests(max_context_len, page_size)
