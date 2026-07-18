import logging
from typing import Any

import jax
from flax import nnx
from jax import numpy as jnp
from jax.sharding import PartitionSpec as P
from transformers import PretrainedConfig

from sgl_jax.srt.configs.model_config import ModelConfig, MoEBackend
from sgl_jax.srt.eplb.expert_location import ExpertLocationMetadata
from sgl_jax.srt.kernels.fused_mlp import apply_fused_mlp_with_padding
from sgl_jax.srt.layers.attention.dsa_types import DsaTopKState
from sgl_jax.srt.layers.embeddings import Embed, ParallelLMHead, RotaryEmbedding
from sgl_jax.srt.layers.layernorm import RMSNorm
from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sgl_jax.srt.layers.moe import (
    EPMoE,
    FusedEPMoE,
    GateLogit,
    TopK,
    create_moe_weights_mapping,
)
from sgl_jax.srt.layers.radix_attention import RadixAttention
from sgl_jax.srt.mem_cache.memory_pool import KVCache
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
from sgl_jax.srt.utils.weight_utils import WeightLoader, WeightMapping

logger = logging.getLogger(__name__)


# No-op: FP32 accumulation logic removed to keep native BF16 execution.


class GlmNorm(nnx.Module):
    def __init__(self, dim: int, dtype: jnp.dtype = jnp.bfloat16):
        self.weight = nnx.Param(jnp.ones((dim,), dtype=dtype))
        self.bias = nnx.Param(jnp.zeros((dim,), dtype=dtype))

    def __call__(self, x: jax.Array) -> jax.Array:
        mean = jnp.mean(x, axis=-1, keepdims=True)
        variance = jnp.var(x, axis=-1, keepdims=True)
        eps = 1e-5
        normalized = (x - mean) / jnp.sqrt(variance + eps)
        return normalized * self.weight.value + self.bias.value


def get_hadamard_matrix(n: int) -> jax.Array:
    if n <= 0 or n & (n - 1):
        raise ValueError(f"Hadamard dimension must be a positive power of two; got {n}")
    if n == 1:
        return jnp.array([[1.0]])
    h = get_hadamard_matrix(n // 2)
    return jnp.block([[h, h], [h, -h]])


class GlmDsaIndexer(nnx.Module):
    def __init__(
        self,
        hidden_size: int,
        q_lora_rank: int,
        index_head_dim: int,
        index_n_heads: int,
        index_topk: int,
        rope_head_dim: int,
        max_position_embeddings: int,
        rope_theta: float,
        rope_scaling: dict[str, Any] | None,
        indexer_rope_interleave: bool,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
        scope_name: str = "indexer",
    ):
        if index_head_dim <= 0 or index_head_dim & (index_head_dim - 1):
            raise ValueError(
                "index_head_dim must be a positive power of two for the "
                f"Hadamard transform; got {index_head_dim}"
            )
        if index_n_heads <= 0:
            raise ValueError(f"index_n_heads must be positive; got {index_n_heads}")
        if index_topk <= 1:
            raise ValueError(f"index_topk must be greater than one; got {index_topk}")
        if rope_head_dim <= 0 or rope_head_dim > index_head_dim:
            raise ValueError(
                "rope_head_dim must be in (0, index_head_dim]; got "
                f"rope_head_dim={rope_head_dim}, index_head_dim={index_head_dim}"
            )

        self.index_head_dim = index_head_dim
        self.index_n_heads = index_n_heads
        self.index_topk = index_topk
        self.rope_head_dim = rope_head_dim
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        # RotaryEmbedding does not yet implement scaled RoPE. Keep the exact
        # checkpoint value visible so the Falcon parity gate cannot silently
        # claim that scaling was applied.
        self.rope_scaling = rope_scaling
        self.indexer_rope_interleave = indexer_rope_interleave
        self.mesh = mesh

        self.wq_b = LinearBase(
            input_size=q_lora_rank,
            output_size=index_head_dim * index_n_heads,
            use_bias=False,
            kernel_axes=(None, None),
            params_dtype=dtype,
            mesh=mesh,
            scope_name="wq_b",
        )
        self.wk = LinearBase(
            input_size=hidden_size,
            output_size=index_head_dim,
            use_bias=False,
            kernel_axes=(None, None),
            params_dtype=dtype,
            mesh=mesh,
            scope_name="wk",
        )
        self.k_norm = GlmNorm(index_head_dim, dtype)

        self.weights_proj = LinearBase(
            input_size=hidden_size,
            output_size=index_n_heads,
            use_bias=False,
            kernel_axes=(None, None),
            params_dtype=dtype,
            mesh=mesh,
            scope_name="weights_proj",
        )
        self.rotary_emb = RotaryEmbedding(
            head_size=index_head_dim,
            rotary_dim=rope_head_dim,
            max_position_embeddings=max_position_embeddings,
            base=rope_theta,
            is_neox_style=not indexer_rope_interleave,
            dtype=dtype,
            mesh=mesh,
        )

    @staticmethod
    def score_candidates(
        q_index: jax.Array,
        head_weights: jax.Array,
        k_index_cache: jax.Array,
        candidate_slots: jax.Array,
    ) -> jax.Array:
        """Compute the unmasked GLM DSA score for each candidate slot."""
        if q_index.ndim != 3:
            raise ValueError(f"q_index must have rank 3; got {q_index.ndim}")
        if head_weights.shape != q_index.shape[:2]:
            raise ValueError(
                "head_weights must match q_index [tokens, heads]; got "
                f"{head_weights.shape} and {q_index.shape}"
            )
        if k_index_cache.ndim != 2 or k_index_cache.shape[1] != q_index.shape[2]:
            raise ValueError(
                "k_index_cache must have shape [slots, q_index_dim]; got "
                f"{k_index_cache.shape} for q_index {q_index.shape}"
            )
        if k_index_cache.shape[0] == 0:
            raise ValueError("k_index_cache must contain at least one safe slot")
        if candidate_slots.ndim != 2 or candidate_slots.shape[0] != q_index.shape[0]:
            raise ValueError(
                "candidate_slots must have shape [tokens, candidates]; got "
                f"{candidate_slots.shape} for q_index {q_index.shape}"
            )
        if candidate_slots.dtype != jnp.int32:
            raise TypeError(f"candidate_slots must have dtype int32; got {candidate_slots.dtype}")

        safe_slots = jnp.clip(candidate_slots, 0, k_index_cache.shape[0] - 1)
        candidate_keys = k_index_cache[safe_slots]
        logits = jnp.einsum(
            "thd,tcd->tch",
            q_index.astype(jnp.float32),
            candidate_keys.astype(jnp.float32),
        )
        scores = jnp.sum(
            jax.nn.relu(logits) * head_weights[:, None, :].astype(jnp.float32),
            axis=-1,
        )
        return scores * (q_index.shape[2] ** -0.5) * (q_index.shape[1] ** -0.5)

    @staticmethod
    def select_topk(
        q_index: jax.Array,
        head_weights: jax.Array,
        k_index_cache: jax.Array,
        candidate_slots: jax.Array,
        candidate_logical_ids: jax.Array,
        candidate_counts: jax.Array,
        *,
        index_topk: int,
    ) -> tuple[jax.Array, jax.Array]:
        """Select logical IDs while reading Index-K from physical candidate slots."""
        if index_topk <= 1:
            raise ValueError(f"index_topk must be greater than one; got {index_topk}")
        if candidate_logical_ids.shape != candidate_slots.shape:
            raise ValueError(
                "candidate_logical_ids must match candidate_slots shape; got "
                f"{candidate_logical_ids.shape} and {candidate_slots.shape}"
            )
        if candidate_logical_ids.dtype != jnp.int32:
            raise TypeError(
                "candidate_logical_ids must have dtype int32; got " f"{candidate_logical_ids.dtype}"
            )
        if candidate_counts.shape != (q_index.shape[0],):
            raise ValueError(
                f"candidate_counts must have shape {(q_index.shape[0],)}; "
                f"got {candidate_counts.shape}"
            )
        if candidate_counts.dtype != jnp.int32:
            raise TypeError(f"candidate_counts must have dtype int32; got {candidate_counts.dtype}")

        candidate_width = candidate_slots.shape[1]
        scores = GlmDsaIndexer.score_candidates(
            q_index,
            head_weights,
            k_index_cache,
            candidate_slots,
        )

        bounded_counts = jnp.clip(candidate_counts, 0, candidate_width)
        candidate_valid = jnp.arange(candidate_width)[None, :] < bounded_counts[:, None]
        scores = jnp.where(candidate_valid, scores, -jnp.inf)

        pad_width = max(0, index_topk - candidate_width)
        if pad_width:
            scores = jnp.pad(scores, ((0, 0), (0, pad_width)), constant_values=-jnp.inf)
            candidate_logical_ids = jnp.pad(
                candidate_logical_ids,
                ((0, 0), (0, pad_width)),
                constant_values=-1,
            )

        _, topk_offsets = jax.lax.top_k(scores, index_topk)
        logical_topk_ids = jnp.take_along_axis(candidate_logical_ids, topk_offsets, axis=1)
        selected_counts = jnp.minimum(bounded_counts, index_topk).astype(jnp.int32)
        selected_valid = jnp.arange(index_topk)[None, :] < selected_counts[:, None]
        logical_topk_ids = jnp.where(selected_valid, logical_topk_ids, -1).astype(jnp.int32)
        return logical_topk_ids, selected_counts

    def _hadamard_rotate(self, value: jax.Array) -> jax.Array:
        matrix = get_hadamard_matrix(self.index_head_dim).astype(value.dtype)
        matrix = matrix * jnp.asarray(self.index_head_dim**-0.5, dtype=value.dtype)
        return jnp.einsum("...d,de->...e", value, matrix)

    def project_query(
        self,
        hidden_states: jax.Array,
        q_lora: jax.Array,
        positions: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        query, _ = self.wq_b(q_lora)
        query = query.reshape(-1, self.index_n_heads, self.index_head_dim)
        query, _ = self.rotary_emb(positions, query, query[:, :1, :])
        query = self._hadamard_rotate(query)
        head_weights, _ = self.weights_proj(hidden_states)
        return query, head_weights

    def project_key(self, hidden_states: jax.Array, positions: jax.Array) -> jax.Array:
        key, _ = self.wk(hidden_states)
        key = self.k_norm(key)
        key_with_head = key[:, None, :]
        _, key_with_head = self.rotary_emb(positions, key_with_head, key_with_head)
        return self._hadamard_rotate(key_with_head[:, 0, :])

    def __call__(
        self, hidden_states: jax.Array, q_lora: jax.Array, positions: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Project current tokens; Task 3 supplies cached candidates for Top-K."""
        query, head_weights = self.project_query(hidden_states, q_lora, positions)
        key = self.project_key(hidden_states, positions)
        return query, head_weights, key


class Glm5Attention(nnx.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position_embeddings: int,
        mesh: jax.sharding.Mesh,
        rope_theta: float = 1000000,
        rope_scaling: dict[str, Any] | None = None,
        head_dim: int | None = None,
        qk_rope_head_dim: int = 64,
        index_head_dim: int = 128,
        index_n_heads: int = 32,
        index_topk: int = 2048,
        indexer_rope_interleave: bool = False,
        rms_norm_eps: float = None,
        use_qk_norm: bool = True,
        rotary_dim: int = 0,
        layer_id: int = 0,
        attention_bias: bool = False,
        dtype: jnp.dtype = jnp.bfloat16,
        use_absorbed: bool = True,
        has_indexer: bool = True,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.mesh = mesh
        self.num_heads = num_heads
        self.kv_head_num = num_kv_heads

        self.qk_nope_head_dim = 192
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.v_head_dim = 256
        self.kv_lora_rank = 512
        self.q_lora_rank = 2048

        self.scaling = 256**-0.5

        self.use_qk_norm = use_qk_norm

        if use_qk_norm:
            self.q_norm = RMSNorm(256, epsilon=rms_norm_eps, param_dtype=dtype, scope_name="q_norm")
            self.k_norm = RMSNorm(256, epsilon=rms_norm_eps, param_dtype=dtype, scope_name="k_norm")
        else:
            self.q_norm = None
            self.k_norm = None

        self.q_a_proj = LinearBase(
            input_size=hidden_size,
            output_size=self.q_lora_rank,
            use_bias=False,
            kernel_axes=(None, None),
            params_dtype=dtype,
            mesh=mesh,
            scope_name="q_a_proj",
        )
        self.q_a_layernorm = RMSNorm(
            self.q_lora_rank, epsilon=rms_norm_eps, param_dtype=dtype, scope_name="q_a_layernorm"
        )
        self.q_b_proj = LinearBase(
            input_size=self.q_lora_rank,
            output_size=num_heads * self.qk_head_dim,
            use_bias=False,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            mesh=mesh,
            scope_name="q_b_proj",
        )
        self.kv_a_proj_with_mqa = LinearBase(
            input_size=hidden_size,
            output_size=self.kv_lora_rank + self.qk_rope_head_dim,
            use_bias=False,
            kernel_axes=(None, None),
            params_dtype=dtype,
            mesh=mesh,
            scope_name="kv_a_proj_with_mqa",
        )
        self.kv_a_layernorm = RMSNorm(
            self.kv_lora_rank, epsilon=rms_norm_eps, param_dtype=dtype, scope_name="kv_a_layernorm"
        )

        self.kv_b_proj = LinearBase(
            input_size=self.kv_lora_rank,
            output_size=num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            use_bias=False,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            mesh=mesh,
            scope_name="kv_b_proj",
        )

        self.o_proj = LinearBase(
            input_size=num_heads * self.v_head_dim,
            output_size=hidden_size,
            use_bias=False,
            kernel_axes=("tensor", None),
            params_dtype=dtype,
            mesh=mesh,
            scope_name="o_proj",
        )

        if has_indexer:
            self.indexer = GlmDsaIndexer(
                hidden_size=hidden_size,
                q_lora_rank=self.q_lora_rank,
                index_head_dim=index_head_dim,
                index_n_heads=index_n_heads,
                index_topk=index_topk,
                rope_head_dim=self.qk_rope_head_dim,
                max_position_embeddings=max_position_embeddings,
                rope_theta=rope_theta,
                rope_scaling=rope_scaling,
                indexer_rope_interleave=indexer_rope_interleave,
                mesh=mesh,
                dtype=dtype,
                scope_name="indexer",
            )
        else:
            self.indexer = None
        self.rotary_emb = RotaryEmbedding(
            head_size=self.qk_rope_head_dim,
            rotary_dim=self.qk_rope_head_dim,
            max_position_embeddings=max_position_embeddings,
            base=rope_theta,
            is_neox_style=False,
            dtype=dtype,
            mesh=mesh,
        )

        self.use_absorbed = use_absorbed

        if use_absorbed:
            uk_axes = (None, "tensor", None)
            self.w_uk = nnx.Param(
                jnp.zeros(
                    (self.kv_lora_rank, num_heads, self.qk_nope_head_dim),
                    dtype=dtype,
                    out_sharding=P(*uk_axes),
                )
            )
            self.w_uv = nnx.Param(
                jnp.zeros(
                    (self.kv_lora_rank, num_heads, self.v_head_dim),
                    dtype=dtype,
                    out_sharding=P(*uk_axes),
                )
            )
            self.attn_mqa = RadixAttention(
                num_heads=num_heads,
                head_dim=self.kv_lora_rank + self.qk_rope_head_dim,
                scaling=self.scaling,
                num_kv_heads=1,
                v_head_dim=self.kv_lora_rank,
                layer_id=layer_id,
            )
        else:
            self.w_uk = None
            self.w_uv = None
            self.attn_mqa = None

        self.attn_mha = RadixAttention(
            num_heads=num_heads,
            head_dim=self.qk_head_dim,
            scaling=self.scaling,
            num_kv_heads=num_heads,
            v_head_dim=self.qk_head_dim,
            layer_id=layer_id,
        )

    def post_load_weights(self):
        if not self.use_absorbed:
            return
        if self.kv_b_proj is None:
            return
        if hasattr(self.kv_b_proj, "weight"):
            raw_weight = self.kv_b_proj.weight.value
        else:
            wq = self.kv_b_proj.weight_q.value
            ws = self.kv_b_proj.weight_scale.value
            wq_f32 = wq.T.astype(jnp.float32)
            if ws.ndim == 3:
                in_blocks, _, n_out = ws.shape
                block_k = wq.shape[1] // in_blocks
                wq_f32 = wq_f32.reshape(in_blocks, block_k, n_out)
                wq_f32 = (wq_f32 * ws.astype(jnp.float32)).reshape(in_blocks * block_k, n_out)
            else:
                wq_f32 = wq_f32 * ws.astype(jnp.float32)[None, :]
            raw_weight = wq_f32.astype(jnp.bfloat16)
        w_kv = raw_weight.reshape(
            self.kv_lora_rank,
            self.num_heads,
            self.qk_nope_head_dim + self.v_head_dim,
        )
        self.w_uk.value = w_kv[:, :, : self.qk_nope_head_dim]
        self.w_uv.value = w_kv[:, :, self.qk_nope_head_dim :]
        self.kv_b_proj = None

    def _forward_mqa(
        self,
        q_nope: jax.Array,
        q_rope: jax.Array,
        compressed: jax.Array,
        k_rope: jax.Array,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
        dsa_state: DsaTopKState | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        # "thd,rhd->thr"
        ql_nope = jax.lax.dot_general(
            q_nope,
            self.w_uk.value,
            (((2,), (2,)), ((1,), (1,))),
        )
        ql_nope = ql_nope.transpose(1, 0, 2)

        c_kv_3d = compressed[:, None, :]
        attention_kwargs = {
            "q_rope": q_rope,
            "k_rope": k_rope,
        }
        if dsa_state is not None:
            attention_kwargs["dsa_state"] = dsa_state
        attn_output, kv_fused = self.attn_mqa(
            ql_nope,
            c_kv_3d,
            c_kv_3d,
            forward_batch=forward_batch,
            token_to_kv_pool=token_to_kv_pool,
            **attention_kwargs,
        )
        # "thr,rhd->thd"
        o_v = jax.lax.dot_general(
            attn_output,
            self.w_uv.value,
            (((2,), (0,)), ((1,), (1,))),
        )
        o_v = o_v.transpose(1, 0, 2)
        attn_output = o_v.reshape(-1, self.num_heads * self.v_head_dim)
        return attn_output, kv_fused

    def _forward_mha(
        self,
        q_nope: jax.Array,
        q_rope: jax.Array,
        compressed: jax.Array,
        k_rope: jax.Array,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
        dsa_state: DsaTopKState | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        kv, _ = self.kv_b_proj(compressed)
        kv = kv.reshape(-1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = jnp.split(kv, [self.qk_nope_head_dim], axis=-1)

        k_rope = jnp.broadcast_to(
            k_rope,
            (k_rope.shape[0], self.num_heads, self.qk_rope_head_dim),
            out_sharding=P("data", "tensor", None),
        )

        q = jnp.concatenate([q_nope, q_rope], axis=-1)
        k = jnp.concatenate([k_nope, k_rope], axis=-1)

        attention_kwargs = {}
        if dsa_state is not None:
            attention_kwargs["dsa_state"] = dsa_state
        attn_output, kv_fused = self.attn_mha(
            q,
            k,
            v,
            forward_batch=forward_batch,
            token_to_kv_pool=token_to_kv_pool,
            **attention_kwargs,
        )
        attn_output = attn_output.reshape(-1, self.num_heads * self.v_head_dim)
        return attn_output, kv_fused

    def _build_or_share_dsa_state(
        self,
        *,
        hidden_states: jax.Array,
        q_lora: jax.Array,
        positions: jax.Array,
        forward_batch: ForwardBatch,
        indexer_k_pool,
        prev_dsa_state: DsaTopKState | None,
    ) -> tuple[DsaTopKState | None, jax.Array | None]:
        """Build a full-layer selection or reuse the previous IndexShare state."""
        if indexer_k_pool is None:
            return (prev_dsa_state if self.indexer is None else None), None

        if self.indexer is None:
            if prev_dsa_state is None:
                raise ValueError(
                    f"shared GLM DSA layer {self.layer_id} requires a previous full-layer state"
                )
            return prev_dsa_state, None

        build_dsa_state = getattr(forward_batch.attn_backend, "build_dsa_state", None)
        if build_dsa_state is None:
            raise TypeError("an Index-K pool requires an attention backend with build_dsa_state()")
        q_index, head_weights, index_k = self.indexer(
            hidden_states,
            q_lora,
            positions,
        )
        state, updated_index_cache = build_dsa_state(
            layer_id=self.layer_id,
            q_index=q_index,
            head_weights=head_weights,
            index_k=index_k,
            forward_batch=forward_batch,
            indexer_k_pool=indexer_k_pool,
            prev_dsa_state=prev_dsa_state,
        )
        if not isinstance(state, DsaTopKState):
            raise TypeError("build_dsa_state() must return a DsaTopKState")
        if state.selection.producer_layer != self.layer_id:
            raise ValueError(
                "full GLM DSA layer must own the produced selection; got "
                f"producer={state.selection.producer_layer}, layer={self.layer_id}"
            )
        return state, updated_index_cache

    def __call__(
        self,
        positions: jax.Array,
        hidden_states: jax.Array,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
        indexer_k_pool=None,
        prev_dsa_state: DsaTopKState | None = None,
    ) -> tuple[jax.Array, jax.Array, DsaTopKState | None, jax.Array | None]:
        q_compressed, _ = self.q_a_proj(hidden_states)
        q_compressed = self.q_a_layernorm(q_compressed)
        q, _ = self.q_b_proj(q_compressed)
        q = q.reshape(-1, self.num_heads, self.qk_head_dim)

        dsa_state, indexer_k_update = self._build_or_share_dsa_state(
            hidden_states=hidden_states,
            q_lora=q_compressed,
            positions=positions,
            forward_batch=forward_batch,
            indexer_k_pool=indexer_k_pool,
            prev_dsa_state=prev_dsa_state,
        )

        q_nope = q[:, :, : self.qk_nope_head_dim]
        q_rope = q[:, :, self.qk_nope_head_dim :]

        latent_cache, _ = self.kv_a_proj_with_mqa(hidden_states)
        compressed, k_rope = jnp.split(latent_cache, [self.kv_lora_rank], axis=-1)
        compressed = self.kv_a_layernorm(compressed)

        k_rope = k_rope.reshape(-1, 1, self.qk_rope_head_dim)
        q_rope, k_rope = self.rotary_emb(positions, q_rope, k_rope)

        if self.use_absorbed:
            attn_output, kv_fused = self._forward_mqa(
                q_nope,
                q_rope,
                compressed,
                k_rope,
                forward_batch,
                token_to_kv_pool,
                dsa_state,
            )
        else:
            attn_output, kv_fused = self._forward_mha(
                q_nope,
                q_rope,
                compressed,
                k_rope,
                forward_batch,
                token_to_kv_pool,
                dsa_state,
            )

        output, _ = self.o_proj(attn_output)
        return output, kv_fused, dsa_state, indexer_k_update


class Glm5MLP(nnx.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        mesh: jax.sharding.Mesh,
        layer_id: int = 0,
        dtype: jnp.dtype = jnp.bfloat16,
        use_fused: bool = True,
    ) -> None:
        self.layer_id = layer_id
        self.mesh = mesh
        self.use_fused = use_fused

        self.gate_proj = LinearBase(
            input_size=hidden_size,
            output_size=intermediate_size,
            kernel_axes=(None, "tensor"),
            use_bias=False,
            params_dtype=dtype,
            mesh=mesh,
            scope_name="gate_proj",
        )

        self.up_proj = LinearBase(
            input_size=hidden_size,
            output_size=intermediate_size,
            kernel_axes=(None, "tensor"),
            use_bias=False,
            params_dtype=dtype,
            mesh=mesh,
            scope_name="up_proj",
        )

        self.down_proj = LinearBase(
            input_size=intermediate_size,
            output_size=hidden_size,
            kernel_axes=("tensor", None),
            use_bias=False,
            params_dtype=dtype,
            mesh=mesh,
            scope_name="down_proj",
        )

        self.act_fn = jax.nn.silu

        if use_fused:
            tp_size = mesh.shape["tensor"]
            local_inter_size = intermediate_size // tp_size

            # Dynamically choose block size (B_INTER) based on local intermediate size
            # to ensure that num_blocks is always a multiple of the TP size.
            if local_inter_size >= 128:
                self.b_inter = 128
            elif local_inter_size >= 64:
                self.b_inter = 64
            else:
                self.b_inter = 32

            pad_inter = (self.b_inter - (local_inter_size % self.b_inter)) % self.b_inter
            local_inter_size_padded = local_inter_size + pad_inter
            global_inter_size_padded = local_inter_size_padded * tp_size

            # Pre-allocate fused parameters with correct global shape and sharding
            # under the active constructor mesh context.
            self.w_gu = nnx.Param(
                jnp.zeros((hidden_size, global_inter_size_padded * 2), dtype=dtype),
                out_sharding=P(None, "tensor"),
            )
            self.w_d = nnx.Param(
                jnp.zeros((global_inter_size_padded, hidden_size), dtype=dtype),
                out_sharding=P("tensor", None),
            )

    def post_load_weights(self):
        if not self.use_fused:
            return
        if not hasattr(self.gate_proj, "weight"):
            # static fp8 checkpoint: gate_proj is already QuantizedLinear
            # (weight_q/weight_scale), fused-merge path from #1344 only
            # handles bf16 LinearBase. Fall back to unfused (forward checks
            # hasattr(self, "w_gu")).
            return

        wg = self.gate_proj.weight.value
        wu = self.up_proj.weight.value
        wd = self.down_proj.weight.value

        # Use dynamically chosen block size
        b_inter = self.b_inter
        hidden_size, local_inter_size = wg.shape

        # Pad local intermediate dimension to a multiple of b_inter
        pad_inter = (b_inter - (local_inter_size % b_inter)) % b_inter
        if pad_inter > 0:
            wg = jnp.pad(wg, ((0, 0), (0, pad_inter)), mode="constant")
            wu = jnp.pad(wu, ((0, 0), (0, pad_inter)), mode="constant")
            wd = jnp.pad(wd, ((0, pad_inter), (0, 0)), mode="constant")
            local_inter_size += pad_inter

        # Combine wg and wu block-by-block using jax.lax.reshape to explicitly
        # specify the sharding for the split/merged dimensions under JAX SPMD.
        num_blocks = local_inter_size // b_inter
        sharding_3d = jax.sharding.NamedSharding(self.mesh, P(None, "tensor", None))
        wg_reshaped = jax.lax.reshape(
            wg, (hidden_size, num_blocks, b_inter), out_sharding=sharding_3d
        )
        wu_reshaped = jax.lax.reshape(
            wu, (hidden_size, num_blocks, b_inter), out_sharding=sharding_3d
        )

        # Concat along block dimension and flatten
        w_gu = jnp.concatenate([wg_reshaped, wu_reshaped], axis=-1)

        sharding_2d = jax.sharding.NamedSharding(self.mesh, P(None, "tensor"))
        w_gu = jax.lax.reshape(w_gu, (hidden_size, local_inter_size * 2), out_sharding=sharding_2d)

        # Assign values directly to pre-allocated sharded parameters
        self.w_gu.value = w_gu
        self.w_d.value = wd

        # Free original projection modules to save HBM
        self.gate_proj = None
        self.up_proj = None
        self.down_proj = None

    def __call__(self, hidden_states: jax.Array):
        if self.use_fused and hasattr(self, "w_gu"):
            seq_len, _ = hidden_states.shape
            b_seq = 64 if seq_len <= 8 else 256

            return apply_fused_mlp_with_padding(
                hidden_states,
                self.w_gu.value,
                self.w_d.value,
                self.mesh,
                b_seq=b_seq,
                b_inter=self.b_inter,
            )

        # Fallback non-fused path
        a1, _ = self.gate_proj(hidden_states)
        a2, _ = self.up_proj(hidden_states)
        intermediate_parallel = a2 * self.act_fn(a1)
        output, _ = self.down_proj(intermediate_parallel)
        return output


class Glm5DecoderLayer(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        layer_id: int = 0,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        rope_params = getattr(config, "rope_parameters", None) or {}
        rope_theta = getattr(config, "rope_theta", None) or rope_params.get("rope_theta", 1000000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings", 131072)
        self.head_dim = getattr(config, "head_dim", None) or 128
        use_qk_norm = getattr(config, "use_qk_norm", True)
        qk_rope_head_dim = getattr(
            config,
            "qk_rope_head_dim",
            getattr(config, "rope_head_dim", 64),
        )
        index_head_dim = getattr(config, "index_head_dim", 128)
        index_n_heads = getattr(config, "index_n_heads", 32)
        index_topk = getattr(config, "index_topk", 2048)
        indexer_rope_interleave = getattr(config, "indexer_rope_interleave", False)

        partial_rotary_factor = getattr(config, "partial_rotary_factor", 0.5)
        rotary_dim = int(self.head_dim * partial_rotary_factor)

        # GLM-5.2 IndexShare: layers tagged "shared" reuse the previous "full"
        # layer's top-k and ship no indexer weights. Dense MLA discards indexer
        # output anyway, so just skip building the module on shared layers.
        indexer_types = getattr(config, "indexer_types", None)
        has_indexer = indexer_types is None or indexer_types[layer_id] == "full"

        self.self_attn = Glm5Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            head_dim=self.head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            index_head_dim=index_head_dim,
            index_n_heads=index_n_heads,
            index_topk=index_topk,
            indexer_rope_interleave=indexer_rope_interleave,
            rms_norm_eps=config.rms_norm_eps,
            use_qk_norm=use_qk_norm,
            rotary_dim=rotary_dim,
            layer_id=layer_id,
            attention_bias=getattr(config, "attention_bias", False),
            dtype=dtype,
            mesh=mesh,
            use_absorbed=True,
            has_indexer=has_indexer,
        )

        first_k_dense_replace = getattr(config, "first_k_dense_replace", 0)
        use_fused_mlp = getattr(config, "_sgl_use_fused_mlp", True)

        if layer_id < first_k_dense_replace:
            self.mlp = Glm5MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                layer_id=layer_id,
                dtype=dtype,
                mesh=mesh,
                use_fused=use_fused_mlp,
            )
            self.is_moe_layer = False
            self.moe_gate = None
        else:
            router_dtype = jnp.float32
            self.moe_gate = GateLogit(
                input_size=config.hidden_size,
                num_experts=config.n_routed_experts,
                enable_expert_bias=True,
                weight_dtype=router_dtype,
                score_func=getattr(config, "scoring_func", "sigmoid"),
            )

            self.moe_backend = getattr(config, "moe_backend", MoEBackend.EPMOE)
            self.use_fused = self.moe_backend == MoEBackend.FUSED

            self.topk = TopK(
                topk=config.num_experts_per_tok,
                renormalize=config.norm_topk_prob,
                num_expert_group=getattr(config, "n_group", 1),
                topk_group=getattr(config, "topk_group", 1),
                routed_scaling_factor=getattr(config, "routed_scaling_factor", 1.0),
                layer_id=layer_id,
                mesh=mesh,
            )

            if self.use_fused:
                self.mlp = FusedEPMoE(
                    hidden_size=config.hidden_size,
                    num_experts=config.n_routed_experts,
                    num_experts_per_tok=config.num_experts_per_tok,
                    intermediate_dim=config.moe_intermediate_size,
                    mesh=mesh,
                    ep_size=getattr(config, "ep_size", 1),
                    weight_dtype=dtype,
                    dtype=dtype,
                    layer_id=layer_id,
                    renormalize_topk_logits=config.norm_topk_prob,
                    routed_scaling_factor=getattr(config, "routed_scaling_factor", 1.0),
                    use_grouped_topk=getattr(config, "n_group", 1) > 1,
                    num_groups=getattr(config, "n_group", 1),
                    top_k_groups=getattr(config, "topk_group", 1),
                    num_shared_experts=getattr(config, "n_shared_experts", 0),
                    moe_shared_expert_intermediate_size=config.moe_intermediate_size,
                    quantization_config=getattr(config, "quantization_config", None),
                )
            else:
                self.mlp = EPMoE(
                    hidden_size=config.hidden_size,
                    num_experts=config.n_routed_experts,
                    num_experts_per_tok=config.num_experts_per_tok,
                    intermediate_dim=config.moe_intermediate_size,
                    mesh=mesh,
                    ep_size=getattr(config, "ep_size", 1),
                    weight_dtype=dtype,
                    dtype=dtype,
                    layer_id=layer_id,
                    quantization_config=getattr(config, "quantization_config", None),
                )

            num_shared_experts = getattr(config, "n_shared_experts", 0)
            if num_shared_experts > 0 and not self.use_fused:
                self.shared_experts = Glm5MLP(
                    hidden_size=config.hidden_size,
                    intermediate_size=config.moe_intermediate_size * num_shared_experts,
                    layer_id=layer_id,
                    dtype=dtype,
                    mesh=mesh,
                    use_fused=use_fused_mlp,
                )
            else:
                self.shared_experts = None
            self.is_moe_layer = True

        self.input_layernorm = RMSNorm(
            config.hidden_size,
            epsilon=config.rms_norm_eps,
            param_dtype=dtype,
            scope_name="input_layernorm",
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            epsilon=config.rms_norm_eps,
            param_dtype=dtype,
            scope_name="post_attention_layernorm",
        )

    def __call__(
        self,
        positions: jax.Array,
        hidden_states: jax.Array,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
        indexer_k_pool=None,
        residual: jax.Array | None = None,
        prev_dsa_state: DsaTopKState | None = None,
        dispatch_info: ExpertLocationMetadata | None = None,
    ):
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states += residual
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)

        hidden_states, kv_fused, dsa_state, indexer_k_update = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
            token_to_kv_pool=token_to_kv_pool,
            indexer_k_pool=indexer_k_pool,
            prev_dsa_state=prev_dsa_state,
        )
        hidden_states += residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        if self.is_moe_layer:
            if self.shared_experts is not None:
                shared_output = self.shared_experts(hidden_states)
            else:
                shared_output = None
            router_logits = self.moe_gate(hidden_states)

            correction_bias = self.moe_gate.bias.value if self.moe_gate.bias is not None else None
            topk_weights, topk_ids = self.topk(
                router_logits,
                correction_bias,
                dispatch_info=dispatch_info,
            )

            hidden_states = self.mlp(hidden_states, topk_weights, topk_ids)

            if shared_output is not None:
                hidden_states = hidden_states + shared_output
        else:
            hidden_states = self.mlp(hidden_states)
            topk_ids = None

        return (
            hidden_states,
            residual,
            kv_fused,
            topk_ids,
            dsa_state,
            indexer_k_update,
        )


class Glm5Model(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = Embed(
            num_embeddings=config.vocab_size,
            features=config.hidden_size,
            dtype=dtype,
            param_dtype=dtype,
            kernel_axes=("tensor", None),
            mesh=mesh,
        )

        self.layers = nnx.data(
            [
                Glm5DecoderLayer(
                    config=config,
                    layer_id=i,
                    dtype=dtype,
                    mesh=mesh,
                )
                for i in range(config.num_hidden_layers)
            ]
        )

        self.norm = RMSNorm(
            config.hidden_size, epsilon=config.rms_norm_eps, param_dtype=dtype, scope_name="norm"
        )

    def __call__(
        self,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
        indexer_k_pool=None,
    ):
        hidden_states = self.embed_tokens(forward_batch.input_ids)
        residual = None
        layers_kv_fused = []
        layers_topk_ids = []
        layers_dsa_states = []
        layers_indexer_k_updates = []
        prev_dsa_state = None
        for layer in self.layers:
            (
                hidden_states,
                residual,
                kv_fused,
                topk_ids,
                dsa_state,
                indexer_k_update,
            ) = layer(
                forward_batch.positions,
                hidden_states,
                forward_batch,
                token_to_kv_pool,
                indexer_k_pool,
                residual,
                prev_dsa_state=prev_dsa_state,
                dispatch_info=forward_batch.expert_location_metadata,
            )
            layers_kv_fused.append(kv_fused)
            layers_topk_ids.append(topk_ids)
            layers_dsa_states.append(dsa_state)
            if indexer_k_update is not None:
                layers_indexer_k_updates.append(indexer_k_update)
            prev_dsa_state = dsa_state

        if residual is not None:
            hidden_states += residual

        hidden_states = self.norm(hidden_states)
        return (
            hidden_states,
            layers_kv_fused,
            layers_topk_ids,
            layers_dsa_states,
            layers_indexer_k_updates,
        )


class Glm5ForCausalLM(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.mesh = mesh
        self.config = config
        self.dtype = dtype
        self.model = Glm5Model(config, dtype=self.dtype, mesh=mesh)
        if not getattr(self.config, "tie_word_embeddings", False):
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                dtype=self.dtype,
                param_dtype=self.dtype,
                kernel_axes=("tensor", None),
            )
        self.logits_processor = LogitsProcessor(
            config.vocab_size,
            mesh=self.mesh,
            soft_cap=getattr(config, "final_logit_softcapping", None),
        )

    def __call__(
        self,
        forward_batch: ForwardBatch,
        memory_pools,
        logits_metadata: LogitsMetadata,
    ):
        kv_pool = memory_pools.token_to_kv_pool
        indexer_k_pool = getattr(memory_pools, "indexer_k_pool", None)
        (
            hidden_states,
            layers_kv_fused,
            layers_topk_ids,
            _,
            layers_indexer_k_updates,
        ) = self.model(
            forward_batch,
            kv_pool,
            indexer_k_pool,
        )

        if not getattr(self.config, "tie_word_embeddings", False):
            output = self.logits_processor(hidden_states, self.lm_head, logits_metadata)
        else:
            output = self.logits_processor(hidden_states, self.model.embed_tokens, logits_metadata)

        pool_updates = {"token_to_kv_pool": layers_kv_fused}
        if indexer_k_pool is not None:
            pool_updates["indexer_k_pool"] = layers_indexer_k_updates
        return output, pool_updates, True, layers_topk_ids

    def load_weights(self, model_config: ModelConfig):
        loader = WeightLoader(
            model=self,
            model_config=model_config,
            mesh=self.mesh,
            dtype=self.dtype,
        )
        weight_mappings = self._create_glm5_weight_mappings(model_config)
        loader.load_weights_from_safetensors(weight_mappings)

        for layer in self.model.layers:
            layer.self_attn.post_load_weights()
            if hasattr(layer, "mlp") and hasattr(layer.mlp, "post_load_weights"):
                layer.mlp.post_load_weights()
            if (
                hasattr(layer, "shared_experts")
                and layer.shared_experts is not None
                and hasattr(layer.shared_experts, "post_load_weights")
            ):
                layer.shared_experts.post_load_weights()
        logger.info("Absorbed MLA weights and Fused MLP weights processed successfully!")

        # Skipping scale inversion for BF16
        logger.info("Skipping scale inversion for BF16 model.")

    def _create_glm5_weight_mappings(self, model_config: ModelConfig) -> dict:
        mappings = {
            "model.embed_tokens.weight": WeightMapping(
                target_path="model.embed_tokens.embedding",
                sharding=("tensor", None),
                transpose=False,
            ),
            "model.norm.weight": WeightMapping(
                target_path="model.norm.scale", sharding=(None,), transpose=False
            ),
        }

        if not getattr(self.config, "tie_word_embeddings", False):
            mappings["lm_head.weight"] = WeightMapping(
                target_path="lm_head.embedding", sharding=("tensor", None), transpose=False
            )

        num_layers = self.config.num_hidden_layers
        first_k_dense_replace = getattr(self.config, "first_k_dense_replace", 0)
        indexer_types = getattr(self.config, "indexer_types", None)

        quant_config = getattr(model_config, "quantization_config", None)
        is_static_quant = quant_config is not None and quant_config.is_static_checkpoint

        hf_layer_indices = list(range(num_layers))
        for layer_idx in range(num_layers):
            target_idx = hf_layer_indices[layer_idx]
            layer_mappings = self._create_moe_layer_mappings(
                layer_idx,
                target_idx,
                target_idx < first_k_dense_replace,
                is_static_quant=is_static_quant,
                has_indexer=indexer_types is None or indexer_types[target_idx] == "full",
            )
            mappings.update(layer_mappings)

        return mappings

    def _create_moe_layer_mappings(
        self,
        layer_idx: int,
        target_idx: int,
        is_mlp_layer: bool,
        is_static_quant: bool = False,
        has_indexer: bool = True,
    ) -> dict:
        prefix = f"model.layers.{target_idx}"
        target_prefix = f"model.layers.{layer_idx}"

        mappings = {
            f"{prefix}.input_layernorm.weight": WeightMapping(
                target_path=f"{target_prefix}.input_layernorm.scale",
                sharding=(None,),
                transpose=False,
            ),
            f"{prefix}.post_attention_layernorm.weight": WeightMapping(
                target_path=f"{target_prefix}.post_attention_layernorm.scale",
                sharding=(None,),
                transpose=False,
            ),
        }

        def add_linear(hf: str, tgt: str, sharding_std: tuple, force_unquant: bool = False):
            """Mirror deepseek_v3._create_layer_mappings.add_linear.

            HF weight is [out, in]. Unquantized → LinearBase.weight [in, out]
            (transpose=True, sharding=kernel_axes). Static FP8 → QuantizedLinear
            .weight_q [out, in] (transpose=False, sharding swapped) plus the
            block-wise weight_scale_inv sidecar. force_unquant covers modules in
            the FP8 checkpoint's modules_to_not_convert (indexer.weights_proj).
            """
            if force_unquant or not is_static_quant:
                mappings[f"{hf}.weight"] = WeightMapping(
                    target_path=f"{tgt}.weight", sharding=sharding_std, transpose=True
                )
                return
            sharding_q = (sharding_std[1], sharding_std[0])
            mappings[f"{hf}.weight"] = WeightMapping(
                target_path=f"{tgt}.weight_q", sharding=sharding_q, transpose=False
            )
            # Load 2D block scale [out_blocks, in_blocks] replicated: GLM-5.1 head_dim
            # 448 → out_blocks not always tp-divisible (kv_b_proj: 224 % 64 ≠ 0).
            # _maybe_expand_linear_block_scale runs after _shard_weight and expands to
            # [in_blocks, 1, n_out]; assignment into model_param then reshards to the
            # QuantizedLinear placeholder's 3D sharding.
            mappings[f"{hf}.weight_scale_inv"] = WeightMapping(
                target_path=f"{tgt}.weight_scale", sharding=(None, None), transpose=False
            )

        ap = f"{prefix}.self_attn"
        tp = f"{target_prefix}.self_attn"
        add_linear(f"{ap}.q_a_proj", f"{tp}.q_a_proj", (None, None))
        mappings[f"{ap}.q_a_layernorm.weight"] = WeightMapping(
            target_path=f"{tp}.q_a_layernorm.scale", sharding=(None,)
        )
        add_linear(f"{ap}.q_b_proj", f"{tp}.q_b_proj", (None, "tensor"))
        add_linear(f"{ap}.kv_a_proj_with_mqa", f"{tp}.kv_a_proj_with_mqa", (None, None))
        mappings[f"{ap}.kv_a_layernorm.weight"] = WeightMapping(
            target_path=f"{tp}.kv_a_layernorm.scale", sharding=(None,)
        )
        add_linear(f"{ap}.kv_b_proj", f"{tp}.kv_b_proj", (None, "tensor"))
        add_linear(f"{ap}.o_proj", f"{tp}.o_proj", ("tensor", None))

        if has_indexer:
            add_linear(f"{ap}.indexer.wq_b", f"{tp}.indexer.wq_b", (None, None))
            add_linear(f"{ap}.indexer.wk", f"{tp}.indexer.wk", (None, None))
            # weights_proj is in modules_to_not_convert (HF: indexers_proj) → unquantized.
            add_linear(
                f"{ap}.indexer.weights_proj",
                f"{tp}.indexer.weights_proj",
                (None, None),
                force_unquant=True,
            )
            mappings[f"{ap}.indexer.k_norm.weight"] = WeightMapping(
                target_path=f"{tp}.indexer.k_norm.weight", sharding=(None,)
            )
            mappings[f"{ap}.indexer.k_norm.bias"] = WeightMapping(
                target_path=f"{tp}.indexer.k_norm.bias", sharding=(None,)
            )

        if is_mlp_layer:
            add_linear(
                f"{prefix}.mlp.gate_proj", f"{target_prefix}.mlp.gate_proj", (None, "tensor")
            )
            add_linear(f"{prefix}.mlp.up_proj", f"{target_prefix}.mlp.up_proj", (None, "tensor"))
            add_linear(
                f"{prefix}.mlp.down_proj", f"{target_prefix}.mlp.down_proj", ("tensor", None)
            )
        else:
            mappings[f"{prefix}.mlp.gate.weight"] = WeightMapping(
                target_path=f"{target_prefix}.moe_gate.kernel",
                sharding=(None, None),
                transpose=True,
            )
            # GLM-4 uses e_score_correction_bias
            mappings[f"{prefix}.mlp.gate.e_score_correction_bias"] = WeightMapping(
                target_path=f"{target_prefix}.moe_gate.bias", sharding=(None,)
            )

            num_logical_experts = self.config.n_routed_experts
            moe_backend = getattr(self.config, "moe_backend", "epmoe")

            moe_mappings = create_moe_weights_mapping(
                prefix=prefix,
                target_prefix=target_prefix,
                num_experts=num_logical_experts,
                expert_type_names=("gate_proj", "up_proj", "down_proj"),
                moe_backend=moe_backend,
                physical_to_logical_map=None,  # Handle physical mapping if needed later
            )

            if is_static_quant:
                new_moe_mappings = {}

                for key, mapping in moe_mappings.items():
                    target_param = mapping.target_path[0]
                    src_paths = mapping.target_path[1:]

                    new_moe_mappings[key] = WeightMapping(
                        target_path=[target_param] + src_paths,
                        sharding=mapping.sharding,
                        transpose=True,
                        concat_axis=mapping.concat_axis,
                        physical_to_logical_map=mapping.physical_to_logical_map,
                    )

                    scale_key = key + "_scale"
                    target_scale_param = target_param + "_scale"
                    scale_src_paths = [p.replace(".weight", ".weight_scale_inv") for p in src_paths]

                    # Stacked HF scale is [E, out_blocks, in_blocks]. Load EP-sharded
                    # and replicated on the block dims (matches deepseek_v3); the
                    # loader's _maybe_convert_epmoe_scale_for_kernel handles the
                    # [E, out_blocks, k_blocks] → [E, k_blocks, 1, n_out] expand.
                    new_moe_mappings[scale_key] = WeightMapping(
                        target_path=[target_scale_param] + scale_src_paths,
                        sharding=("expert", None, None),
                        transpose=False,
                        concat_axis=mapping.concat_axis,
                        physical_to_logical_map=mapping.physical_to_logical_map,
                    )
                moe_mappings = new_moe_mappings

            mappings.update(moe_mappings)

            num_shared = getattr(self.config, "n_shared_experts", 0)
            if num_shared > 0:
                sp = f"{prefix}.mlp.shared_experts"
                st = f"{target_prefix}.shared_experts"
                add_linear(f"{sp}.gate_proj", f"{st}.gate_proj", (None, "tensor"))
                add_linear(f"{sp}.up_proj", f"{st}.up_proj", (None, "tensor"))
                add_linear(f"{sp}.down_proj", f"{st}.down_proj", ("tensor", None))

        return mappings


class GlmMoeDsaForCausalLM(Glm5ForCausalLM):
    @classmethod
    def patch_model_config(cls, mc: ModelConfig) -> None:
        from sgl_jax.srt.configs.model_config import AttentionArch

        # GLM-5 uses 256 for attention head dim (192 nope + 64 pe)
        mc.head_dim = 256
        mc.hf_config.head_dim = 256
        mc.v_head_dim = getattr(mc.hf_text_config, "v_head_dim", 256)
        # GLM-5 uses MLA architecture
        mc.attention_arch = AttentionArch.MLA
        # GLM-5.1-FP8 ships modules_to_not_convert with HF naming (e.g.
        # `self_attn.indexers_proj`); translate to sglang-jax module paths so
        # quantize_model leaves the unquantized indexer head-gate as LinearBase.
        if mc.quantization_config is not None and mc.quantization_config.is_static_checkpoint:
            mc.quantization_config.ignored_layers = list(
                mc.quantization_config.ignored_layers or []
            ) + ["indexer.weights_proj"]
            # indexer.wk has out_dim=128 == block_size_out (single N-block); the
            # narrow-N guard would reject it but the indexer output is currently
            # discarded so accuracy is unaffected. Match deepseek_v3 config.
            mc.quantization_config.allow_narrow_n_blockwise = True
        # Under dynamic (in-framework) quant, Glm5MLP.post_load_weights merges
        # BF16 w_gu/w_d and nulls gate/up/down_proj *before* quantize_model
        # runs, so the fused weights bypass quantization and regress decode
        # TPOT on HBM-bound hardware (#1378). Keep the unfused path there so
        # the LinearBase modules get quantized as before.
        # Static fp8 checkpoint also breaks fused: gate_proj/up_proj/down_proj
        # become QuantizedLinear (no .weight), so post_load_weights cannot
        # populate w_gu/w_d and the abstract ShapeDtypeStruct placeholders
        # leak into jit inputs. Keep fused for bf16-only.
        mc.hf_config._sgl_use_fused_mlp = mc.quantization_config is None


EntryClass = [Glm5ForCausalLM, GlmMoeDsaForCausalLM]
