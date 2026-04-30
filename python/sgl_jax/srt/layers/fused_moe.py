"""Fused Expert-Parallel MoE layer using Pallas kernel."""

import logging

import jax
from flax import nnx
from jax import numpy as jnp
from jax import shard_map
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.eplb.expert_location import get_global_expert_location_metadata
from sgl_jax.srt.kernels.fused_moe.v1.kernel import FusedMoEBlockConfig, fused_ep_moe
from sgl_jax.srt.kernels.gmm.megablox_gmm_backend import gmm
from sgl_jax.srt.utils.quantization.quantization_utils import quantize_tensor

logger = logging.getLogger(__name__)


def _expand_moe_block_scale(scale_3d: jax.Array, n_out: int, block_n: int) -> jax.Array:
    """Expand compact 2D MoE block scales to the kernel's fast 1D-ready layout."""
    scale_per_channel = jnp.repeat(scale_3d, block_n, axis=2)[..., :n_out]
    return scale_per_channel[:, :, None, :]


class FusedEPMoE(nnx.Module):
    """
    Expert Parallel MoE layer using fused TPU kernel.

    This layer wraps the optimized fused_ep_moe kernel which combines Top-K selection,
    expert computation, and aggregation into a single efficient operation.

    Key differences from EPMoE:
    - Weight format: w1/w3 are (num_experts, hidden_size, intermediate_size) for gate/up proj
      and w2 is (num_experts, intermediate_size, hidden_size) for down proj
    - Input: Takes router_logits directly instead of pre-computed topk_weights/topk_ids
    - Implementation: Uses Pallas kernel with manual memory management for TPU optimization

    Args:
        hidden_size: Hidden size of the model
        num_experts: Total number of experts
        num_experts_per_tok: Number of experts to select per token (top_k)
        ep_size: Expert parallel size (number of devices to shard experts across)
        mesh: JAX mesh for distributed execution
        intermediate_dim: Intermediate dimension for expert FFN
        weight_dtype: Data type for weights
        dtype: Data type for computation
        activation: Activation function ("silu", "gelu", "swigluoai")
        layer_id: Layer index (for debugging)
        renormalize_topk_logits: Whether to renormalize top-k weights
        bt, bf, bd1, bd2, btc, bfc, bd1c, bd2c: Tile size parameters (auto-selected if None)
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        num_experts_per_tok: int,
        ep_size: int,
        mesh: Mesh,
        intermediate_dim: int = 2048,
        weight_dtype: jnp.dtype = jnp.bfloat16,
        dtype: jnp.dtype = jnp.bfloat16,
        activation: str = "silu",
        layer_id: int = 0,
        use_grouped_topk: bool = False,
        num_groups: int = 1,
        top_k_groups: int = 1,
        renormalize_topk_logits: bool = False,
        routed_scaling_factor: float | None = None,
        num_shared_experts: int = 0,
        moe_shared_expert_intermediate_size: int | None = None,
        quantization_config=None,
        # Profiling / ablation flags (primarily for microbenching).
        disable_a2a: bool = False,
        disable_dynamic_ffn1: bool = False,
        disable_dynamic_ffn2: bool = False,
        disable_weight_load: bool = False,
        disable_a2a_s_tile_read: bool = False,
        disable_a2a_s_acc_tile_write: bool = False,
        disable_shared_expert: bool = False,
        disable_all_reduce_metadata: bool = False,
        disable_sync_barrier: bool = False,
    ):
        self.hidden_size = hidden_size
        self.num_experts_per_tok = num_experts_per_tok
        self.intermediate_dim = intermediate_dim
        self.weight_dtype = weight_dtype
        self.dtype = dtype
        self.layer_id = layer_id
        self.ep_size = ep_size
        self.activation = activation
        self.use_grouped_topk = use_grouped_topk
        self.num_groups = num_groups
        self.top_k_groups = top_k_groups
        self.renormalize_topk_logits = renormalize_topk_logits
        self.routed_scaling_factor = routed_scaling_factor
        self.num_shared_experts = num_shared_experts
        self.moe_shared_expert_intermediate_size = (
            moe_shared_expert_intermediate_size or intermediate_dim
        )
        self.mesh = mesh
        self.disable_a2a = disable_a2a
        self.disable_dynamic_ffn1 = disable_dynamic_ffn1
        self.disable_dynamic_ffn2 = disable_dynamic_ffn2
        self.disable_weight_load = disable_weight_load
        self.disable_a2a_s_tile_read = disable_a2a_s_tile_read
        self.disable_a2a_s_acc_tile_write = disable_a2a_s_acc_tile_write
        self.disable_shared_expert = disable_shared_expert
        self.disable_all_reduce_metadata = disable_all_reduce_metadata
        self.disable_sync_barrier = disable_sync_barrier
        self.gmm_prefill_threshold = 256

        metadata = get_global_expert_location_metadata()
        if metadata is not None and layer_id is not None:
            self.num_experts = metadata.num_physical_experts
        else:
            self.num_experts = num_experts

        if self.num_experts % self.ep_size != 0:
            raise ValueError(
                f"num_experts({self.num_experts}) must be divisible by ep_size ({self.ep_size})"
            )

        self.quantized_dtype = (
            quantization_config.get_moe_weight_dtype() if quantization_config else None
        )
        self.activation_quantized_dtype = (
            quantization_config.get_moe_activation_dtype() if quantization_config else None
        )

        # Initialize weights.
        self.w1 = nnx.Param(
            jax.random.normal(
                jax.random.key(0),
                (self.num_experts, hidden_size, intermediate_dim),
                dtype=weight_dtype,
                out_sharding=P(("data", "tensor"), None, None),
            )
        )
        self.w3 = nnx.Param(
            jax.random.normal(
                jax.random.key(1),
                (self.num_experts, hidden_size, intermediate_dim),
                dtype=weight_dtype,
                out_sharding=P(("data", "tensor"), None, None),
            )
        )

        self.w2 = nnx.Param(
            jax.random.normal(
                jax.random.key(0),
                (self.num_experts, intermediate_dim, hidden_size),
                dtype=weight_dtype,
                out_sharding=P(("data", "tensor"), None, None),
            )
        )

        self.w1_scale = None
        self.w3_scale = None
        self.w2_scale = None

        if self.num_shared_experts > 0:
            se_inter_dim = self.moe_shared_expert_intermediate_size * self.num_shared_experts

            self.w1_shared = nnx.Param(
                jax.random.normal(
                    jax.random.key(0),
                    (hidden_size, se_inter_dim),
                    dtype=weight_dtype,
                    out_sharding=P(None, None),
                )
            )

            self.w2_shared = nnx.Param(
                jax.random.normal(
                    jax.random.key(0),
                    (se_inter_dim, hidden_size),
                    dtype=weight_dtype,
                    out_sharding=P(None, None),
                )
            )

            self.w3_shared = nnx.Param(
                jax.random.normal(
                    jax.random.key(0),
                    (hidden_size, se_inter_dim),
                    dtype=weight_dtype,
                    out_sharding=P(None, None),
                )
            )
        else:
            self.w1_shared = None
            self.w3_shared = None
            self.w2_shared = None

        self.w1_shared_scale = None
        self.w3_shared_scale = None
        self.w2_shared_scale = None

        # Read block-wise quantization settings from config.
        weight_block_size = (
            getattr(quantization_config, "weight_block_size", None) if quantization_config else None
        )
        if weight_block_size is not None and len(weight_block_size) == 2:
            self.quant_block_k = int(weight_block_size[1])  # block_k
            self.quant_block_n = int(weight_block_size[0])  # block_n
        else:
            self.quant_block_k = None
            self.quant_block_n = None

        # GMM prefill path: create moe_mesh with ("expert", "tensor") axes.
        world_size = mesh.shape.get("data", 1) * mesh.shape.get("tensor", 1)
        self.tp_size = world_size // self.ep_size
        self.experts_per_device = self.num_experts // self.ep_size

        devices = self.mesh.devices.flatten()
        self._gmm_moe_mesh = jax.sharding.Mesh(
            devices.reshape(self.ep_size, self.tp_size),
            axis_names=("expert", "tensor"),
            axis_types=(jax.sharding.AxisType.Explicit, jax.sharding.AxisType.Explicit),
        )
        self._gmm_abstract_mesh = self.mesh.abstract_mesh.update(
            axis_sizes=(self.ep_size, self.tp_size), axis_names=("expert", "tensor")
        )

    def quantize_weights(self, is_static: bool = False):
        """Quantize MoE weights in-place. Call once after model loading."""
        if self.quantized_dtype is None:
            return

        # Default quant_block_k to 256 if not explicitly set.
        wsz = self.quant_block_k if self.quant_block_k is not None else 256
        if hasattr(self, "quant_block_k"):
            del self.quant_block_k
        self.quant_block_k = wsz

        with jax.set_mesh(self.mesh):
            if is_static:
                ep_scale_sharding = P(("data", "tensor"), None, None, None)

                # Scale placeholder shapes are (E, K//block_k, 1, N) for both
                # 1D sub-channel and 2D block-wise quantization.  In the 2D case,
                # _expand_moe_block_scale() expands the compact (E, K//bk, N//bn)
                # scales to the same (E, K//bk, 1, N) layout at weight-loading
                # time, so the kernel always sees the unified 1D shape.
                w1_scale_shape = (
                    self.num_experts,
                    self.hidden_size // wsz,
                    1,
                    self.intermediate_dim,
                )
                w3_scale_shape = w1_scale_shape
                w2_scale_shape = (
                    self.num_experts,
                    self.intermediate_dim // wsz,
                    1,
                    self.hidden_size,
                )

                if hasattr(self, "w1_scale"):
                    del self.w1_scale
                self.w1_scale = nnx.Param(
                    jnp.zeros(w1_scale_shape, dtype=jnp.float32),
                    out_sharding=ep_scale_sharding,
                )

                if hasattr(self, "w3_scale"):
                    del self.w3_scale
                self.w3_scale = nnx.Param(
                    jnp.zeros(w3_scale_shape, dtype=jnp.float32),
                    out_sharding=ep_scale_sharding,
                )

                if hasattr(self, "w2_scale"):
                    del self.w2_scale
                self.w2_scale = nnx.Param(
                    jnp.zeros(w2_scale_shape, dtype=jnp.float32),
                    out_sharding=ep_scale_sharding,
                )

                if self.num_shared_experts > 0:
                    shared_scale_sharding = P(None, None, None)

                    if hasattr(self, "w1_shared_scale"):
                        del self.w1_shared_scale
                    self.w1_shared_scale = nnx.Param(
                        jnp.zeros((1,), dtype=jnp.float32), out_sharding=shared_scale_sharding
                    )

                    if hasattr(self, "w3_shared_scale"):
                        del self.w3_shared_scale
                    self.w3_shared_scale = nnx.Param(
                        jnp.zeros((1,), dtype=jnp.float32), out_sharding=shared_scale_sharding
                    )

                    if hasattr(self, "w2_shared_scale"):
                        del self.w2_shared_scale
                    self.w2_shared_scale = nnx.Param(
                        jnp.zeros((1,), dtype=jnp.float32), out_sharding=shared_scale_sharding
                    )

                return

            # Replace original weights with quantized versions
            if self.quant_block_n is not None:
                # 2D block-wise quantization: scale shape (E, K//block_k, N//block_n)
                w1_value, w1_scale = quantize_tensor(
                    self.quantized_dtype,
                    self.w1.value,
                    axis=(1, 2),
                    block_size=[self.quant_block_k, self.quant_block_n],
                )
                w3_value, w3_scale = quantize_tensor(
                    self.quantized_dtype,
                    self.w3.value,
                    axis=(1, 2),
                    block_size=[self.quant_block_k, self.quant_block_n],
                )
                w2_value, w2_scale = quantize_tensor(
                    self.quantized_dtype,
                    self.w2.value,
                    axis=(1, 2),
                    block_size=[self.quant_block_k, self.quant_block_n],
                )
            else:
                # 1D sub-channel quantization: scale shape (E, K//wsz, N)
                w1_value, w1_scale = quantize_tensor(
                    self.quantized_dtype,
                    self.w1.value,
                    axis=1,
                    block_size=self.quant_block_k,
                )
                w3_value, w3_scale = quantize_tensor(
                    self.quantized_dtype,
                    self.w3.value,
                    axis=1,
                    block_size=self.quant_block_k,
                )
                w2_value, w2_scale = quantize_tensor(
                    self.quantized_dtype,
                    self.w2.value,
                    axis=1,
                    block_size=self.quant_block_k,
                )

            # NOTE: Fused MoE shards the expert dimension across EP=(data*tensor).
            ep_sharding = P(("data", "tensor"), None, None)
            ep_scale_sharding = P(("data", "tensor"), None, None, None)

            self.w1 = nnx.Param(w1_value, out_sharding=ep_sharding)
            self.w3 = nnx.Param(w3_value, out_sharding=ep_sharding)
            self.w2 = nnx.Param(w2_value, out_sharding=ep_sharding)

            # Update scales (reshape to 4D for GMM kernel)
            if self.quant_block_n is not None:
                # 2D block-wise: expand block scales once so forward can run
                # through the fast 1D kernel path without changing semantics.
                w1_scale_4d = _expand_moe_block_scale(
                    w1_scale, self.intermediate_dim, self.quant_block_n
                )
                w3_scale_4d = _expand_moe_block_scale(
                    w3_scale, self.intermediate_dim, self.quant_block_n
                )
                w2_scale_4d = _expand_moe_block_scale(
                    w2_scale, self.hidden_size, self.quant_block_n
                )
            else:
                # (E, K//wsz, N) → (E, K//wsz, 1, N)
                w1_scale_4d = w1_scale.reshape(
                    w1_scale.shape[0], w1_scale.shape[1], 1, w1_scale.shape[2]
                )
                w3_scale_4d = w3_scale.reshape(
                    w3_scale.shape[0], w3_scale.shape[1], 1, w3_scale.shape[2]
                )
                w2_scale_4d = w2_scale.reshape(
                    w2_scale.shape[0], w2_scale.shape[1], 1, w2_scale.shape[2]
                )

            if hasattr(self, "w1_scale"):
                del self.w1_scale
            self.w1_scale = nnx.Param(
                w1_scale_4d,
                out_sharding=ep_scale_sharding,
            )
            if hasattr(self, "w3_scale"):
                del self.w3_scale
            self.w3_scale = nnx.Param(
                w3_scale_4d,
                out_sharding=ep_scale_sharding,
            )
            if hasattr(self, "w2_scale"):
                del self.w2_scale
            self.w2_scale = nnx.Param(
                w2_scale_4d,
                out_sharding=ep_scale_sharding,
            )

            if self.w1_shared is not None:
                w1_shared_value, w1_shared_scale = quantize_tensor(
                    self.quantized_dtype,
                    self.w1_shared.value,
                    axis=0,
                )
                w3_shared_value, w3_shared_scale = quantize_tensor(
                    self.quantized_dtype,
                    self.w3_shared.value,
                    axis=0,
                )
                w2_shared_value, w2_shared_scale = quantize_tensor(
                    self.quantized_dtype,
                    self.w2_shared.value,
                    axis=0,
                )

                self.w1_shared = nnx.Param(w1_shared_value, out_sharding=P(None, None))
                self.w3_shared = nnx.Param(w3_shared_value, out_sharding=P(None, None))
                self.w2_shared = nnx.Param(w2_shared_value, out_sharding=P(None, None))

                if hasattr(self, "w1_shared_scale"):
                    del self.w1_shared_scale
                self.w1_shared_scale = nnx.Param(
                    w1_shared_scale.reshape(
                        1,
                        1,
                        w1_shared_scale.shape[0],
                    ),
                    out_sharding=P(None, None, None),
                )

                if hasattr(self, "w3_shared_scale"):
                    del self.w3_shared_scale
                self.w3_shared_scale = nnx.Param(
                    w3_shared_scale.reshape(
                        1,
                        1,
                        w3_shared_scale.shape[0],
                    ),
                    out_sharding=P(None, None, None),
                )

                if hasattr(self, "w2_shared_scale"):
                    del self.w2_shared_scale
                self.w2_shared_scale = nnx.Param(
                    w2_shared_scale.reshape(
                        1,
                        1,
                        w2_shared_scale.shape[0],
                    ),
                    out_sharding=P(None, None, None),
                )

    def __call__(
        self,
        hidden_states: jax.Array,
        topk_weights: jax.Array,
        topk_ids: jax.Array,
        *,
        block_config: FusedMoEBlockConfig | None = None,
    ) -> jax.Array:
        assert hidden_states.ndim == 2
        num_tokens = hidden_states.shape[0]

        use_gmm = num_tokens > self.gmm_prefill_threshold and self.num_shared_experts == 0
        if use_gmm:
            return self._gmm_prefill_forward(hidden_states, topk_weights, topk_ids)
        return self._fused_forward(hidden_states, topk_weights, topk_ids, block_config=block_config)

    def _fused_forward(
        self,
        hidden_states: jax.Array,
        topk_weights: jax.Array,
        topk_ids: jax.Array,
        *,
        block_config: FusedMoEBlockConfig | None = None,
    ) -> jax.Array:
        """Original fused Pallas kernel path — best for decode (small token count)."""

        w1_shared_val = self.w1_shared.value if self.w1_shared is not None else None
        w3_shared_val = self.w3_shared.value if self.w3_shared is not None else None
        w2_shared_val = self.w2_shared.value if self.w2_shared is not None else None
        w1_scale = self.w1_scale.value if self.w1_scale is not None else None
        w3_scale = self.w3_scale.value if self.w3_scale is not None else None
        w2_scale = self.w2_scale.value if self.w2_scale is not None else None
        w1_shared_scale = self.w1_shared_scale.value if self.w1_shared_scale is not None else None
        w3_shared_scale = self.w3_shared_scale.value if self.w3_shared_scale is not None else None
        w2_shared_scale = self.w2_shared_scale.value if self.w2_shared_scale is not None else None

        quant_block_k = self.quant_block_k if self.quant_block_k is not None else None

        output = fused_ep_moe(
            mesh=self.mesh,
            tokens=hidden_states,
            w1=self.w1.value,
            w2=self.w2.value,
            w3=self.w3.value,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            top_k=self.num_experts_per_tok,
            use_grouped_topk=self.use_grouped_topk,
            num_groups=self.num_groups,
            top_k_groups=self.top_k_groups,
            renormalize_topk_logits=self.renormalize_topk_logits,
            routed_scaling_factor=self.routed_scaling_factor,
            act_fn=self.activation,
            block_config=block_config,
            disable_a2a=self.disable_a2a,
            disable_dynamic_ffn1=self.disable_dynamic_ffn1,
            disable_dynamic_ffn2=self.disable_dynamic_ffn2,
            disable_weight_load=self.disable_weight_load,
            disable_a2a_s_tile_read=self.disable_a2a_s_tile_read,
            disable_a2a_s_acc_tile_write=self.disable_a2a_s_acc_tile_write,
            disable_shared_expert=self.disable_shared_expert,
            disable_all_reduce_metadata=self.disable_all_reduce_metadata,
            disable_sync_barrier=self.disable_sync_barrier,
            # Optional parameters (not used in basic case)
            quant_block_k=quant_block_k,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            w3_scale=w3_scale,
            w1_shared=w1_shared_val,
            w2_shared=w2_shared_val,
            w3_shared=w3_shared_val,
            w1_shared_scale=w1_shared_scale,
            w2_shared_scale=w2_shared_scale,
            w3_shared_scale=w3_shared_scale,
            b1=None,
            b2=None,
            b3=None,
            dp_axis_name="data",
            tp_axis_name="tensor",
        )

        output = jax.sharding.reshard(output, NamedSharding(self.mesh, P("data", None)))
        return output

    def _gmm_prefill_forward(
        self,
        hidden_states: jax.Array,
        topk_weights: jax.Array,
        topk_ids: jax.Array,
    ) -> jax.Array:
        """GMM-based prefill path — better MXU utilization via MegaCore + batched experts."""
        with jax.sharding.use_abstract_mesh(self._gmm_abstract_mesh):
            hidden_r = jax.sharding.reshard(hidden_states, P(None))
            topk_weights_r = jax.sharding.reshard(topk_weights, P(None))
            topk_ids_r = jax.sharding.reshard(topk_ids, P(None))

            # Reshard weights: fused EP sharding -> GMM expert-parallel sharding.
            # For TP=1 this is a zero-copy metadata change.
            w1_r = jax.sharding.reshard(self.w1.value, P("expert", None, "tensor"))
            w3_r = jax.sharding.reshard(self.w3.value, P("expert", None, "tensor"))
            w2_r = jax.sharding.reshard(self.w2.value, P("expert", "tensor", None))

            has_scales = self.w1_scale is not None
            if has_scales:
                w1_scale_r = jax.sharding.reshard(
                    self.w1_scale.value, P("expert", None, None, "tensor")
                )
                w3_scale_r = jax.sharding.reshard(
                    self.w3_scale.value, P("expert", None, None, "tensor")
                )
                w2_scale_r = jax.sharding.reshard(
                    self.w2_scale.value, P("expert", None, None, None)
                )
                in_specs = (
                    P(None),
                    P(None),
                    P(None),
                    P("expert", None, "tensor"),
                    P("expert", None, "tensor"),
                    P("expert", "tensor", None),
                    P("expert", None, None, "tensor"),
                    P("expert", None, None, "tensor"),
                    P("expert", None, None, None),
                )
                args = (
                    hidden_r,
                    topk_weights_r,
                    topk_ids_r,
                    w1_r,
                    w3_r,
                    w2_r,
                    w1_scale_r,
                    w3_scale_r,
                    w2_scale_r,
                )
            else:
                in_specs = (
                    P(None),
                    P(None),
                    P(None),
                    P("expert", None, "tensor"),
                    P("expert", None, "tensor"),
                    P("expert", "tensor", None),
                )
                args = (
                    hidden_r,
                    topk_weights_r,
                    topk_ids_r,
                    w1_r,
                    w3_r,
                    w2_r,
                )

            fn = self._gmm_forward_body if has_scales else self._gmm_forward_body_no_scale
            result = shard_map(
                fn,
                mesh=self._gmm_moe_mesh,
                in_specs=in_specs,
                out_specs=P(None),
                check_vma=False,
            )(*args)

        return jax.sharding.reshard(result, NamedSharding(self.mesh, P("data", None)))

    def _gmm_forward_body(
        self,
        hidden_states,
        topk_weights,
        topk_ids,
        w1,
        w3,
        w2,
        w1_scale,
        w3_scale,
        w2_scale,
    ):
        """GMM compute inside shard_map: permute -> 3x GMM -> unpermute -> psum."""
        return self._gmm_forward_impl(
            hidden_states,
            topk_weights,
            topk_ids,
            w1,
            w3,
            w2,
            w1_scale,
            w3_scale,
            w2_scale,
        )

    def _gmm_forward_body_no_scale(
        self,
        hidden_states,
        topk_weights,
        topk_ids,
        w1,
        w3,
        w2,
    ):
        """GMM compute without quantization scales."""
        return self._gmm_forward_impl(
            hidden_states,
            topk_weights,
            topk_ids,
            w1,
            w3,
            w2,
            None,
            None,
            None,
        )

    def _gmm_forward_impl(
        self,
        hidden_states,
        topk_weights,
        topk_ids,
        w1,
        w3,
        w2,
        w1_scale,
        w3_scale,
        w2_scale,
    ):
        """GMM compute inside shard_map: permute -> 3x GMM -> unpermute -> psum."""
        expert_shard_id = jax.lax.axis_index("expert")
        num_experts = self.num_experts
        num_experts_per_tok = self.num_experts_per_tok
        experts_per_device = self.experts_per_device

        # --- Permute ---
        flatten_experts = jnp.ravel(topk_ids)
        sorted_experts = jnp.argsort(flatten_experts, stable=True)
        token_indices = sorted_experts // num_experts_per_tok
        group_sizes = jnp.bincount(flatten_experts, length=num_experts).astype(jnp.int32)

        group_offset = jnp.array(expert_shard_id * experts_per_device, dtype=jnp.int32)

        # --- Indexed gather ---
        x = hidden_states[token_indices].astype(self.dtype)

        # Pad for sublane alignment (required by GMM kernel).
        from jax.experimental.pallas import tpu as pltpu

        sublane_align = pltpu.get_tpu_info().get_sublane_tiling(x.dtype)
        pad_size = (-x.shape[0]) % sublane_align
        if pad_size > 0:
            x = jnp.pad(x, ((0, pad_size), (0, 0)))
            group_sizes = group_sizes.at[-1].add(pad_size)

        act_q_dtype = self.activation_quantized_dtype
        gmm_kwargs = dict(
            group_sizes=group_sizes,
            preferred_element_type=self.dtype,
            group_offset=group_offset,
            maybe_quantize_lhs=act_q_dtype is not None,
            acc_dtype=jnp.float32,
        )

        # --- 3x GMM: gate, up, down ---
        layer_gate = gmm(
            lhs=x,
            rhs=w1,
            rhs_scale=w1_scale,
            rhs_bias=None,
            zero_initialize=False,
            activation_quantized_dtype=act_q_dtype,
            **gmm_kwargs,
        )
        layer_up = gmm(
            lhs=x,
            rhs=w3,
            rhs_scale=w3_scale,
            rhs_bias=None,
            zero_initialize=False,
            activation_quantized_dtype=act_q_dtype,
            **gmm_kwargs,
        )

        if self.activation == "silu":
            act = jax.nn.silu(layer_gate)
        elif self.activation == "gelu":
            act = jax.nn.gelu(layer_gate)
        else:
            raise ValueError(f"Unsupported activation: {self.activation}")
        intermediate = jnp.multiply(act, layer_up)

        output = gmm(
            lhs=intermediate,
            rhs=w2,
            rhs_scale=w2_scale,
            rhs_bias=None,
            zero_initialize=True,
            activation_quantized_dtype=act_q_dtype,
            **gmm_kwargs,
        )

        # --- Unpermute ---
        expected_tokens = sorted_experts.shape[0]
        actual_tokens = output.shape[0]
        if actual_tokens != expected_tokens:
            if actual_tokens > expected_tokens:
                output = output[:expected_tokens]
            else:
                pad = jnp.zeros(
                    (expected_tokens - actual_tokens, output.shape[1]),
                    dtype=output.dtype,
                )
                output = jnp.concatenate([output, pad], axis=0)

        argsort_inv = jnp.argsort(sorted_experts, stable=True)
        unsorted = jnp.take(output, argsort_inv, axis=0)

        total_tokens = topk_weights.shape[0]
        reshaped_w = jnp.reshape(topk_weights, (total_tokens, num_experts_per_tok))
        reshaped_o = jnp.reshape(unsorted, (total_tokens, num_experts_per_tok, -1))

        combined = jnp.einsum(
            "BKE,BK->BE",
            reshaped_o.astype(jnp.float32),
            reshaped_w.astype(jnp.float32),
        ).astype(self.dtype)

        # --- All-reduce ---
        if self.tp_size > 1:
            combined = jax.lax.psum(combined, "tensor")
        if self.ep_size > 1:
            combined = jax.lax.psum(combined, "expert")

        return combined
