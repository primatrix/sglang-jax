"""Fused MoE layer using optimized TPU kernel."""

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.kernels.fused_moe.v1.kernel import fused_ep_moe


def _get_default_tile_sizes(hidden_size: int, intermediate_size: int) -> dict[str, int]:
    """
    Select appropriate tile sizes based on model dimensions.

    These values are derived from benchmarking in the test suite and optimized
    for TPU performance with different model sizes.

    Args:
        hidden_size: Model hidden dimension
        intermediate_size: MoE intermediate (FFN) dimension

    Returns:
        Dictionary containing tile size parameters for the fused kernel
    """
    if hidden_size >= 4096:
        # Large models (e.g., Qwen 2.5B)
        return {
            "bt": 64,
            "bf": 512,
            "bd1": 2048,
            "bd2": 2048,
            "btc": 64,
            "bfc": 512,
            "bd1c": 2048,
            "bd2c": 2048,
        }
    elif hidden_size >= 2048:
        # Medium models (e.g., Qwen 30B A3B)
        return {
            "bt": 16,
            "bf": 512,
            "bd1": 512,
            "bd2": 512,
            "btc": 16,
            "bfc": 512,
            "bd1c": 256,
            "bd2c": 256,
        }
    else:
        # Small models
        return {
            "bt": 32,
            "bf": 512,
            "bd1": 512,
            "bd2": 512,
            "btc": 32,
            "bfc": 256,
            "bd1c": 256,
            "bd2c": 256,
        }


class FusedEPMoE(nnx.Module):
    """
    Expert Parallel MoE layer using fused TPU kernel.

    This layer wraps the optimized fused_ep_moe kernel which combines Top-K selection,
    expert computation, and aggregation into a single efficient operation.

    Key differences from EPMoE:
    - Weight format: Unfused w1(gate), w3(up), w2(down)
    - Input: Takes router_logits directly instead of pre-computed topk_weights/topk_ids
    - Implementation: Uses Pallas kernel with manual memory management for TPU optimization

    Args:
        config: Model configuration
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
        config,
        num_experts: int,
        num_experts_per_tok: int,
        ep_size: int,
        mesh: Mesh,
        intermediate_dim: int = 2048,
        weight_dtype: jnp.dtype = jnp.bfloat16,
        dtype: jnp.dtype = jnp.bfloat16,
        activation: str = "silu",
        layer_id: int = 0,
        renormalize_topk_logits: bool = False,
        routed_scaling_factor: float | None = None,
        use_grouped_topk: bool = False,
        num_groups: int = 1,
        top_k_groups: int = 1,
        num_shared_experts: int = 0,
        moe_shared_expert_intermediate_size: int | None = None,
        # Tile size parameters - auto-selected if None
        bt: int | None = None,
        bf: int | None = None,
        bd1: int | None = None,
        bd2: int | None = None,
        btc: int | None = None,
        bfc: int | None = None,
        bd1c: int | None = None,
        bd2c: int | None = None,
    ):
        self.config = config
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.intermediate_dim = intermediate_dim
        self.weight_dtype = weight_dtype
        self.dtype = dtype
        self.layer_id = layer_id
        self.ep_size = ep_size
        self.activation = activation
        self.renormalize_topk_logits = renormalize_topk_logits
        self.routed_scaling_factor = routed_scaling_factor
        self.mesh = mesh
        self.use_grouped_topk = use_grouped_topk
        self.num_groups = num_groups
        self.top_k_groups = top_k_groups
        self.num_shared_experts = num_shared_experts
        self.moe_shared_expert_intermediate_size = (
            moe_shared_expert_intermediate_size or intermediate_dim
        )

        if num_experts % self.ep_size != 0:
            raise ValueError(
                f"num_experts({num_experts}) must be divisible by ep_size ({self.ep_size})"
            )

        # Auto-select tile sizes if not provided
        if any(param is None for param in [bt, bf, bd1, bd2, btc, bfc, bd1c, bd2c]):
            default_sizes = _get_default_tile_sizes(config.hidden_size, intermediate_dim)
            bt = bt or default_sizes["bt"]
            bf = bf or default_sizes["bf"]
            bd1 = bd1 or default_sizes["bd1"]
            bd2 = bd2 or default_sizes["bd2"]
            btc = btc or default_sizes["btc"]
            bfc = bfc or default_sizes["bfc"]
            bd1c = bd1c or default_sizes["bd1c"]
            bd2c = bd2c or default_sizes["bd2c"]

        self.bt = bt
        self.bf = bf
        self.bd1 = bd1
        self.bd2 = bd2
        self.btc = btc
        self.bfc = bfc
        self.bd1c = bd1c
        self.bd2c = bd2c

        self.w1 = nnx.Param(
            jax.random.normal(
                jax.random.key(0),
                (num_experts, config.hidden_size, intermediate_dim),
                dtype=weight_dtype,
                out_sharding=P("tensor", None, None),
            )
        )

        self.w3 = nnx.Param(
            jax.random.normal(
                jax.random.key(0),
                (num_experts, config.hidden_size, intermediate_dim),
                dtype=weight_dtype,
                out_sharding=P("tensor", None, None),
            )
        )

        self.w2 = nnx.Param(
            jax.random.normal(
                jax.random.key(0),
                (num_experts, intermediate_dim, config.hidden_size),
                dtype=weight_dtype,
                out_sharding=P("tensor", None, None),
            )
        )

        if self.num_shared_experts > 0:
            se_inter_dim = self.moe_shared_expert_intermediate_size * self.num_shared_experts

            self.w1_shared = nnx.Param(
                jax.random.normal(
                    jax.random.key(0),
                    (config.hidden_size, se_inter_dim),
                    dtype=weight_dtype,
                    out_sharding=P(None, None),
                )
            )

            self.w3_shared = nnx.Param(
                jax.random.normal(
                    jax.random.key(0),
                    (config.hidden_size, se_inter_dim),
                    dtype=weight_dtype,
                    out_sharding=P(None, None),
                )
            )

            self.w2_shared = nnx.Param(
                jax.random.normal(
                    jax.random.key(0),
                    (se_inter_dim, config.hidden_size),
                    dtype=weight_dtype,
                    out_sharding=P(None, None),
                )
            )
        else:
            self.w1_shared = None
            self.w3_shared = None
            self.w2_shared = None

    def __call__(
        self,
        hidden_states: jax.Array,
        router_logits: jax.Array,
        router_bias: jax.Array | None = None,
    ) -> jax.Array:
        """
        Forward pass through the fused MoE layer.

        Args:
            hidden_states: Input tokens, shape (num_tokens, hidden_size) or
                          (batch_size, seq_len, hidden_size)
            router_logits: Router output logits, shape (num_tokens, num_experts)
                          Note: Should be raw logits, not after softmax or top-k

        Returns:
            MoE layer output, same shape as hidden_states
        """
        assert hidden_states.ndim == 2

        hidden_states = jax.sharding.reshard(hidden_states, P("tensor", None))
        router_logits = jax.sharding.reshard(router_logits, P("tensor", None))
        if router_bias is not None:
            router_bias = jax.sharding.reshard(router_bias, P())

        w1_shared_val = self.w1_shared.value if self.w1_shared is not None else None
        w3_shared_val = self.w3_shared.value if self.w3_shared is not None else None
        w2_shared_val = self.w2_shared.value if self.w2_shared is not None else None

        output = fused_ep_moe(
            mesh=self.mesh,
            tokens=hidden_states,
            w1=self.w1.value,
            w3=self.w3.value,
            w2=self.w2.value,
            gating_output=router_logits,
            bias=router_bias,
            top_k=self.num_experts_per_tok,
            renormalize_topk_logits=self.renormalize_topk_logits,
            routed_scaling_factor=self.routed_scaling_factor,
            use_grouped_topk=self.use_grouped_topk,
            num_groups=self.num_groups,
            top_k_groups=self.top_k_groups,
            w1_shared=w1_shared_val,
            w3_shared=w3_shared_val,
            w2_shared=w2_shared_val,
            act_fn=self.activation,
            # Tile sizes
            bt=self.bt,
            bf=self.bf,
            bd1=self.bd1,
            bd2=self.bd2,
            btc=self.btc,
            bfc=self.bfc,
            bd1c=self.bd1c,
            bd2c=self.bd2c,
            # Optional parameters (not used in basic case)
            subc_quant_wsz=None,
            w1_scale=None,
            w3_scale=None,
            w2_scale=None,
            b1=None,
            b3=None,
            b2=None,
            ep_axis_name="tensor",
        )

        final_output = jax.sharding.reshard(output, P(None))
        return final_output
