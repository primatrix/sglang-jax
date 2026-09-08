"""DeepSeek V4 MoE block for the EPMoE backend.

The caller supplies flattened token IDs alongside the collapsed [T, H] mHC
stream. This block does not own scheduler metadata or full-model loading.
"""

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.layers.activation import silu_and_mul_with_clamp
from sgl_jax.srt.layers.gate import GateLogit, TopK
from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.layers.moe import EPMoE


class DeepseekV4SharedMLP(nnx.Module):
    def __init__(self, hidden_size, intermediate_size, mesh, dtype, swiglu_limit):
        self.swiglu_limit = swiglu_limit
        for name in ("gate_proj", "up_proj", "down_proj"):
            down = name == "down_proj"
            setattr(
                self,
                name,
                LinearBase(
                    input_size=intermediate_size if down else hidden_size,
                    output_size=hidden_size if down else intermediate_size,
                    kernel_axes=("tensor", None) if down else (None, "tensor"),
                    use_bias=False,
                    params_dtype=dtype,
                    mesh=mesh,
                    scope_name=name,
                ),
            )

    def __call__(self, hidden_states):
        gate, _ = self.gate_proj(hidden_states)
        up, _ = self.up_proj(hidden_states)
        activated = (
            jax.nn.silu(gate) * up
            if self.swiglu_limit is None
            else silu_and_mul_with_clamp(gate, up, self.swiglu_limit)
        )
        output, _ = self.down_proj(activated)
        return output


class DeepseekV4MoE(nnx.Module):
    """Hash layers and learned routing share scoring, normalization and experts.

    ``load_hash_table`` must be called with the checkpoint's ``gate.tid2eid``
    before real inference; the deterministic initial table is for dummy loads.
    ``route`` exposes weights/IDs for routing diagnostics without running GMM.
    """

    def __init__(self, config, mesh, layer_id, dtype=jnp.bfloat16):
        self.mesh = mesh
        self.hidden_size = config.hidden_size
        self.num_experts = config.n_routed_experts
        self.top_k = config.num_experts_per_tok
        self.vocab_size = config.vocab_size
        self.is_hash_layer = layer_id < config.num_hash_layers
        if not 0 < self.top_k <= self.num_experts:
            raise ValueError("num_experts_per_tok must be in [1, n_routed_experts]")
        if self.vocab_size <= 0 or layer_id < 0:
            raise ValueError("vocab_size must be positive and layer_id nonnegative")
        self.gate = GateLogit(
            self.hidden_size,
            self.num_experts,
            weight_dtype=jnp.float32,
            enable_expert_bias=not self.is_hash_layer,
            score_func=getattr(config, "scoring_func", "sqrtsoftplus"),
        )
        if self.is_hash_layer:
            table = (np.arange(self.vocab_size)[:, None] + np.arange(self.top_k)) % self.num_experts
            self.gate.tid2eid = nnx.Param(
                jax.device_put(table.astype(np.int32), NamedSharding(mesh, P(None, None)))
            )
        self.topk = TopK(
            topk=self.top_k,
            renormalize=config.norm_topk_prob,
            num_expert_group=getattr(config, "n_group", 1),
            topk_group=getattr(config, "topk_group", 1),
            routed_scaling_factor=config.routed_scaling_factor,
            layer_id=layer_id,
            mesh=mesh,
        )
        self.experts = EPMoE(
            hidden_size=self.hidden_size,
            num_experts=self.num_experts,
            num_experts_per_tok=self.top_k,
            intermediate_dim=config.moe_intermediate_size,
            mesh=mesh,
            ep_size=getattr(config, "ep_size", 1),
            moe_dp_size=getattr(config, "moe_dp_size", 1),
            weight_dtype=dtype,
            dtype=dtype,
            layer_id=layer_id,
            quantization_config=getattr(config, "quantization_config", None),
            swiglu_limit=config.swiglu_limit,
        )
        if getattr(config, "n_shared_experts", 0):
            self.shared_experts = DeepseekV4SharedMLP(
                self.hidden_size,
                config.moe_intermediate_size * config.n_shared_experts,
                mesh,
                dtype,
                config.swiglu_limit,
            )
        else:
            self.shared_experts = None

    def load_hash_table(self, table):
        """Load a host checkpoint tensor without floating-point dtype conversion."""
        if not self.is_hash_layer:
            raise ValueError("only hash layers have gate.tid2eid")
        table = np.asarray(table)
        if table.shape != (self.vocab_size, self.top_k):
            raise ValueError("gate.tid2eid must have shape [vocab_size, top_k]")
        if not np.issubdtype(table.dtype, np.integer):
            raise ValueError("gate.tid2eid must contain integer expert IDs")
        if np.any(table < 0) or np.any(table >= self.num_experts):
            raise ValueError("gate.tid2eid expert IDs are outside the logical expert range")
        self.gate.tid2eid.value = jax.device_put(
            table.astype(np.int32), NamedSharding(self.mesh, P(None, None))
        )

    def route(
        self,
        hidden_states,
        input_ids=None,
        *,
        token_valid_mask=None,
        dispatch_info=None,
        routing_sharding=None,
    ):
        if hidden_states.ndim != 2 or hidden_states.shape[1] != self.hidden_size:
            raise ValueError("hidden_states must have shape [tokens, hidden_size]")
        if self.experts.replicate_experts and dispatch_info is not None:
            raise ValueError("replicated experts do not support EPLB dispatch metadata")
        routing_sharding = routing_sharding or NamedSharding(self.mesh, P("data", None))
        if len(routing_sharding.spec) != 2 or routing_sharding.spec[1] is not None:
            raise ValueError("routing sharding must partition tokens only")
        hidden_states = jax.sharding.reshard(hidden_states, routing_sharding)
        token_sharding = NamedSharding(self.mesh, P(routing_sharding.spec[0]))
        tokens = hidden_states.shape[0]
        valid = jnp.ones((tokens,), dtype=jnp.bool_)
        if token_valid_mask is not None:
            if token_valid_mask.shape != (tokens,):
                raise ValueError("token_valid_mask must have shape [tokens]")
            valid = jax.sharding.reshard(token_valid_mask.astype(jnp.bool_), token_sharding)
        selected = None
        if self.is_hash_layer:
            if input_ids is None or input_ids.shape != (tokens,):
                raise ValueError("hash routing requires input_ids with shape [tokens]")
            if not jnp.issubdtype(input_ids.dtype, jnp.integer):
                raise ValueError("input_ids must be integers")
            input_ids = jax.sharding.reshard(input_ids, token_sharding)
            valid = valid & (input_ids >= 0) & (input_ids < self.vocab_size)
            # Padding must never produce negative IDs in EPMoE's bincount/permutation.
            safe_ids = jnp.where(valid, input_ids, 0)
            selected = self.gate.tid2eid.value.at[safe_ids].get(out_sharding=routing_sharding)
        scores = self.gate(jnp.where(valid[:, None], hidden_states, 0))
        weights, ids = self.topk(
            scores,
            None if self.is_hash_layer else self.gate.bias.value,
            dispatch_info,
            routing_sharding,
            selected_experts=selected,
        )
        return jnp.where(valid[:, None], weights, 0), ids

    def __call__(
        self,
        hidden_states,
        input_ids=None,
        *,
        token_valid_mask=None,
        dispatch_info=None,
        out_sharding=None,
    ):
        weights, ids = self.route(
            hidden_states,
            input_ids,
            token_valid_mask=token_valid_mask,
            dispatch_info=dispatch_info,
            routing_sharding=out_sharding,
        )
        # Zero invalid activations too: zero routing weight alone cannot mask NaN.
        valid = jnp.ones(hidden_states.shape[:1], dtype=jnp.bool_)
        if token_valid_mask is not None:
            valid = valid & token_valid_mask.astype(jnp.bool_)
        if self.is_hash_layer:
            valid = valid & (input_ids >= 0) & (input_ids < self.vocab_size)
        hidden_states = jnp.where(valid[:, None], hidden_states, 0)
        output = self.experts(hidden_states, weights, ids, out_sharding=out_sharding)
        if self.shared_experts is not None:
            # routed_scaling_factor applies only to routed weights, exactly once.
            shared = self.shared_experts(hidden_states)
            if out_sharding is not None:
                shared = jax.sharding.reshard(shared, out_sharding)
            output = output + shared
        return jnp.where(valid[:, None], output, 0), ids
