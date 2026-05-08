"""LightningAttnBackend — GLA (Gated Linear Attention) backend for BailingMoeV2.5.

Extends LinearRecurrentAttnBackend to provide:
- Chunked prefill via simple_gla_fwd (Pallas kernel, varlen — kernel pads each
  sequence internally, so cu_seqlens carries real lengths)
- Decode via fused_recurrent_simple_gla (jax.lax.scan)
- Recurrent state management through RecurrentStatePool (no conv state)

Aligns with upstream sglang's LightningAttentionBackend(MambaAttnBackendBase) pattern.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.layers.attention.hybrid_linear_attn_backend import (
    LinearRecurrentAttnBackend,
)
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode

try:
    from sgl_jax.srt.kernels.simple_gla.simple_gla import (
        fused_recurrent_simple_gla,
        simple_gla_fwd,
    )
except ModuleNotFoundError:
    simple_gla_fwd = None
    fused_recurrent_simple_gla = None

if TYPE_CHECKING:
    from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch

_CHUNK_SIZE = 64


class LightningAttnBackend(LinearRecurrentAttnBackend):
    """Attention backend for GLA (Gated Linear Attention) used by BailingMoeV2.5."""

    def __init__(self, mesh: jax.sharding.Mesh = None, chunk_size: int = _CHUNK_SIZE):
        super().__init__(mesh=mesh)
        self.chunk_size = chunk_size

    def __call__(
        self,
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        layer,
        forward_batch: ForwardBatch,
        recurrent_state_pool,
        **kwargs,
    ) -> tuple[jax.Array, tuple]:
        recurrent_indices = self.forward_metadata.recurrent_indices
        ssm_states = self.get_state(recurrent_state_pool, layer.layer_id, recurrent_indices)

        if forward_batch.forward_mode.is_decode():
            output, new_recurrent = self._forward_decode(q, k, v, ssm_states, layer)
        elif forward_batch.forward_mode == ForwardMode.EXTEND:
            output, new_recurrent = self._forward_extend(q, k, v, ssm_states, layer)
        else:
            raise NotImplementedError(
                f"LightningAttnBackend does not support {forward_batch.forward_mode}"
            )

        new_ssm_full = self.set_ssm_state(
            recurrent_state_pool, layer.layer_id, recurrent_indices, new_recurrent
        )
        return output.reshape(output.shape[0], -1), (new_ssm_full, [])

    def get_state(self, recurrent_state_pool, layer_id, recurrent_indices):
        recurrent_buffer, _ = self.get_layer_cache(recurrent_state_pool, layer_id)
        return recurrent_buffer[recurrent_indices]

    def set_ssm_state(self, recurrent_state_pool, layer_id, recurrent_indices, new_recurrent):
        recurrent_buffer, _ = self.get_layer_cache(recurrent_state_pool, layer_id)
        return recurrent_buffer.at[recurrent_indices].set(new_recurrent)

    def _forward_decode(
        self,
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        ssm_states: jax.Array,
        layer,
    ) -> tuple[jax.Array, jax.Array]:
        if fused_recurrent_simple_gla is None:
            raise ImportError("simple_gla kernel is required for GLA decode")

        ssm_states = ssm_states.astype(jnp.float32)
        ssm_states = jax.sharding.reshard(
            ssm_states,
            NamedSharding(layer.mesh, P(None, "tensor", None, None)),
        )

        q_d = q[:, None, :, :]
        k_d = k[:, None, :, :]
        v_d = v[:, None, :, :]
        output_d, new_state = fused_recurrent_simple_gla(
            q_d,
            k_d,
            v_d,
            g_gamma=layer.slope,
            initial_state=ssm_states,
            output_final_state=True,
            scale=None,
        )
        return output_d[:, 0, :, :], new_state

    def _forward_extend(
        self,
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        ssm_states: jax.Array,
        layer,
    ) -> tuple[jax.Array, jax.Array]:
        if simple_gla_fwd is None:
            raise ImportError("simple_gla kernel is required for GLA prefill")

        cu_seqlens = self.forward_metadata.cu_q_lens
        ssm_states = ssm_states.astype(jnp.float32)

        slope_sm = jax.sharding.reshard(layer.slope, NamedSharding(layer.mesh, P("tensor")))
        h0_sm = jax.sharding.reshard(
            ssm_states,
            NamedSharding(layer.mesh, P(None, "tensor", None, None)),
        )

        chunk_size = self.chunk_size

        def _prefill_fn(q_local, k_local, v_local, gamma, h0, cu_seqlens_p):
            return simple_gla_fwd(
                q_local,
                k_local,
                v_local,
                g_gamma=gamma,
                h0=h0,
                cu_seqlens_dev=cu_seqlens_p,
                scale=None,
                use_ht=True,
                chunk_size=chunk_size,
            )

        # q/k/v come in as [T_outer, H, K]; the kernel expects [1, T_outer, H, K].
        q_b = q[None]
        k_b = k[None]
        v_b = v[None]

        output, new_state = jax.shard_map(
            _prefill_fn,
            mesh=layer.mesh,
            in_specs=(
                P(None, None, "tensor", None),
                P(None, None, "tensor", None),
                P(None, None, "tensor", None),
                P("tensor"),
                P(None, "tensor", None, None),
                P(),
            ),
            out_specs=(
                P(None, None, "tensor", None),
                P(None, "tensor", None, None),
            ),
            check_vma=False,
        )(q_b, k_b, v_b, slope_sm, h0_sm, cu_seqlens)

        return output[0], new_state


__all__ = ["LightningAttnBackend"]
