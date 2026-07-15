"""TensorCore attention over a contiguous selected MLA cache."""

from __future__ import annotations

import functools
import math

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from sgl_jax.srt.kernels.mla.v2.kernel import prepare_outputs, prepare_q_inputs


def _selected_mla_attention_kernel(
    valid_counts_ref,
    ql_nope_hbm_ref,
    q_pe_hbm_ref,
    selected_kv_hbm_ref,
    output_hbm_ref,
    ql_nope_vmem_ref,
    q_pe_vmem_ref,
    selected_kv_vmem_ref,
    output_vmem_ref,
    dma_sem,
    *,
    sm_scale: float,
):
    """Compute complete fixed-K softmax attention for one decode request."""
    batch_index = pl.program_id(0)

    def copy_and_wait(source_ref, destination_ref):
        copy = pltpu.make_async_copy(source_ref, destination_ref, dma_sem)
        copy.start()
        copy.wait()

    copy_and_wait(ql_nope_hbm_ref.at[batch_index], ql_nope_vmem_ref)
    copy_and_wait(q_pe_hbm_ref.at[batch_index], q_pe_vmem_ref)
    copy_and_wait(selected_kv_hbm_ref.at[batch_index], selected_kv_vmem_ref)

    num_head_words, q_packing, padded_latent_dim = ql_nope_vmem_ref.shape
    padded_rope_dim = q_pe_vmem_ref.shape[-1]
    num_heads = num_head_words * q_packing
    max_selected = selected_kv_vmem_ref.shape[0]

    ql_nope = ql_nope_vmem_ref[...].reshape(
        (num_heads, padded_latent_dim)
    )
    q_pe = q_pe_vmem_ref[...].reshape((num_heads, padded_rope_dim))
    selected_kv = selected_kv_vmem_ref[...]

    scores = jnp.einsum(
        "hd,kd->hk",
        ql_nope,
        selected_kv[:, :padded_latent_dim],
        preferred_element_type=jnp.float32,
    )
    scores += jnp.einsum(
        "hd,kd->hk",
        q_pe,
        selected_kv[:, padded_latent_dim:],
        preferred_element_type=jnp.float32,
    )
    scores *= jnp.float32(sm_scale)

    valid_count = valid_counts_ref[batch_index]
    selected_positions = lax.broadcasted_iota(
        jnp.int32, (num_heads, max_selected), 1
    )
    scores = jnp.where(selected_positions < valid_count, scores, -jnp.inf)
    row_max = jnp.max(scores, axis=-1, keepdims=True)
    probabilities = jnp.exp(scores - row_max)
    denominator = jnp.sum(probabilities, axis=-1, keepdims=True)
    weighted_values = jnp.einsum(
        "hk,kd->hd",
        probabilities,
        selected_kv[:, :padded_latent_dim],
        preferred_element_type=jnp.float32,
    )
    output = weighted_values / denominator

    output_vmem_ref[...] = output.reshape(output_vmem_ref.shape).astype(
        output_vmem_ref.dtype
    )
    copy_and_wait(output_vmem_ref, output_hbm_ref.at[batch_index])


def selected_mla_attention_unchecked(
    ql_nope: jax.Array,
    q_pe: jax.Array,
    selected_kv: jax.Array,
    valid_counts: jax.Array,
    *,
    sm_scale: float,
    interpret: bool = False,
) -> jax.Array:
    """Run contiguous selected-KV MLA attention without value validation."""
    if not interpret and jax.default_backend() != "tpu":
        raise RuntimeError(
            "selected_mla_attention requires a TPU when interpret=False"
        )

    ql_nope = jnp.asarray(ql_nope)
    q_pe = jnp.asarray(q_pe)
    selected_kv = jnp.asarray(selected_kv)
    valid_counts = jnp.asarray(valid_counts, dtype=jnp.int32)
    actual_num_heads = ql_nope.shape[1]
    actual_latent_dim = ql_nope.shape[-1]

    ql_nope = prepare_q_inputs(ql_nope)
    q_pe = prepare_q_inputs(q_pe)
    if ql_nope.shape[:3] != q_pe.shape[:3]:
        raise ValueError("prepared Q-nope and Q-RoPE packing must match")

    batch_size, num_head_words, q_packing, padded_latent_dim = ql_nope.shape
    padded_rope_dim = q_pe.shape[-1]
    if selected_kv.shape != (
        batch_size,
        selected_kv.shape[1],
        padded_latent_dim + padded_rope_dim,
    ):
        raise ValueError(
            "selected_kv must be [batch, selected, padded_latent + padded_rope]"
        )

    kernel = pl.pallas_call(
        functools.partial(_selected_mla_attention_kernel, sm_scale=float(sm_scale)),
        out_shape=jax.ShapeDtypeStruct(ql_nope.shape, ql_nope.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(batch_size,),
            in_specs=(
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec(memory_space=pltpu.HBM),
            ),
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            scratch_shapes=(
                pltpu.VMEM(
                    (num_head_words, q_packing, padded_latent_dim), ql_nope.dtype
                ),
                pltpu.VMEM(
                    (num_head_words, q_packing, padded_rope_dim), q_pe.dtype
                ),
                pltpu.VMEM(selected_kv.shape[1:], selected_kv.dtype),
                pltpu.VMEM(
                    (num_head_words, q_packing, padded_latent_dim), ql_nope.dtype
                ),
                pltpu.SemaphoreType.DMA,
            ),
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",)
        ),
        interpret=interpret,
        name="dsa-selected-mla-attention",
    )
    packed_output = kernel(valid_counts, ql_nope, q_pe, selected_kv)
    return prepare_outputs(
        packed_output,
        actual_num_heads,
        actual_latent_dim,
    )


def selected_mla_attention(
    ql_nope: jax.Array,
    q_pe: jax.Array,
    selected_kv: jax.Array,
    valid_counts: jax.Array,
    *,
    sm_scale: float,
    interpret: bool = False,
) -> jax.Array:
    """Validate static selected-attention inputs and launch the Pallas kernel."""
    if ql_nope.ndim != 3 or q_pe.ndim != 3 or selected_kv.ndim != 3:
        raise ValueError("queries and selected_kv must be rank-3 arrays")
    if valid_counts.ndim != 1:
        raise ValueError("valid_counts must be rank-1")
    if ql_nope.shape[:2] != q_pe.shape[:2]:
        raise ValueError("Q-nope and Q-RoPE batch/head dimensions must match")
    if selected_kv.shape[0] != ql_nope.shape[0]:
        raise ValueError("selected_kv must have one row per query batch item")
    if valid_counts.shape[0] != ql_nope.shape[0]:
        raise ValueError("valid_counts must have one entry per batch item")
    if not math.isfinite(float(sm_scale)):
        raise ValueError("sm_scale must be finite")
    return selected_mla_attention_unchecked(
        ql_nope,
        q_pe,
        selected_kv,
        valid_counts,
        sm_scale=sm_scale,
        interpret=interpret,
    )


__all__ = ["selected_mla_attention", "selected_mla_attention_unchecked"]
