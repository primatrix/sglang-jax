"""Sparse DSA decode MLA kernels and host-side reference helpers."""

from sgl_jax.srt.kernels.mla.dsa.attention import (
    selected_mla_attention,
    selected_mla_attention_unchecked,
)
from sgl_jax.srt.kernels.mla.dsa.gather import (
    SPARSECORE_COMPILER_OPTIONS,
    materialize_selected_kv_sparsecore,
    materialize_selected_kv_sparsecore_unchecked,
    materialize_selected_kv_xla,
    prepare_safe_topk_slots,
)
from sgl_jax.srt.kernels.mla.dsa.kernel import (
    dsa_decode_mla_attention,
    dsa_decode_mla_attention_unchecked,
)

from sgl_jax.srt.kernels.mla.dsa.reference import (
    dense_selected_mla_attention,
    reference_dsa_decode_mla_attention,
    reference_selected_mla_attention,
)

__all__ = [
    "dense_selected_mla_attention",
    "dsa_decode_mla_attention",
    "dsa_decode_mla_attention_unchecked",
    "materialize_selected_kv_sparsecore",
    "materialize_selected_kv_sparsecore_unchecked",
    "materialize_selected_kv_xla",
    "prepare_safe_topk_slots",
    "reference_dsa_decode_mla_attention",
    "reference_selected_mla_attention",
    "selected_mla_attention",
    "selected_mla_attention_unchecked",
    "SPARSECORE_COMPILER_OPTIONS",
]
