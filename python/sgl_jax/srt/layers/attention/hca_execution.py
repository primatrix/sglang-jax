"""HCA metadata utilities and pool-independent sharded execution."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import numpy as np
from jax.sharding import PartitionSpec as P
from jax.tree_util import register_pytree_node_class

from sgl_jax.srt.kernels.hca.hca import HCAMetadata, fused_projection_weight, hca_step
from sgl_jax.srt.kernels.hca.tuned_block_sizes import HCAKernelSchedule
from sgl_jax.srt.layers.attention.base_attn_backend import AttentionBackendMetadata


@register_pytree_node_class
@dataclass
class HCABackendMetadata(AttentionBackendMetadata):
    """Per-forward HCA metadata plus static framework/optimization choices."""

    kernel: HCAMetadata | None = None
    schedule: HCAKernelSchedule | None = None
    use_uniform_prefill_fast_path: bool = False

    def tree_flatten(self):
        return (self.kernel,), (self.schedule, self.use_uniform_prefill_fast_path)

    @classmethod
    def tree_unflatten(cls, static_options, children):
        schedule, use_uniform_prefill_fast_path = static_options
        return cls(
            kernel=children[0],
            schedule=schedule,
            use_uniform_prefill_fast_path=use_uniform_prefill_fast_path,
        )


# Minimum padded capacities: small floors stop tiny batches re-bucketing every
# few steps, and the page-table floors hold one large batch's tables.
_DECODE_IDS_FLOOR = 8
_BOUNDARY_FLOOR = 8
_WINDOW_TABLE_FLOOR = 512
_COMPRESSED_TABLE_FLOOR = 64


def _query_schedule(cu_q_lens: np.ndarray, query_block_size: int):
    """Build execution-only query blocks after the platform schedule is known.

    Returns possibly-empty arrays; ``get_forward_metadata`` pads them to stable
    capacities."""
    q_lens = np.diff(cu_q_lens).astype(np.int32)
    block_counts = np.where(
        q_lens == 1,
        0,
        (q_lens + query_block_size - 1) // query_block_size,
    )
    request_ids = np.repeat(np.arange(q_lens.size, dtype=np.int32), block_counts)
    offsets = np.concatenate(
        [
            (
                np.arange(0, int(q_len), query_block_size, dtype=np.int32)
                if q_len != 1
                else np.empty((0,), np.int32)
            )
            for q_len in q_lens
        ]
    )
    return request_ids, offsets.astype(np.int32), np.flatnonzero(q_lens == 1).astype(np.int32)


def _pad_capacity(values: np.ndarray, capacity: int, fill) -> np.ndarray:
    """Right-pad to an exact batch-shape-derived capacity with an inert fill."""
    values = np.asarray(values, np.int32)
    if values.shape[0] > capacity:
        raise ValueError(f"HCA metadata length {values.shape[0]} exceeds capacity {capacity}")
    # np.pad costs ~20 us per call; this runs several times per decode tick.
    padded = np.full((capacity,) + values.shape[1:], fill, np.int32)
    padded[: values.shape[0]] = values
    return padded


def _bucket_capacity(length: int, floor: int, bound: int | None = None) -> int:
    """Smallest power-of-two capacity covering ``length``, capped at ``bound``."""
    capacity = floor
    while capacity < length:
        capacity *= 2
    return capacity if bound is None else min(capacity, bound)


def _bucket_max_queries(max_queries: int, floor: int) -> int:
    """Bucket the per-request query capacity to a bounded ladder.

    Decode (1) keeps its dedicated value; longer chunks round up onto powers of
    two interleaved with 1.5x steps (..., 128, 192, 256, 384, ...), so a chunk
    just past a power of two pays 1.5x KV staging instead of 2x.
    """
    if max_queries <= 1:
        return max_queries
    power = 1 << max((max_queries - 1).bit_length() - 1, 0)
    bucket = power * 3 // 2 if max_queries <= power * 3 // 2 else power * 2
    return max(floor, bucket)


def _metadata_partition_spec(metadata: HCAMetadata) -> HCAMetadata:
    """Every metadata leaf rides the leading data axis, like SGLang batch fields.

    Derived rather than hand-listed so a new field cannot silently miss a spec.
    """
    return jax.tree.map(lambda _: P("data"), metadata)


def _check_constants(
    wkv,
    wgate,
    ape,
    norm_weight,
    cos,
    sin,
    attention_sink,
    fused_weight,
    max_context_len,
    *,
    head_dim,
    compressor_hidden_size,
    compress_ratio,
    num_heads,
) -> None:
    """Validate the shapes that cannot change between steps.

    Factored out of the hot path for readability; under jit these run at
    trace time only. All checks run at trace time and do not mutate runtime state.
    """
    if jax.default_backend() != "tpu":
        raise RuntimeError("production HCABackend requires a TPU backend")
    weight_shape = (head_dim, compressor_hidden_size)
    if wkv.shape != weight_shape or wgate.shape != weight_shape:
        raise ValueError("wkv and wgate must both be [512,4096]")
    if ape.shape != (compress_ratio, head_dim):
        raise ValueError("ape must be [128,512]")
    if norm_weight.shape != (head_dim,):
        raise ValueError("norm_weight must be [512]")
    if attention_sink.shape != (num_heads,):
        raise ValueError("attention_sink must be [64]")
    if cos.ndim != 2 or cos.shape[1] != 32 or sin.shape != cos.shape:
        raise ValueError("production HCA RoPE tables must both be [positions,32]")
    # Boundary emission gathers the row at each group start; a shorter table
    # would silently rotate records with clamped or filled frequencies.
    min_rope_rows = max(1, max_context_len - compress_ratio + 1)
    if cos.shape[0] < min_rope_rows:
        raise ValueError(
            f"RoPE tables cover {cos.shape[0]} positions but max_context_len="
            f"{max_context_len} requires at least {min_rope_rows}"
        )
    if fused_weight is not None and fused_weight.shape != (
        compressor_hidden_size,
        2 * head_dim,
    ):
        raise ValueError("fused_weight must be [4096,1024]")


def run_hca(
    q,
    k,
    v,
    *,
    mesh,
    page_size,
    max_context_len,
    positions,
    forward_mode,
    state_arg,
    window_arg,
    compressed_arg,
    metadata,
    compressor_input,
    wkv,
    wgate,
    ape,
    norm_weight,
    cos,
    sin,
    attention_sink,
    fused_weight=None,
    softmax_scale=None,
):
    """Execute HCA on explicit cache arrays; return output and replacement arrays.

    JAX threads cache state through the compiled model instead of mutating pools
    as the PyTorch backend does. Allocation and layer lookup belong to callers.
    """
    num_heads, head_dim, compressor_hidden_size = 64, 512, 4096
    compress_ratio, window_size = 128, 128
    if metadata.kernel is None:
        raise RuntimeError("HCABackend.forward_metadata has not been prepared")
    if metadata.schedule is None:
        raise RuntimeError("HCABackend has no HCA kernel schedule")
    # Only the per-token shapes can change between steps; the model
    # constants are validated in _check_constants below.
    if q.ndim != 3 or q.shape[1:] != (num_heads, head_dim):
        raise ValueError("q must be [T,num_attn_heads,head_dim]")
    if k.ndim == 3 and k.shape[1] == 1:
        new_kv = k[:, 0]
    elif k.ndim == 2:
        new_kv = k
    else:
        raise ValueError("HCA k/v must be [T,D] or [T,1,D]")
    if v.shape != k.shape or new_kv.shape != (q.shape[0], head_dim):
        raise ValueError("HCA k and v must share the same KV shape")
    if compressor_input.shape != (q.shape[0], compressor_hidden_size):
        raise ValueError("compressor_input must be [T,4096]")
    _check_constants(
        wkv,
        wgate,
        ape,
        norm_weight,
        cos,
        sin,
        attention_sink,
        fused_weight,
        max_context_len,
        head_dim=head_dim,
        compressor_hidden_size=compressor_hidden_size,
        compress_ratio=compress_ratio,
        num_heads=num_heads,
    )

    kernel_options = {
        "softmax_scale": head_dim**-0.5 if softmax_scale is None else float(softmax_scale),
        "compress_ratio": compress_ratio,
        "head_dim": head_dim,
        "window_size": window_size,
        "page_size": page_size,
        # Compressed records per original-token page: the kernels need it when
        # the compressed pool is handed over in its flat [rows, D] layout.
        "compressed_page_size": max(1, page_size // compress_ratio),
        "schedule": metadata.schedule,
    }
    fused_weight = fused_projection_weight(wkv, wgate, fused_weight)

    if forward_mode.is_decode():
        kernel_options["mode"] = "decode"
    elif forward_mode.is_extend():
        kernel_options["mode"] = "uniform" if metadata.use_uniform_prefill_fast_path else "ragged"
    else:
        raise ValueError(f"unsupported HCA forward mode: {forward_mode}")

    # Built inline like the MLA and GDN backends: under the model's outer
    # jit this is traced once, so a cached callable buys nothing.
    def rank_local(
        x,
        q_,
        new_kv_,
        state,
        window,
        compressed,
        wkv_,
        wgate_,
        ape_,
        norm_,
        cos_,
        sin_,
        positions_,
        sink_,
        md,
        fused,
    ):
        output, state, window, compressed = hca_step(
            x,
            q_,
            new_kv_,
            state,
            window,
            compressed,
            wkv_,
            wgate_,
            ape_,
            norm_,
            cos_,
            sin_,
            positions_,
            sink_,
            md,
            fused_weight=fused,
            **kernel_options,
        )
        return output.reshape(output.shape[0], -1), state, window, compressed

    output, state, window, compressed = jax.shard_map(
        rank_local,
        mesh=mesh,
        in_specs=(
            P("data", None),  # compressor_input [T, hidden]
            P("data", "tensor", None),  # q                [T, H/tp, D]
            P("data", None),  # new_kv           [T, D]
            _data_spec(state_arg),  # recurrent state pool (rank follows the buffer)
            _data_spec(window_arg),  # window cache
            _data_spec(compressed_arg),  # compressed cache
            P(None, None),  # wkv
            P(None, None),  # wgate
            P(None, None),  # ape
            P(None),  # norm_weight
            P(None, None),  # cos
            P(None, None),  # sin
            P("data"),  # positions        [T]
            P("tensor"),  # attention_sink   [H/tp]
            _metadata_partition_spec(metadata.kernel),
            P(None, None),  # fused_weight
        ),
        out_specs=(
            P("data", "tensor"),  # output [T, H/tp*D]
            _data_spec(state_arg),  # state pool
            _data_spec(window_arg),  # window cache
            _data_spec(compressed_arg),  # compressed cache
        ),
        check_vma=False,
    )(
        compressor_input,
        q,
        new_kv,
        state_arg,
        window_arg,
        compressed_arg,
        wkv,
        wgate,
        ape,
        norm_weight,
        cos,
        sin,
        positions,
        attention_sink,
        metadata.kernel,
        fused_weight,
    )
    return output.astype(q.dtype), (state, window, compressed)


def _data_spec(array) -> P:
    """``P("data", None, ...)`` matching the array rank (flat or 4D pool views)."""
    return P("data", *([None] * (array.ndim - 1)))
