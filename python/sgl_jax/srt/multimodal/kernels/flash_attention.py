# adapted from https://github.com/jax-ml/jax/blob/main/jax/experimental/pallas/ops/tpu/flash_attention.py
# ruff: noqa: E741
"""Flash Attention TPU kernel."""

from __future__ import annotations

import dataclasses
import functools
import math
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from sgl_jax.srt.multimodal.kernels.tuned_block_sizes import (
    get_tuned_block_sizes,
    get_tuned_online_block_sizes,
)

DEFAULT_MASK_VALUE = -0.7 * float(jnp.finfo(jnp.dtype("float32")).max)
NUM_LANES = 128
NUM_SUBLANES = 8
DEFAULT_VMEM_LIMIT_BYTES = 128 * 1024 * 1024

# Pallas also needs VMEM for compiler-generated pipeline buffers and spilled
# vector temporaries.  Do not let the explicitly visible single-step working set
# consume the full logical limit.
_SINGLE_STEP_VMEM_UTILIZATION = 0.8

# Scalar-prefetch arrays live in TPU scalar memory. Fall back to the dense grid
# rather than constructing an oversized sparse schedule for untuned shapes.
_MAX_BLOCK_SPARSE_PREFETCH_ENTRIES = 8192


class SegmentIds(NamedTuple):
    """SegmentIds for Q and KV sequences.

    SegmentIds are used to generate segment mask, which prevents attention between
    different segments in the input sequence. Each array is a list of ids
    (integers).
    Only the token with the same id can attend to each other.

    Attributes:
      q: segment ids along the Q sequence.
      kv: segment ids along the KV sequence.
    """

    q: jax.Array  # [batch_size, q_seq_len]
    kv: jax.Array  # [batch_size, kv_seq_len]


@dataclasses.dataclass(frozen=True)
class BlockSizes:
    """Tile sizes parameterizing FlashAttention kernels.

    Those parameters have negligible effect on numerics, but affect performance
    greatly.
    """

    block_q: int
    block_k_major: int
    block_k: int
    block_b: int

    block_q_major_dkv: int | None = None
    block_k_major_dkv: int | None = None
    block_k_dkv: int | None = None
    block_q_dkv: int | None = None

    block_k_major_dq: int | None = None
    block_k_dq: int | None = None
    block_q_dq: int | None = None

    def __post_init__(self):
        def verify_major_minor(prefix, suffix, major, minor):
            if minor > major:
                raise ValueError(
                    f"{prefix}{suffix}={minor} should be smaller than"
                    f" {prefix}_major{suffix}={major}"
                )
            if major % minor != 0:
                raise ValueError(
                    f"{prefix}{suffix}={minor} should divide {prefix}_major{suffix}={major}"
                )

        verify_major_minor("block_k", "", self.block_k_major, self.block_k)
        if self.block_q_major_dkv is not None and self.block_q_dkv is not None:
            verify_major_minor("block_q", "_dkv", self.block_q_major_dkv, self.block_q_dkv)
        if self.block_k_major_dkv is not None and self.block_k_dkv is not None:
            verify_major_minor("block_k", "_dkv", self.block_k_major_dkv, self.block_k_dkv)
        if self.block_k_major_dq is not None and self.block_k_dq is not None:
            verify_major_minor("block_k", "_dq", self.block_k_major_dq, self.block_k_dq)

    @property
    def has_backward_blocks(self) -> bool:
        backward_blocks = (
            self.block_q_major_dkv,
            self.block_k_major_dkv,
            self.block_q_dkv,
            self.block_k_dkv,
            self.block_k_major_dq,
            self.block_k_dq,
            self.block_q_dq,
        )
        return all(b is not None for b in backward_blocks)

    @classmethod
    def get_default(cls, batch_size, num_heads, q_seq_len, kv_len, d_model):
        # TODO(apaszke,sharadmv): Select better parameters based on a heuristic.
        del batch_size, num_heads, q_seq_len, kv_len, d_model  # Unused.
        return BlockSizes(
            block_q=256,
            block_k_major=128,
            block_k=128,
            block_b=1,
            block_q_major_dkv=128,
            block_k_major_dkv=128,
            block_k_dkv=128,
            block_q_dkv=128,
            block_k_major_dq=128,
            block_k_dq=128,
            block_q_dq=128,
        )


def _single_step_vmem_estimate_bytes(
    q,
    k,
    v,
    ab,
    segment_ids,
    block_sizes: BlockSizes,
) -> int:
    """Conservatively estimate the VMEM working set of the untiled-K kernel."""
    block_b = block_sizes.block_b
    block_q = block_sizes.block_q
    kv_seq_len = k.shape[2]
    head_dim = q.shape[3]

    def tile_bytes(shape, dtype):
        return math.prod(shape) * jnp.dtype(dtype).itemsize

    # Input/output BlockSpecs are staged in VMEM by Pallas. Account for the
    # default double buffering used to overlap those transfers with compute.
    io_working_set = tile_bytes((block_b, 1, block_q, head_dim), q.dtype)
    io_working_set += tile_bytes((block_b, 1, kv_seq_len, head_dim), k.dtype)
    io_working_set += tile_bytes((block_b, 1, kv_seq_len, head_dim), v.dtype)
    io_working_set += tile_bytes((block_b, 1, block_q, head_dim), q.dtype)

    if ab is not None:
        io_working_set += tile_bytes((block_b, 1, block_q, kv_seq_len), ab.dtype)
    if segment_ids is not None:
        io_working_set += tile_bytes((block_b, block_q, NUM_LANES), segment_ids.q.dtype)
        io_working_set += tile_bytes((block_b, NUM_SUBLANES, kv_seq_len), segment_ids.kv.dtype)

    # QK^T and the softmax probabilities are FP32. The compiler can often
    # reuse their storage, but accounting for both keeps the fast-path decision
    # safe across compiler versions and fusion choices.
    logits_bytes = tile_bytes((block_b, block_q, kv_seq_len), jnp.float32)
    return 2 * io_working_set + 2 * logits_bytes


def _select_default_block_sizes(
    q,
    k,
    v,
    ab,
    segment_ids,
    *,
    vmem_limit_bytes: int,
) -> BlockSizes:
    """Select the single-step fast path only when its VMEM working set is safe."""
    batch_size, num_heads, q_seq_len, head_dim = q.shape
    kv_seq_len = k.shape[2]
    tiled = _select_tiled_block_sizes(q, k, v)

    block_q = get_tuned_block_sizes(
        q.dtype,
        k.dtype,
        v.dtype,
        batch_size,
        num_heads,
        q_seq_len,
        kv_seq_len,
        head_dim,
    )
    single_step = BlockSizes(
        block_q=block_q,
        block_b=tiled.block_b,
        block_k_major=kv_seq_len,
        block_k=kv_seq_len,
    )
    estimate = _single_step_vmem_estimate_bytes(q, k, v, ab, segment_ids, single_step)
    safe_limit = int(vmem_limit_bytes * _SINGLE_STEP_VMEM_UTILIZATION)
    return single_step if estimate <= safe_limit else tiled


def _select_tiled_block_sizes(
    q,
    k,
    v,
    *,
    max_segment_len: int | None = None,
) -> BlockSizes:
    """Select a measured online-softmax tile, with a portable safe fallback."""
    block_q, block_k_major, block_k = get_tuned_online_block_sizes(
        q.dtype,
        k.dtype,
        v.dtype,
        q.shape[2],
        k.shape[2],
        q.shape[3],
        max_segment_len=max_segment_len,
    )
    return BlockSizes(
        block_q=block_q,
        block_b=1,
        block_k_major=block_k_major,
        block_k=block_k,
    )


@functools.partial(
    jax.jit,
    static_argnames=[
        "causal",
        "sm_scale",
        "block_sizes",
        "debug",
        "interpret",
        "vmem_limit_bytes",
        "max_segment_len",
        "block_sparse_segments",
    ],
)
def flash_attention(
    q,  # [batch_size, num_heads, q_seq_len, d_model]
    k,  # [batch_size, num_heads, kv_seq_len, d_model]
    v,  # [batch_size, num_heads, kv_seq_len, d_model]
    ab=None,  # [batch_size, num_heads, q_seq_len, kv_seq_len]
    segment_ids=None,  # q of [batch_size, q_seq_len] and kv of [batch_size, kv_seq_len]
    *,
    causal: bool = False,
    sm_scale: float = 1.0,
    block_sizes: BlockSizes | None = None,
    debug: bool = False,
    interpret: bool = False,  # interpret=True for cpu
    vmem_limit_bytes: int = DEFAULT_VMEM_LIMIT_BYTES,
    max_segment_len: int | None = None,
    block_sparse_segments: bool = False,
):
    """Compute attention, optionally restricting sorted segments to a local K band.

    ``max_segment_len`` is an optimization contract for non-causal segmented
    attention. When set, non-negative segment IDs must be monotonically
    non-decreasing, Q/KV positions must be aligned self-attention positions, and
    no segment may exceed the bound. Negative IDs are treated as padding whose
    outputs are not consumed.

    ``block_sparse_segments`` keeps the exact token-level segment mask but skips
    Q/K block pairs whose non-negative segment-id ranges cannot overlap. It is
    useful for packed full attention where each segment is one image.
    """
    batch_size, num_heads, q_seq_len, d_model = q.shape
    batch_size_k, num_heads_k, kv_seq_len, d_model_k = k.shape
    batch_size_v, num_heads_v, kv_seq_len_v, d_model_v = v.shape
    if batch_size != batch_size_k or batch_size != batch_size_v:
        raise ValueError(
            f"Batch size mismatch: got {batch_size}, {batch_size_k} and"
            f" {batch_size_v} (for q, k, v respectively)"
        )
    if num_heads != num_heads_k or num_heads != num_heads_v:
        raise ValueError(
            f"Head count mismatch: got {num_heads}, {num_heads_k},"
            f" {num_heads_v} (for q, k, v respectively)"
        )
    if d_model != d_model_k:
        raise ValueError(
            f"Model dimension mismatch: got {d_model} and {d_model_k} (for q and k respectively)"
        )
    if d_model != d_model_v:
        raise NotImplementedError("V model dimension unequal to KV model dimension unsupported")
    if kv_seq_len != kv_seq_len_v:
        raise ValueError(f"KV sequence length mismatch: got {kv_seq_len} and {kv_seq_len_v}")
    if ab is not None and ab.shape != (batch_size, num_heads, q_seq_len, kv_seq_len):
        raise ValueError(
            f"Attention bias shape mismatch: expected ({batch_size=},"
            f" {num_heads=}, {q_seq_len=}, {kv_seq_len=}), got {ab.shape}"
        )
    if segment_ids is not None:
        if segment_ids.q.shape != (batch_size, q_seq_len):
            raise ValueError(
                f"Q segment ids shape mismatch: expected ({batch_size=},"
                f" {q_seq_len=},), got {segment_ids.q.shape}"
            )
        if segment_ids.kv.shape != (batch_size, kv_seq_len):
            raise ValueError(
                f"KV segment ids shape mismatch: expected ({batch_size=},"
                f" {kv_seq_len=},), got {segment_ids.kv.shape}"
            )
    if max_segment_len is not None:
        if max_segment_len <= 0:
            raise ValueError(f"{max_segment_len=} must be positive")
        if segment_ids is None:
            raise ValueError("max_segment_len requires segment_ids")
        if q_seq_len != kv_seq_len:
            raise ValueError("max_segment_len requires equal Q and KV sequence lengths")
        if causal:
            raise ValueError("max_segment_len is only supported for non-causal attention")
        if ab is not None:
            raise ValueError("max_segment_len is not supported with attention bias")
    if block_sparse_segments:
        if segment_ids is None:
            raise ValueError("block_sparse_segments requires segment_ids")
        if q_seq_len != kv_seq_len:
            raise ValueError("block_sparse_segments requires equal Q and KV sequence lengths")
        if causal:
            raise ValueError("block_sparse_segments is only supported for non-causal attention")
        if ab is not None:
            raise ValueError("block_sparse_segments is not supported with attention bias")
        if max_segment_len is not None:
            raise ValueError("block_sparse_segments and max_segment_len are mutually exclusive")
    if block_sizes is None:
        if max_segment_len is None:
            block_sizes = _select_default_block_sizes(
                q,
                k,
                v,
                ab,
                segment_ids,
                vmem_limit_bytes=vmem_limit_bytes,
            )
        else:
            block_sizes = _select_tiled_block_sizes(
                q,
                k,
                v,
                max_segment_len=max_segment_len,
            )
    return _flash_attention(
        q,
        k,
        v,
        ab,
        segment_ids,
        False,
        causal,
        sm_scale,
        block_sizes,
        debug,
        interpret,
        vmem_limit_bytes,
        max_segment_len,
        block_sparse_segments,
    )


def _flash_attention(
    q,
    k,
    v,
    ab,
    segment_ids,
    save_residuals,
    causal,
    sm_scale,
    block_sizes,
    debug,
    interpret,
    vmem_limit_bytes,
    max_segment_len,
    block_sparse_segments,
):
    return _flash_attention_impl(
        q,
        k,
        v,
        ab,
        segment_ids,
        save_residuals,
        causal,
        sm_scale,
        block_sizes.block_b,
        block_sizes.block_q,
        block_sizes.block_k_major,
        block_sizes.block_k,
        debug,
        interpret,
        vmem_limit_bytes,
        max_segment_len,
        block_sparse_segments,
    )


def _flash_attention_fwd(
    q,
    k,
    v,
    ab,
    segment_ids,
    save_residuals,
    causal,
    sm_scale,
    block_sizes,
    debug,
):
    if save_residuals:
        raise NotImplementedError("Higher-order AD not supported")
    o, l, m = _flash_attention(q, k, v, ab, segment_ids, True, causal, sm_scale, block_sizes, debug)
    return o, (q, k, v, ab, segment_ids, o, l, m)


MIN_BLOCK_SIZE = 128
TRANS_B_DIM_NUMBERS = (((1,), (1,)), ((), ()))


def below_or_on_diag(r, r_blk_size, c, c_blk_size):
    # A block is considered below or on diagonal as long as the bottom left
    # corner of the block is below or on diagonal.
    return ((r + 1) * r_blk_size - 1) > (c * c_blk_size)


def _flash_attention_kernel(q_tile_ref, *args, **kwargs):
    block_b = q_tile_ref.shape[0]
    # If we're not going to tile the softmax, then we can avoid a bunch of VPU ops.
    if kwargs["block_k"] == kwargs["kv_seq_len"]:
        kernel = _flash_attention_kernel_single_batch_single_step
    else:
        kernel = _flash_attention_kernel_single_batch
    for batch_idx in range(block_b):
        kernel((batch_idx, 0), q_tile_ref, *args, **kwargs)


def _flash_attention_block_sparse_kernel(
    block_mask_ref,
    prefetch_k_ref,
    q_tile_ref,
    *args,
    **kwargs,
):
    del prefetch_k_ref
    _flash_attention_kernel(
        q_tile_ref,
        *args,
        block_mask_ref=block_mask_ref,
        **kwargs,
    )


def _flash_attention_kernel_single_batch(
    batch_idx: tuple[int, ...],
    q_tile_ref,
    k_tile_ref,
    v_tile_ref,
    ab_tile_ref,
    q_segment_ids_tile_ref,
    kv_segment_ids_tile_ref,  # Input arrays
    o_tile_ref,  # Output arrays
    l_ref,
    m_ref,
    m_scratch_ref,
    l_scratch_ref,
    acc_scratch_ref,
    *,
    causal,
    sm_scale,
    block_k,
    kv_seq_len,
    kv_grid_size,
    max_segment_len,
    mask_value,
    block_mask_ref=None,
):
    block_k_major = k_tile_ref.shape[2]
    block_q = q_tile_ref.shape[2]
    head_dim = q_tile_ref.shape[-1]

    kv_grid_idx = pl.program_id(3)

    @pl.when(kv_grid_idx == 0)
    def start_new_sequence():
        m_scratch_ref[batch_idx] = jnp.full(m_scratch_ref.shape[2:], -jnp.inf, jnp.float32)
        l_scratch_ref[batch_idx] = jnp.zeros(l_scratch_ref.shape[2:], jnp.float32)
        acc_scratch_ref[batch_idx] = jnp.zeros(acc_scratch_ref.shape[2:], jnp.float32)

    q_seq_idx = pl.program_id(2)
    if max_segment_len is not None:
        halo_blocks = pl.cdiv(max_segment_len - 1, block_k_major)
        kv_seq_idx = q_seq_idx * (block_q // block_k_major) - halo_blocks + kv_grid_idx
        should_run = (kv_seq_idx >= 0) & (kv_seq_idx < (kv_seq_len // block_k_major))
    else:
        kv_seq_idx = kv_grid_idx
        should_run = (
            below_or_on_diag(q_seq_idx, block_q, kv_seq_idx, block_k_major) if causal else True
        )
    if block_mask_ref is not None:
        global_batch_idx = pl.program_id(0) * q_tile_ref.shape[0] + batch_idx[0]
        should_run &= block_mask_ref[global_batch_idx, q_seq_idx, kv_grid_idx] != 0

    @pl.when(should_run)
    def run():
        @pl.loop(0, block_k_major, step=block_k, unroll=True)
        def _body(start_k):
            m_prev = m_scratch_ref[batch_idx]
            l_prev = l_scratch_ref[batch_idx]

            q = q_tile_ref[batch_idx]  # [block_q, head_dim]
            k = k_tile_ref[
                (*batch_idx, pl.dslice(start_k, block_k), slice(None))
            ]  # [block_k, head_dim]

            s = jax.lax.dot_general(
                q, k, TRANS_B_DIM_NUMBERS, preferred_element_type=jnp.float32
            )  # [block_q, block_k]

            # Add attention bias if needed.
            # TODO(tanburn) Should the attention bias be added before or after
            # multiplication by sm_scale?
            if ab_tile_ref is not None:
                ab = ab_tile_ref[(*batch_idx, pl.dslice(None), pl.dslice(start_k, block_k))].astype(
                    jnp.float32
                )
                s += ab

            if sm_scale != 1.0:
                s *= sm_scale

            mask = None
            if q_segment_ids_tile_ref is not None:
                _, rem = divmod(block_k, NUM_LANES)
                if rem:
                    raise NotImplementedError(f"kv block size must be a multiple of {NUM_LANES}")
                q_segment_ids = q_segment_ids_tile_ref[batch_idx[0], :, :1]  # [block_q, 1].
                kv_segment_ids = kv_segment_ids_tile_ref[
                    batch_idx[0], :1, pl.dslice(start_k, block_k)
                ]  # [1, block_k].
                mask = jnp.equal(q_segment_ids, kv_segment_ids).astype(jnp.bool_)

            if causal:
                mask_shape = (block_q, block_k)
                row_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 0)
                row_ids += q_seq_idx * block_q
                col_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 1)
                col_ids += kv_seq_idx * block_k_major + start_k
                causal_mask = col_ids <= row_ids
                mask = causal_mask if mask is None else jnp.logical_and(mask, causal_mask)

            s = s if mask is None else jnp.where(mask, s, mask_value)

            m_curr = jnp.max(s, axis=1)[:, None]  # Row max, shape [block_q, 1].
            m_next = jnp.maximum(m_prev, m_curr)  # Shape [block_q, 128].

            block_k_repeats, rem = divmod(block_k, MIN_BLOCK_SIZE)
            if rem:
                raise NotImplementedError(f"{block_k=} should be a multiple of {MIN_BLOCK_SIZE}")
            p = jnp.exp(s - jnp.tile(m_next, (1, block_k_repeats)))

            alpha = jnp.exp(m_prev - m_next)  # Shape [block_q, 128].

            l_corr = alpha * l_prev

            l_next = jnp.sum(p, axis=1)[:, None] + l_corr  # Shape [block_q, 128]

            head_dim_repeats, rem = divmod(head_dim, MIN_BLOCK_SIZE)
            l_broadcast = lambda l: jnp.tile(l, (1, head_dim_repeats))
            if rem:
                if head_dim_repeats == 0:
                    l_broadcast = lambda l: l[:, :head_dim]
                else:
                    raise NotImplementedError(
                        f"{head_dim=} should be a multiple of {MIN_BLOCK_SIZE} if larger"
                    )
            l_scratch_ref[batch_idx] = l_next
            m_scratch_ref[batch_idx] = m_next

            l_next_inv_safe = jnp.where(l_next == 0.0, 1.0, 1.0 / l_next)
            acc_scratch_ref[batch_idx] *= l_broadcast(l_corr * l_next_inv_safe)
            v = v_tile_ref[(*batch_idx, pl.dslice(start_k, block_k), slice(None))]
            o_curr = jax.lax.dot(p.astype(v.dtype), v, preferred_element_type=jnp.float32)
            acc_scratch_ref[batch_idx] += o_curr * l_broadcast(l_next_inv_safe)

    @pl.when(kv_grid_idx == kv_grid_size - 1)
    def store_output():
        o_tile_ref[batch_idx] = acc_scratch_ref[batch_idx].astype(o_tile_ref.dtype)
        if l_ref is not None:
            l_ref[batch_idx] = l_scratch_ref[batch_idx].astype(l_ref.dtype)
        if m_ref is not None:
            m_ref[batch_idx] = m_scratch_ref[batch_idx].astype(m_ref.dtype)


def _flash_attention_kernel_single_batch_single_step(
    batch_idx: tuple[int, ...],
    q_tile_ref,
    k_tile_ref,
    v_tile_ref,
    ab_tile_ref,
    q_segment_ids_tile_ref,
    kv_segment_ids_tile_ref,  # Input arrays
    o_tile_ref,  # Output arrays
    l_ref: Any | None = None,
    m_ref: Any | None = None,
    *,
    causal,
    sm_scale,
    block_k,
    kv_seq_len,
    kv_grid_size,
    max_segment_len,
    mask_value,
):
    del kv_grid_size, max_segment_len
    block_k_major = k_tile_ref.shape[2]
    block_q = q_tile_ref.shape[2]

    assert kv_seq_len == block_k_major == block_k
    q = q_tile_ref[batch_idx]  # [block_q, head_dim]
    k = k_tile_ref[batch_idx]  # [block_k, head_dim]

    s = jax.lax.dot_general(
        q, k, TRANS_B_DIM_NUMBERS, preferred_element_type=jnp.float32
    )  # [block_q, block_k]

    if ab_tile_ref is not None:
        s += ab_tile_ref[batch_idx].astype(jnp.float32)
    if sm_scale != 1.0:
        s *= sm_scale

    mask = None
    if q_segment_ids_tile_ref is not None:
        repeats, rem = divmod(block_k, NUM_LANES)
        if rem:
            raise NotImplementedError(f"kv block size must be a multiple of {NUM_LANES}")
        q_segment_ids = q_segment_ids_tile_ref[batch_idx[0], :, :1]  # [block_q, 1]
        kv_segment_ids = kv_segment_ids_tile_ref[batch_idx[0], :1]  # [1, block_k].
        mask = jnp.equal(q_segment_ids, kv_segment_ids).astype(jnp.bool_)

    if causal:
        q_seq_idx = pl.program_id(2)
        mask_shape = (block_q, block_k)
        row_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 0)
        row_ids += q_seq_idx * block_q
        col_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 1)
        causal_mask = col_ids <= row_ids
        mask = causal_mask if mask is None else jnp.logical_and(mask, causal_mask)
    s = s if mask is None else jnp.where(mask, s, mask_value)

    m = jnp.max(s, axis=1, keepdims=True)
    p = jnp.exp(s - m)
    l = jnp.sum(p, axis=1, keepdims=True)
    p *= jax.lax.reciprocal(l)

    if m_ref is not None:
        m_ref[batch_idx] = lax.broadcast_in_dim(m, m_ref.shape[2:], range(2))
    if l_ref is not None:
        l_ref[batch_idx] = lax.broadcast_in_dim(l, l_ref.shape[2:], range(2))

    v = v_tile_ref[batch_idx]
    o_tile_ref[batch_idx] = jax.lax.dot(
        p.astype(v.dtype), v, preferred_element_type=jnp.float32
    ).astype(o_tile_ref.dtype)


def _bytes(x: jax.Array | jax.ShapeDtypeStruct) -> int:
    return math.prod(x.shape) * x.dtype.itemsize


def _fwd_cost_estimate(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    ab: jax.Array | None,
    segment_ids: SegmentIds | None,
    *,
    causal: bool,
    sm_scale: jax.Array | None,
    kernel_inputs_specs,
    kernel_outputs_specs,
) -> pl.CostEstimate | None:
    body_cost = pl.estimate_cost(
        mha_reference, q, k, v, ab, segment_ids, causal=causal, sm_scale=sm_scale
    )
    input_bytes = sum(_bytes(x) for x in jax.tree.leaves(kernel_inputs_specs))
    output_bytes = sum(_bytes(x) for x in jax.tree.leaves(kernel_outputs_specs))
    return pl.CostEstimate(
        flops=body_cost.flops,
        transcendentals=body_cost.transcendentals,
        bytes_accessed=input_bytes + output_bytes,
    )


def _segment_block_sparse_schedule(
    q_segment_ids: jax.Array,
    kv_segment_ids: jax.Array,
    *,
    block_q: int,
    block_k_major: int,
) -> tuple[jax.Array, jax.Array]:
    """Build an exact conservative block mask and its next-K prefetch map.

    Negative IDs are padding. For every block we retain the minimum and maximum
    non-negative segment ID. Disjoint ranges prove that a Q/K block pair has no
    matching tokens and can be skipped. Overlapping ranges are conservatively
    computed and still use the token-level equality mask inside the kernel.
    """

    def block_ranges(segment_ids: jax.Array, block_size: int):
        num_blocks = pl.cdiv(segment_ids.shape[1], block_size)
        padded_len = num_blocks * block_size
        if padded_len != segment_ids.shape[1]:
            segment_ids = jnp.pad(
                segment_ids,
                ((0, 0), (0, padded_len - segment_ids.shape[1])),
                constant_values=-1,
            )
        blocks = segment_ids.reshape(segment_ids.shape[0], num_blocks, block_size)
        valid = blocks >= 0
        first = jnp.min(
            jnp.where(valid, blocks, jnp.iinfo(blocks.dtype).max),
            axis=-1,
        )
        last = jnp.max(jnp.where(valid, blocks, -1), axis=-1)
        return first, last

    q_first, q_last = block_ranges(q_segment_ids, block_q)
    kv_first, kv_last = block_ranges(kv_segment_ids, block_k_major)
    block_mask = (
        (q_last[:, :, None] >= 0)
        & (kv_last[:, None, :] >= 0)
        & (q_first[:, :, None] <= kv_last[:, None, :])
        & (kv_first[:, None, :] <= q_last[:, :, None])
    )

    num_kv_blocks = block_mask.shape[-1]
    kv_indices = jnp.arange(num_kv_blocks, dtype=jnp.int32)
    candidates = jnp.where(block_mask, kv_indices, num_kv_blocks)
    next_k = lax.associative_scan(
        jnp.minimum,
        candidates,
        axis=candidates.ndim - 1,
        reverse=True,
    )
    prefetch_k = jnp.where(next_k < num_kv_blocks, next_k, 0).astype(jnp.int32)
    return block_mask.astype(jnp.int32), prefetch_k


def _flash_attention_impl(
    q,
    k,
    v,
    ab,
    segment_ids,
    save_residuals,
    causal,
    sm_scale,
    block_b,
    block_q,
    block_k_major,
    block_k,
    debug,
    interpret,
    vmem_limit_bytes,
    max_segment_len,
    block_sparse_segments,
):
    batch_size, num_heads, q_seq_len, head_dim = q.shape
    _, _, kv_seq_len, _ = k.shape
    _verify_block("block_q", "q_seq_len", block_q, q_seq_len, should_divide=False)
    _verify_block("block_k_major", "kv_seq_len", block_k_major, kv_seq_len)
    _verify_block("block_k", "kv_seq_len", block_k, kv_seq_len)
    _verify_block("block_b", "batch", block_b, batch_size, should_divide=False)

    num_kv_blocks = kv_seq_len // block_k_major
    local_halo_blocks = 0
    if max_segment_len is not None:
        if block_q % block_k_major:
            raise ValueError(
                "Local segmented attention requires block_q to be divisible by "
                f"block_k_major, got {block_q=} and {block_k_major=}."
            )
        local_halo_blocks = pl.cdiv(max_segment_len - 1, block_k_major)
        kv_grid_size = block_q // block_k_major + 2 * local_halo_blocks
    else:
        kv_grid_size = num_kv_blocks

    num_q_blocks = pl.cdiv(q_seq_len, block_q)
    block_sparse_entries = batch_size * num_q_blocks * kv_grid_size
    use_block_sparse_segments = (
        block_sparse_segments
        and block_k != kv_seq_len
        and block_sparse_entries <= _MAX_BLOCK_SPARSE_PREFETCH_ENTRIES
    )
    if use_block_sparse_segments:
        block_mask, prefetch_k = _segment_block_sparse_schedule(
            segment_ids.q,
            segment_ids.kv,
            block_q=block_q,
            block_k_major=block_k_major,
        )
    else:
        block_mask = prefetch_k = None

    # TODO(apaszke): Tile over heads as well.
    grid = (
        pl.cdiv(batch_size, block_b),
        num_heads,
        num_q_blocks,
        kv_grid_size,
    )

    def q_index_map(batch_index, head_index, q_seq_index, _, *scalar_prefetch):
        del scalar_prefetch
        return (batch_index, head_index, q_seq_index, 0)

    def local_kv_index(q_seq_index, kv_grid_index):
        first_kv_index = q_seq_index * (block_q // block_k_major) - local_halo_blocks
        return first_kv_index + kv_grid_index

    def safe_local_kv_index(q_seq_index, kv_grid_index):
        kv_index = local_kv_index(q_seq_index, kv_grid_index)
        return lax.clamp(0, kv_index, num_kv_blocks - 1)

    def kv_index_map(
        batch_index,
        head_index,
        q_seq_index,
        kv_seq_index,
        *scalar_prefetch,
    ):
        if use_block_sparse_segments:
            _, prefetch_k_ref = scalar_prefetch
            next_kv_index = prefetch_k_ref[batch_index, q_seq_index, kv_seq_index]
        elif max_segment_len is not None:
            next_kv_index = safe_local_kv_index(q_seq_index, kv_seq_index)
        elif causal:
            # If the kv block is skipped, prefetch the next valid kv block, i.e. the
            # 0th one to be used for the next block_q rows.
            next_kv_index = lax.select(
                below_or_on_diag(q_seq_index, block_q, kv_seq_index, block_k_major),
                kv_seq_index,
                0,
            )
        else:
            next_kv_index = kv_seq_index
        return (batch_index, head_index, next_kv_index, 0)

    def ab_index_map(
        batch_index,
        head_index,
        q_seq_index,
        kv_seq_index,
        *scalar_prefetch,
    ):
        del scalar_prefetch
        if causal:
            should_run = below_or_on_diag(q_seq_index, block_q, kv_seq_index, block_k_major)
            # If the ab block is skipped, prefetch the next valid ab block, i.e. the
            # 0th kv to be used for the next block_q rows.
            next_q_index = lax.select(
                should_run,
                q_seq_index,
                lax.select(q_seq_index == (q_seq_len // block_q) - 1, 0, q_seq_index + 1),
            )
            next_kv_index = lax.select(should_run, kv_seq_index, 0)
        else:
            next_q_index = q_seq_index
            next_kv_index = kv_seq_index

        return (batch_index, head_index, next_q_index, next_kv_index)

    def o_index_map(batch_index, head_index, q_seq_index, _, *scalar_prefetch):
        del scalar_prefetch
        return (batch_index, head_index, q_seq_index, 0)

    def lm_index_map(batch_index, head_index, q_seq_index, _, *scalar_prefetch):
        del scalar_prefetch
        return (batch_index, head_index, q_seq_index, 0)

    kernel = functools.partial(
        (
            _flash_attention_block_sparse_kernel
            if use_block_sparse_segments
            else _flash_attention_kernel
        ),
        causal=causal,
        mask_value=DEFAULT_MASK_VALUE,
        sm_scale=sm_scale,
        block_k=block_k,
        kv_seq_len=kv_seq_len,
        kv_grid_size=kv_grid_size,
        max_segment_len=max_segment_len,
    )
    out_shape = jax.ShapeDtypeStruct(shape=q.shape, dtype=q.dtype)
    out_shape = [out_shape]
    out_specs = [pl.BlockSpec((block_b, 1, block_q, head_dim), o_index_map)]

    if block_k != kv_seq_len:
        m_scratch = pltpu.VMEM((block_b, 1, block_q, MIN_BLOCK_SIZE), jnp.float32)
        l_scratch = pltpu.VMEM((block_b, 1, block_q, MIN_BLOCK_SIZE), jnp.float32)
        acc_scratch = pltpu.VMEM((block_b, 1, block_q, head_dim), jnp.float32)
        scratch_shapes = [m_scratch, l_scratch, acc_scratch]
    else:
        scratch_shapes = []

    if save_residuals:
        out_specs = [
            *out_specs,
            pl.BlockSpec((block_b, 1, block_q, MIN_BLOCK_SIZE), lm_index_map),
            pl.BlockSpec((block_b, 1, block_q, MIN_BLOCK_SIZE), lm_index_map),
        ]
        l = jax.ShapeDtypeStruct(
            (batch_size, num_heads, q_seq_len, MIN_BLOCK_SIZE), dtype=jnp.float32
        )
        m = jax.ShapeDtypeStruct(
            (batch_size, num_heads, q_seq_len, MIN_BLOCK_SIZE), dtype=jnp.float32
        )
        out_shape = (*out_shape, l, m)
    else:
        out_specs = [*out_specs, None, None]
        out_shape = (*out_shape, None, None)

    ab_block_spec = (
        pl.BlockSpec((block_b, 1, block_q, block_k_major), ab_index_map) if ab is not None else None
    )

    q_segment_ids_spec = kv_segment_ids_spec = None
    q_segment_ids = kv_segment_ids = None
    if segment_ids is not None:

        def q_segment_ids_index_map(
            batch_index,
            head_index,
            q_seq_index,
            _,
            *scalar_prefetch,
        ):
            del head_index, scalar_prefetch
            return (batch_index, q_seq_index, 0)

        def kv_segment_ids_index_map(
            batch_index,
            head_index,
            q_seq_index,
            kv_seq_index,
            *scalar_prefetch,
        ):
            del head_index
            if use_block_sparse_segments:
                _, prefetch_k_ref = scalar_prefetch
                next_kv_index = prefetch_k_ref[batch_index, q_seq_index, kv_seq_index]
            elif max_segment_len is not None:
                next_kv_index = safe_local_kv_index(q_seq_index, kv_seq_index)
            elif causal:
                next_kv_index = lax.select(
                    below_or_on_diag(q_seq_index, block_q, kv_seq_index, block_k_major),
                    kv_seq_index,
                    0,
                )
            else:
                next_kv_index = kv_seq_index
            return (batch_index, 0, next_kv_index)

        q_segment_ids_spec = pl.BlockSpec((block_b, block_q, NUM_LANES), q_segment_ids_index_map)
        kv_segment_ids_spec = pl.BlockSpec(
            (block_b, NUM_SUBLANES, block_k_major), kv_segment_ids_index_map
        )

        q_segment_ids = jax.lax.broadcast_in_dim(
            segment_ids.q,
            (batch_size, q_seq_len, NUM_LANES),
            (
                0,
                1,
            ),
        )
        kv_segment_ids = jax.lax.broadcast_in_dim(
            segment_ids.kv,
            (batch_size, NUM_SUBLANES, kv_seq_len),
            (
                0,
                2,
            ),
        )

    in_specs = [
        pl.BlockSpec((block_b, 1, block_q, head_dim), q_index_map),
        pl.BlockSpec((block_b, 1, block_k_major, head_dim), kv_index_map),
        pl.BlockSpec((block_b, 1, block_k_major, head_dim), kv_index_map),
        ab_block_spec,
        q_segment_ids_spec,
        kv_segment_ids_spec,
    ]

    scalar_prefetch = (block_mask, prefetch_k) if use_block_sparse_segments else ()
    o, *aux = pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=len(scalar_prefetch),
            grid=grid,
            in_specs=in_specs,
            out_specs=out_specs,
            scratch_shapes=scratch_shapes,
        ),
        out_shape=out_shape,
        debug=debug,
        interpret=interpret,
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=(
                "parallel",
                "parallel",
                "parallel",
                "arbitrary",
            ),
            vmem_limit_bytes=vmem_limit_bytes,
        ),
        cost_estimate=_fwd_cost_estimate(
            q,
            k,
            v,
            ab,
            segment_ids,
            causal=causal,
            sm_scale=sm_scale,
            kernel_inputs_specs=(q, k, v, ab, q_segment_ids, kv_segment_ids),
            kernel_outputs_specs=out_shape,
        ),
    )(*scalar_prefetch, q, k, v, ab, q_segment_ids, kv_segment_ids)
    if save_residuals:
        l, m = (v[..., 0] for v in aux[-2:])
        return (o, l, m)
    else:
        return o


# For autograd testing.
def mha_reference_no_custom_vjp(
    q,
    k,
    v,
    ab: jax.Array | None = None,
    segment_ids: SegmentIds | None = None,
    *,
    causal: bool = False,
    mask_value: float = DEFAULT_MASK_VALUE,
    sm_scale: float = 1.0,
    save_residuals: bool = False,
):
    logits = jnp.einsum("bhqc,bhkc->bhqk", q, k)
    if ab is not None:
        logits += ab
    if sm_scale != 1.0:
        logits *= sm_scale

    mask = None
    if segment_ids is not None:
        mask = segment_ids.q[:, :, None] == segment_ids.kv[:, None, :]
        mask = mask[:, None, :, :]

    if causal:
        _, _, q_seq_len, _ = q.shape
        _, _, kv_seq_len, _ = k.shape
        mask_shape = (q_seq_len, kv_seq_len)
        row_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 0)
        col_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 1)
        causal_mask = (col_ids <= row_ids)[None, None, :, :]
        mask = causal_mask if mask is None else jnp.logical_and(mask, causal_mask)

    logits = logits if mask is None else logits + jnp.where(mask, 0.0, mask_value)

    m = logits.max(axis=-1)
    unnormalized = jnp.exp(logits - m[..., None])
    l = unnormalized.sum(axis=-1)
    weights = unnormalized / l[..., None]
    out = jnp.einsum("bhqk,bhkc->bhqc", weights, v)
    if save_residuals:
        return out, l, m
    return out


@functools.partial(jax.jit, static_argnames=["causal", "mask_value", "sm_scale"])
@jax.default_matmul_precision("bfloat16")
def mha_reference(
    q,
    k,
    v,
    ab,
    segment_ids: SegmentIds | None = None,
    causal: bool = False,
    mask_value: float = DEFAULT_MASK_VALUE,
    sm_scale=1.0,
):
    return _mha_reference(
        q,
        k,
        v,
        ab,
        segment_ids,
        causal=causal,
        mask_value=mask_value,
        sm_scale=sm_scale,
        save_residuals=False,
    )


def _mha_reference(
    q,
    k,
    v,
    ab,
    segment_ids: SegmentIds | None,
    causal: bool,
    mask_value: float,
    sm_scale: float,
    save_residuals: bool,
):
    return mha_reference_no_custom_vjp(
        q,
        k,
        v,
        ab,
        segment_ids,
        causal=causal,
        mask_value=mask_value,
        sm_scale=sm_scale,
        save_residuals=save_residuals,
    )


def _mha_reference_fwd(
    q,
    k,
    v,
    ab,
    segment_ids: SegmentIds | None,
    causal: bool,
    mask_value: float,
    sm_scale: float,
    save_residuals: bool,
):
    if save_residuals:
        raise NotImplementedError
    res = _mha_reference(
        q,
        k,
        v,
        ab,
        segment_ids,
        causal=causal,
        mask_value=mask_value,
        sm_scale=sm_scale,
        save_residuals=True,
    )
    assert isinstance(res, tuple)
    out, l, m = res
    return out, (q, k, v, ab, segment_ids, out, l, m)


def _verify_block(block_name, dim_name, block, dim, should_divide=True):
    if block > dim:
        raise ValueError(f"{block_name}={block} should be smaller or equal to {dim_name}={dim}")
    if should_divide and dim % block != 0:
        raise ValueError(f"{dim_name}={dim} should be divisible by {block_name}={block}")
