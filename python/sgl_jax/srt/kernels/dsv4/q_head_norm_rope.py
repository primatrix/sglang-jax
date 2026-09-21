"""Fused per-head q normalisation + interleaved partial RoPE (Pallas TPU).

V4 projects ``q`` with ``wq_b`` to ``[T, H*D]``, normalises each head (no learned
weight), and rotates the trailing ``rope_head_dim`` features of every head. In
XLA that is a reshape to ``[T, H, D]`` (a relayout of the projection's tiles),
an f32 materialisation for the norm, and a HIGHEST-precision ``[D, D]``
permutation matmul for the rope partner: three passes over ``q`` per layer.
This kernel reads the ``[T, H*D]`` bf16 projection once, keeps every head in a
lane-aligned slice, rotates the trailing 128-lane tile with in-register lane
rolls, and writes ``[T*H, D]`` bf16 (row ``t*H + h``) directly.

The rope math is the model's: ``x * cos_full + partner * sin_full`` in f32
with ``partner[2i] = -x[2i+1]``, ``partner[2i+1] = x[2i]`` inside the trailing
block; ``cos_full``/``sin_full`` are built on the host by the same exact
one-hot spread as `interleaved_rope`. The head norm sums squares in a different
order than XLA's reduce, so outputs can differ at rounding level.
"""

from __future__ import annotations

import functools
import os

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

_LANE = 128


def kernel_enabled() -> bool:
    """``DSV4_Q_NORM_ROPE_KERNEL=1`` routes V4's q head-norm + rope through this kernel."""
    return os.environ.get("DSV4_Q_NORM_ROPE_KERNEL", "0") == "1"


def _default_interpret() -> bool:
    return os.environ.get("PALLAS_INTERPRET", "0") == "1" or jax.default_backend() == "cpu"


@functools.cache
def _tail_constants(rope_head_dim: int):
    """``spread [R/2, 128]`` and ``head_ones [128]`` for the trailing 128-lane tile."""
    start = _LANE - rope_head_dim
    half = rope_head_dim // 2
    spread = np.zeros((half, _LANE), np.float32)
    for i in range(half):
        spread[i, start + 2 * i] = 1.0
        spread[i, start + 2 * i + 1] = 1.0
    head_ones = np.zeros((_LANE,), np.float32)
    head_ones[:start] = 1.0
    return spread, head_ones


def rope_tail_tables(cos, sin, rope_head_dim: int):
    """Exact ``[T, 128]`` f32 cos/sin over the trailing lane tile of a head.

    ``cos``/``sin`` are ``[T, rope_head_dim // 2]``; each pair's value lands on
    both of its lanes (one nonzero term per output, so the HIGHEST matmul is exact)
    and non-rope lanes get cos 1 / sin 0, which makes the rotation the identity there.
    """
    spread, head_ones = _tail_constants(rope_head_dim)
    hi = jax.lax.Precision.HIGHEST
    cos = jnp.asarray(cos, jnp.float32)
    sin = jnp.asarray(sin, jnp.float32)
    cos_full = jnp.dot(cos, jnp.asarray(spread), precision=hi) + jnp.asarray(head_ones)
    sin_full = jnp.dot(sin, jnp.asarray(spread), precision=hi)
    return cos_full, sin_full


def _kernel(q_ref, cos_ref, sin_ref, out_ref, acc_ref, *, heads, head_dim, normalize, eps, rows):
    cos_full = cos_ref[...]
    sin_full = sin_ref[...]
    lane = jax.lax.broadcasted_iota(jnp.int32, (rows, _LANE), 1)
    even = (lane % 2) == 0
    # Lane parity as arithmetic masks: under explicit-sharding meshes a select whose
    # predicate is an unsharded iota and whose cases carry the block's sharding is
    # rejected (ShardingTypeError); x * 1 + y * 0 is exact for finite x, y.
    on_even = jnp.where(even, 1.0, 0.0).astype(jnp.float32)
    on_odd = 1.0 - on_even
    keep = head_dim - _LANE
    for h in range(heads):
        xh = q_ref[:, h * head_dim : (h + 1) * head_dim].astype(jnp.float32)
        if normalize:
            sumsq = jnp.sum(xh * xh, axis=1, keepdims=True)
            scale = jax.lax.rsqrt(sumsq * (1.0 / head_dim) + eps)
            # The model rounds the normalised q to bf16 before the rope.
            xh = (xh * scale).astype(jnp.bfloat16).astype(jnp.float32)
        tail = xh[:, keep:]
        # partner[2i] = -x[2i+1] (the lane to the right), partner[2i+1] = x[2i] (left).
        right = pltpu.roll(tail, shift=_LANE - 1, axis=1)
        left = pltpu.roll(tail, shift=1, axis=1)
        partner = left * on_odd - right * on_even
        tail = tail * cos_full + partner * sin_full
        head_out = jnp.concatenate([xh[:, :keep], tail], axis=1) if keep else tail
        # Mosaic's strided (sublane-interleaving) store wants a 128-lane base ref:
        # scatter each 128-lane tile of the head's rows into its own scratch plane.
        for t in range(head_dim // _LANE):
            acc_ref[t, pl.ds(h, rows, stride=heads), :] = head_out[:, t * _LANE : (t + 1) * _LANE]
    out_ref[...] = jnp.concatenate([acc_ref[t] for t in range(head_dim // _LANE)], axis=1).astype(
        out_ref.dtype
    )


def q_head_norm_rope(
    q,
    cos,
    sin,
    *,
    heads: int,
    head_dim: int,
    rope_head_dim: int,
    normalize: bool,
    eps: float = 1e-6,
    rows_per_step: int | None = None,
    out_dtype=jnp.bfloat16,
    interpret: bool | None = None,
):
    """``[T, heads*head_dim]`` -> ``[T*heads, head_dim]`` (row ``t*heads + h``)."""
    if interpret is None:
        interpret = _default_interpret()
    tokens, width = q.shape
    if width != heads * head_dim or head_dim % _LANE:
        raise ValueError("q must be [T, heads*head_dim] with a 128-aligned head_dim")
    if rope_head_dim % 2 or rope_head_dim > _LANE:
        raise ValueError("rope_head_dim must be even and fit the trailing 128-lane tile")
    if cos.shape != (tokens, rope_head_dim // 2) or sin.shape != cos.shape:
        raise ValueError("cos/sin must be [T, rope_head_dim // 2]")
    rows = rows_per_step or int(os.environ.get("DSV4_Q_NORM_ROPE_ROWS", "64"))  # v7x: 64 > 32 > 16
    if rows % 16:
        raise ValueError("rows_per_step must be a multiple of 16")
    padded = -(-tokens // rows) * rows
    cos_full, sin_full = rope_tail_tables(cos, sin, rope_head_dim)
    if padded != tokens:
        pad = (0, padded - tokens)
        q = jnp.pad(q, (pad, (0, 0)))
        cos_full = jnp.pad(cos_full, (pad, (0, 0)), constant_values=1.0)
        sin_full = jnp.pad(sin_full, (pad, (0, 0)))
    out = pl.pallas_call(
        functools.partial(
            _kernel, heads=heads, head_dim=head_dim, normalize=normalize, eps=float(eps), rows=rows
        ),
        grid=(padded // rows,),
        in_specs=[
            pl.BlockSpec((rows, width), lambda i: (i, 0)),
            pl.BlockSpec((rows, _LANE), lambda i: (i, 0)),
            pl.BlockSpec((rows, _LANE), lambda i: (i, 0)),
        ],
        out_specs=pl.BlockSpec((rows * heads, head_dim), lambda i: (i, 0)),
        out_shape=jax.ShapeDtypeStruct((padded * heads, head_dim), out_dtype),
        scratch_shapes=[pltpu.VMEM((head_dim // _LANE, rows * heads, _LANE), jnp.float32)],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",), vmem_limit_bytes=32 * 1024 * 1024
        ),
        interpret=interpret,
        name=f"dsv4-q-{'norm-' if normalize else ''}rope-h{heads}-d{head_dim}-r{rows}",
    )(q, cos_full, sin_full)
    return out[: tokens * heads]
