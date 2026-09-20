# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""mHC fused seam: sublayer ``post`` fused with the next sublayer's ``pre`` (Pallas TPU).

Ported from tpu-inference ``kernels/experimental/deepseek_v4/mhc/fused_post_pre_kernel.py``.
Between two sublayers the unfused path stores the recombined streams, reloads them
for the next ``pre``'s mix projection, and collapses them again. The seam does the
recombine, the mix GEMM (3-pass bf16 split of the f32 ``fn`` == f32 "highest"), the
RMS-scaled pre gate and the collapse in one token-block pass, storing the streams
once. The Sinkhorn gates for the *next* post are an XLA epilogue on the mixes.

Parity with ``post_reference`` followed by ``pre_reference``: the streams are rounded
to bf16 before the GEMM exactly where the unfused path stores them.
"""

from __future__ import annotations

import functools
import os

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

_SUBLANE = 16  # bf16 tiles are (16, 128)


def _round_up(x: int, multiple: int) -> int:
    return (x + multiple - 1) // multiple * multiple


def _fused_kernel(
    x_ref,
    res_ref,
    post_ref,
    comb_ref,
    fn_hi_ref,
    fn_mid_ref,
    fn_lo_ref,
    sc_ref,
    hb_ref,
    newres_ref,
    mixes_ref,
    sqrsum_ref,
    layer_ref,
    *,
    hc_mult,
    hidden_size,
    rms_eps,
    hc_eps,
    interpret,
):
    post = post_ref[...]  # (tb, hc) f32
    comb = comb_ref[...]  # (tb, hc*hc) f32, row-major (i, j)
    x_bf = x_ref[...]  # (tb, hidden) bf16
    old_bf = [res_ref[:, i * hidden_size : (i + 1) * hidden_size] for i in range(hc_mult)]

    streams_bf = []
    for j in range(hc_mult):
        acc = post[:, j : j + 1] * x_bf.astype(jnp.float32)
        for i in range(hc_mult):
            k = i * hc_mult + j
            acc = acc + comb[:, k : k + 1] * old_bf[i].astype(jnp.float32)
        new_bf = acc.astype(jnp.bfloat16)  # the unfused path stores bf16 here
        newres_ref[:, j * hidden_size : (j + 1) * hidden_size] = new_bf
        streams_bf.append(new_bf)

    g_bf = jnp.concatenate(streams_bf, axis=1)  # (tb, hc*hidden)
    dn = (((1,), (1,)), ((), ()))
    mm = jnp.float32 if interpret else None  # XLA:CPU has no bf16 dots (tests only)
    lhs = g_bf.astype(mm) if mm else g_bf
    acc = jax.lax.dot_general(
        lhs,
        fn_hi_ref[...].astype(mm) if mm else fn_hi_ref[...],
        dn,
        preferred_element_type=jnp.float32,
    )
    acc = acc + jax.lax.dot_general(
        lhs,
        fn_mid_ref[...].astype(mm) if mm else fn_mid_ref[...],
        dn,
        preferred_element_type=jnp.float32,
    )
    acc = acc + jax.lax.dot_general(
        lhs,
        fn_lo_ref[...].astype(mm) if mm else fn_lo_ref[...],
        dn,
        preferred_element_type=jnp.float32,
    )
    mixes_ref[...] = acc
    gf = g_bf.astype(jnp.float32)
    sqr = jnp.sum(gf * gf, axis=-1, keepdims=True)
    sqrsum_ref[...] = sqr

    m, h = hc_mult, hidden_size
    scaled = acc[:, :m] * jax.lax.rsqrt(sqr / (m * h) + rms_eps)
    pre_gate = jax.nn.sigmoid(scaled * sc_ref[0, 0] + hb_ref[:, :m]) + hc_eps
    lay = pre_gate[:, 0:1] * gf[:, :h]
    for i in range(1, m):
        lay = lay + pre_gate[:, i : i + 1] * gf[:, i * h : (i + 1) * h]
    layer_ref[...] = lay.astype(layer_ref.dtype)


def mhc_seam_fused(
    y,
    residual_streams,
    post_gate,
    comb,
    fn_next,
    scale_next,
    base_next,
    *,
    hc_mult: int,
    sinkhorn_iters: int,
    norm_eps: float,
    hc_eps: float,
    token_block_size: int | None = None,
    interpret: bool = False,
):
    """``post(y, streams, post_gate, comb)`` then ``pre(new_streams, fn_next, ...)``.

    Returns ``(new_streams [..., hc, d] bf16, layer_input [..., d] bf16,
    post_gate_next [..., hc], comb_next [..., hc, hc])``.
    """
    residual_streams = jnp.asarray(residual_streams)
    outer = residual_streams.shape[:-2]
    hc, hidden = residual_streams.shape[-2:]
    if hc != hc_mult:
        raise ValueError(f"streams have hc={hc}, expected {hc_mult}")
    x2d = jnp.asarray(y).reshape(-1, hidden).astype(jnp.bfloat16)
    res2d = residual_streams.reshape(-1, hc * hidden).astype(jnp.bfloat16)
    post2d = jnp.asarray(post_gate, jnp.float32).reshape(-1, hc)
    comb2d = jnp.asarray(comb, jnp.float32).reshape(-1, hc * hc)
    fn = jnp.asarray(fn_next, jnp.float32)
    hc3 = fn.shape[0]
    n = x2d.shape[0]
    if token_block_size is None:
        token_block_size = int(os.environ.get("DSV4_MHC_SEAM_BLOCK", "64"))

    # Whole-extent blocks for small token counts (decode buckets): no pad/slice ops.
    tb = n if n <= token_block_size else token_block_size
    n_pad = _round_up(n, tb)
    pad = n_pad - n
    if pad:
        x2d, res2d, post2d, comb2d = (
            jnp.pad(a, ((0, pad), (0, 0))) for a in (x2d, res2d, post2d, comb2d)
        )
    # 3-chunk bf16 split of fn == f32 "highest"; reduce_precision, not dtype
    # round-trips, which XLA would fold away and zero the mid/lo chunks.
    fn_hi = jax.lax.reduce_precision(fn, 8, 7)
    rem = fn - fn_hi
    fn_mid = jax.lax.reduce_precision(rem, 8, 7)
    fn_lo = jax.lax.reduce_precision(rem - fn_mid, 8, 7)
    fn_hi, fn_mid, fn_lo = (t.astype(jnp.bfloat16) for t in (fn_hi, fn_mid, fn_lo))
    sc2d = jnp.asarray(scale_next, jnp.float32).reshape(1, 3)
    hb2d = jnp.asarray(base_next, jnp.float32).reshape(1, hc3)

    resident = pl.BlockSpec((hc3, hc * hidden), lambda i: (0, 0))
    new_res, mixes, sqrsum, layer2d = pl.pallas_call(
        functools.partial(
            _fused_kernel,
            hc_mult=hc,
            hidden_size=hidden,
            rms_eps=float(norm_eps),
            hc_eps=float(hc_eps),
            interpret=interpret,
        ),
        grid=(n_pad // tb,),
        in_specs=[
            pl.BlockSpec((tb, hidden), lambda i: (i, 0)),
            pl.BlockSpec((tb, hc * hidden), lambda i: (i, 0)),
            pl.BlockSpec((tb, hc), lambda i: (i, 0)),
            pl.BlockSpec((tb, hc * hc), lambda i: (i, 0)),
            resident,
            resident,
            resident,
            pl.BlockSpec((1, 3), lambda i: (0, 0)),
            pl.BlockSpec((1, hc3), lambda i: (0, 0)),
        ],
        out_specs=[
            pl.BlockSpec((tb, hc * hidden), lambda i: (i, 0)),
            pl.BlockSpec((tb, hc3), lambda i: (i, 0)),
            pl.BlockSpec((tb, 1), lambda i: (i, 0)),
            pl.BlockSpec((tb, hidden), lambda i: (i, 0)),
        ],
        out_shape=[
            jax.ShapeDtypeStruct((n_pad, hc * hidden), jnp.bfloat16),
            jax.ShapeDtypeStruct((n_pad, hc3), jnp.float32),
            jax.ShapeDtypeStruct((n_pad, 1), jnp.float32),
            jax.ShapeDtypeStruct((n_pad, hidden), jnp.bfloat16),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",), vmem_limit_bytes=64 * 1024 * 1024
        ),
        interpret=interpret,
    )(x2d, res2d, post2d, comb2d, fn_hi, fn_mid, fn_lo, sc2d, hb2d)
    new_res, mixes, sqrsum, layer2d = (a[:n] for a in (new_res, mixes, sqrsum, layer2d))

    # Gates for the next post (the pre gate was consumed in-kernel). Same rule as
    # pre: the RMS scale multiplies the projection. On TPU the Sinkhorn runs in the
    # existing gates kernel (in XLA it was ~85 ops per seam); interpret mode (CPU
    # tests) has no device schedule and uses the reference.
    scaled = mixes * jax.lax.rsqrt(sqrsum / (hc * hidden) + norm_eps)
    if interpret:
        from sgl_jax.srt.layers.deepseek_v4_mhc import mhc_gates_reference

        _, post_next, comb_next = mhc_gates_reference(
            scaled, scale_next, base_next, hc_mult=hc, sinkhorn_iters=sinkhorn_iters, eps=hc_eps
        )
    else:
        from sgl_jax.srt.kernels.mhc.mhc import mhc_gates

        post_next, comb_next = mhc_gates(
            scaled,
            jnp.asarray(scale_next, jnp.float32),
            jnp.asarray(base_next, jnp.float32),
            hc_mult=hc,
            sinkhorn_iters=sinkhorn_iters,
            eps=hc_eps,
        )
    return (
        new_res.reshape(*outer, hc, hidden),
        layer2d.reshape(*outer, hidden),
        post_next.reshape(*outer, hc),
        comb_next.reshape(*outer, hc, hc),
    )
