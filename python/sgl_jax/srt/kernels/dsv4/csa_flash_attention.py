"""Fused masked attention for the DSv4 CSA prefill path (Pallas TPU).

The reference (`layers/attention/dsv4/attention.py::dsv4_dense_attention`) scores every
query against every gathered key (`[T, H, N]` f32), masks, takes a softmax with the
attention sink and multiplies back: three full passes over a `[T, H, N]` tensor
per layer. This kernel streams key tiles past query blocks with an online softmax,
never materialising the scores, and skips key tiles that no query of the block
may attend (the sliding-window band and unselected history).

Semantics, per query ``t`` and head ``h`` (``m`` = admissible mask, ``s`` = sink):

    p_e = exp(scale * q.k_e - M) * m[t, e]      M = max(max_e scale*q.k_e, s_h)
    out = sum_e p_e k_e / (sum_e p_e + exp(s_h - M))

which is the reference with the sink folded in as a pseudo-key without a value.
The running maximum is initialised to the sink, so nothing ever underflows to a
NaN and a query that may attend nothing returns exactly zero.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def _kernel(
    q_ref,
    k_ref,
    mask_ref,
    sink_ref,
    out_ref,
    m_sc,
    l_sc,
    acc_sc,
    *,
    heads,
    scale,
):
    j = pl.program_id(1)
    tq_h, tk = q_ref.shape[0], k_ref.shape[0]
    tq = tq_h // heads

    @pl.when(j == 0)
    def _init():
        m_sc[...] = sink_ref[...]  # [tq*H, 1]: the sink is the first "key"
        l_sc[...] = jnp.ones_like(l_sc)
        acc_sc[...] = jnp.zeros_like(acc_sc)

    mask = mask_ref[...] != 0  # [tq, tk]

    @pl.when(jnp.any(mask))
    def _tile():
        q = q_ref[...]
        k = k_ref[...]
        s = jax.lax.dot_general(
            q, k, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32
        )  # [tq*H, tk]
        s = s * scale
        s3 = s.reshape(tq, heads, tk)
        s3 = jnp.where(mask[:, None, :], s3, -jnp.inf)
        m_prev = m_sc[...].reshape(tq, heads, 1)
        m_new = jnp.maximum(m_prev, jnp.max(s3, axis=-1, keepdims=True))
        alpha = jnp.exp(m_prev - m_new)  # m_new >= sink > -inf: always finite
        p3 = jnp.exp(s3 - m_new)  # masked entries: exp(-inf) == 0
        l_new = alpha * l_sc[...].reshape(tq, heads, 1) + jnp.sum(p3, axis=-1, keepdims=True)
        # f32 probabilities against f32-upcast keys: the reference einsum promotes the
        # bf16 keys to f32 here too, so the two agree to f32 rounding.
        p = p3.reshape(tq_h, tk)
        pv = jax.lax.dot_general(
            p,
            k.astype(jnp.float32),
            (((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        )  # [tq*H, D]
        acc_sc[...] = acc_sc[...] * alpha.reshape(tq_h, 1) + pv
        m_sc[...] = m_new.reshape(tq_h, 1)
        l_sc[...] = l_new.reshape(tq_h, 1)

    @pl.when(j == pl.num_programs(1) - 1)
    def _finish():
        out_ref[...] = (acc_sc[...] / l_sc[...]).astype(out_ref.dtype)


def csa_flash_attention(
    q,
    keys,
    mask,
    sink,
    *,
    sm_scale: float,
    block_q: int = 256,
    block_k: int = 512,
    interpret: bool = False,
):
    """``q`` [T, H, D], ``keys`` [N, D], ``mask`` [T, N] (nonzero = admissible),
    ``sink`` [H] f32 -> [T, H, D] f32. ``T`` is padded to ``block_q`` and ``N`` to
    ``block_k`` internally (padded keys are masked)."""
    q = jnp.asarray(q)
    keys = jnp.asarray(keys)
    T, H, D = q.shape
    N = keys.shape[0]
    if keys.shape[1] != D or mask.shape != (T, N) or sink.shape != (H,):
        raise ValueError(
            f"shape mismatch: q{q.shape} keys{keys.shape} mask{mask.shape} sink{sink.shape}"
        )
    kdt = keys.dtype if keys.dtype in (jnp.bfloat16, jnp.float32) else jnp.bfloat16
    tq = min(block_q, -(-T // 8) * 8)
    tk = min(block_k, -(-N // 128) * 128)
    Tp = -(-T // tq) * tq
    Np = -(-N // tk) * tk
    q2 = jnp.pad(q.astype(kdt), ((0, Tp - T), (0, 0), (0, 0))).reshape(Tp * H, D)
    k2 = jnp.pad(keys.astype(kdt), ((0, Np - N), (0, 0)))
    m2 = jnp.pad(jnp.asarray(mask).astype(jnp.int8), ((0, Tp - T), (0, Np - N)))
    sink_rows = jnp.tile(jnp.asarray(sink, jnp.float32), tq)[:, None]  # [tq*H, 1]
    kernel = functools.partial(_kernel, heads=H, scale=float(sm_scale))
    out = pl.pallas_call(
        kernel,
        grid=(Tp // tq, Np // tk),
        in_specs=[
            pl.BlockSpec((tq * H, D), lambda i, j: (i, 0)),
            pl.BlockSpec((tk, D), lambda i, j: (j, 0)),
            pl.BlockSpec((tq, tk), lambda i, j: (i, j)),
            pl.BlockSpec((tq * H, 1), lambda i, j: (0, 0)),
        ],
        out_specs=pl.BlockSpec((tq * H, D), lambda i, j: (i, 0)),
        out_shape=jax.ShapeDtypeStruct((Tp * H, D), jnp.float32),
        scratch_shapes=[
            pltpu.VMEM((tq * H, 1), jnp.float32),
            pltpu.VMEM((tq * H, 1), jnp.float32),
            pltpu.VMEM((tq * H, D), jnp.float32),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary"),
            vmem_limit_bytes=64 * 1024 * 1024,
        ),
        interpret=interpret,
    )(q2, k2, m2, sink_rows)
    return out.reshape(Tp, H, D)[:T]


def _kernel_meta(
    q_ref,
    k_ref,
    qpos_ref,
    qreq_ref,
    qvalid_ref,
    kpos_ref,
    kreq_ref,
    member_ref,
    sink_ref,
    out_ref,
    m_sc,
    l_sc,
    acc_sc,
    *,
    heads,
    scale,
    window_tiles,
    window_size,
    ratio,
):
    """`_kernel` with the admissibility mask built in VMEM from per-row metadata.

    Key tiles ``j < window_tiles`` hold sliding-window rows (``kpos`` = position)
    and use the window rule; later tiles hold compressed records (``kpos`` = entry
    id) and use the completeness rule intersected with the indexer membership.
    Nothing ``[T, N]``-sized is ever written to HBM.
    """
    j = pl.program_id(1)
    tq_h, tk = q_ref.shape[0], k_ref.shape[0]
    tq = tq_h // heads

    @pl.when(j == 0)
    def _init():
        m_sc[...] = sink_ref[...]
        l_sc[...] = jnp.ones_like(l_sc)
        acc_sc[...] = jnp.zeros_like(acc_sc)

    qpos = qpos_ref[:, :1]  # [tq, 1]
    qreq = qreq_ref[:, :1]
    qvalid = qvalid_ref[:, :1] != 0
    kpos = kpos_ref[:1, :]  # [1, tk]
    kreq = kreq_ref[:1, :]
    same = qvalid & (kreq == qreq)
    window_mask = same & (kpos <= qpos) & (kpos > qpos - window_size)
    if ratio > 0:
        compressed_mask = same & (kpos < (qpos + 1) // ratio) & (member_ref[...] != 0)
    else:
        compressed_mask = jnp.zeros_like(window_mask)
    # Boolean algebra rather than a select: Mosaic cannot legalize arith.select on i1
    # vectors. ``j`` is a scalar, so this is one broadcast and two ands.
    in_window = j < window_tiles
    mask = (window_mask & in_window) | (compressed_mask & jnp.logical_not(in_window))  # [tq, tk]

    @pl.when(jnp.any(mask))
    def _tile():
        q = q_ref[...]
        k = k_ref[...]
        s = jax.lax.dot_general(q, k, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32)
        s = s * scale
        s3 = s.reshape(tq, heads, tk)
        s3 = jnp.where(mask[:, None, :], s3, -jnp.inf)
        m_prev = m_sc[...].reshape(tq, heads, 1)
        m_new = jnp.maximum(m_prev, jnp.max(s3, axis=-1, keepdims=True))
        alpha = jnp.exp(m_prev - m_new)
        p3 = jnp.exp(s3 - m_new)
        l_new = alpha * l_sc[...].reshape(tq, heads, 1) + jnp.sum(p3, axis=-1, keepdims=True)
        p = p3.reshape(tq_h, tk)
        pv = jax.lax.dot_general(
            p,
            k.astype(jnp.float32),
            (((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
        acc_sc[...] = acc_sc[...] * alpha.reshape(tq_h, 1) + pv
        m_sc[...] = m_new.reshape(tq_h, 1)
        l_sc[...] = l_new.reshape(tq_h, 1)

    @pl.when(j == pl.num_programs(1) - 1)
    def _finish():
        out_ref[...] = (acc_sc[...] / l_sc[...]).astype(out_ref.dtype)


def _lane_rows(vector, rows, fill):
    """``[n]`` int32 -> ``[rows, 128]`` with the value replicated across lanes."""
    v = jnp.pad(jnp.asarray(vector, jnp.int32), (0, rows - vector.shape[0]), constant_values=fill)
    return jnp.broadcast_to(v[:, None], (rows, 128))


def _sublane_rows(vector, cols, fill):
    """``[n]`` int32 -> ``[8, cols]`` with the value replicated across sublanes."""
    v = jnp.pad(jnp.asarray(vector, jnp.int32), (0, cols - vector.shape[0]), constant_values=fill)
    return jnp.broadcast_to(v[None, :], (8, cols))


def csa_flash_attention_meta(
    q,
    window_keys,
    compressed_keys,
    sink,
    *,
    query_positions,
    query_request_ids,
    valid_token_mask,
    window_positions,
    window_request_ids,
    compressed_entry_ids,
    compressed_request_ids,
    membership=None,
    sm_scale: float,
    window_size: int,
    ratio: int,
    block_q: int = 256,
    block_k: int = 512,
    interpret: bool = False,
):
    """`csa_flash_attention` whose mask is built inside the kernel.

    ``window_keys`` [W, D], ``compressed_keys`` [E, D]; ``membership`` optional
    ``[T, E]`` (nonzero = the indexer selected the record). Same semantics as
    `attention.admissible_mask` followed by `csa_flash_attention`.
    """
    q = jnp.asarray(q)
    T, H, D = q.shape
    W, E = window_keys.shape[0], compressed_keys.shape[0]
    kdt = window_keys.dtype if window_keys.dtype in (jnp.bfloat16, jnp.float32) else jnp.bfloat16
    tq = min(block_q, -(-T // 8) * 8)
    tk = block_k
    Tp = -(-T // tq) * tq
    Wp = -(-W // tk) * tk if W else 0
    Ep = max(tk, -(-E // tk) * tk)
    window_tiles = Wp // tk
    q2 = jnp.pad(q.astype(kdt), ((0, Tp - T), (0, 0), (0, 0))).reshape(Tp * H, D)
    k2 = jnp.concatenate(
        (
            jnp.pad(jnp.asarray(window_keys, kdt), ((0, Wp - W), (0, 0))),
            jnp.pad(jnp.asarray(compressed_keys, kdt), ((0, Ep - E), (0, 0))),
        ),
        axis=0,
    )
    big = jnp.int32(2**30)
    qpos = _lane_rows(query_positions, Tp, -1)
    qreq = _lane_rows(query_request_ids, Tp, -1)
    qvalid = _lane_rows(jnp.asarray(valid_token_mask).astype(jnp.int32), Tp, 0)
    # Padded window rows get an unreachable position, padded records an entry id no
    # query completes, and every pad a request id no query carries.
    kpos = jnp.concatenate(
        (
            jnp.pad(jnp.asarray(window_positions, jnp.int32), (0, Wp - W), constant_values=-big),
            jnp.pad(jnp.asarray(compressed_entry_ids, jnp.int32), (0, Ep - E), constant_values=big),
        )
    )
    kreq = jnp.concatenate(
        (
            jnp.pad(jnp.asarray(window_request_ids, jnp.int32), (0, Wp - W), constant_values=-1),
            jnp.pad(
                jnp.asarray(compressed_request_ids, jnp.int32), (0, Ep - E), constant_values=-1
            ),
        )
    )
    kpos = jnp.broadcast_to(kpos[None, :], (8, Wp + Ep))
    kreq = jnp.broadcast_to(kreq[None, :], (8, Wp + Ep))
    if membership is None:
        member = jnp.ones((Tp, Ep), jnp.int8)
    else:
        member = jnp.pad(jnp.asarray(membership).astype(jnp.int8), ((0, Tp - T), (0, Ep - E)))
    sink_rows = jnp.tile(jnp.asarray(sink, jnp.float32), tq)[:, None]
    kernel = functools.partial(
        _kernel_meta,
        heads=H,
        scale=float(sm_scale),
        window_tiles=window_tiles,
        window_size=int(window_size),
        ratio=int(ratio),
    )

    def member_map(i, j):
        return (i, jnp.maximum(j - window_tiles, 0))

    out = pl.pallas_call(
        kernel,
        grid=(Tp // tq, (Wp + Ep) // tk),
        in_specs=[
            pl.BlockSpec((tq * H, D), lambda i, j: (i, 0)),
            pl.BlockSpec((tk, D), lambda i, j: (j, 0)),
            pl.BlockSpec((tq, 128), lambda i, j: (i, 0)),
            pl.BlockSpec((tq, 128), lambda i, j: (i, 0)),
            pl.BlockSpec((tq, 128), lambda i, j: (i, 0)),
            pl.BlockSpec((8, tk), lambda i, j: (0, j)),
            pl.BlockSpec((8, tk), lambda i, j: (0, j)),
            pl.BlockSpec((tq, tk), member_map),
            pl.BlockSpec((tq * H, 1), lambda i, j: (0, 0)),
        ],
        out_specs=pl.BlockSpec((tq * H, D), lambda i, j: (i, 0)),
        out_shape=jax.ShapeDtypeStruct((Tp * H, D), jnp.float32),
        scratch_shapes=[
            pltpu.VMEM((tq * H, 1), jnp.float32),
            pltpu.VMEM((tq * H, 1), jnp.float32),
            pltpu.VMEM((tq * H, D), jnp.float32),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary"),
            vmem_limit_bytes=64 * 1024 * 1024,
        ),
        interpret=interpret,
        name=f"csa-flash-meta-h{H}-d{D}-q{tq}-k{tk}",
    )(q2, k2, qpos, qreq, qvalid, kpos, kreq, member, sink_rows)
    return out.reshape(Tp, H, D)[:T]
