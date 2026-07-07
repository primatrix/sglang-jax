from __future__ import annotations

import functools
import logging

import jax
import jax.experimental.pallas as pl
import jax.numpy as jnp

# Wrapper helpers reused from v1 (VMEM cap, 128-aligned divisor, interpret toggle).
from sgl_jax.srt.kernels.grouped_topk.v1.kernel import (
    SAFE_AUTO_BT,
    _largest_safe_divisor,
    get_interpret,
)

logger = logging.getLogger(__name__)

NEG_INF = -jnp.inf

def _grouped_topk_kernel_v2(
    logits_ref,  # [BT, E] f32  (router_logits, PRE-bias) — loaded token-major like v1
    bias_ref,  # [E]     f32  (correction_bias)
    w_ref,  # [topk, BT] f32  out: weights (topk in sublane, BT in lane)
    ids_ref,  # [topk, BT] i32  out: expert ids
    *,
    n_group: int,
    topk_group: int,
    topk: int,
    num_experts: int,
    unroll_factor: int,
):
    S = num_experts // n_group
    E = num_experts

    # Transpose the loaded [BT, E] block to [E, BT] so tokens live in the lane/minor dim and every
    # selection reduction below runs over the sublane/major (expert) axis == axis 0.
    logits = logits_ref[...].astype(jnp.float32).T  # [E, BT] pre-bias
    bt = logits.shape[1]
    with jax.named_scope("bias_add"):
        scores = logits + bias_ref[...][:, None]  # [E, BT] post-bias
    
    with jax.named_scope("group_top2"):
        scores_ = jnp.reshape(scores, (n_group, S, bt))
        v1 = jnp.max(scores_, axis=1, keepdims=True) # [n_group, 1, BT]
        i1 = jnp.argmax(scores_, axis=1 , keepdims=True) # [n_group, 1, BT]
        s_iota = jax.lax.broadcasted_iota(jnp.int32, (n_group, S, bt), 1)
        scores_masked = jnp.where(s_iota == i1, NEG_INF, scores_)
        v2 = jnp.max(scores_masked, axis=1, keepdims=True) # [n_group, 1, BT]
        group_scores = jnp.squeeze(v1 + v2, axis=1) # [n_group, BT]

    with jax.named_scope("group_select"):
        group_mask = jnp.zeros((n_group, bt), dtype=jnp.bool_)
        g_iota = jax.lax.broadcasted_iota(jnp.int32, (n_group, bt), 0)
        tmp = group_scores
        for _ in range(topk_group):
            gmax = jnp.max(tmp, axis=0, keepdims=True) # [1, BT]
            gi = jnp.min(jnp.where(tmp == gmax, g_iota, n_group), axis=0, keepdims=True) # [1, BT]
            m = g_iota == gi
            group_mask = jnp.logical_or(group_mask, m)
            tmp = jnp.where(m, NEG_INF, tmp)

    # Apply the group mask ONCE here (dropped groups -> -inf), not inside the pick loop: the mask is
    # loop-invariant, so masking each of the `topk` iterations repeats this reshape+where 8x for no
    # reason. Do it once and let the fori_loop carry the already-masked working array.
    with jax.named_scope("expert_mask"):
        scores_grouped = jnp.reshape(scores, (n_group, S, bt))
        masked = jnp.reshape(
            jnp.where(group_mask[:, None, :], scores_grouped, NEG_INF), (E, bt)
        )  # [E, BT]

    with jax.named_scope("final_select"):
        e_iota = jax.lax.broadcasted_iota(jnp.int32, (E,bt), 0)  # expert index along axis 0
        row_iota = jax.lax.broadcasted_iota(jnp.int32, (topk, bt), 0)  # output-row index
        # buffers are [topk, BT]: topk is the SUBLANE (2nd-minor) dim here, so it only needs 8-align
        # (topk=8 fits exactly) — no 128 padding, unlike v1 where topk is the lane dim.
        ids_init = jnp.full((topk, bt), -1, dtype=jnp.int32) # [topk, bt]
        w_init = jnp.zeros((topk, bt), dtype=jnp.float32) # [topk, bt]

        def _argmax_reducer(a, b):
            # keep the higher post-bias score; tie -> lower expert index (matches jax.lax.top_k)
            sa, wa, ia = a
            sb, wb, ib = b
            take_a = jnp.logical_or(sa > sb, jnp.logical_and(sa == sb, ia < ib))
            return (
                jnp.where(take_a, sa, sb),
                jnp.where(take_a, wa, wb),
                jnp.where(take_a, ia, ib),
            )

        def _pick(k, carry):
            cur, ids_buf, w_buf = carry
            # ONE variadic reduction over (post-bias score, PRE-bias logit, expert index): returns
            # the winner's index AND its weight together, with exact lowest-index tie-break. Replaces
            # the previous 2 reductions/pick (argmax + masked weight-sum). No gather (unsupported).
            _, wval, idx = jax.lax.reduce(
                (cur, logits, e_iota),
                (NEG_INF, jnp.float32(0.0), jnp.int32(E)),
                _argmax_reducer,
                dimensions=(0,),
            )  # each [BT]
            idx = idx[None, :]  # [1, BT]
            write = row_iota == k  # [topk, BT] one-hot on row k (loop index)
            ids_buf = jnp.where(write, idx, ids_buf)
            w_buf = jnp.where(write, wval[None, :], w_buf)
            cur = jnp.where(e_iota == idx, NEG_INF, cur)  # drop the winner before the next pick
            return cur, ids_buf, w_buf
        _, ids_out, w_out = jax.lax.fori_loop(
            0, topk, _pick, (masked, ids_init, w_init), unroll=unroll_factor
        )

    ids_ref[...] = ids_out  # [topk, BT] — written BS-in-lane; wrapper .T is a free bitcast
    w_ref[...] = w_out  # [topk, BT]


def grouped_topk_pallas_v3(
    router_logits: jax.Array,  # [BS, E]
    correction_bias: jax.Array,  # [E]
    *,
    num_expert_group: int,
    topk_group: int,
    topk: int,
    block_tokens: int | str = "auto",
    unroll: int | None = None,
    interpret: bool | None = None,
):
    """Wrapper for kernel3 (`_grouped_topk_kernel_v2`): token-in-lane [E,BT], vectorized group top-2,
    group mask applied ONCE before the pick loop, final-select buffers kept [topk,BT] (topk in the
    sublane dim, only 8-aligned — no 128 pad). Signature-compatible with v1/v2. Outputs [BS, topk]
    with topk as the minor dim; that HBM array is still physically lane-padded to 128, which is
    unavoidable without changing the [BS, topk] output contract (i.e. touching gate.py)."""
    bs, e = router_logits.shape
    router_logits = router_logits.astype(jnp.float32)
    bias = correction_bias.astype(jnp.float32)

    if block_tokens == "auto":
        bt = _largest_safe_divisor(bs, cap=SAFE_AUTO_BT, align=128) or bs
    else:
        bt = min(block_tokens, bs)
        if bs % bt != 0:
            raise ValueError(f"BS={bs} must be divisible by block_tokens={bt}")
    if interpret is None:
        interpret = get_interpret()

    unroll_factor = max(1, min(int(unroll if unroll is not None else topk), topk))

    kernel = functools.partial(
        _grouped_topk_kernel_v2,
        n_group=num_expert_group,
        topk_group=topk_group,
        topk=topk,
        num_experts=e,
        unroll_factor=unroll_factor,
    )
    weights_t, ids_t = pl.pallas_call(
        kernel,
        grid=(bs // bt,),
        in_specs=[
            pl.BlockSpec((bt, e), lambda i: (i, 0)),
            pl.BlockSpec((e,), lambda i: (0,)),
        ],
        out_specs=[
            pl.BlockSpec((topk, bt), lambda i: (0, i)),
            pl.BlockSpec((topk, bt), lambda i: (0, i)),
        ],
        out_shape=[
            jax.ShapeDtypeStruct((topk, bs), jnp.float32),
            jax.ShapeDtypeStruct((topk, bs), jnp.int32),
        ],
        interpret=interpret,
        name="grouped-topk-v3",
    )(router_logits, bias)
    # Kernel emits [topk, BS] (BS in the lane dim — the dense, unpadded layout). The [BS, topk]
    # contract is recovered by .T, which XLA realizes as a FREE bitcast ([topk,BS]{1,0} == [BS,topk]
    # {0,1}) rather than the relayout `copy` it inserts when the kernel emits [BS,topk] with topk in
    # the lanes (topk=8 padded to 128).
    return weights_t.T, ids_t.T








