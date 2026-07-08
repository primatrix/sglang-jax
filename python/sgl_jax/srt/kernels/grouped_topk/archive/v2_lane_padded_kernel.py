"""Token-in-lane Pallas TPU kernel for biased grouped top-k MoE routing (v2, experimental).

Same algorithm as `grouped_topk/v1/kernel.py` — biased grouped top-k (DeepSeek-V3 noaux_tc) done
WITHOUT any `sort`, entirely via `max`/`argmax` selection, matching `gate.py:_biased_grouped_topk`
id-for-id including ties — but with the in-kernel working layout **transposed to `[E, BT]`**:

  * v1 keeps `[BT, E]` (tokens in sublane, experts in the 128-wide lane dim). Every per-token
    reduction is over the expert axis == the lane dim, i.e. a slow *cross-lane* reduction, and the
    128 lanes are wasted collapsing experts instead of carrying independent tokens.
  * v2 transposes the loaded block to `[E, BT]` (experts in sublane/major, tokens in lane/minor).
    All selection reductions become reductions over the **sublane/major** axis, so the 128
    token-lanes are computed in parallel and no cross-lane permute is needed.

Everything else (post-bias scores → group top-2 sum → select `topk_group` groups → mask dropped
groups → select `topk` experts, weights gathered from PRE-bias logits, lowest-index tie-break via
`max` + masked-`min`) mirrors v1 exactly, just with the reduced axis flipped from 1 to 0.

Drop-in signature-compatible with `grouped_topk_pallas`. This is a benchmark/experiment variant;
production routing still goes through v1 until v2 is shown faster on TPU and wired into `gate.py`.
"""

from __future__ import annotations

import functools
import logging

import jax
import jax.experimental.pallas as pl
import jax.numpy as jnp

# Reuse v1's helpers so behavior (VMEM cap, alignment, interpret toggle) stays in one place.
from sgl_jax.srt.kernels.grouped_topk.v1.kernel import (
    NEG_INF,
    SAFE_AUTO_BT,
    _align_to,
    _largest_safe_divisor,
    get_interpret,
)

logger = logging.getLogger(__name__)


def _next_power_of_2(x: int) -> int:
    return 1 if x <= 1 else 1 << (x - 1).bit_length()


# Tuned (block_tokens, unroll) for grouped_topk_pallas_v2, keyed by device_kind then
# (pow2(T_local), E, G, Gtop, k). Autotuned on real TPU by
# `benchmark.kernels.grouped_topk.tune_grouped_topk_bt --variant v2` (v7x: exp-rured4jn0i).
# v2's [E,BT] tile lets BT reach 4096 (v1 caps at 2048 via padded outputs), which large-T wants.
# Lookup misses fall back to the 128-aligned VMEM-safe divisor in the wrapper.
_TUNED_V2_BT: dict[str, dict[tuple, tuple[int, int]]] = {
    "TPU7x": {
        # E=256 (DeepSeek-V3 / Ling) — production config
        (256, 256, 8, 4, 8): (256, 8),
        (512, 256, 8, 4, 8): (512, 8),
        (1024, 256, 8, 4, 8): (1024, 8),
        (2048, 256, 8, 4, 8): (1024, 8),
        (4096, 256, 8, 4, 8): (4096, 8),
        (8192, 256, 8, 4, 8): (4096, 8),
        (16384, 256, 8, 4, 8): (4096, 8),
        (32768, 256, 8, 4, 8): (4096, 8),
        # E=512 (MaxText)
        (256, 512, 8, 4, 8): (256, 8),
        (512, 512, 8, 4, 8): (512, 8),
        (1024, 512, 8, 4, 8): (1024, 8),
        (2048, 512, 8, 4, 8): (1024, 8),
        (4096, 512, 8, 4, 8): (1024, 8),
        (8192, 512, 8, 4, 8): (1024, 8),
        (32768, 512, 8, 4, 8): (1024, 8),
    },
}


def _tuned_v2_config(bs: int, e: int, g: int, gtop: int, k: int) -> tuple[int | None, int | None]:
    """Best (BT, unroll) for this device + routing shape, or (None, None) on a miss."""
    try:
        kind = jax.devices()[0].device_kind
    except Exception:  # noqa: BLE001
        return None, None
    table = _TUNED_V2_BT.get(kind)
    if not table:
        return None, None
    return table.get((_next_power_of_2(bs), e, g, gtop, k), (None, None))


def _grouped_topk_kernel_v2(
    logits_ref,  # [BT, E] f32  (router_logits, PRE-bias) — loaded token-major like v1
    bias_ref,  # [E]     f32  (correction_bias)
    w_ref,  # [BT, padded_topk] f32  out: weights
    ids_ref,  # [BT, padded_topk] i32  out: expert ids
    *,
    n_group: int,
    topk_group: int,
    topk: int,
    num_experts: int,
    padded_topk: int,
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

    # ① group score = sum of top-2 within each group, via 2-pass max (no sort). Mirrors v1 with the
    #    reduced axis flipped 1->0. argmax tie-break here is irrelevant: the top-2 *sum* is the same
    #    whichever of two equal maxima is masked first.
    with jax.named_scope("group_top2"):
        g_scores = []
        for g in range(n_group):
            sl = scores[g * S : (g + 1) * S, :]  # [S, BT]
            v1 = jnp.max(sl, axis=0, keepdims=True)  # [1, BT]
            i1 = jnp.argmax(sl, axis=0, keepdims=True)  # [1, BT]
            io = jax.lax.broadcasted_iota(jnp.int32, sl.shape, 0)  # [S, BT]
            sl_masked = jnp.where(io == i1, NEG_INF, sl)
            v2 = jnp.max(sl_masked, axis=0, keepdims=True)  # [1, BT]
            g_scores.append(v1 + v2)  # [1, BT]
        group_scores = jnp.concatenate(g_scores, axis=0)  # [G, BT]

    # ② select `topk_group` groups, lowest-index tie-break (matches jax.lax.top_k) via max + masked
    #    min(iota) — NOT argmax, since Mosaic's reduction argmax does not break ties to lowest index.
    with jax.named_scope("group_select"):
        group_mask = jnp.zeros((n_group, bt), dtype=jnp.bool_)
        g_iota = jax.lax.broadcasted_iota(jnp.int32, (n_group, bt), 0)  # group index along axis 0
        tmp = group_scores
        for _ in range(topk_group):
            gmax = jnp.max(tmp, axis=0, keepdims=True)  # [1, BT]
            gi = jnp.min(jnp.where(tmp == gmax, g_iota, n_group), axis=0, keepdims=True)  # [1, BT]
            m = g_iota == gi  # [G, BT]
            group_mask = jnp.logical_or(group_mask, m)
            tmp = jnp.where(m, NEG_INF, tmp)

    # mask experts in dropped groups -> -inf (per-group where + concat), mirroring v1 axis-flipped.
    with jax.named_scope("expert_mask"):
        masked_slices = []
        for g in range(n_group):
            gm = group_mask[g : g + 1, :]  # [1, BT]
            masked_slices.append(jnp.where(gm, scores[g * S : (g + 1) * S, :], NEG_INF))  # [S, BT]
        masked = jnp.concatenate(masked_slices, axis=0)  # [E, BT]

    # ③ select `topk` experts, lowest-index tie-break; weight from PRE-bias logits at the selected
    #    id. fori_loop carries a single [E, BT] working array and writes each pick into ROW k of the
    #    [padded_topk, BT] output buffers, keeping per-block VMEM O(E*BT) independent of topk.
    with jax.named_scope("final_select"):
        e_iota = jax.lax.broadcasted_iota(jnp.int32, (E, bt), 0)  # expert index along axis 0
        row_iota = jax.lax.broadcasted_iota(jnp.int32, (padded_topk, bt), 0)  # output-row index
        ids_init = jnp.full((padded_topk, bt), -1, dtype=jnp.int32)
        w_init = jnp.zeros((padded_topk, bt), dtype=jnp.float32)

        def _pick(k, carry):
            cur, ids_buf, w_buf = carry
            cmax = jnp.max(cur, axis=0, keepdims=True)  # [1, BT]
            idx = jnp.min(
                jnp.where(cur == cmax, e_iota, E), axis=0, keepdims=True
            )  # [1, BT] lowest expert id achieving the max
            sel = e_iota == idx  # [E, BT]
            wval = jnp.sum(jnp.where(sel, logits, 0.0), axis=0, keepdims=True)  # [1, BT] pre-bias
            write = row_iota == k  # [padded_topk, BT] one-hot on row k (loop index)
            ids_buf = jnp.where(write, idx.astype(jnp.int32), ids_buf)
            w_buf = jnp.where(write, wval.astype(jnp.float32), w_buf)
            cur = jnp.where(sel, NEG_INF, cur)  # drop the winner before the next pick
            return cur, ids_buf, w_buf

        _, ids_out, w_out = jax.lax.fori_loop(
            0, topk, _pick, (masked, ids_init, w_init), unroll=unroll_factor
        )

    # transpose the [padded_topk, BT] outputs back to [BT, padded_topk] to match out_specs.
    ids_ref[...] = ids_out.T
    w_ref[...] = w_out.T


def grouped_topk_pallas_v2(
    router_logits: jax.Array,  # [BS, E] (any float; cast to f32 inside)
    correction_bias: jax.Array,  # [E]
    *,
    num_expert_group: int,
    topk_group: int,
    topk: int,
    block_tokens: int | str = "auto",
    unroll: int | None = None,
    interpret: bool | None = None,
):
    """Token-in-lane biased grouped top-k. Returns (topk_weights[BS,k], topk_ids[BS,k]).

    Drop-in signature-compatible with `grouped_topk_pallas` (v1). Because tokens land in the lane
    dim after the in-kernel transpose, the token block `block_tokens` should be 128-aligned; the
    `"auto"` path picks the largest 128-aligned divisor of BS (<= SAFE_AUTO_BT), falling back to a
    single whole-batch block when BS has no such divisor.
    """
    bs, e = router_logits.shape
    router_logits = router_logits.astype(jnp.float32)
    bias = correction_bias.astype(jnp.float32)

    tuned_bt, tuned_unroll = _tuned_v2_config(bs, e, num_expert_group, topk_group, topk)
    if block_tokens == "auto":
        if tuned_bt is not None and bs % tuned_bt == 0:
            bt = tuned_bt  # measured to fit v2 VMEM even above SAFE_AUTO_BT (e.g. BT=4096)
        else:
            # 128-aligned (lane dim) divisor of BS, largest that fits VMEM; else whole-batch block.
            bt = _largest_safe_divisor(bs, cap=SAFE_AUTO_BT, align=128) or bs
            if bt > SAFE_AUTO_BT:
                logger.warning(
                    "grouped_topk_v2: auto block_tokens fell back to whole-batch BT=%d (BS=%d has "
                    "no 128-aligned VMEM-safe divisor); a single [%d,%d] tile may exceed VMEM. Pad "
                    "local tokens to a multiple of 128 or pass an explicit block_tokens.",
                    bt,
                    bs,
                    bs,
                    e,
                )
    else:
        bt = min(block_tokens, bs)
        if bs % bt != 0:
            raise ValueError(f"BS={bs} must be divisible by block_tokens={bt}")
    if interpret is None:
        interpret = get_interpret()

    padded_topk = _align_to(topk, 128)  # TPU VMEM output tile needs a 128-multiple minor dim

    # explicit unroll wins, else tuned, else full unroll; clamp to 1..topk
    unroll_factor = max(1, min(int(unroll if unroll is not None else tuned_unroll or topk), topk))

    kernel = functools.partial(
        _grouped_topk_kernel_v2,
        n_group=num_expert_group,
        topk_group=topk_group,
        topk=topk,
        num_experts=e,
        padded_topk=padded_topk,
        unroll_factor=unroll_factor,
    )
    weights, ids = pl.pallas_call(
        kernel,
        grid=(bs // bt,),
        in_specs=[
            pl.BlockSpec((bt, e), lambda i: (i, 0)),
            pl.BlockSpec((e,), lambda i: (0,)),
        ],
        out_specs=[
            pl.BlockSpec((bt, padded_topk), lambda i: (i, 0)),
            pl.BlockSpec((bt, padded_topk), lambda i: (i, 0)),
        ],
        out_shape=[
            jax.ShapeDtypeStruct((bs, padded_topk), jnp.float32),
            jax.ShapeDtypeStruct((bs, padded_topk), jnp.int32),
        ],
        interpret=interpret,
        name="grouped-topk-v2",
    )(router_logits, bias)
    return weights[:, :topk], ids[:, :topk]
