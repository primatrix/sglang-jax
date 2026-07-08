# grouped-topk archived variants

Superseded experiments from the v3 optimization campaign (see `gtopk-v3-optimization-report.md`).
Production kernel: `grouped_topk/v1/kernel.py` (`grouped_topk_pallas`).

| file | what | vs v1(prod) @T=16384 | note |
|---|---|---|---|
| `v1_bt_e_kernel.py` | original `[BT,E]` layout, stable tie-break, padded output | ~221µs (1×, old prod) | experts in lanes → cross-lane reductions |
| `v2_lane_padded_kernel.py` | token-in-lane `[E,BT]`, padded `[BT,128]` output | ~113µs (tuned) | pays a padded-output relayout copy |
| `v3_lane_argmax_kernel.py` | token-in-lane, **hardware argmax** tie-break, unpadded | **~67µs (3.28×)** | ~5% faster than prod but NOT stable tie-break (differs from ref only on exact ties) |

The promoted v1 = v3's structure with the exact **max + masked-min** (lowest-index) tie-break, so it
is a bit-exact drop-in for `jax.lax.top_k` (~3.1× over the old `[BT,E]` kernel). `v3_lane_argmax`
is kept as the fastest non-stable option.
