# Pallas KDA Kernel: NaN with Realistic Gate Magnitudes

**Date**: 2026-04-26
**Affected**: `chunk_kda_fwd` in `python/sgl_jax/srt/kernels/kda/kda.py`
**Kernel origin**: PR #964 (`@pathfinder-pf`)
**Severity**: Blocking — kernel unusable with real HF weights

## Summary

The Pallas chunked KDA kernel (`chunk_kda_fwd`) produces NaN output when gate values have magnitude ≥10. Real HF weights from `moonshotai/Kimi-Linear-48B-A3B-Instruct` produce activated gate values in range `[-1922, 0]`, far exceeding this threshold. The naive JAX kernel (`naive_recurrent_kda`) handles the same inputs correctly.

## What Works vs What Doesn't

| Kernel | Small gate (\|g\| ≤ 5) | Real-weight gate (\|g\| ~ 1900) |
|--------|------------------------|----------------------------------|
| `naive_recurrent_kda` (pure JAX) | ✓ | ✓ |
| `chunk_kda_fwd` (Pallas) | ✓ | **NaN** |

## Where the NaN Comes From

The KDA forward has a 4-stage pipeline. The NaN originates in the **gate cumsum** and **inter-chunk state propagation** stages — NOT in the gate activation step.

### 问题追溯

gate 值在 kernel 里的关键使用点是 `kda_gate_chunk_cumsum`（`kda.py` ~line 1000–1033）：

```python
def kda_gate_chunk_cumsum(g, A_log, chunk_size, scale=None, dt_bias=None, ...):
    g_f32 = g.astype(jnp.float32)
    if dt_bias is not None:
        g_f32 = g_f32 + dt_bias.reshape(H, K)

    A = A_log.astype(jnp.float32)
    # 这一步产生大负值：-exp(2.77~5.3) * softplus(x) → 范围 [-1922, 0]
    g_act = -exp(A).reshape(1, 1, H, 1) * jax.nn.softplus(g_f32)

    # 然后做 chunk-local cumsum，scale = 1/ln(2)
    return chunk_local_cumsum_vector(g_act, chunk_size=chunk_size, scale=scale, ...)
```

其中 `scale = _RCP_LN2 = 1/ln(2) ≈ 1.4427`，是把 log-space 转成 log2-space 的常数，不是问题来源。

真正的问题是 `g_act` 本身的量级：`-exp(A_log) * softplus(g + dt_bias)` 产生的值（~-60 per timestep），经过 64 步 cumsum 累积到 ~-3840，然后下游的 Pallas kernel 对这个 cumsum 做 `exp()` / `log2()` 变换时溢出。

**Workaround 方向**：问题不在于能不能绕过某个参数，而在于 Pallas kernel 缺少 per-chunk normalization。具体来说：

1. **Safe gate normalization** — 在每个 chunk 内，cumsum 前减去 chunk 内最大值（或用 chunk 边界值做 baseline），让 `exp(cumsum - baseline)` 始终在可表示范围内。GPU 的 fla Triton kernel 大概率做了类似处理。
2. **`safe_gate` 参数无效** — kernel 里有个 `safe_gate` 参数（默认 `True`，line 334/456），但从代码看它只影响取 cumsum 的中间值位置（`g_i[BC//2 : BC//2+1]` vs `g_i[0:1]`），不是真正的数值稳定化。

结论：没有简单的参数级 workaround —— 需要改 Pallas kernel 内部的数值处理逻辑。

### Data flow (simplified)

```
raw_gate g  →  gate activation  →  cumsum  →  exp(cumsum)  →  inter-chunk propagation  →  output
                                     ↑                ↑
                              accumulates to      exp(-3840) is fine (≈0),
                              very large negative  but log2/exp intermediate
                              values (-3840+)      arithmetic overflows → NaN
```

### Why `use_gate_in_kernel` doesn't help

Regardless of where gate activation happens (inside kernel via `use_gate_in_kernel=True`, or pre-computed externally with `=False`), the same large negative activated gate values enter the Pallas cumsum + state propagation stages. Both paths produce NaN:

```
use_gate_in_kernel=True:   NaN  (24576/32768 elements)
use_gate_in_kernel=False:  NaN  (21632/32768 elements)
```

### Why the naive kernel works

`naive_recurrent_kda` processes one timestep at a time: `S = S * exp(g_t)`. Each `exp(g_t)` where `g_t ≈ -60` produces a very small but valid float (~1e-26). The state decays toward zero — no overflow, no NaN.

## Reproduction

Tested on: TPU v6e-4, JAX 0.8.1 (libtpu 0.0.30) and JAX 0.9.2 (libtpu 0.0.38). Both produce NaN.

```python
import jax, jax.numpy as jnp
from sgl_jax.srt.kernels.kda import chunk_kda

H, K, V, T = 32, 128, 128, 128
q = jax.random.normal(jax.random.PRNGKey(0), (1, T, H, K), dtype=jnp.float32) * 0.1
k = jax.random.normal(jax.random.PRNGKey(1), (1, T, H, K), dtype=jnp.float32) * 0.1
v = jax.random.normal(jax.random.PRNGKey(2), (1, T, H, V), dtype=jnp.float32) * 0.1
beta = jax.random.uniform(jax.random.PRNGKey(4), (1, T, H), dtype=jnp.float32)
cu = jnp.array([0, T], dtype=jnp.int32)
init = jnp.zeros((1, H, K, V), dtype=jnp.float32)

for gate_scale in [1, 10, 50, 100, 500, 1000, 2000]:
    g = -jnp.abs(jax.random.normal(jax.random.PRNGKey(3), (1, T, H, K), dtype=jnp.float32)) * gate_scale
    o, fs, *_ = chunk_kda(q, k, v, g, beta, scale=K**-0.5,
        initial_state=init, output_final_state=True, cu_seqlens=cu)
    nan_pct = jnp.isnan(o).sum() / o.size * 100
    print(f"gate_scale={gate_scale:5d} | nan: {nan_pct:.0f}%")
```

Output:
```
gate_scale=    1 | nan: 0%
gate_scale=   10 | nan: 100%    ← threshold
gate_scale=   50 | nan: 100%
gate_scale=  100 | nan: 100%
gate_scale=  500 | nan: 100%
gate_scale= 1000 | nan: 100%
gate_scale= 2000 | nan: 100%
```

### Why gate clamping doesn't help

Tested clamping activated gate to [-C, 0] before passing to kernel. Even aggressive clamping (C=3) still NaN:

```
clamp=3:  pallas NaN=True   naive(clamped) vs naive(raw) diff = 4.4e-04
clamp=5:  pallas NaN=True   naive(clamped) vs naive(raw) diff = 6.1e-05
clamp=7:  pallas NaN=True   naive(clamped) vs naive(raw) diff = 8.5e-06
clamp=10: pallas NaN=True   naive(clamped) vs naive(raw) diff = 4.4e-07
```

Further investigation shows the threshold is per-chunk cumsum magnitude ~100, which corresponds to per-element average of ~-1.5:

| Gate values | Per-element avg | Chunk cumsum (64 steps) | Result |
|-------------|-----------------|------------------------|--------|
| scale=1 (range [-5, 0]) | ~-0.8 | ~-51 | ✓ |
| uniform -3 | -3.0 | -192 | **NaN** |
| scale=3 (range [-15, 0]) | ~-2.4 | -154 | **NaN** |
| Real weights | ~-30 | ~-1920 | **NaN** |

Clamping to stay under the threshold (individual gate ~-1.5) would change model behavior from "fully forgotten" to "22% retention per step" — not acceptable.

### Why per-chunk normalization of cumsum doesn't help either

Tested subtracting the first position's cumsum value within each chunk (64 steps) before passing to downstream Pallas functions:

```
raw cumsum:    [-5178, 0]
normed cumsum: [-5151, 0]   ← barely changed
intra-chunk:   all NaN
inter-chunk:   all NaN
```

The problem is **intra-chunk**, not inter-chunk. Within a single chunk of 64 steps, gate ≈ -50 per step (activated), scaled by `1/ln(2)` ≈ 1.44:

- Per-step in log2-space: ~-72
- 16-step sub-chunk (BC=16) cumsum: ~-1154
- `exp2(-1154)` → underflow to 0
- `exp2(+578)` (midpoint reference, reverse direction) → overflow to inf

Even at the finest granularity the kernel uses (BC=16 sub-chunks), the cumsum magnitude (~1154) far exceeds fp32 representable range for exp2 (max ~127). **No reference-subtraction trick can fix this — the per-step gate values themselves are too large for chunked exp2 arithmetic.**

## Suggested Fix Direction

The Pallas kernel needs an **algorithmic rewrite** for gate handling, not a parameter-level fix:

1. **Full log-space arithmetic**: Keep gate cumsums in log-space throughout the intra-chunk and inter-chunk computation. Only materialize `exp()` at the final output stage where relative differences are small.
2. **Per-element online normalization**: Track a running max/normalization factor per element and renormalize at each step, similar to online softmax.
3. **Reference**: Study fla's Triton `chunk_kda` kernel (fla-org/flash-linear-attention) for the numerical strategy used on GPU.

## Additional Kernel Bugs Found During Testing

1. **Missing `max_T` in `chunk_local_cumsum_vector`** (line 216): `prepare_chunk_indices(cu_seqlens, BT)` called without `max_T`, causing `TypeError`. Fixed locally by passing `max_T=T`.

2. **`chunk_kda_fwd` skips `chunk_indices` computation when `None`** (line 1206): The condition `if chunk_indices is not None:` is inverted — `chunk_indices` must always be computed for varlen operation. Downstream functions assert it's not None. Fixed locally by making it unconditional.

## Impact on Sub-3

- **Phase A** (KimiDeltaAttention layer + backend gate changes): Complete and committed.
- **Phase B** (M1 numerical alignment tests): **Blocked**. Tests written but cannot verify output against GPU reference due to NaN.
- **Phase C** (prefill-to-decode invariance): Can proceed using the naive kernel (decode path).
