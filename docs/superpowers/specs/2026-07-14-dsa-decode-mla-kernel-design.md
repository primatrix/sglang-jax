# Decode Sparse MLA Kernel Design

## Goal

Add an isolated TPU/Pallas prototype for the decode portion of DeepSeek Sparse
Attention (DSA).  It will consume already-selected physical KV slots and run
MLA attention over those slots only.  It intentionally does not implement the
indexer, indexer-K cache, Top-K state lifecycle, IndexShare, model-runner
integration, prefill, or MTP.

## Context and constraints

The current absorbed MLA kernel accepts a ragged sequence page table and scans
each sequence from logical token zero to its full KV length.  Passing a
Top-K-derived dense mask to that kernel would retain the scan, KV DMA and QK
work for every token.  The prototype must instead make selected physical slots
the only KV addresses presented to the attention loop.

The production GLM-5.2 configuration uses `index_topk = 2048`.  A full
`int32[160K, 2048]` prefill result is about 1.22 GiB, so this prototype fixes
the scope to decode (`[B, K]`) and treats prefill metadata as a later,
streaming design problem.

## Options considered

1. **Use the existing MLA kernel with a mask.** This is numerically easy but
   still scans and reads all KV pages, so it cannot establish sparse-kernel
   performance. Rejected.
2. **Materialize selected KV in JAX, then call dense MLA.** This gives a useful
   reference and can test the address transform, but adds an HBM gather buffer
   and is not the target Pallas kernel. Retained only as the correctness oracle.
3. **Fixed-K Pallas decode kernel over physical slots.** One program handles
   one decode request; it reads only the supplied slots in fixed-size chunks,
   applies online softmax and emits the latent MLA output. Optional
   deterministic slot sorting is an input-side optimization, not part of the
   correctness contract. Recommended.

## Public kernel contract

Create a kernel module under
`python/sgl_jax/srt/kernels/mla/dsa/` with one public function:

```python
def dsa_decode_mla_attention(
    ql_nope: jax.Array,       # bf16[B, H, kv_lora_rank]
    q_pe: jax.Array,          # bf16[B, H, qk_rope_head_dim]
    cache_kv: jax.Array,      # existing MLA packed paged cache
    topk_slots: jax.Array,    # int32[B, K], physical slot or -1 padding
    valid_counts: jax.Array,  # int32[B], 0 <= count <= K
    *,
    sm_scale: float,
) -> jax.Array:               # bf16[B, H, kv_lora_rank]
```

A physical slot is `physical_page * page_size + page_offset`.  The kernel
derives packed row/column from the existing MLA cache packing.  A negative
slot, or an index at or above `valid_counts[b]`, is masked before softmax and
must never contribute.  A request with zero valid slots is invalid input and
raises before dispatch.  Slot order and duplicate slots are semantically
preserved: reordering is permitted only when the same multiset is retained.

The caller owns causal correctness: it must produce slots only for tokens that
are visible to the query.  This is deliberate because physical slots do not
retain request-local logical positions.  The kernel will validate shapes,
dtypes, page alignment and slot bounds in Python/JAX before Pallas dispatch.

## Implementation boundary

The first implementation uses static `K` and a compile-time chunk size.  It
loads selected packed cache rows directly, computes QK, online-softmax state
and PV for that chunk, then merges the chunk state into the per-request
accumulator.  It does not allocate a `[B, K, D]` HBM gather tensor.

The first version will be correct before it is optimized.  Its benchmark
records both unsorted slots and page-sorted slots; sorted slots must not change
the reference output beyond the documented BF16 reduction tolerance.  A later
iteration may replace direct gathers with per-page buckets and asynchronous
DMA only if profiling shows a benefit.

## Correctness design

There are two independent CPU/JAX references, both operating on the exact
same packed cache and selected slots:

1. `reference_dsa_decode_mla_attention` decodes each physical slot, gathers
   the MLA latent KV and RoPE key, applies the valid-slot mask, and evaluates
   the complete softmax in FP32.
2. A dense-selected cross-check first gathers that same selected KV tensor and
   evaluates the equivalent MQA expression with `einsum` and `jax.nn.softmax`.

The two references must be bitwise equal in FP32 for test inputs.  The Pallas
output is compared to the primary reference after casting to FP32 using
`rtol=2e-2`, `atol=1e-2`, matching existing BF16 MLA tests.  Tests will cover:

- page sizes 8, 16, 32 and 64, including slots on page boundaries;
- heterogeneous batch rows, non-monotonic slot order and duplicate slots;
- `-1` padding and `valid_counts < K` without invalid slots contributing;
- a one-token selected set and a 2048-token selected set;
- invalid shapes, dtypes, out-of-range slots and zero valid count failing
  before kernel dispatch;
- permutation invariance in FP32 reference and tolerance-bounded Pallas output.

The prototype is not a model-logit correctness claim.  Once it is integrated
with the indexer, the next layer of validation will compare GLM outputs and
Top-K rows against GPU SGLang with the same checkpoint and prompts.

## Falcon validation plan

Falcon TPU runs are split into separate gates:

1. **Compile/numerics:** run the targeted unit suite on one accelerator slice
   using BF16 and the GLM MLA dimensions (`H=64 / TP-local heads`,
   `kv_lora_rank=512`, `qk_rope_head_dim=64`, `K=2048`).  Archive command,
   device type, JAX version and max absolute error.
2. **Kernel microbenchmark:** compare sparse and dense MLA with identical
   cache layout and `B in {1, 8, 32, 128}`, contexts
   `{8K, 32K, 128K, 160K}` and `K=2048`.  Record compile time separately from
   50 warm-ups plus 200 timed invocations; report median and p99 latency.
3. **Profile gate:** capture an XProf trace for representative `B=1` and
   `B=32`, reporting HBM read bytes, VMEM spills and the compiler-produced
   program.  Sparse runs must show KV work proportional to K rather than the
   full context length before claiming a sparse-kernel gain.

No end-to-end TPOT or serving claim is made until indexer, cache/state and
model-runner integration are implemented.

## Non-goals

- Prefill and query-specific `[T, K]` metadata;
- indexer projections, Top-K selection, indexer-K cache and IndexShare;
- token-pool lifecycle, continuous batching, prefix reuse and cancellation;
- MTP, distributed sharding, FP8 and page-bucketing DMA optimization.
