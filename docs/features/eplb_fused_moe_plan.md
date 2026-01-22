# EPLB for Fused MoE on JAX/TPU (Code Plan)

This plan is derived from `docs/features/eplb_fused_moe.md` and is organized as implementable
phases with clear checkpoints.

## Phase 0 — Core types + pure-CPU algorithm (no kernel changes)

**Deliverables**

- `python/sgl_jax/srt/eplb/metadata.py`
  - `ExpertPlacement` / `ExpertLocationMetadata` dataclasses (numpy-first)
  - validation helpers for invariants:
    - `E_physical = E_logical + R`, `R<=128`
    - `E_physical % ep_size == 0`
    - bounds checks for ids
- `python/sgl_jax/srt/eplb/algorithm.py`
  - greedy baseline placement algorithm (deterministic)
  - outputs:
    - `physical_to_logical_map[layer, E_physical]`
    - `logical_to_rank_dispatch_physical_map[layer, E_logical, ep_size]`
- `python/sgl_jax/srt/eplb/weight_rebalance.py`
  - compute a per-layer “source physical id for each destination physical slot” mapping
  - build a host-side all-to-all plan (rank-local indices) for distributed weight rebalance

**Checkpoint**

- Unit tests covering mapping invariants and plan correctness (CPU-only).

## Phase 1 — Stats collection hooks for fused MoE

**Deliverables**

- A logical-space top-k path usable for stats collection:
  - Either via an existing router/topk module (preferred), or a lightweight helper.
- A recorder that accumulates:
  - `tokens_per_logical[layer, E_logical]` (sliding window)
  - optional metrics: balancedness ratio, max/mean load, etc.
- `python/sgl_jax/srt/eplb/runtime.py`
  - CPU-side controller to periodically recompute `ExpertLocationMetadata` from recorded routing stats

**Checkpoint**

- Unit tests validating the recorder aggregation (no TPU required).

## Phase 2 — Kernel API: `fused_ep_moe_from_topk`

**Deliverables**

- Modify `python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py`:
  - add a new public entrypoint `fused_ep_moe_from_topk(...)`
  - internal refactor to allow routing metadata (t2e + sizes/stats) to be computed from
    `topk_ids_physical/topk_weights` instead of `gating_output`.
- Update `python/sgl_jax/srt/layers/moe.py`:
  - `FusedEPMoE` supports an “EPLB enabled” mode:
    - compute logical top-k
    - map logical→physical using metadata
    - call `fused_ep_moe_from_topk`

**Checkpoint**

- Manual TPU benchmark:
  - internal top-k path vs external top-k path (no redundancy) should match numerically.

## Phase 3 — Online weight rebalance (TPU path)

**Deliverables**

- `python/sgl_jax/srt/eplb/weight_rebalance.py`
  - implement row exchange using `jax.lax.ragged_all_to_all` for each weight tensor
  - update model parameters in-place without changing shapes
- Hook into runtime:
  - periodic manager (rebalance every N forward passes)
  - layer chunking to bound pause time

**Checkpoint**

- Manual TPU benchmark under imbalanced routing:
  - reduced step-time variance across ranks
  - throughput/latency improvement

## Phase 4 — Redundant experts up to 128

**Deliverables**

- Enable `R > 0` end-to-end:
  - weights are allocated for `E_physical = E_logical + R`
  - metadata maps physical slots to logical experts with duplicates
  - dispatch mapping spreads logical experts across replicas
- Ensure `R` chosen satisfies `E_physical % ep_size == 0`:
  - either enforce at config time or auto-adjust `R` downward.

**Checkpoint**

- Manual TPU benchmark:
  - hotspot routing distributions benefit from redundancy (less straggler).
