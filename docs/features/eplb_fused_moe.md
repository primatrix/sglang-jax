# EPLB for Fused MoE on JAX/TPU (Design)

## Overview

This document describes a design to add **Expert Parallelism Load Balancing (EPLB)** to the
**fused MoE** implementation (`moe_backend=fused`) in `sgl-jax-comp`, targeting JAX/TPU.

The design follows the same high-level idea as DeepSeek's EPLB as integrated in SGLang:
measure expert routing skew, compute an improved expert placement (optionally with redundant
expert replicas), and **rebalance expert weights online** to reduce step-time stragglers.

## Goals

- Reduce end-to-end MoE latency under highly imbalanced routing by minimizing EP stragglers.
- Support **redundant experts** (up to `R<=128` extra physical slots per MoE layer).
- Support **online** rebalancing (periodic, configurable cadence).
- Keep compilation stable by avoiding shape changes across rebalances.
- Assume the EP group uses **all devices in the 2D mesh**: `ep_size = dp_size * tp_size`.

## Non-Goals (initially)

- Topology-aware placement (e.g., node-aware “nearest” selection).
- Multi-model / multi-engine coordination.
- Elastic EP (changing `ep_size` at runtime).
- Guaranteeing that EPLB works on non-TPU platforms (CPU/GPU) with the same kernel.

## Background: Current Fused MoE Constraints

The current fused kernel (`python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py`) assumes:

- The EP group size is derived from mesh axes (`dp_axis_name` and `tp_axis_name`):
  `ep_size = mesh[dp_axis_name] * mesh[tp_axis_name]`.
- Experts are sharded as a **contiguous block** per EP rank:
  global expert id is effectively `e_id = ep_rank * local_num_experts + local_e_id`.

As a result, routing imbalance becomes compute imbalance across ranks, and overall throughput
is dominated by the slowest rank.

## Key Idea: Logical vs Physical Experts

We introduce two expert index spaces per MoE layer:

- **Logical experts**: the model semantics (router outputs `E_logical` logits).
- **Physical expert slots**: the weights that exist in memory and are sharded across EP ranks
  (size `E_physical = E_logical + R`).

Redundant experts are implemented by mapping multiple physical slots to the same logical expert.

### Required invariants

- `E_physical = E_logical + R`, where `0 <= R <= 128`.
- `E_physical % ep_size == 0` so every rank owns exactly `local_E = E_physical / ep_size` slots.
- Compilation stability: `E_physical` must be **static** for a given compiled graph. In practice,
  we recommend fixing `R` to its configured maximum at startup.

## Control Plane vs Data Plane

### Control plane (slow path, periodic)

- Collect per-layer routing statistics in logical space:
  `tokens_per_logical[layer, E_logical]`.
- Run EPLB algorithm on host (numpy) to produce new placement metadata:
  - `physical_to_logical_map[layer, E_physical]`
  - `logical_to_rank_dispatch_physical_map[layer, E_logical, ep_size]` (for static dispatch)
- Broadcast metadata to all ranks and update model weights with an online rebalance.

### Data plane (fast path, every MoE call)

To preserve MoE semantics with redundant experts, **Top-K must be performed in logical space**.
Then logical top-k ids are mapped to physical ids (choosing a replica) before executing the fused
dispatch/compute/combine.

This implies adding a fused-kernel entrypoint that accepts precomputed top-k (ids and weights).

## Kernel/API Design

### New entrypoint: fused_ep_moe_from_topk (recommended)

Add a new public API that skips internal top-k selection:

```
fused_ep_moe_from_topk(
  mesh,
  tokens,                      # (num_tokens, hidden_size)
  w1, w2, w3,                   # weights for E_physical experts
  topk_ids_physical,            # (num_tokens, top_k) int32 in [0, E_physical)
  topk_weights,                 # (num_tokens, top_k) float32/bf16
  *,
  act_fn="silu",
  block_config=...,
  dp_axis_name="data",
  tp_axis_name="tensor",
  ...
) -> (num_tokens, hidden_size)
```

Rationale:

- Avoids materializing `(num_tokens, E_physical)` router logits in HBM.
- Ensures semantics: top-k is computed over `E_logical`, not over replicated columns.
- Supports multiple dispatch algorithms (static/dynamic) by changing the logical→physical map.

### Keeping the existing fused_ep_moe

`fused_ep_moe(...)` remains for non-EPLB mode and for debugging parity:
it computes internal top-k from logits and runs the same fused pipeline.

## Expert Placement Metadata

Define an EPLB metadata object that is cheap to broadcast and static-shape friendly:

- `physical_to_logical_map[layer, E_physical] : int32`
- `logical_to_rank_dispatch_physical_map[layer, E_logical, ep_size] : int32`

For a token located on EP rank `r`, the mapping for a logical expert id `e` is:

```
physical_id = logical_to_rank_dispatch_physical_map[layer, e, r]
```

This physical id is used in `topk_ids_physical` passed into the kernel.

## Online Weight Rebalance

Online rebalance updates the physical expert weights (`w1/w2/w3`) so that each physical slot
contains the weights of its assigned logical expert.

The rebalance process is purely a permutation/copy in the expert-leading dimension:

- If a logical expert is replicated `k` times, its weights are copied into `k` different physical slots.
- Physical slots are sharded by rank, so rebalance may require cross-rank data movement.

### JAX/TPU communication primitive

Use `jax.lax.ragged_all_to_all` to implement a collective “send arbitrary rows to arbitrary ranks”
without requiring fixed-size all-to-all payloads.

Implementation outline:

1. Construct a per-rank rebalance plan: for each destination local slot, choose a source physical slot
   (prefer local, else any).
2. Pack source rows for each destination rank into a contiguous buffer.
3. Perform `ragged_all_to_all` to exchange packed rows.
4. Unpack and write into the destination weight buffer (same shape as before).

## EPLB Algorithm Integration

This repo can either:

1. Reuse DeepSeek EPLB logic conceptually (replicate hot experts, then place/assign to minimize max load),
   implemented in numpy; or
2. Port SGLang’s exact algorithm modules later.

For correctness, the algorithm must output a valid mapping under invariants above.

## Operational Considerations

- **Memory**: increasing `E_physical` increases MoE weights and may reduce batch capacity on TPU.
  Redundancy should be used judiciously.
- **Stability**: avoid changing `R` at runtime; change only mapping and weights in-place.
- **Cadence**: rebalance every `N` forward passes; optionally update layers in chunks to amortize time.
- **Fallbacks**:
  - If statistics are insufficient or rebalance fails, keep prior metadata/weights.
  - Always support disabling EPLB without code changes (`enable_eplb=False`).

## Testing Strategy

Given limited TPU CI availability, tests focus on:

- Mapping invariants (shape, divisibility, bounds, coverage).
- Dispatch mapping correctness (selected physical ids correspond to the intended logical expert).
- Rebalance plan correctness (every destination slot can be sourced from a slot holding the same logical expert).

TPU kernel equivalence (internal top-k vs external top-k) should be validated manually on TPU by benchmarking.
