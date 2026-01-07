from __future__ import annotations

import dataclasses
from typing import Dict, Iterable, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.utils.mesh_utils import create_device_mesh

MARKER = "SGL_BENCH"


@dataclasses.dataclass
class MoEBenchmarkCase:
    name: str
    num_tokens: int
    num_experts: int
    top_k: int
    hidden_size: int
    intermediate_size: int
    seed: int = 0
    activation: str = "silu"
    renormalize_topk_logits: bool = True
    num_expert_group: int = 0
    topk_group: int = 0
    routed_scaling_factor: float | None = None
    # If None, auto-pick based on available devices.
    ep_size: int | None = None
    tp_size: int | None = None


# Bailing MoE defaults (matches the observed precompile shapes).
BAILING_BASE = dict(
    num_experts=256,
    top_k=8,
    hidden_size=8192,
    intermediate_size=2048,
    activation="silu",
    renormalize_topk_logits=True,
    num_expert_group=8,
    topk_group=4,
    # Let benchmarks pick ep_size based on available devices by default.
    ep_size=None,
)

_NUM_TOKENS = (16, 32, 64, 128, 256, 512, 1024, 2048, 4096)

GROUP_GEMM_CASES: Iterable[MoEBenchmarkCase] = tuple(
    MoEBenchmarkCase(
        name=f"bailing_nt{n}_ne256_tk8_h8192_i2048",
        num_tokens=n,
        **BAILING_BASE,
    )
    for n in _NUM_TOKENS
)


def generate_router_logits(
    num_tokens: int,
    num_experts: int,
    scenario: str,
    num_experts_per_tok: int = 2,
    imbalance_factor: float = 3.0,
    *,
    seed: int = 0,
) -> jax.Array:
    """Synthetic router logits with configurable balance; keep generation cheap."""
    token_iota = jnp.arange(num_tokens, dtype=jnp.uint32)[:, None]
    expert_iota = jnp.arange(num_experts, dtype=jnp.uint32)[None, :]

    def _stateless_uniform01_u32(x: jax.Array) -> jax.Array:
        # SplitMix32-style mixing: fast, deterministic, decent bit diffusion.
        x = (x + jnp.uint32(0x9E3779B9)) & jnp.uint32(0xFFFFFFFF)
        x ^= x >> jnp.uint32(16)
        x = (x * jnp.uint32(0x85EBCA6B)) & jnp.uint32(0xFFFFFFFF)
        x ^= x >> jnp.uint32(13)
        x = (x * jnp.uint32(0xC2B2AE35)) & jnp.uint32(0xFFFFFFFF)
        x ^= x >> jnp.uint32(16)
        # Map uint32 -> [0, 1).
        return x.astype(jnp.float32) * (1.0 / 2**32)

    # Deterministic per-(token, expert) noise in [0, 1).
    base = (
        token_iota * jnp.uint32(0xD1B54A35)
        + expert_iota * jnp.uint32(0x94D049BB)
        + jnp.uint32(seed)
    )
    noise01 = _stateless_uniform01_u32(base)

    if scenario == "random":
        # Unbiased pseudo-random logits per token/expert.
        return noise01

    if scenario == "balanced":
        logits = -10.0 * jnp.ones((num_tokens, num_experts), dtype=jnp.float32)
        token_ids = token_iota.astype(jnp.int32)
        cols = (
            token_ids * num_experts_per_tok + jnp.arange(num_experts_per_tok, dtype=jnp.int32)
        ) % num_experts
        logits = logits.at[jnp.arange(num_tokens)[:, None], cols].set(10.0)
        return logits

    if scenario == "imbalanced":
        # Biased logits: per-expert bias + per-token noise.
        #
        # - Larger `imbalance_factor` => stronger skew toward low-index experts.
        # - For imbalance_factor ~= 0, bias goes to ~0 and routing is close to random.
        inv_temp = jnp.float32(max(float(imbalance_factor), 1e-6))
        expert_bias = -(expert_iota.astype(jnp.float32) / jnp.float32(num_experts)) * inv_temp
        noise = (noise01 * 2.0 - 1.0).astype(jnp.float32)
        return expert_bias + noise

    raise ValueError(f"Unknown scenario '{scenario}'. Use random|balanced|imbalanced.")


def generate_fused_router_logits(
    num_tokens: int,
    num_experts: int,
    *,
    top_k: int,
    router_balance_factor: float,
    seed: int = 0,
) -> jax.Array:
    """Router logits for fused_moe benchmarks with a single imbalance knob.

    `router_balance_factor` semantics:
    - 1.0 => balanced cyclic routing (near-uniform expert counts).
    - smaller => increasingly skewed routing toward low-index experts.
    """
    # Clamp to (0, 1] so the mapping is monotonic and avoids div-by-zero.
    f = float(router_balance_factor)
    f = max(min(f, 1.0), 1e-6)
    strength = (1.0 / f) - 1.0  # 0 at f=1, grows as f->0

    # Start from a balanced top-k assignment: for each token, set its top-k experts
    # to +1.0 and others to 0.0. This guarantees balanced routing at strength=0.
    logits = jnp.zeros((num_tokens, num_experts), dtype=jnp.float32)
    token_ids = jnp.arange(num_tokens, dtype=jnp.int32)[:, None]
    cols = (token_ids * top_k + jnp.arange(top_k, dtype=jnp.int32)) % num_experts
    logits = logits.at[jnp.arange(num_tokens)[:, None], cols].set(1.0)

    # Add a global per-expert bias that favors low-index experts. As `strength`
    # grows, this bias overwhelms the balanced pattern and concentrates load.
    expert_rank = jnp.arange(num_experts, dtype=jnp.float32)
    bias = -(expert_rank / jnp.float32(max(num_experts - 1, 1))) * jnp.float32(strength)
    logits = logits + bias[None, :]

    # Add small deterministic noise to break ties, keeping generation cheap and stable.
    token_iota = jnp.arange(num_tokens, dtype=jnp.uint32)[:, None]
    expert_iota = jnp.arange(num_experts, dtype=jnp.uint32)[None, :]
    base = (
        token_iota * jnp.uint32(0xD1B54A35)
        + expert_iota * jnp.uint32(0x94D049BB)
        + jnp.uint32(seed)
    )

    def _stateless_uniform01_u32(x: jax.Array) -> jax.Array:
        x = (x + jnp.uint32(0x9E3779B9)) & jnp.uint32(0xFFFFFFFF)
        x ^= x >> jnp.uint32(16)
        x = (x * jnp.uint32(0x85EBCA6B)) & jnp.uint32(0xFFFFFFFF)
        x ^= x >> jnp.uint32(13)
        x = (x * jnp.uint32(0xC2B2AE35)) & jnp.uint32(0xFFFFFFFF)
        x ^= x >> jnp.uint32(16)
        return x.astype(jnp.float32) * (1.0 / 2**32)

    noise01 = _stateless_uniform01_u32(base)
    logits = logits + (noise01 * 2.0 - 1.0) * 1e-3
    return logits


def build_group_sizes(
    router_logits: jax.Array, top_k: int, num_experts: int
) -> Tuple[jax.Array, jax.Array]:
    token_ids = np.arange(router_logits.shape[0], dtype=np.int32)
    topk_ids_np = np.empty((router_logits.shape[0], top_k), dtype=np.int32)
    for i in range(top_k):
        topk_ids_np[:, i] = (token_ids * top_k + i) % num_experts
    group_sizes = np.bincount(topk_ids_np.reshape(-1), minlength=num_experts).astype(np.int32)
    return jnp.asarray(group_sizes), jnp.asarray(topk_ids_np, dtype=jnp.int32)


def build_grouped_lhs(
    group_sizes: jax.Array, hidden_size: int, dtype: jnp.dtype, seed: int
) -> jax.Array:
    total = int(np.asarray(group_sizes, dtype=np.int32).sum())
    return jnp.empty((total, hidden_size), dtype=dtype)


def prepare_gmm_inputs(
    case: MoEBenchmarkCase,
    scenario: str,
    dtype: jnp.dtype = jnp.bfloat16,
) -> Dict[str, jax.Array]:
    router_logits = generate_router_logits(
        case.num_tokens,
        case.num_experts,
        scenario,
        num_experts_per_tok=case.top_k,
        imbalance_factor=case.routed_scaling_factor or 3.0,
        seed=case.seed,
    ).astype(dtype)
    group_sizes, topk_ids = build_group_sizes(router_logits, case.top_k, case.num_experts)
    lhs = build_grouped_lhs(group_sizes, case.hidden_size, dtype, seed=case.seed + 1)
    rhs = jnp.empty((case.num_experts, case.hidden_size, case.intermediate_size), dtype=dtype)
    return {
        "router_logits": router_logits,
        "group_sizes": group_sizes,
        "topk_ids": topk_ids,
        "gmm_lhs": lhs,
        "gmm_rhs": rhs,
    }


def prepare_fused_moe_inputs(
    case: MoEBenchmarkCase,
    dtype: jnp.dtype = jnp.bfloat16,
    mesh: jax.sharding.Mesh | None = None,
    *,
    ep_axis_name: str = "tensor",
    include_weights: bool = True,
    router_balance_factor: float = 1.0,
    router_imbalance_factor: float | None = None,
) -> Dict[str, jax.Array]:
    # Backward-compat alias: older CLI used --router-imbalance-factor with inverse semantics.
    if router_imbalance_factor is not None:
        router_balance_factor = float(router_imbalance_factor)

    f = float(router_balance_factor)
    if not np.isfinite(f) or f <= 0:
        raise ValueError(f"Expected router_balance_factor to be finite and > 0, got {f}.")

    if mesh is None:
        tokens = jnp.empty((case.num_tokens, case.hidden_size), dtype=dtype)
        out: dict[str, jax.Array] = {"tokens": tokens}
        if include_weights:
            out["w1"] = jnp.empty(
                (case.num_experts, case.hidden_size, case.intermediate_size), dtype=dtype
            )
            out["w3"] = jnp.empty(
                (case.num_experts, case.hidden_size, case.intermediate_size), dtype=dtype
            )
            out["w2"] = jnp.empty(
                (case.num_experts, case.intermediate_size, case.hidden_size),
                dtype=dtype,
            )
        router_logits = generate_fused_router_logits(
            case.num_tokens,
            case.num_experts,
            top_k=case.top_k,
            router_balance_factor=f,
            seed=case.seed,
        ).astype(dtype)
        out["router_logits"] = router_logits
        return out

    ep_size = mesh.shape[ep_axis_name]
    if case.num_tokens % ep_size != 0:
        raise ValueError(
            f"Expected {case.num_tokens=} to be divisible by {ep_size=} for {ep_axis_name=}."
        )
    if case.num_experts % ep_size != 0:
        raise ValueError(
            f"Expected {case.num_experts=} to be divisible by {ep_size=} for {ep_axis_name=}."
        )

    tokens_sharding = NamedSharding(mesh, P(ep_axis_name, None))
    logits_sharding = NamedSharding(mesh, P(ep_axis_name, None))
    w1_sharding = NamedSharding(mesh, P(ep_axis_name, None, None))
    w2_sharding = NamedSharding(mesh, P(ep_axis_name, None, None))
    w3_sharding = NamedSharding(mesh, P(ep_axis_name, None, None))

    # Avoid `jax.device_put(host_array, NamedSharding(...))` for large weights:
    # on multi-host runs it may trigger a cross-host equality check (allgather)
    # of the entire unsharded array and OOM device memory.
    tokens = jax.jit(
        lambda: jnp.zeros((case.num_tokens, case.hidden_size), dtype=dtype),
        out_shardings=tokens_sharding,
    )()
    out: dict[str, jax.Array] = {"tokens": tokens}
    if include_weights:
        out["w1"] = jax.jit(
            lambda: jnp.zeros(
                (case.num_experts, case.hidden_size, case.intermediate_size),
                dtype=dtype,
            ),
            out_shardings=w1_sharding,
        )()
        out["w3"] = jax.jit(
            lambda: jnp.zeros(
                (case.num_experts, case.hidden_size, case.intermediate_size),
                dtype=dtype,
            ),
            out_shardings=w3_sharding,
        )()
        out["w2"] = jax.jit(
            lambda: jnp.zeros(
                (case.num_experts, case.intermediate_size, case.hidden_size),
                dtype=dtype,
            ),
            out_shardings=w2_sharding,
        )()
    router_logits = jax.jit(
        lambda: generate_fused_router_logits(
            case.num_tokens,
            case.num_experts,
            top_k=case.top_k,
            router_balance_factor=f,
            seed=case.seed,
        ).astype(dtype),
        out_shardings=logits_sharding,
    )()
    out["router_logits"] = router_logits
    return out


def format_load_info(group_sizes: jax.Array) -> str:
    sizes = jnp.asarray(group_sizes)
    total = int(sizes.sum())
    avg = float(jnp.mean(sizes))
    return f"dispatch={total}, avg_per_expert={avg:.1f}, " f"min={sizes.min()}, max={sizes.max()}"


def select_cases(cases: Iterable[MoEBenchmarkCase] | None = None) -> Iterable[MoEBenchmarkCase]:
    num_devices = len(jax.devices())
    raw_cases: Iterable[MoEBenchmarkCase] = GROUP_GEMM_CASES if cases is None else cases

    def choose_parallelism(case: MoEBenchmarkCase) -> tuple[int, int]:
        """Pick (ep_size, tp_size) for benchmarks.

        If `case.ep_size` is None, try EP sizes starting from device_count.
        Always return (ep_size, tp_size) such that ep_size * tp_size == device_count.
        """
        if case.ep_size is None:
            target_ep = num_devices
        else:
            target_ep = case.ep_size
        target_ep = min(target_ep, case.num_experts, num_devices)

        for ep in range(target_ep, 0, -1):
            if num_devices % ep != 0:
                continue
            if case.num_tokens % ep != 0:
                continue
            if case.num_experts % ep != 0:
                continue
            return ep, num_devices // ep
        return 1, num_devices

    cases = []
    for case in raw_cases:
        ep_size, tp_size = choose_parallelism(case)
        cases.append(
            MoEBenchmarkCase(
                name=case.name,
                num_tokens=case.num_tokens,
                num_experts=case.num_experts,
                top_k=case.top_k,
                hidden_size=case.hidden_size,
                intermediate_size=case.intermediate_size,
                activation=case.activation,
                renormalize_topk_logits=case.renormalize_topk_logits,
                num_expert_group=case.num_expert_group,
                topk_group=case.topk_group,
                routed_scaling_factor=case.routed_scaling_factor,
                ep_size=ep_size,
                tp_size=tp_size,
            )
        )
    return cases


def build_mesh(ep_size: int = 1, tp_size: int = 1):
    if ep_size <= 0 or tp_size <= 0:
        raise ValueError(f"Expected {ep_size=} and {tp_size=} to be > 0.")
    devices = jax.devices()[: ep_size * tp_size]
    return create_device_mesh(
        ici_parallelism=[tp_size, ep_size],
        dcn_parallelism=[1, 1],
        devices=devices,
        mesh_axes=("data", "tensor"),
    )
