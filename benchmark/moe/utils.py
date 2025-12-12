from __future__ import annotations

import dataclasses
import gzip
import json
import os
import pathlib
import random
import re
import string
from typing import Any, Dict, Iterable, Tuple

import jax
import jax.numpy as jnp
import numpy as np

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
    seed: int = 42
    activation: str = "silu"
    renormalize_topk_logits: bool = True
    num_expert_group: int = 0
    topk_group: int = 0
    routed_scaling_factor: float | None = None
    ep_size: int = 1
    tp_size: int = 1


# Bailing MoE defaults (matches provided config).
GROUP_GEMM_CASES: Iterable[MoEBenchmarkCase] = (
    MoEBenchmarkCase(
        name="bailing_nt16_ne8_tk8_h8192_i2048",
        num_tokens=128,
        num_experts=8,
        top_k=8,
        hidden_size=8192,
        intermediate_size=2048,
        activation="silu",
        renormalize_topk_logits=True,
        num_expert_group=8,
        topk_group=4,
        routed_scaling_factor=2.5,
        ep_size=4,
        tp_size=1,
    ),
    MoEBenchmarkCase(
        name="bailing_nt512_ne8_tk8_h8192_i2048",
        num_tokens=4096,
        num_experts=8,
        top_k=8,
        hidden_size=8192,
        intermediate_size=2048,
        activation="silu",
        renormalize_topk_logits=True,
        num_expert_group=8,
        topk_group=4,
        routed_scaling_factor=2.5,
        ep_size=4,
        tp_size=1,
    ),
)


def generate_router_logits(
    num_tokens: int,
    num_experts: int,
    scenario: str,
    num_experts_per_tok: int = 2,
    imbalance_factor: float = 3.0,
    seed: int = 42,
) -> jax.Array:
    """Synthetic router logits with configurable balance; keep generation cheap."""
    if scenario == "random":
        base = jnp.reshape(
            jnp.arange(num_tokens * num_experts, dtype=jnp.float32),
            (num_tokens, num_experts),
        )
        return base * 0.001

    if scenario == "balanced":
        logits = -10.0 * jnp.ones((num_tokens, num_experts), dtype=jnp.float32)
        token_ids = jnp.arange(num_tokens, dtype=jnp.int32)[:, None]
        cols = (
            token_ids * num_experts_per_tok + jnp.arange(num_experts_per_tok, dtype=jnp.int32)
        ) % num_experts
        logits = logits.at[jnp.arange(num_tokens)[:, None], cols].set(10.0)
        return logits

    if scenario == "imbalanced":
        temperature = num_experts / (imbalance_factor * 2)
        expert_base_logits = jnp.arange(num_experts, dtype=jnp.float32)
        expert_base_logits = 10.0 * jnp.exp(-expert_base_logits / temperature)
        logits = jnp.tile(expert_base_logits, (num_tokens, 1))
        return logits

    raise ValueError(f"Unknown scenario '{scenario}'. Use random|balanced|imbalanced.")


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
    scenario: str,
    dtype: jnp.dtype = jnp.bfloat16,
) -> Dict[str, jax.Array]:
    tokens = jnp.empty((case.num_tokens, case.hidden_size), dtype=dtype)
    w1 = jnp.empty(
        (case.num_experts, 2, case.hidden_size, case.intermediate_size),
        dtype=dtype,
    )
    w2 = jnp.empty(
        (case.num_experts, case.intermediate_size, case.hidden_size),
        dtype=dtype,
    )
    router_logits = generate_router_logits(
        case.num_tokens,
        case.num_experts,
        scenario,
        num_experts_per_tok=case.top_k,
        seed=case.seed,
    ).astype(dtype)
    return {
        "tokens": tokens,
        "w1": w1,
        "w2": w2,
        "router_logits": router_logits,
    }


def compute_gmm_tiling(m: int, k: int, n: int) -> tuple[int, int, int]:
    """Match layer tiling heuristic for gmm (gate/down projections)."""
    default_tile_size = (512, 1024, 1024)
    return (
        min(default_tile_size[0], m),
        min(default_tile_size[1], k),
        min(default_tile_size[2], n),
    )


def format_load_info(group_sizes: jax.Array) -> str:
    sizes = jnp.asarray(group_sizes)
    total = int(sizes.sum())
    avg = float(jnp.mean(sizes))
    return f"dispatch={total}, avg_per_expert={avg:.1f}, " f"min={sizes.min()}, max={sizes.max()}"


def select_cases() -> Iterable[MoEBenchmarkCase]:
    return GROUP_GEMM_CASES


# ---------------- Trace-based timing helpers ---------------- #


def _load_trace(trace_root: str) -> dict[str, Any]:
    trace_dir = pathlib.Path(trace_root) / "plugins" / "profile"
    if not trace_dir.exists():
        raise FileNotFoundError(f"No trace output under {trace_dir}")
    latest_dir = max(trace_dir.iterdir(), key=os.path.getmtime)
    trace_files = list(latest_dir.glob("*.trace.json.gz"))
    if not trace_files:
        raise FileNotFoundError(f"No trace json.gz under {latest_dir}")
    with gzip.open(trace_files[0], "rb") as f:
        return json.load(f)


def _extract_marker_durations_ms(trace: dict[str, Any], task: str | None = None) -> list[float]:
    marker_events: list[dict[str, Any]] = []
    for e in trace.get("traceEvents", []):
        args = e.get("args", {})
        tf_op = args.get("tf_op", "")
        if MARKER in tf_op:
            marker_events.append(e)

    marker_call_done_events = [e for e in marker_events if e.get("name", "").endswith("call-done")]
    if marker_call_done_events:
        marker_events = marker_call_done_events

    if not marker_events:
        if not task:
            return []
        event_matcher = re.compile(task)
        events = []
        for e in trace.get("traceEvents", []):
            if "name" in e and event_matcher.match(e["name"]):
                events.append(e)
        if not events:
            return []
        min_pid = min(e["pid"] for e in events)
        events_from_min_pid = [e for e in events if e["pid"] == min_pid]
        durations_ms: list[float] = []
        for e in events_from_min_pid:
            if e.get("args", {}).get("device_duration_ps"):
                durations_ms.append(float(e["args"]["device_duration_ps"]) / 1e9)
            elif "dur" in e:
                durations_ms.append(float(e["dur"]) / 1e3)
        return durations_ms

    min_pid = min(e["pid"] for e in marker_events)
    events_from_min_pid = [e for e in marker_events if e["pid"] == min_pid]
    durations_ms = []
    for e in events_from_min_pid:
        args = e.get("args", {})
        if "device_duration_ps" in args:
            durations_ms.append(float(args["device_duration_ps"]) / 1e9)
        elif "dur" in e:
            durations_ms.append(float(e["dur"]) / 1e3)
    return durations_ms


def multiple_iteration_timeit_from_trace(
    compute_func,
    data_generator,
    task: str,
    tries: int = 5,
    trace_root: str = "/tmp/sgl_moe_trace",
) -> list[float]:
    """
    Profile multiple iterations and pull per-iteration kernel time from trace.
    """
    trace_name = f"{task}_" + "".join(random.choices(string.ascii_uppercase + string.digits, k=6))
    trace_dir = os.path.join(trace_root, trace_name)
    os.makedirs(trace_dir, exist_ok=True)

    with jax.profiler.trace(trace_dir):
        for i in range(tries):
            data_args = data_generator()
            with jax.profiler.StepTraceAnnotation(task, step_num=i):
                with jax.named_scope(f"{MARKER}_{i}"):
                    out = compute_func(*data_args)
                    jax.block_until_ready(out)

    trace = _load_trace(trace_dir)
    return _extract_marker_durations_ms(trace, task=task)


def build_mesh(ep_size: int = 1):
    return create_device_mesh(
        ici_parallelism=[-1, ep_size],
        dcn_parallelism=[1, 1],
    )
