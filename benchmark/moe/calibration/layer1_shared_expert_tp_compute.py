"""Layer 1 shared-expert tensor-parallel feasibility calibration."""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal

from benchmark.moe.calibration.common import build_observation_row

SCENARIO_LAYER1_SHARED_EXPERT_TP_COMPUTE = "layer1_shared_expert_tp_compute"
SUITE_V7X8_BF16_SHARED_EXPERT_TP_DECODE64 = "v7x8_bf16_shared_expert_tp_decode64"
SUPPORTED_SUITES = (SUITE_V7X8_BF16_SHARED_EXPERT_TP_DECODE64,)

STATUS_MEASURED = "measured"
STATUS_NOT_IMPLEMENTED = "not_implemented"
DEFAULT_WARMUP_RUNS = 3
DEFAULT_SAMPLE_RUNS = 5
DEFAULT_TRACE_DISCARD_RUNS = 1

DTYPE = "bfloat16"
WEIGHT_DTYPE = "bfloat16"
BF16_BYTES = 2
F32_BYTES = 4
T_PACKING = 2
HIDDEN_SIZE = 8192
INTERMEDIATE_SIZE = 2048
KERNEL_PATH = "python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py"

SharedExpertTPPath = Literal[
    "se_tp_compute_only",
    "se_tp_psum_only",
    "se_tp_compute_plus_psum",
]


@dataclass(frozen=True)
class SharedExpertTPShape:
    path: SharedExpertTPPath
    path_class: str
    num_tokens: int
    bt: int
    tp_size: int
    config_label: str
    hidden_size: int = HIDDEN_SIZE
    intermediate_size: int = INTERMEDIATE_SIZE
    t_packing: int = T_PACKING


def build_rows(
    *,
    suite: str,
    shapes: Iterable[SharedExpertTPShape],
    execution_mode: str,
    runtime: dict[str, Any],
    dtype: str,
    weight_dtype: str,
    t_packing: int,
    source: dict[str, Any],
    metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    if suite not in SUPPORTED_SUITES:
        raise ValueError(f"Unsupported Layer 1 shared expert TP suite: {suite}")
    if dtype != DTYPE or weight_dtype != WEIGHT_DTYPE or t_packing != T_PACKING:
        raise ValueError("layer1_shared_expert_tp_compute supports only bf16 with t_packing=2.")

    rows: list[dict[str, Any]] = []
    if execution_mode != "jax_trace" or runtime.get("default_backend") != "tpu":
        return [
            _make_row(
                suite=suite,
                shape=shape,
                execution_mode=execution_mode,
                runtime=runtime,
                dtype=dtype,
                weight_dtype=weight_dtype,
                t_packing=t_packing,
                source=source,
                metadata=metadata,
                status=STATUS_NOT_IMPLEMENTED,
                latency_ms_samples=[],
                implementation_note=_not_implemented_note(execution_mode, runtime),
            )
            for shape in shapes
        ]

    warmup_runs = _positive_int_env("CALIBRATION_LAYER1_SHARED_TP_WARMUP_RUNS", DEFAULT_WARMUP_RUNS)
    sample_runs = _positive_int_env("CALIBRATION_LAYER1_SHARED_TP_SAMPLE_RUNS", DEFAULT_SAMPLE_RUNS)
    discard_runs = _nonnegative_int_env(
        "CALIBRATION_LAYER1_SHARED_TP_TRACE_DISCARD_RUNS", DEFAULT_TRACE_DISCARD_RUNS
    )
    trace_root = os.getenv(
        "CALIBRATION_LAYER1_SHARED_TP_TRACE_ROOT", "/tmp/sglang_jax_layer1_shared_tp"
    )

    for shape in shapes:
        _validate_shape(shape)
        try:
            samples = _measure_shared_tp_ms(
                shape,
                warmup_runs=warmup_runs,
                sample_runs=sample_runs,
                discard_runs=discard_runs,
                trace_root=trace_root,
            )
            status = STATUS_MEASURED
            note = (
                "Measured shared-expert tensor-parallel feasibility with JAX shard_map. "
                "It splits the shared intermediate dimension across tensor ranks, keeps "
                "tokens replicated, and optionally psums the f32 partial output. It "
                "excludes fused-MoE routing, remote DMA, and shared weight prefetch."
            )
        except Exception as exc:
            samples = []
            status = STATUS_NOT_IMPLEMENTED
            note = f"Layer1 shared expert TP measurement failed: {type(exc).__name__}: {exc}"

        rows.append(
            _make_row(
                suite=suite,
                shape=shape,
                execution_mode=execution_mode,
                runtime=runtime,
                dtype=dtype,
                weight_dtype=weight_dtype,
                t_packing=t_packing,
                source=source,
                metadata=metadata,
                status=status,
                latency_ms_samples=samples,
                implementation_note=note,
            )
        )
    return rows


def _make_row(
    *,
    suite: str,
    shape: SharedExpertTPShape,
    execution_mode: str,
    runtime: dict[str, Any],
    dtype: str,
    weight_dtype: str,
    t_packing: int,
    source: dict[str, Any],
    metadata: dict[str, Any],
    status: str,
    latency_ms_samples: list[float],
    implementation_note: str,
) -> dict[str, Any]:
    return build_observation_row(
        scenario=SCENARIO_LAYER1_SHARED_EXPERT_TP_COMPUTE,
        suite=suite,
        layer=1,
        path=shape.path,
        path_class=shape.path_class,
        dtype=dtype,
        weight_dtype=weight_dtype,
        t_packing=t_packing,
        bf=shape.bt,
        bd=shape.hidden_size,
        tile_shape=_tile_shape(shape),
        bytes_hbm=_bytes_hbm(shape),
        bytes_per_fetch=_bytes_hbm(shape),
        dma_count=0,
        status=status,
        execution_mode=execution_mode,
        latency_ms_samples=latency_ms_samples,
        runtime=runtime,
        source=source,
        metadata=_metadata_for_shape(metadata, shape, status=status),
        implementation_note=implementation_note,
    )


def _metadata_for_shape(
    metadata: dict[str, Any], shape: SharedExpertTPShape, *, status: str
) -> dict[str, Any]:
    enriched = dict(metadata)
    shard_intermediate = shape.intermediate_size // shape.tp_size
    enriched["shared_expert_tp_compute"] = {
        "kernel_path": KERNEL_PATH,
        "kernel_reference": f"{KERNEL_PATH}:run_shared_expert_slice",
        "path": shape.path,
        "path_class": shape.path_class,
        "config_label": shape.config_label,
        "num_tokens": shape.num_tokens,
        "bt": shape.bt,
        "tp_size": shape.tp_size,
        "hidden_size": shape.hidden_size,
        "intermediate_size": shape.intermediate_size,
        "shard_intermediate": shard_intermediate,
        "t_packing": shape.t_packing,
        "tile_shape": _tile_shape(shape),
        "flops_per_rank": _flops_per_rank(shape),
        "global_flops": _global_flops(shape),
        "psum_payload_bytes_per_rank": _psum_payload_bytes(shape),
        "measurement_status": "measured" if status == STATUS_MEASURED else "schema_only",
        "measurement_scope": _measurement_scope(shape),
        "includes": _includes(shape),
        "excludes": [
            "shared_expert_weight_hbm_to_vmem_prefetch",
            "shared_expert_token_hbm_to_vmem_prefetch",
            "routed_expert_ffn",
            "remote_dma",
            "metadata_allgather",
            "scatter",
            "gather",
        ],
        "tp_contract": {
            "tokens": "replicated",
            "w1_w3": "intermediate-output sharded on tensor axis",
            "w2": "intermediate-input sharded on tensor axis",
            "output_partial": "f32 [bt, hidden] per tensor rank",
            "merge": "lax.psum over tensor axis for full output",
        },
    }
    return enriched


def _measure_shared_tp_ms(
    shape: SharedExpertTPShape,
    *,
    warmup_runs: int,
    sample_runs: int,
    discard_runs: int,
    trace_root: str,
) -> list[float]:
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P

    from benchmark.utils import multiple_iteration_timeit_from_trace

    devices = jax.devices()[: shape.tp_size]
    if len(devices) < shape.tp_size:
        raise ValueError(f"Need {shape.tp_size} devices, found {len(devices)}.")
    mesh = Mesh(np.asarray(devices).reshape(shape.tp_size), ("tensor",))

    tokens, w1, w3, w2, partial = _make_inputs(
        shape,
        jax=jax,
        jnp=jnp,
        mesh=mesh,
        P=P,
        NamedSharding=NamedSharding,
    )
    jax.block_until_ready((tokens, w1, w3, w2, partial))

    @jax.jit
    def run_shared(tokens_arg, w1_arg, w3_arg, w2_arg, partial_arg):
        with jax.named_scope("SGLANG_JAX_LAYER1_SHARED_EXPERT_TP"):
            return _sharded_shared_tp_call(
                tokens_arg,
                w1_arg,
                w3_arg,
                w2_arg,
                partial_arg,
                shape=shape,
                mesh=mesh,
                jax=jax,
                jnp=jnp,
                P=P,
            )

    jax.block_until_ready(run_shared(tokens, w1, w3, w2, partial))
    task = f"layer1_shared_expert_tp_{shape.path}_tp{shape.tp_size}_{shape.config_label}"
    return multiple_iteration_timeit_from_trace(
        compute_func=run_shared,
        data_generator=lambda: (tokens, w1, w3, w2, partial),
        task=task,
        tries=sample_runs,
        warmup=warmup_runs,
        discard_initial_samples=discard_runs,
        trace_root=trace_root,
    )


def _make_inputs(
    shape: SharedExpertTPShape,
    *,
    jax: Any,
    jnp: Any,
    mesh: Any,
    P: Any,
    NamedSharding: Any,
):
    tokens = jnp.ones((shape.bt, shape.hidden_size), dtype=jnp.bfloat16)
    w1 = jnp.ones((shape.hidden_size, shape.intermediate_size), dtype=jnp.bfloat16)
    w3 = jnp.ones((shape.hidden_size, shape.intermediate_size), dtype=jnp.bfloat16)
    w2 = jnp.ones((shape.intermediate_size, shape.hidden_size), dtype=jnp.bfloat16)
    partial = jnp.ones((shape.bt, shape.hidden_size), dtype=jnp.float32)
    return (
        jax.device_put(tokens, NamedSharding(mesh, P())),
        jax.device_put(w1, NamedSharding(mesh, P(None, "tensor"))),
        jax.device_put(w3, NamedSharding(mesh, P(None, "tensor"))),
        jax.device_put(w2, NamedSharding(mesh, P("tensor", None))),
        jax.device_put(partial, NamedSharding(mesh, P())),
    )


def _sharded_shared_tp_call(
    tokens,
    w1,
    w3,
    w2,
    partial,
    *,
    shape: SharedExpertTPShape,
    mesh: Any,
    jax: Any,
    jnp: Any,
    P: Any,
):
    in_specs = (P(), P(None, "tensor"), P(None, "tensor"), P("tensor", None), P())
    out_specs = P() if shape.path != "se_tp_compute_only" else P("tensor", None)

    @jax.shard_map(mesh=mesh, in_specs=in_specs, out_specs=out_specs, check_vma=False)
    def per_rank(tokens_local, w1_local, w3_local, w2_local, partial_replicated):
        if shape.path == "se_tp_psum_only":
            return jax.lax.psum(partial_replicated, "tensor")

        gate = jnp.dot(tokens_local, w1_local, preferred_element_type=jnp.float32)
        up = jnp.dot(tokens_local, w3_local, preferred_element_type=jnp.float32)
        act = _jax_silu(gate, jnp=jnp) * up
        out_partial = jnp.dot(act, w2_local, preferred_element_type=jnp.float32)
        if shape.path == "se_tp_compute_only":
            return out_partial
        return jax.lax.psum(out_partial, "tensor")

    return per_rank(tokens, w1, w3, w2, partial)


def _tile_shape(shape: SharedExpertTPShape) -> tuple[int, ...]:
    shard_intermediate = shape.intermediate_size // shape.tp_size
    if shape.path == "se_tp_psum_only":
        return (shape.bt, shape.hidden_size, shape.tp_size)
    return (
        shape.bt,
        shape.hidden_size,
        shard_intermediate,
        shape.bt,
        shard_intermediate,
        shape.hidden_size,
        shape.tp_size,
    )


def _bytes_hbm(shape: SharedExpertTPShape) -> int:
    if shape.path == "se_tp_psum_only":
        return shape.bt * shape.hidden_size * F32_BYTES * 2
    shard_intermediate = shape.intermediate_size // shape.tp_size
    token_bytes = shape.bt * shape.hidden_size * BF16_BYTES
    w1_w3_bytes = 2 * shape.hidden_size * shard_intermediate * BF16_BYTES
    w2_bytes = shard_intermediate * shape.hidden_size * BF16_BYTES
    output_bytes = shape.bt * shape.hidden_size * F32_BYTES
    if shape.path == "se_tp_compute_plus_psum":
        output_bytes *= 2
    return token_bytes + w1_w3_bytes + w2_bytes + output_bytes


def _flops_per_rank(shape: SharedExpertTPShape) -> int:
    if shape.path == "se_tp_psum_only":
        return 0
    shard_intermediate = shape.intermediate_size // shape.tp_size
    gate_up = 4 * shape.bt * shape.hidden_size * shard_intermediate
    down = 2 * shape.bt * shard_intermediate * shape.hidden_size
    return gate_up + down


def _global_flops(shape: SharedExpertTPShape) -> int:
    return _flops_per_rank(shape) * shape.tp_size


def _psum_payload_bytes(shape: SharedExpertTPShape) -> int:
    if shape.path == "se_tp_compute_only":
        return 0
    return shape.bt * shape.hidden_size * F32_BYTES


def _includes(shape: SharedExpertTPShape) -> list[str]:
    if shape.path == "se_tp_psum_only":
        return ["f32_partial_output_psum_over_tensor_axis"]
    if shape.path == "se_tp_compute_only":
        return [
            "replicated_token_read",
            "tp_sharded_gate_up_dot",
            "shared_expert_activation",
            "tp_sharded_down_dot",
        ]
    return [
        "replicated_token_read",
        "tp_sharded_gate_up_dot",
        "shared_expert_activation",
        "tp_sharded_down_dot",
        "f32_partial_output_psum_over_tensor_axis",
    ]


def _measurement_scope(shape: SharedExpertTPShape) -> str:
    if shape.path == "se_tp_psum_only":
        return "shared_expert_tp_output_psum_only"
    if shape.path == "se_tp_compute_only":
        return "shared_expert_tp_compute_without_output_psum"
    return "shared_expert_tp_compute_with_output_psum"


def _jax_silu(x, *, jnp: Any):
    return x * jnp.reciprocal(1.0 + jnp.exp(-x))


def _validate_shape(shape: SharedExpertTPShape) -> None:
    if shape.bt <= 0 or shape.tp_size <= 0:
        raise ValueError(f"Expected positive bt/tp_size, got {shape.bt=} {shape.tp_size=}.")
    if shape.hidden_size != HIDDEN_SIZE:
        raise ValueError(f"Expected hidden_size={HIDDEN_SIZE}, got {shape.hidden_size}.")
    if shape.intermediate_size != INTERMEDIATE_SIZE:
        raise ValueError(
            f"Expected intermediate_size={INTERMEDIATE_SIZE}, got {shape.intermediate_size}."
        )
    if shape.t_packing != T_PACKING:
        raise ValueError(f"Expected t_packing={T_PACKING}, got {shape.t_packing}.")
    if shape.intermediate_size % shape.tp_size != 0:
        raise ValueError("intermediate_size must be divisible by tp_size.")
    shard_intermediate = shape.intermediate_size // shape.tp_size
    if shard_intermediate % 128 != 0:
        raise ValueError("intermediate shard must keep TPU dot dimension aligned to 128.")


def _positive_int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if not value:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def _nonnegative_int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if not value:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return parsed if parsed >= 0 else default


def _not_implemented_note(execution_mode: str, runtime: dict[str, Any]) -> str:
    if execution_mode == "local_smoke":
        return "layer1_shared_expert_tp_compute emitted schema-only rows on local_smoke."
    return (
        "layer1_shared_expert_tp_compute requires TPU jax_trace execution; "
        f"observed mode={execution_mode}, backend={runtime.get('default_backend')!r}."
    )
