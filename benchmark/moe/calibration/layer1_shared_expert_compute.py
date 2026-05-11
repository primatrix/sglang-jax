"""Layer 1 fused-MoE shared-expert slice calibration."""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal

from benchmark.moe.calibration.common import build_observation_row

SCENARIO_LAYER1_SHARED_EXPERT_COMPUTE = "layer1_shared_expert_compute"
SUITE_V7X8_BF16_SHARED_EXPERT_TUNED_FAMILY = "v7x8_bf16_shared_expert_tuned_family"
SUPPORTED_SUITES = (SUITE_V7X8_BF16_SHARED_EXPERT_TUNED_FAMILY,)

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
KERNEL_PATH = "python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py"

SharedExpertPath = Literal["shared_expert_slice"]


@dataclass(frozen=True)
class SharedExpertShape:
    path: SharedExpertPath
    path_class: str
    num_tokens: int
    bt: int
    bse: int
    bd1: int
    bd2: int
    config_label: str
    hidden_size: int = HIDDEN_SIZE
    t_packing: int = T_PACKING


def build_rows(
    *,
    suite: str,
    shapes: Iterable[SharedExpertShape],
    execution_mode: str,
    runtime: dict[str, Any],
    dtype: str,
    weight_dtype: str,
    t_packing: int,
    source: dict[str, Any],
    metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    if suite not in SUPPORTED_SUITES:
        raise ValueError(f"Unsupported Layer 1 shared expert suite: {suite}")
    if dtype != DTYPE or weight_dtype != WEIGHT_DTYPE or t_packing != T_PACKING:
        raise ValueError("layer1_shared_expert_compute supports only bf16 with t_packing=2.")

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

    warmup_runs = _positive_int_env("CALIBRATION_LAYER1_SHARED_WARMUP_RUNS", DEFAULT_WARMUP_RUNS)
    sample_runs = _positive_int_env("CALIBRATION_LAYER1_SHARED_SAMPLE_RUNS", DEFAULT_SAMPLE_RUNS)
    discard_runs = _nonnegative_int_env(
        "CALIBRATION_LAYER1_SHARED_TRACE_DISCARD_RUNS", DEFAULT_TRACE_DISCARD_RUNS
    )
    trace_root = os.getenv("CALIBRATION_LAYER1_SHARED_TRACE_ROOT", "/tmp/sglang_jax_layer1_shared")

    for shape in shapes:
        _validate_shape(shape)
        try:
            samples = _measure_shared_ms(
                shape,
                warmup_runs=warmup_runs,
                sample_runs=sample_runs,
                discard_runs=discard_runs,
                trace_root=trace_root,
            )
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
                    status=STATUS_MEASURED,
                    latency_ms_samples=samples,
                    implementation_note=(
                        "Measured one fused-MoE shared-expert block compute slice with JAX trace "
                        "timing: W1/W3 gate/up, activation, W2 down projection, and f32 output "
                        "accumulation. It excludes async weight/token prefetch."
                    ),
                )
            )
        except Exception as exc:
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
                    status=STATUS_NOT_IMPLEMENTED,
                    latency_ms_samples=[],
                    implementation_note=f"Layer1 shared expert measurement failed: {type(exc).__name__}: {exc}",
                )
            )
    return rows


def _make_row(
    *,
    suite: str,
    shape: SharedExpertShape,
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
        scenario=SCENARIO_LAYER1_SHARED_EXPERT_COMPUTE,
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
    metadata: dict[str, Any], shape: SharedExpertShape, *, status: str
) -> dict[str, Any]:
    enriched = dict(metadata)
    enriched["shared_expert_compute"] = {
        "kernel_path": KERNEL_PATH,
        "kernel_reference": f"{KERNEL_PATH}:run_shared_expert_slice",
        "path": shape.path,
        "path_class": shape.path_class,
        "config_label": shape.config_label,
        "num_tokens": shape.num_tokens,
        "bt": shape.bt,
        "bse": shape.bse,
        "hidden_size": shape.hidden_size,
        "t_packing": shape.t_packing,
        "h_per_t_packing": shape.hidden_size // shape.t_packing,
        "bd1": shape.bd1,
        "bd2": shape.bd2,
        "num_bd1": _ceil_div(shape.hidden_size, shape.bd1),
        "num_bd2": _ceil_div(shape.hidden_size, shape.bd2),
        "tile_shape": _tile_shape(shape),
        "flops": _flops(shape),
        "bytes_hbm": _bytes_hbm(shape),
        "traffic_class": "shared_expert_compute_slice",
        "measurement_status": "measured" if status == STATUS_MEASURED else "schema_only",
        "includes": [
            "shared_expert_gate_dot",
            "shared_expert_up_dot",
            "shared_expert_activation",
            "shared_expert_down_dot",
            "shared_expert_f32_accumulate",
        ],
        "excludes": [
            "shared_expert_weight_hbm_to_vmem_prefetch",
            "shared_expert_token_hbm_to_vmem_prefetch",
            "routed_expert_ffn",
            "remote_dma",
        ],
    }
    return enriched


def _measure_shared_ms(
    shape: SharedExpertShape,
    *,
    warmup_runs: int,
    sample_runs: int,
    discard_runs: int,
    trace_root: str,
) -> list[float]:
    import jax
    import jax.numpy as jnp

    from benchmark.utils import multiple_iteration_timeit_from_trace

    inputs = _make_inputs(shape, jnp=jnp)
    jax.block_until_ready(inputs)

    @jax.jit
    def run_shared(*arrays):
        with jax.named_scope("SGLANG_JAX_LAYER1_SHARED_EXPERT"):
            return _shared_loop(shape, arrays, jnp=jnp)

    jax.block_until_ready(run_shared(*inputs))
    task = f"layer1_shared_expert_{shape.config_label}"
    return multiple_iteration_timeit_from_trace(
        compute_func=run_shared,
        data_generator=lambda: inputs,
        task=task,
        tries=sample_runs,
        warmup=warmup_runs,
        discard_initial_samples=discard_runs,
        trace_root=trace_root,
    )


def _make_inputs(shape: SharedExpertShape, *, jnp: Any) -> tuple[Any, ...]:
    hpt = shape.hidden_size // shape.t_packing
    bd1_per_pack = shape.bd1 // shape.t_packing
    bd2_per_pack = shape.bd2 // shape.t_packing
    num_bd1 = _ceil_div(shape.hidden_size, shape.bd1)
    num_bd2 = _ceil_div(shape.hidden_size, shape.bd2)
    tokens = jnp.ones((shape.bt, shape.t_packing, hpt), dtype=jnp.bfloat16)
    w1 = jnp.ones((shape.t_packing, num_bd1, bd1_per_pack, shape.bse), dtype=jnp.bfloat16)
    w3 = jnp.ones((shape.t_packing, num_bd1, bd1_per_pack, shape.bse), dtype=jnp.bfloat16)
    w2 = jnp.ones((shape.t_packing, num_bd2, shape.bse, bd2_per_pack), dtype=jnp.bfloat16)
    out = jnp.ones((shape.bt, shape.t_packing, hpt), dtype=jnp.float32)
    return tokens, w1, w3, w2, out


def _shared_loop(shape: SharedExpertShape, arrays: tuple[Any, ...], *, jnp: Any):
    tokens, w1, w3, w2, out = arrays
    bd1_per_pack = shape.bd1 // shape.t_packing
    bd2_per_pack = shape.bd2 // shape.t_packing
    num_bd1 = _ceil_div(shape.hidden_size, shape.bd1)
    num_bd2 = _ceil_div(shape.hidden_size, shape.bd2)
    gate = jnp.zeros((shape.bt, shape.bse), dtype=jnp.float32)
    up = jnp.zeros((shape.bt, shape.bse), dtype=jnp.float32)

    for bd1_id in range(num_bd1):
        for p_id in range(shape.t_packing):
            k_start = bd1_id * bd1_per_pack
            lhs = tokens[:, p_id, k_start : k_start + bd1_per_pack]
            gate += jnp.dot(lhs, w1[p_id, bd1_id], preferred_element_type=jnp.float32)
            up += jnp.dot(lhs, w3[p_id, bd1_id], preferred_element_type=jnp.float32)

    act = jax_silu(gate, jnp=jnp) * up
    for bd2_id in range(num_bd2):
        for p_id in range(shape.t_packing):
            k_start = bd2_id * bd2_per_pack
            down = jnp.dot(act, w2[p_id, bd2_id], preferred_element_type=jnp.float32)
            out = out.at[:, p_id, k_start : k_start + bd2_per_pack].add(down)
    return out


def jax_silu(x, *, jnp: Any):
    return x * jnp.reciprocal(1.0 + jnp.exp(-x))


def _tile_shape(shape: SharedExpertShape) -> tuple[int, ...]:
    return (shape.bt, shape.t_packing, shape.hidden_size // shape.t_packing, shape.bse)


def _bytes_hbm(shape: SharedExpertShape) -> int:
    token_bytes = shape.bt * shape.hidden_size * BF16_BYTES
    w1w3_bytes = 2 * shape.hidden_size * shape.bse * BF16_BYTES
    w2_bytes = shape.bse * shape.hidden_size * BF16_BYTES
    out_bytes = shape.bt * shape.hidden_size * F32_BYTES
    return token_bytes + w1w3_bytes + w2_bytes + out_bytes


def _flops(shape: SharedExpertShape) -> int:
    return 6 * shape.bt * shape.hidden_size * shape.bse


def _validate_shape(shape: SharedExpertShape) -> None:
    if shape.bt <= 0 or shape.bse <= 0:
        raise ValueError(f"Expected positive bt/bse, got {shape.bt=} {shape.bse=}.")
    if shape.hidden_size != HIDDEN_SIZE:
        raise ValueError(f"Expected hidden_size={HIDDEN_SIZE}, got {shape.hidden_size}.")
    if shape.t_packing != T_PACKING:
        raise ValueError(f"Expected t_packing={T_PACKING}, got {shape.t_packing}.")
    if shape.bd1 % shape.t_packing != 0 or shape.bd2 % shape.t_packing != 0:
        raise ValueError("bd1/bd2 must be divisible by t_packing.")


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


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
        return "layer1_shared_expert_compute emitted schema-only rows on local_smoke."
    return (
        "layer1_shared_expert_compute requires TPU jax_trace execution; "
        f"observed mode={execution_mode}, backend={runtime.get('default_backend')!r}."
    )
