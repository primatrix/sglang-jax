"""Layer 1 fused-MoE FFN loop-context compute calibration.

This scenario measures the compute structure inside the fused-MoE expert loop
without remote DMA or weight prefetch. It intentionally uses JAX trace timing,
not Pallas, so the rows represent the dynamic FFN loop structure rather than a
generic Layer0 GEMM point.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal

from benchmark.moe.calibration.common import build_observation_row

SCENARIO_LAYER1_FFN_COMPUTE = "layer1_ffn_compute"
SUITE_V7X8_BF16_FFN_LOOP_CONTEXT = "v7x8_bf16_ffn_loop_context"
SUPPORTED_SUITES = (SUITE_V7X8_BF16_FFN_LOOP_CONTEXT,)

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
BD1 = 1024
BD2 = 1024
BF = 512
BFC = 512
BTC = 16
KERNEL_PATH = "python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py"

FFNPath = Literal[
    "dynamic_ffn1_init",
    "dynamic_ffn1_accumulate",
    "dynamic_ffn2_first",
    "dynamic_ffn2_later",
]


@dataclass(frozen=True)
class FFNComputeShape:
    path: FFNPath
    path_class: str
    dyn_sz: int
    hidden_size: int = HIDDEN_SIZE
    intermediate_size: int = INTERMEDIATE_SIZE
    t_packing: int = T_PACKING
    bd1: int = BD1
    bd2: int = BD2
    bf: int = BF
    bfc: int = BFC
    btc: int = BTC


def build_rows(
    *,
    suite: str,
    shapes: Iterable[FFNComputeShape],
    execution_mode: str,
    runtime: dict[str, Any],
    dtype: str,
    weight_dtype: str,
    t_packing: int,
    source: dict[str, Any],
    metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    if suite not in SUPPORTED_SUITES:
        raise ValueError(f"Unsupported Layer 1 FFN suite: {suite}")
    if dtype != DTYPE or weight_dtype != WEIGHT_DTYPE or t_packing != T_PACKING:
        raise ValueError("layer1_ffn_compute supports only bf16 tokens/weights with t_packing=2.")

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

    warmup_runs = _positive_int_env("CALIBRATION_LAYER1_FFN_WARMUP_RUNS", DEFAULT_WARMUP_RUNS)
    sample_runs = _positive_int_env("CALIBRATION_LAYER1_FFN_SAMPLE_RUNS", DEFAULT_SAMPLE_RUNS)
    discard_runs = _nonnegative_int_env(
        "CALIBRATION_LAYER1_FFN_TRACE_DISCARD_RUNS", DEFAULT_TRACE_DISCARD_RUNS
    )
    trace_root = os.getenv("CALIBRATION_LAYER1_FFN_TRACE_ROOT", "/tmp/sglang_jax_layer1_ffn")

    for shape in shapes:
        _validate_shape(shape)
        try:
            samples = _measure_ffn_ms(
                shape,
                warmup_runs=warmup_runs,
                sample_runs=sample_runs,
                discard_runs=discard_runs,
                trace_root=trace_root,
            )
            status = STATUS_MEASURED
            note = (
                "Measured fused-MoE dynamic FFN loop-context compute with JAX trace timing: "
                "t_packing=2, bf/bfc/btc blocking, FFN1 dual gate/up dot or FFN2 activation+down dot, "
                "and init/accumulate semantics. It excludes remote DMA and weight prefetch."
            )
        except Exception as exc:
            samples = []
            status = STATUS_NOT_IMPLEMENTED
            note = f"Layer1 FFN measurement failed: {type(exc).__name__}: {exc}"

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
    shape: FFNComputeShape,
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
        scenario=SCENARIO_LAYER1_FFN_COMPUTE,
        suite=suite,
        layer=1,
        path=shape.path,
        path_class=shape.path_class,
        dtype=dtype,
        weight_dtype=weight_dtype,
        t_packing=t_packing,
        bf=shape.dyn_sz,
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
    metadata: dict[str, Any], shape: FFNComputeShape, *, status: str
) -> dict[str, Any]:
    enriched = dict(metadata)
    enriched["ffn_compute"] = {
        "kernel_path": KERNEL_PATH,
        "kernel_reference": _kernel_reference(shape),
        "path": shape.path,
        "path_class": shape.path_class,
        "dyn_sz": shape.dyn_sz,
        "hidden_size": shape.hidden_size,
        "intermediate_size": shape.intermediate_size,
        "t_packing": shape.t_packing,
        "h_per_t_packing": shape.hidden_size // shape.t_packing,
        "bd1": shape.bd1,
        "bd2": shape.bd2,
        "bf": shape.bf,
        "bfc": shape.bfc,
        "btc": shape.btc,
        "tile_shape": _tile_shape(shape),
        "flops": _flops(shape),
        "bytes_hbm": _bytes_hbm(shape),
        "traffic_class": "ffn_loop_context_compute",
        "measurement_status": "measured" if status == STATUS_MEASURED else "schema_only",
        "includes": _includes(shape),
        "excludes": [
            "remote_dma",
            "metadata_allgather",
            "scatter",
            "gather",
            "weight_hbm_to_vmem_prefetch",
            "local_accumulator_hbm_rmw",
        ],
        "loop_context": {
            "num_token_tiles": _ceil_div(shape.dyn_sz, shape.btc) if shape.dyn_sz > 0 else 0,
            "num_bd1": _ceil_div(shape.hidden_size, shape.bd1),
            "num_bd2": _ceil_div(shape.hidden_size, shape.bd2),
            "num_bf": _ceil_div(shape.intermediate_size, shape.bf),
            "num_bfc": _ceil_div(shape.bf, shape.bfc),
        },
    }
    return enriched


def _measure_ffn_ms(
    shape: FFNComputeShape,
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
    def run_ffn(*arrays):
        if shape.path in ("dynamic_ffn1_init", "dynamic_ffn1_accumulate"):
            return _ffn1_loop(shape, arrays, jnp=jnp)
        return _ffn2_loop(shape, arrays, jnp=jnp)

    jax.block_until_ready(run_ffn(*inputs))
    task = f"layer1_ffn_{shape.path}_dyn{shape.dyn_sz}"
    return multiple_iteration_timeit_from_trace(
        compute_func=run_ffn,
        data_generator=lambda: inputs,
        task=task,
        tries=sample_runs,
        warmup=warmup_runs,
        discard_initial_samples=discard_runs,
        trace_root=trace_root,
    )


def _make_inputs(shape: FFNComputeShape, *, jnp: Any) -> tuple[Any, ...]:
    dyn = max(shape.dyn_sz, 1)
    hpt = shape.hidden_size // shape.t_packing
    if shape.path in ("dynamic_ffn1_init", "dynamic_ffn1_accumulate"):
        tokens = jnp.ones((dyn, shape.t_packing, hpt), dtype=jnp.bfloat16)
        w1 = jnp.ones((shape.t_packing, hpt, shape.bf), dtype=jnp.bfloat16)
        w3 = jnp.ones((shape.t_packing, hpt, shape.bf), dtype=jnp.bfloat16)
        acc1 = jnp.ones((dyn, shape.bf), dtype=jnp.float32)
        acc3 = jnp.ones((dyn, shape.bf), dtype=jnp.float32)
        return (tokens, w1, w3, acc1, acc3)
    acc1 = jnp.ones((dyn, shape.bf), dtype=jnp.float32)
    acc3 = jnp.ones((dyn, shape.bf), dtype=jnp.float32)
    w2 = jnp.ones((shape.t_packing, shape.bf, hpt), dtype=jnp.bfloat16)
    res = jnp.ones((dyn, shape.t_packing, hpt), dtype=jnp.bfloat16)
    return (acc1, acc3, w2, res)


def _ffn1_loop(shape: FFNComputeShape, arrays: tuple[Any, ...], *, jnp: Any):
    tokens, w1, w3, acc1_init, acc3_init = arrays
    dyn = max(shape.dyn_sz, 1)
    num_loops = _ceil_div(shape.dyn_sz, shape.btc) if shape.dyn_sz > 0 else 0
    should_init = shape.path == "dynamic_ffn1_init"
    acc1_out = acc1_init
    acc3_out = acc3_init
    for btc_id in range(num_loops):
        start = btc_id * shape.btc
        token_count = min(shape.btc, dyn - start)
        token_count = max(token_count, 1)

        for p_id in range(shape.t_packing):
            lhs = tokens[start : start + token_count, p_id, :]
            d1 = jnp.dot(lhs, w1[p_id], preferred_element_type=jnp.float32)
            d3 = jnp.dot(lhs, w3[p_id], preferred_element_type=jnp.float32)
            old1 = acc1_out[start : start + token_count, :]
            old3 = acc3_out[start : start + token_count, :]
            new1 = d1 if should_init and p_id == 0 else old1 + d1
            new3 = d3 if should_init and p_id == 0 else old3 + d3
            acc1_out = acc1_out.at[start : start + token_count, :].set(new1)
            acc3_out = acc3_out.at[start : start + token_count, :].set(new3)
    return acc1_out + acc3_out


def _ffn2_loop(shape: FFNComputeShape, arrays: tuple[Any, ...], *, jnp: Any):
    acc1, acc3, w2, res_init = arrays
    dyn = max(shape.dyn_sz, 1)
    num_loops = _ceil_div(shape.dyn_sz, shape.btc) if shape.dyn_sz > 0 else 0
    should_init = shape.path == "dynamic_ffn2_first"
    res_out = res_init
    for btc_id in range(num_loops):
        start = btc_id * shape.btc
        token_count = min(shape.btc, dyn - start)
        token_count = max(token_count, 1)
        act = (
            jax_silu(acc1[start : start + token_count, :], jnp=jnp)
            * acc3[start : start + token_count, :]
        )

        for p_id in range(shape.t_packing):
            d = jnp.dot(act, w2[p_id], preferred_element_type=jnp.float32).astype(jnp.bfloat16)
            old = res_out[start : start + token_count, p_id, :]
            new = (
                d
                if should_init
                else (old.astype(jnp.float32) + d.astype(jnp.float32)).astype(jnp.bfloat16)
            )
            res_out = res_out.at[start : start + token_count, p_id, :].set(new)
    return res_out


def jax_silu(x, *, jnp: Any):
    return x * jnp.reciprocal(1.0 + jnp.exp(-x))


def _tile_shape(shape: FFNComputeShape) -> tuple[int, ...]:
    if shape.path in ("dynamic_ffn1_init", "dynamic_ffn1_accumulate"):
        return (shape.dyn_sz, shape.t_packing, shape.hidden_size // shape.t_packing, shape.bf)
    return (shape.dyn_sz, shape.bf, shape.t_packing, shape.hidden_size // shape.t_packing)


def _bytes_hbm(shape: FFNComputeShape) -> int:
    dyn = max(shape.dyn_sz, 1)
    if shape.path in ("dynamic_ffn1_init", "dynamic_ffn1_accumulate"):
        token_bytes = dyn * shape.hidden_size * BF16_BYTES
        weight_bytes = 2 * shape.hidden_size * shape.bf * BF16_BYTES
        acc_bytes = 2 * dyn * shape.bf * F32_BYTES
        return token_bytes + weight_bytes + acc_bytes
    acc_bytes = 2 * dyn * shape.bf * F32_BYTES
    weight_bytes = shape.bf * shape.hidden_size * BF16_BYTES
    output_bytes = dyn * shape.hidden_size * BF16_BYTES
    return acc_bytes + weight_bytes + output_bytes


def _flops(shape: FFNComputeShape) -> int:
    dyn = shape.dyn_sz
    if shape.path in ("dynamic_ffn1_init", "dynamic_ffn1_accumulate"):
        return 4 * dyn * shape.hidden_size * shape.bf
    return 2 * dyn * shape.bf * shape.hidden_size


def _includes(shape: FFNComputeShape) -> list[str]:
    if shape.path in ("dynamic_ffn1_init", "dynamic_ffn1_accumulate"):
        return ["dynamic_ffn1_gate_dot", "dynamic_ffn1_up_dot", "acc1_acc3_update"]
    return ["silu_multiply_activation", "dynamic_ffn2_down_dot", "res_init_or_accumulate"]


def _kernel_reference(shape: FFNComputeShape) -> str:
    if shape.path in ("dynamic_ffn1_init", "dynamic_ffn1_accumulate"):
        return f"{KERNEL_PATH}:dynamic_ffn1/run_gate_up_slices"
    return f"{KERNEL_PATH}:dynamic_ffn2/run_down_slices"


def _validate_shape(shape: FFNComputeShape) -> None:
    if shape.dyn_sz < 0:
        raise ValueError(f"Expected nonnegative dyn_sz, got {shape.dyn_sz}.")
    if shape.hidden_size != HIDDEN_SIZE:
        raise ValueError(f"Expected hidden_size={HIDDEN_SIZE}, got {shape.hidden_size}.")
    if shape.intermediate_size != INTERMEDIATE_SIZE:
        raise ValueError(
            f"Expected intermediate_size={INTERMEDIATE_SIZE}, got {shape.intermediate_size}."
        )
    if shape.t_packing != T_PACKING:
        raise ValueError(f"Expected t_packing={T_PACKING}, got {shape.t_packing}.")
    if shape.hidden_size % shape.t_packing != 0:
        raise ValueError("hidden_size must be divisible by t_packing.")


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
        return "layer1_ffn_compute emitted schema-only rows on local_smoke."
    return (
        "layer1_ffn_compute requires TPU jax_trace execution; "
        f"execution_mode={execution_mode!r}, default_backend={runtime.get('default_backend')!r}."
    )
