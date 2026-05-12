"""Layer 1 fused-MoE Pallas FFN compute calibration.

This scenario measures a Pallas implementation of the routed expert FFN
compute loop. It is intentionally separate from layer1_ffn_compute, which uses
plain JAX loops and can overstate FFN2 costs after lowering.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal

from benchmark.moe.calibration.common import build_observation_row

SCENARIO_LAYER1_FFN_PALLAS_COMPUTE = "layer1_ffn_pallas_compute"
SUITE_V7X8_BF16_FFN_PALLAS_TUNED_FAMILY = "v7x8_bf16_ffn_pallas_tuned_family"
SUPPORTED_SUITES = (SUITE_V7X8_BF16_FFN_PALLAS_TUNED_FAMILY,)

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

FFNPallasPath = Literal[
    "dynamic_ffn1_init",
    "dynamic_ffn1_accumulate",
    "dynamic_ffn2_first",
    "dynamic_ffn2_later",
    "dynamic_ffn1_whole_init",
    "dynamic_ffn1_whole_accumulate",
    "dynamic_ffn2_whole_first",
    "dynamic_ffn2_whole_later",
]


@dataclass(frozen=True)
class FFNPallasShape:
    path: FFNPallasPath
    path_class: str
    dyn_sz: int
    config_label: str
    num_tokens: int
    bt: int
    bts: int
    bf: int
    bfc: int
    btc: int
    bd1: int
    bd2: int
    bd1c: int
    bd2c: int
    hidden_size: int = HIDDEN_SIZE
    intermediate_size: int = INTERMEDIATE_SIZE
    t_packing: int = T_PACKING


def build_rows(
    *,
    suite: str,
    shapes: Iterable[FFNPallasShape],
    execution_mode: str,
    runtime: dict[str, Any],
    dtype: str,
    weight_dtype: str,
    t_packing: int,
    source: dict[str, Any],
    metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    if suite not in SUPPORTED_SUITES:
        raise ValueError(f"Unsupported Layer 1 Pallas FFN suite: {suite}")
    if dtype != DTYPE or weight_dtype != WEIGHT_DTYPE or t_packing != T_PACKING:
        raise ValueError("layer1_ffn_pallas_compute supports only bf16 with t_packing=2.")

    rows: list[dict[str, Any]] = []
    if execution_mode != "pallas" or runtime.get("default_backend") != "tpu":
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

    warmup_runs = _positive_int_env(
        "CALIBRATION_LAYER1_FFN_PALLAS_WARMUP_RUNS", DEFAULT_WARMUP_RUNS
    )
    sample_runs = _positive_int_env(
        "CALIBRATION_LAYER1_FFN_PALLAS_SAMPLE_RUNS", DEFAULT_SAMPLE_RUNS
    )
    discard_runs = _nonnegative_int_env(
        "CALIBRATION_LAYER1_FFN_PALLAS_TRACE_DISCARD_RUNS",
        DEFAULT_TRACE_DISCARD_RUNS,
    )
    trace_root = os.getenv(
        "CALIBRATION_LAYER1_FFN_PALLAS_TRACE_ROOT",
        "/tmp/sglang_jax_layer1_ffn_pallas",
    )

    for shape in shapes:
        _validate_shape(shape)
        try:
            samples = _measure_ffn_pallas_ms(
                shape,
                warmup_runs=warmup_runs,
                sample_runs=sample_runs,
                discard_runs=discard_runs,
                trace_root=trace_root,
            )
            status = STATUS_MEASURED
            note = (
                "Measured fused-MoE routed expert FFN compute with a Pallas kernel aligned to "
                "dynamic_ffn1/dynamic_ffn2 loop structure. It excludes remote DMA, metadata, "
                "A2A scatter/gather, and async weight prefetch."
            )
        except Exception as exc:
            samples = []
            status = STATUS_NOT_IMPLEMENTED
            note = f"Layer1 Pallas FFN measurement failed: {type(exc).__name__}: {exc}"

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
    shape: FFNPallasShape,
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
        scenario=SCENARIO_LAYER1_FFN_PALLAS_COMPUTE,
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
    metadata: dict[str, Any], shape: FFNPallasShape, *, status: str
) -> dict[str, Any]:
    enriched = dict(metadata)
    enriched["ffn_pallas_compute"] = {
        "kernel_path": KERNEL_PATH,
        "kernel_reference": _kernel_reference(shape),
        "path": shape.path,
        "path_class": shape.path_class,
        "dyn_sz": shape.dyn_sz,
        "effective_m": _effective_rows(shape),
        "config_label": shape.config_label,
        "num_tokens": shape.num_tokens,
        "bt": shape.bt,
        "bts": shape.bts,
        "hidden_size": shape.hidden_size,
        "intermediate_size": shape.intermediate_size,
        "t_packing": shape.t_packing,
        "h_per_t_packing": shape.hidden_size // shape.t_packing,
        "bf": shape.bf,
        "bfc": shape.bfc,
        "btc": shape.btc,
        "bd1": shape.bd1,
        "bd2": shape.bd2,
        "bd1c": shape.bd1c,
        "bd2c": shape.bd2c,
        "tile_shape": _tile_shape(shape),
        "flops": _flops(shape),
        "bytes_hbm": _bytes_hbm(shape),
        "traffic_class": "ffn_pallas_compute",
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
            "measurement_scope": _measurement_scope(shape),
            "num_bts_tiles": _ceil_div(shape.dyn_sz, shape.bts) if shape.dyn_sz > 0 else 0,
            "num_btc_tiles": _ceil_div(shape.dyn_sz, shape.btc) if shape.dyn_sz > 0 else 0,
            "num_bd1": _ceil_div(shape.hidden_size, shape.bd1),
            "num_bd2": _ceil_div(shape.hidden_size, shape.bd2),
            "measured_bd1_tiles": _measured_bd1_tiles(shape),
            "measured_bd2_tiles": _measured_bd2_tiles(shape),
            "num_bf": _ceil_div(shape.intermediate_size, shape.bf),
            "num_bfc": _ceil_div(shape.bf, shape.bfc),
            "pallas_aligned": True,
        },
    }
    return enriched


def _measure_ffn_pallas_ms(
    shape: FFNPallasShape,
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
    def run_ffn_pallas(*arrays):
        with jax.named_scope("SGLANG_JAX_LAYER1_FFN_PALLAS"):
            return _pallas_ffn_call(shape, arrays, jax=jax, jnp=jnp)

    jax.block_until_ready(run_ffn_pallas(*inputs))
    task = f"layer1_ffn_pallas_{shape.path}_dyn{shape.dyn_sz}_{shape.config_label}"
    return multiple_iteration_timeit_from_trace(
        compute_func=run_ffn_pallas,
        data_generator=lambda: inputs,
        task=task,
        tries=sample_runs,
        warmup=warmup_runs,
        discard_initial_samples=discard_runs,
        trace_root=trace_root,
    )


def _make_inputs(shape: FFNPallasShape, *, jnp: Any) -> tuple[Any, ...]:
    dyn = _effective_rows(shape)
    h_per_pack = shape.hidden_size // shape.t_packing
    bd1_per_pack = shape.bd1 // shape.t_packing
    bd2_per_pack = shape.bd2 // shape.t_packing
    if _is_ffn1(shape):
        input_width = h_per_pack if _is_whole(shape) else bd1_per_pack
        tokens = jnp.ones((dyn, shape.t_packing, bd1_per_pack), dtype=jnp.bfloat16)
        if _is_whole(shape):
            tokens = jnp.ones((dyn, shape.t_packing, input_width), dtype=jnp.bfloat16)
        w1 = jnp.ones((shape.t_packing, input_width, shape.bf), dtype=jnp.bfloat16)
        w3 = jnp.ones((shape.t_packing, input_width, shape.bf), dtype=jnp.bfloat16)
        acc = jnp.ones((2, dyn, shape.bf), dtype=jnp.float32)
        return tokens, w1, w3, acc
    acc1 = jnp.ones((dyn, shape.bf), dtype=jnp.float32)
    acc3 = jnp.ones((dyn, shape.bf), dtype=jnp.float32)
    output_width = h_per_pack if _is_whole(shape) else bd2_per_pack
    w2 = jnp.ones((shape.t_packing, shape.bf, output_width), dtype=jnp.bfloat16)
    res = jnp.ones((dyn, shape.t_packing, output_width), dtype=jnp.bfloat16)
    return acc1, acc3, w2, res


def _pallas_ffn_call(shape: FFNPallasShape, arrays: tuple[Any, ...], *, jax: Any, jnp: Any):
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    dyn = _effective_rows(shape)
    h_per_pack = shape.hidden_size // shape.t_packing
    bd1_per_pack = shape.bd1 // shape.t_packing
    bd1c_per_pack = shape.bd1c // shape.t_packing
    bd2_per_pack = shape.bd2 // shape.t_packing
    bd2c_per_pack = shape.bd2c // shape.t_packing
    num_bd1 = _ceil_div(shape.hidden_size, shape.bd1)
    num_bd2 = _ceil_div(shape.hidden_size, shape.bd2)
    num_bd1c = _ceil_div(shape.bd1, shape.bd1c)
    num_bd2c = _ceil_div(shape.bd2, shape.bd2c)
    num_bfc = _ceil_div(shape.bf, shape.bfc)
    num_loops = _ceil_div(shape.dyn_sz, shape.btc) if shape.dyn_sz > 0 else 0
    num_token_tiles = _ceil_div(shape.dyn_sz, shape.bts) if shape.dyn_sz > 0 else 0

    if _is_ffn1(shape):
        tokens, w1, w3, acc = arrays
        should_init = shape.path in ("dynamic_ffn1_init", "dynamic_ffn1_whole_init")
        bd1_outer_loops = num_bd1 if _is_whole(shape) else 1

        def kernel(tokens_ref, w1_ref, w3_ref, acc_ref, out_ref, acc1_vmem, acc3_vmem):
            if should_init:
                acc1_vmem[...] = jnp.zeros_like(acc1_vmem)
                acc3_vmem[...] = jnp.zeros_like(acc3_vmem)
            else:
                acc1_vmem[...] = acc_ref[0]
                acc3_vmem[...] = acc_ref[1]

            for btc_id in range(num_loops):
                token_slice = pl.ds(btc_id * shape.btc, shape.btc)
                for bd1_id in range(bd1_outer_loops):
                    bd1_base = bd1_id * bd1_per_pack
                    for bd1c_id in range(num_bd1c):
                        k_slice = pl.ds(bd1_base + bd1c_id * bd1c_per_pack, bd1c_per_pack)
                        for p_id in range(shape.t_packing):
                            t = tokens_ref[token_slice, p_id, k_slice]
                            for bfc_id in range(num_bfc):
                                f_slice = pl.ds(bfc_id * shape.bfc, shape.bfc)
                                w_slice = (
                                    p_id,
                                    k_slice,
                                    f_slice,
                                )
                                d1 = jnp.dot(
                                    t,
                                    w1_ref[w_slice],
                                    preferred_element_type=jnp.float32,
                                )
                                d3 = jnp.dot(
                                    t,
                                    w3_ref[w_slice],
                                    preferred_element_type=jnp.float32,
                                )
                                out_slice = (token_slice, f_slice)
                                acc1_vmem[out_slice] += d1
                                acc3_vmem[out_slice] += d3
            out_ref[0] = acc1_vmem[...]
            out_ref[1] = acc3_vmem[...]

        return pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((2, dyn, shape.bf), jnp.float32),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                in_specs=[
                    pl.BlockSpec(memory_space=pltpu.MemorySpace.VMEM),
                    pl.BlockSpec(memory_space=pltpu.MemorySpace.VMEM),
                    pl.BlockSpec(memory_space=pltpu.MemorySpace.VMEM),
                    pl.BlockSpec(memory_space=pltpu.MemorySpace.VMEM),
                ],
                out_specs=pl.BlockSpec(memory_space=pltpu.MemorySpace.VMEM),
                grid=(1,),
                scratch_shapes=[
                    pltpu.VMEM((dyn, shape.bf), jnp.float32),
                    pltpu.VMEM((dyn, shape.bf), jnp.float32),
                ],
            ),
            compiler_params=pltpu.CompilerParams(has_side_effects=True),
            name=f"layer1_ffn_pallas_{shape.path}_{shape.config_label}",
        )(tokens, w1, w3, acc)

    acc1, acc3, w2, res = arrays
    should_init = shape.path in ("dynamic_ffn2_first", "dynamic_ffn2_whole_first")
    output_width = h_per_pack if _is_whole(shape) else bd2_per_pack
    bd2_outer_loops = num_bd2 if _is_whole(shape) else 1

    def kernel(acc1_ref, acc3_ref, w2_ref, res_ref, out_ref, res_vmem):
        for token_tile_id in range(num_token_tiles):
            token_tile_start = token_tile_id * shape.bts
            global_tile_slice = pl.ds(token_tile_start, shape.bts)
            if should_init:
                res_vmem[...] = jnp.zeros_like(res_vmem)
            else:
                res_vmem[...] = res_ref[global_tile_slice]

            tile_dyn_sz = max(min(shape.dyn_sz - token_tile_start, shape.bts), 0)
            tile_loops = _ceil_div(tile_dyn_sz, shape.btc) if tile_dyn_sz > 0 else 0
            for btc_id in range(tile_loops):
                global_token_slice = pl.ds(
                    token_tile_start + btc_id * shape.btc,
                    shape.btc,
                )
                local_token_slice = pl.ds(btc_id * shape.btc, shape.btc)

                for bd2_id in range(bd2_outer_loops):
                    bd2_base = bd2_id * bd2_per_pack
                    for bd2c_id in range(num_bd2c):
                        out_k_slice = pl.ds(bd2_base + bd2c_id * bd2c_per_pack, bd2c_per_pack)
                        for p_id in range(shape.t_packing):
                            partial = jnp.zeros((shape.btc, bd2c_per_pack), dtype=jnp.float32)
                            for bfc_id in range(num_bfc):
                                f_slice = pl.ds(bfc_id * shape.bfc, shape.bfc)
                                a1 = acc1_ref[global_token_slice, f_slice]
                                a3 = acc3_ref[global_token_slice, f_slice]
                                act = _jax_silu(a1, jnp=jnp) * a3
                                w = w2_ref[p_id, f_slice, out_k_slice]
                                partial += jnp.dot(
                                    act,
                                    w,
                                    preferred_element_type=jnp.float32,
                                )
                            res_slice = (local_token_slice, p_id, out_k_slice)
                            if should_init:
                                res_vmem[res_slice] = partial.astype(jnp.bfloat16)
                            else:
                                res_vmem[res_slice] = (
                                    res_vmem[res_slice].astype(jnp.float32) + partial
                                ).astype(jnp.bfloat16)
            out_ref[global_tile_slice] = res_vmem[...]

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((dyn, shape.t_packing, output_width), jnp.bfloat16),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.MemorySpace.VMEM),
                pl.BlockSpec(memory_space=pltpu.MemorySpace.VMEM),
                pl.BlockSpec(memory_space=pltpu.MemorySpace.VMEM),
                pl.BlockSpec(memory_space=pltpu.MemorySpace.VMEM),
            ],
            out_specs=pl.BlockSpec(memory_space=pltpu.MemorySpace.VMEM),
            grid=(1,),
            scratch_shapes=[
                pltpu.VMEM((shape.bts, shape.t_packing, output_width), jnp.bfloat16),
            ],
        ),
        compiler_params=pltpu.CompilerParams(has_side_effects=True),
        name=f"layer1_ffn_pallas_{shape.path}_{shape.config_label}",
    )(acc1, acc3, w2, res)


def _tile_shape(shape: FFNPallasShape) -> tuple[int, ...]:
    rows = _effective_rows(shape)
    if _is_whole(shape):
        if _is_ffn1(shape):
            return (rows, shape.t_packing, shape.hidden_size // shape.t_packing, shape.bf)
        return (rows, shape.bf, shape.t_packing, shape.hidden_size // shape.t_packing)
    if _is_ffn1(shape):
        return (rows, shape.t_packing, shape.bd1 // shape.t_packing, shape.bf)
    return (rows, shape.bf, shape.t_packing, shape.bd2 // shape.t_packing)


def _bytes_hbm(shape: FFNPallasShape) -> int:
    dyn = _effective_rows(shape)
    input_width = shape.hidden_size if _is_whole(shape) else shape.bd1
    output_width = shape.hidden_size if _is_whole(shape) else shape.bd2
    if _is_ffn1(shape):
        token_bytes = dyn * input_width * BF16_BYTES
        weight_bytes = 2 * input_width * shape.bf * BF16_BYTES
        acc_bytes = 2 * dyn * shape.bf * F32_BYTES
        return token_bytes + weight_bytes + acc_bytes
    acc_bytes = 2 * dyn * shape.bf * F32_BYTES
    weight_bytes = shape.bf * output_width * BF16_BYTES
    output_bytes = dyn * output_width * BF16_BYTES
    return acc_bytes + weight_bytes + output_bytes


def _flops(shape: FFNPallasShape) -> int:
    dyn = _effective_rows(shape)
    if _is_ffn1(shape):
        input_width = shape.hidden_size if _is_whole(shape) else shape.bd1
        return 4 * dyn * input_width * shape.bf
    output_width = shape.hidden_size if _is_whole(shape) else shape.bd2
    return 2 * dyn * shape.bf * output_width


def _effective_rows(shape: FFNPallasShape) -> int:
    if shape.dyn_sz <= 0:
        return 1
    return _ceil_div(shape.dyn_sz, shape.btc) * shape.btc


def _includes(shape: FFNPallasShape) -> list[str]:
    if _is_ffn1(shape):
        if _is_whole(shape):
            return [
                "pallas_dynamic_ffn1_gate_dot_all_bd1_tiles",
                "pallas_dynamic_ffn1_up_dot_all_bd1_tiles",
                "acc1_acc3_update",
            ]
        return ["pallas_dynamic_ffn1_gate_dot", "pallas_dynamic_ffn1_up_dot", "acc1_acc3_update"]
    if _is_whole(shape):
        return [
            "pallas_silu_multiply_activation",
            "pallas_dynamic_ffn2_down_dot_all_bd2_tiles",
            "res_init_or_accumulate",
        ]
    return [
        "pallas_silu_multiply_activation",
        "pallas_dynamic_ffn2_down_dot",
        "res_init_or_accumulate",
    ]


def _kernel_reference(shape: FFNPallasShape) -> str:
    if _is_ffn1(shape):
        return f"{KERNEL_PATH}:dynamic_ffn1/run_gate_up_slices"
    return f"{KERNEL_PATH}:dynamic_ffn2/run_down_slices"


def _measurement_scope(shape: FFNPallasShape) -> str:
    if _is_whole(shape):
        return "single_dynamic_ffn_whole_stage"
    return "single_dynamic_ffn_weight_tile"


def _measured_bd1_tiles(shape: FFNPallasShape) -> int:
    if not _is_ffn1(shape):
        return 0
    return _ceil_div(shape.hidden_size, shape.bd1) if _is_whole(shape) else 1


def _measured_bd2_tiles(shape: FFNPallasShape) -> int:
    if _is_ffn1(shape):
        return 0
    return _ceil_div(shape.hidden_size, shape.bd2) if _is_whole(shape) else 1


def _is_ffn1(shape: FFNPallasShape) -> bool:
    return shape.path in (
        "dynamic_ffn1_init",
        "dynamic_ffn1_accumulate",
        "dynamic_ffn1_whole_init",
        "dynamic_ffn1_whole_accumulate",
    )


def _is_whole(shape: FFNPallasShape) -> bool:
    return "_whole_" in shape.path


def _validate_shape(shape: FFNPallasShape) -> None:
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
    for name, value in (
        ("bf", shape.bf),
        ("bfc", shape.bfc),
        ("btc", shape.btc),
        ("bd1", shape.bd1),
        ("bd2", shape.bd2),
        ("bd1c", shape.bd1c),
        ("bd2c", shape.bd2c),
    ):
        if value <= 0:
            raise ValueError(f"Expected positive {name}, got {value}.")
    if shape.bd1 % (shape.t_packing * 128) != 0:
        raise ValueError("bd1 must be divisible by t_packing * 128.")
    if shape.bd2 % (shape.t_packing * 128) != 0:
        raise ValueError("bd2 must be divisible by t_packing * 128.")
    if shape.bd1c % (shape.t_packing * 128) != 0:
        raise ValueError("bd1c must be divisible by t_packing * 128.")
    if shape.bd2c % (shape.t_packing * 128) != 0:
        raise ValueError("bd2c must be divisible by t_packing * 128.")
    if shape.bf % shape.bfc != 0:
        raise ValueError("bf must be divisible by bfc.")


def _jax_silu(x, *, jnp: Any):
    return x * jnp.reciprocal(1.0 + jnp.exp(-x))


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
        return "layer1_ffn_pallas_compute emitted schema-only rows on local_smoke."
    return (
        "layer1_ffn_pallas_compute requires TPU pallas execution; "
        f"observed mode={execution_mode}, backend={runtime.get('default_backend')!r}."
    )
