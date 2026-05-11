"""Layer 1 fused-MoE shared-expert Pallas compute calibration."""

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
INTERMEDIATE_SIZE = 2048
KERNEL_PATH = "python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py"

SharedExpertPath = Literal[
    "shared_expert_gate_up_init",
    "shared_expert_gate_up_accumulate",
    "shared_expert_down_first",
    "shared_expert_down_later",
    "shared_expert_slice_first",
    "shared_expert_slice_later",
]


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
    intermediate_size: int = INTERMEDIATE_SIZE
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

    warmup_runs = _positive_int_env("CALIBRATION_LAYER1_SHARED_WARMUP_RUNS", DEFAULT_WARMUP_RUNS)
    sample_runs = _positive_int_env("CALIBRATION_LAYER1_SHARED_SAMPLE_RUNS", DEFAULT_SAMPLE_RUNS)
    discard_runs = _nonnegative_int_env(
        "CALIBRATION_LAYER1_SHARED_TRACE_DISCARD_RUNS", DEFAULT_TRACE_DISCARD_RUNS
    )
    trace_root = os.getenv("CALIBRATION_LAYER1_SHARED_TRACE_ROOT", "/tmp/sglang_jax_layer1_shared")

    for shape in shapes:
        _validate_shape(shape)
        try:
            samples = _measure_shared_pallas_ms(
                shape,
                warmup_runs=warmup_runs,
                sample_runs=sample_runs,
                discard_runs=discard_runs,
                trace_root=trace_root,
            )
            status = STATUS_MEASURED
            note = (
                "Measured a Pallas shared-expert compute unit aligned to "
                "run_shared_expert_slice. It excludes async shared weight/token prefetch, "
                "routed expert FFN, and remote DMA."
            )
        except Exception as exc:
            samples = []
            status = STATUS_NOT_IMPLEMENTED
            note = f"Layer1 Pallas shared expert measurement failed: {type(exc).__name__}: {exc}"

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
    num_bd1 = _ceil_div(shape.hidden_size, shape.bd1)
    num_bd2 = _ceil_div(shape.hidden_size, shape.bd2)
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
        "intermediate_size": shape.intermediate_size,
        "t_packing": shape.t_packing,
        "h_per_t_packing": shape.hidden_size // shape.t_packing,
        "bd1": shape.bd1,
        "bd2": shape.bd2,
        "num_bd1": num_bd1,
        "num_bd2": num_bd2,
        "se_total_blocks": _ceil_div(shape.intermediate_size, shape.bse),
        "tile_shape": _tile_shape(shape),
        "flops": _flops(shape),
        "bytes_hbm": _bytes_hbm(shape),
        "traffic_class": _traffic_class(shape),
        "measurement_status": "measured" if status == STATUS_MEASURED else "schema_only",
        "measurement_scope": _measurement_scope(shape),
        "block_cost_formula": {
            "gate_up_tiles": num_bd1,
            "down_tiles": num_bd2,
            "first_block": "num_bd1 * gate_up_init + num_bd2 * down_first",
            "later_block": "num_bd1 * gate_up_init + num_bd2 * down_later",
            "whole_slice_first": "shared_expert_slice_first",
            "whole_slice_later": "shared_expert_slice_later",
        },
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
        "pallas_aligned": True,
    }
    return enriched


def _measure_shared_pallas_ms(
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
    def run_shared_pallas(*arrays):
        with jax.named_scope("SGLANG_JAX_LAYER1_SHARED_EXPERT_PALLAS"):
            return _pallas_shared_call(shape, arrays, jax=jax, jnp=jnp)

    jax.block_until_ready(run_shared_pallas(*inputs))
    task = f"layer1_shared_expert_{shape.path}_{shape.config_label}"
    return multiple_iteration_timeit_from_trace(
        compute_func=run_shared_pallas,
        data_generator=lambda: inputs,
        task=task,
        tries=sample_runs,
        warmup=warmup_runs,
        discard_initial_samples=discard_runs,
        trace_root=trace_root,
    )


def _make_inputs(shape: SharedExpertShape, *, jnp: Any) -> tuple[Any, ...]:
    bd1_per_pack = shape.bd1 // shape.t_packing
    bd2_per_pack = shape.bd2 // shape.t_packing
    h_per_pack = shape.hidden_size // shape.t_packing
    if shape.path in ("shared_expert_slice_first", "shared_expert_slice_later"):
        tokens = jnp.ones((shape.bt, shape.t_packing, h_per_pack), dtype=jnp.bfloat16)
        w1 = jnp.ones((shape.t_packing, h_per_pack, shape.bse), dtype=jnp.bfloat16)
        w3 = jnp.ones((shape.t_packing, h_per_pack, shape.bse), dtype=jnp.bfloat16)
        w2 = jnp.ones((shape.t_packing, shape.bse, h_per_pack), dtype=jnp.bfloat16)
        out = jnp.ones((shape.bt, shape.t_packing, h_per_pack), dtype=jnp.float32)
        return tokens, w1, w3, w2, out
    if shape.path in ("shared_expert_gate_up_init", "shared_expert_gate_up_accumulate"):
        tokens = jnp.ones((shape.bt, shape.t_packing, bd1_per_pack), dtype=jnp.bfloat16)
        w1 = jnp.ones((shape.t_packing, bd1_per_pack, shape.bse), dtype=jnp.bfloat16)
        w3 = jnp.ones((shape.t_packing, bd1_per_pack, shape.bse), dtype=jnp.bfloat16)
        acc = jnp.ones((2, shape.bt, shape.bse), dtype=jnp.float32)
        return tokens, w1, w3, acc
    gate = jnp.ones((shape.bt, shape.bse), dtype=jnp.float32)
    up = jnp.ones((shape.bt, shape.bse), dtype=jnp.float32)
    w2 = jnp.ones((shape.t_packing, shape.bse, bd2_per_pack), dtype=jnp.bfloat16)
    out = jnp.ones((shape.bt, shape.t_packing, bd2_per_pack), dtype=jnp.float32)
    return gate, up, w2, out


def _pallas_shared_call(shape: SharedExpertShape, arrays: tuple[Any, ...], *, jax: Any, jnp: Any):
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    bd1_per_pack = shape.bd1 // shape.t_packing
    bd2_per_pack = shape.bd2 // shape.t_packing
    h_per_pack = shape.hidden_size // shape.t_packing
    num_bd1 = _ceil_div(shape.hidden_size, shape.bd1)
    num_bd2 = _ceil_div(shape.hidden_size, shape.bd2)

    if shape.path in ("shared_expert_slice_first", "shared_expert_slice_later"):
        tokens, w1, w3, w2, out = arrays
        should_init = shape.path == "shared_expert_slice_first"

        def kernel(tokens_ref, w1_ref, w3_ref, w2_ref, out_ref, result_ref, gate_vmem, up_vmem):
            gate_vmem[...] = jnp.zeros_like(gate_vmem)
            up_vmem[...] = jnp.zeros_like(up_vmem)

            for bd1_idx in range(num_bd1):
                hidden_slice = pl.ds(bd1_idx * bd1_per_pack, bd1_per_pack)
                for p_id in range(shape.t_packing):
                    token_slice = tokens_ref[pl.ds(0, shape.bt), p_id, hidden_slice]
                    w1_tile = w1_ref[p_id, hidden_slice, pl.ds(0, shape.bse)]
                    w3_tile = w3_ref[p_id, hidden_slice, pl.ds(0, shape.bse)]
                    gate_vmem[...] += jnp.dot(
                        token_slice,
                        w1_tile,
                        preferred_element_type=jnp.float32,
                    )
                    up_vmem[...] += jnp.dot(
                        token_slice,
                        w3_tile,
                        preferred_element_type=jnp.float32,
                    )

            act = _jax_silu(gate_vmem[...], jnp=jnp) * up_vmem[...]

            for bd2_idx in range(num_bd2):
                hidden_slice = pl.ds(bd2_idx * bd2_per_pack, bd2_per_pack)
                for p_id in range(shape.t_packing):
                    down = jnp.dot(
                        act,
                        w2_ref[p_id, pl.ds(0, shape.bse), hidden_slice],
                        preferred_element_type=jnp.float32,
                    )
                    out_slice = (pl.ds(0, shape.bt), p_id, hidden_slice)
                    if should_init:
                        result_ref[out_slice] = down
                    else:
                        result_ref[out_slice] = out_ref[out_slice] + down

        return pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((shape.bt, shape.t_packing, h_per_pack), jnp.float32),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                in_specs=[
                    pl.BlockSpec(memory_space=pltpu.MemorySpace.VMEM),
                    pl.BlockSpec(memory_space=pltpu.MemorySpace.VMEM),
                    pl.BlockSpec(memory_space=pltpu.MemorySpace.VMEM),
                    pl.BlockSpec(memory_space=pltpu.MemorySpace.VMEM),
                    pl.BlockSpec(memory_space=pltpu.MemorySpace.VMEM),
                ],
                out_specs=pl.BlockSpec(memory_space=pltpu.MemorySpace.VMEM),
                grid=(1,),
                scratch_shapes=[
                    pltpu.VMEM((shape.bt, shape.bse), jnp.float32),
                    pltpu.VMEM((shape.bt, shape.bse), jnp.float32),
                ],
            ),
            compiler_params=pltpu.CompilerParams(has_side_effects=True),
            name=f"layer1_shared_expert_{shape.path}_{shape.config_label}",
        )(tokens, w1, w3, w2, out)

    if shape.path in ("shared_expert_gate_up_init", "shared_expert_gate_up_accumulate"):
        tokens, w1, w3, acc = arrays
        should_init = shape.path == "shared_expert_gate_up_init"

        def kernel(tokens_ref, w1_ref, w3_ref, acc_ref, out_ref):
            if should_init:
                gate_acc = jnp.zeros((shape.bt, shape.bse), dtype=jnp.float32)
                up_acc = jnp.zeros((shape.bt, shape.bse), dtype=jnp.float32)
            else:
                gate_acc = acc_ref[0]
                up_acc = acc_ref[1]

            for p_id in range(shape.t_packing):
                token_slice = tokens_ref[pl.ds(0, shape.bt), p_id, pl.ds(0, bd1_per_pack)]
                w1_tile = w1_ref[p_id, pl.ds(0, bd1_per_pack), pl.ds(0, shape.bse)]
                w3_tile = w3_ref[p_id, pl.ds(0, bd1_per_pack), pl.ds(0, shape.bse)]
                gate_acc += jnp.dot(token_slice, w1_tile, preferred_element_type=jnp.float32)
                up_acc += jnp.dot(token_slice, w3_tile, preferred_element_type=jnp.float32)

            out_ref[0] = gate_acc
            out_ref[1] = up_acc

        return pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((2, shape.bt, shape.bse), jnp.float32),
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
            ),
            compiler_params=pltpu.CompilerParams(has_side_effects=True),
            name=f"layer1_shared_expert_{shape.path}_{shape.config_label}",
        )(tokens, w1, w3, acc)

    gate, up, w2, out = arrays
    should_init = shape.path == "shared_expert_down_first"

    def kernel(gate_ref, up_ref, w2_ref, out_ref, result_ref):
        act = _jax_silu(gate_ref[...], jnp=jnp) * up_ref[...]
        for p_id in range(shape.t_packing):
            down = jnp.dot(
                act,
                w2_ref[p_id, pl.ds(0, shape.bse), pl.ds(0, bd2_per_pack)],
                preferred_element_type=jnp.float32,
            )
            if should_init:
                result_ref[pl.ds(0, shape.bt), p_id, pl.ds(0, bd2_per_pack)] = down
            else:
                result_ref[pl.ds(0, shape.bt), p_id, pl.ds(0, bd2_per_pack)] = (
                    out_ref[pl.ds(0, shape.bt), p_id, pl.ds(0, bd2_per_pack)] + down
                )

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((shape.bt, shape.t_packing, bd2_per_pack), jnp.float32),
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
        ),
        compiler_params=pltpu.CompilerParams(has_side_effects=True),
        name=f"layer1_shared_expert_{shape.path}_{shape.config_label}",
    )(gate, up, w2, out)


def _tile_shape(shape: SharedExpertShape) -> tuple[int, ...]:
    if shape.path in ("shared_expert_slice_first", "shared_expert_slice_later"):
        return (
            shape.bt,
            shape.t_packing,
            shape.hidden_size // shape.t_packing,
            shape.bse,
            shape.t_packing,
            shape.hidden_size // shape.t_packing,
        )
    if shape.path in ("shared_expert_gate_up_init", "shared_expert_gate_up_accumulate"):
        return (shape.bt, shape.t_packing, shape.bd1 // shape.t_packing, shape.bse)
    return (shape.bt, shape.bse, shape.t_packing, shape.bd2 // shape.t_packing)


def _bytes_hbm(shape: SharedExpertShape) -> int:
    if shape.path in ("shared_expert_slice_first", "shared_expert_slice_later"):
        token_bytes = shape.bt * shape.hidden_size * BF16_BYTES
        w1_w3_bytes = 2 * shape.hidden_size * shape.bse * BF16_BYTES
        w2_bytes = shape.bse * shape.hidden_size * BF16_BYTES
        output_bytes = shape.bt * shape.hidden_size * F32_BYTES
        if shape.path == "shared_expert_slice_later":
            output_bytes *= 2
        return token_bytes + w1_w3_bytes + w2_bytes + output_bytes
    if shape.path in ("shared_expert_gate_up_init", "shared_expert_gate_up_accumulate"):
        token_bytes = shape.bt * shape.bd1 * BF16_BYTES
        weight_bytes = 2 * shape.bd1 * shape.bse * BF16_BYTES
        acc_bytes = 2 * shape.bt * shape.bse * F32_BYTES
        return token_bytes + weight_bytes + acc_bytes
    acc_bytes = 2 * shape.bt * shape.bse * F32_BYTES
    weight_bytes = shape.bse * shape.bd2 * BF16_BYTES
    output_bytes = shape.bt * shape.bd2 * F32_BYTES
    return acc_bytes + weight_bytes + output_bytes


def _flops(shape: SharedExpertShape) -> int:
    if shape.path in ("shared_expert_slice_first", "shared_expert_slice_later"):
        gate_up = 4 * shape.bt * shape.hidden_size * shape.bse
        down = 2 * shape.bt * shape.bse * shape.hidden_size
        return gate_up + down
    if shape.path in ("shared_expert_gate_up_init", "shared_expert_gate_up_accumulate"):
        return 4 * shape.bt * shape.bd1 * shape.bse
    return 2 * shape.bt * shape.bse * shape.bd2


def _includes(shape: SharedExpertShape) -> list[str]:
    if shape.path in ("shared_expert_slice_first", "shared_expert_slice_later"):
        return [
            "pallas_shared_expert_gate_up_all_bd1_tiles",
            "pallas_shared_expert_activation",
            "pallas_shared_expert_down_all_bd2_tiles",
            "shared_output_f32_init_or_accumulate",
        ]
    if shape.path in ("shared_expert_gate_up_init", "shared_expert_gate_up_accumulate"):
        return [
            "pallas_shared_expert_gate_dot_single_bd1_tile",
            "pallas_shared_expert_up_dot_single_bd1_tile",
            "gate_up_accumulator_update",
        ]
    return [
        "pallas_shared_expert_activation",
        "pallas_shared_expert_down_dot_single_bd2_tile",
        "shared_output_f32_init_or_accumulate",
    ]


def _measurement_scope(shape: SharedExpertShape) -> str:
    if shape.path in ("shared_expert_slice_first", "shared_expert_slice_later"):
        return "single_run_shared_expert_slice_block"
    if shape.path in ("shared_expert_gate_up_init", "shared_expert_gate_up_accumulate"):
        return "single_shared_expert_gate_up_bd1_tile"
    return "single_shared_expert_down_bd2_tile"


def _traffic_class(shape: SharedExpertShape) -> str:
    if shape.path in ("shared_expert_slice_first", "shared_expert_slice_later"):
        return "shared_expert_pallas_compute_slice"
    return "shared_expert_pallas_compute_tile"


def _jax_silu(x, *, jnp: Any):
    return x * jnp.reciprocal(1.0 + jnp.exp(-x))


def _validate_shape(shape: SharedExpertShape) -> None:
    if shape.bt <= 0 or shape.bse <= 0:
        raise ValueError(f"Expected positive bt/bse, got {shape.bt=} {shape.bse=}.")
    if shape.hidden_size != HIDDEN_SIZE:
        raise ValueError(f"Expected hidden_size={HIDDEN_SIZE}, got {shape.hidden_size}.")
    if shape.intermediate_size != INTERMEDIATE_SIZE:
        raise ValueError(
            f"Expected intermediate_size={INTERMEDIATE_SIZE}, got {shape.intermediate_size}."
        )
    if shape.t_packing != T_PACKING:
        raise ValueError(f"Expected t_packing={T_PACKING}, got {shape.t_packing}.")
    tile_align = shape.t_packing * 128
    if shape.bd1 % tile_align != 0:
        raise ValueError(f"Expected bd1 to be aligned to {tile_align}.")
    if shape.bd2 % tile_align != 0:
        raise ValueError(f"Expected bd2 to be aligned to {tile_align}.")
    if shape.hidden_size % shape.bd1 != 0 or shape.hidden_size % shape.bd2 != 0:
        raise ValueError("hidden_size must be divisible by bd1 and bd2.")
    if shape.intermediate_size % shape.bse != 0:
        raise ValueError("intermediate_size must be divisible by bse.")


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
        "layer1_shared_expert_compute requires TPU pallas execution; "
        f"observed mode={execution_mode}, backend={runtime.get('default_backend')!r}."
    )
