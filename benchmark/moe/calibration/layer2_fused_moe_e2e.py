"""Layer 2 fused-MoE end-to-end and ablation diagnostics."""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp

from benchmark.moe.bench_fused_moe import run_all
from benchmark.moe.calibration.common import build_observation_row

SCENARIO_LAYER2_FUSED_MOE_E2E = "layer2_fused_moe_e2e"
SUITE_V7X32_BF16_FUSED_MOE_E2E_DIAG = "v7x32_bf16_fused_moe_e2e_diag"

DTYPE = "bfloat16"
WEIGHT_DTYPE = "bfloat16"
T_PACKING = 2
NUM_EXPERTS = 256
TOP_K = 8
HIDDEN_SIZE = 8192
INTERMEDIATE_SIZE = 2048
NUM_EXPERT_GROUP = 8
TOPK_GROUP = 4
EP_SIZE = 32
USE_SHARED_EXPERT = True
USE_GROUPED_TOPK = True
IMBALANCE_MODE = "sparse_hotspot"
HOTSPOT_RATIO = 1.0
ZERO_EXPERT_COUNT = 0
NON_HOTSPOT_ALPHA = 100.0
KERNEL_PATH = "python/sgl_jax/srt/kernels/fused_moe/v1/kernel.py"

_DISABLE_FLAGS = (
    "FUSED_MOE_BENCHMARK_ALL_DISABLE",
    "FUSED_MOE_BENCHMARK_DISABLE_A2A",
    "FUSED_MOE_BENCHMARK_DISABLE_DYNAMIC_FFN1",
    "FUSED_MOE_BENCHMARK_DISABLE_DYNAMIC_FFN2",
    "FUSED_MOE_BENCHMARK_DISABLE_WEIGHT_LOAD",
    "FUSED_MOE_BENCHMARK_DISABLE_A2A_S_TILE_READ",
    "FUSED_MOE_BENCHMARK_DISABLE_A2A_S_ACC_TILE_WRITE",
    "FUSED_MOE_BENCHMARK_DISABLE_SHARED_EXPERT",
    "FUSED_MOE_BENCHMARK_DISABLE_ALL_REDUCE_METADATA",
    "FUSED_MOE_BENCHMARK_DISABLE_SYNC_BARRIER",
)


@dataclass(frozen=True)
class FusedMoEE2EShape:
    num_tokens: int
    hotspot_count: int


@dataclass(frozen=True)
class FusedMoEVariant:
    name: str
    disables: tuple[str, ...]
    diagnostic_question: str


VARIANTS: tuple[FusedMoEVariant, ...] = (
    FusedMoEVariant("full", (), "Current full fused-MoE E2E latency with overlap enabled."),
    FusedMoEVariant(
        "disable_a2a",
        ("FUSED_MOE_BENCHMARK_DISABLE_A2A",),
        "How much of the composed critical path disappears when A2A traffic is removed?",
    ),
    FusedMoEVariant(
        "disable_weight_load",
        ("FUSED_MOE_BENCHMARK_DISABLE_WEIGHT_LOAD",),
        "Whether weight HBM->VMEM prefetch is hidden by compute and other work.",
    ),
    FusedMoEVariant(
        "disable_dynamic_ffn1",
        ("FUSED_MOE_BENCHMARK_DISABLE_DYNAMIC_FFN1",),
        "Whether gate/up MXU work is on the composed critical path.",
    ),
    FusedMoEVariant(
        "disable_dynamic_ffn2",
        ("FUSED_MOE_BENCHMARK_DISABLE_DYNAMIC_FFN2",),
        "Whether down-projection MXU and accumulator work is on the composed critical path.",
    ),
    FusedMoEVariant(
        "disable_a2a_s_tile_read",
        ("FUSED_MOE_BENCHMARK_DISABLE_A2A_S_TILE_READ",),
        "Whether local token staging into FFN1 is hidden by compute.",
    ),
    FusedMoEVariant(
        "disable_a2a_s_acc_tile_write",
        ("FUSED_MOE_BENCHMARK_DISABLE_A2A_S_ACC_TILE_WRITE",),
        "Whether FFN2 accumulator HBM traffic is hidden or serialized.",
    ),
    FusedMoEVariant(
        "disable_shared_expert",
        ("FUSED_MOE_BENCHMARK_DISABLE_SHARED_EXPERT",),
        "Whether shared expert work is useful overlap filler or critical-path cost.",
    ),
)


def build_rows(
    *,
    suite: str,
    shapes: tuple[FusedMoEE2EShape, ...],
    execution_mode: str,
    runtime: dict[str, Any],
    source: dict[str, Any],
    metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    if suite != SUITE_V7X32_BF16_FUSED_MOE_E2E_DIAG:
        raise ValueError(f"Unsupported Layer 2 fused MoE E2E suite: {suite}")

    if execution_mode == "local_smoke":
        return [
            _make_row(
                suite=suite,
                shape=shape,
                variant=variant,
                execution_mode=execution_mode,
                runtime=runtime,
                source=source,
                metadata=metadata,
                status="not_implemented",
                latency_ms_samples=[],
                benchmark_result={},
                implementation_note="Layer2 fused MoE E2E local_smoke emits schema rows only; TPU trace execution is required for diagnostic results.",
            )
            for shape in shapes
            for variant in selected_variants()
        ]

    if runtime.get("default_backend") != "tpu":
        raise RuntimeError(
            "Layer2 fused MoE E2E diagnostics require JAX default_backend='tpu'. "
            f"Observed {runtime.get('default_backend')!r}."
        )

    rows: list[dict[str, Any]] = []
    sample_runs = int(os.getenv("CALIBRATION_LAYER2_FUSED_MOE_SAMPLE_RUNS", "5"))
    warmup_runs = int(os.getenv("CALIBRATION_LAYER2_FUSED_MOE_WARMUP_RUNS", "3"))
    for shape in shapes:
        for variant in selected_variants():
            try:
                with _variant_env(variant):
                    result_rows = run_all(
                        sample_runs,
                        weight_dtype=jnp.bfloat16,
                        warmup_iters=warmup_runs,
                        tune_block_config=False,
                        num_tokens=[shape.num_tokens],
                        num_experts=NUM_EXPERTS,
                        top_k=TOP_K,
                        hidden_size=HIDDEN_SIZE,
                        intermediate_size=INTERMEDIATE_SIZE,
                        num_expert_group=NUM_EXPERT_GROUP,
                        topk_group=TOPK_GROUP,
                        use_shared_expert=USE_SHARED_EXPERT,
                        imbalance_mode=IMBALANCE_MODE,
                        hotspot_ratio=HOTSPOT_RATIO,
                        hotspot_count=shape.hotspot_count,
                        zero_expert_count=ZERO_EXPERT_COUNT,
                        non_hotspot_alpha=NON_HOTSPOT_ALPHA,
                        return_results=True,
                    )
                benchmark_result = (result_rows or [{}])[0]
                samples = [
                    float(sample)
                    for sample in benchmark_result.get("latency_ms_samples", [])
                    if sample is not None
                ]
                status = "measured" if samples else "benchmark_error"
                note = "Measured full fused-MoE composed run through benchmark.moe.bench_fused_moe."
                if not samples:
                    note = (
                        "Benchmark returned no trace samples; inspect stdout and trace artifacts."
                    )
            except Exception as exc:
                benchmark_result = {"error": f"{type(exc).__name__}: {exc}"}
                samples = []
                status = "benchmark_error"
                note = f"Benchmark failed: {type(exc).__name__}: {exc}"

            rows.append(
                _make_row(
                    suite=suite,
                    shape=shape,
                    variant=variant,
                    execution_mode=execution_mode,
                    runtime=runtime,
                    source=source,
                    metadata=metadata,
                    status=status,
                    latency_ms_samples=samples,
                    benchmark_result=benchmark_result,
                    implementation_note=note,
                )
            )
    return rows


@contextmanager
def _variant_env(variant: FusedMoEVariant):
    old_values = {key: os.environ.get(key) for key in _DISABLE_FLAGS}
    try:
        for key in _DISABLE_FLAGS:
            os.environ[key] = "False"
        os.environ["FUSED_MOE_BENCHMARK_ALL_DISABLE"] = "False"
        for key in variant.disables:
            os.environ[key] = "True"
        yield
    finally:
        for key, value in old_values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _make_row(
    *,
    suite: str,
    shape: FusedMoEE2EShape,
    variant: FusedMoEVariant,
    execution_mode: str,
    runtime: dict[str, Any],
    source: dict[str, Any],
    metadata: dict[str, Any],
    status: str,
    latency_ms_samples: list[float],
    benchmark_result: dict[str, Any],
    implementation_note: str,
) -> dict[str, Any]:
    row_metadata = dict(metadata)
    row_metadata["target_shape"] = {
        "num_tokens": shape.num_tokens,
        "local_num_tokens": shape.num_tokens // EP_SIZE,
        "num_experts": NUM_EXPERTS,
        "local_num_experts": NUM_EXPERTS // EP_SIZE,
        "top_k": TOP_K,
        "hidden_size": HIDDEN_SIZE,
        "intermediate_size": INTERMEDIATE_SIZE,
        "ep_size": EP_SIZE,
        "num_expert_group": NUM_EXPERT_GROUP,
        "topk_group": TOPK_GROUP,
        "use_shared_expert": USE_SHARED_EXPERT,
        "use_grouped_topk": USE_GROUPED_TOPK,
        "imbalance_mode": IMBALANCE_MODE,
        "hotspot_ratio": HOTSPOT_RATIO,
        "hotspot_count": shape.hotspot_count,
        "zero_expert_count": ZERO_EXPERT_COUNT,
        "non_hotspot_alpha": NON_HOTSPOT_ALPHA,
    }
    row_metadata["variant"] = {
        "name": variant.name,
        "disabled_flags": list(variant.disables),
        "diagnostic_question": variant.diagnostic_question,
        "metadata_algorithm": os.getenv(
            "FUSED_MOE_BENCHMARK_METADATA_ALGORITHM", "recursive_doubling"
        ),
        "disable_metadata_background": os.getenv(
            "FUSED_MOE_BENCHMARK_DISABLE_METADATA_BACKGROUND", "0"
        ),
    }
    row_metadata["benchmark_result"] = benchmark_result
    row_metadata["kernel_mapping"] = {
        "kernel_path": KERNEL_PATH,
        "entry": "fused_moe / run_bt",
        "purpose": "composed full-kernel overlap diagnostic, not primitive cost measurement",
    }
    row_metadata["includes"] = [
        "topk",
        "metadata_allgather",
        "scatter",
        "weight_prefetch",
        "dynamic_ffn1",
        "dynamic_ffn2",
        "gather",
        "shared_expert",
        "output_accumulate_store",
    ]
    row_metadata["excludes"] = ["serving_framework_overhead", "model_layer_stack"]

    return build_observation_row(
        scenario=SCENARIO_LAYER2_FUSED_MOE_E2E,
        suite=suite,
        layer=2,
        path=variant.name,
        path_class="fused_moe_e2e_ablation",
        dtype=DTYPE,
        weight_dtype=WEIGHT_DTYPE,
        t_packing=T_PACKING,
        bf=shape.hotspot_count,
        bd=HIDDEN_SIZE,
        tile_shape=(shape.num_tokens, HIDDEN_SIZE),
        bytes_hbm=0,
        bytes_per_fetch=0,
        dma_count=0,
        status=status,
        execution_mode=execution_mode,
        latency_ms_samples=latency_ms_samples,
        runtime=runtime,
        source=source,
        metadata=row_metadata,
        implementation_note=implementation_note,
    )


def default_shapes() -> tuple[FusedMoEE2EShape, ...]:
    return tuple(
        FusedMoEE2EShape(num_tokens=num_tokens, hotspot_count=hotspot_count)
        for num_tokens in _int_tuple_env("CALIBRATION_LAYER2_FUSED_MOE_NUM_TOKENS", (4096,))
        for hotspot_count in _int_tuple_env(
            "CALIBRATION_LAYER2_FUSED_MOE_HOTSPOT_COUNTS",
            (8, 16, 32, 64, 128, 256),
        )
    )


def selected_variants() -> tuple[FusedMoEVariant, ...]:
    raw = os.getenv("CALIBRATION_LAYER2_FUSED_MOE_VARIANTS")
    if not raw:
        return VARIANTS
    requested = {part.strip() for part in raw.split(",") if part.strip()}
    variants = tuple(variant for variant in VARIANTS if variant.name in requested)
    missing = requested - {variant.name for variant in variants}
    if missing:
        raise ValueError(f"Unknown Layer2 fused MoE variants: {sorted(missing)}")
    return variants


def _int_tuple_env(name: str, default: tuple[int, ...]) -> tuple[int, ...]:
    raw = os.getenv(name)
    if not raw:
        return default
    values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    return values or default
