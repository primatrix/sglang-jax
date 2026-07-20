"""Export deterministic GLM-5.2 DSA CPU PyTorch golden fixtures.

The bundle uses portable ``.npz + manifest.json`` files. BF16 inputs are
quantized by PyTorch first and stored as FP32 values because NumPy does not
provide a portable BF16 dtype; ``semantic_dtype`` in the manifest preserves
the tensor contract.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import io
import json
import shutil
import zipfile
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from benchmark.kernels.mla.glm52_dsa_handoff import GLM52_DSA_CONTRACT
from sgl_jax.srt.kernels.dsa.torch_reference import (
    torch_dsa_sparse_mla,
    torch_glm_dsa_select,
    torch_logical_topk_to_physical_slots,
)


SCHEMA_VERSION = "glm52-dsa-golden-v1"
DEFAULT_CANDIDATE_LENGTHS = (
    1,
    127,
    128,
    129,
    257,
    2047,
    2048,
    2049,
    3072,
    4096,
)


def _bf16_random(shape: tuple[int, ...], *, generator: torch.Generator) -> torch.Tensor:
    return torch.randn(shape, generator=generator, dtype=torch.float32).to(
        torch.bfloat16
    )


def _portable_float(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().float().numpy().astype(np.float32, copy=False)


def _write_deterministic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    """Write a byte-stable NPZ by fixing member order and ZIP timestamps."""
    with zipfile.ZipFile(
        path, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for name in sorted(arrays):
            buffer = io.BytesIO()
            array = np.asarray(arrays[name])
            if array.ndim:
                array = np.ascontiguousarray(array)
            np.lib.format.write_array(buffer, array, allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, buffer.getvalue(), compresslevel=6)


def _array_descriptor(
    array: np.ndarray, *, semantic_dtype: str | None = None
) -> dict[str, object]:
    return {
        "shape": list(array.shape),
        "storage_dtype": str(array.dtype),
        "semantic_dtype": semantic_dtype or str(array.dtype),
    }


def _write_case(
    output_dir: Path,
    *,
    name: str,
    stage: str,
    arrays: Mapping[str, np.ndarray],
    semantic_dtypes: Mapping[str, str],
    metadata: Mapping[str, object] | None = None,
) -> dict[str, object]:
    filename = f"{name}.npz"
    path = output_dir / filename
    _write_deterministic_npz(path, arrays)
    record: dict[str, object] = {
        "name": name,
        "stage": stage,
        "file": filename,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "arrays": {
            key: _array_descriptor(value, semantic_dtype=semantic_dtypes.get(key))
            for key, value in sorted(arrays.items())
        },
    }
    if metadata:
        record["metadata"] = dict(metadata)
    return record


def _selection_case(
    candidate_length: int,
    *,
    generator: torch.Generator,
) -> tuple[dict[str, np.ndarray], dict[str, str], dict[str, object]]:
    contract = GLM52_DSA_CONTRACT
    cache_capacity = candidate_length + 17
    bit_width = max(1, (cache_capacity - 1).bit_length())
    if bit_width > contract.index_head_dim:
        raise ValueError(
            "candidate length exceeds the no-tie fixture encoding capacity"
        )

    # Encode every physical K-cache row as a unique binary score. The sources
    # remain BF16 and accumulation remains FP32, while exact Top-K ordering is
    # independent of tie-breaking details in the candidate implementation.
    q_index = torch.zeros(
        (1, contract.index_heads, contract.index_head_dim), dtype=torch.float32
    )
    q_index[0, 0, :bit_width] = 2.0 ** torch.arange(bit_width, dtype=torch.float32)
    q_index = q_index.to(torch.bfloat16)
    head_weights = torch.zeros((1, contract.index_heads), dtype=torch.bfloat16)
    head_weights[0, 0] = 1.0
    row_ids = torch.arange(cache_capacity, dtype=torch.int64)
    k_index_cache = torch.zeros(
        (cache_capacity, contract.index_head_dim), dtype=torch.float32
    )
    for bit in range(bit_width):
        k_index_cache[:, bit] = ((row_ids >> bit) & 1).to(torch.float32)
    k_index_cache = k_index_cache.to(torch.bfloat16)
    candidate_slots = torch.randperm(
        cache_capacity, generator=generator, dtype=torch.int64
    )[:candidate_length].to(torch.int32)[None, :]
    candidate_logical_ids = torch.arange(candidate_length, dtype=torch.int32)[None, :]
    candidate_counts = torch.tensor([candidate_length], dtype=torch.int32)
    selection_args = {
        "q_index": q_index,
        "head_weights": head_weights,
        "k_index_cache": k_index_cache,
        "candidate_slots": candidate_slots,
        "candidate_logical_ids": candidate_logical_ids,
        "candidate_counts": candidate_counts,
    }
    result = torch_glm_dsa_select(
        **selection_args,
        index_topk=contract.index_topk,
    )
    if candidate_length > contract.index_topk:
        full_result = torch_glm_dsa_select(
            **selection_args,
            index_topk=candidate_length,
        )
        all_scores = full_result.scores[0]
        boundary_margin = float(
            all_scores[contract.index_topk - 1] - all_scores[contract.index_topk]
        )
    else:
        all_scores = result.scores[0, :candidate_length]
        boundary_margin = None
    if torch.unique(all_scores).numel() != candidate_length:
        raise RuntimeError(
            "structured selection fixture unexpectedly contains score ties"
        )
    if boundary_margin is not None and boundary_margin < 1e-3:
        raise RuntimeError(f"Top-K boundary margin is too small: {boundary_margin}")

    arrays = {
        "q_index": _portable_float(q_index),
        "head_weights": _portable_float(head_weights),
        "k_index_cache": _portable_float(k_index_cache),
        "candidate_slots": candidate_slots.numpy(),
        "candidate_logical_ids": candidate_logical_ids.numpy(),
        "candidate_counts": candidate_counts.numpy(),
        "index_topk": np.asarray(contract.index_topk, dtype=np.int32),
        "expected_scores": _portable_float(result.scores),
        "expected_logical_topk_ids": result.logical_topk_ids.numpy(),
        "expected_selected_counts": result.selected_counts.numpy(),
    }
    semantic_dtypes = {
        "q_index": "bfloat16",
        "head_weights": "bfloat16",
        "k_index_cache": "bfloat16",
        "candidate_slots": "int32",
        "candidate_logical_ids": "int32",
        "candidate_counts": "int32",
        "index_topk": "int32",
        "expected_scores": "float32",
        "expected_logical_topk_ids": "int32",
        "expected_selected_counts": "int32",
    }
    metadata = {
        "candidate_length": candidate_length,
        "cache_capacity": cache_capacity,
        "no_score_ties": True,
        "topk_boundary_margin": boundary_margin,
    }
    return arrays, semantic_dtypes, metadata


def _realistic_selection_case(
    *, generator: torch.Generator
) -> tuple[dict[str, np.ndarray], dict[str, str], dict[str, object]]:
    """Exercise every Indexer head/dimension and both sides of ReLU."""
    contract = GLM52_DSA_CONTRACT
    token_count = 2
    candidate_length = 257
    cache_capacity = 521
    candidate_counts = torch.tensor([257, 129], dtype=torch.int32)

    for _ in range(32):
        q_index = _bf16_random(
            (token_count, contract.index_heads, contract.index_head_dim),
            generator=generator,
        )
        head_weights = (
            torch.randn(
                (token_count, contract.index_heads),
                generator=generator,
                dtype=torch.float32,
            )
            * 128.0
        ).to(torch.bfloat16)
        k_index_cache = _bf16_random(
            (cache_capacity, contract.index_head_dim), generator=generator
        )
        candidate_slots = torch.stack(
            [
                torch.randperm(cache_capacity, generator=generator)[:candidate_length]
                for _ in range(token_count)
            ]
        ).to(torch.int32)
        candidate_logical_ids = (
            torch.arange(candidate_length, dtype=torch.int32)[None, :]
            .expand(token_count, -1)
            .contiguous()
        )
        result = torch_glm_dsa_select(
            q_index=q_index,
            head_weights=head_weights,
            k_index_cache=k_index_cache,
            candidate_slots=candidate_slots,
            candidate_logical_ids=candidate_logical_ids,
            candidate_counts=candidate_counts,
            index_topk=contract.index_topk,
        )

        candidate_keys = k_index_cache[candidate_slots.long()]
        logits = torch.einsum("thd,tcd->tch", q_index.float(), candidate_keys.float())
        counted_logits = torch.cat(
            [
                logits[token, :count]
                for token, count in enumerate(candidate_counts.tolist())
            ],
            dim=0,
        )
        counted_keys = torch.cat(
            [
                candidate_keys[token, :count]
                for token, count in enumerate(candidate_counts.tolist())
            ],
            dim=0,
        )
        positive_logit_fraction = float((counted_logits > 0).float().mean())
        relu_both_sides_all_heads = bool(
            torch.all(torch.any(counted_logits > 0, dim=0))
            and torch.all(torch.any(counted_logits <= 0, dim=0))
        )
        uses_all_heads = bool(
            torch.all(torch.any(q_index != 0, dim=(0, 2)))
            and torch.all(torch.any(head_weights != 0, dim=0))
        )
        uses_all_dimensions = bool(
            torch.all(torch.any(q_index != 0, dim=(0, 1)))
            and torch.all(torch.any(counted_keys != 0, dim=0))
        )
        signed_head_weights = bool(
            torch.any(head_weights > 0) and torch.any(head_weights < 0)
        )
        minimum_score_gap = float("inf")
        unique = True
        for token, count in enumerate(candidate_counts.tolist()):
            valid_scores = result.scores[token, :count]
            unique &= torch.unique(valid_scores).numel() == count
            sorted_scores = torch.sort(valid_scores, descending=True).values
            minimum_score_gap = min(
                minimum_score_gap,
                float(torch.min(sorted_scores[:-1] - sorted_scores[1:])),
            )
        if (
            unique
            and minimum_score_gap >= 1e-3
            and 0.25 < positive_logit_fraction < 0.75
            and relu_both_sides_all_heads
            and uses_all_heads
            and uses_all_dimensions
            and signed_head_weights
        ):
            break
    else:
        raise RuntimeError(
            "could not generate a well-separated realistic selection case"
        )

    arrays = {
        "q_index": _portable_float(q_index),
        "head_weights": _portable_float(head_weights),
        "k_index_cache": _portable_float(k_index_cache),
        "candidate_slots": candidate_slots.numpy(),
        "candidate_logical_ids": candidate_logical_ids.numpy(),
        "candidate_counts": candidate_counts.numpy(),
        "index_topk": np.asarray(contract.index_topk, dtype=np.int32),
        "expected_scores": _portable_float(result.scores),
        "expected_logical_topk_ids": result.logical_topk_ids.numpy(),
        "expected_selected_counts": result.selected_counts.numpy(),
    }
    semantic_dtypes = {
        "q_index": "bfloat16",
        "head_weights": "bfloat16",
        "k_index_cache": "bfloat16",
        "candidate_slots": "int32",
        "candidate_logical_ids": "int32",
        "candidate_counts": "int32",
        "index_topk": "int32",
        "expected_scores": "float32",
        "expected_logical_topk_ids": "int32",
        "expected_selected_counts": "int32",
    }
    metadata = {
        "candidate_length": candidate_length,
        "candidate_counts": candidate_counts.tolist(),
        "cache_capacity": cache_capacity,
        "uses_all_heads": uses_all_heads,
        "uses_all_dimensions": uses_all_dimensions,
        "signed_head_weights": signed_head_weights,
        "relu_both_sides_all_heads": relu_both_sides_all_heads,
        "positive_logit_fraction": positive_logit_fraction,
        "minimum_score_gap": minimum_score_gap,
    }
    return arrays, semantic_dtypes, metadata


def _mapping_case() -> tuple[dict[str, np.ndarray], dict[str, str]]:
    width = 2048
    logical_topk_ids = torch.full((2, width), -1, dtype=torch.int32)
    logical_topk_ids[0, :8] = torch.tensor(
        [0, 1, 1, 2, 4, 3, -1, 127], dtype=torch.int32
    )
    logical_topk_ids[1, :8] = torch.tensor(
        [127, 0, 64, 64, 128, -1, 126, 1], dtype=torch.int32
    )
    selected_counts = torch.tensor([8, 8], dtype=torch.int32)
    req_to_token_slots = torch.empty((2, 128), dtype=torch.int32)
    logical = torch.arange(128, dtype=torch.int32)
    req_to_token_slots[0] = (logical * 17 + 5) % 257
    req_to_token_slots[1] = (logical * 19 + 7) % 263
    req_to_token_slots[0, 2] = -1
    query_request_indices = torch.tensor([0, 1], dtype=torch.int32)
    query_positions = torch.tensor([3, 127], dtype=torch.int32)
    producer_layer = 3
    result = torch_logical_topk_to_physical_slots(
        logical_topk_ids=logical_topk_ids,
        selected_counts=selected_counts,
        req_to_token_slots=req_to_token_slots,
        query_request_indices=query_request_indices,
        query_positions=query_positions,
        producer_layer=producer_layer,
    )
    arrays = {
        "logical_topk_ids": logical_topk_ids.numpy(),
        "selected_counts": selected_counts.numpy(),
        "req_to_token_slots": req_to_token_slots.numpy(),
        "query_request_indices": query_request_indices.numpy(),
        "query_positions": query_positions.numpy(),
        "producer_layer": np.asarray(producer_layer, dtype=np.int32),
        "expected_logical_topk_ids": result.logical_topk_ids.numpy(),
        "expected_physical_slots": result.physical_slots.numpy(),
        "expected_selected_counts": result.selected_counts.numpy(),
    }
    return arrays, {name: "int32" for name in arrays}


def _sparse_mla_case(
    *, generator: torch.Generator
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    contract = GLM52_DSA_CONTRACT
    query_rows = 4
    num_heads = 1
    context_length = 2176
    page_count = (context_length + contract.page_size - 1) // contract.page_size
    q_latent = _bf16_random(
        (query_rows, num_heads, contract.latent_dim),
        generator=generator,
    )
    q_rope = _bf16_random(
        (query_rows, num_heads, contract.rope_dim),
        generator=generator,
    )
    cache = _bf16_random(
        (
            page_count,
            contract.page_size // contract.packing,
            contract.packing,
            contract.cache_width,
        ),
        generator=generator,
    )
    selected_counts = torch.tensor([0, 1, 128, 2048], dtype=torch.int32)
    visible_lengths = (1, 1, 128, context_length)
    physical_slots = torch.zeros((query_rows, contract.index_topk), dtype=torch.int32)
    for row, (count, visible_length) in enumerate(
        zip(selected_counts.tolist(), visible_lengths, strict=True)
    ):
        if count == 0:
            continue
        slots = (torch.arange(count, dtype=torch.int64) * visible_length // count).to(
            torch.int32
        )
        order = torch.randperm(count, generator=generator)
        physical_slots[row, :count] = slots[order]

    expected_output = torch_dsa_sparse_mla(
        q_latent=q_latent,
        q_rope=q_rope,
        cache=cache,
        physical_slots=physical_slots,
        selected_counts=selected_counts,
        sm_scale=contract.attention_scale,
        page_size=contract.page_size,
        latent_dim=contract.latent_dim,
        rope_dim=contract.rope_dim,
    )
    arrays = {
        "q_latent": _portable_float(q_latent),
        "q_rope": _portable_float(q_rope),
        "cache": _portable_float(cache),
        "physical_slots": physical_slots.numpy(),
        "selected_counts": selected_counts.numpy(),
        "sm_scale": np.asarray(contract.attention_scale, dtype=np.float32),
        "expected_output": _portable_float(expected_output),
    }
    semantic_dtypes = {
        "q_latent": "bfloat16",
        "q_rope": "bfloat16",
        "cache": "bfloat16",
        "physical_slots": "int32",
        "selected_counts": "int32",
        "sm_scale": "float32",
        "expected_output": "float32",
    }
    return arrays, semantic_dtypes


def export_golden_bundle(
    output_dir: str | Path,
    *,
    candidate_lengths: Sequence[int] = DEFAULT_CANDIDATE_LENGTHS,
    seed: int = 0,
    force: bool = False,
) -> Path:
    """Export all reference stages and return the manifest path."""
    output_dir = Path(output_dir)
    concrete_lengths = tuple(int(length) for length in candidate_lengths)
    if not concrete_lengths or any(length <= 0 for length in concrete_lengths):
        raise ValueError("candidate_lengths must contain positive integers")
    if len(set(concrete_lengths)) != len(concrete_lengths):
        raise ValueError("candidate_lengths must not contain duplicates")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative Python int")

    if output_dir.exists() and any(output_dir.iterdir()):
        if not force:
            raise FileExistsError(
                f"refusing to overwrite non-empty output directory: {output_dir}"
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    cases: list[dict[str, object]] = []
    for candidate_length in concrete_lengths:
        arrays, semantic_dtypes, metadata = _selection_case(
            candidate_length, generator=generator
        )
        cases.append(
            _write_case(
                output_dir,
                name=f"indexer-selection-c{candidate_length}",
                stage="indexer_selection",
                arrays=arrays,
                semantic_dtypes=semantic_dtypes,
                metadata=metadata,
            )
        )

    realistic_arrays, realistic_dtypes, realistic_metadata = _realistic_selection_case(
        generator=generator
    )
    cases.append(
        _write_case(
            output_dir,
            name="indexer-selection-realistic-c257",
            stage="indexer_selection",
            arrays=realistic_arrays,
            semantic_dtypes=realistic_dtypes,
            metadata=realistic_metadata,
        )
    )

    mapping_arrays, mapping_dtypes = _mapping_case()
    cases.append(
        _write_case(
            output_dir,
            name="logical-to-physical-boundaries",
            stage="logical_to_physical",
            arrays=mapping_arrays,
            semantic_dtypes=mapping_dtypes,
            metadata={
                "counted_prefix_only": True,
                "covers_duplicates_future_invalid_and_missing_slots": True,
            },
        )
    )

    sparse_arrays, sparse_dtypes = _sparse_mla_case(generator=generator)
    cases.append(
        _write_case(
            output_dir,
            name="sparse-mla-single-head",
            stage="sparse_mla",
            arrays=sparse_arrays,
            semantic_dtypes=sparse_dtypes,
            metadata={
                "selected_counts": [0, 1, 128, 2048],
                "num_heads": 1,
                "slot_order": "unsorted",
                "fp32_score_softmax_and_accumulation": True,
            },
        )
    )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "seed": seed,
        "candidate_lengths": list(concrete_lengths),
        "contract": dataclasses.asdict(GLM52_DSA_CONTRACT)
        | {"cache_width": GLM52_DSA_CONTRACT.cache_width},
        "abi": {
            "indexer_selection": {
                "q_index_axes": ["query", "index_head", "index_feature"],
                "head_weights_axes": ["query", "index_head"],
                "k_index_cache_axes": ["physical_k_slot", "index_feature"],
                "candidate_slots_axes": ["query", "candidate"],
                "candidate_logical_ids_axes": ["query", "candidate"],
                "candidate_counts_axes": ["query"],
                "index_topk_axes": [],
                "candidate_counts_precondition": (
                    "0 <= candidate_counts[q] <= candidate_width"
                ),
                "counted_candidate_prefix": (
                    "candidate_slots[q, :candidate_counts[q]] and "
                    "candidate_logical_ids[q, :candidate_counts[q]]"
                ),
                "score_formula": (
                    "sum_h(relu(dot(q_index[q,h], "
                    "k_index_cache[candidate_slots[q,c]])) * "
                    "head_weights[q,h]) / sqrt(128 * 32)"
                ),
                "expected_scores_axes": ["query", "selected_rank"],
                "expected_logical_topk_ids_axes": ["query", "selected_rank"],
                "expected_selected_counts_axes": ["query"],
                "output_order": "descending score",
                "selected_counts": "min(candidate_counts, 2048)",
                "reference_accumulation_dtype": "float32",
                "score_padding": "-inf",
                "logical_id_padding": -1,
            },
            "logical_to_physical": {
                "logical_topk_ids_axes": ["query", "selected_rank"],
                "selected_counts_axes": ["query"],
                "req_to_token_slots_axes": ["request", "logical_position"],
                "query_request_indices_axes": ["query"],
                "query_positions_axes": ["query"],
                "producer_layer_axes": [],
                "counted_prefix": ("logical_topk_ids[q, :selected_counts[q]]"),
                "rules": [
                    "remove duplicate logical IDs after the first",
                    "remove logical IDs outside the request mapping",
                    "remove logical IDs greater than the query position",
                    "remove mappings whose physical slot is negative",
                    "compact valid entries while preserving score order",
                ],
                "expected_logical_topk_ids_axes": ["query", "selected_rank"],
                "expected_physical_slots_axes": ["query", "selected_rank"],
                "expected_selected_counts_axes": ["query"],
                "logical_id_padding": -1,
                "physical_slot_padding": 0,
                "output_count": "number of valid unique compacted entries",
            },
            "final_sparse_mla": {
                "q_latent_axes": ["query", "head", "latent_feature"],
                "q_rope_axes": ["query", "head", "rope_feature"],
                "cache_axis_names": ["page", "packed_row", "lane", "feature"],
                "physical_slots_axes": ["query", "selected_rank"],
                "selected_counts_axes": ["query"],
                "sm_scale_axes": [],
                "output_axes": ["query", "head", "latent_feature"],
                "physical_slot_decode": {
                    "page": "slot // 128",
                    "offset": "slot % 128",
                    "packed_row": "offset // 2",
                    "lane": "offset % 2",
                },
                "valid_slot_range": "0 <= slot < pages * 128",
                "counted_prefix": ("physical_slots[q, :selected_counts[q]]"),
                "padding_rule": (
                    "physical slot values at ranks >= selected_counts[q] are ignored; "
                    "slot 0 remains a valid address"
                ),
                "latent_slice": "cache[..., 0:512]",
                "rope_slice": "cache[..., 512:576]",
                "score_formula": (
                    "256^-0.5 * (dot(q_latent, selected_c_kv) + "
                    "dot(q_rope, selected_k_pe))"
                ),
                "softmax_domain": "selected_rank < selected_counts[q]",
                "output_formula": "softmax(score) @ selected_c_kv",
                "reference_accumulation_dtype": "float32",
                "candidate_output_dtype": "bfloat16",
                "zero_count_output": "all zeros",
            },
        },
        "storage_policy": {
            "bfloat16": "stored as float32 values after a PyTorch CPU BF16 round-trip",
            "expected_attention_output": "float32",
            "integer_metadata": "int32",
        },
        "cases": cases,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--candidate-lengths",
        type=int,
        nargs="+",
        default=DEFAULT_CANDIDATE_LENGTHS,
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest_path = export_golden_bundle(
        args.output_dir,
        candidate_lengths=args.candidate_lengths,
        seed=args.seed,
        force=args.force,
    )
    print(manifest_path)


if __name__ == "__main__":
    main()
