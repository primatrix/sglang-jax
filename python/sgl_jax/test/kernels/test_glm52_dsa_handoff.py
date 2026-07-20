"""Host-only contract tests for the GLM-5.2 DSA kernel handoff."""

import hashlib
import json

import numpy as np
import pytest

from benchmark.kernels.mla.glm52_dsa_handoff import (
    GLM52_DSA_CONTRACT,
    PERFORMANCE_CASES,
    SparseMlaPerfCase,
)


def test_production_contract_uses_tp32_local_shape():
    assert GLM52_DSA_CONTRACT.total_query_heads == 64
    assert GLM52_DSA_CONTRACT.tensor_parallel_size == 32
    assert GLM52_DSA_CONTRACT.local_query_heads == 2
    assert GLM52_DSA_CONTRACT.latent_dim == 512
    assert GLM52_DSA_CONTRACT.rope_dim == 64
    assert GLM52_DSA_CONTRACT.cache_width == 640
    assert GLM52_DSA_CONTRACT.page_size == 128
    assert GLM52_DSA_CONTRACT.packing == 2
    assert GLM52_DSA_CONTRACT.index_topk == 2048
    assert GLM52_DSA_CONTRACT.attention_scale == 256**-0.5


def test_performance_manifest_covers_required_decode_and_prefill_shapes():
    cases = {case.name: case for case in PERFORMANCE_CASES}
    assert set(cases) == {
        "decode-bucket-a1-c512",
        "decode-bucket-a1-c1024",
        "decode-bucket-a1-c2048",
        "decode-bucket-a1-c4096",
        "decode-bucket-a8-c4096",
        "decode-bucket-a32-c4096",
        "decode-bucket-a64-c4096",
        "decode-long-a1-c160k",
        "decode-throughput-a8-c32k",
        "prefill-t128-start0",
        "prefill-t128-start2048",
    }

    assert cases["decode-bucket-a1-c512"].shape_tuple == (
        "decode",
        64,
        1,
        512,
        2048,
    )
    assert cases["decode-bucket-a1-c1024"].shape_tuple == (
        "decode",
        64,
        1,
        1024,
        2048,
    )
    assert cases["decode-bucket-a1-c2048"].shape_tuple == (
        "decode",
        64,
        1,
        2048,
        2048,
    )

    assert cases["decode-bucket-a1-c4096"].shape_tuple == (
        "decode",
        64,
        1,
        4096,
        2048,
    )
    assert cases["decode-bucket-a64-c4096"].shape_tuple == (
        "decode",
        64,
        64,
        4096,
        2048,
    )
    assert cases["decode-bucket-a8-c4096"].active_query_rows == 8
    assert cases["decode-bucket-a32-c4096"].active_query_rows == 32
    assert cases["decode-bucket-a64-c4096"].minimum_cache_capacity == 262_144
    assert cases["decode-throughput-a8-c32k"].minimum_cache_capacity == 256_000
    assert cases["prefill-t128-start2048"].minimum_cache_capacity == 2176
    assert cases["decode-long-a1-c160k"].shape_tuple == (
        "decode",
        1,
        1,
        160_000,
        2048,
    )
    assert cases["decode-throughput-a8-c32k"].shape_tuple == (
        "decode",
        8,
        8,
        32_000,
        2048,
    )
    assert cases["prefill-t128-start0"].shape_tuple == (
        "prefill",
        128,
        128,
        128,
        2048,
    )
    assert cases["prefill-t128-start0"].start_position == 0
    assert cases["prefill-t128-start2048"].shape_tuple == (
        "prefill",
        128,
        128,
        2176,
        2048,
    )
    assert cases["prefill-t128-start2048"].start_position == 2048


def test_every_performance_case_uses_production_kernel_dimensions():
    for case in PERFORMANCE_CASES:
        assert case.num_heads == GLM52_DSA_CONTRACT.local_query_heads
        assert case.latent_dim == GLM52_DSA_CONTRACT.latent_dim
        assert case.rope_dim == GLM52_DSA_CONTRACT.rope_dim
        assert case.page_size == GLM52_DSA_CONTRACT.page_size
        if case.mode == "prefill":
            assert case.request_layout == "shared"
        else:
            assert case.request_layout == "disjoint"


def test_performance_manifest_rejects_nonproduction_kernel_dimensions():
    with pytest.raises(ValueError, match="production kernel dimensions"):
        SparseMlaPerfCase(
            name="wrong-head-count",
            mode="decode",
            physical_query_rows=1,
            active_query_rows=1,
            context_length=2048,
            num_heads=4,
        )


def test_cpu_exporter_declares_all_stages_and_round_trips(tmp_path):
    pytest.importorskip("torch")
    from benchmark.kernels.mla.export_glm52_dsa_golden import (
        export_golden_bundle,
    )

    manifest_path = export_golden_bundle(
        tmp_path,
        candidate_lengths=(1, 129, 2049),
        seed=7,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["schema_version"] == "glm52-dsa-golden-v1"
    assert manifest["candidate_lengths"] == [1, 129, 2049]
    sparse_abi = manifest["abi"]["final_sparse_mla"]
    selection_abi = manifest["abi"]["indexer_selection"]
    mapping_abi = manifest["abi"]["logical_to_physical"]
    assert selection_abi["output_order"] == "descending score"
    assert selection_abi["candidate_counts_precondition"] == (
        "0 <= candidate_counts[q] <= candidate_width"
    )
    assert selection_abi["selected_counts"] == "min(candidate_counts, 2048)"
    assert selection_abi["reference_accumulation_dtype"] == "float32"
    assert selection_abi["index_topk_axes"] == []
    assert selection_abi["score_padding"] == "-inf"
    assert selection_abi["logical_id_padding"] == -1
    assert selection_abi["expected_scores_axes"] == ["query", "selected_rank"]
    assert mapping_abi["expected_physical_slots_axes"] == [
        "query",
        "selected_rank",
    ]
    assert mapping_abi["logical_id_padding"] == -1
    assert mapping_abi["physical_slot_padding"] == 0
    assert mapping_abi["producer_layer_axes"] == []
    assert sparse_abi["cache_axis_names"] == [
        "page",
        "packed_row",
        "lane",
        "feature",
    ]
    assert sparse_abi["physical_slot_decode"] == {
        "page": "slot // 128",
        "offset": "slot % 128",
        "packed_row": "offset // 2",
        "lane": "offset % 2",
    }
    assert sparse_abi["counted_prefix"] == ("physical_slots[q, :selected_counts[q]]")
    assert sparse_abi["valid_slot_range"] == "0 <= slot < pages * 128"
    assert sparse_abi["score_formula"] == (
        "256^-0.5 * (dot(q_latent, selected_c_kv) + dot(q_rope, selected_k_pe))"
    )
    assert sparse_abi["softmax_domain"] == "selected_rank < selected_counts[q]"
    assert sparse_abi["reference_accumulation_dtype"] == "float32"
    assert sparse_abi["sm_scale_axes"] == []
    assert {case["stage"] for case in manifest["cases"]} == {
        "indexer_selection",
        "logical_to_physical",
        "sparse_mla",
    }

    for case in manifest["cases"]:
        fixture_path = tmp_path / case["file"]
        assert fixture_path.is_file()
        assert hashlib.sha256(fixture_path.read_bytes()).hexdigest() == case["sha256"]
        with np.load(fixture_path) as arrays:
            assert set(arrays.files) == set(case["arrays"])
            for name, descriptor in case["arrays"].items():
                assert list(arrays[name].shape) == descriptor["shape"]
                assert str(arrays[name].dtype) == descriptor["storage_dtype"]

    sparse_case = next(case for case in manifest["cases"] if case["stage"] == "sparse_mla")
    with np.load(tmp_path / sparse_case["file"]) as arrays:
        assert arrays["expected_output"].dtype == np.float32
        assert arrays["physical_slots"].dtype == np.int32
        assert arrays["selected_counts"].dtype == np.int32
        assert arrays["expected_output"].shape == (7, 2, 512)
        assert arrays["selected_counts"].tolist() == [
            0,
            1,
            127,
            128,
            129,
            2047,
            2048,
        ]
        counted_slots = arrays["physical_slots"][3, :128]
        sorted_slots = np.sort(counted_slots)
        assert all(
            not np.array_equal(counted_slots, np.roll(sorted_slots, shift))
            for shift in range(sorted_slots.size)
        )

    selection_case = next(
        case for case in manifest["cases"] if case["name"] == "indexer-selection-c129"
    )
    with np.load(tmp_path / selection_case["file"]) as arrays:
        assert arrays["k_index_cache"].shape[0] > 129
        assert not np.array_equal(
            arrays["candidate_slots"],
            np.arange(129, dtype=np.int32)[None, :],
        )

    boundary_case = next(
        case for case in manifest["cases"] if case["name"] == "indexer-selection-c2049"
    )
    assert boundary_case["metadata"]["topk_boundary_margin"] >= 1e-3

    realistic_case = next(
        case for case in manifest["cases"] if case["name"] == "indexer-selection-realistic-c257"
    )
    assert realistic_case["metadata"]["uses_all_heads"] is True
    assert realistic_case["metadata"]["uses_all_dimensions"] is True
    assert realistic_case["metadata"]["relu_both_sides_all_heads"] is True
    assert realistic_case["metadata"]["signed_head_weights"] is True
    assert 0.25 < realistic_case["metadata"]["positive_logit_fraction"] < 0.75
    with np.load(tmp_path / realistic_case["file"]) as arrays:
        assert np.all(np.any(arrays["q_index"] != 0, axis=(0, 2)))
        assert np.all(np.any(arrays["q_index"] != 0, axis=(0, 1)))
        assert np.any(arrays["head_weights"] > 0)
        assert np.any(arrays["head_weights"] < 0)


def test_cpu_exporter_refuses_to_overwrite_nonempty_directory(tmp_path):
    pytest.importorskip("torch")
    from benchmark.kernels.mla.export_glm52_dsa_golden import (
        export_golden_bundle,
    )

    (tmp_path / "owned.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError, match="non-empty"):
        export_golden_bundle(tmp_path, candidate_lengths=(1,), seed=0)
