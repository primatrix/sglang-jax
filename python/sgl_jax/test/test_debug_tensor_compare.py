import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import ml_dtypes
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[3]
COMPARE_PATH = ROOT / "scripts/kernels/compare_debug_tensor_dumps.py"


def _load_compare_module():
    assert COMPARE_PATH.is_file(), f"missing comparator: {COMPARE_PATH}"
    spec = importlib.util.spec_from_file_location("debug_tensor_compare", COMPARE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_dump(directory, rows):
    directory.mkdir()
    manifests = {}
    for row, array in rows:
        filename = row["filename"]
        np.save(directory / filename, np.asarray(array), allow_pickle=False)
        manifest_row = {
            **row,
            "shape": list(np.asarray(array).shape),
            "dtype": str(np.asarray(array).dtype),
        }
        manifests.setdefault(row["process"], []).append(manifest_row)
    for process, process_rows in manifests.items():
        manifest = directory / f"manifest-p{process:05d}.jsonl"
        manifest.write_text(
            "".join(json.dumps(row) + "\n" for row in process_rows),
            encoding="utf-8",
        )


def _write_named_dump(root, manifest_name, row, array):
    manifest = root / manifest_name
    manifest.parent.mkdir(parents=True, exist_ok=True)
    host_array = np.asarray(array)
    np.save(manifest.parent / row["filename"], host_array, allow_pickle=False)
    manifest.write_text(
        json.dumps(
            {
                **row,
                "shape": list(host_array.shape),
                "dtype": str(host_array.dtype),
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _write_manifest(directory, row, *, shape, dtype):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "manifest-p00000.jsonl").write_text(
        json.dumps({**row, "shape": list(shape), "dtype": str(dtype)}) + "\n",
        encoding="utf-8",
    )


def _row(
    filename,
    *,
    component="decoder_layer",
    layer=3,
    forward_mode="extend",
    name="hidden_states",
    occurrence=0,
    process=0,
):
    return {
        "filename": filename,
        "component": component,
        "layer": layer,
        "forward_mode": forward_mode,
        "name": name,
        "occurrence": occurrence,
        "process": process,
    }


def test_compare_aligns_semantic_keys_and_reports_raw_metrics(tmp_path):
    compare = _load_compare_module()
    candidate = tmp_path / "candidate"
    baseline = tmp_path / "baseline"
    float_key = _row("candidate-hidden.npy", process=1)
    topk_key = _row(
        "candidate-topk.npy",
        component="dsa_selection",
        name="logical_topk_ids",
        occurrence=2,
    )
    counts_key = _row(
        "candidate-counts.npy",
        component="dsa_selection",
        name="selected_counts",
        occurrence=2,
    )
    _write_dump(
        candidate,
        [
            (float_key, np.array([1.0, 2.5, 2.0, 4.0], dtype=np.float32)),
            (topk_key, np.array([[1, 2, 9, 4]], dtype=np.int32)),
            (counts_key, np.array([4], dtype=np.int32)),
        ],
    )
    _write_dump(
        baseline,
        [
            (
                {**topk_key, "filename": "baseline-topk.npy"},
                np.array([[1, 2, 3, 4]], dtype=np.int32),
            ),
            (
                {**float_key, "filename": "baseline-hidden.npy"},
                np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
            ),
            (
                {**counts_key, "filename": "baseline-counts.npy"},
                np.array([4], dtype=np.int32),
            ),
        ],
    )

    report = compare.compare_dump_directories(candidate, baseline)

    assert report["passed"] is True
    assert report["tensor_count"] == 3
    assert report["missing_from_candidate"] == []
    assert report["missing_from_baseline"] == []
    assert report["manifest_errors"] == []
    comparisons = {item["key"]["name"]: item for item in report["comparisons"]}
    hidden = comparisons["hidden_states"]
    assert hidden["shape_match"] is True
    assert hidden["dtype_match"] is True
    assert hidden["metrics"]["max_abs"] == pytest.approx(1.0)
    assert hidden["metrics"]["mean_abs"] == pytest.approx(0.375)
    assert hidden["metrics"]["p99_abs"] == pytest.approx(0.985)
    assert hidden["metrics"]["cosine"] == pytest.approx(
        float(
            np.dot([1.0, 2.5, 2.0, 4.0], [1.0, 2.0, 3.0, 4.0])
            / (
                np.linalg.norm([1.0, 2.5, 2.0, 4.0])
                * np.linalg.norm([1.0, 2.0, 3.0, 4.0])
            )
        )
    )
    assert hidden["metrics"]["topk_overlap"] is None
    topk = comparisons["logical_topk_ids"]
    assert topk["metrics"]["topk_overlap"] == pytest.approx(0.75)
    json.dumps(report, sort_keys=True, allow_nan=False)


def test_compare_recovers_jax_bfloat16_saved_as_void2(tmp_path):
    compare = _load_compare_module()
    candidate = tmp_path / "candidate"
    baseline = tmp_path / "baseline"
    key = _row("hidden.npy")
    for directory, values in (
        (candidate, [1.0, 2.5]),
        (baseline, [1.0, 2.0]),
    ):
        directory.mkdir()
        bfloat16 = np.asarray(values, dtype=ml_dtypes.bfloat16)
        # JAX host callbacks currently expose bfloat16 to np.save as raw V2.
        np.save(directory / key["filename"], bfloat16.view("V2"), allow_pickle=False)
        _write_manifest(directory, key, shape=bfloat16.shape, dtype="bfloat16")

    report = compare.compare_dump_directories(candidate, baseline)

    assert report["passed"] is True
    comparison = report["comparisons"][0]
    assert comparison["candidate_dtype"] == "bfloat16"
    assert comparison["baseline_dtype"] == "bfloat16"
    assert comparison["dtype_match"] is True
    assert comparison["metrics"]["max_abs"] == pytest.approx(0.5)


def test_compare_uses_debug_context_token_valid_mask_for_metrics(tmp_path):
    compare = _load_compare_module()
    candidate = tmp_path / "candidate"
    baseline = tmp_path / "baseline"
    hidden_key = _row("hidden.npy", occurrence=7)
    mask_key = _row(
        "mask.npy",
        component="debug_context",
        layer=None,
        name="token_valid_mask",
        occurrence=7,
    )
    _write_dump(
        candidate,
        [
            (hidden_key, np.array([[1.0, 2.0], [100.0, 200.0]], dtype=np.float32)),
            (mask_key, np.array([True, False], dtype=np.bool_)),
        ],
    )
    _write_dump(
        baseline,
        [
            (
                {**hidden_key, "filename": "baseline-hidden.npy"},
                np.array([[1.0, 2.0], [-100.0, -200.0]], dtype=np.float32),
            ),
            (
                {**mask_key, "filename": "baseline-mask.npy"},
                np.array([True, False], dtype=np.bool_),
            ),
        ],
    )

    report = compare.compare_dump_directories(candidate, baseline)

    comparison = next(
        item for item in report["comparisons"] if item["key"]["name"] == "hidden_states"
    )
    assert comparison["valid_row_count"] == 1
    assert comparison["metrics"]["max_abs"] == 0.0
    assert comparison["metrics"]["mean_abs"] == 0.0
    assert comparison["metrics"]["p99_abs"] == 0.0


def test_compare_can_drop_cancelled_forwards_without_terminal_marker(tmp_path):
    compare = _load_compare_module()
    candidate = tmp_path / "candidate"
    baseline = tmp_path / "baseline"
    complete_hidden = _row("candidate-complete-hidden.npy", occurrence=10)
    complete_logits = _row(
        "candidate-complete-logits.npy",
        component="logits",
        layer=None,
        name="next_token_logits",
        occurrence=10,
    )
    candidate_partial = _row("candidate-partial.npy", layer=2, occurrence=20)
    baseline_partial = _row("baseline-partial.npy", layer=7, occurrence=30)
    _write_dump(
        candidate,
        [
            (complete_hidden, np.array([1.0], dtype=np.float32)),
            (complete_logits, np.array([2.0], dtype=np.float32)),
            (candidate_partial, np.array([3.0], dtype=np.float32)),
        ],
    )
    _write_dump(
        baseline,
        [
            (
                {**complete_hidden, "filename": "baseline-complete-hidden.npy"},
                np.array([1.0], dtype=np.float32),
            ),
            (
                {**complete_logits, "filename": "baseline-complete-logits.npy"},
                np.array([2.0], dtype=np.float32),
            ),
            (baseline_partial, np.array([4.0], dtype=np.float32)),
        ],
    )

    report = compare.compare_dump_directories(
        candidate,
        baseline,
        complete_forward_marker=("logits", "next_token_logits"),
    )

    assert report["passed"] is True
    assert report["tensor_count"] == 2
    assert report["complete_forward_marker"] == {
        "component": "logits",
        "name": "next_token_logits",
    }
    assert report["candidate_completed_forward_count"] == 1
    assert report["baseline_completed_forward_count"] == 1
    assert report["candidate_dropped_incomplete_forward_count"] == 1
    assert report["baseline_dropped_incomplete_forward_count"] == 1
    assert report["candidate_dropped_tensor_count"] == 1
    assert report["baseline_dropped_tensor_count"] == 1


@pytest.mark.parametrize(
    ("candidate_ids", "baseline_ids", "candidate_count", "baseline_count", "expected"),
    [
        ([7, 9, -1, -1], [9, 7, -1, -1], 2, 2, 1.0),
        ([-1, -1, -1, -1], [-1, -1, -1, -1], 0, 0, 1.0),
        ([7, -1, -1, -1], [-1, -1, -1, -1], 1, 0, 0.0),
        ([1, 2, 3, 4], [1, 2, 9, 4], 4, 4, 0.75),
    ],
)
def test_logical_topk_overlap_uses_selected_count_prefixes(
    tmp_path,
    candidate_ids,
    baseline_ids,
    candidate_count,
    baseline_count,
    expected,
):
    compare = _load_compare_module()
    candidate = tmp_path / "candidate"
    baseline = tmp_path / "baseline"
    ids_key = _row(
        "candidate-ids.npy", component="dsa_selection", name="logical_topk_ids"
    )
    counts_key = _row(
        "candidate-counts.npy", component="dsa_selection", name="selected_counts"
    )
    _write_dump(
        candidate,
        [
            (ids_key, np.array([candidate_ids], dtype=np.int32)),
            (counts_key, np.array([candidate_count], dtype=np.int32)),
        ],
    )
    _write_dump(
        baseline,
        [
            (
                {**ids_key, "filename": "baseline-ids.npy"},
                np.array([baseline_ids], dtype=np.int32),
            ),
            (
                {**counts_key, "filename": "baseline-counts.npy"},
                np.array([baseline_count], dtype=np.int32),
            ),
        ],
    )

    report = compare.compare_dump_directories(candidate, baseline)

    comparison = next(
        item for item in report["comparisons"] if item["key"]["name"] == "logical_topk_ids"
    )
    assert comparison["metrics"]["topk_overlap"] == pytest.approx(expected)


def test_logical_topk_overlap_rejects_duplicate_valid_ids(tmp_path):
    compare = _load_compare_module()
    candidate = tmp_path / "candidate"
    baseline = tmp_path / "baseline"
    ids_key = _row(
        "candidate-ids.npy", component="dsa_selection", name="logical_topk_ids"
    )
    counts_key = _row(
        "candidate-counts.npy", component="dsa_selection", name="selected_counts"
    )
    _write_dump(
        candidate,
        [
            (ids_key, np.array([[3, 3, -1, -1]], dtype=np.int32)),
            (counts_key, np.array([2], dtype=np.int32)),
        ],
    )
    _write_dump(
        baseline,
        [
            (
                {**ids_key, "filename": "baseline-ids.npy"},
                np.array([[3, 4, -1, -1]], dtype=np.int32),
            ),
            (
                {**counts_key, "filename": "baseline-counts.npy"},
                np.array([2], dtype=np.int32),
            ),
        ],
    )

    report = compare.compare_dump_directories(candidate, baseline)

    comparison = next(
        item for item in report["comparisons"] if item["key"]["name"] == "logical_topk_ids"
    )
    assert comparison["passed"] is False
    assert comparison["failures"] == ["candidate valid Top-K prefix contains duplicate IDs"]


def test_compare_discovers_nested_non_process_manifest_names(tmp_path):
    compare = _load_compare_module()
    candidate = tmp_path / "candidate"
    baseline = tmp_path / "baseline"
    key = _row("hidden.npy", process=7)
    array = np.array([1.0, 2.0], dtype=np.float32)
    _write_named_dump(
        candidate,
        "rank-7/run-a/debug_dumps/manifest-worker.jsonl",
        key,
        array,
    )
    _write_named_dump(
        baseline,
        "artifacts/host-7/manifest-reference.jsonl",
        key,
        array,
    )

    report = compare.compare_dump_directories(candidate, baseline)

    assert report["passed"] is True
    assert report["tensor_count"] == 1
    assert report["manifest_errors"] == []


def test_nested_manifest_cannot_reference_sibling_dump_tree(tmp_path):
    compare = _load_compare_module()
    candidate = tmp_path / "candidate"
    baseline = tmp_path / "baseline"
    rank_a = candidate / "rank-a"
    rank_b = candidate / "rank-b"
    rank_a.mkdir(parents=True)
    rank_b.mkdir(parents=True)
    np.save(rank_b / "hidden.npy", np.array([1.0], dtype=np.float32))
    row = _row("../rank-b/hidden.npy")
    (rank_a / "manifest-a.jsonl").write_text(
        json.dumps({**row, "shape": [1], "dtype": "float32"}) + "\n",
        encoding="utf-8",
    )
    _write_dump(
        baseline,
        [(_row("baseline.npy"), np.array([1.0], dtype=np.float32))],
    )

    report = compare.compare_dump_directories(candidate, baseline)

    assert report["passed"] is False
    assert report["manifest_errors"] == [
        "candidate:rank-a/manifest-a.jsonl:1: filename escapes the manifest directory"
    ]


def test_manifest_rejects_symlinked_tensor_even_when_target_is_local(tmp_path):
    compare = _load_compare_module()
    candidate = tmp_path / "candidate"
    baseline = tmp_path / "baseline"
    candidate.mkdir()
    np.save(candidate / "real.npy", np.array([1.0], dtype=np.float32))
    (candidate / "linked.npy").symlink_to("real.npy")
    row = _row("linked.npy")
    _write_manifest(candidate, row, shape=(1,), dtype="float32")
    _write_dump(
        baseline,
        [(_row("baseline.npy"), np.array([1.0], dtype=np.float32))],
    )

    report = compare.compare_dump_directories(candidate, baseline)

    assert report["passed"] is False
    assert report["manifest_errors"] == [
        "candidate:manifest-p00000.jsonl:1: tensor path must not be a symlink"
    ]


def test_symlinked_manifest_is_rejected(tmp_path):
    compare = _load_compare_module()
    candidate = tmp_path / "candidate"
    baseline = tmp_path / "baseline"
    candidate.mkdir()
    real_manifest = candidate / "rows.jsonl"
    real_manifest.write_text("", encoding="utf-8")
    (candidate / "manifest-linked.jsonl").symlink_to(real_manifest.name)
    _write_dump(
        baseline,
        [(_row("baseline.npy"), np.array([1.0], dtype=np.float32))],
    )

    report = compare.compare_dump_directories(candidate, baseline)

    assert report["passed"] is False
    assert report["manifest_errors"] == [
        "candidate:manifest-linked.jsonl: manifest path must not be a symlink"
    ]


def test_compare_reports_non_utf8_manifest_without_crashing(tmp_path):
    compare = _load_compare_module()
    candidate = tmp_path / "candidate"
    baseline = tmp_path / "baseline"
    candidate.mkdir()
    (candidate / "manifest-p00000.jsonl").write_bytes(b"\xff\xfe\x00")
    _write_dump(
        baseline,
        [(_row("baseline.npy"), np.array([1.0], dtype=np.float32))],
    )

    report = compare.compare_dump_directories(candidate, baseline)

    assert report["passed"] is False
    assert report["manifest_errors"] == [
        "candidate:manifest-p00000.jsonl: manifest is not valid UTF-8"
    ]


def test_compare_reports_npz_as_unsupported_loadable(tmp_path):
    compare = _load_compare_module()
    candidate = tmp_path / "candidate"
    baseline = tmp_path / "baseline"
    key = _row("candidate.npz")
    candidate.mkdir()
    np.savez(candidate / key["filename"], values=np.array([1.0], dtype=np.float32))
    _write_manifest(candidate, key, shape=(1,), dtype="float32")
    _write_dump(
        baseline,
        [
            (
                {**key, "filename": "baseline.npy"},
                np.array([1.0], dtype=np.float32),
            )
        ],
    )

    report = compare.compare_dump_directories(candidate, baseline)

    assert report["passed"] is False
    assert report["comparisons"][0]["failures"] == [
        "candidate: loaded object is not an ndarray: NpzFile"
    ]
    json.dumps(report, sort_keys=True, allow_nan=False)


def test_load_array_closes_unsupported_loadable(monkeypatch, tmp_path):
    compare = _load_compare_module()

    class UnsupportedLoadable:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    loadable = UnsupportedLoadable()
    monkeypatch.setattr(compare.np, "load", lambda *_args, **_kwargs: loadable)
    record = {
        "array_path": tmp_path / "unsupported",
        "declared_shape": (1,),
        "declared_dtype": "float32",
    }

    array, errors = compare._load_array(record, "candidate")

    assert array is None
    assert errors == [
        "candidate: loaded object is not an ndarray: UnsupportedLoadable"
    ]
    assert loadable.closed is True


@pytest.mark.parametrize(
    ("filename", "contents", "expected_failure"),
    [
        (
            "empty.npy",
            b"",
            "candidate: cannot load tensor: unexpected end of file",
        ),
        (
            "truncated.npz",
            b"PK\x03\x04truncated",
            "candidate: cannot load tensor: invalid ZIP archive",
        ),
    ],
)
def test_cli_reports_truncated_tensor_files_as_deterministic_json_failure(
    tmp_path, filename, contents, expected_failure
):
    candidate = tmp_path / "candidate"
    baseline = tmp_path / "baseline"
    output = tmp_path / "report.json"
    key = _row(filename)
    _write_manifest(candidate, key, shape=(1,), dtype="float32")
    (candidate / filename).write_bytes(contents)
    _write_dump(
        baseline,
        [
            (
                {**key, "filename": "baseline.npy"},
                np.array([1.0], dtype=np.float32),
            )
        ],
    )
    command = [
        sys.executable,
        str(COMPARE_PATH),
        "--candidate",
        str(candidate),
        "--baseline",
        str(baseline),
        "--output",
        str(output),
    ]

    first = subprocess.run(command, text=True, capture_output=True, check=False)
    first_text = output.read_text(encoding="utf-8")
    second = subprocess.run(command, text=True, capture_output=True, check=False)
    second_text = output.read_text(encoding="utf-8")

    assert first.returncode == second.returncode == 1
    assert first.stderr == second.stderr == ""
    assert first_text == second_text
    assert first.stdout == second.stdout
    report = json.loads(first_text, parse_constant=lambda value: pytest.fail(value))
    assert report["passed"] is False
    assert report["comparisons"][0]["failures"] == [expected_failure]
    assert report["first_failing_key"] == report["comparisons"][0]["key"]


@pytest.mark.parametrize(
    ("dtype", "expected_dtype"),
    [
        (np.float16, np.float32),
        (np.float32, np.float32),
        (np.float64, np.float64),
        (np.int64, np.int64),
        (np.uint32, np.uint32),
        (np.bool_, np.bool_),
    ],
)
def test_metric_values_only_promote_low_precision_floats(dtype, expected_dtype):
    compare = _load_compare_module()

    values = compare._as_metric_values(np.array([0, 1], dtype=dtype))

    assert values.dtype == np.dtype(expected_dtype)


def test_metric_values_promote_bfloat16_to_float32():
    compare = _load_compare_module()
    bfloat16 = pytest.importorskip("ml_dtypes").bfloat16

    values = compare._as_metric_values(np.array([0, 1], dtype=bfloat16))

    assert values.dtype == np.dtype(np.float32)


def test_integer_difference_metrics_preserve_units_above_float32_exact_range():
    compare = _load_compare_module()

    metrics = compare._metrics(
        np.array([16_777_217], dtype=np.int64),
        np.array([16_777_216], dtype=np.int64),
        tensor_name="selected_counts",
    )

    assert metrics["max_abs"] == 1.0


def test_complex_metrics_are_rejected_instead_of_dropping_imaginary_values():
    compare = _load_compare_module()

    with pytest.raises(TypeError, match="complex tensors are unsupported"):
        compare._metrics(
            np.array([1 + 2j], dtype=np.complex64),
            np.array([1 + 3j], dtype=np.complex64),
            tensor_name="hidden_states",
        )


def test_max_float_opposing_signs_produce_finite_strict_json_metrics(tmp_path):
    compare = _load_compare_module()
    candidate = tmp_path / "candidate"
    baseline = tmp_path / "baseline"
    maximum = np.finfo(np.float32).max
    key = _row("candidate.npy")
    _write_dump(candidate, [(key, np.array([maximum], dtype=np.float32))])
    _write_dump(
        baseline,
        [
            (
                {**key, "filename": "baseline.npy"},
                np.array([-maximum], dtype=np.float32),
            )
        ],
    )

    report = compare.compare_dump_directories(candidate, baseline)

    assert report["passed"] is True
    metrics = report["comparisons"][0]["metrics"]
    assert metrics["max_abs"] == pytest.approx(float(maximum) * 2.0)
    assert all(
        value is None or math.isfinite(value) for value in metrics.values()
    )
    json.dumps(report, sort_keys=True, allow_nan=False)


@pytest.mark.parametrize(
    ("candidate_array", "baseline_array", "expected_field"),
    [
        (
            np.ones((2, 3), dtype=np.float32),
            np.ones((3, 2), dtype=np.float32),
            "shape_match",
        ),
        (
            np.ones((2, 3), dtype=np.float16),
            np.ones((2, 3), dtype=np.float32),
            "dtype_match",
        ),
    ],
)
def test_compare_rejects_shape_and_dtype_mismatches(
    tmp_path, candidate_array, baseline_array, expected_field
):
    compare = _load_compare_module()
    candidate = tmp_path / "candidate"
    baseline = tmp_path / "baseline"
    _write_dump(candidate, [(_row("candidate.npy"), candidate_array)])
    _write_dump(baseline, [(_row("baseline.npy"), baseline_array)])

    report = compare.compare_dump_directories(candidate, baseline)

    assert report["passed"] is False
    comparison = report["comparisons"][0]
    assert comparison[expected_field] is False
    assert comparison["metrics"] == {
        "max_abs": None,
        "mean_abs": None,
        "p99_abs": None,
        "cosine": None,
        "topk_overlap": None,
    }
    assert report["first_failing_key"] == comparison["key"]


def test_compare_rejects_missing_and_duplicate_semantic_keys(tmp_path):
    compare = _load_compare_module()
    candidate = tmp_path / "candidate"
    baseline = tmp_path / "baseline"
    duplicate_key = _row("candidate-a.npy")
    _write_dump(
        candidate,
        [
            (duplicate_key, np.array([1.0], dtype=np.float32)),
            (
                {**duplicate_key, "filename": "candidate-b.npy"},
                np.array([1.0], dtype=np.float32),
            ),
        ],
    )
    _write_dump(
        baseline,
        [
            (_row("baseline-a.npy"), np.array([1.0], dtype=np.float32)),
            (
                _row("baseline-extra.npy", name="attention_output"),
                np.array([2.0], dtype=np.float32),
            ),
        ],
    )

    report = compare.compare_dump_directories(candidate, baseline)

    assert report["passed"] is False
    assert len(report["manifest_errors"]) == 1
    assert "duplicate semantic key" in report["manifest_errors"][0]
    assert report["missing_from_candidate"] == [
        {
            "component": "decoder_layer",
            "layer": 3,
            "forward_mode": "extend",
            "name": "attention_output",
            "occurrence": 0,
            "process": 0,
        }
    ]
    assert report["first_failing_key"] is not None


def test_compare_retains_metrics_and_first_failing_key_when_thresholds_fail(tmp_path):
    compare = _load_compare_module()
    candidate = tmp_path / "candidate"
    baseline = tmp_path / "baseline"
    key = _row("candidate.npy")
    _write_dump(candidate, [(key, np.array([1.0, 2.25], dtype=np.float32))])
    _write_dump(
        baseline,
        [({**key, "filename": "baseline.npy"}, np.array([1.0, 2.0], dtype=np.float32))],
    )

    report = compare.compare_dump_directories(candidate, baseline, max_abs=0.1)

    assert report["passed"] is False
    assert report["comparisons"][0]["metrics"]["max_abs"] == pytest.approx(0.25)
    assert report["comparisons"][0]["failures"] == ["max_abs > 0.1"]
    assert report["first_failing_key"] == report["comparisons"][0]["key"]
    assert report["thresholds"] == {
        "max_abs": 0.1,
        "max_mean_abs": None,
        "max_p99_abs": None,
        "min_cosine": None,
        "min_topk_overlap": None,
    }


def test_cli_emits_deterministic_strict_json_report(tmp_path, capsys):
    compare = _load_compare_module()
    candidate = tmp_path / "candidate"
    baseline = tmp_path / "baseline"
    output = tmp_path / "report.json"
    key = _row("candidate.npy")
    _write_dump(candidate, [(key, np.array([0.0, 0.0], dtype=np.float32))])
    _write_dump(
        baseline,
        [({**key, "filename": "baseline.npy"}, np.array([0.0, 0.0], dtype=np.float32))],
    )

    first_exit = compare.main(
        [
            "--candidate",
            str(candidate),
            "--baseline",
            str(baseline),
            "--output",
            str(output),
        ]
    )
    first_text = output.read_text(encoding="utf-8")
    first_stdout = capsys.readouterr().out
    second_exit = compare.main(
        [
            "--candidate",
            str(candidate),
            "--baseline",
            str(baseline),
            "--output",
            str(output),
        ]
    )
    second_text = output.read_text(encoding="utf-8")
    second_stdout = capsys.readouterr().out

    assert first_exit == second_exit == 0
    assert first_text == second_text
    assert first_stdout == second_stdout
    parsed = json.loads(first_text, parse_constant=lambda value: pytest.fail(value))
    assert parsed["passed"] is True
    assert parsed["comparisons"][0]["metrics"]["cosine"] is None
    assert first_text.endswith("\n")
    assert math.isfinite(parsed["comparisons"][0]["metrics"]["max_abs"])
