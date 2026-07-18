import importlib.util
import json
import math
from pathlib import Path

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
    _write_dump(
        candidate,
        [
            (float_key, np.array([1.0, 2.5, 2.0, 4.0], dtype=np.float32)),
            (topk_key, np.array([[1, 2, 9, 4]], dtype=np.int32)),
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
        ],
    )

    report = compare.compare_dump_directories(candidate, baseline)

    assert report["passed"] is True
    assert report["tensor_count"] == 2
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
