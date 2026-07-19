from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).parents[3] / "scripts/kernels/eval_glm52_gpqa_smoke.py"


def _load_module():
    assert SCRIPT.exists(), f"missing GPQA smoke evaluator: {SCRIPT}"
    spec = importlib.util.spec_from_file_location("eval_glm52_gpqa_smoke", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prepare_examples_matches_official_sampling_and_permutation() -> None:
    module = _load_module()
    rows = [
        {
            "Question": f"question-{index}",
            "Correct Answer": f"correct-{index}",
            "Incorrect Answer 1": f"wrong-1-{index}",
            "Incorrect Answer 2": f"wrong-2-{index}",
            "Incorrect Answer 3": f"wrong-3-{index}",
            "High-level domain": "Physics",
        }
        for index in range(8)
    ]

    examples = module.prepare_examples(rows, sample_size=3, seed=0)

    assert [example["row"]["Question"] for example in examples] == [
        "question-6",
        "question-7",
        "question-3",
    ]
    assert [example["permutation"] for example in examples] == [
        [0, 1, 2, 3],
        [2, 1, 3, 0],
        [1, 3, 0, 2],
    ]
    assert [example["correct_answer"] for example in examples] == ["A", "D", "C"]


def test_extract_answer_uses_official_tail_window() -> None:
    module = _load_module()
    response = "ANSWER: A" + ("x" * 1100) + "\nANSWER: C"

    assert module.extract_answer(response) == "C"
    assert module.extract_answer("answer: a") == "a"
    assert module.extract_answer("reasoning without the required final line") is None


def test_build_payload_enables_glm_thinking() -> None:
    module = _load_module()

    payload = module.build_payload(
        model="/models/GLM-5.2",
        prompt="question",
        max_tokens=2048,
        temperature=1.0,
        top_p=0.95,
        seed=4,
    )

    assert payload["messages"] == [{"role": "user", "content": "question"}]
    assert payload["chat_template_kwargs"] == {"enable_thinking": True}
    assert payload["max_tokens"] == 2048
    assert payload["seed"] == 4


def test_wilson_interval_reports_small_sample_uncertainty() -> None:
    module = _load_module()

    lower, upper = module.wilson_interval(correct=8, total=10)

    assert lower == pytest.approx(0.49016, abs=1e-5)
    assert upper == pytest.approx(0.94332, abs=1e-5)


def test_summarize_results_separates_truncation_and_request_errors() -> None:
    module = _load_module()
    results = [
        {
            "score": 1,
            "format_failure": False,
            "finish_reason": "stop",
            "request_error": None,
            "latency_seconds": 2.0,
        },
        {
            "score": 0,
            "format_failure": True,
            "finish_reason": "length",
            "request_error": None,
            "latency_seconds": 3.0,
        },
        {
            "score": 0,
            "format_failure": True,
            "finish_reason": None,
            "request_error": {"type": "TimeoutError", "message": "timed out"},
            "latency_seconds": 4.0,
        },
    ]

    summary = module.summarize_results(results, expected_total=4, wall_seconds=5.0)

    assert summary["correct"] == 1
    assert summary["total"] == 3
    assert summary["expected_total"] == 4
    assert summary["remaining"] == 1
    assert summary["scored"] == 2
    assert summary["official_style_accuracy_percent"] == 50.0
    assert summary["length_truncations"] == 1
    assert summary["request_errors"] == 1
    assert summary["completed"] == 1
    assert summary["completed_only_accuracy_percent"] == 100.0


def test_summarize_results_does_not_score_operational_failure_or_abort() -> None:
    module = _load_module()
    request_error = {
        "score": 0,
        "format_failure": True,
        "finish_reason": None,
        "request_error": {"type": "TimeoutError", "message": "timed out"},
        "latency_seconds": 4.0,
    }
    failed = module.summarize_results(
        [request_error], expected_total=1, wall_seconds=4.0
    )
    assert failed["scored"] == 0
    assert failed["official_style_accuracy_percent"] is None

    aborted = module.summarize_results(
        [
            {
                "score": 0,
                "format_failure": True,
                "finish_reason": "abort",
                "request_error": None,
                "latency_seconds": 1.0,
            }
        ],
        expected_total=1,
        wall_seconds=1.0,
    )
    assert aborted["completed"] == 0
    assert aborted["incomplete_finish_reasons"] == {"abort": 1}


def test_evaluate_one_records_per_item_request_error(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module()

    def fail_request(**_kwargs):
        raise TimeoutError("timed out")

    monkeypatch.setattr(module, "request_completion", fail_request)
    example = {
        "row": {"Record ID": "record-1", "High-level domain": "Physics"},
        "prompt": "question",
        "permutation": [0, 1, 2, 3],
        "correct_answer": "A",
    }

    result = module.evaluate_one(
        0,
        example,
        api_base="http://127.0.0.1:1/v1",
        api_key="EMPTY",
        model="model",
        max_tokens=16,
        temperature=1.0,
        top_p=0.95,
        seed=0,
        timeout=0.01,
    )

    assert result["score"] == 0
    assert result["format_failure"] is True
    assert result["request_error"]["type"] == "TimeoutError"


def test_resume_rejects_a_checkpoint_from_different_request_settings() -> None:
    module = _load_module()
    args = SimpleNamespace(
        api_base="http://127.0.0.1:30280/v1",
        model="/models/GLM-5.2",
        sample_size=4,
        concurrency=2,
        max_tokens=4096,
        temperature=1.0,
        top_p=0.95,
        request_seed=0,
    )
    examples = [
        {
            "row": {"Record ID": f"record-{index}", "High-level domain": "Physics"},
            "prompt": f"question-{index}",
            "permutation": [0, 1, 2, 3],
            "correct_answer": "A",
        }
        for index in range(4)
    ]
    fingerprint = module.prepared_examples_fingerprint(examples)
    checkpoint = module.build_report(
        args, [], wall_seconds=0.0, examples_fingerprint=fingerprint
    )
    module.validate_resume_checkpoint(checkpoint, args, examples)

    checkpoint["request"]["max_tokens"] = 2048
    with pytest.raises(ValueError, match="request settings"):
        module.validate_resume_checkpoint(checkpoint, args, examples)


def test_resume_rejects_different_examples_and_duplicate_indices() -> None:
    module = _load_module()
    args = SimpleNamespace(
        api_base="http://127.0.0.1:30280/v1",
        model="/models/GLM-5.2",
        sample_size=1,
        concurrency=1,
        max_tokens=4096,
        temperature=1.0,
        top_p=0.95,
        request_seed=0,
    )
    examples = [
        {
            "row": {"Record ID": "record-0", "High-level domain": "Physics"},
            "prompt": "question-0",
            "permutation": [0, 1, 2, 3],
            "correct_answer": "A",
        }
    ]
    fingerprint = module.prepared_examples_fingerprint(examples)
    result = {
        "index": 0,
        "record_id": "record-0",
        "domain": "Physics",
        "prompt": "question-0",
        "permutation": [0, 1, 2, 3],
        "correct_answer": "A",
        "request_error": None,
        "score": 1,
        "format_failure": False,
        "finish_reason": "stop",
        "latency_seconds": 1.0,
    }
    checkpoint = module.build_report(
        args, [result], wall_seconds=1.0, examples_fingerprint=fingerprint
    )

    changed_examples = [examples[0] | {"prompt": "different-question"}]
    with pytest.raises(ValueError, match="prepared examples"):
        module.validate_resume_checkpoint(checkpoint, args, changed_examples)

    duplicate_checkpoint = checkpoint | {"results": [result, result]}
    with pytest.raises(ValueError, match="duplicate result index"):
        module.validate_resume_checkpoint(duplicate_checkpoint, args, examples)


def test_build_report_marks_all_request_errors_as_failed() -> None:
    module = _load_module()
    args = SimpleNamespace(
        api_base="http://127.0.0.1:30280/v1",
        model="/models/GLM-5.2",
        sample_size=1,
        concurrency=1,
        max_tokens=4096,
        temperature=1.0,
        top_p=0.95,
        request_seed=0,
    )
    report = module.build_report(
        args,
        [
            {
                "index": 0,
                "score": 0,
                "format_failure": True,
                "finish_reason": None,
                "request_error": {"type": "TimeoutError", "message": "timed out"},
                "latency_seconds": 4.0,
            }
        ],
        wall_seconds=4.0,
        examples_fingerprint="fingerprint",
    )

    assert report["status"] == "failed"
    assert module.report_has_operational_errors(report) is True

    abort_result = {
        "index": 0,
        "score": 0,
        "format_failure": True,
        "finish_reason": "abort",
        "request_error": None,
        "latency_seconds": 1.0,
    }
    abort_report = module.build_report(
        args,
        [abort_result],
        wall_seconds=1.0,
        examples_fingerprint="fingerprint",
    )
    assert abort_report["status"] == "failed"
    assert module.report_has_operational_errors(abort_report) is True


def test_build_report_marks_partial_incomplete_finishes_as_errors() -> None:
    module = _load_module()
    args = SimpleNamespace(
        api_base="http://127.0.0.1:30280/v1",
        model="/models/GLM-5.2",
        sample_size=2,
        concurrency=2,
        max_tokens=4096,
        temperature=1.0,
        top_p=0.95,
        request_seed=0,
    )
    results = [
        {
            "index": 0,
            "score": 1,
            "format_failure": False,
            "finish_reason": "stop",
            "request_error": None,
            "latency_seconds": 1.0,
        },
        {
            "index": 1,
            "score": 0,
            "format_failure": True,
            "finish_reason": "content_filter",
            "request_error": None,
            "latency_seconds": 1.0,
        },
    ]

    report = module.build_report(
        args,
        results,
        wall_seconds=1.0,
        examples_fingerprint="fingerprint",
    )

    assert report["status"] == "complete_with_errors"
    assert module.report_has_operational_errors(report) is True
