import importlib.util
import json
import math
import stat
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
COMPARE_PATH = ROOT / "scripts/kernels/compare_glm52_e2e_results.py"
RUNNER_PATH = ROOT / "scripts/kernels/run_glm52_dsa_v7x32_real_e2e.sh"


def _load_compare_module():
    assert COMPARE_PATH.is_file(), f"missing comparator: {COMPARE_PATH}"
    spec = importlib.util.spec_from_file_location("glm52_e2e_compare", COMPARE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _response(
    output_ids=(42, 43),
    output_logprobs=(-0.10, -0.20),
    top_rows=(
        ((-0.10, 42), (-0.30, 7), (-0.50, 8)),
        ((-0.20, 43), (-0.40, 9), (-0.60, 10)),
    ),
):
    return {
        "output_ids": list(output_ids),
        "meta_info": {
            "prompt_tokens": 4,
            "completion_tokens": len(output_ids),
            "output_token_logprobs": [
                [logprob, token_id, None]
                for logprob, token_id in zip(output_logprobs, output_ids, strict=True)
            ],
            "output_top_logprobs": [
                [[logprob, token_id, None] for logprob, token_id in row]
                for row in top_rows
            ],
        },
    }


def test_compare_accepts_identical_single_and_batched_responses():
    compare = _load_compare_module()

    single = compare.compare_responses(
        _response(),
        _response(),
        max_logprob_abs_error=0.05,
        min_topk_overlap=0.9,
        expected_topk_width=3,
    )
    batched = compare.compare_responses(
        [
            _response(),
            _response(
                output_ids=(11,),
                output_logprobs=(-0.7,),
                top_rows=(((-0.7, 11), (-0.8, 12), (-0.9, 13)),),
            ),
        ],
        [
            _response(),
            _response(
                output_ids=(11,),
                output_logprobs=(-0.7,),
                top_rows=(((-0.7, 11), (-0.8, 12), (-0.9, 13)),),
            ),
        ],
        max_logprob_abs_error=0.05,
        min_topk_overlap=0.9,
        expected_topk_width=3,
    )

    assert single["passed"] is True
    assert single["response_count"] == 1
    assert single["max_output_logprob_abs_error"] == 0.0
    assert single["min_topk_overlap"] == 1.0
    assert batched["passed"] is True
    assert batched["response_count"] == 2


def test_compare_rejects_different_output_ids():
    compare = _load_compare_module()

    report = compare.compare_responses(
        _response(),
        _response(output_ids=(42, 99)),
        max_logprob_abs_error=0.05,
        min_topk_overlap=0.9,
        expected_topk_width=3,
    )

    assert report["passed"] is False
    assert report["output_ids_equal"] is False


def test_compare_rejects_non_finite_output_logprobs():
    compare = _load_compare_module()

    report = compare.compare_responses(
        _response(output_logprobs=(math.nan, -0.2)),
        _response(),
        max_logprob_abs_error=0.05,
        min_topk_overlap=0.9,
        expected_topk_width=3,
    )

    assert report["passed"] is False
    assert report["finite_output_logprobs"] is False


def test_compare_reports_logprob_error_and_token_aligned_topk_overlap():
    compare = _load_compare_module()
    candidate = _response(
        output_logprobs=(-0.12, -0.17),
        top_rows=(
            ((-0.12, 42), (-0.31, 7), (-0.90, 100)),
            ((-0.17, 43), (-0.44, 9), (-0.80, 101)),
        ),
    )

    report = compare.compare_responses(
        candidate,
        _response(),
        max_logprob_abs_error=0.05,
        min_topk_overlap=2 / 3,
        expected_topk_width=3,
    )

    assert report["passed"] is True
    assert math.isclose(report["max_output_logprob_abs_error"], 0.03)
    assert math.isclose(report["min_topk_overlap"], 2 / 3)


def test_compare_rejects_error_missing_metadata_and_incomplete_rows():
    compare = _load_compare_module()
    missing_logprob = _response()
    missing_logprob["meta_info"]["output_token_logprobs"].pop()
    missing_top_row = _response()
    missing_top_row["meta_info"]["output_top_logprobs"].pop()
    truncated_top_rows = _response()
    for row in truncated_top_rows["meta_info"]["output_top_logprobs"]:
        row.pop()

    malformed = (
        {"error": "load failed"},
        {"output_ids": [42]},
        missing_logprob,
        missing_top_row,
        truncated_top_rows,
    )
    for candidate in malformed:
        report = compare.compare_responses(
            candidate,
            candidate,
            max_logprob_abs_error=0.05,
            min_topk_overlap=0.9,
            expected_topk_width=3,
        )
        assert report["passed"] is False
        assert report["schema_valid"] is False


def test_compare_rejects_logprob_token_mismatch_and_emits_strict_json():
    compare = _load_compare_module()
    candidate = _response()
    candidate["meta_info"]["output_token_logprobs"][0][1] = 99

    report = compare.compare_responses(
        candidate,
        _response(),
        max_logprob_abs_error=0.05,
        min_topk_overlap=0.9,
        expected_topk_width=3,
    )

    assert report["passed"] is False
    assert report["schema_valid"] is False
    assert report["max_output_logprob_abs_error"] is None
    json.dumps(report, allow_nan=False)


def test_real_runner_has_checkpoint_backend_request_and_teardown_gates():
    assert RUNNER_PATH.is_file(), f"missing real E2E runner: {RUNNER_PATH}"
    runner = RUNNER_PATH.read_text(encoding="utf-8")

    for required in (
        "GLM52_DSA_RUN_ID",
        "GLM52_ATTENTION_BACKEND",
        'MODEL_PATH="${GLM52_MODEL_PATH:-/models/GLM-5.2}"',
        'COMPLETE_MARKER="${MODEL_PATH}/_DOWNLOAD_COMPLETE"',
        "--load-format safetensors",
        "--attention-backend \"$ATTENTION_BACKEND\"",
        "all_followers_acked",
        "request_names.txt",
        '"return_logprob": True',
        '"top_logprobs_num": 20',
        "GLM52_DSA_REAL_E2E_OK",
    ):
        assert required in runner

    assert 'if [[ "$ATTENTION_BACKEND" != "dsa" && "$ATTENTION_BACKEND" != "fa" ]]' in runner
    assert 'if [[ ! -f "$COMPLETE_MARKER" ]]' in runner
    assert RUNNER_PATH.stat().st_mode & stat.S_IXUSR
    assert COMPARE_PATH.stat().st_mode & stat.S_IXUSR
    assert "setsid" in runner
    assert 'kill -TERM -- "-$SERVER_PGID"' in runner
    assert "expected_topk_width=20" in runner
    assert 'SGLANG_JAX_SKIP_GCSFUSE_WARMUP="${SGLANG_JAX_SKIP_GCSFUSE_WARMUP:-1}"' in runner
    assert "GLM52_DSA_SOURCE_REV" in runner


def test_real_runner_supports_smoke_and_boundary_request_profiles():
    runner = RUNNER_PATH.read_text(encoding="utf-8")

    assert 'REQUEST_PROFILE="${GLM52_DSA_REQUEST_PROFILE:-smoke}"' in runner
    assert 'MAX_NEW_TOKENS="${GLM52_DSA_MAX_NEW_TOKENS:-2}"' in runner
    assert 'if [[ "$REQUEST_PROFILE" != "smoke" && "$REQUEST_PROFILE" != "boundary" ]]' in runner
    for length in (2047, 2048, 2049, 3072):
        assert f'"boundary_{length}"' in runner
    assert '"ignore_eos": True' in runner
    assert 'config["vocab_size"]' in runner
    assert "0 <= token_id < vocab_size" in runner


def test_real_runner_derives_expected_counts_from_generated_requests():
    runner = RUNNER_PATH.read_text(encoding="utf-8")

    assert 'request_names = (out / "request_names.txt").read_text' in runner
    assert 'request = json.loads((out / f"{name}.request.json").read_text())' in runner
    assert 'expected_counts = input_token_counts(request["input_ids"])' in runner
    assert 'expected_completion_tokens = request["sampling_params"]["max_new_tokens"]' in runner
    assert 'expected_prompt_tokens = {"short": [4]' not in runner


def test_real_runner_exports_rank_local_debug_dump_directory_when_enabled():
    runner = RUNNER_PATH.read_text(encoding="utf-8")
    dump_gate = 'if [[ "${SGLANG_JAX_DEBUG_DUMP:-0}" == "1" ]]; then'
    dump_export = 'export SGLANG_JAX_DEBUG_DUMP_DIR="$OUT/debug_dumps"'

    assert dump_gate in runner
    assert dump_export in runner
    assert runner.index(dump_export) < runner.index("setsid")
