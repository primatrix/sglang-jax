import importlib.util
import json
import math
import stat
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
COMPARE_PATH = ROOT / "scripts/kernels/compare_glm52_e2e_results.py"
RUNNER_PATH = ROOT / "scripts/kernels/run_glm52_dsa_v7x32_real_e2e.sh"
REQUEST_GENERATOR_MARKER = (
    '"$PYBIN" - "$OUT" "$REQUEST_PROFILE" "$MAX_NEW_TOKENS" "$MODEL_PATH" '
    "<<'PY'\n"
)


def _load_compare_module():
    assert COMPARE_PATH.is_file(), f"missing comparator: {COMPARE_PATH}"
    spec = importlib.util.spec_from_file_location("glm52_e2e_compare", COMPARE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _request_generator_source():
    runner = RUNNER_PATH.read_text(encoding="utf-8")
    assert REQUEST_GENERATOR_MARKER in runner
    remainder = runner.split(REQUEST_GENERATOR_MARKER, maxsplit=1)[1]
    source, separator, _remainder = remainder.partition("\nPY\n")
    assert separator
    return source


def _run_request_generator(tmp_path, *, profile, max_new_tokens, vocab_size=256):
    model_dir = tmp_path / f"{profile}-model"
    output_dir = tmp_path / f"{profile}-output"
    model_dir.mkdir()
    output_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"vocab_size": vocab_size}), encoding="utf-8"
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-",
            str(output_dir),
            profile,
            str(max_new_tokens),
            str(model_dir),
        ],
        input=_request_generator_source(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    request_names = (output_dir / "request_names.txt").read_text(
        encoding="utf-8"
    ).splitlines()
    requests = {
        name: json.loads(
            (output_dir / f"{name}.request.json").read_text(encoding="utf-8")
        )
        for name in request_names
    }
    return requests


def _input_rows(input_ids):
    return input_ids if input_ids and isinstance(input_ids[0], list) else [input_ids]


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
    assert (
        'if [[ "$REQUEST_PROFILE" != "smoke" && "$REQUEST_PROFILE" != "boundary" '
        '&& "$REQUEST_PROFILE" != "boundary_single" '
        '&& "$REQUEST_PROFILE" != "precompile_repeat" ]]' in runner
    )
    for length in (2047, 2048, 2049, 3072):
        assert f'"boundary_{length}"' in runner
    assert '"ignore_eos": True' in runner
    assert 'config["vocab_size"]' in runner
    assert "0 <= token_id < vocab_size" in runner


@pytest.mark.parametrize(
    ("profile", "max_new_tokens", "expected_lengths"),
    [
        (
            "smoke",
            2,
            {"short": [4], "chunked": [257], "ragged": [9, 133]},
        ),
        (
            "boundary",
            1,
            {
                "boundary_2047": [2047],
                "boundary_2048": [2048],
                "boundary_2049": [2049],
                "boundary_3072": [3072],
            },
        ),
        (
            "boundary_single",
            1,
            {"boundary_3072": [3072]},
        ),
        (
            "precompile_repeat",
            1,
            {"precompile_first": [3072], "precompile_repeat": [3072]},
        ),
    ],
)
def test_real_runner_executes_request_generator_profiles(
    tmp_path, profile, max_new_tokens, expected_lengths
):
    vocab_size = 256

    requests = _run_request_generator(
        tmp_path,
        profile=profile,
        max_new_tokens=max_new_tokens,
        vocab_size=vocab_size,
    )

    assert list(requests) == list(expected_lengths)
    for name, expected in expected_lengths.items():
        payload = requests[name]
        rows = _input_rows(payload["input_ids"])
        assert [len(row) for row in rows] == expected
        assert all(0 <= token_id < vocab_size for row in rows for token_id in row)
        assert payload["sampling_params"]["ignore_eos"] is True
        assert payload["sampling_params"]["max_new_tokens"] == max_new_tokens
        assert payload["return_logprob"] is True

    if profile == "precompile_repeat":
        assert requests["precompile_first"] == requests["precompile_repeat"]
        assert payload["top_logprobs_num"] == 20


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


def test_real_runner_installs_failure_trap_before_rank_local_preflight():
    runner = RUNNER_PATH.read_text(encoding="utf-8")

    trap_index = runner.index("trap finish_server EXIT")
    for rank_local_preflight in (
        'if [[ "$ATTENTION_BACKEND" != "dsa"',
        'if [[ "$REQUEST_PROFILE" != "smoke"',
        'if [[ ! -f "$COMPLETE_MARKER" ]]',
        'mkdir -p "$OUT"',
    ):
        assert trap_index < runner.index(rank_local_preflight)


def test_real_runner_streams_rank_zero_startup_progress_to_falcon_stdout():
    runner = RUNNER_PATH.read_text(encoding="utf-8")

    assert 'SERVER_LOG_MONITOR_PID=""' in runner
    assert "start_server_log_monitor()" in runner
    assert "stop_server_log_monitor()" in runner
    assert 'if [[ "$RANK" != "0" ]]; then' in runner
    assert 'tail -n +1 -F "$SERVER_LOG"' in runner
    assert "tr '\\r' '\\n'" in runner
    assert "Scanning metadata|Starting parallel weight loading" in runner
    assert "Precompile finished|Application startup complete" in runner
    assert runner.index("start_server_log_monitor") < runner.index("health_deadline=")


def test_real_runner_rendezvous_waits_for_every_rank_before_server_launch():
    runner = RUNNER_PATH.read_text(encoding="utf-8")

    ready_index = runner.index('touch "${CONTROL_DIR}/READY-rank-${RANK}"')
    wait_index = runner.index("all_ranks_ready")
    all_ready_index = runner.index('touch "$ALL_READY"')
    launch_index = runner.index('setsid "$PYBIN"')
    assert ready_index < all_ready_index < launch_index
    assert wait_index < launch_index
    assert 'if has_failures; then' in runner[ready_index:launch_index]


def test_real_runner_preserves_ep_capacity_when_skipping_precompile():
    runner = RUNNER_PATH.read_text(encoding="utf-8")

    assert "GLM52_DSA_MAX_RUNNING_REQUESTS" not in runner
    assert 'DISABLE_PRECOMPILE="${GLM52_DSA_DISABLE_PRECOMPILE:-0}"' in runner
    assert "--max-running-requests 64" in runner
    assert 'if [[ "$DISABLE_PRECOMPILE" == "1" ]]; then' in runner
    assert 'SERVER_ARGS+=(--disable-precompile)' in runner
    assert 'if [[ "$DISABLE_PRECOMPILE" != "0" && "$DISABLE_PRECOMPILE" != "1" ]]' in runner


def test_real_runner_precompiles_dsa_context_buckets():
    runner = RUNNER_PATH.read_text(encoding="utf-8")

    assert "--precompile-dsa-context-paddings 512 1024 2048 4096" in runner
    assert "--precompile-top-logprobs 20" in runner
    assert 'echo "dsa_context_paddings=512,1024,2048,4096"' in runner


def test_real_runner_can_isolate_attention_dumps_with_epmoe():
    runner = RUNNER_PATH.read_text(encoding="utf-8")

    assert 'MOE_BACKEND="${GLM52_MOE_BACKEND:-fused}"' in runner
    assert 'if [[ "$MOE_BACKEND" != "fused" && "$MOE_BACKEND" != "epmoe" ]]' in runner
    assert 'echo "moe_backend=$MOE_BACKEND"' in runner
    assert '--moe-backend "$MOE_BACKEND"' in runner
