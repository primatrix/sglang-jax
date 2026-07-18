#!/usr/bin/env bash

set -euo pipefail

ROOT="${GLM52_DSA_ROOT:-/tmp/glm52-dsa-v7x32/sglang-jax}"
PYBIN="${GLM52_DSA_PYBIN:-/opt/venv/bin/python3}"
RANK="${FALCON_RANK:-${FALCON_JAX_PROCESS_ID:-${JOB_COMPLETION_INDEX:-0}}}"
NNODES="${FALCON_WORLD_SIZE:-${FALCON_JAX_PROCESS_COUNT:-4}}"
DIST_ADDR="${FALCON_JAX_COORDINATOR_ADDRESS:?FALCON_JAX_COORDINATOR_ADDRESS missing}"
PORT="${GLM52_DSA_PORT:-30272}"
ARTIFACT_ROOT="${ARTIFACT_LOCAL_DIR:-/tmp/glm52-dsa-artifacts}"
RUN_ID="${GLM52_DSA_RUN_ID:-}"
MODEL_PATH="${GLM52_MODEL_PATH:-/models/GLM-5.2}"
COMPLETE_MARKER="${MODEL_PATH}/_DOWNLOAD_COMPLETE"
ATTENTION_BACKEND="${GLM52_ATTENTION_BACKEND:-dsa}"
START_TIMEOUT_SECONDS="${GLM52_DSA_START_TIMEOUT_SECONDS:-300}"
HEALTH_TIMEOUT_SECONDS="${GLM52_DSA_HEALTH_TIMEOUT_SECONDS:-10800}"
GENERATE_TIMEOUT_SECONDS="${GLM52_DSA_GENERATE_TIMEOUT_SECONDS:-1200}"
SHUTDOWN_TIMEOUT_SECONDS="${GLM52_DSA_SHUTDOWN_TIMEOUT_SECONDS:-90}"
ACK_TIMEOUT_SECONDS="${GLM52_DSA_ACK_TIMEOUT_SECONDS:-180}"
export SGLANG_JAX_SKIP_GCSFUSE_WARMUP="${SGLANG_JAX_SKIP_GCSFUSE_WARMUP:-1}"
SOURCE_REV="${GLM52_DSA_SOURCE_REV:-}"
if [[ -z "$SOURCE_REV" ]]; then
  SOURCE_REV="$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || true)"
fi
SOURCE_REV="${SOURCE_REV:-unknown}"
MIN_FOLLOWER_TIMEOUT_SECONDS=$((
  START_TIMEOUT_SECONDS + HEALTH_TIMEOUT_SECONDS + 3 * GENERATE_TIMEOUT_SECONDS +
  SHUTDOWN_TIMEOUT_SECONDS + ACK_TIMEOUT_SECONDS + 600
))
FOLLOWER_TIMEOUT_SECONDS="${GLM52_DSA_FOLLOWER_TIMEOUT_SECONDS:-$MIN_FOLLOWER_TIMEOUT_SECONDS}"

if [[ -z "$RUN_ID" ]]; then
  echo "GLM52_DSA_RUN_ID must be a unique value shared by all ranks" >&2
  exit 2
fi
if [[ "$ATTENTION_BACKEND" != "dsa" && "$ATTENTION_BACKEND" != "fa" ]]; then
  echo "GLM52_ATTENTION_BACKEND must be dsa or fa, got: ${ATTENTION_BACKEND}" >&2
  exit 2
fi
if [[ ! -f "$COMPLETE_MARKER" ]]; then
  echo "checkpoint completion marker is missing: ${COMPLETE_MARKER}" >&2
  exit 2
fi

OUT="${ARTIFACT_ROOT}/rank-${RANK}/${RUN_ID}"
CONTROL_PARENT="${ARTIFACT_ROOT}/control"
CONTROL_DIR="${CONTROL_PARENT}/${RUN_ID}"
START="${CONTROL_DIR}/START"
SUCCESS="${CONTROL_DIR}/SUCCESS"
STOP="${CONTROL_DIR}/STOP"
FAIL_RANK="${CONTROL_DIR}/FAIL-rank-${RANK}"
ACK_RANK="${CONTROL_DIR}/ACK-rank-${RANK}"
SERVER_LOG="/tmp/tpu_logs/glm52-real-${ATTENTION_BACKEND}-${RUN_ID}-rank${RANK}.log"
SERVER_PID=""
SERVER_PGID=""

mkdir -p "$OUT" "$CONTROL_PARENT"
if [[ "$RANK" == "0" ]]; then
  if ! mkdir "$CONTROL_DIR"; then
    echo "run id already exists; choose a new GLM52_DSA_RUN_ID: ${RUN_ID}" >&2
    exit 2
  fi
  touch "$START"
else
  start_deadline=$(($(date +%s) + START_TIMEOUT_SECONDS))
  while [[ ! -f "$START" ]]; do
    if (( $(date +%s) >= start_deadline )); then
      echo "timed out waiting for rank 0 start marker: ${START}" >&2
      exit 1
    fi
    sleep 2
  done
fi

stop_server() {
  if [[ -z "$SERVER_PID" ]]; then
    return
  fi
  if kill -0 "$SERVER_PID" 2>/dev/null; then
    kill -TERM -- "-$SERVER_PGID" 2>/dev/null || true
    shutdown_deadline=$(($(date +%s) + SHUTDOWN_TIMEOUT_SECONDS))
    while kill -0 "$SERVER_PID" 2>/dev/null; do
      if (( $(date +%s) >= shutdown_deadline )); then
        kill -KILL -- "-$SERVER_PGID" 2>/dev/null || true
        break
      fi
      sleep 1
    done
  fi
  wait "$SERVER_PID" 2>/dev/null || true
  # Catch multiprocessing helpers that outlive the launcher but remain in its
  # dedicated process group.
  kill -KILL -- "-$SERVER_PGID" 2>/dev/null || true
  SERVER_PID=""
  SERVER_PGID=""
}

has_failures() {
  compgen -G "${CONTROL_DIR}/FAIL-rank-*" >/dev/null
}

all_followers_acked() {
  local rank
  for ((rank = 1; rank < NNODES; rank++)); do
    [[ -f "${CONTROL_DIR}/ACK-rank-${rank}" ]] || return 1
  done
}

finish_server() {
  local status=$?
  trap - EXIT
  if (( status != 0 )); then
    touch "$FAIL_RANK" "$STOP" 2>/dev/null || true
  fi
  stop_server
  if [[ -f "$SERVER_LOG" ]]; then
    cp -f "$SERVER_LOG" "$OUT/server-rank${RANK}.log" || true
  fi
  exit "$status"
}
trap finish_server EXIT

touch "${CONTROL_DIR}/READY-rank-${RANK}"
cd "$ROOT"
export PYTHONPATH="$ROOT/python${PYTHONPATH:+:$PYTHONPATH}"

{
  echo "rank=$RANK"
  echo "nnodes=$NNODES"
  echo "dist_addr=$DIST_ADDR"
  echo "run_id=$RUN_ID"
  echo "commit=$SOURCE_REV"
  echo "model=$MODEL_PATH"
  echo "checkpoint_complete=$(cat "$COMPLETE_MARKER")"
  echo "load_format=safetensors"
  echo "parallelism=tp32_dp1_ep32"
  echo "attention_backend=$ATTENTION_BACKEND"
  echo "skip_gcsfuse_warmup=$SGLANG_JAX_SKIP_GCSFUSE_WARMUP"
  echo "created_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} | tee "$OUT/run_context.txt"

"$PYBIN" - "$MODEL_PATH" <<'PY' | tee "$OUT/checkpoint.txt"
import json
import pathlib
import sys

model_dir = pathlib.Path(sys.argv[1])
index = json.loads((model_dir / "model.safetensors.index.json").read_text())
shards = sorted(set(index["weight_map"].values()))
if len(shards) != 282:
    raise SystemExit(f"expected 282 checkpoint shards, got {len(shards)}")
missing = [name for name in shards if not (model_dir / name).is_file()]
empty = [name for name in shards if (model_dir / name).is_file() and (model_dir / name).stat().st_size == 0]
if missing or empty:
    raise SystemExit(f"invalid checkpoint: missing={missing}, empty={empty}")
print(f"GLM52_CHECKPOINT_READY shards={len(shards)} bytes={sum((model_dir / name).stat().st_size for name in shards)}")
PY

"$PYBIN" - <<'PY' | tee "$OUT/environment.txt"
import importlib.metadata as md
import jax

for package in ("jax", "jaxlib", "libtpu", "flax", "transformers", "sglang-jax"):
    try:
        print(f"{package}={md.version(package)}")
    except md.PackageNotFoundError:
        print(f"{package}=MISSING")
print(f"backend={jax.default_backend()}")
print(f"local_device_count={jax.local_device_count()}")
assert md.version("jax") == "0.9.0"
assert md.version("jaxlib") == "0.9.0"
assert md.version("libtpu") == "0.0.34"
assert jax.default_backend() == "tpu"
assert jax.local_device_count() == 8
PY

SERVER_ARGS=(
  --model-path "$MODEL_PATH"
  --trust-remote-code
  --load-format safetensors
  --skip-tokenizer-init
  --attention-backend "$ATTENTION_BACKEND"
  --enable-sequence-parallel
  --tp-size 32
  --dp-size 1
  --ep-size 32
  --moe-backend fused
  --nnodes "$NNODES"
  --node-rank "$RANK"
  --dist-init-addr "$DIST_ADDR"
  --device tpu
  --dtype bfloat16
  --host 0.0.0.0
  --port "$PORT"
  --page-size 128
  --context-length 4096
  --chunked-prefill-size 128
  --max-prefill-tokens 256
  --max-total-tokens 4096
  --max-running-requests 64
  --mem-fraction-static 0.95
  --disable-radix-cache
  --skip-server-warmup
  --log-level info
  --decode-log-interval 1
  --precompile-bs-paddings 1 2
  --precompile-token-paddings 128 256
)

{
  printf 'SERVER_COMMAND='
  printf '%q ' "$PYBIN" -u -m sgl_jax.launch_server "${SERVER_ARGS[@]}"
  printf '\n'
} | tee "$OUT/server_command.txt"

: > "$SERVER_LOG"
setsid "$PYBIN" -u -m sgl_jax.launch_server "${SERVER_ARGS[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
SERVER_PGID=$SERVER_PID

if [[ "$RANK" != "0" ]]; then
  follower_deadline=$(($(date +%s) + FOLLOWER_TIMEOUT_SECONDS))
  while true; do
    if has_failures; then
      echo "rank ${RANK} observed a peer failure" >&2
      tail -200 "$SERVER_LOG" >&2 || true
      exit 1
    fi
    if [[ -f "$STOP" ]]; then
      if [[ ! -f "$SUCCESS" ]]; then
        echo "rank ${RANK} observed stop without success" >&2
        exit 1
      fi
      stop_server
      touch "$ACK_RANK"
      exit 0
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "rank ${RANK} server exited before stop" >&2
      tail -200 "$SERVER_LOG" >&2 || true
      exit 1
    fi
    if (( $(date +%s) >= follower_deadline )); then
      echo "rank ${RANK} timed out waiting for completion" >&2
      exit 1
    fi
    sleep 5
  done
fi

health_deadline=$(($(date +%s) + HEALTH_TIMEOUT_SECONDS))
while true; do
  if has_failures; then
    echo "rank 0 observed a peer failure before health check completed" >&2
    exit 1
  fi
  if curl -sSf --connect-timeout 2 --max-time 5 "http://127.0.0.1:${PORT}/health" >/dev/null; then
    ready_at=$(date +%s)
    echo "server_ready_after_seconds=$((ready_at + HEALTH_TIMEOUT_SECONDS - health_deadline))" \
      | tee -a "$OUT/run_context.txt"
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "rank 0 server exited before health check" >&2
    tail -400 "$SERVER_LOG" >&2 || true
    exit 1
  fi
  if (( $(date +%s) >= health_deadline )); then
    echo "server did not become healthy within ${HEALTH_TIMEOUT_SECONDS} seconds" >&2
    tail -400 "$SERVER_LOG" >&2 || true
    exit 1
  fi
  sleep 10
done

"$PYBIN" - "$OUT" <<'PY'
import json
import pathlib
import sys

out = pathlib.Path(sys.argv[1])
requests = {
    "short": {
        "input_ids": [1, 2, 3, 4],
        "sampling_params": {"temperature": 0.0, "max_new_tokens": 2, "ignore_eos": True},
        "return_logprob": True,
        "top_logprobs_num": 20,
        "return_text_in_logprobs": False,
    },
    "chunked": {
        "input_ids": [100 + (index % 1000) for index in range(257)],
        "sampling_params": {"temperature": 0.0, "max_new_tokens": 2, "ignore_eos": True},
        "return_logprob": True,
        "top_logprobs_num": 20,
        "return_text_in_logprobs": False,
    },
    "ragged": {
        "input_ids": [
            [200 + index for index in range(9)],
            [500 + (index % 1000) for index in range(133)],
        ],
        "sampling_params": {"temperature": 0.0, "max_new_tokens": 2, "ignore_eos": True},
        "return_logprob": True,
        "top_logprobs_num": 20,
        "return_text_in_logprobs": False,
    },
}
for name, payload in requests.items():
    (out / f"{name}.request.json").write_text(json.dumps(payload), encoding="utf-8")
PY

for request_name in short chunked ragged; do
  curl -sS --fail-with-body \
    --connect-timeout 5 \
    --max-time "$GENERATE_TIMEOUT_SECONDS" \
    -X POST "http://127.0.0.1:${PORT}/generate" \
    -H 'Content-Type: application/json' \
    --data-binary "@${OUT}/${request_name}.request.json" \
    | tee "$OUT/${request_name}.json"
done

for request_name in short chunked ragged; do
  "$PYBIN" "$ROOT/scripts/kernels/compare_glm52_e2e_results.py" \
    --candidate "$OUT/${request_name}.json" \
    --baseline "$OUT/${request_name}.json" \
    --max-logprob-abs-error 0 \
    --min-topk-overlap 1 \
    --expected-topk-width 20 \
    --output "$OUT/${request_name}.schema.json"
done

"$PYBIN" - "$OUT" "$ATTENTION_BACKEND" <<'PY' | tee "$OUT/generate_check.txt"
import json
import math
import pathlib
import sys

out = pathlib.Path(sys.argv[1])
backend = sys.argv[2]
expected_prompt_tokens = {"short": [4], "chunked": [257], "ragged": [9, 133]}
expected_completion_tokens = 2
expected_topk_width=20
summary = {}
for name, expected_counts in expected_prompt_tokens.items():
    payload = json.loads((out / f"{name}.json").read_text())
    responses = payload if isinstance(payload, list) else [payload]
    if len(responses) != len(expected_counts):
        raise SystemExit(f"{name}: expected {len(expected_counts)} responses, got {len(responses)}")
    for index, (response, expected_prompt_count) in enumerate(zip(responses, expected_counts, strict=True)):
        if "error" in response:
            raise SystemExit(f"{name}[{index}] generation failed: {response['error']}")
        output_ids = response.get("output_ids")
        meta = response.get("meta_info", {})
        if not isinstance(output_ids, list) or len(output_ids) != expected_completion_tokens:
            raise SystemExit(
                f"{name}[{index}] expected {expected_completion_tokens} output ids: {response}"
            )
        if meta.get("prompt_tokens") != expected_prompt_count:
            raise SystemExit(
                f"{name}[{index}] prompt token mismatch: {meta.get('prompt_tokens')} != {expected_prompt_count}"
            )
        if meta.get("completion_tokens") != expected_completion_tokens:
            raise SystemExit(f"{name}[{index}] completion token count mismatch: {response}")
        output_logprobs = meta.get("output_token_logprobs") or []
        if len(output_logprobs) != expected_completion_tokens:
            raise SystemExit(f"{name}[{index}] output logprob count mismatch")
        if not all(math.isfinite(float(item[0])) for item in output_logprobs):
            raise SystemExit(f"{name}[{index}] has non-finite output logprobs")
        if [int(item[1]) for item in output_logprobs] != output_ids:
            raise SystemExit(f"{name}[{index}] output logprob token ids do not match output ids")
        top_rows = meta.get("output_top_logprobs") or []
        if len(top_rows) != expected_completion_tokens:
            raise SystemExit(f"{name}[{index}] missing output top-logprob rows")
        for row_index, row in enumerate(top_rows):
            if len(row) != expected_topk_width:
                raise SystemExit(
                    f"{name}[{index}] top-logprob row {row_index} has width {len(row)}"
                )
            if len({int(item[1]) for item in row}) != expected_topk_width:
                raise SystemExit(f"{name}[{index}] top-logprob row {row_index} has duplicate ids")
            if not all(math.isfinite(float(item[0])) for item in row):
                raise SystemExit(
                    f"{name}[{index}] top-logprob row {row_index} has non-finite values"
                )
    summary[name] = {
        "response_count": len(responses),
        "prompt_tokens": expected_counts,
        "output_ids": [response["output_ids"] for response in responses],
    }
(out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(f"GLM52_DSA_REAL_E2E_OK backend={backend} requests={len(summary)}")
PY

if has_failures; then
  echo "rank 0 observed a peer failure after generation" >&2
  exit 1
fi
touch "$SUCCESS" "$STOP"

ack_deadline=$(($(date +%s) + ACK_TIMEOUT_SECONDS))
until all_followers_acked; do
  if has_failures; then
    echo "rank 0 observed a follower failure during teardown" >&2
    exit 1
  fi
  if (( $(date +%s) >= ack_deadline )); then
    echo "timed out waiting for follower teardown acknowledgements" >&2
    exit 1
  fi
  sleep 1
done
