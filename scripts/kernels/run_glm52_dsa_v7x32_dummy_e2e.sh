#!/usr/bin/env bash

set -euo pipefail

ROOT="${GLM52_DSA_ROOT:-/tmp/glm52-dsa-v7x32/sglang-jax}"
PYBIN="${GLM52_DSA_PYBIN:-/opt/venv/bin/python3}"
RANK="${FALCON_JAX_PROCESS_ID:-${JOB_COMPLETION_INDEX:-0}}"
NNODES="${FALCON_JAX_PROCESS_COUNT:-4}"
DIST_ADDR="${FALCON_JAX_COORDINATOR_ADDRESS:?FALCON_JAX_COORDINATOR_ADDRESS missing}"
PORT="${GLM52_DSA_PORT:-30272}"
ARTIFACT_ROOT="${ARTIFACT_LOCAL_DIR:-/tmp/glm52-dsa-artifacts}"
RUN_ID="${GLM52_DSA_RUN_ID:-}"
START_TIMEOUT_SECONDS="${GLM52_DSA_START_TIMEOUT_SECONDS:-300}"
HEALTH_TIMEOUT_SECONDS="${GLM52_DSA_HEALTH_TIMEOUT_SECONDS:-7200}"
FOLLOWER_TIMEOUT_SECONDS="${GLM52_DSA_FOLLOWER_TIMEOUT_SECONDS:-7200}"
GENERATE_TIMEOUT_SECONDS="${GLM52_DSA_GENERATE_TIMEOUT_SECONDS:-600}"
SHUTDOWN_TIMEOUT_SECONDS="${GLM52_DSA_SHUTDOWN_TIMEOUT_SECONDS:-60}"
ACK_TIMEOUT_SECONDS="${GLM52_DSA_ACK_TIMEOUT_SECONDS:-120}"

if [[ -z "$RUN_ID" ]]; then
  echo "GLM52_DSA_RUN_ID must be a unique value shared by all ranks" >&2
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
SERVER_LOG="/tmp/tpu_logs/glm52-dsa-dummy-${RUN_ID}-rank${RANK}.log"
SERVER_PID=""

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
    kill "$SERVER_PID" 2>/dev/null || true
    shutdown_deadline=$(($(date +%s) + SHUTDOWN_TIMEOUT_SECONDS))
    while kill -0 "$SERVER_PID" 2>/dev/null; do
      if (( $(date +%s) >= shutdown_deadline )); then
        kill -KILL "$SERVER_PID" 2>/dev/null || true
        break
      fi
      sleep 1
    done
  fi
  wait "$SERVER_PID" 2>/dev/null || true
  SERVER_PID=""
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
    touch "$FAIL_RANK" "$STOP"
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
  echo "commit=$(git rev-parse HEAD)"
  echo "model=zai-org/GLM-5.2"
  echo "load_format=dummy"
  echo "parallelism=tp32_dp1_ep32"
  echo "attention_backend=dsa"
  echo "created_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} | tee "$OUT/run_context.txt"

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
  --model-path zai-org/GLM-5.2
  --trust-remote-code
  --load-format dummy
  --skip-tokenizer-init
  --attention-backend dsa
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
  --max-prefill-tokens 128
  --max-total-tokens 4096
  # Fused EP-MoE requires at least 2 * ep_size request slots.
  --max-running-requests 64
  --mem-fraction-static 0.95
  --disable-radix-cache
  --skip-server-warmup
  --log-level info
  --decode-log-interval 1
  --precompile-bs-paddings 1
  --precompile-token-paddings 128
)

{
  printf 'SERVER_COMMAND='
  printf '%q ' "$PYBIN" -u -m sgl_jax.launch_server "${SERVER_ARGS[@]}"
  printf '\n'
} | tee "$OUT/server_command.txt"

: > "$SERVER_LOG"
"$PYBIN" -u -m sgl_jax.launch_server "${SERVER_ARGS[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

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
    tail -300 "$SERVER_LOG" >&2 || true
    exit 1
  fi
  if (( $(date +%s) >= health_deadline )); then
    echo "server did not become healthy within ${HEALTH_TIMEOUT_SECONDS} seconds" >&2
    tail -300 "$SERVER_LOG" >&2 || true
    exit 1
  fi
  sleep 10
done

curl -sS --fail-with-body \
  --connect-timeout 5 \
  --max-time "$GENERATE_TIMEOUT_SECONDS" \
  -X POST "http://127.0.0.1:${PORT}/generate" \
  -H 'Content-Type: application/json' \
  -d '{"input_ids":[1,2,3,4],"sampling_params":{"temperature":0.0,"max_new_tokens":2}}' \
  | tee "$OUT/generate.json"

"$PYBIN" - "$OUT/generate.json" <<'PY' | tee "$OUT/generate_check.txt"
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    response = json.load(handle)
if "error" in response:
    raise SystemExit(f"generation failed: {response['error']}")
meta_info = response.get("meta_info", {})
if meta_info.get("completion_tokens", 0) < 1:
    raise SystemExit(f"no decode token returned: {response}")
print("GLM52_DSA_DUMMY_PREFILL_DECODE_OK")
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
