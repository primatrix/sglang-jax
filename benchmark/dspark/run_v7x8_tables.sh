#!/usr/bin/env bash
set -euo pipefail

cd /tmp/sglang-jax
export TMPDIR="${TMPDIR:-/tmp/tpu_logs/tmp}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/tmp/tpu_logs/pip-cache}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/tpu_logs/uv-cache}"
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-300}"
export HF_HOME="${HF_HOME:-/tmp/tpu_logs/huggingface}"
export SGLANG_RECORD_STEP_TIME=1
RESULT_ROOT=/tmp/dspark_stage2_tables
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$UV_CACHE_DIR" "$HF_HOME" "$RESULT_ROOT"
ulimit -c 0

uv sync --project python --extra tpu
uv tool install --force evalscope==1.9.0

launch_dspark() {
  local server_log="$1"
  shift
  python/.venv/bin/python -m sgl_jax.launch_server \
    --model-path Qwen/Qwen3-8B \
    --speculative-algorithm DSPARK \
    --speculative-draft-model-path deepseek-ai/dspark_qwen3_8b_block7 \
    --speculative-num-steps 1 \
    --speculative-num-draft-tokens 7 \
    --speculative-eagle-topk 1 \
    --tp-size 8 \
    --dp-size 2 \
    --dtype bfloat16 \
    --attention-backend fa \
    --mem-fraction-static 0.65 \
    --page-size 64 \
    --chunked-prefill-size 2048 \
    --max-running-requests 512 \
    --context-length 16384 \
    --precompile-token-paddings 2048 \
    --decode-log-interval 1 \
    --trust-remote-code \
    --disable-radix-cache \
    --grammar-backend none \
    --host 0.0.0.0 \
    --port 30000 \
    "$@" >"$server_log" 2>&1 &
  SERVER_PID=$!
  export SERVER_PID
  for _ in $(seq 1 600); do
    status=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:30000/health || true)
    if [ "$status" = "200" ]; then
      return
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      tail -300 "$server_log"
      exit 1
    fi
    sleep 2
  done
  echo "Server health check timed out"
  tail -300 "$server_log"
  exit 1
}

stop_dspark() {
  kill "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
}

launch_dspark "$RESULT_ROOT/sps_server.log" \
  --precompile-bs-paddings 2 4 8 16 32 64 128 256 512
python/.venv/bin/python benchmark/dspark/collect_sps.py \
  --python python/.venv/bin/python \
  --base-url http://127.0.0.1:30000 \
  --model Qwen/Qwen3-8B \
  --concurrency 2 4 8 16 32 64 128 256 512 \
  --input-len 256 \
  --output-len 512 \
  --prompts-multiplier 2 \
  --output-dir "$RESULT_ROOT/sps"
stop_dspark

export SGL_JAX_DSPARK_STS_CAPTURE_PATH="$RESULT_ROOT/sts_capture.jsonl"
launch_dspark "$RESULT_ROOT/sts_server.log" --disable-precompile
/root/.local/bin/evalscope eval \
  --model Qwen/Qwen3-8B \
  --eval-type openai_api \
  --api-url http://127.0.0.1:30000/v1/chat/completions \
  --datasets gsm8k \
  --limit 500 \
  --eval-batch-size 64 \
  --generation-config '{"max_tokens":2048,"temperature":0.0,"extra_body":{"chat_template_kwargs":{"enable_thinking":false}}}' \
  --work-dir "$RESULT_ROOT/sts_eval" \
  --no-timestamp
stop_dspark

python/.venv/bin/python benchmark/dspark/fit_sts.py \
  "$RESULT_ROOT/sts_capture.jsonl" \
  --output "$RESULT_ROOT/sts_table.json" \
  --num-bins 15

python/.venv/bin/python - <<'PY'
import json
from importlib.metadata import version

import jax

print("JAX_VERSION=", jax.__version__)
print("LIBTPU_VERSION=", version("libtpu"))
print("SPS_TABLE_BEGIN")
print(open("/tmp/dspark_stage2_tables/sps/sps_table.json", encoding="utf-8").read())
print("SPS_TABLE_END")
print("STS_TABLE_BEGIN")
print(open("/tmp/dspark_stage2_tables/sts_table.json", encoding="utf-8").read())
print("STS_TABLE_END")
report = json.load(
    open(
        "/tmp/dspark_stage2_tables/sts_eval/reports/Qwen3-8B/gsm8k.json",
        encoding="utf-8",
    )
)
print("STS_EVAL_SCORE=", report["score"])
print("STS_EVAL_NUM=", report["num"])
PY
/root/.local/bin/evalscope --version
git rev-parse --short HEAD
git rev-parse --abbrev-ref HEAD
echo "DSPARK_STAGE2_TABLES_READY=$RESULT_ROOT"
while [ ! -f /tmp/dspark-stage2-results-collected.ready ]; do sleep 2; done
