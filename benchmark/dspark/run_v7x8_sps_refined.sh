#!/usr/bin/env bash
set -euo pipefail

cd /tmp/sglang-jax
export SGLANG_RECORD_STEP_TIME=1
export HF_HOME="${HF_HOME:-/tmp/tpu_logs/huggingface}"
RESULT_ROOT=/tmp/dspark_stage2_sps_refined
mkdir -p "$RESULT_ROOT"

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
  --precompile-bs-paddings 2 4 8 16 32 64 128 256 512 \
  --decode-log-interval 1 \
  --trust-remote-code \
  --disable-radix-cache \
  --grammar-backend none \
  --host 0.0.0.0 \
  --port 30000 >"$RESULT_ROOT/server.log" 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT

for _ in $(seq 1 900); do
  status=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:30000/health || true)
  if [ "$status" = "200" ]; then break; fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    tail -300 "$RESULT_ROOT/server.log"
    exit 1
  fi
  sleep 2
done
test "${status:-000}" = "200"

python/.venv/bin/python benchmark/dspark/collect_sps.py \
  --python python/.venv/bin/python \
  --base-url http://127.0.0.1:30000 \
  --model Qwen/Qwen3-8B \
  --concurrency 2 4 8 16 32 64 128 256 512 \
  --input-len 256 \
  --output-len 512 \
  --prompts-multiplier 2 \
  --output-dir "$RESULT_ROOT/sps"

echo "DSPARK_REFINED_SPS_READY=$RESULT_ROOT"
