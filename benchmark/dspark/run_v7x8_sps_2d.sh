#!/usr/bin/env bash
set -euo pipefail

cd /tmp/sglang-jax
export SGLANG_RECORD_STEP_TIME=1
export HF_HOME="${HF_HOME:-/tmp/tpu_logs/huggingface}"
RESULT_ROOT="${DSPARK_RESULT_ROOT:-/tmp/dspark_sps_2d}"
mkdir -p "$RESULT_ROOT/points"

SERVER_PID=""
cleanup_server() {
  if [ -n "$SERVER_PID" ]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=""
  fi
}
trap cleanup_server EXIT

launch_server() {
  local token_bucket=$1
  local server_log="$RESULT_ROOT/server_m${token_bucket}.log"
  export SGL_JAX_DSPARK_FORCE_TOKEN_BUCKET_PER_DP="$token_bucket"
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
    --max-running-requests 128 \
    --context-length 16384 \
    --precompile-token-paddings 2048 \
    --decode-log-interval 1 \
    --trust-remote-code \
    --disable-radix-cache \
    --grammar-backend none \
    --disable-precompile \
    --enable-dspark-tuned-config \
    --host 0.0.0.0 \
    --port 30000 >"$server_log" 2>&1 &
  SERVER_PID=$!

  local status=000
  for _ in $(seq 1 900); do
    status=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:30000/health || true)
    if [ "$status" = "200" ]; then break; fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      tail -300 "$server_log"
      exit 1
    fi
    sleep 2
  done
  test "$status" = "200"
}

collect_point() {
  local requests=$1
  local token_bucket=$2
  python/.venv/bin/python benchmark/dspark/collect_sps_2d.py \
    --python python/.venv/bin/python \
    --global-concurrency "$((requests * 2))" \
    --request-bucket-per-dp "$requests" \
    --verify-token-bucket-per-dp "$token_bucket" \
    --output "$RESULT_ROOT/points/r${requests}_m${token_bucket}.json"
}

for token_bucket in ${DSPARK_TOKEN_BUCKETS:-32 64 128 256 512}; do
  launch_server "$token_bucket"
  if [ "$token_bucket" -le 256 ]; then
    collect_point 32 "$token_bucket"
  fi
  if [ "$token_bucket" -ge 64 ]; then
    collect_point 64 "$token_bucket"
  fi
  cleanup_server
done

python/.venv/bin/python - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ.get("DSPARK_RESULT_ROOT", "/tmp/dspark_sps_2d"))
points = [json.loads(path.read_text()) for path in sorted((root / "points").glob("r*_m*.json"))]
table = {
    "schema_version": 2,
    "kind": "dspark_ragged_sps_t_r_m",
    "gamma": 7,
    "verify_width": 8,
    "tp_size": 8,
    "dp_size": 2,
    "input_len": 256,
    "output_len": 256,
    "points": points,
}
(root / "sps_2d_table.json").write_text(json.dumps(table, indent=2) + "\n")
print(json.dumps(table, indent=2))
PY

touch "$RESULT_ROOT/READY"
echo "DSPARK_SPS_2D_READY=$RESULT_ROOT"
while [ ! -f "$RESULT_ROOT/COLLECTED" ]; do sleep 2; done
