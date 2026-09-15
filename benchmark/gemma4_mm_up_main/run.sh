set -eu
export TMPDIR="${TMPDIR:-/tmp/tpu_logs/tmp}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/tmp/tpu_logs/pip-cache}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/tpu_logs/uv-cache}"
export HF_HOME=/tmp/tpu_logs/huggingface
export HF_HUB_CACHE=/tmp/tpu_logs/huggingface/hub
export TRANSFORMERS_CACHE=/tmp/tpu_logs/huggingface/hub
export XDG_CACHE_HOME=/tmp/tpu_logs/cache
export JAX_COMPILATION_CACHE_DIR=/tmp/tpu_logs/jax-cache-gemma4-up-main
export PYTHONUNBUFFERED=1
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost
export SGLANG_JAX_PROFILER_DIR="${ARTIFACT_LOCAL_DIR:-/tmp/falcon-artifacts}/xprof"
export CANONICAL_MODEL=google/gemma-4-31B-it

ROOT="${ARTIFACT_LOCAL_DIR:-/tmp/falcon-artifacts}"
OUT="$ROOT/rank-0"
mkdir -p "$OUT/benchmark" "$OUT/compiler/llo" "$OUT/variants" \
  "$ROOT/profiling" "$TMPDIR" "$PIP_CACHE_DIR" "$UV_CACHE_DIR" \
  "$HF_HUB_CACHE" "$XDG_CACHE_HOME" "$JAX_COMPILATION_CACHE_DIR"
ulimit -c 0

# Run from the pinned checkout prepared by Falcon's command.
python -m pip show jax jaxlib libtpu flax transformers > "$ROOT/packages.txt"
git rev-parse HEAD > "$ROOT/source-commit.txt"
MODEL_PATH=""
for candidate in \
  /models/models--google--gemma-4-31B-it/snapshots/* \
  /models/hub/models--google--gemma-4-31B-it/snapshots/* \
  /models/gemma-4-31B-it \
  /models/google/gemma-4-31B-it; do
  if test -d "$candidate"; then MODEL_PATH="$candidate"; break; fi
done
if test -z "$MODEL_PATH"; then MODEL_PATH="$CANONICAL_MODEL"; fi
export MODEL_PATH
python -m pip show jax jaxlib libtpu transformers torch > "$ROOT/profiling/python-packages.txt"

SERVER_PID=""
cleanup_server() {
  if test -n "$SERVER_PID"; then
    # The server spawns scheduler/detokenizer workers. Terminate the whole
    # session so a later variant cannot share TPU resources with orphans.
    kill -TERM -- "-$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    for i in $(seq 1 50); do
      if ! kill -0 -- "-$SERVER_PID" 2>/dev/null; then break; fi
      sleep 0.1
    done
    kill -KILL -- "-$SERVER_PID" 2>/dev/null || true
    SERVER_PID=""
  fi
}
trap cleanup_server EXIT

run_variant() {
  VARIANT="$1"
  VARIANT_DIR="$OUT/variants/$VARIANT"
  mkdir -p "$VARIANT_DIR"
  SERVER_LOG="$VARIANT_DIR/server.log"

  setsid python -u -m sgl_jax.launch_server \
    --model-path "$MODEL_PATH" \
    --trust-remote-code --skip-server-warmup \
    --device tpu --tp-size 8 --dp-size 4 --attention-backend fa \
    --dtype bfloat16 --kv-cache-dtype bf16 \
    --context-length 2048 --max-seq-len 2048 \
    --max-running-requests 1024 \
    --max-prefill-tokens 16384 --chunked-prefill-size 4096 \
    --mem-fraction-static 0.9 --page-size 128 \
    --disable-radix-cache --vision-encoder-parallel dp \
    --mm-processor-worker-num 2 \
    --random-seed 0 --download-dir /tmp/tpu_logs/huggingface/hub \
    --host 0.0.0.0 --port 30000 \
    > "$SERVER_LOG" 2>&1 &
  SERVER_PID=$!

  READY=0
  for i in $(seq 1 720); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      tail -n 500 "$SERVER_LOG"
      exit 1
    fi
    if curl -sf http://localhost:30000/get_server_info \
      > "$VARIANT_DIR/server-info.json"; then
      READY=1
      break
    fi
    sleep 10
  done
  if test "$READY" != 1; then
    tail -n 500 "$SERVER_LOG"
    exit 1
  fi

  if python -m sgl_jax.bench_serving \
    --backend sglang-oai-chat \
    --host 127.0.0.1 --port 30000 \
    --model "$MODEL_PATH" --tokenizer "$MODEL_PATH" \
    --dataset-name image --num-prompts 1000 \
    --random-input-len 1024 --random-output-len 500 \
    --image-count 1 --image-resolution 512x512 \
    --image-format jpeg --image-content random \
    --random-range-ratio 1.0 --request-rate inf \
    --seed 0 --warmup-requests 1 --flush-cache --output-details \
    --profile --profile-by-stage --profile-stages prefill decode \
    --profile-num-steps 5 \
    --output-file "$VARIANT_DIR/benchmark.jsonl" \
    > "$VARIANT_DIR/client.log" 2>&1; then
    cat "$VARIANT_DIR/client.log"
  else
    status=$?
    cat "$VARIANT_DIR/client.log"
    tail -n 500 "$SERVER_LOG"
    exit "$status"
  fi

  python3 - "$ROOT/xprof" <<'PROFILE_PY'
import json, time, urllib.request, sys
from pathlib import Path
for _ in range(180):
    with urllib.request.urlopen("http://127.0.0.1:30000/profile_status") as response:
        state = json.load(response)
    if state["status"] == "idle":
        break
    time.sleep(2)
else:
    raise RuntimeError("profile flush timed out")
assert list(Path(sys.argv[1]).rglob("*.xplane.pb")), "missing XPlane capture"
PROFILE_PY
  cleanup_server
  python3 - "$VARIANT" "$VARIANT_DIR" \
    "$OUT/benchmark/metrics.jsonl" <<'PY'
import json
import re
import sys
from pathlib import Path

variant, variant_dir, output_path = sys.argv[1:]
variant_dir = Path(variant_dir)
rows = [json.loads(line) for line in (variant_dir / "benchmark.jsonl").read_text().splitlines()]
result = rows[-1]
if result["completed"] != 1000 or any(result.get("errors", [])):
    raise SystemExit(f"{variant}: expected 1000 completions, got {result['completed']}")
server_log = (variant_dir / "server.log").read_text(errors="replace")
row = {
    "variant": variant,
    "tokens_per_sec": result["output_throughput"],
    "request_throughput_req_s": result["request_throughput"],
    "total_token_throughput_tok_s": result["total_throughput"],
    "median_ttft_ms": result["median_ttft_ms"],
    "p99_ttft_ms": result["p99_ttft_ms"],
    "mean_tpot_ms": result["mean_tpot_ms"],
    "median_tpot_ms": result["median_tpot_ms"],
    "p99_tpot_ms": result["p99_tpot_ms"],
    "median_e2e_latency_ms": result["median_e2e_latency_ms"],
    "p99_e2e_latency_ms": result["p99_e2e_latency_ms"],
    "attention_backend": "fa (upstream RPA v3)",
    "decode_bs_buckets": (re.findall(
        r"\[DECODE\] Begin to precompile bs_paddings=(\[[^\n]+\])", server_log
    ) or [None])[-1],
    "dtype": "bfloat16",
    "kv_cache_dtype": "bfloat16",
    "input_len": 1024,
    "image_count": 1,
    "image_resolution": "512x512",
    "total_input_text_tokens": result.get("total_input_text_tokens"),
    "total_input_vision_tokens": result.get("total_input_vision_tokens"),
    "output_len": 500,
    "prompts": 1000,
}
with open(output_path, "a", encoding="utf-8") as handle:
    handle.write(json.dumps(row, sort_keys=True) + "\n")
print("GEMMA4_MM_RESULT=" + json.dumps(row, sort_keys=True), flush=True)
PY
}

run_variant gemma4-up-main-default-attention

test "$(wc -l < "$OUT/benchmark/metrics.jsonl")" -eq 1
