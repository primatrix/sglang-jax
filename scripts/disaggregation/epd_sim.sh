#!/usr/bin/env bash
#
# One-shot EPD CPU simulation: launch the tiers, drive requests, capture a
# profile, render the flame graph + aligned overlap timeline, open the relevant
# timeline, and tear everything down. One command, no second terminal.
#
#   MODEL_PATH=/path/to/qwen2.5-vl ./scripts/disaggregation/epd_sim.sh
#   ./scripts/disaggregation/epd_sim.sh --serve-only  # launch without profiling
#
# MODEL_PATH is OPTIONAL: if unset, a cached VLM (config+processor) is
# auto-discovered from the HuggingFace cache. Weights are never loaded.
#
# Coefficients (env, all optional; defaults give a readable illustrative graph):
#   SIM_ENC_BASE_MS SIM_ENC_MS_PER_TOKEN
#   SIM_PREFILL_BASE_MS SIM_PREFILL_MS_PER_TOKEN
#   SIM_DECODE_BASE_MS SIM_DECODE_MS_PER_SEQ
#   SIM_TRANSFER_SETUP_MS SIM_TRANSFER_MS_PER_MB SIM_NET_RTT_MS
# Topology / workload (env, optional):
#   NUM_ENCODERS TP_SIZE DP_SIZE N_REQUESTS CONCURRENCY MAX_TOKENS PROFILER_DIR PY_TRACER
#   SIM_MAX_TOTAL_TOKENS SIM_CHUNKED_PREFILL_SIZE
#   ENCODER_MAX_BATCH_SIZE ENCODER_BATCH_COALESCE_MS
#   ENCODER_MAX_INFLIGHT_BATCHES ENCODER_TRANSFER_POOL_SIZE
#   DISAGGREGATION_CHANNEL_NUMBER MM_PROCESSOR_WORKERS
#   PREWARM_REQUESTS PREWARM_CONCURRENCY RANDOM_INPUT_LEN IMAGES_PER_REQ IMAGE_SIZE
#
set -euo pipefail

if [[ $# -gt 1 || ( $# -eq 1 && "$1" != "--serve-only" ) ]]; then
  echo "usage: $0 [--serve-only]" >&2
  exit 2
fi

MODEL_PATH="${MODEL_PATH:-}"

NUM_ENCODERS=${NUM_ENCODERS:-1}
TP_SIZE=${TP_SIZE:-4}
DP_SIZE=${DP_SIZE:-2}
if ((TP_SIZE <= 0 || DP_SIZE <= 0 || TP_SIZE % DP_SIZE != 0)); then
  echo "TP_SIZE and DP_SIZE must be positive, with TP_SIZE divisible by DP_SIZE" >&2
  exit 2
fi
DEVICE_COUNT=${DEVICE_COUNT:-$TP_SIZE}
ENCODER_PORT_BASE=${ENCODER_PORT_BASE:-31001}
LANG_PORT=${LANG_PORT:-30000}
PROFILER_DIR=${PROFILER_DIR:-$(mktemp -d /tmp/epd-sim-XXXXXX)}
N_REQUESTS=${N_REQUESTS:-256}
MAX_TOKENS=${MAX_TOKENS:-16}
CONCURRENCY=${CONCURRENCY:-64}
PREWARM_REQUESTS=${PREWARM_REQUESTS:-32}
PREWARM_CONCURRENCY=${PREWARM_CONCURRENCY:-16}
RANDOM_INPUT_LEN=${RANDOM_INPUT_LEN:-1024}
MAX_RUNNING=${MAX_RUNNING:-512}
SIM_MAX_TOTAL_TOKENS=${SIM_MAX_TOTAL_TOKENS:-32768}  # logical only; no physical sim KV buffers
SIM_MAX_PREFILL_TOKENS=${SIM_MAX_PREFILL_TOKENS:-16384}
SIM_CHUNKED_PREFILL_SIZE=${SIM_CHUNKED_PREFILL_SIZE:-4096}
ENCODER_MAX_BATCH_SIZE=${ENCODER_MAX_BATCH_SIZE:-4}
ENCODER_BATCH_COALESCE_MS=${ENCODER_BATCH_COALESCE_MS:-2}
ENCODER_MAX_INFLIGHT_BATCHES=${ENCODER_MAX_INFLIGHT_BATCHES:-1}
ENCODER_TRANSFER_POOL_SIZE=${ENCODER_TRANSFER_POOL_SIZE:-32}
DISAGGREGATION_CHANNEL_NUMBER=${DISAGGREGATION_CHANNEL_NUMBER:-4}
MM_PROCESSOR_WORKERS=${MM_PROCESSOR_WORKERS:-4}
IMAGES_PER_REQ=${IMAGES_PER_REQ:-1}
IMAGE_SIZE=${IMAGE_SIZE:-512}         # generated benchmark image, in square pixels
PY_TRACER=${PY_TRACER:-0}   # 0 = clean stage view (good for flame graph + timeline)

SIM_ENC_BASE_MS=${SIM_ENC_BASE_MS:-3}
SIM_ENC_MS_PER_TOKEN=${SIM_ENC_MS_PER_TOKEN:-0.014}
SIM_PREFILL_BASE_MS=${SIM_PREFILL_BASE_MS:-3}
SIM_PREFILL_MS_PER_TOKEN=${SIM_PREFILL_MS_PER_TOKEN:-0.08}
SIM_DECODE_BASE_MS=${SIM_DECODE_BASE_MS:-14}
SIM_DECODE_MS_PER_SEQ=${SIM_DECODE_MS_PER_SEQ:-0.05}
SIM_TRANSFER_SETUP_MS=${SIM_TRANSFER_SETUP_MS:-0}
SIM_TRANSFER_MS_PER_MB=${SIM_TRANSFER_MS_PER_MB:-0.12}
SIM_NET_RTT_MS=${SIM_NET_RTT_MS:-0.3}

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
PY=python
[ -x "${ROOT}/.venv/bin/python" ] && PY="${ROOT}/.venv/bin/python"

# Resolve a model directory (config + tokenizer + processor; weights unused).
# Prefer $MODEL_PATH, else auto-discover a cached VLM from the HF cache.
if [ -z "${MODEL_PATH}" ]; then
  MODEL_PATH=$("${PY}" - <<'PY'
import glob, json, os
best, best_score = None, (-1, 1 << 30)
for cfg in glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--*/snapshots/*/config.json")):
    try:
        d = json.load(open(cfg))
    except Exception:
        continue
    mt = str(d.get("model_type", "")).lower()
    arch = ",".join(d.get("architectures") or [])
    is_vlm = ("vision_config" in d) or ("image_token_id" in d) or ("vl" in mt) or ("VL" in arch)
    if not is_vlm:
        continue
    layers = d.get("num_hidden_layers") or (d.get("text_config") or {}).get("num_hidden_layers") or 1 << 20
    score = (2 if "vl" in mt else 1, layers)  # prefer *vl* model_type, then fewer layers
    if (score[0], -score[1]) > (best_score[0], -best_score[1]):
        best, best_score = os.path.dirname(cfg), score
print(best or "")
PY
)
  if [ -z "${MODEL_PATH}" ]; then
    echo "no cached VLM found in ~/.cache/huggingface/hub; set MODEL_PATH=/path/to/vlm" >&2
    exit 2
  fi
  echo ">> auto-discovered model config: ${MODEL_PATH}"
fi

export PYTHONPATH="${ROOT}/python:${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export NO_PROXY="*" no_proxy="*"  # All simulator control traffic is local.
export JAX_PLATFORMS=cpu
export XLA_FLAGS="--xla_force_host_platform_device_count=${DEVICE_COUNT} ${XLA_FLAGS:-}"
export SGLANG_JAX_PROFILER_DIR="${PROFILER_DIR}"
mkdir -p "${PROFILER_DIR}"
if [ -e "${PROFILER_DIR}/language.log" ]; then
  echo "PROFILER_DIR already contains a run: ${PROFILER_DIR}; choose a new directory" >&2
  exit 2
fi

PIDS=()
cleanup() {
  trap - EXIT INT TERM
  echo ">> stopping servers"
  for pid in "${PIDS[@]}"; do kill -- "-${pid}" 2>/dev/null || true; done
  for _ in $(seq 1 50); do
    local running=0
    for pid in "${PIDS[@]}"; do kill -0 -- "-${pid}" 2>/dev/null && running=1; done
    ((running == 0)) && break
    sleep 0.1
  done
  for pid in "${PIDS[@]}"; do kill -KILL -- "-${pid}" 2>/dev/null || true; done
  for pid in "${PIDS[@]}"; do wait "${pid}" 2>/dev/null || true; done
}
trap cleanup EXIT INT TERM

start_server() {
  local log=$1
  shift
  "${PY}" -c 'import os, sys; os.setsid(); os.execv(sys.executable, [sys.executable, "-m", "sgl_jax.launch_server", *sys.argv[1:]])' "$@" > "${log}" 2>&1 &
  PIDS+=($!)
}

wait_for_health() {
  local url=$1 pid=$2
  for _ in $(seq 1 120); do
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "server exited; see ${PROFILER_DIR}/*.log" >&2
      exit 1
    fi
    curl -fsS "${url}" >/dev/null 2>&1 && return 0
    sleep 2
  done
  echo "timed out waiting for ${url}" >&2
  exit 1
}

sim_args=(
  --simulate-compute
  --simulate-compute-encoder-base-ms "${SIM_ENC_BASE_MS}"
  --simulate-compute-encoder-ms-per-token "${SIM_ENC_MS_PER_TOKEN}"
  --simulate-compute-prefill-base-ms "${SIM_PREFILL_BASE_MS}"
  --simulate-compute-prefill-ms-per-token "${SIM_PREFILL_MS_PER_TOKEN}"
  --simulate-compute-decode-base-ms "${SIM_DECODE_BASE_MS}"
  --simulate-compute-decode-ms-per-seq "${SIM_DECODE_MS_PER_SEQ}"
  --simulate-transfer-setup-ms "${SIM_TRANSFER_SETUP_MS}"
  --simulate-transfer-ms-per-mb "${SIM_TRANSFER_MS_PER_MB}"
  --simulate-network-rtt-ms "${SIM_NET_RTT_MS}"
)
common_args=(
  --model-path "${MODEL_PATH}" --tp-size "${TP_SIZE}" --dp-size "${DP_SIZE}"
  --device cpu --load-format dummy --dtype bfloat16 --attention-backend native
  --trust-remote-code --disaggregation-host-ip 127.0.0.1
  --disaggregation-channel-number "${DISAGGREGATION_CHANNEL_NUMBER}"
  --encoder-transfer-pool-size "${ENCODER_TRANSFER_POOL_SIZE}"
  --enable-request-time-stats-logging
)

ENCODER_URLS=()
for ((i = 0; i < NUM_ENCODERS; i++)); do
  port=$((ENCODER_PORT_BASE + i))
  echo ">> starting encoder ${i} on :${port}"
  start_server "${PROFILER_DIR}/encoder_${i}.log" "${common_args[@]}" "${sim_args[@]}" \
    --encoder-only \
    --encoder-max-batch-size "${ENCODER_MAX_BATCH_SIZE}" \
    --encoder-batch-coalesce-ms "${ENCODER_BATCH_COALESCE_MS}" \
    --encoder-max-inflight-batches "${ENCODER_MAX_INFLIGHT_BATCHES}" \
    --vision-encoder-parallel dp \
    --mm-processor-worker-num "${MM_PROCESSOR_WORKERS}" \
    --host 127.0.0.1 --port "${port}"
  ENCODER_URLS+=("http://127.0.0.1:${port}")
done

echo ">> starting language server on :${LANG_PORT}"
start_server "${PROFILER_DIR}/language.log" "${common_args[@]}" "${sim_args[@]}" \
  --language-only --encoder-urls "${ENCODER_URLS[@]}" \
  --disable-radix-cache \
  --max-total-tokens "${SIM_MAX_TOTAL_TOKENS}" \
  --max-prefill-tokens "${SIM_MAX_PREFILL_TOKENS}" \
  --chunked-prefill-size "${SIM_CHUNKED_PREFILL_SIZE}" \
  --context-length 2048 --max-seq-len 2048 \
  --max-running-requests "${MAX_RUNNING}" --mem-fraction-static 0.1 \
  --dp-schedule-policy min_running_queue \
  --vision-encoder-parallel dp \
  --mm-processor-worker-num "${MM_PROCESSOR_WORKERS}" \
  --host 127.0.0.1 --port "${LANG_PORT}"

echo ">> waiting for health (first run compiles; ~30-60s)"
for i in "${!ENCODER_URLS[@]}"; do wait_for_health "${ENCODER_URLS[$i]}/health" "${PIDS[$i]}"; done
wait_for_health "http://127.0.0.1:${LANG_PORT}/health" "${PIDS[$NUM_ENCODERS]}"
echo ">> all healthy; output: ${PROFILER_DIR}"
if [[ "${1:-}" == "--serve-only" ]]; then
  echo "language: http://127.0.0.1:${LANG_PORT}; encoders: ${ENCODER_URLS[*]}"
  echo "Profile with scripts/disaggregation/profile_epd.py --profiler-dir ${PROFILER_DIR}"
  wait
  exit 0
fi

enc_flags=()
for url in "${ENCODER_URLS[@]}"; do enc_flags+=(--encoder-url "${url}"); done

# Footgun guard: python_tracer_level=1 records every Python call; the trace->JSON
# converter caps at ~1M events and silently truncates in Perfetto. Keep level-1
# captures to a tiny slice (1 request, few tokens, no concurrency).
if [ "${PY_TRACER}" -ge 1 ] && { [ "$((N_REQUESTS * MAX_TOKENS))" -gt 40 ] || [ "${CONCURRENCY}" -gt 1 ]; }; then
  echo "WARNING: PY_TRACER=1 with this workload will likely TRUNCATE the trace at ~1M"
  echo "  events (Perfetto shows it cut). For un-truncated function detail use e.g."
  echo "  PY_TRACER=1 CONCURRENCY=1 N_REQUESTS=1 MAX_TOKENS=8 ...; for concurrency/large"
  echo "  workloads use PY_TRACER=0 (stage annotations, never near the cap)."
fi

echo ">> profiling ${N_REQUESTS} requests (concurrency ${CONCURRENCY}, ${IMAGES_PER_REQ} img/req @ ${IMAGE_SIZE}px)"
"${PY}" "${SCRIPT_DIR}/profile_epd.py" \
  --bench-serving --model-path "${MODEL_PATH}" \
  --lang-url "http://127.0.0.1:${LANG_PORT}" "${enc_flags[@]}" \
  --images-per-request "${IMAGES_PER_REQ}" \
  --n-requests "${N_REQUESTS}" --max-tokens "${MAX_TOKENS}" \
  --concurrency "${CONCURRENCY}" \
  --prewarm-requests "${PREWARM_REQUESTS}" \
  --prewarm-concurrency "${PREWARM_CONCURRENCY}" \
  --random-input-len "${RANDOM_INPUT_LEN}" \
  --image-resolution "${IMAGE_SIZE}x${IMAGE_SIZE}" \
  --image-format jpeg --image-content random --seed 0 \
  --python-tracer-level "${PY_TRACER}" --profiler-dir "${PROFILER_DIR}"

echo ">> rendering flame graph"
"${PY}" "${SCRIPT_DIR}/trace_to_flamegraph.py" --profiler-dir "${PROFILER_DIR}"
echo ">> rendering aligned overlap timeline"
"${PY}" "${SCRIPT_DIR}/trace_to_overlap_html.py" --profiler-dir "${PROFILER_DIR}"
# Level-1 traces carry the stdlib/framework firehose; auto-slim to project
# functions so the .slim.trace.json.gz is a readable Perfetto middle ground.
if [ "${PY_TRACER}" -ge 1 ]; then
  echo ">> slimming level-1 trace (project functions only)"
  "${PY}" "${SCRIPT_DIR}/trace_slim.py" --profiler-dir "${PROFILER_DIR}" || true
fi

echo "Artifacts: ${PROFILER_DIR}"
echo "  epd_flamegraph.svg, epd_overlap.html, epd_overlap.trace.json"
echo "  aligned-summary.json, overlap-summary.json, per-tier JAX profiles"
if [[ "${OPEN_BROWSER:-1}" == "1" ]]; then
  if command -v open >/dev/null 2>&1; then open "${PROFILER_DIR}/epd_overlap.html" || true
  elif command -v xdg-open >/dev/null 2>&1; then xdg-open "${PROFILER_DIR}/epd_overlap.html" >/dev/null 2>&1 || true
  fi
fi
