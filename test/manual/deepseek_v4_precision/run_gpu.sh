#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0
export PYTHONUNBUFFERED=1
# GKE mounts the driver here; the image ldconfig cache may point to absent compat libs.
if [ -f /usr/local/nvidia/lib64/libcuda.so.1 ]; then
  export TRITON_LIBCUDA_PATH=/usr/local/nvidia/lib64
fi
TASK_DIR=/workspace/sglang-jax/test/manual/deepseek_v4_precision
ROOT="${ARTIFACT_LOCAL_DIR:?}"
mkdir -p "$ROOT/rank-0/benchmark" "$ROOT/profiling"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > "$ROOT/profiling/gpu.txt"
printf '%s\n' "$SGLANG_SHA" > "$ROOT/sglang-source.txt"
git -C /workspace/sglang-jax rev-parse HEAD > "$ROOT/jax-source.txt"
status=0
python3 "$TASK_DIR/install_gpu.py" /workspace/sglang > "$ROOT/profiling/setup.log" 2>&1 || status=$?
if [ "$status" -eq 0 ]; then
  python3 -m pip freeze > "$ROOT/profiling/python-packages.txt"
  python3 "$TASK_DIR/${GPU_CAPTURE_SCRIPT:-gpu_capture.py}" --model /models/deepseek-v4 --out /tmp/gpu-reference || status=$?
else
  echo "GPU_SETUP_FAILED $status"
  tail -50 "$ROOT/profiling/setup.log"
fi
if [ -d /tmp/gpu-reference ]; then
  tar -czf /tmp/gpu-reference.tar.gz -C /tmp gpu-reference
  cp /tmp/gpu-reference.tar.gz "$ROOT/rank-0/benchmark/gpu-reference.tar.gz"
fi
printf '%s\n' "$status" > /tmp/module-validation-exit
printf '{"schema_version":1,"workflow":"operator-optimization","operator_family":"deepseek-v4","operator_name":"module-precision-gpu-reference","dimensions":{"tokens":128,"tp_size":1,"ep_size":1}}\n' > "$ROOT/manifest.json"
echo "GPU_CAPTURE_READY status=$status; waiting up to 1800s for Falcon transfer/release"
for ((i=0;i<360;i++)); do
  [ -f /tmp/module-validation-release ] && break
  sleep 5
done
exit "$(cat /tmp/module-validation-exit)"
