#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0
unset PIP_CONSTRAINT
export PYTHONUNBUFFERED=1
export TRITON_LIBCUDA_PATH=/usr/local/nvidia/lib64
TASK_DIR=/workspace/sglang-jax/test/manual/deepseek_v4_precision
ROOT="${ARTIFACT_LOCAL_DIR:?}"
mkdir -p "$ROOT/rank-0/benchmark" "$ROOT/profiling" /tmp/gpu-reference
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > "$ROOT/profiling/gpu.txt"
git -C /workspace/sglang-jax rev-parse HEAD > "$ROOT/jax-source.txt"
git -C /workspace/sglang rev-parse HEAD > "$ROOT/sglang-source.txt"
status=0
python3 "$TASK_DIR/install_gpu.py" /workspace/sglang > "$ROOT/profiling/setup.log" 2>&1 || status=$?
if [ "$status" -eq 0 ]; then
  python3 -m pip freeze > "$ROOT/profiling/python-packages.txt"
  for component in attention layer; do
    for layer in 0 2 3; do
      case_status=0
      python3 "$TASK_DIR/gpu_attention_capture.py" --component "$component" --layer "$layer" \
        --model /models/deepseek-v4 --out "/tmp/gpu-reference/$component-l$layer" \
        > "$ROOT/profiling/$component-l$layer.log" 2>&1 || case_status=$?
      printf '%s\n' "$case_status" > "/tmp/gpu-reference/$component-l$layer.exit"
      echo "GPU_CAPTURE_CASE_READY $component layer=$layer status=$case_status"
      if [ "$case_status" -ne 0 ]; then
        status="$case_status"
        tail -40 "$ROOT/profiling/$component-l$layer.log"
      fi
    done
  done
else
  tail -50 "$ROOT/profiling/setup.log"
fi
tar -czf /tmp/gpu-reference.tar.gz -C /tmp gpu-reference
cp /tmp/gpu-reference.tar.gz "$ROOT/rank-0/benchmark/"
sha256sum /tmp/gpu-reference.tar.gz > "$ROOT/rank-0/benchmark/gpu-reference.sha256"
printf '%s\n' "$status" > /tmp/module-validation-exit
echo "GPU_CAPTURE_READY status=$status"
for ((i=0;i<720;i++)); do
  [ -f /tmp/module-validation-release ] && break
  sleep 5
done
exit "$(cat /tmp/module-validation-exit)"
