#!/usr/bin/env bash
# Usage: bash run_flash.sh <node_rank> <pod0_ip> <gh_token>
# Example: bash run_flash.sh 0 10.0.1.5 ghp_xxxx
set -euxo pipefail

NODE_RANK=${1:?usage: run_flash.sh <node_rank> <pod0_ip> <gh_token>}
POD0_IP=${2:?}
GH_TOKEN=${3:?}

export UV_CACHE_DIR="/tmp/tpu_logs/uv_cache"
export UV_PYTHON_INSTALL_DIR="/tmp/tpu_logs/python"
export TMPDIR="/tmp/tpu_logs"
export PIP_CACHE_DIR="/tmp/tpu_logs/pip_cache"
export JAX_COMPILATION_CACHE_DIR="/tmp/tpu_logs/jit_cache"
export LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS:-} \
  --xla_tpu_use_tc_device_shape_on_sc=true \
  --xla_sc_enable_instruction_fusion=false \
  --xla_sc_disjoint_spmem=false \
  --xla_sc_disable_megacore_partitioning=true \
  --xla_tpu_enable_async_collective_fusion_fuse_all_gather=false \
  --xla_tpu_enable_async_collective_fusion_fuse_all_reduce=false \
  --xla_tpu_enable_async_collective_fusion_fuse_reduce_scatter=false \
  --xla_tpu_enable_sparse_core_collective_offload_all_gather=true \
  --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true \
  --xla_tpu_enable_sparse_core_collective_offload_all_reduce=true \
  --xla_tpu_enable_all_gather_offload_tracing=true \
  --xla_tpu_enable_reduce_scatter_offload_tracing=true \
  --xla_tpu_enable_all_reduce_offload_tracing=true"
mkdir -p /tmp/tpu_logs

REPO_DIR=/tmp/tpu_logs/sglang-jax
BRANCH=feat/disable-swa-pool

if [ ! -d "$REPO_DIR/.git" ]; then
  git clone --filter=blob:none "https://${GH_TOKEN}@github.com/primatrix/sglang-jax.git" "$REPO_DIR"
fi
cd "$REPO_DIR"
git fetch origin "$BRANCH"
git checkout "$BRANCH" 2>/dev/null || git checkout -b "$BRANCH" "origin/$BRANCH"
git reset --hard "origin/$BRANCH"

pip install -q uv
cd python
uv venv --clear /tmp/tpu_logs/venv
source /tmp/tpu_logs/venv/bin/activate
uv pip install -e ".[tpu]"
uv pip install evalscope==0.17.1

cd "$REPO_DIR"
python -u -m sgl_jax.launch_server \
  --model-path /models/MiMo-V2-Flash \
  --trust-remote-code \
  --tp-size 8 --ep-size 8 \
  --moe-backend fused \
  --host 0.0.0.0 --port 30271 \
  --page-size 256 \
  --context-length 262144 \
  --disable-radix-cache \
  --chunked-prefill-size 2048 \
  --dtype bfloat16 \
  --mem-fraction-static 0.95 \
  --swa-full-tokens-ratio 0.2 \
  --disable-hybrid-swa-memory \
  --skip-server-warmup \
  --log-level info \
  --max-running-requests 128 \
  --nnodes 4 --node-rank "$NODE_RANK" \
  --dist-init-addr "${POD0_IP}:5000"
