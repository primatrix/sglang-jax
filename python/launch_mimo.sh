#!/bin/bash
set -ex

export RANK=0
export JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache
export LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS} \
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

uv run python -u -m sgl_jax.launch_server \
  --model-path /models/MiMo-V2-Flash-Base \
  --trust-remote-code \
  --tp-size 8 --ep-size 8 \
  --moe-backend fused \
  --host 0.0.0.0 --port 30271 \
  --page-size 256 \
  --context-length 262144 \
  --disable-radix-cache \
  --chunked-prefill-size 4096 \
  --dtype bfloat16 \
  --mem-fraction-static 0.95 \
  --swa-full-tokens-ratio 0.2 \
  --skip-server-warmup \
  --log-level info \
  --max-running-requests 128
