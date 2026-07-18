#!/usr/bin/env bash

set -euxo pipefail

ROOT="${GLM52_DSA_ROOT:-/tmp/glm52-dsa/sglang-jax}"
PYBIN="${GLM52_DSA_PYBIN:-/opt/venv/bin/python3}"
OUTPUT="${GLM52_DSA_BENCH_OUTPUT:-/tmp/glm52-dsa-v7x8-bench.json}"

cd "$ROOT"
export PYTHONPATH="$ROOT/python${PYTHONPATH:+:$PYTHONPATH}"

"$PYBIN" benchmark/kernels/mla/bench_dsa_decode_mla.py \
  --batch-size 1 \
  --context-length 160000 \
  --top-k 2048 \
  --num-heads 8 \
  --latent-dim 512 \
  --rope-dim 64 \
  --page-size 128 \
  --slot-order unsorted \
  --variant sparse \
  --warmup-iters 2 \
  --iters 5 \
  --output "$OUTPUT"

cat "$OUTPUT"
