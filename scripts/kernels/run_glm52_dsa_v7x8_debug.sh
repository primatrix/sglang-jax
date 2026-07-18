#!/usr/bin/env bash

set -euxo pipefail

ROOT="${GLM52_DSA_ROOT:-/tmp/glm52-dsa/sglang-jax}"
PYBIN="${GLM52_DSA_PYBIN:-/opt/venv/bin/python3}"

cd "$ROOT"
export PYTHONPATH="$ROOT/python${PYTHONPATH:+:$PYTHONPATH}"

"$PYBIN" - <<'PY'
import importlib.metadata as md

import jax
import sgl_jax

print(f"jax={jax.__version__}")
print(f"jaxlib={md.version('jaxlib')}")
print(f"libtpu={md.version('libtpu')}")
print(f"backend={jax.default_backend()}")
print(f"local_device_count={jax.local_device_count()}")
print(f"sgl_jax={sgl_jax.__file__}")
assert jax.default_backend() == "tpu"
assert jax.local_device_count() == 8
PY

if ! "$PYBIN" -c 'import pytest'; then
  "$PYBIN" -m pip install --no-cache-dir pytest
fi

if [[ "${1:-targeted}" == "full" ]]; then
  "$PYBIN" -m pytest -q \
    python/sgl_jax/test/kernels/test_dsa_decode_mla.py \
    --maxfail=1
else
  "$PYBIN" -m pytest -q \
    python/sgl_jax/test/kernels/test_dsa_decode_mla.py \
    -k 'tpu_non_interpret' \
    --maxfail=1
fi
