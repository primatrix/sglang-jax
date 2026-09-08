#!/usr/bin/env bash
# Run inside the pinned TPU image; Falcon provides ARTIFACT_LOCAL_DIR.
set -euo pipefail
ulimit -c 0
export TMPDIR=/tmp/tpu_logs/tmp PIP_CACHE_DIR=/tmp/tpu_logs/pip-cache UV_CACHE_DIR=/tmp/tpu_logs/uv-cache
export HF_HOME=/tmp/tpu_logs/huggingface JAX_COMPILATION_CACHE_DIR=/tmp/tpu_logs/qwen3vl-jit
ARTIFACT_DEST="${ARTIFACT_LOCAL_DIR:?}"
OUT=/tmp/tpu_logs/qwen3vl-output
export ARTIFACT_LOCAL_DIR="$OUT"
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$UV_CACHE_DIR" "$HF_HOME" "$JAX_COMPILATION_CACHE_DIR" "$OUT" "$ARTIFACT_DEST"
archive_results() {
  RUN_EXIT=$?
  trap - EXIT
  mkdir -p "$ARTIFACT_DEST"
  cp -a "$OUT/." "$ARTIFACT_DEST/" || { echo "Artifact archive failed"; exit 1; }
  exit "$RUN_EXIT"
}
trap archive_results EXIT
cd "$(dirname "$0")/../.."
git rev-parse HEAD > "$OUT/source-sha.txt"

python -m pip install --no-deps -e ./python
if ! python -c 'import torch, torchvision'; then
  python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi
if ! python -c 'from torchcodec.decoders import decode_image'; then
  python -m pip install 'torchcodec>=0.16.0' --index-url https://download.pytorch.org/whl/cpu
  if ! python -c 'from torchcodec.decoders import decode_image'; then
    apt-get update -qq
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends ffmpeg
  fi
fi
JAX_PLATFORMS=cpu python - <<'IMAGE_PREFLIGHT'
import io,base64
from PIL import Image
from torchcodec.decoders import decode_image
from sgl_jax.srt.multimodal.processors.qwen_vl import QwenVLProcessor
b=io.BytesIO();Image.new('RGB',(512,512)).save(b,format='JPEG')
image=QwenVLProcessor.load_image('data:image/jpeg;base64,'+base64.b64encode(b.getvalue()).decode())
print('IMAGE_DECODE_PREFLIGHT_OK',type(image),getattr(image,'shape',None),flush=True)
IMAGE_PREFLIGHT
python -m pip freeze > "$OUT/packages.txt"
python - <<'DEVICES'
import jax,json,os,importlib.metadata as md
from pathlib import Path
devices=jax.devices()
assert len(devices)==8 and all(d.platform=='tpu' for d in devices),devices
info={'devices':[str(d) for d in devices],'packages':{p:md.version(p) for p in ['jax','jaxlib','libtpu','flax','torch','torchcodec','transformers']}}
Path(os.environ['ARTIFACT_LOCAL_DIR'],'environment.json').write_text(json.dumps(info,indent=2))
print(json.dumps(info),flush=True)
DEVICES
python -u scripts/disaggregation/bench_non_epd.py --launch-server --output-dir "$OUT/baseline" "$@"
