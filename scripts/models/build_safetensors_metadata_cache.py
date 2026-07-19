#!/usr/bin/env python3

import argparse
import glob
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from sgl_jax.srt.utils.weight_utils import (  # noqa: E402
    SAFETENSORS_METADATA_CACHE_BASENAME,
    _scan_safetensors_metadata,
    _write_safetensors_metadata_cache,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a validated sidecar for safetensors header metadata."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output")
    parser.add_argument("--threads", type=int, default=32)
    args = parser.parse_args()

    model_path = os.path.abspath(args.model_path)
    weights_files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
    if not weights_files:
        raise SystemExit(f"no safetensors shards under {model_path}")
    output = args.output or os.path.join(model_path, SAFETENSORS_METADATA_CACHE_BASENAME)

    weight_info = _scan_safetensors_metadata(
        weights_files,
        num_threads=args.threads,
        show_progress=True,
    )
    _write_safetensors_metadata_cache(output, weights_files, weight_info)
    print(
        f"wrote {output}: shards={len(weights_files)} tensors={len(weight_info)}"
    )


if __name__ == "__main__":
    main()
