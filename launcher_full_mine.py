#!/usr/bin/env python3
"""Full model launcher for debug branch (all 48 layers)."""
import os
import sys

sys.path.insert(0, "/tmp/sglang-jax/python")
sys.path.insert(0, "/tmp/sglang-jax")
os.chdir("/tmp/sglang-jax")
node_rank = int(os.environ.get("JOB_COMPLETION_INDEX", "0"))
print(f"[Node {node_rank}] starting full model server (debug branch, 48 layers)", flush=True)
sys.argv = [
    "launch_server",
    "--model-path",
    "/inference-models/MiMo-V2-Flash",
    "--tp-size",
    "16",
    "--port",
    "30000",
    "--quantization-config-path",
    "/tmp/mimo_v2_flash_quant_config.yaml",
    "--trust-remote-code",
    "--nnodes",
    "4",
    "--node-rank",
    str(node_rank),
    "--dist-init-addr",
    "10.31.172.4:5678",
    "--context-length",
    "4096",
    "--attention-backend",
    "native",
    "--page-size",
    "1",
    "--max-running-requests",
    "32",
    "--disable-precompile",
]
import runpy

runpy.run_path("/tmp/sglang-jax/python/sgl_jax/launch_server.py", run_name="__main__")
