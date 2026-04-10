#!/usr/bin/env python3
"""Launcher: EPMoE vs FusedEPMoE under hotspot imbalance (half experts 2x load).

Hotspot config: hotspot_count=128 (half of 256 experts are hot),
hotspot_ratio=0.6667 (2/3 traffic goes to hot half → each hot expert gets 2x).
"""
import os
import sys

sys.path.insert(0, "/tmp/sglang-jax/python")
sys.path.insert(0, "/tmp/sglang-jax")
os.chdir("/tmp/sglang-jax")

import jax  # noqa: E402

jax.distributed.initialize()
proc = jax.process_index()
print(f"[Process {proc}] ready, {jax.device_count()} devices", flush=True)

sys.argv = [
    "bench_moe_compare",
    "--num-tokens",
    "32",
    "64",
    "128",
    "256",
    "512",
    "1024",
    "2048",
    "4096",
    "8192",
    "--num-experts",
    "256",
    "--top-k",
    "8",
    "--hidden-size",
    "4096",
    "--intermediate-size",
    "2048",
    "--activation",
    "silu",
    "--weight-dtype",
    "float8_e4m3fn",
    "--renormalize-topk-logits",
    "--num-expert-group",
    "0",
    "--topk-group",
    "0",
    "--imbalance-mode",
    "hotspot",
    "--hotspot-ratio",
    "0.6667",
    "--hotspot-count",
    "128",
    "--non-hotspot-alpha",
    "10000.0",
    "--iters",
    "3",
    "--warmup-iters",
    "1",
]

import runpy  # noqa: E402

runpy.run_path("/tmp/sglang-jax/benchmark/moe/bench_moe_compare.py", run_name="__main__")
