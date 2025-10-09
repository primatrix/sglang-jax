#!/usr/bin/env python3
"""
Simple script to run JIT cache optimization tests.

Usage:
    python test_jit_cache_optimization.py [--model-path MODEL_PATH] [--tp-size TP_SIZE]

Examples:
    python test_jit_cache_optimization.py
    python test_jit_cache_optimization.py --model-path /path/to/model --tp-size 4
"""

import argparse
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="Test JIT cache optimization")
    parser.add_argument(
        "--model-path", default="/models/Qwen-7B", help="Path to the model"
    )
    parser.add_argument("--tp-size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument(
        "--device", default="tpu", choices=["tpu", "gpu", "cpu"], help="Device type"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument(
        "--specific-test", help="Run a specific test method (e.g., test_cache_hit_rate)"
    )

    args = parser.parse_args()

    # Set environment variables
    env = os.environ.copy()
    env["MODEL_PATH"] = args.model_path
    env["TP_SIZE"] = str(args.tp_size)
    env["JAX_PLATFORMS"] = args.device
    env["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={args.tp_size}"

    # Disable precision tracer for faster testing
    env["ENABLE_PRECISION_TRACER"] = "0"

    # Set cache directory for persistent compilation cache
    env["JAX_COMPILATION_CACHE_DIR"] = "/tmp/jax_test_cache"

    print("🧪 Running JIT Cache Optimization Tests")
    print("=" * 50)
    print(f"Model Path: {args.model_path}")
    print(f"TP Size: {args.tp_size}")
    print(f"Device: {args.device}")
    print(f"Verbose: {args.verbose}")
    if args.specific_test:
        print(f"Specific Test: {args.specific_test}")
    print("=" * 50)

    # Construct test command
    test_module = "python.sgl_jax.test.model_executor.test_jit_cache_optimization"

    if args.specific_test:
        test_target = f"{test_module}.TestJITCacheOptimization.{args.specific_test}"
    else:
        test_target = f"{test_module}.TestJITCacheOptimization"

    cmd = [sys.executable, "-m", "unittest", test_target]

    if args.verbose:
        cmd.append("-v")

    print(f"Running command: {' '.join(cmd)}")
    print()

    try:
        # Run the test
        result = subprocess.run(cmd, env=env, cwd=os.getcwd())

        if result.returncode == 0:
            print("\n✅ All tests passed! JIT cache optimization is working correctly.")
        else:
            print(f"\n❌ Tests failed with return code: {result.returncode}")

        return result.returncode

    except KeyboardInterrupt:
        print("\n⚠️ Tests interrupted by user")
        return 1
    except Exception as e:
        print(f"\n💥 Error running tests: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
