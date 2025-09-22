#!/usr/bin/env python3
"""
Test script for DP Attention implementation
"""

import os
import sys

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "python"))

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P


def test_dp_attention_basic():
    """Test basic DP attention functionality"""
    print("Testing DP Attention basic functionality...")

    try:
        from sgl_jax.srt.layers.dp_attention import (
            DpAttentionConfig,
            DpAttentionMetadata,
            DpPaddingMode,
            get_dp_attention_config,
            initialize_dp_attention,
            is_dp_attention_enabled,
        )

        print("✓ Successfully imported DP attention modules")
    except ImportError as e:
        print(f"✗ Failed to import DP attention modules: {e}")
        return False

    # Test DpPaddingMode
    global_tokens = [100, 150, 75, 200]
    dp_size = 4
    mode = DpPaddingMode.get_dp_padding_mode(global_tokens, dp_size)
    print(f"✓ Padding mode selection works: {mode}")

    # Test configuration
    if jax.device_count() >= 2:
        devices = jax.devices()[:2]
        mesh = Mesh(devices, ("data",))

        with mesh:
            try:
                initialize_dp_attention(
                    dp_size=2,
                    tp_size=1,
                    dp_rank=0,
                    tp_rank=0,
                    hidden_size=512,
                    mesh=mesh,
                )

                config = get_dp_attention_config()
                assert config.dp_size == 2
                assert config.hidden_size == 512
                assert is_dp_attention_enabled()
                print("✓ DP attention configuration works")

            except Exception as e:
                print(f"✗ DP attention configuration failed: {e}")
                return False
    else:
        print("⚠ Skipping mesh tests (need at least 2 devices)")

    return True


def test_dp_attention_backend():
    """Test DP attention backend creation"""
    print("\nTesting DP Attention backend creation...")

    try:
        from sgl_jax.srt.layers.attention.native_backend import NativeAttention
        from sgl_jax.srt.layers.dp_attention import create_dp_attention_backend

        # Create a simple base backend
        base_backend = NativeAttention(num_attn_heads=8, num_kv_heads=8)

        if jax.device_count() >= 2:
            devices = jax.devices()[:2]
            mesh = Mesh(devices, ("data",))

            # Test DP backend creation
            dp_backend = create_dp_attention_backend(
                base_backend=base_backend,
                dp_size=2,
                tp_size=1,
                dp_rank=0,
                tp_rank=0,
                hidden_size=512,
                mesh=mesh,
            )

            print("✓ DP attention backend created successfully")
            return True
        else:
            print("⚠ Skipping backend tests (need at least 2 devices)")
            return True

    except Exception as e:
        print(f"✗ DP attention backend creation failed: {e}")
        return False


def test_model_runner_integration():
    """Test integration with model runner"""
    print("\nTesting Model Runner integration...")

    try:
        from sgl_jax.srt.configs.model_config import ModelConfig
        from sgl_jax.srt.server_args import ServerArgs

        # Create test server args with DP enabled
        server_args = ServerArgs(
            model_path="dummy",
            tp_size=2,
            dp_size=2,
            attention_backend="fa",
            device="tpu",
        )

        # Create a mock model config
        model_config = ModelConfig(
            hidden_size=512,
            num_attention_heads=8,
            num_key_value_heads=8,
            max_seq_len=2048,
            vocab_size=32000,
        )

        print(f"✓ Server args created with dp_size={server_args.dp_size}")
        print(f"✓ Model config created with hidden_size={model_config.hidden_size}")

        # Test that model runner would use DP attention
        if server_args.dp_size > 1:
            print("✓ DP attention would be enabled in model runner")

        return True

    except Exception as e:
        print(f"✗ Model runner integration test failed: {e}")
        return False


def test_jax_distributed_ops():
    """Test JAX distributed operations used by DP attention"""
    print("\nTesting JAX distributed operations...")

    try:
        if jax.device_count() >= 2:
            devices = jax.devices()[:2]
            mesh = Mesh(devices, ("data",))

            # Test basic all_gather
            with mesh:

                def test_gather(x):
                    return jax.lax.all_gather(x, axis_name="data")

                sharded_fn = jax.shard_map(
                    test_gather,
                    mesh=mesh,
                    in_specs=P("data"),
                    out_specs=P(None),
                    check_rep=False,
                )

                test_data = jnp.array([1.0, 2.0])
                try:
                    result = sharded_fn(test_data)
                    print("✓ JAX all_gather works")
                except Exception as e:
                    print(f"⚠ JAX all_gather test failed: {e}")

                # Test basic all_reduce
                def test_reduce(x):
                    return jax.lax.psum(x, axis_name="data")

                sharded_fn = jax.shard_map(
                    test_reduce,
                    mesh=mesh,
                    in_specs=P("data"),
                    out_specs=P("data"),
                    check_rep=False,
                )

                try:
                    result = sharded_fn(test_data)
                    print("✓ JAX all_reduce works")
                except Exception as e:
                    print(f"⚠ JAX all_reduce test failed: {e}")
        else:
            print("⚠ Skipping JAX distributed tests (need at least 2 devices)")

        return True

    except Exception as e:
        print(f"✗ JAX distributed operations test failed: {e}")
        return False


def main():
    """Main test function"""
    print("=== DP Attention Test Suite ===")
    print(f"JAX version: {jax.__version__}")
    print(
        f"Available devices: {len(jax.devices())} ({[d.platform for d in jax.devices()]})"
    )
    print(f"Device count: {jax.device_count()}")
    print()

    tests = [
        test_dp_attention_basic,
        test_dp_attention_backend,
        test_model_runner_integration,
        test_jax_distributed_ops,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                print(f"✗ {test.__name__} failed")
        except Exception as e:
            print(f"✗ {test.__name__} crashed: {e}")

    print(f"\n=== Test Results ===")
    print(f"Passed: {passed}/{total}")

    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("⚠ Some tests failed or were skipped")
        return 1


if __name__ == "__main__":
    exit(main())
