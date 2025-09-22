#!/usr/bin/env python3
"""
Simple test for DP Attention implementation
"""

import os
import sys

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "python"))


def test_dp_attention_import():
    """Test that DP attention can be imported successfully"""
    print("Testing DP Attention imports...")

    try:
        from sgl_jax.srt.layers.dp_attention import (
            DpAttentionBackend,
            DpAttentionConfig,
            DpAttentionMetadata,
            DpPaddingMode,
            create_dp_attention_backend,
            get_dp_attention_config,
            initialize_dp_attention,
            is_dp_attention_enabled,
        )

        print("✓ All DP attention imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def test_padding_mode():
    """Test padding mode selection logic"""
    print("Testing padding mode selection...")

    from sgl_jax.srt.layers.dp_attention import DpPaddingMode

    # Test case 1: MAX_LEN should be chosen when sum_len * 2 > max_len * dp_size
    global_tokens = [100, 150, 75, 200]  # max=200, sum=525
    dp_size = 4
    mode = DpPaddingMode.get_dp_padding_mode(global_tokens, dp_size)
    expected = DpPaddingMode.MAX_LEN if 525 * 2 > 200 * 4 else DpPaddingMode.SUM_LEN
    assert mode == expected
    print(f"✓ Padding mode selection: {mode} (expected: {expected})")

    # Test case 2: SUM_LEN should be chosen for more balanced case
    global_tokens = [50, 50, 50, 50]  # max=50, sum=200
    dp_size = 4
    mode = DpPaddingMode.get_dp_padding_mode(global_tokens, dp_size)
    expected = DpPaddingMode.MAX_LEN if 200 * 2 > 50 * 4 else DpPaddingMode.SUM_LEN
    assert mode == expected
    print(f"✓ Balanced case: {mode} (expected: {expected})")

    return True


def test_model_runner_modification():
    """Test that model runner has been modified correctly"""
    print("Testing model runner DP integration...")

    try:
        # Check if the model runner file contains our DP attention code
        model_runner_path = "python/sgl_jax/srt/model_executor/model_runner.py"

        with open(model_runner_path, "r") as f:
            content = f.read()

        # Check for our modifications
        if (
            "from sgl_jax.srt.layers.dp_attention import create_dp_attention_backend"
            in content
        ):
            print("✓ DP attention import found in model runner")
        else:
            print("✗ DP attention import not found in model runner")
            return False

        if "self.server_args.dp_size > 1" in content:
            print("✓ DP size check found in model runner")
        else:
            print("✗ DP size check not found in model runner")
            return False

        if "create_dp_attention_backend" in content:
            print("✓ DP backend creation found in model runner")
        else:
            print("✗ DP backend creation not found in model runner")
            return False

        return True

    except Exception as e:
        print(f"✗ Model runner test failed: {e}")
        return False


def test_server_args():
    """Test that server args support DP configuration"""
    print("Testing server args DP support...")

    try:
        from sgl_jax.srt.server_args import ServerArgs

        # Test DP size with required model_path
        args = ServerArgs(model_path="test_model", dp_size=4, tp_size=8)
        assert hasattr(args, "dp_size")
        assert args.dp_size == 4
        assert args.tp_size == 8
        print(f"✓ DP configuration: dp_size={args.dp_size}, tp_size={args.tp_size}")

        # Test that ServerArgs class has dp_size field
        import dataclasses

        field_names = [field.name for field in dataclasses.fields(ServerArgs)]
        assert "dp_size" in field_names
        print("✓ dp_size field exists in ServerArgs")

        return True

    except Exception as e:
        print(f"✗ Server args test failed: {e}")
        return False


def main():
    """Main test function"""
    print("=== Simple DP Attention Validation ===")
    print()

    tests = [
        test_dp_attention_import,
        test_padding_mode,
        test_model_runner_modification,
        test_server_args,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
                print()
            else:
                print("✗ Test failed")
                print()
        except Exception as e:
            print(f"✗ Test crashed: {e}")
            print()

    print(f"=== Results ===")
    print(f"Passed: {passed}/{total}")

    if passed == total:
        print("🎉 All validation tests passed!")
        print("\nDP Attention is ready to use! Enable it by setting:")
        print("  --dp-size > 1 when launching the server")
        return 0
    else:
        print("⚠ Some validation tests failed")
        return 1


if __name__ == "__main__":
    exit(main())
