#!/usr/bin/env python3
"""
Test script for layer-wise JIT implementation
"""

import logging
import os
import sys
import time

import jax
import jax.numpy as jnp
from flax import nnx

# Add the python path
sys.path.append("/Users/yuyue/go/src/primatrix/sglang-jax/python")

from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.layers.logits_processor import LogitsMetadata
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sgl_jax.srt.model_executor.model_runner import ModelRunner
from sgl_jax.srt.server_args import ServerArgs

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_mock_config():
    """Create a mock model config for testing"""
    # We'll create a simple test that doesn't require full model loading
    # Just test the JIT compilation mechanism
    return None


def create_mock_forward_batch():
    """Create a mock forward batch for testing"""
    batch_size = 4
    seq_len = 32

    # Mock data
    input_ids = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
    positions = jnp.arange(seq_len)[None, :].repeat(batch_size, axis=0)

    # Create forward batch (simplified)
    forward_batch = ForwardBatch(
        input_ids=input_ids.flatten(),
        positions=positions.flatten(),
        forward_mode=ForwardMode.EXTEND,
        bid=0,
    )

    return forward_batch


def create_mock_logits_metadata():
    """Create mock logits metadata"""
    return LogitsMetadata(
        num_tokens_to_logits=4, token_indices=jnp.arange(4)  # batch size
    )


def test_layer_jit_compilation_time():
    """Test compilation time for layer-wise JIT vs full model JIT"""
    logger.info("Testing layer-wise JIT compilation...")

    # Create mock configurations
    model_config = create_mock_config()

    # Create server args
    server_args = ServerArgs(
        model_path="mock-qwen3",
        tp_size=1,
        device="cpu",
        random_seed=42,
        attention_backend="native",
        disable_jax_precompile=True,  # Disable precompile for timing test
    )

    # Create mesh
    devices = jax.devices()[:1]  # Use one device for testing
    mesh = jax.sharding.Mesh(devices, axis_names=("model",))

    # Create RNG
    rng_key = jax.random.PRNGKey(42)
    rngs = nnx.Rngs(rng_key)

    try:
        # Create ModelRunner with layer-wise JIT
        logger.info("Initializing ModelRunner with layer-wise JIT...")
        start_time = time.time()

        model_runner = ModelRunner(
            model_config=model_config,
            mem_fraction_static=0.8,
            tp_size=1,
            server_args=server_args,
            mesh=mesh,
            rngs=rngs,
        )

        init_time = time.time() - start_time
        logger.info(f"Layer-wise JIT initialization time: {init_time:.3f}s")

        # Test forward pass
        forward_batch = create_mock_forward_batch()
        logits_metadata = create_mock_logits_metadata()

        logger.info("Running first forward pass (compilation)...")
        start_time = time.time()

        # This should trigger compilation
        output, cache_miss_count = model_runner._forward(forward_batch, logits_metadata)

        first_pass_time = time.time() - start_time
        logger.info(
            f"First forward pass time (with compilation): {first_pass_time:.3f}s"
        )
        logger.info(f"Cache miss count: {cache_miss_count}")

        # Run second forward pass (no compilation)
        logger.info("Running second forward pass (no compilation)...")
        start_time = time.time()

        output, cache_miss_count = model_runner._forward(forward_batch, logits_metadata)

        second_pass_time = time.time() - start_time
        logger.info(
            f"Second forward pass time (no compilation): {second_pass_time:.3f}s"
        )
        logger.info(f"Cache miss count: {cache_miss_count}")

        logger.info("Layer-wise JIT test completed successfully!")

        return {
            "init_time": init_time,
            "first_pass_time": first_pass_time,
            "second_pass_time": second_pass_time,
            "output_shape": (
                output.next_token_logits.shape
                if hasattr(output, "next_token_logits")
                else "unknown"
            ),
        }

    except Exception as e:
        logger.error(f"Error during layer-wise JIT test: {e}")
        raise


def main():
    """Main test function"""
    logger.info("Starting layer-wise JIT test...")

    try:
        results = test_layer_jit_compilation_time()

        logger.info("=" * 50)
        logger.info("TEST RESULTS:")
        logger.info(f"Initialization time: {results['init_time']:.3f}s")
        logger.info(
            f"First forward pass (with compilation): {results['first_pass_time']:.3f}s"
        )
        logger.info(
            f"Second forward pass (no compilation): {results['second_pass_time']:.3f}s"
        )
        logger.info(f"Output shape: {results['output_shape']}")
        logger.info("=" * 50)

    except Exception as e:
        logger.error(f"Test failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
