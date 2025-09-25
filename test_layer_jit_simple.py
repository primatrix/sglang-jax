#!/usr/bin/env python3
"""
Simple test script for layer-wise JIT compilation logic
"""

import logging
import time
from functools import partial

import jax
import jax.numpy as jnp
from flax import nnx

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Mock layer class for testing
class MockLayer(nnx.Module):
    def __init__(self, hidden_size=512, rngs=None):
        self.hidden_size = hidden_size
        self.weight = nnx.Param(
            jax.random.normal(rngs.params(), (hidden_size, hidden_size))
        )

    def __call__(self, x, residual=None):
        if residual is None:
            residual = x
        # Simple transformation
        output = jnp.dot(x, self.weight.value)
        kv_fused = jnp.ones((2, x.shape[0], 64))  # Mock KV cache
        callback_flag = []  # Mock callback flags
        return output, residual, kv_fused, callback_flag


def test_layer_jit_compilation():
    """Test layer-wise JIT compilation approach"""
    logger.info("Testing layer-wise JIT compilation logic...")

    # Create mock layers
    num_layers = 4
    hidden_size = 512
    rng_key = jax.random.PRNGKey(42)
    rngs = nnx.Rngs(rng_key)

    layers = [MockLayer(hidden_size, rngs) for _ in range(num_layers)]

    # Test full model approach (baseline)
    logger.info("Testing full model JIT approach...")

    # Create full model function
    def full_model_forward(layers, x):
        residual = None
        layers_kv_fused = []
        for layer in layers:
            x, residual, kv_fused, _ = layer(x, residual)
            layers_kv_fused.append(kv_fused)
        return x, layers_kv_fused

    # JIT the full model
    jitted_full_model = jax.jit(full_model_forward, static_argnames=[])

    # Test input
    batch_size = 4
    seq_len = 32
    x = jax.random.normal(rng_key, (batch_size * seq_len, hidden_size))

    # Time full model JIT compilation
    start_time = time.time()
    output_full, kv_full = jitted_full_model(layers, x)
    full_compile_time = time.time() - start_time
    logger.info(
        f"Full model JIT first run (with compilation): {full_compile_time:.3f}s"
    )

    # Second run (no compilation)
    start_time = time.time()
    output_full, kv_full = jitted_full_model(layers, x)
    full_runtime = time.time() - start_time
    logger.info(f"Full model JIT second run (no compilation): {full_runtime:.3f}s")

    # Test layer-wise JIT approach
    logger.info("Testing layer-wise JIT approach...")

    # Create layer-wise JIT functions
    layer_jits = []
    layer_compile_times = []

    for i, layer in enumerate(layers):
        # Split layer for JIT
        layer_def, layer_state = nnx.split(layer)
        layer_state_leaves, layer_state_def = jax.tree_util.tree_flatten(layer_state)

        @partial(jax.jit, static_argnames=["layer_state_def"])
        def jitted_layer(layer_def, layer_state_def, layer_state_leaves, x, residual):
            layer_state = jax.tree_util.tree_unflatten(
                layer_state_def, layer_state_leaves
            )
            layer = nnx.merge(layer_def, layer_state)
            return layer(x, residual)

        layer_jit = partial(
            jitted_layer, layer_def, layer_state_def, layer_state_leaves
        )

        # Time individual layer compilation
        start_time = time.time()
        residual = None if i == 0 else x  # Mock residual
        output, residual, kv_fused, _ = layer_jit(x, residual)
        layer_compile_time = time.time() - start_time

        layer_jits.append(layer_jit)
        layer_compile_times.append(layer_compile_time)
        logger.info(
            f"Layer {i} JIT first run (with compilation): {layer_compile_time:.3f}s"
        )

    total_layer_compile_time = sum(layer_compile_times)
    logger.info(f"Total layer-wise compilation time: {total_layer_compile_time:.3f}s")

    # Test layer-wise forward pass (no compilation)
    def layer_wise_forward(layer_jits, x):
        residual = None
        layers_kv_fused = []
        for layer_jit in layer_jits:
            x, residual, kv_fused, _ = layer_jit(x, residual)
            layers_kv_fused.append(kv_fused)
        return x, layers_kv_fused

    start_time = time.time()
    output_layer, kv_layer = layer_wise_forward(layer_jits, x)
    layer_runtime = time.time() - start_time
    logger.info(f"Layer-wise JIT forward pass (no compilation): {layer_runtime:.3f}s")

    # Compare results
    logger.info("=" * 50)
    logger.info("COMPARISON RESULTS:")
    logger.info(f"Full model compilation time: {full_compile_time:.3f}s")
    logger.info(f"Layer-wise total compilation time: {total_layer_compile_time:.3f}s")
    logger.info(
        f"Compilation time ratio (layer/full): {total_layer_compile_time/full_compile_time:.2f}"
    )
    logger.info(f"Full model runtime: {full_runtime:.3f}s")
    logger.info(f"Layer-wise runtime: {layer_runtime:.3f}s")
    logger.info(f"Runtime ratio (layer/full): {layer_runtime/full_runtime:.2f}")

    # Check output consistency
    output_diff = jnp.max(jnp.abs(output_full - output_layer))
    logger.info(f"Output difference: {output_diff:.6f}")

    if output_diff < 1e-5:
        logger.info("✓ Outputs are consistent between approaches!")
    else:
        logger.warning("⚠ Outputs differ between approaches!")

    logger.info("=" * 50)

    return {
        "full_compile_time": full_compile_time,
        "layer_compile_time": total_layer_compile_time,
        "full_runtime": full_runtime,
        "layer_runtime": layer_runtime,
        "output_diff": float(output_diff),
    }


def main():
    """Main test function"""
    logger.info("Starting simple layer-wise JIT test...")

    try:
        results = test_layer_jit_compilation()

        # Print summary
        speedup_compile = results["full_compile_time"] / results["layer_compile_time"]
        speedup_runtime = results["full_runtime"] / results["layer_runtime"]

        logger.info("SUMMARY:")
        if results["layer_compile_time"] < results["full_compile_time"]:
            logger.info(
                f"✓ Layer-wise JIT is {speedup_compile:.2f}x faster at compilation!"
            )
        else:
            logger.info(
                f"✗ Layer-wise JIT is {1/speedup_compile:.2f}x slower at compilation"
            )

        if results["output_diff"] < 1e-5:
            logger.info("✓ Results are numerically equivalent")
        else:
            logger.warning("⚠ Results differ numerically")

    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
