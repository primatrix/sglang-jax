#!/usr/bin/env python3
"""
Test script for universal layer JIT optimization
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
    def __init__(self, hidden_size=512, layer_id=0, rngs=None):
        self.hidden_size = hidden_size
        self.layer_id = layer_id
        # Each layer has different weights
        self.weight = nnx.Param(
            jax.random.normal(rngs.params(), (hidden_size, hidden_size))
            * (layer_id + 1)
        )

    def __call__(self, positions, x, forward_batch, residual=None):
        if residual is None:
            residual = x
        # Simple transformation
        output = jnp.dot(x, self.weight.value) + residual
        kv_fused = (
            jnp.ones((2, x.shape[0], 64)) * self.layer_id
        )  # Mock KV cache with layer ID
        callback_flag = []  # Mock callback flags
        return output, residual, kv_fused, callback_flag


def test_universal_vs_individual_jit():
    """Compare universal JIT vs individual layer JIT compilation"""
    logger.info("Comparing Universal JIT vs Individual JIT...")

    # Create mock layers
    num_layers = 8
    hidden_size = 512
    batch_size = 4
    seq_len = 32
    rng_key = jax.random.PRNGKey(42)
    rngs = nnx.Rngs(rng_key)

    # Create different layers with different weights
    layers = [MockLayer(hidden_size, i, rngs) for i in range(num_layers)]

    # Test input
    x = jax.random.normal(rng_key, (batch_size * seq_len, hidden_size))
    positions = jnp.arange(batch_size * seq_len)
    forward_batch = None  # Mock forward batch

    # Method 1: Individual JIT for each layer (current naive approach)
    logger.info("=== Method 1: Individual JIT (N compilations) ===")

    individual_jits = []
    individual_compile_times = []
    total_individual_compile_time = 0

    for i, layer in enumerate(layers):
        layer_def, layer_state = nnx.split(layer)
        layer_state_leaves, layer_state_def = jax.tree_util.tree_flatten(layer_state)

        @partial(jax.jit, static_argnames=["layer_state_def"])
        def jitted_individual_layer(
            layer_def,
            layer_state_def,
            layer_state_leaves,
            positions,
            x,
            forward_batch,
            residual,
        ):
            layer_state = jax.tree_util.tree_unflatten(
                layer_state_def, layer_state_leaves
            )
            layer = nnx.merge(layer_def, layer_state)
            return layer(positions, x, forward_batch, residual)

        layer_jit = partial(
            jitted_individual_layer, layer_def, layer_state_def, layer_state_leaves
        )

        # Time individual layer compilation
        start_time = time.time()
        residual = None if i == 0 else x
        output, residual, kv_fused, _ = layer_jit(positions, x, forward_batch, residual)
        compile_time = time.time() - start_time

        individual_jits.append(layer_jit)
        individual_compile_times.append(compile_time)
        total_individual_compile_time += compile_time

        logger.info(f"Layer {i} individual JIT compilation: {compile_time:.3f}s")

    logger.info(
        f"Total individual JIT compilation time: {total_individual_compile_time:.3f}s"
    )

    # Method 2: Universal JIT (single compilation)
    logger.info("=== Method 2: Universal JIT (1 compilation) ===")

    # Use first layer to define the common structure
    first_layer = layers[0]
    layer_def, layer_state = nnx.split(first_layer)
    layer_state_leaves, layer_state_def = jax.tree_util.tree_flatten(layer_state)

    @partial(jax.jit, static_argnames=["layer_def", "layer_state_def"])
    def universal_jitted_layer(
        layer_def,
        layer_state_def,
        layer_state_leaves,
        positions,
        x,
        forward_batch,
        residual,
    ):
        layer_state = jax.tree_util.tree_unflatten(layer_state_def, layer_state_leaves)
        layer = nnx.merge(layer_def, layer_state)
        return layer(positions, x, forward_batch, residual)

    # Create universal JIT function
    universal_jit = partial(universal_jitted_layer, layer_def, layer_state_def)

    # Store all layer states
    all_layer_states = []
    for layer in layers:
        _, layer_state = nnx.split(layer)
        layer_state_leaves, _ = jax.tree_util.tree_flatten(layer_state)
        all_layer_states.append(layer_state_leaves)

    # Time universal JIT compilation (happens on first call)
    start_time = time.time()
    residual = None
    for i, layer_state_leaves in enumerate(all_layer_states):
        output, residual, kv_fused, _ = universal_jit(
            layer_state_leaves, positions, x, forward_batch, residual
        )
        if i == 0:  # Only the first call triggers compilation
            first_call_time = time.time() - start_time
            logger.info(
                f"Universal JIT first call (with compilation): {first_call_time:.3f}s"
            )

    universal_compile_time = first_call_time
    logger.info(f"Total universal JIT compilation time: {universal_compile_time:.3f}s")

    # Test runtime performance
    logger.info("=== Runtime Performance Test ===")

    # Individual JIT runtime
    start_time = time.time()
    residual = None
    for i, layer_jit in enumerate(individual_jits):
        output_ind, residual, kv_fused, _ = layer_jit(
            positions, x, forward_batch, residual
        )
    individual_runtime = time.time() - start_time
    logger.info(f"Individual JIT runtime: {individual_runtime:.4f}s")

    # Universal JIT runtime
    start_time = time.time()
    residual = None
    for layer_state_leaves in all_layer_states:
        output_uni, residual, kv_fused, _ = universal_jit(
            layer_state_leaves, positions, x, forward_batch, residual
        )
    universal_runtime = time.time() - start_time
    logger.info(f"Universal JIT runtime: {universal_runtime:.4f}s")

    # Compare outputs
    output_diff = jnp.max(jnp.abs(output_ind - output_uni))
    logger.info(f"Output difference: {output_diff:.8f}")

    # Results
    logger.info("=" * 60)
    logger.info("RESULTS SUMMARY:")
    logger.info(
        f"Individual JIT total compilation: {total_individual_compile_time:.3f}s ({num_layers} compilations)"
    )
    logger.info(
        f"Universal JIT total compilation:  {universal_compile_time:.3f}s (1 compilation)"
    )

    speedup_compile = total_individual_compile_time / universal_compile_time
    logger.info(f"Compilation speedup: {speedup_compile:.1f}x faster!")

    if universal_runtime > 0:
        speedup_runtime = individual_runtime / universal_runtime
        logger.info(f"Runtime ratio: {speedup_runtime:.2f}x (universal vs individual)")

    if output_diff < 1e-5:
        logger.info("✓ Outputs are numerically equivalent")
    else:
        logger.warning("⚠ Outputs differ numerically")

    logger.info("=" * 60)

    return {
        "individual_compile_time": total_individual_compile_time,
        "universal_compile_time": universal_compile_time,
        "individual_runtime": individual_runtime,
        "universal_runtime": universal_runtime,
        "output_diff": float(output_diff),
        "compile_speedup": speedup_compile,
    }


def main():
    """Main test function"""
    logger.info("Starting universal layer JIT test...")

    try:
        results = test_universal_vs_individual_jit()

        logger.info("FINAL SUMMARY:")
        logger.info(
            f"✓ Universal JIT is {results['compile_speedup']:.1f}x faster at compilation!"
        )
        logger.info(
            f"✓ Compilation time reduced from {results['individual_compile_time']:.3f}s to {results['universal_compile_time']:.3f}s"
        )

        if results["output_diff"] < 1e-5:
            logger.info("✓ Results are numerically equivalent")
        else:
            logger.warning("⚠ Results differ numerically")

        return 0

    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
