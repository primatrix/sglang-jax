#!/usr/bin/env python3
"""
Test the new layered JIT architecture
"""

import logging
import time

import jax
import jax.numpy as jnp
from flax import nnx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_architecture_workflow():
    """Test the complete workflow: model loading -> JIT -> forward"""
    logger.info("Testing new layered JIT architecture...")

    # === Step 1: Simulate model structure ===
    logger.info("=== Step 1: Model Structure ===")

    # Mock QWen3DecoderLayer
    class MockDecoderLayer(nnx.Module):
        def __init__(self, hidden_size, layer_id, rngs):
            self.layer_id = layer_id
            self.weight = nnx.Param(
                jax.random.normal(rngs.params(), (hidden_size, hidden_size))
            )

        def __call__(self, positions, hidden_states, forward_batch, residual):
            if residual is None:
                residual = hidden_states
            output = jnp.dot(hidden_states, self.weight.value) + residual
            kv_fused = jnp.ones((2, hidden_states.shape[0], 64)) * self.layer_id
            callback_flag = []
            return output, residual, kv_fused, callback_flag

    # Mock QWen3Model
    class MockTransformer(nnx.Module):
        def __init__(self, hidden_size, num_layers, rngs):
            self.layers = [
                MockDecoderLayer(hidden_size, i, rngs) for i in range(num_layers)
            ]
            # Add layer JIT support
            self.use_layer_jit = True
            self.universal_layer_jit = None
            self.layer_states = None

        def initialize_layer_jit(self):
            """Initialize universal layer JIT at model level"""
            from functools import partial

            # Use first layer to create universal JIT function
            first_layer = self.layers[0]
            layer_def, layer_state = nnx.split(first_layer)
            layer_state_leaves, layer_state_def = jax.tree_util.tree_flatten(
                layer_state
            )

            @partial(jax.jit, static_argnames=["layer_def", "layer_state_def"])
            def universal_jitted_layer(
                layer_def,
                layer_state_def,
                layer_state_leaves,
                positions,
                hidden_states,
                forward_batch,
                residual,
            ):
                layer_state = jax.tree_util.tree_unflatten(
                    layer_state_def, layer_state_leaves
                )
                layer = nnx.merge(layer_def, layer_state)
                return layer(positions, hidden_states, forward_batch, residual)

            self.universal_layer_jit = partial(
                universal_jitted_layer, layer_def, layer_state_def
            )

            # Store all layer states
            self.layer_states = []
            for layer in self.layers:
                _, layer_state = nnx.split(layer)
                layer_state_leaves, _ = jax.tree_util.tree_flatten(layer_state)
                self.layer_states.append(layer_state_leaves)

            logger.info(
                f"Universal layer JIT initialized for {len(self.layers)} layers"
            )

        def __call__(self, forward_batch):
            hidden_states = jnp.ones((32, 512))  # Mock embedding output
            residual = None
            layers_kv_fused = []
            layers_callback_flag = []

            if self.use_layer_jit and self.universal_layer_jit is not None:
                # Use universal layer JIT
                logger.info("Using universal layer JIT...")
                for i, layer_state_leaves in enumerate(self.layer_states):
                    hidden_states, residual, kv_fused, callback_flag = (
                        self.universal_layer_jit(
                            layer_state_leaves,
                            forward_batch.get("positions"),
                            hidden_states,
                            forward_batch,
                            residual,
                        )
                    )
                    layers_kv_fused.append(kv_fused)
                    layers_callback_flag.extend(callback_flag)
            else:
                # Original approach
                logger.info("Using original layer-by-layer approach...")
                for layer in self.layers:
                    hidden_states, residual, kv_fused, callback_flag = layer(
                        forward_batch.get("positions"),
                        hidden_states,
                        forward_batch,
                        residual,
                    )
                    layers_kv_fused.append(kv_fused)
                    layers_callback_flag.extend(callback_flag)

            return hidden_states, layers_kv_fused, layers_callback_flag

    # Mock complete model
    class MockModel(nnx.Module):
        def __init__(self, hidden_size, num_layers, rngs):
            self.transformer = MockTransformer(hidden_size, num_layers, rngs)

        def __call__(self, forward_batch, logits_metadata):
            hidden_states, layers_kv_fused, layers_callback_flag = self.transformer(
                forward_batch
            )
            # Mock logits processing - return JAX arrays directly
            next_token_logits = hidden_states[:4, :]  # Mock logits
            return next_token_logits, layers_kv_fused, layers_callback_flag

    # === Step 2: Simulate model loading ===
    logger.info("=== Step 2: Model Loading ===")

    rng_key = jax.random.PRNGKey(42)
    rngs = nnx.Rngs(rng_key)

    # Create model
    model = MockModel(hidden_size=512, num_layers=4, rngs=rngs)
    logger.info("Model created")

    # Initialize layer JIT (happens in load_model)
    model.transformer.initialize_layer_jit()
    logger.info("Layer JIT initialized at model level")

    # === Step 3: Simulate model_runner JIT ===
    logger.info("=== Step 3: Model Runner JIT ===")

    # Split model for JIT
    model_def, model_state = nnx.split(model)
    model_state_leaves, model_state_def = jax.tree_util.tree_flatten(model_state)

    from functools import partial

    @partial(jax.jit, static_argnames=["model_state_def"])
    def jitted_run_model(
        model_def, model_state_def, model_state_leaves, forward_batch, logits_metadata
    ):
        model_state = jax.tree_util.tree_unflatten(model_state_def, model_state_leaves)
        model = nnx.merge(model_def, model_state)
        return model(forward_batch, logits_metadata)

    jitted_model = partial(
        jitted_run_model, model_def, model_state_def, model_state_leaves
    )
    logger.info("Model JIT wrapper created")

    # === Step 4: Test forward passes ===
    logger.info("=== Step 4: Forward Passes ===")

    # Mock forward batch
    forward_batch = {
        "positions": jnp.arange(32),
        "input_ids": jnp.ones((32,), dtype=jnp.int32),
    }
    logits_metadata = None  # Mock

    # First forward pass (compilation happens here)
    logger.info("Running first forward pass (with compilation)...")
    start_time = time.time()
    next_token_logits, layers_kv_fused, _ = jitted_model(forward_batch, logits_metadata)
    first_pass_time = time.time() - start_time
    logger.info(f"First pass time: {first_pass_time:.3f}s")

    # Second forward pass (no compilation)
    logger.info("Running second forward pass (no compilation)...")
    start_time = time.time()
    next_token_logits, layers_kv_fused, _ = jitted_model(forward_batch, logits_metadata)
    second_pass_time = time.time() - start_time
    logger.info(f"Second pass time: {second_pass_time:.4f}s")

    # Verify outputs
    logger.info(f"Output shape: {next_token_logits.shape}")
    logger.info(f"Number of layer KV caches: {len(layers_kv_fused)}")

    logger.info("=" * 50)
    logger.info("ARCHITECTURE TEST RESULTS:")
    logger.info(f"✓ Model loading: successful")
    logger.info(f"✓ Layer JIT initialization: successful")
    logger.info(f"✓ Model JIT wrapper: successful")
    logger.info(f"✓ First pass (compilation): {first_pass_time:.3f}s")
    logger.info(f"✓ Second pass (inference): {second_pass_time:.4f}s")
    logger.info(f"✓ Speedup after compilation: {first_pass_time/second_pass_time:.1f}x")
    logger.info("=" * 50)

    return {
        "first_pass_time": first_pass_time,
        "second_pass_time": second_pass_time,
        "output_shape": next_token_logits.shape,
        "num_layers": len(layers_kv_fused),
    }


def main():
    logger.info("Starting architecture workflow test...")

    try:
        results = test_architecture_workflow()
        logger.info("✅ Architecture test completed successfully!")
        return 0
    except Exception as e:
        logger.error(f"❌ Architecture test failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
