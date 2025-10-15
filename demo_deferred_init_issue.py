#!/usr/bin/env python3
"""
Demo to reproduce the deferred weight initialization issue with tp=4

This reproduces the actual ModelRunner initialization flow with real JAX operations
to observe memory usage patterns during weight initialization.

Run with: python demo_deferred_init_issue.py
"""

import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax import sharding
from jax.sharding import PartitionSpec as P

# Configuration
HIDDEN_SIZE = 4096  # Model hidden size
INTERMEDIATE_SIZE = 11008  # MLP intermediate size
NUM_LAYERS = 4  # Number of transformer layers
TP_SIZE = 4  # Tensor parallelism size


def get_memory_usage():
    """Get actual memory usage if available"""
    try:
        stats = {}
        for i, device in enumerate(jax.devices()):
            try:
                device_stats = device.memory_stats()
                stats[f"device_{i}"] = device_stats.get("bytes_in_use", 0) / (1024**3)
            except:
                stats[f"device_{i}"] = "N/A"
        return stats
    except:
        return {f"device_{i}": "N/A" for i in range(len(jax.devices()))}


def print_memory(stage_name):
    """Print current memory usage"""
    memory = get_memory_usage()
    print(f"\n[{stage_name}] Memory usage:")
    for device, usage in memory.items():
        print(
            f"  {device}: {usage}GB"
            if isinstance(usage, float)
            else f"  {device}: {usage}"
        )
    return memory


def create_mesh(tp_size):
    """Create device mesh for tensor parallelism"""
    devices = jax.devices()
    if len(devices) < tp_size:
        print(
            f"Warning: Only {len(devices)} devices available, requested tp_size={tp_size}"
        )
        tp_size = len(devices)

    if tp_size == 1:
        devices_array = np.array(devices[:1]).reshape(1, 1)
        mesh = sharding.Mesh(devices_array, ["data", "tensor"])
    else:
        devices_array = np.array(devices[:tp_size]).reshape(1, tp_size)
        mesh = sharding.Mesh(devices_array, ["data", "tensor"])

    print(f"Created mesh with devices: {devices[:tp_size]}")
    return mesh


# Mock model components that mimic the actual LinearBase behavior
class MockLinear(nnx.Module):
    """Mock linear layer that mimics LinearBase initialization"""

    def __init__(self, input_size, output_size, kernel_axes, rngs, dtype=jnp.bfloat16):
        print(f"    Initializing MockLinear({input_size}, {output_size})")
        print(
            f"    Expected weight size: {(input_size * output_size * 2) / (1024**3):.3f}GB"
        )

        # Record memory before weight creation
        before_memory = print_memory("Before weight creation")

        # This mimics the actual LinearBase.__init__ process
        print("    Creating weight with nnx.with_partitioning...")
        self.weight = nnx.Param(
            nnx.with_partitioning(nnx.initializers.normal(), kernel_axes)(
                rngs.params(), (input_size, output_size), dtype
            )
        )

        # Record memory after weight creation
        after_memory = print_memory("After weight creation")

        # Show actual sharding info (only outside JIT context)
        try:
            print(f"    Weight sharding: {self.weight.value.sharding}")
            print(f"    Weight devices: {list(self.weight.value.devices())}")
        except:
            print("    Weight sharding info: (unavailable in JIT context)")

        # Calculate memory difference if possible
        try:
            for device in before_memory:
                if isinstance(before_memory[device], float) and isinstance(
                    after_memory[device], float
                ):
                    diff = after_memory[device] - before_memory[device]
                    if diff > 0:
                        print(f"    Memory increase on {device}: +{diff:.3f}GB")
        except:
            pass

    def __call__(self, x):
        return jnp.dot(x, self.weight.value)


class MockMLP(nnx.Module):
    """Mock MLP that mimics QWenMLP"""

    def __init__(
        self, hidden_size, intermediate_size, layer_id, rngs, dtype=jnp.bfloat16
    ):
        print(f"  Creating MLP for layer {layer_id}")

        self.w1 = MockLinear(
            hidden_size,
            intermediate_size,
            kernel_axes=(None, "tensor"),
            rngs=rngs,
            dtype=dtype,
        )

        self.w2 = MockLinear(
            hidden_size,
            intermediate_size,
            kernel_axes=(None, "tensor"),
            rngs=rngs,
            dtype=dtype,
        )

        self.c_proj = MockLinear(
            intermediate_size,
            hidden_size,
            kernel_axes=("tensor", None),
            rngs=rngs,
            dtype=dtype,
        )

    def __call__(self, x):
        a1 = self.w1(x)
        a2 = self.w2(x)
        return self.c_proj(a1 * jax.nn.silu(a2))


class MockAttention(nnx.Module):
    """Mock attention that mimics QWenAttention"""

    def __init__(self, hidden_size, layer_id, rngs, dtype=jnp.bfloat16):
        print(f"  Creating Attention for layer {layer_id}")

        self.q_proj = MockLinear(
            hidden_size,
            hidden_size,
            kernel_axes=(None, "tensor"),
            rngs=rngs,
            dtype=dtype,
        )

        self.k_proj = MockLinear(
            hidden_size,
            hidden_size,
            kernel_axes=(None, "tensor"),
            rngs=rngs,
            dtype=dtype,
        )

        self.v_proj = MockLinear(
            hidden_size,
            hidden_size,
            kernel_axes=(None, "tensor"),
            rngs=rngs,
            dtype=dtype,
        )

        self.o_proj = MockLinear(
            hidden_size,
            hidden_size,
            kernel_axes=("tensor", None),
            rngs=rngs,
            dtype=dtype,
        )

    def __call__(self, x):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        # Simplified attention
        attn_output = q * k * v  # Mock attention computation
        return self.o_proj(attn_output)


class MockTransformerLayer(nnx.Module):
    """Mock transformer layer"""

    def __init__(
        self, hidden_size, intermediate_size, layer_id, rngs, dtype=jnp.bfloat16
    ):
        print(f"Creating TransformerLayer {layer_id}")

        self.self_attn = MockAttention(hidden_size, layer_id, rngs, dtype)
        self.mlp = MockMLP(hidden_size, intermediate_size, layer_id, rngs, dtype)

    def __call__(self, x):
        attn_out = self.self_attn(x)
        mlp_out = self.mlp(x)
        return attn_out + mlp_out


class MockModel(nnx.Module):
    """Mock model that mimics the actual model structure"""

    def __init__(self, config, dtype, rngs, mesh):
        print("=" * 60)
        print("INITIALIZING MOCK MODEL")
        print("=" * 60)
        print(f"Hidden size: {config['hidden_size']}")
        print(f"Intermediate size: {config['intermediate_size']}")
        print(f"Number of layers: {config['num_layers']}")
        print(f"Data type: {dtype}")

        # Record initial memory
        initial_memory = print_memory("Initial state")

        # Create transformer layers
        self.layers = []
        for i in range(config["num_layers"]):
            print(f"\n--- Layer {i+1}/{config['num_layers']} ---")
            layer = MockTransformerLayer(
                config["hidden_size"], config["intermediate_size"], i, rngs, dtype
            )
            self.layers.append(layer)

            # Print memory after each layer
            layer_memory = print_memory(f"After layer {i+1}")

        # Final memory state
        final_memory = print_memory("Model initialization complete")

        print("\n" + "=" * 60)
        print("MEMORY ANALYSIS")
        print("=" * 60)

        # Calculate total expected memory
        total_params = self.calculate_total_params(config)
        total_memory_gb = total_params * 2 / (1024**3)  # bfloat16 = 2 bytes
        expected_per_device = total_memory_gb / len(jax.devices()[:TP_SIZE])

        print(f"Total parameters: {total_params:,}")
        print(f"Total model memory: {total_memory_gb:.3f}GB")
        print(f"Expected per device: {expected_per_device:.3f}GB")

        # Analyze memory spikes if possible
        try:
            max_increase = 0
            max_device = ""
            for device in initial_memory:
                if isinstance(initial_memory[device], float) and isinstance(
                    final_memory[device], float
                ):
                    increase = final_memory[device] - initial_memory[device]
                    if increase > max_increase:
                        max_increase = increase
                        max_device = device

            if max_increase > 0:
                spike_ratio = (
                    max_increase / expected_per_device if expected_per_device > 0 else 0
                )
                print(f"Max memory increase: {max_increase:.3f}GB on {max_device}")
                print(f"Memory spike ratio: {spike_ratio:.1f}x")

                if spike_ratio > 1.5:
                    print("⚠️  MEMORY SPIKE DETECTED - This confirms the issue!")
                else:
                    print("✅ No significant memory spike detected")
        except:
            print("Could not analyze memory differences (memory stats not available)")

    def calculate_total_params(self, config):
        """Calculate total number of parameters"""
        h = config["hidden_size"]
        i = config["intermediate_size"]
        n_layers = config["num_layers"]

        # Per layer: q_proj + k_proj + v_proj + o_proj + w1 + w2 + c_proj
        params_per_layer = h * h + h * h + h * h + h * h + h * i + h * i + i * h
        return n_layers * params_per_layer

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class MockModelRunner:
    """Mock ModelRunner that mimics the actual initialization flow"""

    def __init__(self, tp_size=4):
        self.tp_size = tp_size
        self.mesh = create_mesh(tp_size)

        print("\n" + "=" * 80)
        print("MOCK MODEL RUNNER INITIALIZATION")
        print("=" * 80)

        # Create model config
        self.model_config = {
            "hidden_size": HIDDEN_SIZE,
            "intermediate_size": INTERMEDIATE_SIZE,
            "num_layers": NUM_LAYERS,
        }

        # Initialize model with mesh context (mimics actual flow)
        with self.mesh:
            self.rngs = nnx.Rngs(42)
            self.dtype = jnp.bfloat16

            print(f"Creating model with tp_size={tp_size}")
            start_time = time.time()

            # This mimics the model creation in model_loader.py
            self.model = self.create_model()
            print_memory("Created Model")
            init_time = time.time() - start_time
            print(f"\nModel initialization completed in {init_time:.3f}s")

            # Initialize JIT (mimics ModelRunner.initialize_jit)
            self.initialize_jit()

    def create_model(self):
        """Create model like in the actual code"""

        @nnx.jit
        def create_model_jitted(rngs: nnx.Rngs):
            model = MockModel(self.model_config, self.dtype, rngs, self.mesh)

            # Apply sharding constraint like in the actual code
            state = nnx.state(model)
            pspecs = nnx.get_partition_spec(state)
            sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
            nnx.update(model, sharded_state)

            return model

        return create_model_jitted(self.rngs)

    def initialize_jit(self):
        """Initialize JIT functions like in actual ModelRunner"""
        print("\n" + "=" * 40)
        print("INITIALIZING JIT")
        print("=" * 40)

        # Split model for JIT (mimics actual flow)
        model_def, model_state = nnx.split(self.model)
        model_state_leaves, model_state_def = jax.tree_util.tree_flatten(model_state)

        @partial(jax.jit, static_argnames=["model_state_def"])
        def jitted_forward(model_def, model_state_def, model_state_leaves, x):
            model_state = jax.tree_util.tree_unflatten(
                model_state_def, model_state_leaves
            )
            model = nnx.merge(model_def, model_state)
            return model(x)

        self.jitted_forward = partial(
            jitted_forward, model_def, model_state_def, model_state_leaves
        )

        # Test JIT compilation
        print("Triggering JIT compilation...")
        test_input = jnp.ones((1, 10, HIDDEN_SIZE), dtype=self.dtype)

        compile_memory = print_memory("Before JIT compilation")

        # This will trigger actual JIT compilation
        _ = self.jitted_forward(test_input)

        after_jit_memory = print_memory("After JIT compilation")

        print("JIT compilation completed")


def main():
    """Main function to run the demo"""
    print("JAX Devices:", jax.devices())
    print("JAX Device Count:", len(jax.devices()))

    if len(jax.devices()) < TP_SIZE:
        print(
            f"Note: Requested tp_size={TP_SIZE}, but only {len(jax.devices())} devices available"
        )

    # Run the actual demo
    runner = MockModelRunner(tp_size=TP_SIZE)

    print("\n" + "=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)
    print("This demo shows the actual memory usage pattern during:")
    print("1. Model weight initialization (LinearBase creation)")
    print("2. Sharding constraint application")
    print("3. JIT compilation")
    print("4. Real memory allocation on devices")
    print("\nLook for memory spikes on device_0 during initialization!")


if __name__ == "__main__":
    main()
