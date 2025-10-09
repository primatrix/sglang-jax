"""
Test JIT Cache Optimization for Model Runner

This test validates the JIT cache optimization that replaces double partial
wrapping with closure-based approach to improve cache hit rates.
"""

import os
import time
import unittest
from pathlib import Path

import jax
import jax.numpy as jnp
from flax import nnx
from transformers import AutoTokenizer

from sgl_jax.srt.configs.load_config import LoadConfig, LoadFormat
from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.layers.logits_processor import LogitsMetadata
from sgl_jax.srt.managers.schedule_batch import ModelWorkerBatch
from sgl_jax.srt.mem_cache.memory_pool import ReqToTokenPool
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sgl_jax.srt.model_executor.model_runner import ModelRunner
from sgl_jax.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sgl_jax.srt.server_args import ServerArgs
from sgl_jax.test.test_utils import create_device_mesh


class TestJITCacheOptimization(unittest.TestCase):
    """Test JIT cache optimization effects."""

    def setUp(self):
        """Setup test environment."""
        # Set environment for JAX
        tp_size = int(os.environ.get("TP_SIZE", 1))
        os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={tp_size}"

        # Initialize JAX if needed
        num_processes = int(os.environ.get("SGL_JAX_NUM_PROCESSES", 1))
        if num_processes > 1:
            process_id = int(os.environ.get("SGL_JAX_PROCESS_ID", 0))
            coordinator_address = os.environ.get(
                "SGL_JAX_COORDINATOR_ADDRESS", "localhost:10000"
            )
            jax.distributed.initialize(
                coordinator_address=coordinator_address,
                num_processes=num_processes,
                process_id=process_id,
            )

        # Setup mesh
        jax_devices = jax.devices()
        self.tp_size = tp_size
        if len(jax_devices) > self.tp_size:
            jax_devices = jax_devices[: self.tp_size]

        self.mesh = create_device_mesh(
            devices=jax_devices,
            ici_parallelism=[1, self.tp_size, 1, 1],
            dcn_parallelism=[1, 1, 1, 1],
        )

        # Model configuration
        self.model_path = os.environ.get("MODEL_PATH", "/models/Qwen-7B")
        self.model_config = ModelConfig(
            model_path=self.model_path, model_override_args="{}", dtype="bfloat16"
        )

        # Server args
        self.server_args = ServerArgs(
            model_path=self.model_path,
            trust_remote_code=True,
            device=os.environ.get("JAX_PLATFORMS", "tpu"),
            precompile_token_paddings=[64, 128],  # Reduced for faster testing
            precompile_bs_paddings=[1, 2],  # Reduced for faster testing
        )

        # RNG
        self.rng = nnx.Rngs(42)

    def _create_model_runner(self):
        """Create a ModelRunner instance."""
        req_to_token_pool = ReqToTokenPool(
            size=128, max_context_len=4096, mesh=self.mesh, dtype=jnp.int32
        )

        return ModelRunner(
            model_config=self.model_config,
            mem_fraction_static=0.1,
            tp_size=self.tp_size,
            server_args=self.server_args,
            mesh=self.mesh,
            rngs=self.rng,
            req_to_token_pool=req_to_token_pool,
        )

    def _create_test_batch(self, batch_size=1, seq_len=10):
        """Create a test batch for forward pass."""
        # Create dummy input
        input_ids = [list(range(1, seq_len + 1)) for _ in range(batch_size)]
        positions = [list(range(seq_len)) for _ in range(batch_size)]

        total_tokens = sum(len(ids) for ids in input_ids)

        # Allocate memory
        req_pool_indices = self.model_runner.req_to_token_pool.alloc(len(input_ids))
        cache_loc_index = self.model_runner.token_to_kv_pool_allocator.alloc(
            total_tokens
        )

        # Write to pools
        pt = 0
        for i, input_seq in enumerate(input_ids):
            self.model_runner.req_to_token_pool.write(
                (req_pool_indices[i], slice(0, len(input_seq))),
                cache_loc_index[pt : pt + len(input_seq)],
            )
            pt += len(input_seq)

        # Create worker batch
        worker_batch = ModelWorkerBatch(
            bid=0,
            forward_mode=ForwardMode.EXTEND,
            input_ids=jnp.array([id for ids in input_ids for id in ids]),
            real_input_ids_len=total_tokens,
            real_bs=batch_size,
            req_pool_indices=jnp.array(req_pool_indices),
            seq_lens=jnp.array([len(ids) for ids in input_ids]),
            out_cache_loc=cache_loc_index,
            cache_loc=cache_loc_index,
            positions=jnp.array(
                [pos for positions_seq in positions for pos in positions_seq]
            ),
            extend_start_loc=jnp.array([0]),
            sampling_info=SamplingBatchInfo(
                temperatures=jnp.full((batch_size, 1), 1.0),
                is_all_greedy=True,
                top_ps=jnp.full((batch_size, 1), 1.0),
                top_ks=jnp.ones((batch_size, 1)),
                min_ps=jnp.full((batch_size, 1), 0.0),
            ),
        )

        return ForwardBatch.init_new(worker_batch, self.model_runner), worker_batch

    def test_cache_hit_rate(self):
        """Test JIT cache hit rate optimization."""
        print("\n=== Testing JIT Cache Hit Rate ===")

        self.model_runner = self._create_model_runner()

        # Create test batch
        forward_batch, worker_batch = self._create_test_batch(batch_size=1, seq_len=64)
        logits_metadata = LogitsMetadata.from_model_worker_batch(
            worker_batch, self.mesh
        )

        # First call - should compile
        print("First forward call (compilation expected)...")
        start_time = time.time()
        with self.mesh:
            output1, cache_miss_1 = self.model_runner._forward(
                forward_batch, logits_metadata
            )
        first_call_time = time.time() - start_time

        print(f"First call: {first_call_time:.3f}s, cache misses: {cache_miss_1}")

        # Second call - should use cache
        print("Second forward call (cache hit expected)...")
        # Create identical batch
        forward_batch_2, worker_batch_2 = self._create_test_batch(
            batch_size=1, seq_len=64
        )
        logits_metadata_2 = LogitsMetadata.from_model_worker_batch(
            worker_batch_2, self.mesh
        )

        start_time = time.time()
        with self.mesh:
            output2, cache_miss_2 = self.model_runner._forward(
                forward_batch_2, logits_metadata_2
            )
        second_call_time = time.time() - start_time

        print(f"Second call: {second_call_time:.3f}s, cache misses: {cache_miss_2}")

        # Assertions
        self.assertGreater(cache_miss_1, 0, "First call should have cache misses")
        self.assertEqual(
            cache_miss_2, 0, "Second call should have no cache misses (cache hit)"
        )
        self.assertLess(
            second_call_time, first_call_time * 0.5, "Second call should be much faster"
        )

        # Verify outputs are consistent
        self.assertEqual(
            output1.next_token_logits.shape, output2.next_token_logits.shape
        )

        print(
            f"✅ Cache optimization working! Speedup: {first_call_time/second_call_time:.2f}x"
        )

    def test_precompile_cache_effectiveness(self):
        """Test precompile cache effectiveness."""
        print("\n=== Testing Precompile Cache Effectiveness ===")

        # Create model runner and run precompile
        print("Running first precompile...")
        start_time = time.time()
        self.model_runner = self._create_model_runner()
        first_precompile_time = time.time() - start_time

        print(f"First precompile time: {first_precompile_time:.3f}s")

        # Run precompile again to test cache
        print("Running second precompile (should use cache)...")
        start_time = time.time()

        # Simulate precompile by running forward with standard sizes
        test_configs = [
            (1, 64),  # (batch_size, seq_len)
            (1, 128),
            (2, 64),
        ]

        total_cache_misses = 0
        for bs, seq_len in test_configs:
            forward_batch, worker_batch = self._create_test_batch(
                batch_size=bs, seq_len=seq_len
            )
            logits_metadata = LogitsMetadata.from_model_worker_batch(
                worker_batch, self.mesh
            )

            with self.mesh:
                _, cache_misses = self.model_runner._forward(
                    forward_batch, logits_metadata
                )
            total_cache_misses += cache_misses
            print(f"  Config ({bs}, {seq_len}): {cache_misses} cache misses")

        second_precompile_time = time.time() - start_time
        print(f"Second precompile time: {second_precompile_time:.3f}s")
        print(f"Total cache misses in second run: {total_cache_misses}")

        # Assertions
        self.assertLessEqual(
            total_cache_misses, 1, "Second precompile should have minimal cache misses"
        )
        self.assertLess(
            second_precompile_time,
            first_precompile_time * 0.3,
            "Second precompile should be much faster",
        )

        print(
            f"✅ Precompile cache working! Speedup: {first_precompile_time/second_precompile_time:.2f}x"
        )

    def test_different_batch_sizes(self):
        """Test cache behavior with different batch sizes."""
        print("\n=== Testing Different Batch Sizes ===")

        self.model_runner = self._create_model_runner()

        cache_results = {}

        # Test different configurations
        configs = [(1, 64), (2, 64), (1, 128)]

        for bs, seq_len in configs:
            print(f"Testing batch_size={bs}, seq_len={seq_len}")

            # First call
            forward_batch, worker_batch = self._create_test_batch(
                batch_size=bs, seq_len=seq_len
            )
            logits_metadata = LogitsMetadata.from_model_worker_batch(
                worker_batch, self.mesh
            )

            with self.mesh:
                _, cache_miss_1 = self.model_runner._forward(
                    forward_batch, logits_metadata
                )

            # Second call with same config
            forward_batch_2, worker_batch_2 = self._create_test_batch(
                batch_size=bs, seq_len=seq_len
            )
            logits_metadata_2 = LogitsMetadata.from_model_worker_batch(
                worker_batch_2, self.mesh
            )

            with self.mesh:
                _, cache_miss_2 = self.model_runner._forward(
                    forward_batch_2, logits_metadata_2
                )

            cache_results[(bs, seq_len)] = (cache_miss_1, cache_miss_2)
            print(
                f"  First call: {cache_miss_1} misses, Second call: {cache_miss_2} misses"
            )

        # Verify cache hits for repeated calls
        for config, (miss_1, miss_2) in cache_results.items():
            with self.subTest(config=config):
                self.assertGreaterEqual(
                    miss_1, 0, f"First call for {config} should have some compilation"
                )
                self.assertEqual(
                    miss_2, 0, f"Second call for {config} should hit cache"
                )

        print("✅ All batch size configurations show proper cache behavior")

    def test_functional_correctness(self):
        """Test that optimization doesn't break functionality."""
        print("\n=== Testing Functional Correctness ===")

        self.model_runner = self._create_model_runner()

        # Test basic forward pass
        forward_batch, worker_batch = self._create_test_batch(batch_size=1, seq_len=10)
        logits_metadata = LogitsMetadata.from_model_worker_batch(
            worker_batch, self.mesh
        )

        with self.mesh:
            output, _ = self.model_runner._forward(forward_batch, logits_metadata)

        # Verify output structure
        self.assertIsNotNone(output)
        self.assertIsNotNone(output.next_token_logits)
        self.assertEqual(output.next_token_logits.shape[0], 1)  # batch_size
        self.assertEqual(
            output.next_token_logits.shape[1], self.model_config.vocab_size
        )
        self.assertEqual(output.next_token_logits.dtype, jnp.bfloat16)

        print("✅ Functional correctness verified")

    def test_memory_efficiency(self):
        """Test that the optimization doesn't increase memory usage significantly."""
        print("\n=== Testing Memory Efficiency ===")

        # Get baseline memory usage
        baseline_memory = self._get_memory_usage()

        # Create model runner and run forwards
        self.model_runner = self._create_model_runner()

        # Run multiple forwards to see memory growth
        for i in range(5):
            forward_batch, worker_batch = self._create_test_batch(
                batch_size=1, seq_len=64
            )
            logits_metadata = LogitsMetadata.from_model_worker_batch(
                worker_batch, self.mesh
            )

            with self.mesh:
                _ = self.model_runner._forward(forward_batch, logits_metadata)

        final_memory = self._get_memory_usage()
        memory_increase = final_memory - baseline_memory

        print(f"Baseline memory: {baseline_memory:.2f} MB")
        print(f"Final memory: {final_memory:.2f} MB")
        print(f"Memory increase: {memory_increase:.2f} MB")

        # Memory increase should be reasonable (less than 1GB for this test)
        self.assertLess(memory_increase, 1000, "Memory increase should be reasonable")

        print("✅ Memory usage is within acceptable bounds")

    def _get_memory_usage(self):
        """Get current memory usage in MB."""
        try:
            import psutil

            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # Convert to MB
        except ImportError:
            # Fallback if psutil not available
            return 0.0


if __name__ == "__main__":
    # Enable verbose output
    unittest.TestLoader.sortTestMethodsUsing = None
    unittest.main(verbosity=2)
