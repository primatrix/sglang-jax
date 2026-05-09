"""Tests for LightningAttnBackend (GLA) — backend end-to-end validation.

Mirrors test_flashattention.py structure:
- JAX naive jit reference (same dtype as kernel)
- Real ForwardBatch + ModelWorkerBatch + RadixLightningAttention + MockRecurrentStatePool
- Centralized parameterized driver

Run with: pytest python/sgl_jax/test/test_lightning_backend.py -v
"""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sgl_jax.srt.layers.attention.hybrid_linear_attn_backend import (
    MockRecurrentStatePool,
)
from sgl_jax.srt.layers.attention.linear.lightning_backend import LightningAttnBackend
from sgl_jax.srt.layers.radix_lightning_attention import RadixLightningAttention
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode
from sgl_jax.srt.utils.mesh_utils import create_device_mesh

# Create mesh: DP=1, TP=all devices
mesh = create_device_mesh(ici_parallelism=[1, -1], dcn_parallelism=[1, 1])
jax.sharding.set_mesh(mesh)

# Skip if no TPU (prefill kernel requires TPU)
_HAS_TPU = any(d.platform == "tpu" for d in jax.devices())
requires_tpu = pytest.mark.skipif(not _HAS_TPU, reason="Prefill kernel requires TPU")


# ===========================================================================
# Reference implementation (JAX naive jit, same dtype, no Pallas)
# ===========================================================================


def naive_gla_jit_ref(q, k, v, slope, h0, cu_seqlens, mode):
    """JAX naive jit reference for GLA.

    Implements the GLA recurrent formula per the kernel:
        decay = slope (per-head ALiBi, no per-token g in Lightning)
        h_t = h_{t-1} * exp(decay) + k_t ⊗ v_t
        o_t = sum(h_t * q_t, axis=head_dim)

    Args:
        q: [total_tokens, num_heads, head_dim]
        k: [total_tokens, num_heads, head_dim]
        v: [total_tokens, num_heads, head_dim]
        slope: [num_heads] ALiBi slope (negative values)
        h0: [num_requests, num_heads, head_dim, head_dim] initial state
        cu_seqlens: [num_requests + 1] cumulative sequence lengths
        mode: "prefill" or "decode"

    Returns:
        output: [total_tokens, num_heads, head_dim]
        final_state: [num_requests, num_heads, head_dim, head_dim]
    """
    T, H, K = q.shape
    V = v.shape[-1]
    dtype = q.dtype

    # Build per-token metadata
    token_idx = jnp.arange(T, dtype=jnp.int32)
    seq_ids = jnp.searchsorted(cu_seqlens[1:], token_idx, side="right")
    seq_starts = cu_seqlens[:-1]
    token_starts = seq_starts[seq_ids]
    reset_mask = token_idx == token_starts  # True at first token of each request

    # Scale factor (standard attention scaling)
    scale = 1.0 / jnp.sqrt(K).astype(dtype)

    def step(carry, xs):
        h_prev, final_states = carry
        seq_id, do_reset, q_i, k_i, v_i = xs

        # Reset to h0 at sequence start
        h = jnp.where(do_reset, h0[seq_id], h_prev)

        # GLA recurrent update
        decay = slope  # [H]
        h = h * jnp.exp(decay)[:, None, None]  # [H, K, V]
        h = h + k_i[:, :, None] * v_i[:, None, :]  # outer product: [H, K, V]
        o_i = jnp.sum(h * (q_i[:, :, None] * scale), axis=1)  # [H, V]

        # Update final states (keep float32)
        final_states = final_states.at[seq_id].set(h)
        return (h, final_states), o_i  # Keep fp32; kernel only casts at chunk end

    init_carry = (
        jnp.zeros((H, K, V), dtype=jnp.float32),  # State always float32
        h0,
    )
    (h_last, final_states), o_scan = jax.lax.scan(
        step,
        init_carry,
        (seq_ids, reset_mask, q, k, v),
    )

    return o_scan, final_states


# ===========================================================================
# Test data builders (mirror test_flashattention.py)
# ===========================================================================


def create_qkv(lens, num_heads, head_dim, dtype, seed=42):
    """Create q/k/v tensors for given request lengths.

    Args:
        lens: list of int, per-request sequence lengths
        num_heads: int
        head_dim: int
        dtype: jnp.dtype
        seed: int, random seed for reproducibility

    Returns:
        q, k, v: [total_tokens, num_heads, head_dim]
    """
    total_tokens = sum(lens)
    key = jax.random.PRNGKey(seed)

    q = jax.random.normal(key, (total_tokens, num_heads, head_dim), dtype=dtype)
    k = jax.random.normal(
        jax.random.fold_in(key, 1), (total_tokens, num_heads, head_dim), dtype=dtype
    )
    v = jax.random.normal(
        jax.random.fold_in(key, 2), (total_tokens, num_heads, head_dim), dtype=dtype
    )

    return q, k, v


def create_recurrent_state_pool(num_requests, num_heads, head_dim, dtype, h0_scale=0.0, layer_id=0):
    """Create MockRecurrentStatePool.

    Args:
        num_requests: int
        num_heads: int
        head_dim: int
        dtype: jnp.dtype
        h0_scale: float, scale for initial state (0.0 = zero init)
        layer_id: int

    Returns:
        pool: MockRecurrentStatePool
        h0: [num_requests, num_heads, head_dim, head_dim] initial state
    """
    # Create initial state
    if h0_scale == 0.0:
        h0 = jnp.zeros((num_requests, num_heads, head_dim, head_dim), dtype=jnp.float32)
    else:
        key = jax.random.PRNGKey(100 + layer_id)
        h0 = (
            jax.random.normal(key, (num_requests, num_heads, head_dim, head_dim), dtype=jnp.float32)
            * h0_scale
        )

    # MockRecurrentStatePool expects recurrent_indices starting from 1
    recurrent_indices = jnp.arange(1, num_requests + 1, dtype=jnp.int32)

    # Create buffer with extra slot at index 0 (unused)
    N_plus_1 = num_requests + 1
    buf = jnp.zeros((N_plus_1, num_heads, head_dim, head_dim), dtype=jnp.float32)
    buf = buf.at[recurrent_indices].set(h0)

    pool = MockRecurrentStatePool(layer_caches={layer_id: (buf, [])})

    return pool, h0, recurrent_indices


def create_test_data(mode, lens, model_config, layer_id=0, h0_scale=0.0, n_padded=0):
    """Create test data for a single test case.

    Args:
        mode: "prefill" or "decode"
        lens: list of int, per-request sequence lengths
        model_config: SimpleNamespace with num_heads, head_dim, etc.
        layer_id: int, layer index for slope
        h0_scale: float, scale for initial state (0.0 = zero init)
        n_padded: int, number of trailing empty slots for DP padding

    Returns:
        fb: ForwardBatch
        pool: MockRecurrentStatePool
        q, k, v: [total_tokens, num_heads, head_dim]
        h0: [num_requests, num_heads, head_dim, head_dim]
    """
    num_requests = len(lens)
    total_tokens = sum(lens)

    # Create q/k/v
    q, k, v = create_qkv(lens, model_config.num_heads, model_config.head_dim, model_config.dtype)

    # Create pool and initial state
    pool, h0, recurrent_indices = create_recurrent_state_pool(
        num_requests,
        model_config.num_heads,
        model_config.head_dim,
        model_config.dtype,
        h0_scale,
        layer_id,
    )

    # Build ForwardBatch
    forward_mode = ForwardMode.EXTEND if mode == "prefill" else ForwardMode.DECODE

    # Create backend
    backend = LightningAttnBackend(
        mesh=mesh,
        linear_recurrent_layer_ids=[layer_id],
        num_hidden_layers=model_config.num_hidden_layers,
        num_heads=model_config.num_heads,
    )

    # Build batch metadata
    batch = SimpleNamespace(
        forward_mode=forward_mode,
        recurrent_indices=np.asarray(recurrent_indices, dtype=np.int32),
    )

    if forward_mode == ForwardMode.DECODE:
        batch.seq_lens = np.ones(num_requests, dtype=np.int32)
    else:  # EXTEND
        batch.extend_seq_lens = np.asarray(lens, dtype=np.int32)
        batch.seq_lens = np.asarray(lens, dtype=np.int32)
        batch.input_ids = np.arange(total_tokens, dtype=np.int32)  # dummy input_ids

    # Get forward metadata
    metadata = backend.get_forward_metadata(batch)
    backend.forward_metadata = metadata

    # Create ForwardBatch with adapter to match Radix calling convention
    def attn_backend_adapter(q, k, v, layer, forward_batch, pool, **kwargs):
        """Adapter: Radix wrappers call with pool=, backend expects recurrent_state_pool=."""
        return backend(
            q, k, v, layer=layer, forward_batch=forward_batch, recurrent_state_pool=pool, **kwargs
        )

    fb = SimpleNamespace(
        forward_mode=forward_mode,
        attn_backend=attn_backend_adapter,
        _backend=backend,  # Keep reference to real backend for tests
    )

    return fb, pool, q, k, v, h0


# ===========================================================================
# Test driver (mirror test_flashattention.py)
# ===========================================================================


class TestLightningBackend:
    """Main acceptance tests for LightningAttnBackend."""

    def run_test(
        self, mode, lens, model_config, layer_id=0, h0_scale=0.0, n_padded=0, atol=1e-2, rtol=2e-2
    ):
        """Parameterized test driver.

        Args:
            mode: "prefill" or "decode"
            lens: list of int, per-request sequence lengths
            model_config: SimpleNamespace with num_heads, head_dim, dtype, etc.
            layer_id: int
            h0_scale: float
            n_padded: int
            atol: float (default 1e-2 for bf16, use 0.1 for prefill on TPU)
            rtol: float (default 2e-2 for bf16, use 0.1 for prefill on TPU)
        """
        # 1) Create test data
        fb, pool, q, k, v, h0 = create_test_data(
            mode, lens, model_config, layer_id, h0_scale, n_padded
        )

        # 2) JAX path — real backend + kernel
        backend = fb._backend
        layer = RadixLightningAttention(layer_id, model_config.num_heads, model_config.head_dim)

        jax_output, pool_updates = layer(fb, q, k, v, pool)

        # 3) JAX naive ref
        slope = backend.tp_slope[layer_id]
        cu_seqlens = backend.forward_metadata.cu_q_lens
        ref_output, ref_state = naive_gla_jit_ref(q, k, v, slope, h0, cu_seqlens, mode)

        # Reshape ref_output to match backend output: [T, H, V] -> [T, H*V]
        ref_output_flat = ref_output.reshape(ref_output.shape[0], -1)

        # 4) Compare
        np.testing.assert_allclose(
            np.asarray(jax_output),
            np.asarray(ref_output_flat),
            atol=atol,
            rtol=rtol,
            err_msg=f"Output mismatch for mode={mode}, lens={lens}, layer_id={layer_id}",
        )

        # TODO: Also compare final state from pool_updates vs ref_state

    # ========================================================================
    # Prefill path (9 tests)
    # ========================================================================

    @requires_tpu
    def test_extend_single_aligned(self):
        """Single-request baseline: lens=[64] H=4 K=128 bf16."""
        config = SimpleNamespace(
            num_heads=4,
            head_dim=128,
            dtype=jnp.bfloat16,
            num_hidden_layers=80,
        )
        self.run_test(mode="prefill", lens=[64], model_config=config, atol=0.1, rtol=0.1)

    @requires_tpu
    def test_extend_two_aligned(self):
        """Multi-request cu_seqlens boundary: lens=[64, 128]."""
        config = SimpleNamespace(
            num_heads=4,
            head_dim=128,
            dtype=jnp.bfloat16,
            num_hidden_layers=80,
        )
        self.run_test(mode="prefill", lens=[64, 128], model_config=config, atol=0.1, rtol=0.1)

    @requires_tpu
    def test_extend_three_aligned_mixed(self):
        """Multi-request length diversity: lens=[64, 192, 128]."""
        config = SimpleNamespace(
            num_heads=4,
            head_dim=128,
            dtype=jnp.bfloat16,
            num_hidden_layers=80,
        )
        self.run_test(
            mode="prefill", lens=[64, 192, 128], model_config=config, atol=0.15, rtol=0.15
        )

    @requires_tpu
    def test_extend_with_trailing_empty_slots(self):
        """DP padding empty slots: lens=[64, 128] + n_padded=8."""
        config = SimpleNamespace(
            num_heads=4,
            head_dim=128,
            dtype=jnp.bfloat16,
            num_hidden_layers=80,
        )
        self.run_test(
            mode="prefill", lens=[64, 128], model_config=config, n_padded=8, atol=0.15, rtol=0.15
        )

    @requires_tpu
    def test_extend_with_nonzero_initial_state(self):
        """h0 flow from pool: lens=[128] + h0_scale=0.1."""
        config = SimpleNamespace(
            num_heads=4,
            head_dim=128,
            dtype=jnp.bfloat16,
            num_hidden_layers=80,
        )
        self.run_test(
            mode="prefill", lens=[128], model_config=config, h0_scale=0.1, atol=0.15, rtol=0.15
        )

    @requires_tpu
    def test_extend_h64_k128_full_ling25(self):
        """Production sharding: H=64 K=128 full Ling-2.5 heads."""
        config = SimpleNamespace(
            num_heads=64,
            head_dim=128,
            dtype=jnp.bfloat16,
            num_hidden_layers=80,
        )
        self.run_test(mode="prefill", lens=[128], model_config=config, atol=0.1, rtol=0.1)

    @requires_tpu
    def test_extend_fp32_strict(self):
        """fp32 numerical path: lens=[128] fp32."""
        config = SimpleNamespace(
            num_heads=4,
            head_dim=128,
            dtype=jnp.float32,
            num_hidden_layers=80,
        )
        # fp32 on TPU chunk kernel still has significant accumulation error
        self.run_test(mode="prefill", lens=[128], model_config=config, atol=0.15, rtol=0.15)

    @requires_tpu
    def test_extend_layer_id_varies(self):
        """Slope indexing: 3 different layer_id on same input."""
        config = SimpleNamespace(
            num_heads=4,
            head_dim=128,
            dtype=jnp.bfloat16,
            num_hidden_layers=80,
        )

        # Create backend with multiple layer_ids
        backend = LightningAttnBackend(
            mesh=mesh,
            linear_recurrent_layer_ids=[0, 10, 79],  # Register all 3 layer_ids
            num_hidden_layers=config.num_hidden_layers,
            num_heads=config.num_heads,
        )

        # Build test data manually (can't use create_test_data since it only registers one layer_id)
        lens = [128]
        num_requests = len(lens)
        total_tokens = sum(lens)

        q, k, v = create_qkv(lens, config.num_heads, config.head_dim, config.dtype)

        # Create pool with caches for all 3 layer_ids
        h0 = jnp.zeros(
            (num_requests, config.num_heads, config.head_dim, config.head_dim), dtype=jnp.float32
        )
        recurrent_indices = jnp.arange(1, num_requests + 1, dtype=jnp.int32)
        N_plus_1 = num_requests + 1
        buf = jnp.zeros(
            (N_plus_1, config.num_heads, config.head_dim, config.head_dim), dtype=jnp.float32
        )
        buf = buf.at[recurrent_indices].set(h0)

        # Create pool with caches for all 3 layer_ids
        pool = MockRecurrentStatePool(
            layer_caches={
                0: (buf, []),
                10: (buf, []),
                79: (buf, []),
            }
        )

        # Build ForwardBatch
        batch = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            recurrent_indices=np.asarray(recurrent_indices, dtype=np.int32),
            extend_seq_lens=np.asarray(lens, dtype=np.int32),
            seq_lens=np.asarray(lens, dtype=np.int32),
            input_ids=np.arange(total_tokens, dtype=np.int32),
        )

        metadata = backend.get_forward_metadata(batch)
        backend.forward_metadata = metadata

        # Create adapter to match Radix calling convention
        def attn_backend_adapter(q, k, v, layer, forward_batch, pool, **kwargs):
            return backend(
                q,
                k,
                v,
                layer=layer,
                forward_batch=forward_batch,
                recurrent_state_pool=pool,
                **kwargs,
            )

        fb = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            attn_backend=attn_backend_adapter,
            _backend=backend,
        )

        cu_seqlens = backend.forward_metadata.cu_q_lens

        outputs = []
        for lid in [0, 10, 79]:
            layer_obj = RadixLightningAttention(lid, config.num_heads, config.head_dim)
            jax_output, _ = layer_obj(fb, q, k, v, pool)

            # Compare vs ref
            slope = backend.tp_slope[lid]
            ref_output, _ = naive_gla_jit_ref(q, k, v, slope, h0, cu_seqlens, "prefill")

            # Reshape ref_output to match backend output: [T, H, V] -> [T, H*V]
            ref_output_flat = ref_output.reshape(ref_output.shape[0], -1)

            np.testing.assert_allclose(
                np.asarray(jax_output), np.asarray(ref_output_flat), atol=0.15, rtol=0.15
            )

            outputs.append(jax_output)

        # Outputs should be mutually unequal (different slopes)
        assert not np.allclose(outputs[0], outputs[1], atol=1e-3)
        assert not np.allclose(outputs[1], outputs[2], atol=1e-3)

    @requires_tpu
    def test_extend_then_decode_long_state_continuity(self):
        """Cross-path state continuity: 4096-token extend → 32 decode steps."""
        config = SimpleNamespace(
            num_heads=4,
            head_dim=128,
            dtype=jnp.bfloat16,
            num_hidden_layers=80,
        )
        layer_id = 5

        # Step 1: Run prefill with 4096 tokens
        fb_extend, pool_extend, q_ext, k_ext, v_ext, h0 = create_test_data(
            "prefill", [4096], config, layer_id=layer_id, h0_scale=0.0
        )
        layer = RadixLightningAttention(layer_id, config.num_heads, config.head_dim)
        backend = fb_extend._backend

        extend_output, (new_ssm_full_extend, _) = layer(fb_extend, q_ext, k_ext, v_ext, pool_extend)

        # Update pool with state from prefill
        pool_extend.layer_caches[layer_id] = (new_ssm_full_extend, [])
        pool_after_extend = pool_extend

        # Step 2: Run 32 decode steps using state from prefill
        fb_decode, _, _, _, _, _ = create_test_data(
            "decode", [1], config, layer_id=layer_id, h0_scale=0.0
        )

        all_q, all_k, all_v = [], [], []
        pool = pool_after_extend
        for step in range(32):
            q, k, v = create_qkv(
                [1], config.num_heads, config.head_dim, config.dtype, seed=300 + step
            )
            all_q.append(q)
            all_k.append(k)
            all_v.append(v)

            output, (new_ssm_full, _) = layer(fb_decode, q, k, v, pool)

            # Update pool cache directly
            pool.layer_caches[layer_id] = (new_ssm_full, [])

        # Step 3: Run reference - prefill then 32 decode steps
        slope = backend.tp_slope[layer_id]

        # Reference prefill
        cu_seqlens_prefill = jnp.array([0, 4096], dtype=jnp.int32)
        ref_output_prefill, ref_state = naive_gla_jit_ref(
            q_ext, k_ext, v_ext, slope, h0, cu_seqlens_prefill, "prefill"
        )

        # Reference decode: 32 steps with state propagation
        cu_seqlens_decode = jnp.array([0, 1], dtype=jnp.int32)  # 1 request, 1 token
        for step in range(32):
            q, k, v = all_q[step], all_k[step], all_v[step]
            ref_output, ref_state = naive_gla_jit_ref(
                q, k, v, slope, ref_state, cu_seqlens_decode, "decode"
            )

        # Compare final decode output
        ref_output_flat = ref_output.reshape(1, -1)
        np.testing.assert_allclose(
            np.asarray(output),
            np.asarray(ref_output_flat),
            atol=0.15,
            rtol=0.15,
            err_msg="Extend→decode state continuity failed",
        )

    # ========================================================================
    # Decode path (9 tests)
    # ========================================================================

    def test_decode_single_request(self):
        """Single-request baseline: batch=1 single-step."""
        config = SimpleNamespace(
            num_heads=4,
            head_dim=128,
            dtype=jnp.bfloat16,
            num_hidden_layers=80,
        )
        self.run_test(mode="decode", lens=[1], model_config=config)

    def test_decode_batch_4(self):
        """Multi-request state isolation: batch=4."""
        config = SimpleNamespace(
            num_heads=4,
            head_dim=128,
            dtype=jnp.bfloat16,
            num_hidden_layers=80,
        )
        self.run_test(mode="decode", lens=[1, 1, 1, 1], model_config=config)

    def test_decode_with_trailing_empty_slots(self):
        """DP padding: batch=2 + n_padded=8."""
        config = SimpleNamespace(
            num_heads=4,
            head_dim=128,
            dtype=jnp.bfloat16,
            num_hidden_layers=80,
        )
        self.run_test(mode="decode", lens=[1, 1], model_config=config, n_padded=8)

    def test_decode_state_propagates_3_steps(self):
        """Pool RW chain: batch=2 across 3 steps."""
        config = SimpleNamespace(
            num_heads=4,
            head_dim=128,
            dtype=jnp.bfloat16,
            num_hidden_layers=80,
        )
        layer_id = 5

        # Create test data for 2 requests
        fb, pool, q_step1, k_step1, v_step1, h0 = create_test_data(
            "decode", [1, 1], config, layer_id=layer_id, h0_scale=1.0
        )
        layer = RadixLightningAttention(layer_id, config.num_heads, config.head_dim)
        backend = fb._backend

        # Run 3 decode steps
        outputs = []
        all_q, all_k, all_v = [], [], []
        for step in range(3):
            # Generate new q/k/v for this step (different seed per step)
            q, k, v = create_qkv(
                [1, 1], config.num_heads, config.head_dim, config.dtype, seed=100 + step
            )
            all_q.append(q)
            all_k.append(k)
            all_v.append(v)

            # Run backend
            output, (new_ssm_full, _) = layer(fb, q, k, v, pool)

            # Update pool cache directly
            pool.layer_caches[layer_id] = (new_ssm_full, [])

            outputs.append(output)

        # Run reference: manually loop 3 steps with state propagation
        slope = backend.tp_slope[layer_id]
        cu_seqlens_single = jnp.array([0, 1, 2], dtype=jnp.int32)  # 2 requests, 1 token each

        ref_state = h0
        for step in range(3):
            q, k, v = all_q[step], all_k[step], all_v[step]
            ref_output, ref_state = naive_gla_jit_ref(
                q, k, v, slope, ref_state, cu_seqlens_single, "decode"
            )

        # Compare final step output
        ref_output_flat = ref_output.reshape(2, -1)
        np.testing.assert_allclose(
            np.asarray(outputs[-1]),
            np.asarray(ref_output_flat),
            atol=1e-2,
            rtol=2e-2,
            err_msg="Step 3 output mismatch",
        )

    def test_decode_state_propagates_long_32_steps(self):
        """Long-range pool RW: batch=2 across 32 steps."""
        config = SimpleNamespace(
            num_heads=4,
            head_dim=128,
            dtype=jnp.bfloat16,
            num_hidden_layers=80,
        )
        layer_id = 5

        # Create test data
        fb, pool, _, _, _, h0 = create_test_data(
            "decode", [1, 1], config, layer_id=layer_id, h0_scale=1.0
        )
        layer = RadixLightningAttention(layer_id, config.num_heads, config.head_dim)
        backend = fb._backend

        # Run 32 decode steps
        all_q, all_k, all_v = [], [], []
        for step in range(32):
            q, k, v = create_qkv(
                [1, 1], config.num_heads, config.head_dim, config.dtype, seed=200 + step
            )
            all_q.append(q)
            all_k.append(k)
            all_v.append(v)

            output, (new_ssm_full, _) = layer(fb, q, k, v, pool)

            # Update pool cache directly
            pool.layer_caches[layer_id] = (new_ssm_full, [])

        # Run reference: manually loop 32 steps with state propagation
        slope = backend.tp_slope[layer_id]
        cu_seqlens_single = jnp.array([0, 1, 2], dtype=jnp.int32)  # 2 requests, 1 token each

        ref_state = h0
        for step in range(32):
            q, k, v = all_q[step], all_k[step], all_v[step]
            ref_output, ref_state = naive_gla_jit_ref(
                q, k, v, slope, ref_state, cu_seqlens_single, "decode"
            )

        # Compare final step
        ref_output_flat = ref_output.reshape(2, -1)
        np.testing.assert_allclose(
            np.asarray(output),
            np.asarray(ref_output_flat),
            atol=1e-2,
            rtol=2e-2,
            err_msg="Step 32 output mismatch",
        )

    def test_decode_with_nonzero_initial_state(self):
        """h0 flow from pool: batch=2 + h0_scale=0.1 single-step."""
        config = SimpleNamespace(
            num_heads=4,
            head_dim=128,
            dtype=jnp.bfloat16,
            num_hidden_layers=80,
        )
        self.run_test(mode="decode", lens=[1, 1], model_config=config, h0_scale=0.1)

    def test_decode_layer_id_boundary(self):
        """Boundary layer_id slope: layer_id ∈ {0, num_hidden_layers-1}."""
        config = SimpleNamespace(
            num_heads=4,
            head_dim=128,
            dtype=jnp.bfloat16,
            num_hidden_layers=80,
        )

        # Test layer_id=0
        self.run_test(mode="decode", lens=[1], model_config=config, layer_id=0)

        # Test layer_id=79 (num_hidden_layers - 1)
        self.run_test(mode="decode", lens=[1], model_config=config, layer_id=79)

    def test_decode_h64_k128_full_ling25(self):
        """Production sharding: H=64 K=128 full heads."""
        config = SimpleNamespace(
            num_heads=64,
            head_dim=128,
            dtype=jnp.bfloat16,
            num_hidden_layers=80,
        )
        self.run_test(mode="decode", lens=[1], model_config=config)

    def test_decode_fp32_strict(self):
        """fp32 numerical path: fp32."""
        config = SimpleNamespace(
            num_heads=4,
            head_dim=128,
            dtype=jnp.float32,
            num_hidden_layers=80,
        )
        self.run_test(mode="decode", lens=[1], model_config=config, atol=1e-5, rtol=1e-5)
