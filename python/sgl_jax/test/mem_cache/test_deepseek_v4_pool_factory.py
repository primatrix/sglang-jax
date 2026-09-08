from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh

from sgl_jax.srt.mem_cache.deepseek_v4.capacity import (
    build_deepseek_v4_pools,
    plan_deepseek_v4_pools,
)
from sgl_jax.srt.mem_cache.deepseek_v4.pool import DeepseekV4CacheSpec
from sgl_jax.srt.model_executor.model_runner_kv_cache_mixin import ModelRunnerKVCacheMixin


@pytest.fixture(autouse=True)
def isolated_mesh_context():
    # Legacy test modules set a global mesh at import time. Restore it after
    # each test so tests using different device subsets are order-independent.
    with jax.set_mesh(None):
        yield


@pytest.mark.parametrize("p", [128, 256])
@pytest.mark.parametrize("dp", [1, 2])
def test_budget_matches_real_arrays_including_padding_and_tp_replication(p, dp):
    if jax.device_count() < dp * 2:
        pytest.skip("requires four CPU devices")
    mesh = Mesh(np.array(jax.devices()[: dp * 2]).reshape(dp, 2), ("data", "tensor"))
    spec = DeepseekV4CacheSpec((0, 4, 128), 8, 4)
    budget = plan_deepseek_v4_pools(spec, 150000, 3, p, dp, swa_full_tokens_ratio=0.5)
    req, pools, allocator = build_deepseek_v4_pools(spec, budget, p, mesh, 2048, dp)
    assert req.size == 3
    assert pools.compressor_state_pool.size == req.size
    assert (
        pools.token_to_kv_pool.nbytes + pools.compressor_state_pool.nbytes
    ) // dp == budget.allocated_bytes_per_device
    assert budget.allocated_bytes_per_device <= 150000
    per_rank_history = budget.history_tokens // dp // p
    next_swa_pages = max(1, int(np.ceil((per_rank_history + 1) * 0.5)))
    next_bytes = (
        budget.state_bytes_per_device
        + (per_rank_history + 2) * spec.history_bytes_per_page(p)
        + (next_swa_pages + 1) * p * spec.swa_bytes_per_token
    )
    assert next_bytes > 150000  # largest fitting page count, no fractional-page optimism
    # Two tensor shards replicate each KV/state shard, rather than dividing
    # single-head storage and underestimating per-device consumption.
    array = pools.token_to_kv_pool.get_buffer("swa", 0)
    assert array.addressable_shards[0].data.nbytes == array.nbytes // dp
    assert allocator.size == budget.history_tokens


@pytest.mark.parametrize(
    "available,req,cap", [(1, 1, None), (1000000, 1000, None), (1000000, 1, 127)]
)
def test_insufficient_budget_fails_before_allocating(available, req, cap):
    with pytest.raises(ValueError, match="cannot fit"):
        plan_deepseek_v4_pools(
            DeepseekV4CacheSpec((4, 128), 8, 4), available, req, 128, max_total_tokens=cap
        )


def test_real_flash_auto_request_budget_does_not_assume_thousands_of_slots():
    spec = DeepseekV4CacheSpec((0, 0) + ((4, 128) * 20) + (4,))
    budget = plan_deepseek_v4_pools(spec, 1024**3, None, 128, 2)
    assert 0 < budget.max_num_reqs < 2048
    assert budget.max_num_reqs % 2 == 0
    assert budget.allocated_bytes_per_device <= 1024**3
    assert budget.state_bytes_per_device == (budget.max_num_reqs + 1) * spec.state_bytes_per_request


class V4Runner(ModelRunnerKVCacheMixin):
    linear_recurrent_config = None

    def __init__(self):
        self.server_args = SimpleNamespace(
            disable_overlap_schedule=True,
            disable_radix_cache=True,
            enable_mixed_chunk=False,
            kv_cache_dtype="auto",
            swa_full_tokens_ratio=0.5,
        )
        self.model_config = SimpleNamespace(
            context_len=1024,
            hf_config=SimpleNamespace(
                model_type="deepseek_v4",
                num_hidden_layers=3,
                compress_ratios=[0, 4, 128, 4],
                head_dim=8,
                index_head_dim=4,
            ),
        )
        self.mesh = Mesh(np.array(jax.devices()[:1]).reshape(1, 1), ("data", "tensor"))
        self.page_size = 128
        self.is_draft_worker = False
        self.spec_algorithm = None
        self.req_to_token_pool = self.token_to_kv_pool_allocator = None
        self.mem_fraction_static = 0.8
        self.embedding_pool_bytes = 10000
        # V4 must branch before the existing hybrid MHA factory, which cannot
        # represent SWA and compressed history in the same layer.
        self.is_hybrid = True

    def get_available_device_memory(self):
        return 800000  # after weights, before execution headroom


def test_real_runner_init_branches_before_mha_swa_and_honors_headroom(monkeypatch):
    monkeypatch.delenv("SGLANG_CI_SMALL_KV_SIZE", raising=False)
    runner = V4Runner()
    runner.init_memory_pool(2, 512, 1000000, dp_size=1)
    budget = runner.deepseek_v4_pool_budget
    assert budget.available_bytes_per_device == pytest.approx(590000, abs=1)
    assert runner.max_total_num_tokens == 512
    assert runner.swa_max_total_num_tokens == 256
    assert runner.kv_cache_dtype == jnp.bfloat16
    assert runner.req_to_token_pool.size == 2
    assert runner.memory_pools.compressor_state_pool.size == 2
    a = runner.token_to_kv_pool_allocator
    loc = a.alloc_extend([0], [129], [-1], 129)
    a.free_swa(loc[:128])
    a.free(loc)
    assert a.full_available_size() == 512
    assert a.swa_available_size() == 256


def test_ci_cap_is_applied_to_budget_without_inflation(monkeypatch):
    monkeypatch.setenv("SGLANG_CI_SMALL_KV_SIZE", "256")
    runner = V4Runner()
    runner.init_memory_pool(2, 512, 1000000)
    assert runner.max_total_num_tokens == 256


@pytest.mark.parametrize(
    "field,value",
    [
        ("disable_overlap_schedule", False),
        ("disable_radix_cache", False),
        ("enable_mixed_chunk", True),
        ("kv_cache_dtype", "fp8"),
    ],
)
def test_initial_scope_rejected_before_allocating(field, value):
    runner = V4Runner()
    setattr(runner.server_args, field, value)
    with pytest.raises(ValueError):
        runner.init_memory_pool(2, 512, 1000000)
    assert runner.req_to_token_pool is None


def test_real_request_pool_slot_reuse_and_resource_lifecycle():
    runner = V4Runner()
    runner.init_memory_pool(1, 512, 1000000)
    req_pool = runner.req_to_token_pool
    pools = runner.memory_pools
    allocator = runner.token_to_kv_pool_allocator
    first = SimpleNamespace(req_pool_idx=None)
    assert req_pool.alloc([first]) == [0]
    locs = allocator.alloc_extend([0], [129], [-1], 129)
    req_pool.write((0, slice(0, 129)), locs)
    pools.token_to_kv_pool.write(
        "c128",
        2,
        jnp.array([locs[127] // 128]),
        jnp.full((1, 8), 3, jnp.bfloat16),
        jnp.array([True]),
    )
    pools.compressor_state_pool.write(
        "c4",
        1,
        jnp.array([first.req_pool_idx]),
        jnp.full((1, 8, 32), 7, jnp.float32),
        jnp.array([True]),
    )
    pools.replace_all(
        {
            "token_to_kv_pool": pools.token_to_kv_pool.buffers,
            "compressor_state_pool": pools.compressor_state_pool.buffers,
        }
    )
    allocator.free_swa(req_pool.read(0, 128))
    np.testing.assert_array_equal(
        pools.token_to_kv_pool.get_buffer("c128", 2).reshape(-1, 8)[locs[127] // 128], 3
    )
    np.testing.assert_array_equal(pools.compressor_state_pool.get_buffer("c4", 1)[0], 7)
    allocator.free(req_pool.read(0, 129))
    req_pool.free(first)
    second = SimpleNamespace(req_pool_idx=None)
    assert req_pool.alloc([second]) == [0]
    # Lifecycle/consumer owns this initialization event, not the KV allocator.
    pools.compressor_state_pool.reset(jnp.array([second.req_pool_idx]), jnp.array([True]))
    np.testing.assert_array_equal(pools.compressor_state_pool.get_buffer("c4", 1)[0, :, :16], 0)
    assert jnp.isneginf(pools.compressor_state_pool.get_buffer("c4", 1)[0, :, 16:]).all()
    req_pool.free(second)
    assert req_pool.available_size() == 1
    assert allocator.full_available_size() == 512
    assert allocator.swa_available_size() == 256
