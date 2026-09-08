"""C1 resource acceptance on real JAX arrays, without complete model weights."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh

from sgl_jax.srt.mem_cache.deepseek_v4.pool import DeepseekV4CacheSpec, DeepseekV4TokenToKVPool
from sgl_jax.srt.mem_cache.deepseek_v4.state import DeepseekV4CompressStatePool
from sgl_jax.srt.mem_cache.memory_pool import MemoryPools


@pytest.fixture(autouse=True)
def isolated_mesh_context():
    # Legacy test modules set a global mesh at import time. Restore it after
    # each test so tests using different device subsets are order-independent.
    with jax.set_mesh(None):
        yield


def mesh(dp=1):
    if jax.device_count() < dp:
        pytest.skip(
            f"requires {dp} CPU devices (XLA_FLAGS=--xla_force_host_platform_device_count=4)"
        )
    return Mesh(np.array(jax.devices()[:dp]).reshape(dp, 1), ("data", "tensor"))


def test_backbone_mapping_excludes_draft_and_accounts_every_byte():
    cfg = SimpleNamespace(
        num_hidden_layers=43,
        compress_ratios=[0, 0] + [4, 128] * 20 + [4] + [128, 4, 128],
        head_dim=512,
        index_head_dim=128,
    )
    spec = DeepseekV4CacheSpec.from_config(cfg)
    pool = DeepseekV4TokenToKVPool(128, 128, 128, spec, mesh())
    assert {f: len(v) for f, v in pool.buffers.items()} == {
        "swa": 43,
        "c4": 21,
        "c128": 20,
        "indexer": 21,
    }
    assert pool.get_buffer("swa", 42).shape == (256, 512)
    assert pool.get_buffer("c4", 42).shape == (2, 32, 512)
    assert pool.get_buffer("c128", 41).shape == (2, 1, 512)
    assert pool.get_buffer("indexer", 42).shape == (2, 32, 128)
    assert all(a.dtype == jnp.bfloat16 for group in pool.buffers.values() for a in group)
    expected = 2 * (43 * 256 * 512 + 21 * 2 * 32 * 512 + 20 * 2 * 512 + 21 * 2 * 32 * 128)
    assert pool.nbytes == expected
    with pytest.raises(KeyError):
        pool.get_buffer("c4", 43)
    with pytest.raises(KeyError):
        pool.get_buffer("c128", 2)


@pytest.mark.parametrize("p", [128, 256])
def test_jit_roundtrip_complete_pool_updates_and_padding(p):
    spec = DeepseekV4CacheSpec((0, 4, 128), 8, 4)
    kv = DeepseekV4TokenToKVPool(p, p, p, spec, mesh())
    state = DeepseekV4CompressStatePool(2, spec, mesh())
    pools = MemoryPools(token_to_kv_pool=kv, compressor_state_pool=state)
    traces = []

    @jax.jit
    def step(pools, token, request):
        traces.append(True)
        pools.token_to_kv_pool.write(
            "swa",
            0,
            jnp.array([p, 0]),
            jnp.full((2, 8), token, jnp.bfloat16),
            jnp.array([True, False]),
        )
        pools.compressor_state_pool.write(
            "c4",
            1,
            jnp.array([request, 0]),
            jnp.full((2, 8, 32), token, jnp.float32),
            jnp.array([True, False]),
        )
        return {
            "token_to_kv_pool": pools.token_to_kv_pool.buffers,
            "compressor_state_pool": pools.compressor_state_pool.buffers,
        }

    pools.replace_all(step(pools, 3, 0))
    pools.replace_all(step(pools, 7, 1))
    assert len(traces) == 1
    np.testing.assert_array_equal(kv.get_buffer("swa", 0)[p], 7)
    np.testing.assert_array_equal(kv.get_buffer("swa", 0)[0], 0)
    np.testing.assert_array_equal(state.get_buffer("c4", 1)[0], 3)
    np.testing.assert_array_equal(state.get_buffer("c4", 1)[1], 7)
    assert jnp.isneginf(state.get_buffer("c4", 1)[2, :, 16:]).all()
    with pytest.raises(ValueError, match="exactly match"):
        pools.replace_all({"token_to_kv_pool": kv.buffers})
    with pytest.raises(ValueError, match="every buffer"):
        kv.replace_buffer({"swa": kv.buffers["swa"]})
    with pytest.raises(ValueError, match="shape/dtype"):
        kv.replace_buffer({**kv.buffers, "c4": (jnp.zeros((1,), jnp.float32),)})


def test_state_zero_is_legal_reset_is_empty_and_families_are_independent():
    spec = DeepseekV4CacheSpec((4, 128), 8, 4)
    state = DeepseekV4CompressStatePool(2, spec, mesh())
    indices = state.state_indices(jnp.array([0, 1, -1, 2]), jnp.array([True, False, True, True]))
    np.testing.assert_array_equal(indices, [0, 2, 2, 2])
    state.write("c4", 0, jnp.array([0, 0]), jnp.full((2, 8, 32), 5.0), jnp.array([True, False]))
    np.testing.assert_array_equal(state.get_buffer("c4", 0)[0], 5)
    np.testing.assert_array_equal(state.get_buffer("indexer", 0)[0, :, :8], 0)
    assert jnp.isneginf(state.get_buffer("indexer", 0)[0, :, 8:]).all()
    state.reset(jnp.array([0]), jnp.array([True]))
    for arrays in state.buffers.values():
        for a in arrays:
            np.testing.assert_array_equal(a[0, :, : a.shape[-1] // 2], 0)
            assert jnp.isneginf(a[0, :, a.shape[-1] // 2 :]).all()
    # B reuses A's request slot, without any separate state free list.
    state.write("c128", 1, jnp.array([0]), jnp.full((1, 128, 16), 9.0), jnp.array([True]))
    np.testing.assert_array_equal(state.get_buffer("c128", 1)[0], 9)
    assert state.nbytes == 3 * spec.state_bytes_per_request


def test_dp_shards_keep_rank_local_addresses_and_global_request_slots():
    spec = DeepseekV4CacheSpec((4, 128), 8, 4)
    kv = DeepseekV4TokenToKVPool(256, 256, 128, spec, mesh(2), dp_size=2)
    state = DeepseekV4CompressStatePool(3, spec, mesh(2), dp_size=2)
    for rank in range(2):
        kv.write(
            "c4",
            0,
            jnp.array([32]),
            jnp.full((1, 8), rank + 1, jnp.bfloat16),
            jnp.array([True]),
            rank,
        )
        state.write(
            "c4",
            0,
            jnp.array([0]),
            jnp.full((1, 8, 32), rank + 1, jnp.float32),
            jnp.array([True]),
            rank,
        )
    np.testing.assert_array_equal(kv.get_buffer("c4", 0)[1, 0], 1)
    np.testing.assert_array_equal(kv.get_buffer("c4", 0)[3, 0], 2)
    np.testing.assert_array_equal(state.get_buffer("c4", 0)[0], 1)
    np.testing.assert_array_equal(state.get_buffer("c4", 0)[4], 2)
    assert kv.get_buffer("c4", 0).sharding.spec[0] == "data"
    assert len(kv.get_buffer("c4", 0).addressable_shards) == 2
    assert state.nbytes // 2 == 4 * spec.state_bytes_per_request


@pytest.mark.parametrize(
    "p,size,dtype", [(1, 128, jnp.bfloat16), (128, 127, jnp.bfloat16), (128, 128, jnp.float32)]
)
def test_invalid_resource_configuration(p, size, dtype):
    with pytest.raises(ValueError):
        DeepseekV4TokenToKVPool(size, 128, p, DeepseekV4CacheSpec((4,)), mesh(), dtype=dtype)


@pytest.mark.parametrize("rank", [-1, 1])
def test_invalid_rank_cannot_write_another_shard(rank):
    spec = DeepseekV4CacheSpec((4,), 8, 4)
    kv = DeepseekV4TokenToKVPool(128, 128, 128, spec, mesh())
    state = DeepseekV4CompressStatePool(1, spec, mesh())
    kv.write("swa", 0, jnp.array([128]), jnp.full((1, 8), 7, jnp.bfloat16), jnp.array([True]), rank)
    state.write(
        "c4", 0, jnp.array([0]), jnp.full((1, 8, 32), 7, jnp.float32), jnp.array([True]), rank
    )
    np.testing.assert_array_equal(kv.get_buffer("swa", 0), 0)
    np.testing.assert_array_equal(state.get_buffer("c4", 0)[..., :16], 0)


def test_explicit_production_mesh_jit_scatter_and_reset():
    if jax.device_count() < 4:
        pytest.skip("requires four CPU devices")
    explicit_mesh = Mesh(
        np.array(jax.devices()[:4]).reshape(2, 2),
        ("data", "tensor"),
        axis_types=(jax.sharding.AxisType.Explicit,) * 2,
    )
    spec = DeepseekV4CacheSpec((4, 128), 8, 4)
    with jax.set_mesh(explicit_mesh):
        pools = MemoryPools(
            token_to_kv_pool=DeepseekV4TokenToKVPool(256, 256, 128, spec, explicit_mesh, 2),
            compressor_state_pool=DeepseekV4CompressStatePool(2, spec, explicit_mesh, 2),
        )

        @jax.jit
        def step(pools):
            pools.token_to_kv_pool.write(
                "c4", 0, jnp.array([32]), jnp.full((1, 8), 5, jnp.bfloat16), jnp.array([True]), 1
            )
            pools.compressor_state_pool.write(
                "c4", 0, jnp.array([0]), jnp.full((1, 8, 32), 7, jnp.float32), jnp.array([True]), 1
            )
            pools.compressor_state_pool.reset(jnp.array([1]), jnp.array([True]), 1)
            return {
                "token_to_kv_pool": pools.token_to_kv_pool.buffers,
                "compressor_state_pool": pools.compressor_state_pool.buffers,
            }

        pools.replace_all(step(pools))
    # Transfer to NumPy to inspect global contents without changing their sharding.
    kv = np.asarray(pools.token_to_kv_pool.get_buffer("c4", 0))
    state = np.asarray(pools.compressor_state_pool.get_buffer("c4", 0))
    np.testing.assert_array_equal(kv[1, 0], 0)
    np.testing.assert_array_equal(kv[3, 0], 5)
    np.testing.assert_array_equal(state[0, :, :16], 0)
    np.testing.assert_array_equal(state[3], 7)
    assert np.isneginf(state[4, :, 16:]).all()
