"""CPU-verifiable C1 ownership, metadata and sharded update contracts for HCA."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.kernels.hca.tuned_block_sizes import get_hca_kernel_schedule
from sgl_jax.srt.layers.attention import deepseek_v4_hca_backend as adapter
from sgl_jax.srt.layers.attention.deepseek_v4_hca_backend import DeepseekV4HCABackend
from sgl_jax.srt.mem_cache.deepseek_v4_allocator import DeepseekV4TokenToKVPoolAllocator
from sgl_jax.srt.mem_cache.deepseek_v4_compress_state import DeepseekV4CompressStatePool
from sgl_jax.srt.mem_cache.deepseek_v4_memory_pool import (
    DeepseekV4CacheSpec,
    DeepseekV4TokenToKVPool,
)
from sgl_jax.srt.mem_cache.memory_pool import MemoryPools, ReqToTokenPool
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode


@pytest.fixture(autouse=True)
def host_schedule(monkeypatch):
    # Metadata arithmetic is platform-independent; CPU has no production TPU
    # schedule. Exercise the actual v7x selector without pretending to run TPU.
    real = get_hca_kernel_schedule
    monkeypatch.setattr(adapter, "get_hca_kernel_schedule", lambda _, **kw: real("TPU7x", **kw))


@pytest.fixture
def runtime():
    def create(page_size=128, dp=1, tp=1):
        if jax.device_count() < dp * tp:
            pytest.skip("requires four virtual CPU devices or TPU devices")
        mesh = jax.sharding.Mesh(
            np.asarray(jax.devices()[: dp * tp]).reshape(dp, tp),
            ("data", "tensor"),
            axis_types=(jax.sharding.AxisType.Explicit,) * 2,
        )
        spec = DeepseekV4CacheSpec((0, 4, 128))
        with jax.set_mesh(mesh):
            kv = DeepseekV4TokenToKVPool(2048 * dp, 2048 * dp, page_size, spec, mesh, dp)
            state = DeepseekV4CompressStatePool(4, spec, mesh, dp)
            requests = ReqToTokenPool(4, 1024)
            allocator = DeepseekV4TokenToKVPoolAllocator(kv)
            backend = DeepseekV4HCABackend(
                mesh=mesh, page_size=page_size, max_context_len=1024, request_capacity=4
            )
        return SimpleNamespace(
            mesh=mesh, kv=kv, state=state, requests=requests, allocator=allocator, backend=backend
        )

    return create


def metadata(rt, worker):
    return rt.backend.get_forward_metadata(worker, request_pool=rt.requests, allocator=rt.allocator)


def reserve(rt, length, *, rank=0):
    request = SimpleNamespace(req_pool_idx=None)
    slot = rt.requests.alloc([request])[0]
    locations = rt.allocator.alloc_extend(
        np.array([0]), np.array([length]), np.array([-1]), length, dp_rank=rank
    )
    assert locations is not None
    rt.requests.req_to_token[slot, :length] = locations
    return slot, locations


def batch(mode, slots, lengths, positions, q_lens=None, dp=1):
    return SimpleNamespace(
        forward_mode=mode,
        req_pool_indices=np.asarray(slots, np.int32),
        seq_lens=np.asarray(lengths, np.int32),
        positions=np.asarray(positions, np.int32),
        extend_seq_lens=None if q_lens is None else np.asarray(q_lens, np.int32),
        extend_prefix_lens=None,
        dp_size=dp,
        per_dp_bs_size=len(slots) // dp,
    )


@pytest.mark.parametrize("page_size", [128, 256])
def test_c1_pages_keep_original_token_units_and_request_slot_zero(runtime, page_size):
    rt = runtime(page_size)
    slot, loc = reserve(rt, 382)
    assert slot == 0
    with jax.set_mesh(rt.mesh):
        md = metadata(rt, batch(ForwardMode.EXTEND, [slot], [382], range(382), [382]))
    kernel = md.kernel
    np.testing.assert_array_equal(kernel.state_slots, 0)
    np.testing.assert_array_equal(md.state_init_slots, [0])
    np.testing.assert_array_equal(kernel.compressed_kv_lens, [2])
    pages = loc[::page_size] // page_size
    n = (2 + page_size // 128 - 1) // (page_size // 128)
    np.testing.assert_array_equal(np.asarray(kernel.compressed_page_indices)[:n], pages[:n])
    assert kernel.compressed_cu_kv_lens[1] == n * (page_size // 128)
    assert kernel.window_cu_kv_lens[1] == len(pages) * page_size
    assert md.use_uniform_prefill_fast_path


def test_metadata_pads_each_dp_section_and_keeps_slot_zero_on_rank_one(runtime):
    rt = runtime(dp=2, tp=2)
    slot, _ = reserve(rt, 128, rank=1)
    with jax.set_mesh(rt.mesh):
        md = metadata(
            rt, batch(ForwardMode.DECODE, [-1, -1, slot, -1], [0, 0, 128, 0], [0, 0, 127, 0], dp=2)
        )
    np.testing.assert_array_equal(
        np.asarray(md.kernel.cu_q_lens).reshape(2, 3), [[0, 0, 0], [0, 1, 1]]
    )
    np.testing.assert_array_equal(np.asarray(md.kernel.state_slots).reshape(2, 2), [[4, 4], [0, 4]])
    np.testing.assert_array_equal(
        np.asarray(md.kernel.valid_token_mask).reshape(2, 2), [[False, False], [True, False]]
    )
    np.testing.assert_array_equal(
        np.asarray(md.kernel.boundary_token_indices).reshape(2, -1), [[2, 2], [0, 2]]
    )
    assert not md.use_uniform_prefill_fast_path


def test_same_decode_shape_survives_compression_boundary(runtime):
    rt = runtime()
    slot, _ = reserve(rt, 129)
    signatures = []
    for length in (127, 128, 129):
        with jax.set_mesh(rt.mesh):
            md = metadata(rt, batch(ForwardMode.DECODE, [slot, -1], [length, 0], [length - 1, 0]))
        signatures.append(
            (jax.tree.structure(md), [(x.shape, x.dtype) for x in jax.tree.leaves(md)])
        )
        assert int(md.kernel.compressed_kv_lens[0]) == length // 128
    assert signatures[0] == signatures[1] == signatures[2]


def test_padded_uniform_lengths_use_ragged_path(runtime):
    rt = runtime()
    s0, _ = reserve(rt, 130)
    s1, _ = reserve(rt, 130)
    positions = np.pad(np.tile(np.arange(130), 2), (0, 252))
    with jax.set_mesh(rt.mesh):
        md = metadata(rt, batch(ForwardMode.EXTEND, [s0, s1], [130, 130], positions, [130, 130]))
    assert not md.use_uniform_prefill_fast_path
    assert np.asarray(md.kernel.valid_token_mask).sum() == 260
    np.testing.assert_array_equal(np.asarray(md.kernel.state_slots)[260:], 4)


def test_released_old_swa_is_allowed_but_live_prefix_is_required(runtime):
    rt = runtime()
    slot, loc = reserve(rt, 513)
    rt.allocator.free_swa(loc[:256])
    worker = batch(ForwardMode.DECODE, [slot], [513], [512])
    with jax.set_mesh(rt.mesh):
        md = metadata(rt, worker)
        np.testing.assert_array_equal(np.asarray(md.kernel.window_page_indices)[:2], 0)
        rt.allocator.free_swa(loc[256:384])
        # Position 383 is already outside the 128-token window of query 512.
        metadata(rt, worker)
        rt.allocator.free_swa(loc[384:512])
        with pytest.raises(ValueError, match="SWA pages required"):
            metadata(rt, worker)


@pytest.mark.parametrize("fault", ["positions", "prefix", "slot", "history"])
def test_invalid_host_contracts_fail_before_kernel(runtime, fault):
    rt = runtime()
    slot, _ = reserve(rt, 128)
    worker = batch(ForwardMode.EXTEND, [slot], [128], [127], [1])
    if fault == "positions":
        worker.positions[0] = 126
    elif fault == "prefix":
        worker.extend_prefix_lens = np.array([0])
    elif fault == "slot":
        worker.req_pool_indices[0] = -1
    else:
        rt.requests.req_to_token[slot, 4] = 0
    with jax.set_mesh(rt.mesh), pytest.raises(ValueError):
        metadata(rt, worker)


@pytest.mark.parametrize("kind", ["TPU v6e", "TPU7x", "TPU v7x"])
@pytest.mark.parametrize("page", [1, 2, 128])
def test_platform_schedule_accepts_native_and_standalone_pages(kind, page):
    schedule = get_hca_kernel_schedule(
        kind, page_size=page, max_compressed_entries=8, local_heads=64, head_dim=512
    )
    assert schedule.compressed_tile % page == 0
    assert schedule.query_compute_block_size > 0


def test_sharded_views_reset_reused_slot_and_preserve_other_families(runtime, monkeypatch):
    """This probe validates C1 ownership/sharding, not Pallas numerical math."""
    from sgl_jax.srt.layers.attention import hca_backend as legacy

    rt = runtime(dp=2, tp=2)
    slot, _ = reserve(rt, 1, rank=1)

    def probe(x, q, new_kv, state, window, compressed, *args, **kwargs):
        md = args[-1]
        valid = md.valid_token_mask
        selected = state[md.state_slots, 0, 0, 0]
        output = jnp.broadcast_to(jnp.where(valid, selected, 0)[:, None, None], q.shape).astype(
            q.dtype
        )
        destinations = jnp.where(valid, md.state_slots, state.shape[0])
        state = state.at[destinations, 0, 0, 0].set(
            jnp.full(valid.shape, 101, state.dtype), mode="drop"
        )
        return output, state, window, compressed

    monkeypatch.setattr(legacy, "hca_step", probe)
    monkeypatch.setattr(legacy.HCABackend, "_check_constants", lambda *args, **kw: None)
    with jax.set_mesh(rt.mesh):
        rt.state.buffers["c128"][0].block_until_ready()
        old = rt.state.get_buffer("c128", 2)
        # Slot 0 on rank 0 and rank 1 have different stale content.
        changed = (
            old.at[0, :, :512]
            .set(7, out_sharding=P("data", None, None))
            .at[5, :, :512]
            .set(9, out_sharding=P("data", None, None))
        )
        rt.state.buffers = {**rt.state.buffers, "c128": (changed,)}
        rt.backend.forward_metadata = metadata(
            rt, batch(ForwardMode.EXTEND, [-1, slot], [0, 1], [0, 0], [0, 1], dp=2)
        )

        def put(shape, spec):
            return jax.device_put(jnp.zeros(shape, jnp.bfloat16), NamedSharding(rt.mesh, spec))

        q = put((2, 64, 512), P("data", "tensor", None))
        kv = put((2, 512), P("data", None))
        output, update = rt.backend(
            q,
            kv,
            kv,
            SimpleNamespace(layer_id=2, scaling=None),
            SimpleNamespace(
                forward_mode=ForwardMode.EXTEND, positions=put((2,), P("data")).astype(jnp.int32)
            ),
            rt.kv,
            compressor_state_pool=rt.state,
            compressor_input=put((2, 4096), P("data", None)),
            wkv=put((512, 4096), P(None, None)),
            wgate=put((512, 4096), P(None, None)),
            ape=put((128, 512), P(None, None)),
            norm_weight=put((512,), P(None)),
            cos=put((1024, 32), P(None, None)),
            sin=put((1024, 32), P(None, None)),
            attention_sink=put((64,), P("tensor")),
        )
        np.testing.assert_array_equal(output, 0)  # reset before consumption
        np.testing.assert_array_equal(np.asarray(update[0])[0, :, :512], 7)
        assert np.asarray(update[0])[5, 0, 0] == 101
        np.testing.assert_array_equal(np.asarray(update[0])[5, 1:, :512], 0)
        untouched = rt.kv.buffers["c4"][0]
        replacement = rt.backend.pack_pool_updates({2: update}, rt.kv, rt.state)
        MemoryPools(token_to_kv_pool=rt.kv, compressor_state_pool=rt.state).replace_all(replacement)
        assert rt.kv.buffers["c4"][0] is untouched
        assert rt.state.buffers["c128"][0] is update[0]
