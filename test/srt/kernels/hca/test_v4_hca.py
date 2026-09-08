"""Real HCA execution through C1 pools, checked against independent dense math."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.layers.attention.deepseek_v4_hca_backend import DeepseekV4HCABackend
from sgl_jax.srt.mem_cache.deepseek_v4.allocator import DeepseekV4TokenToKVPoolAllocator
from sgl_jax.srt.mem_cache.deepseek_v4.pool import (
    DeepseekV4CacheSpec,
    DeepseekV4TokenToKVPool,
)
from sgl_jax.srt.mem_cache.deepseek_v4.state import DeepseekV4CompressStatePool
from sgl_jax.srt.mem_cache.memory_pool import MemoryPools, ReqToTokenPool
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode

from .test_hca import HEAD_DIM, HEADS, SOFTMAX_SCALE, _check, _stream, _weights

pytestmark = pytest.mark.skipif(
    jax.default_backend() != "tpu", reason="requires real HCA Mosaic lowering"
)


@pytest.mark.parametrize("page_size", [1, 2])
def test_c1_small_page_dma(page_size):
    """Compile the native tiny-page DMA in isolation before full HCA chains."""
    import jax.experimental.pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    from sgl_jax.srt.kernels.hca.attention import _gather_small_compressed_pages

    cache = (
        jnp.arange(7 * page_size * 512, dtype=jnp.float32)
        .reshape(7, 1, page_size, 512)
        .astype(jnp.bfloat16)
    )
    pages = jnp.asarray([4, 1, 5], jnp.int32)

    def kernel(indices, source, output, scratch, semaphore):
        _gather_small_compressed_pages(
            source,
            indices,
            jnp.int32(0),
            jnp.int32(3 * page_size - 1),
            jnp.int32(0),
            output,
            scratch,
            semaphore,
            page_size=page_size,
            compressed_tile=128,
        )

    actual = pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            in_specs=(pl.BlockSpec(memory_space=pltpu.HBM),),
            out_specs=pl.BlockSpec((128, 512), lambda *_: (0, 0)),
            scratch_shapes=(pltpu.VMEM((page_size, 512), jnp.bfloat16), pltpu.SemaphoreType.DMA),
        ),
        out_shape=jax.ShapeDtypeStruct((128, 512), jnp.bfloat16),
        interpret=jax.default_backend() != "tpu",
    )(pages, cache)
    transferred_pages = (3 * page_size - 1 + page_size - 1) // page_size
    expected = np.zeros((128, 512), np.float32)
    rows = np.asarray(cache, np.float32)[np.asarray(pages)[:transferred_pages]].reshape(-1, 512)
    expected[: rows.shape[0]] = rows
    np.testing.assert_array_equal(np.asarray(actual, np.float32), expected)


class C1Driver:
    def __init__(self, batch, page_size, weights, *, dp=1, tp=1):
        if jax.device_count() < dp * tp:
            pytest.skip("requires DP=2/TP=2 TPU devices")
        self.dp, self.batch = dp, batch
        self.mesh = jax.sharding.Mesh(
            np.asarray(jax.devices()[: dp * tp]).reshape(dp, tp),
            ("data", "tensor"),
            axis_types=(jax.sharding.AxisType.Explicit,) * 2,
        )
        with jax.set_mesh(self.mesh):
            spec = DeepseekV4CacheSpec((128,))
            self.kv = DeepseekV4TokenToKVPool(
                2048 * batch, 2048 * batch, page_size, spec, self.mesh, dp
            )
            self.state = DeepseekV4CompressStatePool(batch, spec, self.mesh, dp)
            self.requests = ReqToTokenPool(batch, 1024)
            self.allocator = DeepseekV4TokenToKVPoolAllocator(self.kv)
            self.backend = DeepseekV4HCABackend(
                mesh=self.mesh, page_size=page_size, max_context_len=1024, request_capacity=batch
            )
            self.owners = [SimpleNamespace(req_pool_idx=None) for _ in range(batch)]
            # Reverse slot assignment so global slot 0 belongs to the last DP
            # rank, never to an inferred slot % slots_per_rank owner.
            self.slots = np.asarray(self.requests.alloc(self.owners[::-1])[::-1], np.int32)
            self.lengths = np.zeros(batch, np.int32)
            self.weights = {
                name: self.put(
                    value,
                    P("tensor") if name == "sink" else P(*([None] * value.ndim)),
                    jnp.float32 if name in ("ape", "cos", "sin", "sink") else jnp.bfloat16,
                )
                for name, value in weights.items()
            }

    def put(self, value, spec, dtype):
        return jax.device_put(np.asarray(value).astype(dtype), NamedSharding(self.mesh, spec))

    def step(self, stream, query_lengths, mode, *, request_padding=0):
        query_lengths = np.asarray(query_lengths, np.int32)
        prefixes = self.lengths.copy()
        ends = prefixes + query_lengths
        plan = [(r, range(int(prefixes[r]), int(ends[r]))) for r in range(self.batch)]
        per_rank = self.batch // self.dp
        counts = query_lengths.reshape(self.dp, per_rank).sum(axis=1)
        capacity = int(counts.max())
        arrays = {name: [] for name in ("hidden", "q", "kv", "positions")}
        for rank in range(self.dp):
            selected = range(rank * per_rank, (rank + 1) * per_rank)
            positions = [np.asarray(list(plan[r][1]), np.int32) for r in selected]
            for r in selected:
                tail = (
                    -1
                    if prefixes[r] == 0
                    else int(self.requests.req_to_token[self.slots[r], prefixes[r] - 1])
                )
                loc = self.allocator.alloc_extend(
                    np.array([prefixes[r]]),
                    np.array([ends[r]]),
                    np.array([tail]),
                    int(query_lengths[r]),
                    dp_rank=rank,
                )
                assert loc is not None
                self.requests.req_to_token[self.slots[r], prefixes[r] : ends[r]] = loc
            for name in arrays:
                local = (
                    np.concatenate(positions)
                    if name == "positions"
                    else np.concatenate([stream[name][r, prefixes[r] : ends[r]] for r in selected])
                )
                padding = [(0, capacity - int(counts[rank]))] + [(0, 0)] * (local.ndim - 1)
                arrays[name].append(np.pad(local, padding))
        arrays = {name: np.concatenate(parts) for name, parts in arrays.items()}
        with jax.set_mesh(self.mesh):

            def pad_requests(values, fill=0):
                return np.pad(
                    np.asarray(values).reshape(self.dp, per_rank),
                    ((0, 0), (0, request_padding)),
                    constant_values=fill,
                ).reshape(-1)

            worker = SimpleNamespace(
                forward_mode=mode,
                req_pool_indices=pad_requests(self.slots, -1),
                seq_lens=pad_requests(ends),
                positions=arrays["positions"],
                extend_seq_lens=pad_requests(query_lengths),
                extend_prefix_lens=pad_requests(prefixes),
                dp_size=self.dp,
                per_dp_bs_size=per_rank + request_padding,
            )
            self.backend.forward_metadata = self.backend.get_forward_metadata(
                worker, request_pool=self.requests, allocator=self.allocator
            )
            q = self.put(arrays["q"], P("data", "tensor", None), jnp.bfloat16)
            kv = self.put(arrays["kv"], P("data", None), jnp.bfloat16)
            output, update = self.backend(
                q,
                kv,
                kv,
                SimpleNamespace(layer_id=0, scaling=SOFTMAX_SCALE),
                SimpleNamespace(
                    forward_mode=mode, positions=self.put(arrays["positions"], P("data"), jnp.int32)
                ),
                self.kv,
                compressor_state_pool=self.state,
                compressor_input=self.put(arrays["hidden"], P("data", None), jnp.bfloat16),
                wkv=self.weights["wkv"],
                wgate=self.weights["wgate"],
                ape=self.weights["ape"],
                norm_weight=self.weights["norm"],
                cos=self.weights["cos"],
                sin=self.weights["sin"],
                attention_sink=self.weights["sink"],
            )
            jax.block_until_ready((output, update))
            MemoryPools(token_to_kv_pool=self.kv, compressor_state_pool=self.state).replace_all(
                self.backend.pack_pool_updates({0: update}, self.kv, self.state)
            )
        self.lengths = ends
        output = np.asarray(output, np.float32).reshape(self.dp, capacity, HEADS, HEAD_DIM)
        for rank in range(self.dp):
            np.testing.assert_array_equal(output[rank, counts[rank] :], 0)
        return np.concatenate([output[r, : counts[r]] for r in range(self.dp)]), plan

    def recycle(self):
        for r, owner in enumerate(self.owners):
            rank = r // (self.batch // self.dp)
            self.allocator.free(
                self.requests.req_to_token[self.slots[r], : self.lengths[r]].copy(), dp_rank=rank
            )
            self.requests.free(owner)
        self.slots = np.asarray(self.requests.alloc(self.owners), np.int32)
        self.lengths.fill(0)
        # Deliberately retain old KV and compressor values. The next fresh
        # metadata event must initialize state before numerical consumption.


@pytest.mark.parametrize("page_size", [128, 256])
def test_c1_hca_boundary_chain_and_slot_reuse(page_size):
    weights = _weights(20260908)
    stream = _stream(1, 386, 81)
    driver = C1Driver(1, page_size, weights)
    output, plan = driver.step(stream, [382], ForwardMode.EXTEND)
    _check(output, stream, plan, weights)
    for _ in range(4):
        output, plan = driver.step(stream, [1], ForwardMode.DECODE)
        _check(output, stream, plan, weights)
    driver.recycle()
    stream = _stream(1, 129, 82)
    output, plan = driver.step(stream, [129], ForwardMode.EXTEND)
    _check(output, stream, plan, weights)


@pytest.mark.parametrize("page_size", [128, 256])
def test_c1_hca_ragged_chunks(page_size):
    weights = _weights(20260909)
    stream = _stream(2, 386, 83)
    driver = C1Driver(2, page_size, weights)
    for query_lengths in ([130, 257], [256, 129]):
        output, plan = driver.step(stream, query_lengths, ForwardMode.EXTEND)
        _check(output, stream, plan, weights)


def test_c1_hca_dp2_tp2_padded_chunks_and_decode():
    weights = _weights(20260910)
    stream = _stream(2, 388, 84)
    driver = C1Driver(2, 128, weights, dp=2, tp=2)
    for query_lengths in ([129, 257], [257, 129]):
        output, plan = driver.step(stream, query_lengths, ForwardMode.EXTEND)
        _check(output, stream, plan, weights)
    output, plan = driver.step(stream, [1, 1], ForwardMode.DECODE)
    _check(output, stream, plan, weights)


def test_c1_hca_request_padding_preserves_continuation_state():
    weights = _weights(20260911)
    stream = _stream(1, 128, 85)
    driver = C1Driver(1, 128, weights)
    output, plan = driver.step(stream, [127], ForwardMode.EXTEND, request_padding=2)
    _check(output, stream, plan, weights)
    expected_contents = stream["hidden"][0, :127] @ weights["wkv"].T
    np.testing.assert_allclose(
        np.asarray(driver.state.get_buffer("c128", 0), np.float32)[0, :127, :512],
        expected_contents,
        rtol=2e-2,
        atol=1e-2,
    )
    output, plan = driver.step(stream, [1], ForwardMode.DECODE)
    _check(output, stream, plan, weights)
