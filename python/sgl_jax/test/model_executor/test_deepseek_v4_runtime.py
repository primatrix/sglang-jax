"""C3 transport, actual ModelRunner donation, and bucket reuse.

The CPU consumer tests device dependencies with C1 buffers. The TPU test below
uses the real C128 HCA kernels through the same ModelRunner entry point.
"""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from jax.sharding import NamedSharding, PartitionSpec as P

from sgl_jax.srt.kernels.hca.tuned_block_sizes import get_hca_kernel_schedule
from sgl_jax.srt.layers.attention import deepseek_v4_hca_backend as hca_adapter
from sgl_jax.srt.layers.attention.dsv4.runtime import DeepseekV4RuntimeBackend
from sgl_jax.srt.mem_cache.deepseek_v4_allocator import DeepseekV4TokenToKVPoolAllocator
from sgl_jax.srt.mem_cache.deepseek_v4_compress_state import DeepseekV4CompressStatePool
from sgl_jax.srt.mem_cache.deepseek_v4_memory_pool import (
    DeepseekV4CacheSpec,
    DeepseekV4TokenToKVPool,
    scatter_sharding,
)
from sgl_jax.srt.mem_cache.memory_pool import MemoryPools, ReqToTokenPool
from sgl_jax.srt.model_executor.compilation_manager import CompilationManager
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sgl_jax.srt.model_executor.model_runner import ModelRunner
from sgl_jax.srt.server_args import ServerArgs


@pytest.fixture(autouse=True)
def cpu_schedule(monkeypatch):
    if jax.default_backend() == "cpu":
        monkeypatch.setattr(
            hca_adapter,
            "get_hca_kernel_schedule",
            lambda _, **kwargs: get_hca_kernel_schedule("TPU7x", **kwargs),
        )


class Harness:
    def __init__(self, page_size=128, dp=1, tp=1, spec=None):
        if jax.device_count() < dp * tp:
            pytest.skip("requires four devices for DP=2/TP=2")
        self.dp, self.tp, self.page_size = dp, tp, page_size
        self.mesh = jax.sharding.Mesh(
            np.asarray(jax.devices()[: dp * tp]).reshape(dp, tp),
            ("data", "tensor"),
            axis_types=(jax.sharding.AxisType.Explicit,) * 2,
        )
        self.spec = spec or DeepseekV4CacheSpec((0, 4, 128))
        self.runner = object.__new__(ModelRunner)
        r = self.runner
        r.mesh, r.tp_size, r.page_size = self.mesh, tp, page_size
        r.server_args = ServerArgs(
            model_path="dummy",
            disable_overlap_schedule=True,
            disable_radix_cache=True,
            page_size=page_size,
            dp_size=dp,
        )
        r.model_config = SimpleNamespace(
            context_len=1024,
            vocab_size=32,
            is_embedding=False,
            hf_config=SimpleNamespace(model_type="deepseek_v4", architectures=[]),
        )
        r.attn_backend = r._get_attention_backend()
        with jax.set_mesh(self.mesh):
            r.req_to_token_pool = ReqToTokenPool(4, 1024)
            r.token_to_kv_pool = DeepseekV4TokenToKVPool(
                4096 * dp, 4096 * dp, page_size, self.spec, self.mesh, dp
            )
            r.memory_pools = MemoryPools(
                token_to_kv_pool=r.token_to_kv_pool,
                compressor_state_pool=DeepseekV4CompressStatePool(4, self.spec, self.mesh, dp),
            )
            r.token_to_kv_pool_allocator = DeepseekV4TokenToKVPoolAllocator(r.token_to_kv_pool)
            r.bind_attention_resources()
        self.compiler = CompilationManager(
            r.server_args,
            2 * dp,
            256 * dp,
            dp,
            tp,
            page_size,
            1024,
            32,
        )
        self.owners = [SimpleNamespace(req_pool_idx=None) for _ in range(dp)]
        # Reverse allocation deliberately places global slot zero on the last rank.
        self.slots = np.asarray(r.req_to_token_pool.alloc(self.owners[::-1])[::-1], np.int32)
        self.lengths = np.zeros(dp, np.int32)

    def batch(self, queries, mode=ForwardMode.EXTEND, capacity=256, *, slots=None):
        r = self.runner
        queries = np.asarray(queries, np.int32)
        ends = self.lengths + queries
        slots = self.slots if slots is None else np.asarray(slots, np.int32)
        bs = self.dp * 2  # one live + one padded request per DP rank
        batch = self.compiler._make_dummy_batch(
            bs,
            capacity * self.dp,
            mode,
            bs * 1024,
            dp_size=self.dp,
            per_dp_bs_size=2,
        )
        batch.seq_lens = np.zeros(bs, np.int32)
        batch.req_pool_indices = np.full(bs, 4, np.int32)
        batch.positions.fill(0)
        batch.out_cache_loc.fill(-1)
        if mode == ForwardMode.EXTEND:
            batch.extend_seq_lens = np.zeros(bs, np.int32)
            batch.extend_prefix_lens = np.zeros(bs, np.int32)
        for rank, n in enumerate(queries):
            pre, end, slot = self.lengths[rank], ends[rank], slots[rank]
            tail = -1 if pre == 0 else r.req_to_token_pool.req_to_token[slot, pre - 1]
            loc = r.token_to_kv_pool_allocator.alloc_extend(
                np.array([pre]),
                np.array([end]),
                np.array([tail]),
                int(n),
                dp_rank=rank,
            )
            assert loc is not None
            r.req_to_token_pool.req_to_token[slot, pre:end] = loc
            i = 2 * rank
            batch.seq_lens[i] = end
            batch.req_pool_indices[i] = slot
            batch.positions[rank * capacity : rank * capacity + n] = np.arange(pre, end)
            batch.out_cache_loc[rank * capacity : rank * capacity + n] = loc
            if mode == ForwardMode.EXTEND:
                batch.extend_seq_lens[i], batch.extend_prefix_lens[i] = n, pre
        self.lengths = ends
        batch.real_bs = self.dp
        batch.real_bs_per_dp = [1] * self.dp
        batch.real_input_ids_len = int(queries.sum())
        return batch

    def forward_batch(self, batch):
        self.runner.attn_backend.forward_metadata = self.runner.get_attention_metadata(batch)
        return ForwardBatch.init_new(batch, self.runner)

    def install(self, model):
        r = self.runner
        r.model, r.sampler = model, nnx.Module()
        r._sampler_base_rng = jax.random.PRNGKey(0)
        r._sampler_step = 0
        r.use_sort_for_toppk_minp = False
        r.initialize_jit()


def make_probe(traces, broken=None):
    class Probe(nnx.Module):
        def __call__(self, batch, pools, logits_metadata):
            traces.append((batch.forward_mode, batch.input_ids.shape))
            md = batch.deepseek_v4_metadata
            kv, state = pools.token_to_kv_pool, pools.compressor_state_pool
            dp = state.dp_size
            ranks = jnp.repeat(jnp.arange(dp, dtype=jnp.int32), md.q_lens.shape[0] // dp)
            index = jnp.where(
                md.request_valid_mask,
                ranks * state.slots_per_rank + md.request_slots,
                state.get_buffer("c128", 2).shape[0],
            )
            old = state.get_buffer("c128", 2)
            previous = old.at[index, 0, 0].get(
                mode="fill", fill_value=0, out_sharding=NamedSharding(state.mesh, P("data"))
            )
            value = jnp.where(md.state_init_mask, 0, previous) + md.q_lens
            new = old.at[index, 0, 0].set(
                value, mode="drop", out_sharding=scatter_sharding(state.mesh, 3)
            )
            # Update every owner's arrays, preserving their shape/dtype. The
            # second step consumes state produced by the first donated call.
            kv_updates = jax.tree.map(
                lambda x: x + jnp.asarray(jnp.any(md.valid_token_mask), x.dtype), kv.buffers
            )
            state_updates = {**state.buffers, "c128": (new,)}
            updates = {"token_to_kv_pool": kv_updates, "compressor_state_pool": state_updates}
            if broken == "owner":
                updates.pop("compressor_state_pool")
            if broken == "family":
                updates["compressor_state_pool"] = {"c128": (new,)}
            return value, updates, None, None

    return Probe()


@pytest.mark.parametrize("page_size", [128, 256])
def test_forward_batch_transport_and_dp_local_indices(page_size):
    h = Harness(page_size, dp=2, tp=2)
    with jax.set_mesh(h.mesh):
        batch = h.batch([129, 3])
        fb = h.forward_batch(batch)
        flat, tree = jax.tree.flatten(fb)
        clone = jax.tree.unflatten(tree, flat)
        md = clone.deepseek_v4_metadata
        np.testing.assert_array_equal(md.q_lens, [129, 0, 3, 0])
        np.testing.assert_array_equal(md.cu_q_lens, [0, 129, 129, 0, 3, 3])
        np.testing.assert_array_equal(md.request_slots, [h.slots[0], 4, 0, 4])
        np.testing.assert_array_equal(md.state_init_mask, [True, False, True, False])
        np.testing.assert_array_equal(md.valid_token_mask.reshape(2, -1).sum(axis=1), [129, 3])
        for leaf in jax.tree.leaves(md):
            assert isinstance(leaf, jax.Array)
            assert leaf.dtype in (jnp.int32, jnp.bool_)
            assert leaf.sharding.spec == P("data")
        boundary = np.asarray(md.c128.boundary_token_indices).reshape(2, -1)
        mask = np.asarray(md.c128.boundary_valid_mask).reshape(2, -1)
        np.testing.assert_array_equal(boundary[0, mask[0]], [127])
        assert np.all(boundary[~mask] == 256)
        # No host NumPy page ledger is part of the transmitted tree.
        assert all(not isinstance(x, np.ndarray) for x in flat)


def test_binding_required_and_ordinary_backend_unchanged():
    h = Harness()
    r = h.runner
    r.attn_backend = r._get_attention_backend()
    assert isinstance(r.attn_backend, DeepseekV4RuntimeBackend)
    with pytest.raises(RuntimeError, match="bound"):
        r.get_attention_metadata(h.batch([1]))
    r.bind_attention_resources()
    sentinel = object()
    r.attn_backend = SimpleNamespace(get_forward_metadata=lambda b: sentinel)
    assert r.get_attention_metadata(None) is sentinel


@pytest.mark.parametrize("bad", ["address", "mixed", "prefix"])
def test_invalid_worker_contract_rejected_before_dispatch(bad):
    h = Harness()
    batch = h.batch([3])
    if bad == "address":
        batch.out_cache_loc[0] += 1
    elif bad == "mixed":
        batch.forward_mode = ForwardMode.MIXED
    else:
        batch.extend_prefix_lens[0] += 1
    with pytest.raises(ValueError):
        h.forward_batch(batch)


def test_same_bucket_is_dynamic_and_different_bucket_changes_structure():
    h = Harness()
    with jax.set_mesh(h.mesh):
        first = jax.tree.map(lambda x: x, h.forward_batch(h.batch([127])))
        second = h.forward_batch(h.batch([2]))

        def signature(fb):
            return jax.tree.structure(fb), [
                (np.shape(x), getattr(x, "dtype", type(x))) for x in jax.tree.leaves(fb)
            ]

        assert signature(first) == signature(second)
        bigger = h.forward_batch(h.batch([3], capacity=384))
        assert signature(first) != signature(bigger)
        assert not first.attn_backend.forward_metadata.use_uniform_prefill_fast_path


@pytest.mark.parametrize("dp,tp", [(1, 1), (2, 2)])
def test_model_runner_donation_reads_new_state_and_reuses_compilation(dp, tp, monkeypatch):
    monkeypatch.setenv("SGLANG_JAX_AOT_DISPATCH", "0")
    h = Harness(dp=dp, tp=tp)
    traces = []
    with jax.set_mesh(h.mesh):
        h.install(make_probe(traces))
        expected = 0
        for n in (127, 2, 5):
            fb = h.forward_batch(h.batch([n] * dp))
            old = h.runner.memory_pools.compressor_state_pool.get_buffer("c128", 2)
            out, _, _ = h.runner._forward(fb, None)
            jax.block_until_ready(out)
            expected += n
            np.testing.assert_array_equal(np.asarray(out).reshape(dp, 2)[:, 0], expected)
            assert old.is_deleted()
        assert len(traces) == 1
        assert np.asarray(h.runner.token_to_kv_pool.get_buffer("swa", 0))[0, 0] == 3
        h.runner._forward(h.forward_batch(h.batch([1] * dp, capacity=384)), None)
        assert len(traces) == 2


@pytest.mark.parametrize("broken", ["owner", "family"])
def test_bad_result_is_rejected_before_donation(broken, monkeypatch):
    monkeypatch.setenv("SGLANG_JAX_AOT_DISPATCH", "0")
    h = Harness()
    with jax.set_mesh(h.mesh):
        h.install(make_probe([], broken))
        old = h.runner.token_to_kv_pool.get_buffer("swa", 0)
        with pytest.raises(ValueError, match="V4"):
            h.runner._forward(h.forward_batch(h.batch([1])), None)
        assert not old.is_deleted()


@pytest.mark.parametrize("mode", [ForwardMode.EXTEND, ForwardMode.DECODE])
def test_dummy_metadata_uses_only_inactive_slots(mode):
    h = Harness(dp=2, tp=2)
    bs, tokens = 4, 512 if mode == ForwardMode.EXTEND else 4
    batch = h.compiler._make_dummy_batch(
        bs,
        tokens,
        mode,
        bs * 1024,
        dp_size=2,
        per_dp_bs_size=2,
    )
    before = h.runner.req_to_token_pool.req_to_token.copy()
    h.runner.prepare_dummy_batch(batch)
    with jax.set_mesh(h.mesh):
        fb = h.forward_batch(batch)
        md = fb.deepseek_v4_metadata
        assert not np.any(md.valid_token_mask)
        assert not np.any(md.request_valid_mask)
        assert not np.any(md.state_init_mask)
        assert np.all(md.request_slots == 4)
        np.testing.assert_array_equal(h.runner.req_to_token_pool.req_to_token, before)


def worker_for(h):
    from queue import SimpleQueue

    from sgl_jax.srt.managers.tp_worker import ModelWorker

    worker = object.__new__(ModelWorker)
    worker.worker = SimpleNamespace(server_args=h.runner.server_args)
    worker.model_runner = h.runner
    worker.model_config = h.runner.model_config
    worker.mesh = h.mesh
    worker._pd_fuse_sample = False
    worker.sync_queue = SimpleQueue()
    worker.dump_topk_ids = lambda *args: None
    h.runner.forward_pass_id = 0
    return worker


def precompile(h, worker, *, embedding_width=None):
    h.compiler.token_buckets = [256 * h.dp]
    h.compiler.bs_buckets = [2 * h.dp]
    h.compiler.cache_loc_buckets = [2 * h.dp * 1024]

    def forward(batch, **kwargs):
        if embedding_width:
            batch.forward_batch.input_embedding = jax.device_put(
                np.zeros((batch.input_ids.size, embedding_width), jnp.bfloat16),
                NamedSharding(h.mesh, P("data", None)),
            )
        return worker.forward_batch_generation(
            batch,
            skip_sample=True,
            sampling_metadata=kwargs["sampling_metadata"],
        )

    h.compiler.precompile_all(forward, h.runner, h.mesh)


def test_precompile_worker_entry_reuses_both_modes(monkeypatch):
    monkeypatch.setenv("SGLANG_JAX_AOT_DISPATCH", "0")
    h = Harness(dp=2, tp=2)
    traces = []
    with jax.set_mesh(h.mesh):
        h.install(make_probe(traces))
        worker = worker_for(h)
        precompile(h, worker)
        assert len(traces) == 2
        for queries, mode, capacity in (
            ([127, 3], ForwardMode.EXTEND, 256),
            ([2, 129], ForwardMode.EXTEND, 256),
            ([1, 1], ForwardMode.DECODE, 2),
        ):
            output, _, _ = worker.forward_batch_generation(
                h.batch(queries, mode, capacity),
                skip_sample=True,
            )
            np.testing.assert_array_equal(np.asarray(output)[::2], h.lengths)
        assert len(traces) == 2


@pytest.mark.skipif(jax.default_backend() != "tpu", reason="requires real HCA Mosaic lowering")
@pytest.mark.parametrize("page_size,dp,tp", [(128, 1, 1), (256, 1, 1), (128, 2, 2)])
def test_real_hca_consumer_precompile_donation_and_reuse(page_size, dp, tp, monkeypatch):
    from test.srt.kernels.hca.test_hca import (
        HEAD_DIM,
        HEADS,
        SOFTMAX_SCALE,
        _check,
        _stream,
        _weights,
    )

    monkeypatch.setenv("SGLANG_JAX_AOT_DISPATCH", "0")
    h = Harness(page_size, dp, tp)
    weights = _weights(20260912)
    traces = []
    width = 4096 + HEADS * HEAD_DIM + HEAD_DIM

    class HCAConsumer(nnx.Module):
        def __init__(self):
            self.weights = nnx.data(
                {
                    name: jax.device_put(
                        np.asarray(value).astype(
                            jnp.float32 if name in ("ape", "cos", "sin", "sink") else jnp.bfloat16
                        ),
                        NamedSharding(h.mesh, P("tensor") if name == "sink" else P()),
                    )
                    for name, value in weights.items()
                }
            )

        def __call__(self, batch, pools, logits_metadata):
            traces.append((batch.forward_mode, batch.input_ids.shape))
            hidden, q, kv = jnp.split(batch.input_embedding, [4096, 4096 + HEADS * HEAD_DIM], 1)
            q = q.reshape(-1, HEADS, HEAD_DIM)
            w = self.weights
            output, updated = batch.attn_backend(
                q,
                kv,
                kv,
                SimpleNamespace(layer_id=2, scaling=SOFTMAX_SCALE),
                batch,
                pools.token_to_kv_pool,
                compressor_state_pool=pools.compressor_state_pool,
                compressor_input=hidden,
                wkv=w["wkv"],
                wgate=w["wgate"],
                ape=w["ape"],
                norm_weight=w["norm"],
                cos=w["cos"],
                sin=w["sin"],
                attention_sink=w["sink"],
            )
            updates = batch.attn_backend.pack_pool_updates(
                {2: updated},
                pools.token_to_kv_pool,
                pools.compressor_state_pool,
            )
            return output, updates, None, None

    def step(stream, queries, mode, capacity):
        prefixes = h.lengths.copy()
        batch = h.batch(queries, mode, capacity)
        embeddings, plan = [], []
        for rank, n in enumerate(queries):
            pre = prefixes[rank]
            chunk = np.concatenate(
                [
                    stream["hidden"][rank, pre : pre + n],
                    stream["q"][rank, pre : pre + n].reshape(n, HEADS * HEAD_DIM),
                    stream["kv"][rank, pre : pre + n],
                ],
                axis=1,
            )
            embeddings.append(np.pad(chunk, ((0, capacity - n), (0, 0))))
            plan.append((rank, range(int(pre), int(pre + n))))
        batch.input_embedding = np.concatenate(embeddings)
        old = h.runner.memory_pools.compressor_state_pool.get_buffer("c128", 2)
        output, _, _ = worker.forward_batch_generation(batch, skip_sample=True)
        output = np.asarray(output, np.float32).reshape(dp, capacity, HEADS, HEAD_DIM)
        assert old.is_deleted()
        for rank, n in enumerate(queries):
            np.testing.assert_array_equal(output[rank, n:], 0)
        _check(
            np.concatenate([output[r, :n] for r, n in enumerate(queries)]), stream, plan, weights
        )

    with jax.set_mesh(h.mesh):
        h.install(HCAConsumer())
        worker = worker_for(h)
        original = jax.tree.map(np.asarray, h.runner.memory_pools)
        precompile(h, worker, embedding_width=width)
        for before, after in zip(jax.tree.leaves(original), jax.tree.leaves(h.runner.memory_pools)):
            np.testing.assert_array_equal(before, after)
        assert len(traces) == 2
        stream = _stream(dp, 388, 91)
        for queries in ([127] * dp, [2] * dp, [128] * dp, [128] * dp):
            step(stream, queries, ForwardMode.EXTEND, 256)
        for _ in range(2):
            step(stream, [1] * dp, ForwardMode.DECODE, 2)
        # Recycle owners and exercise a changed slot without a new compilation.
        r = h.runner
        for rank, owner in enumerate(h.owners):
            r.token_to_kv_pool_allocator.free(
                r.req_to_token_pool.req_to_token[h.slots[rank], : h.lengths[rank]].copy(),
                dp_rank=rank,
            )
            r.req_to_token_pool.free(owner)
        h.slots = np.asarray(r.req_to_token_pool.alloc(h.owners), np.int32)
        h.lengths.fill(0)
        step(_stream(dp, 129, 92), [129] * dp, ForwardMode.EXTEND, 256)
        assert len(traces) == 2
        print(
            {
                "page_size": page_size,
                "dp": dp,
                "tp": tp,
                "model_traces": len(traces),
                "pool_bytes_global": sum(p.nbytes for p in r.memory_pools._pools.values()),
                "device_memory_stats": jax.devices()[0].memory_stats(),
            }
        )


def test_initialize_binds_actual_capacity_before_freezing_graph(monkeypatch):
    h = Harness()
    r = h.runner
    events = []
    r.is_draft_worker = False
    r.sliding_window_size = None
    monkeypatch.setattr(r, "get_available_device_memory", lambda: 1 << 30)
    monkeypatch.setattr(r, "load_model", lambda: events.append("model"))
    monkeypatch.setattr(r, "init_memory_pool", lambda *args, **kw: events.append("pools"))
    bind = r.bind_attention_resources

    def bind_and_record():
        bind()
        events.append("bound")

    monkeypatch.setattr(r, "bind_attention_resources", bind_and_record)

    def freeze():
        assert r.attn_backend.resources_bound
        assert r.attn_backend.request_capacity == r.req_to_token_pool.size == 4
        events.append("jit")

    monkeypatch.setattr(r, "initialize_jit", freeze)
    monkeypatch.setattr(r, "_build_embedding_pool", lambda: None)
    monkeypatch.setattr(r, "init_routed_experts_capturer", lambda: None)
    with jax.set_mesh(h.mesh):
        r.initialize()
    assert events == ["model", "pools", "bound", "jit"]


def test_multiple_live_requests_reorder_boundary_owner():
    h = Harness()
    r = h.runner
    batch = h.batch([129])
    owner = SimpleNamespace(req_pool_idx=None)
    slot = r.req_to_token_pool.alloc([owner])[0]
    loc = r.token_to_kv_pool_allocator.alloc_extend(
        np.array([0]),
        np.array([3]),
        np.array([-1]),
        3,
    )
    r.req_to_token_pool.req_to_token[slot, :3] = loc
    old_loc = batch.out_cache_loc[:129].copy()
    batch.req_pool_indices = np.array([slot, h.slots[0]], np.int32)
    batch.seq_lens = batch.extend_seq_lens = np.array([3, 129], np.int32)
    batch.out_cache_loc[:132] = np.concatenate([loc, old_loc])
    batch.positions[:132] = np.concatenate([np.arange(3), np.arange(129)])
    with jax.set_mesh(h.mesh):
        md = h.forward_batch(batch).deepseek_v4_metadata
        mask = np.asarray(md.c128.boundary_valid_mask)
        np.testing.assert_array_equal(np.asarray(md.c128.boundary_token_indices)[mask], [130])
        np.testing.assert_array_equal(np.asarray(md.c128.boundary_request_ids)[mask], [1])
        np.testing.assert_array_equal(np.asarray(md.c128.boundary_state_slots)[mask], [h.slots[0]])
        np.testing.assert_array_equal(md.query_request_ids[:132], [0] * 3 + [1] * 129)
