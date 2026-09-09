"""Runtime resource setup for the real-weight TPU attention/layer comparisons."""

from types import SimpleNamespace

import jax
import numpy as np

from sgl_jax.srt.mem_cache.deepseek_v4.allocator import DeepseekV4TokenToKVPoolAllocator
from sgl_jax.srt.mem_cache.deepseek_v4.pool import DeepseekV4CacheSpec, DeepseekV4TokenToKVPool
from sgl_jax.srt.mem_cache.deepseek_v4.state import DeepseekV4CompressStatePool
from sgl_jax.srt.mem_cache.memory_pool import MemoryPools, ReqToTokenPool
from sgl_jax.srt.model_executor.compilation_manager import CompilationManager
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sgl_jax.srt.model_executor.model_runner import ModelRunner
from sgl_jax.srt.server_args import ServerArgs


class Harness:
    def __init__(self, page_size=128, dp=1, tp=1, spec=None):
        if jax.device_count() < dp * tp:
            raise RuntimeError(f"requires {dp * tp} devices, found {jax.device_count()}")
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
