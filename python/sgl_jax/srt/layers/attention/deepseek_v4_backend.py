"""C3 runtime bridge: host C1 ownership -> dynamic M2 metadata -> V4 consumer.

Host request/allocator objects stay on ModelRunner. Only their array-derived
metadata enters ForwardBatch, so neither a donated pool nor a mutable host page
ledger is captured in the Flax model graph.

M2.5 filled in the layer dispatch: C128 stays on #349's Pallas kernel and the
SWA-only / CSA routes go through `dsv4.dispatch.run_layer`, which needs the
per-layer weight bundle M1.4 owns.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.layers.attention.base_attn_backend import AttentionBackend
from sgl_jax.srt.layers.attention.deepseek_v4_csa_backend import (
    CompressorWeights,
    DeepseekV4CSABackend,
    padded_read_tables,
)
from sgl_jax.srt.layers.attention.deepseek_v4_hca_backend import (
    DeepseekV4HCABackend,
    DeepseekV4HCAMetadata,
)
from sgl_jax.srt.layers.attention.dsv4.metadata import (
    DeepseekV4AttentionMetadata,
    derive_attention_metadata,
)
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode


@jax.tree_util.register_pytree_node_class
@dataclass
class DeepseekV4RuntimeMetadata(DeepseekV4HCAMetadata):
    # Each leaf concatenates equal-sized DP-local arrays. Indices remain local
    # to a rank, including cu_q_lens and the compression-boundary sentinels.
    attention: DeepseekV4AttentionMetadata | None = None
    read_tables: tuple = ()

    def tree_flatten(self):
        return (self.kernel, self.state_init_slots, self.attention, self.read_tables), (
            self.schedule,
            self.use_uniform_prefill_fast_path,
        )

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(children[0], aux[0], aux[1], children[1], children[2], children[3])


class DeepseekV4AttentionBackend(AttentionBackend):
    def __init__(self, *, mesh, page_size, max_context_len, config=None):
        self.mesh = mesh
        self.page_size = page_size
        self.max_context_len = max_context_len
        self.request_capacity = 1
        self.window_size = int(getattr(config, "sliding_window", 128))
        self.hca = DeepseekV4HCABackend(
            mesh=mesh, page_size=page_size, max_context_len=max_context_len, request_capacity=1
        )
        self.csa = DeepseekV4CSABackend(mesh)
        self.use_pallas_hca = jax.default_backend() == "tpu" and (
            getattr(config, "hidden_size", 4096),
            getattr(config, "num_attention_heads", 64),
            getattr(config, "head_dim", 512),
            getattr(config, "qk_rope_head_dim", 64),
            self.window_size,
        ) == (4096, 64, 512, 64, 128)
        self.resources_bound = False
        self.forward_metadata = nnx.data(DeepseekV4RuntimeMetadata())

    @staticmethod
    def get_max_running_reqests(max_context_len: int, page_size: int) -> int:
        # TpWorker combines this kernel metadata limit with the actual request
        # pool capacity. Reuse the HCA scalar-prefetch budget for mixed V4 layers.
        return DeepseekV4HCABackend.get_max_running_reqests(max_context_len, page_size)

    def bind_resources(self, request_pool, allocator):
        if allocator.dp_size != self.mesh.shape["data"] or allocator.page_size != self.page_size:
            raise ValueError("V4 runtime and resource geometry disagree")
        self.request_capacity = request_pool.size
        self.hca.request_capacity = request_pool.size
        self.resources_bound = True

    def get_forward_metadata(self, batch, *, request_pool, allocator):
        if not self.resources_bound:
            raise RuntimeError("V4 runtime resources must be bound after pool initialization")
        hca = (
            self.hca.get_forward_metadata(
                batch, request_pool=request_pool, allocator=allocator, fixed_bucket=True
            )
            if self.use_pallas_hca
            else DeepseekV4HCAMetadata()
        )
        dp = int(self.mesh.shape["data"])
        lengths = np.asarray(batch.seq_lens, np.int32).reshape(dp, -1)
        slots = np.asarray(batch.req_pool_indices, np.int32).reshape(dp, -1)
        positions = np.asarray(batch.positions, np.int32).reshape(dp, -1)
        history = np.asarray(batch.out_cache_loc, np.int32).reshape(dp, -1)
        if history.shape != positions.shape:
            raise ValueError("V4 output addresses must match the padded query token axis")
        queries = (
            (lengths > 0).astype(np.int32)
            if batch.forward_mode == ForwardMode.DECODE
            else np.asarray(batch.extend_seq_lens, np.int32).reshape(dp, -1)
        )
        if batch.forward_mode not in (ForwardMode.EXTEND, ForwardMode.DECODE):
            raise ValueError("V4 supports ordinary EXTEND and DECODE only")
        active = lengths > 0
        if np.any(lengths < 0) or np.any(lengths > self.max_context_len):
            raise ValueError("V4 request length exceeds the configured context")
        if np.any(queries < 0) or np.any(queries > lengths) or np.any((queries > 0) != active):
            raise ValueError("invalid V4 query lengths")
        if np.any(queries.sum(axis=1) > positions.shape[1]):
            raise ValueError("V4 queries exceed the padded token capacity")
        if np.any(slots[active] < 0) or np.any(slots[active] >= self.request_capacity):
            raise ValueError("active V4 requests require allocated request slots")
        if len(np.unique(slots[active])) != active.sum():
            raise ValueError("active V4 requests must own distinct slots")
        if batch.forward_mode == ForwardMode.EXTEND and not np.array_equal(
            np.asarray(batch.extend_prefix_lens).reshape(lengths.shape), lengths - queries
        ):
            raise ValueError("V4 prefix plus query length must equal sequence length")
        local = []
        tables_by_ratio = {ratio: [] for ratio in (0, 4, 128)}
        # All DP ranks must have identical local array extents for shard_map.
        # Only requests with queries contribute rows to read_tables; bound their
        # completed groups with a power-of-two bucket shared across ranks.
        compressed_capacities = {0: 1}
        for ratio in (4, 128):
            count = int(np.max(np.sum(np.where(queries > 0, lengths // ratio, 0), axis=1)))
            compressed_capacities[ratio] = max(128, 1 << (max(1, count) - 1).bit_length())
        for rank in range(dp):
            live = int(queries[rank].sum())
            mapping = allocator.full_to_swa_index_mapping
            mapping = mapping[rank] if isinstance(mapping, list) else mapping
            writes = history[rank, :live]
            if np.any((writes < self.page_size) | (writes >= len(mapping))):
                raise ValueError("V4 live query addresses must name allocated original-token slots")
            prefixes = lengths[rank] - queries[rank]
            expected = np.concatenate(
                [
                    request_pool.req_to_token[slot, pre:end]
                    for slot, pre, end, n in zip(
                        slots[rank], prefixes, lengths[rank], queries[rank], strict=True
                    )
                    if n
                ]
                or [np.empty(0, np.int32)]
            )
            if not np.array_equal(writes, expected) or np.any(mapping[writes] == 0):
                raise ValueError("V4 query writes disagree with the request/SWA ownership map")
            swa = np.full(positions.shape[1], -1, np.int32)
            swa[:live] = mapping[writes]
            for ratio in tables_by_ratio:
                tables_by_ratio[ratio].append(
                    padded_read_tables(
                        request_pool=request_pool,
                        allocator=allocator,
                        slots=slots[rank],
                        lengths=lengths[rank],
                        q_lens=queries[rank],
                        ratio=ratio,
                        window_size=self.window_size,
                        page_size=self.page_size,
                        max_context_len=self.max_context_len,
                        token_capacity=positions.shape[1],
                        rank=rank,
                        compressed_capacity=compressed_capacities[ratio],
                    )
                )
            local.append(
                derive_attention_metadata(
                    q_lens=queries[rank],
                    prefix_lens=prefixes,
                    positions=positions[rank],
                    request_slots=slots[rank],
                    history_write_loc=history[rank],
                    swa_write_loc=swa,
                    pages_per_request=(lengths[rank] + self.page_size - 1) // self.page_size,
                    page_size=self.page_size,
                    window_size=self.window_size,
                    state_init_mask=(queries[rank] > 0) & (prefixes == 0),
                )
            )
        sharding = NamedSharding(self.mesh, P("data"))
        attention = jax.tree.map(
            lambda *arrays: jax.device_put(np.concatenate(arrays), sharding), *local
        )
        return DeepseekV4RuntimeMetadata(
            hca.kernel,
            hca.schedule,
            hca.use_uniform_prefill_fast_path,
            hca.state_init_slots,
            attention,
            tuple(
                jax.tree.map(
                    lambda *arrays: jax.device_put(np.concatenate(arrays), sharding), *tables
                )
                for tables in tables_by_ratio.values()
            ),
        )

    def layer_ratio(self, layer, token_to_kv_pool) -> int:
        """Compression ratio of a layer, from C1's spec -- the single classification.

        C1's `spec.compress_ratios` is the same list `configs/deepseek_v4.classify_layers`
        reads, so this does not add a third derivation.
        """
        layer_id = int(layer.layer_id)
        ratios = token_to_kv_pool.spec.compress_ratios
        if not 0 <= layer_id < len(ratios):
            raise ValueError(f"layer {layer_id} is outside the V4 backbone")
        return int(ratios[layer_id])

    def __call__(
        self,
        q,
        k,
        v,
        layer,
        forward_batch,
        token_to_kv_pool,
        *,
        compressor_state_pool,
        compressor_input=None,
        compressor=None,
        indexer=None,
        attention_sink=None,
        rope_head_dim=64,
        norm_eps=1e-6,
        index_topk=None,
        **kwargs,
    ):
        ratio = self.layer_ratio(layer, token_to_kv_pool)
        if compressor is None and "wkv" in kwargs:
            # The standalone HCA interface also accepts separate cosine/sine tables.
            compressor = CompressorWeights(
                kwargs["wkv"],
                kwargs["wgate"],
                kwargs["ape"],
                kwargs["norm_weight"],
                jnp.concatenate((kwargs["cos"], kwargs["sin"]), axis=-1),
            )
        if ratio == 128 and self.use_pallas_hca:
            if compressor is None:
                raise ValueError("HCA requires model compressor weights")
            cache = compressor.cos_sin_cache
            output, (state, window, history) = self.hca(
                q,
                k,
                v,
                layer,
                forward_batch,
                token_to_kv_pool,
                compressor_state_pool=compressor_state_pool,
                compressor_input=compressor_input,
                wkv=compressor.wkv,
                wgate=compressor.wgate,
                ape=compressor.ape,
                norm_weight=compressor.norm_weight,
                cos=cache[:, : cache.shape[-1] // 2],
                sin=cache[:, cache.shape[-1] // 2 :],
                attention_sink=attention_sink,
                metadata=self.forward_metadata,
            )
            return output.reshape(q.shape), {"state": state, "swa": window, "c128": history}
        md = self.forward_metadata
        if md.attention is None or not md.read_tables:
            raise RuntimeError("V4 attention metadata has not been prepared")
        return self.csa(
            q,
            k[:, 0] if k.ndim == 3 else k,
            hidden_states=compressor_input,
            layer_id=int(layer.layer_id),
            ratio=ratio,
            metadata=md.attention,
            tables=md.read_tables[(0, 4, 128).index(ratio)],
            token_to_kv_pool=token_to_kv_pool,
            compressor_state_pool=compressor_state_pool,
            compressor=compressor,
            indexer=indexer,
            attention_sink=attention_sink,
            softmax_scale=float(layer.scaling),
            rope_head_dim=rope_head_dim,
            norm_eps=norm_eps,
            index_topk=index_topk,
        )

    @staticmethod
    def pack_pool_updates(layer_updates, token_to_kv_pool, compressor_state_pool):
        kv = {name: list(arrays) for name, arrays in token_to_kv_pool.buffers.items()}
        state = {name: list(arrays) for name, arrays in compressor_state_pool.buffers.items()}
        for layer_id, updates in layer_updates.items():
            ratio = token_to_kv_pool.spec.compress_ratios[layer_id]
            for name, array in updates.items():
                if name in ("state", "indexer_state"):
                    family = f"c{ratio}" if name == "state" else "indexer"
                    state[family][compressor_state_pool.layer_to_buffer[family][layer_id]] = array
                else:
                    kv[name][token_to_kv_pool.layer_to_buffer[name][layer_id]] = array
        return {
            "token_to_kv_pool": {name: tuple(arrays) for name, arrays in kv.items()},
            "compressor_state_pool": {name: tuple(arrays) for name, arrays in state.items()},
        }


def prepare_dummy_batch(batch, backend):
    """Use C1's inactive rows without reserving or writing any live request slot."""
    batch.seq_lens = np.zeros_like(batch.seq_lens, dtype=np.int32)
    batch.req_pool_indices = np.full_like(batch.req_pool_indices, backend.request_capacity)
    batch.out_cache_loc = np.full_like(batch.out_cache_loc, -1)
    batch.positions = np.zeros_like(batch.positions)
    batch.cache_loc = np.zeros_like(batch.cache_loc)
    if batch.forward_mode == ForwardMode.EXTEND:
        batch.extend_prefix_lens = np.zeros_like(batch.extend_prefix_lens, dtype=np.int32)
        batch.extend_seq_lens = np.zeros_like(batch.extend_seq_lens, dtype=np.int32)
