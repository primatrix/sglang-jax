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
import numpy as np
from flax import nnx
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.layers.attention.deepseek_v4_hca_backend import (
    DeepseekV4HCABackend,
    DeepseekV4HCAMetadata,
)
from sgl_jax.srt.layers.attention.dsv4.dispatch import run_layer
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

    def tree_flatten(self):
        return (self.kernel, self.state_init_slots, self.attention), (
            self.schedule,
            self.use_uniform_prefill_fast_path,
        )

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(children[0], aux[0], aux[1], children[1], children[2])


class DeepseekV4RuntimeBackend(DeepseekV4HCABackend):
    def __init__(self, *, mesh, page_size, max_context_len):
        # Constructed before C1 sizing. Bind the actual scalar capacity after
        # allocation; no real forward is allowed to use the placeholder.
        super().__init__(
            mesh=mesh, page_size=page_size, max_context_len=max_context_len, request_capacity=1
        )
        self.resources_bound = False
        self.forward_metadata = nnx.data(DeepseekV4RuntimeMetadata())

    def bind_resources(self, request_pool, allocator):
        if allocator.dp_size != self.mesh.shape["data"] or allocator.page_size != self.page_size:
            raise ValueError("V4 runtime and resource geometry disagree")
        self.request_capacity = request_pool.size
        self.resources_bound = True

    def get_forward_metadata(self, batch, *, request_pool, allocator):
        if not self.resources_bound:
            raise RuntimeError("V4 runtime resources must be bound after pool initialization")
        hca = super().get_forward_metadata(
            batch, request_pool=request_pool, allocator=allocator, fixed_bucket=True
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
        local = []
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

    def __call__(self, q, k, v, layer, forward_batch, token_to_kv_pool, **kwargs):
        ratio = self.layer_ratio(layer, token_to_kv_pool)
        if ratio == 128:
            # Production C128 stays on #349's Pallas kernel.
            return super().__call__(q, k, v, layer, forward_batch, token_to_kv_pool, **kwargs)
        if ratio not in (0, 4):
            raise ValueError(f"unsupported V4 compression ratio {ratio}")

        # M2.5's native route for SWA-only and CSA layers. The per-layer weight bundle
        # (compressor wkv/wgate/ape/norm, the indexer's own compressor and projections,
        # the attention sink) belongs to the model, so M1.4 supplies it; there is no
        # sensible default to invent here.
        bundle = kwargs.get("dsv4_layer")
        if bundle is None:
            raise NotImplementedError(
                f"V4 ratio-{ratio} layers need the per-layer weight bundle from M1.4; "
                "pass it as the `dsv4_layer` keyword. See layers/attention/dsv4/dispatch.py "
                "for the expected contents."
            )
        return run_layer(
            q=q,
            new_kv=k,
            layer_id=int(layer.layer_id),
            ratio=ratio,
            metadata=self.forward_metadata.attention,
            window_size=self.window_size,
            **bundle,
        )


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
