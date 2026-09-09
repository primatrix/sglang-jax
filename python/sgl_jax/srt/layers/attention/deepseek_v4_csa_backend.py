"""Cache-aware native V4 attention, including the CSA compressor and indexer.

Models supply projections and weight values. This backend owns rank-local cache
views, slot resets, sharding and update packaging; no model parameters live here.
The same native implementation supports SWA and small-geometry HCA validation.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.layers.attention.dsv4.dispatch import (
    ReadTables,
    read_tables,
    run_layer,
)


class CompressorWeights(NamedTuple):
    wkv: jax.Array
    wgate: jax.Array
    ape: jax.Array
    norm_weight: jax.Array
    cos_sin_cache: jax.Array


class IndexerInputs(NamedTuple):
    q: jax.Array
    weights: jax.Array
    compressor: CompressorWeights


def padded_read_tables(
    *,
    request_pool,
    allocator,
    slots,
    lengths,
    q_lens,
    ratio,
    window_size,
    page_size,
    max_context_len,
    token_capacity,
    rank,
    compressed_capacity=None,
):
    tables = read_tables(
        request_pool=request_pool,
        allocator=allocator,
        slots=slots,
        lengths=lengths,
        q_lens=q_lens,
        ratio=ratio,
        window_size=window_size,
        page_size=page_size,
        rank=rank,
    )
    # Window reads are bounded by the query bucket and sliding-window halo.
    window_capacity = max(
        1, min(len(slots) * max_context_len, token_capacity + len(slots) * (window_size - 1))
    )
    if compressed_capacity is None:
        # Bucket actual gathered history, not padded request slots times the
        # configured maximum context. The caller shares this bucket across DP.
        count = len(tables.compressed_rows)
        compressed_capacity = max(128, 1 << (max(1, count) - 1).bit_length()) if ratio else 1
    values = {}
    for name in ReadTables.__slots__:
        value = getattr(tables, name)
        capacity = window_capacity if name.startswith("window") else compressed_capacity
        if len(value) > capacity:
            raise ValueError(f"{name} exceeds the configured V4 metadata capacity")
        fill = 0 if name.endswith("rows") else -1
        values[name] = np.pad(value, (0, capacity - len(value)), constant_values=fill)
    return ReadTables(**values)


def _reset_state(state, metadata):
    limit = state.shape[0] - 1  # The last local slot is padding, not a request.
    slots = metadata.request_slots
    valid = metadata.state_init_mask & metadata.request_valid_mask & (slots >= 0) & (slots < limit)
    destinations = jnp.where(valid, slots, state.shape[0])
    empty = jnp.zeros((slots.shape[0], *state.shape[1:]), state.dtype)
    empty = empty.at[..., state.shape[-1] // 2 :].set(-jnp.inf)
    return state.at[destinations].set(empty, mode="drop")


class DeepseekV4CSABackend(nnx.Module):
    def __init__(self, mesh):
        self.mesh = mesh

    def __call__(
        self,
        q,
        new_kv,
        *,
        hidden_states,
        layer_id,
        ratio,
        metadata,
        tables,
        token_to_kv_pool,
        compressor_state_pool,
        compressor,
        indexer,
        attention_sink,
        softmax_scale,
        rope_head_dim,
        norm_eps,
        index_topk,
    ):
        kv_pool = token_to_kv_pool
        states = compressor_state_pool
        family = f"c{ratio}"
        window = kv_pool.get_buffer("swa", layer_id)
        compressed = kv_pool.get_buffer(family, layer_id) if ratio else None
        state = states.get_buffer(family, layer_id) if ratio else None
        index_cache = kv_pool.get_buffer("indexer", layer_id) if ratio == 4 else None
        index_state = states.get_buffer("indexer", layer_id) if ratio == 4 else None
        if ratio and compressor is None:
            raise ValueError("compressed attention requires model compressor weights")
        if ratio == 4 and indexer is None:
            raise ValueError("CSA requires model indexer projections and compressor weights")

        def local(
            q_,
            kv_,
            x_,
            window_,
            compressed_,
            state_,
            index_cache_,
            index_state_,
            md,
            read,
            cw,
            iq,
            iw,
            icw,
            sink,
        ):
            valid = md.valid_token_mask[:, None]
            x_ = jnp.where(valid, x_, 0)
            kv_ = jnp.where(valid, kv_, 0)
            q_ = jnp.where(valid[:, :, None], q_, 0)
            buffers = {"swa": window_.reshape(-1, window_.shape[-1])}
            if ratio:
                buffers["compressed"] = compressed_.reshape(-1, compressed_.shape[-1])
                state_ = _reset_state(state_, md)
            idx = None
            if ratio == 4:
                buffers["indexer"] = index_cache_.reshape(-1, index_cache_.shape[-1])
                idx = dict(
                    q=iq,
                    weights=iw,
                    compressor_input=x_,
                    state=_reset_state(index_state_, md),
                    head_dim=iq.shape[-1],
                    rope_head_dim=rope_head_dim,
                    compressor_weights=icw._asdict(),
                )
            output, updates = run_layer(
                q=q_,
                new_kv=kv_,
                compressor_input=x_,
                layer_id=layer_id,
                ratio=ratio,
                metadata=md,
                tables=read,
                kv_buffers=buffers,
                state=state_,
                compressor_weights=None if cw is None else cw._asdict(),
                indexer=idx,
                attention_sink=sink,
                softmax_scale=softmax_scale,
                window_size=md.window_size,
                head_dim=q_.shape[-1],
                index_topk=index_topk,
                rope_head_dim=rope_head_dim,
                norm_eps=norm_eps,
            )
            out = {"swa": updates["swa"].reshape(window_.shape)}
            if ratio:
                out[family] = updates["compressed"].reshape(compressed_.shape)
                out["state"] = updates["state"]
            if ratio == 4:
                out["indexer"] = updates["indexer"].reshape(index_cache_.shape)
                out["indexer_state"] = updates["indexer_state"]
            return output, out

        replica = lambda tree: jax.tree.map(lambda a: P(*([None] * a.ndim)), tree)
        outputs = {"swa": P("data", None)}
        if ratio:
            outputs.update({family: P("data", None, None), "state": P("data", None, None)})
        if ratio == 4:
            outputs.update(
                {"indexer": P("data", None, None), "indexer_state": P("data", None, None)}
            )
        specs = (P("data", "tensor", None), outputs)
        named = jax.tree.map(lambda p: NamedSharding(self.mesh, p), specs)
        fn = jax.shard_map(
            local,
            mesh=None,
            in_specs=(
                P("data", "tensor", None),
                P("data", None),
                P("data", None),
                P("data", None),
                P("data", None, None) if ratio else None,
                P("data", None, None) if ratio else None,
                P("data", None, None) if ratio == 4 else None,
                P("data", None, None) if ratio == 4 else None,
                jax.tree.map(lambda _: P("data"), metadata),
                jax.tree.map(lambda _: P("data"), tables),
                replica(compressor),
                P("data", None, None) if indexer else None,
                P("data", None) if indexer else None,
                replica(indexer.compressor) if indexer else None,
                P("tensor"),
            ),
            out_specs=specs,
            check_vma=False,
        )
        fn = jax.sharding.auto_axes(fn, axes=self.mesh.axis_names, out_sharding=named)
        return fn(
            q,
            new_kv,
            hidden_states,
            window,
            compressed,
            state,
            index_cache,
            index_state,
            metadata,
            tables,
            compressor,
            indexer.q if indexer else None,
            indexer.weights if indexer else None,
            indexer.compressor if indexer else None,
            attention_sink,
        )
