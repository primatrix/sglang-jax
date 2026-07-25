from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

RELAY_ID_SPEC = P("data", None)
DECODE_STATE_SPEC = P("data", None, None)


class RelayBuffers(NamedTuple):
    next_token_ids: jax.Array


class DecodeRequestState(NamedTuple):
    next_token_ids: jax.Array
    seq_lens: jax.Array
    temperatures: jax.Array
    top_ps: jax.Array
    top_ks: jax.Array
    min_ps: jax.Array
    sampling_seeds: jax.Array


class DecodeBatchDescriptor(NamedTuple):
    req_pool_indices: jax.Array
    valid_mask: jax.Array
    page_indices: jax.Array


class DecodeBatchInputs(NamedTuple):
    input_ids: jax.Array
    seq_lens: jax.Array
    positions: jax.Array
    cu_kv_lens: jax.Array
    distribution: jax.Array
    temperatures: jax.Array
    top_ps: jax.Array
    top_ks: jax.Array
    min_ps: jax.Array
    sampling_seeds: jax.Array


class DecodeWorkspaceBatchSpec(NamedTuple):
    is_all_greedy: bool
    need_min_p_sampling: bool
    has_sampling_seeds: bool


class DecodeWorkspace:
    def __init__(self, mesh, req_to_token_pool, *, dp_size: int):
        self.dp_size = dp_size
        self.capacity = int(req_to_token_pool.req_to_token.shape[0])
        self.input_sharding = NamedSharding(mesh, P("data"))
        self.relay_sharding = NamedSharding(mesh, RELAY_ID_SPEC)
        self.state_sharding = NamedSharding(mesh, DECODE_STATE_SPEC)
        self.request_state = create_decode_request_state(
            mesh,
            req_to_token_pool,
            dp_size=dp_size,
        )
        self._initialized_slots = np.zeros((dp_size, self.capacity), dtype=np.bool_)
        self._descriptor_cache: dict[
            tuple[int, int], dict[tuple[bytes, bytes], DecodeBatchDescriptor]
        ] = {}
        self._write_request_state = jax.jit(
            partial(
                publish_decode_request_state,
                dp_size=dp_size,
                relay_sharding=self.relay_sharding,
                state_sharding=self.state_sharding,
            )
        )

    @property
    def relay_buffers(self) -> RelayBuffers:
        return RelayBuffers(next_token_ids=self.request_state.next_token_ids)

    def contains_request_slots(self, req_pool_indices_per_dp) -> bool:
        for dp_rank, indices in enumerate(req_pool_indices_per_dp):
            if indices is None:
                continue
            indices = np.asarray(indices, dtype=np.int32)
            if indices.size and not self._initialized_slots[dp_rank, indices].all():
                return False
        return True

    def get_descriptor(
        self,
        req_pool_indices,
        page_indices,
    ) -> DecodeBatchDescriptor:
        host_page_indices = np.asarray(page_indices, dtype=np.int32)
        host_indices = np.asarray(req_pool_indices, dtype=np.int32)
        bucket = (len(req_pool_indices), len(host_page_indices))
        entries = self._descriptor_cache.setdefault(bucket, {})
        key = (host_indices.tobytes(), host_page_indices.tobytes())
        descriptor = entries.get(key)
        if descriptor is not None:
            return descriptor

        descriptor = DecodeBatchDescriptor(
            req_pool_indices=jax.device_put(host_indices, self.input_sharding),
            valid_mask=jax.device_put(host_indices >= 0, self.input_sharding),
            page_indices=jax.device_put(host_page_indices, self.input_sharding),
        )
        entries[key] = descriptor
        if len(entries) > 2:
            entries.pop(next(iter(entries)))
        return descriptor

    def publish_request_state(
        self,
        req_pool_indices,
        next_token_ids,
        seq_lens,
        temperatures,
        top_ps,
        top_ks,
        min_ps,
        sampling_seeds,
    ) -> None:
        if sampling_seeds is None:
            sampling_seeds = jnp.zeros_like(seq_lens)
        self.request_state = self._write_request_state(
            self.request_state,
            req_pool_indices,
            next_token_ids,
            seq_lens,
            temperatures,
            top_ps,
            top_ks,
            min_ps,
            sampling_seeds,
        )

    def mark_initialized(self, req_pool_indices, real_bs_per_dp, per_dp_bs_size) -> None:
        req_pool_indices = np.asarray(req_pool_indices, dtype=np.int32)
        for dp_rank, real_bs in enumerate(real_bs_per_dp):
            start = dp_rank * per_dp_bs_size
            indices = req_pool_indices[start : start + int(real_bs)]
            if indices.size:
                self._initialized_slots[dp_rank, indices] = True


def create_decode_request_state(
    mesh,
    req_to_token_pool,
    *,
    dp_size: int,
) -> DecodeRequestState:
    capacity = int(req_to_token_pool.req_to_token.shape[0])
    relay_sharding = NamedSharding(mesh, RELAY_ID_SPEC)
    state_sharding = NamedSharding(mesh, DECODE_STATE_SPEC)
    id_shape = (dp_size, capacity)
    seed_dtype = jnp.int64 if jax.config.x64_enabled else jnp.int32
    return DecodeRequestState(
        next_token_ids=jax.device_put(jnp.zeros(id_shape, dtype=jnp.int32), relay_sharding),
        seq_lens=jax.device_put(jnp.zeros(id_shape, dtype=jnp.int32), relay_sharding),
        temperatures=jax.device_put(
            jnp.ones(id_shape + (1,), dtype=jnp.float32),
            state_sharding,
        ),
        top_ps=jax.device_put(jnp.ones(id_shape, dtype=jnp.float32), relay_sharding),
        top_ks=jax.device_put(jnp.ones(id_shape, dtype=jnp.int32), relay_sharding),
        min_ps=jax.device_put(jnp.zeros(id_shape, dtype=jnp.float32), relay_sharding),
        sampling_seeds=jax.device_put(jnp.zeros(id_shape, dtype=seed_dtype), relay_sharding),
    )


def _gather_request_field(field, indices, *, dp_size: int, output_sharding):
    per_dp_bs = indices.shape[0] // dp_size
    indices_2d = indices.reshape((dp_size, per_dp_bs))
    dp_indices = jnp.arange(dp_size, dtype=jnp.int32)[:, None]
    return (
        field.at[dp_indices, indices_2d]
        .get(out_sharding=output_sharding)
        .reshape(indices.shape + field.shape[2:])
    )


def _update_request_field(
    field,
    indices,
    valid_mask,
    values,
    *,
    dp_size: int,
    output_sharding,
):
    per_dp_bs = indices.shape[0] // dp_size
    indices_2d = indices.reshape((dp_size, per_dp_bs))
    valid_2d = valid_mask.reshape((dp_size, per_dp_bs))
    values = values.astype(field.dtype).reshape((dp_size, per_dp_bs) + values.shape[1:])
    dp_indices = jnp.arange(dp_size, dtype=jnp.int32)[:, None]
    scatter_indices = jnp.where(valid_2d, indices_2d, jnp.full_like(indices_2d, field.shape[1]))
    return field.at[dp_indices, scatter_indices].set(
        values,
        mode="drop",
        out_sharding=output_sharding,
    )


def gather_decode_batch_inputs(
    state: DecodeRequestState,
    descriptor: DecodeBatchDescriptor,
    *,
    dp_size: int,
    page_size: int,
    relay_sharding,
    state_sharding,
    output_sharding,
) -> DecodeBatchInputs:
    safe_indices = jnp.where(descriptor.valid_mask, descriptor.req_pool_indices, 0)
    valid = descriptor.valid_mask

    def gather(field, sharding=relay_sharding):
        value = _gather_request_field(
            field,
            safe_indices,
            dp_size=dp_size,
            output_sharding=sharding,
        )
        return jax.sharding.reshard(value, output_sharding)

    input_ids = jnp.where(valid, gather(state.next_token_ids), 0)
    seq_lens = jnp.where(valid, gather(state.seq_lens), 0)
    positions = jnp.where(valid, seq_lens - 1, 0)
    aligned_seq_lens = ((seq_lens + page_size - 1) // page_size) * page_size
    per_dp_bs = seq_lens.shape[0] // dp_size
    aligned_2d = aligned_seq_lens.reshape((dp_size, per_dp_bs))
    cu_kv_2d = jnp.pad(jnp.cumsum(aligned_2d, axis=1), ((0, 0), (1, 0)))
    cu_kv_lens = jax.sharding.reshard(cu_kv_2d.reshape(-1), output_sharding)
    distribution = jnp.repeat(
        valid.reshape((dp_size, per_dp_bs)).sum(axis=1),
        3,
        out_sharding=output_sharding,
    )
    temperatures = jnp.where(valid[:, None], gather(state.temperatures, state_sharding), 1.0)
    top_ps = jnp.where(valid, gather(state.top_ps), 1.0)
    top_ks = jnp.where(valid, gather(state.top_ks), 1)
    min_ps = jnp.where(valid, gather(state.min_ps), 0.0)
    sampling_seeds = jnp.where(valid, gather(state.sampling_seeds), 0)
    return DecodeBatchInputs(
        input_ids=input_ids,
        seq_lens=seq_lens,
        positions=positions,
        cu_kv_lens=cu_kv_lens,
        distribution=distribution,
        temperatures=temperatures,
        top_ps=top_ps,
        top_ks=top_ks,
        min_ps=min_ps,
        sampling_seeds=sampling_seeds,
    )


def publish_decode_request_state(
    state: DecodeRequestState,
    req_pool_indices,
    next_token_ids,
    seq_lens,
    temperatures,
    top_ps,
    top_ks,
    min_ps,
    sampling_seeds,
    *,
    dp_size: int,
    relay_sharding,
    state_sharding,
) -> DecodeRequestState:
    valid_mask = req_pool_indices >= 0
    safe_indices = jnp.where(valid_mask, req_pool_indices, 0)

    def update(field, values, sharding=relay_sharding):
        return _update_request_field(
            field,
            safe_indices,
            valid_mask,
            values,
            dp_size=dp_size,
            output_sharding=sharding,
        )

    return DecodeRequestState(
        next_token_ids=update(state.next_token_ids, next_token_ids),
        seq_lens=update(state.seq_lens, seq_lens),
        temperatures=update(state.temperatures, temperatures, state_sharding),
        top_ps=update(state.top_ps, top_ps),
        top_ks=update(state.top_ks, top_ks),
        min_ps=update(state.min_ps, min_ps),
        sampling_seeds=update(state.sampling_seeds, sampling_seeds),
    )


def update_decode_result(
    state: DecodeRequestState,
    req_pool_indices,
    next_token_ids,
    current_seq_lens,
    *,
    dp_size: int,
    relay_sharding,
) -> DecodeRequestState:
    valid_mask = req_pool_indices >= 0
    safe_indices = jnp.where(valid_mask, req_pool_indices, 0)

    def update(field, values):
        return _update_request_field(
            field,
            safe_indices,
            valid_mask,
            values,
            dp_size=dp_size,
            output_sharding=relay_sharding,
        )

    return state._replace(
        next_token_ids=update(state.next_token_ids, next_token_ids),
        seq_lens=update(state.seq_lens, current_seq_lens + 1),
    )


def create_relay_buffers(
    mesh,
    req_to_token_pool,
    *,
    dp_size: int,
) -> RelayBuffers:
    capacity = int(req_to_token_pool.req_to_token.shape[0])
    sharding = NamedSharding(mesh, RELAY_ID_SPEC)
    return RelayBuffers(
        next_token_ids=jax.device_put(
            jnp.zeros((dp_size, capacity), dtype=jnp.int32),
            sharding,
        )
    )


def update_relay_buffers(
    buffers: RelayBuffers,
    indices,
    valid_mask,
    next_token_ids,
    *,
    dp_size: int,
    output_sharding,
) -> RelayBuffers:
    per_dp_bs = indices.shape[0] // dp_size
    indices = indices.reshape((dp_size, per_dp_bs))
    valid = valid_mask.reshape((dp_size, per_dp_bs))
    next_token_ids = next_token_ids.reshape((dp_size, per_dp_bs))
    dp_indices = jnp.arange(dp_size, dtype=jnp.int32)[:, None]
    scatter_indices = jnp.where(
        valid,
        indices,
        jnp.full_like(indices, buffers.next_token_ids.shape[1]),
    )
    return RelayBuffers(
        next_token_ids=buffers.next_token_ids.at[dp_indices, scatter_indices].set(
            next_token_ids,
            mode="drop",
            out_sharding=output_sharding,
        )
    )


def gather_relay_buffers(
    buffers: RelayBuffers,
    indices,
    *,
    dp_size: int,
    output_sharding,
) -> jax.Array:
    per_dp_bs = indices.shape[0] // dp_size
    indices_2d = indices.reshape((dp_size, per_dp_bs))
    dp_indices = jnp.arange(dp_size, dtype=jnp.int32)[:, None]
    values = (
        buffers.next_token_ids.at[dp_indices, indices_2d]
        .get(out_sharding=output_sharding)
        .reshape(indices.shape)
    )
    return values


def resolve_relay_inputs(
    buffers: RelayBuffers,
    indices,
    valid_mask,
    input_ids,
    *,
    dp_size: int,
    relay_sharding,
    output_sharding,
) -> jax.Array:
    relay_ids = gather_relay_buffers(
        buffers,
        indices,
        dp_size=dp_size,
        output_sharding=relay_sharding,
    )
    relay_ids = jax.sharding.reshard(relay_ids, output_sharding)
    return jnp.where(valid_mask, relay_ids, input_ids)


def resolve_decode_inputs(
    buffers: RelayBuffers,
    indices,
    input_ids,
    seq_lens,
    *,
    dp_size: int,
    relay_sharding,
    output_sharding,
) -> tuple[jax.Array, jax.Array]:
    valid_mask = indices >= 0
    indices = jnp.where(valid_mask, indices, 0)
    input_ids = resolve_relay_inputs(
        buffers,
        indices,
        valid_mask,
        input_ids,
        dp_size=dp_size,
        relay_sharding=relay_sharding,
        output_sharding=output_sharding,
    )
    positions = jnp.where(valid_mask, seq_lens - 1, 0)
    return input_ids, positions
