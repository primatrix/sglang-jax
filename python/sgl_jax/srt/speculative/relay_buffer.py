from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

RELAY_STATE_SPEC = P("data", None, None)
RELAY_ID_SPEC = P("data", None)
DSPARK_RELAY_STATE_SPEC = P(None, "data", None, None)
DSPARK_RELAY_ID_SPEC = P(None, "data", None)
DSPARK_CAPACITY_LAG = 2
DSPARK_RELAY_RING_SIZE = DSPARK_CAPACITY_LAG + 1


class SpecRelayBuffers(NamedTuple):
    topk_index: jax.Array
    hidden_states: jax.Array
    verified_id: jax.Array
    new_seq_lens: jax.Array


class DFlashRelayBuffers(NamedTuple):
    verified_id: jax.Array
    new_seq_lens: jax.Array


class DSparkConfidenceRelayBuffers(NamedTuple):
    confidence: jax.Array
    slot_generation: jax.Array
    source_decode_round: jax.Array


class DSparkConfidenceRelayHost:
    """Non-blocking host mirror of the req-indexed DSpark confidence ring."""

    def __init__(self, *, dp_size: int, capacity: int, gamma: int):
        self.dp_size = int(dp_size)
        self.capacity = int(capacity)
        self.gamma = int(gamma)
        self._snapshot = (
            np.zeros(
                (DSPARK_RELAY_RING_SIZE, self.dp_size, self.capacity, self.gamma),
                dtype=np.float32,
            ),
            np.full(
                (DSPARK_RELAY_RING_SIZE, self.dp_size, self.capacity),
                -1,
                dtype=np.int32,
            ),
            np.full(
                (DSPARK_RELAY_RING_SIZE, self.dp_size, self.capacity),
                -1,
                dtype=np.int32,
            ),
        )
        self._publisher = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="dspark-confidence-relay",
        )
        self._pending: list[Future] = []

    def publish(self, buffers: DSparkConfidenceRelayBuffers) -> None:
        """Schedule materialization without waiting in the scheduler thread."""
        jax.copy_to_host_async(buffers)
        self._pending = [future for future in self._pending if not future.done()]
        self._pending.append(self._publisher.submit(self._materialize, buffers))

    def _materialize(self, buffers: DSparkConfidenceRelayBuffers) -> None:
        # Tuple replacement is atomic under the GIL, so readers see either the
        # complete old snapshot or the complete new one.
        self._snapshot = (
            np.asarray(buffers.confidence, dtype=np.float32).copy(),
            np.asarray(buffers.slot_generation, dtype=np.int32).copy(),
            np.asarray(buffers.source_decode_round, dtype=np.int32).copy(),
        )

    def gather_lagged_confidence(
        self,
        req_pool_indices: np.ndarray,
        slot_generations: np.ndarray,
        decode_rounds: np.ndarray,
        active_mask: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, int]]:
        """Gather C[t-2], falling back per request without waiting for D2H."""
        req_pool_indices = np.asarray(req_pool_indices, dtype=np.int32)
        slot_generations = np.asarray(slot_generations, dtype=np.int32)
        decode_rounds = np.asarray(decode_rounds, dtype=np.int32)
        active_mask = np.asarray(active_mask, dtype=np.bool_)
        if not (
            req_pool_indices.shape
            == slot_generations.shape
            == decode_rounds.shape
            == active_mask.shape
        ):
            raise ValueError("DSpark confidence relay metadata must have identical shapes.")
        if req_pool_indices.size % self.dp_size != 0:
            raise ValueError("DSpark confidence relay batch must be divisible by dp_size.")

        confidence = np.ones((req_pool_indices.size, self.gamma), dtype=np.float32)
        stats = {
            "hit": 0,
            "stale_warmup": 0,
            "stale_generation": 0,
            "stale_not_ready": 0,
        }
        host_confidence, host_generation, host_round = self._snapshot
        per_dp_bs = req_pool_indices.size // self.dp_size
        for row in np.flatnonzero(active_mask):
            expected_round = int(decode_rounds[row]) - DSPARK_CAPACITY_LAG
            if expected_round < 0:
                stats["stale_warmup"] += 1
                continue
            req_idx = int(req_pool_indices[row])
            if req_idx < 0 or req_idx >= self.capacity:
                stats["stale_not_ready"] += 1
                continue
            dp_rank = int(row) // per_dp_bs
            ring_slot = expected_round % DSPARK_RELAY_RING_SIZE
            if int(host_generation[ring_slot, dp_rank, req_idx]) != int(
                slot_generations[row]
            ):
                stats["stale_generation"] += 1
                continue
            if int(host_round[ring_slot, dp_rank, req_idx]) != expected_round:
                stats["stale_not_ready"] += 1
                continue
            confidence[row] = host_confidence[ring_slot, dp_rank, req_idx]
            stats["hit"] += 1
        return confidence, stats

    def wait_for_pending_for_test(self) -> None:
        """Test-only synchronization; serving code must never call this."""
        for future in self._pending:
            future.result()
        self._pending.clear()


def create_spec_relay_buffers(
    mesh,
    req_to_token_pool,
    *,
    dp_size: int,
    num_steps: int,
    hidden_size: int,
    hidden_dtype,
) -> SpecRelayBuffers:
    """Create DP-local req-indexed buffers for cross-batch draft state relay."""
    capacity = int(req_to_token_pool.req_to_token.shape[0])
    token_sharding = NamedSharding(mesh, RELAY_STATE_SPEC)
    hidden_sharding = NamedSharding(mesh, RELAY_STATE_SPEC)
    id_sharding = NamedSharding(mesh, RELAY_ID_SPEC)
    return SpecRelayBuffers(
        topk_index=jax.device_put(
            jnp.zeros((dp_size, capacity, num_steps), dtype=jnp.int32),
            token_sharding,
        ),
        hidden_states=jax.device_put(
            jnp.zeros((dp_size, capacity, hidden_size), dtype=hidden_dtype),
            hidden_sharding,
        ),
        verified_id=jax.device_put(
            jnp.zeros((dp_size, capacity), dtype=jnp.int32),
            id_sharding,
        ),
        new_seq_lens=jax.device_put(
            jnp.zeros((dp_size, capacity), dtype=jnp.int32),
            id_sharding,
        ),
    )


def create_dflash_relay_buffers(
    mesh,
    req_to_token_pool,
    *,
    dp_size: int,
) -> DFlashRelayBuffers:
    """Create the minimal req-indexed state needed by DFlash overlap."""
    capacity = int(req_to_token_pool.req_to_token.shape[0])
    sharding = NamedSharding(mesh, RELAY_ID_SPEC)
    shape = (dp_size, capacity)
    return DFlashRelayBuffers(
        verified_id=jax.device_put(jnp.zeros(shape, dtype=jnp.int32), sharding),
        new_seq_lens=jax.device_put(jnp.zeros(shape, dtype=jnp.int32), sharding),
    )


def create_dspark_confidence_relay_buffers(
    mesh,
    req_to_token_pool,
    *,
    dp_size: int,
    gamma: int,
) -> DSparkConfidenceRelayBuffers:
    capacity = int(req_to_token_pool.req_to_token.shape[0])
    state_sharding = NamedSharding(mesh, DSPARK_RELAY_STATE_SPEC)
    id_sharding = NamedSharding(mesh, DSPARK_RELAY_ID_SPEC)
    state_shape = (DSPARK_RELAY_RING_SIZE, dp_size, capacity, gamma)
    id_shape = (DSPARK_RELAY_RING_SIZE, dp_size, capacity)
    return DSparkConfidenceRelayBuffers(
        confidence=jax.device_put(jnp.zeros(state_shape, dtype=jnp.float32), state_sharding),
        slot_generation=jax.device_put(
            jnp.full(id_shape, -1, dtype=jnp.int32), id_sharding
        ),
        source_decode_round=jax.device_put(
            jnp.full(id_shape, -1, dtype=jnp.int32), id_sharding
        ),
    )


def update_dspark_confidence_relay_buffers(
    buffers: DSparkConfidenceRelayBuffers,
    req_pool_indices,
    slot_generations,
    decode_rounds,
    valid_mask,
    confidence,
    *,
    dp_size: int,
) -> DSparkConfidenceRelayBuffers:
    per_dp_bs = req_pool_indices.shape[0] // dp_size
    indices = req_pool_indices.reshape((dp_size, per_dp_bs))
    generations = slot_generations.reshape((dp_size, per_dp_bs))
    rounds = decode_rounds.reshape((dp_size, per_dp_bs))
    valid = valid_mask.reshape((dp_size, per_dp_bs))
    values = confidence.reshape((dp_size, per_dp_bs, confidence.shape[-1]))
    dp_indices = jnp.arange(dp_size, dtype=jnp.int32)[:, None]
    ring_slots = jnp.mod(rounds, DSPARK_RELAY_RING_SIZE)
    scatter_indices = jnp.where(valid, indices, buffers.confidence.shape[2])

    return DSparkConfidenceRelayBuffers(
        confidence=buffers.confidence.at[ring_slots, dp_indices, scatter_indices].set(
            values,
            mode="drop",
            out_sharding=DSPARK_RELAY_STATE_SPEC,
        ),
        slot_generation=buffers.slot_generation.at[
            ring_slots, dp_indices, scatter_indices
        ].set(
            generations,
            mode="drop",
            out_sharding=DSPARK_RELAY_ID_SPEC,
        ),
        source_decode_round=buffers.source_decode_round.at[
            ring_slots, dp_indices, scatter_indices
        ].set(
            rounds,
            mode="drop",
            out_sharding=DSPARK_RELAY_ID_SPEC,
        ),
    )


def update_spec_relay_buffers(
    buffers: SpecRelayBuffers,
    future_indices,
    valid_mask,
    topk_index,
    hidden_states,
    verified_id,
    new_seq_lens,
    *,
    dp_size: int,
) -> SpecRelayBuffers:
    """Write DP-padded draft state into relay buffers without touching padded rows."""
    per_dp_bs = future_indices.shape[0] // dp_size
    indices = future_indices.reshape((dp_size, per_dp_bs))
    valid = valid_mask.reshape((dp_size, per_dp_bs))
    dp_indices = jnp.arange(dp_size, dtype=jnp.int32)[:, None]
    scatter_indices = jnp.where(
        valid,
        indices,
        jnp.full_like(indices, buffers.topk_index.shape[1]),
    )

    topk_index = topk_index.reshape((dp_size, per_dp_bs) + topk_index.shape[1:])
    hidden_states = hidden_states.reshape((dp_size, per_dp_bs) + hidden_states.shape[1:])
    verified_id = verified_id.reshape((dp_size, per_dp_bs))
    new_seq_lens = new_seq_lens.reshape((dp_size, per_dp_bs))

    return SpecRelayBuffers(
        topk_index=buffers.topk_index.at[dp_indices, scatter_indices].set(
            topk_index,
            mode="drop",
            out_sharding=RELAY_STATE_SPEC,
        ),
        hidden_states=buffers.hidden_states.at[dp_indices, scatter_indices].set(
            hidden_states,
            mode="drop",
            out_sharding=RELAY_STATE_SPEC,
        ),
        verified_id=buffers.verified_id.at[dp_indices, scatter_indices].set(
            verified_id,
            mode="drop",
            out_sharding=RELAY_ID_SPEC,
        ),
        new_seq_lens=buffers.new_seq_lens.at[dp_indices, scatter_indices].set(
            new_seq_lens,
            mode="drop",
            out_sharding=RELAY_ID_SPEC,
        ),
    )


def update_dflash_relay_buffers(
    buffers: DFlashRelayBuffers,
    future_indices,
    valid_mask,
    verified_id,
    new_seq_lens,
    *,
    dp_size: int,
) -> DFlashRelayBuffers:
    """Publish one DP-padded DFlash round without writing padded slots."""
    per_dp_bs = future_indices.shape[0] // dp_size
    indices = future_indices.reshape((dp_size, per_dp_bs))
    valid = valid_mask.reshape((dp_size, per_dp_bs))
    dp_indices = jnp.arange(dp_size, dtype=jnp.int32)[:, None]
    scatter_indices = jnp.where(
        valid,
        indices,
        jnp.full_like(indices, buffers.verified_id.shape[1]),
    )
    verified_id = verified_id.reshape((dp_size, per_dp_bs))
    new_seq_lens = new_seq_lens.reshape((dp_size, per_dp_bs))
    return DFlashRelayBuffers(
        verified_id=buffers.verified_id.at[dp_indices, scatter_indices].set(
            verified_id,
            mode="drop",
            out_sharding=RELAY_ID_SPEC,
        ),
        new_seq_lens=buffers.new_seq_lens.at[dp_indices, scatter_indices].set(
            new_seq_lens,
            mode="drop",
            out_sharding=RELAY_ID_SPEC,
        ),
    )


def gather_spec_relay_buffers(
    buffers: SpecRelayBuffers,
    future_indices,
    *,
    dp_size: int,
):
    """Gather DP-padded draft state for the next batch."""
    per_dp_bs = future_indices.shape[0] // dp_size
    indices = future_indices.reshape((dp_size, per_dp_bs))
    dp_indices = jnp.arange(dp_size, dtype=jnp.int32)[:, None]

    return (
        buffers.topk_index.at[dp_indices, indices]
        .get(out_sharding=RELAY_STATE_SPEC)
        .reshape(future_indices.shape + buffers.topk_index.shape[2:]),
        buffers.hidden_states.at[dp_indices, indices]
        .get(out_sharding=RELAY_STATE_SPEC)
        .reshape(future_indices.shape + buffers.hidden_states.shape[2:]),
        buffers.verified_id.at[dp_indices, indices]
        .get(out_sharding=RELAY_ID_SPEC)
        .reshape(future_indices.shape),
        buffers.new_seq_lens.at[dp_indices, indices]
        .get(out_sharding=RELAY_ID_SPEC)
        .reshape(future_indices.shape),
    )


def gather_dflash_relay_buffers(
    buffers: DFlashRelayBuffers,
    future_indices,
    *,
    dp_size: int,
):
    """Gather the DFlash seed token and logical length for the next round."""
    per_dp_bs = future_indices.shape[0] // dp_size
    indices = future_indices.reshape((dp_size, per_dp_bs))
    dp_indices = jnp.arange(dp_size, dtype=jnp.int32)[:, None]
    verified_id = (
        buffers.verified_id.at[dp_indices, indices]
        .get(out_sharding=RELAY_ID_SPEC)
        .reshape(future_indices.shape)
    )
    new_seq_lens = (
        buffers.new_seq_lens.at[dp_indices, indices]
        .get(out_sharding=RELAY_ID_SPEC)
        .reshape(future_indices.shape)
    )
    flat_sharding = jax.typeof(future_indices).sharding
    if isinstance(flat_sharding, NamedSharding) and not flat_sharding.mesh.empty:
        verified_id = jax.sharding.reshard(verified_id, flat_sharding)
        new_seq_lens = jax.sharding.reshard(new_seq_lens, flat_sharding)
    return verified_id, new_seq_lens


def make_dp_valid_mask(real_bs_per_dp, *, total_bs: int, per_dp_bs: int) -> np.ndarray:
    mask = np.zeros((total_bs,), dtype=np.bool_)
    for dp_rank, real_bs in enumerate(real_bs_per_dp):
        if real_bs:
            start = dp_rank * per_dp_bs
            mask[start : start + int(real_bs)] = True
    return mask
