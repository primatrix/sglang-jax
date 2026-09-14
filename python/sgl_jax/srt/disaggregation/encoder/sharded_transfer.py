"""Keep encoder lanes local and describe their logical token order to the receiver."""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field

import jax
import numpy as np

from sgl_jax.srt.disaggregation.encoder.raiden_pool import RaidenPool
from sgl_jax.srt.disaggregation.encoder.raiden_transfer import (
    RaidenEncoderServerTransfer,
    _SendReservation,
)
from sgl_jax.srt.disaggregation.encoder.transfer_layout import ENCODER_PAGE_SIZE
from sgl_jax.srt.multimodal.in_model.lane_packing import encoder_num_lanes


@dataclass(slots=True)
class _ShardPart:
    rank: int
    reservation: _SendReservation
    token_indices: np.ndarray
    source_rows: np.ndarray


@dataclass(slots=True)
class _ShardedReservation:
    transfer_id: str
    parts: list[_ShardPart] = field(default_factory=list)


class ShardedEncoderTransfer:
    """One fixed pool per encoder lane; all addressing remains runtime metadata."""

    def __init__(self, host_ip, server_args, model_config, model):
        visual = model.visual
        self.num_lanes = encoder_num_lanes(model.mesh, visual.vision_tp)
        sharding = visual.specs.sharding(visual.specs.batch_axis)
        # Tensor-parallel lanes may have replicated outputs. Register one physical
        # replica per lane, rather than copying that replica to the other lanes.
        devices = {}
        for device, index in sharding.addressable_devices_indices_map((self.num_lanes,)).items():
            devices.setdefault(index[0].start or 0, device)
        self.devices = tuple(devices[rank] for rank in range(self.num_lanes))
        vision = model_config.hf_config.vision_config
        width = model_config.hidden_size * (
            1 + len(getattr(vision, "deepstack_visual_indexes", ()))
        )
        pages = math.ceil(
            server_args.encoder_transfer_max_tokens / (ENCODER_PAGE_SIZE * self.num_lanes)
        )
        self.pools = [
            RaidenPool(
                (ENCODER_PAGE_SIZE, width),
                model_config.dtype,
                jax.sharding.SingleDeviceSharding(device),
                capacity=pages,
                max_batch_size=server_args.encoder_max_batch_size,
            )
            for device in self.devices
        ]
        if not server_args.disable_precompile:
            for capacity in model.get_multimodal_embedding_packed_capacities():
                for pool in self.pools:
                    pool.warmup(capacity // self.num_lanes)
        self.senders = [
            RaidenEncoderServerTransfer(
                host_ip,
                pool,
                parallelism=server_args.disaggregation_channel_number,
                pool_size=server_args.encoder_transfer_pool_size,
                timeout_s=server_args.encoder_request_timeout_seconds,
            )
            for pool in self.pools
        ]
        self._reservations: dict[str, _ShardedReservation] = {}
        self._lock = threading.Lock()

    def batch_capacity(self, token_count):
        return min(sender.batch_capacity(token_count) for sender in self.senders)

    def reserve_batch_sync(self, transfer_ids, token_counts, *, output_indices):
        active = []
        for sender in self.senders:
            sender._reap_completed()
            with sender._lock:
                active.append(set(sender._reservations))
        with self._lock:
            self._reservations = {
                key: value
                for key, value in self._reservations.items()
                if any(part.reservation.transfer_id in active[part.rank] for part in value.parts)
            }
            if len(set(transfer_ids)) != len(transfer_ids) or any(
                key in self._reservations for key in transfer_ids
            ):
                raise ValueError("duplicate Raiden transfer_id")
        local_capacity = output_indices.size // self.num_lanes
        parents = [_ShardedReservation(key) for key in transfer_ids]
        try:
            for rank, sender in enumerate(self.senders):
                plans = []
                offset = 0
                for parent, count in zip(parents, token_counts, strict=True):
                    rows = output_indices[offset : offset + count]
                    positions = np.flatnonzero(rows // local_capacity == rank).astype(np.int32)
                    if positions.size:
                        plans.append((parent, positions, rows[positions] % local_capacity))
                    offset += count
                children = sender.reserve_batch_sync(
                    [f"{parent.transfer_id}:lane:{rank}" for parent, _, _ in plans],
                    tuple(len(positions) for _, positions, _ in plans),
                )
                for (parent, positions, rows), child in zip(plans, children, strict=True):
                    parent.parts.append(_ShardPart(rank, child, positions, rows))
            with self._lock:
                self._reservations.update((parent.transfer_id, parent) for parent in parents)
            return parents
        except BaseException:
            self.cancel_batch(parents)
            raise

    def stage_packed_batch_sync(self, reservations, packed):
        shards = {shard.device: shard.data for shard in packed.addressable_shards}
        try:
            for rank, sender in enumerate(self.senders):
                parts = [
                    part for parent in reservations for part in parent.parts if part.rank == rank
                ]
                if parts:
                    sender.stage_packed_batch_sync(
                        [part.reservation for part in parts],
                        shards[self.devices[rank]],
                        source_rows=np.concatenate([part.source_rows for part in parts]),
                    )
            return reservations
        except BaseException:
            self.cancel_batch(reservations)
            raise

    def publish_batch_sync(self, reservations):
        metadata = []
        try:
            for parent in reservations:
                parts = []
                for part in parent.parts:
                    child = self.senders[part.rank].publish_batch_sync([part.reservation])[0]
                    parts.append(child | {"token_indices": part.token_indices.tolist()})
                metadata.append(
                    {
                        "transfer_id": parent.transfer_id,
                        "transfer_page_size": self.pools[0].page_size,
                        "transfer_parts": parts,
                    }
                )
            return metadata
        except BaseException:
            self.cancel_batch(reservations)
            raise

    def cancel_batch(self, reservations):
        for parent in reservations:
            for part in parent.parts:
                self.senders[part.rank].cancel_batch([part.reservation])
            with self._lock:
                self._reservations.pop(parent.transfer_id, None)

    def release(self, transfer_id):
        with self._lock:
            parent = self._reservations.pop(transfer_id, None)
        if parent is not None:
            for part in parent.parts:
                self.senders[part.rank].release(part.reservation.transfer_id)

    def close(self):
        for sender in self.senders:
            sender.close()
        self._reservations.clear()
