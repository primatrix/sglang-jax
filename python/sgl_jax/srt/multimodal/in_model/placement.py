from __future__ import annotations

from functools import cache

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from jax.typing import ArrayLike


@cache
def dp_submesh(mesh: Mesh, dp_rank: int) -> Mesh:
    if "data" not in mesh.axis_names:
        if dp_rank != 0:
            raise ValueError(f"dp_rank={dp_rank} is invalid for mesh without a data axis")
        return mesh

    data_axis = mesh.axis_names.index("data")
    if not 0 <= dp_rank < int(mesh.shape["data"]):
        raise ValueError(f"invalid dp_rank={dp_rank}")
    devices = np.asarray(np.take(mesh.devices, dp_rank, axis=data_axis))
    axis_names = tuple(name for name in mesh.axis_names if name != "data")
    axis_types = tuple(
        axis_type
        for name, axis_type in zip(mesh.axis_names, mesh.axis_types, strict=True)
        if name != "data"
    )
    return Mesh(devices, axis_names, axis_types=axis_types)


@cache
def dp_local_replicated_sharding(mesh: Mesh, dp_rank: int) -> NamedSharding:
    return NamedSharding(dp_submesh(mesh, dp_rank), PartitionSpec())


@cache
def _single_device_mesh(device: jax.Device) -> Mesh:
    return Mesh(np.asarray(device), ())


def place_on_dp(value: ArrayLike, mesh: Mesh | None, dp_rank: int) -> jax.Array:
    if mesh is None:
        return jnp.asarray(value)

    sharding = dp_local_replicated_sharding(mesh, dp_rank)
    if isinstance(value, jax.Array):
        if value.sharding.is_fully_replicated and value.sharding.device_set == sharding.device_set:
            return value
        if not value.sharding.is_fully_replicated:
            return jax.device_put(value, sharding)

        shards = {shard.device: shard.data for shard in value.addressable_shards}
        devices = tuple(sharding.addressable_devices)
        if not all(device in shards for device in devices):
            if not value.is_fully_addressable:
                raise ValueError("owner DP has no addressable replica of the encoder output")
            replica = value.addressable_shards[0].data
            shards.update((device, jax.device_put(replica, device)) for device in devices)
        buffers = [shards[device] for device in devices]
        shape, dtype = value.shape, value.dtype
    else:
        value = np.asarray(value)
        devices = tuple(sharding.addressable_devices)
        buffers = [jax.device_put(value, device) for device in devices]
        shape, dtype = value.shape, value.dtype

    return jax.make_array_from_single_device_arrays(
        shape,
        sharding,
        buffers,
        dtype=dtype,
    )


def slice_from_dp_batch(
    value: jax.Array,
    mesh: Mesh,
    owner_dp: int,
    lane: int,
    start: int,
    length: int,
    *,
    token_axis: int,
) -> jax.Array:
    dp_size = int(mesh.shape.get("data", 1))
    lanes_per_dp, remainder = divmod(value.shape[0], dp_size)
    if remainder:
        raise ValueError("encoder output lanes must divide the data parallel size")
    local_lane = lane - owner_dp * lanes_per_dp
    if not 0 <= local_lane < lanes_per_dp:
        raise ValueError("encoder item lane does not belong to its owner DP")

    index = [slice(None)] * value.ndim
    index[0] = local_lane
    index[token_axis] = slice(start, start + length)
    output_shape = list(value.shape[1:])
    output_shape[token_axis - 1] = length

    sharding = dp_local_replicated_sharding(mesh, owner_dp)
    shards = {shard.device: shard.data for shard in value.addressable_shards}
    buffers = []
    for device in sharding.addressable_devices:
        with jax.set_mesh(_single_device_mesh(device)):
            buffers.append(shards[device][tuple(index)])
    return jax.make_array_from_single_device_arrays(
        tuple(output_shape),
        sharding,
        buffers,
        dtype=value.dtype,
    )
