from dataclasses import dataclass

import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from sgl_jax.srt.multimodal.in_model.placement import slice_from_dp_batch


def _lpt_lanes(
    item_lengths: list[int] | tuple[int, ...],
    indices: range,
    lane_count: int,
) -> list[list[int]]:
    lanes: list[list[int]] = [[] for _ in range(lane_count)]
    loads = [0] * lane_count
    for index in sorted(indices, key=lambda i: (-item_lengths[i], i)):
        lane = min(range(lane_count), key=lambda i: (loads[i], i))
        lanes[lane].append(index)
        loads[lane] += item_lengths[index]
    return lanes


def apply_data_sharding(x: jax.Array, mesh: Mesh, spec: PartitionSpec) -> jax.Array:
    sharding = NamedSharding(mesh, spec)
    if "data" in mesh.abstract_mesh.explicit_axes:
        return jax.sharding.reshard(x, sharding)
    return jax.lax.with_sharding_constraint(x, sharding)


def resolve_encoder_tp(mesh: Mesh | None, mode: str) -> bool:
    if mode != "tp" or mesh is None:
        return False
    return "tensor" in mesh.shape and int(mesh.shape["tensor"]) > 1


def schedule_vision_lanes(
    item_lengths: list[int] | tuple[int, ...],
    *,
    data_size: int,
    tensor_size: int,
    vision_tp: bool,
    items_per_data_rank: tuple[int, ...],
) -> list[list[int]]:
    """Balance items independently inside each owner DP."""

    lane_count = 1 if vision_tp else tensor_size
    if len(items_per_data_rank) != data_size or sum(items_per_data_rank) != len(item_lengths):
        raise ValueError("vision owner groups do not match data parallel size")
    all_lanes = []
    start = 0
    for item_count in items_per_data_rank:
        all_lanes.extend(_lpt_lanes(item_lengths, range(start, start + item_count), lane_count))
        start += item_count
    return all_lanes


def slice_owner_items(
    value: jax.Array,
    mesh: Mesh,
    items_per_data_rank: tuple[int, ...],
    placements: tuple[tuple[int, int], ...],
    lengths: list[int],
    *,
    token_axis: int,
) -> list[jax.Array]:
    owners = (
        owner for owner, item_count in enumerate(items_per_data_rank) for _ in range(item_count)
    )
    return [
        slice_from_dp_batch(
            value,
            mesh,
            owner,
            placement[0],
            placement[1],
            length,
            token_axis=token_axis,
        )
        for owner, placement, length in zip(owners, placements, lengths, strict=True)
    ]


@dataclass(frozen=True)
class VisionShardSpecs:
    mesh: Mesh | None
    tp: bool

    @property
    def batch_axis(self) -> str | tuple[str, str]:
        if self.tp:
            return "data"
        if self.mesh is not None and "tensor" in self.mesh.axis_names:
            return ("data", "tensor")
        return "data"

    @property
    def head_axis(self) -> str | None:
        return "tensor" if self.tp else None

    @property
    def col_kernel_axes(self) -> tuple[None, str] | tuple[None, None]:
        return (None, "tensor") if self.tp else (None, None)

    @property
    def row_kernel_axes(self) -> tuple[str, None] | tuple[None, None]:
        return ("tensor", None) if self.tp else (None, None)

    def batch_spec(
        self,
        *tail: str | tuple[str, ...] | None,
    ) -> PartitionSpec:
        return PartitionSpec(self.batch_axis, *tail)

    def output_spec(
        self,
        *tail: str | tuple[str, ...] | None,
    ) -> PartitionSpec:
        return PartitionSpec("data", *tail)

    def batch_sharding(
        self,
        *tail: str | tuple[str, ...] | None,
    ) -> NamedSharding | None:
        if self.mesh is None:
            return None
        return NamedSharding(self.mesh, self.batch_spec(*tail))

    def col_out(self, ndim: int) -> NamedSharding | None:
        if self.mesh is None:
            return None
        if self.tp:
            spec = PartitionSpec("data", *([None] * (ndim - 2)), "tensor")
        else:
            spec = PartitionSpec(self.batch_axis, *([None] * (ndim - 1)))
        return NamedSharding(self.mesh, spec)

    def row_out(self, ndim: int) -> NamedSharding | None:
        if self.mesh is None:
            return None
        spec = PartitionSpec(self.batch_axis, *([None] * (ndim - 1)))
        return NamedSharding(self.mesh, spec)

    def qkv_reshape_sharding(self) -> NamedSharding | None:
        if self.mesh is None:
            return None
        spec = (
            PartitionSpec("data", None, "tensor", None)
            if self.tp
            else PartitionSpec(self.batch_axis, None, None, None)
        )
        return NamedSharding(self.mesh, spec)
