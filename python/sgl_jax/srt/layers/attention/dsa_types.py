"""Shared pytree types for DeepSeek Sparse Attention selection state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from jax.tree_util import register_pytree_node_class


def _validate_int32_array(
    name: str,
    value: jax.Array,
    *,
    rank: int,
    mode: Literal["decode", "prefill"],
) -> None:
    if value.ndim != rank:
        raise ValueError(f"{mode} {name} must have rank {rank}; got {value.ndim}")
    if value.dtype != jnp.int32:
        raise TypeError(f"{mode} {name} must have dtype int32; got {value.dtype}")


@register_pytree_node_class
@dataclass
class DsaSelection:
    """Selected token slots produced by one DSA indexer layer."""

    physical_slots: jax.Array
    selected_counts: jax.Array
    producer_layer: int
    logical_topk_ids: jax.Array | None = None

    def validate(self, *, mode: Literal["decode", "prefill"]) -> None:
        """Validate eager selection metadata for a decode or prefill call."""
        if mode not in ("decode", "prefill"):
            raise ValueError(f"unsupported DSA selection mode: {mode!r}")
        if type(self.producer_layer) is not int:
            raise TypeError("producer_layer must be a Python int")
        if self.producer_layer < 0:
            raise ValueError("producer_layer must be nonnegative")
        _validate_int32_array(
            "physical_slots", self.physical_slots, rank=2, mode=mode
        )
        _validate_int32_array(
            "selected_counts", self.selected_counts, rank=1, mode=mode
        )
        if self.logical_topk_ids is not None:
            _validate_int32_array(
                "logical_topk_ids", self.logical_topk_ids, rank=2, mode=mode
            )
            if self.logical_topk_ids.shape != self.physical_slots.shape:
                raise ValueError(
                    f"{mode} logical_topk_ids must match physical_slots shape "
                    f"{self.physical_slots.shape}; got {self.logical_topk_ids.shape}"
                )
        expected_counts_shape = (self.physical_slots.shape[0],)
        if self.selected_counts.shape != expected_counts_shape:
            raise ValueError(
                f"{mode} selected_counts must have shape {expected_counts_shape}; "
                f"got {self.selected_counts.shape}"
            )
        topk_width = self.physical_slots.shape[1]
        selected_counts = np.asarray(self.selected_counts)
        if np.any((selected_counts < 0) | (selected_counts > topk_width)):
            raise ValueError(
                f"{mode} selected_counts entries must be in [0, {topk_width}]"
            )

    def valid_mask(self) -> jax.Array:
        """Return the per-row validity mask without interpreting slot values."""
        topk_positions = jnp.arange(
            self.physical_slots.shape[1], dtype=self.selected_counts.dtype
        )
        return topk_positions[None, :] < self.selected_counts[:, None]

    def tree_flatten(self):
        children = (
            self.physical_slots,
            self.selected_counts,
            self.logical_topk_ids,
        )
        return children, self.producer_layer

    @classmethod
    def tree_unflatten(cls, producer_layer, children):
        physical_slots, selected_counts, logical_topk_ids = children
        return cls(
            physical_slots=physical_slots,
            selected_counts=selected_counts,
            producer_layer=producer_layer,
            logical_topk_ids=logical_topk_ids,
        )


@register_pytree_node_class
@dataclass
class DsaTopKState:
    """Producer selection and ragged request boundaries for IndexShare."""

    selection: DsaSelection
    query_offsets: jax.Array
    request_offsets: jax.Array

    def validate(self, *, mode: Literal["decode", "prefill"]) -> None:
        """Validate eager IndexShare state for a decode or prefill call."""
        self.selection.validate(mode=mode)
        _validate_int32_array(
            "query_offsets", self.query_offsets, rank=1, mode=mode
        )
        _validate_int32_array(
            "request_offsets", self.request_offsets, rank=1, mode=mode
        )

    def tree_flatten(self):
        children = (self.selection, self.query_offsets, self.request_offsets)
        return children, None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del aux_data
        selection, query_offsets, request_offsets = children
        return cls(
            selection=selection,
            query_offsets=query_offsets,
            request_offsets=request_offsets,
        )
