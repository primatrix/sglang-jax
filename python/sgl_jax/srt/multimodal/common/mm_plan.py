"""Containers for scheduler-built in-model multimodal encode/merge plans.

These are modality-agnostic: the encode inputs are an opaque pytree and the
merge arrays are plain index tensors, so the host embed/merge path never needs
to know what modality produced them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, Self

from sgl_jax.srt.multimodal.common.modality_enum import Modality

if TYPE_CHECKING:
    import jax
    import numpy as np


class ModalityEncodeInputs(Protocol):
    """Registered pytree whose array leaves share leading ``[dp,tp]`` axes."""

    def tree_flatten(self) -> tuple[tuple[Any, ...], Any]: ...

    @classmethod
    def tree_unflatten(cls, aux_data: Any, children: tuple[Any, ...]) -> Self: ...


@dataclass
class DeviceMergePlan:
    """Fixed-shape merge arrays routing encoder rows to token embedding rows."""

    src_idx: np.ndarray | jax.Array  # [dp, tp, merge_bucket]
    dst_idx: np.ndarray | jax.Array  # [dp, tp, merge_bucket]
    mask: np.ndarray | jax.Array  # [dp, tp, merge_bucket]


@dataclass
class ModalityEmbedBatch:
    """One fixed-shape encoder invocation and its token merge routing."""

    encode_inputs: ModalityEncodeInputs
    merge: DeviceMergePlan


# One encoder batch per modality for one language-model forward.
MultimodalEmbedPlan = dict[Modality, ModalityEmbedBatch]
