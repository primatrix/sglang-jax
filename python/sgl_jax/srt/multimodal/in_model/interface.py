from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import jax
from jax.sharding import Mesh
from jax.typing import ArrayLike

from sgl_jax.srt.multimodal.common.modality_enum import Modality, MultimodalDataItem


@dataclass(frozen=True)
class MultimodalEmbeddingOutput:
    embeddings: ArrayLike | Sequence[ArrayLike]
    deepstack: ArrayLike | Sequence[ArrayLike] | None = None


MultimodalEmbedding = ArrayLike | Sequence[ArrayLike] | MultimodalEmbeddingOutput


MultimodalEmbeddingFunc = Callable[[list[MultimodalDataItem]], MultimodalEmbedding]
MultimodalEmbeddingFuncs = Mapping[Modality, MultimodalEmbeddingFunc]
MultimodalItemGroups = tuple[tuple[MultimodalDataItem, ...], ...]
PackedMultimodalEmbeddingFunc = Callable[[MultimodalItemGroups], MultimodalEmbedding]
PackedMultimodalEmbeddingFuncs = Mapping[Modality, PackedMultimodalEmbeddingFunc]


class InModelMultimodalContract(ABC):
    mesh: Mesh | None = None
    deepstack_visual_layers: int = 0

    def precompile_multimodal(self) -> None:
        """Warm model-specific multimodal encoders."""
        pass

    @abstractmethod
    def get_input_embeddings(self) -> Callable[[jax.Array], jax.Array]:
        raise NotImplementedError

    def get_multimodal_embedding_funcs(self) -> MultimodalEmbeddingFuncs:
        funcs: dict[Modality, MultimodalEmbeddingFunc] = {}
        image_feature = getattr(self, "get_image_feature", None)
        if image_feature is not None:
            funcs.update(
                {
                    Modality.IMAGE: image_feature,
                    Modality.MULTI_IMAGES: image_feature,
                }
            )
        video_feature = getattr(self, "get_video_feature", None)
        if video_feature is not None:
            funcs[Modality.VIDEO] = video_feature
        audio_feature = getattr(self, "get_audio_feature", None)
        if audio_feature is not None:
            funcs[Modality.AUDIO] = audio_feature
        return funcs

    def get_packed_multimodal_embedding_funcs(self) -> PackedMultimodalEmbeddingFuncs:
        """Optional owner-aware, packed single-dispatch encoder fast paths."""
        return {}
