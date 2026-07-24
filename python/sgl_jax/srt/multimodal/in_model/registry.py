"""Registry for in-model multimodal plan builders."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from sgl_jax.srt.multimodal.in_model.encoder_planning import EncoderPlanBuilder

if TYPE_CHECKING:
    from sgl_jax.srt.configs.model_config import ModelConfig


_BUILDER_FACTORIES: dict[str, Callable[[Any], EncoderPlanBuilder]] = {}


def register_encoder_plan_builder(
    arch: str,
    factory: Callable[[Any], EncoderPlanBuilder],
) -> None:
    _BUILDER_FACTORIES[arch] = factory


def resolve_encoder_plan_builder(
    model_config: ModelConfig,
    input_buckets: Any | None = None,
) -> EncoderPlanBuilder | None:
    hf_config = getattr(model_config, "hf_config", None)
    architectures = getattr(hf_config, "architectures", None) or []
    factory = _BUILDER_FACTORIES.get(architectures[0]) if architectures else None
    if factory is None:
        return None
    builder = factory(model_config)
    if input_buckets is not None:
        builder.input_buckets = tuple(input_buckets)
    return builder
