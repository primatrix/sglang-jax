"""Per-architecture in-model vision adapters."""

from sgl_jax.srt.models.vision_metadata.qwen2_5_vl import (
    Qwen25VLVisionEncodeInputs,
    Qwen25VLVisionMetadata,
    Qwen25VLVisionMetadataBuilder,
    Qwen25VLVisionPlugin,
    register_qwen25vl_vision_encoder,
)

__all__ = [
    "Qwen25VLVisionEncodeInputs",
    "Qwen25VLVisionMetadata",
    "Qwen25VLVisionMetadataBuilder",
    "Qwen25VLVisionPlugin",
    "register_qwen25vl_vision_encoder",
]
