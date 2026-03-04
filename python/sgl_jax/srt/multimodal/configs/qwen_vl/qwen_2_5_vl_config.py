import dataclasses

from sgl_jax.srt.multimodal.configs.multimodal_base_config import MultiModalModelConfigs


@dataclasses.dataclass
class QwenVLModelVitConfig(MultiModalModelConfigs):
    """Unified vision config for Qwen2.5-VL and Qwen3-VL models."""

    model_type: str = "qwen2_5_vl"
    base_config_key: str = "vision_config"

    # Common vision encoder params
    depth: int = 32
    hidden_size: int = 3584
    hidden_act: str = "silu"
    intermediate_size: int = 3420
    num_heads: int = 16
    in_channels: int = 3
    patch_size: int = 14
    spatial_merge_size: int = 2
    temporal_patch_size: int = 2
    out_hidden_size: int = 3584
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-05
    rope_theta: float = 10000.0

    # Qwen2.5-VL specific (unused by Qwen3-VL)
    tokens_per_second: int = 4
    window_size: int = 112
    fullatt_block_indexes: list = dataclasses.field(default_factory=lambda: [7, 15, 23, 31])

    # Qwen3-VL specific (unused by Qwen2.5-VL)
    num_position_embeddings: int = 2304
    deepstack_visual_indexes: tuple = (8, 16, 24)

    # Text config params (for compatibility with weight loading)
    vocab_size: int = 151936
    text_hidden_size: int = 2048

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @property
    def layer_norm_eps(self) -> float:
        return self.rms_norm_eps
