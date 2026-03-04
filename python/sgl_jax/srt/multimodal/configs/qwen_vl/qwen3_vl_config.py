from dataclasses import dataclass, field

from sgl_jax.srt.multimodal.configs.multimodal_base_config import MultiModalModelConfigs
from sgl_jax.srt.multimodal.configs.qwen_vl.qwen_2_5_vl_config import (
    QwenVLModelVitConfig,
)


def _qwen3vl_vision(out_hidden_size: int, **kwargs) -> QwenVLModelVitConfig:
    """Create a Qwen3-VL vision config with model-specific out_hidden_size."""
    defaults = dict(
        model_type="qwen3_vl",
        hidden_act="gelu_pytorch_tanh",
        rms_norm_eps=1e-6,
    )
    defaults.update(kwargs)
    return QwenVLModelVitConfig(out_hidden_size=out_hidden_size, **defaults)


@dataclass
class Qwen3VLConfig(MultiModalModelConfigs):
    """Combined configuration for Qwen3-VL model.

    Text config is not defined here. Use HuggingFace's PretrainedConfig.text_config
    (accessed via get_hf_text_config()) like Qwen2.5-VL does.
    """

    vision_config: QwenVLModelVitConfig = field(
        default_factory=lambda: _qwen3vl_vision(out_hidden_size=2048)
    )
    image_token_id: int = 151655
    video_token_id: int = 151656
    vision_start_token_id: int = 151652
    vision_end_token_id: int = 151653

    @classmethod
    def qwen3vl_2b(cls):
        """Qwen3-VL 2B configuration."""
        return cls(
            vision_config=_qwen3vl_vision(
                depth=24,
                hidden_size=1024,
                intermediate_size=4096,
                out_hidden_size=2048,
                deepstack_visual_indexes=(5, 11, 17),
            ),
        )

    @classmethod
    def qwen3vl_4b(cls):
        """Qwen3-VL 4B configuration."""
        return cls(
            vision_config=_qwen3vl_vision(
                depth=24,
                hidden_size=1024,
                intermediate_size=4096,
                out_hidden_size=2560,
                deepstack_visual_indexes=(5, 11, 17),
            ),
        )

    @classmethod
    def qwen3vl_8b(cls):
        """Qwen3-VL 8B configuration."""
        return cls(
            vision_config=_qwen3vl_vision(
                depth=27,
                hidden_size=1152,
                intermediate_size=4304,
                out_hidden_size=4096,
                deepstack_visual_indexes=(8, 16, 24),
            ),
        )

    @classmethod
    def qwen3vl_32b(cls):
        return cls(
            vision_config=_qwen3vl_vision(
                depth=27,
                hidden_size=1152,
                intermediate_size=4304,
                out_hidden_size=5120,
                deepstack_visual_indexes=(8, 16, 24),
            ),
        )

    @classmethod
    def qwen3vl_30b_a3b(cls):
        """Qwen3-VL-30B-A3B-Thinking MoE configuration."""
        return cls(
            vision_config=_qwen3vl_vision(
                depth=27,
                hidden_size=1152,
                intermediate_size=4304,
                out_hidden_size=2048,
                deepstack_visual_indexes=(8, 16, 24),
            ),
        )
