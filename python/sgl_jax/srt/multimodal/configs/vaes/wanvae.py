# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass

from sgl_jax.srt.multimodal.configs.vaes.vae_base import VAEConfig


@dataclass
class WanVAEConfig(VAEConfig):
    use_feature_cache: bool = True
    use_tiling: bool = False
    use_temporal_tiling: bool = False
    use_parallel_tiling: bool = False
    model_class: None = None

    def __post_init__(self):
        self.blend_num_frames = (
            self.tile_sample_min_num_frames - self.tile_sample_stride_num_frames
        ) * 2
