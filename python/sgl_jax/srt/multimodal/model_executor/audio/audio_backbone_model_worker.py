"""Audio Backbone Model Worker for MiMo Audio."""

from typing import Optional

import jax

from sgl_jax.srt.multimodal.configs.audio.mimo_audio_backbone_config import MiMoSamplerConfig
from sgl_jax.srt.multimodal.manager.schedule_batch import Req
from sgl_jax.srt.multimodal.model_executor.audio.audio_backbone_model_runner import (
    AudioBackboneModelRunner,
)
from sgl_jax.srt.server_args import ServerArgs


class AudioBackboneModelWorker:
    """Worker for MiMo Audio Backbone model execution."""

    def __init__(self, server_args: ServerArgs = None, mesh=None, model_class=None):
        self.mesh = mesh
        self.model_runner = AudioBackboneModelRunner(server_args, mesh, model_class=model_class)

    def forward(
        self,
        batch: Req,
        cache: Optional[list] = None,
        **kwargs,
    ):
        """Forward pass through main transformer.

        Args:
            batch: Request batch containing input_ids
            cache: Optional KV cache

        Returns:
            (text_logits, local_hidden_states, cache), cache_miss_count
        """
        return self.model_runner.forward(batch.input_ids, cache, **kwargs)

    def patch_decode(
        self,
        local_embeds: jax.Array,
        key: jax.Array,
        sampler_config: Optional[MiMoSamplerConfig] = None,
    ):
        """Generate audio tokens for one group.

        Args:
            local_embeds: [B, 1, local_dim]
            key: Random key for sampling
            sampler_config: Sampling configuration

        Returns:
            local_tokens: [B, group_size, audio_channels], cache_miss_count
        """
        return self.model_runner.patch_decode(local_embeds, key, sampler_config)

    def init_cache(self, batch_size: int) -> list:
        """Initialize KV cache for main transformer."""
        return self.model_runner.init_cache(batch_size)
