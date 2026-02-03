"""Audio Backbone Model Worker for MiMo Audio."""

import logging
from typing import Optional

import jax
import jax.numpy as jnp

from sgl_jax.srt.multimodal.configs.audio.mimo_audio_backbone_config import MiMoSamplerConfig
from sgl_jax.srt.multimodal.manager.schedule_batch import Req
from sgl_jax.srt.multimodal.model_executor.audio.audio_backbone_model_runner import (
    AudioBackboneModelRunner,
)
from sgl_jax.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


class AudioBackboneModelWorker:
    """Worker for MiMo Audio Backbone model execution."""

    def __init__(self, server_args: ServerArgs = None, mesh=None, model_class=None):
        self.mesh = mesh
        self.model_runner = AudioBackboneModelRunner(server_args, mesh, model_class=model_class)

    def _prepare_interleaved_input(
        self,
        input_ids: jax.Array,
        audio_codes: jax.Array,
    ) -> jax.Array:
        """Combine text input_ids and audio_codes into interleaved format.

        Args:
            input_ids: Text token IDs [seq_len] or [B, seq_len]
            audio_codes: Audio codes from encoder [n_q, seq_len] or [B, n_q, seq_len]

        Returns:
            Interleaved input [B, 1 + audio_channels, seq_len]
        """
        if input_ids.ndim == 1:
            input_ids = input_ids[None, :]
        if audio_codes.ndim == 2:
            audio_codes = audio_codes[None, :]

        B = input_ids.shape[0]
        text_seq_len = input_ids.shape[1]
        audio_channels = audio_codes.shape[1]
        audio_seq_len = audio_codes.shape[2]

        seq_len = max(text_seq_len, audio_seq_len)

        if text_seq_len < seq_len:
            pad_len = seq_len - text_seq_len
            input_ids = jnp.pad(input_ids, ((0, 0), (0, pad_len)), constant_values=0)

        if audio_seq_len < seq_len:
            pad_len = seq_len - audio_seq_len
            audio_codes = jnp.pad(audio_codes, ((0, 0), (0, 0), (0, pad_len)), constant_values=0)

        text_channel = input_ids[:, None, :]
        interleaved = jnp.concatenate([text_channel, audio_codes], axis=1)

        return interleaved

    def forward(
        self,
        batch: Req,
        cache: Optional[list] = None,
        **kwargs,
    ):
        """Forward pass through main transformer.

        Args:
            batch: Request batch containing input_ids and audio_codes
            cache: Optional KV cache

        Returns:
            (text_logits, local_hidden_states, cache), cache_miss_count
        """
        logger.info(
            "AudioBackboneModelWorker.forward: input_ids=%s, audio_codes=%s",
            batch.input_ids is not None,
            batch.audio_codes is not None,
        )

        if batch.audio_codes is not None and batch.input_ids is not None:
            input_ids = batch.input_ids
            if not jnp.issubdtype(input_ids.dtype, jnp.integer):
                input_ids = input_ids.astype(jnp.int32)
            interleaved_input = self._prepare_interleaved_input(
                input_ids, batch.audio_codes
            )
        elif batch.audio_codes is not None:
            audio_codes = batch.audio_codes
            if audio_codes.ndim == 2:
                audio_codes = audio_codes[None, :]
            B = audio_codes.shape[0]
            audio_channels = audio_codes.shape[1]
            seq_len = audio_codes.shape[2]
            empty_text = jnp.zeros((B, 1, seq_len), dtype=jnp.int32)
            if not jnp.issubdtype(audio_codes.dtype, jnp.integer):
                audio_codes = audio_codes.astype(jnp.int32)
            interleaved_input = jnp.concatenate([empty_text, audio_codes], axis=1)
        elif batch.input_ids is not None:
            input_ids = batch.input_ids
            if not jnp.issubdtype(input_ids.dtype, jnp.integer):
                input_ids = input_ids.astype(jnp.int32)
            if input_ids.ndim == 1:
                interleaved_input = input_ids[None, None, :]
            elif input_ids.ndim == 2:
                interleaved_input = input_ids[:, None, :]
            else:
                interleaved_input = input_ids
        else:
            raise ValueError("Either input_ids or audio_codes must be provided")

        group_size = 4
        seq_len = interleaved_input.shape[2]
        if seq_len % group_size != 0:
            pad_len = group_size - (seq_len % group_size)
            interleaved_input = jnp.pad(
                interleaved_input,
                ((0, 0), (0, 0), (0, pad_len)),
                constant_values=0,
            )
            logger.info(
                "Padded interleaved_input from seq_len=%d to %d",
                seq_len,
                interleaved_input.shape[2],
            )

        return self.model_runner.forward(interleaved_input, cache, **kwargs)

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
