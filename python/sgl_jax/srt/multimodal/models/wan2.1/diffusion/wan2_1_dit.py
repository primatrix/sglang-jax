"""Wan2.1-T2V-1.3B: Text-to-Video Diffusion Transformer Model.

This implements the Wan2.1-T2V-1.3B model, a 1.3B parameter diffusion transformer
for text-to-video generation using Flow Matching framework.

Architecture:
- 30-layer Diffusion Transformer with 1536 hidden dim
- 12 attention heads (128 dim each)
- Vision self-attention + text cross-attention
- AdaLN modulation conditioned on timestep
- UMT5 text encoder for multilingual prompts
- Wan-VAE for video encoding/decoding
"""

import dataclasses
import math
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from flax import nnx
from jax.lax import Precision
from jaxtyping import Array


class Wan2DiT(nnx.Module):
    """
    Wan2.1-T2V-1.3B Diffusion Transformer.
    """

    def __init__(self, cfg: TransformerWanModelConfig, *, rngs: nnx.Rngs):
        self.cfg = cfg


        self.patch_embed = nnx.Conv(
            in_features=cfg.latent_input_dim,
            out_features=cfg.hidden_dim,
            kernel_size=(1, 2, 2),
            strides=(1, 2, 2),
            dtype=cfg.dtype,
            rngs=rngs,
            param_dtype=cfg.weights_dtype,
            precision=cfg.precision,
            kernel_init=nnx.initializers.xavier_uniform(),
        )

        # Text embedding projection: UMT5 (4096) → DiT (1536)
        self.text_proj = nnx.Sequential(
            nnx.Linear(cfg.text_embed_dim, cfg.hidden_dim, rngs=rngs, precision=cfg.precision),
            nnx.gelu,
            nnx.Linear(cfg.hidden_dim, cfg.hidden_dim, rngs=rngs, precision=cfg.precision),
        )

        self.time_embed = TimestepEmbedding(cfg, rngs=rngs)

        self.blocks = nnx.List([WanAttentionBlock(cfg, rngs=rngs) for _ in range(cfg.num_layers)])

        self.final_layer = FinalLayer(cfg, rngs=rngs)

    @jax.named_scope("wan2_dit")
    @jax.jit
    def forward(
        self,
        latents: Array,
        text_embeds: Array,
        timestep: Array,
        deterministic: bool = True,
        debug: bool = True,
    ) -> Array:
        """
        Forward pass of the Diffusion Transformer.

        Args:
            latents: [B, T, H, W, C] noisy video latents from VAE
            text_embeds: [B, seq_len, 4096] from UMT5-XXL encoder (before projection)
            timestep: [B] diffusion timestep (0 to num_steps)
            deterministic: Whether to apply dropout

        Returns:
            predicted_noise: [B, T, H, W, C] predicted noise
        """
        # step_state = {"i": 0}

        text_embeds = self.text_proj(text_embeds)
        # _log_stats("text_proj", text_embeds, step_state, debug)

        # Get time embeddings
        # time_emb: [B, D] for FinalLayer
        # time_proj: [B, 6*D] for AdaLN in blocks
        time_emb, time_proj = self.time_embed(timestep)

        nnx.display(self.patch_embed)

        x = self.patch_embed(latents)
        b, t_out, h_out, w_out, d = x.shape
        x = x.reshape(b, t_out * h_out * w_out, d)

        grid_sizes = (t_out, h_out, w_out)

        max_seq = max(grid_sizes)
        rope_freqs = tuple(
            jax.lax.stop_gradient(arr) for arr in precompute_freqs_cis_3d(dim=self.cfg.head_dim, max_seq_len=max_seq)
        )

        for block_idx, block in enumerate(self.blocks):
            x = block(x, text_embeds, time_proj, rope_state=(rope_freqs, grid_sizes), deterministic=deterministic)

        # Final projection to noise space
        x = self.final_layer(x, time_emb)  # [B, T*H*W, latent_output_dim]

        # Reshape back to video format
        predicted_noise = self.unpatchify(x, grid_sizes)

        return predicted_noise

    def unpatchify(self, x: Array, grid_sizes: tuple[int, int, int]) -> Array:
        """
        Reconstruct video tensors from patch embeddings.

        Args:
            x: [B, T*H*W, patch_t*patch_h*patch_w*C] flattened patch embeddings
            grid_sizes: (T_patches, H_patches, W_patches) grid dimensions

        Returns:
            [B, T, H, W, C] reconstructed video tensor (channel-last)
        """
        b, seq_len, feature_dim = x.shape
        t_patches, h_patches, w_patches = grid_sizes
        c = self.cfg.latent_output_dim
        patch_size = self.cfg.patch_size

        assert seq_len == t_patches * h_patches * w_patches, (
            f"expected: seq_len={seq_len} should be {t_patches * h_patches * w_patches}"
        )
        assert feature_dim == patch_size[0] * patch_size[1] * patch_size[2] * c, (
            f"expected: feature_dim={feature_dim} should be {patch_size[0] * patch_size[1] * patch_size[2] * c}"
        )

        x = x.reshape(
            b,
            t_patches,
            h_patches,
            w_patches,
            patch_size[0],
            patch_size[1],
            patch_size[2],
            c,
        )
        x = jnp.einsum("bthwpqrc->btphqwrc", x)
        x = x.reshape(
            b,
            t_patches * patch_size[0],
            h_patches * patch_size[1],
            w_patches * patch_size[2],
            c,
        )

        return x


__all__ = [
    "TransformerWanModelConfig",
    "Wan2DiT",
]
