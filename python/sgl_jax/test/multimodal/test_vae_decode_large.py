import os
import unittest

import jax
import jax.numpy as jnp
from flax import nnx

from sgl_jax.srt.multimodal.configs.vaes.wan_vae_config import WanVAEConfig
from sgl_jax.srt.multimodal.models.wan.vaes.wanvae import AutoencoderKLWan


class TestWanVaeDecodeLarge(unittest.TestCase):
    @unittest.skipUnless(
        os.getenv("RUN_LARGE_VAE_TEST") == "1",
        "Set RUN_LARGE_VAE_TEST=1 to run the large VAE decode test.",
    )
    def test_decode_480x832_73f(self):
        config = WanVAEConfig()
        model = AutoencoderKLWan(config, dtype=jnp.bfloat16)

        model_def, model_state = nnx.split(model)
        model_state_leaves, model_state_def = jax.tree_util.tree_flatten(model_state)

        @jax.jit
        def decode(latents):
            state = jax.tree_util.tree_unflatten(model_state_def, model_state_leaves)
            m = nnx.merge(model_def, state)
            return m.decode(latents)

        b = 1
        num_frames = 73
        height = 480
        width = 832
        t = (num_frames - 1) // config.scale_factor_temporal + 1
        w = width // config.scale_factor_spatial
        h = height // config.scale_factor_spatial
        # Latents layout follows pipeline: [B, T, W, H, C]
        latents = jnp.zeros((b, t, w, h, config.z_dim), dtype=jnp.float32)
        out = decode(latents)

        expected_t = (t - 1) * config.scale_factor_temporal + 1
        self.assertEqual(out.shape[0], b)
        self.assertEqual(out.shape[1], expected_t)
        self.assertEqual(out.shape[2], w * config.scale_factor_spatial)
        self.assertEqual(out.shape[3], h * config.scale_factor_spatial)
        self.assertEqual(out.shape[4], config.out_channels)


if __name__ == "__main__":
    unittest.main()
