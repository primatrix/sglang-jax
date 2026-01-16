import queue
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.managers.communication import QueueBackend
from sgl_jax.srt.multimodal.common.ServerArgs import MultimodalServerArgs
from sgl_jax.srt.multimodal.manager.schedule_batch import Req
from sgl_jax.srt.multimodal.manager.scheduler.vae_scheduler import VaeScheduler
from sgl_jax.srt.multimodal.models.wan2_1.vaes.wanvae import AutoencoderKLWan as JaxWan
import torch
from diffusers import AutoencoderKLWan
from sgl_jax.srt.multimodal.configs.vaes.wan_vae_config import WanVAEConfig
from sgl_jax.srt.configs.model_config import ModelConfig

class TestWanVaePrecision(unittest.TestCase):
    """Test VaeScheduler full load and forward flow."""

    @classmethod
    def setUpClass(cls):
        cls.mesh = jax.sharding.Mesh(jax.devices(), axis_names=("data",))
        cls.server_args = MultimodalServerArgs(
            model_path="/models/Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        )
        cls.vae = AutoencoderKLWan.from_pretrained(
            cls.server_args.model_path,
            subfolder="vae",
            torch_dtype=torch.float32,
        )
        cls.vae.eval()

        cls.jax_vae = JaxWan(WanVAEConfig(), mesh=cls.mesh)
        cls.jax_vae.load_weights(model_config=ModelConfig(model_path=cls.server_args.model_path + "/vae"))

    def _get_diffusers_encode_output(self):
        input = (
            torch.tensor(np.arange(1 * 5 * 192 * 192 * 3), dtype=torch.float32)
            .reshape(1, 5, 192, 192, 3)
            .permute((0, 4, 1, 2, 3))
        )
        latents = self.vae.encode(input)
        print(latents.latent_dist.parameters.shape)
        with open("wan_vae_diffusers_encode_output.npy", "wb") as f:
            np.save(f, latents.latent_dist.parameters.detach().numpy())

        return latents.latent_dist.parameters.detach().numpy()
    
    def _get_jax_encode_output(self,):
        input = jnp.array(np.arange(1 * 5 * 192 * 192 * 3), dtype=jnp.float32).reshape(1, 5, 192, 192, 3)
        latents = self.jax_vae.encode(input)
        print(latents.parameters.shape)
        return latents.parameters.transpose((0, 4, 1, 2, 3))
    
    def _get_transformer_decode_output(self):
        latents = (
            torch.tensor(np.arange(1 * 5 * 3 * 4 * 16), dtype=torch.float32)
            .reshape(1, 5, 3, 4, 16)
            .permute((0, 4, 1, 2, 3))
        )
        y = self.vae.decode(latents)
        print(y.sample.shape)
        with open("wan_vae_diffusers_decode_output.npy", "wb") as f:
            np.save(f, y.sample.detach().numpy())
        return y.sample.detach().numpy()
    
    def _get_jax_decode_output(self):
        latents = jnp.array(np.arange(1 * 5 * 3 * 4 * 16), dtype=jnp.float32).reshape(1, 5, 3, 4, 16)
        y = self.jax_vae.decode(latents)
        print(y.shape)
        return y.transpose((0, 4, 1, 2, 3))
    
    def test_encode_precision(self):
        torch_output = self._get_transformer_encode_output()
        jax_output = self._get_jax_encode_output()
        np.testing.assert_allclose(torch_output, jax_output, rtol=1e-5, atol=1e-5)

    def test_decode_precision(self):
        torch_output = self._get_transformer_decode_output()
        jax_output = self._get_jax_decode_output()
        np.testing.assert_allclose(torch_output, jax_output, rtol=1e-5, atol=1e-5)

if __name__ == "__main__":
    unittest.main()
