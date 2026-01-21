import dataclasses
import unittest
from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
from flax import nnx

from sgl_jax.srt.multimodal.model_executor.vae.vae_model_runner import VaeModelRunner
from sgl_jax.srt.multimodal.common.ServerArgs import MultimodalServerArgs
from sgl_jax.srt.multimodal.models.wan2_1.vaes.wanvae import AutoencoderKLWan, WanVAEConfig

@dataclasses.dataclass
class SmallWanVAEConfig(WanVAEConfig):
    """Minimal configuration for unit testing."""
    use_feature_cache: bool = True
    use_tiling: bool = False
    use_temporal_tiling: bool = False
    use_parallel_tiling: bool = False
    model_class: None = None
    base_dim: int = 96  # Base channels in the first layer
    decoder_base_dim: int | None = None
    z_dim: int = 16  # Dimensionality of the latent space
    dim_mult: tuple[int, ...] = (1, 2, 4, 4)  # Multipliers for each block
    num_res_blocks: int = 2 
    attn_scales: tuple[float, ...] = ()
    temperal_downsample: tuple[bool, ...] = (False, True, True)
    dropout: float = 0.0
    latents_mean: tuple[float, ...] = (
        -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
        0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
    )
    latents_std: tuple[float, ...] = (
        2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
        3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160,
    )
    is_residual: bool = False
    in_channels: int = 3
    out_channels: int = 3
    patch_size: int | None = None
    scale_factor_temporal: int = 4
    scale_factor_spatial: int = 8
    clip_output: bool = True


class MockModelLoader:
    """Mocks the model loading process to avoid disk I/O and large weights."""
    def __init__(self, model_class):
        self.model_class = model_class

    def load_model(self, model_config):
        return self.model_class(model_config)


class TestVaeModelRunner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("\n>>> Initializing test environment...")
        cls.mesh = jax.sharding.Mesh(jax.devices(), ("data",))
        cls.server_args = MultimodalServerArgs(
            model_path="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
            download_dir="/tmp",
        )

        with patch("sgl_jax.srt.multimodal.model_executor.vae.vae_model_runner.get_model_loader") as mock_get_loader:
            mock_get_loader.return_value = MockModelLoader(AutoencoderKLWan)

            with patch.object(AutoencoderKLWan, "get_config_class", return_value=SmallWanVAEConfig):
                cls.runner = VaeModelRunner(
                    server_args=cls.server_args,
                    mesh=cls.mesh,
                    model_class=AutoencoderKLWan,
                )
        print(">>> VaeModelRunner initialization complete.")

    def test_encode_decode_flow(self):
        """Test the full encoding and decoding pipeline."""

        input_shape = (1, 1, 32, 32, 3) 
        x = jnp.ones(input_shape, dtype=jnp.float32)

        print(f"\nTesting Encode mode, Input shape: {x.shape}")


        posterior, miss_enc = self.runner.forward(x, mode="encode")

        # Extract latent tensor from the distribution
        # Use .mode() (mean) for deterministic results in unit tests
        latent = posterior.mode() 

        print(f"Encode successful! Extracted Latent shape: {latent.shape}, JIT Miss: {miss_enc}")


        print(f"Testing Decode mode, Input shape: {latent.shape}")
        output, miss_dec = self.runner.forward(latent, mode="decode")
        print(f"Decode successful! Output shape: {output.shape}, JIT Miss: {miss_dec}")


        self.assertEqual(output.shape, x.shape, "Decoded output shape should match input shape")
        self.assertIsInstance(output, jax.Array)

    def test_jit_caching(self):
        """Verify if JIT caching works (zero miss count on the second run)."""
        input_data = jnp.zeros((1, 1, 32, 32, 3))


        self.runner.forward(input_data, mode="encode")

        _, miss_count = self.runner.forward(input_data, mode="encode")

        print(f"\nVerifying JIT Cache: Second run Miss Count = {miss_count}")
        self.assertEqual(miss_count, 0, "Second run should not trigger new JIT compilation")

if __name__ == "__main__":
    unittest.main()
