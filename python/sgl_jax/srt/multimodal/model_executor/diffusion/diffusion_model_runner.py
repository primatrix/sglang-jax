from sgl_jax.srt.model_executor.base_model_runner import BaseModelRunner


class DiffusionModelRunner(BaseModelRunner):
    def __init__(self, model_loader, model_config):
        super().__init__(model_loader, model_config)
        # Additional initialization for diffusion model if needed
        # e.g., setting up noise schedulers, diffusion steps, etc.

    def forward(self, batch, mesh):
        # Implement the diffusion model inference logic here
        # This might include steps like adding noise, denoising, etc.
        pass
