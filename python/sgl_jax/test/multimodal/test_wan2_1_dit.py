import contextlib
import os
import unittest
from functools import partial

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.layers.attention.flashattention_backend import (
    FlashAttention,
    FlashAttentionMetadata,
)
from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.multimodal.configs.dits.wan_model_config import WanModelConfig
from sgl_jax.srt.multimodal.models.wan.diffusion.wan_dit import WanTransformer3DModel
from sgl_jax.srt.utils.mesh_utils import create_device_mesh

# Ensure the python directory is in the path


class MockConfig:
    def __init__(self):
        self.patch_size = (1, 2, 2)
        self.hidden_dim = 64
        self.num_heads = 4
        self.num_attention_heads = 4
        self.attention_head_dim = 16
        self.in_channels = 4
        self.out_channels = 4  # Output channels (same as input for diffusion models)
        self.freq_dim = 16
        self.text_dim = 32
        self.image_dim = 32
        self.ffn_dim = 128
        self.qk_norm = "rms_norm"
        self.cross_attn_norm = True
        self.epsilon = 1e-6
        self.added_kv_proj_dim = None
        self.num_layers = 30


class MockRequest:
    def __init__(self, config, batch_size=1, seq_len=256):
        self.attention_backend = FlashAttention(
            num_attn_heads=config.num_attention_heads,
            num_kv_heads=config.num_attention_heads,
            head_dim=config.attention_head_dim,
        )
        # Mock attention metadata
        metadata = FlashAttentionMetadata()
        metadata.num_seqs = jnp.array([batch_size], dtype=jnp.int32)
        metadata.cu_q_lens = jnp.array([0, seq_len], dtype=jnp.int32)
        metadata.cu_kv_lens = jnp.array([0, seq_len], dtype=jnp.int32)
        metadata.page_indices = jnp.arange(seq_len, dtype=jnp.int32)
        metadata.seq_lens = jnp.array([seq_len], dtype=jnp.int32)
        metadata.distribution = jnp.array([0, batch_size, batch_size], dtype=jnp.int32)
        metadata.custom_mask = None
        self.attention_backend.forward_metadata = metadata


class TestWanTransformer3DModel(unittest.TestCase):
    def test_forward(self):
        config = MockConfig()

        model = WanTransformer3DModel(config, mesh=jax.sharding.Mesh(jax.devices(), ("data",)))

        # (batch_size, num_channels, num_frames, height, width) - channel-first format
        input_shape = (1, 4, 1, 32, 32)
        hidden_states = jax.random.normal(jax.random.key(0), input_shape)

        # text embeddings
        encoder_hidden_states = jax.random.normal(jax.random.key(1), (1, 8, 32))  # B, L, D

        # timesteps
        timesteps = jax.numpy.array([1.0])

        # image embeddings
        encoder_hidden_states_image = jax.random.normal(jax.random.key(2), (1, 1, 32))  # B, L, D

        # Run forward (no req needed for diffusion - uses ring attention)
        output = model(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timesteps=timesteps,
            encoder_hidden_states_image=encoder_hidden_states_image,
        )

        # Expected output shape: same as input (B, C, F, H, W) = (1, 4, 1, 32, 32)
        self.assertEqual(output.shape, (1, 4, 1, 32, 32))


@contextlib.contextmanager
def _linear_no_output_sharding():
    # Keep sequence-axis sharding from inputs instead of forcing output sharding.
    original_call = LinearBase.__call__

    def _call(self, x):
        bias = self.bias if not self.skip_bias_add else None
        output = jax.lax.dot_general(
            x,
            self.weight.value,
            (((x.ndim - 1,), (0,)), ((), ())),
            preferred_element_type=self.params_dtype,
        )
        if bias is not None:
            output = output + bias.value
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias

    LinearBase.__call__ = _call
    try:
        yield
    finally:
        LinearBase.__call__ = original_call


class TestWanDiTMaxFramesTwoDevices(unittest.TestCase):
    def test_max_frames_token_shard_two_devices(self):

        devices = jax.devices()
        if len(devices) < 2:
            self.skipTest("Need at least 2 devices for this test.")

        def device_usage_bytes(device):
            try:
                stats = device.memory_stats()
            except Exception:
                return None, None
            return stats.get("bytes_in_use"), stats.get("bytes_limit")

        scored_devices = []
        for device in devices:
            used, limit = device_usage_bytes(device)
            if used is None:
                score = float("inf")
            elif limit:
                score = used / limit
            else:
                score = used
            scored_devices.append((score, used, limit, device))

        scored_devices.sort(key=lambda item: item[0])
        selected = [item[3] for item in scored_devices if item[0] != float("inf")][:2]
        if len(selected) < 2:
            print("Memory stats unavailable; using the first two devices.")
            selected = devices[:2]
        else:
            selected_info = []
            for device in selected:
                used, limit = device_usage_bytes(device)
                selected_info.append((getattr(device, "id", None), used, limit, device.platform))
            print(f"Selected devices (id, used, limit, platform): {selected_info}")

        config = WanModelConfig()
        height = int(os.environ.get("SGLANG_WAN_DIT_TEST_HEIGHT", "480"))
        width = int(os.environ.get("SGLANG_WAN_DIT_TEST_WIDTH", "832"))
        min_frames = int(os.environ.get("SGLANG_WAN_DIT_MIN_FRAMES", "73"))
        max_frames = int(os.environ.get("SGLANG_WAN_DIT_MAX_FRAMES", "200"))
        step = int(os.environ.get("SGLANG_WAN_DIT_FRAME_STEP", str(config.scale_factor_temporal)))

        if height % config.scale_factor_spatial != 0 or width % config.scale_factor_spatial != 0:
            self.skipTest("Height/width must be divisible by scale_factor_spatial.")

        def align_frames(n):
            return n - (n - 1) % config.scale_factor_temporal

        min_frames = align_frames(min_frames)
        max_frames = align_frames(max_frames)
        if max_frames < min_frames:
            self.skipTest("Invalid frame range for stress test.")

        latent_h = height // config.scale_factor_spatial
        latent_w = width // config.scale_factor_spatial

        mesh = create_device_mesh(
            ici_parallelism=[2, 1],
            dcn_parallelism=[1, 1],
            devices=selected,
        )
        data_shards = mesh.shape["data"]
        if latent_w % data_shards == 0:
            token_sharding = NamedSharding(mesh, P(None, None, None, None, "data"))
            shard_axis = "width"
        elif latent_h % data_shards == 0:
            token_sharding = NamedSharding(mesh, P(None, None, None, "data", None))
            shard_axis = "height"
        else:
            self.skipTest(
                "Latent spatial dims must be divisible by data shards for token sharding."
            )
        text_sharding = NamedSharding(mesh, P())
        print(f"Token sharding axis: {shard_axis}")

        with jax.set_mesh(mesh), _linear_no_output_sharding():
            with jax.default_device(selected[0]):
                model = WanTransformer3DModel(config, mesh=mesh)
            model_def, model_state = nnx.split(model)
            model_state_leaves, model_state_def = jax.tree_util.tree_flatten(model_state)

            @partial(jax.jit, static_argnames=["model_state_def"])
            def forward_model(
                model_def,
                model_state_def,
                model_state_leaves,
                hidden_states,
                encoder_hidden_states,
                timesteps,
            ):
                model_state = jax.tree_util.tree_unflatten(model_state_def, model_state_leaves)
                model = nnx.merge(model_def, model_state)
                hidden_states = jax.lax.with_sharding_constraint(hidden_states, token_sharding)
                return model(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    timesteps=timesteps,
                    encoder_hidden_states_image=None,
                    guidance_scale=None,
                )

            best = None
            for frames in range(min_frames, max_frames + 1, step):
                latent_frames = (frames - 1) // config.scale_factor_temporal + 1
                print(f"Testing DiT with {frames} frames ({latent_frames} latent frames)...")
                with jax.default_device(selected[0]):
                    hidden_states = jax.random.normal(
                        jax.random.key(0),
                        (1, config.in_channels, latent_frames, latent_h, latent_w),
                        dtype=jnp.bfloat16,
                    )
                    encoder_hidden_states = jax.random.normal(
                        jax.random.key(1),
                        (1, config.max_text_len, config.text_dim),
                        dtype=jnp.bfloat16,
                    )
                    timesteps = jnp.zeros((1,), dtype=jnp.int32)
                hidden_states = jax.device_put(hidden_states, token_sharding)
                encoder_hidden_states = jax.device_put(encoder_hidden_states, text_sharding)
                try:
                    out = forward_model(
                        model_def,
                        model_state_def,
                        model_state_leaves,
                        hidden_states,
                        encoder_hidden_states,
                        timesteps,
                    )
                    out.block_until_ready()
                    expected_shape = (
                        1,
                        config.in_channels,
                        latent_frames,
                        latent_h,
                        latent_w,
                    )
                    self.assertEqual(out.shape, expected_shape)
                    best = frames
                except Exception as exc:  # pragma: no cover - depends on device OOM
                    message = str(exc).lower()
                    if (
                        "resource exhausted" in message
                        or "out of memory" in message
                        or "oom" in message
                    ):
                        break
                    raise

        if best is None:
            self.fail(f"DiT failed at {min_frames} frames")

        print(f"Max frames supported at {height}x{width} on 2 devices (token-sharded): {best}")
        self.assertGreaterEqual(best, min_frames)


if __name__ == "__main__":
    mesh = jax.sharding.Mesh(jax.devices(), ("data",))
    with jax.set_mesh(mesh):
        unittest.main()
