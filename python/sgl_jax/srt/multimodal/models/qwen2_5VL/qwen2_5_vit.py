import logging
import math
from collections.abc import Callable

import jax
from jax.sharding import Mesh
import jax.numpy as jnp
import numpy as np
from flax import nnx
from functools import partial

from transformers import modeling_flax_utils
from sgl_jax.srt.multimodal.configs.qwen_vl.qwen_2_5_vl_config import QwenVLModelConfig

init_fn = nnx.initializers.uniform()
DEFAULT_BLOCK_K_MAJOR = 128
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Qwen2_5_VisionPatchEmbed(nnx.Module):
    def __init__(
            self,
            rngs: nnx.Rngs = None,
            patch_size: int = 14,
            temporal_patch_size: int = 2,
            in_channels: int = 3,
            hidden_size: int = 1152,
            dtype: jnp.dtype = jnp.bfloat16,
    ) -> None:
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.hidden_size = hidden_size
        kernel_size = (temporal_patch_size, patch_size, patch_size)

        self.proj = nnx.Conv(
            in_features=in_channels,
            out_features=hidden_size,
            kernel_size=kernel_size,
            strides=kernel_size,
            use_bias=False,
            param_dtype=dtype,
            rngs=rngs or nnx.Rngs(0),  # Use dummy rngs if None (for eval_shape)
        )


class Qwen2_5_VisionRotaryEmbedding(nnx.Module):
    def __init__(self, dim: int, theta: float = 10000.0):
        self.dim = dim
        self.theta = theta


class Qwen2_5_VisionMLP(nnx.Module):
    def __init__(self, config: QwenVLModelConfig, dtype: jnp.dtype, rngs: nnx.Rngs = None):
        in_features = config.hidden_size
        hidden_features = config.intermediate_size
        act_fn = modeling_flax_utils.ACT2FN[config.hidden_act]

        # Use dummy rngs if None (for eval_shape)
        _rngs = rngs or nnx.Rngs(0)

        self.gate_proj = nnx.Linear(
            in_features,
            hidden_features,
            use_bias=True,
            param_dtype=dtype,
            rngs=_rngs,
        )
        self.up_proj = nnx.Linear(
            in_features,
            hidden_features,
            use_bias=True,
            param_dtype=dtype,
            rngs=_rngs,
        )
        self.down_proj = nnx.Linear(
            hidden_features,
            in_features,
            use_bias=True,
            param_dtype=dtype,
            rngs=_rngs,
        )
        self.act_fn = act_fn


class Qwen2_5_VisionAttention(nnx.Module):
    def __init__(
            self, config: QwenVLModelConfig, dtype: jnp.dtype, rngs: nnx.Rngs = None, mesh: Mesh = None
    ):
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Use dummy rngs if None (for eval_shape)
        _rngs = rngs or nnx.Rngs(0)

        self.qkv_proj = nnx.Linear(
            self.hidden_size,
            3 * self.hidden_size,
            use_bias=True,
            param_dtype=dtype,
            rngs=_rngs,
        )

        self.proj = nnx.Linear(
            self.hidden_size,
            self.hidden_size,
            use_bias=True,
            param_dtype=dtype,
            rngs=_rngs,
        )


class Qwen2_5_VisionBlock(nnx.Module):
    def __init__(
            self, config: QwenVLModelConfig, dtype: jnp.dtype, rngs: nnx.Rngs = None, mesh: Mesh = None
    ):
        dim = config.hidden_size
        norm_layer = partial(
            nnx.RMSNorm,
            epsilon=config.rms_norm_eps,
            scale_init=nnx.with_partitioning(init_fn, (None,)),
        )

        # Use dummy rngs if None (for eval_shape)
        _rngs = rngs or nnx.Rngs(0)

        self.norm1 = norm_layer(dim, dtype=dtype, rngs=_rngs)
        self.norm2 = norm_layer(dim, dtype=dtype, rngs=_rngs)
        self.attn = Qwen2_5_VisionAttention(config=config, dtype=dtype, rngs=rngs, mesh=mesh)
        self.mlp = Qwen2_5_VisionMLP(config=config, dtype=dtype, rngs=rngs)


class Qwen2_5_VisionPatchMerger(nnx.Module):
    def __init__(
            self,
            d_model: int,
            context_dim: int,
            norm_layer: Callable,
            spatial_merge_size: int,
            dtype: jnp.dtype,
            rngs: nnx.Rngs = None,
    ):
        self.hidden_size = context_dim * (spatial_merge_size ** 2)

        # Use dummy rngs if None (for eval_shape)
        _rngs = rngs or nnx.Rngs(0)

        self.ln_q = norm_layer(
            context_dim, dtype=dtype, rngs=_rngs, scale_init=nnx.with_partitioning(init_fn, (None,))
        )
        self.mlp_fc1 = nnx.Linear(
            self.hidden_size,
            self.hidden_size,
            use_bias=True,
            param_dtype=dtype,
            rngs=_rngs,
        )
        self.mlp_act = modeling_flax_utils.ACT2FN["gelu"]
        self.mlp_fc2 = nnx.Linear(
            self.hidden_size,
            d_model,
            use_bias=True,
            param_dtype=dtype,
            rngs=_rngs,
        )


class Qwen2_5_VL_VisionModel(nnx.Module):
    """Placeholder model class for the ViT stage.
    - Call encode_vision() to get vision embeddings
    - Call get_input_embeddings() to merge vision + text embeddings
    """
    
    def __init__(self, dtype=None, mesh=None):
        super().__init__()

    def load_weights(self, model_config):
      pass

