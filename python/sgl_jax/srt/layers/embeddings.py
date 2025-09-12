#  Copyright 2023 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Embedding Layers."""

import math
from functools import partial
from typing import Optional, Tuple, Union

import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx.nn import dtypes
from flax.nnx.nn.linear import default_embed_init
from flax.typing import PromoteDtypeFn


class Embed(nnx.Module):
    """A parameterized function from integers [0, n) to d-dimensional vectors.

    Attributes:
      num_embeddings: number of embeddings.
      features: number of feature dimensions for each embedding.
      dtype: the dtype of the embedding vectors (default: float32).
      embedding_init: embedding initializer.
    """

    def __init__(
        self,
        num_embeddings: int,
        features: int,
        dtype: Optional[jnp.dtype] = None,
        param_dtype: jnp.dtype = jnp.bfloat16,
        promote_dtype: PromoteDtypeFn = dtypes.promote_dtype,
        rngs: nnx.Rngs = None,
    ):
        """
        Sets up the embedding parameters for the model.

        This method initializes the embedding parameters with logical partitioning.
        The embedding is represented as a parameter with the specified shape and data type.

        Parameters:
        - embedding: The embedding parameter initialized using the specified method,
                     partitioned logically along the 'vocab' and 'embed' dimensions.

        Returns:
        None
        """
        self.embedding = nnx.Param(
            nnx.with_partitioning(default_embed_init, (None, None))(
                rngs.params(), (num_embeddings, features), param_dtype
            )
        )

        self.num_embeddings = num_embeddings
        self.features = features
        self.dtype = dtype or self.embedding.value.dtype
        self.promote_dtype = promote_dtype

    def __call__(self, inputs: jax.Array) -> jax.Array:
        """Embeds the inputs along the last dimension.

        Args:
          inputs: input data, all dimensions are considered batch dimensions.

        Returns:
          Output which is embedded input data.  The output shape follows the input,
          with an additional `features` dimension appended.
        """
        if not jnp.issubdtype(inputs.dtype, jnp.integer):
            raise ValueError("Input type must be an integer or unsigned integer.")
        # Use take because fancy indexing numpy arrays with JAX indices does not
        # work correctly.
        (embedding,) = self.promote_dtype(
            (self.embedding.value,), dtype=self.dtype, inexact=False
        )
        if self.num_embeddings == 1:
            return jnp.broadcast_to(embedding, inputs.shape + (self.features,))
        return jnp.take(embedding, inputs, axis=0)

    def attend(self, query: jax.Array) -> jax.Array:
        """Attend over the embedding using a query array.

        Args:
          query: array with last dimension equal the feature depth `features` of the
            embedding.

        Returns:
          An array with final dim `num_embeddings` corresponding to the batched
          inner-product of the array of query vectors against each embedding.
          Commonly used for weight-sharing between embeddings and logit transform
          in NLP models.
        """
        query, embedding = self.promote_dtype(
            (query, self.embedding.value), dtype=self.dtype
        )
        return jnp.dot(query, embedding.T)


class ParallelLMHead(Embed):
    def __init__(
        self,
        num_embeddings: int,
        features: int,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        rngs: nnx.Rngs = None,
        use_bias: bool = False,
    ):
        super().__init__(
            num_embeddings=num_embeddings,
            features=features,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        if use_bias:
            self.bias = nnx.Param(
                nnx.with_partitioning(nnx.initializers.constant(0.0), (None, "tensor"))(
                    rngs.params(), (self.num_embeddings, self.features), dtype
                )
            )
        else:
            self.bias = None

    def tie_weights(self, embed_tokens: Embed):
        """Tie the weights with word embeddings."""
        self.embedding = embed_tokens.embedding
        return self

    def __call__(self, input_):
        del input_
        raise RuntimeError("LMHead's weights should be used in the sampler.")


class RotaryEmbedding(nnx.Module):
    """Rotary Position Embedding.

    Attributes:
      min_timescale: Start of the geometric index. Determines the periodicity of
        the added signal.
      max_timescale: End of the geometric index. Determines the frequency of the
        added signal.
      embedding_dims: Dimension of the embedding to be generated.
    """

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        dtype: jnp.dtype,
    ):
        super().__init__()
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.is_neox_style = is_neox_style
        self.dtype = dtype

        self.cos_sin_cache = self._compute_cos_sin_cache().astype(dtype=dtype)

    def __call__(
        self,
        positions: jax.Array,
        query: jax.Array,
        key: jax.Array,
    ) -> Tuple[jax.Array, jax.Array]:
        """Generates a jax.Array of sinusoids with different frequencies.

        Args:
          query, key: The input sequence on which to apply the Rotary position
            embedding. Since rotary position embeddings are applied to query and
            keys after projection, it is assumed of shape [B*S, H].
          position: Optional position jax.Array which denotes the position of each
            token in the sequence. This only needs to be supplied when the sequence
            is packed. It is of shape [B, S].

        Returns:
          a Tuple of jax.Array of shape [B*S, H] which includes the inputs together with
          the rotary position embedding incorporated in it.
        """
        return rotary_embedding_forward(
            positions,
            query,
            key,
            self.cos_sin_cache,
            self.rotary_dim,
            self.head_size,
            self.is_neox_style,
        )

    def _compute_inv_freq(self, base: Union[int, float]) -> jax.Array:
        """Compute the inverse frequency."""
        inv_freq = 1.0 / (
            base
            ** (jnp.arange(0, self.rotary_dim, 2, dtype=jnp.float32) / self.rotary_dim)
        )
        return inv_freq

    def _compute_cos_sin_cache(self) -> jax.Array:
        """Compute the cos and sin cache."""
        inv_freq = self._compute_inv_freq(self.base)
        t = jnp.arange(self.max_position_embeddings, dtype=jnp.float32)
        freqs = jnp.outer(t, inv_freq)
        sin, cos = jnp.sin(freqs), jnp.cos(freqs)
        cache = jnp.concatenate((cos, sin), axis=-1)
        return cache


# @partial(jax.jit, static_argnames=["rotary_dim", "head_size", "is_neox_style"])
def rotary_embedding_forward(
    positions: jax.Array,
    query: jax.Array,
    key: jax.Array,
    cos_sin_cache: jax.Array,
    rotary_dim: int,
    head_size: int,
    is_neox_style: bool,
) -> Tuple[jax.Array, jax.Array]:
    """Rotary Position Embedding."""
    positions = positions.flatten()
    num_tokens = positions.shape[0]
    cos_sin = cos_sin_cache.take(positions, axis=0)
    cos, sin = jnp.split(cos_sin, 2, axis=-1)

    query_shape = query.shape
    query = query.reshape(num_tokens, -1, head_size)
    query_rot = query[..., :rotary_dim]
    query_pass = query[..., rotary_dim:]
    query_rot = _apply_rotary_emb(query_rot, cos, sin, is_neox_style)
    query = jnp.concatenate((query_rot, query_pass), axis=-1).reshape(query_shape)

    key_shape = key.shape
    key = key.reshape(num_tokens, -1, head_size)
    key_rot = key[..., :rotary_dim]
    key_pass = key[..., rotary_dim:]
    key_rot = _apply_rotary_emb(key_rot, cos, sin, is_neox_style)
    key = jnp.concatenate((key_rot, key_pass), axis=-1).reshape(key_shape)
    return query, key


# @partial(jax.jit, static_argnames=["is_neox_style"])
def _apply_rotary_emb(
    x: jax.Array,
    cos: jax.Array,
    sin: jax.Array,
    is_neox_style: bool,
) -> jax.Array:
    """
    Args:
        x: [num_tokens, num_heads, head_size]
        cos: [num_tokens, head_size // 2]
        sin: [num_tokens, head_size // 2]
        is_neox_style: Whether to use the Neox-style or GPT-J-style rotary
            positional embeddings.
    """
    cos = jnp.expand_dims(cos, axis=-2).astype(x.dtype)
    sin = jnp.expand_dims(sin, axis=-2).astype(x.dtype)
    if is_neox_style:
        x1, x2 = jnp.split(x, 2, axis=-1)
    else:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    if is_neox_style:
        return jnp.concatenate((o1, o2), axis=-1)
    else:
        stacked = jnp.stack((o1, o2), axis=-1)
        return stacked.reshape(*stacked.shape[:-2], -1)


# Inverse dim formula to find dim based on number of rotations
def _yarn_find_correction_dim(
    num_rotations: int,
    dim: int,
    base: float = 10000,
    max_position_embeddings: int = 2048,
) -> float:
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (
        2 * math.log(base)
    )


# Find dim range bounds based on rotations
def _yarn_find_correction_range(
    low_rot: int,
    high_rot: int,
    dim: int,
    base: float = 10000,
    max_position_embeddings: int = 2048,
) -> Tuple[int, int]:
    low = math.floor(
        _yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings)
    )
    high = math.ceil(
        _yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings)
    )
    return max(low, 0), min(high, dim - 1)  # Clamp values just in case


def _yarn_linear_ramp_mask(
    low: float, high: float, dim: int, dtype: jnp.dtype
) -> jnp.Array:
    if low == high:
        high += 0.001  # Prevent singularity

    linear_func = (jnp.arange(dim, dtype=dtype) - low) / (high - low)
    ramp_func = jax.lax.clamp(linear_func, 0, 1)
    return ramp_func


def _yarn_get_mscale(scale: float = 1.0) -> float:
    if scale <= 1:
        return 1.0
    return 0.1 * math.log(scale) + 1.0


class ScalingRotaryEmbedding(RotaryEmbedding):
    """Scale the RotaryEmbedding in a way similar to YaRN method. https://arxiv.org/pdf/2309.00071."""

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        scaling_factor: float,
        dtype: jnp.dtype,
        *,
        extra_method: str = "yarn_log",
        extrapolation_factor: float = 1.0,
        attn_factor: float = 1.0,
        beta_fast: int = 32,
        beta_slow: int = 1,
    ) -> None:
        self.scaling_factor = scaling_factor
        self.extra_method = extra_method
        self.extrapolation_factor = extrapolation_factor
        self.attn_factor = attn_factor
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        # Get n-d magnitude scaling corrected for interpolation
        self.mscale = float(_yarn_get_mscale(self.scaling_factor) * attn_factor)
        super().__init__(
            head_size, rotary_dim, max_position_embeddings, base, is_neox_style, dtype
        )

    def _compute_inv_freq(self, scaling_factor: float) -> jnp.Array:
        pos_freqs = self.base ** (
            jnp.arange(0, self.rotary_dim, 2, dtype=jnp.float32) / self.rotary_dim
        )
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (scaling_factor * pos_freqs)

        low, high = _yarn_find_correction_range(
            self.beta_fast,
            self.beta_slow,
            self.rotary_dim,
            self.base,
            self.max_position_embeddings,
        )
        # Get n-d rotational scaling corrected for extrapolation
        inv_freq_mask = (
            1
            - _yarn_linear_ramp_mask(low, high, self.rotary_dim // 2, dtype=jnp.float32)
        ) * self.extrapolation_factor
        if self.extra_method in ["original"]:
            inv_freq = inv_freq_extrapolation
        elif self.extra_method in ["yarn", "yarn_linear"]:
            inv_freq = (
                inv_freq_interpolation * (1 - inv_freq_mask)
                + inv_freq_extrapolation * inv_freq_mask
            )
        elif self.extra_method == "yarn_log":
            inv_freq = jnp.exp(
                jnp.log(inv_freq_extrapolation) * inv_freq_mask
                + jnp.log(inv_freq_interpolation) * (1.0 - inv_freq_mask)
            )
        elif self.extra_method == "theta_scale":
            exponents = jnp.arange(0, self.rotary_dim, 2, dtype=jnp.float32)
            theta_scale_exponent = self.base ** (
                math.log(
                    self.max_position_embeddings * self.scaling_factor / (2 * math.pi)
                )
                / math.log(self.max_position_embeddings / (2 * math.pi))
            )
            inv_freq = jnp.array(
                1.0 / (theta_scale_exponent ** (exponents / self.rotary_dim)),
                dtype=jnp.float32,
            )
        else:
            raise ValueError(f"Unknown extrapolation method: {self.extra_method}")
        return inv_freq

    def _compute_cos_sin_cache(self) -> jnp.Array:
        inv_freq = self._compute_inv_freq(self.scaling_factor)
        t = jnp.arange(
            self.max_position_embeddings * self.scaling_factor, dtype=jnp.float32
        )
        freqs = jnp.einsum("i,j -> ij", t, inv_freq)
        # cos = freqs.cos() * self.mscale
        # sin = freqs.sin() * self.mscale
        cos = freqs.cos()
        sin = freqs.sin()
        cache = jnp.concat((cos, sin), dim=-1)
        return cache
