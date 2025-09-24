from functools import partial
from typing import Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
from flax import nnx
from jax import lax


class RMSNorm(nnx.Module):
    """RMS normalization."""

    def __init__(
        self,
        hidden_size: int,
        epsilon: float = 1e-6,
        kernel_axes: Optional[Sequence[str]] = None,
        rngs: nnx.Rngs = None,
    ):
        self.variance_epsilon = epsilon
        self.weight = nnx.Param(
            nnx.with_partitioning(nnx.initializers.ones, kernel_axes)(
                rngs.params(), (hidden_size,)
            )
        )

    def __call__(
        self, x: jax.Array, residual: Optional[jax.Array] = None
    ) -> Union[jax.Array, Tuple[jax.Array, jax.Array]]:
        """Applies layer normalization on the input."""
        orig_dtype = x.dtype
        x_f32 = jnp.asarray(x, jnp.float32)
        if residual is not None:
            x_f32 += jnp.asarray(residual, jnp.float32)
            residual = x_f32.astype(orig_dtype)
        mean2 = jnp.mean(lax.square(x_f32), axis=-1, keepdims=True)
        y = jnp.asarray(x_f32 * lax.rsqrt(mean2 + self.variance_epsilon), jnp.float32)
        output = (y * jnp.asarray(self.weight, jnp.float32)).astype(orig_dtype)
        if residual is None:
            return output
        else:
            return output, residual


class GemmaRMSNorm(nnx.Module):
    """Gemma RMS normalization."""

    def __init__(
        self,
        hidden_size: int,
        epsilon: float = 1e-6,
        kernel_axes: Optional[Sequence[str]] = None,
        rngs: nnx.Rngs = None,
    ):
        self.epsilon = epsilon
        self.weight = nnx.Param(
            nnx.with_partitioning(nnx.initializers.zeros, kernel_axes)(
                rngs.params(), (hidden_size,)
            )
        )

    def __call__(
        self, x: jax.Array, residual: Optional[jax.Array] = None
    ) -> Union[jax.Array, Tuple[jax.Array, jax.Array]]:
        orig_dtype = x.dtype
        if residual is not None:
            if orig_dtype == jnp.float16:
                x = x + jnp.astype(residual, jnp.float16)
            else:
                x = x + residual
            residual = x

        x = x.astype(jnp.float32)
        variance = jnp.mean(lax.square(x), axis=-1, keepdims=True)
        x = x * lax.rsqrt(variance + self.epsilon)
        x = x * (1.0 + jnp.asarray(self.weight, jnp.float32))
        x = x.astype(orig_dtype)
        return x if residual is None else (x, residual)
