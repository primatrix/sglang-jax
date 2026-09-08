import jax
import jax.numpy as jnp
from flax import nnx

from sgl_jax.srt.utils.profiling_utils import named_scope


class GeluAndMul(nnx.Module):
    def __init__(self, approximate: str = "tanh"):
        self.approximate = approximate

    @named_scope
    def __call__(self, gate: jax.Array, up: jax.Array):
        if self.approximate == "tanh":
            gelu = jax.nn.gelu(gate, approximate=True)
        else:
            gelu = jax.nn.gelu(gate, approximate=False)
        out = gelu * up
        return out, None


def silu_and_mul_with_clamp(gate: jax.Array, up: jax.Array, limit: float) -> jax.Array:
    """V4 clamps the pre-activation gate, unlike post-SiLU clamp variants."""
    return jax.nn.silu(jnp.minimum(gate, limit)) * jnp.clip(up, -limit, limit)
