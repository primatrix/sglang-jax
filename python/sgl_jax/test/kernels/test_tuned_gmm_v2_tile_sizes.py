import jax.numpy as jnp

from sgl_jax.srt.kernels.gmm.megablox_gmm_kernel.tuned_tile_sizes import (
    get_tuned_gmm_v2_tile_sizes,
)


def test_ling3_prefill_tiles_on_tpu_v7():
    common = {
        "lhs_dtype": jnp.bfloat16,
        "rhs_dtype": jnp.bfloat16,
        "num_groups": 128,
        "size_m": 2048,
        "device_name": "TPU v7",
    }
    assert get_tuned_gmm_v2_tile_sizes(size_k=1536, size_n=512, **common) == (
        32,
        1536,
        512,
    )
    assert get_tuned_gmm_v2_tile_sizes(size_k=512, size_n=1536, **common) == (
        32,
        512,
        1536,
    )


def test_other_shapes_use_auto_tiler():
    assert (
        get_tuned_gmm_v2_tile_sizes(
            lhs_dtype=jnp.bfloat16,
            rhs_dtype=jnp.bfloat16,
            num_groups=128,
            size_m=64,
            size_k=1536,
            size_n=512,
            device_name="TPU v7",
        )
        is None
    )
    assert (
        get_tuned_gmm_v2_tile_sizes(
            lhs_dtype=jnp.bfloat16,
            rhs_dtype=jnp.bfloat16,
            num_groups=128,
            size_m=2048,
            size_k=1536,
            size_n=512,
            device_name="TPU v6e",
        )
        is None
    )
