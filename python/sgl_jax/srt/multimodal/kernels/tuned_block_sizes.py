"""Auto-tuned block sizes for flash attention."""

import logging

import jax.numpy as jnp

from sgl_jax.srt.utils.jax_utils import get_device_name

logger = logging.getLogger(__name__)
# key
#   - device_name
#     - q dtype
#     - k dtype
#     - v dtype
#     - batch size
#     - head number
#     - q length
#     - kv length
#     - head dim
# value:
#   - (num_queries_per_block,)
TUNED_BLOCK_SIZES = {
    "TPU v6e": {
        ("float32", "float32", "bfloat16", 2, 12, 16896, 16896, 128): (512,),
        ("float32", "float32", "bfloat16", 2, 12, 16896, 17152, 128): (512,),
        ("float32", "float32", "bfloat16", 2, 12, 16896, 17408, 128): (512,),
        ("float32", "float32", "bfloat16", 2, 12, 17152, 16896, 128): (512,),
        ("float32", "float32", "bfloat16", 2, 12, 17152, 17152, 128): (256,),
        ("float32", "float32", "bfloat16", 2, 12, 17152, 17408, 128): (512,),
        ("float32", "float32", "bfloat16", 2, 12, 17408, 16896, 128): (512,),
        ("float32", "float32", "bfloat16", 2, 12, 17408, 17152, 128): (512,),
        ("float32", "float32", "bfloat16", 2, 12, 17408, 17408, 128): (512,),
    }
}

# Online-softmax configurations measured on a TPU v6e core.  These are kept
# separate from ``TUNED_BLOCK_SIZES`` because that table selects the query tile
# for the untiled-K fast path, while these values parameterize all three axes of
# the bounded-VMEM path.
TUNED_ONLINE_BLOCK_SIZES = {
    "TPU v6e": {
        ("bfloat16", "bfloat16", "bfloat16", 80): (2048, 2048, 2048),
    },
}

TUNED_LOCAL_SEGMENT_BLOCK_SIZES = {
    "TPU v6e": {
        ("bfloat16", "bfloat16", "bfloat16", 80, 64): (512, 256, 128),
    },
}


def get_tuned_block_sizes(
    q_dtype,
    k_dtype,
    v_dtype,
    batch_size,
    head_num,
    q_len,
    kv_len,
    head_dim,
) -> int:
    """Look up for the best (num_queries_per_blk,) from auto-tuned table."""

    try:
        keys = get_simplified_key(
            q_dtype,
            k_dtype,
            v_dtype,
            batch_size,
            head_num,
            q_len,
            kv_len,
            head_dim,
        )
    except RuntimeError:
        return 256

    device_name = keys[0]

    # Default block sizes.
    bq = 256
    if device_name in TUNED_BLOCK_SIZES and keys[1:] in TUNED_BLOCK_SIZES[device_name]:
        bq = TUNED_BLOCK_SIZES[device_name][keys[1:]][0]
    else:
        logger.info("Using default block q size: bq=%s.", bq)

    return bq


def get_tuned_online_block_sizes(
    q_dtype,
    k_dtype,
    v_dtype,
    q_len,
    kv_len,
    head_dim,
    *,
    max_segment_len: int | None = None,
) -> tuple[int, int, int]:
    """Return a measured bounded-VMEM ``(block_q, block_k_major, block_k)``.

    The conservative 256x128x128 fallback works on every supported TPU shape.
    Device-specific configurations are used only when the sequence dimensions
    satisfy their tiling requirements.
    """
    fallback = (256, 128, 128)
    try:
        device_name = get_device_name()
    except RuntimeError:
        return fallback
    dtype_key = (
        jnp.dtype(q_dtype).name,
        jnp.dtype(k_dtype).name,
        jnp.dtype(v_dtype).name,
        head_dim,
    )

    if max_segment_len is not None:
        local_key = (*dtype_key, max_segment_len)
        tuned = TUNED_LOCAL_SEGMENT_BLOCK_SIZES.get(device_name, {}).get(local_key)
        if tuned is not None and q_len == kv_len and q_len >= tuned[0] and kv_len % tuned[1] == 0:
            return tuned
        return fallback

    tuned = TUNED_ONLINE_BLOCK_SIZES.get(device_name, {}).get(dtype_key)
    if tuned is not None and q_len == kv_len and q_len >= tuned[0] and kv_len % tuned[1] == 0:
        return tuned
    return fallback


def get_simplified_key(
    q_dtype,
    k_dtype,
    v_dtype,
    batch_size,
    head_num,
    q_len,
    kv_len,
    head_dim,
):
    """Get the simplified key to reduce the number of combinations."""
    device = get_device_name()
    q_dtype_name = jnp.dtype(q_dtype).name
    k_dtype_name = jnp.dtype(k_dtype).name
    v_dtype_name = jnp.dtype(v_dtype).name

    return (
        device,
        q_dtype_name,
        k_dtype_name,
        v_dtype_name,
        batch_size,
        head_num,
        q_len,
        kv_len,
        head_dim,
    )
