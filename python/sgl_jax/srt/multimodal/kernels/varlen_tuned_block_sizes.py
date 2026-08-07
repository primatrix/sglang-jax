"""Auto-tuned block sizes for the ``varlen_attention`` kernel.

Companion to ``tuned_block_sizes.py`` (which serves the block-sparse
``flash_attention`` kernel). ``varlen_attention`` exposes ``num_queries_per_block``
and ``num_kv_per_block``; this table records the fastest correct ``(bq, bkv)``
per shape and attention layout, measured on TPU.

Key:
  - device_name (from ``get_device_name``)
    - (q_dtype, k_dtype, v_dtype, heads, q_len, kv_len, head_dim, layout)
Value:
  - (num_queries_per_block, num_kv_per_block)

``layout`` is ``"full"`` (one contiguous segment, ``window_size=(-1,-1)``) or
``"window"`` (short segments / windowed attention). Per-lane batch is always 1,
so batch is not part of the key.

The ``"TPU v7"`` entries were tuned on a v7x core (64 MiB VMEM) for the
Qwen2.5-VL vision shapes: bf16 MHA, 16 heads, head_dim 80.
"""

import logging

import jax
import jax.numpy as jnp

from sgl_jax.srt.utils.jax_utils import get_device_name

logger = logging.getLogger(__name__)

# Fallback used off-TPU and on table misses (matches varlen_attention defaults).
DEFAULT_Q_BLOCK = 512
DEFAULT_KV_BLOCK = 512

TUNED_VARLEN_BLOCK_SIZES = {
    # bf16 MHA, 16 heads, head_dim 80; tuned on a v7x core.
    "TPU v7": {
        ("bfloat16", "bfloat16", "bfloat16", 16, 256, 256, 80, "full"): (256, 256),
        ("bfloat16", "bfloat16", "bfloat16", 16, 1024, 1024, 80, "full"): (256, 512),
        ("bfloat16", "bfloat16", "bfloat16", 16, 2048, 2048, 80, "full"): (256, 1024),
        ("bfloat16", "bfloat16", "bfloat16", 16, 4096, 4096, 80, "full"): (256, 1024),
        ("bfloat16", "bfloat16", "bfloat16", 16, 8192, 8192, 80, "full"): (256, 1024),
        ("bfloat16", "bfloat16", "bfloat16", 16, 16384, 16384, 80, "full"): (256, 1024),
        ("bfloat16", "bfloat16", "bfloat16", 16, 256, 256, 80, "window"): (128, 256),
        ("bfloat16", "bfloat16", "bfloat16", 16, 1024, 1024, 80, "window"): (128, 256),
        ("bfloat16", "bfloat16", "bfloat16", 16, 2048, 2048, 80, "window"): (128, 256),
        ("bfloat16", "bfloat16", "bfloat16", 16, 4096, 4096, 80, "window"): (128, 256),
        ("bfloat16", "bfloat16", "bfloat16", 16, 8192, 8192, 80, "window"): (128, 256),
        ("bfloat16", "bfloat16", "bfloat16", 16, 16384, 16384, 80, "window"): (128, 256),
    }
}


def get_varlen_tuned_block_sizes(
    q_dtype,
    k_dtype,
    v_dtype,
    heads,
    q_len,
    kv_len,
    head_dim,
    layout,
) -> tuple[int, int]:
    """Look up the best ``(num_queries_per_block, num_kv_per_block)`` for varlen.

    Returns the kernel defaults off-TPU (e.g. CPU interpret) or on a table miss.
    """
    # The tuned table is TPU-only; off-TPU (CPU interpret) fall back to defaults
    # rather than probing for a TPU device name.
    if "TPU" not in jax.devices()[0].device_kind:
        return DEFAULT_Q_BLOCK, DEFAULT_KV_BLOCK

    device_name = get_device_name()
    key = (
        jnp.dtype(q_dtype).name,
        jnp.dtype(k_dtype).name,
        jnp.dtype(v_dtype).name,
        heads,
        q_len,
        kv_len,
        head_dim,
        layout,
    )
    device_table = TUNED_VARLEN_BLOCK_SIZES.get(device_name)
    if device_table is not None and key in device_table:
        return device_table[key]

    logger.info(
        "varlen: using default block sizes bq=%s bkv=%s (no tuned entry for %s %s).",
        DEFAULT_Q_BLOCK,
        DEFAULT_KV_BLOCK,
        device_name,
        key,
    )
    return DEFAULT_Q_BLOCK, DEFAULT_KV_BLOCK
