"""Shape-specific GMM v2 tile sizes measured on supported TPU generations."""

from __future__ import annotations

import jax.numpy as jnp

from sgl_jax.srt.utils.jax_utils import get_device_name

# Key: (lhs dtype, rhs dtype, groups, M, K, N).
# Values are (tile_m, tile_k, tile_n).
# Wildcard-m entries (size_m == -1) apply to calls with at least this many rows;
# size_m == 0 entries apply to calls with fewer rows (decode buckets).
LARGE_M_FLOOR = 4096


# tile_m for the small-m wildcard rows below; 32 measured best on v7x for the
# DeepSeek V4-Flash decode shapes (bs=64: ~12 rows per group).
SMALL_M_TILE = 32


TUNED_TILE_SIZES_GMM_V2 = {
    "TPU v7": {
        # DeepSeek V4-Flash EPMoE, tp8/ep1 (256 replicated experts, inter sharded to
        # 256), bf16 activations x fp8 e4m3 weights, prefill rows (8K chunk: 49152).
        # Swept 2026-09-15 on v7x: 0.524 -> 0.366 ms and 0.483 -> 0.371 ms per call.
        ("bfloat16", "float8_e4m3fn", 256, -1, 4096, 256): (256, 4096, 256),
        ("bfloat16", "float8_e4m3fn", 256, -1, 256, 4096): (256, 256, 4096),
        # DeepSeek V4-Flash EPMoE, tp8/ep8 (32 local experts), decode rows (bs=64: 384
        # rows over 32 groups, ~12 per group). The auto-tiler picks tm=128 with full K/N,
        # so every group multiplies a 128-row tile that is mostly padding: ~30 us of MXU
        # work inside a 72 us call that is otherwise weight-bandwidth bound. Small-m
        # wildcard (size_m == 0): any m below LARGE_M_FLOOR, tile_m = SMALL_M_TILE.
        # The lhs key is the *quantized* activation dtype: gmm_v2 quantizes bf16
        # activations to fp8 when the weights are fp8 and the chip has fp8 MXU ops
        # (maybe_quantize_lhs), so the decode calls look up ("float8_e4m3fn", ...);
        # a "bfloat16" key never matches and the auto-tiler runs silently (kernel
        # name tm_128). Both keys are listed for the no-quantization fallback.
        ("float8_e4m3fn", "float8_e4m3fn", 32, 0, 4096, 2048): (SMALL_M_TILE, 4096, 2048),
        ("float8_e4m3fn", "float8_e4m3fn", 32, 0, 2048, 4096): (SMALL_M_TILE, 2048, 4096),
        ("bfloat16", "float8_e4m3fn", 32, 0, 4096, 2048): (SMALL_M_TILE, 4096, 2048),
        ("bfloat16", "float8_e4m3fn", 32, 0, 2048, 4096): (SMALL_M_TILE, 2048, 4096),
        # Ling-3.0-tiny replicated EPMoE, decode BS=1 hot wi shape.
        # Measured kernel latency: 0.555ms -> 0.382ms (31.1% lower).
        ("bfloat16", "bfloat16", 128, 32, 1536, 512): (32, 768, 512),
        # Ling-3.0-tiny replicated EPMoE, 2K balanced prefill hot shapes.
        ("bfloat16", "bfloat16", 128, 2048, 1536, 512): (32, 1536, 512),
        ("bfloat16", "bfloat16", 128, 2048, 512, 1536): (32, 512, 1536),
    },
}


def get_tuned_gmm_v2_tile_sizes(
    *,
    lhs_dtype: jnp.dtype,
    rhs_dtype: jnp.dtype,
    num_groups: int,
    size_m: int,
    size_k: int,
    size_n: int,
    device_name: str | None = None,
) -> tuple[int, int, int] | None:
    if device_name is None:
        device_name = get_device_name()
    table = TUNED_TILE_SIZES_GMM_V2.get(device_name)
    if table is None:
        return None
    lhs, rhs = jnp.dtype(lhs_dtype).name, jnp.dtype(rhs_dtype).name
    exact = table.get((lhs, rhs, int(num_groups), int(size_m), int(size_k), int(size_n)))
    if exact is not None:
        return exact
    # Prefill rows per call depend on routing (EP) and chunk size, so large-m
    # entries may use size_m == -1 ("any m >= LARGE_M_FLOOR").
    if int(size_m) >= LARGE_M_FLOOR:
        return table.get((lhs, rhs, int(num_groups), -1, int(size_k), int(size_n)))
    small = table.get((lhs, rhs, int(num_groups), 0, int(size_k), int(size_n)))
    if small is not None and small[0] > 0:
        return small
    return None
