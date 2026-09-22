"""Borrowed blockwise batch tiles are aligned to the activation's sublane tile.

jax >= 0.11 rejects a block_m that neither equals the array rows nor divides the
operand's native sublane tiling (fp32 8, bf16 16, fp8/int8 32 rows) with Mosaic
E2002. ``get_safe_blockwise_tuned_value`` therefore rounds a borrowed
``batch_block_size`` up to that tile; this applies to every quantized linear layer
that resolves its tiles through it.
"""

from collections import namedtuple

import jax.numpy as jnp
import pytest

from sgl_jax.srt.kernels.quantized_matmul import blockwise_utils as blockwise_utils

TunedValue = namedtuple(
    "TunedValue", ["batch_block_size", "out_block_size", "in_block_size", "n_lane_multiplier"]
)
TunedKey = namedtuple(
    "TunedKey", ["tpu_version", "x_q_dtype", "w_q_dtype", "n_batch", "n_out", "n_in"]
)


def _install_table(monkeypatch, x_dtype, seed_batch_block):
    key = TunedKey(7, jnp.dtype(x_dtype).name, "float8_e4m3fn", 8, 1024, 1024)
    table = {key: TunedValue(seed_batch_block, 256, 512, 1)}
    monkeypatch.setattr(
        blockwise_utils, "_get_blockwise_tuning_api", lambda: (TunedValue, None, table)
    )
    monkeypatch.setattr(blockwise_utils, "_get_current_tpu_version", lambda: 7)
    # pin the floor off: the tests below assert the table's own choice first
    monkeypatch.setenv("SGLANG_JAX_QMM_MIN_BATCH_BLOCK", "0")


def _resolve(n_batch, x_dtype):
    return blockwise_utils.get_safe_blockwise_tuned_value(
        n_batch=n_batch,
        n_out=1024,
        n_in=1024,
        x_q_dtype=jnp.dtype(x_dtype),
        w_q_dtype=jnp.dtype(jnp.float8_e4m3fn),
        block_size_in=128,
    )


@pytest.mark.parametrize(
    "x_dtype,tile_rows", [(jnp.float32, 8), (jnp.bfloat16, 16), (jnp.float8_e4m3fn, 32)]
)
def test_sublane_tile_rows(x_dtype, tile_rows):
    assert blockwise_utils.sublane_tile_rows(x_dtype) == tile_rows


@pytest.mark.parametrize(
    "x_dtype,seed,n_batch,expected",
    [
        (jnp.float32, 24, 64, 24),  # already a multiple of the 8-row tile
        (jnp.bfloat16, 24, 64, 32),  # rounded up to the 16-row tile
        (jnp.float8_e4m3fn, 24, 64, 32),  # rounded up to the 32-row tile
        (jnp.bfloat16, 24, 12, 12),  # batch below one tile: block == batch
        (jnp.bfloat16, 24, 24, 24),  # block == batch is always legal
    ],
)
def test_borrowed_batch_block_is_tile_aligned(monkeypatch, x_dtype, seed, n_batch, expected):
    _install_table(monkeypatch, x_dtype, seed)
    tuned = _resolve(n_batch, x_dtype)
    assert tuned.batch_block_size == expected


def test_min_batch_block_floor_is_read_at_call_time(monkeypatch):
    _install_table(monkeypatch, jnp.bfloat16, 64)
    assert _resolve(8192, jnp.bfloat16).batch_block_size == 64
    monkeypatch.setenv("SGLANG_JAX_QMM_MIN_BATCH_BLOCK", "512")
    assert _resolve(8192, jnp.bfloat16).batch_block_size == 512
    assert _resolve(256, jnp.bfloat16).batch_block_size == 64  # below the floor's batch


def test_min_batch_block_default_is_512(monkeypatch):
    _install_table(monkeypatch, jnp.bfloat16, 64)
    monkeypatch.delenv("SGLANG_JAX_QMM_MIN_BATCH_BLOCK", raising=False)
    assert _resolve(8192, jnp.bfloat16).batch_block_size == 512
    assert _resolve(256, jnp.bfloat16).batch_block_size == 64  # below the floor's batch


def test_min_batch_block_floor_only_changes_the_batch_tile(monkeypatch):
    # The floor may only widen the row (batch) tile. The reduction tiles that
    # decide the fp32 accumulation order of every output element stay the
    # table's, so a floored call computes each row exactly as the unfloored one.
    _install_table(monkeypatch, jnp.bfloat16, 64)
    monkeypatch.setenv("SGLANG_JAX_QMM_MIN_BATCH_BLOCK", "0")
    table = _resolve(8192, jnp.bfloat16)
    monkeypatch.setenv("SGLANG_JAX_QMM_MIN_BATCH_BLOCK", "512")
    floored = _resolve(8192, jnp.bfloat16)
    assert floored.batch_block_size == 512 and table.batch_block_size == 64
    assert (floored.out_block_size, floored.in_block_size) == (
        table.out_block_size,
        table.in_block_size,
    )
