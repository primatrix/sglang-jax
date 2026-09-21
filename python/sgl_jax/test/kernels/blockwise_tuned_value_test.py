"""Snap constraints in get_safe_blockwise_tuned_value must respect Mosaic sublane tiling.

Regression for the jax 0.11.1 E2002 crash: the nearest-neighbour table match can
borrow batch_block_size=8 (from the n_batch=8 entry) for a padded 16-row decode
bucket, and jax >= 0.11 rejects an 8-row block on a bf16 operand tiled (16, 128).
"""

import os
from unittest import mock

import jax.numpy as jnp
import pytest

from sgl_jax.srt.kernels.quantized_matmul import blockwise_utils


def _tuned(n_batch, x_dtype=jnp.bfloat16):
    with mock.patch.object(blockwise_utils, "_get_current_tpu_version", return_value=7):
        return blockwise_utils.get_safe_blockwise_tuned_value(
            n_batch=n_batch,
            n_out=576,
            n_in=6144,
            x_q_dtype=x_dtype,
            w_q_dtype=jnp.float8_e4m3fn,
            block_size_in=128,
        )


@pytest.mark.parametrize(
    "n_batch,x_dtype,min_ok",
    [
        (16, jnp.bfloat16, 16),  # the crashing bucket: borrowed 8 must align to 16
        (24, jnp.bfloat16, 16),
        (16, jnp.float8_e4m3fn, 16),  # 8-bit activations tile at 32 rows
    ],
)
def test_borrowed_block_respects_sublane_tiling(n_batch, x_dtype, min_ok):
    tuned = _tuned(n_batch, x_dtype)
    if tuned is None:
        pytest.skip("blockwise tuning API unavailable")
    bm = tuned.batch_block_size
    sublane = max(8, 256 // (jnp.dtype(x_dtype).itemsize * 8))
    assert bm == n_batch or bm % sublane == 0, (
        f"batch_block_size={bm} is neither the full batch ({n_batch}) nor a "
        f"multiple of the {sublane}-row sublane tile"
    )
    assert bm <= n_batch


@pytest.mark.parametrize("n_batch", [1, 4, 8, 32])
def test_exact_and_small_batches_unchanged(n_batch):
    tuned = _tuned(n_batch)
    if tuned is None:
        pytest.skip("blockwise tuning API unavailable")
    bm = tuned.batch_block_size
    assert bm == n_batch or bm % 8 == 0
    assert bm <= n_batch


def _tuned_prefill(n_batch, n_out, n_in, x_dtype, min_block):
    with (
        mock.patch.object(blockwise_utils, "_get_current_tpu_version", return_value=7),
        mock.patch.dict(os.environ, {"SGLANG_JAX_QMM_MIN_BATCH_BLOCK": str(min_block)}),
    ):
        return blockwise_utils.get_safe_blockwise_tuned_value(
            n_batch=n_batch,
            n_out=n_out,
            n_in=n_in,
            x_q_dtype=x_dtype,
            w_q_dtype=jnp.float8_e4m3fn,
            block_size_in=128,
        )


def test_min_batch_block_floor_only_lifts_small_borrowed_tiles():
    # bf16 x fp8 at (8192, 4096, 1024) borrows batch_block 64 from the table.
    off = _tuned_prefill(8192, 4096, 1024, jnp.bfloat16, 0)
    if off is None:
        pytest.skip("blockwise tuning API unavailable")
    assert off.batch_block_size == 64
    on = _tuned_prefill(8192, 4096, 1024, jnp.bfloat16, 512)
    assert on.batch_block_size == 512
    assert (on.out_block_size, on.in_block_size) == (off.out_block_size, off.in_block_size)
    # A tile that is already large is left alone.
    big = _tuned_prefill(8192, 4096, 1024, jnp.float8_e4m3fn, 256)
    assert big.batch_block_size == 512
    # Batches below the floor keep the table's choice.
    small = _tuned_prefill(64, 4096, 1024, jnp.bfloat16, 512)
    assert small.batch_block_size <= 64
