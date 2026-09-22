"""The HCA compressed tile bucket is a per-request quantity: a decode batch of 64
requests at ~9K context must size the tile from one request's 72 records (bucket
128), not from the batch total (4608 -> 8192). The per-request rule also matches the
precompile ladder (``precompile_capacities(context_len)[128]``)."""

import types

import numpy as np

from sgl_jax.srt.layers.attention import deepseek_v4_backend as m
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode


class _Mesh:
    shape = {"data": 1, "tensor": 8}


class _Backend:
    """Bare stand-in: hca_entry_bucket only reads ``mesh`` and ``precompile_context_len``."""

    mesh = _Mesh()
    precompile_context_len = None

    def hca_entry_bucket(self, batch):
        return m.DeepseekV4AttentionBackend.hca_entry_bucket(self, batch)


def _backend():
    return _Backend()


def _decode_batch(bs, ctx):
    return types.SimpleNamespace(
        forward_mode=ForwardMode.DECODE,
        seq_lens=np.full((bs,), ctx, np.int32),
        extend_seq_lens=None,
    )


def test_decode_bucket_is_per_request(monkeypatch):
    monkeypatch.delenv("DSV4_HCA_TILE_BUCKET_SUM", raising=False)
    monkeypatch.delenv("DSV4_HCA_TILE_BUCKET", raising=False)
    b = _backend()
    assert b.hca_entry_bucket(_decode_batch(64, 9216)) == 128
    assert b.hca_entry_bucket(_decode_batch(64, 9216)) == m.precompile_capacities(9216)[128]
    assert b.hca_entry_bucket(_decode_batch(1, 32768)) == 256
    assert b.hca_entry_bucket(_decode_batch(16, 200_000)) == 2048


def test_legacy_sum_bucket_env(monkeypatch):
    monkeypatch.setenv("DSV4_HCA_TILE_BUCKET_SUM", "1")
    monkeypatch.delenv("DSV4_HCA_TILE_BUCKET", raising=False)
    b = _backend()
    assert b.hca_entry_bucket(_decode_batch(64, 9216)) == 8192
