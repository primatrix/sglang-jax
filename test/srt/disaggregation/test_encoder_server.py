from __future__ import annotations

import asyncio
from types import MethodType, SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from sgl_jax.srt.disaggregation.encoder.server import MMEncoder
from sgl_jax.srt.disaggregation.encoder.transfer_layout import PackedEmbeddingSlice
from sgl_jax.srt.multimodal.common.modality_enum import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)


def _inputs(token_count: int) -> MultimodalInputs:
    return MultimodalInputs(
        mm_items=[
            MultimodalDataItem(
                modality=Modality.IMAGE,
                placeholder_ranges=[(0, token_count)],
                model_specific_data={
                    "image_grid_thw": np.asarray([[1, 1, token_count]], dtype=np.int32)
                },
            )
        ]
    )


def _encoder(output: jnp.ndarray, processed: list[MultimodalInputs]) -> MMEncoder:
    encoder = object.__new__(MMEncoder)
    encoder.model = SimpleNamespace(get_image_feature=lambda _: output)
    pending = iter(processed)

    async def process_request(self, request, modality):
        return next(pending), {}

    encoder._process_request = MethodType(process_request, encoder)
    return encoder


async def _run_encoder(encoder: MMEncoder, requests: list[dict]):
    return encoder.encode(await encoder.preprocess(requests))


def test_encode_discards_jax_bucket_padding():
    output = jnp.arange(16, dtype=jnp.float32).reshape(8, 2)
    encoder = _encoder(output, [_inputs(2), _inputs(3)])

    results = asyncio.run(_run_encoder(encoder, [{"modality": "IMAGE"}, {"modality": "IMAGE"}]))

    first, second = results[0][0], results[1][0]
    assert isinstance(first, PackedEmbeddingSlice)
    assert isinstance(second, PackedEmbeddingSlice)
    assert first.packed is second.packed
    assert (first.offset, first.rows) == (0, 2)
    assert (second.offset, second.rows) == (2, 3)
    assert [embedding.shape for embedding, _ in results] == [(2, 2), (3, 2)]
    timing = results[0][1]["_encoder_timing"]
    assert timing["encode_server_postprocess_done_ns"] >= timing["encode_done_ns"]
    assert timing["encode_server_postprocess_duration_ns"] >= 0
    assert timing["encode_token_count_duration_ns"] >= 0
    assert timing["encode_embedding_slice_duration_ns"] >= 0
    assert timing["encode_metadata_duration_ns"] >= 0
    assert timing["encode_result_pack_duration_ns"] >= 0
    assert timing["encode_server_postprocess_residual_ns"] >= 0


def test_encode_rejects_incomplete_output():
    encoder = _encoder(jnp.zeros((2, 2)), [_inputs(3)])

    with pytest.raises(ValueError, match="incomplete IMAGE encoder output"):
        asyncio.run(_run_encoder(encoder, [{"modality": "IMAGE"}]))


def test_encode_does_not_wait_for_jax_output(monkeypatch):
    output = jnp.zeros((2, 2))
    encoder = _encoder(output, [_inputs(2)])

    monkeypatch.setattr(
        "sgl_jax.srt.disaggregation.encoder.server.jax.block_until_ready",
        lambda _value: pytest.fail("encode must not block the event loop"),
    )

    asyncio.run(_run_encoder(encoder, [{"modality": "IMAGE"}]))
