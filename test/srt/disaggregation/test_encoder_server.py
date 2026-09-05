from __future__ import annotations

import asyncio
from types import MethodType, SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sgl_jax.srt.disaggregation.encoder.server import MMEncoder
from sgl_jax.srt.multimodal.common.modality_enum import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sgl_jax.srt.multimodal.processors.base_processor import BaseMultimodalProcessor


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
    encoder._simulate_compute = False
    encoder._precompile = False
    pending = iter(processed)

    async def process_request(self, request, modality):
        return next(pending), {}

    encoder._process_request = MethodType(process_request, encoder)
    return encoder


async def _run_encoder(encoder: MMEncoder, requests: list[dict]):
    prepared = await asyncio.gather(*(encoder.preprocess_request(request) for request in requests))
    output = encoder.encode_packed(encoder.build_batch(prepared))
    return output, encoder.metadata_for_packed(output)


def test_encode_preserves_packed_output_and_placeholder_counts():
    output = jnp.arange(16, dtype=jnp.float32).reshape(8, 2)
    encoder = _encoder(output, [_inputs(2), _inputs(3)])

    packed, metadata = asyncio.run(
        _run_encoder(encoder, [{"modality": "IMAGE"}, {"modality": "IMAGE"}])
    )

    assert packed.packed is output
    assert packed.batch.token_counts == (2, 3)
    assert len(metadata) == 2
    timing = metadata[0]["_encoder_timing"]
    assert timing["encode_server_postprocess_done_ns"] >= timing["encode_done_ns"]
    assert timing["encode_server_postprocess_duration_ns"] >= 0
    assert timing["encode_token_count_duration_ns"] >= 0
    assert timing["encode_embedding_slice_duration_ns"] >= 0
    assert timing["encode_split_compile_wait_duration_ns"] >= 0
    assert timing["encode_split_dispatch_duration_ns"] >= 0
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


def test_transfer_precompile_covers_homogeneous_batch_tails():
    processed = [_inputs(2), _inputs(2)]
    for mm_inputs in processed:
        mm_inputs.mm_items[0].feature = np.zeros((8, 3), dtype=np.float32)
    encoder = object.__new__(MMEncoder)
    encoder.model = SimpleNamespace(
        get_multimodal_embedding_packed_capacity=lambda items: 4 * len(items),
        mesh=jax.sharding.Mesh(np.array(jax.devices()), ("x",)),
    )
    encoder.model_config = SimpleNamespace(hidden_size=2, dtype=jnp.float32)
    encoder._max_batch_size = 3
    encoder._simulate_compute = False
    encoder._precompile = True

    specs = encoder._packed_transfer_specs(processed, (2, 2))

    assert [(spec.shape, counts) for spec, counts in specs] == [
        ((4, 2), (2,)),
        ((8, 2), (2, 2)),
        ((12, 2), (2, 2, 2)),
    ]
    encoder._precompile = False
    assert encoder._packed_transfer_specs(processed, (2, 2)) == ()


def test_language_prepare_reuses_encoder_item_hash():
    class Processor(BaseMultimodalProcessor):
        async def process_mm_data_async(self, *args, **kwargs):
            raise NotImplementedError

    processor = object.__new__(Processor)
    processor.hf_config = SimpleNamespace(
        vision_config=SimpleNamespace(spatial_merge_size=2),
        vision_start_token_id=1,
        vision_end_token_id=3,
        image_token_id=2,
        video_token_id=None,
        audio_token_id=None,
        audio_start_token_id=None,
        audio_end_token_id=None,
    )

    result = processor.get_mm_data(
        [1, 2, 3],
        {Modality.IMAGE: jnp.zeros((2, 4))},
        image_grid_thw=np.asarray([[1, 2, 4]], dtype=np.int32),
        item_hashes={Modality.IMAGE: [123]},
    )

    assert result.mm_items[0].hash == 123
    assert result.mm_items[0].pad_value == -124
    assert result.radix_input_ids == [1, -124, -124, 3]
