import ml_dtypes
import numpy as np
import pytest

from sgl_jax.srt.multimodal.common.modality_enum import Modality, MultimodalDataItem
from sgl_jax.srt.multimodal.in_model import lane_packing
from sgl_jax.srt.multimodal.in_model.packing_copy import (
    _copy_f32_to_bf16_bits,
    copy_features,
)


@pytest.mark.parametrize("readonly", [False, True])
def test_compiled_copy_matches_numpy_rounding_and_preserves_source(readonly):
    high = np.arange(1 << 16, dtype=np.uint32)[:, None] << 16
    low = np.asarray([0, 0x7FFF, 0x8000, 0x8001, 0xFFFF], dtype=np.uint32)
    # Cover rounding boundaries for every BF16 bit pattern, plus arbitrary
    # FP32 values: signed NaNs/zero, subnormals, infinities and overflow.
    bits = np.concatenate(
        [
            (high | low).reshape(-1),
            np.random.default_rng(0).integers(0, 2**32, 1_000_000, dtype=np.uint32),
        ]
    )
    original = bits.copy()
    source = bits.view(np.float32)
    source.flags.writeable = not readonly
    storage = np.full(source.size + 2, 0x1234, dtype=np.uint16)
    destination = storage[1:-1].view(ml_dtypes.bfloat16)

    copy_features(destination, source)

    with np.errstate(invalid="ignore", over="ignore"):
        expected = source.astype(ml_dtypes.bfloat16)
    np.testing.assert_array_equal(destination.view(np.uint16), expected.view(np.uint16))
    np.testing.assert_array_equal(bits, original)
    np.testing.assert_array_equal(storage[[0, -1]], 0x1234)


@pytest.mark.parametrize(
    "source",
    [
        np.arange(48, dtype=np.float32).reshape(6, 8)[:, ::2],
        np.arange(24, dtype=np.float32).reshape(6, 4)[::-1],
        np.asfortranarray(np.arange(24, dtype=np.float32).reshape(6, 4)),
        np.arange(6, dtype=np.float32).reshape(6, 1),  # Broadcasting.
        np.arange(24, dtype=np.float64).reshape(6, 4),
        np.arange(24, dtype=np.float16).reshape(6, 4),
        np.arange(24, dtype=np.int32).reshape(6, 4),
        np.arange(24, dtype=np.float32).astype(ml_dtypes.bfloat16).reshape(6, 4),
        np.arange(24, dtype=np.float32).astype(">f4").reshape(6, 4),
        np.ndarray((6, 4), dtype=np.float32, buffer=bytearray(97), offset=1),
    ],
    ids=[
        "strided",
        "reversed",
        "fortran",
        "broadcast",
        "f64",
        "f16",
        "i32",
        "bf16",
        "big-endian",
        "unaligned",
    ],
)
@pytest.mark.parametrize("dtype", [ml_dtypes.bfloat16, np.float32])
def test_copy_preserves_numpy_fallbacks(source, dtype):
    expected = np.zeros((6, 4), dtype=dtype)
    expected[...] = source
    actual = np.zeros_like(expected)

    copy_features(actual, source)

    assert actual.tobytes() == expected.tobytes()


def _items(lengths, width, readonly):
    rng = np.random.default_rng(0)
    result = []
    for index, length in enumerate(lengths):
        feature = rng.normal(size=(length, width)).astype(np.float32)
        feature.flags.writeable = not readonly
        result.append(
            MultimodalDataItem(
                modality=Modality.IMAGE,
                feature=feature,
                hash=index + 1,
                placeholder_ranges=[(0, length // 4)],
                model_specific_data={
                    "image_grid_thw": np.asarray([1, 2, length // 2]),
                    "vision_layout": object(),
                },
            )
        )
    return result


def test_dynamic_packing_reuses_compilation_and_matches_numpy(monkeypatch):
    signatures = tuple(_copy_f32_to_bf16_bits.signatures)
    assert len(signatures) == 1  # Already compiled at module import.

    def unexpected_compilation(*args, **kwargs):
        pytest.fail("packing must not compile on the request path")

    monkeypatch.setattr(_copy_f32_to_bf16_bits, "compile", unexpected_compilation)
    cases = [
        ((4,), 4, 3, False),
        ((12, 4, 8), 2, 7, True),
        ((4, 16, 8, 12, 4), 4, 3, False),
        ((36, 20, 4), 1, 9, True),  # Power-of-two fallback bucket.
    ]
    for lengths, num_lanes, width, readonly in cases:
        items = _items(lengths, width, readonly)
        source_bytes = [item.feature.tobytes() for item in items]
        kwargs = dict(
            num_lanes=num_lanes, buckets=(8, 16, 32), merge_unit=4, dtype=ml_dtypes.bfloat16
        )
        actual = lane_packing.pack_vision_inputs(items, **kwargs)
        with monkeypatch.context() as patcher:
            patcher.setattr(lane_packing, "copy_features", lambda dst, src: np.copyto(dst, src))
            expected = lane_packing.pack_vision_inputs(items, **kwargs)
        for got, want in zip(actual[:3], expected[:3], strict=True):
            assert got.shape == want.shape and got.dtype == want.dtype
            assert got.tobytes() == want.tobytes()
        assert actual[3] == expected[3]

        patches, _, indices, layouts = actual
        valid_indices = indices[indices >= 0]
        restored = patches.reshape(-1, 4, width)[valid_indices].reshape(-1, width)
        original = np.concatenate([item.feature for item in items]).astype(ml_dtypes.bfloat16)
        np.testing.assert_array_equal(restored.view(np.uint16), original.view(np.uint16))
        assert sum(map(len, layouts)) == len(items)
        assert [item.feature.tobytes() for item in items] == source_bytes
        assert [item.hash for item in items] == list(range(1, len(items) + 1))
    assert tuple(_copy_f32_to_bf16_bits.signatures) == signatures
