"""CPU tests for the load-time MXFP4 to resident FP8 conversion path."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from sgl_jax.srt.utils.quantization.mxfp4_fp8_loader import (
    Mxfp4ConversionError,
    convert_mxfp4_pair,
    convert_mxfp4_pair_from_reader,
)


def _pack_codes(codes: np.ndarray) -> np.ndarray:
    return (codes[:, 0::2] | (codes[:, 1::2] << 4)).astype(np.uint8)


def test_reader_is_row_bounded_and_returns_existing_moe_layout() -> None:
    rows, columns = 5, 256
    codes = np.tile(np.arange(16, dtype=np.uint8), rows * columns // 16).reshape(rows, columns)
    packed = _pack_codes(codes)
    scales = np.full((rows, columns // 32), 127, dtype=np.uint8)
    weight_reads: list[int] = []
    scale_reads: list[int] = []

    def read_weight(row_slice: slice) -> np.ndarray:
        weight_reads.append(row_slice.stop - row_slice.start)
        return packed[row_slice]

    def read_scale(row_slice: slice) -> np.ndarray:
        scale_reads.append(row_slice.stop - row_slice.start)
        return scales[row_slice]

    converted = convert_mxfp4_pair_from_reader(
        weight_name="layers.0.ffn.experts.0.w1.weight",
        scale_name="layers.0.ffn.experts.0.w1.scale",
        weight_shape=packed.shape,
        scale_shape=scales.shape,
        weight_dtype="I8",
        scale_dtype="F8_E8M0",
        read_weight_rows=read_weight,
        read_scale_rows=read_scale,
        row_chunk_size=2,
    )

    assert max(weight_reads) <= 2
    assert max(scale_reads) <= 2
    assert converted.report.exact
    assert converted.report.measured_peak_python_bytes is None
    kernel_weight, kernel_scale = converted.as_moe_kernel_layout()
    assert kernel_weight.shape == (1, columns, rows)
    assert kernel_scale.shape == (1, 1, 1, rows)
    assert str(kernel_weight.dtype) == "float8_e4m3fn"
    assert kernel_scale.dtype == np.float32


def test_i8_storage_bits_are_preserved_and_names_are_exact() -> None:
    packed = np.asarray([[0xF1, 0x2E] * 16], dtype=np.uint8)
    scales = np.full((1, 2), 127, dtype=np.uint8)
    converted = convert_mxfp4_pair(
        packed.view(np.int8),
        scales.view(np.int8),
        weight_name="layers.0.ffn.experts.0.w2.weight",
        scale_name="layers.0.ffn.experts.0.w2.scale",
    )
    assert converted.report.exact
    with pytest.raises(Mxfp4ConversionError, match="pair mismatch"):
        convert_mxfp4_pair(
            packed,
            scales,
            weight_name="layers.0.ffn.experts.0.w2.weight",
            scale_name="layers.0.ffn.experts.0.w2.weight_scale",
        )
    with pytest.raises(Mxfp4ConversionError, match="I8/F8_E8M0"):
        convert_mxfp4_pair_from_reader(
            weight_name="layers.0.ffn.experts.0.w2.weight",
            scale_name="layers.0.ffn.experts.0.w2.scale",
            weight_shape=packed.shape,
            scale_shape=scales.shape,
            weight_dtype="F32",
            scale_dtype="F8_E8M0",
            read_weight_rows=lambda rows: packed[rows],
            read_scale_rows=lambda rows: scales[rows],
        )


def test_strict_mode_rejects_nonzero_fp8_underflow() -> None:
    # Two K32 groups with a 2**127 scale ratio cannot share one FP8 row scale
    # without erasing the small group. The default must fail closed.
    codes = np.full((1, 128), 2, dtype=np.uint8)  # E2M1 value 1.0
    packed = _pack_codes(codes)
    scales = np.asarray([[127, 0, 0, 0]], dtype=np.uint8)
    with pytest.raises(Mxfp4ConversionError, match="lossy"):
        convert_mxfp4_pair(
            packed,
            scales,
            weight_name="layers.0.ffn.experts.0.w3.weight",
            scale_name="layers.0.ffn.experts.0.w3.scale",
        )
    converted = convert_mxfp4_pair(
        packed,
        scales,
        weight_name="layers.0.ffn.experts.0.w3.weight",
        scale_name="layers.0.ffn.experts.0.w3.scale",
        strict=False,
    )
    assert converted.report.underflow_count > 0
    assert not converted.report.exact


def test_real_safetensors_06_f8_e8m0_roundtrip(tmp_path: Path) -> None:
    """Exercise the actual safetensors file format, including F8_E8M0 bytes."""

    ml_dtypes = pytest.importorskip("ml_dtypes")
    save_file = pytest.importorskip("safetensors.numpy").save_file
    weight_name = "layers.0.ffn.experts.0.w1.weight"
    scale_name = "layers.0.ffn.experts.0.w1.scale"
    codes = np.tile(np.arange(16, dtype=np.uint8), 16).reshape(1, 256)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).astype(np.uint8).view(np.int8)
    scales = np.full((1, 8), 127, dtype=np.uint8).view(ml_dtypes.float8_e8m0fnu)
    path = tmp_path / "model-00001.safetensors"
    save_file({weight_name: packed, scale_name: scales}, str(path))

    from sgl_jax.srt.utils.quantization.mxfp4_fp8_loader import (
        convert_mxfp4_pair_from_safetensors,
        inspect_mxfp4_checkpoint,
    )

    converted = convert_mxfp4_pair_from_safetensors(
        path,
        weight_name,
        scale_name,
        row_chunk_size=1,
    )
    assert converted.report.exact
    assert converted.scale_fp32.dtype == np.float32
    assert converted.as_moe_kernel_layout()[0].shape == (1, 256, 1)

    index = tmp_path / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {weight_name: path.name, scale_name: path.name}}))
    indexed_report = inspect_mxfp4_checkpoint(tmp_path, row_chunk_size=1)
    assert indexed_report["all_exact"]
    assert indexed_report["checkpoint_coverage"]["coverage_complete"]

    index.unlink()
    sample_report = inspect_mxfp4_checkpoint(tmp_path, row_chunk_size=1)
    assert sample_report["all_converted_tensors_exact"]
    assert not sample_report["all_exact"]
    assert not sample_report["checkpoint_coverage"]["coverage_complete"]


def test_real_sample_roundtrip_when_evidence_dir_is_provided() -> None:
    evidence_dir = os.environ.get("M02_REAL_SAMPLE_DIR")
    if not evidence_dir:
        pytest.skip("set M02_REAL_SAMPLE_DIR to run the captured DeepSeek-V4 sample")
    root = Path(evidence_dir)
    manifests = sorted(root.glob("real-*-sample-manifest.json"))
    assert manifests, f"no sample manifests in {root}"
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text())
        records = {record["name"]: record for record in manifest["tensors"]}
        for weight_name, weight_record in records.items():
            if not weight_name.endswith(".weight"):
                continue
            scale_name = f"{weight_name[:-len('.weight')]}.scale"
            scale_record = records[scale_name]
            assert weight_record["declared_dtype"] == "I8"
            assert scale_record["declared_dtype"] == "F8_E8M0"
            packed = np.fromfile(root / weight_record["raw_file"], dtype=np.uint8).reshape(
                weight_record["shape"]
            )
            scales = np.fromfile(root / scale_record["raw_file"], dtype=np.uint8).reshape(
                scale_record["shape"]
            )
            converted = convert_mxfp4_pair(
                packed,
                scales,
                weight_name=weight_name,
                scale_name=scale_name,
                row_chunk_size=127,
            )
            assert converted.report.exact
            assert converted.report.underflow_count == 0
            assert converted.as_moe_kernel_layout()[0].shape[1:] == (
                converted.logical_shape[1],
                converted.logical_shape[0],
            )


@pytest.mark.parametrize("spread", [0, 1, 4, 8, 14])
def test_exact_shortcut_matches_reference_for_all_codes_and_normal_exponents(spread):
    high = np.arange(27 + spread, 228, dtype=np.uint8)
    scales = np.stack((high, high - spread), axis=1)
    codes = np.tile(np.arange(16, dtype=np.uint8), (len(high), 4))
    packed = _pack_codes(codes)
    kwargs = dict(
        weight_name="layers.0.ffn.experts.0.w1.weight",
        scale_name="layers.0.ffn.experts.0.w1.scale",
        row_chunk_size=17,
    )
    reference = convert_mxfp4_pair(packed, scales, strict=False, **kwargs)
    actual = convert_mxfp4_pair(packed, scales, strict=True, **kwargs)
    assert reference.report.exact and actual.report.exact
    np.testing.assert_array_equal(
        actual.weight_fp8.view(np.uint8), reference.weight_fp8.view(np.uint8)
    )
    np.testing.assert_array_equal(actual.scale_fp32, reference.scale_fp32)
    assert actual.report.rel_l2 == reference.report.rel_l2 == 0


def test_exact_shortcut_preserves_signed_zero_and_loss_rejection():
    codes = np.tile(np.array([0, 8], np.uint8), (3, 32))
    scales = np.array([[27, 227], [126, 127], [227, 27]], np.uint8)
    kwargs = dict(
        weight_name="layers.0.ffn.experts.0.w1.weight",
        scale_name="layers.0.ffn.experts.0.w1.scale",
    )
    reference = convert_mxfp4_pair(_pack_codes(codes), scales, strict=False, **kwargs)
    actual = convert_mxfp4_pair(_pack_codes(codes), scales, **kwargs)
    np.testing.assert_array_equal(
        actual.weight_fp8.view(np.uint8), reference.weight_fp8.view(np.uint8)
    )
    np.testing.assert_array_equal(actual.scale_fp32, np.ones(3, np.float32))
    packed = _pack_codes(np.tile(np.arange(16, dtype=np.uint8), (2, 4)))
    # The first chunk is exact; the second exceeds the FP8 subnormal grid.
    scales = np.array([[127, 127], [127, 112]], np.uint8)
    reference = convert_mxfp4_pair(packed, scales, strict=False, row_chunk_size=1, **kwargs)
    assert not reference.report.exact and reference.report.underflow_count > 0
    with pytest.raises(Mxfp4ConversionError, match="lossy MXFP4->FP8"):
        convert_mxfp4_pair(packed, scales, strict=True, row_chunk_size=1, **kwargs)


@pytest.mark.parametrize("source_scale", [0, 26, 228, 253, 254, 255])
def test_shortcut_fallback_keeps_reference_extreme_exponent_behavior(source_scale):
    packed = _pack_codes(np.tile(np.arange(16, dtype=np.uint8), (1, 2)))
    scales = np.full((1, 1), source_scale, np.uint8)
    kwargs = dict(
        weight_name="layers.0.ffn.experts.0.w1.weight", scale_name="layers.0.ffn.experts.0.w1.scale"
    )
    try:
        reference = convert_mxfp4_pair(packed, scales, strict=False, **kwargs)
    except Mxfp4ConversionError:
        with pytest.raises(Mxfp4ConversionError):
            convert_mxfp4_pair(packed, scales, strict=True, **kwargs)
    else:
        actual = convert_mxfp4_pair(packed, scales, strict=True, **kwargs)
        np.testing.assert_array_equal(
            actual.weight_fp8.view(np.uint8), reference.weight_fp8.view(np.uint8)
        )
        np.testing.assert_array_equal(actual.scale_fp32, reference.scale_fp32)


def test_shortcut_row_scale_covers_each_maximum_magnitude_code():
    rng = np.random.default_rng(340)
    maxima = np.tile(np.arange(8), 16)
    codes = np.stack([rng.integers(0, int(m) + 1, 128, dtype=np.uint8) for m in maxima])
    codes |= rng.integers(0, 2, codes.shape, dtype=np.uint8) << 3
    scales = rng.integers(40, 200, (128, 1), dtype=np.uint8) + rng.integers(
        0, 8, (128, 4), dtype=np.uint8
    )
    kwargs = dict(
        weight_name="layers.0.ffn.experts.0.w1.weight",
        scale_name="layers.0.ffn.experts.0.w1.scale",
        row_chunk_size=17,
    )
    reference = convert_mxfp4_pair(_pack_codes(codes), scales, strict=False, **kwargs)
    actual = convert_mxfp4_pair(_pack_codes(codes), scales, **kwargs)
    np.testing.assert_array_equal(
        actual.weight_fp8.view(np.uint8), reference.weight_fp8.view(np.uint8)
    )
    np.testing.assert_array_equal(actual.scale_fp32, reference.scale_fp32)
