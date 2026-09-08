"""Static artifact integrity, exact conversion and fail-closed publication."""

import json
import struct

import numpy as np
import pytest

from sgl_jax.srt.utils.quantization.deepseek_v4_static_fp8 import (
    COMPLETE,
    CONFIG_KEY,
    FORMAT,
    export_checkpoint,
    read_static_pair,
    validate_static_checkpoint,
)
from sgl_jax.srt.utils.quantization.mxfp4_fp8_loader import (
    Mxfp4ConversionError,
    _read_safetensors_header,
    convert_mxfp4_pair_from_safetensors,
)


def fixture(path, lossy=False):
    path.mkdir()
    tensors = {"keep.scale": (np.array([126, 129], np.uint8), "F8_E8M0")}
    for projection in ["w1", "w2", "w3"]:
        key = f"layers.0.ffn.experts.0.{projection}"
        tensors[key + ".weight"] = (np.full((4, 32), 0xF1, np.uint8).view(np.int8), "I8")
        scale = np.full((4, 2), 127, np.uint8)
        if lossy:
            scale[:, 1] = 0
        tensors[key + ".scale"] = (scale, "F8_E8M0")
    # Different shards for weight and scale exercises cross-shard pairing.
    weight_map = {}
    for number, suffix in enumerate(["weight", "scale"]):
        header, raw, offset = {}, [], 0
        for name, (value, kind) in tensors.items():
            if not name.endswith(suffix):
                continue
            data = value.tobytes()
            header[name] = {
                "dtype": kind,
                "shape": list(value.shape),
                "data_offsets": [offset, offset + len(data)],
            }
            offset += len(data)
            raw.append(data)
            weight_map[name] = f"part{number}.safetensors"
        h = json.dumps(header).encode()
        h += b" " * (-len(h) % 8)
        (path / f"part{number}.safetensors").write_bytes(
            struct.pack("<Q", len(h)) + h + b"".join(raw)
        )
    (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    (path / "config.json").write_text(
        json.dumps(
            {
                "num_hidden_layers": 1,
                "n_routed_experts": 1,
                "quantization_config": {"weight_block_size": [128, 128]},
            }
        )
    )
    (path / "tokenizer_config.json").write_text("{}")
    return weight_map


def export(source, output, **kwargs):
    return export_checkpoint(
        source, output, source_revision="source-sha", converter_revision="converter-sha", **kwargs
    )


def test_exact_roundtrip_nonexpert_bytes_and_resume(tmp_path):
    source, output = tmp_path / "source", tmp_path / "output"
    index = fixture(source)
    proof = export(source, output)
    assert proof["expert_pairs"] == 3 and proof["all_converted_exact"]
    cfg = json.loads((output / "config.json").read_text())
    assert cfg[CONFIG_KEY] == FORMAT and cfg["quantization_config"] == {
        "weight_block_size": [128, 128]
    }
    target_index = json.loads((output / "model.safetensors.index.json").read_text())["weight_map"]
    assert set(index) == set(target_index)
    for w in ["w1", "w2", "w3"]:
        key = f"layers.0.ffn.experts.0.{w}.weight"
        scale = key.removesuffix(".weight") + ".scale"
        expected = convert_mxfp4_pair_from_safetensors(
            source / index[key], key, scale, scale_file=source / index[scale]
        )
        weight, scales = read_static_pair(
            output / target_index[key], key, output / target_index[scale], scale
        )
        np.testing.assert_array_equal(weight.view(np.uint8), expected.weight_fp8.view(np.uint8))
        np.testing.assert_array_equal(scales, expected.scale_fp32)
    p = output / target_index["keep.scale"]
    entry = _read_safetensors_header(p)["keep.scale"]
    assert entry.dtype == "F8_E8M0"
    assert p.read_bytes()[entry.byte_offset : entry.byte_offset + entry.byte_size] == bytes(
        [126, 129]
    )
    assert validate_static_checkpoint(output, verify_files=True) == proof
    assert export(source, output, resume=True) == proof
    (output / COMPLETE).unlink()
    assert export(source, output, resume=True) == proof
    with pytest.raises(FileExistsError):
        export(source, output)


def test_lossy_source_has_no_completion(tmp_path):
    source, output = tmp_path / "source", tmp_path / "output"
    fixture(source, lossy=True)
    with pytest.raises(Mxfp4ConversionError, match="lossy"):
        export(source, output)
    assert not (output / COMPLETE).exists()
    with pytest.raises(FileNotFoundError):
        validate_static_checkpoint(output)


def test_corruption_and_changed_source_rejected(tmp_path):
    source, output = tmp_path / "source", tmp_path / "output"
    fixture(source)
    proof = export(source, output)
    name = next(iter(proof["shards"]))
    p = output / name
    data = bytearray(p.read_bytes())
    data[-1] ^= 1
    p.write_bytes(data)
    with pytest.raises(ValueError, match="checksum"):
        validate_static_checkpoint(output, verify_files=True)
    with pytest.raises(ValueError, match="checksum"):
        export(source, output, resume=True)
    config = source / "config.json"
    config.write_text(config.read_text() + " ")
    with pytest.raises(ValueError, match="identity"):
        export(source, output, resume=True)


def test_modified_config_and_metadata_not_accepted(tmp_path):
    source, output = tmp_path / "source", tmp_path / "output"
    fixture(source)
    export(source, output)
    (output / "config.json").write_text("{}")
    with pytest.raises(ValueError, match="metadata"):
        validate_static_checkpoint(output)


@pytest.mark.parametrize("bad_code", [0x7F, 0xFF])
def test_explicit_pair_validation_rejects_both_fp8_nan_encodings(tmp_path, bad_code):
    source, output = tmp_path / "source", tmp_path / "output"
    fixture(source)
    export(source, output)
    index = json.loads((output / "model.safetensors.index.json").read_text())["weight_map"]
    weight = "layers.0.ffn.experts.0.w1.weight"
    scale = "layers.0.ffn.experts.0.w1.scale"
    path = output / index[weight]
    entry = _read_safetensors_header(path)[weight]
    with path.open("r+b") as handle:
        handle.seek(entry.byte_offset)
        handle.write(bytes([bad_code]))
    with pytest.raises(ValueError, match="nonfinite"):
        read_static_pair(path, weight, output / index[scale], scale)
