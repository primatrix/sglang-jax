import gzip
import json
import struct
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from sgl_jax.srt.utils import weight_utils
from sgl_jax.srt.utils.weight_utils import (
    SAFETENSORS_METADATA_CACHE_BASENAME,
    WeightLoader,
    _load_safetensors_metadata_cache,
    _read_sparse_safetensors_entries,
    _scan_safetensors_metadata,
    _write_safetensors_metadata_cache,
)

ROOT = Path(__file__).resolve().parents[3]
METADATA_CACHE_BUILDER = ROOT / "scripts/models/build_safetensors_metadata_cache.py"


def _write_safetensors_header(path, tensors):
    header = json.dumps(tensors).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(header)) + header)
    return 8 + len(header)


@pytest.mark.parametrize("num_threads", [1, 4])
def test_scan_safetensors_metadata_preserves_file_order(tmp_path, num_threads):
    first = tmp_path / "model-00001-of-00002.safetensors"
    second = tmp_path / "model-00002-of-00002.safetensors"
    first_data_offset = _write_safetensors_header(
        first,
        {
            "shared.weight": {
                "dtype": "BF16",
                "shape": [2, 4],
                "data_offsets": [0, 16],
            },
            "__metadata__": {"format": "pt"},
        },
    )
    second_data_offset = _write_safetensors_header(
        second,
        {
            "shared.weight": {
                "dtype": "F32",
                "shape": [1],
                "data_offsets": [4, 8],
            }
        },
    )

    result = _scan_safetensors_metadata([str(first), str(second)], num_threads=num_threads)

    assert result == {
        "shared.weight": [
            {
                "file": str(first),
                "shape": (2, 4),
                "dtype": "BF16",
                "byte_offset": first_data_offset,
                "byte_size": 16,
            },
            {
                "file": str(second),
                "shape": (1,),
                "dtype": "F32",
                "byte_offset": second_data_offset + 4,
                "byte_size": 4,
            },
        ]
    }


def test_scan_safetensors_metadata_accepts_empty_file_list():
    assert _scan_safetensors_metadata([], num_threads=4) == {}


def test_safetensors_metadata_cache_round_trip(tmp_path):
    shard = tmp_path / "model-00001-of-00001.safetensors"
    _write_safetensors_header(
        shard,
        {
            "model.weight": {
                "dtype": "BF16",
                "shape": [2, 4],
                "data_offsets": [0, 16],
            }
        },
    )
    weights_files = [str(shard)]
    expected = _scan_safetensors_metadata(weights_files, num_threads=1)
    cache = tmp_path / "sglang_jax.safetensors_metadata.v1.json.gz"

    _write_safetensors_metadata_cache(cache, weights_files, expected)

    assert _load_safetensors_metadata_cache(cache, weights_files) == expected


def test_safetensors_metadata_cache_rejects_changed_shard(tmp_path):
    shard = tmp_path / "model-00001-of-00001.safetensors"
    _write_safetensors_header(
        shard,
        {
            "model.weight": {
                "dtype": "BF16",
                "shape": [1],
                "data_offsets": [0, 2],
            }
        },
    )
    weights_files = [str(shard)]
    expected = _scan_safetensors_metadata(weights_files, num_threads=1)
    cache = tmp_path / "sglang_jax.safetensors_metadata.v1.json.gz"
    _write_safetensors_metadata_cache(cache, weights_files, expected)
    shard.write_bytes(shard.read_bytes() + b"changed")

    assert _load_safetensors_metadata_cache(cache, weights_files) is None


def test_safetensors_metadata_cache_rejects_non_object_payload(tmp_path):
    shard = tmp_path / "model-00001-of-00001.safetensors"
    _write_safetensors_header(
        shard,
        {
            "model.weight": {
                "dtype": "BF16",
                "shape": [1],
                "data_offsets": [0, 2],
            }
        },
    )
    cache = tmp_path / "sglang_jax.safetensors_metadata.v1.json.gz"
    with gzip.open(cache, "wt", encoding="utf-8") as fp:
        json.dump([], fp)

    assert _load_safetensors_metadata_cache(cache, [str(shard)]) is None


def test_weight_loader_uses_default_metadata_cache_before_scanning(tmp_path, monkeypatch):
    shard = tmp_path / "model-00001-of-00001.safetensors"
    _write_safetensors_header(
        shard,
        {
            "model.weight": {
                "dtype": "BF16",
                "shape": [2],
                "data_offsets": [0, 4],
            }
        },
    )
    weights_files = [str(shard)]
    expected = _scan_safetensors_metadata(weights_files, num_threads=1)
    cache = tmp_path / SAFETENSORS_METADATA_CACHE_BASENAME
    _write_safetensors_metadata_cache(cache, weights_files, expected)
    loader = WeightLoader.__new__(WeightLoader)
    loader.model_config = SimpleNamespace(model_path=str(tmp_path))
    loader._weight_info_cache = None

    monkeypatch.setattr(weight_utils.jax, "process_index", lambda: 0)
    monkeypatch.setattr(
        weight_utils.multihost_utils,
        "broadcast_one_to_all",
        lambda value, is_source: value,
    )

    def fail_scan(*args, **kwargs):
        raise AssertionError("header scanner should not run on a valid cache hit")

    monkeypatch.setattr(weight_utils, "_scan_safetensors_metadata", fail_scan)

    assert loader._scan_weight_info() == expected


def test_safetensors_metadata_cache_builder_writes_loadable_sidecar(tmp_path):
    shard = tmp_path / "model-00001-of-00001.safetensors"
    _write_safetensors_header(
        shard,
        {
            "model.weight": {
                "dtype": "BF16",
                "shape": [2],
                "data_offsets": [0, 4],
            }
        },
    )
    cache = tmp_path / "metadata.json.gz"

    result = subprocess.run(
        [
            sys.executable,
            str(METADATA_CACHE_BUILDER),
            "--model-path",
            str(tmp_path),
            "--output",
            str(cache),
            "--threads",
            "1",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "shards=1 tensors=1" in result.stdout
    assert _load_safetensors_metadata_cache(cache, [str(shard)]) is not None


def test_read_sparse_safetensors_entries_reads_offsets_concurrently(tmp_path):
    first = tmp_path / "first.safetensors"
    second = tmp_path / "second.safetensors"
    first.write_bytes(b"xx" + bytes([1, 2, 3, 4]) + b"----" + bytes([5, 6, 7, 8]))
    second.write_bytes(b"x" + bytes([9, 10, 11, 12]))

    result = _read_sparse_safetensors_entries(
        [
            (str(first), 7, 2),
            (str(first), 9, 10),
            (str(second), 11, 1),
        ],
        expert_nbytes=4,
        np_read_dtype=np.uint8,
        single_expert_shape=(2, 2),
        max_workers=3,
    )

    np.testing.assert_array_equal(result[7], [[1, 2], [3, 4]])
    np.testing.assert_array_equal(result[9], [[5, 6], [7, 8]])
    np.testing.assert_array_equal(result[11], [[9, 10], [11, 12]])


def test_read_sparse_safetensors_entries_rejects_short_reads(tmp_path):
    path = tmp_path / "short.safetensors"
    path.write_bytes(b"\x01\x02")

    with pytest.raises(OSError, match="short read"):
        _read_sparse_safetensors_entries(
            [(str(path), 0, 0)],
            expert_nbytes=4,
            np_read_dtype=np.uint8,
            single_expert_shape=(4,),
            max_workers=1,
        )
