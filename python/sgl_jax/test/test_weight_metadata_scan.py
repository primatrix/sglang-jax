import json
import struct

import numpy as np
import pytest

from sgl_jax.srt.utils.weight_utils import (
    _read_sparse_safetensors_entries,
    _scan_safetensors_metadata,
)


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
