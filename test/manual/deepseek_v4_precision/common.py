"""Checkpoint bytes, portable tensors and metrics for paired module validation."""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import ml_dtypes
import numpy as np

DTYPES = {
    "F32": np.float32,
    "BF16": ml_dtypes.bfloat16,
    "F8_E4M3": ml_dtypes.float8_e4m3fn,
    "I8": np.int8,
    "U8": np.uint8,
    "F8_E8M0": np.uint8,
    "I32": np.int32,
    "I64": np.int64,
}


class Checkpoint:
    def __init__(self, root):
        self.root = Path(root)
        config_bytes = (self.root / "config.json").read_bytes()
        index_bytes = (self.root / "model.safetensors.index.json").read_bytes()
        self.config = json.loads(config_bytes)
        self.index = json.loads(index_bytes)["weight_map"]
        self.headers = {}
        self.digests = {}
        self.identity = {
            "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "index_sha256": hashlib.sha256(index_bytes).hexdigest(),
        }

    def entry(self, key):
        shard = self.index[key]
        path = self.root / shard
        if shard not in self.headers:
            with path.open("rb") as f:
                size = struct.unpack("<Q", f.read(8))[0]
                if not 0 < size < 64 * 1024**2:
                    raise ValueError(f"invalid header size: {path}")
                self.headers[shard] = (8 + size, json.loads(f.read(size)))
        offset, header = self.headers[shard]
        return path, offset, header[key]

    def read(self, key):
        path, offset, entry = self.entry(key)
        start, end = entry["data_offsets"]
        with path.open("rb") as f:
            f.seek(offset + start)
            raw = f.read(end - start)
        if len(raw) != end - start:
            raise ValueError(f"truncated tensor {key}")
        self.digests[key] = hashlib.sha256(raw).hexdigest()
        return np.frombuffer(raw, DTYPES[entry["dtype"]]).reshape(entry["shape"]).copy()

    def block_fp8(self, stem):
        w = self.read(stem + ".weight").astype(np.float32)
        scale = self.read(stem + ".scale")
        if np.any(scale == 255):
            raise ValueError(f"reserved E8M0 value: {stem}")
        scale = np.ldexp(np.ones(scale.shape, np.float32), scale.astype(np.int16) - 127)
        return w * np.repeat(np.repeat(scale, 128, axis=0), 128, axis=1)[: w.shape[0], : w.shape[1]]

    def expert_reference(self, stem):
        """Independent source-format decoder; NOT the TPU promotion implementation."""
        packed = self.read(stem + ".weight").view(np.uint8)
        exponents = self.read(stem + ".scale")
        if np.any(exponents == 255):
            raise ValueError(f"reserved E8M0 value: {stem}")
        codebook = np.array(
            [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], np.float32
        )
        code = np.empty((packed.shape[0], packed.shape[1] * 2), np.uint8)
        code[:, 0::2], code[:, 1::2] = packed & 15, packed >> 4
        value = codebook[code].reshape(packed.shape[0], -1, 32)
        value = np.ldexp(value, exponents.astype(np.int16)[:, :, None] - 127).reshape(code.shape)
        if not np.isfinite(value).all():
            raise ValueError(f"nonfinite source weight: {stem}")
        return value


class Capture:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.arrays = {}

    def save(self, name, value):
        if hasattr(value, "detach"):
            original_dtype = str(value.dtype)
            value = (
                value.detach().float().cpu().numpy()
                if value.is_floating_point()
                else value.detach().cpu().numpy()
            )
        else:
            value = np.asarray(value)
            original_dtype = str(value.dtype)
            if value.dtype.kind != "i" and value.dtype.kind != "u" and value.dtype.kind != "b":
                value = value.astype(np.float32)
        file = name.replace("/", "__") + ".npy"
        np.save(self.root / file, value, allow_pickle=False)
        self.arrays[name] = {
            "file": file,
            "shape": list(value.shape),
            "original_dtype": original_dtype,
            "sha256": hashlib.sha256((self.root / file).read_bytes()).hexdigest(),
        }
        (self.root / "arrays.json").write_text(json.dumps(self.arrays, indent=2))


def compare(actual, reference):
    a, b = np.asarray(actual), np.asarray(reference)
    if a.shape != b.shape:
        return {"error": "shape_mismatch", "actual": list(a.shape), "reference": list(b.shape)}
    if a.dtype.kind in "iub" and b.dtype.kind in "iub":
        return {"mismatches": int(np.count_nonzero(a != b)), "elements": int(a.size)}
    a, b = a.astype(np.float64), b.astype(np.float64)
    nonfinite = int(np.count_nonzero(~np.isfinite(a)))
    reference_nonfinite = int(np.count_nonzero(~np.isfinite(b)))
    if nonfinite or reference_nonfinite:
        return {"nonfinite": nonfinite, "reference_nonfinite": reference_nonfinite}
    d = a - b
    aa, bb = a.reshape(a.shape[0], -1), b.reshape(b.shape[0], -1)
    an, bn = np.linalg.norm(aa, axis=1), np.linalg.norm(bb, axis=1)
    valid = (an > 0) & (bn > 0)
    cosine = np.sum(aa[valid] * bb[valid], axis=1) / (an[valid] * bn[valid])
    rel = np.linalg.norm(aa - bb, axis=1) / np.maximum(bn, 1e-30)
    return {
        "max_abs": float(np.abs(d).max()),
        "relative_l2": float(np.linalg.norm(d) / max(np.linalg.norm(b), 1e-30)),
        "worst_token_relative_l2": float(rel.max()),
        "worst_token": int(rel.argmax()),
        "cosine": float(np.sum(a * b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30)),
        "worst_token_cosine": float(cosine.min()) if valid.any() else None,
        "zero_norm_tokens": int(np.count_nonzero(~valid)),
        "nonfinite": 0,
        "reference_rms": float(np.sqrt(np.mean(b * b))),
    }
