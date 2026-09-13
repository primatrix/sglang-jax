"""Export an immutable, resumable V4 trunk FP8 checkpoint without changing other tensors.

This is the sglang-jax V4 mixed format, not a generic blockwise-FP8 checkpoint.
Routed trunk experts use logical [N,K] E4M3FN and [N] FP32 power-of-two scales.
All other tensors (including unused MTP) retain their original bytes and dtype.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import struct
from pathlib import Path

import ml_dtypes
import numpy as np

from sgl_jax.srt.utils.quantization.mxfp4_fp8_loader import (
    _checkpoint_locations,
    _read_safetensors_header,
    _SafetensorsTensor,
    convert_mxfp4_pair_from_safetensors,
)

FORMAT = "sglang-jax-deepseek-v4-expert-fp8-per-channel-v1"
CONFIG_KEY = "sglang_jax_expert_format"
COMPLETE = "static-fp8-complete.json"
EXPERT = re.compile(r"^layers\.(\d+)\.ffn\.experts\.(\d+)\.(w[123])\.weight$")
CHUNK = 8 * 1024 * 1024


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def read_static_pair(weight_file, weight_name, scale_file, scale_name, *, entries=None):
    """Read saved FP8/scale payloads; no quantization or MXFP4 decoding occurs."""
    if entries is None:
        wh = _read_safetensors_header(Path(weight_file))[weight_name]
        sh = _read_safetensors_header(Path(scale_file))[scale_name]
    else:
        wh, sh = (
            _SafetensorsTensor(
                path=Path(entry["file"]),
                name=name,
                dtype=entry["dtype"],
                shape=tuple(entry["shape"]),
                byte_offset=entry["byte_offset"],
                byte_size=entry["byte_size"],
            )
            for entry, name in zip(entries, (weight_name, scale_name), strict=True)
        )
    if wh.dtype != "F8_E4M3" or sh.dtype != "F32" or len(wh.shape) != 2:
        raise ValueError(f"invalid static FP8 pair: {weight_name}")
    if (
        sh.shape != (wh.shape[0],)
        or wh.byte_size != np.prod(wh.shape)
        or sh.byte_size != 4 * wh.shape[0]
    ):
        raise ValueError(f"invalid static FP8 shape/bytes: {weight_name}")

    def read(entry, dtype):
        with entry.path.open("rb") as f:
            f.seek(entry.byte_offset)
            raw = f.read(entry.byte_size)
        if len(raw) != entry.byte_size:
            raise ValueError(f"truncated static tensor: {entry.name}")
        return np.frombuffer(raw, dtype=dtype).reshape(entry.shape)

    w = read(wh, ml_dtypes.float8_e4m3fn)
    s = read(sh, "<f4")
    mantissa, _ = np.frexp(s)
    if not np.isfinite(w).all() or not np.isfinite(s).all() or not np.all(mantissa == 0.5):
        raise ValueError(f"nonfinite weight or nonpositive/non-power-of-two scale: {weight_name}")
    return w, s


def validate_static_checkpoint(checkpoint, *, verify_files=False):
    """Require a complete publication and its exact config/index, optionally hash all shards."""
    p = Path(checkpoint)
    proof = json.loads((p / COMPLETE).read_text())
    if proof.get("format") != FORMAT:
        raise ValueError("unsupported static checkpoint format")
    for name, digest in proof["metadata_sha256"].items():
        if Path(name).name != name or sha256(p / name) != digest:
            raise ValueError(f"static checkpoint metadata mismatch: {name}")
    index = json.loads((p / "model.safetensors.index.json").read_text())["weight_map"]
    if set(index.values()) != set(proof["shards"]):
        raise ValueError("static checkpoint shard coverage mismatch")
    for name, record in proof["shards"].items():
        if (
            Path(name).name != name
            or not (p / name).is_file()
            or (p / name).stat().st_size != record["bytes"]
        ):
            raise ValueError(f"static checkpoint shard missing or wrong size: {name}")
        if verify_files and sha256(p / name) != record["sha256"]:
            raise ValueError(f"static checkpoint shard checksum mismatch: {name}")
    return proof


def export_checkpoint(source, output, *, source_revision, converter_revision, resume=False):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output == source or source in output.parents or output in source.parents:
        raise ValueError("source and output must be separate non-nested directories")
    config = json.loads((source / "config.json").read_text())
    if config.get(CONFIG_KEY):
        raise ValueError("source is already a converted checkpoint")
    locations, coverage = _checkpoint_locations(source)
    if not coverage["coverage_complete"]:
        raise ValueError("full indexed source checkpoint required")
    if output.exists() and any(output.iterdir()) and not resume:
        raise FileExistsError("nonempty output requires explicit --resume")
    output.mkdir(parents=True, exist_ok=True)
    headers = {
        path: _read_safetensors_header(path) for path in sorted({v[0] for v in locations.values()})
    }
    source_hashes = {}
    for path in headers:
        source_hashes[path.name] = sha256(path)
        print("SOURCE_SHA256", path.name, source_hashes[path.name], flush=True)
    identity = {
        "format": FORMAT,
        "source_revision": source_revision,
        "converter_revision": converter_revision,
        "config_sha256": sha256(source / "config.json"),
        "index_sha256": sha256(source / "model.safetensors.index.json"),
        "source_shards": source_hashes,
    }
    plan_path = output / "conversion-source.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != identity:
        raise ValueError("resume source/converter identity mismatch")
    write_json(plan_path, identity)
    if (output / COMPLETE).exists():
        return validate_static_checkpoint(output, verify_files=True)
    pairs = {}
    for name, (path, shape, dtype) in locations.items():
        match = EXPERT.fullmatch(name)
        if not match:
            continue
        if (
            int(match[1]) >= config["num_hidden_layers"]
            or int(match[2]) >= config["n_routed_experts"]
        ):
            raise ValueError(f"expert outside configured trunk: {name}")
        scale = name.removesuffix(".weight") + ".scale"
        if dtype != "I8" or scale not in locations or locations[scale][2] != "F8_E8M0":
            raise ValueError(f"invalid source expert: {name}")
        pairs[name] = scale
    expected = config["num_hidden_layers"] * config["n_routed_experts"] * 3
    if len(pairs) != expected:
        raise ValueError(f"expected {expected} expert pairs, found {len(pairs)}")
    paired_scales = set(pairs.values())
    shards, weight_map, total_size = {}, {}, 0
    for shard_number, path in enumerate(headers, 1):
        names = sorted(
            name for name, v in locations.items() if v[0] == path and name not in paired_scales
        )
        if not names:
            continue
        name_out = f"model-{shard_number:05d}.safetensors"
        header, tasks, offset = {}, [], 0
        for name in names:
            entry = headers[path][name]
            if name in pairs:
                n, k2 = entry.shape
                specs = [
                    (name, "F8_E4M3", (n, k2 * 2), n * k2 * 2),
                    (pairs[name], "F32", (n,), n * 4),
                ]
            else:
                specs = [(name, entry.dtype, entry.shape, entry.byte_size)]
            for key, dtype, shape, size in specs:
                header[key] = {
                    "dtype": dtype,
                    "shape": list(shape),
                    "data_offsets": [offset, offset + size],
                }
                weight_map[key] = name_out
                offset += size
            tasks.append(name)
        total_size += offset
        receipt = output / (name_out + ".receipt.json")
        target = output / name_out
        if resume and receipt.exists() and target.exists():
            record = json.loads(receipt.read_text())
            if record["bytes"] == target.stat().st_size and record["sha256"] == sha256(target):
                shards[name_out] = record
                print("RESUME_VERIFIED", name_out, flush=True)
                continue
            raise ValueError(f"resume shard is corrupt: {name_out}")
        raw_header = json.dumps(header, separators=(",", ":")).encode()
        raw_header += b" " * (-len(raw_header) % 8)
        temporary = output / (name_out + ".partial")
        digest = hashlib.sha256()
        tensor_hashes = {}
        with temporary.open("wb") as out, path.open("rb") as original:

            def write(raw, out=out, digest=digest):
                out.write(raw)
                digest.update(raw)

            write(struct.pack("<Q", len(raw_header)))
            write(raw_header)
            for name in tasks:
                if name in pairs:
                    scale = pairs[name]
                    converted = convert_mxfp4_pair_from_safetensors(
                        path, name, scale, scale_file=locations[scale][0], strict=True
                    )
                    for key, array in [
                        (name, converted.weight_fp8),
                        (scale, converted.scale_fp32.astype("<f4", copy=False)),
                    ]:
                        raw = array.tobytes(order="C")
                        expected_bytes = (
                            header[key]["data_offsets"][1] - header[key]["data_offsets"][0]
                        )
                        if len(raw) != expected_bytes:
                            raise ValueError(f"converted size mismatch: {key}")
                        write(raw)
                        tensor_hashes[key] = hashlib.sha256(raw).hexdigest()
                    del converted
                else:
                    entry = headers[path][name]
                    original.seek(entry.byte_offset)
                    remaining, h = entry.byte_size, hashlib.sha256()
                    while remaining:
                        block = original.read(min(CHUNK, remaining))
                        if not block:
                            raise ValueError(f"truncated source tensor: {name}")
                        write(block)
                        h.update(block)
                        remaining -= len(block)
                    tensor_hashes[name] = h.hexdigest()
        if sha256(temporary) != digest.hexdigest():
            raise ValueError(f"output readback mismatch: {name_out}")
        os.replace(temporary, target)
        record = {
            "sha256": digest.hexdigest(),
            "bytes": target.stat().st_size,
            "tensor_sha256": tensor_hashes,
        }
        write_json(receipt, record)
        shards[name_out] = record
        print("SHARD_VERIFIED", name_out, record["bytes"], flush=True)
    if set(weight_map) != set(locations):
        raise ValueError("output tensor coverage differs from source")
    # Explicit allowlist: carry model assets, never credentials or download scratch metadata.
    for name in [
        "tokenizer.json",
        "tokenizer_config.json",
        "generation_config.json",
        "LICENSE",
        "README.md",
        "chat_template.jinja",
    ]:
        if (source / name).is_file():
            shutil.copyfile(
                source / name, output / ("SOURCE_README.md" if name == "README.md" else name)
            )
    for name in ["encoding", "inference"]:
        if (source / name).is_dir():
            shutil.copytree(source / name, output / name, dirs_exist_ok=True)
    config[CONFIG_KEY] = FORMAT
    write_json(output / "config.json", config)
    write_json(
        output / "model.safetensors.index.json",
        {"metadata": {"total_size": total_size}, "weight_map": weight_map},
    )
    (output / "README.md").write_text(
        "# DeepSeek V4 mixed static FP8 checkpoint for sglang-jax\n\n"
        "Trunk routed experts: E4M3FN [N,K], FP32 power-of-two scale [N].\n"
        "Every conversion is strict/exact relative to decoded source MXFP4. Other tensors, including MTP, retain source bytes.\n"
        "Requires the sglang-jax V4 static expert loader; not a generic blockwise FP8 or upstream GPU checkpoint.\n"
        "The original model card and license are preserved. See conversion-source.json for provenance, and static-fp8-complete.json for checksums.\n"
        "MTP inference is not validated. Chat requests require the model's native encoding.\n"
    )
    metadata = ["config.json", "model.safetensors.index.json", "conversion-source.json"]
    proof = {
        "format": FORMAT,
        "expert_pairs": len(pairs),
        "all_converted_exact": True,
        "tensor_count": len(weight_map),
        "metadata_sha256": {n: sha256(output / n) for n in metadata},
        "shards": shards,
    }
    # Completion is published last, only after every shard was closed and read back.
    write_json(output / COMPLETE, proof)
    print("STATIC_CHECKPOINT_COMPLETE", len(pairs), len(weight_map), total_size, flush=True)
    return proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--converter-revision", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    export_checkpoint(
        args.source,
        args.output,
        source_revision=args.source_revision,
        converter_revision=args.converter_revision,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
