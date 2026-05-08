"""Discover Falcon/JAX runtime shape for calibration experiments.

This command intentionally records observations instead of inferring TPU
topology from slice names. Unknown fields are written as null.
"""

from __future__ import annotations

import argparse
import importlib.metadata as md
import os
import platform
import socket
from typing import Any

import jax

from benchmark.moe.calibration.common import collect_env, json_safe, write_jsonl

_DEVICE_ATTRS = (
    "id",
    "process_index",
    "platform",
    "device_kind",
    "coords",
    "core_on_chip",
    "slice_index",
    "task_id",
    "local_hardware_id",
)


def _package_version(name: str) -> str | None:
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


def _call_or_value(value: Any) -> Any:
    if callable(value):
        return value()
    return value


def _device_record(device: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "str": str(device),
        "repr": repr(device),
    }
    for attr in _DEVICE_ATTRS:
        try:
            value = getattr(device, attr)
        except Exception:
            value = None
        try:
            value = _call_or_value(value)
        except Exception:
            value = None
        record[attr] = json_safe(value)
    return record


def _logical_mesh_candidates(device_count: int) -> list[list[int]]:
    if device_count <= 0:
        return []
    candidates: list[list[int]] = [[device_count]]
    for a in range(1, device_count + 1):
        if device_count % a != 0:
            continue
        b = device_count // a
        if a <= b:
            candidates.append([a, b])
    for a in range(1, device_count + 1):
        if device_count % a != 0:
            continue
        rem = device_count // a
        for b in range(a, rem + 1):
            if rem % b != 0:
                continue
            c = rem // b
            if b <= c:
                candidates.append([a, b, c])
    deduped: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    for cand in candidates:
        key = tuple(cand)
        if key not in seen:
            seen.add(key)
            deduped.append(cand)
    return deduped


def build_record() -> dict[str, Any]:
    devices = jax.devices()
    local_devices = jax.local_devices()
    device_count = jax.device_count()
    local_device_count = jax.local_device_count()
    process_count = jax.process_count()

    falcon_env = collect_env(("FALCON_", "JOB_", "KUBERNETES_", "TPU_", "LIBTPU_", "JAX_", "XLA_"))
    falcon_device_count = os.getenv("FALCON_DEVICE_COUNT") or os.getenv(
        "FALCON_OPERATOR_DEVICE_COUNT"
    )
    falcon_device_topo = os.getenv("FALCON_DEVICE_TOPO") or os.getenv("FALCON_OPERATOR_DEVICE_TOPO")

    return {
        "schema_version": 1,
        "scenario": "hardware_discovery",
        "hostname": socket.gethostname(),
        "platform": {
            "python": platform.python_version(),
            "system": platform.platform(),
            "machine": platform.machine(),
        },
        "packages": {
            "jax": _package_version("jax"),
            "jaxlib": _package_version("jaxlib"),
            "libtpu": _package_version("libtpu"),
        },
        "falcon": {
            "device_count": (
                int(falcon_device_count)
                if falcon_device_count and falcon_device_count.isdigit()
                else None
            ),
            "device_topo": falcon_device_topo,
            "cluster_name": os.getenv("FALCON_CLUSTER_NAME"),
            "operator_rank": os.getenv("FALCON_OPERATOR_RANK"),
            "operator_is_leader": os.getenv("FALCON_OPERATOR_IS_LEADER"),
            "env": falcon_env,
        },
        "jax": {
            "default_backend": jax.default_backend(),
            "process_count": process_count,
            "process_index": jax.process_index(),
            "device_count": device_count,
            "local_device_count": local_device_count,
            "devices": [_device_record(device) for device in devices],
            "local_devices": [_device_record(device) for device in local_devices],
        },
        "derived": {
            "chip_count": None,
            "core_count": device_count,
            "logical_mesh_candidates": _logical_mesh_candidates(device_count),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Discover Falcon/JAX runtime shape.")
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to write one JSONL hardware discovery row.",
    )
    parser.add_argument(
        "--print",
        action="store_true",
        help="Also print the JSON object to stdout.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    row = build_record()
    write_jsonl(args.output, [row])
    if getattr(args, "print"):
        import json

        print(json.dumps(row, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
