from __future__ import annotations

import json
import os
import statistics
from pathlib import Path
from typing import Any

_SENSITIVE_KEY_PARTS = ("TOKEN", "SECRET", "PASSWORD", "KEY", "CREDENTIAL")
SCHEMA_VERSION = 1

OBSERVATION_REQUIRED_FIELDS = (
    "schema_version",
    "status",
    "scenario",
    "suite",
    "layer",
    "path",
    "path_class",
    "dtype",
    "weight_dtype",
    "t_packing",
    "bf",
    "bd",
    "tile_shape",
    "bytes_hbm",
    "bytes_per_fetch",
    "dma_count",
    "latency_ms_samples",
    "latency_ms_p50",
    "latency_ms_p90",
    "runtime",
    "source",
    "metadata",
)


def sanitize_env_value(key: str, value: str) -> str:
    upper_key = key.upper()
    if any(part in upper_key for part in _SENSITIVE_KEY_PARTS):
        return "<redacted>"
    return value


def collect_env(prefixes: tuple[str, ...]) -> dict[str, str]:
    result: dict[str, str] = {}
    for key, value in sorted(os.environ.items()):
        if key.startswith(prefixes):
            result[key] = sanitize_env_value(key, value)
    return result


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    return str(value)


def _int_env(*names: str) -> int | None:
    for name in names:
        value = os.getenv(name)
        if value is None or value == "":
            continue
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _str_env(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


def _runtime_identity_from_jax() -> dict[str, Any]:
    try:
        import jax
    except Exception as exc:
        return {
            "jax_available": False,
            "jax_error": f"{type(exc).__name__}: {exc}",
            "default_backend": None,
            "device_type": None,
            "jax_device_count": None,
            "jax_local_device_count": None,
            "jax_process_count": None,
            "jax_process_index": None,
        }

    try:
        devices = jax.devices()
    except Exception:
        devices = []

    device_kind = None
    if devices:
        try:
            device_kind = getattr(devices[0], "device_kind", None)
        except Exception:
            device_kind = None

    try:
        default_backend = jax.default_backend()
    except Exception:
        default_backend = None

    return {
        "jax_available": True,
        "jax_error": None,
        "default_backend": default_backend,
        "device_type": device_kind or default_backend,
        "jax_device_count": _safe_jax_int(jax, "device_count"),
        "jax_local_device_count": _safe_jax_int(jax, "local_device_count"),
        "jax_process_count": _safe_jax_int(jax, "process_count"),
        "jax_process_index": _safe_jax_int(jax, "process_index"),
    }


def _safe_jax_int(jax_module: Any, name: str) -> int | None:
    try:
        return int(getattr(jax_module, name)())
    except Exception:
        return None


def collect_runtime_identity() -> dict[str, Any]:
    """Collect the compact runtime fields used by calibration observations."""

    jax_identity = _runtime_identity_from_jax()
    accelerator_type = _str_env("CALIBRATION_ACCELERATOR_TYPE", "TPU_ACCELERATOR_TYPE")
    device_type = (
        _str_env("CALIBRATION_DEVICE_TYPE", "FALCON_DEVICE_TYPE")
        or accelerator_type
        or jax_identity.get("device_type")
    )
    falcon_device_topo = _str_env(
        "CALIBRATION_FALCON_DEVICE_TOPO",
        "FALCON_DEVICE_TOPO",
        "FALCON_OPERATOR_DEVICE_TOPO",
        "TPU_TOPOLOGY",
    )

    jax_device_count = _int_env("CALIBRATION_EXPECTED_JAX_DEVICE_COUNT")
    if jax_device_count is None:
        jax_device_count = jax_identity.get("jax_device_count")

    jax_local_device_count = _int_env("CALIBRATION_EXPECTED_JAX_LOCAL_DEVICE_COUNT")
    if jax_local_device_count is None:
        jax_local_device_count = jax_identity.get("jax_local_device_count")

    process_count = _int_env("CALIBRATION_EXPECTED_JAX_PROCESS_COUNT")
    if process_count is None:
        process_count = jax_identity.get("jax_process_count")

    falcon_device_count = _int_env(
        "CALIBRATION_FALCON_DEVICE_COUNT",
        "FALCON_DEVICE_COUNT",
        "FALCON_OPERATOR_DEVICE_COUNT",
    )
    if falcon_device_count is None and (device_type == "v7x" or falcon_device_topo):
        falcon_device_count = jax_device_count

    replica = _int_env(
        "CALIBRATION_REPLICA",
        "FALCON_REPLICA",
        "FALCON_OPERATOR_REPLICA",
        "REPLICA",
    )
    if replica is None:
        replica = process_count

    tensorcore_count = _int_env(
        "CALIBRATION_TENSORCORE_OR_JAX_DEVICE_COUNT",
        "FALCON_TENSORCORE_COUNT",
        "TPU_TENSORCORE_COUNT",
    )
    if tensorcore_count is None:
        tensorcore_count = jax_device_count or falcon_device_count

    chip_count = _int_env("CALIBRATION_CHIP_COUNT", "FALCON_CHIP_COUNT", "TPU_CHIP_COUNT")
    if chip_count is None and tensorcore_count is not None and tensorcore_count >= 2:
        chip_count = tensorcore_count // 2

    runtime = {
        "device_type": device_type,
        "accelerator_type": accelerator_type,
        "falcon_device_count": falcon_device_count,
        "falcon_device_topo": falcon_device_topo,
        "replica": replica,
        "jax_device_count": jax_device_count,
        "jax_local_device_count": jax_local_device_count,
        "jax_process_count": process_count,
        "jax_process_index": jax_identity.get("jax_process_index"),
        "chip_count": chip_count,
        "tensorcore_or_jax_device_count": tensorcore_count,
        "default_backend": jax_identity.get("default_backend"),
        "jax_available": jax_identity.get("jax_available"),
        "jax_error": jax_identity.get("jax_error"),
    }
    return json_safe(runtime)


def percentile(samples: list[float], percent: float) -> float | None:
    if not samples:
        return None
    if len(samples) == 1:
        return float(samples[0])
    ordered = sorted(float(sample) for sample in samples)
    index = (len(ordered) - 1) * percent
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def build_observation_row(
    *,
    scenario: str,
    suite: str,
    layer: int,
    path: str,
    path_class: str,
    dtype: str,
    weight_dtype: str,
    t_packing: int,
    bf: int,
    bd: int,
    tile_shape: tuple[int, ...] | list[int],
    bytes_hbm: int,
    bytes_per_fetch: int,
    dma_count: int,
    status: str,
    execution_mode: str,
    latency_ms_samples: list[float] | None = None,
    runtime: dict[str, Any] | None = None,
    source: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    implementation_note: str | None = None,
) -> dict[str, Any]:
    samples = [float(sample) for sample in latency_ms_samples or []]
    row = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "scenario": scenario,
        "suite": suite,
        "layer": layer,
        "path": path,
        "path_class": path_class,
        "dtype": dtype,
        "weight_dtype": weight_dtype,
        "t_packing": t_packing,
        "bf": bf,
        "bd": bd,
        "tile_shape": list(tile_shape),
        "bytes_hbm": bytes_hbm,
        "bytes_per_fetch": bytes_per_fetch,
        "dma_count": dma_count,
        "latency_ms_samples": samples,
        "latency_ms_p50": percentile(samples, 0.50),
        "latency_ms_p90": percentile(samples, 0.90),
        "latency_ms_mean": statistics.fmean(samples) if samples else None,
        "runtime": runtime if runtime is not None else collect_runtime_identity(),
        "source": source or {},
        "metadata": metadata or {},
        "execution_mode": execution_mode,
        "implementation_note": implementation_note,
    }
    for field in OBSERVATION_REQUIRED_FIELDS:
        row.setdefault(field, None)
    return json_safe(row)


def write_jsonl(path: str | os.PathLike[str], rows: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
