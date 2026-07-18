import functools
import json
import os
import re
import tempfile
import threading
from collections import defaultdict
from enum import IntEnum
from pathlib import Path

import jax
import numpy as np


class FrameworkLogLevel(IntEnum):
    ERROR = 0
    WARN = 1
    INFO = 2
    DEBUG = 3
    TRACE = 4


FRAMEWORK_LOG_LEVEL = FrameworkLogLevel(int(os.environ.get("SGLANG_FRAMEWORK_LOG_LEVEL", "0")))

_DEBUG_DUMP_LOCK = threading.Lock()
_DEBUG_DUMP_OCCURRENCES = defaultdict(int)
_UNSAFE_FILENAME_CHARS = re.compile(r"[^A-Za-z0-9_.-]+")


def _debug_dump_filter(name):
    value = os.environ.get(name, "")
    return {item.strip() for item in value.split(",") if item.strip()}


def _sanitize_filename_part(value):
    sanitized = _UNSAFE_FILENAME_CHARS.sub("-", str(value)).strip(".-")
    return sanitized or "unknown"


def _normalize_forward_mode(forward_mode):
    if forward_mode is None:
        return "unknown"
    if hasattr(forward_mode, "name"):
        return str(forward_mode.name).lower()
    return str(forward_mode)


def _write_debug_dump(host_value, *, dump_dir, metadata):
    host_array = np.asarray(host_value)
    semantic_key = (
        metadata["process"],
        metadata["component"],
        metadata["layer"],
        metadata["forward_mode"],
        metadata["name"],
    )

    with _DEBUG_DUMP_LOCK:
        occurrence = _DEBUG_DUMP_OCCURRENCES[semantic_key]
        _DEBUG_DUMP_OCCURRENCES[semantic_key] += 1
        filename = "__".join(
            (
                f"p{metadata['process']:05d}",
                _sanitize_filename_part(metadata["component"]),
                f"l{metadata['layer']:05d}" if metadata["layer"] is not None else "l-none",
                _sanitize_filename_part(metadata["forward_mode"]),
                _sanitize_filename_part(metadata["name"]),
                f"o{occurrence:05d}",
            )
        ) + ".npy"
        output_dir = Path(dump_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / filename

        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=output_dir, prefix=f".{filename}.", delete=False
            ) as temporary:
                temporary_path = Path(temporary.name)
                np.save(temporary, host_array, allow_pickle=False)
            os.replace(temporary_path, output_path)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()

        manifest_row = {
            "filename": filename,
            **metadata,
            "occurrence": occurrence,
            "shape": list(host_array.shape),
            "dtype": str(host_array.dtype),
        }
        manifest_path = output_dir / f"manifest-p{metadata['process']:05d}.jsonl"
        with manifest_path.open("a", encoding="utf-8") as manifest:
            manifest.write(json.dumps(manifest_row, sort_keys=True) + "\n")


def maybe_dump_jax_array(
    value,
    *,
    component,
    name,
    layer_id=None,
    forward_mode=None,
):
    """Optionally dump a JAX value on its host while returning it unchanged."""
    if os.environ.get("SGLANG_JAX_DEBUG_DUMP", "0") != "1":
        return value

    process = jax.process_index()
    components = _debug_dump_filter("SGLANG_JAX_DEBUG_DUMP_COMPONENTS")
    layers = _debug_dump_filter("SGLANG_JAX_DEBUG_DUMP_LAYERS")
    processes = _debug_dump_filter("SGLANG_JAX_DEBUG_DUMP_PROCESSES")
    if components and str(component) not in components:
        return value
    if layers and layer_id is not None and str(layer_id) not in layers:
        return value
    if processes and str(process) not in processes:
        return value

    metadata = {
        "process": process,
        "component": str(component),
        "layer": layer_id,
        "forward_mode": _normalize_forward_mode(forward_mode),
        "name": str(name),
    }
    dump_dir = os.environ.get("SGLANG_JAX_DEBUG_DUMP_DIR", "debug_dumps")
    jax.debug.callback(
        functools.partial(_write_debug_dump, dump_dir=dump_dir, metadata=metadata),
        value,
    )
    return value


def print_parameter_shardings(model):
    if FRAMEWORK_LOG_LEVEL < FrameworkLogLevel.DEBUG:
        return
    for name, param in model.named_parameters():
        print(f"{name}: shape={param.value.shape} sharding={param.value.sharding}")


def log_shardings(name):
    def decorator(fn):
        if FRAMEWORK_LOG_LEVEL < FrameworkLogLevel.DEBUG:
            return fn

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            for i, a in enumerate(args):
                if hasattr(a, "aval") and hasattr(a.aval, "sharding"):
                    print(f"{name} input[{i}]: {a.aval.shape} {a.aval.sharding}")
            result = fn(*args, **kwargs)
            if hasattr(result, "aval") and hasattr(result.aval, "sharding"):
                print(f"{name} output: {result.aval.shape} {result.aval.sharding}")
            elif isinstance(result, tuple):
                for i, r in enumerate(result):
                    if hasattr(r, "aval") and hasattr(r.aval, "sharding"):
                        print(f"{name} output[{i}]: {r.aval.shape} {r.aval.sharding}")
            return result

        return wrapper

    return decorator
