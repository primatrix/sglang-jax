import functools
import hashlib
import json
import os
import re
import tempfile
import threading
from collections import defaultdict
from enum import IntEnum
from pathlib import Path

import jax
import jax.numpy as jnp
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
_DEBUG_DUMP_INITIALIZED = set()
_UNSAFE_FILENAME_CHARS = re.compile(r"[^A-Za-z0-9_.-]+")


def _debug_dump_filter(name):
    value = os.environ.get(name, "")
    return {item.strip() for item in value.split(",") if item.strip()}


def _sanitize_filename_part(value):
    raw_value = str(value)
    sanitized = _UNSAFE_FILENAME_CHARS.sub("-", raw_value).strip(".-") or "unknown"
    digest = hashlib.blake2s(raw_value.encode("utf-8"), digest_size=4).hexdigest()
    return f"{sanitized}-{digest}"


def _normalize_forward_mode(forward_mode):
    if forward_mode is None:
        return "unknown"
    if hasattr(forward_mode, "name"):
        return str(forward_mode.name).lower()
    return str(forward_mode)


def _forward_fingerprint(forward_batch):
    if forward_batch is None:
        return None
    values = [
        getattr(forward_batch, "input_ids", None),
        getattr(forward_batch, "positions", None),
        getattr(forward_batch, "seq_lens", None),
    ]
    if all(value is None for value in values):
        return None

    fingerprint = (
        jnp.asarray(2166136261, dtype=jnp.uint32),
        jnp.asarray(2246822519, dtype=jnp.uint32),
    )

    def mix_word(state, word):
        low, high = state
        word = jnp.asarray(word, dtype=jnp.uint32)
        low = (low ^ word) * jnp.asarray(16777619, dtype=jnp.uint32)
        low = (low ^ (low >> jnp.asarray(16, dtype=jnp.uint32))) * jnp.asarray(
            2246822519, dtype=jnp.uint32
        )
        high = (high + word + jnp.asarray(2654435769, dtype=jnp.uint32)) * jnp.asarray(
            3266489917, dtype=jnp.uint32
        )
        high = high ^ (high >> jnp.asarray(13, dtype=jnp.uint32))
        return low, high

    for field_index, value in enumerate(values):
        if value is None:
            continue
        flattened = jnp.ravel(jnp.asarray(value, dtype=jnp.uint32))
        fingerprint = mix_word(
            fingerprint, jnp.asarray(0xD5A00000 + field_index, dtype=jnp.uint32)
        )
        fingerprint = mix_word(fingerprint, jnp.asarray(value.ndim, dtype=jnp.uint32))
        for dimension in value.shape:
            fingerprint = mix_word(
                fingerprint, jnp.asarray(dimension, dtype=jnp.uint32)
            )
        fingerprint = jax.lax.fori_loop(
            0,
            flattened.size,
            lambda index, state, flattened=flattened: mix_word(state, flattened[index]),
            fingerprint,
        )
    return jnp.stack(fingerprint)


def _write_debug_dump(host_value, host_occurrence=None, *, dump_dir, metadata, enabled=True):
    if not enabled:
        return
    host_array = np.asarray(host_value)

    with _DEBUG_DUMP_LOCK:
        output_dir = Path(dump_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        initialization_key = (str(output_dir.resolve()), metadata["process"])
        manifest_path = output_dir / f"manifest-p{metadata['process']:05d}.jsonl"
        if initialization_key not in _DEBUG_DUMP_INITIALIZED:
            existing_arrays = next(output_dir.glob(f"p{metadata['process']:05d}__*.npy"), None)
            if manifest_path.exists() or existing_arrays is not None:
                raise RuntimeError(
                    f"debug dump directory already contains debug dumps for process "
                    f"{metadata['process']}: {output_dir}"
                )
            _DEBUG_DUMP_INITIALIZED.add(initialization_key)

        semantic_key = (
            str(output_dir.resolve()),
            metadata["process"],
            metadata["component"],
            metadata["layer"],
            metadata["forward_mode"],
            metadata["name"],
        )
        if host_occurrence is None:
            occurrence = _DEBUG_DUMP_OCCURRENCES[semantic_key]
            _DEBUG_DUMP_OCCURRENCES[semantic_key] += 1
        else:
            occurrence_words = np.asarray(host_occurrence, dtype=np.uint32).reshape(-1)
            if occurrence_words.shape != (2,):
                raise RuntimeError(
                    "debug forward occurrence must contain exactly two uint32 words"
                )
            occurrence = int(occurrence_words[0]) | (int(occurrence_words[1]) << 32)
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
        output_path = output_dir / filename
        if output_path.exists():
            raise RuntimeError(
                "debug tensor semantic key already exists; use unique request inputs and a "
                f"fresh dump directory: {output_path}"
            )

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
        with manifest_path.open("a", encoding="utf-8") as manifest:
            manifest.write(json.dumps(manifest_row, sort_keys=True) + "\n")


def maybe_dump_jax_array(
    value,
    *,
    component,
    name,
    layer_id=None,
    forward_mode=None,
    forward_batch=None,
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
    write_on_process = not processes or str(process) in processes

    metadata = {
        "process": process,
        "component": str(component),
        "layer": layer_id,
        "forward_mode": _normalize_forward_mode(forward_mode),
        "name": str(name),
    }
    dump_dir = os.environ.get("SGLANG_JAX_DEBUG_DUMP_DIR", "debug_dumps")
    # Keep the callback in every rank's HLO even when only selected processes
    # write files. Omitting it on other ranks changes multi-controller TPU
    # launch ordering and can halt the device with a launch-id mismatch.
    callback = functools.partial(
        _write_debug_dump,
        dump_dir=dump_dir,
        metadata=metadata,
        enabled=write_on_process,
    )
    forward_fingerprint = _forward_fingerprint(forward_batch)
    if forward_fingerprint is None:
        jax.debug.callback(callback, value)
    else:
        jax.debug.callback(callback, value, forward_fingerprint)
    return value


def maybe_dump_jax_array_sum(left, right, **metadata):
    """Dump a derived sum without constructing it when tensor dumping is disabled."""
    if os.environ.get("SGLANG_JAX_DEBUG_DUMP", "0") != "1":
        return None
    value = left + right
    maybe_dump_jax_array(value, **metadata)
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
