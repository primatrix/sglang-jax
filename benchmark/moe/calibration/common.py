from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

_SENSITIVE_KEY_PARTS = ("TOKEN", "SECRET", "PASSWORD", "KEY", "CREDENTIAL")


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


def write_jsonl(path: str | os.PathLike[str], rows: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
