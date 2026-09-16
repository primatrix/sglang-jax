"""Early loader for the optional tpu-raiden runtime."""

from __future__ import annotations

import importlib
import sys

_RAIDEN_EXTENSION = "tpu_sync.frameworks.jax._tpu_raiden_jax"


def raiden_requested(argv: list[str] | None = None) -> bool:
    args = list(sys.argv[1:] if argv is None else argv)
    pd_requested = False
    encoder_requested = False
    for arg in args:
        if arg == "--disaggregation-use-raiden":
            pd_requested = True
        elif arg == "--no-disaggregation-use-raiden":
            pd_requested = False
        elif arg in ("--encoder-only", "--language-only"):
            encoder_requested = True
    return pd_requested or encoder_requested


def preload_raiden() -> None:
    """Preload the tpu-raiden native extension module.

    Must be called before importing JAX or jaxlib to ensure the native C++
    runtime extensions link and initialize properly before libtpu loads.
    """
    if _RAIDEN_EXTENSION in sys.modules:
        return
    if "jax" in sys.modules or "jaxlib" in sys.modules:
        raise RuntimeError("tpu-raiden must be preloaded before jax/jaxlib")
    try:
        importlib.import_module(_RAIDEN_EXTENSION)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "tpu-raiden is not installed; install a wheel matching JAX and libtpu"
        ) from exc
    except Exception as exc:  # pragma: no cover - native loader failure
        raise RuntimeError(
            "tpu-raiden failed to load; verify that its wheel matches JAX and libtpu"
        ) from exc


def preload_raiden_if_requested(argv: list[str] | None = None) -> None:
    if raiden_requested(argv):
        preload_raiden()


def require_raiden_preloaded() -> None:
    if _RAIDEN_EXTENSION not in sys.modules:
        raise RuntimeError(
            "tpu-raiden was not preloaded. Use sgl_jax.launch_server or call "
            "sgl_jax.raiden.preload_raiden() before importing JAX."
        )
