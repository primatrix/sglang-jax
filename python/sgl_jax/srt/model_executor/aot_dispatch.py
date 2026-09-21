"""AOT execute_sharded dispatch for large-arg-count jitted functions.

Models whose weights flatten to thousands of jit arguments (GLM-5.x /
DeepSeek-V3-class MoE: ~2487 weight leaves + ~99 KV-pool leaves + metadata
~= 2600 args) pay tens of milliseconds of per-step Python dispatch in pjit's
call path: ``_infer_params``, per-argument ``sharding.is_equivalent_to``
checks, and the O(n_args) ``shard_args`` loop. On single-stream decode this
host cost dominates TPOT (measured ~34ms/step on both v6e-64 tp64 and v7x
tp16 for GLM-5.2 753B — device-independent).

``AotDispatcher`` removes the steady-state cost:

1. The first call for each dynamic-shape key lowers and compiles an AOT
   executable, then runs once through the checked ``Compiled.__call__``
   path — this validates input shardings/layouts against the executable
   exactly like normal pjit dispatch.
2. Steady state dispatches via ``xla_executable.execute_sharded`` directly:
   the stable (weight) leaves' sharded buffers are captured once per key,
   and only the small dynamic tail (forward batch, donated pools, metadata)
   goes through ``shard_args``.
3. Executables with ordered/unordered effects, host callbacks, or mutation
   permanently fall back to the checked path for that key.

Donation is unaffected: XLA input-output aliasing is baked into the
executable, and ``ExecuteReplicated`` adds no Python-side donation logic.

Enabling: ``SGLANG_JAX_AOT_DISPATCH`` = ``auto`` (on when the function sees
>= ``_AUTO_MIN_ARGS`` flat args), ``1`` (always), ``0`` (default, off).
"""

from __future__ import annotations

import logging
import os
import time

import jax
import numpy as np
from jax._src.lib import xla_client as _xc

logger = logging.getLogger(__name__)

_ENV = os.environ.get("SGLANG_JAX_AOT_DISPATCH", "auto")
_AUTO_MIN_ARGS = 512

_FALLBACK = object()
_TRACE = bool(os.environ.get("SGLANG_JAX_ITER_TRACE"))


class _DispatchStats:
    """Rolling split of the steady-state AOT dispatch (SGLANG_JAX_ITER_TRACE)."""

    def __init__(self, name: str, every: int = 200):
        self.name, self.every, self.n, self.sums = name, every, 0, {}

    def add(self, **ms):
        for k, v in ms.items():
            self.sums[k] = self.sums.get(k, 0.0) + v
        self.n += 1
        if self.n >= self.every:
            parts = " ".join(f"{k}={v / self.n:.2f}ms" for k, v in self.sums.items())
            logger.info("[iter-trace:aot-dispatch:%s] n=%d %s", self.name, self.n, parts)
            self.n, self.sums = 0, {}


def _leaf_signature(a):
    """(shape, dtype, weak_type) of a dynamic leaf without building an aval.

    ``jax.typeof`` constructs a ShapedArray per leaf (~185 leaves per V4 decode
    step); arrays answer the same question from their attributes directly.
    """
    if isinstance(a, jax.Array):
        return (a.shape, a.dtype, bool(getattr(a, "weak_type", False)))
    if isinstance(a, np.ndarray):
        return (a.shape, a.dtype, False)
    aval = jax.typeof(a)
    return (aval.shape, aval.dtype, aval.weak_type)


def aot_dispatch_requested() -> bool:
    """True when SGLANG_JAX_AOT_DISPATCH is set to "auto" or "1".

    Default is "auto" (on for functions with >= _AUTO_MIN_ARGS flat args); set
    "0" to keep the stock pjit dispatch path.
    """
    return _ENV in ("auto", "1")


def aot_dispatch_enabled(num_flat_args: int) -> bool:
    if _ENV == "auto":
        return num_flat_args >= _AUTO_MIN_ARGS
    return _ENV == "1"


class AotDispatcher:
    """Dispatch ``jit_fn(*stable_call_args, *dyn_args)`` via cached AOT executables.

    Args:
      jit_fn: the ``jax.jit``-wrapped function.
      stable_call_args: positional prefix passed to ``lower``/``Compiled.__call__``
        including any static (hashable, non-array) arguments.
      stable_flat_args: the same prefix with static arguments removed — i.e.
        exactly the prefix pytrees that appear in the executable's flat input
        list. Their leaves must be the same arrays every call (weights).
      name: label for logs.

    If the caller ever replaces the stable containers (LoRA / EPLB reload),
    it must call :meth:`invalidate` (or construct a new dispatcher); the
    cheap ``id()`` guard below also catches replacement of the top-level
    containers between calls.
    """

    def __init__(self, jit_fn, stable_call_args: tuple, stable_flat_args: tuple, name: str):
        self._jit_fn = jit_fn
        self._stable_call_args = stable_call_args
        self._stable_flat_args = stable_flat_args
        self._stable_ids = tuple(id(a) for a in stable_flat_args)
        self._cache = {}
        self._name = name
        self._enabled = None  # decided on first call from flat arg count
        self._stats = _DispatchStats(name) if _TRACE else None

    def invalidate(self) -> None:
        self._cache.clear()

    def ensure_stable_args(self, stable_call_args: tuple, stable_flat_args: tuple) -> None:
        """Rebind the stable containers if the caller replaced them.

        Callers that rebind a stable container to a new object (e.g. LoRA
        weight loading re-flattens ``model_state_leaves`` into a new list in
        ``tp_worker.prepare_lora_batch``) must route every call through this
        check: the dispatcher would otherwise keep executing with the buffers
        captured at construction time. No-op (two identity checks) when the
        containers are unchanged.
        """
        if len(stable_flat_args) == len(self._stable_flat_args) and all(
            a is b for a, b in zip(stable_flat_args, self._stable_flat_args)
        ):
            return
        logger.info("[aot-dispatch:%s] stable args rebound; dropping AOT cache", self._name)
        self._stable_call_args = stable_call_args
        self._stable_flat_args = stable_flat_args
        self._stable_ids = tuple(id(a) for a in stable_flat_args)
        self._cache.clear()

    def __call__(self, *dyn_args):
        if self._enabled is False:
            return self._jit_fn(*self._stable_call_args, *dyn_args)
        if tuple(id(a) for a in self._stable_flat_args) != self._stable_ids:
            logger.info("[aot-dispatch:%s] stable args replaced; recompiling", self._name)
            self._stable_ids = tuple(id(a) for a in self._stable_flat_args)
            self._cache.clear()

        _t0 = time.perf_counter() if self._stats is not None else 0.0
        dyn_leaves, dyn_tree = jax.tree_util.tree_flatten(dyn_args)
        # Batch mode and other pytree metadata can change the program even
        # when all array shapes match. Scalar types also affect compilation.
        key = (dyn_tree, tuple(_leaf_signature(a) for a in dyn_leaves))
        entry = self._cache.get(key)
        _t1 = time.perf_counter() if self._stats is not None else 0.0
        if entry is None:
            return self._compile_and_first_call(key, dyn_args)
        if entry is _FALLBACK:
            return self._jit_fn(*self._stable_call_args, *dyn_args)

        (
            xla_exec,
            out_handlers,
            out_tree,
            static_bufs,
            dyn_kept,
            dyn_shardings,
            dyn_layouts,
            dyn_copy,
        ) = entry
        from jax._src.interpreters import pxla

        kept = [dyn_leaves[i] for i in dyn_kept]
        # Host arrays reach the device one transfer per leaf inside shard_args
        # (~40 us each; V4 has ~170 dynamic leaves per step). Ship every numpy
        # leaf in one batched device_put first, so shard_args only sees device
        # arrays.
        np_pos = [j for j, a in enumerate(kept) if isinstance(a, np.ndarray)]
        if np_pos:
            put = jax.device_put([kept[j] for j in np_pos], [dyn_shardings[j] for j in np_pos])
            for j, arr in zip(np_pos, put):
                kept[j] = arr
        _t2 = time.perf_counter() if self._stats is not None else 0.0
        dyn_bufs = pxla.shard_args(dyn_shardings, dyn_layouts, dyn_copy, kept)
        _t3 = time.perf_counter() if self._stats is not None else 0.0
        results = xla_exec.execute_sharded(static_bufs + list(dyn_bufs))
        out_flat = results.consume_with_handlers(out_handlers)
        out = jax.tree_util.tree_unflatten(out_tree, out_flat)
        if self._stats is not None:
            _t4 = time.perf_counter()
            self._stats.add(
                flatten_key=(_t1 - _t0) * 1e3,
                host_put=(_t2 - _t1) * 1e3,
                shard_args=(_t3 - _t2) * 1e3,
                execute=(_t4 - _t3) * 1e3,
                n_np=float(len(np_pos)),
            )
        return out

    def _compile_and_first_call(self, key, dyn_args):
        from jax._src.interpreters import pxla

        if self._enabled is None:
            n_flat = len(jax.tree_util.tree_leaves((self._stable_flat_args + dyn_args, {})))
            self._enabled = aot_dispatch_enabled(n_flat)
            if not self._enabled:
                logger.info(
                    "[aot-dispatch:%s] disabled (%d flat args, env=%s)",
                    self._name,
                    n_flat,
                    _ENV,
                )
                return self._jit_fn(*self._stable_call_args, *dyn_args)

        compiled = self._jit_fn.lower(*self._stable_call_args, *dyn_args).compile()
        unsafe = compiled._executable.unsafe_call
        if (
            unsafe.ordered_effects
            or unsafe.has_unordered_effects
            or unsafe.has_host_callbacks
            or unsafe.mut is not None
        ):
            logger.warning(
                "[aot-dispatch:%s] effects/mutation present; falling back to "
                "checked dispatch for this shape",
                self._name,
            )
            self._cache[key] = _FALLBACK
            return self._jit_fn(*self._stable_call_args, *dyn_args)

        # Flat layout: [stable leaves][dyn leaves]; kept_var_idx is the
        # DCE-surviving subset, in order, matching in_handler's shardings.
        n_stable = len(jax.tree_util.tree_leaves(self._stable_flat_args))
        kept = sorted(unsafe.kept_var_idx)
        n_static = sum(1 for i in kept if i < n_stable)
        dyn_kept = [i for i in kept if i >= n_stable]
        shardings = unsafe.in_handler.in_shardings
        layouts = unsafe.in_handler.in_layouts

        args_flat, _ = jax.tree_util.tree_flatten((self._stable_flat_args + dyn_args, {}))
        reuse = [_xc.ArrayCopySemantics.REUSE_INPUT] * n_static
        static_bufs = list(
            pxla.shard_args(
                shardings[:n_static],
                layouts[:n_static],
                reuse,
                [args_flat[i] for i in kept[:n_static]],
            )
        )
        self._cache[key] = (
            unsafe.xla_executable,
            unsafe.out_handler.handlers,
            compiled._params.out_tree,
            static_bufs,
            [i - n_stable for i in dyn_kept],
            shardings[n_static:],
            layouts[n_static:],
            [_xc.ArrayCopySemantics.REUSE_INPUT] * len(dyn_kept),
        )
        first_leaves = jax.tree_util.tree_leaves(dyn_args)
        n_np = sum(1 for i in dyn_kept if isinstance(first_leaves[i - n_stable], np.ndarray))
        logger.info(
            "[aot-dispatch:%s] compiled shape key (%d stable + %d dyn kept args, %d numpy)",
            self._name,
            n_static,
            len(dyn_kept),
            n_np,
        )
        # First call goes through the checked path: validates that every
        # input's sharding/layout matches what the executable expects.
        return compiled(*self._stable_flat_args, *dyn_args)
