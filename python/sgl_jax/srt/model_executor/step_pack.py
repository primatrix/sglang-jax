"""Pack the per-step batch arrays into one host vector (``SGLANG_JAX_PACK_STEP_ARGS``).

With ``SGLANG_JAX_LAZY_HOST_ARGS`` the ForwardBatch / SamplingMetadata fields reach
the AOT dispatcher as ~13 separate numpy leaves and its batched ``device_put`` still
pays ~70 us per leaf on an 8-device mesh.  On the fused decode path the overlap
client packs them into a single int32 vector (f32 via bit pattern, bool as 0/1)
carried by ``ForwardBatch.step_packed`` with a static layout; the fused jit
unpacks them at its entry.
"""

from __future__ import annotations

import dataclasses
import os

import jax
import jax.numpy as jnp
import numpy as np

FORWARD_BATCH_FIELDS = (
    "input_ids",
    "seq_lens",
    "out_cache_loc",
    "positions",
    "req_pool_indices",
    "cache_loc",
    "extend_prefix_lens",
    "extend_seq_lens",
)
# ``temperatures`` stays a separate leaf: SamplingMetadata.update_vocab_mask reads its
# shape on the host before the jit (grammar mask sizing).
SAMPLING_FIELDS = ("top_ps", "top_ks", "min_ps", "positions", "sampling_seeds")

_ENABLED = (
    os.environ.get("SGLANG_JAX_PACK_STEP_ARGS", "1") == "1"
)  # default on since pfbase14 (09-19)


def pack_step_enabled() -> bool:
    return _ENABLED


def pack_step_arrays(forward_batch, sampling_metadata):
    """Return ``(forward_batch, sampling_metadata)`` with their host arrays packed.

    Only numpy leaves are packed (device arrays are left alone); the packed
    vector and the static layout live on the returned ForwardBatch.
    """
    specs, chunks, offset = [], [], 0
    fb_updates, sm_updates = {}, {}
    for owner, obj, fields, updates in (
        ("fb", forward_batch, FORWARD_BATCH_FIELDS, fb_updates),
        ("sm", sampling_metadata, SAMPLING_FIELDS, sm_updates),
    ):
        if obj is None:
            continue
        for name in fields:
            value = getattr(obj, name, None)
            if not isinstance(value, np.ndarray) or value.ndim == 0:
                continue
            flat = np.ascontiguousarray(value).reshape(-1)
            if value.dtype == np.float32:
                flat = flat.view(np.int32)
            elif value.dtype == np.bool_:
                flat = flat.astype(np.int32)
            elif value.dtype != np.int32:
                if not np.issubdtype(value.dtype, np.integer):
                    raise TypeError(f"cannot pack step array {name} of dtype {value.dtype}")
                flat = flat.astype(np.int32)  # sglang batch indices fit int32
            specs.append((owner, name, offset, flat.size, tuple(value.shape), value.dtype.str))
            chunks.append(flat)
            offset += flat.size
            updates[name] = None
    if not chunks:
        return forward_batch, sampling_metadata
    packed = np.concatenate(chunks)
    forward_batch = dataclasses.replace(
        forward_batch, **fb_updates, step_packed=packed, step_layout=tuple(specs)
    )
    if sm_updates:
        sampling_metadata = dataclasses.replace(sampling_metadata, **sm_updates)
    return forward_batch, sampling_metadata


def unpack_step_arrays(forward_batch, sampling_metadata):
    """Inverse of :func:`pack_step_arrays`; works on host arrays and inside jit."""
    packed = getattr(forward_batch, "step_packed", None)
    if packed is None:
        return forward_batch, sampling_metadata
    fb_updates, sm_updates = {}, {}
    for owner, name, offset, size, shape, dtype_str in forward_batch.step_layout:
        leaf = jax.lax.slice_in_dim(packed, offset, offset + size).reshape(shape)
        dtype = np.dtype(dtype_str)
        if dtype == np.float32:
            leaf = jax.lax.bitcast_convert_type(leaf, jnp.float32)
        elif dtype == np.bool_:
            leaf = leaf != 0
        # Other integer kinds stay int32: the batch indices are int32 already and
        # x64 is disabled in the server, so a wider dtype could not round-trip.
        (fb_updates if owner == "fb" else sm_updates)[name] = leaf
    forward_batch = dataclasses.replace(
        forward_batch, **fb_updates, step_packed=None, step_layout=None
    )
    if sm_updates:
        sampling_metadata = dataclasses.replace(sampling_metadata, **sm_updates)
    return forward_batch, sampling_metadata
