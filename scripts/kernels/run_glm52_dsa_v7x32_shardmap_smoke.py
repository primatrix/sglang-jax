#!/usr/bin/env python3
"""Run reference and Pallas DSA paths in the explicit 32-device serving mesh."""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.kernels.dsa.reference import dsa_sparse_mla_reference
from sgl_jax.srt.kernels.mla.dsa.kernel import dsa_decode_mla_attention_unchecked


def main() -> None:
    process_count = int(os.environ["FALCON_JAX_PROCESS_COUNT"])
    process_id = int(os.environ["FALCON_JAX_PROCESS_ID"])
    coordinator_address = os.environ["FALCON_JAX_COORDINATOR_ADDRESS"]
    jax.distributed.initialize(
        coordinator_address=coordinator_address,
        num_processes=process_count,
        process_id=process_id,
    )

    devices = np.asarray(jax.devices())
    if devices.size != 32:
        raise AssertionError(f"expected 32 global devices, got {devices.size}")
    mesh = Mesh(
        devices.reshape(1, 32),
        ("data", "tensor"),
        axis_types=(AxisType.Explicit, AxisType.Explicit),
    )

    q_sharding = NamedSharding(mesh, P("data", "tensor", None))
    cache_sharding = NamedSharding(mesh, P("data", None, None, None))
    slots_sharding = NamedSharding(mesh, P("data", None))
    counts_sharding = NamedSharding(mesh, P("data"))

    q_host = ((np.arange(64 * 512, dtype=np.float32) % 13) - 6).reshape(1, 64, 512) / 13
    q_rope_host = ((np.arange(64 * 64, dtype=np.float32) % 11) - 5).reshape(1, 64, 64) / 11
    slot_values = (np.arange(2048, dtype=np.float32) % 23)[:, None] / 23
    width_values = ((np.arange(640, dtype=np.float32) % 7) - 3)[None, :] / 1000
    cache_host = (slot_values + width_values).reshape(16, 64, 2, 640)
    # The uncounted tail points to a dominant sentinel. A second-chunk guard
    # bug will make Pallas diverge from the reference result.
    cache_host.reshape(2048, 640)[-1] = 100
    slots_host = np.full((1, 2048), 2047, dtype=np.int32)
    slots_host[0, :129] = (np.arange(129, dtype=np.int32) * 17) % 2047

    q = jax.device_put(jnp.asarray(q_host, dtype=jnp.bfloat16), q_sharding)
    q_rope = jax.device_put(jnp.asarray(q_rope_host, dtype=jnp.bfloat16), q_sharding)
    cache = jax.device_put(jnp.asarray(cache_host, dtype=jnp.bfloat16), cache_sharding)
    slots = jax.device_put(jnp.asarray(slots_host), slots_sharding)
    counts = jax.device_put(jnp.array([129], dtype=jnp.int32), counts_sharding)

    def reference_kernel(q_, q_rope_, cache_, slots_, counts_):
        return dsa_sparse_mla_reference(
            q_,
            q_rope_,
            cache_,
            slots_,
            counts_,
            sm_scale=0.0625,
            page_size=128,
            latent_dim=512,
            rope_dim=64,
        )

    def local_kernel(q_, q_rope_, cache_, slots_, counts_):
        return dsa_decode_mla_attention_unchecked(
            q_,
            q_rope_,
            cache_,
            slots_,
            counts_,
            sm_scale=0.0625,
            interpret=False,
        )

    mapped_kernel = jax.shard_map(
        local_kernel,
        mesh=mesh,
        in_specs=(
            P("data", "tensor", None),
            P("data", "tensor", None),
            P("data", None, None, None),
            P("data", None),
            P("data"),
        ),
        out_specs=P("data", "tensor", None),
        check_vma=False,
    )
    with jax.set_mesh(mesh):
        reference_output = jax.jit(reference_kernel)(q, q_rope, cache, slots, counts)
        reference_output.block_until_ready()
        output = jax.jit(mapped_kernel)(q, q_rope, cache, slots, counts)
        output.block_until_ready()

    for name, actual in (("reference", reference_output), ("pallas", output)):
        if actual.shape != (1, 64, 512):
            raise AssertionError(f"unexpected {name} output shape: {actual.shape}")
    reference_shards = reference_output.addressable_shards
    pallas_shards = output.addressable_shards
    if len(reference_shards) != len(pallas_shards):
        raise AssertionError("reference and Pallas addressable shard counts differ")
    for reference_shard, pallas_shard in zip(reference_shards, pallas_shards, strict=True):
        expected = np.asarray(reference_shard.data)
        actual = np.asarray(pallas_shard.data)
        if not np.any(expected):
            raise AssertionError("reference smoke output unexpectedly contains only zeros")
        np.testing.assert_allclose(actual, expected, rtol=2e-2, atol=1e-2)
    print(
        "GLM52_DSA_V7X32_REFERENCE_AND_SHARDMAP_OK "
        f"process_id={process_id} local_devices={jax.local_device_count()}"
    )


if __name__ == "__main__":
    main()
