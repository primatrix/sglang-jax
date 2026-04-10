# cd python && USE_DEVICE_TYPE=cpu python -m pytest sgl_jax/test/managers/test_resolve_future_token_ids.py -q

import os
import unittest

if os.environ.get("USE_DEVICE_TYPE") == "cpu":
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
    os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import numpy as np
from jax.experimental import multihost_utils
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.managers.utils import resolve_future_token_ids
from sgl_jax.srt.utils.mesh_utils import create_device_mesh


def _make_dp_mesh():
    return create_device_mesh(ici_parallelism=[-1, 1], dcn_parallelism=[1, 1])


class TestResolveFutureTokenIds(unittest.TestCase):
    def test_resolve_future_keeps_input_ids_data_sharded(self):
        mesh = _make_dp_mesh()
        input_ids_np = np.arange(4096, dtype=np.int32)
        input_ids_np[::1024] = -1
        input_ids_np[1::1024] = -2
        future_token_ids_map_np = np.zeros((16,), dtype=np.int32)
        future_token_ids_map_np[1] = 101
        future_token_ids_map_np[2] = 202

        with mesh:
            input_ids = jax.device_put(input_ids_np, NamedSharding(mesh, P("data")))
            future_token_ids_map = jax.device_put(
                future_token_ids_map_np, NamedSharding(mesh, P(None))
            )

            resolved = resolve_future_token_ids(input_ids, future_token_ids_map, mesh)
            resolved_np = multihost_utils.process_allgather(resolved, tiled=True)

            expected = input_ids_np.copy()
            expected[::1024] = 101
            expected[1::1024] = 202
            np.testing.assert_array_equal(resolved_np, expected)
            self.assertEqual(resolved.sharding.spec, P("data"))

            hlo = (
                resolve_future_token_ids.lower(input_ids, future_token_ids_map, mesh)
                .compiler_ir(dialect="hlo")
                .as_hlo_text()
            )

        # DP input_ids should remain local. The future map is replicated, but
        # resolve itself should not replicate the full input_ids vector.
        self.assertLessEqual(hlo.count("sharding={replicated}"), 2)


if __name__ == "__main__":
    unittest.main()
