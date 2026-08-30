import unittest
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import Mesh

from sgl_jax.srt.layers.moe import EPMoE
from sgl_jax.srt.utils.weight_utils import WeightLoader


class TestMLADCPMoEMesh(unittest.TestCase):
    def test_epmoe_derives_two_dimensional_abstract_mesh_from_dcp_mesh(self):
        if len(jax.devices()) < 4:
            self.skipTest("MLA DCP MoE mesh test requires at least four JAX devices")

        devices = np.asarray(jax.devices()[:4], dtype=object).reshape(1, 2, 2)
        mesh = Mesh(
            devices,
            ("data", "tensor", "dcp"),
            axis_types=(jax.sharding.AxisType.Explicit,) * 3,
        )
        layer = nnx.eval_shape(
            lambda: EPMoE(
                hidden_size=4,
                num_experts=4,
                num_experts_per_tok=1,
                ep_size=2,
                mesh=mesh,
                intermediate_dim=8,
                weight_dtype=jnp.bfloat16,
                dtype=jnp.bfloat16,
            )
        )

        self.assertEqual(layer.moe_mesh.axis_names, ("expert", "tensor"))
        self.assertEqual(tuple(layer.moe_mesh.devices.shape), (2, 2))
        self.assertEqual(layer.updated_mesh.axis_names, ("expert", "tensor"))
        self.assertEqual(tuple(layer.updated_mesh.axis_sizes), (2, 2))
        self.assertEqual(len(layer.updated_mesh.axis_types), 2)

        loader = WeightLoader(
            model=object(),
            model_config=SimpleNamespace(ep_size=2),
            mesh=mesh,
        )
        self.assertEqual(loader.moe_abstract_mesh.axis_names, ("expert", "tensor"))
        self.assertEqual(tuple(loader.moe_abstract_mesh.axis_sizes), (2, 2))
        self.assertEqual(len(loader.moe_abstract_mesh.axis_types), 2)


if __name__ == "__main__":
    unittest.main()
