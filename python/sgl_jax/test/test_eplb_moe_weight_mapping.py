import unittest


class TestEplbMoeWeightMapping(unittest.TestCase):
    def test_create_moe_weights_mapping_supports_physical_to_logical_map_fused(self):
        try:
            from sgl_jax.srt.layers.moe import create_moe_weights_mapping
        except ModuleNotFoundError as e:
            # Some CI environments run EPLB unit tests without optional model deps (e.g. flax).
            self.skipTest(f"Optional dependency missing: {e}")

        m = create_moe_weights_mapping(
            prefix="model.layers.0",
            target_prefix="model.layers.0",
            num_experts=6,
            physical_to_logical_map=[0, 1, 2, 0, 2, 1],
            moe_backend="fused",
            moe_path="mlp",
            source_expert_pattern="experts.{i}",
        )

        # Fused uses w1/w3/w2 names internally.
        w1 = m["__MOE_EXPERTS__model.layers.0.mlp.w1"]
        self.assertEqual(w1.target_path[0], "model.layers.0.mlp.w1")
        self.assertEqual(
            w1.target_path[1:],
            [
                "model.layers.0.mlp.experts.0.gate_proj.weight",
                "model.layers.0.mlp.experts.1.gate_proj.weight",
                "model.layers.0.mlp.experts.2.gate_proj.weight",
                "model.layers.0.mlp.experts.0.gate_proj.weight",
                "model.layers.0.mlp.experts.2.gate_proj.weight",
                "model.layers.0.mlp.experts.1.gate_proj.weight",
            ],
        )


if __name__ == "__main__":
    unittest.main()
