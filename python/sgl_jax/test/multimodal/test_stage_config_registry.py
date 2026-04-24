import unittest

from sgl_jax.srt.multimodal.models.static_configs.yaml_registry import (
    StageConfigRegistry,
)


class TestStageConfigRegistry(unittest.TestCase):
    def test_qwen2_5_vl_3b_uses_tp4_config_for_single_host_ci(self):
        for model_path in (
            "Qwen/Qwen2.5-VL-3B-Instruct",
            "Qwen2.5-VL-3B-Instruct",
            "/models/Qwen2.5-VL-3B-Instruct",
        ):
            with self.subTest(model_path=model_path):
                self.assertEqual(
                    StageConfigRegistry.get_yaml_path(model_path).name,
                    "qwen2_5_vl_stage_config_tp4.yaml",
                )


if __name__ == "__main__":
    unittest.main()
