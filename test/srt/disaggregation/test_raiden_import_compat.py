"""Import compatibility checks without initializing JAX or TPU."""

import sys
import types
import unittest
from unittest.mock import patch

from sgl_jax import raiden


class RaidenImportCompatibilityTest(unittest.TestCase):
    def test_renamed_package_preloads_and_is_recognized(self):
        extension = "tpu_sync.frameworks.jax._tpu_raiden_jax"

        def load(name):
            if name.startswith("tpu_raiden."):
                raise ModuleNotFoundError("legacy package absent", name="tpu_raiden")
            module = types.ModuleType(name)
            sys.modules[name] = module
            return module

        with patch.dict(sys.modules), patch.object(
            raiden.importlib, "import_module", side_effect=load
        ):
            for name in (*raiden._RAIDEN_EXTENSIONS, "jax", "jaxlib"):
                sys.modules.pop(name, None)
            raiden.preload_raiden()
            raiden.require_raiden_preloaded()
            self.assertIn(extension, sys.modules)

    def test_missing_transitive_dependency_is_not_masked(self):
        error = ModuleNotFoundError("dependency absent", name="unrelated_dependency")
        with patch.object(raiden.importlib, "import_module", side_effect=error) as importer:
            with self.assertRaises(ModuleNotFoundError) as caught:
                raiden.import_raiden_module("api.jax.kv_cache_manager")
            self.assertIs(caught.exception, error)
            self.assertEqual(importer.call_count, 1)

    def test_existing_legacy_extension_needs_no_reimport(self):
        extension = raiden._RAIDEN_EXTENSION
        with patch.dict(sys.modules, {extension: types.ModuleType(extension)}):
            with patch.object(raiden.importlib, "import_module") as importer:
                raiden.preload_raiden()
                raiden.require_raiden_preloaded()
                importer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
