import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "fix_pipeline_schema_2_10_compat.py"
spec = importlib.util.spec_from_file_location("fix_pipeline_schema_2_10_compat", SCRIPT)
mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(mod)


class LoaderPatchTests(unittest.TestCase):
    def test_patches_module_level_set(self):
        source = 'SUPPORTED_SCHEMA_VERSIONS = {"2.6", "2.7", "2.8"}\nX = 1\n'
        patched, changed = mod.patch_loader_source(source)
        self.assertTrue(changed)
        self.assertIn("'2.10'", patched)
        self.assertFalse(mod.patch_loader_source(patched)[1])

    def test_patches_local_tuple_inside_load_schema(self):
        source = '''def load_schema(path):
    supported = ("2.6", "2.7", "2.8")
    if version not in supported:
        raise ValueError(f"supports {list(supported)}")
'''
        patched, changed = mod.patch_loader_source(source)
        self.assertTrue(changed)
        self.assertIn("'2.10'", patched)
        self.assertFalse(mod.patch_loader_source(patched)[1])

    def test_patches_inline_set_condition(self):
        source = '''def load_schema(path):
    if version not in {"2.6", "2.7", "2.8"}:
        raise ValueError("unsupported")
'''
        patched, changed = mod.patch_loader_source(source)
        self.assertTrue(changed)
        self.assertIn("'2.10'", patched)

    def test_patches_local_list_and_error_allowlist(self):
        source = '''def load_schema(path):
    supported = ["2.6", "2.7", "2.8"]
    if version not in supported:
        raise ValueError(f"supports {supported}")
'''
        patched, changed = mod.patch_loader_source(source)
        self.assertTrue(changed)
        self.assertIn("'2.10'", patched)

    def test_refuses_unknown_loader_versions(self):
        source = 'SUPPORTED_SCHEMA_VERSIONS = {"1.0", "1.1"}\n'
        with self.assertRaises(mod.CompatibilityPatchError):
            mod.patch_loader_source(source)


class SchemaPatchTests(unittest.TestCase):
    def make_schema(self, version="2.10"):
        return {"schema_version": version, "metadata_policy": {"supported_schema_versions": ["2.6", "2.7", "2.8"]}}

    def test_updates_supported_versions(self):
        text = json.dumps(self.make_schema())
        patched = mod._patch_schema_text(text)
        schema = json.loads(patched)
        self.assertEqual(schema["metadata_policy"]["supported_schema_versions"], ["2.6", "2.7", "2.8", "2.10"])

    def test_refuses_wrong_root_schema_version(self):
        with self.assertRaises(mod.CompatibilityPatchError):
            mod._patch_schema_text(json.dumps(self.make_schema("2.8")))


if __name__ == "__main__":
    unittest.main(verbosity=2)
