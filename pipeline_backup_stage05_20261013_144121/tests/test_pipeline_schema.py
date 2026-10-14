from __future__ import annotations

import ast
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMAGING = ROOT / "backend" / "Imaging"
sys.path.insert(0, str(IMAGING))
from pipeline_schema import (  # noqa: E402
    PipelineSchemaError,
    check_meta_schema_version,
    get_required,
    load_schema,
    repository_root,
    validate_stage_version,
)


class PipelineSchemaTests(unittest.TestCase):
    def setUp(self):
        self.schema = load_schema(schema_path=ROOT / "meta_schema.json")

    def test_current_schema_loads_and_declares_supported_version(self):
        self.assertIn(self.schema["schema_version"], self.schema["metadata_policy"]["supported_schema_versions"])

    def test_log_patches_and_stage03_outputs_are_separate(self):
        stage2 = self.schema["stage_02_detect_and_patch"]
        stage3 = self.schema["stage_03_nodule_classification"]
        self.assertEqual(
            stage2["diagnostic_patch_outputs"]["manifest"],
            stage3["inputs"]["log_diagnostic_patch_manifest"],
        )
        self.assertNotEqual(
            stage2["diagnostic_patch_outputs"]["manifest"],
            stage2["patch_outputs"]["manifest"],
        )
        self.assertNotEqual(
            stage3["outputs"]["log_diagnostic_predictions_json"],
            stage3["outputs"]["predictions_json"],
        )

    def test_stage02_resolves_log_manifest_before_use_and_saves_after_extraction(self):
        stage2 = IMAGING / "02_detect_and_patch.py"
        source = stage2.read_text(encoding="utf-8")
        log_path_assignment = source.index(
            'log_patch_manifest_path = patient_root / STAGE02_LOG_PATCH_OUTPUTS["manifest"]'
        )
        authoritative_manifest = source.index("    manifest = {")
        log_records_extraction = source.index("    log_records = extract_all(")
        actual_count_update = source.index(
            'manifest["diagnostic_log_detector"]["num_log_patches_extracted"] = len(log_records)'
        )
        log_manifest_save = source.index("save_json(log_patch_manifest_path, log_manifest)")
        authoritative_manifest_save = source.index("save_json(patch_manifest_path, manifest)")

        self.assertLess(log_path_assignment, authoritative_manifest)
        self.assertLess(authoritative_manifest, log_records_extraction)
        self.assertLess(log_records_extraction, actual_count_update)
        self.assertLess(actual_count_update, authoritative_manifest_save)
        self.assertLess(log_manifest_save, authoritative_manifest_save)

    def test_stage04_loads_stage03_path_configuration(self):
        stage4 = IMAGING / "04_malignancy_prediction.py"
        tree = ast.parse(stage4.read_text(encoding="utf-8"), filename=str(stage4))
        assigned = {
            target.id
            for node in ast.walk(tree)
            if isinstance(node, (ast.Assign, ast.AnnAssign))
            for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
            if isinstance(target, ast.Name)
        }
        self.assertIn("STAGE03_PATHS", assigned)

    def test_log_candidate_adapter_preserves_source_identity_and_bbox(self):
        import numpy as np
        from types import SimpleNamespace

        stage2_path = IMAGING / "02_detect_and_patch.py"
        tree = ast.parse(stage2_path.read_text(encoding="utf-8"), filename=str(stage2_path))
        function = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "build_log_patch_candidates"
        )
        module = ast.Module(
            body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), function],
            type_ignores=[],
        )
        namespace = {"np": np, "STAGE02_FILES": {"log_diagnostic": "log_candidates_diagnostic.json"}}
        exec(compile(ast.fix_missing_locations(module), str(stage2_path), "exec"), namespace)
        original = SimpleNamespace(
            diagnostic_id="log_00042", center_zyx=[10.0, 20.0, 30.0],
            center_lps_mm=[1.0, 2.0, 3.0], scale_mm=8.0, log_response=0.12,
            purpose="research_comparison_only", coordinate_order="ZYX",
        )
        converted = namespace["build_log_patch_candidates"]([original], np.asarray([2.0, 1.0, 1.0]))[0]
        self.assertEqual(converted["detection_source"], "LoG")
        self.assertEqual(converted["diagnostic_id"], "log_00042")
        self.assertEqual(converted["source_candidates_file"], "log_candidates_diagnostic.json")
        self.assertEqual(converted["bbox_start_zyx"], [8, 16, 26])
        self.assertEqual(converted["bbox_end_zyx"], [12, 24, 34])

    def test_repository_root_beats_nested_stale_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "LungInsight_v2"
            imaging = root / "backend" / "Imaging"
            imaging.mkdir(parents=True)
            (root / "meta_schema.json").write_text("{}", encoding="utf-8")
            (imaging / "meta_schema.json").write_text("{\"schema_version\": \"old\"}", encoding="utf-8")
            script = imaging / "01_hu_and_crop.py"
            script.write_text("# test fixture\n", encoding="utf-8")
            self.assertEqual(repository_root(script), root)

    def test_stage01_legacy_archive_root_is_derived_from_schema_root(self):
        stage01 = ROOT / "backend" / "Imaging" / "01_hu_and_crop.py"
        tree = ast.parse(stage01.read_text(encoding="utf-8"), filename=str(stage01))
        assignments = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        assignments[target.id] = node.value
        self.assertIn("DEFAULT_LIDC_DATA_ROOT", assignments)
        self.assertIn("DEFAULT_LIDC_ROOT", assignments)
        legacy_root = assignments["DEFAULT_LIDC_ROOT"]
        self.assertIsInstance(legacy_root, ast.Attribute)
        self.assertEqual(legacy_root.attr, "parent")
        self.assertIsInstance(legacy_root.value, ast.Name)
        self.assertEqual(legacy_root.value.id, "DEFAULT_LIDC_DATA_ROOT")

    def test_missing_required_key_has_actionable_error(self):
        with self.assertRaisesRegex(PipelineSchemaError, "missing required key"):
            get_required(self.schema, "not_a_real_schema_key")

    def test_wrong_required_type_is_rejected(self):
        with self.assertRaisesRegex(PipelineSchemaError, "expected"):
            get_required(self.schema, "schema_version", expected_type=dict)

    def test_missing_schema_path_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "missing_meta_schema.json"
            with self.assertRaisesRegex(FileNotFoundError, "meta_schema.json.*not found"):
                load_schema(schema_path=missing)

    def test_schema_version_mismatch_is_rejected(self):
        payload = {"meta_schema_version": "0.0"}
        with self.assertRaisesRegex(PipelineSchemaError, "version mismatch"):
            check_meta_schema_version(payload, self.schema, context="test artifact", require=True)

    def test_stage_version_mismatch_is_rejected(self):
        with self.assertRaisesRegex(PipelineSchemaError, "Incompatible"):
            validate_stage_version("old", "new", context="test manifest")


if __name__ == "__main__":
    unittest.main()
