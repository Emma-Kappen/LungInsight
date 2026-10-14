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

    def test_stage05_patch_shape_is_defined_from_schema_and_matches_stage03(self):
        stage5 = IMAGING / "05_gradcam.py"
        source = stage5.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(stage5))
        assigned_names = {
            target.id
            for node in ast.walk(tree)
            if isinstance(node, (ast.Assign, ast.AnnAssign))
            for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
            if isinstance(target, ast.Name)
        }
        self.assertIn("PATCH_SHAPE", assigned_names)
        self.assertIn("STAGE03_PATCH_SHAPE", assigned_names)
        self.assertIn('get_required(STAGE05_SCHEMA, "patch_shape_zyx", expected_type=list)', source)
        self.assertIn("if PATCH_SHAPE != STAGE03_PATCH_SHAPE:", source)
        self.assertEqual(
            tuple(self.schema["stage_05_gradcam"]["patch_shape_zyx"]),
            tuple(self.schema["stage_03_nodule_classification"]["inputs"]["patch_shape_zyx"]),
        )

    def test_stage05_allows_six_requested_attributes_as_subset_of_eight_model_targets(self):
        stage5 = IMAGING / "05_gradcam.py"
        tree = ast.parse(stage5.read_text(encoding="utf-8"), filename=str(stage5))
        function = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "validate_target_contract"
        )
        module = ast.Module(
            body=[function],
            type_ignores=[],
        )
        namespace = {}
        exec(compile(ast.fix_missing_locations(module), str(stage5), "exec"), namespace)
        validate = namespace["validate_target_contract"]
        model_targets = (
            "subtlety", "internal_structure", "calcification", "sphericity",
            "margin", "lobulation", "spiculation", "texture",
        )
        requested = ("subtlety", "sphericity", "margin", "lobulation", "spiculation", "texture")
        positions = validate(model_targets, requested)
        self.assertEqual(tuple(positions), requested)
        self.assertEqual(positions["sphericity"], model_targets.index("sphericity"))
        self.assertEqual(positions["texture"], model_targets.index("texture"))
        with self.assertRaisesRegex(RuntimeError, "absent from the frozen Grad-CAM model"):
            validate(model_targets, ("subtlety", "malignancy"))

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

    def test_schema_defaults_to_top_100_ranked_log_candidates(self):
        stage2 = self.schema["stage_02_detect_and_patch"]["diagnostic_log_detector"]
        self.assertEqual(stage2["patch_limit_default"], 100)
        self.assertIn("--all-log-patches", (IMAGING / "02_detect_and_patch.py").read_text(encoding="utf-8"))

    def test_stage07_csv_columns_are_schema_driven(self):
        stage7 = self.schema["stage_07_visualize"]
        columns = stage7["csv_columns_order"]
        self.assertEqual(columns.count("{attribute_columns}"), 1)
        self.assertTrue(set(self.schema["attribute_contracts"]["stage03_ranked_attributes"]).isdisjoint({"{attribute_columns}"}))
        self.assertIn('get_required(STAGE07, "csv_columns_order", expected_type=list)', (IMAGING / "07_visualize.py").read_text(encoding="utf-8"))

    def test_stage04_log_diagnostic_output_is_separate(self):
        stage4 = self.schema["paths"]["stage_04"]
        stage3 = self.schema["stage_03_nodule_classification"]["outputs"]
        self.assertNotEqual(stage4["output_json"], stage4["log_diagnostic_output_json"])
        self.assertIn("log_diagnostic_predictions_json", stage3)

    def test_stage05_batch_gradcam_uses_manifest_and_cached_checkpoints(self):
        stage5_path = IMAGING / "05_gradcam.py"
        source = stage5_path.read_text(encoding="utf-8")
        self.assertIn('"--manifest"', source)
        self.assertIn('"--batch-limit"', source)
        self.assertIn("model_cache.get(checkpoint_path)", source)
        self.assertIn('get_required(STAGE05_SCHEMA, "batch_processing.patch_output_directory_pattern")', source)

    def test_stage07_excludes_centers_outside_lung_mask(self):
        import numpy as np
        stage7_path = IMAGING / "07_visualize.py"
        tree = ast.parse(stage7_path.read_text(encoding="utf-8"), filename=str(stage7_path))
        fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "sample_mask")
        module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), fn], type_ignores=[])
        namespace = {"np": np}
        exec(compile(ast.fix_missing_locations(module), str(stage7_path), "exec"), namespace)
        mask = np.zeros((5, 5, 5), dtype=bool)
        mask[2, 2, 2] = True
        sample = namespace["sample_mask"]
        self.assertEqual(sample(mask, [2.0, 2.0, 2.0]), (True, (2, 2, 2), None))
        self.assertEqual(sample(mask, [1.0, 1.0, 1.0]), (False, (1, 1, 1), "center_outside_lung_mask"))
        self.assertEqual(sample(mask, [9.0, 1.0, 1.0])[2], "center_outside_ct_volume")
        self.assertEqual(sample(mask, None)[2], "invalid_center_zyx")

    def test_stage07_runner_uses_schema_resolved_output_paths(self):
        runner = (ROOT / "run_pipeline.ps1").read_text(encoding="utf-8")
        self.assertIn("$Schema.paths.stage_07.report_json", runner)
        self.assertIn("$Schema.paths.stage_07.report_csv", runner)
        self.assertIn("$Schema.paths.stage_07.excluded_csv", runner)
        self.assertNotIn("$Schema.stage_07_visualize.outputs.report_json", runner)

    def test_stage07_is_registered_for_static_validation(self):
        validator = (ROOT / "validate_pipeline.py").read_text(encoding="utf-8")
        self.assertIn('"07_visualize.py"', validator)
        self.assertEqual(get_required(self.schema, self.schema["stage_07_visualize"]["outputs"]["report_csv_ref"]), self.schema["paths"]["stage_07"]["report_csv"])

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
