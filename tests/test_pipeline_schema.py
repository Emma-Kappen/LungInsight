from __future__ import annotations

import ast
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMAGING = ROOT / "backend" / "Imaging"
STAGES = IMAGING / "stages"
INFERENCE = IMAGING / "inference"
VISUALIZATION = IMAGING / "visualization"
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

    def test_stage05_patch_shape_is_defined_from_schema_and_matches_stage03(self):
        stage5 = INFERENCE / "05_gradcam.py"
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
        stage5 = INFERENCE / "05_gradcam.py"
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
        stage4 = STAGES / "04_malignancy_prediction.py"
        tree = ast.parse(stage4.read_text(encoding="utf-8"), filename=str(stage4))
        assigned = {
            target.id
            for node in ast.walk(tree)
            if isinstance(node, (ast.Assign, ast.AnnAssign))
            for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
            if isinstance(target, ast.Name)
        }
        self.assertIn("STAGE03_PATHS", assigned)

    def test_stage07_csv_columns_are_schema_driven(self):
        stage7 = self.schema["stage_07_visualize"]
        columns = stage7["csv_columns_order"]
        self.assertEqual(columns.count("{attribute_columns}"), 1)
        self.assertTrue(set(self.schema["attribute_contracts"]["stage03_ranked_attributes"]).isdisjoint({"{attribute_columns}"}))
        self.assertIn('get_required(STAGE07, "csv_columns_order", expected_type=list)', (VISUALIZATION / "07_visualize.py").read_text(encoding="utf-8"))

    def test_fused_candidate_stream_is_the_only_downstream_inference_path(self):
        stage3 = self.schema["stage_03_nodule_classification"]
        self.assertNotIn("process_log_diagnostic_default", stage3["inputs"])
        self.assertNotIn("log_diagnostic_predictions_json", stage3["outputs"])
        runner = (ROOT / "run_pipeline.ps1").read_text(encoding="utf-8")
        self.assertIn("[string]$BrockInputs", runner)
        self.assertIn("Stage 03 - frozen attribute inference for selected fused candidates", runner)
        self.assertIn("Stage 05 - Grad-CAM for all eligible fused candidates", runner)
        self.assertNotIn("$LogStage03", runner)
        self.assertNotIn("$LogStage06", runner)

    def test_stage05_batch_gradcam_uses_manifest_and_cached_checkpoints(self):
        stage5_path = INFERENCE / "05_gradcam.py"
        source = stage5_path.read_text(encoding="utf-8")
        self.assertIn('"--manifest"', source)
        self.assertIn('"--batch-limit"', source)
        self.assertIn("model_cache.get(checkpoint_path)", source)
        self.assertIn("STAGE05_SCHEMA", source)
        self.assertIn("batch_processing", self.schema["stage_05_gradcam"])

    def test_stage07_excludes_centers_outside_lung_mask(self):
        import numpy as np
        stage7_path = VISUALIZATION / "07_visualize.py"
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

    def test_runner_records_brock_formula_identity_without_removed_model_path(self):
        runner = (ROOT / "run_pipeline.ps1").read_text(encoding="utf-8")
        self.assertIn("authoritative_brock_predictions", runner)
        self.assertIn("$Schema.stage_04_malignancy_prediction.formula_version", runner)
        self.assertNotIn("$Schema.stage_04_malignancy_prediction.model_path", runner)

    def test_runner_publishes_completed_contract_after_run_summary(self):
        runner = (ROOT / "run_pipeline.ps1").read_text(encoding="utf-8")
        summary_write = runner.index("$RunSummary | ConvertTo-Json")
        contract_publish = runner.index("Publish completed Imaging-to-Clinical contract")
        self.assertLess(summary_write, contract_publish)
        self.assertIn("$Schema.imaging_clinical_contract.filename", runner)
        self.assertIn("processing_status=''completed''", runner)

    def test_runner_accepts_explicit_uploaded_dicom_input_tree(self):
        runner = (ROOT / "run_pipeline.ps1").read_text(encoding="utf-8")
        self.assertIn("[string]$InputPatientPath", runner)
        self.assertIn("$Stage01Input = if ($InputPatientPath)", runner)

    def test_stage07_is_registered_for_static_validation(self):
        validator = (IMAGING / "validation" / "validate_pipeline.py").read_text(encoding="utf-8")
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
        stage01 = STAGES / "01_hu_and_crop.py"
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
