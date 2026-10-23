#!/usr/bin/env python3
"""Audit supplied Stage 01-07 scripts against meta_schema.json and required files."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

EXPECTED_SCHEMA_SECTIONS = {
    "01_hu_and_crop.py": ["stage_01_hu_and_crop", "paths"],
    "02_detect_and_patch.py": ["stage_01_hu_and_crop", "paths", "defaults"],
    "03_nodule_classification.py": ["paths", "stage_03_nodule_classification.model.attribute_order_ref"],
    "04_malignancy_prediction.py": ["stage_04_malignancy_prediction", "paths"],
    "04_malignancy_classifier.py": ["stage_04_malignancy_classifier"],
    "05_gradcam.py": ["stage_05_gradcam", "visualization"],
    "06_full_ct_gradcam.py": ["stage_06_full_ct_gradcam", "visualization"],
    "07_visualize.py": ["stage_07_visualize", "paths", "attribute_contracts.stage03_ranked_attributes"],
}

REQUIRED_UNSUPPLIED_FILES = [
    "backend/Imaging/pipeline_schema.py",
    "backend/models/malignancy/frozen_attribute_common.py",
    "backend/models/nodule_ranker_3d_distributional_v1/spatial_v1/model_objective.py",
]


def nested_get(obj, dotted):
    current = obj
    for part in dotted.split("."):
        if not isinstance(current, dict) or part not in current:
            return False
        current = current[part]
    return True


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True, help="LungInsight_v2 project root")
    args = p.parse_args()
    root = args.root.resolve()
    schema_path = root / "meta_schema.json"
    if not schema_path.is_file():
        raise FileNotFoundError(f"Schema not found: {schema_path}")
    schema = json.loads(schema_path.read_text(encoding="utf-8-sig"))
    top_level = sorted(schema.keys())
    problems = []
    for filename, paths in EXPECTED_SCHEMA_SECTIONS.items():
        file_path = root / "backend" / "Imaging" / filename
        if not file_path.is_file():
            problems.append(f"missing supplied stage file: {file_path.relative_to(root)}")
            continue
        for key in paths:
            if not nested_get(schema, key):
                problems.append(f"{filename} requires missing schema field: {key}")
    missing_files = [rel for rel in REQUIRED_UNSUPPLIED_FILES if not (root / rel).is_file()]
    for rel in missing_files:
        problems.append(f"missing required project dependency: {rel}")
    report = {
        "project_root": str(root),
        "schema_version": schema.get("schema_version"),
        "schema_top_level_keys": top_level,
        "full_pipeline_compatible_from_supplied_files": not problems,
        "problems": problems,
        "note": "This audit checks interface presence, not checkpoint loading, model validity, clinical validity, or GPU execution.",
    }
    print(json.dumps(report, indent=2))
    return 1 if problems else 0

if __name__ == "__main__":
    raise SystemExit(main())
