from __future__ import annotations

import json
import sys
from pathlib import Path

SCHEMA_PATH = Path("backend/Clinical/schemas/clinical_prediction_v1.schema.json")

EXPLANATION_SCHEMA = {
    "type": "object",
    "required": ["status", "method", "feature_contributions", "limitations"],
    "properties": {
        "status": {"type": "string", "enum": ["available", "unavailable"]},
        "method": {"type": "string", "const": "exact_interventional_shapley"},
        "output_space": {"type": "string"},
        "class_order": {"type": "array", "items": {"type": "string"}},
        "base_values": {"type": "object", "additionalProperties": {"type": "number"}},
        "prediction_values": {"type": "object", "additionalProperties": {"type": "number"}},
        "feature_contributions": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["feature", "values"],
                "properties": {
                    "feature": {"type": "string"},
                    "feature_value": {"type": ["string", "null"]},
                    "values": {"type": "object", "additionalProperties": {"type": "number"}},
                },
                "additionalProperties": True,
            },
        },
        "reference_population": {
            "type": "object",
            "properties": {
                "cohort": {"type": "string"},
                "partition": {"type": "string"},
                "rows": {"type": "integer"},
                "features": {"type": "array", "items": {"type": "string"}},
            },
            "additionalProperties": True,
        },
        "failure_reason": {"type": "string"},
        "limitations": {"type": "array", "items": {"type": "string"}},
    },
    "additionalProperties": True,
}


def main() -> int:
    if not SCHEMA_PATH.is_file():
        print(f"ERROR: schema not found: {SCHEMA_PATH}", file=sys.stderr)
        return 2
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    matches = []

    def walk(node, path="$"):
        if isinstance(node, dict):
            props = node.get("properties")
            if isinstance(props, dict) and {"probabilities", "predicted_class", "feature_values"}.issubset(props):
                matches.append((node, path))
            for key, value in node.items():
                walk(value, f"{path}.{key}")
        elif isinstance(node, list):
            for i, value in enumerate(node):
                walk(value, f"{path}[{i}]")

    walk(schema)
    if len(matches) != 1:
        print(
            f"ERROR: expected exactly one histology result schema containing "
            f"probabilities/predicted_class/feature_values; found {len(matches)}. "
            "No changes made.",
            file=sys.stderr,
        )
        return 3

    node, path = matches[0]
    props = node["properties"]
    existing = props.get("explanation")
    if existing == EXPLANATION_SCHEMA:
        print(f"Schema already supports histology SHAP at {path}.")
        return 0
    backup = SCHEMA_PATH.with_suffix(".schema-backup.json")
    if not backup.exists():
        backup.write_text(json.dumps(schema, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    props["explanation"] = EXPLANATION_SCHEMA
    SCHEMA_PATH.write_text(json.dumps(schema, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Added optional histology SHAP explanation schema at {path}.")
    print(f"Backup: {backup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
