import ast
import json
import re
from pathlib import Path

root = Path.cwd()
clinical = root / "backend" / "Clinical"
candidate_dir = clinical / "artifacts" / "cancer_type_sex_smoking_baseline"
audit_dir = candidate_dir / "integration_audit"
out_path = audit_dir / "stage_removal_dependency_inventory.json"

targets = [
    clinical / "inference.py",
    clinical / "schemas" / "clinical_inference_v1.schema.json",
    clinical / "schemas" / "clinical_prediction_v1.schema.json",
    clinical / "tests" / "test_cancer_type_luad_lusc_v2.py",
    clinical / "tests" / "test_clinical_pipeline.py",
    clinical / "tests" / "test_cancer_type_dataset.py",
    root / "frontend" / "app.js",
    root / "frontend" / "index.html",
]

extensions = {".py", ".json", ".js", ".html", ".ts", ".tsx"}
terms = re.compile(
    r"stage_at_baseline|stage\b|cancer.type|luad|lusc|smoking_history|sex_at_baseline",
    re.IGNORECASE,
)

results = []
for path in targets:
    if not path.is_file():
        results.append({
            "file": str(path.relative_to(root)),
            "exists": False,
            "matches": [],
        })
        continue

    try:
        lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    except OSError as exc:
        results.append({
            "file": str(path.relative_to(root)),
            "exists": True,
            "error": str(exc),
            "matches": [],
        })
        continue

    matches = []
    for i, line in enumerate(lines, 1):
        if terms.search(line):
            matches.append({
                "line": i,
                "text": line.strip()[:300],
            })

    results.append({
        "file": str(path.relative_to(root)),
        "exists": True,
        "matches": matches,
    })

# Search additional Clinical API/contract source files and frontend source files,
# excluding generated data, environments, caches, and model artifacts.
extra_roots = [
    clinical / "api",
    clinical / "schemas",
    clinical / "tests",
    root / "frontend",
]
extra_results = []
skip_parts = {
    ".git", "node_modules", "__pycache__", "l-env", "venv",
    "artifacts", "models", "datasets", "output",
}
already = {Path(item["file"]) for item in results}

for base in extra_roots:
    if not base.exists():
        continue
    for path in base.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        try:
            rel = path.relative_to(root)
        except ValueError:
            continue
        if any(part in skip_parts for part in rel.parts):
            continue
        if rel in already:
            continue
        try:
            lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
        except OSError:
            continue

        matches = []
        for i, line in enumerate(lines, 1):
            if re.search(r"stage_at_baseline|stage\b", line, re.IGNORECASE):
                matches.append({"line": i, "text": line.strip()[:300]})

        if matches:
            extra_results.append({
                "file": str(rel),
                "exists": True,
                "stage_matches": matches,
            })

result = {
    "purpose": "Read-only inventory of Clinical/API/UI dependencies before changing the cancer-type model input contract.",
    "candidate_features": ["sex_at_baseline", "smoking_history"],
    "production_features": ["sex_at_baseline", "smoking_history", "stage"],
    "primary_files": results,
    "additional_stage_references": extra_results,
    "production_model_modified": False,
    "schemas_modified": False,
    "inference_modified": False,
    "tests_run": False,
    "warning": "This inventory is not proof that every runtime dependency has been found. Review the listed references before editing.",
}

audit_dir.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

print(json.dumps(result, indent=2))
print(f"\nSaved: {out_path}")
