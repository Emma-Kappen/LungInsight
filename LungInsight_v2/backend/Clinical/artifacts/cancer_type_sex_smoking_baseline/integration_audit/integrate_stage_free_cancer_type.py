from __future__ import annotations

import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path.cwd().resolve()
AUDIT = ROOT / "backend/Clinical/artifacts/cancer_type_sex_smoking_baseline/integration_audit"
REPORT = AUDIT / "stage_free_inference_integration_report.json"

FILES = [
    ROOT / "backend/Clinical/inference.py",
    ROOT / "backend/Clinical/schemas/clinical_inference_v1.schema.json",
    ROOT / "backend/Clinical/schemas/clinical_prediction_v1.schema.json",
    ROOT / "frontend/app.js",
    ROOT / "frontend/index.html",
]
CANDIDATE = ROOT / "backend/Clinical/artifacts/cancer_type_sex_smoking_baseline/sex-smoking-baseline.joblib"
PRODUCTION = ROOT / "backend/Clinical/models/cancer-type-luad-lusc-v2.joblib"
PRODUCTION_REPORT = ROOT / "backend/Clinical/reports/cancer-type-luad-lusc-v2-evaluation.json"

result = {
    "started_utc": datetime.now(timezone.utc).isoformat(),
    "root": str(ROOT),
    "scope": "stage-free cancer-type candidate integration; production model and report must remain unchanged",
    "edits": [],
    "failures": [],
    "tests": [],
}

def record_failure(where, exc):
    result["failures"].append({
        "where": where,
        "error": f"{type(exc).__name__}: {exc}",
        "traceback": traceback.format_exc(),
    })

try:
    import joblib
    if not CANDIDATE.exists():
        raise FileNotFoundError(f"Candidate model missing: {CANDIDATE}")
    candidate = joblib.load(CANDIDATE)
    if not hasattr(candidate, "predict_proba"):
        raise TypeError(f"Candidate must be a fitted sklearn Pipeline; got {type(candidate).__name__}")
    feature_names = list(getattr(candidate, "feature_names_in_", []))
    expected = ["sex_at_baseline", "smoking_history"]
    if feature_names != expected:
        raise ValueError(f"Candidate feature order mismatch: expected {expected}, got {feature_names}")
    result["candidate_validation"] = {
        "valid": True,
        "artifact": str(CANDIDATE.relative_to(ROOT)),
        "artifact_type": type(candidate).__name__,
        "feature_order": feature_names,
        "classes": [str(x) for x in getattr(candidate, "classes_", [])],
    }
except Exception as exc:
    record_failure("candidate_validation", exc)

# Deliberately inspect current source and available interfaces before patching.
# Avoid broad text replacement: inference contract changes require function-level understanding.
try:
    for path in FILES:
        if not path.exists():
            result["failures"].append({"where": "source_inventory", "error": f"Missing file: {path}"})
            continue
        text = path.read_text(encoding="utf-8")
        result.setdefault("source_inventory", {})[str(path.relative_to(ROOT))] = {
            "exists": True,
            "bytes": len(text.encode("utf-8")),
            "has_stage_reference": "stage_at_baseline" in text,
            "has_smoking_history_reference": "smoking_history" in text,
            "sha256": __import__("hashlib").sha256(text.encode("utf-8")).hexdigest(),
        }

    inference_path = ROOT / "backend/Clinical/inference.py"
    source = inference_path.read_text(encoding="utf-8")
    result["inference_stage_free_integration"] = {
        "candidate_model_feature_order": ["sex_at_baseline", "smoking_history"],
        "implementation_status": "not_patched",
        "reason": (
            "The current modified inference.py contains substantial uncommitted changes. "
            "A blind patch could overwrite unrelated work or break artifact loading. "
            "The next step is to inspect the cancer-type loading and prediction functions "
            "and patch only those functions while preserving the current worktree."
        ),
    }
except Exception as exc:
    record_failure("source_inventory", exc)

result["production_artifacts"] = {
    "model_exists": PRODUCTION.exists(),
    "evaluation_report_exists": PRODUCTION_REPORT.exists(),
    "promotion_performed": False,
}
result["tests"] = [{
    "command": "not run by this integration inspection",
    "status": "not_run",
    "reason": "Implementation needs a narrow function-level patch based on the current inference loader and prediction path.",
}]
result["finished_utc"] = datetime.now(timezone.utc).isoformat()
REPORT.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(result, indent=2, ensure_ascii=False))
sys.exit(1 if result["failures"] else 0)
