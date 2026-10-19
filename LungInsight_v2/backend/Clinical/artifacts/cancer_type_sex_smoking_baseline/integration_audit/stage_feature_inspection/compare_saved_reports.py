import json
from pathlib import Path

root = Path.cwd()
clinical = root / "backend" / "Clinical"
candidate_dir = clinical / "artifacts" / "cancer_type_sex_smoking_baseline"
candidate_path = candidate_dir / "evaluation.json"
production_path = clinical / "reports" / "cancer-type-luad-lusc-v2-evaluation.json"
audit_path = candidate_dir / "integration_audit" / "audit.json"
output_path = candidate_dir / "integration_audit" / "production_comparison.json"

def load(path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)

candidate = load(candidate_path)
production = load(production_path)
audit = load(audit_path)

def flatten(obj, prefix=""):
    found = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            path = f"{prefix}.{key}" if prefix else key
            if isinstance(value, (dict, list)):
                found.update(flatten(value, path))
            else:
                found[path] = value
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            found.update(flatten(value, f"{prefix}[{i}]"))
    return found

def find_metric(obj, aliases):
    aliases = {a.lower() for a in aliases}
    for path, value in flatten(obj).items():
        key = path.rsplit(".", 1)[-1].lower()
        if key in aliases and isinstance(value, (int, float)) and not isinstance(value, bool):
            return {"path": path, "value": float(value)}
    return None

metric_aliases = {
    "holdout_roc_auc": [
        "roc_auc_lusc_positive", "holdout_roc_auc", "roc_auc", "auc"
    ],
    "holdout_balanced_accuracy": [
        "balanced_accuracy", "holdout_balanced_accuracy"
    ],
    "holdout_macro_f1": ["macro_f1", "holdout_macro_f1"],
    "training_cv_roc_auc_mean": [
        "training_cv_roc_auc_mean", "cv_roc_auc_mean"
    ],
    "training_cv_balanced_accuracy_mean": [
        "training_cv_balanced_accuracy_mean", "cv_balanced_accuracy_mean"
    ],
}

# Resolve known nested CV metric structures first, then use recursive fallback.
def cv_value(report, metric):
    cv = report.get("training_only_5fold_cv")
    if isinstance(cv, dict):
        item = cv.get(metric)
        if isinstance(item, dict) and isinstance(item.get("mean"), (int, float)):
            return {"path": f"training_only_5fold_cv.{metric}.mean", "value": float(item["mean"])}
    return None

def resolve(report, key):
    if key == "training_cv_roc_auc_mean":
        result = cv_value(report, "roc_auc")
        if result:
            return result
    if key == "training_cv_balanced_accuracy_mean":
        result = cv_value(report, "balanced_accuracy")
        if result:
            return result

    aliases = metric_aliases[key]
    # Prefer the holdout metrics section to avoid confusing CV AUC with holdout AUC.
    if key.startswith("holdout_"):
        section = report.get("model_holdout_metrics")
        if isinstance(section, dict):
            for alias in aliases:
                value = section.get(alias)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    return {"path": f"model_holdout_metrics.{alias}", "value": float(value)}
    return find_metric(report, aliases)

candidate_holdout = candidate.get("model_holdout_metrics", {})
candidate_cv = candidate.get("training_only_5fold_cv", {})

def candidate_resolve(key):
    if key == "holdout_roc_auc":
        value = candidate_holdout.get("roc_auc_lusc_positive")
        return {"path": "model_holdout_metrics.roc_auc_lusc_positive", "value": float(value)} if isinstance(value, (int, float)) else None
    if key == "holdout_balanced_accuracy":
        value = candidate_holdout.get("balanced_accuracy")
        return {"path": "model_holdout_metrics.balanced_accuracy", "value": float(value)} if isinstance(value, (int, float)) else None
    if key == "holdout_macro_f1":
        value = candidate_holdout.get("macro_f1")
        return {"path": "model_holdout_metrics.macro_f1", "value": float(value)} if isinstance(value, (int, float)) else None
    if key == "training_cv_roc_auc_mean":
        return cv_value(candidate, "roc_auc")
    if key == "training_cv_balanced_accuracy_mean":
        return cv_value(candidate, "balanced_accuracy")
    return None

comparison = {}
for key in metric_aliases:
    cand = candidate_resolve(key)
    prod = resolve(production, key)
    delta = cand["value"] - prod["value"] if cand and prod else None
    comparison[key] = {
        "candidate": cand,
        "production": prod,
        "candidate_minus_production": delta,
        "status": "PASS" if cand and prod else "MISSING_METRIC",
    }

# Correct the previous audit's null production reference using actual saved JSON paths.
audit["production_reference"] = {
    "holdout_roc_auc": comparison["holdout_roc_auc"]["production"]["value"]
        if comparison["holdout_roc_auc"]["production"] else None,
    "holdout_balanced_accuracy": comparison["holdout_balanced_accuracy"]["production"]["value"]
        if comparison["holdout_balanced_accuracy"]["production"] else None,
    "holdout_macro_f1": comparison["holdout_macro_f1"]["production"]["value"]
        if comparison["holdout_macro_f1"]["production"] else None,
    "training_cv_roc_auc_mean": comparison["training_cv_roc_auc_mean"]["production"]["value"]
        if comparison["training_cv_roc_auc_mean"]["production"] else None,
    "training_cv_balanced_accuracy_mean": comparison["training_cv_balanced_accuracy_mean"]["production"]["value"]
        if comparison["training_cv_balanced_accuracy_mean"]["production"] else None,
}
audit["holdout_auc_difference_candidate_minus_production"] = comparison["holdout_roc_auc"]["candidate_minus_production"]
audit["corrected_production_metric_paths"] = {
    key: item["production"]["path"] if item["production"] else None
    for key, item in comparison.items()
}
audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
output_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")

print("PRODUCTION REPORT TOP-LEVEL KEYS:", list(production.keys()))
print("\nMETRIC COMPARISON:")
print(json.dumps(comparison, indent=2))
print(f"\nCorrected audit: {audit_path}")
print(f"Detailed comparison: {output_path}")
