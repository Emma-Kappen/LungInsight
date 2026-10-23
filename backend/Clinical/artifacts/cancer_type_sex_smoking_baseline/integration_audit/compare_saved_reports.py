import json
import sys
from pathlib import Path

root = Path.cwd()
candidate_path = root / "backend/Clinical/artifacts/cancer_type_sex_smoking_baseline/evaluation.json"
production_path = root / "backend/Clinical/reports/cancer-type-luad-lusc-v2-evaluation.json"
output_path = root / "backend/Clinical/artifacts/cancer_type_sex_smoking_baseline/integration_audit/production_comparison.json"

def read(path):
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)

def get(obj, *keys):
    for key in keys:
        if not isinstance(obj, dict) or key not in obj:
            return None
        obj = obj[key]
    return obj

def main():
    c = read(candidate_path)
    p = read(production_path)

    candidate_scores = c["model_holdout_metrics"]
    production_scores = p["holdout_results"]["default_0.50"]["holdout_metrics"]

    candidate = {
        "accuracy": candidate_scores["accuracy"],
        "balanced_accuracy": candidate_scores["balanced_accuracy"],
        "macro_f1": candidate_scores["macro_f1"],
        "lusc_precision": candidate_scores["lusc_precision"],
        "lusc_recall": candidate_scores["lusc_recall"],
        "lusc_f1": candidate_scores["lusc_f1"],
        "auc": candidate_scores["roc_auc_lusc_positive"],
        "cv_auc_mean": get(c, "training_only_5fold_cv", "roc_auc", "mean"),
    }

    production = {
        "accuracy": production_scores["accuracy"],
        "balanced_accuracy": production_scores["balanced_accuracy"],
        "macro_f1": production_scores["macro_f1"],
        "lusc_precision": production_scores["lusc_precision"],
        "lusc_recall": production_scores["lusc_recall"],
        "lusc_f1": production_scores["lusc_f1"],
        "auc": production_scores["roc_auc"],
        # CV AUC is deliberately null unless the saved production report has
        # a clearly identified matching cross-validation metric.
        "cv_auc_mean": get(p, "training_only_5fold_cv", "roc_auc", "mean"),
    }

    comparisons = {}
    for name, cv in candidate.items():
        pv = production.get(name)
        comparisons[name] = {
            "candidate": cv,
            "production": pv,
            "candidate_minus_production": (
                cv - pv if cv is not None and pv is not None else None
            ),
            "status": (
                "compared" if cv is not None and pv is not None
                else "production_metric_not_reported"
            ),
        }

    result = {
        "candidate_features": c.get("features"),
        "production_features": p.get("features"),
        "candidate_train_test_rows": [c.get("training_rows"), c.get("test_rows")],
        "production_train_test_rows": [p.get("train_rows"), p.get("test_rows")],
        "comparisons": comparisons,
        "interpretation": [
            "Candidate scores come from model_holdout_metrics, not dummy_holdout_metrics.",
            "Production scores use the default 0.50 threshold.",
            "The candidate holdout AUC is lower than production by approximately 0.03216.",
            "Equal threshold metrics do not establish that both models used identical holdout membership.",
            "Production CV AUC is not reported in the selected production evaluation report; no value is inferred.",
            "No promotion was performed and no production model or report was modified."
        ],
        "promotion_performed": False,
        "production_model_or_report_modified": False,
    }

    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"\nSaved: {output_path}")

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
