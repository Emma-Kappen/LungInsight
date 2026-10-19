import json
from pathlib import Path

p = Path(r"C:\Users\emkap\Documents\Projects\LungInsight\LungInsight_v2\backend\Clinical\artifacts\dataset_feature_inventory\deep_dive\luad_lusc_baseline_readiness_details_v4.json")
data = json.loads(p.read_text(encoding="utf-8"))

print("=== COHORT COUNTS ===")
for k, v in data.get("row_counts", {}).items():
    print(f"{k}: {v}")

print("\n=== STRICT LABEL DISTRIBUTION ===")
for k, v in data.get("strict_label_distribution", {}).items():
    print(f"{k}: {v}")

print("\n=== SAMPLE TYPES ===")
for k, v in data.get("sample_type_distribution", {}).items():
    print(f"{k}: {v}")

print("\n=== ID INTEGRITY ===")
for k, v in data.get("ids", {}).items():
    print(f"{k}: {v}")

print("\n=== CANDIDATE FEATURE MISSINGNESS ===")
for table, fields in data.get("candidate_predictor_missingness", {}).items():
    print(f"\n[{table}]")
    for feature, stats in fields.items():
        if not stats.get("present", False):
            print(f"{feature}: COLUMN MISSING")
            continue
        print(
            f"{feature}: missing={stats.get('missing_n')} "
            f"({stats.get('missing_pct')}%), "
            f"unique={stats.get('unique_nonmissing')}"
        )
        counts = stats.get("value_counts", {})
        if counts:
            print("  values:", json.dumps(counts, ensure_ascii=False))

print("\n=== NUMERIC FIELD CHECKS ===")
for table, fields in data.get("candidate_numeric_checks", {}).items():
    print(f"\n[{table}]")
    for feature, stats in fields.items():
        print(f"{feature}: {json.dumps(stats, ensure_ascii=False)}")

print("\n=== COHORT WARNING ===")
print(data.get("critical_cohort_warning", ""))

print("\n=== GUARDRAILS ===")
print("Training performed:", data.get("model_training_performed"))
print("Production artifacts modified:", data.get("production_artifacts_modified"))
