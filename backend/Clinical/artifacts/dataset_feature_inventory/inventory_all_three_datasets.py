from __future__ import annotations

import csv
import json
import os
import re
import traceback
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(r"C:\Users\emkap\Documents\Projects\LungInsight\LungInsight_v2")
DATASETS_ROOT = Path(r"C:\Users\emkap\Documents\Projects\LungInsight\datasets")
OUTPUT_DIR = PROJECT_ROOT / "backend" / "Clinical" / "artifacts" / "dataset_feature_inventory"

DATASETS = {
    "MSK-CHORD-2024": DATASETS_ROOT / "msk_chord_2024",
    "TCGA-LUAD": DATASETS_ROOT / "TCGA-LUAD",
    "TCGA-LUSC": DATASETS_ROOT / "TCGA-LUSC",
}

SUPPORTED = {".csv", ".tsv", ".txt", ".xlsx", ".xls", ".parquet"}
SAMPLE_ROWS = 5000
MAX_EXAMPLES = 5

def clean(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return text if text else None

def read_table(path: Path):
    suffix = path.suffix.lower()

    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, nrows=SAMPLE_ROWS, dtype=object)

    if suffix == ".parquet":
        frame = pd.read_parquet(path)
        return frame.head(SAMPLE_ROWS)

    if suffix == ".csv":
        return pd.read_csv(
            path, nrows=SAMPLE_ROWS, dtype=object,
            encoding_errors="replace", low_memory=False
        )

    if suffix in {".tsv", ".txt"}:
        # Most clinical text tables are tab-separated. Detect delimiter
        # from a small prefix when the file is not clearly tabular.
        with path.open("r", encoding="utf-8-sig", errors="replace") as f:
            prefix = f.read(16384)

        try:
            dialect = csv.Sniffer().sniff(prefix, delimiters="\t,;|")
            sep = dialect.delimiter
        except csv.Error:
            sep = "\t" if "\t" in prefix else ","

        return pd.read_csv(
            path, sep=sep, nrows=SAMPLE_ROWS, dtype=object,
            encoding="utf-8-sig", encoding_errors="replace",
            low_memory=False, comment=None
        )

    raise ValueError(f"Unsupported table format: {suffix}")

def likely_data_file(path: Path) -> bool:
    if path.suffix.lower() not in SUPPORTED:
        return False

    excluded_parts = {
        ".git", "node_modules", "__pycache__", ".venv",
        "venv", "l-env", "dicom", "dicoms", "images",
    }
    return not any(part.lower() in excluded_parts for part in path.parts)

def normalize_feature(name):
    value = str(name).strip().lower()
    value = re.sub(r"[\s\-./]+", "_", value)
    value = re.sub(r"[^a-z0-9_]", "", value)
    return re.sub(r"_+", "_", value).strip("_")

def classify_feature(name, values):
    name_l = str(name).lower()
    vals = [str(v).strip().lower() for v in values if v is not None]
    joined = " ".join(vals[:MAX_EXAMPLES])

    if any(k in name_l for k in ["patient_id", "sample_id", "case_id", "submitter_id", "barcode"]):
        return "identifier"
    if any(k in name_l for k in ["diagnosis", "histology", "histologic", "cancer_type", "tumor_type", "disease"]):
        return "target_or_diagnosis_candidate"
    if any(k in name_l for k in ["stage", "tnm", "t_stage", "n_stage", "m_stage"]):
        return "stage"
    if any(k in name_l for k in ["smok", "pack_year", "tobacco", "cessation"]):
        return "smoking"
    if any(k in name_l for k in ["sex", "gender"]):
        return "sex_or_gender"
    if any(k in name_l for k in ["age", "birth", "dob"]):
        return "age_or_demographic"
    if any(k in name_l for k in ["survival", "vital", "death", "follow_up", "followup", "days_to"]):
        return "survival_or_followup"
    if any(k in name_l for k in ["treatment", "therapy", "drug", "radiation", "chemo", "surgery", "neoadjuvant"]):
        return "treatment"
    if any(k in name_l for k in ["ecog", "kps", "performance_status"]):
        return "performance_status"
    if any(k in name_l for k in ["mutation", "variant", "gene", "genomic", "molecular", "expression"]):
        return "molecular_or_genomic"
    if any(k in name_l for k in ["nodule", "diameter", "volume", "spicul", "lobulat", "ground_glass", "solid_component", "attenuation", "radiomic", "texture", "hu_"]):
        return "imaging_or_nodule_candidate"
    if any(k in name_l for k in ["race", "ethnic", "ancestry"]):
        return "race_or_ethnicity"
    if any(k in name_l for k in ["comorbid", "copd", "emphysema", "medical_history", "family_history"]):
        return "medical_history"
    if any(k in name_l for k in ["date", "time", "timestamp"]):
        return "date_or_time"
    if any(k in name_l for k in ["site", "center", "institution", "cohort", "project", "source"]):
        return "dataset_or_site_metadata"
    if vals and all(v in {"0", "1", "true", "false", "yes", "no"} for v in vals):
        return "binary_candidate"
    if len(set(vals)) <= 20:
        return "categorical_or_low_cardinality"
    return "numeric_or_high_cardinality_or_free_text"

def safe_examples(series):
    output = []
    seen = set()
    for raw in series.tolist():
        value = clean(raw)
        if value is not None and value not in seen:
            output.append(value[:180])
            seen.add(value)
        if len(output) >= MAX_EXAMPLES:
            break
    return output

def inspect_json(path: Path):
    try:
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return [], "JSON parse failed or nonstandard JSON"

    records = []
    if isinstance(raw, dict):
        records.append((".".join(map(str, raw.keys())), "json_top_level_keys"))
        for key, value in raw.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    records.append((f"{key}.{subkey}", type(subvalue).__name__))
            elif isinstance(value, list) and value and isinstance(value[0], dict):
                keys = sorted(set().union(*(x.keys() for x in value[:100] if isinstance(x, dict))))
                for key2 in keys:
                    records.append((f"{key}[] . {key2}", "array_object_field"))
    elif isinstance(raw, list) and raw and isinstance(raw[0], dict):
        keys = sorted(set().union(*(x.keys() for x in raw[:100] if isinstance(x, dict))))
        for key in keys:
            records.append((key, "array_object_field"))
    return records, "JSON structure inspected"

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    file_rows = []
    feature_rows = []
    errors = []
    discovered_files = {}

    for dataset_name, dataset_dir in DATASETS.items():
        print(f"Inspecting {dataset_name}: {dataset_dir}")

        if not dataset_dir.exists():
            errors.append({
                "dataset": dataset_name,
                "file": str(dataset_dir),
                "error": "Dataset directory not found",
            })
            discovered_files[dataset_name] = []
            continue

        files = sorted(
            p for p in dataset_dir.rglob("*")
            if p.is_file() and likely_data_file(p)
        )
        discovered_files[dataset_name] = files

        for path in files:
            relative = str(path.relative_to(DATASETS_ROOT))
            size_mb = round(path.stat().st_size / (1024 * 1024), 3)

            if path.suffix.lower() == ".json":
                json_fields, json_note = inspect_json(path)
                file_rows.append({
                    "dataset": dataset_name,
                    "relative_path": relative,
                    "format": path.suffix.lower(),
                    "size_mb": size_mb,
                    "sample_rows_read": None,
                    "column_count": len(json_fields),
                    "inspection_status": json_note,
                })
                for name, kind in json_fields:
                    feature_rows.append({
                        "dataset": dataset_name,
                        "relative_path": relative,
                        "feature_name": name,
                        "normalized_name": normalize_feature(name),
                        "inferred_type": kind,
                        "sample_non_null": None,
                        "sample_missing_pct": None,
                        "sample_unique_count": None,
                        "sample_examples": None,
                        "semantic_group": classify_feature(name, []),
                    })
                continue

            try:
                frame = read_table(path)
                file_rows.append({
                    "dataset": dataset_name,
                    "relative_path": relative,
                    "format": path.suffix.lower(),
                    "size_mb": size_mb,
                    "sample_rows_read": len(frame),
                    "column_count": len(frame.columns),
                    "inspection_status": "ok",
                })

                for col in frame.columns:
                    series = frame[col]
                    non_null = series.map(clean).dropna()
                    examples = safe_examples(series)
                    feature_rows.append({
                        "dataset": dataset_name,
                        "relative_path": relative,
                        "feature_name": str(col),
                        "normalized_name": normalize_feature(col),
                        "inferred_type": str(series.map(clean).dropna().map(type).value_counts().to_dict()),
                        "sample_non_null": int(non_null.size),
                        "sample_missing_pct": round(float(series.map(clean).isna().mean() * 100), 2),
                        "sample_unique_count": int(non_null.nunique()),
                        "sample_examples": json.dumps(examples, ensure_ascii=False),
                        "semantic_group": classify_feature(col, examples),
                    })
            except Exception as exc:
                file_rows.append({
                    "dataset": dataset_name,
                    "relative_path": relative,
                    "format": path.suffix.lower(),
                    "size_mb": size_mb,
                    "sample_rows_read": None,
                    "column_count": None,
                    "inspection_status": f"ERROR: {type(exc).__name__}: {exc}",
                })
                errors.append({
                    "dataset": dataset_name,
                    "file": relative,
                    "error": f"{type(exc).__name__}: {exc}",
                })

    files_df = pd.DataFrame(file_rows)
    features_df = pd.DataFrame(feature_rows)

    files_df.to_csv(OUTPUT_DIR / "source_file_inventory.csv", index=False, encoding="utf-8-sig")
    features_df.to_csv(OUTPUT_DIR / "all_features_long.csv", index=False, encoding="utf-8-sig")

    if not features_df.empty:
        summary = (
            features_df.groupby(["dataset", "feature_name", "normalized_name", "semantic_group"], dropna=False)
            .agg(
                files_containing_feature=("relative_path", "nunique"),
                file_examples=("relative_path", lambda s: " ; ".join(list(dict.fromkeys(s.astype(str)))[:8])),
                observed_examples=("sample_examples", lambda s: " ; ".join(list(dict.fromkeys(s.dropna().astype(str)))[:3])),
            )
            .reset_index()
            .sort_values(["dataset", "semantic_group", "feature_name"])
        )
        summary.to_csv(OUTPUT_DIR / "features_by_dataset.csv", index=False, encoding="utf-8-sig")

        # Exact normalized-name overlap is only a first-pass comparison:
        # matching names do not guarantee matching definitions or units.
        presence = (
            features_df[["dataset", "normalized_name", "feature_name"]]
            .drop_duplicates()
            .assign(present=True)
            .pivot_table(
                index="normalized_name", columns="dataset", values="present",
                aggfunc="any", fill_value=False
            )
            .reset_index()
        )
        presence.to_csv(OUTPUT_DIR / "normalized_feature_overlap.csv", index=False, encoding="utf-8-sig")

        canonical = (
            features_df.groupby(["normalized_name", "dataset"])["feature_name"]
            .agg(lambda s: " | ".join(sorted(set(s.astype(str)))[:8]))
            .unstack("dataset", fill_value="")
            .reset_index()
        )
        canonical.to_csv(OUTPUT_DIR / "feature_name_crosswalk.csv", index=False, encoding="utf-8-sig")

    with (OUTPUT_DIR / "inspection_errors.json").open("w", encoding="utf-8") as f:
        json.dump(errors, f, indent=2, ensure_ascii=False)

    # Include any existing model-dataset audit reports so we can compare
    # raw source schemas against the already-built model-ready dataset.
    prior_audits = []
    artifact_root = PROJECT_ROOT / "backend" / "Clinical" / "artifacts"
    for base in [artifact_root / "cancer_type_dataset", artifact_root / "cancer_type_sex_smoking_baseline"]:
        if base.exists():
            for path in base.rglob("*"):
                if path.is_file() and path.name.lower() in {
                    "dataset_audit.json", "evaluation.json", "manifest.json",
                    "feature_schema.json", "metadata.json"
                }:
                    prior_audits.append({
                        "path": str(path),
                        "size_bytes": path.stat().st_size,
                    })

    (OUTPUT_DIR / "existing_model_audit_files.json").write_text(
        json.dumps(prior_audits, indent=2), encoding="utf-8"
    )

    report = []
    report.append("# Dataset feature inventory")
    report.append("")
    report.append("This is a schema/sample inspection, not a full data-quality or label-validity audit.")
    report.append("Table files are sampled up to 5,000 rows; missingness, examples, and unique counts are sample-based.")
    report.append("Exact or normalized column-name overlap does not establish semantic equivalence.")
    report.append("")
    report.append("## Dataset locations")
    for name, directory in DATASETS.items():
        report.append(f"- **{name}:** `{directory}`")
        report.append(f"  - Exists: `{directory.exists()}`")
        report.append(f"  - Candidate tabular files found: `{len(discovered_files.get(name, []))}`")
    report.append("")
    report.append("## File inventory")
    if not files_df.empty:
        report.append(files_df.groupby(["dataset", "inspection_status"], dropna=False).size().to_string())
    report.append("")
    report.append("## Feature counts")
    if not features_df.empty:
        report.append(features_df.groupby("dataset")["feature_name"].nunique().to_string())
        report.append("")
        report.append("### Semantic group counts")
        report.append(features_df.groupby(["dataset", "semantic_group"])["feature_name"].nunique().to_string())
    report.append("")
    report.append(f"## Inspection errors: {len(errors)}")
    for err in errors[:100]:
        report.append(f"- {err['dataset']} | {err['file']} | {err['error']}")
    report.append("")
    report.append("## Output files")
    for name in [
        "source_file_inventory.csv",
        "all_features_long.csv",
        "features_by_dataset.csv",
        "normalized_feature_overlap.csv",
        "feature_name_crosswalk.csv",
        "inspection_errors.json",
        "existing_model_audit_files.json",
    ]:
        report.append(f"- `{name}`")
    (OUTPUT_DIR / "README.md").write_text("\n".join(report), encoding="utf-8")

    print("\n=== FEATURE INVENTORY SUMMARY ===")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Tabular/JSON files inspected: {len(file_rows)}")
    print(f"Feature/file records: {len(feature_rows)}")
    print(f"Inspection errors: {len(errors)}")

    if not features_df.empty:
        print("\nUnique source feature names by dataset:")
        print(features_df.groupby("dataset")["feature_name"].nunique().to_string())
        print("\nSemantic groups:")
        print(features_df.groupby(["dataset", "semantic_group"])["feature_name"].nunique().to_string())

    print("\nFiles created:")
    for name in [
        "README.md",
        "source_file_inventory.csv",
        "all_features_long.csv",
        "features_by_dataset.csv",
        "normalized_feature_overlap.csv",
        "feature_name_crosswalk.csv",
        "inspection_errors.json",
        "existing_model_audit_files.json",
    ]:
        print(f"  {OUTPUT_DIR / name}")

if __name__ == "__main__":
    main()
