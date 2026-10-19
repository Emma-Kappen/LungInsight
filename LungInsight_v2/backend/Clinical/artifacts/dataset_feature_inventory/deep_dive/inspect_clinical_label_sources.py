from pathlib import Path
import csv
import json
import re
import traceback
import pandas as pd

PROJECT = Path(r"C:\Users\emkap\Documents\Projects\LungInsight\LungInsight_v2")
DATASETS = Path(r"C:\Users\emkap\Documents\Projects\LungInsight\datasets")
OUT = PROJECT / "backend" / "Clinical" / "artifacts" / "dataset_feature_inventory" / "deep_dive"
OUT.mkdir(parents=True, exist_ok=True)

ERRORS = []
LABEL_TERMS = re.compile(
    r"histolog|diagnos|cancer.?type|oncotree|patholog|morpholog|subtype|"
    r"primary.?site|tumou?r.?site|icd.?o|specimen|sample.?type|tissue.?type",
    re.I
)
PREDICTOR_TERMS = re.compile(
    r"age|gender|sex|smok|tobacco|pack.?year|race|ethnic|ecog|performance|"
    r"comorbid|copd|emphysema|weight|height|bmi",
    re.I
)
LEAKAGE_TERMS = re.compile(
    r"treatment|therapy|drug|radiation|surviv|death|vital|follow.?up|"
    r"progress|recurr|response|days_to|date_of_death|tumou?r.?stage|"
    r"clinical.?stage|pathologic.?stage|mutation|gene|expression|"
    r"barcode|patient.?id|sample.?id|submitter|uuid",
    re.I
)

def decode_line(path, line):
    try:
        return line.decode("utf-8-sig").rstrip("\r\n")
    except UnicodeDecodeError:
        return line.decode("latin-1").rstrip("\r\n")

def detect_header(path):
    raw = path.read_bytes().splitlines()
    candidates = []
    for i, bline in enumerate(raw[:40]):
        line = decode_line(path, bline)
        if "\t" not in line and "," not in line:
            continue
        delim = "\t" if line.count("\t") >= line.count(",") else ","
        try:
            fields = next(csv.reader([line], delimiter=delim))
        except Exception:
            continue
        fields = [x.strip().lstrip("\ufeff") for x in fields]
        joined = " ".join(fields).lower()
        # Prioritize real field-name rows over cBioPortal metadata rows.
        score = len(fields)
        if any(x.upper() in {"PATIENT_ID", "SAMPLE_ID", "CASE_ID"} for x in fields):
            score += 100
        if any(x.lower() in {"bcr_patient_barcode", "histologic_diagnosis"} for x in fields):
            score += 100
        if "description" in joined and len(fields) < 5:
            score -= 100
        candidates.append((score, i, delim, fields))
    if not candidates:
        raise ValueError("Could not identify a delimited header in first 40 lines")
    # Choose the highest-scoring plausible header; prefer later rows on ties.
    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    score, header_idx, delim, fields = candidates[0]
    if len(fields) < 2:
        raise ValueError("Detected header has fewer than two columns")
    return header_idx, delim, fields

def read_table(path):
    idx, delim, header = detect_header(path)
    try:
        df = pd.read_csv(
            path, sep=delim, skiprows=idx, dtype=str, low_memory=False,
            encoding="utf-8-sig", on_bad_lines="warn"
        )
    except UnicodeDecodeError:
        df = pd.read_csv(
            path, sep=delim, skiprows=idx, dtype=str, low_memory=False,
            encoding="latin-1", on_bad_lines="warn"
        )
    df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]
    df = df.loc[:, ~pd.Index(df.columns).duplicated()]
    return df, idx, delim

def clean(v):
    if pd.isna(v):
        return ""
    return str(v).strip()

def find_col(columns, pattern):
    return next((c for c in columns if re.search(pattern, str(c), re.I)), None)

def normalized_counts(series, limit=60):
    s = series.fillna("").astype(str).str.strip()
    s = s[s.ne("")]
    counts = s.value_counts(dropna=False)
    return [{"value": str(k), "count": int(v)} for k, v in counts.head(limit).items()]

def describe_table(dataset, path, df, header_idx):
    rel = str(path.relative_to(DATASETS))
    rows = len(df)
    cols = len(df.columns)
    id_cols = [c for c in df.columns if re.search(
        r"patient.?id|sample.?id|bcr_patient_barcode|bcr_sample_barcode|submitter_id|case_id|uuid|barcode",
        c, re.I
    )]
    label_cols = [c for c in df.columns if LABEL_TERMS.search(c)]
    pred_cols = [c for c in df.columns if PREDICTOR_TERMS.search(c)]
    leakage_cols = [c for c in df.columns if LEAKAGE_TERMS.search(c)]

    schema_rows = []
    for c in df.columns:
        s = df[c].astype("string").str.strip()
        nonempty = s.notna() & s.ne("")
        vals = s[nonempty]
        examples = vals.drop_duplicates().head(8).tolist()
        schema_rows.append({
            "dataset": dataset,
            "source_file": rel,
            "column": c,
            "row_count": rows,
            "nonempty_count": int(nonempty.sum()),
            "missing_pct": round(100 * (1 - nonempty.mean()), 2) if rows else 100.0,
            "unique_nonempty": int(vals.nunique()) if len(vals) else 0,
            "example_values": json.dumps(examples, ensure_ascii=False),
            "possible_label": bool(LABEL_TERMS.search(c)),
            "possible_predictor": bool(PREDICTOR_TERMS.search(c)),
            "possible_leakage_or_identifier": bool(LEAKAGE_TERMS.search(c)),
        })

    key_summary = {
        "dataset": dataset,
        "source_file": rel,
        "header_line_1based": header_idx + 1,
        "rows": rows,
        "columns": cols,
        "identifier_columns": id_cols,
        "label_candidate_columns": label_cols,
        "demographic_predictor_candidates": pred_cols,
        "leakage_or_identifier_candidates": leakage_cols,
    }
    return schema_rows, key_summary

def choose_source_files():
    selected = []
    msk = DATASETS / "msk_chord_2024"
    for rel in [
        "data_clinical_patient.txt",
        "data_clinical_sample.txt",
        "data_mutations.txt",
        "data_timeline_diagnosis.txt",
        "msk_chord_out/nsclc_clinical_patient.csv",
        "msk_chord_out/nsclc_clinical_sample.csv",
        "msk_chord_out/nsclc_patient_level_features.csv",
        "msk_chord_out/nsclc_mutations.csv",
        "msk_chord_out/nsclc_data_timeline_diagnosis.txt",
    ]:
        p = msk / rel
        if p.exists():
            selected.append(("MSK-CHORD-2024", p))

    for cohort, folder, suffix in [
        ("TCGA-LUAD", DATASETS / "TCGA-LUAD", "luad"),
        ("TCGA-LUSC", DATASETS / "TCGA-LUSC", "lusc"),
    ]:
        if not folder.exists():
            continue
        # Patient-level clinical table first; include selected tumor sample and follow-up tables.
        pats = list(folder.rglob(f"*clinical_patient_{suffix}.txt"))
        selected.extend((cohort, p) for p in pats)
        selected.extend(
            (cohort, p) for p in folder.rglob(f"*ssf_tumor_samples_{suffix}.txt")
        )
        selected.extend(
            (cohort, p) for p in folder.rglob(f"*clinical_follow_up*_{suffix}.txt")
        )
    return selected

all_schema = []
summaries = []
label_rows = []
value_rows = []
pair_rows = []
patient_sets = {}
table_columns = {}

for dataset, path in choose_source_files():
    try:
        df, header_idx, delim = read_table(path)
        schema, summary = describe_table(dataset, path, df, header_idx)
        all_schema.extend(schema)
        summaries.append(summary)
        table_columns[(dataset, str(path))] = list(df.columns)

        rel = str(path.relative_to(DATASETS))
        print(f"\n[{dataset}] {rel}")
        print(f"  Parsed rows={len(df):,}; columns={len(df.columns)}; header line={header_idx + 1}")
        print("  Columns:", " | ".join(df.columns[:80]))

        # Collect candidate label distributions and demographics/potential confounders.
        relevant_cols = [
            c for c in df.columns
            if LABEL_TERMS.search(c) or PREDICTOR_TERMS.search(c)
        ]
        for c in relevant_cols:
            label_rows.append({
                "dataset": dataset,
                "source_file": rel,
                "column": c,
                "nonempty": int(df[c].fillna("").astype(str).str.strip().ne("").sum()),
                "unique_nonempty": int(df[c].fillna("").astype(str).str.strip().replace("", pd.NA).nunique()),
                "value_counts_top60": json.dumps(normalized_counts(df[c]), ensure_ascii=False),
            })
            for item in normalized_counts(df[c], limit=30):
                value_rows.append({
                    "dataset": dataset,
                    "source_file": rel,
                    "column": c,
                    "value": item["value"],
                    "count": item["count"],
                })

        # Track patient IDs to compare patient-level tables and avoid assuming row=patient.
        id_col = find_col(df.columns, r"^(PATIENT_ID|bcr_patient_barcode|patient_id|submitter_id)$")
        if id_col:
            ids = set(df[id_col].dropna().astype(str).str.strip())
            ids.discard("")
            patient_sets[(dataset, rel)] = ids
            pair_rows.append({
                "dataset": dataset,
                "source_file": rel,
                "id_column": id_col,
                "rows": len(df),
                "unique_ids": len(ids),
                "duplicate_id_rows": int(df[id_col].notna().sum() - df[id_col].astype(str).str.strip().replace("", pd.NA).nunique()),
            })
    except Exception as e:
        ERRORS.append({
            "dataset": dataset,
            "path": str(path),
            "error": repr(e),
            "traceback": traceback.format_exc(),
        })
        print(f"\nERROR [{dataset}] {path}: {e}")

pd.DataFrame(all_schema).to_csv(OUT / "source_table_column_dictionary.csv", index=False)
pd.DataFrame(summaries).to_json(OUT / "source_table_parse_summary.json", orient="records", indent=2)
pd.DataFrame(label_rows).to_csv(OUT / "candidate_label_predictor_distributions.csv", index=False)
pd.DataFrame(value_rows).to_csv(OUT / "candidate_value_counts.csv", index=False)
pd.DataFrame(pair_rows).to_csv(OUT / "source_table_identifier_audit.csv", index=False)
(OUT / "source_table_inspection_errors.json").write_text(
    json.dumps(ERRORS, indent=2), encoding="utf-8"
)

# Compact overlap report of column names among the parsed source tables.
all_names = {}
for rec in all_schema:
    key = (rec["dataset"], rec["source_file"])
    all_names.setdefault(key, set()).add(rec["column"].strip().lower())

overlap = []
for cohort in ["MSK-CHORD-2024", "TCGA-LUAD", "TCGA-LUSC"]:
    cohort_sets = [v for (ds, _), v in all_names.items() if ds == cohort]
    names = set().union(*cohort_sets) if cohort_sets else set()
    overlap.append({
        "dataset": cohort,
        "parsed_source_tables": len(cohort_sets),
        "unique_column_names_across_selected_tables": len(names),
        "names": sorted(names),
    })
(OUT / "selected_source_column_name_sets.json").write_text(
    json.dumps(overlap, indent=2), encoding="utf-8"
)

print("\n========== PARSE SUMMARY ==========")
for x in summaries:
    print(f'{x["dataset"]}: {x["source_file"]} | rows={x["rows"]:,} | columns={x["columns"]}')
print("\n========== OUTPUT FILES ==========")
for name in [
    "source_table_column_dictionary.csv",
    "source_table_parse_summary.json",
    "candidate_label_predictor_distributions.csv",
    "candidate_value_counts.csv",
    "source_table_identifier_audit.csv",
    "source_table_inspection_errors.json",
    "selected_source_column_name_sets.json",
]:
    p = OUT / name
    print(f"{p.name}: {p.stat().st_size:,} bytes" if p.exists() else f"{p.name}: MISSING")
print(f"\nErrors: {len(ERRORS)}")
