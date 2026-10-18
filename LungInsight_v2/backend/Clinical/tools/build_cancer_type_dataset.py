#!/usr/bin/env python
"""Build patient-level clinical-only LUAD/LUSC label datasets.

Target: cancer_type in {LUAD, LUSC}. Uses MSK-CHORD-2024 and TCGA-LUAD/LUSC.
No CT images, imaging-derived features, or genomic data are read.
Keeps cohorts separate in dedicated outputs; the combined export is descriptive
Keeps cohorts separate in dedicated outputs; the combined export is descriptive
only because cohort membership may be confounded with target (TCGA-LUAD is all
LUAD, TCGA-LUSC is all LUSC). Features are normalized, missing values are kept,
and nothing is imputed; see dataset_audit.json["feature_audit"] for per-feature
sources, rules, missingness by cohort and leakage/timing notes.
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
import pandas as pd

DEFAULT_ROOT = Path(r"C:\Users\emkap\Documents\Projects\LungInsight\datasets")
DEFAULT_OUT = Path(r"C:\Users\emkap\Documents\Projects\LungInsight\LungInsight_v2\backend\Clinical\artifacts\cancer_type_dataset")
TARGET = "cancer_type"
# Exact predictor list. Missing values are kept as-is (no imputation here); any
# imputation must be fitted on training data only. Prior treatment is kept as two
# source-specific columns because the MSK and TCGA definitions are not equivalent.
FEATURES = [
    "sex_at_baseline", "age_at_diagnosis", "smoking_history", "smoking_pack_years",
    "ecog_score", "stage",
    "prior_treatment_msk_prior_medication", "prior_treatment_tcga_neoadjuvant",
]
NUMERIC_FEATURES = ["age_at_diagnosis", "smoking_pack_years", "ecog_score"]
TCGA_COHORTS = {"TCGA-LUAD": "LUAD", "TCGA-LUSC": "LUSC"}
MSK_COHORT = "MSK-CHORD-2024"
UNAVAILABLE_MARKERS = {
    "", "[NOT AVAILABLE]", "[UNKNOWN]", "[NOT EVALUATED]", "[NOT APPLICABLE]", "[DISCREPANCY]",
}
AGE_RANGE, PACK_YEARS_RANGE, ECOG_RANGE = (0, 120), (0, 300), (0, 4)
# TCGA tobacco_smoking_history_indicator codes 1-5 are withheld until their meanings are
# verified against the source data dictionary; their order must not be assumed ordinal.
TCGA_SMOKING_CODES_VERIFIED = False

FEATURE_SPECS = {
    "sex_at_baseline": dict(
        msk=["Sex"], tcga=["gender"],
        rule="Female/Male (case-insensitive) -> female/male; every other value -> missing.",
        leakage="Low. Recorded demographic.",
        difference="MSK 'Sex' and TCGA 'gender' are both recorded sex; no known definition difference."),
    "age_at_diagnosis": dict(
        msk=["Current Age"], tcga=["age_at_initial_pathologic_diagnosis"],
        rule="Plain numeric in [0, 120]; markers, metadata and non-numeric text -> missing.",
        leakage="Low for TCGA. MSK 'Current Age' is age at the data snapshot, not at diagnosis, "
                "and is capped at 89 in the source.",
        difference="NOT equivalent: MSK is current age (>= age at diagnosis); TCGA is age at initial diagnosis."),
    "smoking_history": dict(
        msk=["Smoking History (NLP)"], tcga=["tobacco_smoking_history_indicator"],
        rule="MSK: Never -> never; Former/Current Smoker -> former_or_current; Unknown -> missing. "
             "TCGA codes 1-5 -> missing until verified against the data dictionary.",
        leakage="Moderate for MSK: NLP-derived from clinical notes of unspecified timing.",
        difference="TCGA is unpopulated by design; no more specific status is inferred."),
    "smoking_pack_years": dict(
        msk=[], tcga=["tobacco_smoking_pack_years_smoked"],
        rule="Plain numeric in [0, 300]; markers, metadata and non-numeric text -> missing.",
        leakage="Low; self-reported history.",
        difference="Not available in the MSK patient table."),
    "ecog_score": dict(
        msk=[], tcga=["ecog_score"],
        rule="Integer in [0, 4]; markers, metadata and non-numeric text -> missing.",
        leakage="Timing relative to treatment is not documented in the TCGA table.",
        difference="Not available in the MSK patient table (performance status is a separate timeline)."),
    "stage": dict(
        msk=["Stage (Highest Recorded)"], tcga=["ajcc_pathologic_tumor_stage"],
        rule="MSK: 'Stage 1-3' -> stage_1_3, 'Stage 4' -> stage_4. TCGA: 'Stage IA'.. -> stage_ia etc. "
             "Anything else -> missing. TCGA clinical stage is not used (mostly [Not Applicable]).",
        leakage="Post-diagnostic; pathologic stage requires resection.",
        difference="NOT equivalent: MSK is highest recorded stage binned to 1-3/4; TCGA is AJCC pathologic "
                   "stage. Category sets are disjoint, so the column is not comparable across cohorts."),
    "prior_treatment_msk_prior_medication": dict(
        msk=["Prior Treatment to MSK (NLP)"], tcga=[],
        rule="'Prior medications to MSK' -> yes; 'No prior medications' -> no; Unknown -> missing.",
        leakage="NLP-derived from initial consult notes; timing relative to diagnosis is limited.",
        difference="Anti-cancer medication received before MSK care; not equivalent to TCGA neoadjuvant treatment."),
    "prior_treatment_tcga_neoadjuvant": dict(
        msk=[], tcga=["history_neoadjuvant_treatment"],
        rule="Yes (including 'Yes, ...' variants) -> yes; No -> no; markers -> missing.",
        leakage="Describes treatment before resection.",
        difference="Neoadjuvant treatment before surgery; not equivalent to MSK prior medications."),
}

def norm(v):
    if v is None or pd.isna(v): return ""
    return re.sub(r"\s+", " ", str(v).strip()).upper()

def normalize_sex(v):
    v = norm(v)
    if v in {"FEMALE", "F", "WOMAN"}: return "female"
    if v in {"MALE", "M", "MAN"}: return "male"
    return None

def parse_numeric(v, lo, hi, integer=False):
    s = norm(v)
    if s in UNAVAILABLE_MARKERS or not re.fullmatch(r"\d+(\.\d+)?", s): return float("nan")
    x = float(s)
    if not lo <= x <= hi or (integer and x != int(x)): return float("nan")
    return x

def parse_age(v): return parse_numeric(v, *AGE_RANGE)
def parse_pack_years(v): return parse_numeric(v, *PACK_YEARS_RANGE)
def parse_ecog(v): return parse_numeric(v, *ECOG_RANGE, integer=True)

def normalize_msk_smoking(v):
    return {"NEVER": "never", "FORMER/CURRENT SMOKER": "former_or_current"}.get(norm(v))

def normalize_tcga_smoking(v):
    if not TCGA_SMOKING_CODES_VERIFIED: return None
    raise NotImplementedError("Add verified code mappings before enabling TCGA smoking history.")

def normalize_stage(v):
    s = norm(v)
    if s in {"STAGE 1-3", "STAGE 4"}: return "stage_" + s[6:].replace("-", "_")
    m = re.fullmatch(r"STAGE (I{1,3}|IV)([ABC])?", s)
    return "stage_" + (m.group(1) + (m.group(2) or "")).lower() if m else None

def normalize_msk_prior(v):
    return {"PRIOR MEDICATIONS TO MSK": "yes", "NO PRIOR MEDICATIONS": "no"}.get(norm(v))

def normalize_tcga_neoadjuvant(v):
    s = norm(v)
    return "yes" if s == "YES" or s.startswith("YES,") else "no" if s == "NO" else None

def mapped(df, col, fn):
    if col is None or col not in df.columns:
        return pd.Series([fn(None)] * len(df), index=df.index)
    return df[col].map(fn)

def build_model_ready(combined):
    return combined[["patient_id", "cohort", TARGET] + FEATURES + ["label_provenance", "sample_count"]].copy()

def feature_audit(combined):
    cohorts = list(dict.fromkeys(combined["cohort"]))
    out = {}
    for f in FEATURES:
        spec = FEATURE_SPECS[f]
        per = {c: {"populated": int(combined.loc[combined["cohort"] == c, f].notna().sum()),
                   "missing": int(combined.loc[combined["cohort"] == c, f].isna().sum())} for c in cohorts}
        out[f] = {
            "source_columns": {"msk": spec["msk"], "tcga": spec["tcga"]},
            "normalization_rule": spec["rule"],
            "numeric": f in NUMERIC_FEATURES,
            "used_as_predictor": True,
            "predictor_usable_in_cohorts": [c for c, n in per.items() if n["populated"] > 0],
            "populated_missing_by_cohort": per,
            "leakage_timing_concerns": spec["leakage"],
            "cohort_definition_differences": spec["difference"],
        }
    return out

def choose_col(df, *names):
    return next((n for n in names if n in df.columns), None)

def read_cbio(path, required):
    lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    idx = next((i for i, line in enumerate(lines)
                if all(x in line.split("\t") for x in required)), None)
    if idx is None: raise ValueError(f"Could not find header {required} in {path}")
    df = pd.read_csv(
        path, sep="\t", skiprows=idx, dtype=str,
        keep_default_na=False, encoding="utf-8-sig"
    )

    # Filter metadata/repeated-header rows in the MSK clinical tables.
    if "#Patient Identifier" in df.columns:
        ids = df["#Patient Identifier"].astype(str).str.strip()
        df = df.loc[ids.str.fullmatch(r"P-\d+", na=False)].copy()

    if "#Sample Identifier" in df.columns:
        ids = df["#Sample Identifier"].astype(str).str.strip()
        patient_ids = df["Patient Identifier"].astype(str).str.strip()
        valid = (
            ids.str.fullmatch(r"P-\d+-T\d+-.*", na=False)
            & patient_ids.str.fullmatch(r"P-\d+", na=False)
        )
        df = df.loc[valid].copy()

    return df

def read_msk(root):
    base = root / "msk_chord_2024"
    pp, sp = base / "data_clinical_patient.txt", base / "data_clinical_sample.txt"
    if pp.exists() and sp.exists():
        return read_cbio(pp, ("#Patient Identifier", "Sex")), read_cbio(sp, ("#Sample Identifier", "Patient Identifier"))
    pp, sp = base / "msk_chord_out" / "sclc_clinical_patient.csv", base / "msk_chord_out" / "sclc_clinical_sample.csv"
    if not (pp.exists() and sp.exists()):
        raise FileNotFoundError(f"MSK patient/sample clinical files not found under {base}")
    p = pd.read_csv(pp, dtype=str, keep_default_na=False)
    s = pd.read_csv(sp, dtype=str, keep_default_na=False)

    p = p.rename(columns={
        "PATIENT_ID": "#Patient Identifier",
        "GENDER": "Sex",
    })
    s = s.rename(columns={
        "SAMPLE_ID": "#Sample Identifier",
        "PATIENT_ID": "Patient Identifier",
        "CANCER_TYPE": "Cancer Type",
        "CANCER_TYPE_DETAILED": "Cancer Type Detailed",
        "ONCOTREE_CODE": "Oncotree Code",
        "ICDO_HISTOLOGY": "ICD-O Histology",
    })

    patient_col = choose_col(p, "#Patient Identifier")
    sample_col = choose_col(s, "#Sample Identifier")
    sample_patient_col = choose_col(s, "Patient Identifier")

    if not all((patient_col, sample_col, sample_patient_col)):
        raise ValueError(
            "MSK CSV files are missing required patient/sample ID columns."
        )

    p = p[
        p[patient_col].astype(str).str.strip().str.fullmatch(
            r"P-\\d+", na=False
        )
    ].copy()

    s = s[
        s[sample_col].astype(str).str.strip().str.fullmatch(
            r"P-\\d+-T\\d+-.*", na=False
        )
        & s[sample_patient_col].astype(str).str.strip().str.fullmatch(
            r"P-\\d+", na=False
        )
    ].copy()

    print(
        f"MSK rows after ID validation: "
        f"{len(p)} patients, {len(s)} samples"
    )
    return p, s

def label_msk(row):
    # Accept only exact, lung-specific LUAD/LUSC categories or matching OncoTree codes.
    detailed = norm(row.get("Cancer Type Detailed", "")).upper()
    code = norm(row.get("Oncotree Code", "")).upper()

    detailed_map = {
        "LUNG ADENOCARCINOMA": "LUAD",
        "ADENOCARCINOMA OF THE LUNG": "LUAD",
        "LUNG SQUAMOUS CELL CARCINOMA": "LUSC",
        "SQUAMOUS CELL CARCINOMA OF THE LUNG": "LUSC",
    }
    code_map = {"LUAD": "LUAD", "LUSC": "LUSC"}

    label_from_detail = detailed_map.get(detailed, "")
    label_from_code = code_map.get(code, "")

    if label_from_detail and label_from_code and label_from_detail != label_from_code:
        return "", "conflicting_histology_and_code"
    if label_from_detail:
        return label_from_detail, "recorded_sample_histology"
    if label_from_code:
        return label_from_code, "oncotree_code"
    return "", "unresolved_or_non_target"

def load_msk(root):
    patients, samples = read_msk(root)
    pidp = choose_col(patients, "#Patient Identifier", "PATIENT_ID", "Patient Identifier")
    pids = choose_col(samples, "Patient Identifier", "PATIENT_ID")
    sexc = choose_col(patients, "Sex", "GENDER", "Gender")
    if not pidp or not pids: raise ValueError("MSK patient/sample identifier column missing")
    if patients[pidp].duplicated().any(): raise ValueError("Duplicate MSK patient identifiers")
    lab = samples.apply(label_msk, axis=1, result_type="expand")
    samples = samples.copy()
    samples["_target"], samples["_provenance"] = lab[0], lab[1]
    rows, excluded = [], []
    for pid, g in samples.groupby(pids, dropna=False):
        labels = sorted(set(x for x in g["_target"] if x))
        if len(labels) != 1:
            excluded.append({"patient_id":str(pid),"reason":"unresolved_label" if not labels else "conflicting_labels",
                             "resolved_labels":"|".join(labels),"sample_count":len(g)})
            continue
        supporting = sorted(set(
            str(v) for v in g.loc[g["_target"].astype(str).eq(labels[0]), "_provenance"]
            if str(v).strip() not in {"", "unresolved_or_non_target"}
        ))
        rows.append({"patient_id":str(pid),TARGET:labels[0],
                     "label_provenance":";".join(supporting),
                     "sample_count":len(g)})
    labels = pd.DataFrame(rows)
    p = patients.copy()
    feats = pd.DataFrame({"patient_id": p[pidp]}, index=p.index)
    feats["sex_at_baseline"] = mapped(p, sexc, normalize_sex)
    feats["age_at_diagnosis"] = mapped(p, "Current Age", parse_age)
    feats["smoking_history"] = mapped(p, "Smoking History (NLP)", normalize_msk_smoking)
    feats["smoking_pack_years"] = mapped(p, None, parse_pack_years)
    feats["ecog_score"] = mapped(p, None, parse_ecog)
    feats["stage"] = mapped(p, "Stage (Highest Recorded)", normalize_stage)
    feats["prior_treatment_msk_prior_medication"] = mapped(p, "Prior Treatment to MSK (NLP)", normalize_msk_prior)
    feats["prior_treatment_tcga_neoadjuvant"] = mapped(p, None, normalize_tcga_neoadjuvant)
    out = labels.merge(feats, on="patient_id", how="left", validate="one_to_one")
    out.insert(1, "cohort", MSK_COHORT)
    out["source_record_id"] = out["patient_id"]
    return out, pd.DataFrame(excluded)

def read_tcga(path, cohort, target):
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, encoding="utf-8-sig")
    pid = choose_col(df, "bcr_patient_barcode", "patient_id", "submitter_id")
    hist = [c for c in df.columns if c == "histologic_diagnosis" or c.startswith("histologic_diagnosis.")]
    if not pid or not hist: raise ValueError(f"Patient ID or histologic_diagnosis missing in {path}")
    # Discard metadata rows and duplicate patient records; duplicate labels are not silently selected.
    df = df[df[pid].astype(str).str.fullmatch(r"TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}", na=False)].copy()
    dup_ids = set(df.loc[df[pid].duplicated(False), pid])
    df = df[~df[pid].isin(dup_ids)].copy()
    broad = df[hist[0]].map(norm)
    accepted = ({"LUNG ADENOCARCINOMA", "ADENOCARCINOMA OF THE LUNG"} if target == "LUAD"
                else {"LUNG SQUAMOUS CELL CARCINOMA", "SQUAMOUS CELL CARCINOMA OF THE LUNG"})
    valid = broad.isin(accepted)
    sexc = choose_col(df, "gender", "sex", "sex_at_birth")
    kept = df.loc[valid]
    out = pd.DataFrame({
        "patient_id":kept[pid].astype(str).values,
        TARGET:target, "cohort":cohort,
        "label_provenance":"TCGA_recorded_broad_histology",
        "sample_count":1,
        "sex_at_baseline":mapped(kept, sexc, normalize_sex).values,
        "age_at_diagnosis":mapped(kept, "age_at_initial_pathologic_diagnosis", parse_age).values,
        "smoking_history":mapped(kept, "tobacco_smoking_history_indicator", normalize_tcga_smoking).values,
        "smoking_pack_years":mapped(kept, "tobacco_smoking_pack_years_smoked", parse_pack_years).values,
        "ecog_score":mapped(kept, "ecog_score", parse_ecog).values,
        "stage":mapped(kept, "ajcc_pathologic_tumor_stage", normalize_stage).values,
        "prior_treatment_msk_prior_medication":mapped(kept, None, normalize_msk_prior).values,
        "prior_treatment_tcga_neoadjuvant":mapped(kept, "history_neoadjuvant_treatment", normalize_tcga_neoadjuvant).values,
    })
    out["source_record_id"] = out["patient_id"]
    qc = df[[pid] + hist].copy().rename(columns={pid:"patient_id"})
    qc["expected_target"] = target
    qc["broad_histology_matches_expected"] = broad.isin(accepted).values
    return out, qc

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets-root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    root, outdir = args.datasets_root.resolve(), args.output_dir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    msk, excluded = load_msk(root)
    tcga, qc_frames = [], []
    for cohort, target in TCGA_COHORTS.items():
        folder = root / cohort
        files = sorted(folder.rglob("nationwidechildrens.org_clinical_patient_*.txt"))
        if not files: raise FileNotFoundError(f"TCGA clinical patient file not found under {folder}")
        frame, qc = read_tcga(files[0], cohort, target)
        tcga.append(frame); qc_frames.append(qc)
    tcga = pd.concat(tcga, ignore_index=True)
    combined = pd.concat([msk, tcga], ignore_index=True, sort=False)
    model_ready = build_model_ready(combined)
    counts = combined.groupby(["cohort",TARGET], dropna=False).size().rename("patients").reset_index()
    audit = {
        "target":TARGET, "target_values":["LUAD","LUSC"], "candidate_model_features":FEATURES,
        "imaging_data_used":False, "genomic_data_used":False,
        "warning":"Combined file is descriptive only. Do not pool cohorts for fitting without a design addressing cohort-target confounding.",
        "pooled_training_suitable":False,
        "pooled_training_note":"TCGA-LUAD contains only LUAD and TCGA-LUSC only LUSC, so cohort membership is confounded with the target in the TCGA export.",
        "imputation_policy":"None applied. Missing values are retained; any imputation must be fitted on training data only.",
        "excluded_from_predictors":["patient_id","cohort","cancer_type","label_provenance","sample_count","source_record_id","raw source fields","histology","OncoTree code","diagnosis description","survival","vital status","treatment outcome"],
        "feature_audit":feature_audit(combined),
        "counts_by_cohort_and_target":counts.to_dict("records"),
        "missing_sex_count":int(model_ready["sex_at_baseline"].isna().sum()),
        "msk_excluded_or_conflicting_patient_groups":len(excluded),
        "tcga_label_qc_rows":sum(len(x) for x in qc_frames)
    }
    combined.to_csv(outdir/"cancer_type_combined_descriptive.csv", index=False)
    model_ready.to_csv(outdir/"cancer_type_model_ready_candidate.csv", index=False)
    msk.to_csv(outdir/"msk_cancer_type_labeled.csv", index=False)
    tcga.to_csv(outdir/"tcga_cancer_type_labeled_separate.csv", index=False)
    excluded.to_csv(outdir/"msk_excluded_or_conflicting_labels.csv", index=False)
    pd.concat(qc_frames, ignore_index=True, sort=False).to_csv(outdir/"tcga_histology_label_qc.csv", index=False)
    counts.to_csv(outdir/"counts_by_cohort_and_target.csv", index=False)
    (outdir/"dataset_audit.json").write_text(json.dumps(audit,indent=2),encoding="utf-8")
    print("Output directory:", outdir)
    print(counts.to_string(index=False))
    print("MSK excluded/conflicting groups:",len(excluded))
    print("Candidate model features:", FEATURES)
    print("WARNING: combined export is descriptive only; cohorts are not automatically suitable for pooled training.")

if __name__ == "__main__":
    main()


