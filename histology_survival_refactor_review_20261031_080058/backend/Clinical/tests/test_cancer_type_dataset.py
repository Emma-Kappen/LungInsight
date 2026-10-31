from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pandas as pd
import pytest

_PATH = Path(__file__).resolve().parents[1] / "tools" / "build_cancer_type_dataset.py"
_spec = importlib.util.spec_from_file_location("build_cancer_type_dataset", _PATH)
bcd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bcd)

EXPECTED_FEATURES = [
    "sex_at_baseline", "age_at_diagnosis", "smoking_history", "smoking_pack_years",
    "ecog_score", "stage",
    "prior_treatment_msk_prior_medication", "prior_treatment_tcga_neoadjuvant",
]
METADATA_VALUES = ["gender", "CDE_ID:2200604", "ecog_score", "tobacco_smoking_history", "[Not Available]"]


def test_exact_feature_list_and_no_leaky_fields():
    assert bcd.FEATURES == EXPECTED_FEATURES
    forbidden = ("histolog", "oncotree", "cancer_type", "diagnosis_desc", "survival", "vital", "label", "patient_id", "cohort")
    assert not [f for f in bcd.FEATURES if any(x in f.lower() for x in forbidden)]
    assert "prior_treatment" not in bcd.FEATURES  # source-specific columns are kept distinct


def test_sex_normalization():
    assert [bcd.normalize_sex(v) for v in ["Female", " MALE ", "F", "M"]] == ["female", "male", "female", "male"]
    for v in ["Unknown", "Gender at birth", "gender", "", None, float("nan"), "[Not Available]"]:
        assert bcd.normalize_sex(v) is None


@pytest.mark.parametrize("fn,ok,bad", [
    (bcd.parse_age, [("67", 67.0), (" 45.5 ", 45.5), ("0", 0.0)], ["-3", "121", "abc", "", "[Not Available]", "CDE_ID:2006657", "age_at_initial_pathologic_diagnosis", "1e2"]),
    (bcd.parse_pack_years, [("40", 40.0), ("0", 0.0)], ["-1", "301", "[Not Available]", "number_pack_years_smoked", "CDE_ID:2955385", "20 pack"]),
    (bcd.parse_ecog, [("0", 0.0), ("3", 3.0)], ["5", "1.5", "-1", "[Not Evaluated]", "[Unknown]", "ecog_score", "CDE_ID:88", ""]),
])
def test_numeric_parsing(fn, ok, bad):
    for raw, expected in ok:
        assert fn(raw) == expected
    for raw in bad + [None, float("nan")]:
        assert math.isnan(fn(raw)), raw


def test_msk_smoking_only_verified_categories():
    assert bcd.normalize_msk_smoking("Never") == "never"
    assert bcd.normalize_msk_smoking("Former/Current Smoker") == "former_or_current"
    for v in ["Unknown", "Current", "Former", "", "SMOKING_PREDICTIONS_3_CLASSES", "STRING", None]:
        assert bcd.normalize_msk_smoking(v) is None


@pytest.mark.parametrize("code", ["1", "2", "3", "4", "5", "[Not Available]", "tobacco_smoking_history", "CDE_ID:2181650"])
def test_tcga_smoking_codes_are_missing_until_verified(code):
    assert bcd.TCGA_SMOKING_CODES_VERIFIED is False
    assert bcd.normalize_tcga_smoking(code) is None


def test_stage_normalization_keeps_source_categories():
    assert bcd.normalize_stage("Stage 1-3") == "stage_1_3"
    assert bcd.normalize_stage("Stage 4") == "stage_4"
    assert bcd.normalize_stage("Stage IIIA") == "stage_iiia"
    assert bcd.normalize_stage("Stage IV") == "stage_iv"
    assert bcd.normalize_stage("Stage I") == "stage_i"
    for v in ["[Discrepancy]", "[Not Applicable]", "Unknown", "pathologic_stage", "CDE_ID:3203222", "", "DESCRIPTION: Highest recorded stage"]:
        assert bcd.normalize_stage(v) is None
    assert bcd.normalize_stage("Stage 4") != bcd.normalize_stage("Stage IV")  # not asserted equivalent


def test_prior_treatment_sources_are_not_conflated():
    assert bcd.normalize_msk_prior("Prior medications to MSK") == "yes"
    assert bcd.normalize_msk_prior("No prior medications") == "no"
    assert bcd.normalize_msk_prior("Unknown") is None
    assert bcd.normalize_msk_prior("Yes") is None  # TCGA wording is not accepted as MSK semantics
    assert bcd.normalize_tcga_neoadjuvant("Yes") == "yes"
    assert bcd.normalize_tcga_neoadjuvant("Yes, Pharmaceutical Treatment Prior to Resection") == "yes"
    assert bcd.normalize_tcga_neoadjuvant("No") == "no"
    for v in ["[Discrepancy]", "history_of_neoadjuvant_treatment", "CDE_ID:3382737", "", "Prior medications to MSK"]:
        assert bcd.normalize_tcga_neoadjuvant(v) is None


def _write_tcga(path, rows):
    cols = ["bcr_patient_barcode", "histologic_diagnosis", "gender", "age_at_initial_pathologic_diagnosis",
            "tobacco_smoking_history_indicator", "tobacco_smoking_pack_years_smoked", "ecog_score",
            "ajcc_pathologic_tumor_stage", "history_neoadjuvant_treatment"]
    lines = ["\t".join(cols)] + ["\t".join(r) for r in rows]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def test_tcga_metadata_rows_excluded_and_features_normalized(tmp_path):
    meta1 = ["bcr_patient_barcode", "histologic_diagnosis", "gender", "age_at_initial_pathologic_diagnosis",
             "tobacco_smoking_history", "number_pack_years_smoked", "ecog_score", "pathologic_stage",
             "history_of_neoadjuvant_treatment"]
    meta2 = ["CDE_ID:", "CDE_ID:3081934", "CDE_ID:2200604", "CDE_ID:2006657", "CDE_ID:2181650",
             "CDE_ID:2955385", "CDE_ID:88", "CDE_ID:3203222", "CDE_ID:3382737"]
    good = ["TCGA-AA-0001", "Lung Adenocarcinoma", "FEMALE", "61", "4", "40", "1", "Stage IB", "No"]
    sparse = ["TCGA-AA-0002", "Lung Adenocarcinoma", "MALE", "[Not Available]", "[Unknown]",
              "[Not Available]", "[Not Evaluated]", "[Discrepancy]", "Yes, Pharmaceutical Treatment Prior to Resection"]
    wrong_target = ["TCGA-AA-0003", "Lung Squamous Cell Carcinoma", "MALE", "70", "", "", "", "", ""]
    path = _write_tcga(tmp_path / "p.txt", [meta1, meta2, good, sparse, wrong_target])
    out, qc = bcd.read_tcga(path, "TCGA-LUAD", "LUAD")

    assert out["patient_id"].tolist() == ["TCGA-AA-0001", "TCGA-AA-0002"]
    assert set(qc["patient_id"]) == {"TCGA-AA-0001", "TCGA-AA-0002", "TCGA-AA-0003"}
    for col in bcd.FEATURES:
        assert not out[col].isin(METADATA_VALUES).any()
    r = out.set_index("patient_id")
    assert r.loc["TCGA-AA-0001", "sex_at_baseline"] == "female"
    assert r.loc["TCGA-AA-0001", "age_at_diagnosis"] == 61.0
    assert r.loc["TCGA-AA-0001", "smoking_pack_years"] == 40.0
    assert r.loc["TCGA-AA-0001", "ecog_score"] == 1.0
    assert r.loc["TCGA-AA-0001", "stage"] == "stage_ib"
    assert r.loc["TCGA-AA-0001", "prior_treatment_tcga_neoadjuvant"] == "no"
    assert out["smoking_history"].isna().all()
    assert out["prior_treatment_msk_prior_medication"].isna().all()
    s = r.loc["TCGA-AA-0002"]
    assert s[["age_at_diagnosis", "smoking_pack_years", "ecog_score", "stage"]].isna().all()
    assert s["prior_treatment_tcga_neoadjuvant"] == "yes"


MSK_PATIENT_COLS = ["#Patient Identifier", "Sex", "Current Age", "Stage (Highest Recorded)",
                    "Smoking History (NLP)", "Prior Treatment to MSK (NLP)"]
MSK_SAMPLE_COLS = ["#Sample Identifier", "Patient Identifier", "Cancer Type Detailed", "Oncotree Code"]


def _write_msk(root, patients, samples):
    base = root / "msk_chord_2024"
    base.mkdir(parents=True)
    meta_p = ["#Identifier", "#Sex", "#Age", "DESCRIPTION: Highest recorded stage", "DESCRIPTION: NLP", "DESCRIPTION: Binary"]
    meta_s = ["#Identifier", "#Patient", "#Detailed", "#Code"]
    for name, cols, meta, rows in [("data_clinical_patient.txt", MSK_PATIENT_COLS, meta_p, patients),
                                   ("data_clinical_sample.txt", MSK_SAMPLE_COLS, meta_s, samples)]:
        body = ["\t".join(cols), "\t".join(meta), "\t".join(["STRING"] * len(cols))] + ["\t".join(r) for r in rows]
        (base / name).write_text("\n".join(body), encoding="utf-8")


def test_msk_features_labels_and_conflict_exclusion(tmp_path):
    patients = [
        ["P-0000001", "Female", "67", "Stage 1-3", "Never", "No prior medications"],
        ["P-0000002", "Gender at birth", "abc", "Unknown", "Unknown", "Unknown"],
        ["P-0000003", "Male", "72", "Stage 4", "Former/Current Smoker", "Prior medications to MSK"],
        ["P-0000004", "Male", "70", "Stage 4", "Never", "Unknown"],
    ]
    samples = [
        ["P-0000001-T01-IM3", "P-0000001", "Lung Adenocarcinoma", "LUAD"],
        ["P-0000002-T01-IM3", "P-0000002", "Lung Squamous Cell Carcinoma", "LUSC"],
        ["P-0000003-T01-IM3", "P-0000003", "Lung Adenocarcinoma", "LUSC"],  # conflicting histology/code
        ["P-0000004-T01-IM3", "P-0000004", "Lung Adenocarcinoma", "LUAD"],
        ["P-0000004-T02-IM3", "P-0000004", "Lung Squamous Cell Carcinoma", "LUSC"],  # conflicting labels
    ]
    _write_msk(tmp_path, patients, samples)
    out, excluded = bcd.load_msk(tmp_path)

    assert sorted(out["patient_id"]) == ["P-0000001", "P-0000002"]
    assert set(excluded["patient_id"]) == {"P-0000003", "P-0000004"}
    r = out.set_index("patient_id")
    assert r.loc["P-0000001", "cancer_type"] == "LUAD" and r.loc["P-0000002", "cancer_type"] == "LUSC"
    assert r.loc["P-0000001", ["sex_at_baseline", "smoking_history", "stage", "prior_treatment_msk_prior_medication"]].tolist() \
        == ["female", "never", "stage_1_3", "no"]
    assert r.loc["P-0000001", "age_at_diagnosis"] == 67.0
    m = r.loc["P-0000002"]
    assert m[["sex_at_baseline", "age_at_diagnosis", "smoking_history", "stage", "prior_treatment_msk_prior_medication"]].isna().all()
    for col in ["smoking_pack_years", "ecog_score", "prior_treatment_tcga_neoadjuvant"]:
        assert out[col].isna().all()
    for col in bcd.FEATURES:
        assert not out[col].astype(str).str.contains("DESCRIPTION|STRING").any()


def test_model_matrix_and_audit(tmp_path):
    patients = [["P-0000001", "Female", "67", "Stage 1-3", "Never", "No prior medications"]]
    samples = [["P-0000001-T01-IM3", "P-0000001", "Lung Adenocarcinoma", "LUAD"]]
    _write_msk(tmp_path, patients, samples)
    msk, _ = bcd.load_msk(tmp_path)
    tcga_path = _write_tcga(tmp_path / "t.txt", [["TCGA-AA-0001", "Lung Adenocarcinoma", "MALE", "61", "4", "[Not Available]", "2", "Stage IA", "No"]])
    tcga, _ = bcd.read_tcga(tcga_path, "TCGA-LUAD", "LUAD")
    combined = pd.concat([msk, tcga], ignore_index=True, sort=False)

    matrix = bcd.build_model_ready(combined)
    assert list(matrix.columns) == ["patient_id", "cohort", "cancer_type"] + EXPECTED_FEATURES + ["label_provenance", "sample_count"]
    assert not {"Cancer Type Detailed", "Oncotree Code", "source_record_id"} & set(matrix.columns)

    audit = bcd.feature_audit(combined)
    assert list(audit) == EXPECTED_FEATURES
    for f, entry in audit.items():
        assert {"source_columns", "normalization_rule", "populated_missing_by_cohort", "used_as_predictor",
                "leakage_timing_concerns", "cohort_definition_differences"} <= set(entry)
        for c in entry["populated_missing_by_cohort"].values():
            assert c["populated"] + c["missing"] == 1
    assert audit["smoking_history"]["populated_missing_by_cohort"]["TCGA-LUAD"] == {"populated": 0, "missing": 1}
    assert audit["smoking_history"]["populated_missing_by_cohort"]["MSK-CHORD-2024"] == {"populated": 1, "missing": 0}
    assert audit["smoking_pack_years"]["populated_missing_by_cohort"]["TCGA-LUAD"] == {"populated": 0, "missing": 1}
    assert audit["ecog_score"]["predictor_usable_in_cohorts"] == ["TCGA-LUAD"]
