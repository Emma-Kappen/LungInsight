from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from LungInsight_v2.backend.Clinical.harmonization import (
    ClinicalValidationError,
    normalize_performance_status,
    validate_clinician_inputs,
)
from LungInsight_v2.backend.Clinical.api import create_app
from LungInsight_v2.backend.Clinical.imaging import ImagingEvidenceError, validate_imaging_evidence
from LungInsight_v2.backend.Clinical.ingestion import (
    load_tcga_survival_cohort,
    msk_lung_patient_labels,
    parse_tcga_patient_xml,
)
from LungInsight_v2.backend.Clinical.inference import ClinicalInferenceError, run_inference
from LungInsight_v2.backend.Clinical.training import (
    ModelEligibilityError,
    _grouped_three_way_split,
    _supported_horizons,
    fit_cancer_type_model,
    fit_survival_model,
    evaluate_cross_cohort_survival,
    source_target_shortcut_report,
)


def _valid_imaging_contract(patient_id: str = "synthetic-patient") -> dict:
    return {
        "contract_version": "1.0.0",
        "patient_id": patient_id,
        "study_id": "synthetic-study",
        "study_uid": "synthetic-study-uid",
        "series_uid": "synthetic-series-uid",
        "coordinate_metadata": {
            "array_axes": ["z", "y", "x"],
            "coordinate_system": "DICOM_LPS",
            "origin_mm": [0, 0, 0],
            "spacing_zyx_mm": [1, 1, 1],
            "direction_lps": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        },
        "nodules": [
            {
                "stable_nodule_id": "synthetic-nodule",
                "detector": {
                    "candidate_id": "candidate-1",
                    "source": "LoG",
                    "source_members": [],
                    "vitdet_score": None,
                    "log_score": 0.99,
                },
                "bbox_voxel": {"start_zyx": [0, 0, 0], "end_zyx_exclusive": [1, 1, 1]},
                "bbox_lps_mm": {"min_lps_mm": [0, 0, 0], "max_lps_mm": [1, 1, 1]},
                "centroid_voxel": [0.5, 0.5, 0.5],
                "centroid_lps_mm": [0.5, 0.5, 0.5],
                "diameter_mm": 12.5,
                "volume_mm3": 100.0,
                "anatomical_location": {},
                "attributes": {
                    key: {}
                    for key in (
                        "subtlety",
                        "internal_structure",
                        "calcification",
                        "sphericity",
                        "margin",
                        "lobulation",
                        "spiculation",
                        "texture",
                    )
                },
                "malignancy": {"ranking_score": 0.99, "rank": 1},
                "uncertainty": {},
                "quality_flags": [],
                "model_versions": {"stage04": "synthetic-v1"},
                "gradcam_refs": {},
                "shap_refs": {},
                "artifact_refs": {},
            }
        ],
    }


def _imaging_evidence(patient_id: str = "synthetic-patient", baseline: str = "2026-01-01T12:00:00+00:00") -> dict:
    return {
        "contract": _valid_imaging_contract(patient_id),
        "source_timestamp": baseline,
        "baseline_at": baseline,
        "gate_status": "passed",
        "imaging_pipeline_version": "synthetic-imaging-v1",
        "imaging_model_versions": {"stage06": "synthetic-v1"},
    }


def _request() -> dict:
    baseline = "2026-01-01T12:00:00+00:00"
    return {
        "schema_version": "1.0.0",
        "patient_id": "synthetic-patient",
        "baseline_at": baseline,
        "baseline_definition": "initial_pathologic_diagnosis",
        "clinician_inputs": {
            "age_at_baseline_years": 64,
            "sex_at_baseline": "female",
            "stage_at_baseline": "Stage IIA",
            "kps": None,
            "ecog": 1,
        },
        "imaging_evidence": _imaging_evidence(baseline=baseline),
    }


def test_ecog_to_kps_is_interval_and_direct_kps_wins() -> None:
    derived = normalize_performance_status(ecog=1)
    assert derived.kps_value is None
    assert derived.kps_category == "70-80"
    assert derived.provenance == "approximate_ecog_interval"
    direct = normalize_performance_status(kps=60, ecog=1)
    assert direct.kps_value == 60
    assert direct.kps_category == "60"
    assert direct.discrepancy is True
    missing = normalize_performance_status()
    assert missing.kps_category is None
    assert missing.provenance == "missing"


@pytest.mark.parametrize("value", [-1, 101, 70.5, True])
def test_kps_rejects_invalid_values(value: object) -> None:
    with pytest.raises(ClinicalValidationError):
        normalize_performance_status(kps=value)


def test_clinician_input_ranges_and_whitespace_validation() -> None:
    with pytest.raises(ClinicalValidationError):
        validate_clinician_inputs({"age_at_baseline_years": 121})
    with pytest.raises(ClinicalValidationError):
        validate_clinician_inputs({"stage_at_baseline": "   "})


def test_msk_only_assigns_explicit_histology_and_flags_conflicts() -> None:
    samples = pd.DataFrame([
        {"PATIENT_ID": "p1", "SAMPLE_ID": "s1", "CANCER_TYPE": "Non-Small Cell Lung Cancer",
         "CANCER_TYPE_DETAILED": "Lung Adenocarcinoma", "ONCOTREE_CODE": "LUAD"},
        {"PATIENT_ID": "p2", "SAMPLE_ID": "s2", "CANCER_TYPE": "Non-Small Cell Lung Cancer",
         "CANCER_TYPE_DETAILED": "Non-Small Cell Lung Cancer", "ONCOTREE_CODE": "NSCLC"},
        {"PATIENT_ID": "p3", "SAMPLE_ID": "s3", "CANCER_TYPE": "Non-Small Cell Lung Cancer",
         "CANCER_TYPE_DETAILED": "Lung Adenocarcinoma", "ONCOTREE_CODE": "LUAD"},
        {"PATIENT_ID": "p3", "SAMPLE_ID": "s4", "CANCER_TYPE": "Non-Small Cell Lung Cancer",
         "CANCER_TYPE_DETAILED": "Lung Squamous Cell Carcinoma", "ONCOTREE_CODE": "LUSC"},
        {"PATIENT_ID": "p4", "SAMPLE_ID": "s5", "CANCER_TYPE": "Non-Small Cell Lung Cancer",
         "CANCER_TYPE_DETAILED": "Lung Adenocarcinoma", "ONCOTREE_CODE": "LUSC"},
    ])
    result = msk_lung_patient_labels(samples).set_index("patient_id")
    assert result.loc["p1", "histology"] == "LUAD"
    assert result.loc["p2", "label_status"] == "ambiguous_or_other"
    assert result.loc["p3", "label_status"] == "conflicting"
    assert result.loc["p4", "label_status"] == "conflicting"


def test_tcga_patient_xml_right_censoring_and_followup_only(tmp_path) -> None:
    patient_path = tmp_path / "patient.xml"
    patient_path.write_text(
        """<tcga_bcr><patient>
        <bcr_patient_barcode>SYNTHETIC</bcr_patient_barcode>
        <vital_status>Alive</vital_status>
        <days_to_last_followup></days_to_last_followup>
        <days_to_last_known_alive>500</days_to_last_known_alive>
        <age_at_initial_pathologic_diagnosis>64</age_at_initial_pathologic_diagnosis>
        <gender>FEMALE</gender>
        <stage_event><pathologic_stage>Stage IIA</pathologic_stage></stage_event>
        <histological_type>Lung Adenocarcinoma- Not Otherwise Specified (NOS)</histological_type>
        </patient></tcga_bcr>""",
        encoding="utf-8",
    )
    parsed = parse_tcga_patient_xml(patient_path, "TCGA-LUAD")
    assert parsed is not None
    assert parsed["overall_survival_event"] is False
    assert parsed["overall_survival_time_days"] == 500
    assert parsed["time_source"] == "days_to_last_known_alive"
    assert parsed["sex_at_baseline"] == "female"
    followup_path = tmp_path / "followup.xml"
    followup_path.write_text(
        "<tcga_bcr><patient><bcr_patient_barcode>SYNTHETIC</bcr_patient_barcode>"
        "<follow_up><vital_status>Dead</vital_status></follow_up></patient></tcga_bcr>",
        encoding="utf-8",
    )
    assert parse_tcga_patient_xml(followup_path, "TCGA-LUAD") is None


def test_tcga_endpoint_conflicts_are_excluded_not_reinterpreted(tmp_path) -> None:
    patient_path = tmp_path / "conflict.xml"
    patient_path.write_text(
        """<tcga_bcr><patient>
        <bcr_patient_barcode>SYNTHETIC-CONFLICT</bcr_patient_barcode>
        <vital_status>Alive</vital_status>
        <days_to_death>120</days_to_death>
        <days_to_last_followup>500</days_to_last_followup>
        </patient></tcga_bcr>""",
        encoding="utf-8",
    )
    parsed = parse_tcga_patient_xml(patient_path, "TCGA-LUSC")
    assert parsed is not None
    assert parsed["overall_survival_event"] is False
    assert parsed["endpoint_issue"] == "alive_with_days_to_death_conflict"


def test_tcga_loader_rejects_duplicate_conflicting_patient_records(tmp_path) -> None:
    cohort = tmp_path / "TCGA-LUAD"
    cohort.mkdir()
    template = """<tcga_bcr><patient>
      <bcr_patient_barcode>same-patient</bcr_patient_barcode>
      <vital_status>{status}</vital_status>
      <days_to_death>100</days_to_death>
      <age_at_initial_pathologic_diagnosis>60</age_at_initial_pathologic_diagnosis>
      <gender>FEMALE</gender>
      <histological_type>Lung Adenocarcinoma- Not Otherwise Specified (NOS)</histological_type>
    </patient></tcga_bcr>"""
    (cohort / "a.xml").write_text(template.format(status="Dead"), encoding="utf-8")
    (cohort / "b.xml").write_text(template.format(status="Alive"), encoding="utf-8")
    with pytest.raises(ValueError, match="Conflicting"):
        load_tcga_survival_cohort(cohort, "TCGA-LUAD")


def test_imaging_evidence_requires_passed_gate_and_never_counts_rank_as_malignancy() -> None:
    summary = validate_imaging_evidence(
        _imaging_evidence(),
        patient_id="synthetic-patient",
        baseline_at="2026-01-01T12:00:00+00:00",
    )
    assert summary["imaging_candidate_count"] == 1
    assert summary["largest_reported_nodule_diameter_mm"] == 12.5
    assert summary["malignant_nodule_count"] is None
    assert summary["ranking_scores_used_as_malignancy"] is False
    blocked = _imaging_evidence()
    blocked["gate_status"] = "blocked"
    with pytest.raises(ImagingEvidenceError, match="gate"):
        validate_imaging_evidence(
            blocked,
            patient_id="synthetic-patient",
            baseline_at="2026-01-01T12:00:00+00:00",
        )


def test_imaging_timestamp_must_match_prediction_baseline() -> None:
    evidence = _imaging_evidence()
    with pytest.raises(ImagingEvidenceError, match="timestamps"):
        validate_imaging_evidence(
            evidence,
            patient_id="synthetic-patient",
            baseline_at="2026-01-02T12:00:00+00:00",
        )


def test_future_imaging_baseline_is_rejected() -> None:
    future = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()
    evidence = _imaging_evidence(baseline=future)
    with pytest.raises(ImagingEvidenceError, match="future"):
        validate_imaging_evidence(
            evidence,
            patient_id="synthetic-patient",
            baseline_at=future,
        )


def test_unverified_prediction_baseline_blocks_survival_output() -> None:
    request = _request()
    request["baseline_definition"] = "other_or_unverified"
    output = run_inference(
        request,
        artifacts={
            "survival": {
                "estimator": object(),
                "metadata": {
                    "prediction_baseline_definition": "initial_pathologic_diagnosis",
                    "endpoint": "overall survival",
                },
            }
        },
    )
    assert output["survival"]["status"] == "blocked"
    assert "prediction_baseline_does_not_match_training_baseline" in output["survival"]["warnings"]


def test_inference_without_models_is_explicit_and_keeps_imaging_context() -> None:
    output = run_inference(_request())
    assert output["cancer_type"]["status"] == "not_trained"
    assert output["survival"]["status"] == "not_trained"
    assert output["imaging_context"]["malignant_nodule_count"] is None
    assert output["input_validation"]["performance_status"]["provenance"] == "approximate_ecog_interval"


def test_api_serves_frontend_and_reports_no_models_as_untrained() -> None:
    from fastapi.testclient import TestClient

    client = TestClient(create_app())
    status = client.get("/api/v1/clinical/status")
    page = client.get("/")
    assert status.status_code == 200
    assert status.json()["survival_model"] == "not_trained"
    assert page.status_code == 200
    assert b"clinical-form" in page.content


def test_model_failure_does_not_invalidate_independent_survival_output() -> None:
    request = _request()
    request["clinician_inputs"].update({
        "age_at_baseline_years": 65,
        "sex_at_baseline": "female",
        "stage_at_baseline": "Stage I",
        "ecog": None,
    })
    rng = np.random.default_rng(29)
    n = 200
    event = np.arange(n) % 4 != 0
    times = rng.uniform(100, 2500, n)
    frame = pd.DataFrame({
        "patient_id": [f"model-failure-{i}" for i in range(n)],
        "source": np.where(np.arange(n) % 2, "TCGA-LUAD", "TCGA-LUSC"),
        "age_at_baseline_years": rng.normal(65, 6, n),
        "sex_at_baseline": np.where(np.arange(n) % 2, "female", "male"),
        "stage_at_baseline": np.where(np.arange(n) % 3, "Stage I", "Stage III"),
        "overall_survival_time_days": times,
        "overall_survival_event": event,
        "endpoint_issue": None,
    })
    survival_artifact, _ = fit_survival_model(frame, use_authorized=True, random_state=15)
    output = run_inference(
        request,
        artifacts={
            "cancer_type": {
                "estimator": object(),
                "metadata": {
                    "prediction_baseline_definition": "initial_pathologic_diagnosis",
                },
            },
            "survival": survival_artifact,
        },
    )
    assert output["cancer_type"]["status"] == "failed"
    assert output["cancer_type"]["error_code"] == "model_inference_failed"
    assert output["survival"]["status"] == "research_prediction"
    values = output["survival"]["prediction"]["survival_function"]["probability"]
    assert np.all(np.diff(values) <= 1e-12)


def test_input_with_old_unwrapped_contract_is_rejected() -> None:
    request = _request()
    request["imaging_evidence"] = request["imaging_evidence"]["contract"]
    with pytest.raises(ClinicalInferenceError, match="input validation"):
        run_inference(request)


def test_source_shortcut_is_detected_and_prevents_classifier_fit() -> None:
    frame = pd.DataFrame({
        "patient_id": [f"p{i}" for i in range(100)],
        "source": ["TCGA-LUAD"] * 50 + ["TCGA-LUSC"] * 50,
        "histology": ["LUAD"] * 50 + ["LUSC"] * 50,
        "age_at_baseline_years": np.linspace(40, 80, 100),
        "sex_at_baseline": ["female", "male"] * 50,
        "stage_at_baseline": ["Stage I"] * 100,
    })
    assert source_target_shortcut_report(frame, "histology", "source")["perfect_source_target_shortcut"]
    with pytest.raises(ModelEligibilityError, match="source"):
        fit_cancer_type_model(
            frame,
            prediction_baseline_definition="initial_pathologic_diagnosis",
            use_authorized=True,
        )


def test_classifier_uses_grouped_holdouts_and_returns_valid_calibrated_probs() -> None:
    rng = np.random.default_rng(7)
    n = 200
    labels = np.asarray(["LUAD", "LUSC"] * (n // 2))
    frame = pd.DataFrame({
        "patient_id": [f"synthetic-{i}" for i in range(n)],
        "source": ["synthetic-cohort"] * n,
        "histology": labels,
        "age_at_baseline_years": rng.normal(65, 8, n),
        "sex_at_baseline": np.where(np.arange(n) % 3, "female", "male"),
        "stage_at_baseline": np.where(np.arange(n) % 4, "Stage I", "Stage III"),
    })
    artifact, report = fit_cancer_type_model(
        frame,
        prediction_baseline_definition="initial_pathologic_diagnosis",
        use_authorized=True,
        random_state=11,
    )
    assert report["test_patient_ids_used_for_fitting"] is False
    assert artifact["metadata"]["calibration_status"].startswith("sigmoid_calibrated")
    probabilities = artifact["estimator"].predict_proba(frame.iloc[:5][[
        "age_at_baseline_years", "sex_at_baseline", "stage_at_baseline"
    ]])
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert np.isfinite(probabilities).all()


def test_grouped_split_never_separates_a_patient() -> None:
    frame = pd.DataFrame({
        "patient_id": [f"p{i}" for i in range(100)],
        "source": ["cohort"] * 100,
    })
    target = pd.Series(["LUAD", "LUSC"] * 50)
    train, validation, test = _grouped_three_way_split(
        frame,
        target,
        "patient_id",
        random_state=4,
    )
    groups = [set(frame.iloc[index]["patient_id"]) for index in (train, validation, test)]
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])


def test_survival_training_retains_censoring_and_emits_supported_horizons() -> None:
    rng = np.random.default_rng(17)
    n = 240
    event = np.arange(n) % 4 != 0
    latent = rng.exponential(scale=1100, size=n)
    censoring = rng.uniform(450, 2400, size=n)
    times = np.where(event, np.minimum(latent, censoring), censoring)
    observed_event = event & (latent <= censoring)
    frame = pd.DataFrame({
        "patient_id": [f"survival-{i}" for i in range(n)],
        "source": np.where(np.arange(n) % 2, "TCGA-LUAD", "TCGA-LUSC"),
        "age_at_baseline_years": rng.normal(65, 8, n),
        "sex_at_baseline": np.where(np.arange(n) % 2, "female", "male"),
        "stage_at_baseline": np.where(np.arange(n) % 3, "Stage I", "Stage III"),
        "overall_survival_time_days": times,
        "overall_survival_event": observed_event,
        "endpoint_issue": None,
    })
    artifact, report = fit_survival_model(frame, use_authorized=True, random_state=9)
    assert report["eligible_patients"] == n
    assert report["events"] + report["censored"] == n
    assert artifact["metadata"]["endpoint"].startswith("overall survival")
    assert artifact["metadata"]["supported_horizons_days"]
    functions = artifact["estimator"].predict_survival_function(frame.iloc[:2][[
        "age_at_baseline_years", "sex_at_baseline", "stage_at_baseline"
    ]])
    for function in functions:
        times_to_check = np.linspace(function.x[0], function.x[-1], 30)
        values = np.asarray([function(t) for t in times_to_check])
        assert np.all((0 <= values) & (values <= 1))
        assert np.all(np.diff(values) <= 1e-12)


def test_horizons_are_not_extrapolated() -> None:
    dtype = [("event", "?"), ("time", "<f8")]
    target = np.array([(False, 120), (True, 250), (False, 300)], dtype=dtype)
    assert _supported_horizons(target) == []


def test_survival_development_refuses_to_pool_unharmonized_msk_endpoints() -> None:
    frame = pd.DataFrame({
        "patient_id": ["p1"],
        "source": ["MSK-CHORD"],
        "age_at_baseline_years": [65],
        "sex_at_baseline": ["female"],
        "stage_at_baseline": ["Stage I"],
        "overall_survival_time_days": [500],
        "overall_survival_event": [True],
        "endpoint_issue": [None],
    })
    with pytest.raises(ModelEligibilityError, match="TCGA-LUAD/LUSC only"):
        fit_survival_model(frame, use_authorized=True)


def test_cross_cohort_evaluation_keeps_test_source_out_of_fit() -> None:
    rng = np.random.default_rng(21)
    n_per_source = 100
    source = np.repeat(["TCGA-LUAD", "TCGA-LUSC"], n_per_source)
    times = rng.uniform(100, 1800, 2 * n_per_source)
    events = np.arange(2 * n_per_source) % 3 != 0
    frame = pd.DataFrame({
        "patient_id": [f"external-{i}" for i in range(2 * n_per_source)],
        "source": source,
        "age_at_baseline_years": rng.normal(65, 7, 2 * n_per_source),
        "sex_at_baseline": np.where(np.arange(2 * n_per_source) % 2, "female", "male"),
        "stage_at_baseline": np.where(np.arange(2 * n_per_source) % 3, "Stage I", "Stage II"),
        "overall_survival_time_days": times,
        "overall_survival_event": events,
        "endpoint_issue": None,
    })
    result = evaluate_cross_cohort_survival(
        frame,
        training_source="TCGA-LUAD",
        test_source="TCGA-LUSC",
    )
    assert result["training_patients"] == n_per_source
    assert result["test_patients"] == n_per_source
    assert result["test_patient_ids_used_for_training_or_preprocessing"] is False
    assert result["status"] == "cross-cohort_transport_diagnostic_not_external_clinical_validation"
