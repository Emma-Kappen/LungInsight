from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

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
    classify_tcga_histology,
    load_tcga_survival_cohort,
    msk_lung_patient_labels,
    parse_tcga_patient_xml,
)
from LungInsight_v2.backend.Clinical.inference import (
    ClinicalInferenceError,
    load_artifacts,
    run_inference,
)
from LungInsight_v2.backend.Clinical.training import (
    ModelEligibilityError,
    SURVIVAL_FEATURE_ORDER,
    _grouped_three_way_split,
    _supported_horizons,
    build_cancer_type_eligibility_report,
    fit_cancer_type_model,
    fit_survival_model,
    evaluate_cross_cohort_survival,
    save_artifact_bundle,
    source_target_shortcut_report,
)


def _valid_imaging_contract(patient_id: str = "synthetic-patient") -> dict:
    return {
        "contract_version": "2.0.0",
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
        "study": {
            "processing_status": "completed",
            "preprocessing_status": "completed",
            "mask_validation_status": "validated_automated",
            "raw_detection_counts": {"ViTDet3D": None},
            "fused_candidate_count": None,
            "excluded_candidate_count": None,
            "eligible_nodule_count": 1,
            "brock_prediction_counts": {"completed": 0, "incomplete": 1, "failed": 0, "not_run": 0},
            "warnings": [],
            "errors": [],
            "provenance": {
                "run_id": "synthetic-run",
                "meta_schema_version": "2.14",
                "imaging_model_versions": {"stage06": "synthetic-v1"},
                "review_attestation_verified": False,
                "mask_visual_review_status": "not_performed",
            },
            "artifact_refs": {},
        },
        "nodules": [
            {
                "stable_nodule_id": "synthetic-nodule",
                "eligibility_status": "eligible",
                "quality_status": "passed",
                "detector": {
                    "candidate_id": "candidate-1",
                    "source": "ViTDet3D",
                    "source_members": [],
                    "vitdet_score": 0.99,
                },
                "detection_sources": ["ViTDet3D"],
                "bbox_voxel": {"start_zyx": [0, 0, 0], "end_zyx_exclusive": [1, 1, 1]},
                "bbox_lps_mm": {"min_lps_mm": [0, 0, 0], "max_lps_mm": [1, 1, 1]},
                "centroid_voxel": [0.5, 0.5, 0.5],
                "centroid_lps_mm": [0.5, 0.5, 0.5],
                "measurements": {
                    "diameter_mm": 12.5,
                    "volume_mm3": 100.0,
                    "source": "stage02_detector_bbox_estimate",
                    "status": "detector_estimate",
                },
                "anatomical_location": {},
                "classification": {
                    "status": "available",
                    "source": "synthetic-v1",
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
                },
                "malignancy": {
                    "brock": {
                        "status": "incomplete",
                        "method": "brock_formula_decomposition",
                        "model_name": "Brock_PanCan_full",
                        "model_version": "published-coefficients-verified-v1",
                        "target": "pulmonary_nodule_malignancy_probability",
                        "probability": None,
                        "features_missing": ["age_years"],
                        "feature_contributions": [],
                        "limitations": ["Required input was not supplied."],
                    },
                    "selected_prediction": {
                        "method": "brock",
                        "model_version": "published-coefficients-verified-v1",
                        "status": "incomplete",
                        "probability": None,
                        "value": None,
                        "missing_inputs": ["age_years"],
                        "limitations": ["Required input was not supplied."],
                    },
                },
                "uncertainty": {},
                "input_completeness": {"status": "incomplete", "fields_missing": ["age_years"]},
                "quality_flags": [],
                "model_versions": {"stage04": "synthetic-v1"},
                "gradcam_refs": {},
                "shap_refs": {},
                "artifact_refs": {},
            }
        ],
    }


def _imaging_evidence(patient_id: str = "synthetic-patient", scan_datetime: str | None = None) -> dict:
    return {
        "contract": _valid_imaging_contract(patient_id),
        "source_timestamp": "2026-01-01T12:00:00+00:00",
        "scan_datetime": scan_datetime,
        "gate_status": "passed",
        "imaging_pipeline_version": "meta_schema-2.14",
        "imaging_model_versions": _valid_imaging_contract(patient_id)["study"]["provenance"]["imaging_model_versions"],
    }


def _request() -> dict:
    return {
        "schema_version": "1.0.0",
        "patient_id": "synthetic-patient",
        "baseline_definition": "initial_pathologic_diagnosis",
        "clinician_inputs": {
            "age_at_baseline_years": 64,
            "sex_at_baseline": "female",
            "kps": None,
            "ecog": 1,
        },
        "imaging_evidence": _imaging_evidence(),
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
        validate_clinician_inputs({"stage_at_baseline": "Stage IIA"})


def test_clinical_form_preserves_unknown_and_absent_values_and_calculates_pack_years() -> None:
    values = validate_clinician_inputs({
        "age_at_baseline_years": None,
        "age_at_baseline_years_status": "unknown",
        "sex_at_baseline": "unknown",
        "race_at_baseline": "not_reported",
        "ethnicity_at_baseline": "hispanic_or_latino",
        "smoking_status": "former",
        "years_smoked": 20,
        "years_smoked_status": "reported",
        "packs_per_day": 1.5,
        "packs_per_day_status": "reported",
        "years_since_cessation": 4,
        "family_history_lung_cancer": "no",
        "medical_history_status": "reported",
        "medical_history_conditions": [],
        "symptoms_status": "unknown",
        "symptoms": [],
    })
    assert values["age_at_baseline_years_status"] == "unknown"
    assert values["sex_at_baseline"] is None
    assert values["sex_at_baseline_status"] == "unknown"
    assert values["race_at_baseline"] == "not_reported"
    assert values["ethnicity_at_baseline"] == "hispanic_or_latino"
    assert values["pack_years"] == 30
    assert values["medical_history_status"] == "reported"
    assert values["medical_history_conditions"] == []
    assert values["symptoms_status"] == "unknown"
    with pytest.raises(ClinicalValidationError, match="must match"):
        validate_clinician_inputs({
            "smoking_status": "former",
            "years_smoked": 20,
            "years_smoked_status": "reported",
            "packs_per_day": 1,
            "packs_per_day_status": "reported",
            "pack_years": 5,
        })


def test_saved_clinician_record_round_trips_through_validation() -> None:
    saved = validate_clinician_inputs({
        "age_at_baseline_years": 62,
        "sex_at_baseline": "female",
        "race_at_baseline": "white",
        "ethnicity_at_baseline": "not_hispanic_or_latino",
        "smoking_status": "former",
        "years_smoked": 30,
        "packs_per_day": 1,
        "years_since_cessation": 5,
        "family_history_lung_cancer": "no",
        "medical_history_status": "reported",
        "medical_history_conditions": [],
        "symptoms_status": "reported",
        "symptoms": [],
        "kps": 90,
    })
    assert "performance_status" in saved and "kps_category" in saved
    reread = validate_clinician_inputs(saved)
    assert reread == saved
    tampered = dict(saved, kps_category="0-10", sex_at_baseline_status="reported")
    assert validate_clinician_inputs(tampered)["kps_category"] == saved["kps_category"]
    assert validate_clinician_inputs(tampered)["sex_at_baseline_status"] == saved["sex_at_baseline_status"]


def test_prediction_uses_system_utc_timestamp_without_pathological_stage() -> None:
    request = _request()
    output = run_inference(request)
    timestamp = output["prediction_baseline_datetime"]
    assert timestamp.endswith("Z")
    assert datetime.fromisoformat(timestamp.replace("Z", "+00:00")).tzinfo is not None
    assert output["provenance"]["prediction_baseline_datetime"] == timestamp
    assert output["imaging_context"]["scan_datetime"] is None
    assert "stage_at_baseline" not in output["input_validation"]["clinician_fields_supplied"]


def test_pathological_stage_is_not_an_inference_feature() -> None:
    assert "stage_at_baseline" not in SURVIVAL_FEATURE_ORDER
    with pytest.raises(ClinicalInferenceError, match="stage_at_baseline"):
        run_inference({
            **_request(),
            "clinician_inputs": {
                **_request()["clinician_inputs"],
                "stage_at_baseline": "Stage IIA",
            },
        })


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


@pytest.mark.parametrize(
    ("recorded_histology", "expected"),
    [
        ("Lung Adenocarcinoma- Not Otherwise Specified (NOS)", "LUAD"),
        ("Lung Bronchioloalveolar Carcinoma Nonmucinous", "LUAD"),
        ("Lung Squamous Cell Carcinoma- Not Otherwise Specified (NOS)", "LUSC"),
        ("Lung Papillary Squamous Cell Caricnoma", "LUSC"),
        ("Non-Small Cell Lung Cancer", None),
        (None, None),
    ],
)
def test_tcga_histology_mapping_requires_explicit_supported_diagnosis(
    recorded_histology: str | None,
    expected: str | None,
) -> None:
    assert classify_tcga_histology(recorded_histology) == expected


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
        )


def test_missing_scan_datetime_is_not_substituted() -> None:
    evidence = _imaging_evidence()
    assert validate_imaging_evidence(evidence, patient_id="synthetic-patient")["scan_datetime"] is None


def test_scan_datetime_requires_an_aware_timestamp() -> None:
    evidence = _imaging_evidence(scan_datetime="2026-01-01T12:00:00")
    with pytest.raises(ImagingEvidenceError, match="timezone"):
        validate_imaging_evidence(evidence, patient_id="synthetic-patient")


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
    assert b'name="baseline_at"' not in page.content
    assert b'Pathologic stage at baseline' not in page.content
    frontend_script = (Path(__file__).resolve().parents[3] / "frontend" / "app.js").read_text(
        encoding="utf-8"
    )
    # Submission is gated on the ten required items, including a confirmed CT upload for this patient.
    assert 'const ctOk = Boolean(S.upload && S.uploadPid === f.patientId' in frontend_script
    assert "Selected; not uploaded yet." in frontend_script
    assert "CT uploaded and validated" in frontend_script
    assert "/viewer/slices/" in frontend_script
    assert "viewer-lung" in frontend_script
    assert "viewer-previous" in frontend_script
    assert 'addEventListener("wheel"' in frontend_script
    assert "print-report" in frontend_script
    assert "gradcam_nodule_id" in frontend_script and "gradcam_alpha" in frontend_script
    for banned in (b"internal_structure", b"Age status", b"Years smoked information"):
        assert banned not in page.content


def test_analysis_run_rejects_unknown_upload_job() -> None:
    from fastapi.testclient import TestClient

    client = TestClient(create_app())
    response = client.post(
        "/api/v1/analysis/runs",
        json={
            "patient_id": "functional-test-patient",
            "upload_job_id": "missing-upload-job",
            "clinician_inputs": {
                "age_at_baseline_years": 67,
                "sex_at_baseline": "female",
                "kps": 80,
            },
            "requested_malignancy_method": "brock",
            "data_provenance": "synthetic_functional_test",
        },
    )
    assert response.status_code == 422
    assert "not found" in response.json()["detail"].lower()


def test_analysis_viewer_serves_bounded_ct_slices(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from fastapi.testclient import TestClient
    from LungInsight_v2.backend.Clinical import api as clinical_api
    from LungInsight_v2.backend.Imaging.api import analysis as analysis_api

    patient_root = tmp_path / "viewer-patient"
    patient_root.mkdir()
    volume = np.linspace(-1000, 400, 24, dtype=np.float32).reshape(3, 4, 2)
    np.save(patient_root / "volume_hu_cropped.npy", volume)
    np.save(patient_root / "lung_mask.npy", np.ones_like(volume, dtype=bool))
    np.save(patient_root / "heart_mask.npy", np.zeros_like(volume, dtype=bool))
    monkeypatch.setattr(clinical_api, "IMAGING_OUTPUT_ROOT", tmp_path)
    analysis_api._RUNS["viewer-run"] = {
        "run_id": "viewer-run",
        "patient_id": "viewer-patient",
        "status": "completed",
    }
    try:
        client = TestClient(create_app())
        metadata = client.get("/api/v1/analysis/runs/viewer-run/viewer")
        image = client.get("/api/v1/analysis/runs/viewer-run/viewer/slices/1?lung_overlay=true")
        invalid = client.get("/api/v1/analysis/runs/viewer-run/viewer/slices/99")
        assert metadata.status_code == 200
        assert metadata.json()["shape_zyx"] == [3, 4, 2]
        assert metadata.json()["overlays"] == {"lung_mask": True, "heart_mask": True}
        assert image.status_code == 200
        assert image.headers["content-type"] == "image/png"
        assert image.content.startswith(b"\x89PNG")
        assert invalid.status_code == 404
    finally:
        analysis_api._RUNS.pop("viewer-run", None)


def test_analysis_viewer_gradcam_overlay_tolerates_nan_outside_coverage(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import io
    from PIL import Image
    from LungInsight_v2.backend.Imaging.api import analysis as analysis_api

    patient_root = tmp_path / "cam-patient"
    patient_root.mkdir()
    volume = np.full((3, 4, 2), -600.0, dtype=np.float32)
    np.save(patient_root / "volume_hu_cropped.npy", volume)
    cam = np.full((3, 4, 2), np.nan, dtype=np.float32)
    cam[1, 0, 0] = 1.0
    cam[1, 1, 1] = 0.5
    np.save(patient_root / "attribute_ct_cam.npy", cam)
    (patient_root / "imaging_clinical_contract_v2.json").write_text("{}", encoding="utf-8")
    fake_contract = {
        "nodules": [
            {
                "stable_nodule_id": "nodule_test",
                "gradcam_refs": {"spiculation": {"ct_cam_path": "attribute_ct_cam.npy"}},
            }
        ]
    }
    monkeypatch.setattr(analysis_api, "read_json", lambda _path: fake_contract)
    monkeypatch.setattr(analysis_api, "validate_contract", lambda _contract: None)
    analysis_api._RUNS["cam-run"] = {"run_id": "cam-run", "patient_id": "cam-patient", "status": "completed"}
    try:
        def render(index: int, **kwargs: str) -> np.ndarray:
            png = analysis_api.render_viewer_slice("cam-run", index, output_root=tmp_path, **kwargs)
            return np.asarray(Image.open(io.BytesIO(png)).convert("RGB"), dtype=np.int16)

        base = render(1)
        overlay = render(1, gradcam_nodule_id="nodule_test", gradcam_attribute="spiculation")
        assert overlay[0, 0, 0] > base[0, 0, 0]
        assert np.array_equal(overlay[0, 1], base[0, 1])
        nan_slice_base = render(0)
        nan_slice_overlay = render(0, gradcam_nodule_id="nodule_test", gradcam_attribute="spiculation")
        assert np.array_equal(nan_slice_overlay, nan_slice_base)
    finally:
        analysis_api._RUNS.pop("cam-run", None)


def test_analysis_viewer_gradcam_multi_nodule_colormap_and_alpha(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import io
    from PIL import Image
    from LungInsight_v2.backend.Imaging.api import analysis as analysis_api

    patient_root = tmp_path / "cam2-patient"
    patient_root.mkdir()
    np.save(patient_root / "volume_hu_cropped.npy", np.full((2, 4, 4), -600.0, dtype=np.float32))
    first = np.zeros((2, 4, 4), dtype=np.float32)
    first[0, 0, 0] = 1.0
    first[0, 0, 1] = 0.5
    second = np.zeros((2, 4, 4), dtype=np.float32)
    second[0, 3, 3] = 1.0
    np.save(patient_root / "cam_a.npy", first)
    np.save(patient_root / "cam_b.npy", second)
    (patient_root / "imaging_clinical_contract_v2.json").write_text("{}", encoding="utf-8")
    fake_contract = {
        "nodules": [
            {"stable_nodule_id": "n1", "gradcam_refs": {"margin": {"ct_cam_path": "cam_a.npy"}}},
            {"stable_nodule_id": "n2", "gradcam_refs": {"margin": {"ct_cam_path": "cam_b.npy"}}},
        ]
    }
    monkeypatch.setattr(analysis_api, "read_json", lambda _path: fake_contract)
    monkeypatch.setattr(analysis_api, "validate_contract", lambda _contract: None)
    analysis_api._RUNS["cam2-run"] = {"run_id": "cam2-run", "patient_id": "cam2-patient", "status": "completed"}
    try:
        def render(**kwargs: object) -> np.ndarray:
            png = analysis_api.render_viewer_slice("cam2-run", 0, output_root=tmp_path, **kwargs)
            return np.asarray(Image.open(io.BytesIO(png)).convert("RGB"), dtype=np.int16)

        base = render()
        both = render(gradcam_nodule_ids=["n1", "n2"], gradcam_attribute="margin", gradcam_alpha=1.0)
        only_first = render(gradcam_nodule_ids=["n1"], gradcam_attribute="margin", gradcam_alpha=1.0)
        transparent = render(gradcam_nodule_ids=["n1", "n2"], gradcam_attribute="margin", gradcam_alpha=0.0)
        assert np.array_equal(transparent, base)
        # Per-nodule min-max: the maximum is red (#D7191C) and the minimum is blue (#1538D6), never faded out.
        assert both[0, 0].tolist() == [0xD7, 0x19, 0x1C]
        assert both[1, 1].tolist() == [0x15, 0x38, 0xD6]
        assert both[3, 3].tolist() == [0xD7, 0x19, 0x1C]
        assert only_first[3, 3].tolist() == [0x15, 0x38, 0xD6]
    finally:
        analysis_api._RUNS.pop("cam2-run", None)


def test_analysis_viewer_gradcam_is_windowed_around_the_nodule(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import io
    from PIL import Image
    from LungInsight_v2.backend.Imaging.api import analysis as analysis_api

    patient_root = tmp_path / "win-patient"
    patient_root.mkdir()
    np.save(patient_root / "volume_hu_cropped.npy", np.full((3, 48, 48), -600.0, dtype=np.float32))
    cam = np.full((3, 48, 48), 0.1, dtype=np.float32)
    cam[1, 11, 11] = 0.5  # on the nodule box
    cam[1, 45, 45] = 1.0  # strongest value, but ~45 mm from the nodule
    np.save(patient_root / "cam.npy", cam)
    (patient_root / "imaging_clinical_contract_v2.json").write_text("{}", encoding="utf-8")
    fake_contract = {
        "nodules": [{
            "stable_nodule_id": "n1",
            "bbox_voxel": {"start_zyx": [1, 10, 10], "end_zyx_exclusive": [2, 14, 14]},
            "gradcam_refs": {"margin": {"ct_cam_path": "cam.npy"}},
        }]
    }
    monkeypatch.setattr(analysis_api, "read_json", lambda _path: fake_contract)
    monkeypatch.setattr(analysis_api, "validate_contract", lambda _contract: None)
    analysis_api._RUNS["win-run"] = {"run_id": "win-run", "patient_id": "win-patient", "status": "completed"}
    try:
        def render(**kwargs: object) -> np.ndarray:
            png = analysis_api.render_viewer_slice("win-run", 1, output_root=tmp_path, **kwargs)
            return np.asarray(Image.open(io.BytesIO(png)).convert("RGB"), dtype=np.int16)

        base = render()
        shown = render(gradcam_nodule_ids=["n1"], gradcam_attribute="margin", gradcam_alpha=1.0)
        assert shown[11, 11].tolist() == [0xD7, 0x19, 0x1C]  # window maximum inside the box is red
        assert np.array_equal(shown[45, 45], base[45, 45])  # a stronger but distant value is not drawn
        assert not np.array_equal(shown[16, 11], base[16, 11])  # fades out gradually just beyond the box
    finally:
        analysis_api._RUNS.pop("win-run", None)


def test_manual_brock_fills_only_missing_inputs_and_persists(tmp_path: Path) -> None:
    import json
    from LungInsight_v2.backend.Imaging.api import analysis as analysis_api
    from brock import estimate_brock

    patient_root = tmp_path / "man-patient"
    (patient_root / "04_malignancy_prediction").mkdir(parents=True)

    def known(value: object) -> dict[str, object]:
        return {"value": value, "source": "test"}

    row = {
        "stable_nodule_id": "nodule_x",
        "input_provenance": {
            "patient_fields": {
                "age_years": known(68.0), "sex": known("female"),
                "family_history_lung_cancer": known(False), "emphysema": known(True),
            },
            "nodule_fields": {
                "nodule_size_mm": known(5.3), "upper_lobe": known(None), "spiculation": known(0),
                "nodule_type": known("solid"), "nodule_count": known(2),
            },
        },
    }
    (patient_root / "04_malignancy_prediction" / "malignancy_predictions.json").write_text(
        json.dumps({"predictions": [row]}), encoding="utf-8"
    )
    analysis_api._RUNS["a" * 32] = {"run_id": "a" * 32, "patient_id": "man-patient", "status": "completed"}
    try:
        record = analysis_api.compute_manual_brock("a" * 32, "nodule_x", {"upper_lobe": True}, output_root=tmp_path)
        expected = estimate_brock({
            "age_years": 68.0, "sex": "female", "family_history_lung_cancer": False, "emphysema": True,
            "nodule_size_mm": 5.3, "upper_lobe": True, "spiculation": 0, "nodule_type": "solid", "nodule_count": 2,
        })
        assert record["brock"]["status"] == "available"
        assert record["brock"]["probability"] == expected["probability"]
        assert record["supplied"] == {"upper_lobe": True}
        stored = json.loads((patient_root / f"brock_manual_{'a' * 32}.json").read_text(encoding="utf-8"))
        assert stored["nodules"]["nodule_x"]["supplied"] == {"upper_lobe": True}
        with pytest.raises(analysis_api.BrockConflictError):
            analysis_api.compute_manual_brock("a" * 32, "nodule_x", {"emphysema": False}, output_root=tmp_path)
        with pytest.raises(KeyError):
            analysis_api.compute_manual_brock("a" * 32, "other", {"upper_lobe": True}, output_root=tmp_path)
        with pytest.raises(ValueError):
            analysis_api.compute_manual_brock("a" * 32, "nodule_x", {"bogus": 1}, output_root=tmp_path)
    finally:
        analysis_api._RUNS.pop("a" * 32, None)


def test_analysis_run_records_only_observed_completed_stages() -> None:
    from LungInsight_v2.backend.Imaging.api import analysis as analysis_api

    analysis_api._RUNS["status-run"] = {"run_id": "status-run", "completed_stages": []}
    try:
        analysis_api._record_completed_stage("status-run", "[01/07] Stage 01 Complete")
        analysis_api._record_completed_stage("status-run", "[01/07] Stage 01 Complete")
        analysis_api._record_completed_stage("status-run", "[02/07] Stage 03 Complete")
        assert analysis_api.get_analysis("status-run", output_root=Path("."))["completed_stages"] == ["stage01"]
    finally:
        analysis_api._RUNS.pop("status-run", None)


def test_analysis_run_status_survives_process_local_cache_loss(tmp_path: Path) -> None:
    from LungInsight_v2.backend.Imaging.api import analysis as analysis_api

    run_id = "0123456789abcdef0123456789abcdef"
    run = {
        "run_id": run_id,
        "patient_id": "persisted-patient",
        "status": "completed",
        "completed_stages": ["stage01", "stage02"],
        "_state_path": str(tmp_path / ".analysis_runs" / f"{run_id}.json"),
    }
    analysis_api._RUNS[run_id] = run
    try:
        analysis_api._save_run(run)
        analysis_api._RUNS.pop(run_id)
        restored = analysis_api.get_analysis(run_id, output_root=tmp_path)
        assert restored["status"] == "completed"
        assert restored["completed_stages"] == ["stage01", "stage02"]
        assert "_state_path" not in restored
    finally:
        analysis_api._RUNS.pop(run_id, None)


def test_model_failure_does_not_invalidate_independent_survival_output() -> None:
    request = _request()
    request["clinician_inputs"].update({
        "age_at_baseline_years": 65,
        "sex_at_baseline": "female",
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
    assert artifact["metadata"]["feature_order"] == ["sex_at_baseline"]
    assert artifact["metadata"]["feature_contract_version"] == "msk-chord-sex-at-birth-v1"
    assert set(report["model_selection"]["candidates"]) == {
        "regularized_logistic_regression",
        "random_forest",
    }
    assert artifact["metadata"]["calibration_status"].startswith("sigmoid_calibrated")
    probabilities = artifact["estimator"].predict_proba(frame.iloc[:5][[
        "sex_at_baseline"
    ]])
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert np.isfinite(probabilities).all()


def test_cancer_artifact_round_trip_preserves_probabilities(tmp_path) -> None:
    frame = pd.DataFrame({
        "patient_id": [f"round-trip-{index}" for index in range(200)],
        "source": ["authorized-cohort"] * 200,
        "histology": ["LUAD", "LUSC"] * 100,
        "label_status": ["eligible"] * 200,
        "sex_at_baseline": np.where(np.arange(200) % 3, "female", "male"),
    })
    artifact, _ = fit_cancer_type_model(
        frame,
        prediction_baseline_definition="initial_pathologic_diagnosis",
        use_authorized=True,
        random_state=31,
    )
    examples = pd.DataFrame({"sex_at_baseline": ["female", "male"]})
    expected = artifact["estimator"].predict_proba(examples)
    artifact_path = tmp_path / "clinical.joblib"
    save_artifact_bundle(artifact_path, cancer_type=artifact)
    reloaded = load_artifacts(artifact_path)
    assert reloaded["cancer_type"] is not None
    actual = reloaded["cancer_type"]["estimator"].predict_proba(examples)
    assert np.array_equal(expected, actual)
    request = _request()
    request["clinician_inputs"]["sex_at_baseline"] = None
    output = run_inference(request, artifacts=reloaded)
    assert output["cancer_type"]["status"] == "abstained"
    assert output["cancer_type"]["missing_features"] == ["sex_at_baseline"]


def test_classifier_eligibility_report_is_aggregate_and_requires_authorization() -> None:
    frame = pd.DataFrame({
        "patient_id": ["p1", "p2", "p3", "p4"],
        "source": ["approved-cohort"] * 4,
        "histology": ["LUAD", "LUSC", "LUAD", "other"],
        "label_status": ["eligible", "eligible", "eligible", "ambiguous_or_other"],
        "age_at_baseline_years": [60, 61, 62, 63],
        "sex_at_baseline": ["female", "male", "female", "male"],
        "stage_at_baseline": ["Stage I", "Stage II", "Stage I", "Stage II"],
    })
    report = build_cancer_type_eligibility_report(
        frame,
        authorization_status="not_confirmed",
        source_label_provenance="source_recorded",
    )
    assert report["supported_class_patient_counts"] == {"LUAD": 2, "LUSC": 1}
    assert report["eligible_for_training"] is False
    assert "source_data_use_authorization_not_confirmed" in report["blockers"]
    assert report["privacy"]["patient_identifiers_included"] is False


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
