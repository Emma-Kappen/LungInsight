from __future__ import annotations

import numpy as np

from backend.Clinical.histology_v4 import EXPECTED_FEATURES, predict_histology_v4
from backend.Clinical.histology_v4_features import (
    ATTRIBUTE_NAMES,
    aggregate_malignant_nodules,
    build_histology_features,
    validate_histology_v4_inputs,
)
from backend.Imaging.brock_inputs import bbox_in_plane_diameter_mm


class FakeModel:
    def predict_proba(self, frame):
        assert list(frame.columns) == EXPECTED_FEATURES
        assert len(frame) == 1
        return np.asarray([[0.35, 0.65]])


def fake_artifact():
    return {
        "model": FakeModel(),
        "feature_columns": list(EXPECTED_FEATURES),
        "numeric_features": [
            "age_at_diagnosis", "pack_years", "smoking_year_started",
            "smoking_year_stopped", "dlco_percent_predicted",
            "fev1_fvc_post_percent", "fev1_fvc_pre_percent",
            "fev1_percent_predicted_post", "fev1_percent_predicted_pre",
            "ecog_preoperative", "karnofsky_preoperative",
        ],
    }


def nodule(
    nodule_id="n1", *, probability=0.2, primary="bronchus", wall="not_wall_adjacent",
    bbox=(10, 10, 10), attributes=None,
):
    start = [0, 0, 0]
    end = list(bbox)
    attrs = attributes or {name: {"expected_rating": 3.0} for name in ATTRIBUTE_NAMES}
    return {
        "stable_nodule_id": nodule_id,
        "eligibility_status": "eligible",
        "bbox_voxel": {"start_zyx": start, "end_zyx_exclusive": end},
        "anatomical_location": {"primary": primary, "wall_relationship": wall},
        "classification": {"attributes": attrs},
        "malignancy": {"selected_prediction": {"method": "brock", "status": "available", "probability": probability}},
    }


def contract(nodules, spacing=(1.0, 1.0, 1.0)):
    return {"coordinate_metadata": {"spacing_zyx_mm": list(spacing) if spacing is not None else None}, "nodules": nodules}


def test_zero_malignant_nodules_falls_back_to_form_and_imputation():
    result = aggregate_malignant_nodules(contract([nodule(probability=0.099)]))
    assert result["n_malignant_nodules"] == 0
    assert result["mean_nodule_size_mm"] is None
    assert all(value is None for value in result["mean_attribute_ranks"].values())
    built = build_histology_features({}, {"lung_location": "Peripheral Lung"}, contract([nodule(probability=0.099)]))
    assert built["imaging_features_available"] is False
    assert built["values"]["lung_location"] == "Peripheral Lung"
    assert built["provenance"]["lung_location"] == "form"


def test_single_malignant_nodule_mean_size_and_attribute_ranks():
    result = aggregate_malignant_nodules(contract([nodule(bbox=(12, 8, 9))], spacing=(2.0, 1.0, 1.5)))
    assert result["n_malignant_nodules"] == 1
    assert result["mean_nodule_size_mm"] == 13.5
    assert result["mean_attribute_ranks"]["subtlety"] == 3.0
    assert result["lung_location"] == "Central Lung"
    assert result["anatomic_subdivision"] == "Bronchial"


def test_multiple_malignant_nodules_mean_and_mode_tie_break_prefers_largest_bbox():
    a = nodule("a", probability=0.7, primary="bronchus", bbox=(10, 10, 8))
    b = nodule("b", probability=0.6, primary="lung_wall", wall="wall_adjacent", bbox=(10, 15, 8))
    b["classification"]["attributes"]["subtlety"]["expected_rating"] = 5.0
    result = aggregate_malignant_nodules(contract([a, b]))
    assert result["n_malignant_nodules"] == 2
    assert result["lung_location"] == "Peripheral Lung"  # tied mode; larger in-plane BBox wins
    assert result["mean_nodule_size_mm"] == 12.5
    assert result["mean_attribute_ranks"]["subtlety"] == 4.0


def test_partial_ranker_outputs_mean_available_values_and_warns():
    a = nodule("a")
    b = nodule("b")
    b["classification"]["attributes"].pop("texture")
    b["classification"]["attributes"]["subtlety"]["expected_rating"] = None
    result = aggregate_malignant_nodules(contract([a, b]))
    assert result["mean_attribute_ranks"]["subtlety"] == 3.0
    assert result["mean_attribute_ranks"]["texture"] == 3.0
    assert any("subtlety: available for 1/2" in warning for warning in result["warnings"])
    assert any("texture: available for 1/2" in warning for warning in result["warnings"])


def test_spacing_unavailable_flags_size_but_keeps_other_aggregates():
    result = aggregate_malignant_nodules(contract([nodule()], spacing=None))
    assert result["mean_nodule_size_mm"] is None
    assert result["spacing_available"] is False
    assert result["mean_attribute_ranks"]["subtlety"] == 3.0
    assert any("BBox size unavailable" in warning for warning in result["warnings"])


def test_coarse_location_does_not_guess_lobe_and_category_mismatch_becomes_missing():
    row = nodule(primary="lung_parenchyma")
    built = build_histology_features({}, {"anatomic_subdivision": "L-Upper"}, contract([row]))
    assert built["values"]["anatomic_subdivision"] == "L-Upper"  # explicit clinician form wins
    assert built["provenance"]["anatomic_subdivision"] == "form"
    mismatch = build_histology_features({}, {"lung_location": "left upper lobe"}, contract([row]))
    assert mismatch["values"]["lung_location"] is None
    assert mismatch["provenance"]["lung_location"] == "imputed"
    assert any("not a fitted v4 encoder category" in warning for warning in mismatch["warnings"])


def test_form_conflict_wins_and_is_recorded():
    built = build_histology_features({}, {"lung_location": "Peripheral Lung"}, contract([nodule(primary="bronchus")]))
    assert built["values"]["lung_location"] == "Peripheral Lung"
    assert built["provenance"]["lung_location"] == "form"
    assert any("overrides imaging-derived" in warning for warning in built["warnings"])


def test_other_malignancy_history_is_not_inferred_from_nodules_but_can_come_from_form():
    built = build_histology_features({}, {}, contract([nodule(), nodule("n2", primary="lung_wall", wall="wall_adjacent")]))
    assert built["values"]["other_malignancy_history"] is None
    assert built["provenance"]["other_malignancy_history"] == "imputed"
    assert any("not inferred from imaging" in warning for warning in built["warnings"])
    form = build_histology_features({}, {"other_malignancy_history": "Yes, History of Prior Malignancy"}, contract([]))
    assert form["values"]["other_malignancy_history"] == "Yes, History of Prior Malignancy"


def test_v4_uses_unchanged_19_feature_contract_and_returns_imaging_provenance():
    clinician = {
        "prediction_baseline_definition": "initial_pathologic_diagnosis",
        "age_at_baseline_years": 67,
        "sex_at_baseline": "female",
        "race_at_baseline": "asian",
        "ethnicity_at_baseline": "not_hispanic_or_latino",
    }
    result = predict_histology_v4(fake_artifact(), clinician, imaging_contract=contract([nodule()]))
    assert result["status"] == "research_prediction"
    assert result["predicted_class"] == "LUAD"
    assert len(result["model_inputs_used"]) == 19
    assert result["input_provenance"]["lung_location"] == "imaging"
    assert result["imaging_features_available"] is True
    assert result["imaging_derived"]["n_malignant_nodules"] == 1
    assert result["model_inputs_used"]["ethnicity"] is None
    assert result["input_provenance"]["ethnicity"] == "imputed"


def test_v4_abstains_without_initial_diagnosis_baseline():
    result = predict_histology_v4(fake_artifact(), {}, imaging_contract=contract([]))
    assert result["status"] == "abstained"
    assert result["prediction"] is None
    assert result["imaging_features_available"] is False


def test_v4_form_values_must_match_encoder_categories():
    assert validate_histology_v4_inputs({"lung_location": "Central Lung"}) == {"lung_location": "Central Lung"}
    try:
        validate_histology_v4_inputs({"lung_location": "left upper lobe"})
    except ValueError as exc:
        assert "exact v4 encoder category" in str(exc)
    else:
        raise AssertionError("Invalid category should be rejected")


def test_mode_tie_break_uses_malignancy_score_after_equal_bbox_and_alphabetical_last():
    a = nodule("a", probability=0.4, primary="bronchus", bbox=(10, 10, 10))
    b = nodule("b", probability=0.8, primary="lung_wall", wall="wall_adjacent", bbox=(10, 10, 10))
    result = aggregate_malignant_nodules(contract([a, b]))
    assert result["lung_location"] == "Peripheral Lung"  # same BBox, higher Brock score


def test_mode_tie_break_alphabetical_when_bbox_and_score_are_equal():
    a = nodule("a", probability=0.5, primary="bronchus", bbox=(10, 10, 10))
    b = nodule("b", probability=0.5, primary="lung_wall", wall="wall_adjacent", bbox=(10, 10, 10))
    result = aggregate_malignant_nodules(contract([a, b]))
    assert result["lung_location"] == "Central Lung"


def test_anatomic_subdivision_form_conflict_is_recorded_and_form_wins():
    built = build_histology_features(
        {}, {"anatomic_subdivision": "L-Upper"}, contract([nodule(primary="bronchus")])
    )
    assert built["values"]["anatomic_subdivision"] == "L-Upper"
    assert built["provenance"]["anatomic_subdivision"] == "form"
    assert any("anatomic_subdivision" in warning and "overrides imaging-derived" in warning for warning in built["warnings"])


def test_independent_model_runner_forwards_v4_context_kwargs():
    from backend.Clinical.inference import _run_model_independently

    def predictor(artifact, clinician, *, imaging_contract, form_values):
        return {"status": "ok", "n_nodules": len(imaging_contract["nodules"]), "form": form_values}

    result = _run_model_independently(
        predictor, {}, {}, imaging_contract=contract([nodule()]), form_values={"pack_years": 20}
    )
    assert result == {"status": "ok", "n_nodules": 1, "form": {"pack_years": 20}}


def test_histology_size_uses_the_shared_brock_bbox_diameter_helper():
    assert bbox_in_plane_diameter_mm([17.5, 9.375, 8.59375]) == 9.375
    assert bbox_in_plane_diameter_mm([17.5, None, 8.0]) is None
