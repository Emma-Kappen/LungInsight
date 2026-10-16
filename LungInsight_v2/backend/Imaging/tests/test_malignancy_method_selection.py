from __future__ import annotations

import importlib.util
from pathlib import Path


STAGE04_PATH = Path(__file__).resolve().parents[1] / "stages" / "04_malignancy_prediction.py"
SPEC = importlib.util.spec_from_file_location("stage04_malignancy_prediction", STAGE04_PATH)
assert SPEC and SPEC.loader
stage04 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stage04)


def test_brock_is_the_only_supported_method() -> None:
    brock = {
        "status": "incomplete",
        "model_version": "published-coefficients-verified-v1",
        "probability": None,
        "features_missing": ["age_years"],
        "feature_contributions": [],
        "limitations": ["Required input missing."],
    }
    selected = stage04.selected_prediction(brock)
    assert stage04.SUPPORTED_MALIGNANCY_METHODS == ("brock",)
    assert selected["method"] == "brock"
    assert selected["status"] == "incomplete"
    assert selected["probability"] is None
    assert selected["value"] is None
    assert selected["missing_inputs"] == ["age_years"]


def test_legacy_lidc_method_is_rejected_by_parser() -> None:
    parser = stage04.build_parser()
    try:
        parser.parse_args(["--malignancy-method", "lidc_reader_rating"])
    except SystemExit as exc:
        assert exc.code != 0
    else:
        raise AssertionError("legacy LIDC method unexpectedly accepted")
