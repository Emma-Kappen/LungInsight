from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .ingestion import load_msk_cancer_type_cohort
from .inference import load_artifacts
from .training import (
    CANCER_FEATURE_ORDER,
    ModelEligibilityError,
    build_cancer_type_eligibility_report,
    fit_cancer_type_model,
    save_artifact_bundle,
    write_report,
)


def train_authorized_msk_classifier(
    *,
    datasets_root: str | Path,
    artifact_path: str | Path,
    report_path: str | Path,
    random_state: int = 42,
) -> dict[str, Any]:
    """Train, persist, reload, and verify the authorized MSK-scoped classifier."""
    datasets_root = Path(datasets_root)
    cohort = load_msk_cancer_type_cohort(datasets_root / "msk_chord_2024")
    eligibility = build_cancer_type_eligibility_report(
        cohort,
        authorization_status="authorized",
        source_label_provenance="source_recorded",
    )
    if not eligibility["eligible_for_training"]:
        raise ModelEligibilityError(
            f"Cancer-type training is ineligible: {eligibility['blockers']}"
        )

    cancer_type, evaluation = fit_cancer_type_model(
        cohort,
        prediction_baseline_definition="initial_pathologic_diagnosis",
        use_authorized=True,
        random_state=random_state,
    )
    test_metrics = evaluation["test"]
    if test_metrics["log_loss"] >= test_metrics["prevalence_only_baseline"]["log_loss"]:
        raise ModelEligibilityError(
            "The selected classifier does not improve held-out log loss over prevalence-only prediction."
        )

    artifact_path = Path(artifact_path)
    existing_survival = None
    existing_provenance: dict[str, Any] = {}
    if artifact_path.exists():
        import joblib

        existing_bundle = joblib.load(artifact_path)
        if not isinstance(existing_bundle, dict) or existing_bundle.get("schema_version") != "1.0.0":
            raise ModelEligibilityError("The existing Clinical artifact bundle has an unsupported schema.")
        existing_survival = existing_bundle.get("survival")
        existing_provenance = dict(existing_bundle.get("provenance") or {})

    examples = pd.DataFrame(
        [{"sex_at_baseline": "female"}, {"sex_at_baseline": "male"}],
        columns=CANCER_FEATURE_ORDER,
    )
    expected_probabilities = cancer_type["estimator"].predict_proba(examples)
    provenance = {
        **existing_provenance,
        "cancer_type_training": {
            "source": "MSK-CHORD",
            "authorization_status": "authorized",
            "random_state": random_state,
            "eligibility": eligibility,
        },
    }
    save_artifact_bundle(
        artifact_path,
        cancer_type=cancer_type,
        survival=existing_survival,
        provenance=provenance,
    )
    reloaded = load_artifacts(artifact_path)
    reloaded_cancer_type = reloaded["cancer_type"]
    if reloaded_cancer_type is None:
        raise RuntimeError("Saved cancer-type artifact was unavailable after reload.")
    actual_probabilities = reloaded_cancer_type["estimator"].predict_proba(examples)
    round_trip_equal = bool(np.allclose(expected_probabilities, actual_probabilities, rtol=0, atol=0))
    if not round_trip_equal:
        raise RuntimeError("Reloaded cancer-type artifact changed fixed-example probabilities.")

    report = {
        "report_schema_version": "1.0.0",
        "model_status": "trained_research_only",
        "eligibility": eligibility,
        "model_metadata": cancer_type["metadata"],
        "evaluation": evaluation,
        "artifact": {
            "path": artifact_path.name,
            "round_trip_fixed_example_probabilities_identical": round_trip_equal,
            "fixed_examples": ["female", "male"],
        },
        "limitations": [
            "Probabilities are conditional on explicit LUAD/LUSC cancer cases in the MSK-CHORD training population.",
            "The model uses source-recorded sex only; it is not clinically validated and has limited discrimination.",
            "TCGA cohorts were not pooled for classifier fitting because source/cohort membership is target-associated.",
        ],
    }
    write_report(report_path, report)
    return report


def main() -> None:
    clinical_dir = Path(__file__).resolve().parent
    repo_root = clinical_dir.parents[2]
    parser = argparse.ArgumentParser(
        description="Train and verify the authorized MSK-CHORD LUAD/LUSC research classifier."
    )
    parser.add_argument("--datasets-root", type=Path, default=repo_root / "datasets")
    parser.add_argument(
        "--artifact",
        type=Path,
        default=clinical_dir / "models" / "clinical-research-v1.joblib",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=clinical_dir / "reports" / "cancer-type-evaluation.json",
    )
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    report = train_authorized_msk_classifier(
        datasets_root=args.datasets_root,
        artifact_path=args.artifact,
        report_path=args.report,
        random_state=args.random_state,
    )
    print(
        "Cancer-type classifier trained: "
        f"held-out log loss {report['evaluation']['test']['log_loss']:.4f}; "
        f"artifact {args.artifact}"
    )


if __name__ == "__main__":
    main()
