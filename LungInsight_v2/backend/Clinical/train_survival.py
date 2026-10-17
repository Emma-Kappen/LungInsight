from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .ingestion import load_tcga_survival_cohort
from .inference import load_artifacts
from .training import (
    SURVIVAL_FEATURE_ORDER,
    ModelEligibilityError,
    evaluate_cross_cohort_survival,
    fit_survival_model,
    save_artifact_bundle,
    write_report,
)


def train_tcga_survival_model(
    *,
    datasets_root: str | Path,
    artifact_path: str | Path,
    report_path: str | Path,
    cross_cohort_report_path: str | Path,
    random_state: int = 42,
) -> dict[str, Any]:
    """Train, persist, reload, and verify the TCGA-scoped random survival forest.

    The existing cancer-type artifact in the bundle is preserved.
    """
    root = Path(datasets_root)
    rows = pd.concat(
        [
            load_tcga_survival_cohort(root / "TCGA-LUAD", "TCGA-LUAD"),
            load_tcga_survival_cohort(root / "TCGA-LUSC", "TCGA-LUSC"),
        ],
        ignore_index=True,
    )
    survival, report = fit_survival_model(rows, use_authorized=True, random_state=random_state)
    evaluation = report["evaluation"]
    if not evaluation["concordance_index"] > 0.5:
        raise ModelEligibilityError(
            "The held-out concordance index does not exceed random ordering (0.5); the model is not deployed."
        )

    artifact_path = Path(artifact_path)
    existing_cancer_type = None
    existing_provenance: dict[str, Any] = {}
    if artifact_path.exists():
        import joblib

        existing_bundle = joblib.load(artifact_path)
        if not isinstance(existing_bundle, dict) or existing_bundle.get("schema_version") != "1.0.0":
            raise ModelEligibilityError("The existing Clinical artifact bundle has an unsupported schema.")
        existing_cancer_type = existing_bundle.get("cancer_type")
        existing_provenance = dict(existing_bundle.get("provenance") or {})

    examples = pd.DataFrame(
        [
            {"age_at_baseline_years": 55.0, "sex_at_baseline": "female"},
            {"age_at_baseline_years": 70.0, "sex_at_baseline": "male"},
        ],
        columns=SURVIVAL_FEATURE_ORDER,
    )
    expected_risk = survival["estimator"].predict(examples)
    save_artifact_bundle(
        artifact_path,
        cancer_type=existing_cancer_type,
        survival=survival,
        provenance={**existing_provenance, "cohorts": ["TCGA-LUAD", "TCGA-LUSC"]},
    )
    reloaded = load_artifacts(artifact_path)["survival"]
    if reloaded is None:
        raise RuntimeError("Saved survival artifact was unavailable after reload.")
    if not np.array_equal(expected_risk, reloaded["estimator"].predict(examples)):
        raise RuntimeError("Reloaded survival artifact changed fixed-example risk scores.")

    write_report(report_path, report)
    cross_cohort = [
        evaluate_cross_cohort_survival(rows, training_source=train, test_source=test)
        for train, test in (("TCGA-LUAD", "TCGA-LUSC"), ("TCGA-LUSC", "TCGA-LUAD"))
    ]
    write_report(
        cross_cohort_report_path,
        {"status": "transport_diagnostic_not_external_clinical_validation", "results": cross_cohort},
    )
    return report


def main() -> None:
    clinical_dir = Path(__file__).resolve().parent
    repo_root = clinical_dir.parents[2]
    parser = argparse.ArgumentParser(
        description="Train and verify the TCGA-LUAD/LUSC overall-survival research model."
    )
    parser.add_argument("--datasets-root", type=Path, default=repo_root / "datasets")
    parser.add_argument(
        "--artifact", type=Path, default=clinical_dir / "models" / "clinical-research-v1.joblib"
    )
    parser.add_argument(
        "--report", type=Path, default=clinical_dir / "reports" / "survival-evaluation.json"
    )
    parser.add_argument(
        "--cross-cohort-report",
        type=Path,
        default=clinical_dir / "reports" / "cross-cohort-evaluation.json",
    )
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    report = train_tcga_survival_model(
        datasets_root=args.datasets_root,
        artifact_path=args.artifact,
        report_path=args.report,
        cross_cohort_report_path=args.cross_cohort_report,
        random_state=args.random_state,
    )
    evaluation = report["evaluation"]
    print(
        "Survival model trained: "
        f"held-out C-index {evaluation['concordance_index']:.3f}; artifact {args.artifact}"
    )


if __name__ == "__main__":
    main()
