"""
pipeline.py
===========

Backend inference core for the Explainable AI Clinical Lung Cancer pipeline.

This module wraps the artifacts produced by ``train_clinical.ipynb``:

    clinical_preprocessor.joblib   -> sklearn ColumnTransformer
    stage_model_bundle.joblib      -> {"model", "positive_label", "threshold"}
    histology_model_bundle.joblib  -> {"model", "explicit_classes", "rare_label",
                                        "min_train_samples"}
    survival_coxph_model.joblib    -> {"model", "feature_columns", "num_imputer",
                                        "cat_imputer", "categorical_cols",
                                        "numeric_cols"}

and exposes a single ``ClinicalPipeline.predict(patient_data)`` entry point that
returns everything the web app / PDF generator need: stage, histology, survival
risk, and a local SHAP explanation for the stage prediction.

The exact column names, feature order, and artifact schemas below were taken
directly from the training notebook so that the saved artifacts can be dropped
into ``artifacts/`` without any renaming.
"""

from __future__ import annotations

import base64
import io
import logging
import os
from dataclasses import dataclass, field
from typing import Any

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger("clinical_pipeline")
if not logger.handlers:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

# --------------------------------------------------------------------------- #
# Schema constants — must mirror train_clinical.ipynb exactly.
# --------------------------------------------------------------------------- #

NUMERIC_FEATURES: list[str] = [
    "AGE",
    "num_treatment_events",
    "num_distinct_tumor_sites",
    "eastern_cancer_oncology_group",
    "karnofsky_performance_score",
]

CATEGORICAL_FEATURES: list[str] = [
    "SEX",
    "RACE",
    "ETHNICITY",
    "SMOKING_STATUS",
    "ever_pdl1_positive",
    "SOURCE_DATASET",
]

# Columns for which the notebook engineers an explicit missingness flag.
MISSINGNESS_SOURCE_COLUMNS: list[str] = [
    "ever_pdl1_positive",
    "eastern_cancer_oncology_group",
    "karnofsky_performance_score",
]

# Exact column order the preprocessor/models were fit on.
FEATURE_COLUMNS: list[str] = (
    NUMERIC_FEATURES[:1]  # AGE
    + CATEGORICAL_FEATURES[:4]  # SEX, RACE, ETHNICITY, SMOKING_STATUS
    + NUMERIC_FEATURES[1:3]  # num_treatment_events, num_distinct_tumor_sites
    + [CATEGORICAL_FEATURES[4]]  # ever_pdl1_positive
    + NUMERIC_FEATURES[3:]  # eastern_cancer_oncology_group, karnofsky_performance_score
    + [CATEGORICAL_FEATURES[5]]  # SOURCE_DATASET
    + [f"missing_{c}" for c in MISSINGNESS_SOURCE_COLUMNS]
)
assert FEATURE_COLUMNS == [
    "AGE",
    "SEX",
    "RACE",
    "ETHNICITY",
    "SMOKING_STATUS",
    "num_treatment_events",
    "num_distinct_tumor_sites",
    "ever_pdl1_positive",
    "eastern_cancer_oncology_group",
    "karnofsky_performance_score",
    "SOURCE_DATASET",
    "missing_ever_pdl1_positive",
    "missing_eastern_cancer_oncology_group",
    "missing_karnofsky_performance_score",
], "FEATURE_COLUMNS drifted from the training notebook schema."

# The survival model is fit on baseline-safe features only (post-baseline /
# longitudinal variables excluded — see SURVIVAL_EXCLUDE_POST_BASELINE in the
# notebook).
SURVIVAL_EXCLUDE_POST_BASELINE: list[str] = ["num_treatment_events"]
SURVIVAL_FEATURE_COLUMNS: list[str] = [
    c for c in FEATURE_COLUMNS if c not in SURVIVAL_EXCLUDE_POST_BASELINE
]

# Values accepted for each categorical field (per PROJECT spec / dataset).
CATEGORICAL_CHOICES: dict[str, list[str]] = {
    "SEX": ["Female", "Male"],
    "RACE": ["White", "Black", "Asian", "Other", "Unknown"],
    "ETHNICITY": ["Non-Spanish; Non-Hispanic", "Spanish; Hispanic", "Unknown"],
    "SMOKING_STATUS": ["Former/Current Smoker", "Never"],
    "ever_pdl1_positive": ["Yes", "No"],
    "SOURCE_DATASET": ["MSK-CHORD", "TCGA", "Other"],
}

REQUIRED_ARTIFACTS = [
    "clinical_preprocessor.joblib",
    "stage_model_bundle.joblib",
    "histology_model_bundle.joblib",
    "survival_coxph_model.joblib",
]


class ArtifactLoadError(RuntimeError):
    """Raised when a required model artifact is missing or malformed."""


class PatientDataError(ValueError):
    """Raised when incoming patient data cannot be coerced into the model schema."""


def humanize_feature_name(raw_name: str) -> str:
    """
    Convert a ColumnTransformer output feature name (e.g. ``"num__AGE"`` or
    ``"cat__SEX_Male"``) into a clinician-readable label
    (e.g. ``"Age"`` or ``"Sex = Male"``).
    """
    name = raw_name
    for prefix in ("num__", "cat__", "remainder__"):
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break

    # One-hot encoded columns look like "SEX_Male" or "ever_pdl1_positive_Yes".
    # Try to split on the longest matching known base column so the "= value"
    # suffix is only applied to genuinely categorical dummies.
    for base_col in CATEGORICAL_FEATURES:
        prefix = f"{base_col}_"
        if name.startswith(prefix):
            value = name[len(prefix) :]
            label = base_col.replace("_", " ").title()
            return f"{label} = {value}"

    if name.startswith("missing_"):
        base = name[len("missing_") :].replace("_", " ").title()
        return f"{base} — value missing"

    return name.replace("_", " ").strip().title()


@dataclass
class ClinicalPipeline:
    """
    Loads all trained artifacts once at startup and exposes a single
    ``predict`` method used by both the web app and the PDF report generator.
    """

    artifacts_dir: str = field(default_factory=lambda: os.environ.get("ARTIFACTS_DIR", "artifacts"))

    def __post_init__(self) -> None:
        logger.info("Loading clinical pipeline artifacts from %s", self.artifacts_dir)
        self._check_artifacts_present()

        self.preprocessor = self._load("clinical_preprocessor.joblib")

        stage_bundle = self._load("stage_model_bundle.joblib")
        self.stage_model = stage_bundle["model"]
        self.stage_positive_label = stage_bundle.get("positive_label")
        self.stage_threshold = float(stage_bundle.get("threshold", 0.5))
        self.stage_classes = list(self.stage_model.classes_)

        hist_bundle = self._load("histology_model_bundle.joblib")
        self.histology_model = hist_bundle["model"]
        self.histology_explicit_classes = hist_bundle.get("explicit_classes", [])
        self.histology_rare_label = hist_bundle.get("rare_label", "Other/rare")

        surv_bundle = self._load("survival_coxph_model.joblib")
        self.survival_model = surv_bundle["model"]
        self.survival_feature_columns = surv_bundle["feature_columns"]
        self.survival_num_imputer = surv_bundle["num_imputer"]
        self.survival_cat_imputer = surv_bundle["cat_imputer"]
        self.survival_categorical_cols = surv_bundle["categorical_cols"]
        self.survival_numeric_cols = surv_bundle["numeric_cols"]

        # Lazily created on first explanation request (cheap for tree models,
        # no need to hold a background dataset in memory at startup).
        self._shap_explainer = None

        logger.info(
            "Pipeline ready. Stage classes=%s | threshold=%.3f | histology classes=%s",
            self.stage_classes,
            self.stage_threshold,
            list(self.histology_model.classes_),
        )

    # ------------------------------------------------------------------ #
    # Artifact loading
    # ------------------------------------------------------------------ #

    def _check_artifacts_present(self) -> None:
        missing = [
            name
            for name in REQUIRED_ARTIFACTS
            if not os.path.isfile(os.path.join(self.artifacts_dir, name))
        ]
        if missing:
            raise ArtifactLoadError(
                "Missing required model artifact(s) in "
                f"'{self.artifacts_dir}': {missing}. Copy the .joblib files "
                "produced by train_clinical.ipynb into this directory "
                "(see README.md) before starting the app."
            )

    def _load(self, filename: str) -> Any:
        path = os.path.join(self.artifacts_dir, filename)
        try:
            return joblib.load(path)
        except Exception as exc:  # pragma: no cover - defensive
            raise ArtifactLoadError(f"Failed to load artifact '{path}': {exc}") from exc

    # ------------------------------------------------------------------ #
    # Input handling
    # ------------------------------------------------------------------ #

    def _coerce_patient_frame(self, patient_data: dict[str, Any]) -> pd.DataFrame:
        """
        Validate and coerce a raw patient dict into a single-row DataFrame with
        exactly ``FEATURE_COLUMNS`` in the correct order and dtypes, deriving
        the missingness indicator columns automatically.
        """
        if not isinstance(patient_data, dict):
            raise PatientDataError("patient_data must be a dict of feature -> value")

        row: dict[str, Any] = {}

        for col in NUMERIC_FEATURES:
            value = patient_data.get(col, None)
            if value in ("", None):
                row[col] = np.nan
            else:
                try:
                    row[col] = float(value)
                except (TypeError, ValueError) as exc:
                    raise PatientDataError(f"Field '{col}' must be numeric, got {value!r}") from exc

        for col in CATEGORICAL_FEATURES:
            value = patient_data.get(col, None)
            if value in ("", None):
                row[col] = np.nan
            else:
                row[col] = str(value)

        # Derive missingness flags from whatever was actually supplied,
        # exactly mirroring the notebook's feature-engineering step.
        for col in MISSINGNESS_SOURCE_COLUMNS:
            row[f"missing_{col}"] = int(pd.isna(row[col]))

        frame = pd.DataFrame([row], columns=FEATURE_COLUMNS)
        return frame

    # ------------------------------------------------------------------ #
    # Prediction sub-routines
    # ------------------------------------------------------------------ #

    def _predict_stage(self, processed: pd.DataFrame) -> dict[str, Any]:
        proba = self.stage_model.predict_proba(processed)[0]
        proba_by_class = {cls: float(p) for cls, p in zip(self.stage_classes, proba)}

        if self.stage_positive_label and len(self.stage_classes) == 2:
            pos_idx = self.stage_classes.index(self.stage_positive_label)
            pos_proba = float(proba[pos_idx])
            predicted_label = (
                self.stage_positive_label
                if pos_proba >= self.stage_threshold
                else next(c for c in self.stage_classes if c != self.stage_positive_label)
            )
        else:
            pos_proba = None
            predicted_label = self.stage_classes[int(np.argmax(proba))]

        return {
            "predicted_label": predicted_label,
            "probabilities": proba_by_class,
            "positive_label": self.stage_positive_label,
            "positive_label_probability": pos_proba,
            "decision_threshold": self.stage_threshold,
        }

    def _predict_histology(self, processed: pd.DataFrame) -> dict[str, Any]:
        proba = self.histology_model.predict_proba(processed)[0]
        classes = list(self.histology_model.classes_)
        proba_by_class = {cls: float(p) for cls, p in zip(classes, proba)}
        predicted_label = classes[int(np.argmax(proba))]
        return {
            "predicted_label": predicted_label,
            "probabilities": proba_by_class,
            "explicit_classes": self.histology_explicit_classes,
            "rare_label": self.histology_rare_label,
        }

    def _predict_survival(self, raw_row: pd.DataFrame) -> dict[str, Any]:
        surv_raw = raw_row[SURVIVAL_FEATURE_COLUMNS].copy()

        if self.survival_numeric_cols:
            surv_raw[self.survival_numeric_cols] = self.survival_num_imputer.transform(
                surv_raw[self.survival_numeric_cols]
            )
        if self.survival_categorical_cols:
            surv_raw[self.survival_categorical_cols] = self.survival_cat_imputer.transform(
                surv_raw[self.survival_categorical_cols]
            )

        surv_encoded = pd.get_dummies(surv_raw, columns=self.survival_categorical_cols, drop_first=True)
        surv_encoded = surv_encoded.reindex(columns=self.survival_feature_columns, fill_value=0)
        surv_encoded = surv_encoded.astype(np.float64)

        risk_score = float(self.survival_model.predict(surv_encoded)[0])

        survival_curve: list[dict[str, float]] | None = None
        median_survival_months: float | None = None
        try:
            surv_fn = self.survival_model.predict_survival_function(surv_encoded)[0]
            times = surv_fn.x
            probs = surv_fn.y
            # Downsample to ~50 points for a lightweight chart payload.
            step = max(1, len(times) // 50)
            survival_curve = [
                {"months": float(t), "survival_probability": float(p)}
                for t, p in zip(times[::step], probs[::step])
            ]
            below_half = np.where(probs <= 0.5)[0]
            if len(below_half) > 0:
                median_survival_months = float(times[below_half[0]])
        except Exception as exc:  # pragma: no cover - sksurv API can vary by version
            logger.warning("Could not compute survival curve: %s", exc)

        return {
            "risk_score": risk_score,
            "median_survival_months": median_survival_months,
            "survival_curve": survival_curve,
        }

    def _explain_stage(self, processed: pd.DataFrame, top_k: int = 5) -> dict[str, Any]:
        """
        Local SHAP explanation of the stage prediction for this one patient,
        following the same TreeExplainer + multiclass-normalization approach
        used in the training notebook.
        """
        import shap  # deferred import — heavy and only needed for explanations

        if self._shap_explainer is None:
            self._shap_explainer = shap.TreeExplainer(self.stage_model)

        raw_values = self._shap_explainer.shap_values(processed, check_additivity=False)
        n_classes = len(self.stage_classes)
        vals = self._normalize_shap_output(raw_values, n_classes)  # (1, n_features, n_classes)

        if self.stage_positive_label and self.stage_positive_label in self.stage_classes:
            class_idx = self.stage_classes.index(self.stage_positive_label)
        else:
            class_idx = int(np.argmax(self.stage_model.predict_proba(processed)[0]))

        contributions = vals[0, :, class_idx]
        feature_names = list(processed.columns)
        readable_names = [humanize_feature_name(f) for f in feature_names]

        order = np.argsort(contributions)
        top_negative_idx = order[:top_k]  # most negative (push away from positive class)
        top_positive_idx = order[::-1][:top_k]  # most positive (push toward positive class)

        top_positive = [
            {"feature": readable_names[i], "shap_value": float(contributions[i])}
            for i in top_positive_idx
        ]
        top_negative = [
            {"feature": readable_names[i], "shap_value": float(contributions[i])}
            for i in top_negative_idx
        ]

        base = np.asarray(self._shap_explainer.expected_value).reshape(-1)
        base_value = float(base[class_idx] if len(base) > 1 else base[0])

        waterfall_b64 = self._render_waterfall_png(
            contributions=contributions,
            base_value=base_value,
            data_row=processed.iloc[0].to_numpy(),
            feature_names=readable_names,
            class_label=self.stage_classes[class_idx],
        )

        return {
            "explained_class": self.stage_classes[class_idx],
            "base_value": base_value,
            "top_positive_features": top_positive,
            "top_negative_features": top_negative,
            "waterfall_plot_base64": waterfall_b64,
        }

    @staticmethod
    def _normalize_shap_output(values: Any, n_classes: int) -> np.ndarray:
        """Mirror ``normalize_shap_output`` from the training notebook."""
        if isinstance(values, list):
            return np.stack(values, axis=-1)
        arr = np.asarray(values)
        if arr.ndim == 3:
            if arr.shape[-1] == n_classes:
                return arr
            if arr.shape[1] == n_classes:
                return np.moveaxis(arr, 1, -1)
        if arr.ndim == 2 and n_classes == 2:
            # Some SHAP versions return only the positive-class matrix for
            # binary classifiers; synthesize the symmetric negative-class one.
            return np.stack([-arr, arr], axis=-1)
        raise ValueError(f"Unsupported SHAP output shape: {arr.shape}")

    @staticmethod
    def _render_waterfall_png(
        contributions: np.ndarray,
        base_value: float,
        data_row: np.ndarray,
        feature_names: list[str],
        class_label: str,
        max_display: int = 12,
    ) -> str:
        """Render a SHAP waterfall plot to a base64-encoded PNG string."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import shap

        explanation = shap.Explanation(
            values=contributions,
            base_values=base_value,
            data=data_row,
            feature_names=feature_names,
        )
        shap.plots.waterfall(explanation, max_display=max_display, show=False)
        fig = plt.gcf()
        fig.suptitle(f"Feature contributions — {class_label}", fontsize=11, y=1.02)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("ascii")

    # ------------------------------------------------------------------ #
    # Public entry point
    # ------------------------------------------------------------------ #

    def predict(self, patient_data: dict[str, Any]) -> dict[str, Any]:
        """
        Run the full clinical inference pipeline for one patient.

        Parameters
        ----------
        patient_data:
            Dict of raw feature values keyed by the fields listed in
            ``FEATURE_COLUMNS`` (missingness flags are derived automatically —
            do not pass them in). Missing/unknown fields may be omitted or set
            to ``None``/``""``; the fitted imputers handle them.

        Returns
        -------
        dict with keys ``stage``, ``histology``, ``survival``, ``explanation``,
        and the ``input_features`` actually used (post-coercion, pre-transform)
        for audit/reporting purposes.
        """
        raw_row = self._coerce_patient_frame(patient_data)

        try:
            processed = pd.DataFrame(
                self.preprocessor.transform(raw_row),
                columns=self.preprocessor.get_feature_names_out(),
                index=raw_row.index,
            )
        except Exception as exc:
            logger.exception("Preprocessing failed for input: %s", patient_data)
            raise PatientDataError(f"Could not preprocess patient data: {exc}") from exc

        try:
            stage_result = self._predict_stage(processed)
        except Exception:
            logger.exception("Stage inference failed")
            raise

        try:
            histology_result = self._predict_histology(processed)
        except Exception:
            logger.exception("Histology inference failed")
            raise

        try:
            survival_result = self._predict_survival(raw_row)
        except Exception:
            logger.exception("Survival inference failed")
            survival_result = {
                "risk_score": None,
                "median_survival_months": None,
                "survival_curve": None,
                "error": "Survival estimate unavailable for this input.",
            }

        try:
            explanation = self._explain_stage(processed)
        except Exception:
            logger.exception("SHAP explanation failed")
            explanation = {
                "explained_class": stage_result["predicted_label"],
                "top_positive_features": [],
                "top_negative_features": [],
                "waterfall_plot_base64": None,
                "error": "Explanation unavailable for this input.",
            }

        return {
            "input_features": raw_row.iloc[0].to_dict(),
            "stage": stage_result,
            "histology": histology_result,
            "survival": survival_result,
            "explanation": explanation,
        }