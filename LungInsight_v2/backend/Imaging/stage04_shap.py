"""SHAP explanations for the frozen Stage 04 rating regressor."""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Any

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_training_background(
    table_path: Path,
    feature_names: list[str],
    *,
    split_column: str,
    training_split_value: str,
    sample_size: int,
    random_seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load a reproducible sample from the declared training rows only."""
    if sample_size <= 0:
        raise ValueError(f"SHAP background sample_size must be positive, got {sample_size}.")
    if not table_path.is_file():
        raise FileNotFoundError(f"Stage 04 SHAP training table is missing: {table_path}")

    training_rows: list[list[float]] = []
    with table_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        required = [split_column, *feature_names]
        missing = [name for name in required if name not in fieldnames]
        if missing:
            raise ValueError(
                f"SHAP training table {table_path} is missing required columns {missing}; "
                f"found {fieldnames}."
            )
        for line_number, row in enumerate(reader, start=2):
            if row.get(split_column) != training_split_value:
                continue
            try:
                values = [float(row[name]) for name in feature_names]
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid Stage 04 training feature at {table_path}:{line_number}."
                ) from exc
            if not np.isfinite(values).all():
                raise ValueError(
                    f"Non-finite Stage 04 training feature at {table_path}:{line_number}."
                )
            training_rows.append(values)

    if not training_rows:
        raise ValueError(
            f"No rows with {split_column}={training_split_value!r} were found in {table_path}."
        )
    rows = np.asarray(training_rows, dtype=np.float64)
    selected_count = min(sample_size, len(rows))
    selected_indices = np.sort(
        np.random.default_rng(random_seed).choice(len(rows), size=selected_count, replace=False)
    )
    background = rows[selected_indices]
    metadata = {
        "path": str(table_path.resolve()),
        "sha256": sha256_file(table_path),
        "split_column": split_column,
        "split_value": training_split_value,
        "training_row_count": int(len(rows)),
        "sample_row_indices_zero_based": selected_indices.astype(int).tolist(),
        "sample_size": int(selected_count),
        "random_seed": int(random_seed),
        "feature_names_in_order": list(feature_names),
        "target_column_excluded": True,
    }
    return background, metadata


def explain_regression(
    model: Any,
    background: np.ndarray,
    values: np.ndarray,
    *,
    additivity_tolerance: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Compute interventional Tree SHAP values and verify raw-score additivity."""
    try:
        import shap
    except ImportError as exc:
        raise RuntimeError(
            "SHAP is required for Stage 04 explanations but is not installed in the active "
            "environment. Install the declared project dependency with "
            "`python -m pip install -r requirements.txt`."
        ) from exc

    if background.ndim != 2 or values.ndim != 2 or background.shape[1] != values.shape[1]:
        raise ValueError(
            f"Invalid SHAP input shapes: background={background.shape}, values={values.shape}."
        )
    if not np.isfinite(background).all() or not np.isfinite(values).all():
        raise ValueError("SHAP background and inference features must contain only finite values.")
    if additivity_tolerance <= 0:
        raise ValueError("SHAP additivity_tolerance must be positive.")

    explainer = shap.TreeExplainer(
        model,
        data=background,
        feature_perturbation="interventional",
        model_output="raw",
    )
    explanation = explainer(values, check_additivity=True)
    shap_values = np.asarray(explanation.values, dtype=np.float64)
    if shap_values.shape != values.shape or not np.isfinite(shap_values).all():
        raise ValueError(
            f"Unexpected SHAP result shape or non-finite contribution values: "
            f"expected {values.shape}, got {shap_values.shape}."
        )
    base_values = np.asarray(explanation.base_values, dtype=np.float64).reshape(-1)
    if base_values.size == 1:
        base_values = np.repeat(base_values, values.shape[0])
    if base_values.size != values.shape[0] or not np.isfinite(base_values).all():
        raise ValueError(
            f"Unexpected SHAP base-value shape: expected {values.shape[0]}, got {base_values.shape}."
        )
    raw_predictions = np.asarray(model.predict(values), dtype=np.float64).reshape(-1)
    residuals = base_values + shap_values.sum(axis=1) - raw_predictions
    max_abs_error = float(np.max(np.abs(residuals))) if residuals.size else 0.0
    if max_abs_error > additivity_tolerance:
        raise RuntimeError(
            f"Stage 04 SHAP additivity check failed: maximum absolute residual "
            f"{max_abs_error:.8g} exceeds tolerance {additivity_tolerance:.8g}."
        )
    return shap_values, base_values, {
        "explainer": "shap.TreeExplainer",
        "feature_perturbation": "interventional",
        "model_output": "raw_regression_rating",
        "additivity_tolerance": float(additivity_tolerance),
        "max_abs_additivity_error": max_abs_error,
        "shap_version": str(shap.__version__),
    }


def explain_linear_ranker(
    model: Any,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Explain the ranker's decision score with exact linear attributions."""
    if not hasattr(model, "shap_values"):
        raise TypeError("The selected ranking model does not expose linear SHAP values.")
    shap_values, base_values = model.shap_values(values)
    shap_values = np.asarray(shap_values, dtype=np.float64)
    base_values = np.asarray(base_values, dtype=np.float64).reshape(-1)
    if shap_values.shape != values.shape or not np.isfinite(shap_values).all():
        raise ValueError("Ranking SHAP values have an invalid shape or contain non-finite values.")
    if base_values.shape != (values.shape[0],) or not np.isfinite(base_values).all():
        raise ValueError("Ranking SHAP base values have an invalid shape or contain non-finite values.")
    predictions = np.asarray(model.predict(values), dtype=np.float64).reshape(-1)
    residuals = base_values + shap_values.sum(axis=1) - predictions
    max_abs_error = float(np.max(np.abs(residuals))) if residuals.size else 0.0
    if max_abs_error > 1e-10:
        raise RuntimeError(f"Ranking SHAP additivity check failed: {max_abs_error:.8g}")
    return shap_values, base_values, {
        "explainer": "exact_linear_attribution",
        "model_output": "raw_malignancy_priority_score",
        "max_abs_additivity_error": max_abs_error,
    }
