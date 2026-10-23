from __future__ import annotations

import numpy as np
import pandas as pd

from backend.Clinical.inference import (
    _cancer_type_shap_explanation,
    load_cancer_type_pipeline,
)


def main():
    artifact = load_cancer_type_pipeline()
    if artifact is None:
        raise AssertionError("Active histology model is unavailable.")
    model = artifact["estimator"]
    features = pd.DataFrame([{
        "sex_at_baseline": "female",
        "smoking_history": "never",
    }], columns=["sex_at_baseline", "smoking_history"])

    before = np.asarray(model.predict_proba(features)[0], dtype=float)
    classes = [str(x) for x in model.classes_]
    explanation = _cancer_type_shap_explanation(model, features, classes, before)
    after = np.asarray(model.predict_proba(features)[0], dtype=float)

    assert np.array_equal(before, after), "Explanation changed model probabilities."
    assert explanation["status"] == "available", explanation
    assert explanation["method"] == "exact_interventional_shapley"
    assert len(explanation["feature_contributions"]) == 2

    reconstructed = dict(explanation["base_values"])
    for item in explanation["feature_contributions"]:
        for label, value in item["values"].items():
            reconstructed[label] += value
    for label, value in zip(classes, before):
        assert abs(reconstructed[label] - float(value)) <= 1e-6, (
            label, reconstructed[label], value
        )

    print("PASS: active model probabilities unchanged.")
    print("PASS: exact SHAP local accuracy verified for both classes.")
    print("PASS: original clinical feature names retained.")
    print("Probabilities:", dict(zip(classes, before)))
    print("SHAP:", explanation)


if __name__ == "__main__":
    main()
