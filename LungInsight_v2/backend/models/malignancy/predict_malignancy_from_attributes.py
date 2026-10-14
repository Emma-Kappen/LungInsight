#!/usr/bin/env python3
"""Predict a LIDC malignancy/suspicion rating from six attribute-model outputs only."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path

import joblib
import numpy as np

FEATURES = ["subtlety", "sphericity", "margin", "lobulation", "spiculation", "texture"]


def predict_malignancy(attributes: dict[str, float], model_path: str | Path) -> dict:
    """Return continuous and rounded prediction. No CT or image input is accepted."""
    missing = [name for name in FEATURES if name not in attributes]
    extra = [name for name in attributes if name not in FEATURES]
    if missing or extra:
        raise ValueError(f"Expected exactly these six attributes. Missing={missing}, extra={extra}")
    values = np.asarray([float(attributes[name]) for name in FEATURES], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("All six attribute values must be finite numbers")
    if ((values < 1.0) | (values > 5.0)).any():
        raise ValueError("All six attributes must be on the expected 1..5 ordinal-rating scale")

    bundle = joblib.load(model_path)
    if bundle.get("feature_names") != FEATURES:
        raise ValueError(f"Feature-order mismatch in model bundle: {bundle.get('feature_names')}")
    model = bundle["model"]
    prediction = float(np.clip(model.predict(values.reshape(1, -1))[0], 1.0, 5.0))
    rounded = int(np.clip(math.floor(prediction + 0.5), 1, 5))
    return {
        "predicted_malignancy_rating_continuous": prediction,
        "predicted_malignancy_rating_rounded": rounded,
        "rating_scale": "1..5",
        "target_semantics": "LIDC reader-assigned malignancy/suspicion rating; not pathology-confirmed cancer probability",
        "input_features": {name: float(attributes[name]) for name in FEATURES},
        "ct_or_image_used_by_malignancy_predictor": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("backend/models/malignancy/stage03_input_v2/malignancy_from_attributes.joblib"))
    for name in FEATURES:
        parser.add_argument(f"--{name}", required=True, type=float)
    args = parser.parse_args()
    attrs = {name: getattr(args, name) for name in FEATURES}
    result = predict_malignancy(attrs, args.model)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
