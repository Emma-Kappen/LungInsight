"""Stage 04 inference only: convert Stage 03 attribute predictions to radiologist-rating estimates."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
from pipeline_schema import get_required, load_schema, check_meta_schema_version

SCHEMA = load_schema(__file__)
STAGE04_ORDINAL_SCHEMA = get_required(SCHEMA, "stage_04_malignancy_classifier", expected_type=dict)
FEATURES = list(get_required(SCHEMA, str(STAGE04_ORDINAL_SCHEMA["feature_names_ref"]), expected_type=list))
INPUT_RATING_RANGE = tuple(get_required(STAGE04_ORDINAL_SCHEMA, "input_rating_range", expected_type=list))
RATING_CLASSES = tuple(get_required(STAGE04_ORDINAL_SCHEMA, "rating_classes", expected_type=list))
RATING_GROUPS = get_required(STAGE04_ORDINAL_SCHEMA, "rating_groups", expected_type=dict)
CONFIDENCE_THRESHOLD = float(get_required(STAGE04_ORDINAL_SCHEMA, "confidence_threshold"))
CONFIDENCE_FLAGS = get_required(STAGE04_ORDINAL_SCHEMA, "confidence_flags", expected_type=dict)
MODEL_VERSION = str(get_required(STAGE04_ORDINAL_SCHEMA, "model_version"))
LIMITATIONS = list(get_required(STAGE04_ORDINAL_SCHEMA, "limitations", expected_type=list))
if len(RATING_CLASSES) < 2 or len(set(RATING_CLASSES)) != len(RATING_CLASSES):
    raise RuntimeError("meta_schema.json stage_04_malignancy_classifier.rating_classes must contain unique rating classes.")
if set().union(*(set(v) for v in RATING_GROUPS.values())) != set(RATING_CLASSES):
    raise RuntimeError("meta_schema.json Stage 04 rating_groups must cover each rating class exactly at least once.")
V2_ROOT = Path(__file__).resolve().parents[2]
if str(V2_ROOT) not in sys.path:
    sys.path.insert(0, str(V2_ROOT))
def load_ordinal_helpers():
    """Load optional ordinal-rating implementation only when inference is requested."""
    try:
        from backend.models.malignancy.features import ATTRIBUTE_NAMES, SCHEMA_VERSION, transform_features
        from backend.models.malignancy.ordinal_model import RidgeOrdinalLogit
    except ImportError as exc:
        raise ImportError(
            "Could not import optional ordinal inference helpers from "
            "backend/models/malignancy/features.py and ordinal_model.py. Restore those project files. "
            "This alternative is distinct from the primary joblib model in 04_malignancy_prediction.py."
        ) from exc
    if list(ATTRIBUTE_NAMES) != FEATURES:
        raise RuntimeError(
            f"Stage 04 ordinal feature mismatch: implementation has {list(ATTRIBUTE_NAMES)}, "
            f"meta_schema.json requires {FEATURES}."
        )
    return ATTRIBUTE_NAMES, SCHEMA_VERSION, transform_features, RidgeOrdinalLogit

def infer_candidate(candidate, ck, *, attribute_names, schema_version, transform_features, RidgeOrdinalLogit):
    attrs=candidate.get("attributes",{})
    row={}
    for name in attribute_names:
        value=attrs.get(name)
        if isinstance(value,dict):
            value=value.get("expected_rating",value.get("rating_original_scale",value.get("rating_rounded")))
        try: row[name]=float(value)
        except (TypeError,ValueError): row[name]=None
    missing=[a for a,v in row.items() if v is None or not np.isfinite(v)]
    if missing:
        return {"status":"insufficient_inputs","missing_attributes":missing,"model_version":MODEL_VERSION,
                "input_source":"stage03_predicted","limitations":LIMITATIONS}
    lower, upper = float(INPUT_RATING_RANGE[0]), float(INPUT_RATING_RANGE[1])
    bad=[a for a,v in row.items() if not lower <= v <= upper]
    if bad:
        return {"status":"invalid_inputs","out_of_range_attributes":bad,"model_version":MODEL_VERSION,
                "input_source":"stage03_predicted","limitations":LIMITATIONS}
    model=RidgeOrdinalLogit.from_dict(ck["model"])
    X=transform_features([row],ck["feature_mean"],ck["feature_scale"])
    p=np.asarray(model.predict_proba(X)[0], dtype=float)
    if p.shape != (len(RATING_CLASSES),) or not np.isfinite(p).all():
        raise ValueError(
            f"Ordinal checkpoint output must contain {len(RATING_CLASSES)} finite class probabilities; got {p.shape}. "
            "Check checkpoint/model compatibility without modifying the checkpoint."
        )
    class_index={rating:i for i,rating in enumerate(RATING_CLASSES)}
    group={name:float(sum(p[class_index[rating]] for rating in ratings)) for name,ratings in RATING_GROUPS.items()}
    max_group=max(group.values())
    return {"status":"ok","rating_distribution":{str(rating):float(p[class_index[rating]]) for rating in RATING_CLASSES},
      "rating_group_probabilities":group,"expected_rating":float(p@np.asarray(RATING_CLASSES, dtype=float)),
      "confidence_flag":CONFIDENCE_FLAGS["uncertain"] if max_group<CONFIDENCE_THRESHOLD else CONFIDENCE_FLAGS["dominant"],
      "model_version":MODEL_VERSION,"feature_schema":ck.get("schema_version",schema_version),
      "input_source":"stage03_predicted","validation_basis":"human_annotation_validation_only",
      "limitations":LIMITATIONS}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--predictions",required=True, help="Stage 03 predictions.json")
    ap.add_argument("--checkpoint",required=True, help="Existing ordinal_v1 JSON checkpoint; no training is performed")
    ap.add_argument("--out",required=True)
    args=ap.parse_args()
    attribute_names, model_schema_version, transform_features, RidgeOrdinalLogit = load_ordinal_helpers()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    predictions_path = Path(args.predictions).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Ordinal Stage 04 checkpoint not found: {checkpoint_path}. Restore the original JSON checkpoint or use 04_malignancy_prediction.py with its joblib bundle.")
    if not predictions_path.is_file():
        raise FileNotFoundError(f"Stage 03 predictions not found: {predictions_path}. Run Stage 03 first.")
    try:
        ck=json.loads(checkpoint_path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed ordinal checkpoint {checkpoint_path} at line {exc.lineno}: {exc.msg}") from exc
    if not isinstance(ck, dict):
        raise ValueError(f"Ordinal checkpoint must be a JSON object: {checkpoint_path}")
    if ck.get("schema_version")!=model_schema_version:
        raise SystemExit(f"Checkpoint feature schema mismatch: checkpoint={ck.get('schema_version')!r}, expected={model_schema_version!r} from the existing ordinal inference implementation. Restore the compatible checkpoint; do not edit its version.")
    try:
        raw=json.loads(predictions_path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed Stage 03 predictions JSON {predictions_path} at line {exc.lineno}: {exc.msg}") from exc
    if isinstance(raw, dict):
        check_meta_schema_version(raw, SCHEMA, context=str(predictions_path), require=False)
    if isinstance(raw,list): candidates=raw; wrapper=None
    elif isinstance(raw,dict):
        key="predictions" if isinstance(raw.get("predictions"),list) else "candidates"
        candidates=raw.get(key)
        if not isinstance(candidates,list): raise SystemExit("Unrecognized predictions JSON: expected list or predictions/candidates list")
        wrapper=(key,raw)
    else: raise SystemExit("Predictions JSON must be a list or object")
    if not candidates:
        raise ValueError("Stage 03 predictions contain no candidates")
    outputs=[]
    for c in candidates:
        if not isinstance(c, dict):
            raise ValueError(f"Stage 03 candidate at index {len(outputs)} is not a JSON object")
        result={"patch_id":c.get("patch_id"),"candidate_id":c.get("candidate_id"),"stage04":infer_candidate(c,ck,attribute_names=attribute_names,schema_version=model_schema_version,transform_features=transform_features,RidgeOrdinalLogit=RidgeOrdinalLogit)}
        outputs.append(result)
    result={"model_version":MODEL_VERSION,"meta_schema_version":SCHEMA["schema_version"],"input_source":"stage03_predicted","results":outputs,
            "warning":"These are estimated radiologist-rating distributions, not cancer probabilities."}
    output_path = Path(args.out).expanduser().resolve()
    output_path.parent.mkdir(parents=True,exist_ok=True)
    output_path.write_text(json.dumps(result,indent=2,allow_nan=False),encoding="utf-8")
    print(f"Wrote {len(outputs)} Stage 04 results to {args.out}")
if __name__=="__main__": main()
