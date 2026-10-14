#!/usr/bin/env python3
"""SHAP explanation for the frozen Stage 04 ExtraTrees malignancy-rating model.

Consumes Stage 04 malignancy_predictions.json plus the exact joblib model; never
reads CT/H5 or retrains models. Target is a LIDC reader suspicion rating, not
pathology-confirmed cancer probability.
"""
import argparse, json, warnings
from pathlib import Path
import joblib
import numpy as np

FEATURES=["subtlety","sphericity","margin","lobulation","spiculation","texture"]
V2=Path(__file__).resolve().parents[2]
DEFAULT_MODEL=V2/"backend"/"models"/"malignancy"/"xml_nodule_level_v1"/"malignancy_from_nodule_level_xml.joblib"
DEFAULT_OUT=V2/"backend"/"output"
def readj(p): return json.loads(p.read_text(encoding="utf-8-sig"))
def unwrap(b):
    if hasattr(b,"predict"): return b
    if isinstance(b,dict):
        for k in ("model","estimator","regressor","pipeline"):
            if hasattr(b.get(k),"predict"): return b[k]
    raise TypeError("Could not find predict-capable estimator in joblib bundle.")
def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("patient_id",nargs="?")
    p.add_argument("--predictions",type=Path); p.add_argument("--model",type=Path,default=DEFAULT_MODEL)
    p.add_argument("--out-dir",type=Path); p.add_argument("--max-background",type=int,default=100)
    a=p.parse_args()
    if a.predictions is None:
        if not a.patient_id: p.error("provide patient_id or --predictions")
        a.predictions=DEFAULT_OUT/a.patient_id/"04_malignancy_prediction"/"malignancy_predictions.json"
    if not a.predictions.exists(): raise FileNotFoundError(a.predictions)
    if not a.model.exists(): raise FileNotFoundError(a.model)
    payload=readj(a.predictions); valid=[]
    for r in payload.get("predictions",[]):
        d=r.get("feature_values_used",{})
        if all(k in d for k in FEATURES):
            row=[float(d[k]) for k in FEATURES]
            if np.isfinite(row).all(): valid.append((r,row))
    if not valid: raise ValueError("No rows have all six finite feature_values_used.")
    X=np.asarray([r for _,r in valid],dtype=float); model=unwrap(joblib.load(a.model))
    names=list(getattr(model,"feature_names_in_",FEATURES))
    if names!=FEATURES: raise ValueError(f"Model feature order mismatch: {names}")
    try: import shap
    except ImportError as e: raise SystemExit("Install SHAP with: python -m pip install shap") from e
    background=X[:max(1,min(a.max_background,len(X)))]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            explainer=shap.TreeExplainer(model,data=background,feature_perturbation="interventional")
            sv=explainer.shap_values(X,check_additivity=False); base=explainer.expected_value
        except Exception:
            explainer=shap.TreeExplainer(model); sv=explainer.shap_values(X,check_additivity=False); base=explainer.expected_value
    sv=np.asarray(sv)
    if sv.ndim==3 and sv.shape[-1]==1: sv=sv[...,0]
    if sv.shape!=X.shape: raise ValueError(f"Unexpected SHAP shape {sv.shape}; expected {X.shape}")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out=a.out_dir or a.predictions.parent.parent/"05_explainability"/"malignancy"; out.mkdir(parents=True,exist_ok=True)
    for i,(r,_) in enumerate(valid):
        order=np.argsort(np.abs(sv[i])); fig,ax=plt.subplots(figsize=(8,4.8))
        ax.barh([FEATURES[j] for j in order],sv[i,order]); ax.axvline(0,linewidth=.8)
        ax.set_xlabel("SHAP contribution to continuous rating")
        ax.set_title(f"Candidate {r.get('candidate_id',i)} | rating={r.get('predicted_malignancy_rating_continuous',float('nan')):.3f}")
        fig.tight_layout(); fig.savefig(out/f"shap_candidate_{i:03d}.png",dpi=160,bbox_inches="tight"); plt.close(fig)
    mean_abs=np.mean(np.abs(sv),axis=0)
    result={"stage":"05_explainability","component":"stage04_malignancy_shap","patient_id":payload.get("patient_id",a.patient_id),
      "stage04_predictions_json":str(a.predictions.resolve()),"stage04_model":str(a.model.resolve()),
      "features_in_order":FEATURES,"candidate_count":len(valid),"base_value":np.asarray(base).tolist(),
      "candidates":[{"candidate_id":r.get("candidate_id"),"patch_id":r.get("patch_id"),
        "predicted_rating":r.get("predicted_malignancy_rating_continuous"),
        "feature_values":{k:float(X[i,j]) for j,k in enumerate(FEATURES)},
        "shap_values":{k:float(sv[i,j]) for j,k in enumerate(FEATURES)}} for i,(r,_) in enumerate(valid)],
      "mean_absolute_shap":{k:float(mean_abs[i]) for i,k in enumerate(FEATURES)},
      "limitations":["SHAP describes fitted model dependence, not causal effects.",
        "The target is a LIDC reader-assigned suspicion rating, not cancer probability.",
        "A single-patient background is local and not a population reference.",
        "Known weak validation on frozen Stage 03 outputs remains unresolved."]}
    (out/"malignancy_shap.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps({"status":"completed","output_dir":str(out),"candidates_explained":len(valid),
                      "json":str(out/"malignancy_shap.json")},indent=2))
if __name__=="__main__": main()
