#!/usr/bin/env python3
"""Run calibrated XML-trained Stage 04 malignancy inference on Stage 03 predictions.json."""
from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path
import joblib
import numpy as np

FEATURES=["subtlety","sphericity","margin","lobulation","spiculation","texture"]

def read_json(path:Path):
    try: return json.loads(path.read_text(encoding='utf-8-sig'))
    except FileNotFoundError: raise SystemExit(f"ERROR: file not found: {path}")
    except json.JSONDecodeError as e: raise SystemExit(f"ERROR: invalid JSON in {path}: {e}")

def read_features(record):
    attrs=record.get('attributes') if isinstance(record,dict) else None
    if not isinstance(attrs,dict): raise ValueError('missing attributes object')
    values=[]
    for name in FEATURES:
        item=attrs.get(name)
        if not isinstance(item,dict): raise ValueError(f'missing attributes.{name} object')
        raw=item.get('rating_original_scale',item.get('rating_rounded'))
        try: x=float(raw)
        except (TypeError,ValueError): raise ValueError(f'{name} is not numeric')
        if not math.isfinite(x) or x<1 or x>5: raise ValueError(f'{name} outside [1,5]')
        values.append(x)
    return np.asarray(values,dtype=float)

def calibrate(X,calibrators):
    X=np.asarray(X,dtype=float); out=np.empty_like(X)
    for j,name in enumerate(FEATURES):
        c=calibrators[name]
        if c['kind']=='isotonic': out[:,j]=c['estimator'].predict(X[:,j])
        elif c['kind']=='constant': out[:,j]=float(c['value'])
        else: raise ValueError(f'unsupported calibrator for {name}: {c.get("kind")}')
    return np.clip(out,1,5)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('predictions_json',type=Path,help='Existing Stage 03 predictions.json')
    p.add_argument('--model',type=Path,required=True,help='stage04_xml_calibrated_malignancy.joblib')
    p.add_argument('--output',type=Path,default=None)
    p.add_argument('--skip-invalid',action='store_true')
    a=p.parse_args()
    if not a.model.is_file(): raise SystemExit(f'ERROR: model not found: {a.model}')
    bundle=joblib.load(a.model)
    if not isinstance(bundle,dict) or 'model' not in bundle or 'feature_calibrators' not in bundle:
        raise SystemExit('ERROR: model bundle lacks model/feature_calibrators')
    if bundle.get('feature_names_in_order')!=FEATURES: raise SystemExit('ERROR: model feature order mismatch')
    records=read_json(a.predictions_json)
    if not isinstance(records,list) or not records: raise SystemExit('ERROR: expected non-empty Stage 03 predictions JSON array')
    valid=[]; X=[]; errors=[]; seen=set()
    for i,r in enumerate(records):
        try:
            x=read_features(r)
            ident=(str(r.get('patch_id')),str(r.get('candidate_id')))
            if ident in seen: raise ValueError(f'duplicate candidate identity {ident}')
            seen.add(ident); X.append(x); valid.append((i,r))
        except ValueError as e:
            errors.append({'index':i,'error':str(e),'patch_id':r.get('patch_id') if isinstance(r,dict) else None})
            if not a.skip_invalid: raise SystemExit(f'ERROR: candidate index {i}: {e}')
    if not X: raise SystemExit('ERROR: no valid candidate attribute rows')
    X=np.asarray(X,dtype=float); Xcal=calibrate(X,bundle['feature_calibrators'])
    preds=np.clip(np.asarray(bundle['model'].predict(Xcal),dtype=float),1,5)
    outputs=[]
    inferred_patient=a.predictions_json.parent.name
    for j,(index,r) in enumerate(valid):
        attrs={name:float(X[j,k]) for k,name in enumerate(FEATURES)}
        calattrs={name:float(Xcal[j,k]) for k,name in enumerate(FEATURES)}
        score=float(preds[j]); rating=int(np.clip(math.floor(score+0.5),1,5))
        outputs.append({'patch_id':r.get('patch_id'),'candidate_id':r.get('candidate_id'),'patient_id':r.get('patient_id') or r.get('patient') or inferred_patient,
          'predicted_malignancy_suspicion_rating':rating,'predicted_rating_continuous':score,
          'feature_values_received_from_stage03':attrs,'feature_values_after_train_fitted_calibration':calattrs,
          'stage03_prediction_index':index,
          'interpretation':'Predicted LIDC radiologist malignancy/suspicion rating (1-5), not a calibrated cancer probability or pathology diagnosis.'})
    result={'schema_version':'stage04-xml-calibrated-output-v1','model':'XML nodule-median malignancy regressor with per-attribute supervised isotonic calibration','feature_names':FEATURES,'calibration_source':'matched train-split XML reader attribute ratings only','source_predictions_json':str(a.predictions_json),'model_file':str(a.model),'candidate_count':len(records),'valid_candidate_count':len(outputs),'invalid_candidates':errors,'predictions':outputs}
    out=a.output or a.predictions_json.with_name('malignancy_predictions.json')
    out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'status':'completed','output':str(out),'candidate_count':len(records),'valid_candidate_count':len(outputs),'invalid_count':len(errors)},indent=2))
if __name__=='__main__': main()
