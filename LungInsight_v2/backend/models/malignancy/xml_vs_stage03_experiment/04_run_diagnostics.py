#!/usr/bin/env python3
"""Diagnose XML-reader vs frozen Stage 03 attribute inputs for LungInsight v2.

Run from any working directory. Relative CLI paths are resolved against --repo-root
(default: infer the repository root from this script's location, then cwd fallback).
This script is read-only with respect to source artifacts; outputs go only to output-dir.
"""
from __future__ import annotations
import argparse, csv, json, math, sys, warnings
from pathlib import Path
from collections import Counter
from typing import Any

FEATURES = ['subtlety','sphericity','margin','lobulation','spiculation','texture']
SPLITS = ['train','val','test']
ID_PATTERNS = ['patient_id','patientid','subject_id','subjectid']
NODULE_PATTERNS = ['nodule_id','noduleid','seriesuid','series_uid','scan_id','image_id','instance_id']

class DiagnosticError(RuntimeError): pass

def log(msg: str) -> None: print(f'[diagnostics] {msg}', flush=True)

def reader_mean_target(ratings, mask):
    """Return mean valid reader malignancy rating and a valid-row mask."""
    import numpy as np
    ratings = np.asarray(ratings, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if ratings.shape != mask.shape or ratings.ndim != 2:
        raise ValueError(
            f'Reader ratings/mask shape mismatch: {ratings.shape} vs {mask.shape}'
        )
    valid = mask & np.isfinite(ratings) & (ratings >= 1) & (ratings <= 5)
    counts = valid.sum(axis=1)
    sums = np.where(valid, ratings, 0.0).sum(axis=1)
    targets = np.full(ratings.shape[0], np.nan, dtype=np.float64)
    np.divide(sums, counts, out=targets, where=counts > 0)
    return targets, counts > 0

def round_half_up_clamped(values):
    import numpy as np
    return np.clip(np.floor(np.asarray(values, dtype=np.float64) + 0.5), 1, 5).astype(int)

def json_safe(x: Any) -> Any:
    if isinstance(x, dict): return {str(k): json_safe(v) for k,v in x.items()}
    if isinstance(x, (list, tuple)): return [json_safe(v) for v in x]
    if isinstance(x, Path): return str(x)
    if hasattr(x, 'item'):
        try: return json_safe(x.item())
        except Exception: pass
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)): return None
    if x is None or isinstance(x, (str,int,float,bool)): return x
    return str(x)

def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(obj), indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')

def find_feature_columns(columns) -> dict[str,str]:
    norm = {str(c).strip().lower(): str(c) for c in columns}
    out = {}
    for f in FEATURES:
        if f in norm: out[f] = norm[f]
        else:
            candidates = [orig for low,orig in norm.items() if low.replace(' ','_').replace('-','_') == f]
            if len(candidates)==1: out[f]=candidates[0]
    return out

def discover_target_column(df):
    norm = {str(c).strip().lower(): str(c) for c in df.columns}
    for target in ('malignancy','malignancy_rating','suspicion','rating','target','y','label','score'):
        if target in norm: return norm[target]
    return None

def read_csv(path: Path):
    import pandas as pd
    if not path.is_file(): raise DiagnosticError(f'Required XML split is missing: {path}')
    try: df = pd.read_csv(path)
    except Exception as e: raise DiagnosticError(f'Cannot read CSV {path}: {e}') from e
    if df.empty: raise DiagnosticError(f'CSV has no data rows: {path}')
    return df

def quantiles(values):
    import numpy as np
    a=np.asarray(values,dtype=float); a=a[np.isfinite(a)]
    if not len(a): return {str(q):None for q in [0,.05,.25,.5,.75,.95,1]}
    return {str(q):float(np.quantile(a,q)) for q in [0,.05,.25,.5,.75,.95,1]}

def summarize_feature(values, split, source, feature, masked_count=0, invalid_count=0, level=None):
    import numpy as np
    raw=np.asarray(values).reshape(-1)
    try: x=raw.astype(float)
    except Exception: x=np.array([float(v) if str(v).strip() else np.nan for v in raw])
    finite=np.isfinite(x); valid=x[finite]
    row={'source':source,'split':split,'aggregation_level':level,'feature':feature,'n_total':int(len(x)),
         'n_valid':int(len(valid)),'n_missing_or_nonfinite':int((~finite).sum()),'n_masked':int(masked_count),
         'n_invalid_explicit':int(invalid_count),'mean':None,'std':None,'min':None,'max':None,'median':None,
         'quantiles':quantiles(valid),'n_distinct':int(len(np.unique(valid))),
         'proportion_outside_1_5':None,'rating_frequencies':{str(i):int(np.sum(valid==i)) for i in range(1,6)}}
    if len(valid):
        row.update(mean=float(valid.mean()),std=float(valid.std(ddof=1)) if len(valid)>1 else 0.0,
                   min=float(valid.min()),max=float(valid.max()),median=float(np.median(valid)),
                   proportion_outside_1_5=float(np.mean((valid<1)|(valid>5))))
    return row

def get_ids(df, patterns):
    low={str(c).lower():c for c in df.columns}
    for p in patterns:
        if p in low: return df[low[p]]
    return None

def xml_row_counts(df):
    patients=get_ids(df, ID_PATTERNS); nodules=get_ids(df,NODULE_PATTERNS)
    return {'rows':int(len(df)), 'unique_patients':int(patients.nunique(dropna=True)) if patients is not None else None,
            'unique_nodules_or_scans':int(nodules.nunique(dropna=True)) if nodules is not None else None,
            'patient_id_column':str(patients.name) if patients is not None else None,
            'nodule_id_column':str(nodules.name) if nodules is not None else None}

def load_npz(path: Path):
    import numpy as np
    if not path.is_file(): raise DiagnosticError(f'Required Stage 03 archive is missing: {path}')
    try:
        z=np.load(path, allow_pickle=False)
        arrays={k:z[k] for k in z.files}
        z.close()
    except Exception as e: raise DiagnosticError(f'Cannot read NPZ {path}: {e}') from e
    return arrays

def flatten_manifest(obj, names=None):
    """Return possible ordered feature-name lists from common manifest layouts."""
    found=[]
    if isinstance(obj, dict):
        for k,v in obj.items():
            kl=str(k).lower()
            if kl in ('feature_names','features','attribute_names','input_features') and isinstance(v,list) and all(isinstance(x,str) for x in v):
                found.append(v)
            else: found.extend(flatten_manifest(v))
    elif isinstance(obj,list):
        if obj and all(isinstance(x,str) for x in obj): found.append(obj)
        else:
            for v in obj: found.extend(flatten_manifest(v))
    return found

def rating_metrics(y, pred_cont, accuracy_rounding='half_up_clip_1_5'):
    import numpy as np
    from sklearn.metrics import mean_absolute_error, cohen_kappa_score, confusion_matrix
    y=np.asarray(y,dtype=float).reshape(-1); p=np.asarray(pred_cont,dtype=float).reshape(-1)
    ok=np.isfinite(y)&np.isfinite(p); y=y[ok]; p=p[ok]
    if not len(y): return {'n_evaluated':0,'error':'No paired finite targets and predictions.'}
    # Accuracy/QWK operate on integer ratings; continuous predictions remain continuous for MAE.
    yi=round_half_up_clamped(y); pi=round_half_up_clamped(p)
    return {'n_evaluated':int(len(y)),'mae_continuous':float(mean_absolute_error(y,p)),
            'qwk_rounded_clipped':float(cohen_kappa_score(yi,pi,weights='quadratic')) if len(np.unique(yi))>1 or len(np.unique(pi))>1 else None,
            'accuracy_rounded_clipped':float(np.mean(yi==pi)), 'rounding_clipping_policy':accuracy_rounding,
            'prediction_distribution_continuous':{'mean':float(np.mean(p)),'std':float(np.std(p)),'min':float(np.min(p)),'max':float(np.max(p))},
            'prediction_integer_counts_1_to_5':{str(i):int(np.sum(pi==i)) for i in range(1,6)},
            'target_integer_counts_1_to_5':{str(i):int(np.sum(yi==i)) for i in range(1,6)},
            'confusion_matrix_rows_target_1_to_5_cols_prediction_1_to_5':confusion_matrix(yi,pi,labels=[1,2,3,4,5]).tolist()}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo-root', default=None, help='Repository root; defaults to inferred root or current directory.')
    ap.add_argument('--data-dir',default='backend/models/malignancy/xml_vs_stage03_experiment/data')
    ap.add_argument('--feature-dir',default='backend/models/malignancy/frozen_features')
    ap.add_argument('--model',default='backend/models/malignancy/xml_vs_stage03_experiment/models/xml_attribute_malignancy.joblib')
    ap.add_argument('--metadata',default='backend/models/malignancy/xml_vs_stage03_experiment/models/model_metadata.json')
    ap.add_argument('--evaluation-dir',default='backend/models/malignancy/xml_vs_stage03_experiment/evaluation')
    ap.add_argument('--output-dir',default='backend/models/malignancy/xml_vs_stage03_experiment/diagnostics')
    args=ap.parse_args()
    # Installed script is normally in .../xml_vs_stage03_experiment; repo root is four parents up.
    inferred=Path(__file__).resolve().parents[4] if len(Path(__file__).resolve().parents)>4 else Path.cwd()
    root=Path(args.repo_root).expanduser().resolve() if args.repo_root else inferred
    if not (root/'backend').exists() and (Path.cwd()/'backend').exists(): root=Path.cwd().resolve()
    def path_arg(s):
        p=Path(s).expanduser(); return p.resolve() if p.is_absolute() else (root/p).resolve()
    data_dir,feature_dir,model_path,meta_path,eval_dir,out_dir=map(path_arg,[args.data_dir,args.feature_dir,args.model,args.metadata,args.evaluation_dir,args.output_dir])
    out_dir.mkdir(parents=True,exist_ok=True)
    report={'repository_root':str(root),'files':{},'schemas':{},'split_counts':{},'feature_validation':{},'xml_validation':{},'stage03':{},'distribution_shift':[], 'correlations':{},'data_quality_issues':[],'limitations':[], 'interpretation':{'observations':[],'hypotheses_not_established':[]}}
    try:
        import numpy as np, pandas as pd, joblib
        from sklearn.metrics import mean_absolute_error
    except Exception as e: raise DiagnosticError(f'Missing required Python dependency in selected environment: {e}. Install/use numpy, pandas, scikit-learn, and joblib in the existing l-env.') from e
    # Metadata, manifest, prior evaluations are inspected and summarized before analysis.
    manifest_path=feature_dir/'feature_manifest.json'; manifest={}
    for label,p in [('model',model_path),('model_metadata',meta_path),('feature_manifest',manifest_path)]: report['files'][label]={'path':str(p),'exists':p.is_file()}
    if not model_path.is_file(): raise DiagnosticError(f'Trained model not found: {model_path}')
    if not meta_path.is_file(): report['data_quality_issues'].append(f'Model metadata missing: {meta_path}')
    else:
        try: report['schemas']['model_metadata']=json.loads(meta_path.read_text(encoding='utf-8'))
        except Exception as e: raise DiagnosticError(f'Invalid model metadata JSON {meta_path}: {e}') from e
    if manifest_path.is_file():
        try: manifest=json.loads(manifest_path.read_text(encoding='utf-8')); report['schemas']['feature_manifest']=manifest
        except Exception as e: raise DiagnosticError(f'Invalid feature manifest JSON {manifest_path}: {e}') from e
    else: report['data_quality_issues'].append(f'Feature manifest missing: {manifest_path}')
    manifest_lists=flatten_manifest(manifest); manifest_features=next((x for x in manifest_lists if len(x)==6),None)
    report['schemas']['manifest_feature_name_candidates']=manifest_lists
    model=joblib.load(model_path)
    model_features=list(getattr(model,'feature_names_in_',[])) or None
    report['feature_validation']['model_feature_names_in_']=model_features
    report['feature_validation']['expected_shared_features']=FEATURES
    # Load all XML splits and inspect exact schemas.
    xml={}; xml_columns={}; xml_targets={}; xml_feature_cols={}
    for split in SPLITS:
        p=data_dir/f'{split}_xml.csv'; df=read_csv(p); xml[split]=df
        fc=find_feature_columns(df.columns); missing=[f for f in FEATURES if f not in fc]
        if missing: raise DiagnosticError(f'{p} lacks required shared XML feature columns {missing}. Actual columns: {list(df.columns)}')
        tc=discover_target_column(df)
        if tc is None: raise DiagnosticError(f'Cannot identify reader-assigned target column in {p}. Columns: {list(df.columns)}; expected a documented malignancy/suspicion/rating/target column.')
        xml_columns[split]=list(map(str,df.columns)); xml_feature_cols[split]=fc; xml_targets[split]=tc
        report['schemas'][f'xml_{split}']={'columns':xml_columns[split],'dtypes':{str(k):str(v) for k,v in df.dtypes.items()},'feature_columns':fc,'target_column':tc}
        report['split_counts'][f'xml_{split}']=xml_row_counts(df)
    # Load Stage 03 archives and derive targets from valid reader ratings per nodule.
    stageX={}; stageY={}; stage_meta={}
    for split in SPLITS:
        p=feature_dir/f'{split}.npz'
        if not p.is_file(): raise DiagnosticError(f'Required Stage 03 archive is missing: {p}')
        try:
            with np.load(p, allow_pickle=False) as archive:
                required={'X','malignancy_reader_ratings','malignancy_reader_mask'}
                missing=required-set(archive.files)
                if missing: raise DiagnosticError(f'{p} missing arrays: {sorted(missing)}; found={archive.files}')
                info={k:{'shape':list(archive[k].shape),'dtype':str(archive[k].dtype)} for k in archive.files}
                X=np.asarray(archive['X'],dtype=float)
                ratings=archive['malignancy_reader_ratings']
                mask=archive['malignancy_reader_mask']
                names=archive['feature_names'].astype(str).reshape(-1).tolist() if 'feature_names' in archive.files else manifest_features
                if X.ndim!=2 or X.shape[1]!=6:
                    raise DiagnosticError(f'{p}: expected X with shape (N,6), got {X.shape}')
                n_archive_rows=len(X)
                y,keep=reader_mean_target(ratings,mask)
                if len(X)!=len(y):
                    raise DiagnosticError(f'{p}: X rows {len(X)} do not align with reader-rating rows {len(y)}')
                if names is None or [str(name).lower() for name in names] != FEATURES:
                    raise DiagnosticError(f'{p}: expected six features in manifest order {FEATURES}; found {names}')
                if manifest_features and [str(name).lower() for name in manifest_features] != FEATURES:
                    raise DiagnosticError(f'Feature manifest order {manifest_features} differs from required order {FEATURES}.')
                X=X[keep]
                y=y[keep]
        except DiagnosticError:
            raise
        except Exception as e:
            raise DiagnosticError(f'Cannot load Stage 03 archive {p}: {e}') from e
        stageX[split]=X; stageY[split]=y
        target_key='mean(malignancy_reader_ratings[malignancy_reader_mask])'
        stage_meta[split]={'feature_key':'X','feature_names':FEATURES,'n_archive_rows':n_archive_rows,'n_rows':len(X),'target_key':target_key}
        report['schemas'][f'stage03_{split}']={'arrays':info,'feature_order':FEATURES,'target_definition':'Mean of valid finite malignancy_reader_ratings in [1,5], selected by malignancy_reader_mask.'}
        report['split_counts'][f'stage03_{split}']={'n_nodule_rows':int(n_archive_rows),'n_nodules_with_reader_target':int(len(X)),'n_reader_mean_targets':int(len(y)),'target_available':bool(len(y))}
        report['stage03'][split]={'feature_key':'X','feature_names':FEATURES,'target_key':target_key,'feature_shape':list(X.shape),'feature_dtype':str(X.dtype),'target_available':bool(len(y))}
    # Validate training contract, avoid silently reordering mismatched model inputs.
    if model_features:
        normalized=[str(x).lower() for x in model_features]
        if normalized != FEATURES:
            if set(normalized)==set(FEATURES): raise DiagnosticError(f'Model feature order {model_features} differs from verified experiment order {FEATURES}. Existing evaluation/training script must be inspected to establish contract; refusing silent reorder.')
            raise DiagnosticError(f'Model expects feature names {model_features}, not the six shared attributes {FEATURES}.')
    for s in SPLITS:
        if stage_meta[s]['feature_names'] != FEATURES: raise DiagnosticError(f'Stage 03 feature order for {s} is {stage_meta[s]["feature_names"]}, expected {FEATURES}; do not silently reorder.')
        X=stageX[s]
        report['feature_validation'][f'stage03_{s}']={'shape':list(X.shape),'dtype':str(X.dtype),'feature_order':stage_meta[s]['feature_names'],'all_finite_count':int(np.isfinite(X).sum()),'nonfinite_count':int((~np.isfinite(X)).sum()),'rows_with_any_nonfinite':int((~np.isfinite(X)).all(axis=1).sum()),'model_n_features':getattr(model,'n_features_in_',None)}
    if model_features is None and getattr(model,'n_features_in_',6)!=6: raise DiagnosticError(f'Model expects {getattr(model,"n_features_in_",None)} features, expected 6.')
    # Input distributions: XML reader rows and Stage03 nodule rows remain separate units.
    summaries=[]
    for s,df in xml.items():
        for f,c in xml_feature_cols[s].items(): summaries.append(summarize_feature(pd.to_numeric(df[c],errors='coerce').to_numpy(),s,'XML_reader_rows',f,level='reader-level rows'))
    for s,X in stageX.items():
        for j,f in enumerate(FEATURES): summaries.append(summarize_feature(X[:,j],s,'Stage03_nodule_rows',f,level='nodule-level archive rows'))
    pd.DataFrame([{**r, 'q05':r['quantiles']['0.05'],'q25':r['quantiles']['0.25'],'q50':r['quantiles']['0.5'],'q75':r['quantiles']['0.75'],'q95':r['quantiles']['0.95']} for r in summaries]).drop(columns=['quantiles']).to_csv(out_dir/'input_distribution_summary.csv',index=False)
    # Distribution shift by corresponding split context, with robust descriptive metrics.
    shifts=[]
    for s in SPLITS:
        for j,f in enumerate(FEATURES):
            a=pd.to_numeric(xml[s][xml_feature_cols[s][f]],errors='coerce').to_numpy(dtype=float); a=a[np.isfinite(a)]
            b=stageX[s][:,j]; b=b[np.isfinite(b)]
            def stats(v): return {'n':int(len(v)),'mean':float(np.mean(v)) if len(v) else None,'median':float(np.median(v)) if len(v) else None,'std':float(np.std(v,ddof=1)) if len(v)>1 else (0.0 if len(v)==1 else None),'q25':float(np.quantile(v,.25)) if len(v) else None,'q75':float(np.quantile(v,.75)) if len(v) else None}
            # Standardized mean difference uses pooled SD; descriptive only and can be unstable with near-zero variance.
            sa,sb=stats(a),stats(b); pooled=math.sqrt(((len(a)-1)*np.var(a,ddof=1)+(len(b)-1)*np.var(b,ddof=1))/(len(a)+len(b)-2)) if len(a)>1 and len(b)>1 and len(a)+len(b)>2 else float('nan')
            smd=(sb['mean']-sa['mean'])/pooled if np.isfinite(pooled) and pooled>0 else None
            # Wasserstein distance is defined for numeric inputs; interpret as rating-scale distance only.
            from scipy.stats import wasserstein_distance
            wd=float(wasserstein_distance(a,b)) if len(a) and len(b) else None
            row={'split_context':s,'xml_source':'XML_reader_rows','stage03_source':'Stage03_nodule_rows','feature':f,'xml_stats':sa,'stage03_stats':sb,'stage03_minus_xml_mean':(sb['mean']-sa['mean']) if sa['mean'] is not None and sb['mean'] is not None else None,'standardized_mean_difference':float(smd) if smd is not None else None,'wasserstein_distance_rating_units':wd,'interpretation_note':'Descriptive comparison only; XML rows are reader-level while Stage 03 rows are nodule-level and are not paired.'}
            shifts.append(row)
    pd.DataFrame([{**r,'xml_n':r['xml_stats']['n'],'stage03_n':r['stage03_stats']['n'],'xml_mean':r['xml_stats']['mean'],'stage03_mean':r['stage03_stats']['mean'],'mean_difference':r['stage03_minus_xml_mean'],'smd':r['standardized_mean_difference'],'wasserstein_distance':r['wasserstein_distance_rating_units']} for r in shifts]).drop(columns=['xml_stats','stage03_stats']).to_csv(out_dir/'input_distribution_shift.csv',index=False)
    report['distribution_shift']=shifts
    # Correlations per source/split. Pairwise-complete Pearson and Spearman; no false pairing claim.
    corr_rows=[]; report['correlations']={}
    from scipy.stats import spearmanr
    for source in ('XML_reader_rows','Stage03_nodule_rows'):
        for s in SPLITS:
            mat=np.column_stack([pd.to_numeric(xml[s][xml_feature_cols[s][f]],errors='coerce').to_numpy(dtype=float) for f in FEATURES]) if source.startswith('XML') else stageX[s]
            pear=pd.DataFrame(mat,columns=FEATURES).corr(method='pearson',min_periods=2); spear=pd.DataFrame(mat,columns=FEATURES).corr(method='spearman',min_periods=2)
            report['correlations'][f'{source}_{s}']={'n_rows':int(len(mat)),'pearson':pear.where(pd.notna(pear),None).to_dict(),'spearman':spear.where(pd.notna(spear),None).to_dict()}
            for i in range(6):
                for j in range(i+1,6): corr_rows.append({'source':source,'split':s,'feature_a':FEATURES[i],'feature_b':FEATURES[j],'pearson_r':pear.iloc[i,j] if pd.notna(pear.iloc[i,j]) else None,'spearman_rho':spear.iloc[i,j] if pd.notna(spear.iloc[i,j]) else None})
    pd.DataFrame(corr_rows).to_csv(out_dir/'input_distribution_correlations.csv',index=False)
    # XML validation model evaluation at reader-row level using the target column from the split.
    xv=xml['val']; Xv=xv[[xml_feature_cols['val'][f] for f in FEATURES]].apply(pd.to_numeric,errors='coerce').to_numpy(dtype=float); yv=pd.to_numeric(xv[xml_targets['val']],errors='coerce').to_numpy(dtype=float)
    valid=np.isfinite(yv)&np.isfinite(Xv).all(axis=1)
    if not valid.any(): raise DiagnosticError('XML validation has no rows with finite target and all six finite attributes; cannot evaluate model.')
    try: pred_xml=np.asarray(model.predict(pd.DataFrame(Xv[valid],columns=FEATURES)),dtype=float).reshape(-1)
    except Exception as e: raise DiagnosticError(f'Model prediction failed on XML validation inputs with verified feature order: {e}') from e
    xml_metrics=rating_metrics(yv[valid],pred_xml)
    xml_metrics.update({'input_source':'XML validation reader-level rows','target_column':xml_targets['val'],'rows_total':int(len(xv)),'rows_used':int(valid.sum()),'rows_dropped_invalid':int((~valid).sum()),'target_semantics':'Reader-assigned malignancy/suspicion rating, not pathology-confirmed cancer.'})
    report['xml_validation']=xml_metrics
    write_json(out_dir/'xml_validation_metrics.json',xml_metrics)
    # Training target baselines use Stage03 training targets only; absent target means not evaluable.
    baseline_report={'training_target_source':'Mean of valid malignancy_reader_ratings selected by malignancy_reader_mask','training_target_key':stage_meta['train']['target_key'],'train_split_caveat':'Stage 03 train features are in-sample predictions; train-split diagnostics may be optimistic.','test_split_policy':'Report held-out test results once; do not use them for model selection or tuning.','baselines':{},'splits':{},'limitations':[]}
    train_y=stageY['train']; train_y=train_y[np.isfinite(train_y)] if train_y is not None else None
    training_target_mean=None; training_target_mode=None
    if train_y is not None and len(train_y):
        rounded=round_half_up_clamped(train_y); counts=Counter(rounded.tolist()); mode=max(counts, key=lambda v:(counts[v],-v))
        training_target_mean=float(np.mean(train_y)); training_target_mode=int(mode); baseline_report['baselines']={'constant_rating_3':{'value':3},'training_target_mean':{'value':training_target_mean},'rounded_training_target_mode':{'value':training_target_mode,'training_rounded_counts':{str(i):int(counts.get(i,0)) for i in range(1,6)},'tie_policy':'lowest integer rating wins ties'}}
    else:
        baseline_report['limitations'].append('Stage03 training archive contains no nodules with a valid reader-rating target; training-target mean/mode baselines are not computed.')
    # Existing evaluation procedure is continuous model output and rounded/clipped model input.
    for s in ('val','test'):
        X=stageX[s]; y=stageY[s]
        if y is None:
            baseline_report['splits'][s]={'n_nodules':int(stage_meta[s]['n_archive_rows']),'n_with_reader_target':len(X),'target_available':False,'methods':{},'note':'No nodule has a valid masked reader-rating mean target; no metrics fabricated.'}; continue
        valid=np.isfinite(y)&np.isfinite(X).all(axis=1); methods={}
        if not valid.any():
            baseline_report['splits'][s]={'n_nodules':len(X),'target_available':True,'methods':{},'note':'No rows with finite target and all six finite attributes.'}; continue
        yt=y[valid]; methods['constant_rating_3']=rating_metrics(yt,np.full(len(yt),3.0))
        if training_target_mean is not None and training_target_mode is not None:
            methods['training_target_mean']=rating_metrics(yt,np.full(len(yt),training_target_mean))
            methods['rounded_training_target_mode']=rating_metrics(yt,np.full(len(yt),float(training_target_mode)))
        Xcont=X[valid]
        try:
            pc=np.asarray(model.predict(pd.DataFrame(Xcont,columns=FEATURES)),dtype=float).reshape(-1)
            methods['xml_model_stage03_continuous_inputs']=rating_metrics(yt,pc)
            Xround=round_half_up_clamped(Xcont)
            pr=np.asarray(model.predict(pd.DataFrame(Xround,columns=FEATURES)),dtype=float).reshape(-1)
            methods['xml_model_stage03_rounded_clipped_inputs']=rating_metrics(yt,pr)
        except Exception as e: raise DiagnosticError(f'Model prediction failed on Stage03 {s} archive: {e}') from e
        baseline_report['splits'][s]={'n_nodules':int(stage_meta[s]['n_archive_rows']),'n_with_reader_target':int(len(X)),'n_evaluated':int(valid.sum()),'n_dropped_invalid_features':int((~valid).sum()),'target_key':stage_meta[s]['target_key'],'target_semantics':'Mean of valid reader-assigned malignancy/suspicion ratings per nodule; not pathology-confirmed cancer.','methods':methods}
        report['stage03'][s]['evaluation']=baseline_report['splits'][s]
    # Existing artifact comparison, read-only. Find nested validation values defensively.
    eval_path=eval_dir/'continuous_vs_rounded_metrics.json'; pred_path=eval_dir/'heldout_predictions.csv'
    report['files']['existing_evaluation_metrics']={'path':str(eval_path),'exists':eval_path.is_file()}
    report['files']['existing_heldout_predictions']={'path':str(pred_path),'exists':pred_path.is_file()}
    existing={}
    if eval_path.is_file():
        try: existing=json.loads(eval_path.read_text(encoding='utf-8')); report['schemas']['existing_evaluation_metrics']=existing
        except Exception as e: report['data_quality_issues'].append(f'Could not parse existing evaluation JSON: {e}')
    if pred_path.is_file():
        try:
            ep=pd.read_csv(pred_path); report['schemas']['existing_heldout_predictions']={'columns':list(ep.columns),'dtypes':{str(k):str(v) for k,v in ep.dtypes.items()},'rows':len(ep)}
        except Exception as e: report['data_quality_issues'].append(f'Could not inspect existing predictions CSV: {e}')
    # Preserve raw existing values and add comparison only when key names can be identified safely.
    report['existing_evaluation_comparison']={'existing_metrics_path':str(eval_path),'existing_metrics_json':existing,'recomputed_xml_validation':xml_metrics,'note':'Schema-dependent metric values are preserved verbatim above; no values are overwritten. Compare only metrics with matching split, target unit, prediction rounding and aggregation.'}
    # Basic target/input scale checks and interpretation based on observed prediction concentration.
    for s in ('val','test'):
        ev=baseline_report['splits'].get(s,{}).get('methods',{})
        m=ev.get('xml_model_stage03_continuous_inputs')
        if m and m.get('prediction_integer_counts_1_to_5'):
            counts=m['prediction_integer_counts_1_to_5']; n=sum(counts.values()); dominant=max(counts.values())/n if n else None
            report['interpretation']['observations'].append({'split':s,'observation':'Stage03 model predictions after nearest-integer rounding and clipping are concentrated in one rating.','dominant_integer_prediction_fraction':dominant,'prediction_counts':counts})
    report['interpretation']['observations'].append({'observation':'Distribution shift metrics compare XML reader-level rows against Stage03 nodule-level rows; these are not paired samples and aggregation differences confound direct causal interpretation.'})
    report['interpretation']['hypotheses_not_established']=['Feature-distribution mismatch may contribute, but this report alone does not establish causality.','Reader-level target construction and nodule-level archive target aggregation may differ; upstream scripts and archive target provenance are required to establish equivalence.','A nearly constant model response can also arise from feature order/preprocessing mismatch, weak fitted signal, target imbalance, or out-of-range/invalid values.']
    report['limitations'] += ['This script does not train models, alter Stage03, modify splits, or overwrite existing evaluation artifacts.','Stage 03 targets are the mean of valid finite malignancy_reader_ratings selected by malignancy_reader_mask; malignancy_median and malignancy_reader_count are not used as targets.','Stage 03 train features are in-sample predictions, so train-split feature diagnostics may be optimistic.','Held-out test results are reported for evaluation only and must not be used for model selection or tuning.','Reader ratings are not pathology-confirmed malignancy labels.','Only the six features in manifest order (subtlety, sphericity, margin, lobulation, spiculation, texture) are used.']
    write_json(out_dir/'baseline_metrics.json',baseline_report)
    write_json(out_dir/'diagnostics_report.json',report)
    log(f'Wrote diagnostics to {out_dir}')
    for name in ['input_distribution_summary.csv','input_distribution_shift.csv','input_distribution_correlations.csv','baseline_metrics.json','xml_validation_metrics.json','diagnostics_report.json']:
        p=out_dir/name
        if not p.is_file(): raise DiagnosticError(f'Expected output was not created: {p}')
        log(f'  {p.name}: {p.stat().st_size} bytes')
    return 0

if __name__=='__main__':
    try: sys.exit(main())
    except DiagnosticError as e:
        print(f'ERROR: {e}',file=sys.stderr); sys.exit(2)
    except Exception as e:
        print(f'UNEXPECTED ERROR: {type(e).__name__}: {e}',file=sys.stderr); raise
