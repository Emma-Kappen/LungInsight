#!/usr/bin/env python3
"""Train a Stage 04 malignancy model with train-only supervised calibration.

Pairs frozen Stage 03 scores to XML reader attributes for the same physical H5
nodule using exact study/series + reader + full 8-attribute signature matching.
Only the six attributes emitted by Stage 03 are predictors. internalStructure
and calcification are used only as matching fields, never as model inputs.

The script reads H5 annotation/identity datasets for canonical TRAIN and VAL
samples only. It never accesses the H5 patches dataset, CT PixelData, or test.npz.
Stage 03 checkpoints/outputs are never modified. The fixed patient/sample split
is used unchanged.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math, platform, sys, warnings
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import xml.etree.ElementTree as ET

import h5py
import joblib
import numpy as np
import sklearn
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, cohen_kappa_score

FEATURES = ["subtlety", "sphericity", "margin", "lobulation", "spiculation", "texture"]
MATCH_FIELDS = ["subtlety", "internalStructure", "calcification", "sphericity", "margin", "lobulation", "spiculation", "texture"]
XML_KEYS = {"subtlety":"subtlety", "internalStructure":"internalStructure", "calcification":"calcification", "sphericity":"sphericity", "margin":"margin", "lobulation":"lobulation", "spiculation":"spiculation", "texture":"texture"}
RATING_LIMITS = {"subtlety":5, "internalStructure":4, "calcification":6, "sphericity":5, "margin":5, "lobulation":5, "spiculation":5, "texture":5, "malignancy":5}
# Fixed H5's documented reader_labels order; validate against any explicit attr when present.
H5_READER_ORDER_FALLBACK = ["calcification", "internalStructure", "lobulation", "malignancy", "margin", "sphericity", "spiculation", "subtlety", "texture"]


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()

def decode(v):
    if isinstance(v, bytes): return v.decode('utf-8', errors='replace').strip('\x00 ')
    if isinstance(v, np.bytes_): return bytes(v).decode('utf-8', errors='replace').strip('\x00 ')
    return str(v).strip('\x00 ')

def local(tag): return tag.rsplit('}',1)[-1]

def rating(v, maxval):
    try:
        x=float(str(v).strip())
        if not math.isfinite(x) or not x.is_integer() or x<1 or x>maxval: return None
        return int(x)
    except (ValueError, TypeError): return None

def split_ids_and_indices(path: Path):
    obj=json.loads(path.read_text(encoding='utf-8-sig'))
    res={}
    for name in ('train','val'):
        ent=obj.get(name)
        if not isinstance(ent,dict) or not isinstance(ent.get('sample_indices'),list):
            raise ValueError(f"split.json must contain {name}.sample_indices")
        inds=[int(x) for x in ent['sample_indices']]
        if len(inds)!=len(set(inds)): raise ValueError(f"Duplicate H5 sample indices in split {name}")
        res[name]={'indices':set(inds),'patients':set(map(str,ent.get('patients',[])))}
    if res['train']['indices'] & res['val']['indices']: raise ValueError('Train/validation sample-index overlap')
    if res['train']['patients'] & res['val']['patients']: raise ValueError('Train/validation patient overlap')
    return obj,res

def parse_h5_reader_order(h5):
    for attr in ('reader_feature_names','reader_label_names','reader_names_order'):
        if attr in h5.attrs:
            raw=h5.attrs[attr]
            try:
                v=json.loads(decode(raw)) if isinstance(raw,(str,bytes,np.bytes_)) else list(raw)
                v=[decode(x) for x in v]
                if len(v)==h5['reader_labels'].shape[-1] and set(H5_READER_ORDER_FALLBACK).issubset(set(v)):
                    return v
            except Exception: pass
    n=h5['reader_labels'].shape[-1]
    if n != len(H5_READER_ORDER_FALLBACK):
        raise ValueError(f"Cannot identify H5 reader_labels order: last dimension={n}; no usable order attr")
    return H5_READER_ORDER_FALLBACK.copy()

def make_sig(vals):
    if any(vals.get(k) is None for k in MATCH_FIELDS): return None
    return tuple(int(vals[k]) for k in MATCH_FIELDS)

def index_h5_trainval(h5_path: Path, split: dict):
    by_sig=defaultdict(list); sample_meta={}; stats=Counter()
    indices=sorted(split['train']['indices'] | split['val']['indices'])
    with h5py.File(h5_path,'r') as h5:
        required=['reader_labels','reader_mask','patient_id','study_instance_uid','series_instance_uid']
        missing=[k for k in required if k not in h5]
        if missing: raise KeyError(f"Fixed H5 missing annotation/identity datasets: {missing}")
        order=parse_h5_reader_order(h5)
        label_index={k:i for i,k in enumerate(order)}
        for idx in indices:
            if idx<0 or idx>=len(h5['patient_id']): raise ValueError(f"Split sample index {idx} outside H5 sample count")
            meta={'patient_id':decode(h5['patient_id'][idx]),'study_uid':decode(h5['study_instance_uid'][idx]),'series_uid':decode(h5['series_instance_uid'][idx])}
            expected_split='train' if idx in split['train']['indices'] else 'val'
            if meta['patient_id'] not in split[expected_split]['patients']:
                raise ValueError(f"H5 patient mismatch for sample_index={idx}: got {meta['patient_id']} in {expected_split}")
            sample_meta[idx]=meta
            labels=np.asarray(h5['reader_labels'][idx])
            mask=np.asarray(h5['reader_mask'][idx]).astype(bool)
            if labels.ndim!=2 or labels.shape[0]!=len(mask): raise ValueError(f"Unexpected H5 reader label shape at sample {idx}: {labels.shape}")
            for slot,valid in enumerate(mask):
                if not valid: continue
                vals={k:rating(labels[slot,label_index[k]],RATING_LIMITS[k]) for k in MATCH_FIELDS}
                sig=make_sig(vals)
                if sig is None:
                    stats['h5_reader_slots_incomplete_match_signature']+=1; continue
                key=(meta['patient_id'],meta['study_uid'],meta['series_uid'],sig)
                by_sig[key].append({'sample_index':idx,'reader_slot':slot,'malignancy':rating(labels[slot,label_index['malignancy']],5) if 'malignancy' in label_index else None})
                stats['h5_complete_reader_signatures']+=1
        h5_info={'sample_count':len(h5['patient_id']),'trainval_samples_read':len(indices),'reader_label_order':order,'patch_dataset_accessed':False}
    return by_sig,sample_meta,h5_info,dict(stats)

def parse_xml_header(path: Path):
    try:
        for _,el in ET.iterparse(path,events=('end',)):
            if local(el.tag)=='ResponseHeader':
                study=series=None
                for node in el:
                    if local(node.tag)=='StudyInstanceUID': study=(node.text or '').strip()
                    elif local(node.tag)=='SeriesInstanceUid': series=(node.text or '').strip()
                return study or None,series or None
    except Exception: return None,None
    return None,None

def parse_xml_records(path: Path, patient_id: str, study_uid: str, series_uid: str):
    root=ET.parse(path).getroot(); records=[]
    for sess in root:
        if local(sess.tag)!='readingSession': continue
        reader=''
        for child in sess:
            if local(child.tag)=='servicingRadiologistID': reader=(child.text or '').strip()
        for n in (child for child in sess if local(child.tag)=='unblindedReadNodule'):
            chars=next((c for c in n if local(c.tag)=='characteristics'),None)
            if chars is None: continue
            vals={}
            for field in MATCH_FIELDS:
                want=XML_KEYS[field]
                node=next((c for c in chars if local(c.tag)==want),None)
                vals[field]=rating(node.text if node is not None else None,RATING_LIMITS[field])
            malnode=next((c for c in chars if local(c.tag)=='malignancy'),None)
            y=rating(malnode.text if malnode is not None else None,5)
            records.append({'patient_id':patient_id,'study_uid':study_uid,'series_uid':series_uid,'reader_id':reader,'signature':make_sig(vals),'values':vals,'malignancy':y,'annotation_file':str(path)})
    return records

def collect_xml_rows(lidc_root: Path, split: dict):
    metadata_path=lidc_root/'metadata'/'metadata.csv'
    xml_root=lidc_root/'LIDC-XML-only'/'tcia-lidc-xml'
    if not metadata_path.is_file(): raise FileNotFoundError(metadata_path)
    if not xml_root.is_dir(): raise FileNotFoundError(xml_root)
    trainval_patients=split['train']['patients']|split['val']['patients']
    wanted={}
    with metadata_path.open('r',newline='',encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            patient=str(row.get('PatientID','')).strip(); study=str(row.get('StudyInstanceUID','')).strip(); series=str(row.get('SeriesInstanceUID','')).strip()
            if patient in trainval_patients and study and series:
                wanted[(patient,study,series)]=(study,series)
    wanted_pairs={(s,ser) for _,s,ser in wanted}
    files_checked=0; matched_files=[]; duplicates=Counter()
    for path in sorted(xml_root.rglob('*.xml')):
        files_checked+=1
        study,series=parse_xml_header(path)
        if study and series and (study,series) in wanted_pairs:
            matched_files.append((path,study,series))
            duplicates[(study,series)]+=1
    dupkeys=[k for k,v in duplicates.items() if v>1]
    if dupkeys:
        # Multiple XML documents for a single series should not be silently counted twice.
        warnings.warn(f"{len(dupkeys)} study/series identities have multiple XML files; all distinct files are parsed, then signatures are de-duplicated at mapping time")
    pair_to_patient={}
    for (patient,study,series) in wanted:
        key=(study,series)
        if key in pair_to_patient and pair_to_patient[key]!=patient: raise ValueError(f"Metadata maps {key} to multiple patients")
        pair_to_patient[key]=patient
    records=[]; stats=Counter()
    for path,study,series in matched_files:
        patient=pair_to_patient[(study,series)]
        expected='train' if patient in split['train']['patients'] else 'val'
        parsed=parse_xml_records(path,patient,study,series)
        stats[f'{expected}_xml_annotation_records']+=len(parsed)
        for r in parsed:
            r['split']=expected
            records.append(r)
    stats['xml_files_scanned_for_headers']=files_checked
    stats['xml_files_matched_trainval_series']=len(matched_files)
    stats['duplicate_xml_study_series_keys']=len(dupkeys)
    return records,dict(stats)

def archive_data(path: Path, split_name: str, split: dict):
    if not path.is_file(): raise FileNotFoundError(path)
    with np.load(path,allow_pickle=False) as z:
        required={'X','feature_names','patient_id','sample_index','malignancy_reader_ratings','malignancy_reader_mask'}
        missing=required-set(z.files)
        if missing: raise KeyError(f"{split_name} archive missing keys: {sorted(missing)}")
        X=np.asarray(z['X'],dtype=np.float64).copy(); names=[decode(x) for x in z['feature_names'].tolist()]
        patients=np.asarray([decode(x) for x in z['patient_id'].tolist()])
        indices=np.asarray(z['sample_index'],dtype=np.int64).copy()
        rating_arr=np.asarray(z['malignancy_reader_ratings'],dtype=np.float64).copy()
        mask=np.asarray(z['malignancy_reader_mask'],dtype=bool).copy()
    if names!=FEATURES: raise ValueError(f"{split_name} feature order mismatch: {names}")
    if len(np.unique(indices))!=len(indices): raise ValueError(f"Duplicate sample_index in {split_name} archive")
    expected=split[split_name]['indices']
    if set(indices.tolist())!=expected: raise ValueError(f"{split_name} archive indices differ from fixed split: archive={len(indices)} split={len(expected)}")
    if rating_arr.shape!=mask.shape or rating_arr.shape[0]!=len(X) or len(indices)!=len(X) or len(patients)!=len(X): raise ValueError(f"{split_name} feature archive arrays have inconsistent shapes")
    keep={int(sample):i for i,sample in enumerate(indices)}
    med=np.full(len(X),np.nan); mean=np.full(len(X),np.nan)
    for i in range(len(X)):
        valid=mask[i]&np.isfinite(rating_arr[i])&(rating_arr[i]>=1)&(rating_arr[i]<=5)
        if valid.any(): med[i]=np.median(rating_arr[i,valid]); mean[i]=np.mean(rating_arr[i,valid])
    return {'X':X,'patient_id':patients,'sample_index':indices,'y_median':med,'y_mean':mean,'row_by_sample':keep,'archive_hash':sha256(path),'rating_rows_used':int(np.isfinite(med).sum())}

def spearman_safe(a,b):
    from scipy.stats import spearmanr
    if len(a)<3 or len(np.unique(a))<2 or len(np.unique(b))<2: return None
    x=spearmanr(a,b).statistic
    return float(x) if np.isfinite(x) else None

def fit_isotonic(x,y):
    x=np.asarray(x,float); y=np.asarray(y,float); ok=np.isfinite(x)&np.isfinite(y)
    x=x[ok]; y=y[ok]
    if len(x)<15: raise ValueError(f"Too few matched training nodules for isotonic calibration: {len(x)}")
    if len(np.unique(x))<2:
        return {'kind':'constant','value':float(np.median(y)),'n':int(len(x)),'unique_input_values':int(len(np.unique(x)))}
    est=IsotonicRegression(increasing=True,y_min=1,y_max=5,out_of_bounds='clip')
    est.fit(x,y)
    return {'kind':'isotonic','estimator':est,'n':int(len(x)),'unique_input_values':int(len(np.unique(x)))}

def apply_calibrators(X, calibrators):
    X=np.asarray(X,float); out=np.zeros_like(X)
    for j,name in enumerate(FEATURES):
        c=calibrators[name]
        if c['kind']=='isotonic': out[:,j]=c['estimator'].predict(X[:,j])
        else: out[:,j]=c['value']
    return np.clip(out,1,5)

def rounded_half_up(x): return np.clip(np.floor(np.asarray(x,float)+.5),1,5).astype(int)

def score(y,pred,baseline):
    y=np.asarray(y,float); p=np.clip(np.asarray(pred,float),1,5); ok=np.isfinite(y)&np.isfinite(p); y=y[ok]; p=p[ok]
    yi=rounded_half_up(y); pi=rounded_half_up(p); base=np.full(len(y),float(baseline)); bi=rounded_half_up(base)
    q=cohen_kappa_score(yi,pi,weights='quadratic') if len(y)>1 else float('nan')
    return {'n_nodules':int(len(y)),'target_mean':float(np.mean(y)) if len(y) else None,'target_std':float(np.std(y)) if len(y) else None,'prediction_mean':float(np.mean(p)) if len(p) else None,'prediction_std':float(np.std(p)) if len(p) else None,'prediction_min':float(np.min(p)) if len(p) else None,'prediction_max':float(np.max(p)) if len(p) else None,'mae':float(mean_absolute_error(y,p)) if len(y) else None,'rmse':float(np.sqrt(mean_squared_error(y,p))) if len(y) else None,'baseline_value':float(baseline),'baseline_mae':float(mean_absolute_error(y,base)) if len(y) else None,'mae_gain_over_baseline':float(mean_absolute_error(y,base)-mean_absolute_error(y,p)) if len(y) else None,'accuracy_half_up':float(accuracy_score(yi,pi)) if len(y) else None,'qwk_half_up':float(q) if np.isfinite(q) else None,'target_integer_counts':{str(k):int((yi==k).sum()) for k in range(1,6)},'prediction_integer_counts':{str(k):int((pi==k).sum()) for k in range(1,6)}}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--lidc-root',type=Path,default=Path('datasets/LIDC'))
    p.add_argument('--h5',type=Path,default=Path('datasets/LIDC/lidc_patches.h5'))
    p.add_argument('--split-json',type=Path,required=True)
    p.add_argument('--features-dir',type=Path,default=Path('LungInsight_v2/backend/models/malignancy/frozen_features'))
    p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--min-calibration-nodules',type=int,default=30)
    p.add_argument('--overwrite',action='store_true')
    a=p.parse_args()
    for f in (a.lidc_root/'metadata'/'metadata.csv', a.lidc_root/'LIDC-XML-only'/'tcia-lidc-xml', a.h5, a.split_json, a.features_dir/'train.npz', a.features_dir/'val.npz'):
        if not f.exists(): raise FileNotFoundError(f)
    if a.out_dir.exists() and any(a.out_dir.iterdir()) and not a.overwrite: raise FileExistsError(f"Output directory is not empty: {a.out_dir}")
    split_obj,split=split_ids_and_indices(a.split_json)
    h5_sig, h5_meta,h5_info,h5_stats=index_h5_trainval(a.h5,split)
    print(f"H5 annotation-only index: {len(h5_meta)} train/validation samples; patches not read")
    xml_rows, xml_stats=collect_xml_rows(a.lidc_root,split)
    print(f"XML reader annotations parsed: {len(xml_rows)}")
    # Match on the complete non-malignancy signature and exact patient/study/series.
    # Reader IDs are deliberately not part of identity because some H5 builds store reader
    # IDs in compacted slots; a unique physical sample index is the safe requirement.
    per_sample=defaultdict(list); match_stats=Counter()
    for r in xml_rows:
        if r['signature'] is None:
            match_stats['xml_missing_or_invalid_full_attribute_signature']+=1; continue
        key=(r['patient_id'],r['study_uid'],r['series_uid'],r['signature'])
        cand=h5_sig.get(key,[])
        sample_ids={x['sample_index'] for x in cand}
        if len(sample_ids)==1:
            sample=next(iter(sample_ids)); method='unique_full_attribute_signature'
        elif len(sample_ids)==0:
            match_stats['no_h5_signature_match']+=1; continue
        else:
            match_stats['ambiguous_signature_multiple_h5_samples']+=1; continue
        # Avoid duplicated metadata/XML records while retaining multiple readers per nodule.
        dedup_key=(sample,r['reader_id'].casefold(),r['signature'],r['malignancy'])
        if any((q['_dedup_key']==dedup_key) for q in per_sample[sample]):
            match_stats['duplicate_xml_annotation_suppressed']+=1; continue
        rr=dict(r); rr['match_method']=method; rr['_dedup_key']=dedup_key
        per_sample[sample].append(rr); match_stats['matched_reader_annotation_rows']+=1; match_stats[method]+=1
    sample_rows={}
    for sample,records in per_sample.items():
        valid_targets=[r['malignancy'] for r in records if r['malignancy'] is not None]
        if not valid_targets: match_stats['matched_samples_dropped_no_valid_malignancy']+=1; continue
        x={name:float(np.median([r['values'][name] for r in records if r['values'].get(name) is not None])) for name in FEATURES}
        if not all(np.isfinite(v) for v in x.values()): match_stats['matched_samples_dropped_missing_feature']+=1; continue
        sample_rows[sample]={'sample_index':sample,'patient_id':h5_meta[sample]['patient_id'],'split':'train' if sample in split['train']['indices'] else 'val','reader_count':len(records),'target_reader_count':len(valid_targets),'X_xml':x,'y_median':float(np.median(valid_targets)),'y_mean':float(np.mean(valid_targets)),'target_ratings':valid_targets}
    # Load and join only train.npz and val.npz. No test archive path is constructed/opened.
    tr=archive_data(a.features_dir/'train.npz','train',split); va=archive_data(a.features_dir/'val.npz','val',split)
    for arc,split_name in ((tr,'train'),(va,'val')):
        for i,idx in enumerate(arc['sample_index']):
            if arc['patient_id'][i]!=h5_meta[int(idx)]['patient_id']:
                raise ValueError(f"Feature archive/H5 patient mismatch for sample_index={idx}")
    # XML annotation feature/target rows matched to frozen H5 sample indices.
    def table_from_archive(arc, name):
        idxs=[]; xxml=[]; ymed=[]; ymean=[]; xrank=[]; targets=[]
        for i,sample in enumerate(arc['sample_index']):
            row=sample_rows.get(int(sample))
            if row is None or row['split']!=name: continue
            idxs.append(int(sample)); xxml.append([row['X_xml'][f] for f in FEATURES]); ymed.append(row['y_median']); ymean.append(row['y_mean']); xrank.append(arc['X'][i]); targets.append(i)
        if not idxs: return {'indices':np.array([],int),'X_xml':np.empty((0,6)),'y_median':np.array([]),'y_mean':np.array([]),'X_rank':np.empty((0,6)),'archive_indices':np.array([],int)}
        return {'indices':np.asarray(idxs,int),'X_xml':np.asarray(xxml,float),'y_median':np.asarray(ymed,float),'y_mean':np.asarray(ymean,float),'X_rank':np.asarray(xrank,float),'archive_indices':np.asarray(targets,int)}
    dt=table_from_archive(tr,'train'); dv=table_from_archive(va,'val')
    if len(dt['indices'])<a.min_calibration_nodules or len(dv['indices'])<20:
        raise RuntimeError(f"Insufficient exact XML/H5 matches: train={len(dt['indices'])}, val={len(dv['indices'])}. See matching stats; no model was promoted.")
    print(f"Matched usable physical nodules: train={len(dt['indices'])}, validation={len(dv['indices'])}")
    # Fit six monotone score-to-reader-rating calibrators on train only.
    calibrators={}; cal_stats={}
    for j,name in enumerate(FEATURES):
        cal=fit_isotonic(dt['X_rank'][:,j],dt['X_xml'][:,j])
        calibrators[name]=cal
        xval=dv['X_rank'][:,j]; yval=dv['X_xml'][:,j]
        if cal['kind']=='isotonic': valcal=cal['estimator'].predict(xval)
        else: valcal=np.full_like(xval,cal['value'])
        cal_stats[name]={'train_nodules':cal['n'],'train_unique_ranker_values':cal['unique_input_values'],'train_ranker_sd':float(np.std(dt['X_rank'][:,j])),'train_xml_rating_sd':float(np.std(dt['X_xml'][:,j])),'val_nodules_with_xml_match':int(len(yval)),'val_raw_ranker_sd':float(np.std(xval)),'val_xml_rating_sd':float(np.std(yval)),'val_spearman_raw_to_xml':spearman_safe(xval,yval),'val_raw_mae_to_xml_median':float(mean_absolute_error(yval,xval)),'val_calibrated_mae_to_xml_median':float(mean_absolute_error(yval,valcal)),'val_calibrated_spearman_to_xml':spearman_safe(valcal,yval),'val_calibrated_sd':float(np.std(valcal))}
    Xcal_train=apply_calibrators(dt['X_rank'],calibrators)
    # Model fitted against XML median features/target, so raw vs calibrated inference is explicit.
    mal_model=ExtraTreesRegressor(n_estimators=500,min_samples_leaf=8,max_features=1.0,random_state=2026,n_jobs=-1)
    mal_model.fit(dt['X_xml'],dt['y_median'])
    # Stage04 validation on ALL frozen val outputs against fixed reader-target archive; separately report matched-only rows.
    Xval_raw=va['X']; Xval_cal=apply_calibrators(Xval_raw,calibrators)
    pred_raw=np.clip(mal_model.predict(Xval_raw),1,5); pred_cal=np.clip(mal_model.predict(Xval_cal),1,5)
    train_median=float(np.median(dt['y_median'])); train_mean=float(np.mean(dt['y_median']))
    val_rows=[]
    for i,sample in enumerate(va['sample_index']):
        valid_y=np.isfinite(va['y_median'][i])
        val_rows.append({'sample_index':int(sample),'row':i,'has_xml_feature_match':int(sample) in sample_rows,'y_median':va['y_median'][i],'y_mean':va['y_mean'][i]})
    xml_feature_val_index=set(map(int,dv['indices']))
    all_val_mask=np.isfinite(va['y_median']); matched_val_mask=np.asarray([int(i) in xml_feature_val_index for i in va['sample_index']]) & all_val_mask
    out=a.out_dir; out.mkdir(parents=True,exist_ok=True)
    # Report both all validation samples and the exact XML-matched subset.
    stage04_metrics={}
    for subset,mask in [('all_validation_archive_rows',all_val_mask),('xml_matched_validation_nodules',matched_val_mask)]:
        if not mask.any(): continue
        stage04_metrics[subset]={
            'against_median_malignancy_target':{'raw_ranker_scores':score(va['y_median'][mask],pred_raw[mask],train_median),'calibrated_ranker_scores':score(va['y_median'][mask],pred_cal[mask],train_median)},
            'against_mean_malignancy_target':{'raw_ranker_scores':score(va['y_mean'][mask],pred_raw[mask],train_mean),'calibrated_ranker_scores':score(va['y_mean'][mask],pred_cal[mask],train_mean)},
        }
    bundle={'model':mal_model,'feature_calibrators':{k:{kk:vv for kk,vv in v.items() if kk!='estimator'}|({'estimator':v['estimator']} if v['kind']=='isotonic' else {}) for k,v in calibrators.items()},'feature_names_in_order':FEATURES,'target':'median LIDC reader malignancy/suspicion rating per matched physical nodule','target_scale':[1,5],'input_semantics':'six frozen Stage 03 scalar attribute scores; each is calibrated to the matched train XML reader-rating scale before Stage04 regression','model_type':'ExtraTreesRegressor + per-attribute train-only isotonic calibration','stage03_rankers_modified':False,'h5_patches_read':False,'h5_reader_labels_read_train_val_only':True}
    model_path=out/'stage04_xml_calibrated_malignancy.joblib'; joblib.dump(bundle,model_path)
    pred_path=out/'stage04_ranker_validation_predictions.csv'
    with pred_path.open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f); w.writerow(['sample_index','patient_id','has_xml_feature_match','target_median','target_mean','raw_prediction','calibrated_prediction',*[f'raw_{x}' for x in FEATURES],*[f'calibrated_{x}' for x in FEATURES]])
        for i,idx in enumerate(va['sample_index']):
            w.writerow([int(idx),va['patient_id'][i],int(int(idx) in sample_rows),va['y_median'][i],va['y_mean'][i],pred_raw[i],pred_cal[i],*va['X'][i].tolist(),*Xval_cal[i].tolist()])
    table_path=out/'matched_xml_nodule_table.csv'
    with table_path.open('w',newline='',encoding='utf-8') as f:
        cols=['sample_index','patient_id','split','reader_count','target_reader_count','malignancy_median','malignancy_mean']+FEATURES
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader()
        for idx,row in sorted(sample_rows.items()):
            w.writerow({'sample_index':idx,'patient_id':row['patient_id'],'split':row['split'],'reader_count':row['reader_count'],'target_reader_count':row['target_reader_count'],'malignancy_median':row['y_median'],'malignancy_mean':row['y_mean'],**row['X_xml']})
    report={'status':'completed','created_at_utc':datetime.now(timezone.utc).isoformat(),'python_version':platform.python_version(),'scikit_learn_version':sklearn.__version__,'features_in_order':FEATURES,'match_fields_used_only_for_identity':MATCH_FIELDS,'matching_policy':'Exact patient+study+series+full eight non-malignancy attribute signature; accepted only when all matching H5 reader slots point to one physical sample. Reader ID and malignancy target are not used in identity matching.','inputs':{'lidc_root':str(a.lidc_root),'h5_path':str(a.h5),'h5_full_file_sha256_computed':False,'split_json':str(a.split_json),'train_archive_sha256':tr['archive_hash'],'val_archive_sha256':va['archive_hash']},'h5_annotation_index':h5_info,'h5_match_index_stats':h5_stats,'xml_parse_stats':xml_stats,'matching_stats':dict(match_stats),'matched_physical_nodules':{'train':int(len(dt['indices'])),'validation':int(len(dv['indices']))},'feature_calibration_validation':cal_stats,'stage04_ranker_validation':stage04_metrics,'train_target_baselines':{'median_of_train_nodule_medians':train_median,'mean_of_train_nodule_medians':train_mean},'outputs':{'model':str(model_path),'matched_xml_table':str(table_path),'ranker_predictions':str(pred_path)},'test_archive_opened':False,'test_h5_reader_labels_read':False,'h5_patches_or_pixeldata_read':False,'stage03_rankers_modified':False,'fixed_split_modified':False,'warnings':['Stage03 train archive predictions are in-sample outputs from frozen train-fitted rankers; calibration generalization is judged on the disjoint validation split.','Rows without unique train/validation XML/H5 signature match are excluded from calibration and XML malignancy model fitting.']}
    (out/'training_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'status':'completed','model_path':str(model_path),'report_path':str(out/'training_report.json'),'nodule_table_path':str(table_path),'validation_predictions_path':str(pred_path),'matched_train_nodules':len(dt['indices']),'matched_validation_nodules':len(dv['indices']),'feature_calibration_validation':cal_stats,'stage04_ranker_validation':stage04_metrics,'test_archive_opened':False,'test_h5_reader_labels_read':False,'h5_patches_or_pixeldata_read':False,'stage03_rankers_modified':False},indent=2))
    return 0

if __name__=='__main__':
    try: raise SystemExit(main())
    except Exception as e:
        print(f'ERROR: {e}',file=sys.stderr); raise
