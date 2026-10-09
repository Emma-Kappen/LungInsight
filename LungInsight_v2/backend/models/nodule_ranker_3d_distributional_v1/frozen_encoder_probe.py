#!/usr/bin/env python3
"""LungInsight v2 frozen pretrained-encoder probe.

Read-only diagnostic on the fixed H5 + canonical split.  The pretrained
Encoder3D is frozen; only lightweight Ridge and MLP probes are fitted on TRAIN
reader-mean targets and evaluated on VAL.  TEST is never loaded for fitting or
model selection.
"""
from __future__ import annotations

import argparse, csv, hashlib, json, math, os, random, sys, time
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr

try:
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
except ImportError as e:
    raise SystemExit("scikit-learn is required: python -m pip install scikit-learn") from e

# Fixed LIDC/project specification.
TARGETS = ["subtlety","internal_structure","calcification","sphericity","margin","lobulation","spiculation","texture"]
K = {"subtlety":5,"internal_structure":4,"calcification":6,"sphericity":5,"margin":5,"lobulation":5,"spiculation":5,"texture":5}
READER_TO_MODEL = [7,1,0,5,4,2,6,8]
READER_NAMES = ["calcification","internalStructure","lobulation","malignancy","margin","sphericity","spiculation","subtlety","texture"]
INPUT_SHAPE = (64,64,64)
HU_LOW, HU_HIGH, HU_SHIFT = -1000.0, 400.0, 1000.0
FOLD_MEAN, FOLD_STD = 357.87565, 288.47657
ENCODER_CFG = {"stem_channels":16,"channels":[32,64,128,192],"blocks":[1,2,2,1]}
EXPECTED = {"n":2651,"train":1859,"val":437,"test":355}
HASHES = {
    "train":"1814147c1f1f0ccfbda73b56c5dbb392c14412b72c6b0fe316ee7a81957ad922",
    "val":"0831f0b8e0ba4b54de794c914677e783d21197bedc5c6f93c695eb07011051f7",
    "test":"e3ebe6f752237c58240e0888492679295ab068d6929cd75be1df83b821e4b56c",
}

def log(s): print(s, flush=True)
def clean(x):
    if isinstance(x,dict): return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [clean(v) for v in x]
    if isinstance(x,np.ndarray): return clean(x.tolist())
    if isinstance(x,np.generic): return clean(x.item())
    if isinstance(x,Path): return str(x)
    if isinstance(x,float) and not math.isfinite(x): return None
    return x
def save_json(p,obj): Path(p).write_text(json.dumps(clean(obj),indent=2,sort_keys=True),encoding="utf-8")
def hash_indices(x): return hashlib.sha256(np.asarray(x,dtype=np.int64).tobytes()).hexdigest()
def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
def choose_device(s):
    if s=="auto": return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d=torch.device(s)
    if d.type=="cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA requested but unavailable")
    return d

def _gn(c,g=8): return nn.GroupNorm(min(g,c),c)
class ResBlock3D(nn.Module):
    def __init__(self,cin,cout,stride=1):
        super().__init__(); self.conv1=nn.Conv3d(cin,cout,3,stride,1,bias=False); self.n1=_gn(cout)
        self.conv2=nn.Conv3d(cout,cout,3,1,1,bias=False); self.n2=_gn(cout); self.act=nn.GELU(); self.short=None
        if stride!=1 or cin!=cout: self.short=nn.Sequential(nn.Conv3d(cin,cout,1,stride,bias=False),_gn(cout))
    def forward(self,x):
        y=self.act(self.n1(self.conv1(x))); y=self.n2(self.conv2(y)); return self.act(y+(x if self.short is None else self.short(x)))
class Encoder3D(nn.Module):
    def __init__(self,stem_channels=16,channels=(32,64,128,192),blocks=(1,2,2,1)):
        super().__init__(); layers=[]; cin=stem_channels
        self.stem=nn.Sequential(nn.Conv3d(1,stem_channels,3,1,1,bias=False),_gn(stem_channels),nn.GELU())
        for c,nb in zip(channels,blocks):
            for i in range(nb): layers.append(ResBlock3D(cin,c,2 if i==0 else 1)); cin=c
        self.stages=nn.Sequential(*layers); self.out_channels=cin
        for m in self.modules():
            if isinstance(m,nn.Conv3d): nn.init.kaiming_normal_(m.weight,mode="fan_out",nonlinearity="relu")
    def forward(self,x): return self.stages(self.stem(x))

def preprocess(x): return (x.float().clamp(HU_LOW,HU_HIGH)+HU_SHIFT-FOLD_MEAN)/FOLD_STD

def decode(v): return v.decode("utf-8",errors="replace") if isinstance(v,bytes) else str(v)
def split_load(p,n,patients):
    d=json.loads(Path(p).read_text(encoding="utf-8")); out={}
    for name in ("train","val","test"):
        e=d.get(name); 
        if not isinstance(e,dict): raise ValueError(f"{p}: {name} must be an object")
        idx=e.get("sample_indices",e.get("nodule_indices",e.get("indices")))
        if not isinstance(idx,list) or not all(isinstance(i,int) for i in idx): raise ValueError(f"{p}: bad {name} indices")
        idx=np.asarray(idx,dtype=np.int64)
        if len(idx)!=len(np.unique(idx)) or np.any(idx<0) or np.any(idx>=n): raise ValueError(f"{p}: invalid {name} indices")
        listed=e.get("patients",e.get("patient_ids"))
        if listed is not None and sorted(set(map(str,listed)))!=sorted(set(map(str,patients[idx]))): raise ValueError(f"{p}: {name} patient list mismatch")
        out[name]=idx
    for a,b in (("train","val"),("train","test"),("val","test")):
        if set(out[a].tolist())&set(out[b].tolist()): raise ValueError(f"sample leakage: {a}/{b}")
        if set(patients[out[a]])&set(patients[out[b]]): raise ValueError(f"patient leakage: {a}/{b}")
    for name in out:
        if len(out[name])!=EXPECTED[name]: raise ValueError(f"canonical split length mismatch {name}: {len(out[name])} != {EXPECTED[name]}")
        h=hash_indices(out[name])
        if h!=HASHES[name]: raise ValueError(f"canonical split hash mismatch {name}: {h} != {HASHES[name]}")
    if sum(len(x) for x in out.values())!=n: raise ValueError("split does not partition H5")
    return out

def build_targets(labels,mask):
    lab=labels[:,:,READER_TO_MODEL]
    if labels.shape[1:]!=(8,9) or mask.shape!=labels.shape[:2]: raise ValueError(f"unexpected label/mask shapes {labels.shape} {mask.shape}")
    n=labels.shape[0]; km=max(K.values()); dist=np.zeros((n,8,km),np.float32); cnt=np.zeros((n,8),np.float32)
    for h,name in enumerate(TARGETS):
        k=K[name]; v=lab[:,:,h]; ok=mask & np.isfinite(v) & (v>=1) & (v<=k); cnt[:,h]=ok.sum(1)
        den=np.maximum(cnt[:,h],1)
        for r in range(1,k+1): dist[:,h,r-1]=(ok & (v==r)).sum(1)/den
        invalid=int((mask & np.isfinite(v) & ((v<1)|(v>k))).sum())
        if invalid: log(f"  {name}: excluding {invalid} invalid active rating(s) outside 1..{k}")
    vals=np.arange(1,km+1,dtype=np.float32); mean=(dist*vals[None,None,:]).sum(2); valid=cnt>0
    for h,name in enumerate(TARGETS):
        if valid[:,h].any() and not np.allclose(dist[valid[:,h],h,:K[name]].sum(1),1,atol=1e-6): raise ValueError(f"{name}: target distributions do not sum to 1")
    return {"mean":mean.astype(np.float32),"count":cnt,"valid":valid,"distribution":dist}

def load_encoder(cp,train_idx,device):
    ck=torch.load(cp,map_location="cpu",weights_only=False)
    if not isinstance(ck,dict) or "encoder_state_dict" not in ck or "encoder_config" not in ck: raise ValueError("checkpoint is missing encoder_state_dict/encoder_config")
    cfg={"stem_channels":int(ck["encoder_config"]["stem_channels"]),"channels":[int(x) for x in ck["encoder_config"]["channels"]],"blocks":[int(x) for x in ck["encoder_config"]["blocks"]]}
    if cfg!=ENCODER_CFG: raise ValueError(f"encoder config mismatch: {cfg} != {ENCODER_CFG}")
    pre=ck.get("train_nodule_indices")
    if pre is None: raise ValueError("checkpoint has no train_nodule_indices; refusing to run without provenance")
    outside=set(map(int,pre))-set(map(int,train_idx.tolist()))
    if outside: raise RuntimeError(f"pretrained encoder leakage: {len(outside)} indices outside canonical TRAIN")
    m=Encoder3D(**cfg); m.load_state_dict(ck["encoder_state_dict"],strict=True); m.to(device).eval()
    for p in m.parameters(): p.requires_grad_(False)
    meta={"keys":sorted(map(str,ck.keys())),"encoder_config":cfg,"n_pretrain_nodules":len(set(map(int,pre))),"train_only_provenance":True,"epochs_completed":ck.get("epochs_completed"),"complete":ck.get("complete"),"pretrain_config":ck.get("pretrain_config"),"final_pretrain_loss":ck["history"][-1]["loss"] if isinstance(ck.get("history"),list) and ck["history"] else None}
    return m,meta

@torch.inference_mode()
def extract(h5p,enc,idx,bs,device):
    out=np.empty((len(idx),enc.out_channels*2),np.float32); pin=device.type=="cuda"
    with h5py.File(h5p,"r") as f:
        ds=f["patches"]
        for s in range(0,len(idx),bs):
            e=min(len(idx),s+bs); ids=idx[s:e]; order=np.argsort(ids); xs=np.asarray(ds[ids[order]],np.float32)[np.argsort(order)]
            x=torch.from_numpy(xs)
            if pin: x=x.pin_memory()
            x=x.to(device,non_blocking=pin); y=enc(preprocess(x).unsqueeze(1))
            z=torch.cat([y.mean((2,3,4)),y.amax((2,3,4))],1).cpu().numpy().astype(np.float32); out[s:e]=z
            if s==0 or e==len(idx) or s//bs%20==0: log(f"  embeddings {e}/{len(idx)} ({100*e/max(1,len(idx)):.1f}%)")
    return out

def metrics(p,t):
    p=np.asarray(p,float); t=np.asarray(t,float); ok=np.isfinite(p)&np.isfinite(t); p,t=p[ok],t[ok]
    ps=float(np.std(p)) if len(p) else float("nan"); ts=float(np.std(t)) if len(t) else float("nan")
    if len(p)>=3 and ps>1e-12 and ts>1e-12:
        pr=float(pearsonr(p,t).statistic); sr=float(spearmanr(p,t).statistic)
    else: pr=sr=float("nan")
    return {"n":len(p),"mae":float(np.mean(np.abs(p-t))),"rmse":float(np.sqrt(np.mean((p-t)**2))),"pearson":pr,"spearman":sr,"pred_std":ps,"true_std":ts,"sd_ratio":ps/ts if ts>1e-12 else float("nan")}

def base(train_y,val_y):
    v=float(np.median(train_y[np.isfinite(train_y)])); return v,metrics(np.full(len(val_y),v),val_y)

def main(a):
    set_seed(a.seed); h5=Path(a.h5).resolve(); splitp=Path(a.split_file).resolve(); cp=Path(a.pretrained_encoder).resolve(); out=Path(a.out_dir).resolve(); out.mkdir(parents=True,exist_ok=True); dev=choose_device(a.device)
    for p in (h5,splitp,cp):
        if not p.is_file(): raise FileNotFoundError(p)
    log("="*72); log("LUNGINSIGHT v2 / FROZEN PRETRAINED-ENCODER PROBE"); log("READ-ONLY: fixed H5 + canonical split + pretrained checkpoint"); log(f"H5: {h5}\nSplit: {splitp}\nCheckpoint: {cp}\nOutput: {out}\nDevice: {dev}")
    if dev.type=="cuda": log(f"CUDA: {torch.cuda.get_device_name(dev)}")
    with h5py.File(h5,"r") as f:
        n=int(f["patches"].shape[0]); ps=tuple(f["patches"].shape[1:]); patients=np.array([decode(v) for v in f["patient_id"][:]],dtype=object); labels=f["reader_labels"][:].astype(np.float32); mask=f["reader_mask"][:].astype(bool)
        if n!=EXPECTED["n"] or ps!=INPUT_SHAPE: raise ValueError(f"unexpected H5 patches {n}, {ps}")
    split=split_load(splitp,n,patients); log(f"Canonical split: {len(split['train'])}/{len(split['val'])}/{len(split['test'])}; TEST will not be loaded further")
    log("\n[1/4] Build reader-mean targets"); tg=build_targets(labels,mask)
    log("\n[2/4] Load + freeze encoder"); enc,ckmeta=load_encoder(cp,split["train"],dev); log(f"Encoder output channels={enc.out_channels}; raw embedding=384-D; pretrain nodules={ckmeta['n_pretrain_nodules']}; provenance=PASSED")
    log("\n[3/4] Extract TRAIN/VAL frozen embeddings"); t=time.time(); Xtr=extract(h5,enc,split["train"],a.batch_size,dev); Xva=extract(h5,enc,split["val"],a.batch_size,dev); sec=time.time()-t
    np.save(out/"train_embeddings.npy",Xtr); np.save(out/"val_embeddings.npy",Xva); np.save(out/"train_indices.npy",split["train"]); np.save(out/"val_indices.npy",split["val"])
    results=[]; decisions=[]
    log("\n[4/4] Fit TRAIN-only Ridge + MLP probes; evaluate VAL")
    for h,name in enumerate(TARGETS):
        mt=tg["valid"][split["train"],h]; mv=tg["valid"][split["val"],h]; yt=tg["mean"][split["train"],h][mt]; yv=tg["mean"][split["val"],h][mv]; xt=Xtr[mt]; xv=Xva[mv]
        bval,bm=base(yt,yv)
        ridge=make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),Ridge(alpha=a.ridge_alpha)); ridge.fit(xt,yt); pr=ridge.predict(xv); mr=metrics(pr,yv)
        results.append({"attribute":name,"probe":"ridge_linear","alpha":a.ridge_alpha,**mr,"baseline_value":bval,"baseline_mae":bm["mae"],"baseline_rmse":bm["rmse"],"mae_improvement_vs_baseline":bm["mae"]-mr["mae"]})
        mlp=make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),MLPRegressor(hidden_layer_sizes=(128,64),activation="relu",solver="adam",alpha=1e-3,batch_size=64,learning_rate_init=1e-3,max_iter=500,early_stopping=True,validation_fraction=.2,n_iter_no_change=25,random_state=a.seed+h)); mlp.fit(xt,yt); pm=mlp.predict(xv); mm=metrics(pm,yv)
        results.append({"attribute":name,"probe":"mlp_nonlinear","hidden_layers":"128,64",**mm,"baseline_value":bval,"baseline_mae":bm["mae"],"baseline_rmse":bm["rmse"],"mae_improvement_vs_baseline":bm["mae"]-mm["mae"],"mlp_iterations":int(mlp.named_steps["mlpregressor"].n_iter_)})
        best=max([r for r in (mr["spearman"],mm["spearman"]) if math.isfinite(r)],default=float("nan")); bestmae=min(mr["mae"],mm["mae"])
        if not math.isfinite(best): interp="NO_VALID_PROBE_RESULT"
        elif best>=.20: interp="CLEAR_FROZEN_SIGNAL"
        elif best>=.10: interp="WEAK_FROZEN_SIGNAL"
        else: interp="LITTLE_OR_NO_FROZEN_RANK_SIGNAL"
        decisions.append({"attribute":name,"best_probe_spearman":best,"best_probe_mae":bestmae,"baseline_mae":bm["mae"],"interpretation":interp})
        log(f"  {name:20s} Ridge rho={mr['spearman']:+.4f} MAE={mr['mae']:.4f} SD={mr['sd_ratio']:.3f} | MLP rho={mm['spearman']:+.4f} MAE={mm['mae']:.4f} SD={mm['sd_ratio']:.3f}")
    fields=sorted({k for r in results for k in r});
    with (out/"probe_results.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); [w.writerow(clean(r)) for r in results]
    valid_rho=[r["best_probe_spearman"] for r in decisions if math.isfinite(r["best_probe_spearman"])]; avg=float(np.mean(valid_rho)) if valid_rho else float("nan")
    strong=sum(r["interpretation"]=="CLEAR_FROZEN_SIGNAL" for r in decisions); weak=sum(r["interpretation"]=="WEAK_FROZEN_SIGNAL" for r in decisions)
    overall=("FROZEN_ENCODER_HAS_USEFUL_SIGNAL: retain the encoder and focus next on objective/head design." if strong>=2 else "FROZEN_ENCODER_HAS_SOME_SIGNAL: representation is not obviously the sole bottleneck; use controlled objective/head experiments." if strong+weak>=3 else "FROZEN_ENCODER_SIGNAL_IS_WEAK: representation quality remains a leading suspect; do not conclude from this alone that the full encoder must be replaced.")
    report={"probe_version":"1.0","read_only":True,"test_used_for_fitting_or_selection":False,"h5":str(h5),"split_file":str(splitp),"pretrained_encoder":ckmeta,"device":str(dev),"batch_size":a.batch_size,"seed":a.seed,"preprocessing":{"clip_low_hu":HU_LOW,"clip_high_hu":HU_HIGH,"shift_after_clip":HU_SHIFT,"mean":FOLD_MEAN,"std":FOLD_STD,"formula":"(clip(HU,-1000,400)+1000-357.87565)/288.47657"},"encoder":{"architecture":ENCODER_CFG,"out_channels":enc.out_channels,"pooling":"global_mean + global_max","embedding_dim":enc.out_channels*2,"frozen":True},"targets":{"names":TARGETS,"reader_names":READER_NAMES,"reader_to_model":READER_TO_MODEL,"authoritative_cardinality":K},"split":{"train_n":len(split["train"]),"val_n":len(split["val"]),"test_n":len(split["test"]),"train_hash":hash_indices(split["train"]),"val_hash":hash_indices(split["val"]),"test_hash":hash_indices(split["test"])},"embeddings":{"train_shape":list(Xtr.shape),"val_shape":list(Xva.shape),"extraction_seconds":sec},"results":results,"decision_by_attribute":decisions,"aggregate_mean_best_spearman":avg,"overall_interpretation":overall}
    save_json(out/"frozen_encoder_probe.json",report)
    txt=["LUNGINSIGHT v2 — FROZEN PRETRAINED-ENCODER PROBE","="*60,f"Mean best VAL Spearman: {avg:.4f}","","PER-ATTRIBUTE DECISION"]+[f"  {r['attribute']:20s} rho={r['best_probe_spearman']:+.4f} MAE={r['best_probe_mae']:.4f} baseline={r['baseline_mae']:.4f} | {r['interpretation']}" for r in decisions]+["","OVERALL",overall,"","FILES","  frozen_encoder_probe.json","  probe_results.csv","  train_embeddings.npy","  val_embeddings.npy"]
    (out/"FROZEN_PROBE_SUMMARY.txt").write_text("\n".join(txt)+"\n",encoding="utf-8"); log("\n"+"\n".join(txt)); return 0

def args():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--h5",required=True); p.add_argument("--split-file",required=True); p.add_argument("--pretrained-encoder",required=True); p.add_argument("--out-dir",required=True); p.add_argument("--device",default="auto"); p.add_argument("--batch-size",type=int,default=8); p.add_argument("--ridge-alpha",type=float,default=10.0); p.add_argument("--seed",type=int,default=123); return p.parse_args()
if __name__=="__main__":
    try: raise SystemExit(main(args()))
    except KeyboardInterrupt: raise SystemExit(130)
