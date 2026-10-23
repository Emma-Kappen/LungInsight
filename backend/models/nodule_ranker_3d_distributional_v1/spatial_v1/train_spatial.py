#!/usr/bin/env python3
"""Train one matched LungInsight spatial-stride model.

Use --spatial-variant stride8 or stride4. All other settings are identical.
H5 and canonical split are read-only; TEST patches/labels are not loaded.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,math,random,time
from pathlib import Path
from typing import Any
import h5py,numpy as np,torch
import torch.nn.functional as F
from scipy.stats import pearsonr,spearmanr
from torch.utils.data import Dataset,DataLoader
from model_spatial import MODEL_TARGETS,RATING_MAX,ORDERED_TARGETS,READER_TO_MODEL,SpatialNoduleRanker3D,architecture_config,preprocess_hu,HU_LOW,HU_HIGH,HU_SHIFT,FOLD_MEAN,FOLD_STD

EXP_N={"train":1859,"val":437,"test":355}
EXP_HASH={"train":"1814147c1f1f0ccfbda73b56c5dbb392c14412b72c6b0fe316ee7a81957ad922","val":"0831f0b8e0ba4b54de794c914677e783d21197bedc5c6f93c695eb07011051f7","test":"e3ebe6f752237c58240e0888492679295ab068d6929cd75be1df83b821e4b56c"}

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def hsh(x): return hashlib.sha256(np.asarray(x,dtype=np.int64).tobytes()).hexdigest()
def dec(v): return v.decode("utf-8",errors="replace") if isinstance(v,bytes) else str(v)

def load_split(path,n,patients):
    raw=json.loads(Path(path).read_text(encoding="utf-8")); out={}
    for name in ("train","val","test"):
        e=raw[name]; idx=e.get("sample_indices",e.get("nodule_indices",e.get("indices")))
        if idx is None: raise ValueError(f"{name}: missing sample indices")
        idx=np.asarray(idx,dtype=np.int64)
        if len(idx)!=EXP_N[name] or hsh(idx)!=EXP_HASH[name]: raise ValueError(f"{name}: canonical split mismatch")
        out[name]=idx
    for a,b in (("train","val"),("train","test"),("val","test")):
        if set(out[a].tolist()) & set(out[b].tolist()): raise ValueError(f"sample leakage {a}/{b}")
        if set(patients[out[a]]) & set(patients[out[b]]): raise ValueError(f"patient leakage {a}/{b}")
    if sum(map(len,out.values()))!=n: raise ValueError("split does not partition H5")
    return out

def targets_from_h5(reader_labels,reader_mask):
    lab=reader_labels[:,:,READER_TO_MODEL].astype(np.float32); mask=reader_mask.astype(bool)
    valid=np.zeros_like(lab,dtype=bool); invalid=[]
    for h,name in enumerate(MODEL_TARGETS):
        k=RATING_MAX[name]
        active=mask & np.isfinite(lab[:,:,h])
        v=active & (lab[:,:,h]>=1) & (lab[:,:,h]<=k)
        nbad=int((active & ~v).sum())
        if nbad: invalid.append((name,nbad))
        valid[:,:,h]=v
    return lab,valid,invalid

class H5DS(Dataset):
    def __init__(self,h5,idx,labs,valid):
        self.path=str(h5); self.idx=np.asarray(idx); self.labs=labs[self.idx]; self.valid=valid[self.idx]; self.f=None
    def _open(self):
        if self.f is None: self.f=h5py.File(self.path,"r")
        return self.f
    def __len__(self): return len(self.idx)
    def __getitem__(self,j):
        x=np.asarray(self._open()["patches"][int(self.idx[j])],dtype=np.float32)
        return torch.from_numpy(x),torch.from_numpy(self.labs[j]),torch.from_numpy(self.valid[j])
    def __del__(self):
        try:
            if self.f is not None:self.f.close()
        except Exception: pass

def collate(b):
    x,y,v=zip(*b); return torch.stack(x),torch.stack(y),torch.stack(v)

def ordinal_loss(logits,y,v,k):
    # Each reader contributes; mean losses within a nodule, then mean nodules.
    q=logits[:,None,:].expand(-1,y.shape[1],-1)
    thr=torch.arange(1,k,device=y.device,dtype=y.dtype)[None,None,:]
    target=(y[:,:,None]>thr).float(); m=v[:,:,None].float().expand_as(target)
    per=F.binary_cross_entropy_with_logits(q,target,reduction="none")
    per=(per*m).sum(-1)/m.sum(-1).clamp_min(1)
    use=v.any(1); return per[use].mean() if use.any() else logits.sum()*0

def nominal_loss(logits,y,v,k):
    b,r=y.shape; L=logits[:,None,:].expand(-1,r,-1).reshape(b*r,k); yy=(y.reshape(-1).long()-1).clamp(0,k-1)
    ce=F.cross_entropy(L,yy,reduction="none").reshape(b,r)*v.float(); per=ce.sum(1)/v.float().sum(1).clamp_min(1); use=v.any(1)
    return per[use].mean() if use.any() else logits.sum()*0

def loss_all(model,outs,y,v):
    vals=[]; detail={}
    for h,name in enumerate(MODEL_TARGETS):
        k=RATING_MAX[name]; lh=ordinal_loss(outs[h],y[:,:,h],v[:,:,h],k) if name in ORDERED_TARGETS else nominal_loss(outs[h],y[:,:,h],v[:,:,h],k)
        vals.append(lh); detail[name]=float(lh.detach())
    return torch.stack(vals).mean(),detail

def mean_targets(y, v):
    # Reader arrays contain NaN for inactive/missing readers.  Multiplying
    # NaN by False/0 still produces NaN in NumPy, so never mask by y*v.
    # Select only valid entries and explicitly replace inactive entries with 0.
    safe_y = np.where(v, y, 0.0)
    den = v.sum(1).clip(min=1)
    return (safe_y.sum(1) / den).astype(np.float32)

def corr(fn, p, t, mask):
    # Be defensive against missing/invalid labels or non-finite model outputs.
    ok = mask & np.isfinite(p) & np.isfinite(t)
    p = p[ok]
    t = t[ok]
    if len(p) < 3 or np.std(p) < 1e-12 or np.std(t) < 1e-12:
        return float("nan")
    return float(fn(p, t).statistic)

def metrics(model,outs,y,v):
    p=model.expected_rating(outs).detach().cpu().numpy(); yy=y.detach().cpu().numpy(); vv=v.detach().cpu().numpy(); means=mean_targets(yy,vv)
    rows={}; rhos=[]; rhos3=[]
    for h,name in enumerate(MODEL_TARGETS):
        ok=vv[:,:,h].any(1); cnt=vv[:,:,h].sum(1); m3=ok&(cnt>=3); true=means[:,h]; pred=p[:,h]
        rows[name]={"n":int(ok.sum()),"n_ge3":int(m3.sum()),"mae":float(np.mean(np.abs(pred[ok]-true[ok]))),"rmse":float(np.sqrt(np.mean((pred[ok]-true[ok])**2))),"spearman":corr(spearmanr,pred,true,ok),"spearman_ge3":corr(spearmanr,pred,true,m3),"pearson":corr(pearsonr,pred,true,ok),"pred_std":float(np.std(pred[ok & np.isfinite(pred) & np.isfinite(true)])),"true_std":float(np.std(true[ok & np.isfinite(pred) & np.isfinite(true)])),"sd_ratio":float(np.std(pred[ok & np.isfinite(pred) & np.isfinite(true)])/np.std(true[ok & np.isfinite(pred) & np.isfinite(true)])) if np.std(true[ok & np.isfinite(pred) & np.isfinite(true)])>1e-12 else float("nan")}
        if name in ORDERED_TARGETS:
            if math.isfinite(rows[name]["spearman"]):rhos.append(rows[name]["spearman"])
            if math.isfinite(rows[name]["spearman_ge3"]):rhos3.append(rows[name]["spearman_ge3"])
    return rows,(float(np.mean(rhos3)) if rhos3 else float("nan")),(float(np.mean(rhos)) if rhos else float("nan"))

@torch.inference_mode()
def evaluate(model,loader,device):
    model.eval(); total=n=0; outs=[[] for _ in MODEL_TARGETS]; ys=[]; vs=[]
    for x,y,v in loader:
        x=preprocess_hu(x.to(device,non_blocking=True)).unsqueeze(1); y=y.to(device); v=v.to(device); o=model(x); l,_=loss_all(model,o,y,v)
        total+=float(l)*x.shape[0]; n+=x.shape[0]
        for h,t in enumerate(o):outs[h].append(t.cpu())
        ys.append(y.cpu());vs.append(v.cpu())
    o=tuple(torch.cat(z) for z in outs); y=torch.cat(ys);v=torch.cat(vs); met,m6,m6all=metrics(model,o,y,v);return total/max(n,1),met,m6,m6all

def train_epoch(model,loader,opt,scaler,device,amp_dtype,clip):
    model.train();total=n=0
    for x,y,v in loader:
        x=preprocess_hu(x.to(device,non_blocking=True)).unsqueeze(1);y=y.to(device);v=v.to(device);opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type,dtype=amp_dtype,enabled=amp_dtype is not None):o=model(x);l,_=loss_all(model,o,y,v)
        if scaler is not None:
            scaler.scale(l).backward();scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(model.parameters(),clip);scaler.step(opt);scaler.update()
        else:
            l.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),clip);opt.step()
        total+=float(l.detach())*x.shape[0];n+=x.shape[0]
    return total/max(n,1)

def clean(x):
    if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
    if isinstance(x,list):return [clean(v) for v in x]
    if isinstance(x,float) and not math.isfinite(x):return None
    if isinstance(x,np.generic):return clean(x.item())
    return x

def main(a):
    seed_all(a.seed); h5=Path(a.h5).resolve(); split_path=Path(a.split_file).resolve(); out=Path(a.out_dir).resolve()/a.spatial_variant/f"seed_{a.seed}";out.mkdir(parents=True,exist_ok=True)
    device=torch.device("cuda" if a.device=="auto" and torch.cuda.is_available() else ("cpu" if a.device=="auto" else a.device))
    print(f"\nLUNGINSIGHT SPATIAL_V1 | variant={a.spatial_variant} | seed={a.seed} | device={device}",flush=True)
    if device.type=="cuda":print(f"GPU={torch.cuda.get_device_name(device)}",flush=True)
    with h5py.File(h5,"r") as f:
        n=int(f["patches"].shape[0]); patients=np.asarray([dec(v) for v in f["patient_id"][:]],dtype=object)
    if n!=2651:raise ValueError(f"Expected 2651 nodules, got {n}")
    split=load_split(split_path,n,patients)
    # Read reader supervision only for TRAIN + VAL. TEST labels are never loaded.
    use_idx=np.concatenate([split["train"],split["val"]])
    # h5py requires fancy indices to be strictly increasing. The canonical
    # split preserves its own ordering, which is not guaranteed to be sorted.
    # Read TRAIN+VAL labels in sorted H5 order, then restore the original
    # TRAIN/VAL concatenation order so no target-to-patch alignment changes.
    read_order=np.argsort(use_idx, kind="stable")
    sorted_idx=use_idx[read_order]
    inverse_order=np.empty_like(read_order)
    inverse_order[read_order]=np.arange(len(read_order),dtype=read_order.dtype)
    with h5py.File(h5,"r") as f:
        rl_sorted=np.asarray(f["reader_labels"][sorted_idx],dtype=np.float32)
        rm_sorted=np.asarray(f["reader_mask"][sorted_idx],dtype=bool)
    rl=rl_sorted[inverse_order]
    rm=rm_sorted[inverse_order]
    labs,valid,invalid=targets_from_h5(rl,rm)
    train_local=np.arange(len(split["train"]),dtype=np.int64)
    val_local=np.arange(len(split["train"]),len(use_idx),dtype=np.int64)
    for name,c in invalid:print(f"Excluding {c} invalid {name} rating(s)",flush=True)
    # H5DS receives labels aligned to a compact TRAIN+VAL array but indexes patches by canonical H5 indices.
    # Build a patch-label lookup for only the used indices.
    class UsedH5DS(H5DS):
        def __init__(self,h5,patch_indices,labels,valid):
            self.path=str(h5);self.idx=np.asarray(patch_indices);self.labs=labels;self.valid=valid;self.f=None
    tr=UsedH5DS(h5,split["train"],labs[train_local],valid[train_local]);va=UsedH5DS(h5,split["val"],labs[val_local],valid[val_local])
    kwargs=dict(batch_size=a.batch_size,num_workers=a.num_workers,pin_memory=device.type=="cuda",collate_fn=collate,persistent_workers=a.num_workers>0)
    tl=DataLoader(tr,shuffle=True,drop_last=False,**kwargs);vl=DataLoader(va,shuffle=False,drop_last=False,**kwargs)
    model=SpatialNoduleRanker3D(a.spatial_variant,a.attn_dim,a.head_hidden,a.dropout).to(device);print(f"parameters={model.parameter_count():,}",flush=True);spatial_desc={"spatial16":"64x16x16x16 (4096 tokens)","stride8":"128x8x8x8 (512 tokens)","stride4":"192x4x4x4 (64 tokens)"}
    print(f"spatial output={spatial_desc[a.spatial_variant]}",flush=True);print("TEST is not loaded.",flush=True)
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=a.weight_decay); steps=max(1,len(tl)); total_steps=a.epochs*steps; warm=max(1,a.warmup_epochs*steps)
    def lrfn(s):
        if s<warm:return (s+1)/warm
        q=(s-warm)/max(1,total_steps-warm);return a.min_lr_factor+(1-a.min_lr_factor)*0.5*(1+math.cos(math.pi*min(1,q)))
    sched=torch.optim.lr_scheduler.LambdaLR(opt,lrfn)
    amp_dtype=torch.bfloat16 if device.type=="cuda" and a.amp=="bf16" and torch.cuda.is_bf16_supported() else torch.float16 if device.type=="cuda" and a.amp=="fp16" else None
    scaler=torch.cuda.amp.GradScaler(enabled=amp_dtype==torch.float16)
    best=-float("inf");best_epoch=-1;best_metric_name=None;bad=0;hist=[];t0=time.time()
    for ep in range(1,a.epochs+1):
        trl=train_epoch(model,tl,opt,scaler if amp_dtype==torch.float16 else None,device,amp_dtype,a.grad_clip)
        for _ in range(steps):sched.step()
        vl_loss,vm,m6,m6all=evaluate(model,vl,device)
        finite_heads=sum(math.isfinite(vm[n]["spearman"]) for n in ORDERED_TARGETS)
        collapsed=[n for n in MODEL_TARGETS if vm[n]["pred_std"]<1e-4]
        if math.isfinite(m6):
            score=m6; selection_metric="M6>=3"
        elif math.isfinite(m6all):
            score=m6all; selection_metric="M6all"
        else:
            # Ranking is undefined when every ordered head is constant.
            # Still checkpoint the best finite validation-loss model so that
            # this diagnostic run cannot finish without a reproducible model.
            score=-vl_loss; selection_metric="-val_loss_fallback"
        hist.append({"epoch":ep,"train_loss":trl,"val_loss":vl_loss,"m6_spearman_ge3":m6,"m6_spearman_all":m6all,"finite_ordered_head_spearman":finite_heads,"collapsed_heads":collapsed,"selection_metric":selection_metric,"selection_score":score,"head_metrics":vm,"lr":opt.param_groups[0]["lr"]})
        print(f"epoch {ep:03d} | train={trl:.4f} val={vl_loss:.4f} M6>=3={m6:+.4f} M6all={m6all:+.4f} finite_heads={finite_heads}/6 collapsed={len(collapsed)}/8 [{','.join(collapsed) if collapsed else 'none'}]",flush=True)
        if score>best:
            best=score;best_epoch=ep;best_metric_name=selection_metric;bad=0;torch.save({"model_state_dict":model.state_dict(),"architecture":architecture_config(a.spatial_variant,a.attn_dim,a.head_hidden,a.dropout),"epoch":ep,"seed":a.seed,"best_metric":score,"validation_metrics":vm,"training_config":vars(a),"test_used_for_selection":False,"split_hashes":{"train":hsh(split["train"]),"val":hsh(split["val"]),"test":hsh(split["test"])}},out/"best.pth")
        else:
            bad+=1
            if bad>=a.patience:print(f"Early stopping; best epoch={best_epoch}",flush=True);break
    report={"variant":a.spatial_variant,"seed":a.seed,"best_epoch":best_epoch,"best_selection_score":best,"best_selection_metric":best_metric_name,"best_metric_semantics":"M6>=3, then M6all, then -val_loss fallback when all ordered Spearman values are undefined","elapsed_seconds":time.time()-t0,"history":hist,"checkpoint":str((out/"best.pth").resolve()),"h5_modified":False,"split_modified":False,"test_used_for_selection":False}
    (out/"run.json").write_text(json.dumps(clean(report),indent=2),encoding="utf-8")
    if hist:
        with (out/"history.csv").open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=list(hist[0]));w.writeheader();w.writerows(hist)
    print(f"Saved {out/'best.pth'} | best {best_metric_name}={best:+.4f} epoch={best_epoch}",flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--h5",required=True);p.add_argument("--split-file",required=True);p.add_argument("--out-dir",required=True);p.add_argument("--spatial-variant",choices=["spatial16","stride8","stride4"],required=True);p.add_argument("--device",default="auto");p.add_argument("--batch-size",type=int,default=8);p.add_argument("--num-workers",type=int,default=2);p.add_argument("--epochs",type=int,default=150);p.add_argument("--patience",type=int,default=25);p.add_argument("--warmup-epochs",type=int,default=5);p.add_argument("--lr",type=float,default=5e-4);p.add_argument("--weight-decay",type=float,default=0.05);p.add_argument("--min-lr-factor",type=float,default=0.05);p.add_argument("--grad-clip",type=float,default=1.0);p.add_argument("--attn-dim",type=int,default=96);p.add_argument("--head-hidden",type=int,default=64);p.add_argument("--dropout",type=float,default=0.2);p.add_argument("--amp",choices=["none","bf16","fp16"],default="bf16");p.add_argument("--seed",type=int,default=123);main(p.parse_args())
