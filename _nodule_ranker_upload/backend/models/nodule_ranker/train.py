#!/usr/bin/env python3
"""Fast HDF5-based LIDC 8-attribute training. Malignancy is not a neural target."""
import argparse,json,random,time
from pathlib import Path
from typing import cast
import h5py,numpy as np,torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from scipy.stats import pearsonr,spearmanr

ATTRIBUTE_NAMES=["subtlety","internalStructure","calcification","sphericity","margin","lobulation","spiculation","texture"]
RANGES={"subtlety":(1,5),"internalStructure":(1,4),"calcification":(1,6),"sphericity":(1,5),"margin":(1,5),"lobulation":(1,5),"spiculation":(1,5),"texture":(1,5)}
H5_ORDER=["calcification","internalStructure","lobulation","margin","sphericity","spiculation","subtlety","texture"]
H5_TO_MODEL=[H5_ORDER.index(a) for a in ATTRIBUTE_NAMES]
DEPTH=16; SIZE=64; HU_MIN=-1000.; HU_MAX=400.

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

class H5Dataset(Dataset):
    def __init__(self,patches,y,indices,train=False):
        self.x=patches; self.y=y; self.idx=np.asarray(indices); self.train=train
    def __len__(self): return len(self.idx)
    def __getitem__(self,k):
        i=int(self.idx[k]); v=self.x[i]
        ids=np.linspace(0,v.shape[0]-1,DEPTH).round().astype(np.int64); v=v[ids]
        if self.train:
            if random.random()<.5: v=v[:,:,::-1]
            if random.random()<.5: v=v[:,::-1,:]
        v=np.asarray(v,dtype=np.float32)
        mn=float(np.nanmin(v)); mx=float(np.nanmax(v))
        if mn < -2.0 or mx > 2.0:
            v=np.clip(v,HU_MIN,HU_MAX); v=(v-HU_MIN)/(HU_MAX-HU_MIN)
        v=np.nan_to_num(v,nan=0.0,posinf=1.0,neginf=0.0)
        return torch.from_numpy(v).unsqueeze(1),torch.from_numpy(self.y[i].astype(np.float32))

class Block(nn.Module):
    def __init__(self,ci,co):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(ci,co,3,padding=1,bias=False),nn.BatchNorm2d(co),nn.ReLU(inplace=True),
                               nn.Conv2d(co,co,3,padding=1,bias=False),nn.BatchNorm2d(co),nn.ReLU(inplace=True))
    def forward(self,x): return self.net(x)

class Net(nn.Module):
    def __init__(self,base=24):
        super().__init__()
        self.enc=nn.Sequential(Block(1,base),nn.MaxPool2d(2),Block(base,base*2),nn.MaxPool2d(2),
                                Block(base*2,base*4),nn.MaxPool2d(2),Block(base*4,base*8))
        c=base*8
        self.score=nn.Sequential(nn.Linear(c,c//4),nn.Tanh(),nn.Linear(c//4,1))
        self.norm=nn.LayerNorm(c); self.emb=nn.Parameter(torch.randn(8,c)*.02)
        self.heads=nn.ModuleList([nn.Sequential(nn.Linear(c,c//2),nn.ReLU(inplace=True),nn.Dropout(.20),
                                                nn.Linear(c,RANGES[a][1]-1)) for a in ATTRIBUTE_NAMES])
    def forward(self,x):
        b,s=x.shape[:2]; f=self.enc(x.reshape(b*s,1,SIZE,SIZE))
        f=F.adaptive_avg_pool2d(f,1).flatten(1).reshape(b,s,-1)
        w=self.score(f).squeeze(-1).softmax(1); f=self.norm((f*w.unsqueeze(-1)).sum(1))
        z=f.unsqueeze(1)+self.emb.unsqueeze(0)
        return [h(z[:,j]) for j,h in enumerate(self.heads)]

def soft_target(y,a):
    _,hi=RANGES[a]; k=hi-1
    return torch.stack([torch.clamp(y-(t+1)+1.0,0,1) for t in range(k)],dim=1)

def loss_fn(logits,y):
    return sum(F.binary_cross_entropy_with_logits(logits[j],soft_target(y[:,j],a))
               for j,a in enumerate(ATTRIBUTE_NAMES))/len(ATTRIBUTE_NAMES)

def decode(logits,a):
    return RANGES[a][0]+torch.sigmoid(logits).sum(1)

@torch.no_grad()
def evaluate(model,loader,device):
    model.eval(); P=[]; Y=[]
    for x,y in loader:
        out=model(x.to(device,non_blocking=True))
        P.append(np.stack([decode(out[j],a).cpu().numpy() for j,a in enumerate(ATTRIBUTE_NAMES)],1)); Y.append(y.numpy())
    p=np.concatenate(P); y=np.concatenate(Y); m={}
    for j,a in enumerate(ATTRIBUTE_NAMES):
        d=p[:,j]-y[:,j]; lo,hi=RANGES[a]; pr=np.clip(np.rint(p[:,j]),lo,hi); yr=np.rint(y[:,j])
        m[a]={"MAE":float(np.mean(abs(d))),"RMSE":float(np.sqrt(np.mean(d*d))),
              "Pearson":float(pearsonr(y[:,j],p[:,j]).statistic),"Spearman":float(spearmanr(y[:,j],p[:,j]).statistic), # type: ignore
              "ExactMatch":float(np.mean(pr==yr)),"Within1":float(np.mean(abs(pr-yr)<=1))}
    m["_aggregate"]={k:float(np.mean([v[k] for v in m.values()])) for k in ["MAE","RMSE","Pearson","Spearman","ExactMatch","Within1"]}
    return m

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--h5",type=Path,required=True); ap.add_argument("--output-dir",type=Path,required=True)
    ap.add_argument("--epochs",type=int,default=20); ap.add_argument("--batch-size",type=int,default=16)
    ap.add_argument("--lr",type=float,default=3e-4); ap.add_argument("--patience",type=int,default=4)
    ap.add_argument("--seed",type=int,default=42); ap.add_argument("--base",type=int,default=24)
    ap.add_argument("--workers",type=int,default=2); args=ap.parse_args()
    seed_all(args.seed); args.output_dir.mkdir(parents=True,exist_ok=True)
    with h5py.File(args.h5,"r") as f:
        patches_ds: h5py.Dataset = f["patches"] # type: ignore
        patients_ds: h5py.Dataset = f["patient_id"] # type: ignore
        y0_ds: h5py.Dataset = f["attribute_mean"] # type: ignore
        patches=np.asarray(patches_ds[...], dtype=np.float32)
        patients=np.asarray(patients_ds[...], dtype=str)
        y0=np.asarray(y0_ds[...], dtype=np.float32)
        raw_names=f.attrs["attribute_names"]
        if isinstance(raw_names, bytes):
            raw_names=raw_names.decode("utf-8")
        names=json.loads(str(raw_names))
    assert names==H5_ORDER,(names,H5_ORDER)
    y=y0[:,H5_TO_MODEL]; pids=np.unique(patients).tolist(); random.Random(args.seed).shuffle(pids)
    n=len(pids); nt=max(1,round(.15*n)); nv=max(1,round(.15*n))
    test=set(pids[:nt]); val=set(pids[nt:nt+nv]); train=set(pids[nt+nv:])
    ti=np.where(np.isin(patients,list(train)))[0]; vi=np.where(np.isin(patients,list(val)))[0]; ei=np.where(np.isin(patients,list(test)))[0]
    split={"seed":args.seed,"train_patients":sorted(train),"val_patients":sorted(val),"test_patients":sorted(test),
           "train_samples":len(ti),"val_samples":len(vi),"test_samples":len(ei)}
    (args.output_dir/"split.json").write_text(json.dumps(split,indent=2))
    pin=torch.cuda.is_available()
    tr=DataLoader(H5Dataset(patches,y,ti,True),batch_size=args.batch_size,shuffle=True,num_workers=args.workers,pin_memory=pin,persistent_workers=args.workers>0)
    va=DataLoader(H5Dataset(patches,y,vi),batch_size=args.batch_size,shuffle=False,num_workers=args.workers,pin_memory=pin,persistent_workers=args.workers>0)
    te=DataLoader(H5Dataset(patches,y,ei),batch_size=args.batch_size,shuffle=False,num_workers=args.workers,pin_memory=pin,persistent_workers=args.workers>0)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); model=Net(args.base).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=2e-5)
    sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode="min",factor=.5,patience=2,min_lr=1e-6)
    scaler=torch.amp.GradScaler("cuda",enabled=device.type=="cuda"); best=1e9; bad=0; hist=[]; start=time.time()
    print(f"Device={device} samples train/val/test={len(ti)}/{len(vi)}/{len(ei)} patients={len(train)}/{len(val)}/{len(test)}",flush=True)
    for ep in range(1,args.epochs+1):
        model.train(); total=0.; nb=0; t0=time.time()
        for x,yb in tr:
            x=x.to(device,non_blocking=True); yb=yb.to(device,non_blocking=True); opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda",dtype=torch.float16,enabled=device.type=="cuda"):
                loss=loss_fn(model(x),yb)
            scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(),2.0) #type: ignore   
            scaler.step(opt); scaler.update(); total+=float(loss.detach()); nb+=1 # type: ignore
        vm=evaluate(model,va,device)["_aggregate"]; sched.step(vm["MAE"])
        hist.append({"epoch":ep,"train_loss":total/max(1,nb),"val_metrics":vm,"seconds":time.time()-t0})
        print(f"Epoch {ep}: loss={hist[-1]['train_loss']:.5f} val_MAE={vm['MAE']:.4f} val_Spearman={vm['Spearman']:.4f} time={hist[-1]['seconds']:.1f}s",flush=True)
        ck={"model_state_dict":model.state_dict(),"model_config":{"base":args.base},"attribute_names":ATTRIBUTE_NAMES,
            "attribute_ranges":RANGES,"preprocessing":{"patch_size":SIZE,"depth":DEPTH,"hu_min":HU_MIN,"hu_max":HU_MAX,"h5_to_model":H5_TO_MODEL},
            "ordinal":True,"epoch":ep,"val_mae":vm["MAE"]}
        torch.save(ck,args.output_dir/"last.pth")
        if vm["MAE"]<best-1e-4:
            best=vm["MAE"]; bad=0; torch.save(ck,args.output_dir/"best.pth"); print("saved best.pth",flush=True)
        else: bad+=1
        (args.output_dir/"training_history.json").write_text(json.dumps(hist,indent=2))
        if bad>=args.patience: print("Early stopping",flush=True); break
    ck=torch.load(args.output_dir/"best.pth",map_location=device); model.load_state_dict(ck["model_state_dict"])
    tm=evaluate(model,te,device)
    summary={"epochs_completed":len(hist),"best_epoch":ck["epoch"],"best_val_mae":ck["val_mae"],"test_metrics":tm,
             "train_samples":len(ti),"val_samples":len(vi),"test_samples":len(ei),"train_patients":len(train),
             "val_patients":len(val),"test_patients":len(test),"seconds":time.time()-start,"malignancy_is_neural_target":False}
    (args.output_dir/"training_summary.json").write_text(json.dumps(summary,indent=2)); print(json.dumps(tm,indent=2))
    print(f"TOTAL_SECONDS={summary['seconds']:.1f}",flush=True)

if __name__=="__main__": main()
